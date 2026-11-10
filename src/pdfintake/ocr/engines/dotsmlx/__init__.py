"""dots.mocr in-process on Apple silicon, on mlx alone: no mlx-vlm, transformers or huggingface_hub.

Written for pdfintake (MIT) around code ported from mlx-vlm 0.7.4 (MIT, Copyright (c) 2025 Prince
Canuma) and transformers 5.18 (Apache-2.0, Copyright 2024 The Qwen team, Alibaba Group and the
HuggingFace Inc. team); each module's header says what came from where. The model is dots.mocr
(MIT, Copyright (c) 2025 rednote-hilab), as mlx-community's 8-bit conversion.

    model = load("/path/to/dots.mocr-8bit")         # the snapshot folder, nothing is fetched
    text, tokens, stopped = model.read(image, prompt, max_tokens=6000)

``read`` reproduces mlx-vlm's greedy (temperature 0) output for the same image and prompt token
for token: same image patches, same prompt ids, same prefill chunks, same argmax. The image is
read as given, so any downscaling is the caller's. Runtime needs: ``mlx`` and ``tokenizers``, plus
numpy and Pillow. Every module in this package imports without mlx (``_mlx.py``); ``load`` raises
``ModuleNotFoundError`` where it's missing.

mlx-vlm's licence, which covers the code this package ports from it:

    MIT License

    Copyright © 2025 Prince Canuma

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

from . import _mlx
from .config import ModelConfig, read_config

if TYPE_CHECKING:
    import PIL.Image

    from .model import DotsOCR
    from .tokenizer import Tokenizer

Stopped = Literal["eos", "length", "stop", "deadline"]


class DotsRead(NamedTuple):
    text: str                # what the model wrote (end token excluded)
    tokens: int              # tokens generated, counted as mlx-vlm's generation_tokens: an end token counts
    stopped: Stopped         # eos: it ended; length: max_tokens; stop: the callback; deadline: out of time


class DotsModel:
    """A loaded dots.mocr. ``read`` may be called from several threads; calls run one at a time."""

    def __init__(self, model: DotsOCR, tokenizer: Tokenizer, config: ModelConfig) -> None:
        self._model = model
        self._tok = tokenizer
        self.config = config
        ids = list(config.eos_token_ids)
        if tokenizer.eos_id is not None and tokenizer.eos_id not in ids:
            ids.append(tokenizer.eos_id)
        self.eos_token_ids = frozenset(ids)
        self._lock = threading.Lock()

    def prompt_ids(self, prompt: str, image_tokens: int) -> list[int]:
        from .tokenizer import render_prompt

        return self._tok.encode(render_prompt(prompt, image_tokens))

    def read(self, image: PIL.Image.Image, prompt: str, max_tokens: int,
             stop: Callable[[str], bool] | None = None, deadline: float | None = None) -> DotsRead:
        """Read ``image`` with ``prompt``, greedily, up to ``max_tokens`` tokens.

        ``stop(text_so_far)`` is called each time the text grows, including once more with the
        last of it when the model ends; returning True stops generation there (``stopped="stop"``
        unless the model had already ended). The text is released a word at a time, exactly as
        mlx-vlm's stream releases it, so a stop rule sees the same text at the same token. A stop
        returns the text released so far; an end or ``max_tokens`` returns all of it.
        ``deadline`` is a ``time.monotonic()`` value; past it the read stops (checked when it
        starts, again once it has the model, and after every token; the image and prompt are
        read in one go, so the first check after that comes with the first token). Raises
        ``ValueError`` for an image smart_resize refuses (one side over 200 times the other).
        """
        from ._mlx import mx
        from .generate import greedy_tokens, wired_limit
        from .preprocess import image_token_count, preprocess

        if max_tokens < 0:
            raise ValueError("max_tokens must be 0 or more")
        if deadline is not None and time.monotonic() >= deadline:
            return DotsRead("", 0, "deadline")
        pixels, grid = preprocess(image, self.config.image)
        ids = self.prompt_ids(prompt, image_token_count(grid, self.config.image.merge_size))
        with self._lock, wired_limit(self._model):
            if deadline is not None and time.monotonic() >= deadline:
                return DotsRead("", 0, "deadline")
            decoder = self._tok.decoder()
            text, n, why = "", 0, None
            gen = greedy_tokens(self._model, mx.array([ids]), mx.array(pixels), grid, max_tokens)
            try:
                for token in gen:
                    n += 1
                    if token in self.eos_token_ids:
                        why = "eos"
                        break
                    decoder.add(token)
                    segment = decoder.last_segment
                    if segment:
                        text += segment
                        if stop is not None and stop(text):
                            return DotsRead(text, n, "stop")
                    if deadline is not None and time.monotonic() >= deadline:
                        return DotsRead(text, n, "deadline")
            finally:
                gen.close()
                mx.clear_cache()
            decoder.finalize()
            segment = decoder.last_segment
            if segment:
                text += segment
                if stop is not None:
                    stop(text)
            return DotsRead(text, n, why or "length")


def load(model_dir: str | Path) -> DotsModel:
    """dots.mocr from its model folder (config.json, preprocessor_config.json, tokenizer.json,
    tokenizer_config.json, generation_config.json, model.safetensors). Raises
    ``ModuleNotFoundError`` without mlx, ``FileNotFoundError`` for a missing file, and
    ``ValueError`` for a model this port doesn't cover."""
    if not _mlx.AVAILABLE:
        raise ModuleNotFoundError("mlx isn't installed; dots.mocr runs only on Apple silicon", name="mlx")
    from .model import build
    from .tokenizer import Tokenizer

    model_dir = Path(model_dir)
    config = read_config(model_dir)
    tokenizer = Tokenizer(model_dir / "tokenizer.json", config.eos_token)
    return DotsModel(build(model_dir, config), tokenizer, config)


__all__ = ["DotsModel", "DotsRead", "load"]
