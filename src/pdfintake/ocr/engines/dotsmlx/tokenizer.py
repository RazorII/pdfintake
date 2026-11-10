"""dots.mocr's prompt, its token ids, and the streaming decoder for what it writes.

The chat template is dots.ocr's (``chat_template.jinja`` in the model folder, MIT, Copyright (c)
2025 rednote-hilab), rendered here as the fixed string it produces for one user turn holding one
image and then the prompt text, as mlx-vlm's ``apply_chat_template`` builds that turn. The image
placeholder is then repeated once per image token, as transformers' Qwen2.5-VL processor does
(Apache-2.0, Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team).
``StreamDecoder`` is ported from mlx-vlm 0.7.4 (``mlx_vlm/tokenizer_utils.py``,
``BPEStreamingDetokenizer``), MIT, Copyright (c) 2025 Prince Canuma, after the GPT-2 byte decoder
it credits. Tokenising uses the ``tokenizers`` library on the model folder's ``tokenizer.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

USER, END_USER, ASSISTANT = "<|user|>", "<|endofuser|>", "<|assistant|>"
IMAGE_START, IMAGE_PAD, IMAGE_END = "<|img|>", "<|imgpad|>", "<|endofimg|>"
REPLACEMENT_CHAR = "�"


def render_prompt(prompt: str, image_tokens: int) -> str:
    """The chat-templated text: one user turn with the image first, then the prompt, then the
    assistant's turn opening, with the image placeholder repeated ``image_tokens`` times."""
    return USER + IMAGE_START + IMAGE_PAD * image_tokens + IMAGE_END + prompt + END_USER + ASSISTANT


def _byte_decoder() -> dict[str, int]:
    """GPT-2's map from the printable stand-in characters back to bytes."""
    limits = [0, ord("!"), ord("~") + 1, ord("¡"), ord("¬") + 1, ord("®"), ord("ÿ") + 1]
    out: dict[str, int] = {}
    n = 0
    for i, (start, stop) in enumerate(zip(limits, limits[1:])):
        if i % 2 == 0:
            for b in range(start, stop):
                out[chr(2**8 + n)] = b
                n += 1
        else:
            for b in range(start, stop):
                out[chr(b)] = b
    return out


class Tokenizer:
    """The model folder's tokenizer.json, plus the id -> token text map the decoder needs."""

    def __init__(self, path: Path, eos_token: str | None = None) -> None:
        from tokenizers import Tokenizer as _Tok

        self._tok = _Tok.from_file(str(path))
        vocab: dict[str, int] = self._tok.get_vocab(with_added_tokens=True)
        self.token_text: list[str | None] = [None] * (max(vocab.values()) + 1)
        for text, tid in vocab.items():
            self.token_text[tid] = text
        self.image_pad_id = self._tok.token_to_id(IMAGE_PAD)
        if self.image_pad_id is None:
            raise ValueError(f"{path} has no {IMAGE_PAD} token")
        self.eos_id = self._tok.token_to_id(eos_token) if eos_token else None

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=True).ids

    def decoder(self) -> StreamDecoder:
        return StreamDecoder(self.token_text)


class StreamDecoder:
    """Turns token ids into text as they come, a word at a time.

    Text is released when a token that starts with a space arrives (the bytes before it are then
    complete), and ``last_segment`` holds back a release that ends in a broken character. So the
    segments, and when they arrive, are mlx-vlm's: a stop rule fed them sees what it would see
    there. ``finalize`` releases the rest, dropping undecodable bytes.
    """

    _BYTES: dict[str, int] | None = None

    def __init__(self, token_text: list[str | None]) -> None:
        if StreamDecoder._BYTES is None:
            StreamDecoder._BYTES = _byte_decoder()
        self._bytes = StreamDecoder._BYTES
        self._token_text = token_text
        self.text = ""
        self._unflushed = ""
        self._offset = 0

    def _decode(self, chars: str, errors: str) -> str:
        # A character outside the byte map (only in added tokens) has no bytes of its own; it is
        # skipped, where mlx-vlm would raise.
        return bytes(self._bytes[c] for c in chars if c in self._bytes).decode("utf-8", errors=errors)

    def add(self, token: int) -> None:
        v: Any = self._token_text[token] if 0 <= token < len(self._token_text) else None
        if not v:                         # an id the vocabulary doesn't have (the model's padding rows)
            return
        if self._bytes.get(v[0]) == 32:   # starts with a space: what came before is whole
            self.text += self._decode(self._unflushed, "replace")
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self) -> None:
        self.text += self._decode(self._unflushed, "ignore")
        self._unflushed = ""

    @property
    def last_segment(self) -> str:
        """The text released since the last call, or "" while it ends in a broken character."""
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self._offset:]
            self._offset = len(text)
            return segment
        return ""
