"""dots.mocr wired together, and its weights loaded and quantised from explicit paths.

Ported from mlx-vlm 0.7.4 (``mlx_vlm/models/dots_ocr/dots_ocr.py`` for the wiring and the image
feature merge; ``mlx_vlm/utils.py`` ``load_model`` for the weight loading and the quantisation
predicate), MIT, Copyright (c) 2025 Prince Canuma. dots.mocr's architecture and weights are
rednote-hilab's (MIT, Copyright (c) 2025 rednote-hilab); the 8-bit conversion is mlx-community's.
The checkpoint is already in mlx's layout and key names, so loading is: build the modules,
quantise every module the checkpoint has ``.scales`` for, load the tensors strictly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ._mlx import mx, nn
from .language import LanguageModel
from .vision import VisionModel

if TYPE_CHECKING:
    from .config import ModelConfig


class DotsOCR(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision)
        self.language_model = LanguageModel(config.text)

    def embed(self, input_ids, pixel_values, grid: tuple[int, int, int]):
        """The prompt's input embeddings (1, L, D), the image tokens replaced by the vision
        tower's features for ``pixel_values`` in order."""
        lm = self.language_model.model
        inputs_embeds = lm.embed_tokens(input_ids)
        dtype = self.vision_tower.patch_embed.patchifier.proj.weight.dtype
        features = self.vision_tower(pixel_values.astype(dtype), grid)
        image_mask = input_ids[0] == self.config.image_token_id
        n_positions = mx.sum(image_mask).item()
        if n_positions != features.shape[0]:
            raise ValueError(f"the prompt has {n_positions} image tokens for {features.shape[0]} image features")
        cumsum = mx.cumsum(image_mask.astype(mx.int32))
        gathered = features[mx.where(image_mask, cumsum - 1, 0)]
        merged = mx.where(mx.expand_dims(image_mask, axis=-1), gathered, inputs_embeds[0])
        return mx.stack([merged], axis=0)


def _weight_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        files = [model_dir / name for name in sorted(set(weight_map.values()))]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"dots model folder {model_dir} lacks {', '.join(missing)}")
        if files:
            return files
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"dots model folder {model_dir} has no model.safetensors")
    return [single]


def build(model_dir: Path, config: ModelConfig) -> DotsOCR:
    """The model with its weights loaded and evaluated."""
    weights: dict = {}
    for path in _weight_files(Path(model_dir)):
        weights.update(mx.load(str(path)))
    model = DotsOCR(config)
    q = config.quantization
    if q:
        if q.get("mode", "affine") != "affine":
            raise ValueError(f"quantisation mode {q.get('mode')!r} isn't supported by this port")

        def predicate(path: str, module) -> bool:
            return hasattr(module, "to_quantized") and f"{path}.scales" in weights

        nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode="affine", class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model
