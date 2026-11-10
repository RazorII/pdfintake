"""dots.mocr's configuration, read from the model folder's JSON files.

Ported from mlx-vlm 0.7.4 (``mlx_vlm/models/dots_ocr/config.py``), MIT, Copyright (c) 2025 Prince
Canuma, cut down to the fields dots.mocr-8bit uses. The defaults are dots.ocr's own (MIT, Copyright
(c) 2025 rednote-hilab). Every file is read by an explicit path inside ``model_dir``; nothing is
looked up or fetched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The files a model folder must hold besides its weights (model.py finds those: the shards
# model.safetensors.index.json names, else model.safetensors).
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json")


@dataclass(frozen=True)
class TextConfig:
    """The Qwen2 language model."""

    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 151936
    attention_bias: bool = True


@dataclass(frozen=True)
class VisionConfig:
    """The dots ViT."""

    embed_dim: int = 1536
    hidden_size: int = 1536
    intermediate_size: int = 4224
    num_hidden_layers: int = 42
    num_attention_heads: int = 12
    num_channels: int = 3
    patch_size: int = 14
    post_norm: bool = True
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    use_bias: bool = False


@dataclass(frozen=True)
class ImageConfig:
    """The Qwen2-VL image processor's settings (preprocessor_config.json)."""

    min_pixels: int = 3136
    max_pixels: int = 11289600
    patch_size: int = 14
    temporal_patch_size: int = 1
    merge_size: int = 2
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    rescale_factor: float = 1 / 255


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    vision: VisionConfig
    image: ImageConfig
    image_token_id: int
    eos_token_ids: tuple[int, ...]        # config.json and generation_config.json
    eos_token: str | None                 # tokenizer_config.json's end-of-text token, by its text
    quantization: dict[str, Any] | None


def _pick(cls: type, params: dict[str, Any]) -> Any:
    names = cls.__dataclass_fields__
    return cls(**{k: v for k, v in params.items() if k in names})


def _ids(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int)]
    return []


def read_config(model_dir: Path) -> ModelConfig:
    """The configuration in ``model_dir``. Raises ``FileNotFoundError`` or ``ValueError``."""
    model_dir = Path(model_dir)
    for name in REQUIRED_FILES:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"dots model folder {model_dir} has no {name}")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "dots_ocr":
        raise ValueError(f"{model_dir}/config.json is model_type {cfg.get('model_type')!r}, not 'dots_ocr'")
    text_params = {k: v for k, v in cfg.items() if k != "vision_config"}       # the LM's keys sit at the top
    text = _pick(TextConfig, text_params)
    if cfg.get("rope_scaling"):
        raise ValueError("dots config sets rope_scaling, which this port doesn't implement")
    vision = _pick(VisionConfig, cfg.get("vision_config") or {})
    pre = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    image = _pick(ImageConfig, {k: tuple(v) if isinstance(v, list) else v for k, v in pre.items()})
    if image.temporal_patch_size != vision.temporal_patch_size or image.patch_size != vision.patch_size:
        raise ValueError("preprocessor_config.json and config.json disagree on the patch layout")

    # Stop ids: the model's and the generation config's. The tokenizer adds its end-of-text token
    # (tokenizer.py), the union mlx-vlm's StoppingCriteria uses (resolve_eos_token_ids).
    eos = _ids(cfg.get("eos_token_id"))
    gen_path = model_dir / "generation_config.json"
    if gen_path.is_file():
        eos += _ids(json.loads(gen_path.read_text(encoding="utf-8")).get("eos_token_id"))
    eos_text = None
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        eos_text = json.loads(tok_cfg_path.read_text(encoding="utf-8")).get("eos_token")
        if isinstance(eos_text, dict):
            eos_text = eos_text.get("content")
    return ModelConfig(text=text, vision=vision, image=image,
                       image_token_id=int(cfg.get("image_token_id", 151665)),
                       eos_token_ids=tuple(dict.fromkeys(eos)),
                       eos_token=eos_text if isinstance(eos_text, str) else None,
                       quantization=cfg.get("quantization"))
