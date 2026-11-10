# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Modified for pdfintake: only the path dots.mocr takes is kept (see the docstring below).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A page image to dots.mocr's pixel patches, exactly as the Qwen2-VL image processor makes them.

Ported from Hugging Face transformers 5.18 (``models/qwen2_vl/image_processing_pil_qwen2_vl.py``:
``smart_resize``, ``patchify``; ``image_transforms.py``: ``resize``, ``rescale``, ``normalize``),
Apache-2.0, Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. Only the
path dots.mocr takes is kept: one RGB image, Pillow bicubic resize, rescale, normalise, patchify.
The arithmetic is the processor's own, step for step (rescale in float64 then float32, normalise
in float32), so the patches match it bit for bit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import PIL.Image

    from .config import ImageConfig

MAX_ASPECT = 200                 # smart_resize refuses anything longer than this to one


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """(height, width) for the model: both divisible by ``factor``, the area within
    [min_pixels, max_pixels], the aspect ratio kept as closely as that allows."""
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_ASPECT}, "
                         f"got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(image: PIL.Image.Image, cfg: ImageConfig) -> tuple[np.ndarray, tuple[int, int, int]]:
    """(pixel_values, (t, h, w)): float32 patches of shape (h * w, C * T * P * P), and the patch
    grid. The model sees (h * w) / merge_size**2 image tokens."""
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    p, m, t = cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size
    rh, rw = smart_resize(image.height, image.width, p * m, cfg.min_pixels, cfg.max_pixels)
    resized = image.resize((rw, rh), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(resized)                                        # (H, W, C) uint8
    arr = (arr.astype(np.float64) * cfg.rescale_factor).astype(np.float32)
    mean = np.array(cfg.image_mean, dtype=np.float32)
    std = np.array(cfg.image_std, dtype=np.float32)
    arr = (arr - mean) / std                                         # float32, per channel
    chw = np.transpose(arr, (2, 0, 1))
    c = chw.shape[0]
    gh, gw = rh // p, rw // p
    patches = chw.reshape(c, gh // m, m, p, gw // m, m, p)
    patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6))           # (gh/m, gw/m, m, m, C, P, P)
    patches = np.broadcast_to(patches[:, :, :, :, :, None, :, :],
                              (*patches.shape[:5], t, *patches.shape[5:]))
    flat = np.ascontiguousarray(patches.reshape(gh * gw, c * t * p * p), dtype=np.float32)
    return flat, (1, gh, gw)


def image_token_count(grid: tuple[int, int, int], merge_size: int) -> int:
    t, h, w = grid
    return t * h * w // (merge_size * merge_size)
