"""The in-process dots.mocr port (``pdfintake.ocr.engines.dotsmlx``).

Two kinds of test live here:

* **Everywhere**, no model and no mlx: every module of the package imports with mlx and
  tokenizers blocked, and ``load`` then says mlx is missing; ``smart_resize`` keeps its bounds;
  the patches have the Qwen2-VL processor's shape, values and merge-block order.
* **With the model**, on darwin-arm64 only, with mlx and tokenizers installed and the 8-bit
  dots.mocr snapshot on disk: ``PDFINTAKE_DOTS_MODEL`` names the folder, else the Hugging Face
  cache's copy of mlx-community/dots.mocr-8bit at revision ``REVISION`` is used. Nothing is
  downloaded; without the snapshot these tests skip and say so. A synthetic page reads back, the
  stop callback and the deadline stop it, and two reads are identical.

All text is made up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from pdfintake import platforms
from pdfintake.ocr.engines.dotsmlx.config import ImageConfig
from pdfintake.ocr.engines.dotsmlx.preprocess import image_token_count, preprocess, smart_resize

SRC = Path(__file__).resolve().parents[1] / "src"
REVISION = "6d9e750ba44cf99e3c608cfdf47e59df209dce13"
OCR_PROMPT = "Extract the text content from this image."
SAMPLE = "Sample invoice 4521"
MODULES = ("", "._mlx", ".config", ".preprocess", ".tokenizer", ".language", ".vision", ".model", ".generate")


REQUIRE = os.environ.get("PDFINTAKE_REQUIRE_THOROUGH") == "1"      # scripts/thorough-local.sh: fail, never skip


def _model_dir() -> Path:
    named = os.environ.get("PDFINTAKE_DOTS_MODEL")
    if named:
        return Path(named)
    from pdfintake import thorough

    if not thorough.model_problems():                # thorough mode's own copy, installed and verified
        return thorough.model_dir()
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub")
    return hub / "models--mlx-community--dots.mocr-8bit" / "snapshots" / REVISION


def _page(text: str = SAMPLE, size: tuple[int, int] = (850, 1100)) -> Image.Image:
    """A letter-shaped page with one line of text near the top, in Pillow's own font (every OS has
    it). At 72 px on a short strip dots.mocr misreads this font; at 36 px on a page it reads it."""
    img = Image.new("RGB", size, "white")
    font = ImageFont.load_default(size=36)
    ImageDraw.Draw(img).text((80, 120), text, fill="black", font=font)
    return img


# --------------------------------------------------------------------------- everywhere

_BLOCKED_IMPORT = r"""
import importlib, importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx", "tokenizers"}:
            raise ModuleNotFoundError("blocked: " + fullname, name=fullname)
        return None
sys.meta_path.insert(0, Block())
out = {}
for suffix in json.loads(sys.argv[2]):
    importlib.import_module("pdfintake.ocr.engines.dotsmlx" + suffix)
    out[suffix] = "ok"
import pdfintake.ocr.engines.dotsmlx as dots
try:
    dots.load("/nonexistent")
    out["load"] = "loaded"
except ModuleNotFoundError as exc:
    out["load"] = "missing " + str(exc.name)
print("@@" + json.dumps(out))
"""


def test_the_package_imports_without_mlx_and_load_says_so():
    run = subprocess.run([sys.executable, "-I", "-c", _BLOCKED_IMPORT, str(SRC), json.dumps(MODULES)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(next(ln for ln in run.stdout.splitlines() if ln.startswith("@@"))[2:])
    assert out == {**{m: "ok" for m in MODULES}, "load": "missing mlx"}


@pytest.mark.parametrize("h,w", [(1600, 1131), (3508, 2480), (50, 900), (2936, 613), (28, 28)])
def test_smart_resize_keeps_the_factor_and_the_pixel_bounds(h: int, w: int):
    cfg = ImageConfig()
    rh, rw = smart_resize(h, w, 28, cfg.min_pixels, cfg.max_pixels)
    assert rh % 28 == 0 and rw % 28 == 0
    assert cfg.min_pixels <= rh * rw <= cfg.max_pixels
    assert abs(rh / rw - h / w) / (h / w) < 0.6           # the shape survives (loosely: tiny images round hard)


def test_smart_resize_refuses_an_extreme_aspect_ratio():
    with pytest.raises(ValueError):
        smart_resize(10, 2100, 28, 3136, 11289600)


def test_preprocess_gives_the_processor_layout():
    cfg = ImageConfig()
    img = _page(size=(1131, 1600))
    pixels, grid = preprocess(img, cfg)
    assert grid == (1, 114, 80)                                       # 1596 x 1120 after smart_resize
    assert pixels.shape == (114 * 80, 3 * 1 * 14 * 14) and pixels.dtype == np.float32
    assert image_token_count(grid, cfg.merge_size) == 114 * 80 // 4
    white = (1.0 - np.array(cfg.image_mean, np.float32)) / np.array(cfg.image_std, np.float32)
    first = pixels[0].reshape(3, 1, 14, 14)
    assert np.allclose(first[:, 0], white[:, None, None])             # the top-left patch is paper


def test_patches_come_in_merge_block_order():
    cfg = ImageConfig()
    arr = np.zeros((56, 56, 3), np.uint8)                            # 4 x 4 patches of 14 px
    for r in range(4):
        for c in range(4):
            arr[r * 14:(r + 1) * 14, c * 14:(c + 1) * 14] = 10 * (r * 4 + c)
    pixels, grid = preprocess(Image.fromarray(arr), cfg)
    assert grid == (1, 4, 4)
    std, mean = np.array(cfg.image_std, np.float32), np.array(cfg.image_mean, np.float32)
    values = [round(float(p.reshape(3, 14, 14)[0, 0, 0] * std[0] + mean[0]) * 255) // 10 for p in pixels]
    # each 2 x 2 block of patches in turn, row-major inside the block
    assert values == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]


# --------------------------------------------------------------------------- with the model

def _skip(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def model():
    if platforms.current().key != "darwin-arm64":
        _skip(f"dots.mocr runs on darwin-arm64 only, this is {platforms.current().key}")
    for lib in ("mlx.core", "tokenizers"):
        try:
            __import__(lib)
        except ImportError:
            _skip(f"{lib.split('.')[0]} isn't installed (pdfintake[thorough])")
    folder = _model_dir()
    if not (folder / "model.safetensors").is_file():
        _skip(f"no dots.mocr-8bit snapshot at {folder}; set PDFINTAKE_DOTS_MODEL to its folder")
    from pdfintake.ocr.engines import dotsmlx as dots

    return dots.load(folder)


@pytest.mark.thorough
def test_a_synthetic_page_reads_back(model):
    out = model.read(_page(), OCR_PROMPT, max_tokens=64)
    assert "4521" in out.text and "invoice" in out.text.lower(), out
    assert out.stopped in ("eos", "length") and 0 < out.tokens <= 64


@pytest.mark.thorough
def test_the_prompt_holds_one_token_per_merged_patch(model):
    pixels, grid = preprocess(_page(), model.config.image)
    ids = model.prompt_ids(OCR_PROMPT, image_token_count(grid, model.config.image.merge_size))
    assert ids.count(model.config.image_token_id) == grid[1] * grid[2] // 4


@pytest.mark.thorough
def test_two_reads_are_identical(model):
    assert model.read(_page(), OCR_PROMPT, max_tokens=32) == model.read(_page(), OCR_PROMPT, max_tokens=32)


@pytest.mark.thorough
def test_the_stop_callback_stops_it(model):
    seen: list[str] = []

    def stop(text: str) -> bool:
        seen.append(text)
        return True

    out = model.read(_page(), OCR_PROMPT, max_tokens=64, stop=stop)
    assert out.stopped == "stop" and len(seen) == 1 and out.text == seen[0] and out.tokens < 64


@pytest.mark.thorough
def test_a_passed_deadline_reads_nothing(model):
    assert tuple(model.read(_page(), OCR_PROMPT, max_tokens=64, deadline=time.monotonic() - 1)) == ("", 0, "deadline")
