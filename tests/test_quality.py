"""quality.py: the SSIMULACRA 2 port, its float32 recursive Gaussian, tile choice and inspection.

The port follows libjxl's own ``ssimulacra2`` tool (decided 2026-11-05, DESIGN §11.2), so the
scores are held to that tool:

* ``CPP``: libjxl 0.12.0's ``ssimulacra2 ref.png dist.png`` on the synthetic pairs below
  (8-bit sRGB PNGs, no colour profile), committed as numbers. The numpy-only pairs are held
  within 0.01; the JPEG pairs within 0.1, because a different JPEG encoder build makes different
  files.
* ``test_matches_the_installed_tool`` runs the tool itself on the same pairs and holds the port
  within 0.1. It skips where the tool isn't installed.

The exact-tap port this replaces (``ssimulacra2-exact/1``) scored the text pairs at JPEG q88,
q90, q92 and q95 0.45, 0.49, 0.56 and 0.84 above the tool; this port matches the tool there to
1e-8.

``_recursive_gaussian`` and ``_fast_gaussian_1d`` below are an independent double-precision port
of libjxl's ``CreateRecursiveGaussian`` and the scalar ``FastGaussian1D`` (tools/gauss_blur.cc,
BSD-3-Clause, Copyright the JPEG XL Project Authors; the full notice is in
src/pdfintake/quality.py), used to check the module's float32 tables and blur.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fixtures import build as fb
from pdfintake import quality

# --------------------------------------------------------------------------- the recursion (libjxl)


def _recursive_gaussian(sigma: float) -> tuple[int, list[float], list[float]]:
    radius = round(3.2795 * sigma + 0.2546)
    pi_div_2r = math.pi / (2.0 * radius)
    omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r]
    p1, p3, p5 = 1.0 / math.tan(0.5 * omega[0]), -1.0 / math.tan(0.5 * omega[1]), 1.0 / math.tan(0.5 * omega[2])
    r1, r3, r5 = p1 * p1 / math.sin(omega[0]), -p3 * p3 / math.sin(omega[1]), p5 * p5 / math.sin(omega[2])
    rho = [math.exp(-0.5 * sigma * sigma * w * w) / radius for w in omega]
    d13, d35, d51 = p1 * r3 - r1 * p3, p3 * r5 - r3 * p5, p5 * r1 - r5 * p1
    z15, z35 = d35 / d13, d51 / d13
    a = np.linalg.inv(np.array([[p1, p3, p5], [r1, r3, r5], [z15, z35, 1.0]]))
    gamma = np.array([1.0, radius * radius - sigma * sigma, z15 * rho[0] + z35 * rho[1] + rho[2]])
    beta = a @ gamma
    assert abs(beta[0] * p1 + beta[1] * p3 + beta[2] * p5 - 1.0) < 1e-12   # (39): weights normalised
    n2 = [-beta[i] * math.cos(omega[i] * (radius + 1.0)) for i in range(3)]
    d1 = [-2.0 * math.cos(omega[i]) for i in range(3)]
    return radius, n2, d1


def _fast_gaussian_1d(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius, n2, d1 = _recursive_gaussian(sigma)
    xs = len(x)
    out = np.zeros(xs)
    prev = [0.0, 0.0, 0.0]
    prev2 = [0.0, 0.0, 0.0]
    for n in range(-radius + 1, xs):
        left, right = n - radius - 1, n + radius - 1
        s = (x[left] if left >= 0 else 0.0) + (x[right] if right < xs else 0.0)
        total = 0.0
        for k in range(3):
            y = n2[k] * s - d1[k] * prev[k] - prev2[k]
            prev2[k], prev[k] = prev[k], y
            total += y
        if n >= 0:
            out[n] = total
    return out


def _blur64(a: np.ndarray) -> np.ndarray:
    rec = np.apply_along_axis(_fast_gaussian_1d, 1, a.astype(np.float64))
    return np.apply_along_axis(_fast_gaussian_1d, 0, rec)


def _ulps_from_a_tie(v: float) -> float:
    """How far ``v`` lies from the nearest float32 rounding tie, in float32 ulps (0.5 = on a float32)."""
    f = np.float32(v)
    if float(f) == v:
        return 0.5
    other = np.nextafter(f, np.float32(np.inf) if v > float(f) else np.float32(-np.inf))
    ulp = abs(float(other) - float(f))
    return abs(abs(v - float(f)) - ulp / 2) / ulp


def test_tables_match_an_independent_derivation() -> None:
    rg = quality.create_recursive_gaussian(1.5)
    radius, n2, d1 = _recursive_gaussian(1.5)
    assert rg.radius == radius == 5
    assert rg.n2.dtype == rg.d1.dtype == rg.mul_in.dtype == np.float32
    assert np.array_equal(rg.n2, np.array(n2, np.float32))
    assert np.array_equal(rg.d1, np.array(d1, np.float32))
    for k in range(3):
        d, n = d1[k], n2[k]
        mul_in = [n, -d * n, d * d * n - n, -d * d * d * n + 2 * d * n]
        mul_prev = [-d, d * d - 1, -d * d * d + 2 * d, d ** 4 - 3 * d * d + 1]
        mul_prev2 = [-1, d, -d * d + 1, d * d * d - 2 * d]
        assert np.array_equal(rg.mul_in[k], np.array(mul_in, np.float32))
        assert np.array_equal(rg.mul_prev[k], np.array(mul_prev, np.float32))
        assert np.array_equal(rg.mul_prev2[k], np.array(mul_prev2, np.float32))


def test_tables_are_far_from_float32_ties() -> None:
    # Every double coefficient lies well clear of a float32 rounding tie, so neither a compiler
    # fusing libjxl's double arithmetic nor a different inverse can change a float32 table entry.
    radius, n2, d1 = _recursive_gaussian(1.5)
    values = []
    for d, n in zip(d1, n2):
        values += [n, d, d * d - 1, -d * d * d + 2 * d, d ** 4 - 3 * d * d + 1, -d * d + 1, d * d * d - 2 * d,
                   -d * n, d * d * n - n, -d * d * d * n + 2 * d * n]
    assert min(_ulps_from_a_tie(v) for v in values) > 0.01


# --------------------------------------------------------------------------- the float32 FMA


def _exact_fma32(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    lo = np.float32(float(exact))                     # within an ulp of the answer; settle it exactly
    best = min((lo, np.nextafter(lo, np.float32(np.inf)), np.nextafter(lo, np.float32(-np.inf))),
               key=lambda f: (abs(Fraction(float(f)) - exact), int(np.array(f).view(np.uint32)) & 1))
    return best


def test_fma_rounds_once() -> None:
    rng = np.random.default_rng(5)
    a = (rng.standard_normal(3000) * 10.0 ** rng.integers(-6, 6, 3000)).astype(np.float32)
    b = (rng.standard_normal(3000) * 10.0 ** rng.integers(-6, 6, 3000)).astype(np.float32)
    c = (rng.standard_normal(3000) * 10.0 ** rng.integers(-6, 6, 3000)).astype(np.float32)
    got = quality._fma(a, b, c)
    assert got.dtype == np.float32
    want = np.array([_exact_fma32(x, y, z) for x, y, z in zip(a, b, c)], np.float32)
    assert np.array_equal(got, want)


def test_fma_settles_a_double_rounding_tie() -> None:
    # a * b = 2^-24 - 2^-70; plus c = 1 + 2^-23 the float64 sum rounds to the float32 tie
    # 1 + 2^-23 + 2^-24, which rounds to even (up). The exact value is just below it: down.
    a = np.array([np.float32(2.0 ** -24) * np.float32(1 + 2.0 ** -23)], np.float32)
    b = np.array([1 - 2.0 ** -23], np.float32)
    c = np.array([1 + 2.0 ** -23], np.float32)
    naive = (a.astype(np.float64) * b + c).astype(np.float32)
    assert naive[0] == np.float32(1 + 2.0 ** -22)
    assert quality._fma(a, b, c)[0] == np.float32(1 + 2.0 ** -23) == _exact_fma32(a[0], b[0], c[0])
    # mirrored: the float64 sum is the tie -(1 + 2^-24), the exact value lies beyond it
    assert quality._fma(a, b, -c)[0] == -np.float32(1 + 2.0 ** -23) == _exact_fma32(a[0], b[0], -c[0])


def test_fma_in_the_subnormal_range() -> None:
    tiny = np.float32(2.0 ** -140)
    a = np.array([tiny, tiny, np.float32(3.0)], np.float32)
    b = np.array([np.float32(0.75), np.float32(-1.25), np.float32(2.0 ** -130)], np.float32)
    c = np.array([np.float32(2.0 ** -149), np.float32(0.0), np.float32(-2.0 ** -129)], np.float32)
    want = np.array([_exact_fma32(x, y, z) for x, y, z in zip(a, b, c)], np.float32)
    assert np.array_equal(quality._fma(a, b, c), want)


# --------------------------------------------------------------------------- the blur


def test_blur_follows_the_recursion() -> None:
    impulse = np.zeros((33, 64), np.float32)
    impulse[16, 32] = 1.0
    got = quality.blur(impulse)
    assert got.dtype == np.float32
    assert np.abs(got - _blur64(impulse)).max() < 1e-7
    a = np.random.default_rng(0).random((23, 31)).astype(np.float32)
    assert np.abs(quality.blur(a) - _blur64(a)).max() < 2e-6


@pytest.mark.parametrize("h, w", [(1, 1), (3, 5), (8, 7), (9, 12), (13, 16), (17, 9), (64, 3)])
def test_blur_of_every_size_and_both_paths(h: int, w: int) -> None:
    # widths under 8 run scalar steps only; 9 to 12 add the remainder; 13 and up the 4-lane form.
    a = np.random.default_rng(h * 100 + w).random((2, h, w)).astype(np.float32) + 0.5
    got = quality.blur(a)
    assert got.shape == a.shape and got.dtype == np.float32
    for p in range(2):
        assert np.abs(got[p] - _blur64(a[p])).max() < 4e-6


def test_blur_is_libjxl_float32_bit_for_bit() -> None:
    # Pinned when the port was made: these planes' blur, as float32 bytes. The scores this blur
    # gives match libjxl's tool to 1e-8, so any change here is a change of arithmetic.
    a = np.random.default_rng(11).random((2, 37, 53)).astype(np.float32)
    digest = hashlib.sha256(quality.blur(a).tobytes()).hexdigest()
    assert digest == "a45aec5ff330498ea1009baa0d1590dd26b76299201fbe05c993114efe635ff7"


def test_srgb_curve_and_cube_root_stay_close_to_exact() -> None:
    v = np.arange(256) / 255.0
    exact = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    assert quality._SRGB_TO_LINEAR.dtype == np.float32
    assert np.abs(quality._SRGB_TO_LINEAR - exact).max() < 2e-6
    assert np.all(np.diff(quality._SRGB_TO_LINEAR) > 0)
    # libjxl only takes cube roots of opsin values, which the bias keeps at 0.0038 or more
    x = (0.0037 + np.random.default_rng(4).random(10000) * 1.1).astype(np.float32)
    x[:2] = (0.0037, 1.0)
    want = np.cbrt(x.astype(np.float64))
    got = quality._cube_root_and_add(x, np.float32(0.0)).astype(np.float64)
    ulp = np.spacing(want.astype(np.float32)).astype(np.float64)
    assert np.all(np.abs(got - want) <= 6 * ulp)                    # libjxl: "6 ulp max error"
    assert quality._cube_root_and_add(np.zeros(4, np.float32), np.float32(0.25)).tolist() == [0.25] * 4


# --------------------------------------------------------------------------- synthetic pairs


def _scene(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([150 + 70 * np.sin(xx / 23.0 + seed) * np.cos(yy / 31.0),
                    140 + 60 * np.cos(xx / 41.0 - yy / 19.0),
                    130 + 50 * np.sin((xx + yy) / 27.0)], axis=-1)
    for _ in range(12):
        y0, x0 = int(rng.integers(0, h - 8)), int(rng.integers(0, w - 30))
        img[y0:y0 + 6, x0:x0 + int(rng.integers(10, 30))] = rng.integers(10, 60)
    img += rng.normal(0.0, 3.0, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _box3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")
    s = sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(3) for dx in range(3)) / 9.0
    return np.clip(np.rint(s), 0, 255).astype(np.uint8)


def _quantize(a: np.ndarray, step: int) -> np.ndarray:
    return np.clip(np.rint(a.astype(np.float64) / step) * step, 0, 255).astype(np.uint8)


def _noisy(a: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    n = np.random.default_rng(seed).normal(0.0, sigma, a.shape)
    return np.clip(np.rint(a + n), 0, 255).astype(np.uint8)


def _blocky(a: np.ndarray, k: int, mix: float) -> np.ndarray:
    h, w = a.shape[:2]
    m = a.astype(np.float64).reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))
    m = np.repeat(np.repeat(m, k, axis=0), k, axis=1)
    return np.clip(np.rint((1 - mix) * a + mix * m), 0, 255).astype(np.uint8)


def _half_up(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    m = a.astype(np.float64).reshape(h // 2, 2, w // 2, 2, -1).mean(axis=(1, 3))
    return np.clip(np.rint(np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)), 0, 255).astype(np.uint8)


def _jpeg(img: Image.Image, q: int) -> np.ndarray:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert(img.mode))


def _pairs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Numpy-only pairs: every platform builds the same samples."""
    a = _scene(256, 256, 1)
    g = _scene(224, 224, 2).mean(axis=2).round().astype(np.uint8)
    o = _scene(101, 147, 3)
    bright = (255 - a // 5).astype(np.uint8)
    return {
        "box3": (a, _box3(a)),
        "quant8": (a, _quantize(a, 8)),
        "quant24": (a, _quantize(a, 24)),
        "noise2": (a, _noisy(a, 2.0, 7)),
        "noise6": (a, _noisy(a, 6.0, 8)),
        "blocky8": (a, _blocky(a, 8, 0.35)),
        "half": (a, _half_up(a)),
        "gray_box3": (g, _box3(g[:, :, None])[:, :, 0]),
        "gray_noise3": (g, _noisy(g, 3.0, 9)),
        "odd_quant16": (o, _quantize(o, 16)),
        "bright_noise2": (bright, _noisy(bright, 2.0, 1)),
    }


def _jpeg_pairs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Text on white and a colour scene as JPEG near the bar (q88 to q95), where the old port drifted."""
    text = fb.text_like(512, 512, lines=14)
    scene = Image.fromarray(_scene(512, 384, 6))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for q in (88, 90, 92, 95):
        out[f"text_q{q}"] = (np.asarray(text.convert("RGB")), _jpeg(text.convert("RGB"), q))
        out[f"scene_q{q}"] = (np.asarray(scene), _jpeg(scene, q))
    out["text_grey_q92"] = (np.asarray(text), _jpeg(text, 92))
    return out


# libjxl 0.12.0 `ssimulacra2 ref.png dist.png` on these pairs.
CPP = {"box3": 67.682542, "quant8": 88.376188, "quant24": 39.814466, "noise2": 90.266066,
       "noise6": 69.816475, "blocky8": 53.496868, "half": 72.003037, "gray_box3": 65.693198,
       "gray_noise3": 78.599415, "odd_quant16": 64.625367, "bright_noise2": 87.804778}
CPP_JPEG = {"text_q88": 89.160361, "text_q90": 90.243784, "text_q92": 91.660294, "text_q95": 94.101846,
            "text_grey_q92": 91.660294, "scene_q88": 85.124764, "scene_q90": 85.982192,
            "scene_q92": 86.605799, "scene_q95": 88.450104}


@pytest.fixture(scope="module")
def pairs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return _pairs()


@pytest.fixture(scope="module")
def jpeg_pairs() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return _jpeg_pairs()


@pytest.mark.parametrize("name", sorted(CPP))
def test_matches_libjxl(pairs: dict[str, tuple[np.ndarray, np.ndarray]], name: str) -> None:
    assert abs(quality.ssimulacra2(*pairs[name]) - CPP[name]) < 0.01


@pytest.mark.parametrize("name", sorted(CPP_JPEG))
def test_matches_libjxl_near_the_bar(jpeg_pairs: dict[str, tuple[np.ndarray, np.ndarray]], name: str) -> None:
    assert abs(quality.ssimulacra2(*jpeg_pairs[name]) - CPP_JPEG[name]) < 0.1


def _tool() -> str | None:
    found = shutil.which("ssimulacra2")
    if found:
        return found
    brew = "/opt/homebrew/bin/ssimulacra2"
    return brew if os.access(brew, os.X_OK) else None


LIVE = ["text_q88", "text_q90", "text_q92", "text_q95", "text_grey_q92", "scene_q90", "bright_noise2",
        "gray_noise3", "odd_quant16", "quant8"]


@pytest.mark.parametrize("name", LIVE)
def test_matches_the_installed_tool(pairs: dict[str, tuple[np.ndarray, np.ndarray]],
                                    jpeg_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
                                    name: str, tmp_path: Path) -> None:
    tool = _tool()
    if tool is None:
        pytest.skip("libjxl's ssimulacra2 tool isn't installed")
    ref, dist = {**pairs, **jpeg_pairs}[name]
    Image.fromarray(ref).save(tmp_path / "ref.png")
    Image.fromarray(dist).save(tmp_path / "dist.png")
    run = subprocess.run([tool, str(tmp_path / "ref.png"), str(tmp_path / "dist.png")],
                         capture_output=True, text=True, timeout=60, check=True)
    assert abs(quality.ssimulacra2(ref, dist) - float(run.stdout.split()[0])) < 0.1


@pytest.mark.parametrize("shape", [(64, 64, 3), (37, 53, 3), (80, 120), (8, 8, 3)])
def test_identical_scores_100(shape: tuple[int, ...]) -> None:
    a = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    assert quality.ssimulacra2(a, a.copy()) == 100.0


def test_grey_is_scored_as_three_equal_channels(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    g, d = pairs["gray_noise3"]
    rgb = quality.ssimulacra2(np.repeat(g[:, :, None], 3, 2), np.repeat(d[:, :, None], 3, 2))
    assert quality.ssimulacra2(g, d) == rgb


def test_more_distortion_scores_lower(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    s = {k: quality.ssimulacra2(*pairs[k]) for k in ("quant8", "quant24", "noise2", "noise6")}
    assert s["quant8"] > s["quant24"] and s["noise2"] > s["noise6"]


def test_rejects_bad_inputs() -> None:
    a = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError):
        quality.ssimulacra2(a, np.zeros((16, 17, 3), np.uint8))
    with pytest.raises(ValueError):
        quality.ssimulacra2(a.astype(np.float32), a)
    with pytest.raises(ValueError):
        quality.ssimulacra2(np.zeros((16, 16, 4), np.uint8), np.zeros((16, 16, 4), np.uint8))


@pytest.mark.parametrize("shape", [(7, 64, 3), (64, 3), (3, 3)])
def test_images_under_8px_are_refused_not_scored_100(shape: tuple[int, ...]) -> None:
    a = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        quality.ssimulacra2(a, 255 - a)
    with pytest.raises(ValueError, match="too small"):
        quality.worst_tile(a, 255 - a)


def test_metric_id() -> None:
    assert quality.METRIC_ID == "ssimulacra2-libjxl/1;worst-of-4@512;inspect-300ppi"


def test_metric_id_names_the_defaults_a_page_is_scored_with() -> None:
    # The record's smaller.metric carries METRIC_ID, so its tile and inspection parts must say
    # what worst_tile and at_inspection actually do by default (smaller.build passes neither).
    import inspect

    scorer, tiles, inspect_part = quality.METRIC_ID.split(";")
    assert scorer == "ssimulacra2-libjxl/1"
    wt = inspect.signature(quality.worst_tile).parameters
    dt = inspect.signature(quality.detail_tiles).parameters
    assert tiles == f"worst-of-{wt['n'].default}@{wt['size'].default}"
    assert (dt["n"].default, dt["size"].default) == (wt["n"].default, wt["size"].default)
    ppi = inspect.signature(quality.at_inspection).parameters["inspect_ppi"].default
    assert inspect_part == f"inspect-{ppi:g}ppi"


# --------------------------------------------------------------------------- tiles


def _page_with_detailed_corner() -> np.ndarray:
    page = np.full((2048, 1536), 240, np.uint8)
    rng = np.random.default_rng(3)
    corner = page[1024:, 512:]                                 # bottom-right 1024 x 1024
    for y in range(0, 1024, 24):                               # text-like bars
        x = 0
        while x < 1000:
            w = int(rng.integers(8, 60))
            corner[y:y + 10, x:x + w] = int(rng.integers(0, 60))
            x += w + int(rng.integers(4, 14))
    return page


def test_detailed_tiles_are_chosen() -> None:
    page = _page_with_detailed_corner()
    tiles = quality.detail_tiles(page, 4, 512)
    assert len(tiles) == 4
    for y, x in tiles:
        # every tile lies mostly inside the detailed corner
        inside_h = max(0, min(y + 512, 2048) - max(y, 1024))
        inside_w = max(0, min(x + 512, 1536) - max(x, 512))
        assert inside_h * inside_w >= 0.5 * 512 * 512, (y, x)
    for i, (y, x) in enumerate(tiles):                         # no two tiles overlap
        for (y2, x2) in tiles[i + 1:]:
            assert abs(y - y2) >= 512 or abs(x - x2) >= 512
    first = tiles[0]
    assert first[0] >= 1024 and first[1] >= 512                # the best tile is fully inside
    assert quality.detail_tiles(page, 4, 512) == tiles          # deterministic


def test_small_image_is_one_tile() -> None:
    a = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    assert quality.detail_tiles(a, 4, 512) == [(0, 0)]
    strip = np.random.default_rng(0).integers(0, 256, (300, 1600), dtype=np.uint8)
    tiles = quality.detail_tiles(strip, 4, 512)
    assert 1 <= len(tiles) <= 3 and all(y == 0 for y, _ in tiles)


def test_worst_tile_is_the_lowest_tile_score() -> None:
    page = _page_with_detailed_corner()
    dist = page.copy()
    y, x = quality.detail_tiles(page)[1]
    dist[y:y + 512, x:x + 512] = _noisy(page[y:y + 512, x:x + 512, None], 8.0, 1)[:, :, 0]
    ts = quality.worst_tile(page, dist)
    assert len(ts.tiles) == 4
    assert ts.score == min(t[2] for t in ts.tiles)
    worst = min(ts.tiles, key=lambda t: t[2])
    assert (worst[0], worst[1]) == (y, x) and ts.score < 90
    assert quality.worst_tile(page, page).score == 100.0


# --------------------------------------------------------------------------- inspection


def test_at_inspection_brings_both_to_300ppi() -> None:
    ref = Image.fromarray(_scene(1200, 900, 4))                # 600 ppi: 1.5 x 2 inches
    dist = ref.resize((600, 800), Image.Resampling.LANCZOS)    # 400 ppi
    a, b = quality.at_inspection(ref, dist, 600.0, 400.0)
    assert a.shape == b.shape == (600, 450, 3) and a.dtype == np.uint8


def test_at_inspection_never_upsamples_the_original() -> None:
    ref = Image.fromarray(_scene(400, 300, 5)).convert("L")    # 200 ppi
    a, b = quality.at_inspection(ref, ref.copy(), 200.0, 200.0)
    assert a.shape == (400, 300) and np.array_equal(a, np.asarray(ref)) and np.array_equal(a, b)


def test_at_inspection_refuses_different_areas() -> None:
    ref = Image.new("RGB", (600, 600))
    with pytest.raises(ValueError):
        quality.at_inspection(ref, Image.new("RGB", (600, 600)), 300.0, 150.0)
