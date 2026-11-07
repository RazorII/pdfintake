"""SSIMULACRA 2 quality score, worst tile of a page, compared at an inspection resolution.

This module is a numpy-only, line-by-line port of ``tools/ssimulacra2.cc`` from libjxl 0.12.0
(the JPEG XL reference implementation), including the float32 arithmetic it runs in, so that it
scores what libjxl's own ``ssimulacra2`` tool scores (decided 2026-11-05: the bar of 90 was
measured with that tool). Ported with it, in float32 and in libjxl's order of operations:

* reading 8-bit samples (``LoadFloatRow``) and undoing the sRGB curve with libjxl's rational
  polynomial (``TF_SRGB::DisplayFromEncoded``, ``lib/jxl/cms/transfer_functions-inl.h``);
* ``LinearRGBToXYB`` (``lib/jxl/enc_xyb.cc``) with its fast cube root (``CubeRootAndAdd``,
  ``lib/jxl/base/fast_math-inl.h``), then ``MakePositiveXYB``, ``Downsample`` and the maps;
* the recursive (IIR) Gaussian of ``tools/gauss_blur.cc``: ``CreateRecursiveGaussian(1.5)``,
  then ``FastGaussian``, which runs ``FastGaussian1D`` along each row (its 4-lane unrolled form)
  and the vertical recursion down each column, both with fused multiply-adds.

Each fused multiply-add is computed exactly (``_fma``: one rounding to float32), as the FMA
instructions libjxl's Highway code uses on arm64 (NEON) and on x86 with AVX2 or later. A build
for a target without FMA rounds twice and can score slightly differently. Plane averages are
summed in float64, as libjxl does, though not in its order; on the test pairs the scores match
the tool's eight printed decimals to within 1e-8.

Licence of the ported code (libjxl, BSD-3-Clause):

    Copyright (c) the JPEG XL Project Authors.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

What the score means here (``METRIC_ID``): a page image and its re-encoded copy are both
brought to 300 ppi (never above the original's own resolution), the four most detailed
512 px tiles of the original are found, and the page scores what its worst tile scores.
100 is identical; 90 or more means a difference shows only when flicking between the two.

Inputs are 8-bit sRGB arrays: ``(H, W)`` grey or ``(H, W, 3)`` RGB. A grey image is scored as
RGB with three equal channels, which is what libjxl does with a grey ``ImageBundle``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import PIL.Image

# decided 2026-11-05: libjxl's float32 recursive Gaussian replaces the exact 9-tap kernel
# ("ssimulacra2-exact/1"), which scored 0.3 to 1.1 above libjxl's tool on text near the bar
# (JPEG q88 to q95). The first component names the scorer; the rest name how a page is scored
# (worst of the 4 most detailed 512 px tiles, at 300 ppi), and must match worst_tile's and
# at_inspection's defaults, which smaller.build uses.
METRIC_ID = "ssimulacra2-libjxl/1;worst-of-4@512;inspect-300ppi"

_F32 = np.float32
_F64 = np.float64

# --------------------------------------------------------------------------- float32 FMA

_LOW29 = np.uint64((1 << 29) - 1)     # the float64 mantissa bits a float32 drops
_HALF29 = np.uint64(1 << 28)          # ... when they are exactly half a float32 ulp
_TINY = 2.0 ** -125                   # below this a float32 result may be subnormal


def _fma(a: np.ndarray | np.float32, b: np.ndarray | np.float32, c: np.ndarray | np.float32) -> np.ndarray:
    """``a * b + c`` rounded once to float32, as an FMA instruction computes it.

    ``a``, ``b`` and ``c`` are float32 (arrays or scalars, broadcast together). The product of two
    float32 values is exact in float64, so ``s = a * b + c`` in float64 carries one rounding, and
    rounding ``s`` to float32 is correct unless ``s`` landed exactly on a float32 rounding tie
    that the exact value isn't on (or sits in the subnormal range). Those rare elements get the
    exact remainder (TwoSum) and are rounded by hand.
    """
    p = np.multiply(a, b, dtype=_F64)
    s = p + np.asarray(c, dtype=_F64)
    r = np.asarray(s.astype(_F32))
    bits = s.view(np.uint64)
    risky = ((bits & _LOW29) == _HALF29) | (np.abs(s) < _TINY)
    if risky.any():
        idx = np.flatnonzero(risky)
        ss = s.ravel()[idx]
        pp = np.broadcast_to(p, s.shape).ravel()[idx]
        cc = np.broadcast_to(np.asarray(c, dtype=_F64), s.shape).ravel()[idx]
        bb = ss - pp
        err = (pp - (ss - bb)) + (cc - bb)          # ss + err == pp + cc exactly
        flat = r.reshape(-1)
        rr = flat[idx]
        other = np.nextafter(rr, np.where(ss > rr, np.inf, -np.inf).astype(_F32))
        tie = (ss == (rr.astype(_F64) + other.astype(_F64)) * 0.5) & (err != 0)
        flat[idx] = np.where(tie, np.where(err > 0, np.maximum(rr, other), np.minimum(rr, other)), rr)
    return r


# --------------------------------------------------------------------------- constants (libjxl)

_K_C2 = _F32(0.0009)
_NUM_SCALES = 6

# lib/jxl/cms/opsin_params.h: kOpsinAbsorbanceMatrix and kOpsinAbsorbanceBias, constexpr float
_M02, _M00 = _F32(0.078), _F32(0.30)
_M01 = _F32(1.0) - _M02 - _M00
_M12, _M10 = _F32(0.078), _F32(0.23)
_M11 = _F32(1.0) - _M12 - _M10
_M20, _M21 = _F32(0.24342268924547819), _F32(0.20476744424496821)
_M22 = _F32(1.0) - _M20 - _M21
_OPSIN = ((_M00, _M01, _M02), (_M10, _M11, _M12), (_M20, _M21, _M22))
_BIAS = _F32(0.0037930732552754493)
_NEG_BIAS_CBRT = _F32(-np.cbrt(_F64(_BIAS)))           # -cbrtf(bias), ComputePremulAbsorb

# lib/jxl/base/fast_math-inl.h, CubeRootAndAdd
_CBRT_EXP_BIAS = np.int32(0x54800000)
_CBRT_EXP_MUL = np.int32(0x002AAAAA)
_K1_3 = _F32(1.0) / _F32(3.0)
_K4_3 = _F32(4.0) / _F32(3.0)

# tools/ssimulacra2.cc, MakePositiveXYB
_X_MUL, _X_ADD, _Y_ADD, _B_ADD = _F32(14.0), _F32(0.42), _F32(0.01), _F32(0.55)

# tools/ssimulacra2.cc, Msssim::Score(): 108 weights, ordered channel (X, Y, B), scale (6),
# norm (1, 4), map (ssim, ringing, blurring).
_WEIGHTS = np.array([
    0.0, 0.0007376606707406586, 0.0, 0.0, 0.0007793481682867309, 0.0, 0.0, 0.0004371155730107379,
    0.0, 1.1041726426657346, 0.00066284834129271, 0.00015231632783718752, 0.0, 0.0016406437456599754,
    0.0, 1.8422455520539298, 11.441172603757666, 0.0, 0.0007989109436015163, 0.000176816438078653,
    0.0, 1.8787594979546387, 10.94906990605142, 0.0, 0.0007289346991508072, 0.9677937080626833,
    0.0, 0.00014003424285435884, 0.9981766977854967, 0.00031949755934435053, 0.0004550992113792063,
    0.0, 0.0, 0.0013648766163243398, 0.0, 0.0, 0.0, 0.0, 0.0, 7.466890328078848, 0.0,
    17.445833984131262, 0.0006235601634041466, 0.0, 0.0, 6.683678146179332, 0.00037724407979611296,
    1.027889937768264, 225.20515300849274, 0.0, 0.0, 19.213238186143016, 0.0011401524586618361,
    0.001237755635509985, 176.39317598450694, 0.0, 0.0, 24.43300999870476, 0.28520802612117757,
    0.0004485436923833408, 0.0, 0.0, 0.0, 34.77906344483772, 44.835625328877896, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0008680556573291698, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005313191874358747, 0.0,
    0.00016533814161379112, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004179171803251336, 0.0017290828234722833,
    0.0, 0.0020827005846636437, 0.0, 0.0, 8.826982764996862, 23.19243343998926, 0.0, 95.1080498811086,
    0.9863978034400682, 0.9834382792465353, 0.0012286405048278493, 171.2667255897307,
    0.9807858872435379, 0.0, 0.0, 0.0, 0.0005130064588990679, 0.0, 0.00010854057858411537,
], dtype=np.float64)
assert _WEIGHTS.shape == (108,)


def _srgb_to_linear_lut() -> np.ndarray:
    """Every 8-bit sample through LoadFloatRow (``u8 * (1.0f / 255)``) and TF_SRGB, in float32."""
    x = np.arange(256).astype(_F32) * (_F32(1.0) / _F32(255.0))
    p = [_F32(v) for v in (2.200248328e-04, 1.043637593e-02, 1.624820318e-01, 7.961564959e-01, 8.210152774e-01)]
    q = [_F32(v) for v in (2.631846970e-01, 1.076976492e+00, 4.987528350e-01, -5.512498495e-02, 6.521209011e-03)]
    yp = np.full_like(x, p[4])
    yq = np.full_like(x, q[4])
    for k in (3, 2, 1, 0):                                  # EvalRationalPolynomial: Horner with MulAdd
        yp = _fma(yp, x, p[k])
        yq = _fma(yq, x, q[k])
    poly = yp / yq
    linear = x * (_F32(1.0) / _F32(12.92))
    return np.where(x > _F32(0.04045), poly, linear).astype(_F32)


_SRGB_TO_LINEAR = _srgb_to_linear_lut()


# --------------------------------------------------------------------------- the recursive Gaussian

class RecursiveGaussian(NamedTuple):
    """``CreateRecursiveGaussian``'s float32 tables; index ``[k]`` is the k = 1, 3, 5 term."""

    radius: int
    n2: np.ndarray            # (3,)
    d1: np.ndarray            # (3,)
    mul_in: np.ndarray        # (3, 4)
    mul_prev: np.ndarray      # (3, 4)
    mul_prev2: np.ndarray     # (3, 4)


def _inv3x3(m: list[list[float]]) -> list[list[float]]:
    """lib/jxl/base/matrix_ops.h, Inv3x3Matrix, in double as there."""
    t = [[0.0] * 3 for _ in range(3)]
    t[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    t[0][1] = m[0][2] * m[2][1] - m[0][1] * m[2][2]
    t[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    t[1][0] = m[1][2] * m[2][0] - m[1][0] * m[2][2]
    t[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0]
    t[1][2] = m[0][2] * m[1][0] - m[0][0] * m[1][2]
    t[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    t[2][1] = m[0][1] * m[2][0] - m[0][0] * m[2][1]
    t[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det = m[0][0] * t[0][0] + m[0][1] * t[1][0] + m[0][2] * t[2][0]
    idet = 1.0 / det
    return [[t[j][i] * idet for i in range(3)] for j in range(3)]


def create_recursive_gaussian(sigma: float) -> RecursiveGaussian:
    """tools/gauss_blur.cc, CreateRecursiveGaussian: Charalampidis [2016], truncated cosines.

    The coefficients are computed in double and stored as float32, as there. Every float32
    value lies at least 0.03 ulp from a float32 rounding tie for sigma 1.5 (a test checks it), so
    the double rounding of this computation, or of a compiler that fuses it, can't change them.
    """
    radius = float(round(3.2795 * sigma + 0.2546))                 # (57), "N"
    pi_div_2r = 3.141592653589793238 / (2.0 * radius)
    omega = (pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r)
    p_1 = +1.0 / math.tan(0.5 * omega[0])                            # (37)
    p_3 = -1.0 / math.tan(0.5 * omega[1])
    p_5 = +1.0 / math.tan(0.5 * omega[2])
    r_1 = +p_1 * p_1 / math.sin(omega[0])                            # (44)
    r_3 = -p_3 * p_3 / math.sin(omega[1])
    r_5 = +p_5 * p_5 / math.sin(omega[2])
    neg_half_sigma2 = -0.5 * sigma * sigma                           # (50)
    recip_radius = 1.0 / radius
    rho = [math.exp(neg_half_sigma2 * w * w) * recip_radius for w in omega]
    d_13 = p_1 * r_3 - r_1 * p_3                                     # (52)
    d_35 = p_3 * r_5 - r_3 * p_5
    d_51 = p_5 * r_1 - r_5 * p_1
    recip_d13 = 1.0 / d_13
    zeta_15 = d_35 * recip_d13
    zeta_35 = d_51 * recip_d13
    a = _inv3x3([[p_1, p_3, p_5], [r_1, r_3, r_5], [zeta_15, zeta_35, 1.0]])   # (56)
    gamma = (1.0, radius * radius - sigma * sigma, zeta_15 * rho[0] + zeta_35 * rho[1] + rho[2])  # (55)
    beta = []
    for y in range(3):                                               # (53), Mul3x3Vector
        e = 0.0
        for x in range(3):
            e += a[y][x] * gamma[x]
        beta.append(e)
    n2, d1, mul_in, mul_prev, mul_prev2 = [], [], [], [], []
    for i in range(3):
        n = -beta[i] * math.cos(omega[i] * (radius + 1.0))           # (33)
        d = -2.0 * math.cos(omega[i])
        d_2 = d * d
        n2.append(n)
        d1.append(d)
        mul_prev.append((-d, d_2 - 1.0, -d_2 * d + 2.0 * d, d_2 * d_2 - 3.0 * d_2 + 1.0))
        mul_prev2.append((-1.0, d, -d_2 + 1.0, d_2 * d - 2.0 * d))
        mul_in.append((n, -d * n, d_2 * n - n, -d_2 * d * n + 2.0 * d * n))
    f = lambda v: np.array(v, dtype=_F32)  # noqa: E731
    return RecursiveGaussian(int(radius), f(n2), f(d1), f(mul_in), f(mul_prev), f(mul_prev2))


_RG = create_recursive_gaussian(1.5)
_LANES = 4                                           # JXL_GAUSS_MAX_LANES
# The unrolled step's ShiftLeftLanes<i>(mul_in): lane j holds mul_in[j - i], or 0 below i.
_SHIFTED_MUL_IN = np.zeros((4, 3, 4, 1), _F32)
for _i in range(4):
    for _j in range(_i, 4):
        _SHIFTED_MUL_IN[_i, :, _j, 0] = _RG.mul_in[:, _j - _i]
del _i, _j


def _gaussian_rows(x: np.ndarray) -> np.ndarray:
    """FastGaussian1D along axis 0 of ``x`` (length, lines), every line at once, in float32.

    Scalar steps with bounds checks up to the first aligned output (8), then the 4-lane unrolled
    form (each block of four outputs from the four input sums and the last two outputs), then
    scalar steps again for the remainder; zeros outside the line.
    """
    rg = _RG
    rad = rg.radius
    xs, m = x.shape
    out = np.empty_like(x)
    zero = np.zeros(m, _F32)
    prev = np.zeros((3, m), _F32)              # lane 0 (or the broadcast lane) of prev_1/3/5
    prev2 = np.zeros((3, m), _F32)
    n2 = rg.n2[:, None]
    mul_prev_0 = rg.mul_prev[:, 0][:, None]
    mul_prev2_0 = rg.mul_prev2[:, 0][:, None]

    def scalar_step(n: int) -> None:
        nonlocal prev, prev2
        left, right = n - rad - 1, n + rad - 1
        s = (x[left] if left >= 0 else zero) + (x[right] if right < xs else zero)
        o = s[None, :] * n2
        o = _fma(mul_prev2_0, prev2, o)
        prev2 = prev
        o = _fma(mul_prev_0, prev, o)
        prev = o
        if n >= 0:
            out[n] = o[0] + (o[1] + o[2])

    n = -rad + 1
    first_aligned = -(-(rad + 1) // _LANES) * _LANES   # RoundUpTo(N + 1, Lanes(d))
    while n < min(first_aligned, xs):
        scalar_step(n)
        n += 1
    sh = _SHIFTED_MUL_IN
    mul_prev = rg.mul_prev[:, :, None]
    mul_prev2 = rg.mul_prev2[:, :, None]
    while n < xs - rad + 1 - (_LANES - 1):
        s = x[n - rad - 1:n - rad + 3] + x[n + rad - 1:n + rad + 3]          # (4, lines)
        o = s[0] * sh[0]                                                   # (3, 4, lines)
        for i in (1, 2, 3):
            o = _fma(sh[i], s[i], o)
        o = _fma(mul_prev2, prev2[:, None, :], o)
        o = _fma(mul_prev, prev[:, None, :], o)
        prev2 = o[:, _LANES - 2].copy()
        prev = o[:, _LANES - 1].copy()
        out[n:n + _LANES] = o[0] + (o[1] + o[2])
        n += _LANES
    while n < xs:
        scalar_step(n)
        n += 1
    return out


def _gaussian_columns(x: np.ndarray) -> np.ndarray:
    """FastGaussianVertical along axis 0 of ``x`` (rows, columns): recursion (35) with FMAs."""
    rg = _RG
    rad = rg.radius
    ys, m = x.shape
    out = np.empty_like(x)
    zero = np.zeros(m, _F32)
    y_n1 = np.zeros((3, m), _F32)
    y_n2 = np.zeros((3, m), _F32)
    n2 = rg.n2[:, None]
    neg_d1 = -rg.d1[:, None]
    for n in range(-rad + 1, ys):
        top, bottom = n - rad - 1, n + rad - 1
        s = (x[top] if top >= 0 else zero) + (x[bottom] if bottom < ys else zero)
        y = _fma(n2, s[None, :], _fma(neg_d1, y_n1, -y_n2))   # MulAdd(n2, sum, NegMulSub(d1, y_n1, y_n2))
        y_n2, y_n1 = y_n1, y
        if n >= 0:
            out[n] = y[0] + (y[1] + y[2])
    return out


def blur(planes: np.ndarray) -> np.ndarray:
    """libjxl's FastGaussian (sigma 1.5) of each float32 plane: along rows, then down columns.

    ``planes`` is ``(H, W)`` or ``(P, H, W)``; the result is float32 in the same shape.
    """
    a = np.asarray(planes, dtype=_F32)
    single = a.ndim == 2
    if single:
        a = a[None]
    p, h, w = a.shape
    rows = np.ascontiguousarray(a.transpose(2, 0, 1).reshape(w, p * h))
    tmp = _gaussian_rows(rows).reshape(w, p, h).transpose(1, 2, 0)
    cols = np.ascontiguousarray(tmp.transpose(1, 0, 2).reshape(h, p * w))
    out = _gaussian_columns(cols).reshape(h, p, w).transpose(1, 0, 2)
    out = np.ascontiguousarray(out)
    return out[0] if single else out


# --------------------------------------------------------------------------- the port

def _as_rgb_u8(img: np.ndarray, what: str) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError(f"{what}: expected uint8 samples, got {a.dtype}")
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{what}: expected an (H, W) or (H, W, 3) array, got shape {a.shape}")
    return a


def _to_linear(u8: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` uint8 -> ``(3, H, W)`` float32 linear sRGB (TransformTo(LinearSRGB))."""
    return np.ascontiguousarray(np.moveaxis(_SRGB_TO_LINEAR[u8], -1, 0))


def _cube_root_and_add(x: np.ndarray, add: np.float32) -> np.ndarray:
    """CubeRootAndAdd: a bit-trick estimate of x^(-1/3), Newton steps, then cbrt(x) + add."""
    xa_3 = _K1_3 * x
    m1 = x.view(np.int32)
    m2 = np.where(m1 == 0, np.int32(0), _CBRT_EXP_BIAS - (m1 >> 23) * _CBRT_EXP_MUL).astype(np.int32)
    r = m2.view(_F32)
    for _ in range(3):
        r2 = r * r
        r = _fma(-xa_3, r2 * r2, _K4_3 * r)                 # NegMulAdd(xa_3, r2 * r2, k4_3 * r)
    r2 = r * r
    r = _fma(_K1_3, _fma(-x, r2 * r2, r), r)              # MulAdd(k1_3, NegMulAdd(xa, r2 * r2, r), r)
    r2 = r * r
    return _fma(r2, x, add)


def _to_positive_xyb(lin: np.ndarray) -> np.ndarray:
    """LinearRGBToXYB (OpsinAbsorbance, ZeroIfNegative, CubeRootAndAdd), then MakePositiveXYB."""
    r, g, b = lin
    mixed = []
    for row in _OPSIN:
        mx = _fma(row[0], r, _fma(row[1], g, _fma(row[2], b, _BIAS)))
        mixed.append(_cube_root_and_add(np.maximum(mx, _F32(0.0)), _NEG_BIAS_CBRT))
    half = _F32(0.5)
    x = half * (mixed[0] - mixed[1])
    y = half * (mixed[0] + mixed[1])
    bb = mixed[2]
    # rowB = (rowB - rowY) + 0.55f; rowX = rowX * 14.f + 0.42f (contracted to an FMA); rowY += 0.01f
    return np.stack([_fma(x, _X_MUL, _X_ADD), y + _Y_ADD, (bb - y) + _B_ADD])


def _downsample(lin: np.ndarray) -> np.ndarray:
    """Downsample(in, 2, 2) of ``(3, H, W)``: ceil sizes, the last row/column repeated, float32."""
    _, h, w = lin.shape
    if h % 2:
        lin = np.concatenate([lin, lin[:, -1:]], axis=1)
    if w % 2:
        lin = np.concatenate([lin, lin[:, :, -1:]], axis=2)
    total = ((lin[:, 0::2, 0::2] + lin[:, 0::2, 1::2]) + lin[:, 1::2, 0::2]) + lin[:, 1::2, 1::2]
    return total * _F32(0.25)


def _norms(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per plane of ``d`` (float64, (3, H, W)): the mean, and the 4-norm (mean of quartic)^(1/4)."""
    n = d.shape[1] * d.shape[2]
    q = d * d
    q = q * q
    one = 1.0 / n
    return one * d.sum(axis=(1, 2)), np.sqrt(np.sqrt(one * q.sum(axis=(1, 2))))


def _ssim_map(m1: np.ndarray, m2: np.ndarray, s11: np.ndarray, s22: np.ndarray, s12: np.ndarray) -> np.ndarray:
    mu11, mu22, mu12 = m1 * m1, m2 * m2, m1 * m2
    dm = m1 - m2
    num_m = (1.0 - (dm * dm).astype(_F64)).astype(_F32)
    num_s = _F32(2.0) * (s12 - mu12) + _K_C2
    denom_s = ((s11 - mu11) + (s22 - mu22)) + _K_C2
    d = np.maximum(1.0 - ((num_m * num_s) / denom_s).astype(_F64), 0.0)
    out = np.empty(6)
    out[0::2], out[1::2] = _norms(d)
    return out


def _edge_diff_map(i1: np.ndarray, m1: np.ndarray, i2: np.ndarray, m2: np.ndarray) -> np.ndarray:
    d1 = (1.0 + np.abs(i2 - m2).astype(_F64)) / (1.0 + np.abs(i1 - m1).astype(_F64)) - 1.0
    artifact = np.maximum(d1, 0.0)        # distorted has an edge where the original is smooth
    detail_lost = np.maximum(-d1, 0.0)    # the original has an edge where the distorted is smooth
    out = np.empty(12)
    out[0::4], out[1::4] = _norms(artifact)
    out[2::4], out[3::4] = _norms(detail_lost)
    return out


def _score(scales: list[tuple[np.ndarray, np.ndarray]]) -> float:
    ssim = 0.0
    i = 0
    for c in range(3):
        for avg_ssim, avg_edge in scales:
            for n in range(2):
                ssim += _WEIGHTS[i] * abs(avg_ssim[c * 2 + n]); i += 1
                ssim += _WEIGHTS[i] * abs(avg_edge[c * 4 + n]); i += 1
                ssim += _WEIGHTS[i] * abs(avg_edge[c * 4 + n + 2]); i += 1
    ssim *= 0.9562382616834844
    ssim = 2.326765642916932 * ssim - 0.020884521182843837 * ssim * ssim + 6.248496625763138e-05 * ssim * ssim * ssim
    if ssim > 0:
        return float(100.0 - 10.0 * ssim ** 0.6276336467831387)
    return 100.0


def ssimulacra2(ref: np.ndarray, dist: np.ndarray) -> float:
    """SSIMULACRA 2 of ``dist`` against ``ref`` (equal-size 8-bit sRGB arrays); 100 is identical.

    Images under 8 x 8 px raise ``ValueError``, as libjxl's tool refuses them: with no scale
    left to compare, the formula would otherwise score them 100.
    """
    a = _as_rgb_u8(ref, "ref")
    b = _as_rgb_u8(dist, "dist")
    if a.shape != b.shape:
        raise ValueError(f"ref and dist differ in size: {a.shape[:2]} vs {b.shape[:2]}")
    if a.shape[0] < 8 or a.shape[1] < 8:                # tools/ssimulacra2_main.cc refuses these too
        raise ValueError(f"too small to score: {a.shape[1]} x {a.shape[0]} px, the minimum is 8 x 8")
    lin1, lin2 = _to_linear(a), _to_linear(b)
    img1, img2 = _to_positive_xyb(lin1), _to_positive_xyb(lin2)
    scales: list[tuple[np.ndarray, np.ndarray]] = []
    for scale in range(_NUM_SCALES):
        if img1.shape[1] < 8 or img1.shape[2] < 8:      # checked on the previous scale's size, as libjxl does
            break
        if scale:
            lin1, lin2 = _downsample(lin1), _downsample(lin2)
            img1, img2 = _to_positive_xyb(lin1), _to_positive_xyb(lin2)
        blurred = blur(np.concatenate([img1 * img1, img2 * img2, img1 * img2, img1, img2]))
        sigma1_sq, sigma2_sq, sigma12, mu1, mu2 = blurred.reshape(5, 3, *img1.shape[1:])
        scales.append((_ssim_map(mu1, mu2, sigma1_sq, sigma2_sq, sigma12), _edge_diff_map(img1, mu1, img2, mu2)))
    return _score(scales)


# --------------------------------------------------------------------------- tiles

def _detail_map(img: np.ndarray) -> np.ndarray:
    """Per-pixel detail: |vertical| + |horizontal| difference of the channel mean, (H-1, W-1)."""
    a = np.asarray(img)
    g = a.astype(np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if g.shape[0] < 2 or g.shape[1] < 2:
        return np.zeros((max(g.shape[0] - 1, 0), max(g.shape[1] - 1, 0)))
    return np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]


def _positions(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile // 2)
    out = list(range(0, length - tile + 1, step))
    if out[-1] != length - tile:
        out.append(length - tile)                       # the far edge is always covered
    return out


def detail_tiles(img: np.ndarray, n: int = 4, size: int = 512) -> list[tuple[int, int]]:
    """Top-left corners ``(y, x)`` of the ``n`` most detailed, non-overlapping ``size`` px tiles.

    Candidates sit on a half-tile grid (plus the far edges). Detail is the mean absolute
    gradient of the channel mean inside the tile. The most detailed candidate is taken first,
    then the most detailed one that doesn't overlap any taken tile, and so on; ties go to the
    smaller ``y``, then the smaller ``x``. A tile is clipped to the image, so an image no larger
    than ``size`` in both directions is one tile at ``(0, 0)``. Fewer than ``n`` tiles come back
    when fewer fit without overlapping.
    """
    if n < 1 or size < 1:
        raise ValueError("n and size must be at least 1")
    a = np.asarray(img)
    h, w = a.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty image")
    th, tw = min(size, h), min(size, w)
    e = _detail_map(a)
    integral = np.zeros((e.shape[0] + 1, e.shape[1] + 1))
    if e.size:
        integral[1:, 1:] = e.cumsum(axis=0).cumsum(axis=1)
    cands: list[tuple[float, int, int]] = []
    for y in _positions(h, th):
        for x in _positions(w, tw):
            y1, x1 = min(y + th - 1, e.shape[0]), min(x + tw - 1, e.shape[1])
            area = max(1, (y1 - y) * (x1 - x))
            total = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            cands.append((total / area, y, x))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    chosen: list[tuple[int, int]] = []
    for _, y, x in cands:
        if all(abs(y - cy) >= th or abs(x - cx) >= tw for cy, cx in chosen):
            chosen.append((y, x))
            if len(chosen) == n:
                break
    return chosen


@dataclass
class TileScore:
    """The page's score (its worst tile) and every tile scored: ``(y, x, score)``."""

    score: float
    tiles: list[tuple[int, int, float]] = field(default_factory=list)


def worst_tile(ref: np.ndarray, dist: np.ndarray, n: int = 4, size: int = 512) -> TileScore:
    """Score the ``n`` most detailed ``size`` px tiles of ``ref`` (chosen on ``ref`` alone)."""
    a, b = np.asarray(ref), np.asarray(dist)
    if a.shape != b.shape:
        raise ValueError(f"ref and dist differ in size: {a.shape} vs {b.shape}")
    h, w = a.shape[:2]
    th, tw = min(size, h), min(size, w)
    tiles: list[tuple[int, int, float]] = []
    for y, x in detail_tiles(a, n, size):
        s = ssimulacra2(a[y:y + th, x:x + tw], b[y:y + th, x:x + tw])
        tiles.append((y, x, s))
    return TileScore(score=min(t[2] for t in tiles), tiles=tiles)


# --------------------------------------------------------------------------- inspection

def _common_mode(*images: PIL.Image.Image) -> str:
    return "L" if all(im.mode in ("1", "L", "LA", "I", "I;16", "F") for im in images) else "RGB"


def at_inspection(ref: PIL.Image.Image, dist: PIL.Image.Image, ref_ppi: float, dist_ppi: float,
                  inspect_ppi: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """Both images at the inspection resolution, as equal-size 8-bit arrays ready to score.

    The inspection resolution is ``inspect_ppi``, or the original's own resolution when that is
    lower: pixels are never invented. ``ref`` and ``dist`` must show the same area (their sizes
    divided by their ppi agree within 1%); both are resampled with Lanczos to the size ``ref``
    has at the inspection resolution. Grey stays grey when both are grey; otherwise both are RGB.
    """
    from PIL import Image

    if ref_ppi <= 0 or dist_ppi <= 0 or inspect_ppi <= 0:
        raise ValueError("ppi values must be positive")
    for axis in (0, 1):
        a_in = ref.size[axis] / ref_ppi
        b_in = dist.size[axis] / dist_ppi
        if abs(a_in - b_in) > 0.01 * max(a_in, b_in) + 1.0 / min(ref_ppi, dist_ppi):
            raise ValueError("ref and dist don't show the same area at the given ppi")
    target = min(float(inspect_ppi), float(ref_ppi))
    scale = target / float(ref_ppi)
    size = (max(1, round(ref.size[0] * scale)), max(1, round(ref.size[1] * scale)))
    mode = _common_mode(ref, dist)

    def prep(im: PIL.Image.Image) -> np.ndarray:
        if im.mode != mode:
            im = im.convert(mode)
        if im.size != size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)

    return prep(ref), prep(dist)
