"""intake/images.py: routes, IDAT passthrough, exact decodes, Flate payloads and the one encode."""

from __future__ import annotations

import io
import random
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageStat

from fixtures import build
from pdfintake.errors import LosslessUnavailable, UsageError
from pdfintake.intake import images as I
from pdfintake.intake import jpeg as J
from pdfintake.intake.images import (DecodedImage, ImagePayload, decode_exact, encode_jpeg, flate_payload,
                                     jpeg_payload, png_idat_payload, route_for, split_alpha)
from pdfintake.intake.sniff import sniff


# --------------------------------------------------------------------------- helpers

def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def save(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def pdf_decode(p: ImagePayload) -> bytes:
    """The payload's samples as a PDF reader gets them: qpdf decodes Flate and the predictor."""
    import pikepdf

    pdf = pikepdf.new()
    s = pikepdf.Stream(pdf, p.data)
    s.Filter = pikepdf.Name("/" + p.filter)
    if p.decode_parms:
        s.DecodeParms = pikepdf.Dictionary({"/" + k: v for k, v in p.decode_parms.items()})
    return s.read_bytes()


def samples(p: ImagePayload, channels: int) -> np.ndarray:
    """Decoded samples as an (h, w, channels) integer array, unpacking 1/2/4-bit rows."""
    raw = pdf_decode(p)
    if p.bpc == 16:
        return np.frombuffer(raw, ">u2").reshape(p.height, p.width, channels).astype(np.int64)
    if p.bpc == 8:
        return np.frombuffer(raw, np.uint8).reshape(p.height, p.width, channels).astype(np.int64)
    rowbytes = (p.width * channels * p.bpc + 7) // 8
    rows = np.frombuffer(raw, np.uint8).reshape(p.height, rowbytes)
    bits = np.unpackbits(rows, axis=1)[:, :p.width * channels * p.bpc]
    weights = 2 ** np.arange(p.bpc - 1, -1, -1)
    vals = (bits.reshape(p.height, p.width * channels, p.bpc) * weights).sum(axis=2)
    return vals.reshape(p.height, p.width, channels).astype(np.int64)


def as_array(img: Image.Image) -> np.ndarray:
    a = np.asarray(img).astype(np.int64)
    return a[:, :, None] if a.ndim == 2 else a


def fake_icc(space: bytes, size: int = 300) -> bytes:
    head = bytearray(128)
    struct.pack_into(">I", head, 0, size)
    head[12:16] = b"prtr"
    head[16:20] = space
    head[20:24] = b"Lab "
    head[36:40] = b"acsp"
    return bytes(head) + bytes((i * 11) % 256 for i in range(size - 128))


def tiff_by_hand(w: int, h: int, arr: np.ndarray, photometric: int, bps: int = 16) -> bytes:
    """An uncompressed big-endian TIFF of ``arr`` (h, w, spp) at ``bps`` bits per sample."""
    spp = arr.shape[2]
    data = arr.astype(">u2" if bps == 16 else "u1").tobytes()
    n = 10
    extra = 8 + 2 + 12 * n + 4
    inline = spp * 2 <= 4
    data_at = extra + (0 if inline else 2 * spp)
    out = b"MM\x00*" + struct.pack(">I", 8) + struct.pack(">H", n)
    out += struct.pack(">HHIHH", 256, 3, 1, w, 0) + struct.pack(">HHIHH", 257, 3, 1, h, 0)
    if inline:
        out += struct.pack(">HHIHH", 258, 3, spp, bps, bps if spp == 2 else 0)
    else:
        out += struct.pack(">HHII", 258, 3, spp, extra)
    out += struct.pack(">HHIHH", 259, 3, 1, 1, 0) + struct.pack(">HHIHH", 262, 3, 1, photometric, 0)
    out += struct.pack(">HHII", 273, 4, 1, data_at) + struct.pack(">HHIHH", 277, 3, 1, spp, 0)
    out += struct.pack(">HHIHH", 278, 3, 1, h, 0) + struct.pack(">HHII", 279, 4, 1, len(data))
    out += struct.pack(">HHIHH", 284, 3, 1, 1, 0) + b"\x00\x00\x00\x00"
    if not inline:
        out += struct.pack(">" + "H" * spp, *([bps] * spp))
    return out + data


def mean_diff(a: Image.Image, b: Image.Image) -> float:
    assert a.size == b.size, (a.size, b.size)
    return sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3


def marked(width: int, height: int) -> Image.Image:
    """Continuous tone with coloured corner blocks, so any wrong turn or flip shows."""
    img = build.photo_like(width, height)
    img.paste((220, 20, 20), (0, 0, width // 5, height // 6))
    img.paste((20, 20, 220), (width - width // 6, height - height // 5, width, height))
    return img


INVERSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
           5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE,
           8: Image.Transpose.ROTATE_270}


def stored(upright: Image.Image, orientation: int) -> Image.Image:
    """How a camera stores ``upright`` when it writes EXIF ``orientation``."""
    return upright.transpose(INVERSE[orientation]) if orientation in INVERSE else upright.copy()


# --------------------------------------------------------------------------- an arithmetic-coding JPEG encoder
# ITU-T T.81 Annex D (the QM coder, Table D.2) and F.1.4 (the statistical model), sequential, 8-bit,
# 1x1 sampling. It exists only to make SOF9 fixtures; the test checks libjpeg decodes them as meant.

QE = [(0x5A1D, 1, 1, 1), (0x2586, 14, 2, 0), (0x1114, 16, 3, 0), (0x080B, 18, 4, 0), (0x03D8, 20, 5, 0),
      (0x01DA, 23, 6, 0), (0x00E5, 25, 7, 0), (0x006F, 28, 8, 0), (0x0036, 30, 9, 0), (0x001A, 33, 10, 0),
      (0x000D, 35, 11, 0), (0x0006, 9, 12, 0), (0x0003, 10, 13, 0), (0x0001, 12, 13, 0), (0x5A7F, 15, 15, 1),
      (0x3F25, 36, 16, 0), (0x2CF2, 38, 17, 0), (0x207C, 39, 18, 0), (0x17B9, 40, 19, 0), (0x1182, 42, 20, 0),
      (0x0CEF, 43, 21, 0), (0x09A1, 45, 22, 0), (0x072F, 46, 23, 0), (0x055C, 48, 24, 0), (0x0406, 49, 25, 0),
      (0x0303, 51, 26, 0), (0x0240, 52, 27, 0), (0x01B1, 54, 28, 0), (0x0144, 56, 29, 0), (0x00F5, 57, 30, 0),
      (0x00B7, 59, 31, 0), (0x008A, 60, 32, 0), (0x0068, 62, 33, 0), (0x004E, 63, 34, 0), (0x003B, 32, 35, 0),
      (0x002C, 33, 9, 0), (0x5AE1, 37, 37, 1), (0x484C, 64, 38, 0), (0x3A0D, 65, 39, 0), (0x2EF1, 67, 40, 0),
      (0x261F, 68, 41, 0), (0x1F33, 69, 42, 0), (0x19A8, 70, 43, 0), (0x1518, 72, 44, 0), (0x1177, 73, 45, 0),
      (0x0E74, 74, 46, 0), (0x0BFB, 75, 47, 0), (0x09F8, 77, 48, 0), (0x0861, 78, 49, 0), (0x0706, 79, 50, 0),
      (0x05CD, 48, 51, 0), (0x04DE, 50, 52, 0), (0x040F, 50, 53, 0), (0x0363, 51, 54, 0), (0x02D4, 52, 55, 0),
      (0x025C, 53, 56, 0), (0x01F8, 54, 57, 0), (0x01A4, 55, 58, 0), (0x0160, 56, 59, 0), (0x0125, 57, 60, 0),
      (0x00F6, 58, 61, 0), (0x00CB, 59, 62, 0), (0x00AB, 61, 63, 0), (0x008F, 61, 32, 0), (0x5B12, 65, 65, 1),
      (0x4D04, 80, 66, 0), (0x412C, 81, 67, 0), (0x37D8, 82, 68, 0), (0x2FE8, 83, 69, 0), (0x293C, 84, 70, 0),
      (0x2379, 86, 71, 0), (0x1EDF, 87, 72, 0), (0x1AA9, 87, 73, 0), (0x174E, 72, 74, 0), (0x1424, 72, 75, 0),
      (0x119C, 74, 76, 0), (0x0F6B, 74, 77, 0), (0x0D51, 75, 78, 0), (0x0BB6, 77, 79, 0), (0x0A40, 77, 48, 0),
      (0x5832, 80, 81, 1), (0x4D1C, 88, 82, 0), (0x438E, 89, 83, 0), (0x3BDD, 90, 84, 0), (0x34EE, 91, 85, 0),
      (0x2EAE, 92, 86, 0), (0x299A, 93, 87, 0), (0x2516, 86, 71, 0), (0x5570, 88, 89, 1), (0x4CA9, 95, 90, 0),
      (0x44D9, 96, 91, 0), (0x3E22, 97, 92, 0), (0x3824, 99, 93, 0), (0x32B4, 99, 94, 0), (0x2E17, 93, 86, 0),
      (0x56A8, 95, 96, 1), (0x4F46, 101, 97, 0), (0x47E5, 102, 98, 0), (0x41CF, 103, 99, 0), (0x3C3D, 104, 100, 0),
      (0x375E, 99, 93, 0), (0x5231, 105, 102, 0), (0x4C0F, 106, 103, 0), (0x4639, 107, 104, 0), (0x415E, 103, 99, 0),
      (0x5627, 105, 106, 1), (0x50E7, 108, 107, 0), (0x4B85, 109, 103, 0), (0x5597, 110, 109, 0), (0x504F, 111, 107, 0),
      (0x5A10, 110, 111, 1), (0x5522, 112, 109, 0), (0x59EB, 112, 111, 1), (0x5A1D, 113, 113, 0)]


class _QMCoder:
    def __init__(self) -> None:
        self.c, self.a, self.sc, self.zc, self.ct, self.buffer = 0, 0x10000, 0, 0, 11, -1
        self.out = bytearray()

    def _zeros(self) -> None:
        self.out += b"\x00" * self.zc
        self.zc = 0

    def _byte(self, v: int) -> None:
        self.out.append(v)
        if v == 0xFF:
            self.out.append(0x00)

    def _release(self, temp: int) -> None:
        if temp > 0xFF:                                  # a carry into the stacked bytes
            if self.buffer >= 0:
                self._zeros()
                self._byte(self.buffer + 1)
            self.zc += self.sc
            self.sc = 0
            self.buffer = temp & 0xFF
        elif temp == 0xFF:
            self.sc += 1
        else:
            if self.buffer == 0:
                self.zc += 1
            elif self.buffer >= 0:
                self._zeros()
                self._byte(self.buffer)
            if self.sc:
                self._zeros()
                self.out += b"\xff\x00" * self.sc
                self.sc = 0
            self.buffer = temp & 0xFF

    def code(self, st: bytearray, i: int, bit: int) -> None:
        sv = st[i]
        qe, nlps, nmps, switch = QE[sv & 0x7F]
        self.a -= qe
        if bit != (sv >> 7):
            if self.a >= qe:
                self.c += self.a
                self.a = qe
            st[i] = (sv & 0x80) ^ (nlps | (switch << 7))
        else:
            if self.a >= 0x8000:
                return
            if self.a < qe:
                self.c += self.a
                self.a = qe
            st[i] = (sv & 0x80) ^ nmps
        while True:
            self.a <<= 1
            self.c <<= 1
            self.ct -= 1
            if self.ct == 0:
                self._release(self.c >> 19)
                self.c &= 0x7FFFF
                self.ct += 8
            if self.a >= 0x8000:
                return

    def finish(self) -> bytes:
        temp = (self.a - 1 + self.c) & 0xFFFF0000
        self.c = temp + 0x8000 if temp < self.c else temp
        self.c <<= self.ct
        if self.c & 0xF8000000:
            if self.buffer >= 0:
                self._zeros()
                self._byte(self.buffer + 1)
            self.zc += self.sc
            self.sc = 0
        else:
            self._release(0)                             # flush buffer and stacked 0xFF bytes
        if self.c & 0x7FFF800:
            self._zeros()
            self._byte((self.c >> 19) & 0xFF)
            if self.c & 0x7F800:
                self._byte((self.c >> 11) & 0xFF)
        return bytes(self.out)


def arith_scan(blocks: list[list[list[int]]]) -> bytes:
    """Entropy-coded data for ``blocks[component][block]`` = 64 quantised coefficients in zigzag order."""
    coder = _QMCoder()
    dc = [bytearray(64) for _ in range(2)]
    ac = [bytearray(256) for _ in range(2)]
    fixed = bytearray([113])
    last = [0] * len(blocks)
    ctx = [0] * len(blocks)
    for b in range(len(blocks[0])):
        for ci, comp in enumerate(blocks):
            blk, t = comp[b], (0 if ci == 0 else 1)
            st, s0 = dc[t], ctx[ci]
            diff = blk[0] - last[ci]
            if diff == 0:
                coder.code(st, s0, 0)
                ctx[ci] = 0
            else:
                last[ci] = blk[0]
                coder.code(st, s0, 1)
                coder.code(st, s0 + 1, 0 if diff > 0 else 1)
                at = s0 + (2 if diff > 0 else 3)
                ctx[ci] = 4 if diff > 0 else 8
                v = abs(diff)
                m = 0
                if v - 1:
                    m = 1
                    w = (v - 1) >> 1
                    while w:
                        m <<= 1
                        w >>= 1
                _dc_magnitude(coder, st, at, v)
                if m > 1:                                # U = 1: a large difference
                    ctx[ci] += 8
            st = ac[t]
            ke = 63
            while ke > 0 and blk[ke] == 0:
                ke -= 1
            k = 1
            while k <= ke:
                at = 3 * (k - 1)
                coder.code(st, at, 0)
                while blk[k] == 0:
                    coder.code(st, at + 1, 0)
                    at += 3
                    k += 1
                coder.code(st, at + 1, 1)
                coder.code(fixed, 0, 0 if blk[k] > 0 else 1)
                _ac_magnitude(coder, st, at + 2, 189 if k <= 5 else 217, abs(blk[k]))
                k += 1
            if k <= 63:
                coder.code(st, 3 * (k - 1), 1)
    return coder.finish()


def _dc_magnitude(coder: _QMCoder, st: bytearray, at: int, v: int) -> None:
    m, v = 0, v - 1
    if v:
        coder.code(st, at, 1)
        m, w, at = 1, v >> 1, 20
        while w:
            coder.code(st, at, 1)
            m, at, w = m << 1, at + 1, w >> 1
    coder.code(st, at, 0)
    at += 14
    m >>= 1
    while m:
        coder.code(st, at, 1 if (m & v) else 0)
        m >>= 1


def _ac_magnitude(coder: _QMCoder, st: bytearray, at: int, large: int, v: int) -> None:
    m, v = 0, v - 1
    if v:
        coder.code(st, at, 1)
        m, w = 1, v >> 1
        if w:
            coder.code(st, at, 1)
            m, at, w = m << 1, large, w >> 1
            while w:
                coder.code(st, at, 1)
                m, at, w = m << 1, at + 1, w >> 1
    coder.code(st, at, 0)
    at += 14
    m >>= 1
    while m:
        coder.code(st, at, 1 if (m & v) else 0)
        m >>= 1


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def arith_jpeg(blocks: list[list[list[int]]], bw: int, bh: int, *, q: int = 2) -> bytes:
    """A SOF9 (extended sequential, arithmetic) JPEG of bw x bh 8x8 blocks per component."""
    n = len(blocks)
    out = b"\xff\xd8"
    if n == 3:
        out += _segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    out += _segment(0xDB, b"\x00" + bytes([q] * 64)) + (_segment(0xDB, b"\x01" + bytes([q] * 64)) if n > 1 else b"")
    comps = b"".join(bytes([ci + 1, 0x11, 0 if ci == 0 else 1]) for ci in range(n))
    out += _segment(0xC9, struct.pack(">BHHB", 8, bh * 8, bw * 8, n) + comps)
    out += _segment(0xDA, bytes([n]) + b"".join(bytes([ci + 1, 0x00 if ci == 0 else 0x11]) for ci in range(n)) + b"\x00\x3f\x00")
    return out + arith_scan(blocks) + b"\xff\xd9"


def zigzag() -> list[int]:
    order: list[int] = []
    for s in range(15):
        cells = [(r, s - r) for r in range(max(0, s - 7), min(s, 7) + 1)]
        cells.sort(key=lambda rc: -rc[0] if s % 2 == 0 else rc[0])
        order += [r * 8 + c for r, c in cells]
    return order


def random_blocks(n: int, *, seed: int, dc: int = 60) -> list[list[int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        zz = [0] * 64
        zz[0] = rng.randint(-dc, dc)
        for k in range(1, 64):
            if rng.random() < (0.5 if k < 10 else 0.08):
                zz[k] = rng.choice([-1, 1]) * rng.choice([1, 1, 2, 3, 5, 9, 17, 40])
        out.append(zz)
    return out


def float_idct(zz: list[int], q: int) -> np.ndarray:
    nat = np.zeros(64)
    for k, pos in enumerate(zigzag()):
        nat[pos] = zz[k] * q
    c = nat.reshape(8, 8)
    n = np.arange(8)
    basis = np.where(n[:, None] == 0, np.sqrt(1 / 8), np.sqrt(2 / 8)) * np.cos((2 * n[None, :] + 1) * n[:, None] * np.pi / 16)
    return basis.T @ c @ basis + 128


# --------------------------------------------------------------------------- routes

def test_route_for_each_kind(tmp_path: Path):
    img = build.photo_like(32, 24)
    cases = {
        "passthrough": ("a.jpg", build.jpeg_bytes(img)),
        "png_idat": ("b.png", build.raw_png(8, 6, color_type=2, bit_depth=16)),
        "lossless": ("c.png", save(img.convert("RGBA"), "PNG")),
    }
    for want, (name, data) in cases.items():
        path = write(tmp_path, name, data)
        assert route_for(sniff(path), path) == want
    for name, data, want in (
        ("i.png", build.raw_png(8, 6, color_type=0, bit_depth=8, interlace=True), "lossless"),
        ("t.png", build.png_bytes(build.blank(8, 6, mode="P")), "png_idat"),
        ("x.tif", save(img, "TIFF"), "lossless"),
        ("x.gif", save(img.convert("P"), "GIF"), "lossless"),
        ("x.bmp", save(img, "BMP"), "lossless"),
        ("ll.webp", save(img, "WEBP", lossless=True), "lossless"),
        ("ly.webp", save(img, "WEBP", quality=80), "encoded_once"),
        ("arith.jpg", arith_jpeg([random_blocks(1, seed=1)], 1, 1), "lossless"),
    ):
        path = write(tmp_path, name, data)
        assert route_for(sniff(path), path) == want, name
    trns = Image.new("RGB", (4, 4), (1, 2, 3))
    path = write(tmp_path, "trns.png", save(trns, "PNG", transparency=(1, 2, 3)))
    assert route_for("png", path) == "lossless"                         # tRNS is alpha
    with pytest.raises(UsageError):
        route_for("pdf", write(tmp_path, "x.pdf", build.pdf_bytes(1)))


# --------------------------------------------------------------------------- PNG IDAT passthrough

def test_a_16_bit_rgb_png_goes_png_idat_with_samples_equal_to_the_source(tmp_path: Path):
    import pikepdf
    import pypdfium2 as pdfium

    rows = build.png_samples(50, 40, color_type=2, bit_depth=16, seed=1)
    data = build.raw_png(50, 40, color_type=2, bit_depth=16, rows=rows)
    path = write(tmp_path, "rgb16.png", data)
    assert route_for(sniff(path), path) == "png_idat"
    p = png_idat_payload(data)
    assert p is not None and p.bpc == 16 and p.colorspace == "DeviceRGB" and p.smask is None
    idat = b"".join(pl for t, pl in I._png_chunks(data) if t == b"IDAT")
    assert p.data == idat                                               # no decode, no re-encode
    assert p.decode_parms == {"Predictor": 15, "Colors": 3, "BitsPerComponent": 16, "Columns": 50}
    src = np.asarray(rows, dtype=np.int64).reshape(40, 50, 3)
    assert np.array_equal(samples(p, 3), src)
    with Image.open(path, formats=["PNG"]) as im:                       # what Pillow would have kept
        assert im.mode == "RGB"
    from pdfintake.intake.assemble import assemble

    out = tmp_path / "rgb16.pdf"
    assemble([path], out)
    with pikepdf.open(out) as pdf:
        im0 = pdf.pages[0].Resources.XObject["/Im0"]
        assert im0.read_raw_bytes() == idat and int(im0.BitsPerComponent) == 16
    render = pdfium.PdfDocument(str(out))[0].render(scale=50 / 842).to_numpy()[:, :, :3][:, :, ::-1]
    assert render.shape[:2] == (40, 50)
    assert int(np.abs(render.astype(np.int64) - (src >> 8)).max()) <= 1


@pytest.mark.parametrize("ct,depth", [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (2, 8), (3, 1), (3, 2), (3, 4), (3, 8)])
def test_png_idat_passes_every_depth_through_exactly(ct: int, depth: int):
    import pikepdf
    import pypdfium2 as pdfium

    w, h = 37, 11                                                        # rows that end mid-byte
    rows = build.png_samples(w, h, color_type=ct, bit_depth=depth, seed=depth)
    data = build.raw_png(w, h, color_type=ct, bit_depth=depth, rows=rows)
    p = png_idat_payload(data)
    assert p is not None and p.bpc == depth
    channels = 3 if ct == 2 else 1
    assert np.array_equal(samples(p, channels), np.asarray(rows, dtype=np.int64).reshape(h, w, channels))
    if ct == 3:
        plte = next(pl for t, pl in I._png_chunks(data) if t == b"PLTE")
        assert p.colorspace == "Indexed" and p.palette == plte
    # pdfium shows what Pillow shows
    pdf = pikepdf.new()
    from pdfintake.intake.assemble import _Writer

    page = pdf.add_blank_page(page_size=(w, h))
    page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=_Writer(pdf).image(p)))
    page.obj.Contents = pdf.make_stream(f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode())
    buf = io.BytesIO()
    pdf.save(buf)
    render = pdfium.PdfDocument(buf.getvalue())[0].render(scale=1).to_pil().convert("RGB")
    if depth == 16:                                                     # Pillow's I;16 -> RGB clips; pdfium scales
        ref = np.repeat((np.asarray(rows, dtype=np.int64).reshape(h, w, 1) >> 8), 3, axis=2)
    else:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as im:
            ref = np.asarray(im.convert("RGB"), np.int64)
    assert np.abs(np.asarray(render, np.int64) - ref).max() <= 1


def test_png_idat_payload_declines_what_it_cannot_pass_through():
    assert png_idat_payload(build.raw_png(8, 8, color_type=0, interlace=True)) is None
    assert png_idat_payload(build.raw_png(8, 8, color_type=6)) is None
    assert png_idat_payload(build.raw_png(8, 8, color_type=4)) is None
    assert png_idat_payload(save(Image.new("L", (4, 4)), "PNG", transparency=0)) is None
    good = build.raw_png(8, 8, color_type=2)
    at = good.index(b"IDAT")
    assert png_idat_payload(good[:at + 6] + bytes([good[at + 6] ^ 0xFF]) + good[at + 7:]) is None   # CRC
    short = build.raw_png(8, 4, color_type=2)
    lie = short.replace(struct.pack(">II", 8, 4), struct.pack(">II", 8, 5), 1)          # one row missing
    ihdr = lie.index(b"IHDR")
    body = lie[ihdr:ihdr + 17]
    lie = lie[:ihdr + 17] + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF) + lie[ihdr + 21:]
    assert png_idat_payload(lie) is None
    assert png_idat_payload(b"not a png") is None


def test_a_png_profile_is_used_only_when_its_space_matches(tmp_path: Path):
    def with_iccp(data: bytes, icc: bytes) -> bytes:
        body = b"example\x00\x00" + zlib.compress(icc)
        chunk = struct.pack(">I", len(body)) + b"iCCP" + body + struct.pack(">I", zlib.crc32(b"iCCP" + body) & 0xFFFFFFFF)
        return data[:33] + chunk + data[33:]

    rgb = with_iccp(build.raw_png(6, 4, color_type=2), fake_icc(b"RGB "))
    p = png_idat_payload(rgb)
    assert p is not None and p.colorspace == "ICCBased" and p.icc == fake_icc(b"RGB ")
    grey_wrong = with_iccp(build.raw_png(6, 4, color_type=0), fake_icc(b"RGB "))
    p = png_idat_payload(grey_wrong)
    assert p is not None and p.colorspace == "DeviceGray" and p.icc is None
    pal = with_iccp(build.raw_png(6, 4, color_type=3, bit_depth=8), fake_icc(b"RGB "))
    p = png_idat_payload(pal)
    assert p is not None and p.colorspace == "Indexed" and p.icc == fake_icc(b"RGB ")


# --------------------------------------------------------------------------- what can't be held exactly

def test_16_bit_rgba_png_and_16_bit_rgb_tiff_raise_naming_the_file(tmp_path: Path):
    rng = np.random.default_rng(2)
    cases = {
        "rgba16.png": build.raw_png(9, 7, color_type=6, bit_depth=16),
        "greyalpha16.png": build.raw_png(9, 7, color_type=4, bit_depth=16),
        "rgb16-interlaced.png": build.raw_png(9, 7, color_type=2, bit_depth=16, interlace=True),
        "rgb16.tif": tiff_by_hand(6, 4, rng.integers(0, 65536, (4, 6, 3)), photometric=2),
    }
    for name, data in cases.items():
        path = write(tmp_path, name, data)
        kind = sniff(path)
        assert route_for(kind, path) == "lossless"
        with pytest.raises(LosslessUnavailable) as err:
            decode_exact(path, kind)
        assert err.value.code == "bit_depth" and name in str(err.value) and err.value.detail["file"] == str(path)


def _with_frame(data: bytes, sof: int, precision: int | None = None) -> bytes:
    at = data.index(b"\xff\xc0")
    head = bytearray(data[at:at + 5])
    head[1] = sof
    if precision is not None:
        head[4] = precision
    return data[:at] + bytes(head) + data[at + 5:]


def test_12_bit_and_sof3_jpegs_raise_lossless_unavailable(tmp_path: Path):
    base = build.jpeg_bytes(build.photo_like(16, 16))
    for name, data, code in (("twelve.jpg", _with_frame(base, 0xC1, 12), "jpeg_precision"),
                             ("lossless.jpg", _with_frame(base, 0xC3), "jpeg_process")):
        path = write(tmp_path, name, data)
        assert route_for("jpeg", path) == "lossless"
        with pytest.raises(LosslessUnavailable) as err:
            decode_exact(path, "jpeg")
        assert err.value.code == code and name in str(err.value)


def test_more_than_one_frame_and_modes_a_pdf_cannot_take_raise(tmp_path: Path):
    frames = [build.photo_like(10, 8, seed=s).convert("P") for s in range(3)]
    gif = io.BytesIO()
    frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:])
    path = write(tmp_path, "anim.gif", gif.getvalue())
    with pytest.raises(LosslessUnavailable) as err:
        decode_exact(path, "gif")
    assert err.value.code == "multi_frame" and err.value.detail["frames"] == 3
    path = write(tmp_path, "float.tif", save(Image.new("F", (4, 4), 0.5), "TIFF"))
    with pytest.raises(LosslessUnavailable) as err:
        decode_exact(path, "tiff")
    assert err.value.code == "image_mode"
    path = write(tmp_path, "broken.png", build.raw_png(8, 8, color_type=2)[:60])
    with pytest.raises(LosslessUnavailable) as err:
        decode_exact(path, "png")
    assert err.value.code == "unreadable_image"
    with pytest.raises(UsageError):
        decode_exact(write(tmp_path, "x.pdf", build.pdf_bytes(1)), "pdf")


# --------------------------------------------------------------------------- exact decodes

def test_a_palette_png_keeps_its_exact_colours(tmp_path: Path):
    palette = [((i * 37) % 256, (i * 91) % 256, (i * 13) % 256) for i in range(16)]
    rows = build.png_samples(13, 9, color_type=3, bit_depth=4, seed=4, palette_size=16)
    expected = np.asarray([[palette[v] for v in r] for r in rows], dtype=np.int64)
    interlaced = build.raw_png(13, 9, color_type=3, bit_depth=4, rows=rows, palette=palette, interlace=True)
    path = write(tmp_path, "pal-interlaced.png", interlaced)
    assert route_for("png", path) == "lossless"
    p = flate_payload(decode_exact(path, "png"))
    assert p.colorspace == "DeviceRGB" and p.bpc == 8
    assert np.array_equal(samples(p, 3), expected)
    # with tRNS the palette's alpha comes along as an /SMask
    alpha = bytes((i * 17) % 256 for i in range(16))
    plain = build.raw_png(13, 9, color_type=3, bit_depth=4, rows=rows, palette=palette)
    trns = struct.pack(">I", len(alpha)) + b"tRNS" + alpha + struct.pack(">I", zlib.crc32(b"tRNS" + alpha) & 0xFFFFFFFF)
    at = plain.index(b"IDAT") - 4
    path = write(tmp_path, "pal-trns.png", plain[:at] + trns + plain[at:])
    assert route_for("png", path) == "lossless"
    p = flate_payload(decode_exact(path, "png"))
    assert np.array_equal(samples(p, 3), expected)
    assert np.array_equal(samples(p.smask, 1)[:, :, 0], np.asarray([[alpha[v] for v in r] for r in rows]))


def test_an_8_bit_rgba_png_gets_an_8_bit_smask_and_identical_colour_samples(tmp_path: Path):
    src = build.photo_like(31, 17)
    a = Image.linear_gradient("L").resize((31, 17))
    rgba = src.copy()
    rgba.putalpha(a)
    path = write(tmp_path, "rgba.png", save(rgba, "PNG"))
    dec = decode_exact(path, "png")
    assert dec.image.mode == "RGB" and dec.alpha is not None and dec.alpha.mode == "L"
    p = flate_payload(dec)
    assert p.colorspace == "DeviceRGB" and p.bpc == 8
    assert p.smask is not None and p.smask.bpc == 8 and p.smask.colorspace == "DeviceGray" and p.smask.smask is None
    assert np.array_equal(samples(p, 3), as_array(src))
    assert np.array_equal(samples(p.smask, 1), as_array(a))


def test_a_cmyk_tiff_with_a_profile_becomes_iccbased_n4_with_the_same_samples(tmp_path: Path):
    cmyk = build.photo_like(23, 19).convert("CMYK")
    icc = fake_icc(b"CMYK")
    path = write(tmp_path, "cmyk.tif", save(cmyk, "TIFF", icc_profile=icc, compression="tiff_lzw"))
    dec = decode_exact(path, "tiff")
    assert dec.icc == icc and dec.image.mode == "CMYK"
    p = flate_payload(dec)
    assert p.colorspace == "ICCBased" and p.icc == icc and p.bpc == 8
    assert np.array_equal(samples(p, 4), as_array(cmyk))
    from pdfintake.intake.assemble import assemble
    import pikepdf

    out = tmp_path / "cmyk.pdf"
    assemble([path], out)
    with pikepdf.open(out) as pdf:
        cs = pdf.pages[0].Resources.XObject["/Im0"].ColorSpace
        assert cs[0] == pikepdf.Name.ICCBased and int(cs[1].N) == 4 and cs[1].read_bytes() == icc
        assert cs[1].Alternate == pikepdf.Name.DeviceCMYK


@pytest.mark.parametrize("name,img,channels,bpc", [
    ("bilevel.tif", build.text_like(64, 40, mode="1"), 1, 1),
    ("grey16.tif", Image.fromarray((np.arange(64 * 40, dtype=np.uint16) * 17).reshape(40, 64)), 1, 16),
    ("grey8.bmp", build.photo_like(64, 40, mode="L"), 1, 8),
    ("rgb.bmp", build.photo_like(64, 40), 3, 8),
])
def test_lossless_decodes_keep_depth_and_samples(tmp_path: Path, name: str, img: Image.Image, channels: int, bpc: int):
    fmt = {"tif": "TIFF", "bmp": "BMP"}[name.rsplit(".", 1)[1]]
    kw = {"compression": "group4"} if bpc == 1 else {}
    path = write(tmp_path, name, save(img, fmt, **kw))
    p = flate_payload(decode_exact(path, sniff(path)))
    assert p.bpc == bpc
    got = samples(p, channels)
    want = as_array(img)
    if bpc == 1:
        want = (want > 0).astype(np.int64)
    assert np.array_equal(got, want)


def test_a_big_endian_16_bit_grey_tiff_stays_16_bit(tmp_path: Path):
    arr = np.random.default_rng(5).integers(0, 65536, (7, 11, 1))
    path = write(tmp_path, "g16be.tif", tiff_by_hand(11, 7, arr, photometric=1))
    dec = decode_exact(path, "tiff")
    assert dec.image.mode == "I;16B"
    p = flate_payload(dec)
    assert p.bpc == 16 and np.array_equal(samples(p, 1), arr)


def test_a_16_bit_grey_png_off_the_idat_route_stays_16_bit(tmp_path: Path):
    rows = build.png_samples(21, 13, color_type=0, bit_depth=16, seed=7)
    path = write(tmp_path, "g16i.png", build.raw_png(21, 13, color_type=0, bit_depth=16, rows=rows, interlace=True))
    dec = decode_exact(path, "png")
    assert dec.image.mode.startswith("I;16")
    p = flate_payload(dec)
    assert p.bpc == 16 and np.array_equal(samples(p, 1)[:, :, 0], np.asarray(rows))


def test_an_arithmetic_coded_8_bit_jpeg_decodes_exactly_to_flate(tmp_path: Path):
    bw, bh, q = 3, 2, 2
    blocks = random_blocks(bw * bh, seed=3)
    data = arith_jpeg([blocks], bw, bh, q=q)
    path = write(tmp_path, "arith.jpg", data)
    info = J.jpeg_info(data)
    assert info.sof == 0xC9 and not J.passthrough_ok(info)
    assert route_for(sniff(path), path) == "lossless"
    with Image.open(path, formats=["JPEG"]) as im:
        im.load()
        decoded = np.asarray(im).astype(np.int64)
    # the fixture is a real arithmetic JPEG: libjpeg decodes it to what its coefficients say
    want = np.zeros((bh * 8, bw * 8))
    for b, zz in enumerate(blocks):
        y, x = divmod(b, bw)
        want[y * 8:y * 8 + 8, x * 8:x * 8 + 8] = float_idct(zz, q)
    assert np.abs(decoded - np.clip(np.round(want), 0, 255)).max() <= 1
    p = flate_payload(decode_exact(path, "jpeg"))
    assert p.filter == "FlateDecode" and p.colorspace == "DeviceGray" and p.bpc == 8
    assert np.array_equal(samples(p, 1)[:, :, 0], decoded)
    # three components (YCbCr), interleaved
    colour = arith_jpeg([random_blocks(bw * bh, seed=s, dc=20) for s in (4, 5, 6)], bw, bh)
    path = write(tmp_path, "arith-rgb.jpg", colour)
    with Image.open(path, formats=["JPEG"]) as im:
        im.load()
        ref = np.asarray(im).astype(np.int64)
    p = flate_payload(decode_exact(path, "jpeg"))
    assert p.colorspace == "DeviceRGB" and np.array_equal(samples(p, 3), ref)


def test_an_mpo_decodes_its_primary_image(tmp_path: Path):
    from test_intake_jpeg import rich_jpeg

    gain = build.jpeg_bytes(build.photo_like(12, 8, seed=9))
    primary = build.photo_like(48, 32)
    path = write(tmp_path, "mpo.jpg", rich_jpeg(primary, mpf=True, trailing=gain))
    dec = decode_exact(path, "jpeg")
    assert dec.image.size == (48, 32) and dec.orientation == 6
    assert {J._kind(s) for s in dec.segments} >= {"exif", "xmp", "iptc", "com"}


def test_split_alpha_turns_a_transparent_colour_into_a_mask():
    rgb = Image.new("RGB", (4, 2), (10, 20, 30))
    rgb.putpixel((1, 0), (1, 2, 3))
    rgb.info["transparency"] = (1, 2, 3)
    colour, alpha = split_alpha(rgb)
    assert colour.mode == "RGB" and "transparency" not in colour.info
    assert np.array_equal(np.asarray(alpha), np.array([[255, 0, 255, 255], [255, 255, 255, 255]]))
    g16 = Image.fromarray(np.array([[5, 700], [700, 9]], dtype=np.uint16))
    g16.info["transparency"] = 700
    colour, alpha = split_alpha(g16)
    assert colour.mode == "I;16" and np.array_equal(np.asarray(colour), np.asarray(g16))
    assert np.array_equal(np.asarray(alpha), np.array([[255, 0], [0, 255]]))
    la = Image.merge("LA", (Image.new("L", (2, 2), 40), Image.new("L", (2, 2), 128)))
    colour, alpha = split_alpha(la)
    assert colour.mode == "L" and alpha.getpixel((0, 0)) == 128
    assert split_alpha(Image.new("RGB", (2, 2)))[1] is None


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_a_grey_png_trns_marks_the_right_pixels_at_every_depth(tmp_path: Path, depth: int):
    """Pillow 13 leaves a 2- and 4-bit grey PNG's tRNS value unscaled; decode_exact scales it."""
    top = 2 ** depth - 1
    rows = [[(x * 3 + y) % (top + 1) if depth < 16 else (x * 1500 + y) for x in range(9)] for y in range(4)]
    trns = rows[1][2]
    data = build.raw_png(9, 4, color_type=0, bit_depth=depth, rows=rows)
    t = struct.pack(">H", trns)
    chunk = struct.pack(">I", 2) + b"tRNS" + t + struct.pack(">I", zlib.crc32(b"tRNS" + t) & 0xFFFFFFFF)
    at = data.index(b"IDAT") - 4
    path = write(tmp_path, f"trns{depth}.png", data[:at] + chunk + data[at:])
    assert route_for("png", path) == "lossless"
    dec = decode_exact(path, "png")
    want = np.where(np.asarray(rows) == trns, 0, 255)
    assert dec.alpha is not None and np.array_equal(np.asarray(dec.alpha), want)
    p = flate_payload(dec)
    assert p.smask is not None and np.array_equal(samples(p.smask, 1)[:, :, 0], want)


def test_a_profile_that_does_not_fit_is_dropped(tmp_path: Path):
    grey = build.photo_like(8, 8, mode="L")
    path = write(tmp_path, "g.tif", save(grey, "TIFF", icc_profile=fake_icc(b"RGB ")))
    dec = decode_exact(path, "tiff")
    p = flate_payload(dec)
    assert dec.icc is not None and p.icc is None and p.colorspace == "DeviceGray"


# --------------------------------------------------------------------------- Flate details

def test_flate_keeps_the_smaller_of_plain_and_predictor_and_both_decode_exactly():
    smooth = build.photo_like(120, 80)
    dec = DecodedImage(smooth, None, None, 1, [], "tiff")
    p = flate_payload(dec)
    assert p.decode_parms is not None                                   # predictor wins on continuous tone
    noisy = flate_payload(DecodedImage(build.noise(120, 80), None, None, 1, [], "tiff"))
    assert noisy.decode_parms is None                                   # nothing to gain on noise
    for img, ch in ((smooth, 3), (smooth.convert("CMYK"), 4), (smooth.convert("L"), 1)):
        raw = img.tobytes()
        rowbytes = img.width * ch
        filtered = I._png_filter(raw, img.height, rowbytes, ch)
        pay = ImagePayload("FlateDecode", zlib.compress(filtered), img.width, img.height, "DeviceRGB", 8,
                           None, None, None, {"Predictor": 15, "Colors": ch, "BitsPerComponent": 8,
                                              "Columns": img.width}, None)
        assert pdf_decode(pay) == raw
    g16 = Image.fromarray((np.arange(90 * 30, dtype=np.uint16) * 23).reshape(30, 90))
    p16 = flate_payload(DecodedImage(g16, None, None, 1, [], "png"))
    assert np.array_equal(samples(p16, 1)[:, :, 0], np.asarray(g16))
    assert flate_payload(dec) == p                                       # deterministic


# --------------------------------------------------------------------------- JPEG payload

def test_jpeg_payload_keeps_bytes_and_inverts_adobe_cmyk():
    cmyk = save(build.photo_like(16, 16).convert("CMYK"), "JPEG")
    p = jpeg_payload(cmyk, J.jpeg_info(cmyk))
    assert p.data == cmyk and p.colorspace == "DeviceCMYK" and p.decode == (1.0, 0.0) * 4
    icc = fake_icc(b"RGB ")
    rgb = save(build.photo_like(16, 16), "JPEG", icc_profile=icc)
    p = jpeg_payload(rgb, J.jpeg_info(rgb))
    assert p.colorspace == "ICCBased" and p.icc == icc and p.decode is None and p.data == rgb
    grey = save(build.photo_like(16, 16, mode="L"), "JPEG", icc_profile=icc)
    p = jpeg_payload(grey, J.jpeg_info(grey))
    assert p.colorspace == "DeviceGray" and p.icc is None


# --------------------------------------------------------------------------- the one encode

def _boxes(data: bytes) -> set[bytes]:
    found: set[bytes] = set()

    def walk(start: int, end: int) -> None:
        i = start
        while i + 8 <= end:
            size, typ = struct.unpack_from(">I4s", data, i)
            if size < 8 or i + size > end:
                return
            found.add(typ)
            if typ == b"meta":
                walk(i + 12, i + size)
            elif typ in (b"iprp", b"ipco"):
                walk(i + 8, i + size)
            i += size

    walk(0, len(data))
    return found


def _exif(make: str = "ExampleCam", orientation: int = 6) -> bytes:
    ex = Image.Exif()
    ex[0x0112] = orientation
    ex[0x010F] = make
    return ex.tobytes()


def test_a_heic_with_irot_comes_out_upright_encoded_once_with_exif_carried(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    upright = marked(160, 96)
    try:
        data = save(stored(upright, 6), "HEIF", quality=95, exif=_exif())
    except Exception as exc:                                             # noqa: BLE001
        pytest.skip(f"pillow-heif can't encode HEIC here: {exc}")
    assert b"irot" in _boxes(data), "the fixture must carry its turn as irot"
    path = write(tmp_path, "photo.heic", data)
    assert sniff(path) == "heif" and route_for("heif", path) == "encoded_once"
    dec = decode_exact(path, "heif")
    assert dec.orientation == 1 and dec.image.size == upright.size   # pillow-heif applied irot
    jpeg = encode_jpeg(dec)
    info = J.jpeg_info(jpeg)
    assert (info.width, info.height, info.orientation) == (160, 96, 1)
    assert J.check_jpeg(jpeg, passthrough=False) == []
    with Image.open(io.BytesIO(jpeg), formats=["JPEG"]) as im:
        assert im.getexif()[0x010F] == "ExampleCam"
        assert mean_diff(im, upright) < 3.0


def test_a_10_bit_heic_takes_the_one_encode_instead_of_failing(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    arr = (np.random.default_rng(0).integers(0, 1024, (48, 64, 3)) << 6).astype(np.uint16)
    try:
        heif = pillow_heif.from_bytes(mode="RGB;16", size=(64, 48), data=arr.tobytes())
        buf = io.BytesIO()
        heif.save(buf, quality=90)
    except Exception as exc:                                             # noqa: BLE001
        pytest.skip(f"pillow-heif can't encode a 10-bit HEIC here: {exc}")
    path = write(tmp_path, "ten.heic", buf.getvalue())
    from pdfintake.intake.sniff import declared_depth

    assert declared_depth(path, "heif") == 10 and route_for("heif", path) == "encoded_once"
    jpeg = encode_jpeg(decode_exact(path, "heif"))
    assert J.jpeg_info(jpeg).precision == 8 and J.check_jpeg(jpeg, passthrough=False) == []


def test_an_avif_is_encoded_once_upright(tmp_path: Path):
    from PIL import features

    if not features.check("avif"):
        pytest.skip("this Pillow has no AVIF plugin")
    upright = marked(96, 64)
    try:
        data = save(stored(upright, 6), "AVIF", quality=90, exif=_exif())
    except Exception as exc:                                             # noqa: BLE001
        pytest.skip(f"no AVIF encoder here: {exc}")
    path = write(tmp_path, "photo.avif", data)
    assert sniff(path) == "avif" and route_for("avif", path) == "encoded_once"
    dec = decode_exact(path, "avif")
    jpeg = encode_jpeg(dec)
    info = J.jpeg_info(jpeg)
    assert (info.width, info.height, info.orientation) == (96, 64, 1)
    with Image.open(io.BytesIO(jpeg), formats=["JPEG"]) as im:
        assert im.getexif()[0x010F] == "ExampleCam"
        assert mean_diff(im, upright) < 4.0


def test_a_lossless_webp_goes_the_lossless_route(tmp_path: Path):
    src = build.photo_like(40, 30)
    path = write(tmp_path, "x.webp", save(src, "WEBP", lossless=True))
    assert route_for("webp", path) == "lossless"
    p = flate_payload(decode_exact(path, "webp"))
    assert np.array_equal(samples(p, 3), as_array(src))


def test_a_lossy_webp_is_encoded_once(tmp_path: Path):
    upright = marked(64, 48)
    path = write(tmp_path, "x.webp", save(stored(upright, 8), "WEBP", quality=90, exif=_exif(orientation=8)))
    assert route_for("webp", path) == "encoded_once"
    dec = decode_exact(path, "webp")
    assert dec.orientation == 8
    jpeg = encode_jpeg(dec)
    info = J.jpeg_info(jpeg)
    assert (info.width, info.height, info.orientation, info.sof) == (64, 48, 1, 0xC0)
    with Image.open(io.BytesIO(jpeg), formats=["JPEG"]) as im:
        assert mean_diff(im, upright) < 4.0
        assert im.getexif()[0x010F] == "ExampleCam"


def test_encode_jpeg_splices_iptc_and_writes_the_profile():
    icc = fake_icc(b"RGB ")
    iptc = _segment(0xED, b"Photoshop 3.0\x008BIM\x04\x04" + bytes(10))
    exif = _segment(0xE1, _exif())                                      # Exif.tobytes() carries the Exif\0\0 id
    dec = DecodedImage(build.photo_like(32, 16), None, icc, 1, [exif, iptc], "webp")
    jpeg = encode_jpeg(dec)
    kinds = [J._kind(s) for s in J.split_jpeg(jpeg)[0]]
    assert kinds[:3] == ["jfif", "exif", "iptc"] and "icc" in kinds
    assert J.jpeg_info(jpeg).icc == icc
    assert encode_jpeg(dec) == jpeg                                      # deterministic


def test_metadata_format_limit_is_a_record_warning_code():
    try:
        from pdfintake import record
    except ImportError:
        pytest.skip("record.py (U2) isn't built yet")
    assert I.METADATA_FORMAT_LIMIT in record.WARNING_CODES


# --------------------------------------------------------------------------- review fixes

def test_an_animated_png_is_refused_rather_than_cut_to_its_first_frame(tmp_path: Path):
    frames = [Image.new("RGB", (24, 16), c) for c in ((200, 0, 0), (0, 200, 0), (0, 0, 200))]
    data = save(frames[0], "PNG", save_all=True, append_images=frames[1:])
    assert b"acTL" in data and b"tRNS" not in data
    path = write(tmp_path, "anim.png", data)
    assert png_idat_payload(data) is None
    assert route_for("png", path) == "lossless"
    with pytest.raises(LosslessUnavailable) as err:
        decode_exact(path, "png")
    assert err.value.code == "multi_frame" and str(path) in str(err.value)


def _tiff_with_extra_tags(extra: list[tuple[int, int, int, int]], spp: int = 3) -> bytes:
    """An 8x6 RGB(A) uncompressed little-endian TIFF whose IFD0 also carries ``extra`` (tag, type, count, value)."""
    w, h = 8, 6
    pixels = bytes(i % 256 for i in range(w * h * spp))
    entries = sorted([(256, 3, 1, w), (257, 3, 1, h), (258, 3, spp, 0), (259, 3, 1, 1), (262, 3, 1, 2),
                      (273, 4, 1, 0), (277, 3, 1, spp), (278, 3, 1, h), (279, 4, 1, len(pixels))] + extra)
    ifd_size = 2 + 12 * len(entries) + 4
    bps_at = 8 + ifd_size
    data_at = bps_at + 2 * spp
    out = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(entries))
    for tag, typ, cnt, val in entries:
        if tag == 258:
            val = bps_at
        elif tag == 273:
            val = data_at
        if typ == 3 and cnt == 1:
            out += struct.pack("<HHIHH", tag, typ, cnt, val, 0)
        else:
            out += struct.pack("<HHII", tag, typ, cnt, val)
    return out + b"\x00\x00\x00\x00" + struct.pack("<" + "H" * spp, *([8] * spp)) + pixels


@pytest.mark.parametrize("extra,code", [
    ([(254, 4, 1, 1)], "sub_images"),                   # NewSubfileType: a reduced-resolution IFD0
    ([(330, 4, 1, 4096)], "sub_images"),                # SubIFDs, as in DNG
    ([(338, 3, 1, 1)], "associated_alpha"),             # premultiplied alpha
], ids=["reduced", "subifds", "premultiplied"])
def test_a_tiff_whose_first_image_may_not_be_the_picture_is_refused(tmp_path: Path, extra, code):
    data = _tiff_with_extra_tags(extra, spp=4 if code == "associated_alpha" else 3)
    path = write(tmp_path, "cam.tif", data)
    with pytest.raises(LosslessUnavailable) as err:
        decode_exact(path, "tiff")
    assert err.value.code == code and str(path) in str(err.value)
    plain = write(tmp_path, "plain.tif", _tiff_with_extra_tags([(338, 3, 1, 2)], spp=4))   # unassociated alpha
    dec = decode_exact(plain, "tiff")
    assert dec.image.size == (8, 6) and dec.alpha is not None


def _tiff_from_path(tmp_path: Path, img: Image.Image, orientation: int) -> Path:
    path = tmp_path / f"{img.mode.replace(';', '_')}-o{orientation}.tif"
    img.save(path, format="TIFF", tiffinfo={274: orientation})
    return path


@pytest.mark.parametrize("orientation", [2, 3, 5, 6, 7, 8])
@pytest.mark.parametrize("mode", ["L", "I;16", "P", "RGB", "RGBA", "CMYK", "LA", "1"])
def test_an_uncompressed_tiff_decodes_to_its_stored_pixels_whatever_its_orientation(tmp_path: Path, mode, orientation):
    # Pillow 13 turns a TIFF itself on load, and for an uncompressed file opened by path it scrambles
    # the pixels at orientations 5-8. decode_exact must hand back the stored pixels and the tag.
    rng = np.random.default_rng(orientation)
    if mode == "I;16":
        base = Image.fromarray(rng.integers(0, 65535, (30, 44), dtype=np.uint16))
    elif mode == "1":
        base = Image.fromarray(rng.integers(0, 2, (30, 44)).astype(bool))
    else:
        base = Image.fromarray(rng.integers(0, 255, (30, 44, 4), dtype=np.uint8), "RGBA").convert(mode)
        if mode == "P":
            base = Image.fromarray(rng.integers(0, 255, (30, 44, 3), dtype=np.uint8)).convert("P")
    path = _tiff_from_path(tmp_path, base, orientation)
    dec = decode_exact(path, "tiff")
    assert dec.orientation == orientation and dec.image.size == base.size
    got = np.asarray(dec.image.convert("RGB") if mode == "P" else dec.image)
    want = np.asarray(base.convert("RGB") if mode == "P" else base)
    if dec.alpha is not None:
        want_alpha = np.asarray(base.getchannel("A"))
        assert np.array_equal(np.asarray(dec.alpha), want_alpha)
        want = np.asarray(base.convert("RGB" if mode == "RGBA" else "L"))
    assert np.array_equal(got, want)


def test_a_16_bit_grey_heic_is_scaled_to_8_bits_not_clipped(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    ramp = np.tile(np.linspace(0, 65535, 64), (32, 1)).astype(np.uint16)
    try:
        data = save(Image.fromarray(ramp), "HEIF", quality=-1)
    except Exception as exc:                                             # noqa: BLE001
        pytest.skip(f"pillow-heif can't encode a 16-bit grey HEIC here: {exc}")
    path = write(tmp_path, "grey.heic", data)
    dec = decode_exact(path, "heif")
    if not dec.image.mode.startswith("I;16"):
        pytest.skip(f"pillow-heif opened it as {dec.image.mode}, not I;16")
    with Image.open(io.BytesIO(encode_jpeg(dec)), formats=["JPEG"]) as im:
        row = np.asarray(im.convert("L"), np.int64)[16]
    want = np.round(np.linspace(0, 255, 64))
    assert np.abs(row - want).max() <= 12 and row[:8].mean() < 40 and row[-8:].mean() > 215


def test_sixteen_to_eight_rounds_the_full_range():
    img = Image.fromarray(np.array([[0, 128, 257, 32767, 65535]], dtype=np.uint16))
    assert np.asarray(I._sixteen_to_eight(img)).tolist() == [[0, 0, 1, 127, 255]]


def test_a_heic_whose_colour_lives_only_in_nclx_warns(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    img = build.photo_like(32, 32)
    try:
        p3 = save(img, "HEIF", quality=90, nclx_profile={"color_primaries": 12, "transfer_characteristics": 13,
                                                          "matrix_coefficients": 6, "full_range_flag": 1})
        srgb = save(img, "HEIF", quality=90)
    except Exception as exc:                                             # noqa: BLE001
        pytest.skip(f"pillow-heif can't write an nclx box here: {exc}")
    dec = decode_exact(write(tmp_path, "p3.heic", p3), "heif")
    assert dec.icc is None and I.METADATA_FORMAT_LIMIT in dec.warnings
    assert I.METADATA_FORMAT_LIMIT not in decode_exact(write(tmp_path, "srgb.heic", srgb), "heif").warnings
    assert I._nclx_is_srgb(None) and I._nclx_is_srgb({"color_primaries": 1, "transfer_characteristics": 13})
    assert not I._nclx_is_srgb({"color_primaries": 9, "transfer_characteristics": 16})
