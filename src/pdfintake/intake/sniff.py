"""What an input file is, decided from its bytes, never its name.

``sniff`` reads the first bytes of a file and names its kind. It reads only a regular file: the
file is opened without blocking and checked before a byte is read, so a FIFO or a device such as
``/dev/stdin`` is refused (``input_not_file``) and can never make a run wait.

``open_image`` is the only way pdfintake opens an image with Pillow: it passes exactly one format,
so a file can only ever be decoded by the plugin its bytes call for. That keeps Pillow's EPS
plugin, which shells out to Ghostscript, from ever loading: an EPS renamed ``scan.jpg`` sniffs as
``unknown`` and is refused, and even when it is opened as a JPEG on purpose, Pillow raises
``UnidentifiedImageError``.

``declared_depth`` reads the bits per sample a file's header declares (the index depth for a
palette image, the channel depth otherwise), so ``images.decode_exact`` can tell when Pillow's
decoded mode holds fewer bits than the file has.
"""

from __future__ import annotations

import os
import stat
import struct
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, get_args

from ..errors import LosslessUnavailable, UsageError

if TYPE_CHECKING:
    import PIL.Image

InputKind = Literal["pdf", "jpeg", "heif", "avif", "png", "tiff", "webp", "gif", "bmp", "unknown"]
INPUT_KINDS: tuple[str, ...] = get_args(InputKind)

# The one Pillow format each image kind may open as ("HEIF" is registered by pillow-heif).
PIL_FORMAT: Mapping[InputKind, str] = MappingProxyType({
    "jpeg": "JPEG",
    "heif": "HEIF",
    "avif": "AVIF",
    "png": "PNG",
    "tiff": "TIFF",
    "webp": "WEBP",
    "gif": "GIF",
    "bmp": "BMP",
})

# Decompression-bomb guard (DESIGN §9.3): Pillow's own limit is set to this, and any image above
# it is refused by name before a pixel is decoded.
MAX_IMAGE_PIXELS = 200_000_000

_HEAD = 4096
_PDF_WINDOW = 1024                      # readers accept a %PDF- header within the first 1024 bytes
_HEIF_MAJOR = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx"}
_HEIF_COMPAT = _HEIF_MAJOR | {b"mif1", b"msf1", b"mif2"}
_AVIF_BRANDS = {b"avif", b"avis"}
_BMP_DIB_SIZES = {12, 16, 40, 52, 56, 64, 108, 124}


def sniff(path: Path) -> InputKind:
    """The kind of ``path``, from its first bytes. Anything not recognised is ``"unknown"``.

    Only a regular file is read. ``path`` is opened without blocking and the open descriptor is
    checked before a byte is read, so a FIFO, a device or ``/dev/stdin`` (even one swapped in after
    the caller's own ``os.stat``) is refused with UsageError ``input_not_file`` instead of making
    the run wait for a writer: a run never waits (DESIGN §12, decided 2026-11-06). OSError when
    ``path`` can't be opened.
    """
    # Read-only; O_NONBLOCK so opening a FIFO never waits for a writer; O_NOCTTY so a terminal
    # can't become the controlling terminal; O_BINARY so Windows reads bytes untranslated (text
    # mode would turn the PNG signature's \r\n into \n). Each only where the platform has it. The
    # flags stay written out here so a reader (and a scan) can see the open can't write.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0)
                 | getattr(os, "O_BINARY", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UsageError("input_not_file", f"The input {path} isn't a regular file.", path=str(path))
        chunks: list[bytes] = []
        need = _HEAD
        while need > 0:
            chunk = os.read(fd, need)
            if not chunk:
                break
            chunks.append(chunk)
            need -= len(chunk)
    finally:
        os.close(fd)
    return _sniff_bytes(b"".join(chunks))


def _sniff_bytes(head: bytes) -> InputKind:
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:2] == b"BM" and len(head) >= 18 and struct.unpack_from("<I", head, 14)[0] in _BMP_DIB_SIZES:
        return "bmp"
    if head[4:8] == b"ftyp" and len(head) >= 12:
        kind = _bmff_kind(head)
        if kind is not None:
            return kind
    if _pdf_header(head):
        return "pdf"
    return "unknown"


def _pdf_header(head: bytes) -> bool:
    """A ``%PDF-`` header at byte 0, or within the first 1024 bytes after binary junk or at the
    start of a line (readers skip such a prefix), but never inside text: an EPS whose comment
    mentions ``%PDF-``, or notes that quote it mid-line, aren't PDFs."""
    if head.startswith(b"%PDF-"):
        return True
    if head.startswith((b"%!PS", b"\xc5\xd0\xd3\xc6")):
        return False
    window = head[:_PDF_WINDOW]
    at = window.find(b"%PDF-")
    while at > 0:
        before = window[at - 1]
        if before in b"\r\n" or before < 0x20 or before >= 0x7F:
            return True
        at = window.find(b"%PDF-", at + 1)
    return False


def _bmff_kind(head: bytes) -> InputKind | None:
    """HEIF or AVIF from an ISO BMFF ``ftyp`` box: the major brand first, then the compatible ones."""
    size = struct.unpack_from(">I", head, 0)[0]
    end = min(size if size >= 16 else len(head), len(head))
    major = head[8:12]
    compat = {head[k:k + 4] for k in range(16, end - 3, 4)}
    if major in _AVIF_BRANDS:
        return "avif"
    if major in _HEIF_MAJOR:
        return "heif"
    if compat & _AVIF_BRANDS:
        return "avif"
    if major in _HEIF_COMPAT or compat & _HEIF_COMPAT:
        return "heif"
    return None


def open_image(path: Path, kind: InputKind) -> "PIL.Image.Image":
    """Open ``path`` with Pillow as exactly ``PIL_FORMAT[kind]`` (lazy: pixels load on first use).

    UsageError for a kind that isn't an image (``pdf``, ``unknown``); PIL's UnidentifiedImageError
    when the bytes aren't that format; LosslessUnavailable above ``MAX_IMAGE_PIXELS``.
    """
    fmt = PIL_FORMAT.get(kind)
    if fmt is None:
        raise UsageError("not_an_image", f"{path}: this file isn't an image pdfintake reads ({kind}).",
                         file=str(path), kind=kind)
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    if kind == "heif":
        _register_heif()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(path, formats=[fmt])
    except Image.DecompressionBombError:
        raise _too_many_pixels(path, None) from None
    if img.width * img.height > MAX_IMAGE_PIXELS:
        size = img.size
        img.close()
        raise _too_many_pixels(path, size)
    return img


def _too_many_pixels(path: Path, size: tuple[int, int] | None) -> LosslessUnavailable:
    what = f"{size[0]}x{size[1]} pixels" if size else "too many pixels"
    return LosslessUnavailable(
        "too_many_pixels",
        f"{path}: the image has {what}; pdfintake refuses images above {MAX_IMAGE_PIXELS // 1_000_000} megapixels.",
        file=str(path), limit=MAX_IMAGE_PIXELS)


_HEIF_REGISTERED = False


def _register_heif() -> None:
    global _HEIF_REGISTERED
    if not _HEIF_REGISTERED:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _HEIF_REGISTERED = True


def declared_depth(path: Path, kind: InputKind) -> int:
    """Bits per sample as the file's header declares them.

    The index depth for a palette image (PNG colour type 3, a palette BMP or TIFF), the channel
    depth otherwise: PNG IHDR, TIFF BitsPerSample (largest value; 1 when absent), the JPEG frame's
    precision, BMP bit fields, HEIF/AVIF ``pixi`` (or ``hvcC``/``av1C``). GIF and WebP are 8.
    ValueError for a kind with no image header (``pdf``, ``unknown``) or a header it can't read.
    """
    if kind not in PIL_FORMAT:
        raise ValueError(f"{path}: a {kind} file has no image header")
    data = Path(path).read_bytes()
    try:
        if kind == "png":
            return _png_depth(data)
        if kind == "tiff":
            return _tiff_depth(data)
        if kind == "jpeg":
            from .jpeg import _frame

            return _frame(data)[1]
        if kind == "bmp":
            return _bmp_depth(data)
        if kind in ("heif", "avif"):
            return _bmff_depth(data)
    except (struct.error, IndexError) as exc:
        raise ValueError(f"{path}: unreadable {kind} header ({exc})") from None
    return 8                                             # gif, webp


def _png_depth(data: bytes) -> int:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("no PNG IHDR")
    return data[24]


def _tiff_depth(data: bytes) -> int:
    bo = {b"II": "<", b"MM": ">"}.get(data[:2])
    if bo is None:
        raise ValueError("no TIFF byte order")
    magic = struct.unpack_from(bo + "H", data, 2)[0]
    sizes = {3: 2, 4: 4, 16: 8}
    if magic == 42:
        ifd = struct.unpack_from(bo + "I", data, 4)[0]
        count = struct.unpack_from(bo + "H", data, ifd)[0]
        entry, head, inline, cfmt, ofmt = 12, 2, 4, "I", "I"
    elif magic == 43:
        ifd = struct.unpack_from(bo + "Q", data, 8)[0]
        count = struct.unpack_from(bo + "Q", data, ifd)[0]
        entry, head, inline, cfmt, ofmt = 20, 8, 8, "Q", "Q"
    else:
        raise ValueError("not a TIFF")
    for k in range(count):
        e = ifd + head + entry * k
        tag, typ = struct.unpack_from(bo + "HH", data, e)
        if tag != 258:
            continue
        n = struct.unpack_from(bo + cfmt, data, e + 4)[0]
        width = sizes.get(typ, 2)
        at = e + 4 + struct.calcsize(cfmt)
        if n * width > inline:
            at = struct.unpack_from(bo + ofmt, data, at)[0]
        code = {2: "H", 4: "I", 8: "Q"}[width]
        values = struct.unpack_from(bo + code * n, data, at)
        return max(values) if values else 1
    return 1                                             # the TIFF default for BitsPerSample


def _bmp_depth(data: bytes) -> int:
    dib = struct.unpack_from("<I", data, 14)[0]
    if dib == 12:                                        # OS/2 core header
        return min(struct.unpack_from("<H", data, 24)[0], 8)
    bits, compression = struct.unpack_from("<HI", data, 28)
    if bits <= 8:
        return bits                                      # palette indices
    if bits == 64:
        return 16
    if compression in (3, 6):                            # BI_BITFIELDS, BI_ALPHABITFIELDS
        n = 4 if (compression == 6 or dib >= 56) else 3
        at = 54 if dib >= 52 else 14 + dib
        masks = struct.unpack_from("<" + "I" * n, data, at)
        return max(bin(m).count("1") for m in masks) or 8
    return 5 if bits == 16 else 8                        # 16-bit BI_RGB is 5-5-5


def _bmff_depth(data: bytes) -> int:
    """The largest channel depth an ISO BMFF image's properties declare (pixi, else hvcC/av1C)."""
    pixi: list[int] = []
    codec: list[int] = []

    def walk(buf: bytes, start: int, end: int) -> None:
        i = start
        while i + 8 <= end:
            size, typ = struct.unpack_from(">I4s", buf, i)
            hdr = 8
            if size == 1:
                size = struct.unpack_from(">Q", buf, i + 8)[0]
                hdr = 16
            elif size == 0:
                size = end - i
            if size < hdr or i + size > end:
                return
            body = i + hdr
            if typ == b"meta":
                walk(buf, body + 4, i + size)            # a FullBox: version and flags first
            elif typ in (b"iprp", b"ipco"):
                walk(buf, body, i + size)
            elif typ == b"pixi" and size >= hdr + 5:
                n = buf[body + 4]
                pixi.extend(buf[body + 5:body + 5 + n])
            elif typ == b"hvcC" and size >= hdr + 18:
                codec.append((buf[body + 17] & 0x07) + 8)  # bitDepthLumaMinus8
            elif typ == b"av1C" and size >= hdr + 3:
                b = buf[body + 2]
                codec.append(12 if (b & 0x40 and b & 0x20) else 10 if b & 0x40 else 8)
            i += size

    walk(data, 0, len(data))
    if pixi:
        return max(pixi)
    return max(codec) if codec else 8


def webp_is_lossless(data: bytes) -> bool:
    """True when a WebP's image data is a VP8L (lossless) bitstream and none of it is VP8 (lossy).

    An animated WebP counts as lossless only when every frame is. ALPH (lossy-file alpha) alone
    doesn't make a file lossless.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return False
    found: set[bytes] = set()

    def walk(start: int, end: int) -> None:
        i = start
        while i + 8 <= end:
            fourcc = data[i:i + 4]
            size = struct.unpack_from("<I", data, i + 4)[0]
            body = i + 8
            if fourcc in (b"VP8L", b"VP8 "):
                found.add(fourcc)
            elif fourcc == b"ANMF" and size >= 16:
                walk(body + 16, min(body + size, end))
            i = body + size + (size & 1)

    walk(12, len(data))
    return b"VP8L" in found and b"VP8 " not in found
