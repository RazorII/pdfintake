"""How each image becomes a PDF image stream without losing anything.

Four routes, chosen by ``route_for`` from what the file is:

``passthrough``
    A JPEG a PDF's DCTDecode reads (SOF C0/C1/C2, 8-bit, 1/3/4 components). Its bytes go into
    the PDF untouched, trailing MPF data and camera Orientation tag included.
``png_idat``
    A non-interlaced PNG without alpha or tRNS, colour type 0, 2 or 3 at any bit depth. Its IDAT
    data is the stream (FlateDecode with Predictor 15), with nothing decoded at all.
``lossless``
    Everything else that can be held exactly: decoded by Pillow, checked against the depth the
    header declares, and stored as Flate samples. Alpha becomes an /SMask; depth and colour space
    are kept. An input Pillow can't hold exactly fails with ``LosslessUnavailable``.
``encoded_once``
    HEIC, AVIF and lossy WebP, which a PDF can't carry and which are lossy already: one Pillow
    JPEG encode at q94, 4:2:0, with the ICC profile and the source's metadata carried over.

Orientation is never applied to pixels on the first three routes: ``assemble`` places the
stored image through the page matrix. (Pillow 13 turns a TIFF's pixels itself on load;
``decode_exact`` takes the tag away first, so it can't.) The one encode turns its pixels while
it encodes anyway, so its JPEG says Orientation 1 and is placed upright. Every route reads the
orientation by one rule, ``jpeg.exif_orientation``: EXIF IFD0 Orientation, SHORT or LONG, and
never XMP, so the same picture is turned the same way whichever route it takes.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from ..errors import LosslessUnavailable, PdfIntakeError, UsageError
from . import jpeg as _jpeg
from .jpeg import JpegInfo, carried_segments, exif_orientation, jpeg_info, passthrough_ok, settle_metadata
from .sniff import InputKind, declared_depth, open_image, webp_is_lossless

if TYPE_CHECKING:
    import PIL.Image

Route = Literal["passthrough", "png_idat", "lossless", "encoded_once"]
ROUTES: tuple[str, ...] = get_args(Route)

JPEG_QUALITY = 94
# zlib level for image streams. Measured on a 24 MP continuous-tone RGB image: predictor stream
# 23.2 MB in 6.3 s at level 6 against 22.2 MB in 20.4 s at level 9; pdf/shrink may recompress later.
FLATE_LEVEL = 6
METADATA_FORMAT_LIMIT = "metadata_format_limit"     # record.WARNING_CODES

_DEVICE = {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}
_ICC_SPACES = {b"GRAY": 1, b"RGB ": 3, b"CMYK": 4}
# The bits per sample each Pillow mode holds exactly; any other mode can't be placed losslessly.
_MODE_DEPTH = {"1": 1, "L": 8, "P": 8, "LA": 8, "PA": 8, "RGB": 8, "RGBA": 8, "CMYK": 8,
               "I;16": 16, "I;16B": 16, "I;16L": 16, "I;16N": 16}
_ENCODE_ONLY = frozenset({"heif", "avif"})
_PNG_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8)}
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1}
# Chunks that keep a PNG off the IDAT route: tRNS (alpha) and acTL (an APNG, whose other frames
# the IDAT data doesn't hold; decode_exact refuses more than one frame by name).
_NOT_IDAT = frozenset({b"tRNS", b"acTL"})
# EXIF orientation -> the Pillow transpose that turns stored pixels upright (as ImageOps.exif_transpose)
_UPRIGHT = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM", 5: "TRANSPOSE",
            6: "ROTATE_270", 7: "TRANSVERSE", 8: "ROTATE_90"}


@dataclass
class DecodedImage:
    image: "PIL.Image.Image"            # colour samples only, as stored (not turned)
    alpha: "PIL.Image.Image | None"     # "L" (or "1") alpha at the source's depth
    icc: bytes | None
    orientation: int                    # EXIF orientation to place it with, 1..8
    segments: list[bytes]               # metadata segments an encode carries over (jpeg.carried_segments shape)
    source_kind: InputKind
    # Additive to DESIGN §2.2: warning codes the one encode inherits (a metadata block too big for a
    # JPEG segment, a HEIF colour space only an nclx box declares). ``assemble`` reports them on the
    # encoded_once route only: the lossless routes carry no EXIF or XMP into the PDF at all.
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImagePayload:
    filter: Literal["DCTDecode", "FlateDecode"]
    data: bytes
    width: int
    height: int
    colorspace: Literal["DeviceGray", "DeviceRGB", "DeviceCMYK", "ICCBased", "Indexed"]
    bpc: Literal[1, 2, 4, 8, 16]
    icc: bytes | None                   # ICCBased profile (for Indexed, the base's profile)
    palette: bytes | None               # Indexed lookup: RGB triples
    decode: tuple[float, ...] | None
    decode_parms: dict[str, int] | None
    smask: "ImagePayload | None"


# --------------------------------------------------------------------------- routes

def route_for(kind: InputKind, path: Path) -> Route:
    """Which route ``path`` (already sniffed as ``kind``) takes. UsageError for a non-image kind."""
    path = Path(path)
    if kind == "jpeg":
        try:
            info = jpeg_info(path.read_bytes())
        except ValueError:
            return "lossless"                            # decode_exact names what's wrong with it
        return "passthrough" if passthrough_ok(info) else "lossless"
    if kind == "png":
        return "png_idat" if _png_idat_candidate(path.read_bytes()) else "lossless"
    if kind in _ENCODE_ONLY:
        return "encoded_once"
    if kind == "webp":
        return "lossless" if webp_is_lossless(path.read_bytes()) else "encoded_once"
    if kind in ("tiff", "gif", "bmp"):
        return "lossless"
    raise UsageError("not_an_image", f"{path}: this file isn't an image pdfintake reads ({kind}).",
                     file=str(path), kind=kind)


def _icc_components(icc: bytes | None) -> int | None:
    """The component count an ICC profile's colour space has (1, 3, 4), None for anything else."""
    if not icc or len(icc) < 128 or icc[36:40] != b"acsp":
        return None
    return _ICC_SPACES.get(bytes(icc[16:20]))


def _colorspace(n: int, icc: bytes | None) -> tuple[str, bytes | None]:
    if icc is not None and _icc_components(icc) == n:
        return "ICCBased", icc
    return _DEVICE[n], None


def jpeg_payload(data: bytes, info: JpegInfo) -> ImagePayload:
    """The JPEG's bytes untouched as a DCTDecode image. Adobe CMYK/YCCK (stored inverted) gets /Decode."""
    cs, icc = _colorspace(info.components, info.icc)
    decode = (1.0, 0.0) * 4 if info.components == 4 and info.adobe_transform is not None else None
    return ImagePayload(filter="DCTDecode", data=data, width=info.width, height=info.height,
                        colorspace=cs, bpc=8, icc=icc, palette=None, decode=decode,
                        decode_parms=None, smask=None)


# --------------------------------------------------------------------------- PNG IDAT passthrough

def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """[(type, payload)] through IEND, CRCs checked. ValueError for anything malformed."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    i, n, out = 8, len(data), []
    while True:
        if i + 12 > n:
            raise ValueError("PNG cut off before IEND")
        length, typ = struct.unpack_from(">I4s", data, i)
        end = i + 12 + length
        if end > n:
            raise ValueError(f"PNG chunk {typ!r} runs past the end")
        payload = data[i + 8:i + 8 + length]
        crc = struct.unpack_from(">I", data, i + 8 + length)[0]
        if zlib.crc32(typ + payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"PNG chunk {typ!r} has a bad CRC")
        out.append((typ, payload))
        i = end
        if typ == b"IEND":
            return out


def _png_ihdr(chunks: list[tuple[bytes, bytes]]) -> tuple[int, int, int, int, int, int, int]:
    if not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise ValueError("PNG doesn't start with IHDR")
    return struct.unpack(">IIBBBBB", chunks[0][1])


def _png_idat_candidate(data: bytes) -> bool:
    try:
        chunks = _png_chunks(data)
        w, h, depth, ct, comp, filt, inter = _png_ihdr(chunks)
    except ValueError:
        return False
    return (ct in _PNG_DEPTHS and depth in _PNG_DEPTHS[ct] and not inter and not comp and not filt
            and not any(t in _NOT_IDAT for t, _ in chunks))


def _png_icc(chunks: list[tuple[bytes, bytes]]) -> bytes | None:
    for typ, payload in chunks:
        if typ == b"iCCP":
            nul = payload.find(b"\x00")
            if nul < 0 or nul + 2 > len(payload) or payload[nul + 1] != 0:
                return None
            try:
                return zlib.decompress(payload[nul + 2:])
            except zlib.error:
                return None
    return None


def _png_orientation(chunks: list[tuple[bytes, bytes]]) -> int:
    for typ, payload in chunks:
        if typ == b"eXIf":
            return exif_orientation(payload)
    return 1


def _idat_ok(idat: bytes, rows: int, stride: int) -> bool:
    """The zlib stream inflates to exactly ``rows`` rows of ``stride`` bytes with valid filter types."""
    expected = rows * stride
    d = zlib.decompressobj()
    total, pending = 0, idat
    try:
        while True:
            out = d.decompress(pending, 1 << 20)
            pending = d.unconsumed_tail
            if out:
                first = (-total) % stride
                tags = np.frombuffer(out, np.uint8)[first::stride]
                if tags.size and int(tags.max()) > 4:
                    return False
                total += len(out)
                if total > expected:
                    return False
            if d.eof:
                break
            if not out and not pending:
                return False                             # truncated stream
    except zlib.error:
        return False
    return total == expected


def png_idat_payload(data: bytes) -> ImagePayload | None:
    """The PNG's IDAT data as the image stream, for colour types 0, 2 and 3 at any bit depth.

    FlateDecode with Predictor 15, Colors, BitsPerComponent and Columns from IHDR; PLTE becomes an
    Indexed colour space. Nothing is decoded; the zlib stream is only checked to inflate to the
    right number of rows. None when the PNG is interlaced, has alpha or tRNS, is animated (acTL),
    or is malformed.
    """
    try:
        chunks = _png_chunks(data)
        w, h, depth, ct, comp, filt, inter = _png_ihdr(chunks)
    except ValueError:
        return None
    if ct not in _PNG_DEPTHS or depth not in _PNG_DEPTHS[ct] or inter or comp or filt or not w or not h:
        return None
    types = [t for t, _ in chunks]
    if _NOT_IDAT.intersection(types) or b"IDAT" not in types:
        return None
    channels = _PNG_CHANNELS[ct]
    stride = 1 + (w * channels * depth + 7) // 8
    idat = b"".join(p for t, p in chunks if t == b"IDAT")
    if not _idat_ok(idat, h, stride):
        return None
    icc = _png_icc(chunks)
    palette = None
    if ct == 3:
        palette = next((p for t, p in chunks if t == b"PLTE"), None)
        if not palette or len(palette) % 3 or len(palette) // 3 > 2 ** depth:
            return None
        base_icc = icc if _icc_components(icc) == 3 else None
        cs: str = "Indexed"
        icc = base_icc
    else:
        cs, icc = _colorspace(channels, icc)
    parms = {"Predictor": 15, "Colors": channels, "BitsPerComponent": depth, "Columns": w}
    return ImagePayload(filter="FlateDecode", data=idat, width=w, height=h, colorspace=cs,  # type: ignore[arg-type]
                        bpc=depth, icc=icc, palette=palette, decode=None,  # type: ignore[arg-type]
                        decode_parms=parms, smask=None)


# --------------------------------------------------------------------------- exact decode

def _unreadable(path: Path, exc: BaseException) -> LosslessUnavailable:
    return LosslessUnavailable("unreadable_image", f"{path}: the image can't be decoded ({type(exc).__name__}).",
                               file=str(path))


def decode_exact(path: Path, kind: InputKind) -> DecodedImage:
    """Decode ``path`` with nothing lost: the samples as stored, alpha split off, profile and
    orientation read (never applied: a TIFF's tag is taken away before Pillow can turn it), the
    metadata an encode would carry.

    LosslessUnavailable when Pillow's mode can't hold the depth the header declares (Pillow 13
    reads 16-bit colour PNG and TIFF as 8-bit), for a 12-bit or lossless/hierarchical JPEG, a
    mode a PDF can't take exactly, more than one frame, a TIFF whose first image may not be the
    picture (a reduced-resolution IFD0 or SubIFDs) or whose alpha is premultiplied, or an image
    that can't be decoded.
    """
    path = Path(path)
    data = path.read_bytes()
    if kind == "jpeg":
        try:
            sof, precision, _, _, _ = _jpeg._frame(data)
        except ValueError as exc:
            raise _unreadable(path, exc) from None
        if precision != 8:
            raise LosslessUnavailable(
                "jpeg_precision",
                f"{path}: a {precision}-bit JPEG can't be carried in a PDF or decoded exactly, so pdfintake can't place it.",
                file=str(path), precision=precision)
        if sof in _jpeg.LOSSLESS_SOF or sof in _jpeg.HIERARCHICAL_SOF:
            raise LosslessUnavailable(
                "jpeg_process",
                f"{path}: a lossless or hierarchical JPEG (SOF 0x{sof:02X}) can't be decoded exactly, so pdfintake can't place it.",
                file=str(path), sof=sof)
    try:
        img = open_image(path, kind)
    except PdfIntakeError:
        raise
    except Exception as exc:                             # noqa: BLE001 - UnidentifiedImageError and decoder errors
        raise _unreadable(path, exc) from None
    try:
        frames = getattr(img, "n_frames", 1)
        if frames > 1 and img.format != "MPO":           # MPO: the primary image; the rest are auxiliary
            raise LosslessUnavailable(
                "multi_frame", f"{path}: the file holds {frames} images; pdfintake places one image per file.",
                file=str(path), frames=frames)
        tiff_orientation = _tiff_prepare(path, img) if kind == "tiff" else None
        img.load()
    except PdfIntakeError:
        img.close()
        raise
    except Exception as exc:                             # noqa: BLE001
        img.close()
        raise _unreadable(path, exc) from None
    if kind == "tiff" and img.size != _tiff_stored_size(img):
        size = img.size
        img.close()
        raise LosslessUnavailable(
            "unreadable_image",
            f"{path}: Pillow decoded the TIFF at {size[0]}x{size[1]}, not the size its header stores, "
            "so pdfintake can't tell the pixels are as stored.", file=str(path))
    mode = img.mode
    cap = _MODE_DEPTH.get(mode)
    if cap is None:
        img.close()
        raise LosslessUnavailable(
            "image_mode", f"{path}: Pillow decodes this image as {mode}, which pdfintake can't place exactly.",
            file=str(path), mode=mode)
    try:
        depth = declared_depth(path, kind)
    except ValueError as exc:
        img.close()
        raise _unreadable(path, exc) from None
    # HEIC and AVIF only ever take the one encode (route encoded_once), where a 10- or 12-bit source
    # becoming 8-bit is the lossy exception DESIGN §0.8 allows; every other kind must decode exactly.
    if depth > cap and kind not in _ENCODE_ONLY:
        img.close()
        raise LosslessUnavailable(
            "bit_depth",
            f"{path}: the file has {depth} bits per sample but Pillow reads it as {mode} ({cap}-bit), "
            "so placing it would lose data.",
            file=str(path), declared=depth, mode=mode)
    trns = img.info.get("transparency")
    if kind == "png" and mode == "L" and depth in (2, 4) and isinstance(trns, int) and 0 <= trns < 2 ** depth:
        # Pillow 13 scales a 2- or 4-bit grey PNG's samples to 8 bits but not its tRNS value (measured):
        # scale it the same way, or the transparent colour would mark the wrong pixels.
        img.info["transparency"] = trns * (255 // (2 ** depth - 1))
    icc = img.info.get("icc_profile") or None
    orientation = tiff_orientation if tiff_orientation is not None else exif_orientation(img.info.get("exif"))
    warnings: list[str] = []
    if kind == "jpeg":
        segments = carried_segments(data)
    else:
        segments, dropped = _info_segments(img)
        if dropped:
            warnings.append(METADATA_FORMAT_LIMIT)
    if kind == "heif" and icc is None and not _nclx_is_srgb(img.info.get("nclx_profile")):
        warnings.append(METADATA_FORMAT_LIMIT)           # the JPEG would be read as sRGB (no profile to carry)
    color, alpha = split_alpha(img)
    return DecodedImage(image=color, alpha=alpha, icc=bytes(icc) if icc else None, orientation=orientation,
                        segments=segments, source_kind=kind, warnings=warnings)


def _tiff_prepare(path: Path, img: "PIL.Image.Image") -> int:
    """Before a TIFF loads: refuse what can't be placed as stored, read its Orientation, and take
    the tag away so Pillow doesn't turn the pixels.

    Pillow 13 reports a TIFF with Orientation 5-8 at its turned size from the moment it opens, and
    runs ``ImageOps.exif_transpose(in_place=True)`` at the end of the load. An uncompressed file
    opened by path is memory-mapped at that turned size, so its pixels come out scrambled
    (measured: modes L, I;16, P, RGBA and CMYK at orientations 5-8). So the size goes back to the
    stored one and the tag leaves the cached EXIF: nothing is turned, and the page matrix places
    the stored pixels as on every route. ``decode_exact`` checks the loaded size against the
    header, so a Pillow that changes this fails by name rather than placing wrong pixels.
    """
    tags = img.tag_v2                                    # type: ignore[attr-defined]
    subfile = tags.get(254, 0)
    if (isinstance(subfile, int) and subfile & 1) or 330 in tags:
        raise LosslessUnavailable(
            "sub_images",
            f"{path}: the TIFF's first image is a reduced-resolution preview or the file holds sub-images "
            "(as camera raw files do), so pdfintake can't tell which image is the picture.",
            file=str(path))
    extra = tags.get(338, ())
    extra = (extra,) if isinstance(extra, int) else tuple(extra or ())
    if 1 in extra:
        raise LosslessUnavailable(
            "associated_alpha",
            f"{path}: the TIFF stores premultiplied (associated) alpha, which Pillow un-premultiplies on "
            "decode, so its samples can't be placed as stored.", file=str(path))
    value, typ = tags.get(274, 1), tags.tagtype.get(274)
    orientation = value if typ in (3, 4) and isinstance(value, int) and 1 <= value <= 8 else 1
    stored = _tiff_stored_size(img)
    if stored is not None and img.size != stored:
        img._size = stored                               # type: ignore[attr-defined] - Pillow's own swap, undone
    try:
        img.getexif().pop(0x0112, None)
    except Exception:                                    # noqa: BLE001 - an unreadable EXIF can't turn anything
        pass
    return orientation


def _tiff_stored_size(img: "PIL.Image.Image") -> tuple[int, int] | None:
    w, h = img.tag_v2.get(256), img.tag_v2.get(257)      # type: ignore[attr-defined]
    return (w, h) if isinstance(w, int) and isinstance(h, int) else None


# nclx (ISO/IEC 23091-2) codes a JPEG without a profile matches: primaries BT.709/sRGB (1) or
# unspecified (2); transfer BT.709 (1), unspecified (2), BT.601 (6), sRGB (13), BT.2020 SDR (14, 15).
_SRGB_PRIMARIES = frozenset({1, 2})
_SRGB_TRANSFER = frozenset({1, 2, 6, 13, 14, 15})


def _nclx_is_srgb(nclx: object) -> bool:
    if not isinstance(nclx, dict):
        return True                                      # no nclx box: the HEIF default is sRGB
    return (nclx.get("color_primaries", 2) in _SRGB_PRIMARIES
            and nclx.get("transfer_characteristics", 2) in _SRGB_TRANSFER)


def _info_segments(img: "PIL.Image.Image") -> tuple[list[bytes], bool]:
    """EXIF and XMP that Pillow read from a non-JPEG source, as JPEG APP1 segments.

    The flag is true when a block was too big for one JPEG segment and had to be left out.
    """
    segs: list[bytes] = []
    dropped = False
    exif = img.info.get("exif")
    if exif:
        exif = bytes(exif)
        if not exif.startswith(_jpeg.EXIF_ID):
            exif = _jpeg.EXIF_ID + exif
        try:
            segs.append(_jpeg._segment(0xE1, exif))
        except ValueError:
            dropped = True
    xmp = img.info.get("xmp")
    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")
    if xmp:
        xmp = bytes(xmp)
        if not xmp.lstrip().startswith(b"<?xpacket"):    # Pillow hands some formats' XMP over unwrapped
            xmp = (b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>' + xmp
                   + b'<?xpacket end="w"?>')
        try:
            segs.append(_jpeg._segment(0xE1, _jpeg.XMP_ID + xmp))
        except ValueError:
            dropped = True
    return segs, dropped


def split_alpha(img: "PIL.Image.Image") -> tuple["PIL.Image.Image", "PIL.Image.Image | None"]:
    """(the colour samples, the alpha or None), with no depth or colour change.

    RGBA and LA split by channel. A single transparent colour (``info["transparency"]`` on 1, L,
    I;16 or RGB) becomes a 0/255 alpha mask. A palette image with transparency expands to its
    exact RGBA values first, since its alpha lives in the palette.
    """
    from PIL import Image

    mode = img.mode
    if mode in ("PA",) or (mode == "P" and img.info.get("transparency") is not None):
        img = img.convert("RGBA")
        mode = "RGBA"
    if mode == "RGBA":
        r, g, b, a = img.split()
        return Image.merge("RGB", (r, g, b)), a
    if mode == "LA":
        lum, a = img.split()
        return lum, a
    trns = img.info.get("transparency")
    if trns is not None and (mode in ("1", "L", "RGB") or mode.startswith("I;16")):
        arr = np.asarray(img)
        if mode == "1":
            arr = arr.astype(np.uint8) * 255
        if mode == "RGB":
            if not isinstance(trns, tuple) or len(trns) != 3:
                return img, None
            opaque = np.any(arr != np.asarray(trns, dtype=arr.dtype), axis=2)
        else:
            value = 255 if (mode == "1" and trns) else trns
            if not isinstance(value, int):
                return img, None
            opaque = arr != value
        alpha = Image.fromarray(np.where(opaque, 255, 0).astype(np.uint8))
        color = img.copy()
        color.info.pop("transparency", None)
        return color, alpha
    return img, None


# --------------------------------------------------------------------------- Flate payloads

def _png_filter(raw: bytes, height: int, rowbytes: int, bpp: int) -> bytes:
    """PNG-filtered rows (a tag byte, then the row), each row's filter the one with the smallest
    sum of absolute signed residuals (libpng's heuristic). Works on bytes, as PNG and PDF do."""
    a = np.frombuffer(raw, np.uint8).reshape(height, rowbytes)
    out = np.empty((height, rowbytes + 1), np.uint8)
    chunk = max(1, (4 << 20) // max(rowbytes, 1))
    zero = np.zeros(rowbytes, np.uint8)
    for start in range(0, height, chunk):
        stop = min(height, start + chunk)
        cur = a[start:stop].astype(np.int16)
        prev = np.vstack([a[start - 1] if start else zero, a[start:stop - 1]]).astype(np.int16)
        left = np.zeros_like(cur)
        upleft = np.zeros_like(cur)
        if bpp < rowbytes:
            left[:, bpp:] = cur[:, :-bpp]
            upleft[:, bpp:] = prev[:, :-bpp]
        p = left + prev - upleft
        pa, pb, pc = np.abs(p - left), np.abs(p - prev), np.abs(p - upleft)
        paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, upleft))
        cands = np.stack([cur, cur - left, cur - prev, cur - ((left + prev) >> 1), cur - paeth]) & 0xFF
        cands = cands.astype(np.uint8)
        score = np.abs(cands.view(np.int8).astype(np.int32)).sum(axis=2)
        best = score.argmin(axis=0)
        rows = np.arange(stop - start)
        out[start:stop, 0] = best
        out[start:stop, 1:] = cands[best, rows]
    return out.tobytes()


def _flate(raw: bytes, width: int, height: int, channels: int, bpc: int) -> tuple[bytes, dict[str, int] | None]:
    """Plain Flate or Flate over PNG Predictor 15, whichever is smaller (plain on a tie)."""
    rowbytes = (width * channels * bpc + 7) // 8
    bpp = max(1, (channels * bpc) // 8)
    plain = zlib.compress(raw, FLATE_LEVEL)
    pred = zlib.compress(_png_filter(raw, height, rowbytes, bpp), FLATE_LEVEL)
    if len(pred) < len(plain):
        return pred, {"Predictor": 15, "Colors": channels, "BitsPerComponent": bpc, "Columns": width}
    return plain, None


def _samples(img: "PIL.Image.Image") -> tuple[int, int, bytes]:
    """(components, bits per component, packed big-endian sample bytes) for a placeable mode."""
    mode = img.mode
    if mode == "1":
        return 1, 1, img.tobytes()
    if mode == "L":
        return 1, 8, img.tobytes()
    if mode.startswith("I;16"):
        return 1, 16, np.asarray(img).astype(">u2").tobytes()
    if mode == "RGB":
        return 3, 8, img.tobytes()
    if mode == "CMYK":
        return 4, 8, img.tobytes()
    raise LosslessUnavailable("image_mode", f"a {mode} image can't be placed exactly", mode=mode)


def _gray_payload(img: "PIL.Image.Image") -> ImagePayload:
    n, bpc, raw = _samples(img)
    data, parms = _flate(raw, img.width, img.height, n, bpc)
    return ImagePayload(filter="FlateDecode", data=data, width=img.width, height=img.height,
                        colorspace="DeviceGray", bpc=bpc, icc=None, palette=None, decode=None,  # type: ignore[arg-type]
                        decode_parms=parms, smask=None)


def flate_payload(dec: DecodedImage) -> ImagePayload:
    """The decoded samples as a Flate image: 1/8/16 bpc; Gray, RGB or CMYK (ICCBased with the
    source profile when its component count matches; otherwise the device space, and the profile
    is dropped, which ``assemble`` reports as ``metadata_format_limit``). A palette image expands
    to its exact RGB values. Alpha becomes an /SMask at its own depth. No other conversion."""
    img = dec.image
    if img.mode == "P":
        img = img.convert("RGB")                         # an exact palette lookup
    n, bpc, raw = _samples(img)
    cs, icc = _colorspace(n, dec.icc)
    data, parms = _flate(raw, img.width, img.height, n, bpc)
    smask = _gray_payload(dec.alpha) if dec.alpha is not None else None
    return ImagePayload(filter="FlateDecode", data=data, width=img.width, height=img.height,
                        colorspace=cs, bpc=bpc, icc=icc, palette=None, decode=None,  # type: ignore[arg-type]
                        decode_parms=parms, smask=smask)


# --------------------------------------------------------------------------- the one encode

def _upright(img: "PIL.Image.Image", orientation: int) -> "PIL.Image.Image":
    """``img`` turned the way EXIF ``orientation`` says (a copy; 1 returns a copy unchanged)."""
    from PIL import Image

    name = _UPRIGHT.get(orientation)
    return img.transpose(getattr(Image.Transpose, name)) if name else img.copy()


def _sixteen_to_eight(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """A 16-bit grey image scaled (rounded) to 8-bit L: 0 -> 0, 65535 -> 255."""
    from PIL import Image

    arr = np.asarray(img).astype(np.uint32)
    return Image.fromarray(((arr * 255 + 32767) // 65535).astype(np.uint8), "L")


def encode_jpeg(dec: DecodedImage, *, quality: int = JPEG_QUALITY) -> bytes:
    """The one lossy encode, for routes ``encoded_once`` only (HEIC, AVIF, lossy WebP).

    The pixels are turned upright first (pillow-heif has already applied a HEIC's irot and reset
    its tag; AVIF and WebP keep theirs), then Pillow encodes at ``quality``, 4:2:0, optimized
    Huffman tables, with the source's ICC profile. The source's metadata segments (EXIF, XMP,
    IPTC, comments) are spliced in after JFIF, since Pillow can't write all of them, and
    ``settle_metadata`` makes them say Orientation 1 and the new size. Alpha isn't encoded:
    ``assemble`` turns ``dec.alpha`` the same way and adds it as a Flate /SMask.
    """
    img = _upright(dec.image, dec.orientation)
    if img.mode.startswith("I;16"):
        img = _sixteen_to_eight(img)                     # never Pillow's convert, which clips at 255
    elif img.mode == "1":
        img = img.convert("L")
    elif img.mode not in ("L", "RGB", "CMYK"):
        img = img.convert("RGB")                         # P: an exact palette lookup
    img.info = {}
    n = {"L": 1, "RGB": 3, "CMYK": 4}[img.mode]
    kwargs: dict[str, object] = {"quality": quality, "subsampling": 2, "optimize": True}
    if dec.icc is not None and _icc_components(dec.icc) == n:
        kwargs["icc_profile"] = dec.icc
    buf = io.BytesIO()
    img.save(buf, format="JPEG", **kwargs)
    out = _jpeg._splice(buf.getvalue(), list(dec.segments))
    return settle_metadata(out, size=img.size)

