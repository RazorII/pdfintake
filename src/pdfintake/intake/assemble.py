"""Image files to one upright PDF, one page per image, with /Rotate 0 on every page.

Each page shows one image XObject named ``/Im0``, placed by ``q a b c d e f cm /Im0 Do Q``. The
matrix maps the stored image onto the page so it displays upright for its EXIF orientation:
the stored bytes are never turned (all 8 orientations, verified in pdfium and Quartz by the
tests). The page's long side is ``long_side_pt`` (A4 height by default) and keeps the image's
displayed aspect ratio.

What ``AssembledPage`` records, for ``verify.passthrough_identical``:
- ``embedded_sha256`` is the SHA-256 of the raw (still encoded) ``/Im0`` stream as written;
- ``source_sha256`` is the SHA-256 of the source file, except on the ``png_idat`` route, where it
  is that of the source's IDAT data joined in order. The two are equal on the ``passthrough``
  and ``png_idat`` routes, and ``assemble`` re-reads the saved file to prove it.
- ``orientation`` is the EXIF orientation the page matrix applies: the source's on the first
  three routes, 1 on ``encoded_once`` (that encode turns its pixels upright).
- ``bytes`` is the size of the image data written: the stream plus its /SMask.

The PDF gets no Info dictionary and no XMP; ``meta.set_built_info`` is the caller's step.

``dest`` is a work file, never an input: ``assemble`` refuses a ``dest`` that is one of
``images`` (DESIGN §0.6), and writes it through a ``.partial`` beside it that replaces ``dest``
only once the re-read check passes, so a failed or concurrent build never leaves a half-written
or unchecked file at ``dest``. A check that fails inside ``assemble`` raises
``LosslessUnavailable`` naming the file, never a bare exception.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ..errors import LosslessUnavailable, UsageError
from .images import (METADATA_FORMAT_LIMIT, DecodedImage, ImagePayload, Route, _gray_payload, _icc_components,
                     _png_chunks, _png_orientation, _upright, decode_exact, encode_jpeg, flate_payload, jpeg_payload,
                     png_idat_payload, route_for)
from .jpeg import _kind, check_jpeg, jpeg_info, split_jpeg
from .sniff import MAX_IMAGE_PIXELS, sniff

LONG_SIDE_PT = 842.0
XOBJECT_NAME = "/Im0"
_WIN_BUSY = frozenset({5, 32})      # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION: a handle has the file
_REPLACE_PATIENCE_S = 2.0           # how long a busy replace is tried again on Windows (_replace)

# Where each stored-image corner lands, per EXIF orientation: the page matrix for a page of
# width W and height H is (a*W, b*H, c*W, d*H, e*W, f*H) with these unit coefficients.
_UNIT_MATRIX: dict[int, tuple[int, int, int, int, int, int]] = {
    1: (1, 0, 0, 1, 0, 0),
    2: (-1, 0, 0, 1, 1, 0),
    3: (-1, 0, 0, -1, 1, 1),
    4: (1, 0, 0, -1, 0, 1),
    5: (0, -1, -1, 0, 1, 1),
    6: (0, -1, 1, 0, 0, 1),
    7: (0, 1, 1, 0, 0, 0),
    8: (0, 1, -1, 0, 1, 0),
}


@dataclass
class AssembledPage:
    source: str
    route: Route
    orientation: int
    bytes: int
    embedded_sha256: str
    source_sha256: str      # equal for passthrough (JPEG) and png_idat (IDAT data)
    warnings: list[str]


def orientation_matrix(orientation: int, page_w: float, page_h: float) -> tuple[float, float, float, float, float, float]:
    """The ``cm`` matrix that places an image stored with EXIF ``orientation`` upright on a page
    of ``page_w`` x ``page_h`` points (the displayed size). ValueError outside 1..8."""
    unit = _UNIT_MATRIX.get(orientation)
    if unit is None:
        raise ValueError(f"EXIF orientation must be 1..8, not {orientation!r}")
    a, b, c, d, e, f = unit
    return (a * page_w + 0.0, b * page_h + 0.0, c * page_w + 0.0, d * page_h + 0.0,
            e * page_w + 0.0, f * page_h + 0.0)


def page_size(width_px: int, height_px: int, orientation: int, *, long_side_pt: float = LONG_SIDE_PT) -> tuple[float, float]:
    """The displayed page size in points for an image stored at ``width_px`` x ``height_px``:
    sides swapped for orientations 5-8, the long side ``long_side_pt``, rounded to 1/10000 pt."""
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"an image of {width_px}x{height_px} pixels has no page size")
    w, h = (height_px, width_px) if orientation in (5, 6, 7, 8) else (width_px, height_px)
    scale = long_side_pt / max(w, h)
    return round(w * scale, 4), round(h * scale, 4)


def _num(v: float) -> str:
    text = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Writer:
    """Builds the pikepdf objects for payloads; one ICC stream per distinct profile."""

    def __init__(self, pdf: "object") -> None:
        self.pdf = pdf
        self._icc: dict[bytes, object] = {}

    def icc_space(self, icc: bytes, n: int) -> object:
        import pikepdf

        obj = self._icc.get(icc)
        if obj is None:
            stream = pikepdf.Stream(self.pdf, icc)
            stream.N = n
            stream.Alternate = pikepdf.Name("/" + {1: "DeviceGray", 3: "DeviceRGB", 4: "DeviceCMYK"}[n])
            obj = self.pdf.make_indirect(stream)
            self._icc[icc] = obj
        return pikepdf.Array([pikepdf.Name.ICCBased, obj])

    def colorspace(self, p: ImagePayload) -> object:
        import pikepdf

        if p.colorspace == "Indexed":
            if p.palette is None:
                raise ValueError("an Indexed payload needs a palette")
            base = self.icc_space(p.icc, 3) if p.icc is not None else pikepdf.Name.DeviceRGB
            return pikepdf.Array([pikepdf.Name.Indexed, base, len(p.palette) // 3 - 1, pikepdf.String(p.palette)])
        if p.colorspace == "ICCBased":
            comps = _icc_components(p.icc)
            if p.icc is None or comps is None:
                raise ValueError("an ICCBased payload needs a GRAY, RGB or CMYK profile")
            return self.icc_space(p.icc, comps)
        return pikepdf.Name("/" + p.colorspace)

    def image(self, p: ImagePayload) -> object:
        import pikepdf

        stream = pikepdf.Stream(self.pdf, p.data)
        stream.Type = pikepdf.Name.XObject
        stream.Subtype = pikepdf.Name.Image
        stream.Width = p.width
        stream.Height = p.height
        stream.BitsPerComponent = p.bpc
        stream.ColorSpace = self.colorspace(p)
        stream.Filter = pikepdf.Name("/" + p.filter)
        if p.decode_parms:
            stream.DecodeParms = pikepdf.Dictionary({"/" + k: v for k, v in p.decode_parms.items()})
        if p.decode is not None:
            stream.Decode = pikepdf.Array([Decimal(_num(v)) for v in p.decode])
        if p.smask is not None:
            stream.SMask = self.image(p.smask)
        return self.pdf.make_indirect(stream)


def _place(path: Path) -> tuple[ImagePayload, Route, int, str, list[str]]:
    """(payload, route, orientation for the matrix, source sha256, warnings) for one image file."""
    kind = sniff(path)
    route = route_for(kind, path)
    data = path.read_bytes()
    warnings: list[str] = []
    if route == "passthrough":
        info = jpeg_info(data)
        _check_pixels(path, info.width, info.height)
        problems = check_jpeg(data, passthrough=True)
        if problems:
            raise LosslessUnavailable("unreadable_image", f"{path}: the JPEG can't go into a PDF as it is: "
                                      + "; ".join(problems) + ".", file=str(path))
        payload = jpeg_payload(data, info)
        if info.icc is not None and payload.icc is None:
            warnings.append(METADATA_FORMAT_LIMIT)
        return payload, route, info.orientation, _sha(data), warnings
    if route == "png_idat":
        payload = png_idat_payload(data)
        if payload is not None:
            _check_pixels(path, payload.width, payload.height)
            chunks = _png_chunks(data)
            if any(t == b"iCCP" for t, _ in chunks) and payload.icc is None:
                warnings.append(METADATA_FORMAT_LIMIT)
            return payload, route, _png_orientation(chunks), _sha(payload.data), warnings
        route = "lossless"
    dec = decode_exact(path, kind)
    if route == "encoded_once":
        warnings += dec.warnings                         # what the one encode couldn't carry
        jpeg = encode_jpeg(dec)
        problems = check_jpeg(jpeg, passthrough=False)
        if problems and dec.segments:
            # Metadata settle_metadata couldn't make true to the upright pixels: leave it out
            # rather than refuse the picture, and say so.
            warnings.append(METADATA_FORMAT_LIMIT)
            dec = dataclasses.replace(dec, segments=[])
            jpeg = encode_jpeg(dec)
            problems = check_jpeg(jpeg, passthrough=False)
        if problems:
            raise _failed(path, "pdfintake's own JPEG encode failed its check: " + "; ".join(problems))
        info = jpeg_info(jpeg)
        if _kinds(dec.segments) - _kinds(split_jpeg(jpeg)[0]):
            warnings.append(METADATA_FORMAT_LIMIT)       # a block settle_metadata couldn't patch was dropped
        smask = _alpha_payload(dec) if dec.alpha is not None else None
        payload = dataclasses.replace(jpeg_payload(jpeg, info), smask=smask)
        if dec.icc is not None and payload.icc is None:
            warnings.append(METADATA_FORMAT_LIMIT)
        orientation = 1
    else:
        payload = flate_payload(dec)
        if dec.icc is not None and payload.icc is None:
            warnings.append(METADATA_FORMAT_LIMIT)
        orientation = dec.orientation
    return payload, route, orientation, _sha(data), warnings


def _failed(path: Path | str, what: str) -> LosslessUnavailable:
    return LosslessUnavailable("assemble_check", f"{path}: {what}.", file=str(path))


def _same_file(a: Path, b: Path) -> bool:
    """True when ``a`` and ``b`` name one file: the same inode, or the same resolved path once
    case is folded (Scan.png and scan.png are one file on APFS and NTFS)."""
    try:
        if a.exists() and b.exists() and os.path.samefile(a, b):
            return True
    except OSError:
        pass
    norm = lambda p: os.path.normcase(str(p.resolve())).casefold()  # noqa: E731
    return norm(a) == norm(b)


def _replace(partial: Path, dest: Path) -> None:
    """``os.replace(partial, dest)``, tried again on Windows for a short while when a file is busy.

    POSIX's rename replaces ``dest`` whoever has it open. Windows refuses while the file being
    replaced, or the one being moved, is open elsewhere: ERROR_ACCESS_DENIED (5) or
    ERROR_SHARING_VIOLATION (32), both gone once that handle closes. CI saw WinError 5 with eight
    builds replacing one ``dest`` at once (2026-11-07); a virus scanner or an indexer reading the
    new file can hold it too. So there the replace is tried again for up to ``_REPLACE_PATIENCE_S``
    seconds before the error is raised; every other error, and every error elsewhere, is raised
    at once.
    """
    deadline = time.monotonic() + _REPLACE_PATIENCE_S
    pause = 0.005
    while True:
        try:
            os.replace(partial, dest)
            return
        except PermissionError as exc:
            if (sys.platform != "win32" or getattr(exc, "winerror", None) not in _WIN_BUSY
                    or time.monotonic() >= deadline):
                raise
        time.sleep(pause)
        pause = min(pause * 2, 0.1)


def _check_pixels(path: Path, width: int, height: int) -> None:
    if width * height > MAX_IMAGE_PIXELS:
        raise LosslessUnavailable(
            "too_many_pixels",
            f"{path}: the image has {width}x{height} pixels; pdfintake refuses images "
            f"above {MAX_IMAGE_PIXELS // 1_000_000} megapixels.", file=str(path), limit=MAX_IMAGE_PIXELS)


def _kinds(segments: Sequence[bytes]) -> set[str]:
    return {_kind(s) for s in segments}


def _alpha_payload(dec: DecodedImage) -> ImagePayload:
    """The alpha of an encoded image, turned like its colour samples, as a Flate /SMask."""
    return _gray_payload(_upright(dec.alpha, dec.orientation))  # type: ignore[arg-type]


def _stream_bytes(p: ImagePayload) -> int:
    return len(p.data) + (_stream_bytes(p.smask) if p.smask is not None else 0)


def _needs_version(payloads: Sequence[ImagePayload]) -> str | None:
    def walk(p: ImagePayload) -> list[ImagePayload]:
        return [p] + (walk(p.smask) if p.smask is not None else [])

    flat = [q for p in payloads for q in walk(p)]
    if any(q.bpc == 16 for q in flat):
        return "1.5"                                     # 16 bits per component
    if any(p.smask is not None for p in payloads):
        return "1.4"                                     # soft masks
    return None


def assemble(images: Sequence[Path], dest: Path) -> list[AssembledPage]:
    """Write ``images`` (in order, one page each) to ``dest`` as an upright PDF. /Rotate 0 asserted.

    UsageError for an empty list, a file that isn't an image, or a ``dest`` that is one of the
    images; LosslessUnavailable for an image that can't be placed without losing data. ``dest``
    is written only once every image is placed and the saved file re-reads as built.
    """
    import pikepdf

    if not images:
        raise UsageError("no_inputs", "assemble needs at least one image.")
    dest = Path(dest)
    for item in images:
        if _same_file(Path(item), dest):
            raise UsageError("output_is_input", f"{dest}: assemble won't write over one of its own inputs.",
                             file=str(dest))
    pdf = pikepdf.new()
    writer = _Writer(pdf)
    pages: list[AssembledPage] = []
    payloads: list[ImagePayload] = []
    for item in images:
        path = Path(item)
        payload, route, orientation, source_sha, warnings = _place(path)
        _check_pixels(path, payload.width, payload.height)
        page_w, page_h = page_size(payload.width, payload.height, orientation)
        matrix = orientation_matrix(orientation, page_w, page_h)
        content = ("q " + " ".join(_num(v) for v in matrix) + f" cm {XOBJECT_NAME} Do Q").encode("ascii")
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page,
            MediaBox=pikepdf.Array([0, 0, Decimal(_num(page_w)), Decimal(_num(page_h))]),
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary({XOBJECT_NAME: writer.image(payload)})),
            Contents=pdf.make_stream(content),
        )
        pdf.pages.append(pikepdf.Page(page))
        payloads.append(payload)
        pages.append(AssembledPage(source=str(path), route=route, orientation=orientation,
                                   bytes=_stream_bytes(payload), embedded_sha256=_sha(payload.data),
                                   source_sha256=source_sha, warnings=list(dict.fromkeys(warnings))))
    _assert_upright(pdf)
    version = _needs_version(payloads)
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".partial", dir=dest.parent)
    os.close(fd)
    partial = Path(name)
    try:
        if version is None:
            pdf.save(partial, deterministic_id=True)
        else:
            pdf.save(partial, deterministic_id=True, min_version=version)
        pdf.close()
        _check_written(partial, pages)
        _replace(partial, dest)
    except BaseException:
        pdf.close()
        partial.unlink(missing_ok=True)
        raise
    return pages


def _assert_upright(pdf: "object") -> None:
    rotate = pdf.Root.Pages.get("/Rotate", 0)  # type: ignore[attr-defined]
    bad = [i + 1 for i, p in enumerate(pdf.pages) if int(p.obj.get("/Rotate", rotate)) % 360 != 0]  # type: ignore[attr-defined]
    if bad or int(rotate) % 360:
        raise _failed("the assembled PDF", f"pages {bad or 'all'} carry /Rotate other than 0")


def _check_written(dest: Path, pages: Sequence[AssembledPage]) -> None:
    """Re-open the saved PDF: /Rotate 0 everywhere, and each /Im0 stream is the bytes we meant."""
    import pikepdf

    with pikepdf.open(dest) as pdf:
        _assert_upright(pdf)
        if len(pdf.pages) != len(pages):
            raise _failed(dest, f"wrote {len(pdf.pages)} pages for {len(pages)} images")
        for i, (page, rec) in enumerate(zip(pdf.pages, pages)):
            raw = page.obj.Resources.XObject[XOBJECT_NAME].read_raw_bytes()
            if _sha(raw) != rec.embedded_sha256:
                raise _failed(rec.source, f"page {i + 1}: the saved image stream differs from the one built")
            if rec.route in ("passthrough", "png_idat") and rec.embedded_sha256 != rec.source_sha256:
                raise _failed(rec.source, f"page {i + 1}: a {rec.route} image isn't byte-identical to its source")
