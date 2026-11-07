"""Lossless shrink: a smaller PDF with identical content, or a byte copy.

Every step is lossless by construction:

* ``recompress_streams`` re-deflates FlateDecode streams at level 9. For 8-bit images it also
  tries PNG prediction (Up or Paeth, chosen per row) and keeps whichever encoding is smallest.
  The decoded bytes never change, and a predictor image is read back through qpdf and must
  match its samples before it is kept. A new encoding is kept only when it is at least 16 bytes
  smaller. XMP packets, DCT, JPX, JBIG2, CCITT and multi-filter streams are never touched.
* The save generates object streams, drops objects nothing references and earlier revisions,
  and writes a deterministic ``/ID`` (qpdf keeps an existing ``/ID[0]``). It never re-serialises
  XMP: when object streams raise the file's effective version (the higher of the header and the
  catalog ``/Version``, which overrides the header), ``meta.patch_xmp_pdfversion`` raises the
  packet's ``pdf:PDFVersion`` in place at the same length. A version the save doesn't raise is
  left alone, so the packet is never patched down.
* Never grow: the result is used only when it is smaller than the source, and by at least 1%
  when the build didn't change any content (a saving of exactly 1% is kept). Otherwise ``dest``
  is a byte copy of ``src``.
* Never encrypted: every save passes ``encryption=False`` explicitly rather than relying on
  pikepdf's default (decided 2026-11-05, DESIGN §10.4, §11.1), and ``deterministic_id`` stays on,
  which qpdf allows only on an unencrypted write. An encrypted source (``flow`` decrypts once, so
  this is a guard) is always rewritten, never byte-copied, so ``dest`` is never encrypted.
"""

from __future__ import annotations

import mmap
import os
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import meta

if TYPE_CHECKING:
    import numpy as np
    import pikepdf

__all__ = ["ShrinkReport", "recompress_streams", "lossless_shrink"]

MIN_GAIN_BYTES = 16                # a re-encoded stream is kept only if at least this much smaller
MIN_SAVING_PERCENT = 1             # below this, with no content change, the source is byte-copied
OBJECT_STREAM_VERSION = "1.5"      # object streams need PDF 1.5
MAX_INFLATE_BYTES = 1 << 30        # a stream that inflates beyond this is left alone
_ROW_BLOCK = 512                   # rows per numpy block when predicting (bounds memory on big images)
_SKIP_TYPES = frozenset({"/Metadata", "/ObjStm", "/XRef"})
_PNG_UP, _PNG_PAETH = 2, 4
_DEFAULT_FLATE_LEVEL = -1          # zlib's default, qpdf's unless set


@dataclass
class ShrinkReport:
    before: int; after: int; byte_copy: bool; streams_recompressed: int; predictor_images: int
    objects_dropped: int; revisions_dropped: int; version_from: str; version_to: str; xmp_pdfversion_patched: bool


# --------------------------------------------------------------------------- stream helpers

def _filters(stream: "pikepdf.Stream") -> list[str] | None:
    """The stream's filter names, [] when unfiltered, None when /Filter is malformed."""
    import pikepdf

    raw = stream.get("/Filter")
    if raw is None:
        return []
    if isinstance(raw, pikepdf.Name):
        return [str(raw)]
    if isinstance(raw, pikepdf.Array):
        names = [str(f) for f in raw if isinstance(f, pikepdf.Name)]
        return names if len(names) == len(raw) else None
    return None


def _inflate(raw: bytes) -> bytes | None:
    """The zlib payload of a Flate stream, or None unless it decodes cleanly with nothing after it."""
    d = zlib.decompressobj()
    out = bytearray()
    try:
        chunk = d.decompress(raw, MAX_INFLATE_BYTES + 1)
        out += chunk
        if len(out) > MAX_INFLATE_BYTES or d.unconsumed_tail:
            return None
        out += d.flush()
    except zlib.error:
        return None
    if not d.eof or d.unused_data.strip(b"\r\n \t\x00"):
        return None
    return bytes(out)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _image_geometry(stream: "pikepdf.Stream", decoded_len: int) -> tuple[int, int, int] | None:
    """(width, height, bytes per pixel) of an 8-bit image whose samples fill the stream exactly."""
    import pikepdf

    if stream.get("/Subtype") != pikepdf.Name.Image or stream.get("/ImageMask") is True:
        return None
    width, height, bpc = _int(stream.get("/Width")), _int(stream.get("/Height")), _int(stream.get("/BitsPerComponent"))
    if not width or not height or bpc != 8 or width <= 0 or height <= 0:
        return None
    pixels = width * height
    if decoded_len % pixels:
        return None
    bpp = decoded_len // pixels
    return (width, height, bpp) if 1 <= bpp <= 32 else None


def _paeth(a: "np.ndarray", b: "np.ndarray", c: "np.ndarray") -> "np.ndarray":
    import numpy as np

    a16, b16, c16 = a.astype(np.int16), b.astype(np.int16), c.astype(np.int16)
    pa = np.abs(b16 - c16)
    pb = np.abs(a16 - c16)
    pc = np.abs(a16 + b16 - 2 * c16)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def png_predict(samples: bytes, width: int, height: int, bpp: int) -> bytes:
    """PNG-predicted rows (Predictor 15 layout): per row the smaller of Up and Paeth, tag byte first.

    The row choice uses the usual minimum-sum-of-absolute-differences heuristic. Filtering only
    reads raw samples, so the whole image is predicted with vectorised numpy, one block of rows at
    a time.
    """
    import numpy as np

    stride = width * bpp
    img = np.frombuffer(samples, dtype=np.uint8).reshape(height, stride)
    out = bytearray()
    prev_last = np.zeros(stride, dtype=np.uint8)
    for top in range(0, height, _ROW_BLOCK):
        rows = img[top:top + _ROW_BLOCK]
        up = np.vstack([prev_last[None, :], rows[:-1]])
        left = np.zeros_like(rows)
        left[:, bpp:] = rows[:, :-bpp]
        upleft = np.zeros_like(rows)
        upleft[:, bpp:] = up[:, :-bpp]
        f_up = rows - up                               # uint8 arithmetic wraps mod 256, as PNG wants
        f_paeth = rows - _paeth(left, up, upleft)
        cost_up = np.abs(f_up.view(np.int8).astype(np.int32)).sum(axis=1)
        cost_paeth = np.abs(f_paeth.view(np.int8).astype(np.int32)).sum(axis=1)
        use_paeth = cost_paeth < cost_up
        chosen = np.where(use_paeth[:, None], f_paeth, f_up)
        tags = np.where(use_paeth, _PNG_PAETH, _PNG_UP).astype(np.uint8)
        block = np.hstack([tags[:, None], chosen])
        out += block.tobytes()
        prev_last = rows[-1].copy()
    return bytes(out)


def _try_stream(stream: "pikepdf.Stream") -> int:
    """Re-encode one stream if that saves at least MIN_GAIN_BYTES. 0 untouched, 1 recompressed, 2 predicted."""
    import pikepdf

    if stream.get("/Type") is not None and str(stream.get("/Type")) in _SKIP_TYPES:
        return 0
    if "/F" in stream:                                 # external file stream: its data isn't here
        return 0
    filters = _filters(stream)
    if filters is None or filters not in ([], ["/FlateDecode"]):
        return 0
    raw = bytes(stream.read_raw_bytes())
    old_parms = stream.get("/DecodeParms")

    candidates: list[tuple[bytes, Any, bool]] = []    # (encoded, decode_parms, predicted)
    if filters == ["/FlateDecode"]:
        inner = _inflate(raw)
        if inner is None:
            return 0
        candidates.append((zlib.compress(inner, 9), old_parms, False))
        try:
            samples = bytes(stream.read_bytes()) if old_parms is not None else inner
        except Exception:  # noqa: BLE001 - a predictor qpdf can't decode: keep the zlib-only candidate
            samples = None
    else:
        samples = raw

    geometry = _image_geometry(stream, len(samples)) if samples is not None else None
    if geometry is None and filters == []:
        return 0                                       # qpdf deflates plain unfiltered streams on save
    if geometry is not None and samples is not None:
        width, height, bpp = geometry
        if not candidates or old_parms is not None:   # else the zlib-only candidate already is the plain one
            candidates.append((zlib.compress(samples, 9), None, False))
        predicted = zlib.compress(png_predict(samples, width, height, bpp), 9)
        parms = pikepdf.Dictionary(Predictor=15, Colors=bpp, BitsPerComponent=8, Columns=width)
        candidates.append((predicted, parms, True))

    best = min(candidates, key=lambda c: len(c[0]))   # min keeps the first of equals: no predictor wins ties
    if len(best[0]) > len(raw) - MIN_GAIN_BYTES:
        return 0
    data, parms, predicted = best
    stream.write(data, filter=pikepdf.Name.FlateDecode, decode_parms=parms)
    if predicted:
        try:
            ok = bytes(stream.read_bytes()) == samples
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:                                     # qpdf must read back the exact samples, or we revert
            restore = pikepdf.Name.FlateDecode if filters else None
            stream.write(raw, filter=restore, decode_parms=old_parms if filters else None)
            return 0
        return 2
    return 1


def recompress_streams(pdf: "pikepdf.Pdf") -> tuple[int, int]:
    """Re-deflate Flate streams and predict 8-bit images, in place. Returns (streams, predictor images)."""
    import pikepdf

    streams = predicted = 0
    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Stream):
            continue
        result = _try_stream(obj)
        if result:
            streams += 1
            predicted += result == 2
    return streams, predicted


# --------------------------------------------------------------------------- whole-file helpers

_STARTXREF = re.compile(rb"startxref\s+(\d+)")


def _revisions(path: Path) -> int:
    """Cross-reference sections the file is built from (a linearized file's first-page "startxref 0" isn't one)."""
    if path.stat().st_size == 0:
        return 0
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return sum(1 for m in _STARTXREF.finditer(data) if int(m.group(1)) > 0)


def _object_count(pdf: "pikepdf.Pdf") -> int:
    """Indirect objects that carry content: not null, not an object or xref stream."""
    import pikepdf

    count = 0
    for obj in pdf.objects:
        if obj is None or getattr(obj, "_type_code", None) == pikepdf.ObjectType.null:
            continue
        if isinstance(obj, pikepdf.Stream) and str(obj.get("/Type")) in ("/ObjStm", "/XRef"):
            continue
        count += 1
    return count


def _vt(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in version.split("."))
    except ValueError:
        return (0,)


def _catalog_version(pdf: "pikepdf.Pdf") -> str | None:
    """The catalog's ``/Version`` (ISO 32000 7.7.2) when it is a plain version name, else None."""
    import pikepdf

    value = pdf.Root.get("/Version")
    if not isinstance(value, pikepdf.Name):
        return None
    text = str(value)[1:]
    return text if meta._version_tuple(text) is not None else None


def _effective_version(header: str, catalog: str | None) -> str:
    """The version a reader applies: the catalog's ``/Version`` when it is higher than the header."""
    return max(header, catalog, key=_vt) if catalog is not None else header


def _keep_rewrite(before: int, after: int, content_changed: bool) -> bool:
    """Never grow: a rewrite is kept only when smaller, and at least 1% smaller with no content change."""
    if content_changed:
        return after < before
    return after < before and (before - after) * 100 >= before * MIN_SAVING_PERCENT


def _save(pdf: "pikepdf.Pdf", dest: Path, min_version: str) -> None:
    import pikepdf

    pikepdf.settings.set_flate_compression_level(9)
    try:
        pdf.save(dest, object_stream_mode=pikepdf.ObjectStreamMode.generate, fix_metadata_version=False,
                 deterministic_id=True, recompress_flate=False, min_version=min_version, encryption=False)
    finally:
        pikepdf.settings.set_flate_compression_level(_DEFAULT_FLATE_LEVEL)


def lossless_shrink(src: Path, dest: Path, *, content_changed: bool) -> ShrinkReport:
    """Write ``dest``: ``src`` saved losslessly smaller, or a byte copy of ``src``. ``src`` is never modified.

    ``content_changed`` says the build changed page content (a layer added, a page turned, an
    old layer stripped). Then any saving is taken; without it, a saving under 1% gives a byte copy.
    An encrypted ``src`` (one that opens without a password) is always rewritten, without
    encryption, even when that is larger: a byte copy would keep the encryption.
    ``dest`` belongs in the work or staging folder: this isn't the commit to the user's folder.
    """
    import pikepdf

    src, dest = Path(src), Path(dest)
    before = src.stat().st_size
    tmp = dest.with_name(dest.name + ".shrink-partial")
    try:
        with pikepdf.open(src) as pdf:
            encrypted = bool(pdf.is_encrypted)
            version_from = str(pdf.pdf_version)
            catalog = _catalog_version(pdf)            # the save keeps it as it is
            effective_from = _effective_version(version_from, catalog)
            objects_before = _object_count(pdf)
            streams, predicted = recompress_streams(pdf)
            target = max(version_from, OBJECT_STREAM_VERSION, key=_vt)
            patched = False
            if _vt(_effective_version(target, catalog)) > _vt(effective_from):
                patched = meta.patch_xmp_pdfversion(pdf, target)
            _save(pdf, tmp, target)
            with pikepdf.open(tmp) as out:
                version_to = str(out.pdf_version)
                objects_after = _object_count(out)
            if version_to != target:                  # qpdf chose a higher version than asked: patch to it
                if _vt(_effective_version(version_to, catalog)) > _vt(effective_from):
                    patched = meta.patch_xmp_pdfversion(pdf, version_to) or patched
                _save(pdf, tmp, version_to)
        after = tmp.stat().st_size
        if encrypted or _keep_rewrite(before, after, content_changed):
            os.replace(tmp, dest)
            return ShrinkReport(
                before=before, after=after, byte_copy=False, streams_recompressed=streams,
                predictor_images=predicted, objects_dropped=max(0, objects_before - objects_after),
                revisions_dropped=max(0, _revisions(src) - 1), version_from=version_from,
                version_to=version_to, xmp_pdfversion_patched=patched)
        shutil.copyfile(src, dest)
        return ShrinkReport(
            before=before, after=before, byte_copy=True, streams_recompressed=0, predictor_images=0,
            objects_dropped=0, revisions_dropped=0, version_from=version_from, version_to=version_from,
            xmp_pdfversion_patched=False)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
