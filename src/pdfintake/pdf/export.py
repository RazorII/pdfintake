"""Export: a copy to share (``export clean``) and a non-editable copy (``export flatten``).

Both build a new file and never touch the input. ``dest`` is a build path inside pdfintake's
cache (a staging file): writing into the user's folders is ``flow.commit``'s job alone, so the
caller commits the built file there afterwards. A ``dest`` outside the cache is a UsageError.

**clean** is Share Clean's wheel-only subset. It removes the Info dictionary except Title (and
the dates, unless ``strip_dates``), every XMP packet, ``/PieceInfo`` (pdfintake's own layer
marker included) and page thumbnails, through ``meta.strip_for_export``. Every JPEG reachable
from the trailer (image XObjects, and inline images in page, form, appearance, pattern and
Type 3 content) loses its EXIF, XMP, IPTC, comment and other application segments through
``jpeg.strip_jpeg(keep_icc=True)``; its ICC profile, its Adobe colour-transform segment, its
tables, frame and entropy-coded data stay byte for byte. An unreferenced image is neither
cleaned nor compared: the save drops it. An inline JPEG behind a second filter is left as it is
(warning ``inline_jpeg_metadata_kept``). The copy's file ID is derived afresh from its content,
so it can't be matched to the original by ID. Page content, the text layer, form fields and
annotations are kept.

**flatten** bakes form fields and annotations into page content with pikepdf (qpdf's
``flattenAnnotations``, the on-screen selection), so the copy looks as the file looks on screen
and nothing in it can be edited. The original page content, the text layer among it, is kept
byte for byte; the new appearance content is appended after it. Links stay (they edit nothing);
popups go with the annotations they belong to. Fields that have no appearance yet get one from
qpdf first (warning ``appearances_generated``; NeedAppearances is set for qpdf when a field lacks
one, and goes with the AcroForm). An annotation qpdf can't bake (one with no appearance at all)
stops the export: nothing is half-flattened.

**Refusals.** Both refuse signed, certified, usage-rights and timestamped files (any rewrite
breaks the signature), encrypted files, and the provenances ``counterparty`` and ``evidence``:
an evidence copy goes out as the original. Flatten also refuses XFA forms, whose visible pages
are generated at view time. Two detectors decide, and either one is enough to refuse:
``refusals.read`` (an object walk plus a raw byte scan) and this module's own walk of the opened
file. An input that isn't a PDF is a UsageError (``not_a_pdf``); a PDF that pikepdf or pdfium
can't open is refused as ``damaged`` (exit 4).

**Verify.** Before an export is returned, the built file is re-opened and checked: nothing the
export removes is left, every JPEG's image data is identical, each page renders identically in
pdfium (72 dpi, grey, forms and annotations drawn; for flatten, pages whose field appearances
qpdf had to generate are left out), and the text is kept (clean: identical per page; flatten:
every character of each original page is still there, and the original content streams and the
page's form XObjects are unchanged). Any failure deletes the built file and raises VerifyFailed,
whose ``checks`` are ``pdf.verify.Check`` objects, the one check type a run uses too (decided
2026-11-05).

**Copies.** A searchable copy (DESIGN §10.2) carries a ``pdfintake:DerivedFrom`` block in its XMP,
written by ``meta.mark_derived``. Exporting one is allowed, with warning ``exporting_a_copy``:
the original is the file that counts. The provenance still decides, so a copy of an evidence
original is refused like the original (the original is what goes out). Clean removes the block
with the rest of the XMP; flatten keeps it.

Signing or certifying an exported copy (pyHanko) is a later version.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import PdfIntakeError, Refused, UsageError, VerifyFailed
from .verify import Check

if TYPE_CHECKING:
    import pikepdf

__all__ = [
    "ExportReport", "export_clean", "export_flatten",
    "PROVENANCES", "REFUSAL_CODES", "WARNING_CODES", "WARNING_MESSAGES",
]

PROVENANCES = ("ordinary", "counterparty", "evidence")

# The refusal codes export can raise, in the order the first one found is named (the most specific
# kind of signature first: a certified file is signed too). They are refusals.CODES' names, plus
# "damaged" (a PDF neither pikepdf nor pdfium can read; DESIGN §3.3 makes that exit 4).
REFUSAL_CODES = ("certified", "usage_rights", "timestamped", "signed", "encrypted", "xfa",
                 "provenance_counterparty", "provenance_evidence", "damaged")

_REFUSAL_MESSAGES: Mapping[str, str] = {
    "signed": "This PDF is digitally signed, and any change to it breaks the signature, so pdfintake doesn't export it. Share the original.",
    "certified": "This PDF is certified, and any change to it breaks the certification, so pdfintake doesn't export it. Share the original.",
    "usage_rights": "This PDF carries usage rights (a signature from the program that made it), and any change breaks them, so pdfintake doesn't export it. Share the original.",
    "timestamped": "This PDF carries a document timestamp, and any change to it breaks the timestamp, so pdfintake doesn't export it. Share the original.",
    "encrypted": "This PDF is encrypted, so pdfintake doesn't export it.",
    "xfa": "This PDF is an XFA form: its pages are built when it is opened, so pdfintake can't flatten what you see.",
    "provenance_counterparty": "This file came from the other side of a contract, deal or dispute, so it goes out exactly as you received it: share the original, not an export.",
    "provenance_evidence": "This file is an evidence original, so it goes out exactly as it is: share the original, not an export.",
    "damaged": "This PDF is damaged and can't be read reliably, so pdfintake doesn't export it.",
}

# Warnings an export can carry (ExportReport.warnings holds these codes; core words them here).
WARNING_MESSAGES: Mapping[str, str] = {
    "conformance_claim_dropped": "This PDF claimed PDF/A or PDF/UA conformance in its XMP metadata; removing the metadata drops that claim.",
    "jpx_metadata_kept": "This PDF has JPEG 2000 images; metadata inside them is not cleaned.",
    "embedded_files_kept": "This PDF has file attachments; their own metadata is not cleaned.",
    "annotation_authors_kept": "Some comments or markups carry an author name, which stays in the copy.",
    "optional_content_kept": "This PDF has layers (optional content); hidden layers stay in the copy.",
    "input_repaired": "The input was damaged and was repaired while it was read.",
    "appearances_generated": "Some form fields had no drawn appearance, so one was generated before flattening; they may look slightly different.",
    "annotations_not_shown_dropped": "Some annotations weren't shown on screen (hidden or print-only), so they aren't in the flattened copy.",
    "inline_jpeg_metadata_kept": "Some small JPEG images are stored inside the page content behind a second filter; metadata inside them is not cleaned.",
    "exporting_a_copy": "This file is a searchable copy pdfintake made, not the original; where it matters, the original is the one that counts.",
}
WARNING_CODES: frozenset[str] = frozenset(WARNING_MESSAGES)


@dataclass
class ExportReport:
    dest: Path
    removed: list[str]          # what the export took out, e.g. "Info/Author", "XMP", "jpeg/Exif ×2", "annotations/Widget ×3"
    kept: list[str]             # what it deliberately kept, e.g. "Info/Title", "jpeg/ICC ×1", "text_layer", "annotations/Link ×1"
    warnings: list[str]         # codes from WARNING_CODES, each once, sorted


# --------------------------------------------------------------------------- arguments

def _fold(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path)).casefold()


def _check_args(src: Path, dest: Path, provenance: str) -> None:
    if provenance not in PROVENANCES:
        raise UsageError("bad_provenance", f"--provenance must be one of {', '.join(PROVENANCES)}.",
                         value=str(provenance)[:40])
    if provenance in ("counterparty", "evidence"):
        code = f"provenance_{provenance}"
        raise Refused(code, _REFUSAL_MESSAGES[code], path=str(src))
    if not src.is_file():
        raise UsageError("input_missing", f"The input {src} doesn't exist or isn't a file.", path=str(src))
    from .. import platforms

    cache = _fold(platforms.cache_root())
    target = _fold(dest)
    if not (target == cache or target.startswith(cache.rstrip(os.sep) + os.sep)):
        raise UsageError("export_dest_outside_cache",
                         f"The export is built inside pdfintake's cache, not at {dest}; the caller commits it.",
                         path=str(dest))
    if target == _fold(src):
        raise UsageError("output_is_input", f"The export {dest} would replace its input.", path=str(dest))
    if dest.is_dir() or not dest.parent.is_dir():
        raise UsageError("export_dest_invalid", f"The export can't be built at {dest}.", path=str(dest))
    with open(src, "rb") as f:
        head = f.read(_PDF_WINDOW)
    if b"%PDF-" not in head:
        raise UsageError("not_a_pdf", f"The input {src} isn't a PDF.", path=str(src))


_PDF_WINDOW = 1024      # readers accept a %PDF- header within the first 1024 bytes (as intake.sniff does)


def _damaged(src: Path, reason: str) -> Refused:
    return Refused("damaged", _REFUSAL_MESSAGES["damaged"], path=str(src), codes=["damaged"], reason=reason)


# --------------------------------------------------------------------------- refusals

def _walk(root: Any, visit: Callable[[Any], None]) -> None:
    """Visit every dictionary and stream reachable from ``root``, each indirect object once."""
    import pikepdf

    seen: set[tuple[int, int]] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream, pikepdf.Array)):
            if obj.is_indirect:
                if obj.objgen in seen:
                    continue
                seen.add(obj.objgen)
            if isinstance(obj, pikepdf.Array):
                stack.extend(list(obj))
                continue
            visit(obj)
            for key in list(obj.keys()):
                try:
                    stack.append(obj[key])
                except Exception:  # noqa: BLE001 - an unreadable value can't hold a signature we could act on
                    continue


def _report_codes(report: Any) -> set[str]:
    """The export-relevant findings of ``refusals.read``: both of its detectors count."""
    walk = set(getattr(report, "walk_codes", ()) or ())
    raw = set(getattr(report, "raw_markers", ()) or ())
    codes: set[str] = set()
    if report.signed_fields or report.sig_dicts or "signed" in walk or "/ByteRange [" in raw:
        codes.add("signed")
    if report.certified_p is not None or "certified" in walk or "/DocMDP" in raw:
        codes.add("certified")
    if report.usage_rights or "usage_rights" in walk or "/UR3" in raw:
        codes.add("usage_rights")
    if report.doc_timestamps or "timestamped" in walk or "/DocTimeStamp" in raw:
        codes.add("timestamped")
    if report.encrypted or "encrypted" in walk or "/Encrypt" in raw:
        codes.add("encrypted")
    if report.xfa or "xfa" in walk or "/XFA" in raw:
        codes.add("xfa")
    return codes


def _own_codes(pdf: "pikepdf.Pdf") -> set[str]:
    """This module's own detector, on the opened file."""
    import pikepdf

    codes: set[str] = set()
    if pdf.is_encrypted:
        codes.add("encrypted")
    perms = pdf.Root.get("/Perms")
    if isinstance(perms, pikepdf.Dictionary):
        if "/DocMDP" in perms:
            codes.add("certified")
        if "/UR3" in perms or "/UR" in perms:
            codes.add("usage_rights")
    acro = pdf.Root.get("/AcroForm")
    if isinstance(acro, pikepdf.Dictionary) and acro.get("/XFA") is not None:
        codes.add("xfa")

    def visit(obj: Any) -> None:
        if "/ByteRange" in obj and "/Contents" in obj:
            codes.add("timestamped" if obj.get("/Type") == pikepdf.Name.DocTimeStamp else "signed")
        if obj.get("/FT") == pikepdf.Name.Sig and obj.get("/V") is not None:
            codes.add("signed")

    _walk(pdf.trailer, visit)
    return codes


def _refuse(codes: Iterable[str], src: Path) -> None:
    found = set(codes)
    for code in REFUSAL_CODES:
        if code in found:
            raise Refused(code, _REFUSAL_MESSAGES[code], path=str(src), codes=sorted(found))


def _pre_open_refusals(src: Path, *, xfa: bool) -> Any:
    from .. import refusals

    report = refusals.read(src)
    codes = _report_codes(report)
    if not xfa:
        codes.discard("xfa")
    _refuse(codes, src)
    if getattr(refusals, "WALK_OPEN_FAILED", "open_failed") in (getattr(report, "walk_codes", ()) or ()):
        raise _damaged(src, "pikepdf")
    return report


def _open(src: Path, *, xfa: bool) -> "pikepdf.Pdf":
    import pikepdf

    try:
        pdf = pikepdf.open(src)
    except pikepdf.PasswordError:
        raise Refused("encrypted", _REFUSAL_MESSAGES["encrypted"], path=str(src), codes=["encrypted"]) from None
    except pikepdf.PdfError:
        raise _damaged(src, "pikepdf") from None
    try:
        codes = _own_codes(pdf)
        if not xfa:
            codes.discard("xfa")
        _refuse(codes, src)
    except BaseException:
        pdf.close()
        raise
    return pdf


# --------------------------------------------------------------------------- JPEG segments (verification only)

_STANDALONE = set(range(0xD0, 0xD8)) | {0x01}


@dataclass
class _JpegParts:
    meta: list[tuple[str, bytes]]   # application and comment segments: (kind, bytes)
    image: bytes                    # every other segment from SOI to EOI, in order (tables, frame, scans)
    trailing: int                   # bytes after EOI


def _app_kind(marker: int, payload: bytes) -> str:
    if marker == 0xE0:
        return "JFIF" if payload.startswith((b"JFIF\x00", b"JFXX\x00")) else "APP0"
    if marker == 0xE1:
        if payload.startswith(b"Exif\x00"):
            return "Exif"
        return "XMP" if payload.startswith(b"http://ns.adobe.com/") else "APP1"
    if marker == 0xE2:
        if payload.startswith(b"ICC_PROFILE\x00"):
            return "ICC"
        return "MPF" if payload.startswith(b"MPF\x00") else "APP2"
    if marker == 0xED:
        return "IPTC"
    if marker == 0xEE:
        return "Adobe" if payload.startswith(b"Adobe") else "APP14"
    if marker == 0xFE:
        return "COM"
    return f"APP{marker - 0xE0}"


def _jpeg_parts(data: bytes) -> _JpegParts | None:
    """The segments of one JPEG, or None when it can't be walked (then it is left untouched)."""
    if data[:2] != b"\xff\xd8":
        return None
    meta: list[tuple[str, bytes]] = []
    image = bytearray(data[:2])
    i, n = 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            return None
        while i + 1 < n and data[i + 1] == 0xFF:
            i += 1
        if i + 1 >= n:
            return None
        marker = data[i + 1]
        if marker == 0xD9:
            image += data[i:i + 2]
            return _JpegParts(meta, bytes(image), n - (i + 2))
        if marker in _STANDALONE:
            image += data[i:i + 2]
            i += 2
            continue
        if i + 4 > n:
            return None
        end = i + 2 + int.from_bytes(data[i + 2:i + 4], "big")
        if end > n:
            return None
        if marker == 0xDA:
            j = end
            while True:
                j = data.find(b"\xff", j)
                if j < 0 or j + 1 >= n:
                    return None
                nxt = data[j + 1]
                if nxt == 0x00 or nxt == 0xFF or 0xD0 <= nxt <= 0xD7:
                    j += 1 if nxt == 0xFF else 2
                    continue
                break
            end = j
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            meta.append((_app_kind(marker, data[i + 4:end]), data[i:end]))
        else:
            image += data[i:end]
        i = end
    return None


_JPEG_ALLOWED_AFTER = {"JFIF", "ICC", "Adobe"}


def _jpeg_problems(before: _JpegParts, after: _JpegParts | None) -> list[str]:
    if after is None:
        return ["the cleaned JPEG can't be read"]
    problems = []
    if after.image != before.image:
        problems.append("JPEG image data changed")
    if after.trailing:
        problems.append("bytes left after the JPEG's end")
    left = sorted({kind for kind, _ in after.meta if kind not in _JPEG_ALLOWED_AFTER})
    if left:
        problems.append("JPEG metadata left: " + ", ".join(left))
    for kind in ("ICC", "Adobe"):
        if [s for k, s in before.meta if k == kind] != [s for k, s in after.meta if k == kind]:
            problems.append(f"JPEG {kind} segment changed")
    return problems


def _filters(stream: Any) -> list[str]:
    import pikepdf

    f = stream.get("/Filter")
    if f is None:
        return []
    return [str(x) for x in f] if isinstance(f, pikepdf.Array) else [str(f)]


def _dct_parms(stream: Any, filters: list[str]) -> Any:
    import pikepdf

    dp = stream.get("/DecodeParms")
    if isinstance(dp, pikepdf.Array):
        return dp[len(filters) - 1] if len(dp) >= len(filters) else None
    return dp if len(filters) == 1 else None


def _dct_bytes(stream: Any, filters: list[str]) -> bytes:
    """The JPEG inside an image stream whose last filter is DCTDecode (a filter chain such as
    [/FlateDecode /DCTDecode] is read by decoding the leading filters only)."""
    import pikepdf

    if len(filters) == 1:
        return bytes(stream.read_raw_bytes())
    old_filter, old_parms = stream.get("/Filter"), stream.get("/DecodeParms")
    try:
        stream.Filter = pikepdf.Array([pikepdf.Name(x) for x in filters[:-1]])
        if isinstance(old_parms, pikepdf.Array):
            stream.DecodeParms = pikepdf.Array(list(old_parms)[:-1])
        return bytes(stream.read_bytes(decode_level=pikepdf.StreamDecodeLevel.generalized))
    finally:
        stream.Filter = old_filter
        if old_parms is not None:
            stream.DecodeParms = old_parms


def _image_streams(pdf: "pikepdf.Pdf") -> list[Any]:
    """The image XObjects reachable from the trailer: an unreferenced (orphan) image, as an
    incremental edit leaves behind, is dropped when the copy is saved, so it is neither cleaned
    nor compared."""
    import pikepdf

    out: list[Any] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == pikepdf.Name.Image:
            out.append(obj)

    _walk(pdf.trailer, visit)
    return sorted(out, key=lambda s: s.objgen)


def _content_streams(pdf: "pikepdf.Pdf") -> list[Any]:
    """Every reachable stream that holds content operators: page contents, form XObjects (field and
    annotation appearances among them), tiling patterns and Type 3 glyph procedures."""
    import pikepdf

    found: dict[tuple[int, int], Any] = {}

    def add(obj: Any) -> None:
        if isinstance(obj, pikepdf.Stream) and obj.objgen not in found:
            found[obj.objgen] = obj

    for page in pdf.pages:
        contents = page.obj.get("/Contents")
        for s in (list(contents) if isinstance(contents, pikepdf.Array) else [contents]):
            add(s)

    def visit(obj: Any) -> None:
        if isinstance(obj, pikepdf.Stream) and (obj.get("/Subtype") == pikepdf.Name.Form
                                                or obj.get("/PatternType") == 1):
            add(obj)
        ap = obj.get("/AP")
        if isinstance(ap, pikepdf.Dictionary):
            for key in ("/N", "/R", "/D"):
                value = ap.get(key)
                if isinstance(value, pikepdf.Dictionary):
                    for state in list(value.keys()):
                        add(value[state])
                else:
                    add(value)
        procs = obj.get("/CharProcs")
        if isinstance(procs, pikepdf.Dictionary):
            for name in list(procs.keys()):
                add(procs[name])

    _walk(pdf.trailer, visit)
    return [found[k] for k in sorted(found)]


_PDF_WS = b" \t\n\r\x0c\x00"


def _inline_dct(stream: Any) -> tuple[list[Any], list[tuple[int, Any, list[str]]]] | None:
    """The instructions of a content stream and its inline images whose last filter is DCTDecode
    (index, image, filters), or None when it holds none. A stream that names /DCT and can't be
    parsed raises: what can't be read can't be cleaned."""
    import pikepdf

    try:
        data = bytes(stream.read_bytes())
    except Exception:  # noqa: BLE001 - undecodable content draws nothing, in pdfium too
        return None
    if b"/DCT" not in data or b"ID" not in data:
        return None
    try:
        instructions = list(pikepdf.parse_content_stream(stream))
    except Exception as exc:  # noqa: BLE001 - any parse failure means the inline images can't be checked
        raise PdfIntakeError("export_jpeg_unreadable",
                             "Some page content holds an inline image pdfintake can't read, so it can't clean it; nothing was exported.",
                             object=list(stream.objgen), reason=type(exc).__name__) from None
    images: list[tuple[int, Any, list[str]]] = []
    for index, ins in enumerate(instructions):
        if isinstance(ins, pikepdf.ContentStreamInlineImage):
            try:
                filters = [str(f) for f in ins.iimage.filters]
            except Exception:  # noqa: BLE001 - a filter list we can't read: treat it as a chain
                filters = ["?", "/DCTDecode"]
            if filters and filters[-1] == "/DCTDecode":
                images.append((index, ins.iimage, filters))
    return instructions, images


def _inline_jpeg(iimage: Any) -> bytes:
    """An inline image's JPEG, without the whitespace that separates it from EI."""
    return bytes(iimage.read_raw_bytes()).rstrip(_PDF_WS)


def _dct_images(pdf: "pikepdf.Pdf") -> list[tuple[str, bytes | None]]:
    """Every JPEG pdfintake cleans, as (where, bytes): reachable image XObjects whose last filter is
    DCTDecode, and inline images whose only filter is DCTDecode (chained ones are warned about).
    None marks content that couldn't be read."""
    out: list[tuple[str, bytes | None]] = []
    for stream in _image_streams(pdf):
        filters = _filters(stream)
        if filters and filters[-1] == "/DCTDecode":
            out.append((f"object {stream.objgen[0]}", _dct_bytes(stream, filters)))
    for stream in _content_streams(pdf):
        where = f"object {stream.objgen[0]} inline"
        try:
            parsed = _inline_dct(stream)
        except PdfIntakeError:
            out.append((where, None))
            continue
        if parsed is not None:
            out += [(where, _inline_jpeg(img)) for _, img, filters in parsed[1] if len(filters) == 1]
    return out


def _counted(counter: Counter[str], prefix: str) -> list[str]:
    return [f"{prefix}/{kind} ×{n}" for kind, n in sorted(counter.items())]


@dataclass
class _JpegTally:
    removed: Counter[str]
    kept: Counter[str]
    before: list[_JpegParts]
    jpx: int = 0
    inline_chained: int = 0


def _strip_one(raw: bytes, objgen: Any, tally: _JpegTally) -> bytes:
    """One JPEG stripped and checked segment by segment; counts what went and what stayed."""
    from ..intake.jpeg import strip_jpeg

    before = _jpeg_parts(raw)
    if before is None:
        raise PdfIntakeError("export_jpeg_unreadable",
                             "An embedded JPEG can't be read, so pdfintake can't clean it; nothing was exported.",
                             object=list(objgen))
    try:
        new = strip_jpeg(raw, keep_icc=True)
    except Exception as exc:  # noqa: BLE001 - any failure to strip means this JPEG can't be cleaned
        raise PdfIntakeError("export_jpeg_unreadable",
                             "An embedded JPEG can't be cleaned, so nothing was exported.",
                             object=list(objgen), reason=type(exc).__name__) from None
    after = _jpeg_parts(new)
    problems = _jpeg_problems(before, after)
    if problems:
        raise VerifyFailed("export_verify_failed",
                           "Cleaning an embedded JPEG would change more than its metadata; nothing was exported.",
                           checks=[Check("jpeg_image_data", False, "; ".join(problems))])
    assert after is not None
    tally.before.append(before)
    tally.removed.update(Counter(k for k, _ in before.meta) - Counter(k for k, _ in after.meta))
    if before.trailing:
        tally.removed["trailing_data"] += 1
    tally.kept.update(k for k, _ in after.meta if k in ("ICC", "Adobe"))
    return new


def _clean_jpegs(pdf: "pikepdf.Pdf") -> _JpegTally:
    """Strip every reachable JPEG: image XObjects, and inline images in any content stream."""
    import pikepdf

    tally = _JpegTally(Counter(), Counter(), [])
    for stream in _image_streams(pdf):
        filters = _filters(stream)
        if filters and filters[-1] == "/JPXDecode":
            tally.jpx += 1
            continue
        if not filters or filters[-1] != "/DCTDecode":
            continue
        raw = _dct_bytes(stream, filters)
        new = _strip_one(raw, stream.objgen, tally)
        if new != raw:
            parms = _dct_parms(stream, filters)
            if parms is None:
                stream.write(new, filter=pikepdf.Name.DCTDecode)
            else:
                stream.write(new, filter=pikepdf.Name.DCTDecode, decode_parms=parms)

    for stream in _content_streams(pdf):
        parsed = _inline_dct(stream)
        if parsed is None:
            continue
        instructions, images = parsed
        replaced: dict[int, bytes] = {}
        for index, iimage, filters in images:
            if len(filters) != 1:
                tally.inline_chained += 1
                continue
            raw = _inline_jpeg(iimage)
            new = _strip_one(raw, stream.objgen, tally)
            if new == raw:
                continue
            # The instruction as qpdf writes it ends with the image data and EI; swap the data.
            written = bytes(pikepdf.unparse_content_stream([instructions[index]]))
            data = bytes(iimage.read_raw_bytes())
            if not written.endswith(data + b"EI"):
                raise PdfIntakeError("export_jpeg_unreadable",
                                     "An inline JPEG can't be rewritten, so nothing was exported.",
                                     object=list(stream.objgen))
            replaced[index] = written[:len(written) - len(data) - 2] + new + b"\nEI"
        if replaced:
            stream.write(b"\n".join(replaced.get(i) or bytes(pikepdf.unparse_content_stream([ins]))
                                    for i, ins in enumerate(instructions)))
    return tally


# --------------------------------------------------------------------------- pdfium views

def _pdfium_view(path: Path) -> list[tuple[str, str]]:
    """Per page: (sha256 of the 72 dpi grey render with forms and annotations, the page text)."""
    import pypdfium2 as pdfium

    from .render import PDFIUM_LOCK

    out: list[tuple[str, str]] = []
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(path))
        try:
            try:
                doc.init_forms()
            except Exception:  # noqa: BLE001 - a file without a form environment renders without one
                pass
            for index in range(len(doc)):
                page = doc[index]
                try:
                    bitmap = page.render(scale=1, grayscale=True, may_draw_forms=True, draw_annots=True)
                    try:
                        image = bitmap.to_pil()
                        digest = hashlib.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest()
                    finally:
                        bitmap.close()
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_bounded()
                    finally:
                        textpage.close()
                finally:
                    page.close()
                out.append((digest, text))
        finally:
            doc.close()
    return out


def _view_input(src: Path) -> list[tuple[str, str]]:
    """The input's pdfium view; an input pdfium can't open is damaged (Refused, exit 4)."""
    import pypdfium2 as pdfium

    try:
        return _pdfium_view(src)
    except pdfium.PdfiumError:
        raise _damaged(src, "pdfium") from None


def _view_output(dest: Path, checks: list[Check], message: str) -> list[tuple[str, str]]:
    """The built copy's pdfium view; a copy pdfium can't open fails its checks (with ``checks``)."""
    import pypdfium2 as pdfium

    try:
        return _pdfium_view(dest)
    except pdfium.PdfiumError:
        _fail(dest, checks + [Check("renders_identical", False, "the copy can't be opened")], message)
        raise AssertionError("unreachable") from None  # _fail always raises


def _reopen(dest: Path, message: str) -> "pikepdf.Pdf":
    """The built copy opened with pikepdf; a copy pikepdf can't open fails its checks."""
    import pikepdf

    try:
        return pikepdf.open(dest)
    except (pikepdf.PdfError, OSError):
        _fail(dest, [Check("renders_identical", False, "the copy can't be opened")], message)
        raise AssertionError("unreachable") from None  # _fail always raises


def _fail(dest: Path, checks: list[Check], message: str) -> None:
    try:
        dest.unlink(missing_ok=True)
    except OSError:
        pass
    raise VerifyFailed("export_verify_failed", message, checks=checks)


def _check_renders(before: list[tuple[str, str]], after: list[tuple[str, str]], skip: set[int]) -> list[Check]:
    checks: list[Check] = []
    if len(before) != len(after):
        return [Check("renders_identical", False, f"{len(after)} pages, expected {len(before)}")]
    for index, ((a, _), (b, _)) in enumerate(zip(before, after)):
        if index not in skip and a != b:
            checks.append(Check("renders_identical", False, "the page renders differently", index + 1))
    return checks


def _nonspace(text: str) -> Counter[str]:
    return Counter(ch for ch in text if not ch.isspace())


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted(set(items))


# --------------------------------------------------------------------------- copies

def _is_searchable_copy(pdf: "pikepdf.Pdf") -> bool:
    """Whether the document XMP carries pdfintake's ``DerivedFrom`` block (DESIGN §10.2, §11.1).

    One detector for the block: ``meta.derived_from``, which decodes UTF-8 and BOM-marked UTF-16
    packets, parses them and matches the block by its namespace. So XMP's standard
    ``xmpMM:DerivedFrom`` (which editing apps write), or the words in a value or a comment, never
    count (decided 2026-11-05).
    """
    from . import meta

    return meta.derived_from(pdf) is not None


# --------------------------------------------------------------------------- clean

def _claims_conformance(pdf: "pikepdf.Pdf", report: Any) -> bool:
    import pikepdf

    if getattr(report, "pdfa", None) or getattr(report, "pdfua", None):
        return True
    meta = pdf.Root.get("/Metadata")
    if isinstance(meta, pikepdf.Stream):
        try:
            data = bytes(meta.read_bytes())
        except Exception:  # noqa: BLE001 - an undecodable packet is removed all the same
            return False
        return bool(re.search(rb"pdf(?:a|ua)id:part\s*[=>]", data))
    return False


def _clean_warnings(pdf: "pikepdf.Pdf", report: Any) -> list[str]:
    import pikepdf

    warnings: list[str] = []
    if getattr(report, "repaired", False):
        warnings.append("input_repaired")
    if "/OCProperties" in pdf.Root:
        warnings.append("optional_content_kept")
    authors = embedded = 0

    def visit(obj: Any) -> None:
        nonlocal authors, embedded
        if "/EF" in obj:
            embedded += 1
        if (obj.get("/Type") == pikepdf.Name.Annot or "/Subtype" in obj and "/Rect" in obj) and "/T" in obj:
            if obj.get("/Subtype") not in (pikepdf.Name.Widget, pikepdf.Name.Link, pikepdf.Name.Popup):
                authors += 1

    _walk(pdf.Root, visit)
    if embedded:
        warnings.append("embedded_files_kept")
    if authors:
        warnings.append("annotation_authors_kept")
    return warnings


def _leftovers(path: Path, keep_dates: bool, images_before: list[bytes], message: str) -> list[Check]:
    """What a cleaned file must not hold, and its JPEGs' image data against ``images_before``."""
    import pikepdf

    checks: list[Check] = []
    allowed_info = {"/Title"} | ({"/CreationDate", "/ModDate"} if keep_dates else set())
    with _reopen(path, message) as out:
        info = out.trailer.get("/Info")
        if isinstance(info, pikepdf.Dictionary):
            extra = sorted(set(str(k) for k in info.keys()) - allowed_info)
            if extra:
                checks.append(Check("info_clean", False, "Info still has " + ", ".join(extra)))
        found: set[str] = set()

        def visit(obj: Any) -> None:
            for key in ("/Metadata", "/PieceInfo"):
                if key in obj:
                    found.add(key)
            if obj.get("/Type") == pikepdf.Name.Metadata:
                found.add("XMP stream")

        _walk(out.trailer, visit)
        for page in out.pages:
            if "/Thumb" in page.obj:
                found.add("/Thumb")
            if not keep_dates and "/LastModified" in page.obj:
                found.add("/LastModified")
        if found:
            checks.append(Check("metadata_removed", False, "still present: " + ", ".join(sorted(found))))
        images_after: list[bytes] = []
        for where, data in _dct_images(out):
            parts = _jpeg_parts(data) if data is not None else None
            bad = sorted({k for k, _ in parts.meta if k not in _JPEG_ALLOWED_AFTER}) if parts else ["unreadable"]
            if bad or (parts is not None and parts.trailing):
                checks.append(Check("jpeg_metadata_removed", False, f"{where}: " + ", ".join(bad or ["trailing data"])))
            if parts is not None:
                images_after.append(parts.image)
    if sorted(images_after) != sorted(images_before):
        checks.append(Check("images_identical", False, "embedded JPEG image data changed"))
    return checks


def export_clean(src: Path, dest: Path, *, provenance: str, strip_dates: bool = False) -> ExportReport:
    """A copy to share: identifying metadata removed, content, layer, forms and ICC kept.

    Refuses signed, certified, usage-rights, timestamped and encrypted files and the provenances
    ``counterparty`` and ``evidence`` (Refused, nothing built). ``dest`` must lie in the cache.
    """
    from . import meta

    src, dest = Path(src), Path(dest)
    _check_args(src, dest, provenance)
    report = _pre_open_refusals(src, xfa=False)
    keep_dates = not strip_dates
    view_before = _view_input(src)

    with _open(src, xfa=False) as pdf:
        warnings = _clean_warnings(pdf, report)
        if _is_searchable_copy(pdf):            # read before strip_for_export removes the XMP
            warnings.append("exporting_a_copy")
        if _claims_conformance(pdf, report):
            warnings.append("conformance_claim_dropped")
        info_before = meta.snapshot(pdf).info
        removed = list(meta.strip_for_export(pdf, keep_dates=keep_dates))
        tally = _clean_jpegs(pdf)
        if tally.jpx:
            warnings.append("jpx_metadata_kept")
        if tally.inline_chained:
            warnings.append("inline_jpeg_metadata_kept")
        removed += _counted(tally.removed, "jpeg")
        info_after = meta.snapshot(pdf).info
        kept = [f"Info/{k}" for k in sorted(info_after) if k in info_before]
        kept += _counted(tally.kept, "jpeg")
        # Both halves of the file ID are derived from the cleaned content, so the copy can't be
        # matched to the original by ID[0] (qpdf keeps the original's first half otherwise).
        if "/ID" in pdf.trailer:
            del pdf.trailer["/ID"]
        # Every save of a copy names its encryption and XMP handling rather than relying on
        # pikepdf's defaults: never encrypted, the XMP never rewritten (decided 2026-11-05).
        pdf.save(dest, encryption=False, deterministic_id=True, fix_metadata_version=False)

    failed = "The cleaned copy didn't pass its checks; nothing was exported."
    checks = _leftovers(dest, keep_dates, [part.image for part in tally.before], failed)
    view_after = _view_output(dest, checks, failed)
    checks += _check_renders(view_before, view_after, set())
    for index, ((_, a), (_, b)) in enumerate(zip(view_before, view_after)):
        if a != b:
            checks.append(Check("text_identical", False, "the page text changed", index + 1))
    if checks:
        _fail(dest, checks, failed)

    if any(text.strip() for _, text in view_after):
        kept.append("text_layer")
    return ExportReport(dest=dest, removed=removed, kept=kept, warnings=_sorted_unique(warnings))


# --------------------------------------------------------------------------- flatten

_F_INVISIBLE, _F_HIDDEN, _F_NOVIEW = 1, 2, 32


def _annots(page: Any) -> list[Any]:
    import pikepdf

    raw = page.obj.get("/Annots")
    if not isinstance(raw, pikepdf.Array):
        return []
    return [a for a in raw if isinstance(a, pikepdf.Dictionary)]


def _subtype(annot: Any) -> str:
    sub = annot.get("/Subtype")
    return str(sub)[1:] if sub is not None else "unknown"


def _has_appearance(annot: Any) -> bool:
    import pikepdf

    ap = annot.get("/AP")
    if not isinstance(ap, pikepdf.Dictionary):
        return False
    normal = ap.get("/N")
    if isinstance(normal, pikepdf.Stream):
        return True
    if isinstance(normal, pikepdf.Dictionary):          # appearance states: the one /AS names
        state = annot.get("/AS")
        return state is not None and isinstance(normal.get(str(state)), pikepdf.Stream)
    return False


def _flag(annot: Any) -> int:
    try:
        return int(annot.get("/F", 0))
    except (TypeError, ValueError):
        return 0


def _decoded(stream: Any) -> bytes:
    try:
        return bytes(stream.read_bytes())
    except Exception:  # noqa: BLE001 - an undecodable stream is compared by its raw bytes
        return bytes(stream.read_raw_bytes())


def _page_fingerprint(page: Any) -> tuple[Counter[str], dict[str, str]]:
    """The page's content streams (hashes) and its form and image XObjects (name -> hash)."""
    import pikepdf

    contents = page.obj.get("/Contents")
    streams = list(contents) if isinstance(contents, pikepdf.Array) else ([contents] if contents is not None else [])
    content = Counter(hashlib.sha256(_decoded(s)).hexdigest() for s in streams if isinstance(s, pikepdf.Stream))
    xobjects: dict[str, str] = {}
    resources = page.obj.get("/Resources")
    xo = resources.get("/XObject") if isinstance(resources, pikepdf.Dictionary) else None
    if isinstance(xo, pikepdf.Dictionary):
        for name in xo.keys():
            obj = xo[name]
            if isinstance(obj, pikepdf.Stream):
                xobjects[str(name)] = hashlib.sha256(_decoded(obj)).hexdigest()
    return content, xobjects


def export_flatten(src: Path, dest: Path, *, provenance: str) -> ExportReport:
    """A non-editable copy: fields and annotations baked into page content, the text layer kept.

    Refuses what ``export_clean`` refuses, and XFA forms. An annotation that can't be baked (it
    has no appearance at all) stops the export with ``flatten_incomplete``. ``dest`` must lie in
    the cache.
    """
    import pikepdf

    from . import meta

    src, dest = Path(src), Path(dest)
    _check_args(src, dest, provenance)
    report = _pre_open_refusals(src, xfa=True)
    view_before = _view_input(src)
    warnings: list[str] = []
    if getattr(report, "repaired", False):
        warnings.append("input_repaired")

    with _open(src, xfa=True) as pdf:
        if _is_searchable_copy(pdf):
            warnings.append("exporting_a_copy")
        meta_before = meta.snapshot(pdf)
        prints_before = [_page_fingerprint(page) for page in pdf.pages]
        acro = pdf.Root.get("/AcroForm")
        need_appearances = isinstance(acro, pikepdf.Dictionary) and bool(acro.get("/NeedAppearances", False))
        if isinstance(acro, pikepdf.Dictionary) and not need_appearances and any(
                _subtype(a) == "Widget" and not _has_appearance(a) for page in pdf.pages for a in _annots(page)):
            # qpdf generates appearances only when NeedAppearances is set; the key goes with the
            # AcroForm below. Every field is then redrawn, so every page with a field is generated.
            acro.NeedAppearances = True
            need_appearances = True
        baked: Counter[str] = Counter()
        not_shown = 0
        generated_pages: set[int] = set()
        for index, page in enumerate(pdf.pages):
            for annot in _annots(page):
                sub, flags = _subtype(annot), _flag(annot)
                if sub == "Widget" and (need_appearances or not _has_appearance(annot)):
                    generated_pages.add(index)
                if sub in ("Link", "Popup"):
                    continue
                if flags & (_F_INVISIBLE | _F_HIDDEN | _F_NOVIEW):
                    not_shown += 1
                else:
                    baked[sub] += 1
        if generated_pages:
            pdf.generate_appearance_streams()
            warnings.append("appearances_generated")
        pdf.flatten_annotations("screen")

        popups = 0
        links = 0
        left: Counter[str] = Counter()
        for page in pdf.pages:
            remaining = []
            for annot in _annots(page):
                sub = _subtype(annot)
                if sub == "Popup":
                    popups += 1
                    continue
                if sub == "Link":
                    links += 1
                else:
                    left[sub] += 1
                remaining.append(annot)
            if "/Annots" in page.obj:
                if remaining:
                    page.obj.Annots = pikepdf.Array(remaining)
                else:
                    del page.obj["/Annots"]
        if left:
            raise PdfIntakeError("flatten_incomplete",
                                 "Some annotations have no drawn appearance, so they can't be baked into the page; nothing was exported.",
                                 path=str(src), annotations=dict(sorted(left.items())))
        removed = _counted(baked, "annotations")
        if not_shown:
            removed.append(f"annotations/not_shown ×{not_shown}")
            warnings.append("annotations_not_shown_dropped")
        if popups:
            removed.append(f"annotations/Popup ×{popups}")
        if "/AcroForm" in pdf.Root:
            del pdf.Root["/AcroForm"]
            removed.append("AcroForm")
        elif isinstance(acro, pikepdf.Dictionary):
            removed.append("AcroForm")
        # The metadata is kept byte for byte: pikepdf's default fix_metadata_version would rewrite
        # the XMP packet into its own form (and fail metadata_kept on any packet it didn't write).
        # The copy is never encrypted, named explicitly rather than left to the default (decided 2026-11-05).
        pdf.save(dest, encryption=False, deterministic_id=True, fix_metadata_version=False)

    failed = "The flattened copy didn't pass its checks; nothing was exported."
    checks: list[Check] = []
    with _reopen(dest, failed) as out:
        meta_after = meta.snapshot(out)
        if "/AcroForm" in out.Root:
            checks.append(Check("no_form_fields", False, "AcroForm still present"))
        for index, page in enumerate(out.pages):
            subs = {_subtype(a) for a in _annots(page)} - {"Link"}
            if subs:
                checks.append(Check("annotations_baked", False, "still has " + ", ".join(sorted(subs)), index + 1))
            content, xobjects = _page_fingerprint(page)
            old_content, old_xobjects = prints_before[index] if index < len(prints_before) else (Counter(), {})
            if old_content - content or any(xobjects.get(k) != v for k, v in old_xobjects.items()):
                checks.append(Check("layer_kept", False, "the original page content changed", index + 1))
    view_after = _view_output(dest, checks, failed)
    checks += _check_renders(view_before, view_after, generated_pages)
    for index, ((_, a), (_, b)) in enumerate(zip(view_before, view_after)):
        if _nonspace(a) - _nonspace(b):
            checks.append(Check("text_kept", False, "page text was lost", index + 1))
    md = meta.diff(meta_before, meta_after)
    if md.changed or md.dropped:
        checks.append(Check("metadata_kept", False, "changed: " + ", ".join(md.changed + md.dropped)))
    if checks:
        _fail(dest, checks, failed)

    kept = list(md.kept)
    if any(text.strip() for _, text in view_before):
        kept.append("text_layer")
    if links:
        kept.append(f"annotations/Link ×{links}")
    return ExportReport(dest=dest, removed=removed, kept=kept, warnings=_sorted_unique(warnings))
