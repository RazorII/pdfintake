"""The verify suite: every build converges here before anything is written.

Each check compares the built PDF with its reference (the input PDF, or the assembled PDF for
image inputs) and returns a ``Check``. ``verify_build`` runs them all; ``flow`` raises
``VerifyFailed`` when any comes back ``ok=False``, and then nothing is written.

A check never raises for what it finds in a file: a file it can't open or parse gives a failed
Check whose detail says why. Every pypdfium2 call holds ``render.PDFIUM_LOCK``.

Conventions:

* Every page key the checks take (``turns``, ``touched``, ``expected``) is a 0-based page index,
  as everywhere in pdfintake. ``Check.page`` counts from 1, like every page number in a decision
  record (``flow`` copies Checks into ``verify[]`` unchanged), and so does "page N" in a detail.
* Details are core-worded and never carry text the file controls: no document text, no qpdf or
  exception messages, no names or keys from the file (beyond the standard Info keys), and no
  paths. Counts, ratios, page numbers and closed categories only.
* ``metadata_kept`` takes the names in ``allowed`` as listed in ``META_NAMES``.

Copies (§10.2, decided 2026-11-05): in copy mode ``BuildContext.original_sha256`` is set and the
reference is the work copy before the layer. ``renders_identical`` draws form appearances on both
sides, so each signature stamp ``strip.flatten_signatures`` drew into its page must look exactly as
the widget did. ``images_identical`` then counts, for a signature widget of the reference, only the
appearance the flattening draws. Two checks run on copies only: ``not_encrypted`` and
``derived_from_present``. ``no_signature_material`` runs on every build, with
``refusals.output_signature_material`` (its raw scan and its walk) as well as its own walk.
"""

from __future__ import annotations

import contextlib
import decimal
import functools
import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..intake.assemble import AssembledPage

# text_retention thresholds
DOC_MIN_RETENTION = 0.98
PAGE_MIN_RETENTION = 0.90
PAGE_MIN_CHARS = 20                 # below this a page is too sparse for its own ratio to mean anything

RENDER_DPI = 72.0
RENDER_MAX_PIXELS = 60_000_000      # a 72 dpi render of a huge page is scaled down (both sides alike)
STRUCTURE_RENDER_SCALE = 0.25       # structure_ok only proves every page renders

IMAGE_FILTERS = frozenset({"/DCTDecode", "/JBIG2Decode", "/JPXDecode", "/CCITTFaxDecode", "/DCT", "/CCF"})
JBIG2 = "/JBIG2Decode"

# Names metadata_kept accepts in ``allowed`` (the names meta.diff reports):
#   an Info key as "Info/Producer" (or "Producer", or "/Producer"), "Info" for any change to the
#   Info dictionary, and "XMP" for any change to the XMP packet.
# A same-length patch of the XMP pdf:PDFVersion value to the built file's header version (shrink's
# patch_xmp_pdfversion) is always accepted and noted; the header version itself isn't metadata.
META_NAMES = ("Info", "XMP", "Info/<Key>")
STANDARD_INFO_KEYS = frozenset({"Title", "Author", "Subject", "Keywords", "Creator", "Producer",
                                "CreationDate", "ModDate", "Trapped"})

CHECK_NAMES = ("text_retention", "off_page_glyphs", "renders_identical", "images_identical",
               "passthrough_identical", "upright", "no_jbig2_written", "no_signature_material",
               "metadata_kept", "structure_ok", "not_encrypted", "derived_from_present",
               "text_layer_identical")

# refusals.output_signature_material's labels, as no_signature_material words them
_OUTPUT_SIG_WORDS = {"/ByteRange": "/ByteRange", "/Perms": "/Perms", "/Sig field": "a signed /Sig field",
                     "unreadable": "a part the scan couldn't read"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    page: int | None = None


@dataclass
class BuildContext:
    reference: Path            # the input PDF, or the assembled PDF for image inputs
    built: Path
    turns: Mapping[int, int]                          # page -> clockwise turn baked or cleared
    touched: frozenset[int]                           # pages baked, cleared or layered
    expected: Mapping[int, str]                       # page -> the exact text the layer wrote
    assembled: Sequence["AssembledPage"] | None
    allowed_meta_changes: frozenset[str]
    original_sha256: str | None = None                # copy mode: the original's SHA-256 (§10.2)
    original_name: str | None = None                  # copy mode: the original's file name, when known


# --------------------------------------------------------------------------- shared helpers


def _guard(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Turn an exception from a malformed file into a failed Check (or a one-item list of one)."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except ImportError:
                raise
            except Exception as exc:              # noqa: BLE001 - a check reports, it never crashes the build
                failed = Check(name, False, f"the check could not run ({_exc_kind(exc)})")
                return [failed] if name == "text_retention" else failed
        return inner

    return wrap


def _exc_kind(exc: BaseException) -> str:
    """The exception's class name only (never its message, which can quote the file or a path)."""
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) else "error"


def _pg(index: int) -> int:
    """A 0-based page index as ``Check.page`` carries it (counting from 1, as in the record)."""
    return int(index) + 1


def _human(page: int) -> str:
    return f"page {page + 1}"


def _pages_text(pages: Sequence[int], limit: int = 8) -> str:
    shown = ", ".join(str(p + 1) for p in pages[:limit])
    more = f" and {len(pages) - limit} more" if len(pages) > limit else ""
    word = "page" if len(pages) == 1 else "pages"
    return f"{word} {shown}{more}"


def _floor3(value: float) -> str:
    """Three decimals, rounded down (so 0.9799 never shows as a passing 0.980); 0.98 stays 0.980."""
    return f"{math.floor(value * 1000 + 1e-9) / 1000:.3f}"


def _lock() -> Any:
    from .render import PDFIUM_LOCK

    return PDFIUM_LOCK


@contextlib.contextmanager
def _pdfium(path: Path, *, forms: bool = False) -> Iterator[Any]:
    """A pypdfium2 document read through a file object (safe for any path on any OS).

    ``forms=True`` initialises the form environment first, so renders can draw widget appearances.
    """
    import pypdfium2 as pdfium

    handle = open(path, "rb")
    try:
        doc = pdfium.PdfDocument(handle, autoclose=True)
    except BaseException:
        handle.close()
        raise
    try:
        if forms:
            doc.init_forms()
        yield doc
    finally:
        doc.close()


@contextlib.contextmanager
def _pike(path: Path) -> Iterator[Any]:
    import pikepdf

    with pikepdf.open(path) as pdf:
        yield pdf


def _inherited(obj: Any, key: str) -> Any:
    seen: set[tuple[int, int]] = set()
    node = obj
    while node is not None:
        value = node.get(key)
        if value is not None:
            return value
        og = node.objgen
        if og != (0, 0):
            if og in seen:
                return None
            seen.add(og)
        node = node.get("/Parent")
    return None


def _norm_chars(text: str) -> Counter[str]:
    return Counter(c for c in unicodedata.normalize("NFC", text or "").casefold() if not c.isspace())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- text


@_guard("text_retention")
def text_retention(built: Path, expected: Mapping[int, str]) -> list[Check]:
    """The layer text pdfium hands back from each page's CropBox, against what was written.

    A character-multiset ratio (case-folded, NFC, whitespace ignored): at least 98% for the document
    and 90% on every page with 20 or more expected characters. Zero OCR text on every page fails.
    The first Check is the document's; a failed page adds its own Check.
    """
    expected = {int(i): t for i, t in expected.items()}
    pages = sorted(expected)
    per_page: dict[int, tuple[int, int]] = {}
    missing: list[int] = []
    with _lock(), _pdfium(Path(built)) as doc:
        count = len(doc)
        for i in pages:
            want = _norm_chars(expected[i])
            total = sum(want.values())
            if not 0 <= i < count:
                missing.append(i)
                per_page[i] = (0, total)
                continue
            page = doc[i]
            try:
                textpage = page.get_textpage()
                try:
                    got = _norm_chars(textpage.get_text_bounded(*page.get_cropbox()))
                finally:
                    textpage.close()
            finally:
                page.close()
            per_page[i] = (sum((want & got).values()), total)

    total_all = sum(t for _k, t in per_page.values())
    if total_all == 0:
        what = f"{len(pages)} layered page(s)" if pages else "any page"
        return [Check("text_retention", False, f"no OCR text at all on {what}: the read found nothing")]
    kept_all = sum(k for k, _t in per_page.values())
    doc_ratio = kept_all / total_all
    failed = [i for i, (k, t) in per_page.items() if t >= PAGE_MIN_CHARS and k / t < PAGE_MIN_RETENTION]
    failed += [i for i in missing if i not in failed]
    rated = [(k / t, i) for i, (k, t) in per_page.items() if t > 0]
    worst_ratio, worst_page = min(rated)
    ok = doc_ratio >= DOC_MIN_RETENTION and not failed
    detail = f"{_floor3(doc_ratio)} document, worst {_human(worst_page)} at {_floor3(worst_ratio)}"
    if doc_ratio < DOC_MIN_RETENTION:
        detail += f"; the document needs {DOC_MIN_RETENTION:.2f}"
    if failed:
        detail += f"; {_pages_text(sorted(failed))} below {PAGE_MIN_RETENTION:.2f}"
    checks = [Check("text_retention", ok, detail)]
    for i in sorted(failed):
        k, t = per_page[i]
        if i in missing:
            checks.append(Check("text_retention", False, f"{_human(i)} is not in the built file", _pg(i)))
        else:
            checks.append(Check("text_retention", False,
                                f"{_floor3(k / t)}: {k} of {t} characters came back, below {PAGE_MIN_RETENTION:.2f}",
                                _pg(i)))
    return checks


def _box_off(box: tuple[float, float, float, float], crop: tuple[float, float, float, float]) -> bool:
    left, bottom, right, top = box
    c_left, c_bottom, c_right, c_top = crop
    return right < c_left or left > c_right or top < c_bottom or bottom > c_top


@_guard("off_page_glyphs")
def off_page_glyphs(built: Path, *, pages: Collection[int] | None = None) -> Check:
    """Characters whose box lies wholly outside the page's CropBox must be 0.

    ``pages`` limits the check to those pages (``verify_build`` passes the layered and stripped
    ones, since the other pages carry the input's own text); None checks every page.
    """
    import pypdfium2 as pdfium

    bad: dict[int, int] = {}
    checked = 0
    with _lock(), _pdfium(Path(built)) as doc:
        count = len(doc)
        wanted = range(count) if pages is None else sorted(int(i) for i in pages)
        for i in wanted:
            if not 0 <= i < count:
                bad[i] = bad.get(i, 0)
                continue
            checked += 1
            page = doc[i]
            try:
                crop = page.get_cropbox()
                textpage = page.get_textpage()
                try:
                    n = 0
                    for k in range(textpage.count_chars()):
                        if pdfium.raw.FPDFText_IsGenerated(textpage.raw, k) == 1:
                            continue
                        code = pdfium.raw.FPDFText_GetUnicode(textpage.raw, k)
                        if code == 0 or chr(code).isspace():
                            continue
                        if _box_off(textpage.get_charbox(k), crop):
                            n += 1
                    if n:
                        bad[i] = n
                finally:
                    textpage.close()
            finally:
                page.close()
    out_of_range = [i for i, n in bad.items() if n == 0]
    if out_of_range:
        return Check("off_page_glyphs", False, f"{_pages_text(sorted(out_of_range))} not in the built file",
                     _pg(min(out_of_range)))
    if bad:
        first = min(bad)
        total = sum(bad.values())
        return Check("off_page_glyphs", False,
                     f"{total} character(s) outside the CropBox on {_pages_text(sorted(bad))}", _pg(first))
    return Check("off_page_glyphs", True, f"0 characters outside the CropBox on {checked} page(s)")


# --------------------------------------------------------------------------- pixels


def _render_gray(page: Any, scale: float, rotation: int, *, forms: bool = False) -> Any:
    """``forms=True`` draws form-field widgets too; the document must have had ``init_forms()``."""
    import numpy as np

    bitmap = page.render(scale=scale, rotation=rotation, grayscale=True, may_draw_forms=forms)
    try:
        width, height, stride = bitmap.width, bitmap.height, bitmap.stride
        pixels = np.frombuffer(bitmap.buffer, dtype=np.uint8, count=stride * height).reshape(height, stride)
        return pixels[:, :width].copy()
    finally:
        bitmap.close()


def _size(page: Any) -> tuple[float, float]:
    w, h = page.get_size()
    return max(float(w), 0.0), max(float(h), 0.0)


def _capped_scale(scale: float, size: tuple[float, float]) -> float:
    """``scale``, lowered so a page of ``size`` points renders to at most RENDER_MAX_PIXELS pixels."""
    w, h = size
    if w <= 0 or h <= 0:
        return scale
    while math.ceil(w * scale) * math.ceil(h * scale) > RENDER_MAX_PIXELS:
        scale = min(scale * 0.999, math.sqrt(RENDER_MAX_PIXELS / (w * h)))
    return scale


@_guard("renders_identical")
def renders_identical(reference: Path, built: Path, turns: Mapping[int, int]) -> Check:
    """Every page of ``built`` renders exactly like ``reference`` turned by ``turns[page]`` clockwise.

    pdfium, 72 dpi, grayscale, with form-field widgets drawn (what a viewer shows); the reference
    page is rendered with pdfium's own rotation = turn, the built page with none. Equality is exact.
    A huge page is scaled down to ``RENDER_MAX_PIXELS``, from the larger of the two pages.
    """
    import numpy as np

    for page, turn in turns.items():
        if int(turn) % 90:
            return Check("renders_identical", False, f"{_human(int(page))} has a turn of {int(turn)}, not a quarter turn",
                         _pg(int(page)))
    differ: list[tuple[int, str]] = []
    turns = {int(k): v for k, v in turns.items()}
    with _lock(), _pdfium(Path(reference), forms=True) as ref, _pdfium(Path(built), forms=True) as out:
        if len(ref) != len(out):
            return Check("renders_identical", False,
                         f"the built file has {len(out)} page(s), the reference {len(ref)}")
        count = len(ref)
        for i in range(count):
            turn = int(turns.get(i, 0)) % 360
            rp, bp = ref[i], out[i]
            try:
                scale = min(_capped_scale(RENDER_DPI / 72.0, _size(rp)), _capped_scale(RENDER_DPI / 72.0, _size(bp)))
                a = _render_gray(rp, scale, turn, forms=True)
                b = _render_gray(bp, scale, 0, forms=True)
            finally:
                rp.close()
                bp.close()
            if a.shape != b.shape:
                differ.append((i, f"{a.shape[1]}x{a.shape[0]} px against {b.shape[1]}x{b.shape[0]} px"))
            elif not np.array_equal(a, b):
                delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
                differ.append((i, f"{int(np.count_nonzero(delta))} pixel(s) differ, by up to {int(delta.max())}"))
    if differ:
        first, why = differ[0]
        pages = [p for p, _w in differ]
        return Check("renders_identical", False,
                     f"{_pages_text(pages)} render differently ({_human(first)}: {why})", _pg(first))
    return Check("renders_identical", True, f"{count}/{count} pages")


# --------------------------------------------------------------------------- images


def _as_list(obj: Any) -> list[Any]:
    import pikepdf

    if obj is None:
        return []
    if isinstance(obj, pikepdf.Array):
        return list(obj)
    return [obj]


def _canon(obj: Any, depth: int = 0) -> Any:
    """A hashable, number-normalised form of a PDF object (streams by the hash of their data)."""
    import pikepdf

    if depth > 16:
        return ("deep",)
    if obj is None:
        return None
    if isinstance(obj, bool):
        return ("b", obj)
    if isinstance(obj, (int, float, decimal.Decimal)):
        return ("n", float(obj))
    if isinstance(obj, pikepdf.Name):
        return ("N", str(obj))
    if isinstance(obj, pikepdf.String):
        return ("S", bytes(obj))
    if isinstance(obj, pikepdf.Stream):
        try:
            data = obj.read_bytes()
        except Exception:                          # noqa: BLE001 - an image-coded stream: its stored bytes
            data = obj.read_raw_bytes()
        extra = {k: v for k, v in obj.stream_dict.items()
                 if k not in ("/Length", "/Filter", "/DecodeParms", "/DL") and v is not None}
        return ("stream", _sha(data), tuple(sorted((k, _canon(v, depth + 1)) for k, v in extra.items())))
    if isinstance(obj, pikepdf.Array):
        return ("A", tuple(_canon(v, depth + 1) for v in obj))
    if isinstance(obj, pikepdf.Dictionary):
        # items() lists null-valued keys, which mean no key (ISO 32000-1 §7.3.7) and which qpdf never writes
        return ("D", tuple(sorted((str(k), _canon(v, depth + 1)) for k, v in obj.items() if v is not None)))
    try:
        return ("?", bytes(obj.unparse()))
    except Exception:                              # noqa: BLE001
        return ("?", repr(obj))


def _image_data(pdf: Any, img: Any) -> tuple[str, bytes]:
    """("encoded <filter>", stored image-coded bytes) or ("samples", decoded samples)."""
    import pikepdf

    filters = [str(f) for f in _as_list(img.get("/Filter"))]
    if filters and filters[-1] in IMAGE_FILTERS:
        lead = filters[:-1]
        raw = img.read_raw_bytes()
        if not lead:
            return f"encoded {filters[-1]}", raw
        parms = _as_list(img.get("/DecodeParms"))
        tmp = pikepdf.Stream(pdf, raw)
        tmp.Filter = pikepdf.Array([pikepdf.Name(f) for f in lead])
        if parms:
            lead_parms = parms[:len(lead)]
            tmp.DecodeParms = pikepdf.Array([p if p is not None else pikepdf.Dictionary() for p in lead_parms])
        return f"encoded {filters[-1]}", tmp.read_bytes()
    return "samples", img.read_bytes()


def _image_fp(pdf: Any, img: Any, memo: dict[tuple[int, int], tuple[Any, ...]], depth: int = 0) -> tuple[Any, ...]:
    """What an image looks like, independent of where it sits in the file."""
    import pikepdf

    og = img.objgen
    if og != (0, 0) and og in memo:
        return memo[og]
    d = img.stream_dict
    try:
        kind, data = _image_data(pdf, img)
        digest = _sha(data)
    except Exception as exc:                       # noqa: BLE001 - undecodable data is itself a fingerprint
        kind, digest = f"undecodable {type(exc).__name__}", _sha(img.read_raw_bytes())
    filters = [str(f) for f in _as_list(d.get("/Filter"))]
    parms = _as_list(d.get("/DecodeParms"))
    image_parms = _canon(parms[-1]) if kind.startswith("encoded") and len(parms) == len(filters) and parms else None
    smask = d.get("/SMask")
    mask = d.get("/Mask")
    fp = (
        kind, digest, image_parms,
        _canon(d.get("/Width")), _canon(d.get("/Height")), _canon(d.get("/BitsPerComponent")),
        _canon(d.get("/ColorSpace")), _canon(d.get("/Decode")), _canon(d.get("/ImageMask")),
        _image_fp(pdf, smask, memo, depth + 1) if isinstance(smask, pikepdf.Stream) and depth < 4 else _canon(smask),
        _image_fp(pdf, mask, memo, depth + 1) if isinstance(mask, pikepdf.Stream) and depth < 4 else _canon(mask),
        _canon(d.get("/SMaskInData")),
    )
    if og != (0, 0):
        memo[og] = fp
    return fp


def _describe(fp: tuple[Any, ...]) -> str:
    kind = fp[0]
    if kind == "inline":
        return "an inline image"
    w = fp[3][1] if isinstance(fp[3], tuple) and len(fp[3]) == 2 else "?"
    h = fp[4][1] if isinstance(fp[4], tuple) and len(fp[4]) == 2 else "?"
    size = f"{int(w)}x{int(h)}" if isinstance(w, float) and isinstance(h, float) else "?"
    what = kind.replace("encoded /", "").replace("samples", "decoded samples")
    return f"{size} ({what})"


class _Drawn:
    """The images each page draws, inline images included.

    Walked from the page content through every way content can paint an image: forms drawn with
    ``Do``, tiling patterns selected with ``scn``/``SCN``, soft-mask groups set with ``gs``, the
    glyph procedures of Type 3 fonts selected with ``Tf``, and every annotation appearance (all
    states). Each is walked with its own Resources, or its caller's when it has none.
    """

    def __init__(self, pdf: Any, *, flattened_signatures: bool = False) -> None:
        self.pdf = pdf
        self.flattened_signatures = flattened_signatures
        self.memo: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.cache: dict[Any, tuple[Counter[tuple[Any, ...]], list[Any]]] = {}

    def page(self, index: int) -> tuple[Counter[tuple[Any, ...]], list[Any]]:
        """(images drawn by the page's content, image streams in drawing order); appearances included."""
        import pikepdf

        page = self.pdf.pages[index]
        found: Counter[tuple[Any, ...]] = Counter()
        streams: list[Any] = []
        if page.obj.get("/Contents") is not None:
            self._walk(page, _inherited(page.obj, "/Resources"), found, streams, set(), 0)
        annots = page.obj.get("/Annots")
        if isinstance(annots, pikepdf.Array):
            for annot in annots:
                for ap in self._annot_appearances(annot):
                    self._holder(ap, ("ap",), _own_res(ap), None, found, streams, set(), 0)
        return found, streams

    def _annot_appearances(self, annot: Any) -> list[Any]:
        """Every appearance; for a signature widget whose stamp was flattened, only the one drawn."""
        if self.flattened_signatures:
            from . import strip

            if strip.is_signature_widget(annot):
                drawn = strip.drawn_appearance(annot)
                return [drawn] if drawn is not None else []
        return _appearances(annot)

    def _holder(self, holder: Any, tag: tuple[Any, ...], own: Any, caller: Any, found: Counter[tuple[Any, ...]],
                streams: list[Any], active: set[Any], depth: int, content: Any = None) -> None:
        """Walk one content-bearing object; cached by object when it has Resources of its own."""
        og = holder.objgen
        key = (og, tag)
        if key in active:
            return
        if own is not None and og != (0, 0) and key in self.cache:
            sub, sub_streams = self.cache[key]
            found.update(sub)
            streams.extend(sub_streams)
            return
        sub = Counter()
        sub_streams: list[Any] = []
        active.add(key)
        try:
            self._walk(content if content is not None else holder, own if own is not None else caller,
                       sub, sub_streams, active, depth + 1)
        finally:
            active.discard(key)
        if own is not None and og != (0, 0):
            self.cache[key] = (sub, sub_streams)
        found.update(sub)
        streams.extend(sub_streams)

    def _walk(self, holder: Any, resources: Any, found: Counter[tuple[Any, ...]], streams: list[Any],
              active: set[Any], depth: int) -> None:
        import pikepdf

        if depth > MAX_WALK_DEPTH:
            raise ValueError("content nests more deeply than the walk follows")
        res = resources if isinstance(resources, pikepdf.Dictionary) else None

        def category(name: str) -> Any:
            value = res.get(name) if res is not None else None
            return value if isinstance(value, pikepdf.Dictionary) else None

        for ins in pikepdf.parse_content_stream(holder, "Do BI ID EI scn SCN gs Tf"):
            if isinstance(ins, pikepdf.ContentStreamInlineImage):
                found[("inline", _sha(bytes(ins.iimage.unparse())))] += 1
                continue
            op = str(ins.operator)
            operands = list(ins.operands)
            if op == "Do" and len(operands) == 1:
                xobjects = category("/XObject")
                xobj = xobjects.get(str(operands[0])) if xobjects is not None else None
                if not isinstance(xobj, pikepdf.Stream):
                    continue
                subtype = xobj.stream_dict.get("/Subtype")
                if subtype == pikepdf.Name.Image:
                    found[_image_fp(self.pdf, xobj, self.memo)] += 1
                    streams.append(xobj)
                elif subtype == pikepdf.Name.Form:
                    self._holder(xobj, ("form",), _own_res(xobj), res, found, streams, active, depth)
            elif op in ("scn", "SCN") and operands and isinstance(operands[-1], pikepdf.Name):
                patterns = category("/Pattern")
                pat = patterns.get(str(operands[-1])) if patterns is not None else None
                if isinstance(pat, pikepdf.Stream):
                    self._holder(pat, ("pattern",), _own_res(pat), res, found, streams, active, depth)
            elif op == "gs" and len(operands) == 1:
                states = category("/ExtGState")
                gs = states.get(str(operands[0])) if states is not None else None
                smask = gs.get("/SMask") if isinstance(gs, pikepdf.Dictionary) else None
                group = smask.get("/G") if isinstance(smask, pikepdf.Dictionary) else None
                if isinstance(group, pikepdf.Stream):
                    self._holder(group, ("smask",), _own_res(group), res, found, streams, active, depth)
            elif op == "Tf" and operands:
                fonts = category("/Font")
                font = fonts.get(str(operands[0])) if fonts is not None else None
                if not (isinstance(font, pikepdf.Dictionary) and font.get("/Subtype") == pikepdf.Name.Type3):
                    continue
                procs = font.get("/CharProcs")
                if not isinstance(procs, pikepdf.Dictionary):
                    continue
                own = _own_res(font)
                for glyph, proc in sorted(procs.items(), key=lambda kv: str(kv[0])):
                    if isinstance(proc, pikepdf.Stream):
                        self._holder(font, ("glyph", str(glyph)), own, res, found, streams, active, depth,
                                     content=proc)


MAX_WALK_DEPTH = 64                 # content drawn inside content (forms, patterns, glyphs, masks)


def _own_res(obj: Any) -> Any:
    import pikepdf

    d = obj.stream_dict if isinstance(obj, pikepdf.Stream) else obj
    value = d.get("/Resources") if isinstance(d, pikepdf.Dictionary) else None
    return value if isinstance(value, pikepdf.Dictionary) else None


def _appearances(annot: Any) -> list[Any]:
    """Every appearance stream of an annotation: /N, /R and /D, each state of each."""
    import pikepdf

    out: list[Any] = []
    if not isinstance(annot, pikepdf.Dictionary):
        return out
    ap = annot.get("/AP")
    if not isinstance(ap, pikepdf.Dictionary):
        return out
    for key in ("/N", "/R", "/D"):
        value = ap.get(key)
        if isinstance(value, pikepdf.Stream):
            out.append(value)
        elif isinstance(value, pikepdf.Dictionary):
            out.extend(v for _k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
                       if isinstance(v, pikepdf.Stream))
    return out


@_guard("images_identical")
def images_identical(reference: Path, built: Path, *, flattened_signatures: bool = False) -> Check:
    """Every page draws the same images: stored bytes for DCT, JBIG2, JPX and CCITT images,
    decoded samples for Flate and unfiltered ones, with the same size, depth, colour space, decode
    array and masks. Walked from the pages through forms, patterns, soft masks, Type 3 glyphs and
    annotation appearances (``_Drawn``); inline images compared as written.

    ``flattened_signatures`` (copies): a signature widget of the reference counts only the
    appearance ``strip.flatten_signatures`` draws into the page, none when it is hidden."""
    with _pike(Path(reference)) as ref, _pike(Path(built)) as out:
        if len(ref.pages) != len(out.pages):
            return Check("images_identical", False,
                         f"the built file has {len(out.pages)} page(s), the reference {len(ref.pages)}")
        a, b = _Drawn(ref, flattened_signatures=flattened_signatures), _Drawn(out)
        total = 0
        for i in range(len(ref.pages)):
            want, _ = a.page(i)
            got, _ = b.page(i)
            total += sum(want.values())
            if want != got:
                lost = want - got
                added = got - want
                parts = []
                if lost:
                    parts.append(f"{sum(lost.values())} image(s) changed or gone, first {_describe(next(iter(lost)))}")
                if added:
                    parts.append(f"{sum(added.values())} new or changed image(s), first {_describe(next(iter(added)))}")
                return Check("images_identical", False, f"{_human(i)}: " + "; ".join(parts), _pg(i))
    return Check("images_identical", True, f"{total} image(s) on {len(ref.pages)} page(s)")


@_guard("passthrough_identical")
def passthrough_identical(built: Path, assembled: Sequence["AssembledPage"]) -> Check:
    """Image inputs placed without decoding still carry their exact bytes.

    A ``passthrough`` page (a JPEG) must draw a DCT image whose stored bytes hash to the source's.
    A ``png_idat`` page must have been assembled from the source's IDAT bytes exactly; in the built
    file its Flate stream may have been recompressed losslessly by shrink, and its samples are
    then proved by ``images_identical`` against the assembled PDF.
    """
    pages = list(assembled)
    with _pike(Path(built)) as pdf:
        if len(pdf.pages) != len(pages):
            return Check("passthrough_identical", False,
                         f"the built file has {len(pdf.pages)} page(s) for {len(pages)} assembled image(s)")
        drawn = _Drawn(pdf)
        jpeg = idat = recompressed = 0
        for i, ap in enumerate(pages):
            route = getattr(ap, "route", None)
            if route not in ("passthrough", "png_idat"):
                continue
            if ap.embedded_sha256 != ap.source_sha256:
                return Check("passthrough_identical", False,
                             f"{_human(i)}: the embedded image isn't the source's bytes ({route})", _pg(i))
            _, streams = drawn.page(i)
            hashes = []
            for s in streams:
                filters = [str(f) for f in _as_list(s.stream_dict.get("/Filter"))]
                hashes.append((filters, _sha(s.read_raw_bytes())))
            if route == "passthrough":
                if not any(f == ["/DCTDecode"] and h == ap.embedded_sha256 for f, h in hashes):
                    return Check("passthrough_identical", False,
                                 f"{_human(i)}: the JPEG's bytes are no longer in the file unchanged", _pg(i))
                jpeg += 1
            else:
                if any(h == ap.embedded_sha256 for _f, h in hashes):
                    idat += 1
                elif any("/DCTDecode" not in f and "/JPXDecode" not in f for f, _h in hashes):
                    recompressed += 1
                else:
                    return Check("passthrough_identical", False,
                                 f"{_human(i)}: the PNG's image is no longer a lossless stream", _pg(i))
    detail = f"{jpeg} JPEG(s) and {idat} PNG IDAT stream(s) byte-identical"
    if recompressed:
        detail += f"; {recompressed} PNG stream(s) recompressed losslessly (samples checked by images_identical)"
    return Check("passthrough_identical", True, detail)


# --------------------------------------------------------------------------- structure and metadata


@_guard("upright")
def upright(built: Path, touched: frozenset[int]) -> Check:
    """/Rotate 0 on every touched page; untouched pages keep whatever flag they had."""
    bad: list[tuple[int, int]] = []
    with _pike(Path(built)) as pdf:
        count = len(pdf.pages)
        for i in sorted(int(p) for p in touched):
            if not 0 <= i < count:
                return Check("upright", False, f"{_human(i)} is not in the built file", _pg(i))
            value = _inherited(pdf.pages[i].obj, "/Rotate")
            rotate = int(value) % 360 if value is not None else 0
            if rotate:
                bad.append((i, rotate))
    if bad:
        first, rot = bad[0]
        return Check("upright", False,
                     f"{_pages_text([p for p, _r in bad])} still {'carries' if len(bad) == 1 else 'carry'} /Rotate (first at {rot})", _pg(first))
    return Check("upright", True, f"/Rotate 0 on {len(touched)} touched page(s)")


def _jbig2_hashes(pdf: Any) -> Counter[str]:
    import pikepdf

    found: Counter[str] = Counter()
    for obj in pdf.objects:
        if isinstance(obj, pikepdf.Stream):
            filters = [str(f) for f in _as_list(obj.stream_dict.get("/Filter"))]
            if JBIG2 in filters:
                try:
                    kind, data = _image_data(pdf, obj)          # the JBIG2-coded bytes, under any lead filter
                except Exception:                              # noqa: BLE001 - undecodable lead: the stored bytes
                    kind, data = "raw", obj.read_raw_bytes()
                if not kind.startswith("encoded"):
                    data = obj.read_raw_bytes()
                found[_sha(data)] += 1
    return found


@_guard("no_jbig2_written")
def no_jbig2_written(reference: Path, built: Path) -> Check:
    """Every JBIG2 stream in ``built`` is one the reference already carried, byte for byte."""
    with _pike(Path(reference)) as ref, _pike(Path(built)) as out:
        before = _jbig2_hashes(ref)
        after = _jbig2_hashes(out)
    new = after - before
    if new:
        return Check("no_jbig2_written", False, f"{sum(new.values())} JBIG2 stream(s) the input didn't have")
    kept = sum(after.values())
    return Check("no_jbig2_written", True, f"no new JBIG2 streams ({kept} passed through from the input)")


def _dicts_in(obj: Any, depth: int = 0) -> Iterator[Any]:
    """``obj`` and every dictionary nested in it directly (indirect references aren't followed)."""
    import pikepdf

    if depth > 32:
        return
    if isinstance(obj, pikepdf.Stream):
        obj = obj.stream_dict
    if isinstance(obj, pikepdf.Dictionary):
        yield obj
        children = list(obj.values())
    elif isinstance(obj, pikepdf.Array):
        children = list(obj)
    else:
        return
    for child in children:
        if isinstance(child, (pikepdf.Dictionary, pikepdf.Array)) and not child.is_indirect:
            yield from _dicts_in(child, depth + 1)


def _field_type(d: Any) -> Any:
    seen = 0
    node = d
    while node is not None and seen < 64:
        ft = node.get("/FT")
        if ft is not None:
            return ft
        node = node.get("/Parent")
        seen += 1
    return None


@_guard("no_signature_material")
def no_signature_material(built: Path) -> Check:
    """No /ByteRange, no /Perms, no signature or timestamp dictionary, and no signed /Sig field.

    Two detectors: ``refusals.output_signature_material`` (a raw byte scan, which also sees byte
    ranges nothing references, and a walk from the catalog; a file it can't read in full fails
    closed), and this check's own walk over every object, which also catches a /Sig or
    /DocTimeStamp dictionary on its own. An empty signature field (``/FT /Sig`` with no value)
    isn't signature material: refusals let such a file through, and it passes here unchanged.
    """
    import pikepdf

    from .. import refusals

    other = {_OUTPUT_SIG_WORDS.get(label, "signature material")
             for label in refusals.output_signature_material(Path(built))}
    found: Counter[str] = Counter()
    with _pike(Path(built)) as pdf:
        roots: list[Any] = [pdf.trailer]
        roots.extend(pdf.objects)
        for root in roots:
            for d in _dicts_in(root):
                if "/ByteRange" in d:
                    found["/ByteRange"] += 1
                typ = d.get("/Type")
                if typ in (pikepdf.Name("/Sig"), pikepdf.Name("/DocTimeStamp")):
                    found[f"a {str(typ)[1:]} dictionary"] += 1
                if d.get("/V") is not None and _field_type(d) == pikepdf.Name("/Sig"):
                    found["a signed /Sig field"] += 1
        if "/Perms" in pdf.Root:
            found["/Perms"] += 1
    for what in other:
        found[what] = max(found[what], 1)            # a label only the refusals detector saw counts once
    if found:
        return Check("no_signature_material", False,
                     "found " + ", ".join(f"{what} ({n})" for what, n in sorted(found.items())))
    return Check("no_signature_material", True, "no signature material")


_PDF_NS = b"http://ns.adobe.com/pdf/1.3/"


def _xmp_version_spans(data: bytes) -> list[tuple[int, int]]:
    """Where each ``<prefix>:PDFVersion`` value sits, for every prefix bound to the Adobe PDF schema."""
    prefixes = {m.group(1) for m in re.finditer(
        rb"xmlns:([A-Za-z_][A-Za-z0-9_.\-]*)\s*=\s*([\"'])" + re.escape(_PDF_NS) + rb"\2", data)}
    spans: list[tuple[int, int]] = []
    for prefix in sorted(prefixes):
        name = re.escape(prefix + b":PDFVersion")
        attr = re.compile(rb"\s" + name + rb"\s*=\s*([\"'])([^\"'<>]*)\1")
        elem = re.compile(rb"<" + name + rb"(?:\s[^>]*)?>\s*([^<\s]*)\s*</" + name + rb"\s*>")
        spans += [m.span(2) for m in attr.finditer(data)]
        spans += [m.span(1) for m in elem.finditer(data)]
    return sorted(spans)


def _only_version_patched(before: bytes, after: bytes, version: str) -> bool:
    """``after`` is ``before`` with only pdf:PDFVersion value(s) rewritten in place, to ``version``."""
    if len(before) != len(after):
        return False
    spans = _xmp_version_spans(after)
    if not spans or _xmp_version_spans(before) != spans:
        return False
    a, b = bytearray(before), bytearray(after)
    changed = False
    for start, end in spans:
        if a[start:end] != b[start:end]:
            changed = True
            if bytes(b[start:end]) != version.encode("ascii", "replace"):
                return False
        a[start:end] = b"\0" * (end - start)
        b[start:end] = b"\0" * (end - start)
    return changed and a == b


def _info(pdf: Any) -> dict[str, Any]:
    import pikepdf

    info = pdf.trailer.get("/Info")
    if not isinstance(info, pikepdf.Dictionary):
        return {}
    # A key whose value is null is the same as no key (ISO 32000-1 §7.3.7), and qpdf writes neither,
    # so a build never drops one. items() lists them; keys() and meta.snapshot don't.
    return {str(k)[1:]: _canon(v) for k, v in info.items() if v is not None}


def _xmp(pdf: Any) -> bytes | None:
    import pikepdf

    meta = pdf.Root.get("/Metadata")
    if not isinstance(meta, pikepdf.Stream):
        return None
    return meta.read_bytes()


def _meta_allow(allowed: Collection[str]) -> tuple[set[str], set[str]]:
    """(whole parts waived, Info keys waived): two namespaces, kept apart.

    A whole part is waived only by the bare names "Info" and "XMP". An Info key is waived by
    "Info/<Key>", "/<Key>" or a bare "<Key>" other than those two, so an Info key that happens to be
    named XMP or Info ("Info/XMP", "Info/Info") waives that key and nothing more.
    """
    parts: set[str] = set()
    keys: set[str] = set()
    for name in allowed:
        name = str(name)
        if name in ("Info", "XMP"):
            parts.add(name)
        elif name.startswith("Info/"):
            keys.add(name[len("Info/"):])
        elif name.startswith("/"):
            keys.add(name[1:])
        else:
            keys.add(name)
    return parts, keys


def _key_words(keys: Sequence[str]) -> str:
    """Info keys for a detail: the standard ones by name, any other only counted (the file names those)."""
    named = [k for k in keys if k in STANDARD_INFO_KEYS]
    other = len(keys) - len(named)
    words = list(named)
    if other:
        words.append(f"{other} other key(s)")
    return ", ".join(words)


@_guard("metadata_kept")
def metadata_kept(reference: Path, built: Path, allowed: frozenset[str]) -> Check:
    """The Info dictionary and the XMP packet are unchanged, except for the names in ``allowed``.

    ``allowed`` holds Info keys ("Info/Producer"; "/Producer" and "Producer" work too), "Info" for the
    whole dictionary and "XMP" for the packet (``META_NAMES``, the names ``meta.diff`` reports); the
    two namespaces are kept apart (``_meta_allow``). A same-length XMP pdf:PDFVersion patch to the
    built header version is always accepted and noted. A detail names only the standard Info keys;
    any other key is counted, since its name comes from the file.
    """
    parts, keys = _meta_allow(allowed)
    problems: list[str] = []
    notes: list[str] = []
    with _pike(Path(reference)) as ref, _pike(Path(built)) as out:
        before, after = _info(ref), _info(out)
        version = str(out.pdf_version)
        if "Info" not in parts:
            dropped, added, changed = [], [], []
            for key in sorted(set(before) | set(after)):
                if key in keys:
                    continue
                if key not in after:
                    dropped.append(key)
                elif key not in before:
                    added.append(key)
                elif before[key] != after[key]:
                    changed.append(key)
            for what, found in (("dropped", dropped), ("added", added), ("changed", changed)):
                if found:
                    problems.append(f"Info {_key_words(found)} {what}")
        xmp_before, xmp_after = _xmp(ref), _xmp(out)
        if "XMP" not in parts and xmp_before != xmp_after:
            if xmp_before is None:
                problems.append("XMP added")
            elif xmp_after is None:
                problems.append("XMP dropped")
            elif _only_version_patched(xmp_before, xmp_after, version):
                notes.append(f"XMP pdf:PDFVersion patched in place to {_safe_version(version)}")
            else:
                problems.append("XMP changed")
    if problems:
        return Check("metadata_kept", False, "; ".join(problems))
    kept = f"Info ({len(after)} key(s)) and " + ("XMP" if xmp_after is not None else "no XMP") + " kept"
    return Check("metadata_kept", True, "; ".join([kept] + notes))


def _safe_version(version: str) -> str:
    return version if re.fullmatch(r"[0-9]\.[0-9]", version) else "the header version"


def _syntax_category(warning: object) -> str:
    """A closed, core-worded category for one qpdf warning (its text is never shown)."""
    text = str(warning).casefold()
    if "content" in text:
        return "content stream"
    if "xref" in text or "cross-reference" in text or "startxref" in text or "trailer" in text:
        return "cross-reference table"
    if "stream" in text:
        return "stream data"
    return "object syntax"


@_guard("structure_ok")
def structure_ok(built: Path) -> Check:
    """qpdf finds no syntax problem (``check_pdf_syntax``, open-time repairs included), and pdfium
    opens the file and renders every page (scaled down to ``RENDER_MAX_PIXELS`` for a huge page).
    A detail counts qpdf's problems by closed category and never quotes them."""
    with _pike(Path(built)) as pdf:
        problems = pdf.check_pdf_syntax()
        count = len(pdf.pages)
    if problems:
        cats = Counter(_syntax_category(w) for w in problems)
        what = ", ".join(f"{n} in {cat}" for cat, n in sorted(cats.items()))
        return Check("structure_ok", False, f"qpdf reports {len(problems)} syntax problem(s) ({what})")
    with _lock(), _pdfium(Path(built)) as doc:
        if len(doc) != count:
            return Check("structure_ok", False, f"pdfium sees {len(doc)} page(s), qpdf {count}")
        for i in range(count):
            page = doc[i]
            try:
                _render_gray(page, _capped_scale(STRUCTURE_RENDER_SCALE, _size(page)), 0)
            finally:
                page.close()
    return Check("structure_ok", True, f"syntax clean; {count} page(s) open and render")


@_guard("not_encrypted")
def not_encrypted(built: Path) -> Check:
    """A copy is saved without encryption, open password or edit lock, so it can be searched (§10.4)."""
    import pikepdf

    try:
        pdf = pikepdf.open(Path(built))
    except pikepdf.PasswordError:
        return Check("not_encrypted", False, "the copy needs a password to open")
    with pdf:
        if pdf.is_encrypted or "/Encrypt" in pdf.trailer:
            return Check("not_encrypted", False, "the copy is encrypted (it carries an edit lock)")
    return Check("not_encrypted", True, "no encryption")


@_guard("derived_from_present")
def derived_from_present(built: Path, original_sha256: str, original_name: str | None = None) -> Check:
    """The copy's XMP carries the DerivedFrom block ``meta.mark_derived`` writes (§10.2).

    Read with ``meta.derived_from``, the one detector for the block (decided 2026-11-05): the
    element must sit in ``meta.PDFINTAKE_NS`` exactly, carry the original's SHA-256 (64 hex
    digits, compared exactly), a file name (``original_name`` when it is given) and the note
    ``meta.DERIVED_NOTE`` word for word. XMP's own ``xmpMM:DerivedFrom`` never counts.
    """
    from . import meta

    want = str(original_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", want):
        return Check("derived_from_present", False, "the original's SHA-256 isn't 64 hex digits")
    with _pike(Path(built)) as pdf:
        if _xmp(pdf) is None:
            return Check("derived_from_present", False, "the copy has no XMP, so it doesn't name its original")
        found = meta.derived_from(pdf)
    if found is None:
        return Check("derived_from_present", False, "the copy's XMP has no DerivedFrom block")
    if found.get("sha256") != want:
        return Check("derived_from_present", False, "the DerivedFrom block doesn't carry the original's SHA-256")
    name = found.get("name") or ""
    if not name.strip() or (original_name is not None and name != original_name):
        return Check("derived_from_present", False, "the DerivedFrom block doesn't carry the original's file name")
    if found.get("note") != meta.DERIVED_NOTE:
        return Check("derived_from_present", False,
                     "the DerivedFrom block doesn't say it is a searchable copy, not the original")
    return Check("derived_from_present", True, "DerivedFrom names the original by its file name and SHA-256")


@_guard("text_layer_identical")
def text_layer_identical(a: Path, b: Path) -> Check:
    """Both files hand back the same text with the same character boxes on every page (the smaller copy)."""
    with _lock(), _pdfium(Path(a)) as da, _pdfium(Path(b)) as db:
        if len(da) != len(db):
            return Check("text_layer_identical", False, f"{len(da)} page(s) against {len(db)}")
        chars = 0
        for i in range(len(da)):
            pa, pb = da[i], db[i]
            try:
                ta, tb = pa.get_textpage(), pb.get_textpage()
                try:
                    if ta.get_text_range() != tb.get_text_range():
                        return Check("text_layer_identical", False, f"{_human(i)}: the text differs", _pg(i))
                    n = ta.count_chars()
                    for k in range(n):
                        if ta.get_charbox(k) != tb.get_charbox(k):
                            return Check("text_layer_identical", False,
                                         f"{_human(i)}: character positions differ", _pg(i))
                    chars += n
                finally:
                    ta.close()
                    tb.close()
            finally:
                pa.close()
                pb.close()
        pages = len(da)
    return Check("text_layer_identical", True, f"{pages} page(s), {chars} character(s) identical")


# --------------------------------------------------------------------------- the suite


def verify_build(ctx: BuildContext) -> list[Check]:
    """Every check above, in the order listed; ``flow`` raises VerifyFailed on any ``ok=False``.

    ``text_retention`` runs when a layer was written (``expected`` not empty); ``off_page_glyphs``
    covers the layered and touched pages; ``passthrough_identical`` runs for image inputs. In copy
    mode (``original_sha256`` set) the reference is the work copy before the layer, signature stamps
    are compared as flattened, and ``not_encrypted`` and ``derived_from_present`` run after
    ``structure_ok``.
    """
    copy = ctx.original_sha256 is not None
    checks: list[Check] = []
    if ctx.expected:
        checks.extend(text_retention(ctx.built, ctx.expected))
    checks.append(off_page_glyphs(ctx.built, pages=set(ctx.touched) | {int(i) for i in ctx.expected}))
    checks.append(renders_identical(ctx.reference, ctx.built, ctx.turns))
    checks.append(images_identical(ctx.reference, ctx.built, flattened_signatures=copy))
    if ctx.assembled is not None:
        checks.append(passthrough_identical(ctx.built, ctx.assembled))
    checks.append(upright(ctx.built, ctx.touched))
    checks.append(no_jbig2_written(ctx.reference, ctx.built))
    checks.append(no_signature_material(ctx.built))
    checks.append(metadata_kept(ctx.reference, ctx.built, ctx.allowed_meta_changes))
    checks.append(structure_ok(ctx.built))
    if copy:
        checks.append(not_encrypted(ctx.built))
        checks.append(derived_from_present(ctx.built, str(ctx.original_sha256), ctx.original_name))
    return checks
