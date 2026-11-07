"""Which pages get OCR: a class for every page, from pdfium's text and image objects.

The rule that matters most: a page with any visible text is never OCR'd, so printed words can
never be covered or erased. Text is visible when it is drawn (render mode not 3 or 7), carries at
least one non-space character, and its paint isn't fully transparent: a fill mode needs fill
alpha above 0, a stroke mode stroke alpha above 0. Objects are read through Form XObjects to the
same depth pdfium's own walk uses; annotation and widget appearances aren't page objects and
don't count. Text objects holding only whitespace print nothing and are ignored altogether.

Classes (DESIGN §2.2, U5):

* ``born_digital``: visible text, garble ratio under 0.05. No OCR; warning ``image_text_not_read``
  when images cover at least half the page.
* ``garbled``: visible text, garble ratio 0.05 or more. No OCR; warning ``garbled_text``.
* ``clip_text``: no visible text, some mode-7 text. No OCR; warning ``clip_text_kept``.
* ``invisible_layer``: text present, none visible, and none of it adds to the clipping path: every
  piece is mode 3, or a fill or stroke mode (0, 1, 2) whose paint is fully transparent (Scanner
  Pro, for one, draws its OCR at fill alpha 0; decided 2026-11-07, DESIGN Q8). Not ours. OCR'd with
  the old layer stripped (``existing_layer_replaced``), unless ``keep_existing_layer``
  (``existing_layer_kept``). Dropping that text changes no pixel, and verify's
  ``renders_identical`` proves it on every build.
* ``our_layer``: mode-3 text only, plus our ``/PieceInfo /RazorII_pdfintake`` marker. Left alone
  unless ``redo_ocr``.
* ``hidden_text``: text present, none visible, some of it transparent text in a clipping mode (4,
  5 or 6), which shapes what the page draws next. No OCR; warning ``existing_layer_kept``, because
  it can't be stripped safely.
* ``image_only``: no text, images present. OCR'd.
* ``empty``: neither.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .render import MAX_FORM_DEPTH, Renderer, assert_locked

if TYPE_CHECKING:
    import pikepdf

PageKind = Literal["image_only", "invisible_layer", "our_layer", "hidden_text", "born_digital",
                   "garbled", "clip_text", "empty"]

GARBLE_LIMIT = 0.05           # visible text at or above this garble ratio has a broken font map
IMAGE_TEXT_COVERAGE = 0.5     # a born-digital page this covered by images may carry unread image text
MARKER_KEY = "/RazorII_pdfintake"   # our /PieceInfo entry (written by pdf/textlayer.write_layer)

_FILL_MODES = frozenset({0, 2, 4, 6})
_STROKE_MODES = frozenset({1, 2, 5, 6})
_LAYER_MODES = frozenset({0, 1, 2, 3})    # invisible text in these modes draws nothing and clips nothing
_GARBLE_CATEGORIES = frozenset({"Cc", "Co", "Cn"})


@dataclass
class PageClass:
    index: int
    kind: PageKind
    visible_chars: int
    invisible_chars: int
    garble_ratio: float
    image_coverage: float
    needs_ocr: bool
    strip_old_layer: bool
    warnings: list[str] = field(default_factory=list)


def garble_ratio(text: str) -> float:               # U+FFFD, Cc/Co/Cn, PUA share of non-space characters
    """The share of non-space characters that are U+FFFD, controls, private use or unassigned."""
    total = bad = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ch == "�" or unicodedata.category(ch) in _GARBLE_CATEGORIES:
            bad += 1
    return bad / total if total else 0.0


def _nonspace(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


@dataclass
class _Tally:
    visible: list[str] = field(default_factory=list)
    visible_chars: int = 0
    invisible_chars: int = 0
    hidden_text: list[str] = field(default_factory=list)
    modes: set[int] = field(default_factory=set)        # render modes of every non-visible text object
    images: int = 0


def _alphas(obj: Any) -> tuple[int, int]:
    """(fill alpha, stroke alpha), 0-255; 255 when pdfium can't say (counted as visible)."""
    import ctypes

    import pypdfium2.raw as pdfium_c

    out = []
    for getter in (pdfium_c.FPDFPageObj_GetFillColor, pdfium_c.FPDFPageObj_GetStrokeColor):
        r, g, b, a = (ctypes.c_uint() for _ in range(4))
        ok = getter(obj, r, g, b, a)
        out.append(int(a.value) if ok else 255)
    return out[0], out[1]


def _visible(mode: int, fill_alpha: int, stroke_alpha: int) -> bool:
    if mode in (3, 7):
        return False
    if mode not in range(8):                        # unknown mode: assume it prints
        return True
    return (mode in _FILL_MODES and fill_alpha > 0) or (mode in _STROKE_MODES and stroke_alpha > 0)


def _tally(page: Any) -> _Tally:
    """Walk one pdfium page's objects (through forms); the caller holds PDFIUM_LOCK."""
    import pypdfium2.raw as pdfium_c

    assert_locked()
    t = _Tally()
    textpage = page.get_textpage()
    try:
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE],
                                    max_depth=MAX_FORM_DEPTH, textpage=textpage):
            if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                t.images += 1
                continue
            try:
                text = obj.extract()
            except Exception:
                text = ""
            n = _nonspace(text)
            if n == 0:
                continue                             # only whitespace prints nothing
            mode = int(pdfium_c.FPDFTextObj_GetTextRenderMode(obj))
            fill_a, stroke_a = _alphas(obj)
            if _visible(mode, fill_a, stroke_a):
                t.visible.append(text)
                t.visible_chars += n
            else:
                t.hidden_text.append(text)
                t.invisible_chars += n
                t.modes.add(mode)
    finally:
        textpage.close()
    return t


def _has_marker(holder: Any) -> bool:
    import pikepdf

    try:
        info = holder.get("/PieceInfo")
        return isinstance(info, pikepdf.Dictionary) and MARKER_KEY in info
    except Exception:
        return False


def _our_marker(pdf: "pikepdf.Pdf", index: int) -> bool:
    """Our marker on the page, on a Form XObject the page draws directly, or on the document."""
    import pikepdf

    page = pdf.pages[index].obj
    if _has_marker(page):
        return True
    try:
        res = page.get("/Resources")
        xobjects = res.get("/XObject") if isinstance(res, pikepdf.Dictionary) else None
        if isinstance(xobjects, pikepdf.Dictionary):
            for _, xo in xobjects.items():
                if isinstance(xo, pikepdf.Stream) and str(xo.get("/Subtype", "")) == "/Form" and _has_marker(xo):
                    return True
    except Exception:
        pass
    return _has_marker(pdf.Root)


def _decide(index: int, t: _Tally, coverage: float, marker: bool, *, redo_ocr: bool,
            keep_existing_layer: bool) -> PageClass:
    warnings: list[str] = []
    if t.visible_chars:
        ratio = garble_ratio("".join(t.visible))
        if ratio >= GARBLE_LIMIT:
            kind: PageKind = "garbled"
            warnings.append("garbled_text")
        else:
            kind = "born_digital"
            if coverage >= IMAGE_TEXT_COVERAGE:
                warnings.append("image_text_not_read")
        return PageClass(index, kind, t.visible_chars, t.invisible_chars, ratio, coverage, False, False, warnings)
    ratio = garble_ratio("".join(t.hidden_text))
    needs = strip = False
    if t.invisible_chars:
        if 7 in t.modes:
            kind = "clip_text"
            warnings.append("clip_text_kept")
        elif t.modes == {3} and marker:
            kind = "our_layer"
            needs = strip = redo_ocr
            if redo_ocr:
                warnings.append("existing_layer_replaced")
        elif t.modes <= _LAYER_MODES:
            kind = "invisible_layer"
            needs = strip = not keep_existing_layer
            warnings.append("existing_layer_kept" if keep_existing_layer else "existing_layer_replaced")
        else:
            kind = "hidden_text"
            warnings.append("existing_layer_kept")
    elif t.images or coverage > 0:
        kind = "image_only"
        needs = True
    else:
        kind = "empty"
    return PageClass(index, kind, 0, t.invisible_chars, ratio, coverage, needs, strip, warnings)


def classify(pdf: Path, *, redo_ocr: bool, keep_existing_layer: bool) -> list[PageClass]:
    """One ``PageClass`` per page, in page order."""
    out: list[PageClass] = []
    with Renderer(Path(pdf)) as r:
        for index in range(r.page_count()):
            with r.pdfium_page(index) as page:
                tally = _tally(page)
            coverage = r.image_coverage(index)
            marker = _our_marker(r._pikepdf(), index) if tally.invisible_chars and not tally.visible_chars else False
            out.append(_decide(index, tally, coverage, marker, redo_ocr=redo_ocr,
                               keep_existing_layer=keep_existing_layer))
    return out
