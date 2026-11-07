"""The reading side of a PDF: page renders, placed image geometry and the OCR render dpi.

pdfium isn't thread-safe, so every pypdfium2 call in the package runs under ``PDFIUM_LOCK``.
``Renderer``'s public methods take the lock themselves (it is re-entrant, so a caller that already
holds it is fine), and every internal pdfium touch point asserts that it is held, so a path that
reaches pdfium without the lock fails loudly in debug runs (``assert_locked``). Engine workers
never get pdfium objects: they get the PNG a render wrote (``PageImage``).

Image geometry comes from a walk of the page's content stream with pikepdf, composing ``cm``
operators and each Form XObject's ``/Matrix``, so an image's pixels per inch come from the lengths
of its placed axes. That stays right when an image is placed turned 90 degrees, where the
width-over-box metadata dpi is wrong, and when an image sits inside a form.

The OCR render dpi (``Renderer.ocr_dpi``) is the native resolution of the scan that covers the
page (the largest image covering at least half of it, the higher of its two axes), never below a
200 dpi floor, never past the dpi at which the long side reaches RapidOCR's 2000 px input limit,
never above 1200, and never above 60 MP. The 60 MP cap wins over the floor; every other limit
gives way to the floor.

No render side is ever longer than 32767 px, Tesseract's limit (decided 2026-11-05). A page that
would exceed it, such as a long till receipt, is rendered at the largest whole dpi that fits, and
``Renderer.render_scaled_down`` reports it (record note ``render_scaled_down``). This limit wins
over the floor too, and applies to every render and to ``upscale_for_osd``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..types import PageImage

if TYPE_CHECKING:
    import PIL.Image
    import pikepdf
    import pypdfium2

PDFIUM_LOCK: threading.RLock = threading.RLock()
    # pdfium isn't thread-safe. Every pypdfium2 call in the package (Renderer, classify, verify, smaller,
    # export) holds this lock or runs in the main thread under it. Engine workers get only rendered PNGs.

DPI_FLOOR = 200.0               # an OCR render is never coarser than this (except under the pixel cap)
DPI_MAX = 1200.0                # sanity cap on any OCR render
RAPID_MAX_SIDE = 2000           # RapidOCR resizes a longer input down to this, so pixels past it are wasted
MAX_RENDER_PIXELS = 60_000_000  # every render: at most 60 MP (memory bound)
MAX_RENDER_SIDE = 32767         # every render: no side longer than this, Tesseract's limit (decided 2026-11-05)
COVER_MIN = 0.5                 # a page's "scan" covers at least half of it
MAX_FORM_DEPTH = 15             # the same depth pdfium's object walk uses
MAX_UNION_RECTS = 2000          # above this many images, coverage is a capped sum, not an exact union

_IDENTITY: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
Matrix = tuple[float, float, float, float, float, float]
Rect = tuple[float, float, float, float]

# Inline-image abbreviations (PDF 32000-1, tables 93 and 94).
_INLINE_FILTERS = {"AHx": "ASCIIHexDecode", "A85": "ASCII85Decode", "LZW": "LZWDecode", "Fl": "FlateDecode",
                   "RL": "RunLengthDecode", "CCF": "CCITTFaxDecode", "DCT": "DCTDecode"}
_INLINE_SPACES = {"G": "DeviceGray", "RGB": "DeviceRGB", "CMYK": "DeviceCMYK", "I": "Indexed"}


def assert_locked() -> None:
    """Debug assertion: the calling thread holds ``PDFIUM_LOCK``. Gone under ``python -O``."""
    assert PDFIUM_LOCK._is_owned(), "pdfium was called without holding PDFIUM_LOCK"  # type: ignore[attr-defined]


@dataclass(frozen=True)
class PlacedImage:
    """One image as a page shows it.

    ``xobject`` is the image XObject's (object number, generation); an inline image has no object
    and reports ``(0, 0)``. ``ppi`` is the higher of the two placed axes' pixels per inch.
    ``coverage`` is the share of the page's CropBox its placed bounding box covers (clipped to the
    CropBox and to enclosing form boxes). ``filter`` is the filter chain without slashes, joined
    with ``+`` in stream order (``""`` when unfiltered). ``colorspace`` is the colour space name or
    the family of an array colour space (``"ICCBased"``, ``"Indexed"``; ``""`` for image masks).
    ``bytes`` is the length of the stored (still encoded) stream.
    """

    page: int
    xobject: tuple[int, int]
    width_px: int
    height_px: int
    ppi: float
    coverage: float
    filter: str
    bpc: int
    colorspace: str
    bytes: int


# --------------------------------------------------------------------------- matrix helpers

def _mul(m: Matrix, n: Matrix) -> Matrix:
    """m then n (PDF row-vector convention: p' = p x m x n)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)


def _apply(m: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (x * a + y * c + e, x * b + y * d + f)


def _bbox(m: Matrix, rect: Rect = (0.0, 0.0, 1.0, 1.0)) -> Rect:
    x0, y0, x1, y1 = rect
    pts = [_apply(m, x, y) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _intersect(r: Rect, s: Rect) -> Rect:
    x0, y0 = max(r[0], s[0]), max(r[1], s[1])
    x1, y1 = min(r[2], s[2]), min(r[3], s[3])
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x0, y0)
    return (x0, y0, x1, y1)


def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _norm_rect(raw: Any) -> Rect | None:
    try:
        vals = [float(v) for v in raw]
    except Exception:
        return None
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _matrix(raw: Any) -> Matrix | None:
    try:
        vals = tuple(float(v) for v in raw)
    except Exception:
        return None
    if len(vals) != 6 or not all(math.isfinite(v) for v in vals):
        return None
    return vals  # type: ignore[return-value]


def _union_area(rects: Sequence[Rect]) -> float:
    """The area the union of axis-aligned rectangles covers (a sweep over x; exact)."""
    rs = [r for r in rects if _area(r) > 0]
    if not rs:
        return 0.0
    xs = sorted({r[0] for r in rs} | {r[2] for r in rs})
    total = 0.0
    for xa, xb in zip(xs, xs[1:]):
        spans = sorted((r[1], r[3]) for r in rs if r[0] <= xa and r[2] >= xb)
        covered, cur0, cur1 = 0.0, None, None
        for y0, y1 in spans:
            if cur1 is None or y0 > cur1:
                if cur1 is not None:
                    covered += cur1 - cur0  # type: ignore[operator]
                cur0, cur1 = y0, y1
            else:
                cur1 = max(cur1, y1)
        if cur1 is not None:
            covered += cur1 - cur0  # type: ignore[operator]
        total += covered * (xb - xa)
    return total


# --------------------------------------------------------------------------- content walk (pikepdf)

def _name(obj: Any) -> str:
    s = str(obj)
    return s[1:] if s.startswith("/") else s


def _inherited(page_obj: Any, key: str) -> Any:
    node, seen = page_obj, 0
    while node is not None and seen < 64:
        if key in node:
            return node[key]
        node = node.get("/Parent")
        seen += 1
    return None


def _filters(raw: Any) -> str:
    if raw is None:
        return ""
    import pikepdf

    items = list(raw) if isinstance(raw, pikepdf.Array) else [raw]
    return "+".join(_name(x) for x in items)


def _colorspace(raw: Any) -> str:
    if raw is None:
        return ""
    import pikepdf

    if isinstance(raw, pikepdf.Array):
        if len(raw) == 0:
            return ""
        raw = raw[0]
    return _name(raw)


def _int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _ppi(m: Matrix, w: int, h: int) -> float:
    a, b, c, d, _, _ = m
    lx, ly = math.hypot(a, b), math.hypot(c, d)       # the image's axes, in page points
    if lx <= 0 or ly <= 0 or w <= 0 or h <= 0:
        return 0.0
    return max(w * 72.0 / lx, h * 72.0 / ly)


def _walk_images(pdf: "pikepdf.Pdf", index: int, cropbox: Rect) -> list[tuple[PlacedImage, Rect]]:
    """Every image a page's content (and its forms, to ``MAX_FORM_DEPTH``) places, in drawing order,
    with the rectangle (pt, user space, clipped to the CropBox and form boxes) it shows in."""
    import pikepdf

    page_obj = pdf.pages[index].obj
    page_area = _area(cropbox) or 1.0
    found: list[tuple[PlacedImage, Rect]] = []

    def visit(content: Any, resources: Any, ctm: Matrix, clip: Rect, depth: int, stack: tuple[tuple[int, int], ...]) -> None:
        try:
            instructions = pikepdf.parse_content_stream(content)
        except Exception:
            return
        xobjects = resources.get("/XObject") if isinstance(resources, pikepdf.Dictionary) else None
        gstack: list[Matrix] = []
        cur = ctm
        for inst in instructions:
            if isinstance(inst, pikepdf.ContentStreamInlineImage):
                img = inst.iimage
                try:
                    w, h = int(img.width), int(img.height)
                except Exception:
                    continue
                mask = bool(_safe(lambda: img.image_mask, False))
                filt = "+".join(_INLINE_FILTERS.get(_name(f), _name(f)) for f in (_safe(lambda: img.filters, []) or []))
                cs = "" if mask else _INLINE_SPACES.get(_name(_safe(lambda: img.colorspace, "") or ""),
                                                        _name(_safe(lambda: img.colorspace, "") or ""))
                bpc = 1 if mask else _int(_safe(lambda: img.bits_per_component, 0))
                size = len(_safe(lambda: img.read_raw_bytes(), b"") or b"")
                shown = _intersect(_bbox(cur), clip)
                found.append((PlacedImage(index, (0, 0), w, h, _ppi(cur, w, h), min(1.0, _area(shown) / page_area),
                                          filt, bpc, cs, size), shown))
                continue
            op = str(inst.operator)
            if op == "q":
                gstack.append(cur)
            elif op == "Q":
                if gstack:
                    cur = gstack.pop()
            elif op == "cm":
                m = _matrix(inst.operands)
                if m is not None:
                    cur = _mul(m, cur)
            elif op == "Do" and xobjects is not None and inst.operands:
                xo = xobjects.get(str(inst.operands[0]))
                if not isinstance(xo, pikepdf.Stream):
                    continue
                sub = _name(xo.get("/Subtype", ""))
                key = xo.objgen
                if sub == "Image":
                    w, h = _int(xo.get("/Width")), _int(xo.get("/Height"))
                    mask = bool(xo.get("/ImageMask", False))
                    try:
                        size = len(xo.read_raw_bytes())
                    except Exception:
                        size = 0
                    shown = _intersect(_bbox(cur), clip)
                    found.append((PlacedImage(
                        index, (int(key[0]), int(key[1])), w, h, _ppi(cur, w, h),
                        min(1.0, _area(shown) / page_area), _filters(xo.get("/Filter")),
                        1 if mask else _int(xo.get("/BitsPerComponent")),
                        "" if mask else _colorspace(xo.get("/ColorSpace")), size), shown))
                elif sub == "Form" and depth < MAX_FORM_DEPTH and key not in stack:
                    fm = _matrix(xo.get("/Matrix", list(_IDENTITY))) or _IDENTITY
                    inner = _mul(fm, cur)
                    box = _norm_rect(xo.get("/BBox"))
                    inner_clip = _intersect(clip, _bbox(inner, box)) if box is not None else clip
                    res = xo.get("/Resources", resources)
                    visit(xo, res, inner, inner_clip, depth + 1, stack + (key,))

    resources = _inherited(page_obj, "/Resources")
    if "/Contents" in page_obj:
        visit(pikepdf.Page(page_obj), resources, _IDENTITY, cropbox, 0, ())
    return found


# --------------------------------------------------------------------------- the renderer

def _pixels(width_pt: float, height_pt: float, dpi: float) -> tuple[int, int]:
    """The bitmap size pypdfium2 makes for a page of this displayed size at this dpi."""
    scale = dpi / 72.0
    return math.ceil(width_pt * scale), math.ceil(height_pt * scale)


def _pixel_cap(width_pt: float, height_pt: float, dpi: float, max_pixels: int) -> float:
    """``dpi``, lowered (to a whole number) until the render holds at most ``max_pixels``."""
    w, h = _pixels(width_pt, height_pt, dpi)
    if w * h <= max_pixels:
        return dpi
    cap = math.floor(72.0 * math.sqrt(max_pixels / max(width_pt * height_pt, 1e-9)))
    while cap > 1:
        w, h = _pixels(width_pt, height_pt, cap)
        if w * h <= max_pixels:
            break
        cap -= 1
    return float(max(cap, 1))


def _side_cap(width_pt: float, height_pt: float, dpi: float, max_side: int) -> float:
    """``dpi``, lowered (to a whole number) until neither side of the render exceeds ``max_side``."""
    if max(_pixels(width_pt, height_pt, dpi)) <= max_side:
        return dpi
    cap = math.floor(72.0 * max_side / max(width_pt, height_pt, 1e-9))
    while cap > 1 and max(_pixels(width_pt, height_pt, cap)) > max_side:
        cap -= 1
    return float(max(cap, 1))


def _fit_dpi(width_pt: float, height_pt: float, dpi: float, max_pixels: int = MAX_RENDER_PIXELS,
             max_side: int = MAX_RENDER_SIDE) -> tuple[float, bool, bool]:
    """(the dpi to render at, lowered by the pixel cap, lowered further by the side limit).

    The pixel cap applies first; ``scaled_down`` is true only when the side limit then lowered the
    dpi below what the pixel cap allowed.
    """
    by_pixels = _pixel_cap(width_pt, height_pt, dpi, max_pixels)
    used = _side_cap(width_pt, height_pt, by_pixels, max_side)
    return used, by_pixels < dpi, used < by_pixels


def _cap_dpi(width_pt: float, height_pt: float, dpi: float, max_pixels: int = MAX_RENDER_PIXELS) -> float:
    """``dpi``, lowered (to a whole number) until the render holds at most ``max_pixels`` and no
    side is longer than ``MAX_RENDER_SIDE``."""
    return _fit_dpi(width_pt, height_pt, dpi, max_pixels)[0]


class Renderer:                                          # one pypdfium2 document per run
    """One open PDF for a run: renders (pdfium) and image geometry (pikepdf, read only)."""

    def __init__(self, pdf: Path) -> None:
        self.path = Path(pdf)
        self._doc: "pypdfium2.PdfDocument | None" = None
        self._pike: "pikepdf.Pdf | None" = None
        self._sizes: dict[int, tuple[tuple[float, float], Rect]] = {}
        self._images: dict[int, list[tuple[PlacedImage, Rect]]] = {}
        self._capped: dict[int, bool] = {}
        self._scaled: dict[int, bool] = {}           # sticky: some render of the page hit MAX_RENDER_SIDE

    # ---- lifetime

    def __enter__(self) -> "Renderer":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        import pikepdf
        import pypdfium2

        with PDFIUM_LOCK:
            if self._doc is None:
                self._doc = pypdfium2.PdfDocument(str(self.path))
        if self._pike is None:
            self._pike = pikepdf.open(self.path)

    def close(self) -> None:
        with PDFIUM_LOCK:
            if self._doc is not None:
                self._doc.close()
                self._doc = None
        if self._pike is not None:
            self._pike.close()
            self._pike = None

    # ---- pdfium access (lock held)

    def _document(self) -> "pypdfium2.PdfDocument":
        assert_locked()
        if self._doc is None:
            raise RuntimeError("Renderer is not open; use it as a context manager")
        return self._doc

    def _page(self, index: int) -> "pypdfium2.PdfPage":
        """The pdfium page (the caller holds PDFIUM_LOCK and closes the page)."""
        doc = self._document()
        if not 0 <= index < len(doc):
            raise IndexError(f"page {index} is out of range (the document has {len(doc)} pages)")
        return doc[index]

    @contextmanager
    def pdfium_page(self, index: int) -> Iterator["pypdfium2.PdfPage"]:
        """For this package's own pdfium readers (classify): the page, under the lock, closed after."""
        with PDFIUM_LOCK:
            page = self._page(index)
            try:
                yield page
            finally:
                page.close()

    def _geometry(self, index: int) -> tuple[tuple[float, float], Rect]:
        """((displayed width, height) in pt, the CropBox in user space), cached."""
        if index not in self._sizes:
            with self.pdfium_page(index) as page:
                size = (float(page.get_width()), float(page.get_height()))
                box = _norm_rect(page.get_bbox()) or (0.0, 0.0, size[0], size[1])
            self._sizes[index] = (size, box)
        return self._sizes[index]

    def _pikepdf(self) -> "pikepdf.Pdf":
        if self._pike is None:
            raise RuntimeError("Renderer is not open; use it as a context manager")
        return self._pike

    # ---- public

    def page_count(self) -> int:
        with PDFIUM_LOCK:
            return len(self._document())

    def cropbox(self, index: int) -> Rect:
        """The page's effective CropBox (CropBox within MediaBox), pt, user space."""
        return self._geometry(index)[1]

    def _walked(self, index: int) -> list[tuple[PlacedImage, Rect]]:
        if index not in self._images:
            box = self.cropbox(index)
            self._images[index] = _walk_images(self._pikepdf(), index, box)
        return self._images[index]

    def placed_images(self, index: int) -> list[PlacedImage]:   # ppi from matrix axes composed through forms
        return [im for im, _ in self._walked(index)]

    def image_coverage(self, index: int) -> float:
        """The share of the CropBox that the union of the page's placed images covers (0 to 1)."""
        rects = [r for _, r in self._walked(index)]
        box = self.cropbox(index)
        area = _area(box) or 1.0
        if len(rects) > MAX_UNION_RECTS:                        # pathological pages: an upper bound
            return min(1.0, sum(_area(r) for r in rects) / area)
        return min(1.0, _union_area(rects) / area)

    def native_dpi(self, index: int) -> float | None:
        """The ppi of the largest image covering at least half the page, or None (no scan on it)."""
        best, best_cover = None, 0.0
        for im in self.placed_images(index):
            if im.coverage < COVER_MIN or im.coverage <= best_cover or im.ppi <= 0:
                continue
            best, best_cover = im.ppi, im.coverage
        return best

    def ocr_dpi(self, index: int) -> float:                     # native dpi, floor 200, cap 72*2000/long side, <=1200, <=60 MP
        (w, h), _ = self._geometry(index)
        long_side = max(w, h, 1e-9)
        native = self.native_dpi(index)
        if native is None:
            dpi = DPI_FLOOR
        else:
            rapid_cap = math.floor(72.0 * RAPID_MAX_SIDE / long_side)
            dpi = float(int(max(DPI_FLOOR, min(native, rapid_cap, DPI_MAX))))
        used, capped, scaled = _fit_dpi(w, h, dpi)
        self._capped[index] = capped
        self._note_scaled(index, scaled)
        return used

    def render_capped(self, index: int) -> bool:
        """Whether ``ocr_dpi`` for this page was lowered by the 60 MP cap (record warning ``render_capped``)."""
        if index not in self._capped:
            self.ocr_dpi(index)
        return self._capped[index]

    def _note_scaled(self, index: int, scaled: bool) -> None:
        self._scaled[index] = self._scaled.get(index, False) or scaled

    def render_scaled_down(self, index: int) -> bool:
        """Whether this page's ``ocr_dpi``, or any render of it this Renderer made, was lowered so that
        no side exceeds ``MAX_RENDER_SIDE`` (record note ``render_scaled_down``). Sticky for the run."""
        if index not in self._capped:
            self.ocr_dpi(index)
        return self._scaled.get(index, False)

    def _bitmap(self, index: int, dpi: float, rotation: int, gray: bool) -> "PIL.Image.Image":
        if not (isinstance(dpi, (int, float)) and math.isfinite(dpi) and dpi > 0):
            raise ValueError(f"dpi must be a positive number, not {dpi!r}")
        if rotation not in (0, 90, 180, 270):
            raise ValueError(f"rotation must be 0, 90, 180 or 270, not {rotation!r}")
        with PDFIUM_LOCK:
            page = self._page(index)
            try:
                bitmap = page.render(scale=dpi / 72.0, rotation=rotation, grayscale=gray, may_draw_forms=False)
                try:
                    img = bitmap.to_pil()
                    if img.mode in ("L", "RGBA", "RGBX"):
                        img = img.copy()          # never keep a view of pdfium's buffer past close()
                    if img.mode not in ("L", "RGB"):
                        img = img.convert("L" if gray else "RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
        return img

    def render(self, index: int, dpi: float, dest: Path) -> PageImage:   # PNG written with pHYs dpi
        """Render the displayed page (its /Rotate honoured, CropBox only) to a PNG carrying its dpi.

        The dpi is lowered to fit 60 MP and ``MAX_RENDER_SIDE`` if it would exceed them;
        ``PageImage.dpi`` is the dpi used.
        """
        (w, h), box = self._geometry(index)
        used, _, scaled = _fit_dpi(w, h, float(dpi))
        self._note_scaled(index, scaled)
        img = self._bitmap(index, used, 0, False)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            img.save(dest, format="PNG", dpi=(used, used), compress_level=1)
            size = img.size
        finally:
            img.close()
        return PageImage(index=index, path=dest, width=size[0], height=size[1], dpi=used, cropbox=box)

    def render_scaled(self, index: int, dpi: float, *, rotation: int = 0, gray: bool = False) -> "PIL.Image.Image":
        """An in-memory render (for the orientation vote and OSD); ``rotation`` is clockwise, on top of
        /Rotate. Lowered to fit 60 MP and ``MAX_RENDER_SIDE`` like ``render``."""
        (w, h), _ = self._geometry(index)
        used, _, scaled = _fit_dpi(w, h, float(dpi))
        img = self._bitmap(index, used, rotation, gray)
        self._note_scaled(index, scaled)
        return img


def upscale_for_osd(image: "PIL.Image.Image", dpi: float, target_dpi: float = 300.0) -> "PIL.Image.Image":
    """A copy scaled up to about ``target_dpi`` (OSD is unreliable on coarser renders).

    An image already at or above the target comes back as is (the same object). The result is
    never larger than 60 MP, and no side of it is longer than ``MAX_RENDER_SIDE``.
    """
    from PIL import Image

    if not (math.isfinite(dpi) and dpi > 0 and math.isfinite(target_dpi) and target_dpi > 0):
        raise ValueError(f"dpi and target_dpi must be positive numbers, not {dpi!r} and {target_dpi!r}")
    if dpi >= target_dpi:
        return image
    factor = target_dpi / dpi
    w, h = image.size
    if w * h * factor * factor > MAX_RENDER_PIXELS:
        factor = math.sqrt(MAX_RENDER_PIXELS / max(w * h, 1))
    if max(w, h) * factor > MAX_RENDER_SIDE:
        factor = MAX_RENDER_SIDE / max(w, h, 1)
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    if nw * nh > MAX_RENDER_PIXELS or max(nw, nh) > MAX_RENDER_SIDE:
        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.Resampling.LANCZOS)
