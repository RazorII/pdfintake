"""The invisible OCR text layer.

The layer writer gets only ``LayerPage`` objects (text plus pixel boxes of the shared render) and
knows nothing about engines. For every page it is given it:

1. filters each line by the fonts' cmap coverage (``coverage_filter``): a character no layer font
   has is dropped and counted, never written as a missing glyph;
2. splits the line into runs by font (``split_runs``): a character goes to a font that claims its
   script, other characters (digits, punctuation) to the font with the best ``priority``, and a
   combining mark stays with its base character;
3. sizes and places the line inside its box (``fit_line``): widths are the fonts' hmtx advances,
   rounded exactly as fpdf2 rounds them, so a run lands where pdfium will put it; every glyph is
   kept inside the CropBox with 0.5 pt of padding, and a line that can't fit at the 1.0 pt floor
   is moved left or wrapped rather than allowed to overflow.

``write_layer`` then draws every page's runs into one ``FPDF(unit="pt")`` document in
``TextMode.INVISIBLE`` only, so each font is subset once for the whole file. Each page of that
document becomes our own Form XObject, named ``/PdfIntakeOCR0`` (``/PdfIntakeOCR1`` and up when
the name is taken on that page) rather than pikepdf's ``Page.add_overlay``, which names its form
with ``Name.random()`` and so made every build's bytes different. The form is drawn first, in
the page's initial graphics state; the page's existing content follows, wrapped in ``q ... Q``
and left byte for byte as it was. The layer document's Info is never copied.
Then ``retag_subsets`` gives each font subset this call copied in a tag derived from its embedded
font file (fpdf2 writes the same ``MPDFAA+`` tag on every font), and the page gets our
``/PieceInfo`` marker, which ``classify`` uses to recognise our own layer later.

Coordinates. A ``PlacedRun``'s ``x`` and ``y`` are points measured from the top-left corner of the
page as the render shows it (fpdf2's convention: y down, ``y`` is the baseline). Pages with a
``/Rotate`` get a form matrix that maps the displayed orientation back to user space.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import struct
import threading
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import IntegrityError, PackError, SetupError
from ..types import LayerLine, LayerPage

if TYPE_CHECKING:
    import pikepdf

OVERLAY_NAME = "/PdfIntakeOCR0"     # a page-unique suffix (/PdfIntakeOCR1, ...) if the name is taken
MARKER_KEY = "/RazorII_pdfintake"   # the /PieceInfo key that marks a page carrying our layer

LAYER_PAD = 0.5          # pt of slack kept inside the CropBox on every side
MIN_SIZE = 1.0           # pt; the floor before a line is moved or wrapped
LINE_FILL = 0.85         # font size as a share of the box height (one visual line per source line)
_INK_SLACK = 2.0         # 1000-unit em slack on every ink edge, for pdfium's whole-unit rounding

FPDF_FONT_SUFFIXES = (".otb", ".otf", ".otc", ".ttf", ".ttc", ".woff", ".woff2")   # what fpdf2 2.9.0 opens
FPDF_TAG = "MPDFAA"      # the subset tag fpdf2 2.9.0 writes on every TrueType font

_OVERLAY_RE = re.compile(r"^/PdfIntakeOCR[0-9]+$")
_TAG_RE = re.compile(r"^/?([A-Z]{6})\+")
_MARKER_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
_MARKER_VALUE_RE = re.compile(r"^[\x20-\x7e]{0,200}$")
_PDF_DATE_RE = re.compile(r"^D:[0-9]{4}([0-9]{2}){0,5}(Z|[+-][0-9]{2}'?([0-9]{2}'?)?)?$")
_MISSING_GLYPHS = "is missing the following glyphs"
_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
# The marker date when the caller gives none: fixed, so two builds of one input have the same
# bytes (the layer document's own creation date is the same instant).
DEFAULT_MARKER_DATE = "D:20000101000000Z"
# Unicode's Default_Ignorable_Code_Point property (DerivedCoreProperties.txt, Unicode 15.1). These
# carry no text of their own; pdfium extracts several of them (U+200B to U+200D, U+2066 to U+2069,
# U+FE00) as a line break or not at all, which splits a word or leaves the layer disagreeing with
# what it was told it wrote. So they are left out of the layer, and not counted as dropped.
_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
    (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
    (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)
_IGNORABLE = frozenset(chr(cp) for lo, hi in _IGNORABLE_RANGES for cp in range(lo, hi + 1))
_IGNORABLE_TABLE = {ord(c): None for c in _IGNORABLE}


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class FontSpec:
    key: str
    path: Path
    sha256: str
    covers: tuple[str, ...]      # script names (ocr.scripts.SCRIPTS) this font claims
    priority: int                # lower wins among fonts that don't claim a character's script


@dataclass(frozen=True)
class Run:
    font_key: str
    text: str


@dataclass(frozen=True)
class PlacedRun:
    font_key: str
    text: str
    x: float          # pt from the left edge of the displayed CropBox
    y: float          # pt from the top edge of the displayed CropBox down to the baseline
    size: float       # pt, rounded to 0.01 as fpdf2 writes it


@dataclass
class LayerReport:
    chars_written: dict[int, int]          # page index -> non-whitespace characters written
    expected_text: dict[int, str]          # page index -> exactly what was written, one visual line per row
    # "U+4E2D" -> count of characters left out of the layer: ones no layer font covers, and every
    # character of a line that couldn't be placed (a box with a non-finite corner, a page too small
    # to hold any text, a line that would need a size under 0.01 pt), so a loss always reaches the
    # record. Not counted: whitespace (it becomes a space), default-ignorable characters (U+200B,
    # ZWJ, variation selectors ...), and a combining mark repeated in place, which pdfium would
    # extract once anyway; none of these carry text.
    dropped: Counter[str]
    runs: int                              # text runs written over all pages
    font_tags: list[str]                   # the new subset tags, in the order retag_subsets gave them


@dataclass
class _Font:
    """One verified layer font and the metrics the writer needs, in 1000-unit em space."""

    key: str
    path: Path
    sha256: str
    covers: frozenset[str]
    priority: int
    rank: int                              # position after sorting by (priority, input order)
    cmap: dict[int, str]
    cw: dict[int, int]                     # advance per code point, rounded exactly as fpdf2 rounds it
    default_w: int                         # fpdf2's width for a code point the cmap lacks (.notdef)
    asc: float                             # highest ink or ascent above the baseline
    desc: float                            # lowest ink or descent below the baseline (positive)
    hhea_desc: float                       # hhea descent (positive), for placing the baseline
    _ttfont: Any = field(repr=False)
    _scale: float = field(repr=False)
    _bounds: dict[str, tuple[float, float] | None] = field(default_factory=dict, repr=False)
    # The TTFont is opened lazily, and fontTools' lazy glyf table is not safe to touch from two
    # threads at once (measured: KeyError on a glyph name). Every glyph read goes through this lock.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def ink(self, cp: int) -> tuple[float, float]:
        """(left, right) of the glyph's ink relative to its pen position, widened to its advance."""
        adv = float(self.cw.get(cp, self.default_w))
        name = self.cmap.get(cp)
        b = None
        if name is not None:
            try:
                b = self._bounds[name]
            except KeyError:
                with self._lock:
                    if name not in self._bounds:
                        self._bounds[name] = self._glyph_bounds(name)
                    b = self._bounds[name]
        if b is None:
            return -_INK_SLACK, adv + _INK_SLACK
        return min(0.0, b[0]) - _INK_SLACK, max(adv, b[1]) + _INK_SLACK

    def _glyph_bounds(self, name: str) -> tuple[float, float] | None:
        """The glyph's (xMin, xMax) in 1000 units, None for an empty glyph. Only a glyph the font
        doesn't hold, or whose outline data fontTools can't decode, falls back to the font's head
        bbox (wider, so still safe); any other error raises."""
        from fontTools.ttLib import TTLibError

        font = self._ttfont
        head = font["head"]
        fallback = (head.xMin * self._scale, head.xMax * self._scale)
        try:
            if "glyf" in font:
                glyf = font["glyf"]
                if name not in glyf:
                    return fallback
                g = glyf[name]
                if getattr(g, "numberOfContours", 0) == 0:
                    return None
                if not hasattr(g, "xMin"):
                    g.recalcBounds(glyf)
                return g.xMin * self._scale, g.xMax * self._scale
            from fontTools.pens.boundsPen import BoundsPen

            gs = font.getGlyphSet()
            if name not in gs:
                return fallback
            pen = BoundsPen(gs)
            gs[name].draw(pen)
            if pen.bounds is None:
                return None
            return pen.bounds[0] * self._scale, pen.bounds[2] * self._scale
        except (struct.error, TTLibError, IndexError, ValueError, AssertionError):
            return fallback                     # outline data fontTools can't decode


@dataclass
class _Cluster:
    text: str
    font: _Font
    adv: int                  # 1000-unit advance of the whole cluster
    ink_l: float              # ink extent relative to the cluster's pen, 1000 units
    ink_r: float
    space: bool


# --------------------------------------------------------------------------- the font set


def _scripts() -> Any:
    from ..ocr import scripts

    return scripts


class FontSet:
    """The verified layer fonts, in priority order. Build it with ``from_specs``."""

    def __init__(self, fonts: Sequence[_Font], script_of: Any) -> None:
        self._fonts: tuple[_Font, ...] = tuple(sorted(fonts, key=lambda f: f.rank))
        self._by_key: dict[str, _Font] = {f.key: f for f in self._fonts}
        self._script_of = script_of
        self._choice: dict[str, tuple[_Font | None, str]] = {}
        self._script: dict[str, str | None] = {}
        self._metrics: dict[tuple[int, str], tuple[int, float, float, bool]] = {}

    @classmethod
    def from_specs(cls, specs: Sequence[FontSpec]) -> "FontSet":
        """Load and check every font: its sha256, that fpdf2 can open its format, and that its cmap
        covers the required set of every script it claims (a disguised font, such as a Latin font
        renamed NotoSansThai, is refused with SetupError)."""
        from fontTools.ttLib import TTFont

        if not specs:
            raise PackError("no_layer_fonts", "The text layer needs at least one font, and none was given.")
        keys = [s.key for s in specs]
        if len(set(keys)) != len(keys) or any(not isinstance(k, str) or not k for k in keys):
            raise PackError("layer_font_keys", "The layer fonts need distinct, non-empty keys.", keys=keys)
        for spec in specs:
            if not isinstance(spec.sha256, str) or not _SHA256_RE.fullmatch(spec.sha256):
                raise PackError("layer_font_spec", f"The layer font {spec.key!r} needs a sha256 of 64 hex characters.",
                                key=spec.key)
            if not isinstance(spec.covers, (tuple, list)) or not all(isinstance(s, str) for s in spec.covers):
                raise PackError("layer_font_spec", f"The layer font {spec.key!r} needs its scripts as a list of names.",
                                key=spec.key)
            if isinstance(spec.priority, bool) or not isinstance(spec.priority, int):
                raise PackError("layer_font_spec", f"The layer font {spec.key!r} needs a whole-number priority.",
                                key=spec.key)
        scripts = _scripts()
        order = sorted(range(len(specs)), key=lambda i: (specs[i].priority, i))
        fonts: list[_Font] = []
        for rank, i in enumerate(order):
            spec = specs[i]
            # Resolved now: fpdf2 opens the file again at write time, and a relative path would be
            # looked up against whatever the working directory (or FPDF_FONT_DIR) is by then.
            path = Path(spec.path).resolve()
            if path.suffix.lower() not in FPDF_FONT_SUFFIXES:
                raise SetupError("layer_font_format", f"The layer font {path} isn't a font file pdfintake can embed.",
                                 path=str(path))
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise IntegrityError("layer_font_unreadable", f"The layer font {path} can't be read ({exc}).",
                                     path=str(path)) from exc
            digest = hashlib.sha256(data).hexdigest()
            if digest != spec.sha256.lower():
                raise IntegrityError("layer_font_hash", f"The layer font {path} doesn't match its pinned sha256.",
                                     path=str(path), expected=spec.sha256, got=digest)
            try:
                tt = TTFont(io.BytesIO(data), recalcTimestamp=False, lazy=True)
                cmap = dict(tt.getBestCmap() or {})
                upem = float(tt["head"].unitsPerEm)
                hmtx = tt["hmtx"].metrics
                head, hhea = tt["head"], tt["hhea"]
            except Exception as exc:
                raise SetupError("layer_font_unreadable", f"The layer font {path} can't be parsed ({type(exc).__name__}).",
                                 path=str(path)) from exc
            if not cmap:
                raise SetupError("layer_font_cmap", f"The layer font {path} has no Unicode cmap.", path=str(path))
            for script in spec.covers:
                info = scripts.SCRIPTS.get(script)
                if info is None:
                    raise PackError("unknown_script", f"The layer font {path} claims the script {script!r}, "
                                    "which this core doesn't know.", path=str(path), script=script)
                missing = sorted(set(info.required) - cmap.keys())
                if missing:
                    shown = ", ".join(f"U+{cp:04X}" for cp in missing[:8])
                    more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
                    raise SetupError("layer_font_coverage",
                                     f"The layer font {path} claims {script} but lacks {len(missing)} of the "
                                     f"characters {script} needs ({shown}{more}).",
                                     path=str(path), script=script, missing=len(missing))
            scale = 1000.0 / upem
            try:
                cw = {cp: _fpdf_width(hmtx[name][0], scale) for cp, name in cmap.items()}
                default_w = _fpdf_width(hmtx[".notdef"][0], scale) if ".notdef" in hmtx else 0
            except KeyError as exc:
                raise SetupError("layer_font_unreadable", f"The layer font {path} lacks a width for a mapped glyph.",
                                 path=str(path)) from exc
            # pdfium reports a CID TrueType glyph's top as top + top // 64 in 1000-unit space
            # (measured: a 700-unit box comes back as 710, an 800-unit one as 812), and rounds
            # every edge to whole units, so the extents carry that growth and _INK_SLACK units more.
            asc = max(float(head.yMax), float(hhea.ascent)) * scale * 65.0 / 64.0 + _INK_SLACK
            desc = max(-float(head.yMin), -float(hhea.descent), 0.0) * scale + _INK_SLACK
            fonts.append(_Font(key=spec.key, path=path, sha256=digest, covers=frozenset(spec.covers),
                               priority=int(spec.priority), rank=rank, cmap=cmap, cw=cw, default_w=default_w,
                               asc=asc, desc=desc, hhea_desc=max(-float(hhea.descent), 0.0) * scale,
                               _ttfont=tt, _scale=scale))
        return cls(fonts, scripts.script_of)

    # -- lookups

    @property
    def keys(self) -> tuple[str, ...]:
        """Font keys in priority order."""
        return tuple(f.key for f in self._fonts)

    def font_for(self, ch: str) -> str | None:
        """The key of the font for ``ch``: one character, or a base character followed by its
        combining marks (the base decides, so a mark follows its base). None when no font has it."""
        if not ch:
            return None
        font, _ = self._choose(ch)
        return font.key if font is not None else None

    def advance(self, font_key: str, text: str, size: float) -> float:
        """Width of ``text`` in ``font_key`` at ``size`` pt: hmtx sums, shaping off, rounded per
        code point exactly as fpdf2 rounds them (so it equals fpdf2's get_string_width)."""
        f = self._by_key[font_key]
        cw, dw = f.cw, f.default_w
        return sum(cw.get(ord(c), dw) for c in text) * size / 1000.0

    # -- internals

    def _script_for(self, ch: str) -> str | None:
        if ch not in self._script:
            try:
                self._script[ch] = self._script_of(ch)
            except Exception:
                self._script[ch] = None
        return self._script[ch]

    def _choose(self, cluster: str) -> tuple[_Font | None, str]:
        """(font, kept cluster text) for a base character plus its marks. Marks the chosen font
        lacks are left out of the kept text."""
        hit = self._choice.get(cluster)
        if hit is not None:
            return hit
        base = cluster[0]
        cp = ord(base)
        cands = [f for f in self._fonts if cp in f.cmap]
        if not cands:
            result: tuple[_Font | None, str] = (None, "")
        else:
            script = self._script_for(base)
            if script is not None and any(script in f.covers for f in cands):
                cands.sort(key=lambda f: (script not in f.covers, f.rank))
            marks = cluster[1:]
            chosen = next((f for f in cands if all(ord(m) in f.cmap for m in marks)), cands[0])
            kept = base + "".join(m for m in marks if ord(m) in chosen.cmap)
            result = (chosen, kept)
        if len(self._choice) < 65536:
            self._choice[cluster] = result
        return result

    def _segment(self, text: str) -> tuple[list[tuple[_Font, str]], Counter[str]]:
        """Clusters of ``text`` with their fonts, plus what was dropped. Whitespace that is a
        control character (tab, newline), or that no layer font has (U+3000, U+1680 ...), becomes a
        plain space, so the words on either side stay apart; other control characters are dropped.
        Default-ignorable characters and a mark repeated in place are left out uncounted."""
        out: list[tuple[_Font, str]] = []
        dropped: Counter[str] = Counter()
        current: _Font | None = None
        if _IGNORABLE.intersection(text):
            text = text.translate(_IGNORABLE_TABLE)
        for cluster in _clusters(text):
            if len(cluster) > 1:                        # a mark repeated in place: pdfium extracts it once
                dup = [i for i in range(1, len(cluster)) if cluster[i] == cluster[i - 1]]
                if dup:
                    cluster = "".join(c for i, c in enumerate(cluster) if i not in dup)
            base = cluster[0]
            if base.isspace() and base != " ":
                if unicodedata.category(base) == "Cc" or not any(ord(base) in f.cmap for f in self._fonts):
                    cluster = " " + cluster[1:]
                    base = " "
            elif unicodedata.category(base) == "Cc":
                dropped.update(_code(c) for c in cluster)
                continue
            if base.isspace() and current is not None and all(ord(c) in current.cmap for c in cluster):
                font, kept = current, cluster
            else:
                font, kept = self._choose(cluster)
            if font is None:
                dropped.update(_code(c) for c in cluster)
                continue
            if len(kept) != len(cluster):
                lost = Counter(cluster)
                lost.subtract(Counter(kept))
                dropped.update({_code(c): n for c, n in lost.items() if n > 0})
            out.append((font, kept))
            current = font
        return out, dropped


def _fpdf_width(units: int, scale: float) -> int:
    w = 0 if units == 65535 else units
    return round(scale * w + 0.001)       # fpdf2 2.9.0 TTFFont: ROUND_HALF_UP


def _code(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def _clusters(text: str) -> Iterator[str]:
    """A base character followed by its combining marks (category M*); a leading mark stands alone."""
    start = 0
    n = len(text)
    i = 1
    while i <= n:
        if i == n or not unicodedata.category(text[i]).startswith("M"):
            yield text[start:i]
            start = i
        i += 1


# --------------------------------------------------------------------------- filtering and runs


def coverage_filter(text: str, fonts: FontSet) -> tuple[str, Counter[str]]:
    """``text`` as the layer can carry it, and a count of what was dropped ("U+4E2D" -> 2).

    A character is kept when a layer font covers it; a combining mark is kept when the font chosen
    for its base covers it. Whitespace control characters, and whitespace no layer font has,
    become a space; other control characters are dropped. Default-ignorable characters (U+200B,
    ZWJ, ZWNJ, variation selectors ...) and a mark repeated in place are left out, uncounted.
    """
    segs, dropped = fonts._segment(text)
    return "".join(t for _, t in segs), dropped


def split_runs(text: str, fonts: FontSet) -> list[Run]:
    """``text`` split into runs by font. Characters no font covers are left out, exactly as
    ``coverage_filter`` leaves them out; whitespace stays in the run it follows."""
    segs, _ = fonts._segment(text)
    runs: list[list[Any]] = []
    for font, t in segs:
        if runs and runs[-1][0] is font:
            runs[-1][1].append(t)
        else:
            runs.append([font, [t]])
    return [Run(f.key, "".join(parts)) for f, parts in runs]


def _cluster_metrics(segs: Sequence[tuple[_Font, str]], fonts: FontSet) -> list[_Cluster]:
    cache = fonts._metrics
    out: list[_Cluster] = []
    for font, t in segs:
        key = (font.rank, t)
        m = cache.get(key)
        if m is None:
            pen = 0.0
            lo, hi = 0.0, 0.0
            for c in t:
                cp = ord(c)
                il, ir = font.ink(cp)
                lo = min(lo, pen + il)
                hi = max(hi, pen + ir)
                pen += font.cw.get(cp, font.default_w)
            m = (int(pen), lo, max(hi, pen), t.isspace())
            if len(cache) < 65536:
                cache[key] = m
        out.append(_Cluster(t, font, m[0], m[1], m[2], m[3]))
    return out


def _extent(clusters: Sequence[_Cluster]) -> tuple[float, float, float]:
    """(left, right, advance) of a sequence of clusters, 1000 units from its first pen position."""
    pen = 0.0
    lo, hi = math.inf, -math.inf
    for c in clusters:
        lo = min(lo, pen + c.ink_l)
        hi = max(hi, pen + c.ink_r)
        pen += c.adv
    if lo == math.inf:
        return 0.0, 0.0, 0.0
    return lo, hi, pen


def _wrap(clusters: Sequence[_Cluster], size: float, limit: float) -> list[list[_Cluster]]:
    """Greedy word wrap to ``limit`` pt at ``size``; a word wider than the limit is split between
    clusters. Extents are kept incrementally, so a long line costs one pass."""
    k = size / 1000.0
    words: list[tuple[list[_Cluster], list[_Cluster]]] = []     # (spaces before, word)
    spaces: list[_Cluster] = []
    word: list[_Cluster] = []
    for c in clusters:
        if c.space:
            if word:
                words.append((spaces, word))
                spaces, word = [], []
            spaces.append(c)
        else:
            word.append(c)
    if word:
        words.append((spaces, word))

    pieces: list[list[_Cluster]] = []
    cur: list[_Cluster] = []
    lo = hi = pen = 0.0

    def start(seq: Sequence[_Cluster]) -> None:
        nonlocal cur, lo, hi, pen
        cur = []
        lo, hi, pen = math.inf, -math.inf, 0.0
        extend(seq)

    def extend(seq: Sequence[_Cluster]) -> None:
        nonlocal lo, hi, pen
        for c in seq:
            lo = min(lo, pen + c.ink_l)
            hi = max(hi, pen + c.ink_r)
            pen += c.adv
            cur.append(c)

    def fits_with(seq: Sequence[_Cluster]) -> bool:
        l2, h2, p = lo, hi, pen
        for c in seq:
            l2 = min(l2, p + c.ink_l)
            h2 = max(h2, p + c.ink_r)
            p += c.adv
        return (h2 - l2) * k <= limit

    for sp, w in words:
        if cur and fits_with(sp + w):
            extend(sp + w)
            continue
        if cur:
            pieces.append(cur)
        start([])
        for c in w:                                 # split between clusters, one pass
            one = (c,)
            if cur and not fits_with(one):
                pieces.append(cur)
                start([])
            extend(one)
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]


def _strip_spaces(clusters: list[_Cluster]) -> list[_Cluster]:
    a, b = 0, len(clusters)
    while a < b and clusters[a].space:
        a += 1
    while b > a and clusters[b - 1].space:
        b -= 1
    return clusters[a:b]


def _floor2(v: float) -> float:
    return math.floor(v * 100.0 + 1e-9) / 100.0


def fit_line(line: LayerLine, scale: tuple[float, float], cropbox: tuple[float, float, float, float],
             fonts: FontSet) -> list[PlacedRun]:
    """Size and place one source line inside its box and inside the page.

    ``scale`` is (pt per px across, pt per px down) of the render; ``cropbox`` is the page as the
    render shows it (only its width and height are used). The size is 0.85 of the box height,
    shrunk to the box width, floored at 1.0 pt and capped so the font's ink fits the page height.
    A line still wider than the room to the right of its box moves left, and wraps only when it is
    wider than the page; wrapped rows never overprint, so a line whose rows would be taller than
    the page shrinks below the floor instead. Every glyph's ink stays at least 0.5 pt inside the
    page. A line that can't be placed at all (a non-finite or malformed box, a page under about
    1 pt, a size under 0.01 pt) gives [], and ``write_layer`` counts its characters as dropped.
    """
    segs, _ = fonts._segment(line.text)
    return _fit(segs, line.box, scale, cropbox, fonts)


def _fit(segs: Sequence[tuple[_Font, str]], box: Any, scale: tuple[float, float],
         cropbox: tuple[float, float, float, float], fonts: FontSet) -> list[PlacedRun]:
    sx, sy = float(scale[0]), float(scale[1])
    page_w = abs(float(cropbox[2]) - float(cropbox[0]))
    page_h = abs(float(cropbox[3]) - float(cropbox[1]))
    if not all(math.isfinite(v) for v in (sx, sy, page_w, page_h)) or sx <= 0 or sy <= 0:
        return []
    inner_w = page_w - 2 * LAYER_PAD
    inner_h = page_h - 2 * LAYER_PAD
    if inner_w <= 0 or inner_h <= 0:
        return []
    try:
        xs = [float(p[0]) * sx for p in box]
        ys = [float(p[1]) * sy for p in box]
    except (TypeError, ValueError, IndexError):
        return []
    if len(xs) != 4 or not all(math.isfinite(v) for v in xs + ys):
        return []
    clusters = _strip_spaces(_cluster_metrics(segs, fonts))
    if not clusters:
        return []
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)

    used = {c.font.key: c.font for c in clusters}
    top = max(f.asc for f in used.values())
    bottom = max(f.desc for f in used.values())
    base_desc = max(f.hhea_desc for f in used.values())

    x0 = min(max(bx0, LAYER_PAD), page_w - LAYER_PAD)
    avail = max(1.0, page_w - LAYER_PAD - x0)
    box_w = max(1.0, min(min(bx1, page_w - LAYER_PAD) - x0, avail))
    box_h = max(1.0, by1 - by0)

    lo, hi, _ = _extent(clusters)
    ext1 = (hi - lo) / 1000.0                       # ink width at 1 pt
    size = box_h * LINE_FILL
    if ext1 > 0:
        size = min(size, box_w / ext1)
    size = max(MIN_SIZE, size)
    if top + bottom > 0:
        size = min(size, inner_h / ((top + bottom) / 1000.0))
    size = _floor2(size)
    if size < 0.01:
        return []

    pieces = _pieces(clusters, size, ext1, inner_w)
    row1 = (top + bottom) / 1000.0                  # one row's ink height at 1 pt
    # Wrapped rows are stacked at least one ink height apart, so they never overprint: pdfium
    # extracts a glyph drawn twice at one spot only once. When the rows at this size are taller
    # than the page, the size shrinks (below the 1.0 pt floor if it must) and the line re-wraps.
    while len(pieces) > 1 and row1 > 0 and len(pieces) * row1 * size > inner_h:
        ratio = inner_h / (len(pieces) * row1 * size)
        smaller = _floor2(size * min(0.95, max(0.5, math.sqrt(ratio))))
        if smaller < 0.01 or smaller >= size:
            return []
        size = smaller
        pieces = _pieces(clusters, size, ext1, inner_w)
    if not pieces:
        return []

    out: list[PlacedRun] = []
    n = len(pieces)
    step = box_h / n
    first = by0
    if n > 1:
        # The rows run on below the box, and the stack moves up if it would leave the page.
        step = max(step, row1 * size)
        if n * step > inner_h:
            step = inner_h / n                      # still at least one ink height: see the loop above
        first = min(max(by0, LAYER_PAD), page_h - LAYER_PAD - n * step)
    for i, piece in enumerate(pieces):
        plo, phi, _ = _extent(piece)
        width1 = (phi - plo) / 1000.0
        psize = size
        if width1 * psize > inner_w:                # one cluster wider than the page: shrink it
            psize = _floor2(inner_w / width1)
            if psize < 0.01:
                continue
        k = psize / 1000.0
        left = x0
        if left + width1 * psize > page_w - LAYER_PAD:
            left = page_w - LAYER_PAD - width1 * psize
        left = max(LAYER_PAD, left)
        pen0 = left - plo * k                       # the pen position that puts the ink at `left`
        if n == 1:
            y = by0 + step - base_desc * k          # the descent sits on the box's bottom edge
        else:
            y = first + (i + 1) * step - bottom * k
        y = min(max(y, LAYER_PAD + top * k), page_h - LAYER_PAD - bottom * k)
        y = round(y, 2)
        offset = 0
        for font, text in _merge_runs(piece):
            x = round(pen0 + offset * k, 2)
            out.append(PlacedRun(font.key, text, x, y, psize))
            offset += sum(font.cw.get(ord(c), font.default_w) for c in text)
    return out


def _pieces(clusters: list[_Cluster], size: float, ext1: float, inner_w: float) -> list[list[_Cluster]]:
    if ext1 * size <= inner_w:
        return [clusters]
    pieces = [_strip_spaces(p) for p in _wrap(clusters, size, inner_w)]
    return [p for p in pieces if p]


def _merge_runs(piece: Sequence[_Cluster]) -> list[tuple[_Font, str]]:
    runs: list[list[Any]] = []
    for c in piece:
        if runs and runs[-1][0] is c.font:
            runs[-1][1].append(c.text)
        else:
            runs.append([c.font, [c.text]])
    return [(f, "".join(parts)) for f, parts in runs]


# --------------------------------------------------------------------------- page geometry


def _inherited(page_obj: Any, key: str) -> Any:
    node = page_obj
    for _ in range(64):
        if node is None:
            return None
        value = node.get(key)
        if value is not None:
            return value
        node = node.get("/Parent")
    return None


def _box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _page_box(page_obj: Any) -> tuple[float, float, float, float]:
    """The CropBox clipped to the MediaBox, as pdfium shows the page."""
    media = _box(_inherited(page_obj, "/MediaBox")) or (0.0, 0.0, 612.0, 792.0)
    crop = _box(_inherited(page_obj, "/CropBox"))
    if crop is None:
        return media
    x0, y0 = max(crop[0], media[0]), max(crop[1], media[1])
    x1, y1 = min(crop[2], media[2]), min(crop[3], media[3])
    if x1 <= x0 or y1 <= y0:
        return media
    return x0, y0, x1, y1


def _rotation(page_obj: Any) -> int:
    """The page's /Rotate as pdfium reads it: truncated to a quarter turn, 0..270."""
    value = _inherited(page_obj, "/Rotate")
    try:
        quarter = int(int(value) / 90)
    except (TypeError, ValueError):
        return 0
    return (quarter % 4) * 90


def _form_matrix(rot: int, box: tuple[float, float, float, float]) -> list[float]:
    """Maps the layer's own space (the page as displayed, origin bottom-left) to user space."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if rot == 90:
        return [0.0, 1.0, -1.0, 0.0, x0 + w, y0]
    if rot == 180:
        return [-1.0, 0.0, 0.0, -1.0, x0 + w, y0 + h]
    if rot == 270:
        return [0.0, -1.0, 1.0, 0.0, x0, y0 + h]
    return [1.0, 0.0, 0.0, 1.0, x0, y0]


# --------------------------------------------------------------------------- the writer


@dataclass
class _PagePlan:
    index: int
    width: float
    height: float
    matrix: list[float]
    runs: list[PlacedRun]
    rows: list[str]


def _check_marker(marker: Mapping[str, str]) -> tuple[str, list[tuple[str, str]]]:
    items: list[tuple[str, str]] = []
    date: str | None = None
    for key in sorted(marker, key=str):
        value = marker[key]
        if not isinstance(key, str) or not _MARKER_KEY_RE.fullmatch(key):
            raise ValueError(f"marker key {key!r} must be a short ASCII name")
        if not isinstance(value, str):
            raise ValueError(f"marker value for {key!r} must be a string")
        if key == "LastModified":
            if not _PDF_DATE_RE.fullmatch(value):
                raise ValueError("marker LastModified must be a PDF date such as D:YYYYMMDDHHmmSSZ")
            date = value
            continue
        if not _MARKER_VALUE_RE.fullmatch(value):
            raise ValueError(f"marker value for {key!r} must be printable ASCII of at most 200 characters")
        items.append((key, value))
    return (date if date is not None else DEFAULT_MARKER_DATE), items


@contextmanager
def _missing_glyphs_raise() -> Iterator[list[str]]:
    """Collects fpdf2's missing-glyph warnings while the layer document is built."""
    found: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            if _MISSING_GLYPHS in msg:
                found.append(msg)

    logger = logging.getLogger("fpdf.output")
    handler = _Catch(logging.WARNING)
    old_level = logger.level
    if not logger.isEnabledFor(logging.WARNING):
        logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield found
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _layer_document(plans: Sequence[_PagePlan], fonts: FontSet) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import TextMode

    keys = {r.font_key for p in plans for r in p.runs}
    used = [(i, f) for i, f in enumerate(fonts._fonts) if f.key in keys]
    # fpdf2 opens each font file again by path, so the file is re-checked just before it does, and
    # again after (and on any failure in between): a file swapped, truncated or removed meanwhile
    # is named as that, never as fpdf2's or fontTools' own error or as a missing glyph.
    for _, f in used:
        _check_font_file(f)
    try:
        family: dict[str, str] = {}
        doc = FPDF(unit="pt")
        doc.set_auto_page_break(False)
        doc.set_margins(0, 0, 0)
        doc.set_creation_date(datetime(2000, 1, 1, tzinfo=timezone.utc))   # never copied; keeps the bytes stable
        for i, f in used:
            family[f.key] = f"pdfintake{i}"
            doc.add_font(family[f.key], fname=str(f.path))
        with _missing_glyphs_raise() as found:
            for plan in plans:
                doc.add_page(format=(plan.width, plan.height))
                doc.text_mode = TextMode.INVISIBLE
                for r in plan.runs:
                    doc.set_font(family[r.font_key], size=r.size)
                    doc.text(r.x, r.y, r.text)
            missing = sorted({cp for font in doc.fonts.values() for cp in getattr(font, "missing_glyphs", [])})
            data = bytes(doc.output())
    except Exception:
        for _, f in used:
            _check_font_file(f)
        raise
    for _, f in used:                               # the file fpdf2 read must still be the verified one
        _check_font_file(f)
    if missing:
        raise SetupError("layer_missing_glyph",
                         f"A layer font lacks {len(missing)} glyph(s) it claimed to have "
                         f"({', '.join(f'U+{cp:04X}' for cp in missing[:8])}).", missing=len(missing))
    if found:
        raise SetupError("layer_missing_glyph", "fpdf2 reported a missing glyph while writing the layer.",
                         missing=len(found))
    return data


def _check_font_file(f: _Font) -> None:
    try:
        digest = hashlib.sha256(f.path.read_bytes()).hexdigest()
    except OSError as exc:
        raise IntegrityError("layer_font_unreadable", f"The layer font {f.path} can't be read ({exc}).",
                             path=str(f.path)) from exc
    if digest != f.sha256:
        raise IntegrityError("layer_font_hash", f"The layer font {f.path} changed after it was verified.",
                             path=str(f.path), expected=f.sha256, got=digest)


def _q_depth(page: "pikepdf.Page") -> int:
    """How many q the page content leaves open, so the wrapper can close them first."""
    import pikepdf

    try:
        ops = pikepdf.parse_content_stream(page, "q Q")
    except Exception:
        return 0
    depth = 0
    for _, op in ops:
        name = str(op)
        if name == "q":
            depth += 1
        elif name == "Q" and depth > 0:
            depth -= 1
    return depth


def _copy_dict(value: Any) -> Any:
    """A new direct dictionary with ``value``'s entries. Every dictionary the writer changes is
    rebuilt this way, never edited in place: an indirect one may be shared with other pages, and
    pikepdf shares direct sub-dictionaries between shallow copies (measured on pikepdf 10.17)."""
    import pikepdf

    if not isinstance(value, pikepdf.Dictionary):
        return pikepdf.Dictionary()
    return pikepdf.Dictionary({k: value[k] for k in value.keys()})


def _free_name(resources: Any) -> str:
    import pikepdf

    taken: set[str] = set()
    for _, sub in resources.items():
        if isinstance(sub, pikepdf.Dictionary):
            taken.update(str(k) for k in sub.keys())
    n = 0
    name = OVERLAY_NAME
    while name in taken:
        n += 1
        name = f"/PdfIntakeOCR{n}"
    return name


def _place(pdf: "pikepdf.Pdf", page: "pikepdf.Page", form: "pikepdf.Object") -> str:
    import pikepdf

    obj = page.obj
    old = obj.get("/Resources")
    if old is None:
        old = _inherited(obj, "/Resources")          # pushed down to the page, as qpdf would
    resources = _copy_dict(old)
    xobjects = _copy_dict(resources.get("/XObject"))
    name = _free_name(resources)
    xobjects[name] = form
    resources.XObject = xobjects
    obj.Resources = resources
    draw = f"q {name} Do Q\n".encode("ascii")
    if obj.get("/Contents") is None:
        page.contents_add(pdf.make_indirect(pikepdf.Stream(pdf, draw)))
        return name
    # The layer is drawn first, in the page's clean initial graphics state, and the original
    # content follows inside q ... Q. Drawn last, it was lost whenever a damaged page's content
    # ends inside an unterminated token (an open string, an inline image without EI): pdfium
    # swallows everything after it. The text is invisible, so stacking order doesn't matter.
    closing = b"\n" + b"Q\n" * (_q_depth(page) + 1)
    page.contents_add(pdf.make_indirect(pikepdf.Stream(pdf, draw + b"q\n")), prepend=True)
    page.contents_add(pdf.make_indirect(pikepdf.Stream(pdf, closing)))
    return name


def _mark(pdf: "pikepdf.Pdf", page: "pikepdf.Page", date: str, items: Sequence[tuple[str, str]]) -> None:
    import pikepdf

    obj = page.obj
    pieces = _copy_dict(obj.get("/PieceInfo"))       # other applications' entries are kept
    private = pikepdf.Dictionary({f"/{k}": pikepdf.String(v) for k, v in items})
    pieces[MARKER_KEY] = pikepdf.Dictionary(LastModified=pikepdf.String(date), Private=private)
    obj.PieceInfo = pieces
    obj.LastModified = pikepdf.String(date)          # required beside /PieceInfo (ISO 32000-1, 14.5)


def write_layer(pdf: "pikepdf.Pdf", pages: Sequence[LayerPage], fonts: FontSet, *,
                marker: Mapping[str, str]) -> LayerReport:
    """Write the invisible layer for ``pages`` into ``pdf`` (in place) and report what was written.

    One FPDF(unit="pt") document holds every page's layer, in TextMode.INVISIBLE only; fpdf2's
    missing-glyph warning raises SetupError. Each page's layer becomes our Form XObject at the
    CropBox origin, drawn before the existing content, which is wrapped in q...Q. The layer
    document's Info is never copied. Then retag_subsets (on the fonts copied in here only), then
    the /PieceInfo marker: ``marker`` entries go into its /Private dictionary, and
    ``marker["LastModified"]`` (a PDF date) is the marker's date. When it is absent the date is
    the fixed DEFAULT_MARKER_DATE, never the clock, so the same input gives the same bytes.
    A page with nothing to write gets no form and no marker. Characters of a line that can't be
    placed are counted in ``dropped`` with the uncovered ones.
    """
    import pikepdf

    date, items = _check_marker(marker)
    report = LayerReport({}, {}, Counter(), 0, [])
    plans: list[_PagePlan] = []
    seen: set[int] = set()
    count = len(pdf.pages)
    for lp in pages:
        if not 0 <= lp.index < count:
            raise ValueError(f"layer page {lp.index} is outside the document's {count} pages")
        if lp.index in seen:
            raise ValueError(f"layer page {lp.index} is given twice")
        if lp.render_w <= 0 or lp.render_h <= 0:
            raise ValueError(f"layer page {lp.index} has an empty render size")
        seen.add(lp.index)
        page_obj = pdf.pages[lp.index].obj
        box = _page_box(page_obj)
        rot = _rotation(page_obj)
        bw, bh = box[2] - box[0], box[3] - box[1]
        width, height = (bw, bh) if rot in (0, 180) else (bh, bw)
        shown = (box[0], box[1], box[0] + width, box[1] + height)
        scale = (width / lp.render_w, height / lp.render_h)
        runs: list[PlacedRun] = []
        rows: list[str] = []
        for line in lp.lines:
            segs, dropped = fonts._segment(line.text)     # coverage_filter and fit_line, segmented once
            report.dropped.update(dropped)
            placed = _fit(segs, line.box, scale, shown, fonts)
            want = [c for _, t in segs for c in t if not c.isspace()]
            have = sum(1 for r in placed for c in r.text if not c.isspace())
            if have != len(want):                       # a line (or a row of it) that couldn't be placed
                lost = Counter(want)
                lost.subtract(c for r in placed for c in r.text if not c.isspace())
                report.dropped.update({_code(c): n for c, n in lost.items() if n > 0})
            row: list[str] = []
            last: PlacedRun | None = None
            for r in placed:                            # a wrapped line restarts at the left
                if last is not None and (r.y != last.y or r.x <= last.x):
                    rows.append("".join(row))
                    row = []
                row.append(r.text)
                last = r
            if row:
                rows.append("".join(row))
            runs.extend(placed)
        report.expected_text[lp.index] = "\n".join(rows)
        report.chars_written[lp.index] = sum(1 for row in rows for c in row if not c.isspace())
        if runs:
            plans.append(_PagePlan(lp.index, width, height, _form_matrix(rot, box), runs, rows))
            report.runs += len(runs)
    if not plans:
        return report

    data = _layer_document(plans, fonts)
    with pikepdf.open(io.BytesIO(data)) as layer:
        if len(layer.pages) != len(plans):
            raise SetupError("layer_pages", "The layer document has the wrong number of pages.")
        copied: dict[tuple[int, int], Any] = {}
        ours: list[Any] = []                            # the Type0 fonts this call copied in
        for plan, lpage in zip(plans, layer.pages):
            resources = lpage.obj.get("/Resources")
            if resources is None:
                resources = pikepdf.Dictionary()
            elif not resources.is_indirect:
                resources = layer.make_indirect(resources)
            key = resources.objgen
            if key not in copied:
                copied[key] = pdf.copy_foreign(resources)
                fdict = copied[key].get("/Font")
                if isinstance(fdict, pikepdf.Dictionary):
                    ours += [fdict[k] for k in sorted(str(k) for k in fdict.keys())]
            content = _page_content(lpage)
            form = pikepdf.Stream(pdf, content)
            form.Type = pikepdf.Name.XObject
            form.Subtype = pikepdf.Name.Form
            form.FormType = 1
            form.BBox = pikepdf.Array([0, 0, plan.width, plan.height])
            form.Matrix = pikepdf.Array(plan.matrix)
            form.Resources = copied[key]
            _place(pdf, pdf.pages[plan.index], pdf.make_indirect(form))
    report.font_tags = retag_subsets(pdf, only=ours)
    for plan in plans:
        _mark(pdf, pdf.pages[plan.index], date, items)
    return report


def _page_content(page: "pikepdf.Page") -> bytes:
    import pikepdf

    contents = page.obj.get("/Contents")
    if contents is None:
        return b""
    if isinstance(contents, pikepdf.Array):
        return b"\n".join(s.read_bytes() for s in contents)
    return contents.read_bytes()


# --------------------------------------------------------------------------- subset tags


def _subset_tag(raw: bytes, taken: set[str]) -> str:
    digest = hashlib.sha256(raw).digest()
    while True:
        tag = "".join(chr(65 + b % 26) for b in digest[:6])
        if tag not in taken:
            return tag
        digest = hashlib.sha256(digest).digest()


def _named_layer_fonts(pdf: "pikepdf.Pdf") -> Iterator[Any]:
    """The fonts in every ``/PdfIntakeOCRn`` form on every page, in page, form and font-name order."""
    import pikepdf

    for page in pdf.pages:
        res = page.obj.get("/Resources")
        xobjects = res.get("/XObject") if isinstance(res, pikepdf.Dictionary) else None
        if not isinstance(xobjects, pikepdf.Dictionary):
            continue
        for name in sorted(str(k) for k in xobjects.keys()):
            if not _OVERLAY_RE.fullmatch(name):
                continue
            form = xobjects[name]
            if not isinstance(form, pikepdf.Stream) or form.get("/Subtype") != pikepdf.Name.Form:
                continue
            fres = form.get("/Resources")
            fdict = fres.get("/Font") if isinstance(fres, pikepdf.Dictionary) else None
            if isinstance(fdict, pikepdf.Dictionary):
                for fname in sorted(str(k) for k in fdict.keys()):
                    yield fdict[fname]


def _font_file(descriptor: Any) -> Any:
    for key in ("/FontFile2", "/FontFile3", "/FontFile"):
        stream = descriptor.get(key)
        if stream is not None:
            return stream
    return None


def retag_subsets(pdf: "pikepdf.Pdf", *, only: Sequence[Any] | None = None) -> list[str]:
    """Give fpdf2's font subsets a tag of six letters derived from the SHA-256 of each embedded
    font file's raw bytes, unique in the document (fpdf2 writes the same ``MPDFAA+`` on every font).

    ``only`` is the list of Type0 font objects to retag; ``write_layer`` passes exactly the fonts it
    copied in, so a font the input already had is never touched, even one that sits in a form an
    earlier tool happened to name ``/PdfIntakeOCRn``. Called without it, the fonts are found by
    name: every Type0 font still tagged ``MPDFAA+`` in a ``/PdfIntakeOCRn`` form on any page.
    Returns the new tags in order."""
    import pikepdf

    targets: list[tuple[Any, Any, Any, Any]] = []        # (Type0 font, descendant, descriptor, file)
    seen: set[tuple[int, int]] = set()
    for font in (_named_layer_fonts(pdf) if only is None else only):
        if not isinstance(font, pikepdf.Dictionary) or font.get("/Subtype") != pikepdf.Name.Type0:
            continue
        if font.is_indirect:
            if font.objgen in seen:
                continue
            seen.add(font.objgen)
        if not str(font.get("/BaseFont", "")).startswith(f"/{FPDF_TAG}+"):
            continue
        kids = font.get("/DescendantFonts")
        if not isinstance(kids, pikepdf.Array) or len(kids) == 0:
            continue
        cid = kids[0]
        desc = cid.get("/FontDescriptor") if isinstance(cid, pikepdf.Dictionary) else None
        ffile = _font_file(desc) if isinstance(desc, pikepdf.Dictionary) else None
        if ffile is None:
            continue
        targets.append((font, cid, desc, ffile))
    if not targets:
        return []
    mine = {o.objgen for t in targets for o in t[:3] if o.is_indirect}
    taken: set[str] = {FPDF_TAG}
    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Dictionary) or obj.objgen in mine:
            continue
        for key in ("/FontName", "/BaseFont"):
            value = obj.get(key)
            if isinstance(value, pikepdf.Name):
                m = _TAG_RE.match(str(value))
                if m:
                    taken.add(m.group(1))
    tags: list[str] = []
    for font, cid, desc, ffile in targets:
        tag = _subset_tag(ffile.read_raw_bytes(), taken)
        taken.add(tag)
        rest = str(font.BaseFont)[len(FPDF_TAG) + 2:]
        new = pikepdf.Name(f"/{tag}+{rest}")
        font.BaseFont = new
        cid.BaseFont = new
        desc.FontName = new
        tags.append(tag)
    return tags
