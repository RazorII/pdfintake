"""Repairing broken font maps: the missing ``/ToUnicode`` entries, proven, added losslessly (DESIGN §15).

A born-digital page can look right and still read as garbage, because its fonts carry no map from
glyph codes to letters (no ``/ToUnicode``, or one that misses the codes in use). Drawing needs only
the glyphs; searching and copying need the map. This module finds the codes that need one, works
out which letter each glyph draws, and writes the map. Nothing else in the file changes: content
streams, font programs, images and annotations stay byte for byte, and verify proves every page
renders as before.

The steps (``analyse``, then ``decide``, then ``write``):

1. **Fonts.** Every font a page (or a form it draws) uses is described (``Font``): its program,
   code to glyph, the mappings it already has, and whether a map may be written (``scope``; DESIGN
   §15.1: only where no common renderer picks glyphs through ``/ToUnicode``).
2. **The probe.** On a work copy each font that needs a map gets a temporary ``/ToUnicode`` that
   maps every code to its own private-use code point, so pdfium's text API reports every drawn
   glyph's font, code, box, page and visibility (``Glyph``) through its own layout. More codes than
   private-use points take more passes. A second pass over the input itself gives what pdfium reads
   there today (``Glyph.baseline``), joined by box.
3. **Evidence, strongest first** (``Evidence``): the program's own cmap or glyph names; an installed
   font of the same family (outline hash, GID when the subset kept the numbering, GSUB read
   backwards); another mapped font of the same family in the document; then OCR on chosen pages
   (``ocr_pages`` and ``add_reads``), aligned per line and voted per code.
4. **Lookalikes** (§15.3) are widened with their twins in Latin, Greek and Cyrillic and decided by
   the script of the other letters in their words; a code with no deciding context stays open.
5. **Decide and write.** A code is written only when proven; a used code still open in a font that
   gets a map reads as U+FFFD. A font whose proven entries equal what pdfium already reads isn't
   changed.

Nothing here asks, starts a process, or touches the network; installed fonts are only read.
Pages are 0-based, as everywhere in pdfintake.
"""

from __future__ import annotations

import ctypes
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from . import fontprog
from .fontprog import Program, ProgramError, Reference, ReferenceIndex
from .render import MAX_FORM_DEPTH, PDFIUM_LOCK

if TYPE_CHECKING:
    import pikepdf

    from ..types import EngineRead, PageImage

PUA_FIRST = 0xE000
SENTINEL = 0xF8FF                 # a glyph of a probed font whose code this pass doesn't cover
PUA_SLOTS = SENTINEL - PUA_FIRST  # 6,399 codes per pass
MAX_PASSES = 16                   # more codes than this many passes cover leave the rest out of scope

OCR_PAGES_CAP = 12                # pages read for OCR evidence, at most, per document
OCR_TARGET = 5                    # stop adding pages for a code once it has been seen this many times
VOTE_MIN = 2                      # votes a code needs from OCR
VOTE_SHARE = 0.8                  # share of its counted votes the winner needs (all of them when only 2)
ONE_ENGINE_OCCURRENCES = 3        # a winner backed by one engine alone needs this many occurrences
LINE_MATCH_MIN = 0.6              # a paired line whose known glyphs match less than this casts no vote
OCR_ROUNDS = 4                    # re-alignments with the letters each round decided as new anchors
BAND = 12                         # the alignment's band around the diagonal, in characters

KNOWN_COLLECTIONS = frozenset({"Japan1", "GB1", "CNS1", "Korea1", "KR"})
UNPROVEN = "�"

Evidence = Literal["existing", "program", "installed", "document", "context", "ocr", "space"]
EVIDENCE_COUNTED: tuple[str, ...] = ("program", "installed", "document", "ocr")


# --------------------------------------------------------------------------- fonts

@dataclass
class Font:
    """One font dictionary of the document and what pdfintake knows about it."""

    ref: tuple[int, int]              # its object number and generation
    base: str
    family: str
    style: str
    two_byte: bool
    scope: str | None                 # None: a map may be written; else why not (a closed code)
    program: Program | None = None
    code_gid: dict[int, int] = field(default_factory=dict)
    existing: dict[int, str] = field(default_factory=dict)       # usable mappings it already has
    tounicode: dict[int, str | None] = field(default_factory=dict)
    implicit: bool = False            # symbolic TrueType without names: a printable reading is presumed right

    def probe_codes(self) -> list[int]:
        """The codes the probe must cover: every code with a glyph and no usable mapping yet."""
        return sorted(c for c in self.code_gid if c not in self.existing)


def _name(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text[1:] if text.startswith("/") else text


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


def _font_dicts(pdf: "pikepdf.Pdf") -> Iterator[Any]:
    """Every font dictionary page content (and the forms it draws, to pdfium's depth) can use, once each."""
    import pikepdf

    seen: set[tuple[int, int]] = set()
    seen_res: set[tuple[int, int]] = set()
    for page in pdf.pages:
        stack = [(_inherited(page.obj, "/Resources"), 0)]
        while stack:
            res, depth = stack.pop()
            if not isinstance(res, pikepdf.Dictionary) or depth > MAX_FORM_DEPTH:
                continue
            if res.objgen != (0, 0):
                if res.objgen in seen_res:
                    continue
                seen_res.add(res.objgen)
            fonts = res.get("/Font")
            if isinstance(fonts, pikepdf.Dictionary):
                for _key, font in fonts.items():
                    if not isinstance(font, pikepdf.Dictionary):
                        continue
                    og = font.objgen
                    if og == (0, 0) or og in seen:
                        continue
                    seen.add(og)
                    yield font
            xobjects = res.get("/XObject")
            if isinstance(xobjects, pikepdf.Dictionary):
                for _key, xo in xobjects.items():
                    if isinstance(xo, pikepdf.Stream) and _name(xo.get("/Subtype")) == "Form":
                        stack.append((xo.get("/Resources"), depth + 1))


def _program(descriptor: Any) -> tuple[Program | None, str | None]:
    import pikepdf

    if not isinstance(descriptor, pikepdf.Dictionary):
        return None, "not_embedded"
    if "/FontFile2" in descriptor:
        stream, subtype = descriptor["/FontFile2"], "FontFile2"
    elif "/FontFile3" in descriptor:
        stream = descriptor["/FontFile3"]
        subtype = _name(stream.get("/Subtype")) if isinstance(stream, pikepdf.Stream) else ""
    elif "/FontFile" in descriptor:
        return None, "type1_program"
    else:
        return None, "not_embedded"
    if not isinstance(stream, pikepdf.Stream):
        return None, "program_unreadable"
    try:
        length = int(stream.get("/Length1", 0) or 0)
        if length > fontprog.MAX_PROGRAM_BYTES:
            return None, "program_too_large"
        data = stream.read_bytes()
        return fontprog.load_program(data, subtype), None
    except ProgramError as exc:
        return None, str(exc) or "program_unreadable"
    except Exception:  # noqa: BLE001 - a stream that won't decode
        return None, "program_unreadable"


def _tounicode(font: Any) -> tuple[dict[int, str | None], bool]:
    """The font's existing /ToUnicode entries, and whether a /ToUnicode is there and readable."""
    import pikepdf

    stream = font.get("/ToUnicode")
    if not isinstance(stream, pikepdf.Stream):
        return {}, False
    try:
        return fontprog.parse_cmap(stream.read_bytes()), True
    except Exception:  # noqa: BLE001 - an unreadable map is no map
        return {}, False


def _usable(entries: Mapping[int, str | None]) -> dict[int, str]:
    return {c: t for c, t in entries.items() if t is not None and fontprog.usable_text(t)}


def _encoding_names(font: Any, program: Program | None) -> dict[int, str]:
    """code -> glyph name for a simple font: base encoding, then /Differences; a CFF program's own
    encoding when the font names none."""
    import pikepdf

    from fontTools.agl import UV2AGL

    names: dict[int, str] = {}
    enc = font.get("/Encoding")
    base = None
    diffs = None
    if isinstance(enc, pikepdf.Name):
        base = _name(enc)
    elif isinstance(enc, pikepdf.Dictionary):
        base = _name(enc.get("/BaseEncoding")) or None
        diffs = enc.get("/Differences")
    if base in ("WinAnsiEncoding", "MacRomanEncoding"):
        codec = "cp1252" if base == "WinAnsiEncoding" else "mac_roman"
        for code in range(32, 256):
            try:
                ch = bytes([code]).decode(codec)
            except UnicodeDecodeError:
                continue
            name = UV2AGL.get(ord(ch))
            if name:
                names[code] = name
    elif base == "StandardEncoding" or (base is None and program is None):
        from fontTools.encodings.StandardEncoding import StandardEncoding

        names.update({i: n for i, n in enumerate(StandardEncoding) if n != ".notdef"})
    elif base is None and program is not None:
        names.update(program.builtin_encoding())
    if isinstance(diffs, pikepdf.Array):
        code = None
        for item in list(diffs)[:4096]:
            if isinstance(item, pikepdf.Name):
                if code is None:
                    continue

                if 0 <= code < 256:
                    names[code] = _name(item)
                code += 1
            elif isinstance(item, (int, float)) or type(item).__name__ in ("Decimal",):
                try:
                    code = int(item)
                except (TypeError, ValueError):
                    code = None
            else:
                code = None
    return names


def describe(font: Any) -> Font:
    """What pdfintake knows about one font dictionary (never raises for what the file holds)."""
    base = _name(font.get("/BaseFont"))
    sub = _name(font.get("/Subtype"))
    f = Font(font.objgen, base, fontprog.family_key(base), fontprog.style_key(base), sub == "Type0", None)
    tu, has_tu = _tounicode(font)
    f.tounicode = tu
    f.existing = _usable(tu)
    try:
        if sub == "Type0":
            _describe_type0(font, f, has_tu)
        elif sub == "TrueType":
            _describe_truetype(font, f)
        elif sub in ("Type1", "MMType1"):
            _describe_type1(font, f)
        elif sub == "Type3":
            f.scope = "type3"
        else:
            f.scope = "font_kind"
    except Exception:  # noqa: BLE001 - a malformed font dictionary is out of scope, never fatal
        f.scope = "font_unreadable"
        f.code_gid = {}
    if f.program is None and f.scope is None:
        f.scope = "not_embedded"
    return f


def _describe_type0(font: Any, f: Font, has_tu: bool) -> None:
    import pikepdf

    desc = font.get("/DescendantFonts")
    cid = desc[0] if isinstance(desc, pikepdf.Array) and len(desc) else None
    if not isinstance(cid, pikepdf.Dictionary):
        f.scope = "font_unreadable"
        return
    if _name(font.get("/Encoding")) not in ("Identity-H", "Identity-V"):
        f.scope = "encoding_not_identity"
        return
    info = cid.get("/CIDSystemInfo")
    ordering = str(info.get("/Ordering", "")) if isinstance(info, pikepdf.Dictionary) else ""
    if ordering in KNOWN_COLLECTIONS and not has_tu:
        f.scope = "known_collection"
        return
    program, why = _program(cid.get("/FontDescriptor"))
    f.program = program
    if program is None:
        f.scope = why
        return
    kind = _name(cid.get("/Subtype"))
    if kind == "CIDFontType2":
        c2g = cid.get("/CIDToGIDMap")
        if isinstance(c2g, pikepdf.Name) and _name(c2g) == "Identity":
            f.code_gid = {c: c for c in range(min(program.num_glyphs, 65536))}
        elif isinstance(c2g, pikepdf.Stream):
            data = c2g.read_bytes()[: 2 * 65536]
            f.code_gid = {i // 2: int.from_bytes(data[i:i + 2], "big") for i in range(0, len(data) - 1, 2)
                          if int.from_bytes(data[i:i + 2], "big")}
        else:
            f.scope = "cidtogid_missing"
            f.code_gid = {c: c for c in range(min(program.num_glyphs, 65536))}
    elif kind == "CIDFontType0":
        if program.cid_keyed:
            f.code_gid = {c: g for c, g in program.cid_gid.items() if c < 65536}
        else:
            f.code_gid = {c: c for c in range(min(program.num_glyphs, 65536))}
    else:
        f.scope = "font_kind"


def _flags(font: Any) -> int:
    import pikepdf

    fd = font.get("/FontDescriptor")
    try:
        return int(fd.get("/Flags", 0)) if isinstance(fd, pikepdf.Dictionary) else 0
    except (TypeError, ValueError):
        return 0


def _describe_truetype(font: Any, f: Font) -> None:
    program, why = _program(font.get("/FontDescriptor"))
    f.program = program
    if program is None:
        f.scope = why
        return
    symbolic = bool(_flags(font) & 4)
    has_enc = font.get("/Encoding") is not None
    if symbolic and not has_enc:
        f.code_gid = program.symbol_cmap()
        f.implicit = True
        return
    # A trusted source only: poppler picks some of this font's glyphs through /ToUnicode (§15.1).
    f.scope = "simple_truetype_encoded"
    names = _encoding_names(font, None)
    cmap = program._ttfont["cmap"] if program._ttfont is not None and "cmap" in program._ttfont else None
    uni = cmap.getcmap(3, 1) if cmap is not None else None
    order = {n: i for i, n in enumerate(program.glyph_order)}
    for code, name in names.items():
        text = fontprog.agl_text(name)
        if text and len(text) == 1 and uni is not None and ord(text) in uni.cmap:
            gid = order.get(uni.cmap[ord(text)])
            if gid is not None:
                f.code_gid[code] = gid
                f.existing.setdefault(code, text)


def _describe_type1(font: Any, f: Font) -> None:
    program, why = _program(font.get("/FontDescriptor"))
    f.program = program
    if program is None:
        f.scope = why
        return
    if program.kind != "cff" or program.cid_keyed:
        f.scope = "font_kind"
        return
    for code, name in _encoding_names(font, program).items():
        gid = program.gid_for_name(name)
        if gid is None:
            continue
        f.code_gid[code] = gid
        if code not in f.existing:
            text = fontprog.agl_text(name)
            if text:
                f.existing[code] = text


# --------------------------------------------------------------------------- the probe

@dataclass
class Glyph:
    """One drawn glyph of a probed font."""

    page: int
    font: int                         # index into Analysis.fonts
    code: int
    box: tuple[float, float, float, float]   # l, b, r, t in page space (pt)
    visible: bool
    baseline: str | None = None       # what pdfium reads for it in the input itself
    map_error: bool = False           # pdfium itself says that reading has no map


@dataclass
class PageItems:
    """A page's text as pdfium lays it out: known characters, and glyph ids where a probed font draws."""

    items: list[str | int] = field(default_factory=list)       # str: a known character; int: Glyph id
    boxes: list[tuple[float, float, float, float] | None] = field(default_factory=list)
    visible: list[bool] = field(default_factory=list)


def _alphas(obj: Any) -> tuple[int, int]:
    import pypdfium2.raw as pdfium_c

    out = []
    for getter in (pdfium_c.FPDFPageObj_GetFillColor, pdfium_c.FPDFPageObj_GetStrokeColor):
        r, g, b, a = (ctypes.c_uint() for _ in range(4))
        ok = getter(obj, r, g, b, a)
        out.append(int(a.value) if ok else 255)
    return out[0], out[1]


def _char_visible(tp: Any, i: int, memo: dict[int, bool]) -> bool:
    import pypdfium2.raw as pdfium_c

    from .classify import _visible

    obj = pdfium_c.FPDFText_GetTextObject(tp, i)
    key = ctypes.cast(obj, ctypes.c_void_p).value if obj else None
    if key is None:
        return True
    if key not in memo:
        mode = int(pdfium_c.FPDFTextObj_GetTextRenderMode(obj))
        fa, sa = _alphas(obj)
        memo[key] = _visible(mode, fa, sa)
    return memo[key]


def _box(tp: Any, i: int) -> tuple[float, float, float, float] | None:
    import pypdfium2.raw as pdfium_c

    left, right, bottom, top = (ctypes.c_double() for _ in range(4))
    if not pdfium_c.FPDFText_GetCharBox(tp, i, left, right, bottom, top):
        return None
    return float(left.value), float(bottom.value), float(right.value), float(top.value)


def _key(page: int, box: tuple[float, float, float, float] | None) -> tuple[Any, ...] | None:
    if box is None:
        return None
    return (page, *(round(v, 2) for v in box))


@dataclass
class _Slot:
    font: int
    code: int


def _batches(fonts: Sequence[Font], probed: Sequence[int]) -> list[dict[int, _Slot]]:
    """Private-use point -> (font, code), one dict per pdfium pass."""
    out: list[dict[int, _Slot]] = []
    cur: dict[int, _Slot] = {}
    for fi in probed:
        for code in fonts[fi].probe_codes():
            if len(cur) >= PUA_SLOTS:
                out.append(cur)
                cur = {}
                if len(out) >= MAX_PASSES:
                    return out
            cur[PUA_FIRST + len(cur)] = _Slot(fi, code)
    if cur:
        out.append(cur)
    return out


def _probe_file(src: Path, dest: Path, fonts: Sequence[Font], probed: Sequence[int], batch: Mapping[int, _Slot]) -> None:
    import pikepdf

    by_font: dict[int, dict[int, str]] = defaultdict(dict)
    for point, slot in batch.items():
        by_font[slot.font][slot.code] = chr(point)
    with pikepdf.open(src) as pdf:
        for fi in probed:
            f = fonts[fi]
            entries = dict(f.existing)
            for code in f.probe_codes():
                entries[code] = by_font[fi].get(code, chr(SENTINEL))
            obj = pdf.get_object(f.ref)
            obj["/ToUnicode"] = pikepdf.Stream(pdf, fontprog.cmap_bytes(entries, two_byte=f.two_byte))
        pdf.save(dest)


def _read_pages(path: Path, visit: Callable[[int, Any], None]) -> None:
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(doc)):
                page = doc[i]
                try:
                    tp = page.get_textpage()
                    try:
                        visit(i, tp)
                    finally:
                        tp.close()
                finally:
                    page.close()
        finally:
            doc.close()


# --------------------------------------------------------------------------- the analysis

@dataclass
class Decision:
    text: str
    evidence: Evidence


@dataclass
class Analysis:
    """Everything known about a document's fonts, glyphs and the letters they draw."""

    fonts: list[Font]
    probed: list[int]                       # fonts whose codes the probe covered
    glyphs: list[Glyph]
    pages: dict[int, PageItems]
    probe: Path | None                      # the probe file of the first pass (for page geometry)
    decided: dict[tuple[int, int], Decision] = field(default_factory=dict)
    pending: dict[tuple[int, int], set[str]] = field(default_factory=dict)   # lookalikes awaiting context
    pending_evidence: dict[tuple[int, int], Evidence] = field(default_factory=dict)
    votes: dict[tuple[int, int], list[tuple[str, int, str]]] = field(default_factory=dict)  # (text, glyph, witness)
    reads: dict[int, tuple[dict[int, tuple[float, float, float, float]], list[Any]]] = field(default_factory=dict)
    tallies: dict[tuple[int, int], dict[str, int]] = field(default_factory=dict)   # an OCR win's counted votes
    ocr_read: list[int] = field(default_factory=list)
    out_of_passes: bool = False

    def occurrences(self) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = defaultdict(list)
        for gi, g in enumerate(self.glyphs):
            out[(g.font, g.code)].append(gi)
        return out

    def needs(self) -> set[tuple[int, int]]:
        """Used codes that need a map (§15.1): no usable mapping, and no presumption a printable
        reading of a symbolic TrueType font is right."""
        out: set[tuple[int, int]] = set()
        for key, gids in self.occurrences().items():
            f = self.fonts[key[0]]
            if key[1] in f.existing:
                continue
            if not any(self.glyphs[g].visible for g in gids):
                continue
            if f.implicit and key not in self.decided and all(_printable(self.glyphs[g].baseline) and
                                                              not self.glyphs[g].map_error for g in gids):
                continue
            out.add(key)
        return out

    def open_codes(self) -> set[tuple[int, int]]:
        return {k for k in self.needs() if k not in self.decided}


def _printable(text: str | None) -> bool:
    return bool(text) and fontprog.usable_text(text or "") and not any(
        unicodedata.category(ch) == "Co" for ch in text or "")


def analyse(src: Path, work: Path) -> Analysis:
    """Describe the fonts of ``src``, probe the glyphs of the ones that need a map, and read what
    pdfium reads for them today. ``work`` is a private folder for the probe files."""
    import pikepdf

    base = _baseline_read(src)
    if not any(err for chars in base.values() for _k, _t, err in chars):
        return Analysis([], [], [], {}, None)          # pdfium maps every glyph: nothing to repair
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with pikepdf.open(src) as pdf:
        fonts = [describe(fd) for fd in _font_dicts(pdf)]
    probed = [i for i, f in enumerate(fonts) if f.scope is None and f.probe_codes()]
    if not probed:
        return Analysis(fonts, [], [], {}, None)
    batches = _batches(fonts, probed)
    glyphs: list[Glyph] = []
    pages: dict[int, PageItems] = {}
    first: Path | None = None
    for n, batch in enumerate(batches):
        probe = work / f"probe-{n}.pdf"
        _probe_file(src, probe, fonts, probed, batch)
        if first is None:
            first = probe

        def visit(i: int, tp: Any, batch: Mapping[int, _Slot] = batch, n: int = n) -> None:
            import pypdfium2.raw as pdfium_c

            memo: dict[int, bool] = {}
            count = pdfium_c.FPDFText_CountChars(tp)
            if n == 0:
                pi = PageItems()
                pages[i] = pi
            else:
                pi = pages.get(i) or PageItems()
                if len(pi.items) != count:              # a pass that lays the page out otherwise adds nothing
                    return
            for k in range(count):
                u = int(pdfium_c.FPDFText_GetUnicode(tp, k))
                slot = batch.get(u)
                if n == 0:
                    box = _box(tp, k)
                    vis = _char_visible(tp, k, memo)
                    pi.boxes.append(box)
                    pi.visible.append(vis)
                    if slot is not None and box is not None:
                        glyphs.append(Glyph(i, slot.font, slot.code, box, vis))
                        pi.items.append(len(glyphs) - 1)
                    elif u == SENTINEL or (slot is not None):
                        pi.items.append(-1)
                    else:
                        pi.items.append(chr(u) if 0 <= u <= 0x10FFFF else "�")
                elif slot is not None and pi.items[k] == -1 and pi.boxes[k] is not None:
                    glyphs.append(Glyph(i, slot.font, slot.code, pi.boxes[k], pi.visible[k]))
                    pi.items[k] = len(glyphs) - 1

        _read_pages(probe, visit)
    analysis = Analysis(fonts, probed, glyphs, pages, first, out_of_passes=len(batches) >= MAX_PASSES)
    _join_baseline(base, analysis)
    return analysis


def _baseline_read(src: Path) -> dict[int, list[tuple[tuple[Any, ...] | None, str | None, bool]]]:
    """What pdfium reads in the input itself: per page, each drawn character's box key, text and
    whether pdfium says it has no map (``FPDFText_HasUnicodeMapError``)."""
    out: dict[int, list[tuple[tuple[Any, ...] | None, str | None, bool]]] = {}

    def visit(i: int, tp: Any) -> None:
        import pypdfium2.raw as pdfium_c

        chars = out.setdefault(i, [])
        for k in range(pdfium_c.FPDFText_CountChars(tp)):
            if pdfium_c.FPDFText_IsGenerated(tp, k) == 1:
                continue
            u = int(pdfium_c.FPDFText_GetUnicode(tp, k))
            chars.append((_key(i, _box(tp, k)), chr(u) if 0 < u <= 0x10FFFF else None,
                          pdfium_c.FPDFText_HasUnicodeMapError(tp, k) == 1))

    _read_pages(src, visit)
    return out


def _join_baseline(base: Mapping[int, Sequence[tuple[tuple[Any, ...] | None, str | None, bool]]], a: Analysis) -> None:
    """Join each probed glyph with what pdfium reads at the same box in the input itself."""
    index: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for gi, g in enumerate(a.glyphs):
        k = _key(g.page, g.box)
        if k is not None:
            index[k].append(gi)
    for chars in base.values():
        for key, text, err in chars:
            if key is None or not index.get(key):
                continue
            gi = index[key].pop(0)
            a.glyphs[gi].baseline = text
            a.glyphs[gi].map_error = err


# --------------------------------------------------------------------------- evidence 1-3

class _References:
    """Installed faces per family, loaded once per run."""

    def __init__(self, index: ReferenceIndex | None) -> None:
        self.index = index
        self._loaded: dict[str, list[Reference]] = {}

    def family(self, family: str) -> list[Reference]:
        if self.index is None or not family:
            return []
        if family not in self._loaded:
            refs = [fontprog.load_reference(face) for face in self.index.faces(family)]
            self._loaded[family] = [r for r in refs if r is not None]
        return self._loaded[family]


def _hash_of(f: Font, code: int) -> tuple[str | None, bool]:
    """(outline hash, drawable): None with drawable True is an empty glyph (a space)."""
    if f.program is None or code not in f.code_gid:
        return None, False
    gid = f.code_gid[code]
    if gid == 0 or not f.program.has_glyph(gid) or not f.program.drawable(gid):
        return None, False
    return f.program.glyph_hash(gid), True


def _document_texts(a: Analysis) -> dict[str, dict[str, set[str]]]:
    """family -> outline hash -> texts, from every font's own usable mappings."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for f in a.fonts:
        if f.program is None or not f.family:
            continue
        for code, text in f.existing.items():
            h, ok = _hash_of(f, code)
            if ok and h is not None:
                out[f.family][h].add(unicodedata.normalize("NFC", text))
    return out


def _program_texts(f: Font, code: int, cmap: dict[int, set[str]]) -> set[str]:
    if f.program is None or code not in f.code_gid:
        return set()
    gid = f.code_gid[code]
    texts = set(cmap.get(gid, ()))
    name = f.program.glyph_name(gid)
    agl = fontprog.agl_text(name)
    if agl:
        texts.add(agl)
    return texts


def _preserved(f: Font, codes: Sequence[int], ref: Reference) -> bool:
    """Whether ``f``'s subset kept ``ref``'s glyph numbers: of its used glyphs whose outline the face
    holds, at least 3, and at least 90%, sit at the same GID there."""
    found = same = 0
    for code in codes:
        h, ok = _hash_of(f, code)
        if not ok or h is None or h not in ref.hash_text:
            continue
        found += 1
        if ref.gid_hash.get(f.code_gid[code]) == h:
            same += 1
    return same >= 3 and same >= 0.9 * found


def gather_evidence(a: Analysis, references: ReferenceIndex | None) -> None:
    """Evidence 1-3 for every used code that needs a map, then the lookalike rule (§15.2, §15.3)."""
    refs = _References(references)
    docs = _document_texts(a)
    used: dict[int, list[int]] = defaultdict(list)
    for (fi, code) in a.occurrences():
        used[fi].append(code)
    for fi, codes in used.items():
        f = a.fonts[fi]
        cmap = f.program.cmap_text() if f.program is not None else {}
        faces = refs.family(f.family)
        preserved = [r for r in faces if _preserved(f, codes, r)]
        for code in sorted(codes):
            key = (fi, code)
            if code in f.existing or key in a.decided:
                continue
            h, ok = _hash_of(f, code)
            if not ok:
                continue                              # .notdef, a missing glyph: unprovable
            if h is None:
                a.decided[key] = Decision(" ", "space")
                continue
            evidence: Evidence | None = None
            by_gid = False
            cands = _program_texts(f, code, cmap)
            if cands:
                evidence = "program"
            if not cands:
                for r in preserved:
                    at = r.text_at(f.code_gid[code], h)
                    if at:
                        cands, evidence, by_gid = at, "installed", True
                        break
            if not cands:
                for r in faces:
                    cands |= r.hash_text.get(h, set())
                if cands:
                    evidence = "installed"
            if not cands and h in docs.get(f.family, {}):
                cands, evidence = set(docs[f.family][h]), "document"
            if not cands or evidence is None:
                continue
            _offer(a, key, cands, evidence, widen=not by_gid)
    settle_context(a)


def _offer(a: Analysis, key: tuple[int, int], cands: set[str], evidence: Evidence, *, widen: bool = True) -> None:
    """Record a code's candidate texts: decided when they settle, pending on context otherwise.
    ``widen=False`` for a decision by glyph id (step 2), which twins can't overturn."""
    settled, pool = settle(cands, widen=widen)
    if settled is not None:
        a.decided[key] = Decision(settled, evidence)
        a.pending.pop(key, None)
    elif pool:
        a.pending[key] = pool
        a.pending_evidence[key] = evidence


def settle(cands: set[str], *, widen: bool = True) -> tuple[str | None, set[str]]:
    """Steps 1, 3 and 6 of the lookalike rule: (the text, when the candidates settle without context;
    else the widened pool to decide by context)."""
    pool = set()
    for t in cands:
        t = unicodedata.normalize("NFC", t)
        if all(unicodedata.category(ch) == "Cf" for ch in t):
            continue
        pool.add(t)
    pool = {t for t in pool if not (unicodedata.normalize("NFKC", t) != t and unicodedata.normalize("NFKC", t) in pool)}
    for t in list(pool) if widen else ():
        if len(t) == 1:
            pool |= fontprog.TWINS.get(t, frozenset())
    if not pool:
        return None, set()
    if len(pool) == 1:
        return next(iter(pool)), pool
    scripts = {_script(t) for t in pool}
    if len(scripts) == 1:
        return _lowest(pool), pool
    return None, pool


def _script(text: str) -> str | None:
    for ch in text:
        s = fontprog.script_of(ch)
        if s is not None:
            return s
    return None


def _lowest(pool: set[str]) -> str:
    return min(pool, key=lambda t: (len(t), [ord(c) for c in t]))


def _blank(a: Analysis, item: str | int) -> bool:
    """Whitespace: a known space, or a glyph decided as one (an empty glyph)."""
    text = _item_text(a, item) if isinstance(item, int) else item
    return text is not None and (not text or text.isspace())


def _words(a: Analysis, pi: PageItems) -> Iterator[list[int]]:
    """Positions of each visible word on a page (split at whitespace and at hidden text)."""
    cur: list[int] = []
    for k, item in enumerate(pi.items):
        if not pi.visible[k] or _blank(a, item):
            if cur:
                yield cur
            cur = []
            continue
        cur.append(k)
    if cur:
        yield cur


def _item_text(a: Analysis, item: str | int) -> str | None:
    if isinstance(item, str):
        return item
    if item < 0:
        return None
    g = a.glyphs[item]
    f = a.fonts[g.font]
    if g.code in f.existing:
        return f.existing[g.code]
    d = a.decided.get((g.font, g.code))
    return d.text if d is not None else None


def settle_context(a: Analysis) -> None:
    """Steps 4-6 of the lookalike rule: decide pending codes by the script of their words, until
    nothing changes. A code no word gives context for stays pending."""
    for _round in range(6):
        if not a.pending:
            return
        votes: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
        for pi in a.pages.values():
            for word in _words(a, pi):
                waiting = [pi.items[k] for k in word
                           if isinstance(pi.items[k], int) and pi.items[k] >= 0
                           and (a.glyphs[pi.items[k]].font, a.glyphs[pi.items[k]].code) in a.pending]
                if not waiting:
                    continue
                scripts: Counter[str] = Counter()
                for k in word:
                    item = pi.items[k]
                    if isinstance(item, int) and item >= 0 and \
                            (a.glyphs[item].font, a.glyphs[item].code) in a.pending:
                        continue
                    text = _item_text(a, item)
                    if text and len(text) == 1 and fontprog.TWINS.get(text):
                        continue                      # a lookalike itself says nothing about its word
                    s = _script(text or "")
                    if s is not None:
                        scripts[s] += 1
                if not scripts:
                    continue
                top = scripts.most_common(1)[0][0]
                for gi in waiting:
                    votes[(a.glyphs[gi].font, a.glyphs[gi].code)][top] += 1
        changed = False
        for key, pool in list(a.pending.items()):
            for script, _n in votes.get(key, Counter()).most_common():
                inside = {t for t in pool if _script(t) == script}
                if inside:
                    a.decided[key] = Decision(_lowest(inside), a.pending_evidence.pop(key, "document"))
                    del a.pending[key]
                    changed = True
                    break
        if not changed:
            return


# --------------------------------------------------------------------------- OCR evidence

def ocr_pages(a: Analysis, cap: int = OCR_PAGES_CAP) -> list[int]:
    """The pages worth reading for the codes still open: greedily, the page that adds the most
    sightings of open codes (each counted up to OCR_TARGET in all) first."""
    open_ = a.open_codes() | set(a.pending)
    if not open_:
        return []
    per_page: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    for g in a.glyphs:
        key = (g.font, g.code)
        if key in open_ and g.visible:
            per_page[g.page][key] += 1
    seen: Counter[tuple[int, int]] = Counter()
    chosen: list[int] = []
    while len(chosen) < cap:
        best, gain = None, 0
        for page, counts in per_page.items():
            if page in chosen:
                continue
            g = sum(min(n, max(OCR_TARGET - seen[k], 0)) for k, n in counts.items())
            if g > gain or (g == gain and best is not None and page < best):
                best, gain = page, g
        if best is None or gain == 0:
            break
        chosen.append(best)
        seen.update(per_page[best])
    return sorted(chosen)


def _rect(box: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over the smaller rectangle's area."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / small if small > 0 else 0.0


def _device_boxes(probe: Path, page: int, image: "PageImage",
                  boxes: Mapping[int, tuple[float, float, float, float]]) -> dict[int, tuple[float, float, float, float]]:
    """Page-space boxes as pixel rectangles of ``image`` (pdfium's own page-to-device mapping)."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    out: dict[int, tuple[float, float, float, float]] = {}
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(probe))
        try:
            pg = doc[page]
            try:
                dx, dy = ctypes.c_int(), ctypes.c_int()
                for k, (left, bottom, right, top) in boxes.items():
                    xs, ys = [], []
                    for px, py in ((left, bottom), (right, top)):
                        pdfium_c.FPDF_PageToDevice(pg.raw, 0, 0, image.width, image.height, 0, px, py, dx, dy)
                        xs.append(dx.value)
                        ys.append(dy.value)
                    out[k] = (min(xs), min(ys), max(xs), max(ys))
            finally:
                pg.close()
        finally:
            doc.close()
    return out


def _lines(a: Analysis, page: int) -> list[list[int]]:
    """A page's visible glyph lines: item positions between pdfium's line breaks, whitespace left out."""
    pi = a.pages.get(page)
    if pi is None:
        return []
    out: list[list[int]] = []
    cur: list[int] = []
    for k, item in enumerate(pi.items):
        if isinstance(item, str) and item in ("\r", "\n"):
            if cur:
                out.append(cur)
            cur = []
            continue
        if not pi.visible[k] or pi.boxes[k] is None or _blank(a, item):
            continue
        cur.append(k)
    if cur:
        out.append(cur)
    return out


def _align(tokens: Sequence[str | None], text: str) -> list[tuple[int, int]]:
    """A banded global alignment of glyph tokens (a known character, or None for an open code) with
    ``text``; free gaps at both ends of ``text``. Returns the (token, character) pairs on the path."""
    n, m = len(tokens), len(text)
    if n == 0 or m == 0:
        return []
    neg = -10**9
    ratio = m / n
    band = BAND + abs(m - n)
    score = [[neg] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        score[0][j] = 0                         # free leading gap in text
    for i in range(1, n + 1):
        centre = i * ratio
        lo = max(0, int(centre) - band)
        hi = min(m, int(centre) + band)
        score[i][lo] = score[i - 1][lo] - 1 if score[i - 1][lo] > neg else neg
        move[i][lo] = 2
        tok = tokens[i - 1]
        for j in range(max(lo, 1), hi + 1):
            ch = text[j - 1]
            sub = 1 if tok is None else (2 if tok == ch else -1)
            best, mv = score[i - 1][j - 1] + sub, 1
            up = score[i - 1][j] - 1
            if up > best:
                best, mv = up, 2
            left = score[i][j - 1] - 1
            if left > best:
                best, mv = left, 3
            score[i][j] = best
            move[i][j] = mv
    j = max(range(m + 1), key=lambda jj: score[n][jj])   # free trailing gap in text
    i = n
    path: list[tuple[int, int]] = []
    while i > 0 and j >= 0:
        mv = move[i][j]
        if mv == 1:
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif mv == 2:
            path.append((i - 1, -1))
            i -= 1
        elif mv == 3:
            path.append((-1, j - 1))
            j -= 1
        else:
            break
    path.reverse()
    return path


def add_reads(a: Analysis, page: int, image: "PageImage", reads: Sequence["EngineRead"]) -> None:
    """Keep one page's OCR reads, with the page's glyph boxes in the render's pixels, for voting."""
    if a.probe is None:
        return
    pi = a.pages.get(page)
    if pi is None:
        return
    boxes = {k: box for k, box in enumerate(pi.boxes) if box is not None and pi.visible[k]}
    a.reads[page] = (_device_boxes(a.probe, page, image, boxes), [r for r in reads if r.status == "ok"])
    if page not in a.ocr_read:
        a.ocr_read.append(page)


def cast_votes(a: Analysis) -> int:
    """Every kept read's votes (§15.4), from scratch, aligned against what is decided now; returns
    how many were cast."""
    a.votes = {}
    cast = 0
    for page, (dev, reads) in sorted(a.reads.items()):
        pi = a.pages[page]
        lines = _lines(a, page)
        for read in reads:
            witness = f"{read.use_id}@{page}"
            ocr = [(ln, _rect(ln.box)) for ln in read.lines if ln.text and ln.text.strip()]
            for line in lines:
                rects = [dev[k] for k in line if k in dev]
                if not rects:
                    continue
                lr = (min(r[0] for r in rects), min(r[1] for r in rects),
                      max(r[2] for r in rects), max(r[3] for r in rects))
                best = max(ocr, key=lambda o: _overlap(lr, o[1]), default=None)
                if best is None or _overlap(lr, best[1]) < 0.5:
                    continue
                cast += _vote_line(a, pi, line, best[0].text, best[1], lr, dev, witness)
    return cast


def _vote_line(a: Analysis, pi: PageItems, line: list[int], text: str, ocr_rect: tuple[float, float, float, float],
               line_rect: tuple[float, float, float, float], dev: Mapping[int, Any], witness: str) -> int:
    texts = [unicodedata.normalize("NFC", text)]
    nfd = unicodedata.normalize("NFD", text)
    if nfd != texts[0]:
        texts.append(nfd)
    tokens: list[str | None] = []
    owners: list[int | None] = []                  # glyph id behind each token (None: known text)
    for k in line:
        item = pi.items[k]
        if isinstance(item, int) and item >= 0:
            g = a.glyphs[item]
            key = (g.font, g.code)
            known = _item_text(a, item) if key not in a.pending else None
            if known is not None and key in a.decided and a.decided[key].text == UNPROVEN:
                known = None
            if known is None:
                tokens.append(None)
                owners.append(item)
            else:
                for ch in unicodedata.normalize("NFC", known):
                    tokens.append(ch)
                    owners.append(None)
        elif isinstance(item, str):
            for ch in unicodedata.normalize("NFC", item):
                tokens.append(ch)
                owners.append(None)
        else:
            tokens.append(None)
            owners.append(-1)                      # a glyph of another pass: open, but can't vote
    if not any(o is not None and o >= 0 for o in owners):
        return 0
    # The part of the read across this line's width (reads may run past it).
    ow = max(ocr_rect[2] - ocr_rect[0], 1e-6)
    best_cast = 0
    best_votes: list[tuple[tuple[int, int], str, int]] = []
    for t in texts:
        t = "".join(ch for ch in t if not ch.isspace())
        if not t:
            continue
        lo = int(max(0.0, (line_rect[0] - ocr_rect[0]) / ow) * len(t)) - 3
        hi = int(min(1.0, (line_rect[2] - ocr_rect[0]) / ow) * len(t)) + 3
        cropped = t[max(lo, 0):max(hi, 0)] or t
        best_cast, best_votes = _best_part(a, tokens, owners, [cropped] if cropped == t else [cropped, t],
                                           best_cast, best_votes)
    for key, ch, gi in best_votes:
        a.votes.setdefault(key, []).append((ch, gi, witness))
    return best_cast


def _best_part(a: Analysis, tokens: Sequence[str | None], owners: Sequence[int | None], parts: Sequence[str],
               best_cast: int, best_votes: list[tuple[tuple[int, int], str, int]]
               ) -> tuple[int, list[tuple[tuple[int, int], str, int]]]:
    """The votes of the read text (cropped to the line's width, or whole) that aligns best."""
    for part in parts:
        known = [i for i, tok in enumerate(tokens) if tok is not None]
        path = _align(tokens, part)
        pairs = {ti: cj for ti, cj in path if ti >= 0}
        if known:
            hits = sum(1 for i in known if pairs.get(i, -1) >= 0 and tokens[i] == part[pairs[i]])
            if hits < LINE_MATCH_MIN * len(known):
                continue
        elif len(tokens) != len(part):
            continue                               # nothing known: only a line whose counts agree votes
        votes: list[tuple[tuple[int, int], str, int]] = []
        steps = list(path)
        for s, (ti, cj) in enumerate(steps):
            if ti < 0 or cj < 0 or tokens[ti] is not None or owners[ti] is None or owners[ti] < 0:
                continue
            ok = True
            for ns in (s - 1, s + 1):
                if 0 <= ns < len(steps):
                    nti, ncj = steps[ns]
                    if nti < 0 or ncj < 0 or (tokens[nti] is not None and tokens[nti] != part[ncj]):
                        ok = False
            if not ok:
                continue
            gi = owners[ti]
            g = a.glyphs[gi]                       # type: ignore[index]
            votes.append(((g.font, g.code), part[cj], gi))  # type: ignore[arg-type]
        if len(votes) > best_cast:
            best_cast, best_votes = len(votes), votes
    return best_cast, best_votes


def decide_ocr(a: Analysis) -> None:
    """Decide open codes from the kept reads (§15.4): vote, decide what clears the bar, then align
    again with those letters as anchors, until a round decides nothing new."""
    for _round in range(OCR_ROUNDS):
        cast_votes(a)
        if not _decide_votes(a):
            break


def _decide_votes(a: Analysis) -> int:
    """One round: the codes the votes prove, with the two guards of §15.4, then context; returns how
    many were decided."""
    before = len(a.decided)
    owned: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))   # style -> letter -> hashes
    for fi, f in enumerate(a.fonts):
        if f.program is None:
            continue
        for code, text in f.existing.items():
            h, ok = _hash_of(f, code)
            if ok and h is not None:
                owned[f.style][text].add(h)
    for (fi, code), d in a.decided.items():
        f = a.fonts[fi]
        h, ok = _hash_of(f, code)
        if ok and h is not None and d.text != UNPROVEN:
            owned[f.style][d.text].add(h)
    wins: dict[tuple[int, int], str] = {}
    for key, votes in a.votes.items():
        if key in a.decided:
            continue
        f = a.fonts[key[0]]
        h, ok = _hash_of(f, key[1])
        if not ok or h is None:
            continue
        counted: list[tuple[str, int, str]] = []
        aside: list[str] = []
        for ch, gi, witness in votes:
            others = owned[f.style].get(ch, set()) - {h}
            if others and not unicodedata.category(ch).startswith("M"):
                aside.append(ch)
            else:
                counted.append((ch, gi, witness))
        if len(counted) < VOTE_MIN:
            continue
        tally = Counter(ch for ch, _gi, _w in counted)
        win, n = tally.most_common(1)[0]
        if n < (VOTE_SHARE * len(counted) if len(counted) > 2 else len(counted)):
            continue
        backing = [(gi, w) for ch, gi, w in counted if ch == win]
        engines = {w.split("@")[0] for _gi, w in backing}
        seen = len({gi for gi, _w in backing})
        if len(engines) < 2 and seen < ONE_ENGINE_OCCURRENCES:
            continue                               # one engine's habit isn't proof (§15.4)
        if all(unicodedata.category(ch).startswith("M") for ch in win) and (len(counted) < 3 or seen < 2):
            continue                               # a mark is OCR's least sure read: 3 votes, 2 sightings
        shape = fontprog.skeleton(win)
        alike = sum(1 for ch, _gi, _w in votes if fontprog.skeleton(ch) == shape)
        if alike < VOTE_SHARE * len(votes):
            continue                               # the set-aside votes saw another shape
        wins[key] = win
        a.tallies[key] = dict(tally)
    by_letter: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for key, win in wins.items():
        if not unicodedata.category(win[0]).startswith("M"):
            by_letter[(key[0], win)].append(key)   # one program, one letter, one shape
    for (_font, _win), keys in by_letter.items():
        hashes = {_hash_of(a.fonts[k[0]], k[1])[0] for k in keys}
        if len(hashes) > 1:
            for k in keys:
                wins.pop(k, None)
    for key, win in wins.items():
        _offer(a, key, {win}, "ocr")
    settle_context(a)
    return len(a.decided) - before


# --------------------------------------------------------------------------- decide and write

@dataclass
class FontMapResult:
    """What the repair did; counts and page numbers only (record rule 1)."""

    fonts_repaired: int = 0
    fonts_left: int = 0
    codes: dict[str, int] = field(default_factory=dict)        # evidence -> codes written
    pages_repaired: list[int] = field(default_factory=list)    # every needed code on the page proven
    pages_partial: list[int] = field(default_factory=list)     # some proven, some not
    pages_unrepaired: list[int] = field(default_factory=list)  # codes needed, none proven
    ocr_pages: list[int] = field(default_factory=list)
    maps: dict[tuple[int, int], dict[int, str]] = field(default_factory=dict)   # font ref -> new /ToUnicode
    expected: dict[int, list[tuple[tuple[float, float, float, float], str]]] = field(default_factory=dict)
    # page -> (box, text) of every repaired glyph: what verify's fonts_mapped reads back
    scopes: dict[str, int] = field(default_factory=dict)       # out-of-scope fonts with needed codes, by reason

    @property
    def changed(self) -> bool:
        return bool(self.maps)


def decide(a: Analysis) -> FontMapResult:
    """The maps to write and what each page will read (§15.5)."""
    res = FontMapResult()
    needs = a.needs()
    occ = a.occurrences()
    per_font: dict[int, list[int]] = defaultdict(list)
    for fi, code in needs:
        per_font[fi].append(code)
    written: dict[tuple[int, int], str] = {}
    for fi, codes in per_font.items():
        f = a.fonts[fi]
        proven = {c: a.decided[(fi, c)] for c in codes if (fi, c) in a.decided}
        if not any(d.text.strip() for d in proven.values()):
            proven = {}                            # a space alone proves nothing worth a map
        differs = any(any(a.glyphs[g].baseline != d.text or a.glyphs[g].map_error for g in occ[(fi, c)])
                      for c, d in proven.items())
        if not proven or not differs:
            if not proven:
                res.fonts_left += 1
            continue
        entries: dict[int, str] = {c: t for c, t in f.tounicode.items() if t is not None}
        for code in f.existing:
            entries[code] = f.existing[code]
        for c in codes:
            d = proven.get(c)
            entries[c] = d.text if d is not None else UNPROVEN
            written[(fi, c)] = entries[c]
            if d is not None:
                bucket = {"space": "program", "context": "document", "existing": "program"}.get(d.evidence, d.evidence)
                res.codes[bucket] = res.codes.get(bucket, 0) + 1
            else:
                res.codes["unproven"] = res.codes.get("unproven", 0) + 1
        res.maps[f.ref] = entries
        res.fonts_repaired += 1
    res.scopes = out_of_scope(a)
    # pages
    page_need: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for gi, g in enumerate(a.glyphs):
        key = (g.font, g.code)
        if key in needs and g.visible:
            page_need[g.page].add(key)
    for page, keys in sorted(page_need.items()):
        fixed = [k for k in keys if k in written and written[k] != UNPROVEN]
        if len(fixed) == len(keys):
            res.pages_repaired.append(page)
        elif fixed:
            res.pages_partial.append(page)
        else:
            res.pages_unrepaired.append(page)
    for g in a.glyphs:
        key = (g.font, g.code)
        if key in written:
            res.expected.setdefault(g.page, []).append((g.box, written[key]))
    res.ocr_pages = sorted(a.ocr_read)
    return res


def out_of_scope(a: Analysis) -> dict[str, int]:
    """Fonts pdfintake may not write a map for, by reason (counts only)."""
    return dict(Counter(f.scope for f in a.fonts if f.scope is not None))


def write(pdf: "pikepdf.Pdf", result: FontMapResult) -> int:
    """Set each repaired font's ``/ToUnicode`` to its new map; returns how many fonts changed.

    Only the ``/ToUnicode`` key of those font dictionaries changes.
    """
    import pikepdf

    n = 0
    for ref, entries in sorted(result.maps.items()):
        font = pdf.get_object(ref)
        two_byte = _name(font.get("/Subtype")) == "Type0"
        font["/ToUnicode"] = pikepdf.Stream(pdf, fontprog.cmap_bytes(entries, two_byte=two_byte))
        n += 1
    return n


# --------------------------------------------------------------------------- one call

Reader = Callable[[Sequence[int]], Mapping[int, tuple["PageImage", Sequence["EngineRead"]]]]


def repair(src: Path, work: Path, *, references: ReferenceIndex | None = None, read: Reader | None = None,
           plan_only: bool = False) -> tuple[FontMapResult, Analysis]:
    """Analyse ``src``, gather evidence (OCR through ``read`` for the pages ``ocr_pages`` picks), and
    decide. Writes nothing; ``write`` applies the result to an open copy."""
    a = analyse(src, work)
    if not a.glyphs:
        return FontMapResult(), a
    gather_evidence(a, references)
    if not plan_only and read is not None:
        pages = ocr_pages(a)
        if pages:
            got = read(pages)
            for page in pages:
                if page in got:
                    image, reads = got[page]
                    add_reads(a, page, image, reads)
            decide_ocr(a)
    elif plan_only:
        a.ocr_read = ocr_pages(a)
    return decide(a), a


def cleanup(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)
