"""Born-digital PDFs whose fonts have no working map to letters, built in code (DESIGN §15).

Everything is made up: fonts come from fontTools ``FontBuilder`` with a distinct polygon per letter
(``shape_font``), or are cut from the Noto Sans the English data wheel pins (``noto_subset``); text is
example text. ``garbled_pdf`` draws lines with such a font as a Type0 font (Identity-H, no
``/ToUnicode`` unless given), so pdfium can't tell what the glyphs say.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UPM = 1000
ADVANCE = 600
SPACE_ADVANCE = 300


def _shape(cp: int) -> list[tuple[int, int]]:
    """A closed polygon that only this code point draws (two notch parameters, unique below 9797)."""
    a, b = cp % 101, cp % 97
    return [(50, 0), (50, 600), (150 + a, 600), (250, 280 + b), (330, 600), (500, 600), (500, 0)]


def _glyph_name(cp: int) -> str:
    return f"uni{cp:04X}"


@dataclass
class ShapeFont:
    data: bytes
    gid: dict[str, int]                 # character -> glyph id in this program


def shape_font(chars: str, *, family: str = "Example Serif", style: str = "Regular", order: str | None = None,
               cmap: bool = True, names: bool = True, same_shape: Mapping[str, str] | None = None,
               ligatures: Mapping[str, str] | None = None, encoded_variants: Mapping[str, str] | None = None) -> ShapeFont:
    """A TrueType font whose every letter has its own outline.

    ``order`` fixes the glyph order (default: ``chars``), so a test can renumber a "subset".
    ``cmap=False`` and ``names=False`` leave out the Unicode cmap and the glyph names (post format 3),
    as an embedded subset often does. ``same_shape`` maps a character to another whose outline it
    copies (a Cyrillic а drawn exactly like Latin a). ``ligatures`` adds a glyph per ligature text
    (``{"fi": "f_i"}``) with a GSUB ligature lookup; ``encoded_variants`` adds a single substitution
    from one encoded letter to another (``{"i": "ı"}``, as Times New Roman's ``dtls`` does).
    A space draws nothing.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    same_shape = dict(same_shape or {})
    ligatures = dict(ligatures or {})
    seq = list(dict.fromkeys(order if order is not None else chars))
    for ch in chars:
        if ch not in seq:
            seq.append(ch)
    glyph_order = [".notdef"] + [_glyph_name(ord(c)) for c in seq] + list(ligatures.values())
    glyphs: dict[str, Any] = {}
    widths: dict[str, int] = {}

    def poly(points: Sequence[tuple[int, int]]) -> Any:
        pen = TTGlyphPen(None)
        pen.moveTo(points[0])
        for p in points[1:]:
            pen.lineTo(p)
        pen.closePath()
        return pen.glyph()

    glyphs[".notdef"] = poly([(50, 0), (50, 700), (550, 700), (550, 0)])
    widths[".notdef"] = ADVANCE
    for c in seq:
        name = _glyph_name(ord(c))
        if c.isspace():
            glyphs[name] = TTGlyphPen(None).glyph()
            widths[name] = SPACE_ADVANCE
        else:
            glyphs[name] = poly(_shape(ord(same_shape.get(c, c))))
            widths[name] = ADVANCE
    for k, (text, name) in enumerate(ligatures.items()):
        glyphs[name] = poly(_shape(0x2000 + 7 * k + len(text)))
        widths[name] = ADVANCE
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({ord(c): _glyph_name(ord(c)) for c in seq})   # OS/2 needs it; dropped below
    fb.setupGlyf(glyphs)
    glyf = fb.font["glyf"]
    fb.setupHorizontalMetrics({g: (widths[g], getattr(glyf[g], "xMin", 0)) for g in glyph_order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    ps = "".join(ch for ch in family if ch.isalnum()) + "-" + "".join(ch for ch in style if ch.isalnum())
    fb.setupNameTable({"familyName": family, "styleName": style, "uniqueFontIdentifier": ps,
                       "fullName": f"{family} {style}", "psName": ps, "version": "Version 1.000"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, sTypoLineGap=0, usWinAscent=800, usWinDescent=200,
                fsType=0, achVendID="NONE")
    fb.setupPost(keepGlyphNames=names)
    fb.setupHead(unitsPerEm=UPM, created=0, modified=0)
    if ligatures or encoded_variants:
        lines = ["languagesystem DFLT dflt;", "feature liga {"]
        for text, name in ligatures.items():
            lines.append(f"  sub {' '.join(_glyph_name(ord(c)) for c in text)} by {name};")
        lines.append("} liga;")
        if encoded_variants:
            lines.append("feature dtls {")
            for src, dst in encoded_variants.items():
                lines.append(f"  sub {_glyph_name(ord(src))} by {_glyph_name(ord(dst))};")
            lines.append("} dtls;")
        fb.addOpenTypeFeatures("\n".join(lines))
    if not cmap and "cmap" in fb.font:
        del fb.font["cmap"]
    fb.font.recalcTimestamp = False
    buf = io.BytesIO()
    fb.save(buf)
    gid = {c: glyph_order.index(_glyph_name(ord(c))) for c in seq}
    gid.update({text: glyph_order.index(name) for text, name in ligatures.items()})
    return ShapeFont(buf.getvalue(), gid)


def noto_subset(noto: Path, chars: str) -> ShapeFont:
    """Noto Sans cut down to ``chars`` (the subsetter renumbers it), with no cmap, glyph names or GSUB:
    real letter shapes that OCR reads, and nothing in the program that says which letter is which."""
    from fontTools import subset
    from fontTools.ttLib import TTFont

    font = TTFont(str(noto))
    opts = subset.Options()
    opts.glyph_names = False
    opts.notdef_outline = True
    opts.layout_features = []
    opts.drop_tables += ["GSUB", "GPOS", "GDEF"]
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=[ord(c) for c in set(chars)])
    sub.subset(font)
    best = font.getBestCmap()
    order_now = font.getGlyphOrder()
    gid = {c: order_now.index(best[ord(c)]) for c in set(chars) if ord(c) in best}
    if "cmap" in font:
        del font["cmap"]
    font["post"].formatType = 3.0
    buf = io.BytesIO()
    font.save(buf)
    return ShapeFont(buf.getvalue(), gid)


@dataclass
class FontSpec:
    """One font on the page: its program, how text maps to glyphs, and an optional /ToUnicode."""

    font: ShapeFont
    base: str = "ABCDEF+ExampleSerif-Regular"
    tounicode: Mapping[str, str] | None = None     # character -> text written into /ToUnicode (None: none)
    cidtogid: str | None = "Identity"              # None leaves /CIDToGIDMap out


@dataclass
class Run:
    font: int                                      # index into the fonts list
    text: str


@dataclass
class Page:
    lines: list[list[Run]] = field(default_factory=list)
    size: float = 24.0


def garbled_pdf(path: Path, pages: Sequence[Page], fonts: Sequence[FontSpec]) -> Path:
    """A PDF whose text runs are drawn with Type0 Identity-H fonts (2-byte codes = glyph ids)."""
    import pikepdf

    pdf = pikepdf.new()
    font_objs = []
    for k, spec in enumerate(fonts):
        prog = pikepdf.Stream(pdf, spec.font.data)
        prog["/Length1"] = len(spec.font.data)
        desc = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.FontDescriptor, FontName=pikepdf.Name("/" + spec.base), Flags=4,
            FontBBox=[0, -200, 1000, 800], ItalicAngle=0, Ascent=800, Descent=-200, CapHeight=600, StemV=80,
            FontFile2=prog))
        cid = pikepdf.Dictionary(
            Type=pikepdf.Name.Font, Subtype=pikepdf.Name.CIDFontType2, BaseFont=pikepdf.Name("/" + spec.base),
            CIDSystemInfo=pikepdf.Dictionary(Registry=pikepdf.String("Adobe"), Ordering=pikepdf.String("Identity"),
                                             Supplement=0),
            FontDescriptor=desc, DW=ADVANCE)
        if spec.cidtogid is not None:
            cid["/CIDToGIDMap"] = pikepdf.Name("/" + spec.cidtogid)
        font = pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type0,
                                  BaseFont=pikepdf.Name("/" + spec.base), Encoding=pikepdf.Name("/Identity-H"),
                                  DescendantFonts=[pdf.make_indirect(cid)])
        if spec.tounicode is not None:
            from pdfintake.pdf.fontprog import cmap_bytes

            entries = {spec.font.gid[c]: t for c, t in spec.tounicode.items()}
            font["/ToUnicode"] = pikepdf.Stream(pdf, cmap_bytes(entries, two_byte=True))
        font_objs.append(pdf.make_indirect(font))
    for page in pages:
        ops = []
        y = 760.0
        for line in page.lines:
            ops.append(f"BT 1 0 0 1 60 {y:.2f} Tm")
            for run in line:
                spec = fonts[run.font]
                hexes = "".join(f"{spec.font.gid[c]:04X}" for c in run.text)
                ops.append(f"/F{run.font} {page.size:g} Tf <{hexes}> Tj")
            ops.append("ET")
            y -= page.size * 1.6
        content = pikepdf.Stream(pdf, "\n".join(ops).encode("ascii"))
        resources = pikepdf.Dictionary(Font=pikepdf.Dictionary({f"/F{k}": o for k, o in enumerate(font_objs)}))
        pdf.pages.append(pikepdf.Page(pikepdf.Dictionary(
            Type=pikepdf.Name.Page, MediaBox=[0, 0, 612, 792], Resources=resources, Contents=content)))
    path = Path(path)
    pdf.save(path, deterministic_id=True)
    return path


def lines_of(texts: Sequence[str], font: int = 0) -> list[list[Run]]:
    return [[Run(font, t)] for t in texts]
