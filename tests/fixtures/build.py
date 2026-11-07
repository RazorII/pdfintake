"""Deterministic fixture builders: images, PDFs, box-glyph fonts, synthetic reads and a fake engine.

Every builder gives the same bytes for the same arguments on the same machine: random content
comes from ``random.Random(seed)``, PDFs are saved with fixed dates and deterministic ids, and
fonts carry fixed timestamps. Nothing here reads a real document, and every text is made up.

Import it as ``from fixtures import build`` (pyproject puts ``tests/`` on the path).
"""

from __future__ import annotations

import io
import random
import struct
import time
import unicodedata
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from pdfintake.ocr.engines import Availability, ContractCase
from pdfintake.params import Param, ParamTable
from pdfintake.types import EngineRead, Line, PageImage, Quad, Word

FIXED_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_HEAD_TIMESTAMP = int(FIXED_DATE.timestamp()) + 2082844800   # OpenType dates count seconds from 1904-01-01

# Made-up sample text, safe for any fixture.
LATIN_SAMPLE = "The example invoice lists seven sample items and a total of 4521 units."
THAI_SAMPLE = "ข้อความตัวอย่างสำหรับการทดสอบ"


# --------------------------------------------------------------------------- images

def blank(width: int = 400, height: int = 300, *, mode: str = "RGB", color: Any = None) -> Image.Image:
    """A flat image; white by default."""
    if color is None:
        color = {"1": 1, "L": 255, "RGB": (255, 255, 255), "RGBA": (255, 255, 255, 255),
                 "CMYK": (0, 0, 0, 0), "I;16": 65535, "P": 0, "LA": (255, 255)}.get(mode, 255)
    return Image.new(mode, (width, height), color)


def noise(width: int, height: int, *, mode: str = "RGB", seed: int = 0) -> Image.Image:
    """Uniform random samples: incompressible, for size tests."""
    rng = random.Random(seed)
    channels = {"L": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "LA": 2}[mode]
    return Image.frombytes(mode, (width, height), rng.randbytes(width * height * channels))


def photo_like(width: int, height: int, *, mode: str = "RGB", seed: int = 0) -> Image.Image:
    """Smooth gradients plus a little noise: continuous tone that compresses poorly when lossless."""
    rng = random.Random(seed)
    base = Image.linear_gradient("L").resize((width, height))
    other = base.transpose(Image.Transpose.ROTATE_90).resize((width, height))
    grain = Image.frombytes("L", (width, height), rng.randbytes(width * height)).point(lambda v: v // 8)
    if mode == "L":
        return Image.blend(base, grain, 0.15)
    r = Image.blend(base, grain, 0.15)
    g = Image.blend(other, grain, 0.2)
    b = Image.blend(base.transpose(Image.Transpose.FLIP_LEFT_RIGHT), grain, 0.1)
    img = Image.merge("RGB", (r, g, b))
    return img.convert(mode) if mode != "RGB" else img


def text_like(width: int = 1200, height: int = 800, *, lines: int = 5, seed: int = 0,
              mode: str = "L", margin: int | None = None) -> Image.Image:
    """Dark word-like bars in lines on white, like a page of text seen from afar."""
    rng = random.Random(seed)
    img = blank(width, height, mode="L")
    draw = ImageDraw.Draw(img)
    m = margin if margin is not None else max(4, width // 12)
    pitch = (height - 2 * m) / max(lines, 1)
    bar_h = max(2, int(pitch * 0.45))
    for k in range(lines):
        y0 = int(m + k * pitch)
        x = m
        while x < width - m - 10:
            w = rng.randint(max(3, width // 40), max(4, width // 10))
            x1 = min(x + w, width - m)
            draw.rectangle((x, y0, x1, y0 + bar_h), fill=rng.randint(0, 60))
            x = x1 + rng.randint(max(2, width // 120), max(3, width // 50))
    return img.convert(mode) if mode != "L" else img


def png_bytes(img: Image.Image, *, dpi: float | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {}
    if dpi is not None:
        kwargs["dpi"] = (dpi, dpi)
    img.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def write_png(img: Image.Image, path: Path, *, dpi: float | None = None) -> Path:
    path = Path(path)
    path.write_bytes(png_bytes(img, dpi=dpi))
    return path


def jpeg_bytes(img: Image.Image, *, quality: int = 90, orientation: int | None = None,
               subsampling: int = 2, progressive: bool = False) -> bytes:
    """A baseline (or progressive) JPEG; ``orientation`` writes an EXIF Orientation tag."""
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {"quality": quality, "subsampling": subsampling, "progressive": progressive}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def png_samples(width: int, height: int, *, color_type: int = 2, bit_depth: int = 8, seed: int = 0,
                palette_size: int | None = None) -> list[list[int]]:
    """Seeded sample rows for ``raw_png``: each row holds width x channels integers."""
    rng = random.Random(seed)
    if color_type == 3:
        top = (palette_size or min(256, 2 ** bit_depth)) - 1
    else:
        top = 2 ** bit_depth - 1
    per_row = width * _PNG_CHANNELS[color_type]
    return [[rng.randint(0, top) for _ in range(per_row)] for _ in range(height)]


def raw_png(width: int, height: int, *, color_type: int = 2, bit_depth: int = 8, rows: Sequence[Sequence[int]] | None = None,
            seed: int = 0, palette: Sequence[tuple[int, int, int]] | None = None, dpi: float | None = None,
            interlace: bool = False) -> bytes:
    """A PNG written by hand (any colour type and bit depth Pillow can't write, Adam7 if asked).

    ``rows`` are sample rows as ``png_samples`` returns them; without them, ``png_samples(seed=seed)``.
    """
    if bit_depth not in _PNG_DEPTHS.get(color_type, ()):
        raise ValueError(f"PNG colour type {color_type} can't have bit depth {bit_depth}")
    channels = _PNG_CHANNELS[color_type]
    if color_type == 3 and palette is None:
        n = min(256, 2 ** bit_depth)
        palette = [((i * 67) % 256, (i * 151) % 256, (i * 29) % 256) for i in range(n)]
    if rows is None:
        rows = png_samples(width, height, color_type=color_type, bit_depth=bit_depth, seed=seed,
                           palette_size=len(palette) if palette else None)
    if len(rows) != height or any(len(r) != width * channels for r in rows):
        raise ValueError("rows don't match width, height and channels")

    def pack(samples: Sequence[int]) -> bytes:
        if bit_depth == 16:
            return b"".join(struct.pack(">H", s) for s in samples)
        if bit_depth == 8:
            return bytes(samples)
        out = bytearray()
        per_byte = 8 // bit_depth
        for i in range(0, len(samples), per_byte):
            chunk = list(samples[i:i + per_byte]) + [0] * (per_byte - len(samples[i:i + per_byte]))
            b = 0
            for s in chunk:
                b = (b << bit_depth) | s
            out.append(b)
        return bytes(out)

    raw = bytearray()
    if not interlace:
        for r in rows:
            raw += b"\x00" + pack(r)
    else:
        for x0, y0, dx, dy in _ADAM7:
            xs = range(x0, width, dx)
            if not xs:
                continue
            for y in range(y0, height, dy):
                sub = [rows[y][x * channels + c] for x in xs for c in range(channels)]
                raw += b"\x00" + pack(sub)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 1 if interlace else 0))
    if dpi is not None:
        ppm = round(dpi / 0.0254)
        out += chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
    if color_type == 3:
        out += chunk(b"PLTE", b"".join(bytes(c) for c in palette))  # type: ignore[union-attr]
    out += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += chunk(b"IEND", b"")
    return out


# --------------------------------------------------------------------------- PDFs

@dataclass(frozen=True)
class TextSpec:
    """One line of base-14 Helvetica text. ``mode`` is the PDF render mode (3 is invisible, 7 is clip)."""

    text: str
    x: float = 72.0
    y: float = 720.0
    size: float = 12.0
    mode: int = 0
    fill_alpha: float | None = None     # ExtGState /ca; 0 hides the text
    gray: float | None = None           # fill colour, 0 black to 1 white


@dataclass(frozen=True)
class ImageSpec:
    """A PIL image (Flate, exact samples) or JPEG bytes (DCT, bytes untouched), placed at x, y, w, h in pt."""

    image: Any
    x: float = 0.0
    y: float = 0.0
    w: float | None = None              # None: the page width
    h: float | None = None              # None: the page height


@dataclass(frozen=True)
class PageSpec:
    width: float = 612.0
    height: float = 792.0
    rotate: int = 0
    cropbox: tuple[float, float, float, float] | None = None
    texts: tuple[TextSpec, ...] = ()
    images: tuple[ImageSpec, ...] = ()
    in_form: bool = False               # wrap the page's content in a Form XObject
    extra: bytes = b""                  # raw content operators appended after the rest


def _pdf_string(text: str) -> str:
    try:
        raw = text.encode("cp1252")
    except UnicodeEncodeError:
        raise ValueError("TextSpec text must be WinAnsi-encodable; use fpdf_text_pdf for other scripts") from None
    out = []
    for b in raw:
        ch = chr(b)
        if ch in "()\\":
            out.append("\\" + ch)
        elif 32 <= b < 127:
            out.append(ch)
        else:
            out.append(f"\\{b:03o}")
    return "".join(out)


def _image_xobject(pdf: Any, image: Any) -> Any:
    import pikepdf

    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(image), formats=["JPEG"]) as probe:
            mode, (w, h) = probe.mode, probe.size
        cs = {"L": "/DeviceGray", "RGB": "/DeviceRGB"}.get(mode)
        if cs is None:
            raise ValueError(f"ImageSpec JPEG bytes must be grey or RGB, not {mode}")
        stream = pikepdf.Stream(pdf, bytes(image))
        stream.Filter = pikepdf.Name.DCTDecode
        bpc = 8
    else:
        mode, (w, h) = image.mode, image.size
        if mode == "1":
            cs, bpc, data = "/DeviceGray", 1, image.tobytes()
        elif mode == "L":
            cs, bpc, data = "/DeviceGray", 8, image.tobytes()
        elif mode == "I;16":
            cs, bpc, data = "/DeviceGray", 16, image.tobytes("raw", "I;16B")
        elif mode == "RGB":
            cs, bpc, data = "/DeviceRGB", 8, image.tobytes()
        elif mode == "CMYK":
            cs, bpc, data = "/DeviceCMYK", 8, image.tobytes()
        else:
            raise ValueError(f"ImageSpec images must be 1, L, I;16, RGB or CMYK, not {mode}")
        stream = pikepdf.Stream(pdf, zlib.compress(data, 9))
        stream.Filter = pikepdf.Name.FlateDecode
    stream.Type = pikepdf.Name.XObject
    stream.Subtype = pikepdf.Name.Image
    stream.Width = w
    stream.Height = h
    stream.ColorSpace = pikepdf.Name(cs)
    stream.BitsPerComponent = bpc
    return stream


def pdf_bytes(pages: Sequence[PageSpec] | int = 1) -> bytes:
    """A PDF built with pikepdf from page specs (or that many blank Letter pages), deterministic bytes."""
    import pikepdf

    specs: Sequence[PageSpec] = [PageSpec() for _ in range(pages)] if isinstance(pages, int) else pages
    pdf = pikepdf.new()
    for spec in specs:
        page = pdf.add_blank_page(page_size=(spec.width, spec.height))
        ops: list[str] = []
        xobjects = pikepdf.Dictionary()
        states = pikepdf.Dictionary()
        fonts = pikepdf.Dictionary()
        for i, img in enumerate(spec.images):
            name = f"/Im{i}"
            xobjects[name] = pdf.make_indirect(_image_xobject(pdf, img.image))
            w = spec.width if img.w is None else img.w
            h = spec.height if img.h is None else img.h
            ops.append(f"q {w:g} 0 0 {h:g} {img.x:g} {img.y:g} cm {name} Do Q")
        if spec.texts:
            fonts["/F1"] = pdf.make_indirect(pikepdf.Dictionary(
                Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica, Encoding=pikepdf.Name.WinAnsiEncoding))
        for j, t in enumerate(spec.texts):
            parts = ["q"]
            if t.fill_alpha is not None:
                gs = f"/GS{j}"
                states[gs] = pikepdf.Dictionary(Type=pikepdf.Name.ExtGState, ca=t.fill_alpha, CA=t.fill_alpha)
                parts.append(f"{gs} gs")
            if t.gray is not None:
                parts.append(f"{t.gray:g} g")
            parts.append(f"BT /F1 {t.size:g} Tf {t.mode} Tr {t.x:g} {t.y:g} Td ({_pdf_string(t.text)}) Tj ET Q")
            ops.append(" ".join(parts))
        content = "\n".join(ops).encode("latin-1")
        if spec.extra:
            content += b"\n" + spec.extra
        resources = pikepdf.Dictionary()
        if len(fonts):
            resources.Font = fonts
        if len(xobjects):
            resources.XObject = xobjects
        if len(states):
            resources.ExtGState = states
        if spec.in_form:
            form = pikepdf.Stream(pdf, content)
            form.Type = pikepdf.Name.XObject
            form.Subtype = pikepdf.Name.Form
            form.BBox = pikepdf.Array([0, 0, spec.width, spec.height])
            form.Resources = resources
            page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=pdf.make_indirect(form)))
            page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, b"q /Fm0 Do Q"))
        else:
            page.obj.Resources = resources
            page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, content))
        if spec.rotate:
            page.obj.Rotate = spec.rotate
        if spec.cropbox is not None:
            page.obj.CropBox = pikepdf.Array(list(spec.cropbox))
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


def write_pdf(path: Path, pages: Sequence[PageSpec] | int = 1) -> Path:
    path = Path(path)
    path.write_bytes(pdf_bytes(pages))
    return path


def with_null_info(path: Path, title: str, null_keys: Sequence[str]) -> Path:
    """Give the PDF at ``path`` an Info dictionary with ``/Title`` and each of ``null_keys`` set to
    the PDF null, as some scanner apps and PDF writers leave them. pikepdf can't hold a null in a
    dictionary, so each key is written with a four-byte placeholder string and swapped for ``null``
    in the file's bytes; the lengths match, so the cross-reference table stays valid."""
    import pikepdf

    path = Path(path)
    with pikepdf.open(path) as pdf:
        info = pikepdf.Dictionary(Title=pikepdf.String(title))
        for key in null_keys:
            info[f"/{key}"] = pikepdf.String("NN")
        pdf.trailer.Info = pdf.make_indirect(info)
        buf = io.BytesIO()
        pdf.save(buf, deterministic_id=True, object_stream_mode=pikepdf.ObjectStreamMode.disable)
    data = buf.getvalue()
    for key in null_keys:
        old = f"/{key} (NN)".encode()
        assert data.count(old) == 1, key
        data = data.replace(old, f"/{key} null".encode())
    path.write_bytes(data)
    return path


def fpdf_text_pdf(path: Path, lines: Sequence[str], font: Path | bytes, *, size: float = 14.0,
                  page: tuple[float, float] = (595.28, 841.89), invisible: bool = False,
                  x: float = 72.0, y: float = 100.0, leading: float = 1.5) -> Path:
    """A PDF whose text uses an embedded TrueType font (a box font from ``box_font``), via fpdf2.

    Any script the font covers works; ``invisible`` writes the text in render mode 3.
    """
    from fpdf import FPDF
    from fpdf.enums import TextMode

    path = Path(path)
    font_path = Path(font) if not isinstance(font, (bytes, bytearray)) else None
    if font_path is None:
        font_path = path.with_suffix(".font.ttf")
        font_path.write_bytes(bytes(font))  # type: ignore[arg-type]
    pdf = FPDF(unit="pt", format=page)
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_auto_page_break(False)
    pdf.add_page()
    pdf.add_font("FixtureFont", fname=str(font_path))
    pdf.set_font("FixtureFont", size=size)
    if invisible:
        pdf.text_mode = TextMode.INVISIBLE
    for i, text in enumerate(lines):
        pdf.text(x, y + i * size * leading, text)
    path.write_bytes(bytes(pdf.output()))
    return path


# --------------------------------------------------------------------------- fonts

def _codepoints(chars: str | Iterable[str | int]) -> list[int]:
    cps: list[int] = []
    for c in chars:
        cp = c if isinstance(c, int) else ord(c)
        if cp not in cps:
            cps.append(cp)
    return sorted(cps)


def _glyph_name(cp: int) -> str:
    return f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:05X}"


def box_font(chars: str | Iterable[str | int], *, family: str = "Example Box", style: str = "Regular",
             advance: int = 600, mark_advance: int = 0, advances: Mapping[str | int, int] | None = None,
             units_per_em: int = 1000, ascent: int = 800, descent: int = -200) -> bytes:
    """A TrueType font with a box glyph for each character, built with fontTools FontBuilder.

    Combining marks (category Mn) get ``mark_advance`` (0) and a small box above the baseline to the
    left of the pen; spaces get an empty glyph. ``advances`` overrides per character.
    The ``family`` can be any name, which is how a test builds a font that claims a script it lacks.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    cps = _codepoints(chars)
    overrides = {(k if isinstance(k, int) else ord(k)): v for k, v in (advances or {}).items()}
    order = [".notdef"] + [_glyph_name(cp) for cp in cps]
    glyphs: dict[str, Any] = {}
    widths: dict[str, int] = {".notdef": advance}

    def box(x0: int, y0: int, x1: int, y1: int) -> Any:
        pen = TTGlyphPen(None)
        pen.moveTo((x0, y0))
        pen.lineTo((x0, y1))
        pen.lineTo((x1, y1))
        pen.lineTo((x1, y0))
        pen.closePath()
        return pen.glyph()

    glyphs[".notdef"] = box(50, 0, advance - 50, ascent - 100)
    for cp in cps:
        name = _glyph_name(cp)
        cat = unicodedata.category(chr(cp))
        if cat == "Mn":
            w = overrides.get(cp, mark_advance)
            glyphs[name] = box(-advance + 150, ascent - 150, -150, ascent)
        elif cat.startswith("Z"):
            w = overrides.get(cp, advance // 2)
            glyphs[name] = TTGlyphPen(None).glyph()
        else:
            w = overrides.get(cp, advance)
            right = max(w - 50, 60)
            glyphs[name] = box(50, 0, right, ascent - 100)
        widths[name] = w
    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({cp: _glyph_name(cp) for cp in cps})
    fb.setupGlyf(glyphs)
    glyf = fb.font["glyf"]
    fb.setupHorizontalMetrics({g: (widths[g], getattr(glyf[g], "xMin", 0)) for g in order})
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    ps = "".join(ch for ch in family if ch.isalnum()) + "-" + "".join(ch for ch in style if ch.isalnum())
    fb.setupNameTable({"familyName": family, "styleName": style, "uniqueFontIdentifier": ps,
                       "fullName": f"{family} {style}", "psName": ps, "version": "Version 1.000"})
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=descent, sTypoLineGap=0,
                usWinAscent=ascent, usWinDescent=abs(descent), fsType=0, achVendID="NONE")
    fb.setupPost()
    fb.setupHead(unitsPerEm=units_per_em, created=_HEAD_TIMESTAMP, modified=_HEAD_TIMESTAMP)
    fb.font.recalcTimestamp = False
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue()


def write_box_font(path: Path, chars: str | Iterable[str | int], **kwargs: Any) -> Path:
    path = Path(path)
    path.write_bytes(box_font(chars, **kwargs))
    return path


# --------------------------------------------------------------------------- synthetic reads

def quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    """An axis-aligned quad: clockwise from the top-left, in px, y down."""
    return ((float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1)))


def word(text: str, box: Quad | None = None, *, conf: float = 0.9) -> Word:
    return Word(text, conf, box if box is not None else quad(0, 0, 18 * max(len(text), 1), 40))


def line(text: str, box: Quad | None = None, *, conf: float = 0.9, with_words: bool = False,
         char_w: float = 18.0) -> Line:
    box = box if box is not None else quad(100, 100, 100 + char_w * max(len(text), 1), 140)
    words: tuple[Word, ...] = ()
    if with_words:
        (x0, y0), _, (_, y1), _ = box
        out = []
        x = x0
        for part in text.split():
            out.append(Word(part, conf, quad(x, y0, x + char_w * len(part), y1)))
            x += char_w * (len(part) + 1)
        words = tuple(out)
    return Line(text, conf, box, words)


def lines_for(texts: Sequence[str], *, x: float = 100.0, y: float = 100.0, height: float = 40.0, gap: float = 20.0,
              char_w: float = 18.0, conf: float = 0.9, with_words: bool = False) -> list[Line]:
    """One Line per text, laid out top to bottom with boxes that depend only on the text."""
    out = []
    for i, text in enumerate(texts):
        top = y + i * (height + gap)
        out.append(line(text, quad(x, top, x + char_w * max(len(text), 1), top + height),
                        conf=conf, with_words=with_words, char_w=char_w))
    return out


def engine_read(use_id: str, texts: Sequence[str] = (), *, engine: str | None = None, spec: str | None = None,
                role: str = "reader", status: str | None = None, seconds: float = 0.0, conf: float = 0.9,
                error: str | None = None, meta: Mapping[str, Any] | None = None, lines: Sequence[Line] | None = None,
                with_words: bool = False) -> EngineRead:
    """A synthetic EngineRead. Status defaults to "ok" with text and "empty" without."""
    lns = list(lines) if lines is not None else lines_for(texts, conf=conf, with_words=with_words)
    if status is None:
        status = "ok" if lns else "empty"
    return EngineRead(use_id=use_id, engine=engine or use_id, spec=spec or f"{engine or use_id}:synthetic",
                      role=role, status=status, lines=lns, seconds=seconds,  # type: ignore[arg-type]
                      error=error, meta=dict(meta or {}))


def read_to_json(read: EngineRead) -> dict[str, Any]:
    from pdfintake.types import to_jsonable

    return to_jsonable(read, include_text=True)


def read_from_json(data: Mapping[str, Any]) -> EngineRead:
    """An EngineRead from its JSON shape (made-up reads stored as JSON)."""
    def q(box: Sequence[Sequence[float]]) -> Quad:
        pts = tuple((float(p[0]), float(p[1])) for p in box)
        if len(pts) != 4:
            raise ValueError("a box has four points")
        return pts  # type: ignore[return-value]

    lines = []
    for ln in data.get("lines", []):
        words = tuple(Word(w["text"], float(w["conf"]), q(w["box"])) for w in ln.get("words", []))
        lines.append(Line(ln["text"], float(ln["conf"]), q(ln["box"]), words))
    return EngineRead(use_id=data["use_id"], engine=data["engine"], spec=data["spec"], role=data["role"],
                      status=data["status"], lines=lines, seconds=float(data.get("seconds", 0.0)),
                      error=data.get("error"), meta=dict(data.get("meta", {})))


def page_image(folder: Path, *, image: Image.Image | None = None, index: int = 0, dpi: float = 300.0,
               cropbox: tuple[float, float, float, float] | None = None, name: str | None = None) -> PageImage:
    """Write ``image`` (default: a text-like page) as a PNG with dpi and return its PageImage."""
    img = image if image is not None else text_like()
    path = Path(folder) / (name or f"page-{index:04d}.png")
    write_png(img if img.mode in ("1", "L", "RGB", "RGBA", "I;16", "P", "LA") else img.convert("RGB"), path, dpi=dpi)
    w, h = img.size
    box = cropbox if cropbox is not None else (0.0, 0.0, w * 72.0 / dpi, h * 72.0 / dpi)
    return PageImage(index=index, path=path, width=w, height=h, dpi=float(dpi), cropbox=box)


# --------------------------------------------------------------------------- a fake engine

class FakeSession:
    """Reads dark horizontal bands as lines; deterministic, thread-safe, honours timeouts."""

    def __init__(self, engine: FakeEngine, use_id: str, role: str, params: Mapping[str, Any],
                 models: Mapping[str, Path]) -> None:
        self.engine = engine
        self.use_id = use_id
        self.role = role
        self.params = dict(params)
        self.models = dict(models)
        self.closed = False

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        import numpy as np

        start = time.perf_counter()
        spec = f"fake:{self.params.get('lang', 'en')}"

        def result(status: str, lines: list[Line]) -> EngineRead:
            return EngineRead(self.use_id, self.engine.name, spec, self.role, status, lines,  # type: ignore[arg-type]
                              time.perf_counter() - start)

        delay = float(self.params.get("delay_s") or 0.0)
        while time.perf_counter() - start < delay:
            if time.perf_counter() - start >= timeout_s:
                return result("timeout", [])
            time.sleep(min(0.005, max(timeout_s / 4, 0.0005)))
        if time.perf_counter() - start >= timeout_s:
            return result("timeout", [])
        with page.open() as img:
            dark = np.asarray(img.convert("L")) < 128
        rows = dark.any(axis=1)
        bands: list[tuple[int, int]] = []
        y = 0
        n = len(rows)
        while y < n:
            if rows[y]:
                y0 = y
                while y < n and rows[y]:
                    y += 1
                bands.append((y0, y))
            else:
                y += 1
        lines: list[Line] = []
        for k, (y0, y1) in enumerate(bands):
            cols = np.flatnonzero(dark[y0:y1].any(axis=0))
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            texts = self.engine.texts
            text = texts[k] if texts is not None and k < len(texts) else f"fake line {k + 1}"
            conf = round(max(0.5, 0.95 - 0.01 * k), 4)
            lines.append(Line(text, conf, quad(x0, y0, x1, y1)))
        return result("ok" if lines else "empty", lines)

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    """An engine that satisfies the Engine protocol without OCR, for contract and pipeline tests."""

    name = "fake"
    api = 1
    PARAMS = ParamTable(
        Param("lang", "str", pattern=r"[a-z]{2,3}", default="en"),
        Param("delay_s", "float", min=0.0, max=60.0, default=0.0),
    )
    MODEL_KINDS = frozenset({"traineddata"})
    platforms = frozenset({"fake-os"})
    session_per_thread = False

    def __init__(self, *, required_models: Sequence[str] = (), languages: Sequence[str] = ("en",),
                 texts: Sequence[str] | None = None, platforms: Iterable[str] | None = None,
                 case: ContractCase | None = None) -> None:
        self.required_models = tuple(required_models)
        self.languages = tuple(languages)
        self.texts = list(texts) if texts is not None else None
        if platforms is not None:
            self.platforms = frozenset(platforms)  # type: ignore[misc]
        self.case = case
        self.opened: list[FakeSession] = []

    def version(self) -> str | None:
        return "1.0"

    def availability(self, params: Mapping[str, Any], models: Mapping[str, Path], platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform", f"the fake engine runs only on {', '.join(sorted(self.platforms))}")
        for key in self.required_models:
            path = models.get(key)
            if path is None or not Path(path).is_file():
                return Availability(False, "missing_model", f"model {key} is missing")
        if params.get("lang", "en") not in self.languages:
            return Availability(False, "unsupported_language", f"language {params.get('lang')} is not supported")
        return Availability(True, "available", "")

    def open(self, use_id: str, role: str, params: Mapping[str, Any], models: Mapping[str, Path]) -> FakeSession:
        session = FakeSession(self, use_id, role, params, models)
        self.opened.append(session)
        return session

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence[Any]) -> list[dict[str, Any]]:
        return [{"lang": lang_tag}] if lang_tag in self.languages else []

    def contract_case(self) -> ContractCase | None:
        return self.case
