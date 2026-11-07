"""Bench pages and canaries, rendered from the public corpus with the pinned OFL fonts.

``render_pages`` lays out the corpus text as four page kinds (DESIGN §4.5 step 4):

* ``prose``: paragraphs at 11 pt;
* ``form``: label and value rows (labels from the corpus, values made up: dates, amounts, ids);
* ``microprint``: prose at 5 pt;
* ``mixed_script``: the pack's lines alternating with made-up lines in a second script, when a
  corpus font covers one (that's what the mixed-script trigger is measured on).

Each page is built once as a PDF with fpdf2, with HarfBuzz shaping (uharfbuzz, the ``[kit]``
extra), and rendered with pdfium at three conditions: ``clean`` (300 dpi), ``scan`` (200 dpi, then a
seeded ``degrade``) and ``small`` (110 dpi). The truth is exactly the lines drawn, so a bench page
never holds text that isn't on it. The same corpus, fonts and seed give the same pages: the layout
uses Python's seeded ``random``, the degrade numpy's seeded generator, and the PDFs a fixed date.

The pages and their truth (``pages.json``) live in the kit's work folder in the cache, never in the
repo: the truth is corpus text, fetched at bench time and never committed.

``render_canary`` draws the canary line a pack's wheel carries: same route, one line, 300 dpi,
greyscale, deterministic bytes. The pixels come out the same on every OS; the PNG is encoded here
(``png_bytes``) rather than by Pillow, because Pillow's zlib differs by platform (zlib-ng in the
macOS and Windows wheels, classic zlib on Linux) and the same pixels compressed to other bytes on
Linux, so a pin made on one OS failed to rebuild on another.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import struct
import unicodedata
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..errors import PackError, PdfIntakeError, SetupError

if TYPE_CHECKING:
    import PIL.Image

    from .corpus import Corpus

CONDITIONS: Mapping[str, float] = {"clean": 300.0, "scan": 200.0, "small": 110.0}
# The one seed the kit renders with. `kit render` and `kit verify --rebench` both use it, so CI
# re-renders exactly the pages the claimed numbers came from (decided 2026-11-05).
BENCH_SEED = 0
MIN_FONTS_PER_SCRIPT = 2                 # DESIGN §3.6: at least two corpus fonts per script
PAGES_FILE = "pages.json"
PAGE_SIZE = (595.28, 841.89)            # A4 in pt
MARGIN = 56.0
FIXED_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
PROSE_SIZE, FORM_SIZE, MICRO_SIZE = 11.0, 10.0, 5.0
CANARY_SIZE = 28.0
CANARY_MAX_BYTES = 2 * 1024 * 1024

# Made-up lines for the second script of a mixed-script page (never document text), tried in this
# order: the first script other than the pack's own that a corpus font covers. A non-Latin pack gets
# Latin (its corpus pins Latin fonts anyway); a Latin pack gets Greek, else Cyrillic, which NotoSans
# and NotoSerif both cover, so it needs no extra font (kit trial es, item 8).
MIXED_SAMPLES: Mapping[str, tuple[str, ...]] = {
    "Latin": ("The sample order lists seven example items for the test account.",
              "Please send the example form back before the end of the month."),
    "Greek": ("Ένα παράδειγμα κειμένου για τη δοκιμή.",),
    "Cyrillic": ("Пример текста для проверки.",),
    "Thai": ("ข้อความตัวอย่างสำหรับการทดสอบ", "บริษัทตัวอย่าง จำกัด เลขที่ 4521"),
}


@dataclass(frozen=True)
class BenchPage:
    id: str
    cond: Literal["clean", "scan", "small"]
    kind: Literal["prose", "form", "microprint", "mixed_script"]
    png: Path
    truth: str
    dpi: float


# --------------------------------------------------------------------------- fonts

@dataclass(frozen=True)
class _Font:
    path: Path
    family: str
    cmap: frozenset[int]


def _load_fonts(fonts: Sequence[Path]) -> list[_Font]:
    from fontTools.ttLib import TTFont

    out = []
    for i, p in enumerate(fonts):
        try:
            with TTFont(str(p), lazy=True) as tt:
                cmap = frozenset(tt.getBestCmap() or {})
        except Exception as exc:
            raise PackError("kit_font", f"the font {Path(p).name} can't be read: {type(exc).__name__}") from None
        out.append(_Font(Path(p), f"KitFont{i}", cmap))
    if not out:
        raise PackError("kit_font", "the bench needs at least one font (corpus.json fonts)")
    return out


def font_coverage_problems(fonts: Sequence[Path], scripts: Sequence[str]) -> list[str]:
    """Each of ``scripts`` that fewer than ``MIN_FONTS_PER_SCRIPT`` of ``fonts`` cover, as a
    problem. A font covers a script when its cmap holds the script's ``required`` code points,
    the test the layer writer uses."""
    from ..ocr import scripts as S

    loaded = _load_fonts(fonts) if fonts else []
    out = []
    for name in scripts:
        sc = S.SCRIPTS.get(name)
        if sc is None:
            continue
        n = sum(1 for f in loaded if sc.required <= f.cmap)
        if n < MIN_FONTS_PER_SCRIPT:
            out.append(f"{name}: {n} corpus font{'' if n == 1 else 's'} cover{'s' if n == 1 else ''} it, "
                       f"the bench needs at least {MIN_FONTS_PER_SCRIPT}")
    return out


def _covers(font: _Font, text: str) -> bool:
    return all(ord(ch) in font.cmap for ch in text if not ch.isspace())


def _script_of_text(text: str) -> str | None:
    from ..ocr import scripts

    counts: dict[str, int] = {}
    for ch in text:
        s = scripts.script_of(ch)
        if s is not None and s in scripts.SCRIPTS:
            counts[s] = counts.get(s, 0) + 1
    return max(sorted(counts), key=lambda k: counts[k]) if counts else None


def _new_pdf(fonts: Sequence[_Font]) -> Any:
    try:
        import uharfbuzz  # noqa: F401  (fpdf2's shaping engine; the [kit] extra)
    except ImportError:
        raise SetupError("kit_extra_missing", "the kit needs HarfBuzz shaping: install pdfintake[kit]") from None
    from fpdf import FPDF

    pdf = FPDF(unit="pt", format=PAGE_SIZE)
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_auto_page_break(False)
    pdf.set_margin(0)
    pdf.c_margin = 0                     # a cell's text starts at its left edge (_draw_line)
    for f in fonts:
        pdf.add_font(f.family, fname=str(f.path))
    pdf.set_text_shaping(True)
    return pdf


def _render_pdf(pdf_path: Path, dpi: float, dest: Path, *, gray: bool = True) -> tuple[int, int]:
    """Render page 0 of ``pdf_path`` at ``dpi`` to a greyscale PNG; its pixel size."""
    from ..pdf.render import Renderer

    with Renderer(pdf_path) as r:
        img = r.render_scaled(0, dpi, gray=gray)
    try:
        img = img.convert("L")
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="PNG", dpi=(dpi, dpi), compress_level=6)
        return img.size
    finally:
        img.close()


# --------------------------------------------------------------------------- text

_WORD_RE = re.compile(r"\S+")


def _words(text: str) -> list[str]:
    """The corpus text's words: every run of non-space characters holding a letter or a digit, so a
    number ("300", "79)") stays in the sentence, or a bracket that isn't half of an empty pair ("%)"
    in "(0,5 %)"), so a sentence keeps both its brackets. Other bare punctuation ("–", "|", the "()"
    a dropped template leaves) is left out."""
    return [w for w in _WORD_RE.findall(text)
            if any(ch.isalnum() for ch in w) or any(ch in "()" for ch in w.replace("()", ""))]


def _wrap(pdf: Any, words: Sequence[str], width: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}" if cur else w
        if pdf.get_string_width(cand) <= width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        cur = w
        while pdf.get_string_width(cur) > width and len(cur) > 1:   # a word wider than the line
            cur = cur[:-1]
    if cur:
        lines.append(cur)
    return lines


def _font_for(fonts: Sequence[_Font], text: str, prefer: int) -> _Font | None:
    order = list(fonts[prefer % len(fonts):]) + list(fonts[:prefer % len(fonts)])
    for f in order:
        if _covers(f, text):
            return f
    return None


def _filter_covered(words: Sequence[str], font: _Font) -> list[str]:
    return [w for w in words if _covers(font, w)]


def _value(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        return f"{rng.randrange(1, 29):02d}.{rng.randrange(1, 13):02d}.20{rng.randrange(20, 30)}"
    if kind == 1:
        return f"{rng.randrange(1, 99)} {rng.randrange(0, 1000):03d},{rng.randrange(0, 100):02d}"
    if kind == 2:
        letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
        return f"{letters}-{rng.randrange(1000, 10000)}-{rng.randrange(100, 1000)}"
    return str(rng.randrange(10000, 1000000))


@dataclass
class _Layout:
    lines: list[tuple[str, _Font, float]]       # text, font, size


def _prose(pdf: Any, words: Sequence[str], rng: random.Random, font: _Font, size: float) -> _Layout:
    pdf.set_font(font.family, size=size)
    usable = [w for w in words if _covers(font, w)]
    if not usable:
        return _Layout([])
    leading = size * 1.45
    rows = int((PAGE_SIZE[1] - 2 * MARGIN) // leading)
    start = rng.randrange(max(1, len(usable) - 40)) if len(usable) > 40 else 0
    take = usable[start:] + usable[:start]
    lines = _wrap(pdf, take[: rows * 20], PAGE_SIZE[0] - 2 * MARGIN)[:rows]
    return _Layout([(ln, font, size) for ln in lines])


def _form(pdf: Any, words: Sequence[str], rng: random.Random, font: _Font) -> _Layout:
    pdf.set_font(font.family, size=FORM_SIZE)
    labels = [w.strip(".,;:()[]\"'«»„“”") for w in words if _covers(font, w)]
    labels = [w for w in labels if len(w) >= 4 and any(ch.isalpha() for ch in w)]   # a label is a word, not a number
    if not labels:
        return _Layout([])
    out = []
    for _ in range(24):
        a, b = rng.choice(labels), rng.choice(labels)
        text = f"{a} {b}: {_value(rng)}"
        if pdf.get_string_width(text) <= PAGE_SIZE[0] - 2 * MARGIN:
            out.append((text, font, FORM_SIZE))
    return _Layout(out)


def _script_part(text: str, script: str) -> str:
    from ..ocr import scripts

    return "".join(ch for ch in text if scripts.script_of(ch) == script)


def _mixed(pdf: Any, words: Sequence[str], rng: random.Random, fonts: Sequence[_Font], font: _Font,
           script: str | None) -> _Layout:
    other = None
    for name, samples in MIXED_SAMPLES.items():
        if name == script:
            continue
        f = next((x for x in fonts if all(_covers(x, _script_part(s, name)) for s in samples)), None)
        if f is not None and all(_covered(fonts, s) for s in samples):
            other = (samples, f)
            break
    if other is None:
        return _Layout([])
    prose = _prose(pdf, words, rng, font, PROSE_SIZE).lines[:12]
    out: list[tuple[str, _Font, float]] = []
    for i, line in enumerate(prose):
        out.append(line)
        if i % 3 == 1:
            samples, f = other
            out.append((samples[(i // 3) % len(samples)], f, PROSE_SIZE))
    return _Layout(out)


def _runs(text: str, fonts: Sequence[_Font], prefer: _Font) -> list[tuple[_Font, str]] | None:
    """``text`` cut into runs, each drawn with one font: the preferred font where it has the
    character, else the first font that has it (spaces stay with the run they're in). None when a
    character is in no font. fpdf2's own fallback fonts don't work with text shaping, so the kit cuts
    the runs itself."""
    out: list[tuple[_Font, str]] = []
    for ch in text:
        if ch.isspace() and out:
            out[-1] = (out[-1][0], out[-1][1] + ch)
            continue
        cur = out[-1][0] if out else None
        if cur is not None and ord(ch) in cur.cmap and (ord(ch) not in prefer.cmap or cur is prefer):
            out[-1] = (cur, out[-1][1] + ch)
            continue
        f = prefer if ord(ch) in prefer.cmap else next((x for x in fonts if ord(ch) in x.cmap), None)
        if f is None:
            if ch.isspace():
                continue
            return None
        out.append((f, ch))
    return out


def _covered(fonts: Sequence[_Font], text: str) -> bool:
    return all(any(ord(ch) in f.cmap for f in fonts) for ch in text if not ch.isspace())


def _text_width(pdf: Any, runs: Sequence[tuple[_Font, str]], size: float) -> float:
    w = 0.0
    for f, s in runs:
        pdf.set_font(f.family, size=size)
        w += pdf.get_string_width(s)
    return w


def _draw_line(pdf: Any, x: float, y: float, text: str, fonts: Sequence[_Font], prefer: _Font, size: float) -> None:
    """Draw ``text`` with its left end at ``x`` and its baseline at ``y``, run by run, each run shaped.

    fpdf2's ``text()`` places glyphs straight from the cmap with no shaping, even with
    ``set_text_shaping(True)``: a Thai tone mark over an upper vowel lands on the vowel, and kerning
    and ligatures are lost. ``cell()`` goes through HarfBuzz, so every run of every script is drawn
    with it. A cell puts its baseline at top + h/2 + 0.3 * font size, so a cell of height ``size``
    whose top is at ``y - 0.8 * size`` has its baseline at ``y``; ``_new_pdf`` sets the cell margin
    to 0, so the run starts exactly at ``x``. Each run advances by its shaped width, as before."""
    runs = _runs(text, fonts, prefer) or []
    for f, s in runs:
        pdf.set_font(f.family, size=size)
        w = pdf.get_string_width(s)
        pdf.set_xy(x, y - 0.8 * size)
        pdf.cell(w=w, h=size, text=s)
        x += w


def _draw(layout: _Layout, fonts: Sequence[_Font], dest: Path) -> None:
    pdf = _new_pdf(fonts)
    pdf.add_page()
    y = MARGIN
    for text, font, size in layout.lines:
        y += size * 1.45
        _draw_line(pdf, MARGIN, y, text, fonts, font, size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(pdf.output()))


# --------------------------------------------------------------------------- degrade

def degrade(img: "PIL.Image.Image", seed: int) -> "PIL.Image.Image":
    """A scanned look, the same for the same image and seed: a small tilt, blur and sensor noise."""
    import numpy as np
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(int(seed))
    g = img.convert("L")
    angle = float(rng.uniform(-0.8, 0.8))
    g = g.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    g = g.filter(ImageFilter.GaussianBlur(radius=0.7))
    arr = np.asarray(g, dtype=np.float32)
    arr = arr * 0.92 + 12.0 + rng.normal(0.0, 9.0, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), mode="L")


# --------------------------------------------------------------------------- the bench pages

def _texts(corpus: "Corpus", texts: Mapping[str, str] | None, cache: Path | None) -> list[str]:
    from . import corpus as C

    out = []
    for t in corpus.texts:
        if texts is not None and t.sha256 in texts:
            out.append(texts[t.sha256])
        else:
            out.append(C.fetch_text(t, cache=cache))
    return out


def render_pages(corpus: "Corpus", fonts: Sequence[Path], outdir: Path, *, seed: int = BENCH_SEED,
                 texts: Mapping[str, str] | None = None, cache: Path | None = None,
                 scripts: Sequence[str] | None = None) -> list[BenchPage]:
    """Render every bench page of ``corpus`` into ``outdir`` and write ``pages.json``.

    ``texts`` maps a text's SHA-256 to its plain words (for a caller that already has them);
    otherwise each text is fetched and checked (``corpus.fetch_text``, cached in ``cache``).
    ``scripts`` (the pack's, as the kit commands pass them) must each be covered by at least
    ``MIN_FONTS_PER_SCRIPT`` fonts, or ``kit_fonts_too_few`` (kit exit 1) before anything is drawn.
    """
    if scripts:
        problems = font_coverage_problems(fonts, scripts)
        if problems:
            raise PdfIntakeError("kit_fonts_too_few", "corpus.json pins too few fonts: " + "; ".join(problems)
                            + " (pin more with `pdfintake kit corpus --font URL`)", problems=problems)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loaded = _load_fonts(fonts)
    plain = _texts(corpus, texts, cache)
    words = _words(unicodedata.normalize("NFC", "\n".join(plain)))
    if not words:
        raise PackError("kit_corpus_empty", "the corpus has no words to render (kit corpus pins the texts)")
    script = _script_of_text(" ".join(words[:2000]))
    sample = words[:200]
    # The fonts that can draw the pack's own text; the others serve the second script of mixed pages.
    covering = [f for f in loaded if sum(_covers(f, w) for w in sample) * 2 >= len(sample)] or loaded
    pages: list[BenchPage] = []
    for kind in ("prose", "form", "microprint", "mixed_script"):
        for i in range(int(corpus.pages.get(kind, 0))):
            rng = random.Random(f"{seed}:{kind}:{i}")
            font = covering[i % len(covering)]
            sizer = _new_pdf(loaded)
            sizer.add_page()
            if kind == "prose":
                layout = _prose(sizer, words, rng, font, PROSE_SIZE)
            elif kind == "microprint":
                layout = _prose(sizer, words, rng, font, MICRO_SIZE)
            elif kind == "form":
                layout = _form(sizer, words, rng, font)
            else:
                layout = _mixed(sizer, words, rng, loaded, font, script)
            if not layout.lines:
                continue
            base = f"{kind}-{i}"
            pdf_path = outdir / "pdf" / f"{base}.pdf"
            _draw(layout, loaded, pdf_path)
            truth = "\n".join(t for t, _f, _s in layout.lines)
            for cond, dpi in CONDITIONS.items():
                png = outdir / f"{base}-{cond}.png"
                _render_pdf(pdf_path, dpi, png)
                if cond == "scan":
                    from PIL import Image

                    with Image.open(png) as im:
                        out = degrade(im, int(hashlib.sha256(f"{seed}:{base}".encode()).hexdigest()[:8], 16))
                    out.save(png, format="PNG", dpi=(dpi, dpi), compress_level=6)
                pages.append(BenchPage(f"{base}-{cond}", cond, kind, png, truth, dpi))  # type: ignore[arg-type]
    _write_index(outdir, pages, seed)
    return pages


def _write_index(outdir: Path, pages: Sequence[BenchPage], seed: int) -> None:
    data = {"seed": seed, "pages": [{"id": p.id, "cond": p.cond, "kind": p.kind, "png": p.png.name,
                                     "dpi": p.dpi, "truth": p.truth} for p in pages]}
    (Path(outdir) / PAGES_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def load_pages(outdir: Path) -> list[BenchPage]:
    """The pages ``render_pages`` wrote into ``outdir`` (PackError when there are none)."""
    path = Path(outdir) / PAGES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PackError("kit_no_pages", f"no bench pages in {outdir}; run `pdfintake kit render` first") from None
    except (OSError, ValueError) as exc:
        raise PackError("kit_no_pages", f"{path} isn't readable: {exc}") from None
    out = []
    for p in data.get("pages", []):
        png = Path(outdir) / str(p["png"])
        if not png.is_file():
            raise PackError("kit_no_pages", f"{png} is missing; run `pdfintake kit render` again")
        out.append(BenchPage(str(p["id"]), p["cond"], p["kind"], png, str(p["truth"]), float(p["dpi"])))
    return out


# --------------------------------------------------------------------------- the canary

def render_canary(text: str, fonts: Sequence[Path], dest: Path, *, dpi: int = 300) -> Path:
    """One line of ``text`` (HarfBuzz shaping; each run in a font that has it), rendered at
    ``dpi`` to a greyscale PNG. The same text, fonts and dpi give the same bytes."""
    if not text or not text.strip():
        raise PackError("kit_canary", "the canary needs some text")
    loaded = _load_fonts(fonts)
    uncovered = sorted({ch for ch in text if not ch.isspace() and not any(ord(ch) in f.cmap for f in loaded)})
    if uncovered:
        raise PackError("kit_canary", "no font covers " + ", ".join(f"U+{ord(c):04X}" for c in uncovered))
    first = next(ch for ch in text if not ch.isspace())
    font = next(f for f in loaded if ord(first) in f.cmap)
    runs = _runs(text, loaded, font) or []
    sizer = _new_pdf(loaded)
    sizer.add_page()
    width = _text_width(sizer, runs, CANARY_SIZE) + 2 * 24.0
    height = CANARY_SIZE * 2.4
    pdf = _new_pdf(loaded)
    pdf.add_page(format=(max(width, 72.0), height))
    _draw_line(pdf, 24.0, CANARY_SIZE * 1.55, text, loaded, font, CANARY_SIZE)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdf = dest.with_suffix(".pdf.tmp")
    tmp_pdf.write_bytes(bytes(pdf.output()))
    try:
        from ..pdf.render import Renderer

        with Renderer(tmp_pdf) as r:
            img = r.render_scaled(0, float(dpi), gray=True)
        try:
            gray = img.convert("L")
            dest.write_bytes(png_bytes(gray.tobytes(), gray.width, gray.height, dpi=dpi))
            gray.close()
        finally:
            img.close()
    finally:
        tmp_pdf.unlink(missing_ok=True)
    if dest.stat().st_size > CANARY_MAX_BYTES:
        raise PackError("kit_canary", f"the canary image is larger than {CANARY_MAX_BYTES} bytes")
    return dest


# --------------------------------------------------------------------------- deterministic PNG

# A canary's bytes are pinned, so its PNG must come out the same from any zlib build: this writes
# the deflate stream itself (RFC 1951, one block with the fixed Huffman codes, greedy matches from
# a table of the last place each 3-byte string was seen). Only the checksums come from ``zlib``
# (crc32 and adler32 are arithmetic, the same everywhere); ``zlib.compress`` is never used.

_LEN_BASE = (3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
             131, 163, 195, 227, 258)
_LEN_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0)
_DIST_BASE = (1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
              2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577)
_DIST_EXTRA = (0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13)
_WINDOW = 32768
_MAX_MATCH = 258


def _rev(code: int, n: int) -> int:
    out = 0
    for _ in range(n):
        out = (out << 1) | (code & 1)
        code >>= 1
    return out


def _fixed_lit(sym: int) -> tuple[int, int]:
    """The fixed Huffman code of a literal/length symbol (RFC 1951 3.2.6), bit-reversed for LSB-first output."""
    if sym < 144:
        return _rev(0x30 + sym, 8), 8
    if sym < 256:
        return _rev(0x190 + sym - 144, 9), 9
    if sym < 280:
        return _rev(sym - 256, 7), 7
    return _rev(0xC0 + sym - 280, 8), 8


_LIT = tuple(_fixed_lit(s) for s in range(288))
_DIST = tuple((_rev(d, 5), 5) for d in range(30))


def _code_of(value: int, bases: tuple[int, ...]) -> int:
    i = len(bases) - 1
    while bases[i] > value:
        i -= 1
    return i


def deflate_fixed(data: bytes) -> bytes:
    """A raw deflate stream for ``data``: deterministic, independent of any zlib build."""
    acc = 0
    nbits = 0
    out = bytearray()

    def put(value: int, n: int) -> None:
        nonlocal acc, nbits
        acc |= value << nbits
        nbits += n
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8

    put(1, 1)                                            # BFINAL
    put(1, 2)                                            # BTYPE 01: fixed Huffman codes
    last: dict[bytes, int] = {}
    n = len(data)
    i = 0
    while i < n:
        length = 0
        dist = 0
        if i + 3 <= n:
            key = data[i:i + 3]
            j = last.get(key)
            if j is not None and i - j <= _WINDOW:
                limit = min(_MAX_MATCH, n - i)
                k = 3
                while k < limit and data[j + k] == data[i + k]:
                    k += 1
                length, dist = k, i - j
            last[key] = i
        if length:
            li = _code_of(length, _LEN_BASE)
            put(*_LIT[257 + li])
            if _LEN_EXTRA[li]:
                put(length - _LEN_BASE[li], _LEN_EXTRA[li])
            di = _code_of(dist, _DIST_BASE)
            put(*_DIST[di])
            if _DIST_EXTRA[di]:
                put(dist - _DIST_BASE[di], _DIST_EXTRA[di])
            for p in range(i + 1, min(i + length, n - 2)):
                last[data[p:p + 3]] = p
            i += length
        else:
            put(*_LIT[data[i]])
            i += 1
    put(*_LIT[256])                                      # end of block
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def png_bytes(pixels: bytes, width: int, height: int, *, dpi: int) -> bytes:
    """An 8-bit greyscale PNG of ``pixels`` (row-major, one byte each) with its dpi, the same bytes
    on every OS: IHDR, pHYs, one IDAT (filter 0 on every row, ``deflate_fixed``), IEND."""
    if len(pixels) != width * height:
        raise ValueError(f"{len(pixels)} pixel bytes for a {width} x {height} image")
    raw = b"".join(b"\x00" + pixels[y * width:(y + 1) * width] for y in range(height))
    stream = b"\x78\x01" + deflate_fixed(raw) + struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF)
    ppm = int(dpi / 0.0254 + 0.5)                        # pixels per metre, as Pillow writes it
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + _chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
            + _chunk(b"IDAT", stream)
            + _chunk(b"IEND", b""))
