"""The invisible text layer (pdf/textlayer.py), on synthetic box fonts and made-up text.

The real Noto fonts are used only when they are present: the files packs/en and packs/th pin, from
``conftest.pinned_asset(pack, key)`` (``<PDFINTAKE_TEST_MODELS or ~/.pdfintake-dev/models>/<pack>/<file>``,
checked against the pinned size and SHA-256; ``tools/fetch_test_models.py`` fetches them).
Otherwise those tests skip. Until ocr/scripts.py exists, a
stand-in with its §2.2 interface takes its place.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import random
import re
import subprocess
import sys
import time
import types
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from fixtures import build
from pdfintake.errors import IntegrityError, PackError, SetupError
from pdfintake.types import LayerLine, LayerPage

TESTS = Path(__file__).resolve().parent
# Made-up PDF dates, built from pieces so no long digit run in the file looks like a real identifier.
FIXED_DATE = "D:" + "1998" + "0706" + "050403Z"
OTHER_DATE = "D:" + "1999" + "1231" + "235959Z"
MARKER = {"version": "0.1.0", "LastModified": FIXED_DATE}

# Made-up text only.
THAI_WORDS = ["ข้อความ", "ตัวอย่าง", "ทดสอบ", "น้ำ", "ผู้", "เครื่อง", "กิ่งไม้", "ฤดู", "๑๒๓", "สำหรับ"]
LATIN_WORDS = ["Total", "4521", "sample", "invoice", "45.00", "fűrész", "őrlő", "Straße", "SN-0042", "units"]


# --------------------------------------------------------------------------- the scripts table (U11)


@dataclass(frozen=True)
class _FakeScript:
    name: str
    ranges: tuple[tuple[int, int], ...]
    trigram_ranges: tuple[tuple[int, int], ...]
    spaced: bool
    osd_names: tuple[str, ...]
    required: frozenset[int]


def _fake_scripts_module() -> types.ModuleType:
    """A stand-in for pdfintake.ocr.scripts with the §2.2 interface, used only while U11 is absent."""
    latin_req = frozenset(list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)) + [ord(c) for c in "őűŐŰäöüÄÖÜßé"])
    thai_req = frozenset(list(range(0x0E01, 0x0E3B)) + list(range(0x0E3F, 0x0E5C)))
    scripts = {
        "Latin": _FakeScript("Latin", ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F)), (), True, ("Latin",), latin_req),
        "Thai": _FakeScript("Thai", ((0x0E00, 0x0E7F),), ((0x0E01, 0x0E4F),), False, ("Thai",), thai_req),
        "Han": _FakeScript("Han", ((0x4E00, 0x9FFF),), (), False, ("Han",), frozenset({0x4E00})),
    }

    def script_of(ch: str) -> str | None:
        cp = ord(ch)
        if 0x0E00 <= cp <= 0x0E7F:
            return "Thai"
        if 0x4E00 <= cp <= 0x9FFF:
            return "Han"
        if unicodedata.category(ch).startswith("M"):
            return "Inherited"
        if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A or (0xC0 <= cp <= 0x24F and cp not in (0xD7, 0xF7)):
            return "Latin"
        return None

    mod = types.ModuleType("pdfintake.ocr.scripts")
    mod.SCRIPTS = scripts
    mod.script_of = script_of
    mod.Script = _FakeScript
    return mod


def _real_scripts() -> types.ModuleType | None:
    try:
        from pdfintake.ocr import scripts
    except ImportError:
        return None
    if hasattr(scripts, "SCRIPTS") and hasattr(scripts, "script_of") and "Latin" in scripts.SCRIPTS:
        return scripts
    return None


def install_scripts_if_missing() -> types.ModuleType:
    """The real scripts module, or the stand-in installed in its place (used by the subprocess build too)."""
    real = _real_scripts()
    if real is not None:
        return real
    import pdfintake.ocr

    mod = _fake_scripts_module()
    sys.modules["pdfintake.ocr.scripts"] = mod
    pdfintake.ocr.scripts = mod
    return mod


@pytest.fixture(scope="module", autouse=True)
def scripts_module():
    real = _real_scripts()
    if real is not None:
        yield real
        return
    import pdfintake.ocr

    mp = pytest.MonkeyPatch()
    mod = _fake_scripts_module()
    mp.setitem(sys.modules, "pdfintake.ocr.scripts", mod)
    mp.setattr(pdfintake.ocr, "scripts", mod, raising=False)
    try:
        yield mod
    finally:
        mp.undo()


# --------------------------------------------------------------------------- fonts


THAI_BLOCK = [cp for cp in range(0x0E01, 0x0E5C) if unicodedata.category(chr(cp)) != "Cn"]


def latin_chars(scripts) -> list[int]:
    cps = set(scripts.SCRIPTS["Latin"].required)
    cps |= set(range(0x20, 0x7F))
    cps |= {cp for cp in range(0xA0, 0x180) if unicodedata.category(chr(cp))[0] in "LNPSZ"}
    cps |= {0x0303, 0x2013}
    return sorted(cps)


def thai_chars(scripts) -> list[int]:
    cps = set(scripts.SCRIPTS["Thai"].required) | set(THAI_BLOCK) | {0x20, 0x2D, 0x0303}
    return sorted(cps)


def _spec(tl, path: Path, key: str, covers: tuple[str, ...], priority: int):
    return tl.FontSpec(key, path, hashlib.sha256(path.read_bytes()).hexdigest(), covers, priority)


def box_fontset(tl, folder: Path, scripts, *, upem: int = 1000, latin_advance: int = 560, thai_advance: int = 640):
    lat = build.write_box_font(folder / "ExampleSans-Regular.ttf", latin_chars(scripts), family="Example Sans",
                               advance=latin_advance, units_per_em=upem,
                               advances={"i": latin_advance // 3, "m": latin_advance + 233, "W": latin_advance + 301})
    th = build.write_box_font(folder / "ExampleThai-Regular.ttf", thai_chars(scripts), family="Example Thai",
                              advance=thai_advance, units_per_em=upem, advances={"ำ": thai_advance - 177})
    specs = [_spec(tl, lat, "lat", ("Latin",), 0), _spec(tl, th, "th", ("Thai",), 1)]
    return tl.FontSet.from_specs(specs), specs


@pytest.fixture
def tl():
    from pdfintake.pdf import textlayer

    return textlayer


@pytest.fixture
def fonts(tl, tmp_path, scripts_module):
    return box_fontset(tl, tmp_path, scripts_module)[0]


REAL_FONTS = (("en", "noto-sans"), ("th", "noto-sans-thai"))   # the base pack's font first: better priority


def real_fontset(tl):
    """The pinned launch fonts as a FontSet, or a skip naming what's missing."""
    from conftest import manifest_entry, pinned_assets

    found = pinned_assets(*(f"{pack}:{key}" for pack, key in REAL_FONTS))
    specs = []
    for i, (pack, key) in enumerate(REAL_FONTS):
        pin = manifest_entry(pack, key)
        specs.append(tl.FontSpec(key, found[f"{pack}:{key}"], pin["sha256"], tuple(pin.get("covers", ())), i))
    return tl.FontSet.from_specs(specs)


# --------------------------------------------------------------------------- helpers


def q(x0: float, y0: float, x1: float, y1: float):
    return build.quad(x0, y0, x1, y1)


def mixed_line(rng: random.Random, words: int = 9) -> str:
    return " ".join(rng.choice(THAI_WORDS + LATIN_WORDS) for _ in range(words))


def blank_pdf(n: int = 1, size: tuple[float, float] = (595.0, 842.0)):
    import pikepdf

    pdf = pikepdf.new()
    for _ in range(n):
        pdf.add_blank_page(page_size=size)
    return pdf


def image_pdf(n: int = 1, *, size: tuple[float, float] = (595.0, 842.0), cropbox=None, rotate: int = 0, seed: int = 0):
    """Pages that show a text-like image (Flate, exact samples)."""
    import pikepdf

    specs = [build.PageSpec(width=size[0], height=size[1], rotate=rotate, cropbox=cropbox,
                            images=(build.ImageSpec(build.text_like(600, 800, seed=seed + i)),)) for i in range(n)]
    return pikepdf.open(io.BytesIO(build.pdf_bytes(specs)))


def save(pdf) -> bytes:
    buf = io.BytesIO()
    pdf.save(buf, deterministic_id=True)
    return buf.getvalue()


def render_gray(data: bytes, index: int = 0, scale: float = 1.0) -> bytes:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    try:
        return doc[index].render(scale=scale, grayscale=True).to_pil().tobytes()
    finally:
        doc.close()


def page_text(data: bytes, index: int = 0) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    try:
        page = doc[index]
        tp = page.get_textpage()
        return tp.get_text_bounded()
    finally:
        doc.close()


def squash(s: str) -> str:
    return "".join(s.split())


def overlay_forms(page):
    import pikepdf

    res = page.obj.get("/Resources")
    xo = res.get("/XObject") if res is not None else None
    if not isinstance(xo, pikepdf.Dictionary):
        return {}
    return {str(k): v for k, v in xo.items() if str(k).startswith("/PdfIntakeOCR")}


# --------------------------------------------------------------------------- fonts and runs


def test_synthetic_fonts_cover_latin_and_thai(tl, fonts):
    assert fonts.keys == ("lat", "th")
    assert fonts.font_for("a") == "lat"
    assert fonts.font_for("ő") == "lat"
    assert fonts.font_for("ก") == "th"
    assert fonts.font_for("๑") == "th"
    assert fonts.font_for("1") == "lat"                 # digits go to the Latin font
    assert fonts.font_for("-") == "lat"                 # both fonts have it: priority decides
    assert fonts.font_for("中") is None
    assert fonts.font_for("") is None
    # a cluster: the base decides, so the mark follows it
    assert fonts.font_for("ก\u0303") == "th"
    assert fonts.font_for("a\u0303") == "lat"
    assert fonts.font_for("\u0303") == "lat"            # a lone mark has no base: priority


def test_run_splitting_keeps_marks_with_their_base_and_sends_digits_to_latin(tl, fonts):
    runs = tl.split_runs("ผู้รับ 123 เงิน", fonts)
    assert [(r.font_key, r.text) for r in runs] == [("th", "ผู้รับ "), ("lat", "123 "), ("th", "เงิน")]
    # U+0303 is in both fonts; after a Thai base it must stay in the Thai run, after a Latin one in the Latin run
    runs = tl.split_runs("กก\u0303a\u0303b", fonts)
    assert [(r.font_key, r.text) for r in runs] == [("th", "กก\u0303"), ("lat", "a\u0303b")]
    for text in ["น้ำดื่ม ทดสอบ 4521 ข้อความ", "ผู้ ปู่ ฤดู แม่น้ำ ๑๒๓ SN-0042", "เครื่อง Total 45.00 units"]:
        runs = tl.split_runs(text, fonts)
        assert "".join(r.text for r in runs) == tl.coverage_filter(text, fonts)[0] == text
        for r in runs:
            assert not unicodedata.category(r.text[0]).startswith("M"), f"a run starts with a mark: {r}"
            for ch in r.text:
                if 0x0E00 <= ord(ch) <= 0x0E7F:
                    assert r.font_key == "th"
                elif ch.isdigit() and ord(ch) < 0x80:
                    assert r.font_key == "lat"
        assert all(a.font_key != b.font_key for a, b in zip(runs, runs[1:])), "neighbouring runs share a font"


def test_whitespace_stays_with_the_run_it_follows(tl, fonts):
    runs = tl.split_runs("ทดสอบ  ข้อความ", fonts)
    assert [(r.font_key, r.text) for r in runs] == [("th", "ทดสอบ  ข้อความ")]
    runs = tl.split_runs(" ab", fonts)                    # leading space: the space's own best font
    assert [(r.font_key, r.text) for r in runs] == [("lat", " ab")]


def test_coverage_filter_reports_a_cjk_character(tl, fonts):
    kept, dropped = tl.coverage_filter("Total 中 4521 中文", fonts)
    assert kept == "Total  4521 "
    assert dropped == Counter({"U+4E2D": 2, "U+6587": 1})


def test_coverage_filter_controls_marks_and_zero_width(tl, fonts):
    kept, dropped = tl.coverage_filter("a\tb\nc\x01d", fonts)
    assert kept == "a b cd"                             # whitespace controls become spaces
    assert dropped == Counter({"U+0001": 1})
    kept, dropped = tl.coverage_filter("a\u0e49b", fonts)  # a Thai tone mark on a Latin base: the base's font lacks it
    assert kept == "ab" and dropped == Counter({"U+0E49": 1})
    kept, dropped = tl.coverage_filter("ทด\u200bสอบ", fonts)  # pdfium reads U+200B as a line break
    assert kept == "ทดสอบ" and not dropped
    kept, dropped = tl.coverage_filter("中\u0303x", fonts)   # a dropped base takes its marks with it
    assert kept == "x" and dropped == Counter({"U+4E2D": 1, "U+0303": 1})
    # every default-ignorable character is left out uncounted: ZWNJ, ZWJ, VS1, the bidi isolates ...
    for ch in "\u200c\u200d\ufe00\u2066\u2067\u2068\u2069\u200e\u202a\u00ad\u034f\U000e0100":
        kept, dropped = tl.coverage_filter(f"ab{ch}cd ทด{ch}สอบ", fonts)
        assert kept == "abcd ทดสอบ" and not dropped, hex(ord(ch))
    # a mark repeated in place is left out once, and isn't reported as a character no font has
    kept, dropped = tl.coverage_filter("ก\u0e49\u0e49า", fonts)
    assert kept == "ก\u0e49า" and not dropped
    # whitespace no layer font has becomes a space, so the words stay apart
    for ch in "\u3000\u1680\u2028\u2003":
        kept, dropped = tl.coverage_filter(f"alpha{ch}beta ข้อความ{ch}ทดสอบ", fonts)
        assert kept == "alpha beta ข้อความ ทดสอบ" and not dropped, hex(ord(ch))
    kept, dropped = tl.coverage_filter("a\u00a0b", fonts)          # whitespace a font has is kept as it is
    assert kept == "a\u00a0b" and not dropped


def test_a_font_claiming_thai_without_covering_it_is_refused(tl, tmp_path, scripts_module):
    fake = build.write_box_font(tmp_path / "NotoSansThai-Regular.ttf", latin_chars(scripts_module),
                                family="Noto Sans Thai")
    with pytest.raises(SetupError) as err:
        tl.FontSet.from_specs([_spec(tl, fake, "th", ("Thai",), 0)])
    assert err.value.code == "layer_font_coverage"
    assert "Thai" in str(err.value) and "NotoSansThai-Regular.ttf" in str(err.value)
    # the same font claiming only what it has is fine
    tl.FontSet.from_specs([_spec(tl, fake, "lat", ("Latin",), 0)])


def test_font_checks_name_the_file(tl, tmp_path, scripts_module):
    font = build.write_box_font(tmp_path / "ExampleSans-Regular.ttf", latin_chars(scripts_module))
    good = _spec(tl, font, "lat", ("Latin",), 0)
    with pytest.raises(IntegrityError, match="ExampleSans-Regular.ttf") as err:
        tl.FontSet.from_specs([tl.FontSpec("lat", font, "0" * 64, ("Latin",), 0)])
    assert err.value.code == "layer_font_hash"
    with pytest.raises(IntegrityError, match="missing.ttf"):
        tl.FontSet.from_specs([tl.FontSpec("lat", tmp_path / "missing.ttf", good.sha256, ("Latin",), 0)])
    with pytest.raises(PackError) as err:
        tl.FontSet.from_specs([tl.FontSpec("lat", font, good.sha256, ("Klingon",), 0)])
    assert err.value.code == "unknown_script"
    odd = tmp_path / "font.bin"
    odd.write_bytes(font.read_bytes())
    with pytest.raises(SetupError) as err:
        tl.FontSet.from_specs([tl.FontSpec("lat", odd, good.sha256, ("Latin",), 0)])
    assert err.value.code == "layer_font_format"
    junk = tmp_path / "junk.ttf"
    junk.write_bytes(b"<html>not a font</html>")
    with pytest.raises(SetupError):
        tl.FontSet.from_specs([_spec(tl, junk, "lat", (), 0)])
    with pytest.raises(PackError):
        tl.FontSet.from_specs([])
    with pytest.raises(PackError):
        tl.FontSet.from_specs([good, good])


def test_fpdf2_missing_glyph_warning_raises(tl, fonts, tmp_path):
    font = fonts._by_key["lat"]
    font.cmap[0x4E2D] = "uni4E2D"                      # the writer now believes a glyph exists that fpdf2 lacks
    font.cw[0x4E2D] = 500
    pdf = blank_pdf()
    page = LayerPage(0, 1000, 1400, (LayerLine("ok 中 ok", q(100, 100, 600, 140)),))
    with pytest.raises(SetupError) as err:
        tl.write_layer(pdf, [page], fonts, marker=MARKER)
    assert err.value.code == "layer_missing_glyph"
    assert overlay_forms(pdf.pages[0]) == {}           # nothing reached the document


def test_the_missing_glyph_warning_is_caught_even_when_fpdf_logging_is_quiet(tl, fonts):
    from fpdf import FPDF

    logger = logging.getLogger("fpdf.output")
    old = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with tl._missing_glyphs_raise() as found:
            doc = FPDF(unit="pt")
            doc.add_font("x", fname=str(fonts._by_key["lat"].path))
            doc.add_page()
            doc.set_font("x", size=10)
            doc.text(10, 10, "a中")
            doc.output()
        assert found and "missing" in found[0]
        assert logger.level == logging.ERROR            # restored
    finally:
        logger.setLevel(old)


@pytest.mark.parametrize("upem", [1000, 2048])
def test_hmtx_widths_equal_fpdf2_get_string_width(tl, tmp_path, scripts_module, upem):
    from fpdf import FPDF

    fonts, specs = box_fontset(tl, tmp_path, scripts_module, upem=upem, latin_advance=1187, thai_advance=1301)
    _check_widths(tl, fonts, specs)


def test_hmtx_widths_equal_fpdf2_on_real_fonts(tl):
    fonts = real_fontset(tl)
    specs = [tl.FontSpec(f.key, f.path, f.sha256, tuple(f.covers), f.priority) for f in fonts._fonts]
    _check_widths(tl, fonts, specs)


def _check_widths(tl, fonts, specs) -> None:
    from fpdf import FPDF

    doc = FPDF(unit="pt")
    for s in specs:
        doc.add_font(s.key, fname=str(s.path))
    doc.add_page()
    samples = {"lat": ["Total 4521 units", "fűrész őrlő Straße", "iiiWWWmmm", "SN-0042 45.00"],
               "th": ["ข้อความตัวอย่าง", "น้ำดื่ม ๑๒๓", "ผู้ ปู่ ฤดู", "สำหรับ ทดสอบ"]}
    for s in specs:
        for text in samples.get(s.key, []):
            kept, _ = tl.coverage_filter(text, fonts)
            for size in (7.3, 10.0, 12.37, 31.0):
                doc.set_font(s.key, size=size)
                assert fonts.advance(s.key, kept, size) == pytest.approx(doc.get_string_width(kept), abs=1e-9)


# --------------------------------------------------------------------------- placement


def _ink_boxes(data: bytes, index: int):
    """Every non-generated character's tight and loose box, from pdfium."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(data)
    try:
        page = doc[index]
        tp = page.get_textpage()
        out = []
        for i in range(tp.count_chars()):
            if raw.FPDFText_IsGenerated(tp.raw, i) == 1:
                continue
            out.append((tp.get_charbox(i), tp.get_charbox(i, loose=True)))
        return out
    finally:
        doc.close()


def _random_line(rng: random.Random) -> str:
    alphabet = THAI_WORDS + LATIN_WORDS + ["\u0e49\u0e49", "กั้", "W" * 40, "ก" * 300, "a\u0303", "中文"]
    if rng.random() < 0.05:
        return "".join(rng.choice("ABCDEFGHkmnopq0123ก้ำ") for _ in range(rng.randint(500, 1500)))
    return " ".join(rng.choice(alphabet) for _ in range(rng.randint(1, 14)))


def _random_box(rng: random.Random, w: int, h: int):
    kind = rng.random()
    if kind < 0.15:                                   # partly or wholly off the page
        x0, y0 = rng.uniform(-0.5 * w, 1.4 * w), rng.uniform(-0.5 * h, 1.4 * h)
        return q(x0, y0, x0 + rng.uniform(1, w), y0 + rng.uniform(1, h * 0.5))
    if kind < 0.25:                                   # tiny or degenerate
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        return q(x0, y0, x0 + rng.uniform(0, 3), y0 + rng.uniform(0, 3))
    if kind < 0.32:                                   # taller than the page
        return q(rng.uniform(0, w * 0.5), -h * 0.2, rng.uniform(w * 0.5, w), h * 1.3)
    if kind < 0.4:                                    # corners given in another order
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        x1, y1 = rng.uniform(0, w), rng.uniform(0, h)
        return ((x1, y1), (x0, y1), (x0, y0), (x1, y0))
    x0, y0 = rng.uniform(0, w * 0.9), rng.uniform(0, h * 0.95)
    return q(x0, y0, min(w, x0 + rng.uniform(10, w)), y0 + rng.uniform(8, 60))


@pytest.mark.parametrize("seed", [1, 2, 3, 16, 26, 32])   # 16, 26, 32: huge marks met pdfium's top + top//64
def test_nothing_lands_outside_the_page(tl, fonts, seed):
    """Property test: random boxes, random text, offset CropBoxes and every /Rotate."""
    import pikepdf

    rng = random.Random(seed)
    specs = []
    for i in range(4):
        x0, y0 = rng.uniform(0, 40), rng.uniform(0, 40)
        specs.append(build.PageSpec(width=500 + 40 * i, height=380 + 60 * i, rotate=90 * i,
                                    cropbox=(x0, y0, x0 + 300 + 30 * i, y0 + 250 + 40 * i)))
    pdf = pikepdf.open(io.BytesIO(build.pdf_bytes(specs)))
    pages = []
    for i in range(4):
        w, h = rng.randint(400, 2400), rng.randint(400, 3000)
        pages.append(LayerPage(i, w, h, tuple(LayerLine(_random_line(rng), _random_box(rng, w, h)) for _ in range(60))))
    rep = tl.write_layer(pdf, pages, fonts, marker=MARKER)
    data = save(pdf)
    eps = 0.01
    for i in range(4):
        x0, y0, x1, y1 = tl._page_box(pdf.pages[i].obj)
        boxes = _ink_boxes(data, i)
        assert len(boxes) >= rep.chars_written[i]
        for tight, loose in boxes:
            for left, bottom, right, top in (tight, loose):
                assert x0 - eps <= left <= right <= x1 + eps, (i, tight, loose, (x0, y0, x1, y1))
                assert y0 - eps <= bottom <= top <= y1 + eps, (i, tight, loose, (x0, y0, x1, y1))
        # Random boxes overlap, so pdfium may order the lines differently, and it extracts once a
        # glyph that two lines draw at the same spot (its overprint dedupe; measured: two "A" 0.01 pt
        # apart). Nothing may appear that wasn't written, and almost nothing may go missing.
        got, wrote = Counter(squash(page_text(data, i))), Counter(squash(rep.expected_text[i]))
        assert not (got - wrote)
        assert sum((wrote - got).values()) <= 0.001 * sum(wrote.values())


def test_fit_line_sizes_to_the_box_and_keeps_its_floor(tl, fonts):
    page = (0.0, 0.0, 595.0, 842.0)
    runs = tl.fit_line(LayerLine("sample", q(100, 100, 400, 140)), (0.5, 0.5), page, fonts)
    assert len(runs) == 1 and runs[0].font_key == "lat"
    width = fonts.advance("lat", "sample", 1.0) + 2 * tl._INK_SLACK / 1000     # the box fonts' ink is their advance
    assert runs[0].size == pytest.approx(min(20 * 0.85, 150 / width), abs=0.011)
    assert runs[0].x == pytest.approx(50.0 + tl._INK_SLACK * runs[0].size / 1000, abs=0.01)
    assert runs[0].y == pytest.approx(70.0 - 200 * runs[0].size / 1000, abs=0.01)   # the descent on the box's bottom
    # a box far too narrow: the 1.0 pt floor holds, and the line moves left rather than wrapping
    long = "sample " * 30
    runs = tl.fit_line(LayerLine(long, q(1150, 100, 1170, 140)), (0.5, 0.5), page, fonts)
    assert {r.size for r in runs} == {1.0} and len({r.y for r in runs}) == 1
    right = runs[-1].x + fonts.advance("lat", runs[-1].text.rstrip(), 1.0)
    assert right <= 595.0 - tl.LAYER_PAD + 0.01
    # wider than the page even at 1 pt: wrapped into several rows
    huge = "ก" * 2000
    runs = tl.fit_line(LayerLine(huge, q(10, 10, 20, 30)), (0.5, 0.5), page, fonts)
    assert len({r.y for r in runs}) > 1 and "".join(r.text for r in runs) == huge
    # nothing to write
    assert tl.fit_line(LayerLine("   ", q(10, 10, 200, 30)), (0.5, 0.5), page, fonts) == []
    assert tl.fit_line(LayerLine("中", q(10, 10, 200, 30)), (0.5, 0.5), page, fonts) == []
    assert tl.fit_line(LayerLine("x", q(float("nan"), 10, 200, 30)), (0.5, 0.5), page, fonts) == []


def test_wrapped_rows_never_overprint(tl, fonts):
    """Rows of a wrapped line are stacked apart: pdfium extracts a glyph drawn twice at one spot once."""
    pdf = blank_pdf(1, (300, 400))
    text = "ก" * 3000 + " " + "A" * 2000
    rep = tl.write_layer(pdf, [LayerPage(0, 600, 800, (LayerLine(text, q(500, 700, 520, 720)),))], fonts,
                         marker=MARKER)
    rows = rep.expected_text[0].split("\n")
    assert len(rows) > 5
    data = save(pdf)
    assert squash(page_text(data, 0)) == squash(rep.expected_text[0]) == squash(text)
    for tight, loose in _ink_boxes(data, 0):
        assert 0 <= tight[0] and tight[2] <= 300 and 0 <= tight[1] and tight[3] <= 400


def test_rotated_and_offset_pages_put_the_text_where_the_render_shows_it(tl, fonts):
    """The layer, made visible, inks inside the boxes given in render pixels, for every /Rotate."""
    import numpy as np
    import pikepdf
    import pypdfium2 as pdfium

    for rot in (0, 90, 180, 270):
        pdf = pikepdf.open(io.BytesIO(build.pdf_bytes([build.PageSpec(width=500, height=300, rotate=rot,
                                                                      cropbox=(40, 20, 440, 280))])))
        doc = pdfium.PdfDocument(save(pdf))
        w, h = doc[0].render(scale=2).to_pil().size
        doc.close()
        box = (int(w * 0.2), int(h * 0.3), int(w * 0.7), int(h * 0.42))
        tl.write_layer(pdf, [LayerPage(0, w, h, (LayerLine("HHHH ขขขข", q(*box)),))], fonts, marker=MARKER)
        (form,) = overlay_forms(pdf.pages[0]).values()
        form.write(form.read_bytes().replace(b" 3 Tr", b" 0 Tr"))
        img = np.asarray(pdfium.PdfDocument(save(pdf))[0].render(scale=2).to_pil().convert("L")) < 128
        ys, xs = np.nonzero(img)
        assert len(xs), f"nothing rendered at /Rotate {rot}"
        slack = 3
        assert box[0] - slack <= xs.min() and xs.max() <= box[2] + slack, (rot, box, xs.min(), xs.max())
        assert box[1] - slack <= ys.min() and ys.max() <= box[3] + slack, (rot, box, ys.min(), ys.max())


# --------------------------------------------------------------------------- the document


def test_subset_tags_unique_retention_full_and_no_visible_text(tl, fonts, tmp_path):
    import pikepdf
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    # page 0: a born-digital page made by fpdf2 (its font carries fpdf2's MPDFAA+ tag), page 1-2: scans
    digital = build.fpdf_text_pdf(tmp_path / "digital.pdf", ["sample invoice"], fonts._by_key["lat"].path)
    pdf = pikepdf.open(digital)
    scans = image_pdf(2)
    pdf.pages.extend(scans.pages)
    before = save(pdf)
    rng = random.Random(7)
    pages = [LayerPage(i, 1200, 1700, tuple(LayerLine(mixed_line(rng), q(80, 60 + 40 * k, 1100, 92 + 40 * k))
                                            for k in range(30))) for i in (1, 2)]
    rep = tl.write_layer(pdf, pages, fonts, marker=MARKER)
    data = save(pdf)

    assert len(rep.font_tags) == 2 and len(set(rep.font_tags)) == 2
    assert all(re.fullmatch(r"[A-Z]{6}", t) for t in rep.font_tags) and "MPDFAA" not in rep.font_tags
    out = pikepdf.open(io.BytesIO(data))
    names = []
    for obj in out.objects:
        if isinstance(obj, pikepdf.Dictionary) and obj.get("/Type") == "/FontDescriptor":
            names.append(str(obj.FontName))
    tags = [n[1:7] for n in names]
    assert len(tags) == len(set(tags)) == 3, names
    assert any(n.startswith("/MPDFAA+") for n in names), "the born-digital page's font must be left alone"
    for obj in out.objects:
        if isinstance(obj, pikepdf.Dictionary) and obj.get("/Subtype") == "/Type0" and not str(obj.BaseFont).startswith("/MPDFAA"):
            cid = obj.DescendantFonts[0]
            assert obj.BaseFont == cid.BaseFont == cid.FontDescriptor.FontName
            assert str(obj.BaseFont)[1:7] in rep.font_tags

    for i in (1, 2):
        assert squash(page_text(data, i)) == squash(rep.expected_text[i]), "retention must be 100%"
        assert rep.chars_written[i] == len(squash(rep.expected_text[i])) > 500
        assert render_gray(data, i) == render_gray(before, i), "the layer must not change the render"
    assert squash(page_text(data, 0)) == "sampleinvoice"

    doc = pdfium.PdfDocument(data)
    try:
        for i in (1, 2):
            modes = [raw.FPDFTextObj_GetTextRenderMode(o.raw) for o in doc[i].get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT])]
            assert modes and set(modes) == {raw.FPDF_TEXTRENDERMODE_INVISIBLE}
    finally:
        doc.close()
    for i in (1, 2):
        for form in overlay_forms(out.pages[i]).values():
            content = form.read_bytes()
            assert b" 0 Tr" not in content
            assert content.count(b"BT ") == content.count(b" 3 Tr") + content.count(b"Tf ET")
    assert "/Producer" not in out.docinfo or "PyFPDF" not in str(out.docinfo.get("/Producer", ""))


def test_fixed_name_overlay_and_existing_content_kept(tl, fonts):
    import pikepdf

    pdf = image_pdf(2)
    original = [bytes(pg.obj.Contents.read_raw_bytes()) for pg in pdf.pages]
    original_objgen = [pg.obj.Contents.objgen for pg in pdf.pages]
    pg1 = pdf.pages[1]
    taken = pikepdf.Dictionary({k: pg1.obj.Resources.XObject[k] for k in pg1.obj.Resources.XObject.keys()})
    taken["/PdfIntakeOCR0"] = pg1.obj.Resources.XObject["/Im0"]          # the name is taken on page 1 only
    pg1.obj.Resources.XObject = taken
    pg1.obj.Resources.ExtGState = pikepdf.Dictionary({"/PdfIntakeOCR1": pikepdf.Dictionary(ca=1)})
    pages = [LayerPage(i, 600, 800, (LayerLine("sample 4521", q(50, 50, 400, 90)),)) for i in (0, 1)]
    tl.write_layer(pdf, pages, fonts, marker=MARKER)
    assert list(overlay_forms(pdf.pages[0])) == ["/PdfIntakeOCR0"]
    assert sorted(overlay_forms(pdf.pages[1])) == ["/PdfIntakeOCR0", "/PdfIntakeOCR2"]
    for i, name in ((0, b"/PdfIntakeOCR0"), (1, b"/PdfIntakeOCR2")):
        contents = pdf.pages[i].obj.Contents
        assert isinstance(contents, pikepdf.Array) and len(contents) == 3
        assert contents[0].read_bytes() == b"q " + name + b" Do Q\nq\n"     # the layer first, in a clean state
        assert contents[1].objgen == original_objgen[i]
        assert bytes(contents[1].read_raw_bytes()) == original[i]          # untouched, byte for byte
        assert contents[2].read_bytes() == b"\nQ\n"
        form = pdf.pages[i].obj.Resources.XObject[name.decode()]
        assert form.Subtype == pikepdf.Name.Form and list(form.Matrix) == [1, 0, 0, 1, 0, 0]


def test_unbalanced_q_in_the_page_is_closed_before_the_layer(tl, fonts):
    """A page that leaves q open with a scale must not shrink the layer."""
    pdf = blank_pdf(1, (600, 800))
    data = b"q 0.5 0 0 0.5 0 0 cm q 1 0 0 1 3 3 cm"
    pdf.pages[0].obj.Contents = pdf.make_stream(data)
    rep = tl.write_layer(pdf, [LayerPage(0, 600, 800, (LayerLine("sample", q(300, 400, 500, 440)),))], fonts,
                         marker=MARKER)
    assert pdf.pages[0].obj.Contents[2].read_bytes() == b"\nQ\nQ\nQ\n"
    out = save(pdf)
    boxes = _ink_boxes(out, 0)
    assert boxes and all(290 <= t[0] and t[2] <= 510 for t, _ in boxes), boxes
    assert squash(page_text(out, 0)) == squash(rep.expected_text[0]) == "sample"


def test_shared_and_inherited_resources_change_only_the_layered_page(tl, fonts):
    import pikepdf

    pdf = image_pdf(1)
    shared = pdf.make_indirect(pdf.pages[0].obj.Resources)
    pdf.pages[0].obj.Resources = shared
    pdf.pages.append(pikepdf.Page(pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, 595, 842],
                                                     Resources=shared, Contents=pdf.pages[0].obj.Contents)))
    inherit = pikepdf.Page(pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, 595, 842],
                                              Contents=pdf.pages[0].obj.Contents))
    pdf.pages.append(inherit)
    assert "/Resources" not in pdf.pages[2].obj
    pdf.Root.Pages.Resources = shared                          # page 2 inherits its resources
    before = save(pdf)
    pages = [LayerPage(i, 600, 800, (LayerLine("sample", q(50, 50, 400, 90)),)) for i in (0, 2)]
    tl.write_layer(pdf, pages, fonts, marker=MARKER)
    after = save(pdf)
    assert "/PdfIntakeOCR0" in pdf.pages[0].obj.Resources.XObject
    assert "/PdfIntakeOCR0" in pdf.pages[2].obj.Resources.XObject
    assert "/PdfIntakeOCR0" not in shared.XObject, "a shared resource dictionary must not change"
    assert "/PdfIntakeOCR0" not in pdf.pages[1].obj.Resources.XObject
    for i in range(3):
        assert render_gray(after, i) == render_gray(before, i)
    assert squash(page_text(after, 1)) == ""
    assert squash(page_text(after, 2)) == "sample"


def test_the_piece_info_marker(tl, fonts):
    import pikepdf

    pdf = image_pdf(2)
    pdf.pages[0].obj.PieceInfo = pikepdf.Dictionary({"/ExampleApp": pikepdf.Dictionary(
        LastModified=pikepdf.String(OTHER_DATE), Private=pikepdf.Name.Kept)})
    pages = [LayerPage(0, 600, 800, (LayerLine("sample", q(50, 50, 400, 90)),)),
             LayerPage(1, 600, 800, (LayerLine("中中", q(50, 50, 400, 90)),))]       # nothing writable
    rep = tl.write_layer(pdf, pages, fonts, marker={"version": "0.1.0", "pack": "th", "LastModified": FIXED_DATE})
    info = pdf.pages[0].obj.PieceInfo
    assert info["/ExampleApp"].Private == pikepdf.Name.Kept
    mine = info[tl.MARKER_KEY]
    assert str(mine.LastModified) == FIXED_DATE and str(pdf.pages[0].obj.LastModified) == FIXED_DATE
    assert {str(k): str(v) for k, v in mine.Private.items()} == {"/pack": "th", "/version": "0.1.0"}
    # a page with nothing written gets neither a form nor a marker
    assert "/PieceInfo" not in pdf.pages[1].obj and overlay_forms(pdf.pages[1]) == {}
    assert rep.chars_written[1] == 0 and rep.expected_text[1] == "" and rep.dropped["U+4E2D"] == 2
    # without a date the marker gets the fixed default, never the clock
    pdf2 = image_pdf(1)
    tl.write_layer(pdf2, [pages[0]], fonts, marker={})
    assert str(pdf2.pages[0].obj.PieceInfo[tl.MARKER_KEY].LastModified) == tl.DEFAULT_MARKER_DATE
    assert str(pdf2.pages[0].obj.LastModified) == tl.DEFAULT_MARKER_DATE
    assert re.fullmatch(r"D:\d{14}Z", tl.DEFAULT_MARKER_DATE)
    for bad in ({"bad key": "x"}, {"k": "x" * 201}, {"k": "ö"}, {"LastModified": "yesterday"}, {"k": 3}):
        with pytest.raises(ValueError):
            tl.write_layer(image_pdf(1), [pages[0]], fonts, marker=bad)


def test_report_and_page_checks(tl, fonts):
    pdf = image_pdf(3)
    rng = random.Random(3)
    lines = tuple(LayerLine(mixed_line(rng, 5), q(40, 40 + 35 * k, 560, 70 + 35 * k)) for k in range(12))
    pages = [LayerPage(0, 600, 800, lines), LayerPage(2, 600, 800, ()),
             LayerPage(1, 600, 800, (LayerLine("Total 中 4521", q(40, 40, 400, 70)),))]
    rep = tl.write_layer(pdf, pages, fonts, marker=MARKER)
    assert set(rep.chars_written) == set(rep.expected_text) == {0, 1, 2}
    assert rep.expected_text[1] == "Total  4521" and rep.chars_written[1] == 9
    assert rep.expected_text[2] == "" and overlay_forms(pdf.pages[2]) == {}
    assert len(rep.expected_text[0].split("\n")) == 12
    assert rep.dropped == Counter({"U+4E2D": 1})
    assert rep.runs >= 13
    for bad in ([LayerPage(3, 600, 800, ())], [LayerPage(0, 600, 800, ()), LayerPage(0, 600, 800, ())],
                [LayerPage(0, 0, 800, ())]):
        with pytest.raises(ValueError):
            tl.write_layer(image_pdf(3), bad, fonts, marker=MARKER)
    empty = image_pdf(1)
    rep = tl.write_layer(empty, [], fonts, marker=MARKER)
    assert rep.runs == 0 and rep.font_tags == [] and overlay_forms(empty.pages[0]) == {}


def test_retag_leaves_fonts_outside_our_forms_alone(tl, fonts, tmp_path):
    import pikepdf

    digital = build.fpdf_text_pdf(tmp_path / "digital.pdf", ["sample"], fonts._by_key["lat"].path)
    pdf = pikepdf.open(digital)
    assert tl.retag_subsets(pdf) == []
    names = [str(o.BaseFont) for o in pdf.objects if isinstance(o, pikepdf.Dictionary) and o.get("/Subtype") == "/Type0"]
    assert names and all(n.startswith("/MPDFAA+") for n in names)


def test_subset_tag_is_derived_and_avoids_collisions(tl):
    a = tl._subset_tag(b"font bytes", set())
    assert re.fullmatch(r"[A-Z]{6}", a) and tl._subset_tag(b"font bytes", set()) == a
    b = tl._subset_tag(b"font bytes", {a})
    assert b != a and re.fullmatch(r"[A-Z]{6}", b)


# --------------------------------------------------------------------------- budgets and reproducibility


def _thirty_by_seventy():
    rng = random.Random(11)
    w, h = 1654, 2339
    return [LayerPage(i, w, h, tuple(LayerLine(mixed_line(rng), q(100, 30 + 32 * k, 1550, 56 + 32 * k))
                                     for k in range(70))) for i in range(30)]


def _budget(tl, fonts) -> float:
    import fpdf  # noqa: F401  (imported once, as in a real run, before the clock starts)

    pages = _thirty_by_seventy()
    pdf = blank_pdf(30, (595.44, 842.04))
    start = time.perf_counter()
    rep = tl.write_layer(pdf, pages, fonts, marker=MARKER)
    seconds = time.perf_counter() - start
    assert rep.runs > 2100 and sum(rep.chars_written.values()) > 100_000
    return seconds


def test_performance_budget_thirty_pages_of_seventy_lines(tl, fonts):
    seconds = _budget(tl, fonts)
    print(f"write_layer, 30 pages x 70 lines, box fonts: {seconds:.3f} s")
    assert seconds < 1.0


def test_performance_budget_on_real_fonts(tl):
    fonts = real_fontset(tl)
    seconds = _budget(tl, fonts)
    print(f"write_layer, 30 pages x 70 lines, real fonts: {seconds:.3f} s")
    assert seconds < 1.0


def hash_build(folder: str) -> str:
    """Build a two-page layered PDF from scratch and return its sha256 (run in a subprocess)."""
    scripts = install_scripts_if_missing()
    from pdfintake.pdf import textlayer as tl

    path = Path(folder)
    fonts, _ = box_fontset(tl, path, scripts)
    pdf = image_pdf(2)
    rng = random.Random(5)
    pages = [LayerPage(i, 1200, 1700, tuple(LayerLine(mixed_line(rng), q(80, 60 + 40 * k, 1100, 92 + 40 * k))
                                            for k in range(25))) for i in range(2)]
    tl.write_layer(pdf, pages, fonts, marker=MARKER)
    return hashlib.sha256(save(pdf)).hexdigest()


def test_reproducible_bytes_across_hash_seeds(tmp_path):
    digests = []
    for seed in ("1", "2"):
        folder = tmp_path / f"seed{seed}"
        folder.mkdir()
        env = dict(os.environ)
        env.update(PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import test_textlayer as t; "
                "print('@@' + t.hash_build(sys.argv[2]))")
        proc = subprocess.run([sys.executable, "-B", "-c", code, str(TESTS), str(folder)], capture_output=True,
                              text=True, encoding="utf-8", env=env, timeout=300, cwd=str(tmp_path))
        marks = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
        assert marks, f"the build failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
        digests.append(marks[-1][2:])
    assert digests[0] == digests[1], digests
    # and the same bytes in this process
    assert hash_build(str(tmp_path)) == digests[0]


# --------------------------------------------------------------------------- review fixes


def test_default_marker_date_gives_the_same_bytes_twice(tl, fonts):
    """Without a LastModified key the build must not depend on the clock."""
    digests = []
    for _ in range(2):
        pdf = image_pdf(1)
        tl.write_layer(pdf, [LayerPage(0, 600, 800, (LayerLine("sample 4521", q(50, 50, 400, 90)),))], fonts,
                       marker={})
        digests.append(hashlib.sha256(save(pdf)).hexdigest())
        time.sleep(1.05)                               # a whole second apart, as the clock would differ
    assert digests[0] == digests[1]


def test_default_ignorable_characters_extract_as_written(tl, fonts):
    """ZWNJ, ZWJ, VS1 and the bidi isolates are left out, so pdfium gives back exactly what was written."""
    pdf = image_pdf(1)
    text = "x\u200cy sample a\u200db \u2066Total\u2069 4521\ufe00 ทด\u200dสอบ"
    rep = tl.write_layer(pdf, [LayerPage(0, 600, 800, (LayerLine(text, q(40, 40, 560, 80)),))], fonts, marker=MARKER)
    assert rep.expected_text[0] == "xy sample ab Total 4521 ทดสอบ" and not rep.dropped
    got = page_text(save(pdf), 0)
    assert squash(got) == squash(rep.expected_text[0])
    assert "\n" not in got.strip() and "\r" not in got.strip()          # no word was split


def test_a_line_that_cannot_be_placed_is_counted_as_dropped(tl, fonts):
    nan = float("nan")
    lines = (LayerLine("Total 4521", ((nan, 10.0), (200.0, 10.0), (200.0, 30.0), (10.0, 30.0))),
             LayerLine("ab", ((10.0, 10.0), (200.0, 10.0), (200.0, float("inf")), (10.0, 30.0))),
             LayerLine("ok", q(10, 40, 200, 70)))
    pdf = image_pdf(1)
    rep = tl.write_layer(pdf, [LayerPage(0, 600, 800, lines)], fonts, marker=MARKER)
    assert rep.expected_text[0] == "ok"
    assert rep.dropped == Counter({"U+0054": 1, "U+006F": 1, "U+0074": 1, "U+0061": 2, "U+006C": 1,
                                   "U+0034": 1, "U+0035": 1, "U+0032": 1, "U+0031": 1, "U+0062": 1})
    tiny = blank_pdf(1, (3.0, 3.0))
    tiny.pages[0].obj.MediaBox = [0, 0, 1, 1]          # no room inside the 0.5 pt padding
    rep = tl.write_layer(tiny, [LayerPage(0, 10, 10, (LayerLine("ab", q(1, 1, 9, 9)),))], fonts, marker=MARKER)
    assert rep.expected_text[0] == "" and rep.dropped == Counter({"U+0061": 1, "U+0062": 1})


def test_retag_never_touches_an_input_font_in_a_form_with_our_name(tl, fonts, tmp_path):
    """An fpdf2-made input whose own form happens to be called /PdfIntakeOCR0 keeps its font's tag."""
    import pikepdf

    digital = build.fpdf_text_pdf(tmp_path / "digital.pdf", ["sample invoice"], fonts._by_key["lat"].path)
    pdf = pikepdf.open(digital)
    page0 = pdf.pages[0]
    form = pikepdf.Stream(pdf, page0.obj.Contents.read_bytes())
    form.Type, form.Subtype = pikepdf.Name.XObject, pikepdf.Name.Form
    form.BBox = page0.obj.MediaBox
    form.Resources = pikepdf.Dictionary(Font=page0.obj.Resources.Font)
    page0.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary({"/PdfIntakeOCR0": pdf.make_indirect(form)}))
    page0.obj.Contents = pdf.make_stream(b"q /PdfIntakeOCR0 Do Q")
    assert squash(page_text(save(pdf), 0)) == "sampleinvoice"
    pdf.pages.extend(image_pdf(1).pages)
    rep = tl.write_layer(pdf, [LayerPage(1, 600, 800, (LayerLine("Total 4521", q(50, 50, 400, 90)),))], fonts,
                         marker=MARKER)
    assert len(rep.font_tags) == 1
    out = pikepdf.open(io.BytesIO(save(pdf)))
    names = sorted(str(o.FontName) for o in out.objects
                   if isinstance(o, pikepdf.Dictionary) and o.get("/Type") == "/FontDescriptor")
    assert len(names) == 2
    assert sum(n.startswith("/MPDFAA+") for n in names) == 1, names           # the input's font, untouched
    assert sum(n.startswith(f"/{rep.font_tags[0]}+") for n in names) == 1
    assert squash(page_text(save(pdf), 0)) == "sampleinvoice"


@pytest.mark.parametrize("tail", [b" BT /F9 12 Tf (abc", b" BI /W 2 /H 2 /BPC 8 /CS /G ID \x00\x01\x02\x03"])
def test_the_layer_survives_a_page_whose_content_ends_mid_token(tl, fonts, tail):
    """A damaged page that ends inside an open string or an inline image without EI still gets its layer."""
    pdf = blank_pdf(1, (600, 800))
    pdf.pages[0].obj.Contents = pdf.make_stream(b"0 0 1 rg 10 10 50 50 re f" + tail)
    rep = tl.write_layer(pdf, [LayerPage(0, 600, 800, (LayerLine("Total 4521", q(100, 100, 400, 140)),))], fonts,
                         marker=MARKER)
    assert squash(page_text(save(pdf), 0)) == squash(rep.expected_text[0]) == "Total4521"


def _ink_table(font_set) -> dict:
    return {(f.key, cp): f.ink(cp) for f in font_set._fonts for cp in sorted(f.cmap)}


@pytest.mark.parametrize("which", ["box", "real"])
def test_glyph_bounds_are_the_same_from_many_threads(tl, tmp_path, scripts_module, which):
    """The lazy glyf table is read under a lock: twelve threads on one fresh FontSet agree with a serial run."""
    import threading

    def make():
        if which == "real":
            return real_fontset(tl)
        return box_fontset(tl, tmp_path, scripts_module)[0]

    serial = _ink_table(make())
    for _ in range(12 if which == "real" else 3):      # the unlocked race showed in about half of 3 trials
        shared = make()
        results, errors = [], []

        def work(seed):
            try:
                rng = random.Random(seed)
                items = [(f, cp) for f in shared._fonts for cp in sorted(f.cmap)]
                rng.shuffle(items)
                for f, cp in items:
                    f.ink(cp)
                results.append(_ink_table(shared))
            except Exception as exc:                  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(s,)) for s in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and len(results) == 12
        assert all(r == serial for r in results)


def test_glyph_bounds_let_an_unexpected_error_through(tl, fonts):
    font = fonts._by_key["lat"]

    class Broken:
        def __contains__(self, key):
            raise RuntimeError("not a measurement problem")

        def __getitem__(self, key):
            return font._ttfont[key]

    font._ttfont = Broken()
    font._bounds.clear()
    with pytest.raises(RuntimeError):
        font.ink(ord("a"))


def _swap(path: Path, how: str, other: Path) -> None:
    import shutil

    if how == "truncated":
        path.write_bytes(path.read_bytes()[:2000])
    elif how == "deleted":
        path.unlink()
    elif how == "directory":
        path.unlink()
        path.mkdir()
    elif how == "other_font":
        shutil.copyfile(other, path)


@pytest.mark.parametrize("how,code", [("truncated", "layer_font_hash"), ("deleted", "layer_font_unreadable"),
                                      ("directory", "layer_font_unreadable"), ("other_font", "layer_font_hash")])
def test_a_font_changed_after_verification_is_named_as_that(tl, tmp_path, scripts_module, how, code):
    font_set, specs = box_fontset(tl, tmp_path, scripts_module)
    other = build.write_box_font(tmp_path / "Other.ttf", latin_chars(scripts_module), advance=611)
    _swap(Path(specs[0].path), how, other)
    pdf = image_pdf(1)
    with pytest.raises(IntegrityError) as err:
        tl.write_layer(pdf, [LayerPage(0, 600, 800, (LayerLine("sample", q(50, 50, 400, 90)),))], font_set,
                       marker=MARKER)
    assert err.value.code == code
    assert overlay_forms(pdf.pages[0]) == {}


def test_font_spec_paths_are_resolved_and_fields_checked(tl, tmp_path, scripts_module, monkeypatch):
    font = build.write_box_font(tmp_path / "ExampleSans-Regular.ttf", latin_chars(scripts_module))
    good = _spec(tl, font, "lat", ("Latin",), 0)
    monkeypatch.chdir(tmp_path)
    font_set = tl.FontSet.from_specs([tl.FontSpec("lat", Path("ExampleSans-Regular.ttf"), good.sha256, ("Latin",), 0)])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rep = tl.write_layer(image_pdf(1), [LayerPage(0, 600, 800, (LayerLine("sample", q(50, 50, 400, 90)),))], font_set,
                         marker=MARKER)
    assert rep.expected_text[0] == "sample"
    for bad in (tl.FontSpec("lat", font, None, ("Latin",), 0), tl.FontSpec("lat", font, "abc", ("Latin",), 0),
                tl.FontSpec("lat", font, good.sha256, "Latin", 0), tl.FontSpec("lat", font, good.sha256, ("Latin",), "0")):
        with pytest.raises(PackError) as err:
            tl.FontSet.from_specs([bad])
        assert err.value.code == "layer_font_spec"


def test_text_larger_than_the_page_shrinks_rather_than_overprinting(tl, fonts):
    """Rows that would be taller than the page re-wrap at a smaller size; every character comes back."""
    pdf = blank_pdf(1, (100, 100))
    rng = random.Random(4)
    text = "".join(rng.choice("ABCDEFGHkmnopq0123กขคง") for _ in range(40000))
    start = time.perf_counter()
    rep = tl.write_layer(pdf, [LayerPage(0, 200, 200, (LayerLine(text, q(10, 10, 30, 30)),))], fonts, marker=MARKER)
    assert time.perf_counter() - start < 5.0
    rows = rep.expected_text[0].split("\n")
    assert len(rows) > 10 and not rep.dropped
    data = save(pdf)
    assert Counter(squash(page_text(data, 0))) == Counter(squash(text))
    for tight, _ in _ink_boxes(data, 0):
        assert 0 <= tight[0] and tight[2] <= 100 and 0 <= tight[1] and tight[3] <= 100
