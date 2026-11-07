"""kit/render.py: bench pages and canaries, rendered deterministically from the corpus.

This file also holds the helpers the other kit tests import: box fonts built with fontTools, a fake
web (``FakeWeb``) that answers the kit's downloads with made-up bytes, and ``fake_pack``, a base pack
in ``packs/en`` read by fake engines registered for the test. Every text is made up.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from fixtures import build
from pdfintake import platforms
from pdfintake.errors import PackError, PdfIntakeError, SetupError
from pdfintake.kit import corpus as C
from pdfintake.kit import render as R
from pdfintake.ocr import engines, scripts

pytest.importorskip("uharfbuzz")

TEXT = ("The example invoice lists seven sample items and a total of four thousand units for the test "
        "customer account. The company will send the order to the address given on the form before Friday, "
        "as agreed on the phone last week, and the invoice is attached for your records. ") * 12
WIKITEXT = ("'''Example''' is a [[sample|made-up]] article.<ref>A note</ref> {{Infobox|x=1}}\n"
            "== History ==\n* " + TEXT + "\n[[Category:Examples]]\n")
FONT_URL = "https://github.com/notofonts/latin-greek-cyrillic/releases/download/example-v1/ExampleSans-Regular.ttf"
FONT2_URL = "https://github.com/notofonts/latin-greek-cyrillic/releases/download/example-v1/ExampleSerif-Regular.ttf"
THAI_FONT_URL = "https://github.com/notofonts/thai/releases/download/example-v1/ExampleThai-Regular.ttf"


# --------------------------------------------------------------------------- helpers for every kit test

def latin_chars() -> str:
    cps = set(range(0x20, 0x7F)) | set(scripts.SCRIPTS["Latin"].required)
    return "".join(chr(c) for c in sorted(cps))


def latin_font() -> bytes:
    return build.box_font(latin_chars(), family="Example Sans")


def latin_font2() -> bytes:
    """A second Latin font: the bench needs two per script (DESIGN §3.6)."""
    return build.box_font(latin_chars(), family="Example Serif")


def thai_font() -> bytes:
    cps = set(scripts.SCRIPTS["Thai"].required) | {0x20}
    return build.box_font("".join(chr(c) for c in sorted(cps)), family="Example Thai")


class FakeWeb:
    """Answers the kit's downloads (Wikidata, Wikipedia, fonts, tessdata) with made-up bytes."""

    def __init__(self, files: dict[str, bytes] | None = None, *, wikitext: str = WIKITEXT, revid: int = 4242) -> None:
        self.files = dict(files or {})
        self.wikitext = wikitext
        self.revid = revid
        self.calls: list[str] = []

    def __call__(self, url: str, *, cap: int) -> bytes:
        self.calls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        if "wikidata.org" in url:
            item = q["ids"][0]
            site = q["sitefilter"][0]
            return json.dumps({"entities": {item: {"sitelinks": {site: {"title": f"Example {item}"}}}}}).encode()
        if "/w/api.php" in url:
            n = int("".join(ch for ch in q["titles"][0] if ch.isdigit()) or 0)   # one revision per item
            return json.dumps({"query": {"pages": [{"revisions": [{"revid": self.revid + n - 1}]}]}}).encode()
        if "action=raw" in url:
            return self.wikitext.encode("utf-8")
        if url in self.files:
            return self.files[url]
        raise PackError("kit_download", f"couldn't download {url}: HTTP Error 404", url=url, status=404)


def all_platforms() -> frozenset[str]:
    return frozenset(p.key for p in platforms.PLATFORMS)


def register_fake(monkeypatch: pytest.MonkeyPatch, name: str = "fake", texts: list[str] | None = None) -> Any:
    eng = build.FakeEngine(languages=("en",), texts=texts, platforms=all_platforms())
    if name != "fake":
        eng.name = name
    monkeypatch.setitem(engines._REGISTRY, name, eng)
    return eng


def fake_manifest(font: bytes, *, engine: str = "fake", must: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": 1, "id": "en", "version": "1.0.0", "requires_core": ">=0.1.0", "base": True, "title": "English",
        "languages": [{"tag": "en", "iso639_3": "eng", "role": "primary"}], "scripts": ["Latin"],
        "detect": {"osd_scripts": ["Latin"], "langid": {"codes": ["en"]}},
        "readers": {"default": {"engines": [{"id": engine, "engine": engine, "role": "reader",
                                             "params": {"lang": "en"}, "models": []}],
                                "pick": {"rule": "single", "version": 1, "params": {"reader": engine}},
                                "fallback": "none"}},
        "normalize": ["nfc"], "repairs": [], "fills": [], "models": [],
        "fonts": [{"key": "example-sans", "file": "ExampleSans-Regular.ttf", "covers": ["Latin"], "bytes": len(font),
                   "sha256": hashlib.sha256(font).hexdigest(), "source": FONT_URL, "licence": "OFL-1.1"}],
        "must_survive": must if must is not None else [{"name": "accents", "chars": "éá", "unit": "word", "floor": 0.0}],
        "bench": {"kit": "1.0", "corpus": "none", "platforms": {}},
    }


def fake_pack(tmp: Path, monkeypatch: pytest.MonkeyPatch, *, texts: list[str] | None = None,
              second: bool = False) -> tuple[Path, Path, FakeWeb]:
    """``packs/en`` read by fake engines, its font already in the work folder; (packs, work, web)."""
    register_fake(monkeypatch, "fake", texts)
    if second:
        register_fake(monkeypatch, "fakeb", texts)
    font = latin_font()
    packs, work = tmp / "packs", tmp / "work"
    (packs / "en").mkdir(parents=True)
    (packs / "en" / "manifest.json").write_text(json.dumps(fake_manifest(font), indent=2), encoding="utf-8")
    (packs / "en" / "OFL.txt").write_text("Example licence text for the test font.\n", encoding="utf-8")
    files = work / "en" / "files"
    files.mkdir(parents=True)
    (files / "ExampleSans-Regular.ttf").write_bytes(font)
    web = FakeWeb({FONT_URL: font, FONT2_URL: latin_font2(), THAI_FONT_URL: thai_font()})
    monkeypatch.setattr(C, "_http_get", web)
    return packs, work, web


def corpus_for(text: str = TEXT, *, pages: dict[str, int] | None = None, fonts: tuple = ()) -> C.Corpus:
    raw = text.encode("utf-8")
    t = C.CorpusText("url", "en", "Example", 0, "https://www.gutenberg.org/cache/epub/0/example.txt", hashlib.sha256(raw).hexdigest(),
                     "CC0-1.0")
    return C.Corpus("test-v1", (t,), fonts, pages or {"prose": 1, "form": 1, "microprint": 1, "mixed_script": 1})


def font_files(tmp: Path, *, thai: bool = False) -> list[Path]:
    out = [build.write_box_font(tmp / "ExampleSans-Regular.ttf", latin_chars(), family="Example Sans")]
    if thai:
        p = tmp / "ExampleThai-Regular.ttf"
        p.write_bytes(thai_font())
        out.append(p)
    return out


# --------------------------------------------------------------------------- the tests

def _render(tmp: Path, name: str, *, seed: int = 0, thai: bool = False) -> list[R.BenchPage]:
    c = corpus_for()
    (tmp / f"fonts-{name}").mkdir()
    fonts = font_files(tmp / f"fonts-{name}", thai=thai)
    return R.render_pages(c, fonts, tmp / name, seed=seed, texts={c.texts[0].sha256: TEXT})


def test_render_is_deterministic_for_a_seed(tmp_path: Path):
    a = _render(tmp_path, "a")
    b = _render(tmp_path, "b")
    assert [p.id for p in a] == [p.id for p in b]
    assert [p.truth for p in a] == [p.truth for p in b]
    assert [p.png.read_bytes() for p in a] == [p.png.read_bytes() for p in b]
    c = _render(tmp_path, "c", seed=7)
    assert [p.truth for p in c] != [p.truth for p in a]


def test_pages_cover_every_kind_and_condition(tmp_path: Path):
    pages = _render(tmp_path, "a", thai=True)
    kinds = {p.kind for p in pages}
    assert kinds == {"prose", "form", "microprint", "mixed_script"}
    for p in pages:
        assert p.dpi == R.CONDITIONS[p.cond]
        assert p.truth.strip()
        with Image.open(p.png) as im:
            assert im.mode == "L"
            assert abs(im.info["dpi"][0] - p.dpi) < 0.01
    clean = next(p for p in pages if p.id == "prose-0-clean")
    small = next(p for p in pages if p.id == "prose-0-small")
    with Image.open(clean.png) as a, Image.open(small.png) as b:
        assert a.size[0] > b.size[0] * 2
    form = next(p for p in pages if p.kind == "form")
    assert any(ch.isdigit() for ch in form.truth)
    mixed = next(p for p in pages if p.kind == "mixed_script")
    assert any(scripts.script_of(ch) == "Thai" for ch in mixed.truth)


def test_no_mixed_page_without_a_font_for_a_second_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(R, "MIXED_SAMPLES", {"Thai": R.MIXED_SAMPLES["Thai"]})
    pages = _render(tmp_path, "a", thai=False)
    assert not [p for p in pages if p.kind == "mixed_script"]


def test_load_pages_round_trips(tmp_path: Path):
    pages = _render(tmp_path, "a")
    again = R.load_pages(tmp_path / "a")
    assert [(p.id, p.truth, p.dpi, p.png) for p in pages] == [(p.id, p.truth, p.dpi, p.png) for p in again]
    with pytest.raises(PackError, match="kit render"):
        R.load_pages(tmp_path / "nothing")


def test_degrade_is_seeded():
    img = build.text_like(400, 300).convert("L")
    a, b, c = R.degrade(img, 1), R.degrade(img, 1), R.degrade(img, 2)
    assert a.mode == "L" and a.size == img.size
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


def test_canary_renders_to_the_same_bytes(tmp_path: Path):
    fonts = font_files(tmp_path, thai=True)
    text = "ตัวอย่าง example 4521"
    a = R.render_canary(text, fonts, tmp_path / "a.png")
    b = R.render_canary(text, list(fonts), tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()
    with Image.open(a) as im:
        assert im.mode == "L" and round(im.info["dpi"][0]) == 300
    with pytest.raises(PackError, match="no font covers"):
        R.render_canary("中文", fonts, tmp_path / "c.png")


# --------------------------------------------------------------------------- shaping

THAI_LINE = "ที่ เมื่อ น้ำ ญี่ปุ่น"
UPPER_MARKS = (0x0E31, 0x0E34, 0x0E35, 0x0E36, 0x0E37, 0x0E47, 0x0E4D)   # above-base vowels and nikhahit
TONE_MARKS = (0x0E48, 0x0E49, 0x0E4A, 0x0E4B)


def shaping_font(tmp: Path) -> Path:
    """A box font (Latin and Thai) whose GPOS does what a real Thai font does to a tone mark: one
    that follows an upper vowel is lifted above it (+300 units), where unshaped text leaves it on the
    vowel. It also kerns "AV" (-250), so Latin runs need shaping too."""
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
    from fontTools.ttLib import TTFont

    chars = latin_chars() + "".join(chr(c) for c in sorted(scripts.SCRIPTS["Thai"].required))
    tt = TTFont(io.BytesIO(build.box_font(chars, family="Example Shaped")))
    upper = " ".join(build._glyph_name(c) for c in UPPER_MARKS)
    tone = " ".join(build._glyph_name(c) for c in TONE_MARKS)
    addOpenTypeFeaturesFromString(tt, f"""
languagesystem DFLT dflt;
languagesystem thai dflt;
languagesystem latn dflt;
@UPPER = [{upper}];
@TONE = [{tone}];
lookup LIFT {{ pos @TONE <0 300 0 0>; }} LIFT;
feature mark {{ pos @UPPER @TONE' lookup LIFT; }} mark;
feature kern {{ pos uni0041 uni0056 -250; }} kern;
""")
    path = tmp / "ExampleShaped-Regular.ttf"
    tt.save(str(path))
    return path


def _reference_line(font: Path, text: str, size: float, x: float, y: float, page: tuple[float, float],
                    dest: Path) -> Path:
    """``text`` drawn where HarfBuzz itself puts each glyph: shaped with uharfbuzz directly (not through
    fpdf2), then each glyph placed on its own with ``FPDF.text`` and rendered like a kit page."""
    import uharfbuzz as hb
    from fontTools.ttLib import TTFont
    from fpdf import FPDF

    face = hb.Face(font.read_bytes())
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(face), buf, {})
    tt = TTFont(str(font))
    char_of = {tt.getGlyphID(name): chr(cp) for cp, name in tt.getBestCmap().items()}
    pdf = FPDF(unit="pt", format=page)
    pdf.set_margin(0)
    pdf.set_auto_page_break(False)
    pdf.add_page()
    pdf.add_font("Reference", fname=str(font))
    pdf.set_font("Reference", size=size)
    scale = size / face.upem
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        pdf.text(x + pos.x_offset * scale, y - pos.y_offset * scale, char_of[info.codepoint])
        x += pos.x_advance * scale
    pdf_path = dest.with_suffix(".pdf")
    pdf_path.write_bytes(bytes(pdf.output()))
    R._render_pdf(pdf_path, 300.0, dest)
    return dest


def _kit_line(font: Path, text: str, size: float, x: float, y: float, page: tuple[float, float], dest: Path, *,
              unshaped: bool = False) -> Path:
    """``text`` drawn by the kit's own line drawing (or, ``unshaped``, by a single ``FPDF.text``)."""
    fonts = R._load_fonts([font])
    pdf = R._new_pdf(fonts)
    pdf.add_page(format=page)
    if unshaped:
        pdf.set_font(fonts[0].family, size=size)
        pdf.text(x, y, text)
    else:
        R._draw_line(pdf, x, y, text, fonts, fonts[0], size)
    pdf_path = dest.with_suffix(".pdf")
    pdf_path.write_bytes(bytes(pdf.output()))
    R._render_pdf(pdf_path, 300.0, dest)
    return dest


def _ink_mismatch(got: Path, want: Path) -> float:
    """Pixels inked in one image and not the other, as a share of ``want``'s ink."""
    import numpy as np

    with Image.open(got) as a, Image.open(want) as b:
        x, y = np.asarray(a.convert("L")) < 128, np.asarray(b.convert("L")) < 128
    assert x.shape == y.shape and y.sum() > 1000
    return float((x ^ y).sum()) / float(y.sum())


@pytest.mark.parametrize("text", [THAI_LINE, "AVA example VAT 4521"], ids=["thai-marks", "latin-kerning"])
def test_lines_are_drawn_shaped(tmp_path: Path, text: str):
    font = shaping_font(tmp_path)
    geometry = (24.0, 12.0, 50.0, (420.0, 80.0))
    ref = _reference_line(font, text, *geometry, tmp_path / "reference.png")
    kit = _kit_line(font, text, *geometry, tmp_path / "kit.png")
    assert _ink_mismatch(kit, ref) < 0.005
    # The font's GPOS really moves glyphs: the same line drawn unshaped (the kit before 2026-11-06,
    # which lost the Thai pack's tone marks) is far off.
    unshaped = _kit_line(font, text, *geometry, tmp_path / "unshaped.png", unshaped=True)
    assert _ink_mismatch(unshaped, ref) > 0.03


def test_canary_is_drawn_shaped(tmp_path: Path):
    font = shaping_font(tmp_path)
    png = R.render_canary(THAI_LINE, [font], tmp_path / "canary.png")
    fonts = R._load_fonts([font])
    sizer = R._new_pdf(fonts)
    sizer.add_page()
    width = R._text_width(sizer, R._runs(THAI_LINE, fonts, fonts[0]) or [], R.CANARY_SIZE) + 2 * 24.0
    ref = _reference_line(font, THAI_LINE, R.CANARY_SIZE, 24.0, R.CANARY_SIZE * 1.55,
                          (max(width, 72.0), R.CANARY_SIZE * 2.4), tmp_path / "reference.png")
    assert _ink_mismatch(png, ref) < 0.005


def test_shaping_needs_the_kit_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "uharfbuzz", None)
    with pytest.raises(SetupError, match=r"pdfintake\[kit\]"):
        R.render_canary("example", font_files(tmp_path), tmp_path / "c.png")


def test_kit_render_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    from pdfintake.kit import cli

    packs, work, web = fake_pack(tmp_path, monkeypatch)
    common = ["--packs", str(packs), "--work", str(work)]
    assert cli.main(["kit", "corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL,
                     "--pages", "prose=1,form=1,microprint=0,mixed_script=0"]) == 0
    capsys.readouterr()
    # One Latin font: the bench needs two per script (DESIGN §3.6), so render stops before drawing.
    assert cli.main(["kit", "render", "en", *common, "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"]["code"] == "kit_fonts_too_few" and out["error"]["problems"] == [
        "Latin: 1 corpus font covers it, the bench needs at least 2"]
    assert not (work / "en" / "pages" / R.PAGES_FILE).exists()
    assert cli.main(["kit", "corpus", "en", *common, "--font", FONT2_URL]) == 0
    capsys.readouterr()
    assert cli.main(["kit", "render", "en", *common, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["pages"] == 6 and out["by_kind"] == {"prose": 3, "form": 3}
    pages = R.load_pages(work / "en" / "pages")
    assert "Infobox" not in pages[0].truth and "Category" not in pages[0].truth
    # One seed for every render, so CI re-renders exactly these pages: there is no --seed.
    assert json.loads((work / "en" / "pages" / R.PAGES_FILE).read_text(encoding="utf-8"))["seed"] == R.BENCH_SEED
    assert cli.main(["kit", "render", "en", *common, "--seed", "7"]) == 2


def test_font_coverage_counts_fonts_per_script(tmp_path: Path):
    fonts = font_files(tmp_path, thai=True)
    assert R.font_coverage_problems(fonts, ["Latin"]) == ["Latin: 1 corpus font covers it, the bench needs at least 2"]
    second = build.write_box_font(tmp_path / "ExampleSerif-Regular.ttf", latin_chars(), family="Example Serif")
    assert R.font_coverage_problems([*fonts, second], ["Latin"]) == []
    assert R.font_coverage_problems([*fonts, second], ["Thai", "Latin"]) == [
        "Thai: 1 corpus font covers it, the bench needs at least 2"]
    c = corpus_for()
    with pytest.raises(PdfIntakeError) as exc:
        R.render_pages(c, fonts, tmp_path / "p", texts={c.texts[0].sha256: TEXT}, scripts=["Thai", "Latin"])
    assert exc.value.code == "kit_fonts_too_few" and exc.value.exit_code == 1


# --------------------------------------------------------------------------- the canary's PNG encoding

def _pattern(w: int = 40, h: int = 12) -> bytes:
    return bytes(((x * 7 + y * 13) % 256) if (x // 5 + y // 3) % 2 else 255 for y in range(h) for x in range(w))


@pytest.mark.parametrize("data", [b"", b"a", b"abcabcabcabc" * 100, bytes(range(256)) * 9, b"\xff" * 70000 + b"x"])
def test_deflate_fixed_round_trips_through_any_inflater(data: bytes):
    import zlib

    assert zlib.decompress(R.deflate_fixed(data), -15) == data


def test_the_canary_png_is_the_same_bytes_from_any_zlib_build():
    """The Linux CI runners drew canary-en.png with the same pixels as macOS and Windows (one pixel
    digest on all three) but 7412 bytes against 7333: Pillow's wheels compress with zlib-ng on macOS
    and Windows and classic zlib on Linux. png_bytes writes its own deflate stream, so this digest
    is what every OS gets. Changing it re-pins every pack's canary (`pdfintake kit wheel <id>`)."""
    png = R.png_bytes(_pattern(), 40, 12, dpi=300)
    assert hashlib.sha256(png).hexdigest() == "1ace8e66e2c30797199fb6c1cf78dd805fe3fc73e3126321ef7bc06256b17087"
    with Image.open(io.BytesIO(png)) as im:
        assert (im.mode, im.size, im.tobytes()) == ("L", (40, 12), _pattern())
        assert im.info["dpi"] == pytest.approx((300, 300), abs=0.01)


def test_render_canary_never_saves_the_png_through_pillow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pillow's PNG encoder compresses with the platform's zlib build; the canary goes through
    png_bytes instead. (fpdf2 still deflates the throwaway PDF, which only changes how the pages are
    stored, not the pixels pdfium draws from them.)"""
    def no(*a: Any, **k: Any) -> Any:
        raise AssertionError("the canary was saved through Pillow")

    monkeypatch.setattr(Image.Image, "save", no)
    fonts = font_files(tmp_path)
    png = R.render_canary("Sunlight 4521", fonts, tmp_path / "canary-xx.png")
    first = png.read_bytes()
    assert first.startswith(b"\x89PNG") and R.render_canary("Sunlight 4521", fonts, tmp_path / "again.png").read_bytes() == first
