"""The Tesseract engine (``pdfintake.ocr.engines.tesseract``).

Three kinds of test:

* Logic that needs no model: ``pair_lines``, parameters, availability states, broken-model
  naming, the shared-folder rule, and the read loop against a stand-in ``PyTessBaseAPI``.
* Real reads that need the base pack's ``eng`` and ``osd``: English word recall, a blank page, the
  1 ms timeout, a dpi-less PNG, a 16-bit grey PNG, a truncated model, a model in a non-ASCII
  folder (DESIGN §9.3; the Windows job's cache path is non-ASCII too), OSD.
* Real reads that need the Thai and Hungarian packs, all marked ``network`` (the CI network job
  runs ``pytest -m network tests/test_tesseract.py``; with ``PDFINTAKE_NETWORK_TESTS=1`` a missing
  th or hu file is fetched first, and a download that fails fails the test with its error rather than
  skipping it): the per-instance parameter test, the Thai gap test, and a Thai
  read whose models sit in two folders joined by the shared tessdata folder.

Every model and font comes from ``conftest.pinned_asset(pack, key)``: the file the pack manifest
pins, under ``PDFINTAKE_TEST_MODELS`` (default ``~/.pdfintake-dev/models``) as ``<pack>/<file>``,
checked against its pinned size and SHA-256 (``tools/fetch_test_models.py`` downloads them). A test
whose files aren't there or don't match is skipped, naming what is missing.

Every page is drawn here from made-up text: Latin with Pillow's built-in font, Thai shaped with
uharfbuzz and filled from the font's outlines (Pillow has no complex-script layout).
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import pinned_asset, pinned_assets
from fixtures import build
from pdfintake.errors import EngineFailure, PackError, SetupError
from pdfintake.ocr import engines
from pdfintake.ocr.engines import Availability, ContractCase
from pdfintake.ocr.engines import tesseract as T
from pdfintake.params import ValidationContext
from pdfintake.protocols import OsdSession
from pdfintake.types import EngineRead, Line, PageImage, to_jsonable

REPO = Path(__file__).resolve().parents[1]
HERE_KEY = "darwin-arm64"          # any supported key: availability only checks membership
CTX = ValidationContext(frozenset(), frozenset(), {}, frozenset(), "test.params")

needs_tesserocr = pytest.mark.skipif(T.PyTessBaseAPI is None, reason=f"tesserocr isn't importable: {T._IMPORT_ERROR}")

# Made-up text only.
EN_LINES = ("The example invoice lists seven sample items",
            "and a total of 4521 units shipped today.",
            "Every line here is made up for a test page,",
            "so nothing in it describes a real document.")
OSD_EXTRA = ("Each sample row names a made up product and",
             "a quantity, a unit price and a short remark.")
HU_LINES = ("Ez egy kitalalt mondat a teszt oldalhoz,",
            "amely csak peldakent szolgal minden gepen.")
THAI_LINES = ("ข้อความตัวอย่าง สำหรับการทดสอบ",
              "บริษัทตัวอย่าง จำกัด",
              "ภาษาไทยเขียนต่อกัน ไม่เว้นวรรค ระหว่างคำ",
              "หน้าที่หนึ่งของเอกสารทดสอบ",
              "ระบบอ่านข้อความ ทำงานได้ดี")


# --------------------------------------------------------------------------- pages

def latin_page(folder: Path, lines: Sequence[str] = EN_LINES, *, size: int = 42, dpi: float | None = 300.0,
               page_dpi: float = 300.0, name: str = "latin.png", rotate: int = 0) -> PageImage:
    """Made-up Latin text in Pillow's built-in font (Aileron, ASCII), as a PNG with (or without) pHYs."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=size)
    width = int(size * 26 + 2 * size)
    height = int(size * 2.4 * len(lines) + 2 * size)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((size, size + i * size * 2.4), text, font=font, fill=0)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=255)
    path = folder / name
    if dpi is None:
        img.save(path, format="PNG")
    else:
        img.save(path, format="PNG", dpi=(dpi, dpi))
    w, h = img.size
    return PageImage(0, path, w, h, float(page_dpi), (0.0, 0.0, w * 72.0 / page_dpi, h * 72.0 / page_dpi))


def thai_page(folder: Path, font: Path, lines: Sequence[str] = THAI_LINES, *, size_px: float = 50.0,
              dpi: float = 300.0) -> PageImage:
    """Made-up Thai text shaped by uharfbuzz and filled from the font's outlines, at ``dpi``."""
    pytest.importorskip("uharfbuzz")
    width, height = 1800, int(120 + size_px * 2.0 * len(lines) + 80)
    img = _shaped_render(font, lines, size_px=size_px, width=width, height=height, x=80.0, y=120.0, leading=2.0)
    path = folder / "thai.png"
    img.save(path, format="PNG", dpi=(dpi, dpi))
    return PageImage(0, path, width, height, float(dpi), (0.0, 0.0, width * 72.0 / dpi, height * 72.0 / dpi))


def _shaped_render(font_path: Path, lines: Sequence[str], *, size_px: float, width: int, height: int,
                   x: float, y: float, leading: float, ss: int = 3) -> Any:
    import numpy as np
    import uharfbuzz as hb
    from fontTools.pens.basePen import BasePen
    from fontTools.ttLib import TTFont
    from PIL import Image

    class FlatPen(BasePen):
        def __init__(self, glyphset: Any) -> None:
            super().__init__(glyphset)
            self.contours: list[list[tuple[float, float]]] = []
            self.cur: list[tuple[float, float]] = []

        def _moveTo(self, pt: tuple[float, float]) -> None:
            self.cur = [pt]

        def _lineTo(self, pt: tuple[float, float]) -> None:
            self.cur.append(pt)

        def _qCurveToOne(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
            p0 = self.cur[-1]
            for i in range(1, 9):
                t = i / 8
                a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
                self.cur.append((a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1]))

        def _curveToOne(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> None:
            p0 = self.cur[-1]
            for i in range(1, 9):
                t = i / 8
                m = 1 - t
                self.cur.append((m ** 3 * p0[0] + 3 * m * m * t * p1[0] + 3 * m * t * t * p2[0] + t ** 3 * p3[0],
                                 m ** 3 * p0[1] + 3 * m * m * t * p1[1] + 3 * m * t * t * p2[1] + t ** 3 * p3[1]))

        def _closePath(self) -> None:
            if len(self.cur) > 1:
                self.contours.append(self.cur)
            self.cur = []

        _endPath = _closePath

    data = Path(font_path).read_bytes()
    face = hb.Face(data)
    hbfont = hb.Font(face)
    tt = TTFont(str(font_path))
    glyphs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    scale = size_px * ss / face.upem
    edges: list[tuple[float, float, float, float]] = []
    for li, text in enumerate(lines):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {})
        pen_x = x * ss
        base = (y + li * size_px * leading) * ss
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            pen = FlatPen(glyphs)
            glyphs[order[info.codepoint]].draw(pen)
            ox, oy = pen_x + pos.x_offset * scale, base - pos.y_offset * scale
            for contour in pen.contours:
                pts = [(ox + px * scale, oy - py * scale) for px, py in contour]
                edges.extend((ax, ay, bx, by) for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]))
            pen_x += pos.x_advance * scale
    w, h = width * ss, height * ss
    cover = np.zeros((h, w), dtype=bool)
    e = np.asarray(edges, dtype=np.float64)
    e = e[e[:, 1] != e[:, 3]]
    x0, y0, x1, y1 = e.T
    direction = np.where(y1 > y0, 1, -1)
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    for row in range(max(0, int(lo.min())), min(h, int(np.ceil(hi.max())) + 1)):   # non-zero winding fill
        yc = row + 0.5
        on = (lo <= yc) & (yc < hi)
        if not on.any():
            continue
        xs = x0[on] + (yc - y0[on]) * (x1[on] - x0[on]) / (y1[on] - y0[on])
        order_x = np.argsort(xs, kind="stable")
        xs, winding = xs[order_x], np.cumsum(direction[on][order_x])
        diff = np.zeros(w + 1, dtype=np.int32)
        for k in range(len(xs) - 1):
            if winding[k] != 0:
                a = int(np.clip(np.ceil(xs[k] - 0.5), 0, w))
                b = int(np.clip(np.ceil(xs[k + 1] - 0.5), 0, w))
                if b > a:
                    diff[a] += 1
                    diff[b] -= 1
        cover[row] = np.cumsum(diff[:-1]) > 0
    grey = cover.reshape(height, ss, width, ss).mean(axis=(1, 3))
    return Image.fromarray((255 - np.round(grey * 255)).astype(np.uint8), "L")


# --------------------------------------------------------------------------- helpers

def _params(**raw: Any) -> dict[str, Any]:
    return T.ENGINE.PARAMS.validate(raw, CTX)


def _comparable(read: EngineRead) -> dict[str, Any]:
    data = to_jsonable(read, include_text=True)
    data.pop("seconds")
    return data


def _read(models: dict[str, Path], page: PageImage, *, lang: str, psm: int = 3, timeout_s: float = 120.0) -> EngineRead:
    session = T.ENGINE.open("tesseract", "reader", _params(lang=lang, psm=psm), models)
    try:
        return session.read(page, timeout_s=timeout_s)
    finally:
        session.close()


def _words(text: str) -> list[str]:
    return [w.casefold() for w in re.findall(r"[^\W_]+", text)]


def word_recall(truth: Iterable[str], read: str) -> float:
    want = Counter(_words("\n".join(truth)))
    got = Counter(_words(read))
    return sum(min(n, got[w]) for w, n in want.items()) / max(1, sum(want.values()))


def thai_gaps(text: str) -> int:
    """Whitespace runs between two Thai characters, counted inside each line."""
    return sum(len(re.findall(r"(?<=[\u0e00-\u0e7f])[ \t\u00a0]+(?=[\u0e00-\u0e7f])", ln)) for ln in text.splitlines())


def _assert_lines_in_image(read: EngineRead, page: PageImage) -> None:
    for ln in read.lines:
        assert 0.0 <= ln.conf <= 1.0
        for x, y in ln.box:
            assert 0.0 <= x <= page.width and 0.0 <= y <= page.height
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = ln.box
        assert ax <= bx and ay <= cy and ay == by and cy == dy and ax == dx and bx == cx   # clockwise from top-left


# --------------------------------------------------------------------------- pair_lines (today's strings, made up)

# From a read of the synthetic Thai page below: page-level text, then the TEXTLINE words.
PAGE_TEXT = ("ข้อความตัวอย่าง สําหรับการทดสอบ\n\nบริษัทตัวอย่าง จําคัด\n\n"
             "ภาษาไทยเขียนต่อกัน ไม่เว้นวรรค ระหว่างคํา\nหน้าที่หนึ่งของเอกสารทดสอบ\n")
LINE_WORDS = (
    ("ข้", "อ", "ค", "ว", "า", "ม", "ต", "ั", "ว", "อ", "ย", "่", "า", "ง", "ส", "ํ", "า", "ห", "ร", "ั", "บ",
     "ก", "า", "ร", "ท", "ด", "ส", "อ", "บ"),
    ("บ", "ร", "ิ", "ษั", "ท", "ต", "ั", "ว", "อ", "ย", "่", "า", "ง", "จ", "ํ", "า", "ค", "ั", "ด"),
    ("ภา", "ษา", "ไท", "ย", "เข", "ี", "ย", "น", "ต", "่", "อ", "ก", "ั", "น", "ไม", "่", "เว", "้", "น", "ว", "ร",
     "ร", "ค", "ร", "ะ", "ห", "ว", "่", "า", "ง", "ค", "ํ", "า"),
    ("ห", "น", "้", "า", "ท", "ี", "่", "ห", "น", "ึ", "่", "ง", "ขอ", "ง", "เอ", "ก", "ส", "า", "ร", "ท", "ด", "ส",
     "อ", "บ"),
)


def _iter(words: Sequence[str], k: int, conf: float = 0.9) -> T.IterLine:
    return T.IterLine(tuple(words), build.quad(10, 10 + 60 * k, 900, 50 + 60 * k), conf)


def test_pair_lines_takes_the_page_text_for_thai():
    lines = T.pair_lines(PAGE_TEXT, [_iter(w, k, 0.9 + k / 100) for k, w in enumerate(LINE_WORDS)])
    assert [ln.text for ln in lines] == [
        "ข้อความตัวอย่าง สําหรับการทดสอบ", "บริษัทตัวอย่าง จําคัด",
        "ภาษาไทยเขียนต่อกัน ไม่เว้นวรรค ระหว่างคํา", "หน้าที่หนึ่งของเอกสารทดสอบ"]
    assert [ln.conf for ln in lines] == [0.9, 0.91, 0.92, 0.93]
    assert lines[2].box == build.quad(10, 130, 900, 170)
    assert all(ln.words == () for ln in lines)
    # the TEXTLINE text itself would have spaced every cluster
    assert thai_gaps(" ".join(LINE_WORDS[0])) == 28 and thai_gaps(lines[0].text) == 1


def test_pair_lines_falls_back_to_the_words_without_thai_gaps():
    other_page = "ข้อความอื่น\nบริษัทตัวอย่าง จําคัด\n"            # line 0 doesn't match its words
    lines = T.pair_lines(other_page, [_iter(LINE_WORDS[0], 0), _iter(LINE_WORDS[1], 1)])
    assert lines[0].text == "ข้อความตัวอย่างสําหรับการทดสอบ"     # joined, no Thai-Thai space
    assert lines[1].text == "บริษัทตัวอย่าง จําคัด"               # still paired by position


def test_pair_lines_latin_fallback_and_mixed_words():
    lines = T.pair_lines("", [_iter(("Total", "4521", "บาท", "ถ้วน"), 0), _iter(("Example", "Ltd."), 1)])
    assert [ln.text for ln in lines] == ["Total 4521 บาทถ้วน", "Example Ltd."]


def test_pair_lines_skips_empty_iterator_lines_and_blank_page_lines():
    page = "\n\n  first line here \n\n\nsecond   line\n   \n"
    iters = [_iter(("first", "line", "here"), 0), _iter(("", "  "), 1), _iter((), 2), _iter(("second", "line"), 3)]
    lines = T.pair_lines(page, iters)
    assert [ln.text for ln in lines] == ["first line here", "second line"]     # whitespace runs collapsed
    assert lines[1].box == build.quad(10, 190, 900, 230)


def test_pair_lines_more_lines_than_the_page_text():
    lines = T.pair_lines("one two\n", [_iter(("one", "two"), 0), _iter(("three",), 1)])
    assert [ln.text for ln in lines] == ["one two", "three"]
    assert T.pair_lines("", []) == []


def test_pair_lines_needs_the_exact_letters():
    lines = T.pair_lines("one tw0\n", [_iter(("one", "two"), 0)])
    assert lines[0].text == "one two"                           # a page line that differs is not taken


# --------------------------------------------------------------------------- the engine object

def test_registry_resolves_the_engine():
    assert engines.get("tesseract") is T.ENGINE
    eng = T.ENGINE
    assert isinstance(eng, engines.Engine)
    assert eng.name == "tesseract" and eng.api == 1 and eng.session_per_thread is True
    assert eng.MODEL_KINDS == frozenset({"traineddata"})
    assert eng.platforms == {"darwin-arm64", "win-amd64", "linux-x86_64", "linux-aarch64"}
    assert "win-arm64" not in eng.platforms


def test_platforms_are_every_supported_platform():
    platforms = pytest.importorskip("pdfintake.platforms")
    assert T.ENGINE.platforms == {p.key for p in platforms.PLATFORMS if p.supported}


def test_params_table():
    assert _params(lang="tha+eng") == {"lang": "tha+eng", "psm": 3, "timeout_s": 300.0}
    assert _params(lang="hun", psm=6, timeout_s=10)["timeout_s"] == 10.0
    for lang in ("chi_sim", "aze_cyrl", "Thai", "tha+eng+deu+hun"):
        assert _params(lang=lang)["lang"] == lang
    bad = [{"lang": "eng", "psm": 5}, {"lang": "eng", "timeout_s": 5}, {"lang": "eng", "timeout_s": 601},
           {"lang": "~eng"}, {"lang": "eng;rm"}, {"lang": "../eng"}, {"lang": "a+b"}, {"lang": "eng+"},
           {"lang": "e1+e2+e3+e4+e5"}, {}, {"lang": "eng", "oem": 1}]
    for raw in bad:
        with pytest.raises(PackError):
            T.ENGINE.PARAMS.validate(raw, CTX)


@needs_tesserocr
def test_version_names_tesseract_and_tesserocr():
    version = T.ENGINE.version()
    assert re.fullmatch(r"\d+\.\d+\.\d+ \(tesserocr \d+\.\d+\.\d+\)", version or ""), version


def test_candidate_params_for_the_kit():
    def asset(file: str, engine: str = "tesseract", kind: str = "traineddata") -> Any:
        return SimpleNamespace(key=file.split(".")[0], file=file, engine=engine, kind=kind)

    th = [asset("tha.traineddata"), asset("th_rec.onnx", "rapidocr", "onnx_rec")]
    assert T.ENGINE.candidate_params("th", "Thai", th) == [{"lang": "tha", "psm": 3}, {"lang": "tha+eng", "psm": 3}]
    assert T.ENGINE.candidate_params("hu", "Latin", [asset("hun.traineddata")]) == [{"lang": "hun", "psm": 3}]
    assert T.ENGINE.candidate_params("en", "Latin", [asset("osd.traineddata"), asset("eng.traineddata")]) == \
        [{"lang": "eng", "psm": 3}]
    assert T.ENGINE.candidate_params("xx", "Latin", [asset("xx.onnx", "rapidocr", "onnx_rec")]) == []
    for params in T.ENGINE.candidate_params("th", "Thai", th):
        T.ENGINE.PARAMS.validate(params, CTX)


def test_contract_case_needs_the_base_pack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(T, "_contract_models", lambda: None)
    assert T.ENGINE.contract_case() is None
    if T.PyTessBaseAPI is None:
        return
    model = tmp_path / "eng.traineddata"
    model.write_bytes(b"x")
    monkeypatch.setattr(T, "_contract_models", lambda: {"eng": model})
    case = T.ENGINE.contract_case()
    assert isinstance(case, ContractCase)
    assert T.ENGINE.PARAMS.validate(case.params, CTX)["lang"] == "eng"
    assert case.model_kinds == {"eng": "traineddata"} and case.models == {"eng": model}
    assert case.unsupported_params is not None and case.page is not None
    page = case.page(tmp_path)
    assert page.path.is_file() and page.dpi == 300.0
    with page.open() as img:
        assert img.size == (page.width, page.height) and img.info.get("dpi") is not None


# --------------------------------------------------------------------------- availability without models

def test_not_on_platform_is_never_failed(tmp_path: Path):
    for key in ("nowhere-none", "win-arm64", "darwin-x86_64"):
        avail = T.ENGINE.availability(_params(lang="eng"), {}, key)
        assert avail == Availability(False, "not_on_platform", avail.detail) and key in avail.detail


@needs_tesserocr
def test_missing_model_is_named(tmp_path: Path):
    avail = T.ENGINE.availability(_params(lang="tha+eng"), {}, HERE_KEY)
    assert avail.state == "missing_model" and not avail.ok and "tha.traineddata" in avail.detail
    gone = tmp_path / "tha.traineddata"
    avail = T.ENGINE.availability(_params(lang="tha"), {"tha": gone}, HERE_KEY)
    assert avail.state == "missing_model" and str(gone) in avail.detail and "tha" in avail.detail
    elsewhere = {"tha": tmp_path / "some-other-name.bin"}
    (tmp_path / "some-other-name.bin").write_bytes(b"x")
    assert T.ENGINE.availability(_params(lang="tha"), elsewhere, HERE_KEY).state == "missing_model"


@needs_tesserocr
def test_osd_is_not_a_reading_language(tmp_path: Path):
    model = tmp_path / "osd.traineddata"
    model.write_bytes(b"x")
    avail = T.ENGINE.availability(_params(lang="osd"), {"osd": model}, HERE_KEY)
    assert avail.state == "unsupported_language" and not avail.ok
    assert T.ENGINE.availability(_params(lang="eng+osd"), {}, HERE_KEY).state == "unsupported_language"


@needs_tesserocr
@pytest.mark.parametrize("kind", ["zero", "html", "html_bom"])
def test_empty_and_html_models_are_broken_and_named(tmp_path: Path, kind: str):
    model = tmp_path / "eng.traineddata"
    html = b"<!DOCTYPE html>\n<html><body>Not Found</body></html>\n"
    model.write_bytes({"zero": b"", "html": html, "html_bom": b"\xef\xbb\xbf  " + html}[kind])
    avail = T.ENGINE.availability(_params(lang="eng"), {"en:eng": model}, HERE_KEY)
    assert avail.state == "broken" and not avail.ok
    assert str(model) in avail.detail
    assert ("empty" if kind == "zero" else "HTML") in avail.detail
    with pytest.raises(EngineFailure) as info:
        T.ENGINE.open("tesseract", "reader", _params(lang="eng"), {"en:eng": model})
    assert info.value.code == "model_broken" and str(model) in info.value.message
    with pytest.raises(SetupError) as osd_info:
        osd = tmp_path / "osd" / "osd.traineddata"
        osd.parent.mkdir()
        osd.write_bytes(model.read_bytes())
        T.open_osd({"osd": osd})
    assert osd_info.value.code == "model_broken" and str(osd) in osd_info.value.message


@needs_tesserocr
def test_the_broken_file_is_named_among_several(tmp_path: Path):
    good = pinned_asset("en", "eng")
    folder = tmp_path / "td"
    folder.mkdir()
    try:
        os.link(good, folder / "eng.traineddata")
    except OSError:
        shutil.copyfile(good, folder / "eng.traineddata")
    (folder / "tha.traineddata").write_bytes(b"")
    models = {"en:eng": folder / "eng.traineddata", "tha": folder / "tha.traineddata"}
    avail = T.ENGINE.availability(_params(lang="tha+eng"), models, HERE_KEY)
    assert avail.state == "broken" and str(folder / "tha.traineddata") in avail.detail
    assert "eng.traineddata" not in avail.detail


def _header(offsets: Sequence[int], size: int) -> bytes:
    """A traineddata-shaped file: an int32 count, int64 offsets, then filler up to ``size`` bytes."""
    head = len(offsets).to_bytes(4, "little") + b"".join(o.to_bytes(8, "little", signed=True) for o in offsets)
    return head + b"\x00" * (size - len(head))


class _RefusingAPI:
    """A PyTessBaseAPI that refuses every model, as "Failed to init API" does."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Failed to init API, possibly an invalid tessdata path")


@needs_tesserocr
def test_a_load_failure_blames_truncation_only_on_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(T, "PyTessBaseAPI", _RefusingAPI)
    intact = _touch(tmp_path / "a" / "eng.traineddata", _header([-1, 60, 100, 150], 200))
    cut = _touch(tmp_path / "b" / "eng.traineddata", _header([-1, 60, 100, 150], 120))   # last component past the end
    shuffled = _touch(tmp_path / "c" / "eng.traineddata", _header([100, 60], 200))
    junk = _touch(tmp_path / "d" / "eng.traineddata", b"\xff" * 300)
    detail = T.ENGINE.availability(_params(lang="eng"), {"en:eng": intact}, HERE_KEY).detail
    assert "truncated" not in detail and "header is intact" in detail and str(intact.parent) in detail
    for bad in (cut, shuffled, junk):
        avail = T.ENGINE.availability(_params(lang="eng"), {"en:eng": bad}, HERE_KEY)
        assert avail.state == "broken" and "truncated" in avail.detail and str(bad) in avail.detail


@needs_tesserocr
def test_a_windows_folder_without_a_short_name_is_blamed_not_the_model(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(T, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(T, "_short_path", lambda s: None)       # 8.3 names off on this volume
    monkeypatch.setattr(T, "PyTessBaseAPI", _RefusingAPI)
    model = _touch(tmp_path / "modèles ไทย" / "eng.traineddata", _header([-1, 60, 100], 200))
    avail = T.ENGINE.availability(_params(lang="eng"), {"en:eng": model}, HERE_KEY)
    assert avail.state == "broken" and "8.3 short name" in avail.detail and "truncated" not in avail.detail
    assert T._folder_problem(tmp_path / "plain") is None
    monkeypatch.setattr(T, "_short_path", lambda s: "C:\\Users\\RUNNER~1\\MODLES~1")
    assert T._folder_problem(model.parent) is None


class _AnsiOnlyAPI:
    """A PyTessBaseAPI that opens only an ASCII folder, as Tesseract's ANSI file API does on Windows."""

    opened: list[str] = []

    def __init__(self, path: str = "", lang: str = "", psm: int = 3, **_: Any) -> None:
        if not str(path).isascii():
            raise RuntimeError(f"Failed to init API, possibly an invalid tessdata path: {path}")
        _AnsiOnlyAPI.opened.append(str(path))

    def End(self) -> None:
        pass


@needs_tesserocr
@pytest.mark.parametrize("short", [None, "C:\\PDFINT~1\\TESSDATA"])
def test_a_shared_folder_tesseract_cant_open_is_named_before_any_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      short: str | None):
    # The Windows CI jobs: tha and eng each load from their own ASCII folder, but tha+eng loads from the
    # shared tessdata folder in a non-ASCII cache, and a volume with 8.3 names off gives that folder no
    # ASCII form. Tesseract can't open it, so availability must say so, naming it, before any page is read.
    monkeypatch.setattr(T, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(T, "_short_path", lambda s: short)
    monkeypatch.setattr(T, "PyTessBaseAPI", _AnsiOnlyAPI)
    monkeypatch.setattr(_AnsiOnlyAPI, "opened", [])
    tha = _touch(tmp_path / "packs" / "th" / "tha.traineddata", _header([-1, 60, 100], 200))
    eng = _touch(tmp_path / "site" / "en" / "eng.traineddata", _header([-1, 60, 120], 220))
    shared = tmp_path / "pdfintake-çache-ü" / "tessdata"
    for src in (tha, eng):
        shutil.copyfile(src, _touch(shared / src.name))         # copies, as across volumes
    monkeypatch.setattr(T, "_store_tessdata_dir", lambda: shared)
    models = {"tha": tha, "en:eng": eng}
    assert T._plan("tha+eng", models).tessdata == shared
    avail = T.ENGINE.availability(_params(lang="tha+eng"), models, HERE_KEY)
    if short is None:
        assert avail.state == "broken" and str(shared) in avail.detail, avail
        assert "8.3 short name" in avail.detail and "PDFINTAKE_CACHE" in avail.detail
        assert "truncated" not in avail.detail and "damaged" not in avail.detail
        with pytest.raises(EngineFailure) as info:
            T.ENGINE.open("tesseract", "reader", _params(lang="tha+eng"), models)
        assert str(shared) in info.value.message and "8.3 short name" in info.value.message
    else:
        assert avail == Availability(True, "available", "")
        T.ENGINE.open("tesseract", "reader", _params(lang="tha+eng"), models).close()
        assert _AnsiOnlyAPI.opened[-1].startswith(short)         # Tesseract got the folder's 8.3 form


def test_native_dir_ends_in_this_platform_separator(tmp_path: Path):
    assert T._native_dir(tmp_path).endswith(os.sep)
    assert T._native_dir(Path(str(tmp_path) + os.sep)) == str(tmp_path) + os.sep
    if os.sep == "/":                                          # a backslash is part of a POSIX name
        assert T._native_dir(tmp_path / "ends-in-backslash\\") == str(tmp_path / "ends-in-backslash\\") + "/"


def test_open_osd_without_a_model(tmp_path: Path):
    if T.PyTessBaseAPI is None:
        with pytest.raises(SetupError):
            T.open_osd({})
        return
    with pytest.raises(SetupError) as info:
        T.open_osd({"eng": tmp_path / "eng.traineddata"})
    assert info.value.code == "missing_model" and "osd.traineddata" in info.value.message
    with pytest.raises(SetupError) as info:
        T.open_osd({"osd": tmp_path / "osd.traineddata"})
    assert info.value.code == "missing_model" and str(tmp_path / "osd.traineddata") in info.value.message


# --------------------------------------------------------------------------- one folder, or the shared one

def _touch(path: Path, data: bytes = b"model bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_models_in_one_folder_load_from_it(tmp_path: Path):
    tha = _touch(tmp_path / "a" / "tha.traineddata")
    eng = _touch(tmp_path / "a" / "eng.traineddata")
    plan = T._plan("tha+eng", {"tha": tha, "en:eng": eng})
    assert plan.state == "available" and plan.tessdata == tha.parent
    assert dict(plan.files) == {"tha": tha, "eng": eng}


def test_models_in_two_folders_use_the_shared_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tha = _touch(tmp_path / "packs" / "th" / "tha.traineddata", b"thai model")
    eng = _touch(tmp_path / "site" / "en" / "eng.traineddata", b"english model")
    shared = tmp_path / "cache" / "tessdata"
    monkeypatch.setattr(T, "_store_tessdata_dir", lambda: shared)
    models = {"tha": tha, "en:eng": eng}
    plan = T._plan("tha+eng", models)
    assert plan.state == "broken" and "doesn't hold" in plan.detail and str(shared) in plan.detail
    _touch(shared / "tha.traineddata", b"thai model")           # a copy (where linking fails)
    _touch(shared / "eng.traineddata", b"other model")          # same name, other bytes
    plan = T._plan("tha+eng", models)
    assert plan.state == "broken" and "eng.traineddata" in plan.detail
    (shared / "eng.traineddata").unlink()
    try:
        os.link(eng, shared / "eng.traineddata")                  # a hard link, as the store makes
    except OSError:
        _touch(shared / "eng.traineddata", b"english model")
    plan = T._plan("tha+eng", models)
    assert plan.state == "available" and plan.tessdata == shared


def _windows_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows' ``st_ctime`` semantics on any OS: a file's ctime is its creation time, so no write moves it."""
    real = Path.stat
    created: dict[str, int] = {}
    in_sequence = {"st_mode", "st_ino", "st_dev", "st_nlink", "st_uid", "st_gid", "st_size"}   # 3.14 refuses them twice

    def stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        st = real(self, follow_symlinks=follow_symlinks)
        ctime_ns = created.setdefault(os.path.realpath(self), st.st_ctime_ns)
        fields = {k: getattr(st, k) for k in dir(st) if k.startswith("st_") and k not in in_sequence}
        fields.update(st_ctime_ns=ctime_ns, st_ctime=ctime_ns / 1e9)
        visible = list(tuple(st)[:10])
        visible[9] = ctime_ns // 10**9
        return os.stat_result(visible, fields)

    monkeypatch.setattr(Path, "stat", stat)


@pytest.mark.parametrize("platform", ["native", "windows"])
def test_an_in_place_rewrite_of_the_shared_copy_is_seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str):
    if platform == "windows":                                    # on a Windows runner "native" is this case too
        monkeypatch.setattr(T, "sys", SimpleNamespace(platform="win32"))
        _windows_stat(monkeypatch)
    tha = _touch(tmp_path / "packs" / "th" / "tha.traineddata", b"thai model")
    eng = _touch(tmp_path / "site" / "en" / "eng.traineddata", b"english model")
    shared = tmp_path / "cache" / "tessdata"
    monkeypatch.setattr(T, "_store_tessdata_dir", lambda: shared)
    for src in (tha, eng):
        shutil.copyfile(src, _touch(shared / src.name))         # copies, so the hashes decide
    models = {"tha": tha, "en:eng": eng}
    assert T._plan("tha+eng", models).state == "available"
    target = shared / "eng.traineddata"
    st = target.stat()
    key = T._stat_key(target)
    time.sleep(0.01)
    with open(target, "r+b") as fh:                              # same size, other bytes, mtime put back
        fh.write(b"ENGLISH")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert target.stat().st_mtime_ns == st.st_mtime_ns and target.stat().st_size == st.st_size
    if platform == "windows":
        assert T._stat_key(target) == key                        # what the CI runners showed: no stat field moved
    plan = T._plan("tha+eng", models)
    assert plan.state == "broken" and "eng.traineddata" in plan.detail


def test_the_hash_cache_is_used_only_where_a_write_moves_the_stat_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _touch(tmp_path / "m" / "eng.traineddata", b"english model")
    reads: list[Path] = []
    real_open = open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        reads.append(Path(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(T, "open", counting_open, raising=False)
    monkeypatch.setattr(T, "sys", SimpleNamespace(platform="linux"))
    assert T._sha256(model) == T._sha256(model)
    assert reads == [model]                                      # POSIX: read once, then answered from the cache
    reads.clear()
    monkeypatch.setattr(T, "sys", SimpleNamespace(platform="win32"))
    assert T._sha256(model) == T._sha256(model)
    assert reads == [model, model]                               # Windows: read in full every time


def test_no_shared_folder_is_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tha = _touch(tmp_path / "x" / "tha.traineddata")
    eng = _touch(tmp_path / "y" / "eng.traineddata")

    def no_store() -> Path:
        raise SetupError("no_store", "the store isn't set up")

    monkeypatch.setattr(T, "_store_tessdata_dir", no_store)
    plan = T._plan("tha+eng", {"tha": tha, "eng": eng})
    assert plan.state == "broken" and "isn't available" in plan.detail


# --------------------------------------------------------------------------- the read loop, with a stand-in API

@dataclass
class _Word:
    text: str
    conf: float
    box: tuple[int, int, int, int]
    line_start: bool = False
    block_start: bool = False
    line_box: tuple[int, int, int, int] | None = None


class _FakeIterator:
    def __init__(self, words: list[_Word]) -> None:
        self.words = words
        self.i = 0

    def Begin(self) -> None:
        self.i = 0

    def Empty(self, level: int) -> bool:
        return not self.words

    def IsAtBeginningOf(self, level: int) -> bool:
        w = self.words[self.i]
        return w.block_start if level == T.RIL.BLOCK else w.line_start if level == T.RIL.TEXTLINE else True

    def BoundingBox(self, level: int) -> tuple[int, int, int, int] | None:
        w = self.words[self.i]
        return w.line_box if level == T.RIL.TEXTLINE else w.box

    def GetUTF8Text(self, level: int) -> str:
        return self.words[self.i].text

    def Confidence(self, level: int) -> float:
        return self.words[self.i].conf

    def Next(self, level: int) -> bool:
        self.i += 1
        return self.i < len(self.words)


class _FakeAPI:
    """Stands in for PyTessBaseAPI: records calls, returns a scripted recognition."""

    script: dict[str, Any] = {}
    instances: list["_FakeAPI"] = []

    def __init__(self, path: str = "", lang: str = "eng", psm: int = 3, **kwargs: Any) -> None:
        self.path, self.lang, self.psm = path, lang, psm
        self.calls: list[tuple[Any, ...]] = []
        _FakeAPI.instances.append(self)

    def SetImage(self, img: Any) -> None:
        self.calls.append(("SetImage", img.mode, img.size))

    def SetSourceResolution(self, dpi: int) -> None:
        self.calls.append(("SetSourceResolution", dpi))

    def ClearAdaptiveClassifier(self) -> None:
        self.calls.append(("ClearAdaptiveClassifier",))

    def Recognize(self, timeout: int = 0) -> bool:
        self.calls.append(("Recognize", timeout))
        delay = self.script.get("delay_s", 0.0)
        if delay:
            time.sleep(delay)
        return self.script.get("ok", True)

    def GetUTF8Text(self) -> str:
        return self.script.get("text", "")

    def GetIterator(self) -> _FakeIterator:
        return _FakeIterator(list(self.script.get("words", [])))

    def DetectOrientationScript(self) -> dict[str, Any] | None:
        self.calls.append(("DetectOrientationScript",))
        return self.script.get("osd")

    def Clear(self) -> None:
        self.calls.append(("Clear",))

    def End(self) -> None:
        self.calls.append(("End",))


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    if T.PyTessBaseAPI is None:
        pytest.skip("tesserocr isn't importable (RIL constants)")
    _FakeAPI.script = {}
    _FakeAPI.instances = []
    monkeypatch.setattr(T, "PyTessBaseAPI", _FakeAPI)
    models = {"tha": _touch(tmp_path / "td" / "tha.traineddata"), "en:eng": _touch(tmp_path / "td" / "eng.traineddata")}
    page = build.page_image(tmp_path, image=build.text_like(1000, 600), dpi=200, name="p.png")
    return SimpleNamespace(models=models, page=page, tmp=tmp_path)


def _thai_words() -> list[_Word]:
    out: list[_Word] = []
    for k, words in enumerate(LINE_WORDS[:2]):
        for j, w in enumerate(words):
            out.append(_Word(w, 90.0 + k, (10 + 20 * j, 10 + 60 * k, 30 + 20 * j, 50 + 60 * k), line_start=j == 0,
                             block_start=(k, j) in ((0, 0), (1, 0)), line_box=(10, 10 + 60 * k, 600, 50 + 60 * k)))
    out[3] = _Word(out[3].text, -1.0, out[3].box)               # no confidence: left out of the means
    return out


def test_read_pairs_page_text_with_iterator_boxes(fake_api: Any):
    _FakeAPI.script = {"text": PAGE_TEXT, "words": _thai_words()}
    session = T.ENGINE.open("tesseract", "reader", _params(lang="tha+eng", psm=4), fake_api.models)
    read = session.read(fake_api.page, timeout_s=50)
    session.close()
    assert read.status == "ok" and read.engine == "tesseract" and read.use_id == "tesseract"
    assert read.spec == "tesseract:tha+eng:psm4" and read.role == "reader" and read.error is None
    assert [ln.text for ln in read.lines] == ["ข้อความตัวอย่าง สําหรับการทดสอบ", "บริษัทตัวอย่าง จําคัด"]
    assert read.lines[0].box == build.quad(10, 10, 600, 50) and read.lines[1].box == build.quad(10, 70, 600, 110)
    assert read.lines[0].conf == 0.9 and read.lines[1].conf == 0.91
    assert read.meta["line_blocks"] == [0, 1] and read.meta["dpi"] == 200
    assert read.meta["words"] == len(LINE_WORDS[0]) + len(LINE_WORDS[1]) - 1
    api = _FakeAPI.instances[0]
    assert api.lang == "tha+eng" and api.psm == 4 and api.path.endswith(os.sep)
    names = [c[0] for c in api.calls]
    assert names[:4] == ["SetImage", "SetSourceResolution", "ClearAdaptiveClassifier", "Recognize"]
    assert ("SetSourceResolution", 200) in api.calls and names[-1] == "End"


def test_boxes_are_clamped_to_the_image(fake_api: Any):
    w, h = fake_api.page.width, fake_api.page.height
    _FakeAPI.script = {"text": "edge\n", "words": [_Word("edge", 80, (-5, -3, w + 9, h + 2), True, True,
                                                          (-5, -3, w + 9, h + 2))]}
    read = _read_fake(fake_api)
    assert read.lines[0].box == build.quad(0, 0, w, h)


def _read_fake(fake_api: Any, *, timeout_s: float = 50.0, **params: Any) -> EngineRead:
    session = T.ENGINE.open("t", "checker", _params(lang="eng", **params), fake_api.models)
    try:
        return session.read(fake_api.page, timeout_s=timeout_s)
    finally:
        session.close()


def test_nothing_recognised_is_empty(fake_api: Any):
    _FakeAPI.script = {"text": "\n\n", "words": []}
    read = _read_fake(fake_api)
    assert read.status == "empty" and read.lines == [] and read.meta["mean_conf"] is None and read.role == "checker"


def test_the_smaller_timeout_wins(fake_api: Any):
    _FakeAPI.script = {"text": "", "words": []}
    session =T.ENGINE.open("t", "reader", _params(lang="eng", timeout_s=20), fake_api.models)
    session.read(fake_api.page, timeout_s=600)
    session.read(fake_api.page, timeout_s=5)
    session.close()
    budgets = [c[1] for c in _FakeAPI.instances[0].calls if c[0] == "Recognize"]
    assert 19000 <= budgets[0] <= 20000 and 4000 <= budgets[1] <= 5000


def test_running_out_of_time_is_timeout(fake_api: Any):
    _FakeAPI.script = {"ok": False, "delay_s": 0.05}
    read = _read_fake(fake_api, timeout_s=0.02)
    assert read.status == "timeout" and read.lines == [] and "timed out" in (read.error or "")
    assert ("Clear",) in _FakeAPI.instances[0].calls
    assert _read_fake(fake_api, timeout_s=0.0).status == "timeout"
    assert _read_fake(fake_api, timeout_s=float("nan")).status == "timeout"


def test_a_quick_refusal_is_an_engine_failure(fake_api: Any):
    _FakeAPI.script = {"ok": False}
    with pytest.raises(EngineFailure) as info:
        _read_fake(fake_api, timeout_s=50)
    assert info.value.code == "engine_failed"


def test_a_closed_session_raises(fake_api: Any):
    session = T.ENGINE.open("t", "reader", _params(lang="eng"), fake_api.models)
    session.close()
    session.close()                                            # twice is fine
    with pytest.raises(EngineFailure) as info:
        session.read(fake_api.page, timeout_s=10)
    assert info.value.code == "session_closed"


def test_an_unreadable_render_raises(fake_api: Any):
    gone = PageImage(3, fake_api.tmp / "missing.png", 10, 10, 300.0, (0, 0, 2.4, 2.4))
    session = T.ENGINE.open("t", "reader", _params(lang="eng"), fake_api.models)
    with pytest.raises(EngineFailure) as info:
        session.read(gone, timeout_s=10)
    session.close()
    assert info.value.code == "page_unreadable"


def test_image_modes_reach_tesseract_as_grey_or_rgb(fake_api: Any):
    from PIL import Image

    for mode in ("1", "L", "RGB", "RGBA", "LA", "P"):
        img = build.blank(64, 48, mode=mode)
        assert T._tess_image(img).mode in ("L", "RGB")
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert T._tess_image(rgba).getpixel((0, 0)) == (255, 255, 255)   # transparent reads as paper


def test_wide_samples_are_scaled_to_eight_bits_not_clipped():
    import numpy as np
    from PIL import Image

    grey = np.array([[90, 230]], dtype=np.uint16)
    sixteen = Image.fromarray(grey * 257)                     # how Pillow opens a 16-bit grey PNG
    assert sixteen.mode == "I;16"
    for img in (sixteen, sixteen.convert("I"), Image.fromarray((grey * 257).astype(np.float32))):
        out = T._tess_image(img)
        assert out.mode == "L" and [out.getpixel((0, 0)), out.getpixel((1, 0))] == [90, 230], img.mode
    for img in (Image.fromarray(grey.astype(np.int32)), Image.fromarray(grey.astype(np.float32))):
        out = T._tess_image(img)                               # already 0..255: passed through
        assert [out.getpixel((0, 0)), out.getpixel((1, 0))] == [90, 230], img.mode
    wide = T._tess_image(Image.fromarray(np.array([[0, 200000]], dtype=np.int32)))
    assert [wide.getpixel((0, 0)), wide.getpixel((1, 0))] == [0, 255]


def _png(folder: Path, name: str, size: tuple[int, int], *, dpi: float = 300.0) -> PageImage:
    from PIL import Image

    path = folder / name
    Image.new("1", size, 1).save(path, format="PNG")
    return PageImage(0, path, size[0], size[1], dpi, (0.0, 0.0, size[0] * 72.0 / dpi, size[1] * 72.0 / dpi))


def test_a_decompression_bomb_is_a_named_failure(fake_api: Any, monkeypatch: pytest.MonkeyPatch):
    from PIL import Image

    page = _png(fake_api.tmp, "bomb.png", (200, 200))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)    # 40,000 px is over twice the cap
    session = T.ENGINE.open("t", "reader", _params(lang="eng"), fake_api.models)
    try:
        with pytest.raises(EngineFailure) as info:
            session.read(page, timeout_s=10)
    finally:
        session.close()
    assert info.value.code == "page_unreadable" and str(page.path) in info.value.message
    assert "too many pixels" in info.value.message


def test_dpi_is_clamped_and_marked(fake_api: Any):
    session = T.ENGINE.open("t", "reader", _params(lang="eng"), fake_api.models)
    try:
        for given, used in ((1e12, T.MAX_DPI), (0.0, 1), (-5.0, 1), (float("nan"), 300), (float("inf"), 300),
                            (200.0, 200)):
            page = PageImage(0, fake_api.page.path, fake_api.page.width, fake_api.page.height, given, (0, 0, 1, 1))
            read = session.read(page, timeout_s=10)
            assert read.meta["dpi"] == used and read.meta.get("dpi_substituted", False) is (given != used), given
            assert ("SetSourceResolution", used) in _FakeAPI.instances[0].calls
    finally:
        session.close()


def test_a_side_over_tesseracts_limit_is_refused_by_name(fake_api: Any):
    session = T.ENGINE.open("t", "reader", _params(lang="eng"), fake_api.models)
    try:
        ok = session.read(_png(fake_api.tmp, "edge.png", (T.TESS_MAX_SIDE, 2)), timeout_s=10)
        assert ok.status == "empty"
        for size in ((T.TESS_MAX_SIDE + 1, 2), (2, T.TESS_MAX_SIDE + 1)):
            with pytest.raises(EngineFailure) as info:
                session.read(_png(fake_api.tmp, "long.png", size), timeout_s=10)
            assert info.value.code == "page_too_large" and str(T.TESS_MAX_SIDE) in info.value.message
    finally:
        session.close()


def test_osd_edge_images(fake_api: Any):
    from PIL import Image

    osd = T.TessOsd(fake_api.tmp)
    try:
        assert osd.detect(Image.new("L", (0, 0)), 300.0) is None
        assert osd.detect(Image.new("L", (0, 50)), 300.0) is None
        _FakeAPI.script = {"osd": {"script_name": "Latin", "script_conf": 2.0, "orient_deg": 90, "orient_conf": 9.0}}
        found = osd.detect(Image.new("L", (40, 40), 255), 1e12)
        assert found is not None and found.orient_deg == 90 and found.script == "Latin"
        assert ("SetSourceResolution", T.MAX_DPI) in _FakeAPI.instances[-1].calls
        with pytest.raises(EngineFailure) as info:
            osd.detect(Image.new("L", (T.TESS_MAX_SIDE + 1, 2)), 300.0)
        assert info.value.code == "page_too_large"

        def refuse(img: Any) -> None:
            raise ValueError("cannot write empty image")

        _FakeAPI.instances[-1].SetImage = refuse  # type: ignore[method-assign]
        with pytest.raises(EngineFailure) as info:
            osd.detect(Image.new("L", (40, 40), 255), 300.0)
        assert info.value.code == "osd_failed" and "ValueError" in info.value.message
    finally:
        osd.close()


# --------------------------------------------------------------------------- real reads: the base pack

@pytest.fixture
def eng(tmp_path: Path) -> dict[str, Path]:
    if T.PyTessBaseAPI is None:
        pytest.skip(f"tesserocr isn't importable: {T._IMPORT_ERROR}")
    return {"en:eng": pinned_asset("en", "eng")}


def test_available_with_a_real_model(eng: dict[str, Path]):
    assert T.ENGINE.availability(_params(lang="eng"), eng, HERE_KEY) == Availability(True, "available", "")


def test_english_word_recall(eng: dict[str, Path], tmp_path: Path):
    page = latin_page(tmp_path)
    read = _read(eng, page, lang="eng")
    recall = word_recall(EN_LINES, read.text())
    print(f"\nEnglish word recall {recall:.3f} in {read.seconds:.2f} s, mean_conf {read.meta['mean_conf']}")
    assert read.status == "ok" and recall >= 0.9
    _assert_lines_in_image(read, page)
    assert len(read.lines) == len(EN_LINES)
    assert read.meta["version"] and 0.0 < read.meta["mean_conf"] <= 1.0
    assert read.meta["line_blocks"] == sorted(read.meta["line_blocks"])


def test_dpi_less_png_reads_like_one_with_dpi(eng: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with_dpi = latin_page(tmp_path, dpi=300, name="dpi.png")
    without = latin_page(tmp_path, dpi=None, name="nodpi.png")
    with without.open() as img:
        assert "dpi" not in img.info
    a, b = _read(eng, with_dpi, lang="eng"), _read(eng, without, lang="eng")
    assert _comparable(a) == _comparable(b)
    assert a.meta["dpi"] == b.meta["dpi"] == 300
    # the engine passes the render's dpi itself, after SetImage
    seen: list[tuple[str, Any]] = []
    real = T.PyTessBaseAPI

    class Spy(real):  # type: ignore[misc, valid-type]
        def SetImage(self, img: Any) -> None:
            seen.append(("SetImage", img.info.get("dpi")))
            super().SetImage(img)

        def SetSourceResolution(self, dpi: int) -> None:
            seen.append(("SetSourceResolution", dpi))
            super().SetSourceResolution(dpi)

    monkeypatch.setattr(T, "PyTessBaseAPI", Spy)
    _read(eng, without, lang="eng")
    assert seen == [("SetImage", None), ("SetSourceResolution", 300)]


def test_blank_page_is_empty(eng: dict[str, Path], tmp_path: Path):
    page = build.page_image(tmp_path, image=build.blank(900, 600), dpi=300, name="blank.png")
    read = _read(eng, page, lang="eng")
    assert read.status == "empty" and read.lines == [] and read.error is None


def test_one_millisecond_is_a_timeout(eng: dict[str, Path], tmp_path: Path):
    page = latin_page(tmp_path)
    session = T.ENGINE.open("tesseract", "reader", _params(lang="eng"), eng)
    try:
        start = time.perf_counter()
        late = session.read(page, timeout_s=0.001)
        elapsed = time.perf_counter() - start
        again = session.read(page, timeout_s=120)            # the session recovers
    finally:
        session.close()
    print(f"\n1 ms timeout returned {late.status} after {elapsed * 1000:.1f} ms")
    assert late.status == "timeout" and late.lines == [] and elapsed < 2.0
    assert again.status == "ok" and word_recall(EN_LINES, again.text()) >= 0.9


def test_truncated_model_is_broken_and_named(eng: dict[str, Path], tmp_path: Path):
    data = eng["en:eng"].read_bytes()
    bad = tmp_path / "eng.traineddata"
    bad.write_bytes(data[: len(data) // 2])
    avail = T.ENGINE.availability(_params(lang="eng"), {"en:eng": bad}, HERE_KEY)
    assert avail.state == "broken" and str(bad) in avail.detail and "truncated" in avail.detail
    with pytest.raises(EngineFailure) as info:
        T.ENGINE.open("tesseract", "reader", _params(lang="eng"), {"en:eng": bad})
    assert info.value.code == "model_broken" and str(bad) in info.value.message


def _link(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return dst


def test_sixteen_bit_grey_page_reads_like_eight_bit(eng: dict[str, Path], tmp_path: Path):
    """Grey text on a grey ground, saved as an 8-bit and a 16-bit PNG: the same read, not a blank page."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (1700, 560), 230)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(EN_LINES):
        draw.text((60, 60 + i * 110), text, font=ImageFont.load_default(size=42), fill=90)
    pages = {}
    for bits, im in (("8", img), ("16", Image.fromarray(np.asarray(img, dtype=np.uint16) * 257))):
        path = tmp_path / f"grey{bits}.png"
        im.save(path, format="PNG", dpi=(300, 300))
        pages[bits] = PageImage(0, path, img.width, img.height, 300.0, (0, 0, img.width * 0.24, img.height * 0.24))
    with pages["16"].open() as opened:
        assert opened.mode.startswith("I")
    a, b = _read(eng, pages["8"], lang="eng"), _read(eng, pages["16"], lang="eng")
    assert a.status == "ok" and word_recall(EN_LINES, a.text()) >= 0.9
    assert _comparable(b) == _comparable(a)


@pytest.mark.parametrize("folder", ["modèles ไทย", "ends-in-backslash\\"])
def test_a_model_in_an_unusual_folder_loads(eng: dict[str, Path], tmp_path: Path, folder: str):
    """DESIGN §9.3: a non-ASCII folder (8.3 short path on Windows), and on POSIX a name ending in a backslash."""
    if folder.endswith("\\") and os.sep == "\\":
        pytest.skip("a backslash is a separator on Windows")
    model = _link(eng["en:eng"], tmp_path / folder / "eng.traineddata")
    models = {"en:eng": model}
    assert T.ENGINE.availability(_params(lang="eng"), models, HERE_KEY) == Availability(True, "available", "")
    read = _read(models, latin_page(tmp_path), lang="eng")
    assert read.status == "ok" and word_recall(EN_LINES, read.text()) >= 0.9


def test_osd_on_an_upscaled_latin_page(tmp_path: Path):
    if T.PyTessBaseAPI is None:
        pytest.skip(f"tesserocr isn't importable: {T._IMPORT_ERROR}")
    osd_model = pinned_asset("en", "osd")
    from PIL import Image

    session = T.open_osd({"osd": osd_model})
    assert isinstance(session, OsdSession) and isinstance(session, T.TessOsd)
    try:
        # A 150 dpi render of 12 lines of 12 pt text, upscaled to about 300 dpi. (Four lines are too few:
        # at some sizes OSD then calls the script Cyrillic with a confidence under 1; measured 2026-11-05.)
        small = latin_page(tmp_path, (EN_LINES + OSD_EXTRA) * 2, size=25, dpi=150, page_dpi=150)
        with small.open() as img:
            up = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)   # ... upscaled to about 300
        found = session.detect(up, 300.0)
        print(f"\nOSD upscaled Latin page: {found}")
        assert found is not None and found.script == "Latin" and found.orient_deg == 0
        turned = up.rotate(90, expand=True, fillcolor=255)               # content turned 90 degrees counter-clockwise
        found = session.detect(turned, 300.0)
        assert found is not None and found.orient_deg == 270 and found.script == "Latin"
        assert session.detect(build.blank(800, 600, mode="L"), 300.0) is None   # too little text
    finally:
        session.close()


# --------------------------------------------------------------------------- real reads: the Thai and Hungarian packs

def _shared_tessdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *models: Path) -> None:
    """Link models from different pack folders into one shared tessdata folder, as the store does."""
    shared = tmp_path / "cache" / "tessdata"
    for src in models:
        _link(src, shared / src.name)
    monkeypatch.setattr(T, "_store_tessdata_dir", lambda: shared)


@pytest.mark.network
def test_per_instance_parameters_stay_apart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A tha+eng and a hun API alive at once, and across 4 threads, give the reads of their solo runs."""
    if T.PyTessBaseAPI is None:
        pytest.skip(f"tesserocr isn't importable: {T._IMPORT_ERROR}")
    files = pinned_assets("th:tha", "en:eng", "hu:hun", "th:noto-sans-thai", fetch=True)
    th_models = {"tha": files["th:tha"], "en:eng": files["en:eng"]}
    hu_models = {"hun": files["hu:hun"]}
    _shared_tessdata(tmp_path, monkeypatch, files["th:tha"], files["en:eng"])
    (tmp_path / "th").mkdir()
    (tmp_path / "hu").mkdir()
    th_page = thai_page(tmp_path / "th", files["th:noto-sans-thai"])
    hu_page = latin_page(tmp_path / "hu", HU_LINES, name="hu.png")
    jobs = {"th": (th_models, th_page, "tha+eng", 3), "hu": (hu_models, hu_page, "hun", 6)}

    solo = {k: _comparable(_read(m, p, lang=lang, psm=psm)) for k, (m, p, lang, psm) in jobs.items()}
    assert solo["th"]["status"] == "ok" and solo["hu"]["status"] == "ok"

    def open_job(k: str) -> Any:
        m, _, lang, psm = jobs[k]
        return T.ENGINE.open("tesseract", "reader", _params(lang=lang, psm=psm), m)

    th, hu = open_job("th"), open_job("hu")                    # alive at once, interleaved
    try:
        for _ in range(2):
            assert _comparable(th.read(th_page, timeout_s=120)) == solo["th"]
            assert _comparable(hu.read(hu_page, timeout_s=120)) == solo["hu"]
    finally:
        th.close()
        hu.close()

    kinds = ["th", "hu", "th", "hu"]
    results: list[list[dict[str, Any]]] = [[] for _ in kinds]
    errors: list[BaseException] = []
    gate = threading.Barrier(len(kinds))

    def work(i: int) -> None:
        session = open_job(kinds[i])
        try:
            gate.wait(timeout=60)
            for _ in range(3):
                results[i].append(_comparable(session.read(jobs[kinds[i]][1], timeout_s=120)))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            session.close()

    start = time.perf_counter()
    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(kinds))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    print(f"\n4 threads x 3 reads in {time.perf_counter() - start:.2f} s")
    assert not errors, repr(errors[0])
    for i, k in enumerate(kinds):
        assert len(results[i]) == 3 and all(r == solo[k] for r in results[i]), f"thread {i} ({k}) differs from its solo run"


@pytest.mark.network
def test_thai_no_intraword_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """On a shaped Thai page, the Thai-Thai gaps in the read equal the true word gaps."""
    if T.PyTessBaseAPI is None:
        pytest.skip(f"tesserocr isn't importable: {T._IMPORT_ERROR}")
    files = pinned_assets("th:tha", "en:eng", "th:noto-sans-thai", fetch=True)
    models = {"tha": files["th:tha"], "en:eng": files["en:eng"]}
    _shared_tessdata(tmp_path, monkeypatch, files["th:tha"], files["en:eng"])
    page = thai_page(tmp_path, files["th:noto-sans-thai"])
    read = _read(models, page, lang="tha+eng")
    assert read.status == "ok" and len(read.lines) == len(THAI_LINES)
    truth = sum(thai_gaps(ln) for ln in THAI_LINES)
    got = sum(thai_gaps(ln.text) for ln in read.lines)
    # what TEXTLINE text would have given, read straight from tesserocr on the same page
    plan = T._plan("tha+eng", models)
    assert plan.tessdata is not None
    api = T.PyTessBaseAPI(path=T._native_dir(plan.tessdata), lang="tha+eng", psm=3)
    try:
        with page.open() as img:
            api.SetImage(img)
        api.SetSourceResolution(300)
        assert api.Recognize()
        page_text = api.GetUTF8Text()
        it = api.GetIterator()
        it.Begin()
        parts = []
        while True:
            parts.append(it.GetUTF8Text(T.RIL.TEXTLINE) or "")
            if not it.Next(T.RIL.TEXTLINE):
                break
    finally:
        api.End()
    textline = sum(thai_gaps(p) for p in parts)
    print(f"\nThai-Thai gaps: true {truth}, pdfintake {got}, page text {thai_gaps(page_text)}, TEXTLINE text {textline}")
    assert got == truth
    assert textline > 5 * max(truth, 1)


@pytest.mark.network
def test_thai_read_with_models_in_two_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A pack's tha beside the base pack's eng in another folder: Tesseract loads both from the shared folder."""
    if T.PyTessBaseAPI is None:
        pytest.skip(f"tesserocr isn't importable: {T._IMPORT_ERROR}")
    files = pinned_assets("th:tha", "en:eng", "th:noto-sans-thai", fetch=True)
    tha = _link(files["th:tha"], tmp_path / "packs" / "th" / "tha.traineddata")
    eng = _link(files["en:eng"], tmp_path / "site" / "en" / "eng.traineddata")
    shared = tmp_path / "cache" / "tessdata"
    for src in (tha, eng):
        _link(src, shared / src.name)
    monkeypatch.setattr(T, "_store_tessdata_dir", lambda: shared)
    models = {"tha": tha, "en:eng": eng}
    plan = T._plan("tha+eng", models)
    assert plan.state == "available" and plan.tessdata == shared
    assert T.ENGINE.availability(_params(lang="tha+eng"), models, HERE_KEY) == Availability(True, "available", "")
    read = _read(models, thai_page(tmp_path, files["th:noto-sans-thai"]), lang="tha+eng")
    assert read.status == "ok" and len(read.lines) == len(THAI_LINES)
