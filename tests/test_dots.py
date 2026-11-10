"""ocr/engines/dots.py: thorough mode's reader (DESIGN §14.4), without the model.

Everything here runs on every OS: the image sizing, the parsing of the model's layout output (cut
off, fenced, looping), the loop watch that stops a read, the flattening of an element's Markdown,
HTML or LaTeX into lines, the boxes, availability, and a session's read path with a stand-in for
the loaded model. The real model's reads are in ``tests/test_dotsmlx.py`` and the contract suite,
which skip without the 4.4 GB download. All text is made up.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from pdfintake.errors import EngineFailure
from pdfintake.ocr.engines import dots
from pdfintake.ocr.engines.dotsmlx import DotsRead
from pdfintake.ocr.engines.dotsmlx.config import ImageConfig
from pdfintake.ocr.engines.dotsmlx.preprocess import smart_resize
from pdfintake.types import PageImage


def el(text: str | None, bbox: list[float] | None = None, category: str = "Text") -> dict[str, Any]:
    out: dict[str, Any] = {"bbox": bbox or [10, 10, 200, 40], "category": category}
    if text is not None:
        out["text"] = text
    return out


def dumps(elements: list[dict[str, Any]]) -> str:
    return json.dumps(elements, ensure_ascii=False)


# --------------------------------------------------------------------------- the image

@pytest.mark.parametrize("size,want", [
    ((2480, 3508), (1131, 1600)),          # A4 at 300 dpi: the long side cut to 1,600 px
    ((631, 3022), (613, 2936)),            # a till receipt: the 1.8 MP budget keeps its width
    ((800, 600), (800, 600)),              # never upscaled
])
def test_target_size(size: tuple[int, int], want: tuple[int, int]):
    assert dots.target_size(*size) == want


def test_a_page_is_cut_with_lanczos_and_turned_to_rgb():
    img = Image.new("L", (2480, 3508), 255)
    out = dots._model_image(img)
    assert out.mode == "RGB" and out.size == (1131, 1600)


# --------------------------------------------------------------------------- the model's output

def test_complete_elements_of_a_closed_and_a_cut_array():
    whole = dumps([el("Example one"), el("Example two")])
    assert dots.complete_elements(whole) == ([el("Example one"), el("Example two")], True)
    cut = whole[:-25]
    elements, closed = dots.complete_elements(cut)
    assert elements == [el("Example one")] and closed is False
    assert dots.complete_elements("```json\n" + whole + "\n```")[1] is True
    assert dots.complete_elements("no array here") == ([], False)


def test_read_elements_keeps_a_whole_read_and_drops_a_trailing_loop():
    clean = [el("Line"), el("Line"), el("Line"), el("Other")]
    assert dots.read_elements(dumps(clean), cut=False) == (clean, 0)
    loop = [el("Example heading"), *[el("Stuck line", [10, 50 + i, 200, 60 + i]) for i in range(7)]]
    kept, dropped = dots.read_elements(dumps(loop)[:-1] + ",", cut=True)
    assert [e["text"] for e in kept] == ["Example heading", "Stuck line"] and dropped == 6
    # a block of two lines repeated
    pair = [el("First"), *[el(t) for _ in range(4) for t in ("Alpha row", "Beta row")]]
    kept, dropped = dots.read_elements(dumps(pair), cut=True)
    assert [e["text"] for e in kept] == ["First", "Alpha row", "Beta row"] and dropped == 6


def test_two_repeats_are_not_a_loop():
    elements = [el("Total"), el("0.00"), el("0.00")]
    assert dots.trailing_repeat(elements) is None
    assert dots.read_elements(dumps(elements), cut=True) == (elements, 0)


def test_the_loop_watch_stops_on_repeated_elements():
    # A stuck model moves the box a little each time, so the text never repeats exactly.
    def stuck(i: int) -> str:
        return dumps([el("Stuck line", [10, 50 + 3 * i, 200, 60 + 3 * i])])[1:-1]

    watch = dots.LoopWatch()
    text = "[" + ", ".join(stuck(i) for i in range(5))
    assert watch(text) is False                       # five repeats: not yet
    text += ", " + stuck(5)
    assert watch(text) is True and watch.stopped == "loop"


def test_identical_repeated_elements_stop_as_a_phrase_loop():
    watch = dots.LoopWatch()
    same = dumps([el("Stuck line")])[1:-1]
    assert watch("[" + ", ".join(same for _ in range(4))) is True and watch.stopped == "phrase"


def test_the_loop_watch_stops_on_a_phrase_loop_but_not_on_dot_leaders():
    watch = dots.LoopWatch()
    head = '[{"bbox": [1, 2, 3, 4], "category": "Text", "text": "Terms: '
    assert watch(head + "the example clause applies " * 3) is False
    assert watch(head + "the example clause applies " * 12) is True and watch.stopped == "phrase"
    leaders = dots.LoopWatch()
    assert leaders(head + "Chapter one " + "." * 600) is False
    assert leaders(head + "<td></td>" * 80) is False


def test_flatten_turns_markup_into_lines():
    assert dots.flatten("## Example **Title**", "Section-header") == ["Example Title"]
    table = "<table><tr><td>Item</td><td>4521</td></tr><tr><td>Fee &amp; tax</td><td>12.50</td></tr></table>"
    assert dots.flatten(table, "Table") == ["Item 4521", "Fee & tax 12.50"]
    assert dots.flatten(r"$x^{2} + \frac{1}{2}$", "Formula") == ["x 2 + 1 2"]
    assert dots.flatten("one\n\ntwo", "Text") == ["one", "two"]
    assert dots.flatten("   ", "Text") == []


def test_elements_become_lines_in_the_renders_pixels():
    elements = [el("Example one\nExample two", [100, 200, 300, 240]), el(None, [0, 0, 50, 50], "Picture"),
                el("Bad box", [1, 2, 3]), el("Off the page", [900, 900, 2000, 2000])]
    lines = dots.elements_to_lines(elements, seen=(1000, 1000), page=(2000, 3000))
    assert [ln.text for ln in lines] == ["Example one", "Example two", "Off the page"]
    assert lines[0].box == ((200.0, 600.0), (600.0, 600.0), (600.0, 660.0), (200.0, 660.0))
    assert lines[1].box == ((200.0, 660.0), (600.0, 660.0), (600.0, 720.0), (200.0, 720.0))
    assert lines[2].box[2] == (2000.0, 3000.0)       # clamped to the render
    assert all(ln.conf == 1.0 for ln in lines)


# --------------------------------------------------------------------------- availability

def model_folder(tmp_path: Path) -> dict[str, Path]:
    folder = tmp_path / "model"
    folder.mkdir(parents=True)
    out = {}
    for name in (*dots.REQUIRED_FILES, "README.md"):
        (folder / name).write_bytes(b"made-up model file")
        out[name] = folder / name
    return out


@pytest.fixture
def apple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dots, "_on_apple_silicon", lambda: True)
    monkeypatch.setattr(dots, "_missing_libraries", lambda: [])


def test_availability(tmp_path: Path, apple: None, monkeypatch: pytest.MonkeyPatch):
    models = model_folder(tmp_path)
    eng = dots.ENGINE
    assert eng.availability({}, models, "darwin-arm64").state == "available"
    assert eng.availability({}, models, "linux-x86_64").state == "not_on_platform"
    without = {k: v for k, v in models.items() if k != "model.safetensors"}
    assert eng.availability({}, without, "darwin-arm64").state == "missing_model"
    monkeypatch.setattr(dots, "_missing_libraries", lambda: ["mlx"])
    broken = eng.availability({}, models, "darwin-arm64")
    assert broken.state == "broken" and "pdfintake[thorough]" in broken.detail


def test_off_apple_silicon_a_session_reads_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dots, "_on_apple_silicon", lambda: False)
    page = page_image(tmp_path)
    read = dots.ENGINE.open("dots", "reader", {}, {}).read(page, timeout_s=10)
    assert read.status == "unavailable" and read.lines == []
    assert dots.ENGINE.contract_case() is None
    assert dots.canary({}) is not None


def test_open_names_a_missing_model(tmp_path: Path, apple: None):
    with pytest.raises(EngineFailure) as info:
        dots.ENGINE.open("dots", "reader", {}, {})
    assert info.value.code == "missing_model"


# --------------------------------------------------------------------------- a read, with a stand-in model

class FakeModel:
    """Answers like DotsModel.read: releases its text a few characters at a time to ``stop``."""

    def __init__(self, text: str = "", *, stopped: str = "eos", error: BaseException | None = None) -> None:
        self.config = SimpleNamespace(image=ImageConfig())
        self.text = text
        self.stopped = stopped
        self.error = error
        self.calls: list[tuple[tuple[int, int], str, int]] = []

    def read(self, image: Image.Image, prompt: str, max_tokens: int, stop: Any = None,
             deadline: float | None = None) -> DotsRead:
        self.calls.append((image.size, prompt, max_tokens))
        if self.error is not None:
            raise self.error
        if deadline is not None and time.monotonic() >= deadline:
            return DotsRead("", 0, "deadline")
        out, n = "", 0
        for i in range(0, len(self.text), 7):
            out += self.text[i:i + 7]
            n += 1
            if stop is not None and stop(out):
                return DotsRead(out, n, "stop")
        return DotsRead(out, n, self.stopped)  # type: ignore[arg-type]


def page_image(tmp_path: Path, size: tuple[int, int] = (1700, 2200)) -> PageImage:
    path = tmp_path / "page.png"
    Image.new("L", size, 255).save(path)
    return PageImage(0, path, size[0], size[1], 200.0, (0.0, 0.0, size[0] * 0.36, size[1] * 0.36))


def session_with(model: FakeModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dots.DotsSession:
    monkeypatch.setattr(dots, "_load", lambda folder: model)
    return dots.ENGINE.open("dots", "reader", {}, model_folder(tmp_path))


def test_a_read_scales_boxes_to_the_render(tmp_path: Path, apple: None, monkeypatch: pytest.MonkeyPatch):
    model = FakeModel(dumps([el("Invoice 2026-0457", [100, 200, 500, 240]), el(None, [0, 0, 9, 9], "Picture")]))
    page = page_image(tmp_path)
    read = session_with(model, tmp_path, monkeypatch).read(page, timeout_s=60)
    assert read.status == "ok" and read.engine == "dots" and [ln.text for ln in read.lines] == ["Invoice 2026-0457"]
    sent = dots.target_size(1700, 2200)
    assert model.calls == [(sent, dots.PROMPT, dots.MAX_TOKENS)]
    cfg = ImageConfig()
    seen_h, seen_w = smart_resize(sent[1], sent[0], cfg.patch_size * cfg.merge_size, cfg.min_pixels, cfg.max_pixels)
    assert read.meta["size"] == [seen_w, seen_h] and read.meta["stopped"] == "eos" and read.meta["elements"] == 2
    (x0, y0), _, (x1, y1), _ = read.lines[0].box
    assert (x0, y0) == (round(100 * 1700 / seen_w, 1), round(200 * 2200 / seen_h, 1))
    assert (x1, y1) == (round(500 * 1700 / seen_w, 1), round(240 * 2200 / seen_h, 1))


def test_a_looping_read_stops_and_keeps_one_copy(tmp_path: Path, apple: None, monkeypatch: pytest.MonkeyPatch):
    stuck = [el("Example heading"), *[el("Stuck line", [10, 50 + i, 200, 60 + i]) for i in range(40)]]
    model = FakeModel(dumps(stuck), stopped="length")
    read = session_with(model, tmp_path, monkeypatch).read(page_image(tmp_path), timeout_s=60)
    assert read.status == "ok" and [ln.text for ln in read.lines] == ["Example heading", "Stuck line"]
    assert read.meta["stopped"] == "loop" and read.meta["repeats_dropped"] == 5


def test_empty_timeout_and_failure(tmp_path: Path, apple: None, monkeypatch: pytest.MonkeyPatch):
    page = page_image(tmp_path)
    assert session_with(FakeModel("[]"), tmp_path / "a", monkeypatch).read(page, timeout_s=60).status == "empty"
    late = session_with(FakeModel("[]"), tmp_path / "b", monkeypatch).read(page, timeout_s=0.0)
    assert late.status == "timeout"
    failed = session_with(FakeModel(error=RuntimeError("made-up GPU error")), tmp_path / "c", monkeypatch)
    out = failed.read(page, timeout_s=60)
    assert out.status == "failed" and "made-up GPU error" in (out.error or "")
    page.path.unlink()
    assert failed.read(page, timeout_s=60).status == "failed"


def test_the_canary(tmp_path: Path, apple: None, monkeypatch: pytest.MonkeyPatch):
    models = model_folder(tmp_path)
    monkeypatch.setattr(dots, "_load", lambda folder: FakeModel(dumps([el(dots.CANARY_TEXT)])))
    assert dots.canary(models) is None
    monkeypatch.setattr(dots, "_load", lambda folder: FakeModel(dumps([el("Something else")])))
    assert dots.canary(models) == "the test page's text didn't come back"
    monkeypatch.setattr(dots, "_load", lambda folder: FakeModel("[]"))
    assert "empty" in (dots.canary(models) or "")


def test_the_model_is_loaded_once_per_folder(tmp_path: Path, apple: None, monkeypatch: pytest.MonkeyPatch):
    loads: list[Path] = []

    class Port:
        @staticmethod
        def load(folder: Path) -> FakeModel:
            loads.append(folder)
            return FakeModel("[]")

    import pdfintake.ocr.engines as engines_pkg

    monkeypatch.setattr(engines_pkg, "dotsmlx", Port, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "pdfintake.ocr.engines.dotsmlx", Port)
    dots._reset_for_tests()
    try:
        models = model_folder(tmp_path)
        a = dots.ENGINE.open("dots", "reader", {}, models)
        b = dots.ENGINE.open("dots", "checker", {}, models)
        assert a.model is b.model and len(loads) == 1
    finally:
        dots._reset_for_tests()


def test_a_pack_cant_name_thorough_modes_reader():
    """dots is registered (thorough mode adds it, DESIGN §14), but no pack may list it."""
    from pdfintake.langpacks.manifest import check
    from pdfintake.ocr import engines

    assert "dots" in engines.names() and "dots" in engines.NOT_FOR_PACKS
    fx = Path(__file__).resolve().parent / "fixtures" / "packs" / "en" / "manifest.json"
    data = json.loads(fx.read_text(encoding="utf-8"))
    data["readers"]["darwin-arm64"] = {
        "engines": [{"id": "dots", "engine": "dots", "role": "reader", "params": {}, "models": []}],
        "pick": {"rule": "single", "version": 1, "params": {"reader": "dots"}}, "fallback": "none"}
    found = check(data, base=None)
    assert any(p.startswith('$.readers["darwin-arm64"].engines[0].engine:') and "thorough mode" in p for p in found), found


def test_the_engine_declares_what_the_registry_needs():
    eng = dots.ENGINE
    assert eng.name == "dots" and eng.platforms == frozenset({"darwin-arm64"})
    assert eng.session_per_thread is False and eng.MODEL_KINDS == frozenset()
    assert eng.candidate_params("en", "Latin", []) == []
