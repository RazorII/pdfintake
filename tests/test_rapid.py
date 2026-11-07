"""The RapidOCR engine (ocr/engines/rapid.py).

Model-free tests run everywhere. Tests that read need the pinned RapidOCR models: the base pack's
det, cls and latin rec and, for the Thai probe case, the th pack's Thai rec. They come from
``conftest.pinned_asset(pack, key)`` (``<PDFINTAKE_TEST_MODELS or ~/.pdfintake-dev/models>/<pack>/<file>``,
checked against the manifest's size and SHA-256; ``tools/fetch_test_models.py`` fetches them);
without them those tests skip.

What is proved: every model path is explicit, nothing touches the network at open or read (the
conftest socket guard records every attempt; it can't prove the paths explicit on its own, because
rapidocr's wheel bundles default models it loads without a network call, so the
``cfg.*.model_path`` assertions are what prove it), ``min_score`` reaches rapidocr, boxes come back as clockwise quads in pixels of the
page image, broken and keyless models are refused before rapidocr could download anything, the
contract suite passes on the real engine, and ``open_voter`` reads upright text and not text
turned 180 degrees.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont, features

from conftest import pinned_assets
from fixtures import build
from pdfintake.errors import EngineFailure, PackError
from pdfintake.ocr import engines
from pdfintake.ocr.engines import rapid
from pdfintake.params import ValidationContext
from pdfintake.types import EngineRead, PageImage

KEYS = {"det": "rapid-det", "cls": "rapid-cls", "latin": "rapid-rec-latin", "th": "rapid-rec-th"}
PACK_OF = {"det": "en", "cls": "en", "latin": "en", "th": "th"}     # which manifest pins each role
LINES = ("The example invoice lists seven items", "Total amount 4521 units", "Reference number AB123456")


def _ctx(models: dict[str, Path], kinds: dict[str, str]) -> ValidationContext:
    return ValidationContext(use_ids=frozenset({"rapidocr"}), asset_keys=frozenset(models), asset_kinds=kinds,
                             scripts=frozenset({"Latin", "Thai"}), where="test.params")


def _need(*roles: str) -> dict[str, Path]:
    found = pinned_assets(*(f"{PACK_OF[r]}:{KEYS[r]}" for r in roles))
    return {r: found[f"{PACK_OF[r]}:{KEYS[r]}"] for r in roles}


def _models(*roles: str) -> dict[str, Path]:
    found = _need(*roles)
    return {KEYS[r]: found[r] for r in roles}


def _params(rec: str = "rapid-rec-latin", **extra: Any) -> dict[str, Any]:
    models = {k: Path("x") for k in KEYS.values()}
    kinds = {"rapid-det": "onnx_det", "rapid-cls": "onnx_cls", "rapid-rec-latin": "onnx_rec", "rapid-rec-th": "onnx_rec"}
    return rapid.ENGINE.PARAMS.validate({"rec": rec, **extra}, _ctx(models, kinds))


def _font(size: int = 44) -> Any:
    if not features.check("freetype2"):
        pytest.skip("Pillow was built without FreeType, so it can't draw a readable test page")
    return ImageFont.load_default(size=size)


def _text_page(folder: Path, *, turn: int = 0, mode: str = "RGB", name: str = "page.png") -> tuple[PageImage, list]:
    """Made-up lines on white; returns the page and each line's drawn bounding box (before ``turn``)."""
    img = Image.new("RGB", (1700, 1100), "white")
    draw = ImageDraw.Draw(img)
    font = _font()
    boxes = []
    for i, text in enumerate(LINES):
        xy = (120, 150 + i * 200)
        draw.text(xy, text, fill="black", font=font)
        boxes.append(draw.textbbox(xy, text, font=font))
    if turn:
        img = img.rotate(turn, expand=True)
    if mode != "RGB":
        img = img.convert(mode)
    return build.page_image(folder, image=img, dpi=300, name=name), boxes


def _read(session: Any, page: PageImage, timeout_s: float = 120.0) -> EngineRead:
    try:
        return session.read(page, timeout_s=timeout_s)
    finally:
        session.close()


# --------------------------------------------------------------------------- without models

def test_registered_and_protocol():
    eng = engines.get("rapidocr")
    assert eng is rapid.ENGINE
    assert isinstance(eng, engines.Engine)
    assert eng.MODEL_KINDS == frozenset({"onnx_det", "onnx_cls", "onnx_rec"})
    assert eng.session_per_thread is True
    assert eng.platforms == frozenset({"darwin-arm64", "win-amd64", "linux-x86_64", "linux-aarch64"})
    import importlib.metadata

    v = eng.version()
    assert v == importlib.metadata.version("rapidocr") and v.split(".")[0] == "3"


def test_params_table():
    p = _params()
    assert p == {"rec": "rapid-rec-latin", "use_cls": False, "min_score": 0.5}
    assert _params("rapid-rec-th", use_cls=True, min_score=0.7)["min_score"] == 0.7
    models = {"rapid-det": Path("x"), "rapid-rec-th": Path("y")}
    kinds = {"rapid-det": "onnx_det", "rapid-rec-th": "onnx_rec"}
    for bad in ({}, {"rec": "rapid-det"}, {"rec": "rapid-rec-th", "min_score": 1.5},
                {"rec": "rapid-rec-th", "lang": "th"}, {"rec": "nope"}, {"rec": "rapid-rec-th", "use_cls": "no"}):
        with pytest.raises(PackError):
            rapid.ENGINE.PARAMS.validate(bad, _ctx(models, kinds))


def test_not_on_platform(tmp_path: Path):
    avail = rapid.ENGINE.availability(_params(), {}, "plan9-mips")
    assert not avail.ok and avail.state == "not_on_platform" and "plan9-mips" in avail.detail


def test_missing_and_garbage_models_are_named(tmp_path: Path):
    zero = tmp_path / "zero.onnx"
    zero.write_bytes(b"")
    html = tmp_path / "html.onnx"
    html.write_bytes(b"<!DOCTYPE html><html><body>Not Found</body></html>")
    cases = [
        ({}, "missing_model", "rapid-rec-latin"),
        ({"rapid-rec-latin": tmp_path / "absent.onnx"}, "missing_model", "rapid-rec-latin"),
        ({"rapid-rec-latin": zero}, "broken", "zero.onnx"),
        ({"rapid-rec-latin": html}, "broken", "html.onnx"),
    ]
    for models, state, named in cases:
        avail = rapid.ENGINE.availability(_params(), models, "linux-x86_64")
        assert not avail.ok and avail.state == state, (models, avail)
        assert named in avail.detail
        with pytest.raises(EngineFailure) as info:
            rapid.ENGINE.open("rapidocr", "checker", _params(), models)
        assert named in info.value.message


def test_quad_conversion_is_clockwise_and_clamped():
    q = rapid._quad([[10, 5], [100, 5], [100, 40], [10, 40]], 200, 100)
    assert q == ((10.0, 5.0), (100.0, 5.0), (100.0, 40.0), (10.0, 40.0))
    # anticlockwise input, starting elsewhere, partly outside the image
    q = rapid._quad([[110, 50], [-4, 50], [-4, 10], [110, 10]], 100, 60)
    assert q == ((0.0, 10.0), (100.0, 10.0), (100.0, 50.0), (0.0, 50.0))
    area = sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))
    assert area > 0


def test_lines_from_rapidocr_output():
    import numpy as np

    class Out:
        txts = ("first", " ", "third", "fourth")
        scores = (0.98, 0.9, 1.2, float("nan"))
        boxes = np.array([[[1, 1], [50, 1], [50, 9], [1, 9]], [[0, 0], [1, 0], [1, 1], [0, 1]],
                          [[1, 20], [60, 20], [60, 30], [1, 30]], [[1, 40], [60, 40], [60, 50], [1, 50]]], dtype=float)
    lines = rapid._Session._lines(Out(), 64, 64)
    assert [ln.text for ln in lines] == ["first", "third", "fourth"]
    assert [ln.conf for ln in lines] == [0.98, 1.0, 0.0]
    assert lines[0].box == ((1.0, 1.0), (50.0, 1.0), (50.0, 9.0), (1.0, 9.0)) and lines[0].words == ()
    assert rapid._Session._lines(type("Empty", (), {"txts": None, "scores": None, "boxes": None})(), 10, 10) == []


@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA", "LA", "I;16"])
def test_image_conversion_to_bgr(mode: str):
    img = Image.new("RGB", (8, 4), (10, 20, 30))
    if mode == "I;16":
        src = Image.new("I;16", (8, 4), 0x8040)
        arr = rapid._to_bgr(src)
        assert arr.shape == (4, 8, 3) and arr.dtype.name == "uint8" and int(arr[0, 0, 0]) == 0x80
        return
    src = img.convert(mode)
    if mode in ("RGBA", "LA"):
        src.putalpha(0)                          # fully transparent: composited over white
    arr = rapid._to_bgr(src)
    assert arr.shape == (4, 8, 3) and arr.dtype.name == "uint8" and arr.flags["C_CONTIGUOUS"]
    if mode == "RGB":
        assert tuple(int(v) for v in arr[0, 0]) == (30, 20, 10)       # BGR
    if mode in ("RGBA", "LA"):
        assert tuple(int(v) for v in arr[0, 0]) == (255, 255, 255)


def test_candidate_params():
    class A:
        def __init__(self, key: str, engine: str, kind: str) -> None:
            self.key, self.engine, self.kind = key, engine, kind
    assets = [A("rapid-rec-th", "rapidocr", "onnx_rec"), A("rapid-det", "rapidocr", "onnx_det"),
              A("tha", "tesseract", "traineddata"), A("a-rec", "rapidocr", "onnx_rec")]
    assert rapid.ENGINE.candidate_params("th", "Thai", assets) == [{"rec": "a-rec"}, {"rec": "rapid-rec-th"}]
    assert rapid.ENGINE.candidate_params("th", "Thai", []) == []


def test_contract_case_is_none_without_a_base_pack(monkeypatch: pytest.MonkeyPatch):
    import pdfintake.langpacks as lp

    def broken_base():  # pragma: no cover - called through contract_case
        raise RuntimeError("no base pack here")
    fake = type(sys)("pdfintake.langpacks.store")
    fake.base = broken_base
    monkeypatch.setitem(sys.modules, "pdfintake.langpacks.store", fake)
    monkeypatch.setattr(lp, "store", fake, raising=False)
    assert rapid.ENGINE.contract_case() is None


def _fake_session(out: Any) -> Any:
    """A _Session around a stand-in for rapidocr's callable, for tests that need no models."""
    session = object.__new__(rapid._Session)
    session.use_id, session.role, session.spec = "rapidocr", "checker", "rapidocr:fake"
    session._use_cls, session._rec_ref, session._closed = False, "rapid-rec-latin", False
    session._lock = threading.Lock()
    session._ocr = lambda arr, use_cls=False: out  # type: ignore[assignment]
    return session


@pytest.mark.parametrize("timeout_s", [float("inf"), 1e12, threading.TIMEOUT_MAX * 10])
def test_huge_timeouts_never_raise(tmp_path: Path, timeout_s: float):
    """Lock.acquire and Thread.join refuse waits over TIMEOUT_MAX; read must clamp, not raise."""
    import numpy as np

    class Out:
        txts = ("Invented line",)
        scores = (0.97,)
        boxes = np.array([[[1, 1], [50, 1], [50, 9], [1, 9]]], dtype=float)
    page = build.page_image(tmp_path, image=build.blank(64, 64), dpi=300, name="huge.png")
    read = _read(_fake_session(Out()), page, timeout_s=timeout_s)
    assert read.status == "ok" and [ln.text for ln in read.lines] == ["Invented line"]


def test_voter_takes_only_its_two_keys(tmp_path: Path):
    """No model loads here: absent keys fail before any file is opened, other entries never count."""
    junk = tmp_path / "junk.traineddata"
    junk.write_bytes(b"\x00" * 64)
    for models in ({}, {"en:eng": junk, "en:osd": junk},
                   {"rapid-det": tmp_path / "absent.onnx", "rapid-rec-th": junk},          # Thai rec only
                   {"rapid-rec-latin": junk, "other-det": junk}):                          # det under another key
        with pytest.raises(EngineFailure) as info:
            rapid.open_voter(models)
        assert info.value.detail["state"] == "missing_model", (models, info.value.message)
        assert "rapid-det" in info.value.message or "rapid-rec-latin" in info.value.message


def test_module_never_names_a_download_host():
    source = Path(rapid.__file__).read_text(encoding="utf-8")
    assert "http://" not in source and "https://" not in source


# --------------------------------------------------------------------------- with the pinned models

def test_explicit_paths_no_network_and_quads(tmp_path: Path, network_attempts: list):
    models = _models("det", "cls", "latin")
    page, drawn = _text_page(tmp_path)
    assert rapid.ENGINE.availability(_params(), models, "linux-x86_64").ok
    session = rapid.ENGINE.open("rapidocr", "checker", _params(), models)
    assert session._ocr.cfg.Det.model_path == str(models["rapid-det"])
    assert session._ocr.cfg.Rec.model_path == str(models["rapid-rec-latin"])
    assert session._ocr.cfg.Cls.model_path == str(models["rapid-cls"])
    read = _read(session, page)
    assert network_attempts == [], network_attempts
    assert read.status == "ok" and read.engine == "rapidocr" and read.use_id == "rapidocr" and read.role == "checker"
    assert read.spec == "rapidocr:rapid-rec-latin:cls0:min0.5"
    assert [ln.text for ln in read.lines] == list(LINES)
    assert read.meta["version"] == rapid.ENGINE.version() and 0.9 < read.meta["mean_conf"] <= 1.0
    for ln, (x0, y0, x1, y1) in zip(read.lines, drawn):
        assert isinstance(ln.box, tuple) and len(ln.box) == 4
        (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = ln.box
        assert tlx < trx and bly > tly, ln.box                             # clockwise from top-left
        xs, ys = [p[0] for p in ln.box], [p[1] for p in ln.box]
        assert abs(min(xs) - x0) < 12 and abs(max(xs) - x1) < 12, (ln.box, (x0, y0, x1, y1))
        assert abs(min(ys) - y0) < 14 and abs(max(ys) - y1) < 14, (ln.box, (x0, y0, x1, y1))
        assert 0.9 <= ln.conf <= 1.0


def test_boxes_are_in_pixels_of_a_large_render(tmp_path: Path):
    """rapidocr shrinks pages over 2000 px internally; boxes must come back in the page's pixels."""
    models = _models("det", "latin")
    small, drawn = _text_page(tmp_path, name="small.png")
    with small.open() as img:
        big = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    page = build.page_image(tmp_path, image=big, dpi=600, name="big.png")
    read = _read(rapid.ENGINE.open("rapidocr", "checker", _params(), models), page)
    assert [ln.text for ln in read.lines] == list(LINES)
    for ln, (x0, y0, x1, y1) in zip(read.lines, drawn):
        xs, ys = [p[0] for p in ln.box], [p[1] for p in ln.box]
        assert abs(min(xs) - 2 * x0) < 24 and abs(max(xs) - 2 * x1) < 24
        assert abs(min(ys) - 2 * y0) < 28 and abs(max(ys) - 2 * y1) < 28


def test_grey_and_alpha_pages_read_the_same(tmp_path: Path):
    models = _models("det", "latin")
    rgb, _ = _text_page(tmp_path, name="rgb.png")
    grey, _ = _text_page(tmp_path, mode="L", name="grey.png")
    rgba, _ = _text_page(tmp_path, mode="RGBA", name="rgba.png")
    texts = []
    for page in (rgb, grey, rgba):
        read = _read(rapid.ENGINE.open("rapidocr", "checker", _params(), models), page)
        texts.append([ln.text for ln in read.lines])
    assert texts[0] == texts[1] == texts[2] == list(LINES)


def test_min_score_reaches_rapidocr(tmp_path: Path):
    """min_score is rapidocr's text_score: at 1.0 no line clears it, at 0.0 every line does."""
    models = _models("det", "latin")
    page, _ = _text_page(tmp_path)
    strict = rapid.ENGINE.open("rapidocr", "checker", _params(min_score=1.0), models)
    assert strict._ocr.cfg.Global.text_score == 1.0
    read = _read(strict, page)
    assert read.status == "empty" and read.lines == [] and read.spec.endswith(":min1")
    loose = rapid.ENGINE.open("rapidocr", "checker", _params(min_score=0.0), models)
    assert loose._ocr.cfg.Global.text_score == 0.0
    read = _read(loose, page)
    assert read.status == "ok" and [ln.text for ln in read.lines] == list(LINES)
    mid = rapid.ENGINE.open("rapidocr", "checker", _params(min_score=0.7), models)
    try:
        assert mid._ocr.cfg.Global.text_score == 0.7
    finally:
        mid.close()


def test_classifier_is_used_only_when_asked(tmp_path: Path):
    models = _models("det", "cls", "latin")
    page, _ = _text_page(tmp_path)
    on = rapid.ENGINE.open("rapidocr", "checker", _params(use_cls=True), models)
    assert on._use_cls and on._ocr.text_cls is not None
    read = _read(on, page)
    assert read.spec.endswith(":cls1:min0.5") and [ln.text for ln in read.lines] == list(LINES)
    no_cls = {k: v for k, v in models.items() if k != "rapid-cls"}
    avail = rapid.ENGINE.availability(_params(use_cls=True), no_cls, "linux-x86_64")
    assert avail.state == "missing_model" and "classifier" in avail.detail
    off = rapid.ENGINE.open("rapidocr", "probe", _params(), no_cls)        # the Thai probe's shape: no cls
    assert not off._use_cls and off._ocr.text_cls is None
    off.close()


def test_base_prefixed_keys_resolve(tmp_path: Path):
    found = _need("det", "latin")
    models = {"en:rapid-det": found["det"], "en:rapid-rec-latin": found["latin"]}
    params = rapid.ENGINE.PARAMS.validate({"rec": "en:rapid-rec-latin"},
                                          _ctx(models, {"en:rapid-det": "onnx_det", "en:rapid-rec-latin": "onnx_rec"}))
    assert rapid.ENGINE.availability(params, models, "darwin-arm64").ok
    # and a store that drops the prefix still serves a prefixed ref
    stripped = {"rapid-det": found["det"], "rapid-rec-latin": found["latin"]}
    assert rapid.ENGINE.availability(params, stripped, "darwin-arm64").ok


def test_wrong_kind_and_ambiguous_models_are_broken(tmp_path: Path):
    found = _need("det", "cls", "latin")
    raw = {"rec": "rapid-det", "use_cls": False, "min_score": 0.5}       # what the loader would refuse
    avail = rapid.ENGINE.availability(raw, {"rapid-det": found["det"]}, "linux-x86_64")
    assert avail.state == "broken" and "not a recognition model" in avail.detail
    twice = {"rapid-det": found["det"], "other-det": found["det"], "rapid-rec-latin": found["latin"]}
    avail = rapid.ENGINE.availability(_params(), twice, "linux-x86_64")
    assert avail.state == "broken" and "detection models" in avail.detail
    only_rec = {"rapid-rec-latin": found["latin"], "rapid-cls": found["cls"]}
    avail = rapid.ENGINE.availability(_params(), only_rec, "linux-x86_64")
    assert avail.state == "missing_model" and "detection" in avail.detail


def test_truncated_model_is_broken(tmp_path: Path):
    found = _need("det", "latin")
    cut = tmp_path / "cut.onnx"
    cut.write_bytes(found["det"].read_bytes()[:100_000])
    avail = rapid.ENGINE.availability(_params(), {"rapid-det": cut, "rapid-rec-latin": found["latin"]}, "linux-x86_64")
    assert avail.state == "broken" and "cut.onnx" in avail.detail


def test_recogniser_without_keys_is_refused_not_downloaded(tmp_path: Path, network_attempts: list):
    """rapidocr downloads a character list for a recogniser that carries none; we refuse it first."""
    found = _need("det", "latin")
    data = found["latin"].read_bytes()
    marker = b"\x0a\x09character"            # StringStringEntryProto key "character"
    assert marker in data
    keyless = tmp_path / "keyless.onnx"
    keyless.write_bytes(data.replace(marker, b"\x0a\x09characteX"))
    models = {"rapid-det": found["det"], "rapid-rec-latin": keyless}
    avail = rapid.ENGINE.availability(_params(), models, "linux-x86_64")
    assert avail.state == "broken" and "character list" in avail.detail
    with pytest.raises(EngineFailure):
        rapid.ENGINE.open("rapidocr", "checker", _params(), models)
    assert network_attempts == []


def test_timeout_reports_timeout_and_session_recovers(tmp_path: Path):
    models = _models("det", "latin")
    page, _ = _text_page(tmp_path)
    session = rapid.ENGINE.open("rapidocr", "checker", _params(), models)
    try:
        start = time.perf_counter()
        quick = session.read(page, timeout_s=0.001)
        assert quick.status == "timeout" and quick.error and time.perf_counter() - start < 1.0
        assert session.read(page, timeout_s=0).status == "timeout"
        later = session.read(page, timeout_s=120)                 # waits for the timed-out read, then reads
        assert later.status == "ok" and [ln.text for ln in later.lines] == list(LINES)
    finally:
        session.close()


def test_blank_page_is_empty_and_errors_are_failed(tmp_path: Path):
    models = _models("det", "latin")
    blank = build.page_image(tmp_path, image=build.blank(900, 600), dpi=300, name="blank.png")
    read = _read(rapid.ENGINE.open("rapidocr", "checker", _params(), models), blank)
    assert read.status == "empty" and read.lines == [] and read.meta["mean_conf"] is None
    session = rapid.ENGINE.open("rapidocr", "checker", _params(), models)
    try:
        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("inference exploded")
        type_patch = type(session._ocr)       # call syntax looks __call__ up on the type
        original = type_patch.__call__
        type_patch.__call__ = boom            # type: ignore[method-assign]
        try:
            read = session.read(blank, timeout_s=30)
        finally:
            type_patch.__call__ = original    # type: ignore[method-assign]
        assert read.status == "failed" and "inference exploded" in read.error
        lost = PageImage(index=0, path=tmp_path / "gone.png", width=10, height=10, dpi=300, cropbox=(0, 0, 2.4, 2.4))
        read = session.read(lost, timeout_s=30)
        assert read.status == "failed" and "page image" in read.error
    finally:
        session.close()
    with pytest.raises(EngineFailure):
        session.read(blank, timeout_s=30)


def test_thai_probe_shape_reads(tmp_path: Path, network_attempts: list):
    """The Thai pack's probe use: det + the Thai recogniser, no classifier listed."""
    models = _models("det", "th")
    page, _ = _text_page(tmp_path)
    read = _read(rapid.ENGINE.open("probe", "probe", _params("rapid-rec-th"), models), page)
    assert read.status in ("ok", "empty") and read.role == "probe"
    assert read.spec == "rapidocr:rapid-rec-th:cls0:min0.5"
    assert network_attempts == []


def test_open_voter(tmp_path: Path, network_attempts: list):
    found = _need("det", "cls", "latin", "th")
    upright, _ = _text_page(tmp_path, name="up.png")
    flipped, _ = _text_page(tmp_path, turn=180, name="flip.png")
    models = {"en:rapid-det": found["det"], "en:rapid-cls": found["cls"],
              "en:rapid-rec-latin": found["latin"], "rapid-rec-th": found["th"]}
    voter = rapid.open_voter(models)
    try:
        assert not voter._use_cls and voter._ocr.cfg.Rec.model_path == str(found["latin"])
        assert voter.use_id == "orient_vote" and voter.spec == "rapidocr:voter:en:rapid-rec-latin"
        up = voter.read(upright, timeout_s=120)
        down = voter.read(flipped, timeout_s=120)
    finally:
        voter.close()
    assert [ln.text for ln in up.lines] == list(LINES)
    score = lambda r: sum(len(ln.text) * ln.conf for ln in r.lines)  # noqa: E731
    assert score(up) >= 10 * max(score(down), 1e-9) or not down.lines
    assert network_attempts == []
    with pytest.raises(EngineFailure):
        rapid.open_voter({"rapid-det": found["det"]})
    with pytest.raises(EngineFailure):
        rapid.open_voter({"rapid-det": found["det"], "a": found["latin"], "b": found["th"]})
    with pytest.raises(EngineFailure) as info:                     # det + Thai rec is not a voter
        rapid.open_voter({"rapid-det": found["det"], "rapid-rec-th": found["th"]})
    assert info.value.detail["state"] == "missing_model" and "rapid-rec-latin" in info.value.message


def test_open_voter_ignores_the_rest_of_the_base_pack(tmp_path: Path, network_attempts: list):
    """The base pack's whole model map, Tesseract's osd and eng included, makes a voter."""
    found = _need("det", "cls", "latin", "th")
    junk = tmp_path / "invented.traineddata"
    junk.write_bytes(b"\x00" * 4096)                    # not ONNX: opening it would fail
    upright, _ = _text_page(tmp_path, name="up.png")
    models = {"en:rapid-det": found["det"], "en:rapid-cls": found["cls"], "en:rapid-rec-latin": found["latin"],
              "en:eng": junk, "en:osd": junk, "th:rapid-rec-th": found["th"]}
    voter = rapid.open_voter(models)
    assert voter._ocr.cfg.Det.model_path == str(found["det"])
    assert voter._ocr.cfg.Rec.model_path == str(found["latin"]) and not voter._use_cls
    read = _read(voter, upright)
    assert [ln.text for ln in read.lines] == list(LINES)
    assert network_attempts == []


# --------------------------------------------------------------------------- the contract suite, on the real engine

def _contract_module() -> Any:
    name = "_rapid_engine_contract"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "contract" / "test_engine_contract.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # dataclasses look their module up while it executes
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


CONTRACT_TESTS = ("test_member_declares_the_protocol", "test_contract_case_is_available",
                  "test_read_returns_quads_inside_the_image", "test_blank_image_is_empty",
                  "test_parallel_sessions_give_identical_reads", "test_same_input_same_output",
                  "test_timeout_is_honoured", "test_missing_model_is_named", "test_unavailable_is_never_failed")


@pytest.mark.parametrize("name", CONTRACT_TESTS)
def test_contract_suite_on_the_real_engine(name: str, tmp_path: Path, network_attempts: list):
    found = _need("det", "latin")
    contract = _contract_module()
    case = rapid._contract_case(found["det"], found["latin"])
    if case is None or case.page is None:
        pytest.skip("Pillow has no FreeType, so the contract page can't be drawn")
    params = rapid.ENGINE.PARAMS.validate(case.params, contract._context("rapidocr", case))
    member = contract.Member("rapidocr", rapid.ENGINE, case, "linux-x86_64", params, tmp_path)
    getattr(contract, name)(member)
    assert network_attempts == []
