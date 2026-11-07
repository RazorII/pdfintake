"""The Apple Vision engine (DESIGN §7.3, "How Vision is tested on macOS").

Two kinds of test live here:

* **Stubbed**, unmarked, on every OS: the parameter table, ``not_on_platform`` elsewhere, the
  language gate, the box conversion, the timeout path with a stubbed hang, and failure handling.
  These patch the module's few pyobjc-touching functions, so no Vision is needed.
* **Real**, marked ``vision``, on macOS: the import, the supported-language list (printed to the
  log), ``{}`` options rejected and ``None`` accepted, ``hu-HU`` caught by ``availability`` before
  any ``performRequests``, a synthetic Thai and English page read with Thai trigram recall of at
  least 0.8 at 300 dpi, the real timeout and failure paths, and the th canary when that pack is
  installed. ``scripts/release.sh`` runs ``pytest -m vision`` and refuses to tag on a skip.

With ``PDFINTAKE_REQUIRE_VISION=1`` (the macOS CI jobs and the release Mac), a failed import, a
failed request and ``th-TH`` missing from revision 3 accurate are failures, never skips.

All text is made up.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake.errors import EngineFailure, PackError
from pdfintake.ocr import engines
from pdfintake.ocr.engines import Availability, vision
from pdfintake.params import ValidationContext
from pdfintake.types import EngineRead, PageImage

REQUIRE = os.environ.get("PDFINTAKE_REQUIRE_VISION") == "1"
CTX = ValidationContext(use_ids=frozenset({"vision"}), asset_keys=frozenset(), asset_kinds={},
                        scripts=frozenset({"Latin", "Thai"}), where="test.vision.params")

THAI_LINES = (build.THAI_SAMPLE, "ราคาสินค้าตัวอย่าง 4521 บาท")
LATIN_LINE = "The example invoice lists seven sample items."
# Thai fonts every macOS install carries (base system, not a downloadable asset).
SYSTEM_THAI_FONTS = tuple(Path("/System/Library/Fonts/Supplemental") / f
                          for f in ("Ayuthaya.ttf", "Sathu.ttf", "Silom.ttf", "Krungthep.ttf"))


def validate(raw: dict[str, Any]) -> dict[str, Any]:
    return vision.ENGINE.PARAMS.validate(raw, CTX)


@pytest.fixture(autouse=True)
def _fresh_module_state():
    vision._reset_for_tests()
    yield
    vision._reset_for_tests()


# --------------------------------------------------------------------------- stubs


class FakeMac:
    """Patches the module so it believes it runs on a macOS with the given Vision languages."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, languages: dict[tuple[int, str], tuple[str, ...]] | None = None,
                 revisions: tuple[int, ...] = (1, 2, 3)) -> None:
        self.languages = languages if languages is not None else {
            (3, "accurate"): ("en-US", "de-DE", "th-TH", "zh-Hans", "zh-Hant"),
            (3, "fast"): ("en-US", "de-DE"),
            (2, "accurate"): ("en-US", "de-DE"),
        }
        self.performed: list[tuple[Path, Any]] = []
        self.raws: list[Any] = []
        monkeypatch.setattr(vision, "_on_darwin", lambda: True)
        monkeypatch.setattr(vision, "_supported_revisions", lambda: revisions)
        monkeypatch.setattr(vision, "supported_languages", lambda rev, level: self.languages.get((rev, level), ()))
        monkeypatch.setattr(vision, "_macos_version", lambda: "15.4")
        monkeypatch.setattr(vision, "_perform", self.perform)

    def perform(self, path: Path, opts: Any, job: Any) -> list[Any]:
        self.performed.append((path, opts))
        return list(self.raws)


@pytest.fixture
def fake_mac(monkeypatch: pytest.MonkeyPatch) -> FakeMac:
    return FakeMac(monkeypatch)


@pytest.fixture
def page(tmp_path: Path) -> PageImage:
    return build.page_image(tmp_path, image=build.text_like(1000, 500, lines=3), dpi=300)


def raw(text: str, x: float, y: float, w: float, h: float, conf: float = 0.9) -> Any:
    return vision._Raw(text, conf, x, y, w, h)


# --------------------------------------------------------------------------- the engine object


def test_engine_declares_the_protocol():
    eng = vision.ENGINE
    assert isinstance(eng, engines.Engine)
    assert eng.name == "vision" and eng.api == 1
    assert eng.MODEL_KINDS == frozenset()
    assert eng.platforms == frozenset({"darwin-arm64"})
    assert eng.session_per_thread is False
    assert engines.get("vision") is eng


def test_import_does_not_load_vision():
    # vision.py imports Vision lazily; importing the module (done above) must not pull in pyobjc.
    import subprocess

    src = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import pdfintake.ocr.engines.vision; "
            "print(sorted(m for m in ('Vision', 'objc', 'Foundation') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(src)], capture_output=True, text=True,
                         encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


# --------------------------------------------------------------------------- parameters


def test_param_defaults():
    p = validate({"languages": ["th-TH", "en-US"]})
    assert p == {"languages": ["th-TH", "en-US"], "revision": 3, "level": "accurate",
                 "language_correction": True, "timeout_s": 120.0}
    assert vision.DEFAULT_TIMEOUT_S == 120.0 and vision.DEFAULT_REVISION == 3
    assert validate(p) == p


@pytest.mark.parametrize("tag", ["en-US", "th-TH", "zh-Hans", "yue-Hant", "ars-SA", "vi-VT", "es-419"])
def test_vision_language_tags_accepted(tag: str):
    assert validate({"languages": [tag]})["languages"] == [tag]


@pytest.mark.parametrize("raw_params", [
    {},                                                        # languages are required
    {"languages": []},
    {"languages": ["thai"]},
    {"languages": ["th_TH"]},
    {"languages": ["TH-th"]},
    {"languages": ["en-US", "en-US"]},
    {"languages": [f"e{c}-US" for c in "abcdefghi"]},          # 9 > 8
    {"languages": ["en-US"], "level": "medium"},
    {"languages": ["en-US"], "revision": 4},
    {"languages": ["en-US"], "revision": "3"},
    {"languages": ["en-US"], "language_correction": "yes"},
    {"languages": ["en-US"], "timeout_s": 1.0},
    {"languages": ["en-US"], "options": {}},                   # unknown key
])
def test_params_refused(raw_params: dict[str, Any]):
    with pytest.raises(PackError) as info:
        validate(raw_params)
    assert info.value.code == "invalid_params"


# --------------------------------------------------------------------------- not on platform


@pytest.mark.parametrize("key", ["linux-x86_64", "linux-aarch64", "win-amd64", "darwin-x86_64", "nowhere-none"])
def test_not_on_platform_elsewhere(key: str, fake_mac: FakeMac):
    avail = vision.ENGINE.availability(validate({"languages": ["en-US"]}), {}, key)
    assert avail == Availability(False, "not_on_platform", avail.detail) and avail.detail


def test_off_macos_everything_is_unavailable_never_failed(monkeypatch: pytest.MonkeyPatch, page: PageImage):
    monkeypatch.setattr(vision, "_on_darwin", lambda: False)
    called: list[Any] = []
    monkeypatch.setattr(vision, "_perform", lambda *a: called.append(a) or [])
    params = validate({"languages": ["en-US"]})
    avail = vision.ENGINE.availability(params, {}, "darwin-arm64")
    assert not avail.ok and avail.state == "not_on_platform"
    assert vision.supported_languages(3, "accurate") == ()
    assert vision.ENGINE.version() is None
    assert vision.ENGINE.candidate_params("th", "Thai", []) == []
    assert vision.ENGINE.contract_case() is None
    with pytest.raises(EngineFailure):
        vision._vision()
    session = vision.ENGINE.open("vision", "reader", params, {})
    read = session.read(page, timeout_s=5)
    assert read.status == "unavailable" and read.lines == [] and read.error
    assert called == []


# --------------------------------------------------------------------------- the language gate


def test_language_gate(fake_mac: FakeMac):
    eng = vision.ENGINE
    ok = eng.availability(validate({"languages": ["th-TH", "en-US"]}), {}, "darwin-arm64")
    assert ok == Availability(True, "available", "")
    hu = eng.availability(validate({"languages": ["hu-HU", "en-US"]}), {}, "darwin-arm64")
    assert not hu.ok and hu.state == "unsupported_language" and "hu-HU" in hu.detail and "en-US" not in hu.detail
    fast = eng.availability(validate({"languages": ["th-TH"], "level": "fast"}), {}, "darwin-arm64")
    assert fast.state == "unsupported_language" and "fast" in fast.detail
    rev2 = eng.availability(validate({"languages": ["th-TH"], "revision": 2}), {}, "darwin-arm64")
    assert rev2.state == "unsupported_language"
    assert eng.availability({}, {}, "darwin-arm64").state == "unsupported_language"


def test_unsupported_language_never_reaches_perform(fake_mac: FakeMac, page: PageImage):
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["hu-HU"]}), {})
    read = session.read(page, timeout_s=5)
    assert read.status == "unavailable" and "hu-HU" in (read.error or "")
    assert fake_mac.performed == []


def test_missing_revision_is_not_on_platform(monkeypatch: pytest.MonkeyPatch, page: PageImage):
    fake = FakeMac(monkeypatch, revisions=(1, 2))
    avail = vision.ENGINE.availability(validate({"languages": ["en-US"]}), {}, "darwin-arm64")
    assert avail.state == "not_on_platform" and "revision 3" in avail.detail
    read = vision.ENGINE.open("v", "reader", validate({"languages": ["en-US"]}), {}).read(page, timeout_s=5)
    assert read.status == "unavailable" and fake.performed == []
    assert vision.ENGINE.version() == "macOS 15.4"


def test_broken_vision_install(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vision, "_on_darwin", lambda: True)

    def broken() -> Any:
        raise EngineFailure("vision_unavailable", "pyobjc-framework-Vision could not be loaded: ImportError: made up")

    monkeypatch.setattr(vision, "_vision", broken)
    params = validate({"languages": ["en-US"]})
    avail = vision.ENGINE.availability(params, {}, "darwin-arm64")
    assert avail.state == "broken" and "could not be loaded" in avail.detail
    with pytest.raises(EngineFailure):
        vision.ENGINE.open("vision", "reader", params, {})
    assert vision.ENGINE.version() is None
    assert vision.ENGINE.contract_case() is None
    assert vision.ENGINE.candidate_params("en", "Latin", []) == []


def test_import_failure_is_remembered(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vision, "_on_darwin", lambda: True)
    monkeypatch.setitem(sys.modules, "Vision", None)        # makes `import Vision` raise ImportError
    with pytest.raises(EngineFailure) as first:
        vision._vision()
    assert first.value.code == "vision_unavailable" and "Error" in first.value.message
    monkeypatch.delitem(sys.modules, "Vision")
    with pytest.raises(EngineFailure):
        vision._vision()                                   # not retried within the process


def test_version_names_revision_and_macos(fake_mac: FakeMac):
    assert vision.ENGINE.version() == "rev3 (macOS 15.4)"


def test_supported_languages_refuses_an_unknown_level():
    with pytest.raises(ValueError):
        vision.supported_languages(3, "medium")


def test_candidate_params(fake_mac: FakeMac):
    th = vision.ENGINE.candidate_params("th", "Thai", [])
    assert [c["languages"] for c in th] == [["th-TH", "en-US"], ["th-TH", "en-US"]]
    assert {c["language_correction"] for c in th} == {True, False}
    for c in th:
        assert validate(c) == {**c, "timeout_s": 120.0}
    assert [c["languages"] for c in vision.ENGINE.candidate_params("en", "Latin", [])] == [["en-US"], ["en-US"]]
    assert len(vision.ENGINE.candidate_params("zh", "Han", [])) == 4
    assert vision.ENGINE.candidate_params("hu", "Latin", []) == []


# --------------------------------------------------------------------------- reads (stubbed)


def test_boxes_become_top_left_clockwise_quads(fake_mac: FakeMac, page: PageImage):
    # Vision: normalized, origin bottom left. Line A sits near the top of the image, B lower.
    fake_mac.raws = [raw("lower line", 0.1, 0.2, 0.5, 0.1, conf=0.5), raw("upper line", 0.2, 0.8, 0.4, 0.1, conf=1.0)]
    read = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {}).read(page, timeout_s=5)
    assert read.status == "ok"
    assert [ln.text for ln in read.lines] == ["upper line", "lower line"]
    w, h = page.width, page.height                          # 1000 x 500
    assert read.lines[0].box == ((200.0, 50.0), (600.0, 50.0), (600.0, 100.0), (200.0, 100.0))
    assert read.lines[1].box == ((100.0, 350.0), (600.0, 350.0), (600.0, 400.0), (100.0, 400.0))
    for ln in read.lines:
        (x0, y0), (x1, _), (_, y1), _ = ln.box
        assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h
    assert read.meta["mean_conf"] == 0.75
    assert read.meta["version"] == "rev3 (macOS 15.4)"
    assert read.spec == "vision:en-US:rev3:accurate"
    assert read.engine == "vision" and read.use_id == "vision" and read.role == "reader"


def test_boxes_clamped_and_confidence_bounded():
    lines = vision._to_lines([raw("edge", -0.05, -0.02, 1.2, 0.3, conf=1.7), raw("", 0.1, 0.1, 0.1, 0.1),
                              raw("neg", 0.5, 0.5, 0.1, 0.1, conf=-0.2)], 800, 600)
    assert [ln.text for ln in lines] == ["neg", "edge"]       # the empty observation is dropped
    edge = lines[1]
    assert edge.box == ((0.0, 432.0), (800.0, 432.0), (800.0, 600.0), (0.0, 600.0))   # top 0.28 -> 432 px
    assert edge.conf == 1.0 and lines[0].conf == 0.0


def test_empty_result_is_empty(fake_mac: FakeMac, page: PageImage):
    read = vision.ENGINE.open("vision", "checker", validate({"languages": ["en-US"]}), {}).read(page, timeout_s=5)
    assert read.status == "empty" and read.lines == [] and read.role == "checker"
    assert "mean_conf" not in read.meta


def test_options_reach_the_request(fake_mac: FakeMac, page: PageImage):
    params = validate({"languages": ["th-TH", "en-US"], "language_correction": False})
    session = vision.ENGINE.open("vision", "reader", params, {})
    session.read(page, timeout_s=5)
    (path, opts), = fake_mac.performed
    assert path == page.path
    assert opts == vision._Options(("th-TH", "en-US"), 3, "accurate", False)
    assert session.spec == "vision:th-TH+en-US:rev3:accurate:nolc"


def test_stubbed_hang_hits_the_timeout(fake_mac: FakeMac, monkeypatch: pytest.MonkeyPatch, page: PageImage):
    released = threading.Event()
    cancelled: list[bool] = []

    class FakeRequest:
        def cancel(self) -> None:
            cancelled.append(True)
            released.set()

    def hang(path: Path, opts: Any, job: Any) -> list[Any]:
        if not job.attach(FakeRequest()):
            raise vision._Cancelled()
        released.wait(30)                    # a request that never finishes on its own
        job.detach()
        raise vision._Cancelled()

    monkeypatch.setattr(vision, "_perform", hang)
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {})
    start = time.perf_counter()
    read = session.read(page, timeout_s=0.2)  # the caller's deadline, below the pack's 120 s
    elapsed = time.perf_counter() - start
    assert read.status == "timeout" and read.lines == [] and "0.2 s" in (read.error or "")
    assert 0.15 <= elapsed < 2.0
    assert cancelled == [True]               # the request was told to stop
    assert read.seconds >= 0.15


def test_zero_timeout_is_a_timeout(fake_mac: FakeMac, monkeypatch: pytest.MonkeyPatch, page: PageImage):
    monkeypatch.setattr(vision, "_perform", lambda p, o, job: time.sleep(0.5) or [])
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {})
    for none_left in (0, -1.0, float("nan")):
        read = session.read(page, timeout_s=none_left)
        assert read.status == "timeout" and read.error == "no time left for the read"
    assert fake_mac.performed == []          # no request was started


def _deadline_spy(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the deadline each read waits for, and answer as a request that ran out of time."""
    waits: list[float] = []

    def spy(work: Any, timeout_s: float, on_timeout: Any) -> tuple[str, Any]:
        waits.append(timeout_s)
        on_timeout()
        return "timeout", None

    monkeypatch.setattr(vision, "_run_with_deadline", spy)
    return waits


def test_pack_timeout_caps_a_longer_caller_deadline(fake_mac: FakeMac, monkeypatch: pytest.MonkeyPatch,
                                                    page: PageImage):
    # A pack that writes timeout_s: 10 is obeyed even when the pipeline offers 60 s, as Tesseract does.
    waits = _deadline_spy(monkeypatch)
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"], "timeout_s": 10}), {})
    assert session.limit_s == 10.0
    read = session.read(page, timeout_s=60)
    assert waits == [10.0]
    assert read.status == "timeout" and "within 10 s" in (read.error or "")


def test_default_pack_timeout_is_120_s(fake_mac: FakeMac, monkeypatch: pytest.MonkeyPatch, page: PageImage):
    waits = _deadline_spy(monkeypatch)
    validated = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {})
    unvalidated = vision.ENGINE.open("vision", "reader", {"languages": ["en-US"]}, {})   # param absent
    assert validated.limit_s == unvalidated.limit_s == vision.DEFAULT_TIMEOUT_S == 120.0
    validated.read(page, timeout_s=600)
    unvalidated.read(page, timeout_s=600)
    validated.read(page, timeout_s=30)                       # a shorter caller deadline still wins
    validated.read(page, timeout_s="soon")                   # an unusable one falls back to the cap
    assert waits == [120.0, 120.0, 30.0, 120.0]


def test_pack_timeout_cuts_off_a_slow_request(fake_mac: FakeMac, monkeypatch: pytest.MonkeyPatch, page: PageImage):
    # The real deadline path, with the pack's cap patched down from its 10 s floor so the test is fast.
    monkeypatch.setattr(vision, "_perform", lambda p, o, job: time.sleep(3) or [vision._Raw("late", 0.9, 0, 0, 1, 1)])
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"], "timeout_s": 10}), {})
    session.limit_s = 0.2
    start = time.perf_counter()
    read = session.read(page, timeout_s=60)
    assert read.status == "timeout" and time.perf_counter() - start < 2.0


@pytest.mark.parametrize("error, expect", [
    (vision._VisionRequestError("com.example.vision 9: made-up failure"), "Vision: com.example.vision 9"),
    (RuntimeError("something odd"), "RuntimeError: something odd"),
    (EngineFailure("vision_unavailable", "made-up load failure"), "made-up load failure"),
    (vision._Cancelled(), "cancelled"),
])
def test_request_errors_are_reported_not_raised(fake_mac: FakeMac, monkeypatch: pytest.MonkeyPatch, page: PageImage,
                                                error: BaseException, expect: str):
    def boom(path: Path, opts: Any, job: Any) -> list[Any]:
        raise error

    monkeypatch.setattr(vision, "_perform", boom)
    read = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {}).read(page, timeout_s=5)
    assert read.status == ("timeout" if isinstance(error, vision._Cancelled) else "failed")
    assert expect in (read.error or "") and read.lines == []


def test_missing_page_image_is_failed(fake_mac: FakeMac, tmp_path: Path):
    ghost = PageImage(0, tmp_path / "gone.png", 100, 100, 300.0, (0.0, 0.0, 24.0, 24.0))
    read = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {}).read(ghost, timeout_s=5)
    assert read.status == "failed" and read.error and fake_mac.performed == []


def test_closed_session_refuses(fake_mac: FakeMac, page: PageImage):
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {})
    session.close()
    with pytest.raises(EngineFailure):
        session.read(page, timeout_s=5)


def test_job_cancel_before_attach_stops_the_request():
    job = vision._Job()
    job.cancel()
    assert job.attach(object()) is False and job.cancelled


# --------------------------------------------------------------------------- real Vision (macOS)


def need_vision(reason: str) -> None:
    """Fail under PDFINTAKE_REQUIRE_VISION=1, skip otherwise."""
    if REQUIRE:
        pytest.fail(f"PDFINTAKE_REQUIRE_VISION=1: {reason}")
    pytest.skip(reason)


@pytest.fixture
def real_vision() -> vision._Bridge:
    if sys.platform != "darwin":
        need_vision("Apple Vision exists only on macOS")
    try:
        return vision._vision()
    except EngineFailure as exc:
        need_vision(f"Vision could not be imported: {exc.message}")
        raise  # unreachable


def thai_trigrams(text: str) -> Counter[str]:
    """Thai character trigrams (U+0E01-U+0E4F, no Thai digits) over the text's Thai runs joined."""
    t = "".join(re.findall("[ก-๏]+", unicodedata.normalize("NFKC", text)))
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def trigram_recall(expected: str, got: str) -> float:
    want, have = thai_trigrams(expected), thai_trigrams(got)
    total = sum(want.values())
    return sum(min(n, have[g]) for g, n in want.items()) / total if total else 0.0


def shaped_page(folder: Path, lines: tuple[str, ...], *, dpi: float = 300.0) -> PageImage:
    """Lines of text shaped by uharfbuzz (fpdf2's text shaping), rendered by pypdfium2 at ``dpi``."""
    pytest.importorskip("uharfbuzz")
    import pypdfium2 as pdfium
    from fpdf import FPDF

    font = next((f for f in SYSTEM_THAI_FONTS if f.is_file()), None)
    if font is None:
        pytest.skip("no Thai system font found (looked in /System/Library/Fonts/Supplemental)")
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    pdf = FPDF(unit="pt", format=(595.28, 100 + 60 * len(lines)))
    pdf.set_creation_date(build.FIXED_DATE)
    pdf.set_auto_page_break(False)
    pdf.add_page()
    pdf.add_font("ThaiSystem", fname=str(font))
    pdf.set_font("ThaiSystem", size=20)
    pdf.set_text_shaping(True)
    for i, text in enumerate(lines):
        pdf.text(40, 70 + 60 * i, text)
    doc = pdfium.PdfDocument(bytes(pdf.output()))
    try:
        img = doc[0].render(scale=dpi / 72.0).to_pil().convert("RGB")
    finally:
        doc.close()
    return build.page_image(folder, image=img, dpi=dpi, name="shaped.png")


@pytest.mark.vision
def test_vision_imports_and_lists_languages(real_vision: vision._Bridge, record_property: Any):
    revisions = vision._supported_revisions()
    listing = {f"rev{r} {lvl}": list(vision.supported_languages(r, lvl)) for r in revisions for lvl in vision.LEVELS}
    for key, tags in listing.items():
        print(f"Vision {key} on macOS {vision._macos_version()}: {', '.join(tags)}")
    record_property("vision_languages", listing)
    assert 3 in revisions, f"revision 3 missing (have {revisions})"
    assert vision.ENGINE.version() == f"rev3 (macOS {vision._macos_version()})"


@pytest.mark.vision
def test_th_TH_in_rev3_accurate(real_vision: vision._Bridge):
    langs = vision.supported_languages(3, "accurate")
    if "th-TH" not in langs:
        need_vision(f"th-TH is missing from Vision rev3 accurate: {', '.join(langs)}")
    assert vision.ENGINE.availability(validate({"languages": ["th-TH", "en-US"]}), {}, "darwin-arm64").ok


@pytest.mark.vision
def test_empty_dict_options_rejected_none_accepted(real_vision: vision._Bridge, tmp_path: Path):
    """Why vision.py passes ``None``: an empty dict raises "key does not exist" on some macOS
    releases (27 does; the macOS 26 CI runner accepts it), and ``None`` works on every one."""
    png = build.write_png(build.blank(64, 64), tmp_path / "w.png", dpi=300)
    url = real_vision.NSURL.fileURLWithPath_(str(png))
    handlers = real_vision.Vision.VNImageRequestHandler
    try:
        handlers.alloc().initWithURL_options_(url, {})
    except Exception as exc:  # noqa: BLE001 - pyobjc raises its own error type for this
        assert "key does not exist" in str(exc), exc
    handler = handlers.alloc().initWithURL_options_(url, None)
    request = real_vision.Vision.VNRecognizeTextRequest.alloc().init()
    ok, err = handler.performRequests_error_([request], None)
    assert ok and err is None


@pytest.mark.vision
def test_hu_HU_caught_before_perform(real_vision: vision._Bridge, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    if "hu-HU" in vision.supported_languages(3, "accurate"):
        pytest.skip("this macOS now lists hu-HU for Vision rev3: re-measure and update the Latin packs")
    params = validate({"languages": ["hu-HU", "en-US"]})
    avail = vision.ENGINE.availability(params, {}, "darwin-arm64")
    assert avail.state == "unsupported_language" and "hu-HU" in avail.detail
    performed: list[Any] = []
    monkeypatch.setattr(vision, "_perform", lambda *a: performed.append(a) or [])
    page = build.page_image(tmp_path, image=build.text_like(600, 300), dpi=300)
    read = vision.ENGINE.open("vision", "reader", params, {}).read(page, timeout_s=30)
    assert read.status == "unavailable" and performed == []


@pytest.mark.vision
def test_reads_synthetic_thai_and_english(real_vision: vision._Bridge, tmp_path: Path):
    if "th-TH" not in vision.supported_languages(3, "accurate"):
        need_vision("th-TH is missing from Vision rev3 accurate")
    page = shaped_page(tmp_path, (THAI_LINES[0], LATIN_LINE, THAI_LINES[1]))
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["th-TH", "en-US"]}), {})
    read = session.read(page, timeout_s=vision.DEFAULT_TIMEOUT_S)
    if read.status == "failed":
        need_vision(f"the Vision request failed: {read.error}")
    assert read.status == "ok", read.status
    text = read.text()
    recall = trigram_recall("".join(THAI_LINES), text)
    print(f"Thai trigram recall {recall:.3f} at 300 dpi; {read.seconds:.2f} s; {len(read.lines)} lines")
    assert recall >= 0.8, f"Thai trigram recall {recall:.3f}"
    assert "4521" in text and "example invoice" in text
    for ln in read.lines:
        for x, y in ln.box:
            assert 0 <= x <= page.width and 0 <= y <= page.height
        assert 0.0 <= ln.conf <= 1.0
    tops = [ln.box[0][1] for ln in read.lines]
    assert tops == sorted(tops)
    again = session.read(page, timeout_s=vision.DEFAULT_TIMEOUT_S)
    assert [(ln.text, ln.box, ln.conf) for ln in again.lines] == [(ln.text, ln.box, ln.conf) for ln in read.lines]


@pytest.mark.vision
def test_real_timeout_then_recovers(real_vision: vision._Bridge, tmp_path: Path):
    page = shaped_page(tmp_path, (LATIN_LINE,))
    session = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {})
    start = time.perf_counter()
    read = session.read(page, timeout_s=0.001)
    assert read.status == "timeout" and time.perf_counter() - start < 2.0
    after = session.read(page, timeout_s=vision.DEFAULT_TIMEOUT_S)
    if after.status == "failed":
        need_vision(f"the Vision request failed: {after.error}")
    assert after.status == "ok" and "example invoice" in after.text()


@pytest.mark.vision
def test_real_request_failure_is_failed_not_raised(real_vision: vision._Bridge, tmp_path: Path):
    bad = tmp_path / "not-a-png.png"
    bad.write_bytes(b"made-up bytes that are not an image")
    page = PageImage(0, bad, 100, 100, 300.0, (0.0, 0.0, 24.0, 24.0))
    read = vision.ENGINE.open("vision", "reader", validate({"languages": ["en-US"]}), {}).read(page, timeout_s=30)
    assert isinstance(read, EngineRead) and read.status == "failed" and read.error


@pytest.mark.vision
def test_contract_case_on_this_mac(real_vision: vision._Bridge, tmp_path: Path):
    case = vision.ENGINE.contract_case()
    assert case is not None and case.platform_key == "darwin-arm64"
    params = validate(dict(case.params))
    assert vision.ENGINE.availability(params, {}, "darwin-arm64").ok
    page = case.page(tmp_path)
    read = vision.ENGINE.open("contract", "reader", params, {}).read(page, timeout_s=case.timeout_s)
    if read.status == "failed":
        need_vision(f"the Vision request failed: {read.error}")
    assert read.status == "ok" and "4521" in read.text()
    assert case.unsupported_params is not None
    bad = validate(dict(case.unsupported_params))
    assert vision.ENGINE.availability(bad, {}, "darwin-arm64").state == "unsupported_language"


@pytest.mark.vision
def test_th_canary(real_vision: vision._Bridge):
    try:
        from pdfintake.langpacks import store
    except ImportError:
        pytest.skip("langpacks/store.py is not built yet")
    try:
        pack = store.locate("th")
    except Exception as exc:          # a broken store is store's test's business
        pytest.skip(f"the th pack can't be located: {exc}")
    if pack is None or getattr(pack.manifest, "canary", None) is None:
        pytest.skip("the th pack (and its canary) isn't installed")
    canary = pack.manifest.canary
    path = Path(pack.root) / canary.file
    from PIL import Image

    with Image.open(path, formats=["PNG"]) as img:
        width, height = img.size
    page = PageImage(0, path, width, height, float(canary.dpi), (0.0, 0.0, width * 72.0 / canary.dpi,
                                                                  height * 72.0 / canary.dpi))
    use = next((u for u in pack.manifest.readers_for("darwin-arm64").engines if u.engine == "vision"), None)
    if use is None:
        pytest.skip("the th pack's darwin-arm64 reader set has no Vision use")
    params = validate(dict(use.params))
    read = vision.ENGINE.open(use.id, use.role, params, {}).read(page, timeout_s=vision.DEFAULT_TIMEOUT_S)
    if read.status == "failed":
        need_vision(f"the Vision request failed: {read.error}")
    text = read.text()
    missing = [s for s in canary.must_find if s not in text]
    assert not missing, f"the th canary lacks {missing}"
