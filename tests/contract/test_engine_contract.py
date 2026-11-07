"""The engine contract, run against every registered engine that offers a contract case on this OS.

Each engine in ``pdfintake.ocr.engines._REGISTRY`` is a member. An engine whose module isn't built
yet, or that offers no contract case here, is skipped with the reason. A built-in fake member
always runs, so the suite itself is tested.

Tesseract and RapidOCR find their models in the installed base pack, which a test run doesn't have
(``PDFINTAKE_CACHE`` is a fresh folder). So when ``contract_case()`` is None, the suite builds the
engine's own case from the pinned test models instead, through ``conftest.pinned_assets`` (decided
2026-11-05): ``en:eng`` for Tesseract, ``en:rapid-det`` and ``en:rapid-rec-latin`` for RapidOCR.
``tools/fetch_test_models.py`` puts them in place (CI runs it before the suite); without them the
member skips, naming the missing files and that command.

What every member must do:
* ``read`` returns quads inside the image, clockwise, and confidences in [0, 1];
* a blank image gives ``empty``, not an exception;
* two sessions in parallel threads give identical reads;
* the timeout is honoured, and running out is ``timeout``, never an exception;
* the same input gives the same output;
* a missing model gives ``missing_model``;
* ``unavailable`` is never reported as ``failed``.

Below the members, the registry itself: lazy, explicit, and precise about unknown names.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
import types as pytypes
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from conftest import pinned_assets
from fixtures import build
from pdfintake.errors import EXIT_SETUP, PackError, SetupError
from pdfintake.ocr import engines
from pdfintake.ocr.engines import Availability, ContractCase
from pdfintake.params import ParamTable, ValidationContext
from pdfintake.types import READ_STATUSES, ROLES, EngineRead, PageImage, to_jsonable

SRC = Path(__file__).resolve().parents[2] / "src"
MODEL_KINDS = frozenset({"traineddata", "onnx_det", "onnx_cls", "onnx_rec", "rec_keys"})
AVAILABILITY_STATES = frozenset({"available", "not_on_platform", "missing_model", "unsupported_language", "broken"})
NAME_PATTERN = r"[A-Za-z][A-Za-z0-9_]{0,39}"
USE_ID = "contract"
EPS = 0.5          # px of slack for boxes converted from normalized coordinates


@dataclass
class Member:
    name: str
    engine: Any
    case: ContractCase
    platform_key: str
    params: dict[str, Any]
    folder: Path

    def page(self) -> PageImage:
        if self.case.page is not None:
            return self.case.page(self.folder)
        return build.page_image(self.folder, image=build.text_like(1200, 800), dpi=300, name="synthetic.png")

    def open(self, params: dict[str, Any] | None = None) -> Any:
        return self.engine.open(USE_ID, self.case.role, params or self.params, self.case.models)


def _known_scripts() -> frozenset[str]:
    try:
        from pdfintake.ocr.scripts import SCRIPTS
    except ImportError:
        return frozenset({"Latin", "Thai", "Cyrillic", "Greek", "Arabic", "Hebrew", "Han", "Hangul", "Devanagari"})
    return frozenset(SCRIPTS)


def _current_platform_key() -> str:
    try:
        from pdfintake import platforms
    except ImportError:
        pytest.skip("platforms.py is not built yet")
    try:
        return platforms.current().key
    except SetupError as exc:
        pytest.skip(f"unsupported platform: {exc}")


def _context(name: str, case: ContractCase) -> ValidationContext:
    return ValidationContext(
        use_ids=frozenset({USE_ID}), asset_keys=frozenset(case.models), asset_kinds=dict(case.model_kinds),
        scripts=_known_scripts(), where=f"contract.{name}.params",
    )


def _built(name: str) -> bool:
    module = engines._REGISTRY[name].partition(":")[0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _fake_member(folder: Path) -> Member:
    model = folder / "fake.model"
    model.write_bytes(b"made-up model bytes")
    engine = build.FakeEngine(required_models=("fake-model",), languages=("en",))
    case = ContractCase(
        params={"lang": "en"}, models={"fake-model": model}, model_kinds={"fake-model": "traineddata"},
        page=lambda d: build.page_image(d, image=build.text_like(900, 600, lines=4), dpi=300, name="fake-page.png"),
        unsupported_params={"lang": "zz"}, platform_key="fake-os", timeout_s=30.0,
    )
    params = engine.PARAMS.validate(case.params, _context("fake", case))
    return Member("fake", engine, case, "fake-os", params, folder)


def _engine_module(name: str) -> Any:
    return importlib.import_module(engines._REGISTRY[name].partition(":")[0])


def _pinned_tesseract(engine: Any, monkeypatch: pytest.MonkeyPatch) -> ContractCase | None:
    """Tesseract's own contract case, with ``eng`` taken from the pinned test models."""
    module = _engine_module("tesseract")
    if module.PyTessBaseAPI is None:
        pytest.skip(f"tesserocr can't be imported: {module._IMPORT_ERROR}")
    eng = pinned_assets("en:eng")["en:eng"]
    monkeypatch.setattr(module, "_contract_models", lambda: {"eng": eng})
    return engine.contract_case()


def _pinned_rapidocr(engine: Any, monkeypatch: pytest.MonkeyPatch) -> ContractCase | None:
    """RapidOCR's own contract case, with the base pack's detector and Latin recogniser pinned."""
    module = _engine_module("rapidocr")
    det_key, rec_key = module.VOTER_DET_KEY, module.VOTER_REC_KEY
    found = pinned_assets(f"en:{det_key}", f"en:{rec_key}")
    return module._contract_case(found[f"en:{det_key}"], found[f"en:{rec_key}"], det_key=det_key, rec_key=rec_key)


PINNED_CASES = {"tesseract": _pinned_tesseract, "rapidocr": _pinned_rapidocr}


def _real_member(name: str, folder: Path, monkeypatch: pytest.MonkeyPatch) -> Member:
    if not _built(name):
        pytest.skip(f"{engines._REGISTRY[name]} is not built yet")
    engine = engines.get(name)
    case = engine.contract_case()
    if case is None and name in PINNED_CASES:
        case = PINNED_CASES[name](engine, monkeypatch)      # skips, naming the files, when they're absent
    if case is None:
        pytest.skip(f"{name} offers no contract case on this machine")
    key = case.platform_key or _current_platform_key()
    params = engine.PARAMS.validate(case.params, _context(name, case))
    return Member(name, engine, case, key, params, folder)


@pytest.fixture(params=[*engines.names(), "fake"])
def member(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Member:
    if request.param == "fake":
        return _fake_member(tmp_path)
    return _real_member(request.param, tmp_path, monkeypatch)


def _comparable(read: EngineRead) -> dict[str, Any]:
    data = to_jsonable(read, include_text=True)
    data.pop("seconds")
    data.get("meta", {}).pop("seconds", None)
    return data


def _assert_read_shape(read: Any, member: Member, page: PageImage) -> None:
    assert isinstance(read, EngineRead)
    assert read.status in READ_STATUSES
    assert read.use_id == USE_ID
    assert read.engine == member.engine.name
    assert read.role == member.case.role and read.role in ROLES
    assert isinstance(read.spec, str) and read.spec
    assert read.seconds >= 0
    if read.status == "failed":
        assert isinstance(read.error, str) and read.error, "a failed read must say why"
    for ln in read.lines:
        assert isinstance(ln.text, str)
        assert 0.0 <= ln.conf <= 1.0, f"line confidence {ln.conf} is outside [0, 1]"
        _assert_quad(ln.box, page)
        for w in ln.words:
            assert isinstance(w.text, str)
            assert 0.0 <= w.conf <= 1.0, f"word confidence {w.conf} is outside [0, 1]"
            _assert_quad(w.box, page)


def _assert_quad(box: Any, page: PageImage) -> None:
    assert len(box) == 4, "a box is four points"
    for x, y in box:
        assert -EPS <= x <= page.width + EPS and -EPS <= y <= page.height + EPS, \
            f"point ({x}, {y}) lies outside the {page.width}x{page.height} image"
    area = sum(box[i][0] * box[(i + 1) % 4][1] - box[(i + 1) % 4][0] * box[i][1] for i in range(4))
    assert area >= -EPS, "box points must run clockwise from the top-left (y down)"


# --------------------------------------------------------------------------- members


def test_member_declares_the_protocol(member: Member):
    eng = member.engine
    assert isinstance(eng, engines.Engine)
    assert eng.name == member.name
    assert isinstance(eng.api, int) and not isinstance(eng.api, bool) and eng.api >= 1
    assert isinstance(eng.PARAMS, ParamTable)
    assert isinstance(eng.MODEL_KINDS, frozenset) and eng.MODEL_KINDS <= MODEL_KINDS
    assert isinstance(eng.platforms, frozenset) and all(isinstance(p, str) for p in eng.platforms)
    assert isinstance(eng.session_per_thread, bool)
    version = eng.version()
    assert version is None or isinstance(version, str)
    assert set(member.case.model_kinds.values()) <= eng.MODEL_KINDS


def test_contract_case_is_available(member: Member):
    avail = member.engine.availability(member.params, member.case.models, member.platform_key)
    assert isinstance(avail, Availability)
    if avail.state == "not_on_platform":
        pytest.skip(f"{member.name} is not on this platform: {avail.detail}")
    assert avail.ok and avail.state == "available", f"{member.name}: {avail.state}: {avail.detail}"


def test_read_returns_quads_inside_the_image(member: Member):
    page = member.page()
    session = member.open()
    try:
        read = session.read(page, timeout_s=member.case.timeout_s)
    finally:
        session.close()
    _assert_read_shape(read, member, page)
    assert read.status != "failed", f"{member.name} failed on its own contract page: {read.error}"
    if member.case.page is not None:
        assert read.status == "ok" and read.lines, f"{member.name} read nothing on its contract page"


def test_blank_image_is_empty(member: Member):
    page = build.page_image(member.folder, image=build.blank(800, 600, mode="RGB"), dpi=300, name="blank.png")
    session = member.open()
    try:
        read = session.read(page, timeout_s=member.case.timeout_s)
    finally:
        session.close()
    _assert_read_shape(read, member, page)
    assert read.status == "empty", f"a blank page gave {read.status!r}"
    assert read.lines == []


def test_parallel_sessions_give_identical_reads(member: Member):
    page = member.page()
    sessions = [member.open(), member.open()]
    results: list[Any] = [None, None]
    errors: list[BaseException] = []
    gate = threading.Barrier(2)

    def work(i: int) -> None:
        try:
            gate.wait(timeout=30)
            results[i] = sessions[i].read(page, timeout_s=member.case.timeout_s)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=member.case.timeout_s * 2 + 30)
    for s in sessions:
        s.close()
    assert not errors, f"a session raised: {errors[0]!r}"
    assert all(isinstance(r, EngineRead) for r in results), "a thread didn't finish"
    assert _comparable(results[0]) == _comparable(results[1])


def test_same_input_same_output(member: Member):
    page = member.page()
    first = member.open()
    try:
        a = first.read(page, timeout_s=member.case.timeout_s)
        b = first.read(page, timeout_s=member.case.timeout_s)
    finally:
        first.close()
    second = member.open()
    try:
        c = second.read(page, timeout_s=member.case.timeout_s)
    finally:
        second.close()
    assert _comparable(a) == _comparable(b) == _comparable(c)


def test_timeout_is_honoured(member: Member):
    page = member.page()
    params = dict(member.params)
    if member.name == "fake":
        params["delay_s"] = 5.0                  # a slow engine
    session = member.open(params)
    try:
        start = time.perf_counter()
        read = session.read(page, timeout_s=0.001)
        elapsed = time.perf_counter() - start
    finally:
        session.close()
    _assert_read_shape(read, member, page)
    assert read.status in ("timeout", "ok", "empty"), f"running out of time gave {read.status!r}"
    assert elapsed < 10.0, f"a 1 ms timeout took {elapsed:.1f} s"
    if member.name == "fake":
        assert read.status == "timeout" and elapsed < 1.0


def test_missing_model_is_named(member: Member):
    models = dict(member.case.models)
    if not models:
        pytest.skip(f"{member.name} needs no model files")
    for key in sorted(models):
        without = {k: v for k, v in models.items() if k != key}
        avail = member.engine.availability(member.params, without, member.platform_key)
        assert not avail.ok and avail.state == "missing_model", f"dropping {key}: {avail.state}"
        gone = dict(models, **{key: member.folder / "no-such-model-file"})
        avail = member.engine.availability(member.params, gone, member.platform_key)
        assert not avail.ok and avail.state == "missing_model", f"{key} pointing nowhere: {avail.state}"


def test_unavailable_is_never_failed(member: Member):
    eng = member.engine
    elsewhere = eng.availability(member.params, member.case.models, "nowhere-none")
    assert isinstance(elsewhere, Availability)
    assert not elsewhere.ok and elsewhere.state == "not_on_platform", elsewhere
    here = eng.availability(member.params, member.case.models, member.platform_key)
    assert here.state in AVAILABILITY_STATES and here.ok == (here.state == "available")
    if member.case.unsupported_params is not None:
        params = eng.PARAMS.validate(member.case.unsupported_params, _context(member.name, member.case))
        avail = eng.availability(params, member.case.models, member.platform_key)
        assert not avail.ok and avail.state == "unsupported_language", avail


# --------------------------------------------------------------------------- the registry


def test_registry_names_in_order():
    assert engines.names() == ("tesseract", "vision", "rapidocr")
    for name, target in engines._REGISTRY.items():
        assert re.fullmatch(NAME_PATTERN, name), name
        module, _, attr = target.partition(":")
        assert module.startswith("pdfintake.ocr.engines.") and attr == "ENGINE", target


@pytest.mark.parametrize("name", engines.names())
def test_registered_engine_resolves_to_itself(name: str):
    if not _built(name):
        pytest.skip(f"{engines._REGISTRY[name]} is not built yet")
    eng = engines.get(name)
    assert eng.name == name
    assert engines.get(name) is eng


def test_unknown_engine_is_named_precisely():
    with pytest.raises(PackError) as info:
        engines.get("tesserakt")
    err = info.value
    assert err.code == "unknown_engine"
    assert err.exit_code == EXIT_SETUP
    assert "'tesserakt'" in err.message
    for known in engines.names():
        assert known in err.message
    assert err.detail["engine"] == "tesserakt" and err.detail["known"] == list(engines.names())


def test_registry_is_lazy():
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); import pdfintake.ocr.engines as e; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('pdfintake.ocr.engines.') "
        "or m.split('.')[0] in ('tesserocr', 'onnxruntime', 'rapidocr', 'Vision', 'cv2'))))"
    )
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(SRC)], capture_output=True, text=True,
                         encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch):
    engines._clear_cache()
    yield monkeypatch
    engines._clear_cache()


def test_override_with_an_object(clean_registry: pytest.MonkeyPatch):
    fake = build.FakeEngine()
    clean_registry.setitem(engines._REGISTRY, "fake", fake)
    assert engines.get("fake") is fake
    assert "fake" in engines.names()


def test_override_with_a_module_target(clean_registry: pytest.MonkeyPatch):
    mod = pytypes.ModuleType("pdfintake_test_fake_engines")
    mod.ENGINE = build.FakeEngine()
    clean_registry.setitem(sys.modules, mod.__name__, mod)
    clean_registry.setitem(engines._REGISTRY, "fake", f"{mod.__name__}:ENGINE")
    assert engines.get("fake") is mod.ENGINE


def test_import_failure_is_a_named_setup_error(clean_registry: pytest.MonkeyPatch):
    clean_registry.setitem(engines._REGISTRY, "fake", "pdfintake_test_no_such_module:ENGINE")
    with pytest.raises(SetupError) as info:
        engines.get("fake")
    assert not isinstance(info.value, PackError)
    assert info.value.code == "engine_import_failed"
    assert "pdfintake_test_no_such_module" in info.value.message and "'fake'" in info.value.message


def test_warm_up_remembers_failures(clean_registry: pytest.MonkeyPatch):
    clean_registry.setattr(engines, "_REGISTRY", {"fake": "pdfintake_test_late_module:ENGINE"})
    assert engines.warm_up() is None
    mod = pytypes.ModuleType("pdfintake_test_late_module")
    mod.ENGINE = build.FakeEngine()
    clean_registry.setitem(sys.modules, mod.__name__, mod)
    with pytest.raises(SetupError):
        engines.get("fake")              # not imported again after warm_up failed


def test_missing_attribute_and_name_mismatch(clean_registry: pytest.MonkeyPatch):
    mod = pytypes.ModuleType("pdfintake_test_odd_engines")
    mod.OTHER = build.FakeEngine()
    clean_registry.setitem(sys.modules, mod.__name__, mod)
    clean_registry.setitem(engines._REGISTRY, "fake", f"{mod.__name__}:ENGINE")
    with pytest.raises(SetupError) as info:
        engines.get("fake")
    assert "has no ENGINE" in info.value.message
    clean_registry.setitem(engines._REGISTRY, "renamed", f"{mod.__name__}:OTHER")
    with pytest.raises(SetupError) as info:
        engines.get("renamed")
    assert info.value.code == "engine_name_mismatch"


def test_contract_case_defaults():
    case = ContractCase(params={})
    assert case.models == {} and case.role == "reader" and case.page is None and case.timeout_s > 0
    assert replace(case, timeout_s=1.0).timeout_s == 1.0
