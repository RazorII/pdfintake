"""ocr/pipeline.py: the two-pass OCR flow, its threading rules and its page results.

Most tests use fake engines registered through a test-only override of ``engines._REGISTRY`` and
packs built in code (manifests as dataclasses, model files of made-up bytes under a temp folder),
so the reads are deterministic and every call can be counted. The fakes answer by page index, not
by pixels. The tests at the end use the real engines and pinned models on synthetic English,
Hungarian and Thai pages (``conftest.pinned_asset``; they skip, naming the files, when the models
aren't fetched). Every text is made up.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake import record
from pdfintake.errors import EngineFailure, IntegrityError, PackError
from pdfintake.langpacks import index as I
from pdfintake.langpacks import manifest as M
from pdfintake.langpacks import store
from pdfintake.langpacks.manifest import (
    Asset,
    Detect,
    EngineUse,
    Manifest,
    MustSurvive,
    Probe,
    ReaderSet,
    RuleUse,
    StepUse,
)
from pdfintake.ocr import detect, engines, langplan, pipeline
from pdfintake.ocr.detect import MixedScript
from pdfintake.ocr.engines import Availability
from pdfintake.ocr.langplan import Calibration, PagePlan
from pdfintake.params import Param, ParamTable
from pdfintake.pdf.classify import PageClass
from pdfintake.types import EngineRead, LangCall, Line, OsdResult, PageImage, ScriptCall

KEY = "test-os"                          # a made-up platform key: every fake engine runs only here
REPO = Path(__file__).resolve().parents[1]
FIXTURE_INDEX = REPO / "tests" / "fixtures" / "index.test.json"

# Made-up text only.
EN_TEXT = ["The example invoice lists seven sample items and a total of four thousand units for the",
           "test customer account. The company will send the order to the address given on the form",
           "before Friday, as agreed on the phone last week, and the invoice is attached for your records."]
HU_TEXT = ["A kertészek tapasztalják tavasszal, hogy a régi fák és a rózsák illata minden reggel egyre erősebben",
           "árad, illetve a nyári zápor után szellő támad, amely a meteorológusok szerint az egyik hétvégén",
           "sem lesz erős. A könyvtár egy épületben működik, és a polcok már csak vagy még ezt kell tudni."]
THAI_TEXT = ["ข้อความตัวอย่าง สำหรับการทดสอบ", "บริษัทตัวอย่าง จำกัด"]
THAI_OSD = OsdResult("Thai", 8.0, 0, 5.0)


# --------------------------------------------------------------------------- fake engines

Script = Callable[[str, int], Any]       # (lang, page index) -> list[str] | list[Line] | "failed"/"timeout" | Exception


class FakeSession:
    def __init__(self, engine: FakeOcr, use_id: str, role: str, params: Mapping[str, Any]) -> None:
        self.engine = engine
        self.use_id = use_id
        self.role = role
        self.params = dict(params)
        self.threads: set[int] = set()
        self.closed = False

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        eng = self.engine
        if self.closed:
            raise EngineFailure("closed", "the fake session is closed")
        lang = str(self.params.get("lang", "en"))
        with eng.lock:
            eng.reads.append((self.use_id, lang, page.index, threading.get_ident(), timeout_s))
        self.threads.add(threading.get_ident())
        if eng.delay:
            time.sleep(eng.delay)
        out = eng.script(lang, page.index) if eng.script is not None else []
        if isinstance(out, BaseException):
            raise out
        spec = f"{eng.name}:{lang}"
        if isinstance(out, str):
            return EngineRead(self.use_id, eng.name, spec, self.role, out, [], 0.001, f"made-up {out}", {})  # type: ignore[arg-type]
        if out and isinstance(out[0], Line):
            lines = list(out)
        else:
            lines = build.lines_for(list(out), conf=eng.conf, with_words=True)
        return EngineRead(self.use_id, eng.name, spec, self.role, "ok" if lines else "empty",  # type: ignore[arg-type]
                          lines, 0.001, None, {"mean_conf": eng.mean_conf})

    def close(self) -> None:
        self.closed = True


class FakeOcr:
    """An engine that reads by page index; counts every open and read, with the thread it ran in."""

    api = 1
    MODEL_KINDS = frozenset({"onnx_rec"})

    def __init__(self, name: str, script: Script | None = None, *, per_thread: bool = True, conf: float = 0.9,
                 mean_conf: float | None = 0.9, avail: tuple[str, str] = ("available", ""),
                 open_error: str | None = None, delay: float = 0.0) -> None:
        self.name = name
        self.PARAMS = ParamTable(
            Param("lang", "str", pattern=r"[a-z]{2,3}", default="en"),
            Param("rec", "asset_ref"),
            Param("timeout_s", "float", min=1.0, max=600.0),
        )
        self.platforms = frozenset({KEY})
        self.session_per_thread = per_thread
        self.script = script
        self.conf = conf
        self.mean_conf = mean_conf
        self.avail = avail
        self.open_error = open_error
        self.delay = delay
        self.lock = threading.Lock()
        self.reads: list[tuple[str, str, int, int, float]] = []
        self.opens: list[tuple[str, int, dict[str, Any], dict[str, Path]]] = []
        self.sessions: list[FakeSession] = []

    def version(self) -> str | None:
        return "1.0"

    def availability(self, params: Mapping[str, Any], models: Mapping[str, Path], platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform", f"the fake engine runs only on {KEY}")
        state, detail = self.avail
        return Availability(state == "available", state, detail)  # type: ignore[arg-type]

    def open(self, use_id: str, role: str, params: Mapping[str, Any], models: Mapping[str, Path]) -> FakeSession:
        with self.lock:
            self.opens.append((use_id, threading.get_ident(), dict(params), dict(models)))
        if self.open_error is not None:
            raise EngineFailure("fake_open", self.open_error)
        session = FakeSession(self, use_id, role, params)
        with self.lock:
            self.sessions.append(session)
        return session

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence[Any]) -> list[dict[str, Any]]:
        return []

    def contract_case(self) -> None:
        return None

    def calls(self, lang: str | None = None) -> list[int]:
        return sorted(r[2] for r in self.reads if lang is None or r[1] == lang)


def register(monkeypatch: pytest.MonkeyPatch, *fakes: FakeOcr) -> None:
    for f in fakes:
        monkeypatch.setitem(engines._REGISTRY, f.name, f)


# --------------------------------------------------------------------------- packs built in code

def use(uid: str, engine: str, role: str = "reader", models: Sequence[str] = (), **params: Any) -> EngineUse:
    return EngineUse(uid, engine, role, dict(params), tuple(models))  # type: ignore[arg-type]


def make_pack(root: Path, pid: str, readers: Mapping[str, ReaderSet], *, models: Mapping[str, str] = (),  # type: ignore[assignment]
              base: bool = False, scripts: Sequence[str] = ("Latin",), osd_scripts: Sequence[str] = ("Latin",),
              normalize: Sequence[str] = ("nfc",), repairs: Sequence[StepUse] = (), fills: Sequence[StepUse] = (),
              must: Sequence[MustSurvive] = (), probe: Probe | None = None, tag: str | None = None) -> store.InstalledPack:
    """A pack folder with made-up model files (``models``: key -> engine) and its InstalledPack."""
    folder = root / f"{pid}-1.0.0"
    (folder / "models").mkdir(parents=True, exist_ok=True)
    assets = []
    for key, engine in dict(models).items():
        data = f"made-up model {pid}:{key}".encode()
        file = f"{key}.onnx"
        (folder / "models" / file).write_bytes(data)
        assets.append(Asset(key, file, engine, "onnx_rec", len(data), hashlib.sha256(data).hexdigest(),
                            "https://example.com/model", "MIT", None))
    lang = tag or pid
    m = Manifest(schema=1, id=pid, version="1.0.0", requires_core=">=0.1.0", base=base, title=pid.upper(),
                 languages=({"tag": lang, "iso639_3": "xxx", "role": "primary"},), scripts=tuple(scripts),
                 detect=Detect(tuple(osd_scripts), (lang,), (), frozenset(), frozenset(), 3, 1 / 3, probe),
                 readers=dict(readers), normalize=tuple(normalize), repairs=tuple(repairs), fills=tuple(fills),
                 models=tuple(assets), fonts=(), canary=None, must_survive=tuple(must), bench={}, raw_sha256="0" * 64)
    return store.InstalledPack(m, folder)


def floor_set(primary: EngineUse, checker: EngineUse, *, fallback: str = "none", **params: Any) -> ReaderSet:
    return ReaderSet((primary, checker), RuleUse("primary_with_floor", 1, {"primary": primary.id, "checker": checker.id,
                                                                          **params}), fallback)  # type: ignore[arg-type]


def single_set(*uses: EngineUse, reader: str, fallback: str = "none") -> ReaderSet:
    return ReaderSet(tuple(uses), RuleUse("single", 1, {"reader": reader}), fallback)  # type: ignore[arg-type]


def base_pack(root: Path) -> store.InstalledPack:
    return make_pack(root, "en", {"default": floor_set(
        use("reader", "fk_tess", models=("eng",), lang="en"),
        use("checker", "fk_chk", "checker", models=("chk",), rec="chk"))},
        models={"eng": "fk_tess", "chk": "fk_chk"}, base=True, fills=(StepUse("numbers", 1, {}),))


def thai_pack(root: Path, *, probe_min_chars: int = 20, probe_min_conf: float = 0.8, probe_min_line_chars: int = 2,
              script: str = "Thai") -> store.InstalledPack:
    probe = Probe(use("probe", "fk_probe", "probe", models=("en:chk", "probe-rec"), rec="probe-rec"),
                  probe_min_chars, probe_min_conf, probe_min_line_chars)
    return make_pack(root, "th", {"default": single_set(
        use("thr", "fk_thai", models=("th-rec",), lang="th"),
        use("checker", "fk_chk", "checker", models=("en:chk",), rec="en:chk"), reader="thr")},
        models={"th-rec": "fk_thai", "probe-rec": "fk_probe"}, scripts=(script, "Latin"), osd_scripts=(script,),
        normalize=("nfc", "thai_sara_am"), fills=(StepUse("numbers", 1, {}),), probe=probe,
        must=(MustSurvive("tone-marks", "่้๊๋ำ", "trigram", 0.75),))


def hu_pack(root: Path) -> store.InstalledPack:
    return make_pack(root, "hu", {"default": floor_set(
        use("tesseract", "fk_tess", models=("hun",), lang="hu"),
        use("rapidocr", "fk_chk", "checker", models=("en:chk",), rec="en:chk"))},
        models={"hun": "fk_tess"}, fills=(StepUse("numbers", 1, {}),),
        must=(MustSurvive("double-acute", "őűŐŰ", "word", 0.9),))


def page_class(index: int, *, needs_ocr: bool = True) -> PageClass:
    return PageClass(index, "image_only" if needs_ocr else "born_digital", 0, 0, 0.0, 1.0, needs_ocr, False)


def image_pdf(path: Path, pages: int, *, seed: int = 0) -> Path:
    """A PDF of ``pages`` image-only pages (2 x 1 in, a 300 x 150 px scan: rendered at the 200 dpi floor)."""
    specs = [build.PageSpec(width=144, height=72,
                            images=(build.ImageSpec(build.text_like(300, 150, lines=2, seed=seed + i)),))
             for i in range(pages)]
    return build.write_pdf(path, specs)


def plan_for(index: int, pack: str | None, *, composition: str = "script", lang: str | None = None,
             script: str | None = "Latin", warnings: Sequence[str] = (), languages: Sequence[LangCall] | None = None
             ) -> PagePlan:
    langs = list(languages) if languages is not None else [LangCall(lang or pack or "en", 1.0, "pack_script", 0)]
    return PagePlan(index, pack, composition, langs, True, "made up for a test",
                    script=ScriptCall(script, 0.0, "osd", 90.0) if script else None, warnings=list(warnings))


def pages_for(folder: Path, n: int) -> list[PageImage]:
    return [build.page_image(folder, image=build.text_like(300, 150, lines=2, seed=i), index=i, dpi=200.0)
            for i in range(n)]


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeOcr]:
    """The fake engines of the base, Thai and Hungarian packs, registered; page 0 Thai, 1 Hungarian, 2 English."""
    def tess(lang: str, i: int) -> Any:
        if lang == "en":
            return [THAI_TEXT[0].replace("ส", "?")] if i == 0 else (HU_TEXT if i == 1 else EN_TEXT)
        return HU_TEXT if i == 1 else EN_TEXT

    def chk(lang: str, i: int) -> Any:
        return ["4521 ?"] if i == 0 else (HU_TEXT + ["Ref 4521"] if i == 1 else EN_TEXT + ["Ref 4521"])

    def thai(lang: str, i: int) -> Any:
        return [t.replace("ำ", "ํา") for t in THAI_TEXT]          # sara am as the engines write it

    def probe(lang: str, i: int) -> Any:
        return ["Latin text only here"]

    out = {"fk_tess": FakeOcr("fk_tess", tess), "fk_chk": FakeOcr("fk_chk", chk),
           "fk_thai": FakeOcr("fk_thai", thai), "fk_probe": FakeOcr("fk_probe", probe)}
    register(monkeypatch, *out.values())
    return out


# --------------------------------------------------------------------------- the two passes

def test_the_two_passes_read_the_planned_packs_only_after_planning(tmp_path: Path, fakes: dict[str, FakeOcr]):
    root = tmp_path / "packs"
    en, th, hu = base_pack(root), thai_pack(root), hu_pack(root)
    pdf = image_pdf(tmp_path / "work.pdf", 3)
    sig = pipeline.gather(pdf, [page_class(i) for i in range(3)], en, {"en": en, "th": th, "hu": hu}, KEY,
                          tmp_path / "work", workers=2, osd={0: THAI_OSD, 1: None, 2: None})

    # Pass 1: the base pack's eng and checker reads on every page, the Thai probe on the Latin pages only.
    assert fakes["fk_tess"].calls("en") == [0, 1, 2] and fakes["fk_tess"].calls("hu") == []
    assert fakes["fk_chk"].calls() == [0, 1, 2]
    assert fakes["fk_probe"].calls() == [1, 2]
    assert fakes["fk_thai"].reads == [] and fakes["fk_thai"].opens == []      # no pack reader set yet
    assert [p.index for p in sig.pages] == [0, 1, 2] and all(p.path.is_file() for p in sig.pages)
    assert [s.script.name for s in sig.signals] == ["Thai", "Latin", "Latin"]
    assert sig.signals[1].script.eng_mean_conf == pytest.approx(90.0)      # mean_conf 0.9 on the 0-100 scale
    assert sig.signals[1].probes["th"].min_chars == 20                      # Probe.min_chars reaches run_probe
    assert sig.signals[0].probes == {} and sig.signals[0].windows == []
    assert sig.osd == {0: THAI_OSD, 1: None, 2: None}

    # Between the passes: planning, with nothing asked (§12).
    idx = I.load(FIXTURE_INDEX)
    plan = langplan.resolve_unsure(langplan.plan(sig.signals, idx, lang=(), force_lang=(),
                                                 installed={"en", "th", "hu"}, fetch="auto"), idx)
    assert [p.pack for p in plan.pages] == ["th", "hu", "en"]

    # Pass 2: each page with its pack; the pass-1 reads with the same configuration are reused.
    ocr = pipeline.read_pages(sig.pages, plan.pages, {"th": th, "hu": hu}, en, KEY, signals=sig, workers=2)
    assert [o.pack for o in ocr] == ["th", "hu", "en"]
    assert fakes["fk_thai"].calls() == [0]
    assert fakes["fk_tess"].calls("hu") == [1]
    assert fakes["fk_tess"].calls("en") == [0, 1, 2]                       # page 2's English reader came from pass 1
    assert fakes["fk_chk"].calls() == [0, 1, 2]                            # every checker read came from pass 1
    reused = {(o.index, r.use_id) for o in ocr for r in o.reads if r.meta.get("pass1")}
    assert reused == {(0, "checker"), (1, "rapidocr"), (2, "reader"), (2, "checker")}
    assert [r.role for r in ocr[1].reads] == ["reader", "checker"]
    assert [(r.use_id, r.engine) for r in ocr[1].reads] == [("tesseract", "fk_tess"), ("rapidocr", "fk_chk")]


def test_pass_1_reads_with_the_base_packs_default_set_whatever_the_platform_reads_english_with(
        tmp_path: Path, fakes: dict[str, FakeOcr]):
    """Detection's thresholds are measured on the default set's reads (DESIGN §1, pass 1), so a platform
    entry the kit picks for reading English changes how an English page is read, never what pass 1 reads.
    The platform set reuses the use id ``reader`` for another engine: a pass-1 read is reused only by a use
    configured exactly like the one that made it."""
    root = tmp_path / "packs"
    en = make_pack(root, "en", {
        "default": floor_set(use("reader", "fk_tess", models=("eng",), lang="en"),
                             use("checker", "fk_chk", "checker", models=("chk",), rec="chk")),
        KEY: single_set(use("reader", "fk_probe", models=("plat-rec",), rec="plat-rec"),
                        use("checker", "fk_chk", "checker", models=("chk",), rec="chk"), reader="reader")},
        models={"eng": "fk_tess", "chk": "fk_chk", "plat-rec": "fk_probe"}, base=True)
    sig = pipeline.gather(image_pdf(tmp_path / "work.pdf", 1), [page_class(0)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=1, osd={0: None})
    assert fakes["fk_tess"].calls("en") == [0] and fakes["fk_chk"].calls() == [0]
    assert fakes["fk_probe"].calls() == []
    assert sig.signals[0].eng_read.engine == "fk_tess" and sig.signals[0].checker_read.engine == "fk_chk"

    [o] = pipeline.read_pages(sig.pages, [plan_for(0, "en")], {}, en, KEY, signals=sig, workers=1)
    assert fakes["fk_probe"].calls() == [0]                                # the platform's reader reads in pass 2
    assert fakes["fk_tess"].calls("en") == [0] and fakes["fk_chk"].calls() == [0]
    assert [(r.use_id, r.engine, bool(r.meta.get("pass1"))) for r in o.reads] == [
        ("reader", "fk_probe", False), ("checker", "fk_chk", True)]


def test_gather_without_ocr_pages_renders_nothing(tmp_path: Path, fakes: dict[str, FakeOcr]):
    en = base_pack(tmp_path / "packs")
    sig = pipeline.gather(tmp_path / "missing.pdf", [page_class(0, needs_ocr=False)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=2, osd={})
    assert sig.pages == [] and sig.signals == [] and fakes["fk_tess"].reads == []


def test_osd_runs_on_the_render_when_orientation_gave_none(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                           monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[int, float]] = []

    class Osd:
        def detect(self, image: Any, dpi: float) -> OsdResult | None:
            seen.append((image.size[0], dpi))
            return THAI_OSD

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "_open_osd", lambda base: Osd())
    en = base_pack(tmp_path / "packs")
    pdf = image_pdf(tmp_path / "work.pdf", 2)
    sig = pipeline.gather(pdf, [page_class(0), page_class(1)], en, {"en": en}, KEY, tmp_path / "work",
                          workers=1, osd={1: None})
    assert sig.osd == {0: THAI_OSD, 1: None}
    assert [s.script.name for s in sig.signals] == ["Thai", "Latin"]
    assert len(seen) == 1 and seen[0][1] == pytest.approx(300.0, abs=2)   # the 200 dpi render scaled to about 300


def test_a_fired_probe_skips_the_mixed_script_check(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                    monkeypatch: pytest.MonkeyPatch):
    fakes["fk_probe"].script = lambda lang, i: THAI_TEXT * 2
    calls: list[int] = []
    real = detect.mixed_script_check
    monkeypatch.setattr(detect, "mixed_script_check", lambda page, eng, osd, calib: calls.append(page.index) or real(page, eng, osd, calib))
    root = tmp_path / "packs"
    en, th = base_pack(root), thai_pack(root)
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 1), [page_class(0)], en, {"en": en, "th": th}, KEY,
                          tmp_path / "work", workers=1, osd={0: None})
    assert sig.signals[0].probes["th"].chars >= 20 and calls == []


# --------------------------------------------------------------------------- the mixed-script trigger

LOW = [build.line(t, build.quad(10, 10 + 40 * k, 200, 40 + 40 * k), conf=0.2) for k, t in enumerate(["?? ??", "?? ?"])]


def test_an_engine_failure_in_the_mixed_script_check_leaves_the_page_unchecked(tmp_path: Path,
                                                                               fakes: dict[str, FakeOcr],
                                                                               monkeypatch: pytest.MonkeyPatch):
    fakes["fk_tess"].script = lambda lang, i: LOW
    calib = Calibration(0.5, 2.0, 1, 4, "kit 1.0 test")
    monkeypatch.setattr(langplan, "load_calibration", lambda: calib)

    class Osd:
        def detect(self, image: Any, dpi: float) -> OsdResult | None:
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "_open_osd", lambda base: Osd())

    def boom(page: Any, eng: Any, osd: Any, c: Any) -> MixedScript:
        if c is None:
            return MixedScript(False, None, 2, None)
        raise EngineFailure("page_unreadable", "the render can't be read for the mixed-script check")

    monkeypatch.setattr(detect, "mixed_script_check", boom)
    en = base_pack(tmp_path / "packs")
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 1), [page_class(0)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=1, osd={0: None})
    assert sig.signals[0].mixed == MixedScript(False, None, 2, None)
    idx = I.load(FIXTURE_INDEX)
    plan = langplan.plan(sig.signals, idx, lang=(), force_lang=(), installed={"en"}, fetch="auto")
    assert "mixed_script_unchecked" in plan.notes                          # the page is reported, not lost


def test_osd_that_cant_open_leaves_the_page_unchecked(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                      monkeypatch: pytest.MonkeyPatch):
    fakes["fk_tess"].script = lambda lang, i: LOW
    monkeypatch.setattr(langplan, "load_calibration", lambda: Calibration(0.5, 2.0, 1, 4, "kit 1.0 test"))
    en = base_pack(tmp_path / "packs")                     # no osd.traineddata: OSD can't open
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 1), [page_class(0)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=1, osd={})
    assert sig.osd == {0: None}
    m = sig.signals[0].mixed
    assert not m.checked and m.script is None and m.blocks == 1             # both low lines form one block


def test_a_calibrated_trigger_runs_with_the_threads_osd(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                        monkeypatch: pytest.MonkeyPatch):
    fakes["fk_tess"].script = lambda lang, i: LOW
    monkeypatch.setattr(langplan, "load_calibration", lambda: Calibration(0.5, 2.0, 1, 4, "kit 1.0 test"))

    class Osd:
        def detect(self, image: Any, dpi: float) -> OsdResult | None:
            return OsdResult("Thai", 9.0, 0, 4.0)

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "_open_osd", lambda base: Osd())
    en = base_pack(tmp_path / "packs")
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 1), [page_class(0)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=1, osd={0: None})
    assert sig.signals[0].mixed.checked and sig.signals[0].mixed.script == "Thai"


# --------------------------------------------------------------------------- threading

def test_renders_happen_in_the_calling_thread_at_most_two_per_worker_ahead(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                           monkeypatch: pytest.MonkeyPatch):
    workers, n = 2, 12
    caller = threading.get_ident()
    state = {"rendered": 0, "finished": 0, "peak": 0, "threads": set()}
    lock = threading.Lock()

    class Spy(pipeline.Renderer):
        def render(self, index: int, dpi: float, dest: Path) -> PageImage:
            with lock:
                state["threads"].add(threading.get_ident())
                state["peak"] = max(state["peak"], state["rendered"] - state["finished"])
                state["rendered"] += 1
                state.setdefault("locked", []).append(pipeline.PDFIUM_LOCK._is_owned())  # type: ignore[attr-defined]
            return super().render(index, dpi, dest)

    real = pipeline._page_signals

    def slow(ctx: Any, page: PageImage, given: Any) -> Any:
        time.sleep(0.03)
        try:
            return real(ctx, page, given)
        finally:
            with lock:
                state["finished"] += 1

    monkeypatch.setattr(pipeline, "Renderer", Spy)
    monkeypatch.setattr(pipeline, "_page_signals", slow)
    en = base_pack(tmp_path / "packs")
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", n), [page_class(i) for i in range(n)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=workers, osd={i: None for i in range(n)})
    assert state["threads"] == {caller}
    assert state["locked"] == [True] * n                       # every render under PDFIUM_LOCK (§2.2)
    assert state["peak"] <= 2 * workers - 1                    # with the one being rendered: at most 2 x workers
    assert state["peak"] >= workers                            # the renders do run ahead of the workers
    assert len(sig.signals) == n
    reader_threads = {r[3] for r in fakes["fk_tess"].reads}
    assert caller not in reader_threads                        # engines read only in the workers


def test_one_session_per_thread_and_language_string_and_one_shared_session(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch):
    tess = FakeOcr("fk_tess", lambda lang, i: EN_TEXT, delay=0.01)
    shared = FakeOcr("fk_shared", lambda lang, i: EN_TEXT, per_thread=False, delay=0.01)
    register(monkeypatch, tess, shared)
    root = tmp_path / "packs"
    en = make_pack(root, "en", {"default": floor_set(use("reader", "fk_tess", models=("eng",), lang="en"),
                                                     use("checker", "fk_shared", "checker", models=("chk",), rec="chk"))},
                   models={"eng": "fk_tess", "chk": "fk_shared"}, base=True)
    xa = make_pack(root, "xa", {"default": floor_set(use("reader", "fk_tess", models=("xam",), lang="xa"),
                                                     use("checker", "fk_shared", "checker", models=("en:chk",),
                                                         rec="en:chk"))},
                   models={"xam": "fk_tess"})
    n = 12
    pages = pages_for(tmp_path, n)
    plans = [plan_for(i, "en" if i % 2 else "xa") for i in range(n)]
    pipeline.read_pages(pages, plans, {"xa": xa}, en, KEY, signals=None, workers=3)
    for lang in ("en", "xa"):
        sessions = [s for s in tess.sessions if s.params["lang"] == lang]
        assert 1 <= len(sessions) <= 3
        assert all(len(s.threads) == 1 for s in sessions)                       # never shared across threads
        assert len({next(iter(s.threads)) for s in sessions}) == len(sessions)  # one per thread
    assert len(shared.sessions) == 1                       # the same configuration (en:chk is chk): one session
    assert len(shared.sessions[0].threads) >= 2
    assert all(s.closed for s in tess.sessions + shared.sessions)           # every session closed at the end


class ClosableOsd:
    def __init__(self, opened: list["ClosableOsd"]) -> None:
        self.closed = False
        opened.append(self)

    def detect(self, image: Any, dpi: float) -> OsdResult | None:
        return None

    def close(self) -> None:
        self.closed = True


def test_gather_closes_every_session_it_opened_osd_included(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                            monkeypatch: pytest.MonkeyPatch):
    osds: list[ClosableOsd] = []
    lock = threading.Lock()

    def open_osd(base: Any) -> ClosableOsd:
        with lock:
            return ClosableOsd(osds)

    monkeypatch.setattr(pipeline, "_open_osd", open_osd)
    for f in fakes.values():
        f.delay = 0.01
    root = tmp_path / "packs"
    en, th = base_pack(root), thai_pack(root)
    n = 6
    pipeline.gather(image_pdf(tmp_path / "w.pdf", n), [page_class(i) for i in range(n)], en, {"en": en, "th": th},
                    KEY, tmp_path / "work", workers=3, osd={})
    opened = [s for f in fakes.values() for s in f.sessions]
    assert osds and opened                                    # OSD and the engines did open in pass 1
    assert all(o.closed for o in osds)
    assert all(s.closed for s in opened)


def test_osd_that_cant_open_is_tried_once_per_pass_across_threads(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                   monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []
    lock = threading.Lock()

    def open_osd(base: Any) -> Any:
        with lock:
            calls.append(threading.get_ident())
        time.sleep(0.05)
        raise EngineFailure("osd_unavailable", "made-up: OSD can't run here")

    monkeypatch.setattr(pipeline, "_open_osd", open_osd)
    en = base_pack(tmp_path / "packs")
    n = 8
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", n), [page_class(i) for i in range(n)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=4, osd={})
    assert sig.osd == {i: None for i in range(n)}
    assert len(calls) == 1


def test_models_are_checked_immediately_before_each_open(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                         monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[int, str, str]] = []
    real_paths = store.model_paths

    def paths(pack: Any, base: Any, u: EngineUse) -> dict[str, Path]:
        out = real_paths(pack, base, u)
        events.append((threading.get_ident(), "checked", u.id))
        return out

    monkeypatch.setattr(store, "model_paths", paths)
    for f in fakes.values():
        real_open = f.open

        def spy(use_id: str, role: str, params: Any, models: Any, _real: Any = real_open) -> Any:
            events.append((threading.get_ident(), "open", use_id))
            return _real(use_id, role, params, models)

        monkeypatch.setattr(f, "open", spy)
    root = tmp_path / "packs"
    en, th = base_pack(root), thai_pack(root)
    pipeline.read_pages(pages_for(tmp_path, 4), [plan_for(i, "th" if i % 2 else "en") for i in range(4)],
                        {"th": th}, en, KEY, signals=None, workers=2)
    opens = 0
    for k, (thread, kind, uid) in enumerate(events):
        if kind != "open":
            continue
        opens += 1
        before = [e for e in events[:k] if e[0] == thread]
        assert before and before[-1] == (thread, "checked", uid)       # this thread checked this use's models last
    assert opens >= 3


def test_a_tampered_model_stops_the_file_before_any_engine_opens_it(tmp_path: Path, fakes: dict[str, FakeOcr]):
    root = tmp_path / "packs"
    en, th = base_pack(root), thai_pack(root)
    model = th.root / "models" / "th-rec.onnx"
    model.write_bytes(model.read_bytes()[::-1])                            # same size, other bytes
    with pytest.raises(IntegrityError) as info:
        pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "th")], {"th": th}, en, KEY, signals=None, workers=1)
    assert str(model) in str(info.value)
    assert fakes["fk_thai"].opens == []


def test_workers_must_be_a_whole_number(tmp_path: Path, fakes: dict[str, FakeOcr]):
    en = base_pack(tmp_path / "packs")
    with pytest.raises(ValueError):
        pipeline.read_pages([], [], {}, en, KEY, signals=None, workers=2.5)  # type: ignore[arg-type]
    assert pipeline.read_pages([], [], {}, en, KEY, signals=None, workers=0) == []


# --------------------------------------------------------------------------- pass 2 on its own (the kit's way)

def test_read_pages_reads_a_draft_manifest_without_index_store_or_signals(tmp_path: Path,
                                                                          fakes: dict[str, FakeOcr]):
    root = tmp_path / "packs"
    en = base_pack(root)
    draft = make_pack(tmp_path / "draft", "xq", {"default": floor_set(
        use("tesseract", "fk_tess", models=("xqm",), lang="xq"),
        use("rapidocr", "fk_chk", "checker", models=("en:chk",), rec="en:chk"))}, models={"xqm": "fk_tess"})
    pages = pages_for(tmp_path, 2)
    plans = [plan_for(i, "xq", composition="forced") for i in range(2)]
    ocr = pipeline.read_pages(pages, plans, {"xq": draft}, en, KEY, signals=None, workers=2)
    assert [o.pack for o in ocr] == ["xq", "xq"]
    assert fakes["fk_tess"].calls("xq") == [0, 1]
    assert ocr[0].pick.rule == "primary_with_floor" and ocr[0].pick.chosen == "tesseract"

    # Another candidate set for the same draft, as the kit's bench passes it.
    other = single_set(use("only", "fk_chk", models=("en:chk",), rec="en:chk"), reader="only")
    variant = store.InstalledPack(dataclasses.replace(draft.manifest, readers={KEY: other}), draft.root)
    ocr2 = pipeline.read_pages(pages, plans, {"xq": variant}, en, KEY, signals=None, workers=1)
    assert ocr2[0].pick.rule == "single" and ocr2[0].pick.chosen == "only"
    assert [r.use_id for r in ocr2[0].reads] == ["only"]


def test_a_plan_with_no_pack_gets_no_reads_and_no_layer(tmp_path: Path, fakes: dict[str, FakeOcr]):
    en = base_pack(tmp_path / "packs")
    pages = pages_for(tmp_path, 1)
    [o] = pipeline.read_pages(pages, [plan_for(0, None, composition="none")], {}, en, KEY, signals=None, workers=1)
    assert o.reads == [] and o.pick.chosen is None and o.warnings == [] and o.pack is None
    assert fakes["fk_tess"].reads == []
    [lp] = pipeline.layer_pages([o], pages, [page_class(0)])
    assert lp.lines == ()


@pytest.mark.parametrize("composition", ["undecided", "script", "strongest"])
def test_a_plan_with_no_pack_that_isnt_lang_none_is_refused(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                           composition: str):
    en = base_pack(tmp_path / "packs")
    pages = pages_for(tmp_path, 2)
    pp = PagePlan(1, None, composition, [], False, "made up for a test", script=ScriptCall("Latin", 0.0, "osd", 90.0))
    with pytest.raises(ValueError, match="resolve_unsure"):
        pipeline.read_pages(pages, [plan_for(0, "en"), pp], {}, en, KEY, signals=None, workers=1)
    assert fakes["fk_tess"].reads == [] and fakes["fk_tess"].opens == []   # refused before any page is read


def test_bad_plans_are_refused(tmp_path: Path, fakes: dict[str, FakeOcr]):
    en = base_pack(tmp_path / "packs")
    pages = pages_for(tmp_path, 1)
    with pytest.raises(ValueError, match="no render"):
        pipeline.read_pages(pages, [plan_for(3, "en")], {}, en, KEY, signals=None, workers=1)
    with pytest.raises(ValueError, match="twice"):
        pipeline.read_pages(pages, [plan_for(0, "en"), plan_for(0, "en")], {}, en, KEY, signals=None, workers=1)
    with pytest.raises(ValueError, match="wasn't supplied"):
        pipeline.read_pages(pages, [plan_for(0, "zz")], {}, en, KEY, signals=None, workers=1)


def test_an_unknown_normaliser_or_fill_is_a_pack_error_before_any_read(tmp_path: Path, fakes: dict[str, FakeOcr]):
    root = tmp_path / "packs"
    en = base_pack(root)
    for kw in ({"normalize": ("nfc", "no_such")}, {"fills": (StepUse("no_such", 1, {}),)}):
        bad = make_pack(root / str(len(kw)), "xa", {"default": single_set(use("r", "fk_tess", models=("m",)), reader="r")},
                        models={"m": "fk_tess"}, **kw)  # type: ignore[arg-type]
        with pytest.raises(PackError):
            pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "xa")], {"xa": bad}, en, KEY,
                                signals=None, workers=1)
    assert fakes["fk_tess"].reads == []


# --------------------------------------------------------------------------- what a page holds

def test_normalisers_run_on_every_read_and_never_on_the_pass_1_signals(tmp_path: Path, fakes: dict[str, FakeOcr]):
    decomposed = "ทํา"                       # nikhahit + sara aa, as engines write sara am
    fakes["fk_thai"].script = lambda lang, i: [decomposed + " 4521"]
    fakes["fk_chk"].script = lambda lang, i: [decomposed + " 4521"]
    root = tmp_path / "packs"
    en, th = base_pack(root), thai_pack(root)
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 1), [page_class(0)], en, {"en": en, "th": th}, KEY,
                          tmp_path / "work", workers=1, osd={0: THAI_OSD})
    [o] = pipeline.read_pages(sig.pages, [plan_for(0, "th", script="Thai")], {"th": th}, en, KEY,
                              signals=sig, workers=1)
    assert [r.text() for r in o.reads] == ["ทำ 4521"] * 2           # the checker's read too
    assert o.pick.lines[0].text == "ทำ 4521"
    assert sig.signals[0].checker_read.text() == decomposed + " 4521"        # pass 1 keeps its raw read


def test_the_layer_is_the_chosen_lines_then_the_fills(tmp_path: Path, fakes: dict[str, FakeOcr]):
    fakes["fk_tess"].script = lambda lang, i: ["Example invoice for the sample order"]
    fakes["fk_chk"].script = lambda lang, i: ["Example invoice for the sample order", "Reference 4521-7788"]
    en = base_pack(tmp_path / "packs")
    pages = pages_for(tmp_path, 1)
    [o] = pipeline.read_pages(pages, [plan_for(0, "en")], {}, en, KEY, signals=None, workers=1)
    assert [f.kind for f in o.fills] == ["number"] and o.fills[0].source == "checker"
    [lp] = pipeline.layer_pages([o], pages, [page_class(0)])
    assert [ln.text for ln in lp.lines] == ["Example invoice for the sample order", o.fills[0].text]
    assert "4521" in o.fills[0].text
    assert lp.lines[0].box == o.pick.lines[0].box and lp.lines[1].box == o.fills[0].box
    assert (lp.render_w, lp.render_h) == (pages[0].width, pages[0].height)


def test_layer_pages_refuses_a_page_with_printed_text(tmp_path: Path, fakes: dict[str, FakeOcr]):
    en = base_pack(tmp_path / "packs")
    pages = pages_for(tmp_path, 1)
    ocr = pipeline.read_pages(pages, [plan_for(0, "en")], {}, en, KEY, signals=None, workers=1)
    with pytest.raises(ValueError, match="never gets a layer"):
        pipeline.layer_pages(ocr, pages, [page_class(0, needs_ocr=False)])


@pytest.mark.parametrize("classes", [[], [1]], ids=["no-classes", "page-missing"])
def test_layer_pages_fails_closed_for_a_page_without_its_class(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                               classes: list[int]):
    en = base_pack(tmp_path / "packs")
    pages = pages_for(tmp_path, 2)
    ocr = pipeline.read_pages(pages[:1], [plan_for(0, "en")], {}, en, KEY, signals=None, workers=1)
    assert ocr[0].pick.lines                                    # there is something it could have written
    with pytest.raises(ValueError, match="never gets a layer"):
        pipeline.layer_pages(ocr, pages, [page_class(i) for i in classes])


def test_must_survive_is_counted_per_engine_and_for_the_layer(tmp_path: Path, fakes: dict[str, FakeOcr]):
    fakes["fk_tess"].script = lambda lang, i: ["Az erdő fűzfája szép", "egy tűz"]
    fakes["fk_chk"].script = lambda lang, i: ["Az erdo fűzfája szép", "egy tűz és 4521"]
    root = tmp_path / "packs"
    en, hu = base_pack(root), hu_pack(root)
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "hu")], {"hu": hu}, en, KEY,
                              signals=None, workers=1)
    assert o.must_survive == {"double-acute": {"tesseract": 3, "rapidocr": 2, "layer": 3}}

    ms = {u: MustSurvive("x", "่้ำ", u, 0.5) for u in ("char", "word", "trigram")}
    text = "กา น้ำ"           # two Thai words; the second has a tone mark and sara am
    assert pipeline.must_survive_count(text, ms["char"]) == 2
    assert pipeline.must_survive_count(text, ms["word"]) == 1
    # The joined Thai run, NFKC (sara am as nikhahit + sara aa), has four trigrams; the three after the
    # first hold the tone mark or the sara am.
    assert pipeline.must_survive_count(text, ms["trigram"]) == 3
    assert pipeline.must_survive_count("กานา", ms["trigram"]) == 0


def test_a_second_language_pack_adds_its_must_survive_counts(tmp_path: Path, fakes: dict[str, FakeOcr]):
    fakes["fk_tess"].script = lambda lang, i: ["The example form and its szőlő fűzfa"]
    root = tmp_path / "packs"
    en, hu = base_pack(root), hu_pack(root)
    plan = plan_for(0, "en", languages=[LangCall("en", 0.9, "langid", 4), LangCall("hu", 0.8, "langid", 3)],
                    warnings=["mixed_language_page"])
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan], {"hu": hu}, en, KEY, signals=None, workers=1)
    assert o.must_survive["double-acute"]["layer"] == 2
    assert o.warnings == ["mixed_language_page"]


def test_repairs_run_on_the_chosen_lines_with_their_source(tmp_path: Path, fakes: dict[str, FakeOcr]):
    fakes["fk_tess"].script = lambda lang, i: ["Die Bar ist offen"]
    fakes["fk_chk"].script = lambda lang, i: ["Die Bär ist offen"]
    root = tmp_path / "packs"
    en = base_pack(root)
    de = make_pack(root, "de", {"default": floor_set(use("tesseract", "fk_tess", models=("deu",), lang="de"),
                                                     use("rapidocr", "fk_chk", "checker", models=("en:chk",),
                                                         rec="en:chk"))},
                   models={"deu": "fk_tess"},
                   repairs=(StepUse("diacritic_word", 1, {"source": "rapidocr", "table": {"ä": ["a", "á"]}}),))
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "de")], {"de": de}, en, KEY,
                              signals=None, workers=1)
    assert o.pick.lines[0].text == "Die Bär ist offen"
    assert o.repairs == {"diacritic_word.fixed": 1, "diacritic_word.unpartnered": 0}
    assert o.reads[0].text() == "Die Bar ist offen"                          # the read itself is left as it was


def test_a_repair_sees_the_pages_detected_languages(tmp_path: Path, fakes: dict[str, FakeOcr]):
    # diacritic_word v2 runs only where German was detected: the same read is repaired on a
    # Hungarian-German page and left alone on a Hungarian one
    fakes["fk_tess"].script = lambda lang, i: ["Die Gártnerin ist hier"]
    fakes["fk_chk"].script = lambda lang, i: ["Die Gärtnerin ist hier"]
    root = tmp_path / "packs"
    en = base_pack(root)
    params = {"source": "rapidocr", "table": {"ä": ["a", "á"]}, "languages": ["de"], "never": "őű",
              "context": ["die", "ist"]}
    hu = make_pack(root, "hu", {"default": floor_set(use("tesseract", "fk_tess", models=("hun",), lang="hu"),
                                                     use("rapidocr", "fk_chk", "checker", models=("en:chk",),
                                                         rec="en:chk"))},
                   models={"hun": "fk_tess"}, repairs=(StepUse("diacritic_word", 2, params),))
    both = plan_for(0, "hu", languages=[LangCall("hu", 0.7, "langid", 3), LangCall("de", 0.3, "langid", 1)])
    one = plan_for(1, "hu", languages=[LangCall("hu", 0.9, "langid", 3)])
    o_both, o_one = pipeline.read_pages(pages_for(tmp_path, 2), [both, one], {"hu": hu}, en, KEY,
                                        signals=None, workers=1)
    assert o_both.pick.lines[0].text == "Die Gärtnerin ist hier"
    assert o_both.repairs["diacritic_word.fixed"] == 1
    assert o_one.pick.lines[0].text == "Die Gártnerin ist hier"
    assert o_one.repairs["diacritic_word.fixed"] == 0


# --------------------------------------------------------------------------- warnings

def two_readers(root: Path, *, fallback: str = "none") -> tuple[store.InstalledPack, store.InstalledPack]:
    en = base_pack(root)
    xa = make_pack(root, "xa", {"default": ReaderSet(
        (use("first", "fk_tess", models=("am",), lang="xa"), use("second", "fk_two", models=("bm",), lang="xa"),
         use("checker", "fk_chk", "checker", models=("en:chk",), rec="en:chk")),
        RuleUse("corroborated_trigrams", 1, {"script": "Latin", "tie_order": ["first", "second"]}),
        fallback)}, models={"am": "fk_tess", "bm": "fk_two"})  # type: ignore[arg-type]
    return en, xa


@pytest.mark.parametrize("failure,code", [
    (EngineFailure("engine_failed", "made-up failure: the engine stopped"), "reader_failed"),
    ("failed", "reader_failed"),
    ("timeout", "reader_timeout"),
])
def test_a_degraded_reader_is_warned_in_core_wording(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                     monkeypatch: pytest.MonkeyPatch, failure: Any, code: str):
    fakes["fk_tess"].script = lambda lang, i: failure
    register(monkeypatch, FakeOcr("fk_two", lambda lang, i: EN_TEXT))
    en, xa = two_readers(tmp_path / "packs")
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "xa")], {"xa": xa}, en, KEY,
                              signals=None, workers=1)
    assert o.warnings == [code]
    assert o.pick.chosen == "second"
    first = o.reads[0]
    if isinstance(failure, EngineFailure):
        assert first.status == "failed" and first.error == "made-up failure: the engine stopped"
    entry = record.page_entry(o)
    assert entry["warnings"] == [code]
    assert entry["readers"][0]["status"] == ("timeout" if code == "reader_timeout" else "failed")
    msg = record.warning(code, page=1)
    assert msg.message == record.WARNING_MESSAGES[code]
    assert "comes from the others" in msg.message


def test_no_reader_working_gives_no_layer_and_one_warning(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                          monkeypatch: pytest.MonkeyPatch):
    fakes["fk_tess"].script = lambda lang, i: "failed"
    register(monkeypatch, FakeOcr("fk_two", lambda lang, i: "timeout"))
    en, xa = two_readers(tmp_path / "packs")
    pages = pages_for(tmp_path, 1)
    [o] = pipeline.read_pages(pages, [plan_for(0, "xa")], {"xa": xa}, en, KEY, signals=None, workers=1)
    assert o.pick.chosen is None and o.warnings == ["no_layer_readers_failed"] and o.fills == []
    assert pipeline.layer_pages([o], pages, [page_class(0)])[0].lines == ()


def test_the_checker_fallback_is_named(tmp_path: Path, fakes: dict[str, FakeOcr], monkeypatch: pytest.MonkeyPatch):
    fakes["fk_tess"].script = lambda lang, i: "failed"
    register(monkeypatch, FakeOcr("fk_two", lambda lang, i: "failed"))
    en, xa = two_readers(tmp_path / "packs", fallback="checker")
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "xa")], {"xa": xa}, en, KEY,
                              signals=None, workers=1)
    assert o.pick.chosen == "checker" and o.pick.fallback
    assert o.warnings == ["reader_failed", "fallback_used"]


def test_the_rules_own_warning_is_copied_onto_the_page(tmp_path: Path, fakes: dict[str, FakeOcr]):
    words = " ".join(f"sample{k:02d}word" for k in range(30))
    fakes["fk_tess"].script = lambda lang, i: ["only a short read"]
    fakes["fk_chk"].script = lambda lang, i: [words]
    en = base_pack(tmp_path / "packs")
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "en")], {}, en, KEY, signals=None, workers=1)
    assert o.pick.chosen == "reader" and o.pick.warnings == ["agreement_low"]
    assert o.warnings == ["agreement_low"]


def test_plan_and_render_warnings_come_first(tmp_path: Path, fakes: dict[str, FakeOcr],
                                             monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline.Renderer, "render_capped", lambda self, index: index == 0)
    en = base_pack(tmp_path / "packs")
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 2), [page_class(0), page_class(1)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=1, osd={0: None, 1: None})
    assert sig.warnings == {0: ["render_capped"]}
    plans = [plan_for(0, "en", warnings=["language_uncertain"]), plan_for(1, "en")]
    ocr = pipeline.read_pages(sig.pages, plans, {}, en, KEY, signals=sig, workers=1)
    assert ocr[0].warnings == ["language_uncertain", "render_capped"] and ocr[1].warnings == []
    assert all(w in record.WARNING_CODES for o in ocr for w in o.warnings)


@pytest.mark.parametrize("state,status,warnings,notes", [
    ("not_on_platform", "unavailable", [], ["vision_not_on_platform"]),
    ("unsupported_language", "unavailable", ["vision_language_unsupported"], []),
    ("broken", "failed", ["reader_failed"], []),
    ("missing_model", "failed", ["reader_failed"], []),
])
def test_an_unusable_vision_use(tmp_path: Path, fakes: dict[str, FakeOcr], monkeypatch: pytest.MonkeyPatch,
                                state: str, status: str, warnings: list[str], notes: list[str]):
    vision = FakeOcr("vision", lambda lang, i: EN_TEXT, avail=(state, f"made-up {state}"), per_thread=False)
    register(monkeypatch, vision)
    root = tmp_path / "packs"
    en = base_pack(root)
    xa = make_pack(root, "xa", {"default": ReaderSet(
        (use("vision", "vision"), use("first", "fk_tess", models=("am",), lang="xa"),
         use("checker", "fk_chk", "checker", models=("en:chk",), rec="en:chk")),
        RuleUse("corroborated_trigrams", 1, {"script": "Latin", "tie_order": ["vision", "first"]}), "none")},
        models={"am": "fk_tess"})
    [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "xa")], {"xa": xa}, en, KEY,
                              signals=None, workers=1)
    v = o.reads[0]
    assert (v.use_id, v.engine, v.status, v.meta["state"]) == ("vision", "vision", status, state)
    assert o.pick.chosen == "first" and o.warnings == warnings
    assert pipeline.notes_for(o) == notes
    assert vision.opens == [] and vision.reads == []
    assert all(n in record.NOTE_CODES for n in notes)


def test_an_engine_that_cant_open_fails_its_reads_and_is_tried_once(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                    monkeypatch: pytest.MonkeyPatch):
    register(monkeypatch, FakeOcr("fk_two", open_error="made-up: the model can't be loaded"))
    fakes["fk_tess"].script = lambda lang, i: EN_TEXT
    en, xa = two_readers(tmp_path / "packs")
    ocr = pipeline.read_pages(pages_for(tmp_path, 4), [plan_for(i, "xa") for i in range(4)], {"xa": xa}, en, KEY,
                              signals=None, workers=1)
    assert all(o.reads[1].status == "failed" and o.reads[1].error == "made-up: the model can't be loaded" for o in ocr)
    assert all(o.warnings == ["reader_failed"] and o.pick.chosen == "first" for o in ocr)
    assert len(engines.get("fk_two").opens) == 1


@pytest.mark.parametrize("workers", [2, 4])
def test_an_engine_that_cant_open_is_tried_once_per_pass_across_threads(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                        monkeypatch: pytest.MonkeyPatch, workers: int):
    two = FakeOcr("fk_two", open_error="made-up: the model can't be loaded")
    real_open = two.open

    def slow_open(*a: Any, **k: Any) -> Any:
        time.sleep(0.05)                                        # every thread reaches the open before it fails
        return real_open(*a, **k)

    monkeypatch.setattr(two, "open", slow_open)
    register(monkeypatch, two)
    fakes["fk_tess"].script = lambda lang, i: EN_TEXT
    en, xa = two_readers(tmp_path / "packs")
    ocr = pipeline.read_pages(pages_for(tmp_path, 8), [plan_for(i, "xa") for i in range(8)], {"xa": xa}, en, KEY,
                              signals=None, workers=workers)
    assert all(o.reads[1].status == "failed" for o in ocr)
    assert len(two.opens) == 1
    assert len({r[3] for r in fakes["fk_tess"].reads}) >= 2     # the working reader still opens per thread


def test_pass_1_reads_are_reused_only_on_the_render_they_read(tmp_path: Path, fakes: dict[str, FakeOcr]):
    en = base_pack(tmp_path / "packs")
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 1), [page_class(0)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=1, osd={0: None})
    fakes["fk_tess"].reads.clear()
    fakes["fk_chk"].reads.clear()
    [same] = pipeline.read_pages(sig.pages, [plan_for(0, "en")], {}, en, KEY, signals=sig, workers=1)
    assert all(r.meta.get("pass1") for r in same.reads) and fakes["fk_tess"].reads == []

    g = sig.pages[0]
    (tmp_path / "other").mkdir()
    other_path = build.page_image(tmp_path / "other", image=build.text_like(g.width, g.height, lines=2, seed=0),
                                  index=0, dpi=g.dpi)
    other_dpi = build.page_image(tmp_path / "other", image=build.text_like(g.width * 2, g.height * 2, lines=2, seed=0),
                                 index=0, dpi=g.dpi * 2, name="hi.png")
    for page in (other_path, other_dpi):
        fakes["fk_tess"].reads.clear()
        fakes["fk_chk"].reads.clear()
        [o] = pipeline.read_pages([page], [plan_for(0, "en")], {}, en, KEY, signals=sig, workers=1)
        assert not any(r.meta.get("pass1") for r in o.reads)
        assert fakes["fk_tess"].calls() == [0] and fakes["fk_chk"].calls() == [0]   # read again, on this render


def test_an_engine_module_that_cant_load_degrades_only_its_reader(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(engines._REGISTRY, "fk_two", "pdfintake.no_such_module_for_a_test:ENGINE")
    engines._clear_cache()
    fakes["fk_tess"].script = lambda lang, i: EN_TEXT
    en, xa = two_readers(tmp_path / "packs")
    try:
        [o] = pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "xa")], {"xa": xa}, en, KEY,
                                  signals=None, workers=1)
    finally:
        engines._clear_cache()
    assert o.reads[1].status == "failed" and "could not be loaded" in (o.reads[1].error or "")
    assert o.warnings == ["reader_failed"]


def test_an_unknown_engine_name_is_a_pack_error(tmp_path: Path, fakes: dict[str, FakeOcr]):
    root = tmp_path / "packs"
    en = base_pack(root)
    xa = make_pack(root, "xa", {"default": single_set(use("r", "no_such_engine"), reader="r")})
    with pytest.raises(PackError):
        pipeline.read_pages(pages_for(tmp_path, 1), [plan_for(0, "xa")], {"xa": xa}, en, KEY, signals=None, workers=1)


def test_nothing_in_a_page_result_reaches_the_record_as_text(tmp_path: Path, fakes: dict[str, FakeOcr]):
    root = tmp_path / "packs"
    en, hu = base_pack(root), hu_pack(root)
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 3), [page_class(i) for i in range(3)], en, {"en": en}, KEY,
                          tmp_path / "work", workers=2, osd={0: None, 1: None, 2: None})
    ocr = pipeline.read_pages(sig.pages, [plan_for(i, "hu" if i == 1 else "en") for i in range(3)], {"hu": hu}, en,
                              KEY, signals=sig, workers=2)
    for o in ocr:
        entry = record.page_entry(o)
        blob = repr(entry)
        assert "tapasztalják" not in blob and "invoice" not in blob


# --------------------------------------------------------------------------- the detection cache (DESIGN §9.1)

class Disk:
    """flow's detection cache on disk, in a pdfintake cache under the test's folder, and run work
    folders inside that cache (a hit copies the renders into the call's work folder, as in a run)."""

    def __init__(self) -> None:
        self.runs = 0

    def cache(self) -> Any:
        from pdfintake import flow, fsutil

        return flow._DetectionCache(fsutil.work_dir("work", "ab" * 32 + "-0.1.0"))

    def work(self) -> Path:
        from pdfintake import fsutil

        self.runs += 1
        return fsutil.work_dir("work", f"run-{self.runs}") / "ocr"

    def entries(self) -> list[Path]:
        return sorted(p for p in self.cache().folder.iterdir() if p.name.startswith("e-"))

    def entry_file(self) -> Path:
        (entry,) = self.entries()
        return entry / pipeline.CACHE_FILE


@pytest.fixture
def disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Disk:
    monkeypatch.setenv("PDFINTAKE_CACHE", str(tmp_path / "cache"))
    return Disk()


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every pass-1 render, by page index."""
    out: list[int] = []

    class Counting(pipeline.Renderer):
        def render(self, index: int, dpi: float, dest: Path) -> PageImage:
            out.append(index)
            return super().render(index, dpi, dest)

    monkeypatch.setattr(pipeline, "Renderer", Counting)
    return out


def reads_of(fakes: Mapping[str, FakeOcr]) -> dict[str, int]:
    return {name: len(f.reads) for name, f in fakes.items()}


def without_paths(sig: pipeline.DocumentSignals) -> list[PageImage]:
    return [dataclasses.replace(p, path=Path(Path(p.path).name)) for p in sig.pages]


def test_a_second_gather_takes_pass_1_from_the_cache_and_renders_and_reads_nothing(
        tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk, renders: list[int], monkeypatch: pytest.MonkeyPatch):
    osd_calls: list[int] = []

    class Osd:
        def detect(self, image: Any, dpi: float) -> OsdResult | None:
            osd_calls.append(1)
            return THAI_OSD

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "_open_osd", lambda base: Osd())
    root = tmp_path / "packs"
    en, th, hu = base_pack(root), thai_pack(root), hu_pack(root)
    installed = {"en": en, "th": th, "hu": hu}
    pdf = image_pdf(tmp_path / "work.pdf", 3)
    classes = [page_class(i) for i in range(3)]
    first = pipeline.gather(pdf, classes, en, installed, KEY, disk.work(), workers=2, osd={1: None, 2: None},
                            cache=disk.cache())
    assert not first.cached and renders == [0, 1, 2] and len(osd_calls) == 1
    assert len(disk.entries()) == 1
    before = reads_of(fakes)
    opened = {name: len(f.opens) for name, f in fakes.items()}

    work = disk.work()
    again = pipeline.gather(pdf, classes, en, installed, KEY, work, workers=2, osd={1: None, 2: None},
                            cache=disk.cache())
    assert again.cached
    assert renders == [0, 1, 2] and len(osd_calls) == 1        # nothing rendered, OSD not run again
    assert reads_of(fakes) == before                           # no eng, checker or probe read
    assert {name: len(f.opens) for name, f in fakes.items()} == opened   # no engine even opened
    # The same result, its renders copied into this call's own work folder.
    assert [p.path for p in again.pages] == [work / f"ocr-{i:05d}.png" for i in range(3)]
    assert without_paths(again) == without_paths(first)
    assert [p.path.read_bytes() for p in again.pages] == [p.path.read_bytes() for p in first.pages]
    assert again.signals == first.signals
    assert again.osd == first.osd == {0: THAI_OSD, 1: None, 2: None}
    assert (again.warnings, again.notes, again.classes) == (first.warnings, first.notes, classes)

    # Planning and pass 2 on the cached result go exactly as on the fresh one, pass 1's reads reused.
    idx = I.load(FIXTURE_INDEX)
    plans = [langplan.resolve_unsure(langplan.plan(s.signals, idx, lang=(), force_lang=(), installed=set(installed),
                                                   fetch="auto"), idx) for s in (first, again)]
    assert [p.pack for p in plans[1].pages] == [p.pack for p in plans[0].pages] == ["th", "hu", "en"]
    ocr = pipeline.read_pages(again.pages, plans[1].pages, {"th": th, "hu": hu}, en, KEY, signals=again, workers=2)
    reused = {(o.index, r.use_id) for o in ocr for r in o.reads if r.meta.get("pass1")}
    assert reused == {(0, "checker"), (1, "rapidocr"), (2, "reader"), (2, "checker")}
    assert fakes["fk_tess"].calls("en") == [0, 1, 2]           # still only the first call's eng reads


def _bumped(pack: store.InstalledPack, **changes: Any) -> store.InstalledPack:
    return dataclasses.replace(pack, manifest=dataclasses.replace(pack.manifest, **changes))


def _key_args(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "packs"
    en, th = base_pack(root), thai_pack(root)
    return {"pdf": image_pdf(tmp_path / "w.pdf", 2), "classes": [page_class(0), page_class(1)], "base": en,
            "installed": {"en": en, "th": th}, "platform_key": KEY, "osd": {0: None, 1: None}}


def _with_base(a: dict[str, Any], base: store.InstalledPack) -> None:
    a.update(base=base, installed={**a["installed"], "en": base})


def _other_platform(a: dict[str, Any], mp: pytest.MonkeyPatch, tmp: Path, fk: dict[str, FakeOcr]) -> None:
    for f in fk.values():
        f.platforms = frozenset({KEY, "test-os-2"})
    a.update(platform_key="test-os-2")


def _reader_timeout(a: dict[str, Any], mp: pytest.MonkeyPatch, tmp: Path, fk: dict[str, FakeOcr]) -> None:
    base = make_pack(tmp / "packs", "en", {"default": floor_set(
        use("reader", "fk_tess", models=("eng",), lang="en", timeout_s=100.0),
        use("checker", "fk_chk", "checker", models=("chk",), rec="chk"))},
        models={"eng": "fk_tess", "chk": "fk_chk"}, base=True, fills=(StepUse("numbers", 1, {}),))
    _with_base(a, base)


def _library(a: dict[str, Any], mp: pytest.MonkeyPatch, tmp: Path, fk: dict[str, FakeOcr]) -> None:
    real = pipeline._library_versions
    mp.setattr(pipeline, "_library_versions", lambda: {**real(), "pypdfium2": "0.0.1"})


def _tesseract_version(a: dict[str, Any], mp: pytest.MonkeyPatch, tmp: Path, fk: dict[str, FakeOcr]) -> None:
    from pdfintake.ocr.engines import tesseract

    mp.setattr(tesseract.ENGINE, "version", lambda: "made-up 9.9")


KeyChange = Callable[[dict[str, Any], pytest.MonkeyPatch, Path, dict[str, FakeOcr]], None]
KEY_PARTS: dict[str, KeyChange] = {
    "the work PDF": lambda a, mp, tmp, fk: a.update(pdf=image_pdf(tmp / "other.pdf", 2, seed=5)),
    "a page class": lambda a, mp, tmp, fk: a.update(
        classes=[dataclasses.replace(a["classes"][0], image_coverage=0.5), a["classes"][1]]),
    "a page class that needs no OCR": lambda a, mp, tmp, fk: a.update(
        classes=[*a["classes"], page_class(2, needs_ocr=False)]),
    "the OSD orientation gave": lambda a, mp, tmp, fk: a.update(osd={0: THAI_OSD, 1: None}),
    "the platform key": _other_platform,
    "the OS release": lambda a, mp, tmp, fk: mp.setattr(pipeline, "_os_release", lambda: "Example OS 99.0"),
    "the base pack's version": lambda a, mp, tmp, fk: _with_base(a, _bumped(a["base"], version="1.0.1")),
    "the base pack's manifest": lambda a, mp, tmp, fk: _with_base(a, _bumped(a["base"], raw_sha256="1" * 64)),
    "a pack with a probe removed": lambda a, mp, tmp, fk: a.update(installed={"en": a["base"]}),
    "a probe pack's version": lambda a, mp, tmp, fk: a.update(
        installed={**a["installed"], "th": _bumped(a["installed"]["th"], version="1.0.1")}),
    "a probe's threshold": lambda a, mp, tmp, fk: a.update(
        installed={**a["installed"], "th": thai_pack(tmp / "packs", probe_min_chars=30)}),
    # The probe's script and line bar come from the pack's manifest, whose hash the key also holds;
    # each is still a part of its own, so a change in how core derives them can't reuse a stale result.
    "a probe's script": lambda a, mp, tmp, fk: a.update(
        installed={**a["installed"], "th": thai_pack(tmp / "packs", script="Greek")}),
    "a probe's minimum confidence": lambda a, mp, tmp, fk: a.update(
        installed={**a["installed"], "th": thai_pack(tmp / "packs", probe_min_conf=0.7)}),
    "a probe's minimum line length": lambda a, mp, tmp, fk: a.update(
        installed={**a["installed"], "th": thai_pack(tmp / "packs", probe_min_line_chars=3)}),
    # Core's share bar, not the manifest's: an entry written before the bar replays the old counts.
    "the probe line share": lambda a, mp, tmp, fk: mp.setattr(detect, "PROBE_LINE_MIN_SHARE", 0.4),
    "the reader's parameters": _reader_timeout,
    "the reader engine's version": lambda a, mp, tmp, fk: mp.setattr(fk["fk_tess"], "version", lambda: "2.0"),
    "the checker engine's version": lambda a, mp, tmp, fk: mp.setattr(fk["fk_chk"], "version", lambda: "2.0"),
    "the probe engine's version": lambda a, mp, tmp, fk: mp.setattr(fk["fk_probe"], "version", lambda: "2.0"),
    "the reader's availability": lambda a, mp, tmp, fk: setattr(fk["fk_tess"], "avail", ("missing_model", "made up")),
    "Tesseract's version (OSD)": _tesseract_version,
    "the calibration": lambda a, mp, tmp, fk: mp.setattr(langplan, "load_calibration",
                                                         lambda: Calibration(0.5, 2.0, 1, 4, "kit 1.0 test")),
    "the core version": lambda a, mp, tmp, fk: mp.setattr(pipeline, "__version__", "0.1.1"),
    "a library's version": _library,
}


@pytest.mark.parametrize("part", sorted(KEY_PARTS))
def test_every_part_of_the_key_makes_a_fresh_detection(part: str, tmp_path: Path, fakes: dict[str, FakeOcr],
                                                       disk: Disk, renders: list[int], monkeypatch: pytest.MonkeyPatch):
    a = _key_args(tmp_path)
    first = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    same = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert (first.cached, same.cached, len(renders)) == (False, True, 2)
    KEY_PARTS[part](a, monkeypatch, tmp_path, fakes)
    before = reads_of(fakes)
    changed = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert not changed.cached and len(renders) == 4, part      # rendered and detected again
    assert reads_of(fakes)["fk_tess"] > before["fk_tess"] or fakes["fk_tess"].avail[0] != "available"
    again = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert again.cached and len(renders) == 4                  # the new result is kept under its own key
    assert len(disk.entries()) == 1                            # and the input's older entry is gone


def test_a_pack_without_a_probe_takes_no_part_in_pass_1_so_installing_one_keeps_the_cache(
        tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk, renders: list[int]):
    # A run that fetched a Latin pack (no probe) and then apply --lang: pass 1 is unchanged, so reused.
    a = _key_args(tmp_path)
    first = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    a["installed"] = {**a["installed"], "hu": hu_pack(tmp_path / "packs")}
    again = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert again.cached and renders == [0, 1] and again.signals == first.signals


# A failure of the moment (DESIGN §9.1, decided 2026-11-06): each makes the first pass unstorable, and
# the pass after the engine recovers detects again rather than reusing the failure.

class _Flaky:
    """Fails while ``on``; the test turns it off once the engine has "recovered"."""

    def __init__(self) -> None:
        self.on = True


def _eng_raises(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    real = fk["fk_tess"].script
    fk["fk_tess"].script = lambda lang, i: (EngineFailure("engine_failed", "made-up failure under load")
                                            if flaky.on and i == 0 else real(lang, i))


def _eng_times_out(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    real = fk["fk_tess"].script
    fk["fk_tess"].script = lambda lang, i: "timeout" if flaky.on and i == 1 else real(lang, i)


def _checker_fails(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    real = fk["fk_chk"].script
    fk["fk_chk"].script = lambda lang, i: "failed" if flaky.on and i == 0 else real(lang, i)


def _probe_raises(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    real = fk["fk_probe"].script
    fk["fk_probe"].script = lambda lang, i: (EngineFailure("engine_failed", "made-up probe failure")
                                             if flaky.on else real(lang, i))


def _probe_cant_open(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    real = fk["fk_probe"].open

    def opener(*args: Any) -> Any:
        if flaky.on:
            raise EngineFailure("fake_open", "made-up: the probe's model can't be loaded just now")
        return real(*args)

    mp.setattr(fk["fk_probe"], "open", opener)


class _FlakyOsd:
    def __init__(self, flaky: _Flaky) -> None:
        self.flaky = flaky

    def detect(self, image: Any, dpi: float) -> OsdResult | None:
        if self.flaky.on:
            raise EngineFailure("osd_failed", "made-up OSD failure under load")
        return None

    def close(self) -> None:
        pass


def _recovered() -> _Flaky:
    f = _Flaky()
    f.on = False
    return f


def _with_osd_model(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(pipeline, "_osd_key", lambda base: "osd")     # as if the base pack had osd.traineddata


def _osd_cant_open(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    _with_osd_model(mp)

    def opener(base: Any) -> Any:
        if flaky.on:
            raise EngineFailure("osd_unavailable", "made-up: OSD can't open just now")
        return _FlakyOsd(_recovered())

    mp.setattr(pipeline, "_open_osd", opener)


def _osd_detect_raises(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    _with_osd_model(mp)
    mp.setattr(pipeline, "_open_osd", lambda base: _FlakyOsd(flaky))


def _mixed_check_osd_raises(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    # The calibrated trigger gives OSD crops; mixed_script_check swallows a crop's failure, the watch sees it.
    fk["fk_tess"].script = lambda lang, i: LOW
    fk["fk_probe"].script = lambda lang, i: []
    mp.setattr(langplan, "load_calibration", lambda: Calibration(0.5, 2.0, 1, 4, "kit 1.0 test"))
    _osd_detect_raises(fk, mp, flaky)


def _mixed_check_unreadable(fk: dict[str, FakeOcr], mp: pytest.MonkeyPatch, flaky: _Flaky) -> None:
    mp.setattr(langplan, "load_calibration", lambda: Calibration(0.5, 2.0, 1, 4, "kit 1.0 test"))
    _with_osd_model(mp)
    mp.setattr(pipeline, "_open_osd", lambda base: _FlakyOsd(_recovered()))
    real = detect.mixed_script_check

    def check(page: Any, eng: Any, osd: Any, calib: Any) -> MixedScript:
        if calib is not None and flaky.on:
            raise EngineFailure("page_unreadable", "made-up: the render can't be read just now")
        return real(page, eng, osd, calib)

    mp.setattr(detect, "mixed_script_check", check)


TRANSIENT: dict[str, Callable[[dict[str, FakeOcr], pytest.MonkeyPatch, _Flaky], None]] = {
    "an eng read that raises": _eng_raises,
    "an eng read that times out": _eng_times_out,
    "a checker read that fails": _checker_fails,
    "a probe read that raises": _probe_raises,
    "a probe that can't open": _probe_cant_open,
    "OSD that can't open": _osd_cant_open,
    "an OSD call that raises": _osd_detect_raises,
    "an OSD crop in the mixed-script check that raises": _mixed_check_osd_raises,
    "a mixed-script check that falls back": _mixed_check_unreadable,
}


@pytest.mark.parametrize("case", sorted(TRANSIENT))
def test_a_pass_that_met_a_failure_of_the_moment_is_never_stored(case: str, tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                 disk: Disk, renders: list[int],
                                                                 monkeypatch: pytest.MonkeyPatch):
    flaky = _Flaky()
    TRANSIENT[case](fakes, monkeypatch, flaky)
    a = _key_args(tmp_path)
    if "OSD" in case:
        a["osd"] = {}                                          # OSD runs on the renders in pass 1
    first = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert not first.cached and len(first.signals) == 2       # the run itself goes on with what it got
    assert disk.entries() == [], case                          # but nothing of it is kept
    flaky.on = False                                           # the engine works again; nothing in the key changed
    again = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert not again.cached and len(renders) == 4, case        # detected again, never the stored failure
    assert len(disk.entries()) == 1
    third = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert third.cached and len(renders) == 4 and third.signals == again.signals


def test_a_failure_the_key_holds_is_stored(tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk, renders: list[int]):
    # A reader whose models are missing fails on every run alike; its state is part of the key.
    fakes["fk_tess"].avail = ("missing_model", "made up: the model file isn't installed")
    a = _key_args(tmp_path)
    first = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert first.signals[0].eng_read.status == "failed" and first.signals[0].eng_read.meta["state"] == "missing_model"
    again = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert again.cached and renders == [0, 1]


# --------------------------------------------------------------------------- the orientation cache (DESIGN §9.1)

def _votes() -> tuple[dict[int, Any], dict[int, OsdResult | None]]:
    from pdfintake.pdf.orient import OrientVote

    votes = {0: OrientVote(0, 90, 3.5, 4, {0: 1.0, 90: 3.5, 180: 0.0, 270: 0.5}),
             2: OrientVote(2, 0, 0.0, 0, {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0})}
    return votes, {0: OsdResult("Latin", 4.0, 90, 6.5), 2: None}


def _orient_key_args(tmp_path: Path) -> dict[str, Any]:
    return {"pdf": image_pdf(tmp_path / "w.pdf", 3), "pages": [0, 2], "base": base_pack(tmp_path / "packs"),
            "platform_key": KEY, "settings": {"vote_dpi": 150.0, "osd_dpi": 300.0}}


def _rapid_version(a: dict[str, Any], mp: pytest.MonkeyPatch, tmp: Path) -> None:
    from pdfintake.ocr.engines import rapid

    mp.setattr(rapid.ENGINE, "version", lambda: "made-up 9.9")


def _voter_pin(a: dict[str, Any], mp: pytest.MonkeyPatch, tmp: Path) -> None:
    from pdfintake.ocr.engines import rapid

    base = a["base"]
    det = M.Asset(rapid.VOTER_DET_KEY, "det.onnx", "rapidocr", "onnx_det", 1, "d" * 64, "https://example.com/m",
                  "Apache-2.0", None)
    a["base"] = _bumped(base, models=(*base.manifest.models, det))


ORIENT_KEY_PARTS: dict[str, Callable[[dict[str, Any], pytest.MonkeyPatch, Path], None]] = {
    "the work PDF": lambda a, mp, tmp: a.update(pdf=image_pdf(tmp / "other.pdf", 3, seed=5)),
    "the pages": lambda a, mp, tmp: a.update(pages=[0, 1, 2]),
    "a render setting": lambda a, mp, tmp: a.update(settings={"vote_dpi": 150.0, "osd_dpi": 400.0}),
    "the platform key": lambda a, mp, tmp: a.update(platform_key="test-os-2"),
    "the OS release": lambda a, mp, tmp: mp.setattr(pipeline, "_os_release", lambda: "Example OS 99.0"),
    "the base pack's manifest": lambda a, mp, tmp: a.update(base=_bumped(a["base"], raw_sha256="1" * 64)),
    "a voter model's pin": _voter_pin,
    "RapidOCR's version": _rapid_version,
    "Tesseract's version (OSD)": lambda a, mp, tmp: _tesseract_version(a, mp, tmp, {}),
    "a library's version": lambda a, mp, tmp: _library(a, mp, tmp, {}),
    "the core version": lambda a, mp, tmp: mp.setattr(pipeline, "__version__", "0.1.1"),
}


@pytest.mark.parametrize("part", sorted(ORIENT_KEY_PARTS))
def test_every_part_of_the_orientation_key_changes_it(part: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    a = _orient_key_args(tmp_path)
    key = pipeline.orientation_key(**a)
    assert key is not None and key == pipeline.orientation_key(**a)
    ORIENT_KEY_PARTS[part](a, monkeypatch, tmp_path)
    assert pipeline.orientation_key(**a) not in (None, key), part


def test_an_orientation_entry_reads_back_exactly_and_nothing_else_reads(tmp_path: Path):
    votes, osd = _votes()
    key = "e" * 64
    data = pipeline.orientation_entry(votes, osd, key)
    assert data is not None and set(data) == {"v", "kind", "key", "sha256", "body"}
    raw = json.dumps(data).encode("utf-8")
    back = pipeline.read_orientation(raw, key)
    assert back is not None
    assert back[0] == votes and {i: dict(v.scores) for i, v in back[0].items()} == {i: dict(v.scores) for i, v in votes.items()}
    assert back[1] == osd
    assert pipeline.read_orientation(raw, "f" * 64) is None                       # another key
    assert pipeline.read_orientation(raw[:-5], key) is None                       # cut short
    assert pipeline.read_orientation(b" " * (pipeline.ORIENT_MAX_BYTES + 1), key) is None
    for change in (lambda d: d.update(kind="detection"), lambda d: d.update(v=pipeline.CACHE_VERSION + 1),
                   lambda d: d["body"]["pages"][0]["vote"].update(turn_cw=45)):
        bad = json.loads(raw)
        change(bad)
        bad["sha256"] = hashlib.sha256(pipeline._canonical(bad["body"])).hexdigest()
        assert pipeline.read_orientation(json.dumps(bad).encode("utf-8"), key) is None
    tampered = json.loads(raw)
    tampered["body"]["pages"][0]["vote"]["margin"] = 9.0                         # the checksum no longer matches
    assert pipeline.read_orientation(json.dumps(tampered).encode("utf-8"), key) is None


def test_orientation_entry_refuses_what_it_cant_store_faithfully():
    votes, osd = _votes()
    assert pipeline.orientation_entry(votes, osd, "not a key") is None
    assert pipeline.orientation_entry(votes, {0: osd[0]}, "e" * 64) is None       # a page without its OSD result
    assert pipeline.orientation_entry({}, {}, "e" * 64) is None
    nan = {**votes, 0: dataclasses.replace(votes[0], margin=float("nan"))}
    assert pipeline.orientation_entry(nan, osd, "e" * 64) is None


def test_the_watch_sees_what_a_session_swallowed():
    watch = pipeline.FailureWatch()

    class Session:
        def __init__(self, status: str) -> None:
            self.status = status

        def read(self, page: Any, *, timeout_s: float) -> Any:
            if self.status == "raise":
                raise EngineFailure("engine_failed", "made up")
            return EngineRead("v", "fake", "fake:en", "reader", self.status, [], 0.0, None, {})  # type: ignore[arg-type]

    for status in ("ok", "empty"):
        pipeline.watch_session(Session(status), watch).read(None, timeout_s=1.0)
    assert not watch.failed
    pipeline.watch_session(Session("timeout"), watch).read(None, timeout_s=1.0)
    assert watch.failed
    other = pipeline.FailureWatch()
    with pytest.raises(EngineFailure):
        pipeline.watch_session(Session("raise"), other).read(None, timeout_s=1.0)
    assert other.failed and pipeline.watch_session(None, other) is None


def _rewrite(path: Path, change: Callable[[dict[str, Any]], None], *, rehash: bool) -> None:
    data = json.loads(path.read_text("utf-8"))
    change(data)
    if rehash:
        data["sha256"] = hashlib.sha256(pipeline._canonical(data["body"])).hexdigest()
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")


def _first_text(data: dict[str, Any]) -> None:
    line = data["body"]["signals"][0]["eng_read"]["lines"][0]
    line["text"] = line["text"] + "x"                     # any change at all, however small


def _render(disk: Disk, index: int) -> Path:
    return disk.entry_file().parent / f"ocr-{index:05d}.png"


def _linked_to_a_copy(path: Path) -> None:
    copy = path.with_name(path.name + ".copy")
    shutil.copyfile(path, copy)
    path.unlink()
    path.symlink_to(copy)


def _flip(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


DAMAGE: dict[str, Callable[[Disk], None]] = {
    "the cache file cut short": lambda d: d.entry_file().write_bytes(d.entry_file().read_bytes()[:200]),
    "a text changed, the checksum not": lambda d: _rewrite(d.entry_file(), _first_text, rehash=False),
    "the checksum redone over a body with a field it never writes": lambda d: _rewrite(
        d.entry_file(), lambda x: x["body"].update(extra=[]), rehash=True),
    "the checksum redone over a page out of place": lambda d: _rewrite(
        d.entry_file(), lambda x: x["body"]["pages"].reverse(), rehash=True),
    "the checksum redone over a render's name": lambda d: _rewrite(
        d.entry_file(), lambda x: x["body"]["pages"][0].update(file="../ocr-00000.png"), rehash=True),
    "another key": lambda d: _rewrite(d.entry_file(), lambda x: x.update(key="0" * 64), rehash=False),
    "another format version": lambda d: _rewrite(d.entry_file(), lambda x: x.update(v=pipeline.CACHE_VERSION + 1),
                                                 rehash=False),
    "the format version as true": lambda d: _rewrite(d.entry_file(), lambda x: x.update(v=True), rehash=False),
    # Equal to the current version but not an int (True stopped being equal to it when the version left 1).
    "the format version as a float": lambda d: _rewrite(
        d.entry_file(), lambda x: x.update(v=float(pipeline.CACHE_VERSION)), rehash=False),
    "not JSON": lambda d: d.entry_file().write_bytes(b"\x00 not a cache file"),
    "a key written twice": lambda d: d.entry_file().write_text(
        d.entry_file().read_text("utf-8").replace("{", '{"v": 1, ', 1), "utf-8"),
    "a NaN": lambda d: d.entry_file().write_text(
        d.entry_file().read_text("utf-8").replace('"seconds": 0.001', '"seconds": NaN', 1), "utf-8"),
    "a render changed": lambda d: _flip(_render(d, 0)),
    "a render missing": lambda d: _render(d, 1).unlink(),
    "a render replaced by a link to the same bytes": lambda d: _linked_to_a_copy(_render(d, 0)),
    "the cache file replaced by a link to the same bytes": lambda d: _linked_to_a_copy(d.entry_file()),
}


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_a_damaged_entry_is_a_fresh_detection_never_a_reuse(damage: str, tmp_path: Path, fakes: dict[str, FakeOcr],
                                                            disk: Disk, renders: list[int]):
    if "link" in damage and sys.platform == "win32":
        pytest.skip("symbolic links need privileges on Windows")
    a = _key_args(tmp_path)
    first = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    DAMAGE[damage](disk)
    before = reads_of(fakes)
    fresh = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert not fresh.cached and len(renders) == 4, damage
    assert reads_of(fakes)["fk_tess"] == before["fk_tess"] + 2  # detected again
    assert fresh.signals == first.signals and without_paths(fresh) == without_paths(first)
    third = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert third.cached and len(renders) == 4                  # the damaged entry was replaced
    assert len(disk.entries()) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs are POSIX only")
def test_a_cache_file_that_is_a_fifo_is_never_waited_on(tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk,
                                                        renders: list[int]):
    a = _key_args(tmp_path)
    pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    f = disk.entry_file()
    f.unlink()
    os.mkfifo(f)                                               # no writer: a blocking open would hang here
    fresh = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert not fresh.cached and len(renders) == 4


def test_a_cache_file_over_the_size_cap_is_never_read(tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk,
                                                      renders: list[int], monkeypatch: pytest.MonkeyPatch):
    a = _key_args(tmp_path)
    pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    monkeypatch.setattr(pipeline, "MAX_CACHE_BYTES", disk.entry_file().stat().st_size - 1)
    fresh = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    assert not fresh.cached and len(renders) == 4
    assert pipeline.read_cache(b" " * (pipeline.MAX_CACHE_BYTES + 1), "0" * 64) is None


def test_a_cache_that_fails_or_answers_wrongly_never_fails_the_gather(tmp_path: Path, fakes: dict[str, FakeOcr],
                                                                      renders: list[int]):
    class Broken:
        def load(self, key: str, work: Path) -> Any:
            raise RuntimeError("made-up disk error")

        def keep(self, key: str, signals: Any) -> None:
            raise OSError("made-up full disk")

    class Elsewhere:
        """Hands back a result whose pages aren't this call's (another document's, or in another folder)."""

        def __init__(self, sig: pipeline.DocumentSignals) -> None:
            self.sig = sig

        def load(self, key: str, work: Path) -> Any:
            return self.sig

        def keep(self, key: str, signals: Any) -> None:
            pass

    a = _key_args(tmp_path)
    sig = pipeline.gather(**a, work=tmp_path / "w1", workers=1, cache=Broken())
    assert not sig.cached and renders == [0, 1]
    for wrong in (pipeline.DocumentSignals([], [], [], {}), sig):   # no pages; this call's pages in another folder
        again = pipeline.gather(**a, work=tmp_path / "w2", workers=1, cache=Elsewhere(wrong))
        assert not again.cached
    assert renders == [0, 1, 0, 1, 0, 1]


def _texts(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text":
                yield v
            else:
                yield from _texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _texts(v)


def test_the_cache_holds_pass_1_and_nothing_pass_2_reads(tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk):
    pass_two = "Rivers carry water from the hills down to the sea in every season"
    thai_two = "ดวงจันทร์โคจรรอบโลก"                         # made up: the Moon goes round the Earth
    tess = fakes["fk_tess"].script
    assert tess is not None
    fakes["fk_tess"].script = lambda lang, i: [pass_two] if lang == "hu" else tess(lang, i)
    fakes["fk_thai"].script = lambda lang, i: [thai_two]
    root = tmp_path / "packs"
    en, th, hu = base_pack(root), thai_pack(root), hu_pack(root)
    installed = {"en": en, "th": th, "hu": hu}
    classes = [page_class(i) for i in range(3)]
    sig = pipeline.gather(image_pdf(tmp_path / "w.pdf", 3), classes, en, installed, KEY, disk.work(), workers=2,
                          osd={0: THAI_OSD, 1: None, 2: None}, cache=disk.cache())
    idx = I.load(FIXTURE_INDEX)
    plan = langplan.resolve_unsure(langplan.plan(sig.signals, idx, lang=(), force_lang=(), installed=set(installed),
                                                 fetch="auto"), idx)
    ocr = pipeline.read_pages(sig.pages, plan.pages, {"th": th, "hu": hu}, en, KEY, signals=sig, workers=2)
    pipeline.layer_pages(ocr, sig.pages, classes)
    pass_2_text = "\n".join(r.text() for o in ocr for r in o.reads)
    assert pass_two in pass_2_text and thai_two in pass_2_text  # pass 2 did read them

    (entry,) = disk.entries()
    assert sorted(p.name for p in entry.iterdir()) == [pipeline.CACHE_FILE, "ocr-00000.png", "ocr-00001.png",
                                                       "ocr-00002.png"]
    everything = b"".join(p.read_bytes() for p in disk.cache().folder.rglob("*") if p.is_file())
    for text in (pass_two, thai_two, "Rivers", "ดวงจันทร์"):
        assert text.encode("utf-8") not in everything
    data = json.loads(disk.entry_file().read_text("utf-8"))
    assert set(data) == {"v", "key", "sha256", "body"}
    assert set(data["body"]) == {"pages", "signals", "osd", "warnings", "notes"}
    assert all(set(s) == {f.name for f in dataclasses.fields(detect.PageSignals)} for s in data["body"]["signals"])
    # Every text in it is a line or a word of the two reads detection reads: the base pack's eng and checker reads.
    allowed = {t for s in sig.signals for r in (s.eng_read, s.checker_read) if r is not None
               for ln in r.lines for t in (ln.text, *(w.text for w in ln.words))}
    found = list(_texts(data["body"]))
    assert found and set(found) <= allowed
    assert all(set(w) == {"start", "code", "prob"} for s in data["body"]["signals"] for w in s["windows"])


def test_cache_entry_refuses_what_it_cant_store_faithfully(tmp_path: Path, fakes: dict[str, FakeOcr]):
    a = _key_args(tmp_path)
    sig = pipeline.gather(**a, work=tmp_path / "work", workers=1)
    key = "c" * 64
    assert pipeline.cache_entry(sig, key) is not None

    def with_read(**changes: Any) -> pipeline.DocumentSignals:
        s0 = sig.signals[0]
        return dataclasses.replace(sig, signals=[dataclasses.replace(s0, eng_read=dataclasses.replace(s0.eng_read,
                                                                                                    **changes)),
                                                 *sig.signals[1:]])

    assert pipeline.cache_entry(with_read(meta={"blocks": (1, 2)}), key) is None   # a tuple JSON can't carry
    assert pipeline.cache_entry(with_read(seconds=float("nan")), key) is None
    assert pipeline.cache_entry(with_read(error="made up \udcff"), key) is None      # not writable as UTF-8
    # A failure of the moment is never stored; a use that can't run here (its state is in the key) is.
    assert pipeline.cache_entry(with_read(status="timeout", lines=[]), key) is None
    assert pipeline.cache_entry(with_read(status="failed", lines=[], meta={}), key) is None
    assert pipeline.cache_entry(with_read(status="timeout", lines=[], meta={"state": "broken"}), key) is None
    assert pipeline.cache_entry(with_read(status="failed", lines=[], meta={"state": "missing_model"}), key) is not None
    assert pipeline.cache_entry(sig, "not a key") is None
    other = tmp_path / "elsewhere"
    other.mkdir()
    shutil.copyfile(sig.pages[1].path, other / sig.pages[1].path.name)
    split = dataclasses.replace(sig, pages=[sig.pages[0], dataclasses.replace(sig.pages[1], path=other / sig.pages[1].path.name)])
    assert pipeline.cache_entry(split, key) is None             # renders in two folders
    shutil.copyfile(sig.pages[0].path, tmp_path / "work" / "page.png")
    renamed = dataclasses.replace(sig, pages=[dataclasses.replace(sig.pages[0], path=tmp_path / "work" / "page.png"),
                                              sig.pages[1]])
    assert pipeline.cache_entry(renamed, key) is None           # a render under a name gather never gives


def test_read_cache_hands_back_only_renders_it_checked(tmp_path: Path, fakes: dict[str, FakeOcr], disk: Disk):
    a = _key_args(tmp_path)
    sig = pipeline.gather(**a, work=disk.work(), workers=1, cache=disk.cache())
    raw = disk.entry_file().read_bytes()
    key = json.loads(raw)["key"]
    assert pipeline.read_cache(raw, "0" * 64) is None           # another key
    found = pipeline.read_cache(raw, key)
    assert found is not None and found.renders == ("ocr-00000.png", "ocr-00001.png")
    entry = disk.entry_file().parent
    back = found.signals(entry)
    assert back is not None and back.signals == sig.signals and [p.path.parent for p in back.pages] == [entry] * 2
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in found.renders:
        shutil.copyfile(entry / name, copy / name)
    assert found.signals(copy) is not None
    if sys.platform != "win32":
        _linked_to_a_copy(copy / "ocr-00001.png")
        assert found.signals(copy) is None                      # a link, even to the same bytes
    (copy / "ocr-00001.png").unlink()
    assert found.signals(copy) is None                          # a render missing
    shutil.copyfile(entry / "ocr-00000.png", copy / "ocr-00001.png")
    assert found.signals(copy) is None                          # another page's render


# --------------------------------------------------------------------------- the real engines

def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def real_packs(tmp: Path, *ids: str) -> dict[str, store.InstalledPack]:
    """The launch packs' real manifests with their pinned models (skips, naming them, when they aren't fetched)."""
    from conftest import pinned_asset

    base_m = M.parse((REPO / "packs" / "en" / "manifest.json").read_bytes())
    out: dict[str, store.InstalledPack] = {}
    for pid in ids:
        m = base_m if pid == "en" else M.parse((REPO / "packs" / pid / "manifest.json").read_bytes(), base=base_m)
        root = tmp / "real" / f"{pid}-{m.version}"
        for a in m.models:
            _link(pinned_asset(pid, a.key), root / "models" / a.file)
        out[pid] = store.InstalledPack(m, root)
    return out


def _font(pack: str, key: str, size: int) -> Any:
    from PIL import ImageFont

    from conftest import pinned_asset

    return ImageFont.truetype(str(pinned_asset(pack, key)), size, layout_engine=ImageFont.Layout.BASIC)


def text_page(lines: Sequence[str], font: Any, *, width: int = 1800, pitch: int = 110) -> Any:
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, 120 + pitch * len(lines)), 255)
    draw = ImageDraw.Draw(img)
    for k, text in enumerate(lines):
        draw.text((60, 50 + k * pitch), text, font=font, fill=0)
    return img


REAL_EN = ["The example invoice lists seven sample items and a total",
           "of 4521 units for the test customer. The company will send",
           "the order to the address on the form before Friday, as agreed",
           "on the phone last week. Every line here is made up for a test."]
REAL_HU = ["A kertészek tapasztalják tavasszal, hogy a régi fák és a",
           "virágok minden reggel illatoznak, illetve az erdő felett",
           "csak akkor ér eső a fűre, ha már egy másik vagy még több",
           "felhő érkezett. Ezt kell tudni, és nem többet, köszönöm."]
REAL_TH = ["ข้อความตัวอย่าง สำหรับการทดสอบ", "บริษัทตัวอย่าง จำกัด",
           "ภาษาไทยเขียนต่อกัน ไม่เว้นวรรค ระหว่างคำ", "หน้าที่หนึ่งของเอกสารทดสอบ",
           "ระบบอ่านข้อความ ทำงานได้ดี"]


def real_pdf(path: Path) -> Path:
    """Three image-only pages at 300 dpi: Thai, Hungarian, English (made-up text)."""
    imgs = [text_page(REAL_TH, _font("th", "noto-sans-thai", 56), pitch=150),
            text_page(REAL_HU, _font("en", "noto-sans", 52)),
            text_page(REAL_EN, _font("en", "noto-sans", 52))]
    specs = [build.PageSpec(width=img.size[0] * 72 / 300, height=img.size[1] * 72 / 300,
                            images=(build.ImageSpec(img),)) for img in imgs]
    return build.write_pdf(path, specs)


def run_real(tmp_path: Path, platform_key: str) -> tuple[list[Any], list[Any], Any]:
    pytest.importorskip("tesserocr")
    packs = real_packs(tmp_path, "en", "th", "hu")
    pdf = real_pdf(tmp_path / "real.pdf")
    en = packs["en"]
    classes = [page_class(i) for i in range(3)]
    sig = pipeline.gather(pdf, classes, en, packs, platform_key, tmp_path / "work", workers=2, osd={})
    idx = I.load(FIXTURE_INDEX)
    plan = langplan.resolve_unsure(langplan.plan(sig.signals, idx, lang=(), force_lang=(),
                                                 installed=set(packs), fetch="auto"), idx)
    ocr = pipeline.read_pages(sig.pages, plan.pages, packs, en, platform_key, signals=sig, workers=2)
    return ocr, pipeline.layer_pages(ocr, sig.pages, classes), sig


def _layer_text(lp: Any) -> str:
    return "\n".join(ln.text for ln in lp.lines)


def _reads_report(ocr: Sequence[Any], layer: Sequence[Any]) -> str:
    """Why each page's layer came out as it did, for a failure message (counts and errors, no text):
    the pack, pick, warnings and layer line count, then every read's use, status, line count, state and
    error; and the shared tessdata folder as Tesseract is handed it, which on Windows is its 8.3 form
    when the volume gives the folder one and the non-ASCII path itself when it doesn't."""
    from pdfintake.ocr.engines import tesseract as T

    out: list[str] = []
    for o, lp in zip(ocr, layer):
        out.append(f"page {o.index} ({o.pack}): pick {o.pick.rule} chose {o.pick.chosen!r}, fallback "
                   f"{o.pick.fallback}, warnings {o.warnings}, {len(lp.lines)} layer lines")
        for r in o.reads:
            out.append(f"  {r.use_id} [{r.engine} {r.role}] {r.status}, {len(r.lines)} lines, "
                       f"state {r.meta.get('state')!r}, error {r.error!r}")
    shared = store.tessdata_dir()
    out.append(f"shared tessdata {shared} is handed to Tesseract as {T._native_dir(shared)!r}")
    return "\n".join(out)


def test_real_engines_read_thai_hungarian_and_english_pages(tmp_path: Path):
    # The default reader sets (Tesseract and RapidOCR), as every platform without its own set reads them.
    # Thai reads tha+eng, whose models sit in two folders, so Tesseract loads them from the store's shared
    # tessdata folder inside PDFINTAKE_CACHE (non-ASCII in the Windows CI jobs); hun and eng load from
    # their own folders. The report names each read's status and error when a page comes out wrong.
    ocr, layer, sig = run_real(tmp_path, "linux-x86_64")
    assert [s.script.name for s in sig.signals] == ["Thai", "Latin", "Latin"]
    assert sig.osd[0] is not None and sig.osd[0].script == "Thai"
    assert [o.pack for o in ocr] == ["th", "hu", "en"]
    report = _reads_report(ocr, layer)

    th, hu, en = ocr
    th_text = _layer_text(layer[0]).replace(" ", "")
    assert "ตัวอย่าง" in th_text and "ทดสอบ" in th_text, report
    assert "ํา" not in th_text and "ำ" in th_text                 # sara am normalised
    assert th.must_survive["tone-marks"]["layer"] > 0 and th.pick.chosen == "tesseract"
    assert th.warnings == [] or set(th.warnings) <= {"mixed_script_page"}

    hu_text = _layer_text(layer[1])
    assert "tapasztalják" in hu_text and ("erdő" in hu_text or "fűre" in hu_text), report
    assert hu.must_survive["double-acute"]["layer"] >= 1
    assert any(r.meta.get("pass1") for r in hu.reads if r.role == "checker")    # the Latin checker read reused

    en_text = _layer_text(layer[2]).lower()
    assert "invoice" in en_text and "4521" in en_text, report
    assert all(r.meta.get("pass1") for r in en.reads)                           # pass 1 already read it
    for o in ocr:
        assert all(r.status == "ok" for r in o.reads), [(r.use_id, r.status, r.error) for r in o.reads]


@pytest.mark.vision
@pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision exists only on macOS")
def test_real_engines_with_vision_on_this_mac(tmp_path: Path):
    from pdfintake.ocr.engines import vision

    if not vision.ENGINE.availability({"languages": ["th-TH", "en-US"], "revision": 3, "level": "accurate",
                                       "language_correction": True}, {}, "darwin-arm64").ok:
        pytest.skip("Vision on this Mac doesn't read th-TH at revision 3")
    ocr, layer, _sig = run_real(tmp_path, "darwin-arm64")
    th = ocr[0]
    assert th.pack == "th" and [r.use_id for r in th.reads] == ["vision", "tesseract", "rapidocr"]
    assert th.pick.rule == "corroborated_trigrams" and th.pick.chosen in ("vision", "tesseract")
    assert set(th.pick.scores) == {"vision", "tesseract"}
    assert "ตัวอย่าง" in _layer_text(layer[0]).replace(" ", "")


def test_real_engines_pass_1_goes_through_the_cache_unchanged(tmp_path: Path, disk: Disk, renders: list[int]):
    # Real Tesseract and RapidOCR reads (their meta, boxes and words) are stored faithfully, and pass 2 on
    # the cached result writes the same layer as on the fresh one.
    pytest.importorskip("tesserocr")
    packs = real_packs(tmp_path, "en", "th", "hu")
    pdf = real_pdf(tmp_path / "real.pdf")
    en = packs["en"]
    classes = [page_class(i) for i in range(3)]
    idx = I.load(FIXTURE_INDEX)
    sigs, texts = [], []
    for _ in range(2):
        sig = pipeline.gather(pdf, classes, en, packs, "linux-x86_64", disk.work(), workers=2, osd={},
                              cache=disk.cache())
        plan = langplan.resolve_unsure(langplan.plan(sig.signals, idx, lang=(), force_lang=(),
                                                     installed=set(packs), fetch="auto"), idx)
        ocr = pipeline.read_pages(sig.pages, plan.pages, packs, en, "linux-x86_64", signals=sig, workers=2)
        sigs.append(sig)
        texts.append([_layer_text(lp) for lp in pipeline.layer_pages(ocr, sig.pages, classes)])
    first, again = sigs
    assert (first.cached, again.cached) == (False, True) and len(disk.entries()) == 1
    assert renders == [0, 1, 2]                                # rendered once
    assert again.signals == first.signals and again.osd == first.osd
    assert texts[0] == texts[1] and "tapasztalják" in texts[1][1]
