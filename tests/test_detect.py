"""Pass-1 detection (``ocr/detect.py``) on synthetic reads and pages. Every text is made up."""

from __future__ import annotations

import hashlib
import io
import math
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from fixtures import build
from pdfintake.errors import EngineFailure, PackError
from pdfintake.ocr import detect, langplan
from pdfintake.ocr.detect import (FWScore, LangWindow, MixedScript, PageSignals, ProbeResult, function_word_score,
                                  language_windows, mixed_script_check, read_mean_conf, run_probe, script_call)
from pdfintake.ocr.langplan import Calibration
from pdfintake.protocols import LanguageIdentifier, OsdSession
from pdfintake.types import EngineRead, Line, OsdResult, ScriptCall, Word

SPANISH = [
    "El ejemplo de factura incluye siete artículos de muestra y un total de cuatro mil unidades para el cliente de prueba.",
    "La empresa enviará el pedido a la dirección indicada en el formulario antes del viernes, según lo acordado por teléfono.",
]
HUNGARIAN = [
    "A kertészek tapasztalják tavasszal, hogy a régi fák és a rózsák illata minden reggel egyre erősebben árad,",
    "illetve a nyári zápor után szellő támad, amely a meteorológusok szerint az egyik hétvégén sem lesz erős.",
    "A könyvtár egy régi épületben működik, továbbá a polcok között minden olvasó esetén a térkép segít, ha kell.",
]
THAI_LINE = "ข้อความตัวอย่างสำหรับการทดสอบระบบ"     # made up; every character is Thai
THANKS = "ขอบคุณที่ใช้บริการ"


# --------------------------------------------------------------------------- stubs

class StubIdentifier:
    """Calls every text by its first word: "es-word ..." -> [("es", 0.9)]; records what it saw."""

    name: ClassVar[str] = "stub"

    def __init__(self, prob: float = 0.9) -> None:
        self.prob = prob
        self.seen: list[str] = []

    def classify(self, text: str) -> list[tuple[str, float]]:
        self.seen.append(text)
        if not text.strip():
            return []
        code = text.split()[0].split("-")[0]
        return [(code, self.prob), ("xx", 1.0 - self.prob)]


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubIdentifier:
    s = StubIdentifier()
    monkeypatch.setitem(detect._IDENTIFIERS, "stub", s)
    return s


class StubOsd:
    """An OSD session that answers from a function of the crop; records every call."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[tuple[tuple[int, int], float]] = []
        self.lock = threading.Lock()

    def detect(self, image: Image.Image, dpi: float) -> OsdResult | None:
        with self.lock:
            self.calls.append((image.size, dpi))
        return self.answer(image, dpi)


class StubSession:
    def __init__(self, read: EngineRead | None = None, *, fail: bool = False) -> None:
        self._read = read
        self.fail = fail
        self.timeouts: list[float] = []

    def read(self, page, *, timeout_s: float) -> EngineRead:
        self.timeouts.append(timeout_s)
        if self.fail:
            raise EngineFailure("made_up", "a made-up engine failure")
        assert self._read is not None
        return self._read

    def close(self) -> None:
        pass


def calib(**kw: Any) -> Calibration:
    base = dict(block_max_conf=0.5, osd_min_conf=5.0, min_block_px=20, max_blocks=4, measured="test")
    base.update(kw)
    return Calibration(**base)


# --------------------------------------------------------------------------- registry and protocols

def test_identifier_registry_is_read_only_and_has_py3langid():
    assert set(detect.IDENTIFIERS) == {"py3langid"}
    assert detect.DEFAULT_IDENTIFIER == "py3langid"
    with pytest.raises(TypeError):
        detect.IDENTIFIERS["other"] = StubIdentifier()            # type: ignore[index]
    ident = detect.IDENTIFIERS["py3langid"]
    assert isinstance(ident, LanguageIdentifier)
    assert ident.name == "py3langid"


def test_stubs_satisfy_the_protocols():
    assert isinstance(StubIdentifier(), LanguageIdentifier)
    assert isinstance(StubOsd(lambda i, d: None), OsdSession)


def test_unknown_identifier_is_a_pack_error_naming_it():
    with pytest.raises(PackError) as e:
        language_windows(build.engine_read("eng", SPANISH), identifier="nope")
    assert e.value.code == "unknown_identifier" and "nope" in str(e.value)


def test_page_signals_defaults_and_fields():
    read = build.engine_read("eng", ["made up"])
    sig = PageSignals(3, None, ScriptCall(None, 0.0, "none", None), read, None)
    assert sig.probes == {} and sig.windows == [] and sig.mixed == MixedScript(False, None, 0, None)
    full = PageSignals(3, None, ScriptCall(None, 0.0, "none", None), read, None, {"th": ProbeResult("th", 1, 1)},
                       [LangWindow(0, "en", 0.9)], MixedScript(True, None, 0, None))
    assert full.probes["th"].chars == 1 and full.windows[0].code == "en"


# --------------------------------------------------------------------------- windows

def _words_line(pairs: Sequence[tuple[str, float]], y: float = 100.0) -> Line:
    x = 10.0
    words = []
    for text, conf in pairs:
        words.append(Word(text, conf, build.quad(x, y, x + 10 * len(text), y + 30)))
        x += 10 * len(text) + 10
    return Line(" ".join(t for t, _ in pairs), sum(c for _, c in pairs) / len(pairs),
                build.quad(10, y, x, y + 30), tuple(words))


def test_windows_cover_whole_words_each_at_least_the_size(stub: StubIdentifier):
    words = [f"es{i:03d}" for i in range(60)]                     # 60 words of 5 letters: 359 characters joined
    read = build.engine_read("eng", [" ".join(words[i:i + 10]) for i in range(0, 60, 10)])
    ws = language_windows(read, size=120, identifier="stub")
    joined = " ".join(words)
    assert len(ws) == 2                                           # 120 + a tail of the rest
    assert [w.start for w in ws] == [0, joined.index(stub.seen[1].split()[0])]
    assert all(len(t) >= 120 for t in stub.seen)
    assert " ".join(stub.seen) == joined                          # nothing lost, nothing repeated
    assert all(w.code.startswith("es") and w.prob == 0.9 for w in ws)


def test_a_short_text_has_no_window_and_a_tail_joins_the_last(stub: StubIdentifier):
    assert language_windows(build.engine_read("eng", ["es-short text only"]), identifier="stub") == []
    read = build.engine_read("eng", ["es-" + "a" * 118, "tail"])                # 121 characters, then a 4-letter tail
    ws = language_windows(read, identifier="stub")
    assert len(ws) == 1 and stub.seen[-1].endswith(" tail")


def test_only_confident_latin_words_count(stub: StubIdentifier):
    line = _words_line([("es-uno", 0.9), ("lowconf", 0.39), ("ok", 0.4), ("1234", 0.99), (THAI_LINE, 0.99),
                        ("mixedกข", 0.95), ("Ünïcödé", 0.9), ("x" * 120, 0.9)])
    read = build.engine_read("eng", lines=[line])
    language_windows(read, identifier="stub")
    seen = stub.seen[-1].split()
    assert seen == ["es-uno", "ok", "Ünïcödé", "x" * 120]


def test_lines_without_word_boxes_use_the_line_confidence(stub: StubIdentifier):
    good = build.engine_read("eng", ["es-" + "word " * 30], conf=0.5)
    bad = build.engine_read("eng", ["es-" + "word " * 30], conf=0.3)
    assert len(language_windows(good, identifier="stub")) == 1
    assert language_windows(bad, identifier="stub") == []


def test_a_read_that_failed_has_no_windows(stub: StubIdentifier):
    failed = build.engine_read("eng", ["es-" + "word " * 40], status="failed")
    assert language_windows(failed, identifier="stub") == []
    with pytest.raises(ValueError):
        language_windows(build.engine_read("eng", SPANISH), size=0)


def test_the_real_identifier_on_made_up_spanish_and_hungarian():
    ident = detect.IDENTIFIERS["py3langid"]
    es = ident.classify(" ".join(SPANISH))
    hu = ident.classify(" ".join(HUNGARIAN))
    assert es[0][0] == "es" and es[0][1] >= langplan.LANGID_MIN_PROB
    assert hu[0][0] == "hu" and hu[0][1] >= langplan.LANGID_MIN_PROB
    assert all(0.0 <= p <= 1.0 for _, p in es) and math.isclose(sum(p for _, p in es), 1.0, abs_tol=1e-3)
    assert es == sorted(es, key=lambda t: -t[1])
    assert ident.classify("") == [] and ident.classify("   ") == []
    ws_es = language_windows(build.engine_read("eng", SPANISH))
    ws_hu = language_windows(build.engine_read("eng", HUNGARIAN))
    assert ws_es and {w.code for w in ws_es} == {"es"}
    assert ws_hu and {w.code for w in ws_hu} == {"hu"}
    assert all(w.prob >= langplan.LANGID_MIN_PROB for w in ws_es + ws_hu)
    assert detect.page_language(build.engine_read("eng", HUNGARIAN))[0][0] == "hu"


@pytest.mark.parametrize("content", [None, b"", b"\xfd7zXZ\x00\x00truncated"])
def test_a_broken_identifier_model_is_a_setup_error(tmp_path, monkeypatch: pytest.MonkeyPatch, content):
    import py3langid.langid as pl

    from pdfintake.errors import SetupError

    model = tmp_path / "model.plzma"
    if content is not None:
        model.write_bytes(content)
    monkeypatch.setattr(pl, "MODEL_FILE", str(model))
    with pytest.raises(SetupError) as e:
        detect.Py3LangId().classify("una frase de ejemplo")
    assert e.value.code == "identifier_broken" and "py3langid" in str(e.value)


def _installed_model() -> Path:
    import py3langid.langid as pl

    path = Path(pl.MODEL_FILE)
    return path if path.is_absolute() else Path(pl.MODEL_DIR) / path


def test_the_pin_matches_the_installed_py3langid_model():
    from importlib import metadata

    assert metadata.version("py3langid") == detect.PY3LANGID_VERSION == "0.4.0"
    raw = _installed_model().read_bytes()
    assert len(raw) == detect.PY3LANGID_MODEL_SIZE
    assert hashlib.sha256(raw).hexdigest() == detect.PY3LANGID_MODEL_SHA256


def test_a_copy_of_the_pinned_model_loads_from_anywhere(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import py3langid.langid as pl

    copy = tmp_path / "model.npz.xz"
    shutil.copyfile(_installed_model(), copy)
    monkeypatch.setattr(pl, "MODEL_FILE", str(copy))
    ranked = detect.Py3LangId().classify(" ".join(SPANISH))
    assert ranked[0][0] == "es" and ranked[0][1] >= langplan.LANGID_MIN_PROB


@pytest.mark.parametrize("damage", ["flip_byte", "append_byte", "same_size_zeros"])
def test_a_model_that_differs_from_the_pin_is_a_setup_error_before_it_is_parsed(
        tmp_path, monkeypatch: pytest.MonkeyPatch, damage: str):
    import py3langid.langid as pl
    import py3langid.modelio as pm

    from pdfintake.errors import SetupError

    raw = bytearray(_installed_model().read_bytes())
    if damage == "flip_byte":
        raw[len(raw) // 2] ^= 0x01                                 # same size, one bit different
    elif damage == "append_byte":
        raw += b"\x00"
    else:
        raw = bytearray(len(raw))
    model = tmp_path / "model.npz.xz"
    model.write_bytes(bytes(raw))
    monkeypatch.setattr(pl, "MODEL_FILE", str(model))
    parsed: list[Any] = []
    monkeypatch.setattr(pm, "load_model", lambda *a, **k: parsed.append(a) or pytest.fail("parsed an unchecked model"))
    with pytest.raises(SetupError) as e:
        detect.Py3LangId().classify("una frase de ejemplo")
    assert e.value.code == "identifier_broken" and e.value.exit_code == 3
    assert e.value.detail["file"] == str(model)
    assert e.value.detail["expected_sha256"] == detect.PY3LANGID_MODEL_SHA256
    assert e.value.detail["sha256"] == hashlib.sha256(bytes(raw)).hexdigest() != detect.PY3LANGID_MODEL_SHA256
    assert detect.PY3LANGID_MODEL_SHA256 in str(e.value) and "py3langid 0.4.0" in str(e.value)
    assert parsed == []


def test_the_model_is_parsed_from_the_bytes_that_were_checked(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """What is parsed is what was checked: the parser gets the hashed bytes, never the path to re-read."""
    import py3langid.langid as pl
    import py3langid.modelio as pm

    model = tmp_path / "model.npz.xz"
    shutil.copyfileobj(_installed_model().open("rb"), model.open("wb"))
    monkeypatch.setattr(pl, "MODEL_FILE", str(model))
    real_load = pm.load_model
    seen: list[Any] = []

    def recording_load(src: Any) -> Any:
        seen.append(src)
        model.write_bytes(b"swapped on disk after the check")     # a re-read from disk would now fail
        return real_load(src)

    def no_path_load(*a: Any, **k: Any) -> Any:
        pytest.fail("the model was re-read from its path after the check")

    monkeypatch.setattr(pm, "load_model", recording_load)
    monkeypatch.setattr(pl, "_load_model_file", no_path_load)           # what from_model_file calls
    monkeypatch.setattr(pl.LanguageIdentifier, "from_model_file", classmethod(no_path_load))
    ranked = detect.Py3LangId().classify(" ".join(SPANISH))
    assert ranked[0][0] == "es"
    assert len(seen) == 1 and isinstance(seen[0], io.BytesIO)
    assert hashlib.sha256(seen[0].getvalue()).hexdigest() == detect.PY3LANGID_MODEL_SHA256


def test_a_failed_load_is_retried_on_the_next_call(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import py3langid.langid as pl

    from pdfintake.errors import SetupError

    bad = tmp_path / "bad.npz.xz"
    bad.write_bytes(b"not a model")
    good = tmp_path / "good.npz.xz"
    shutil.copyfile(_installed_model(), good)
    ident = detect.Py3LangId()
    monkeypatch.setattr(pl, "MODEL_FILE", str(bad))
    with pytest.raises(SetupError):
        ident.classify("una frase de ejemplo")
    monkeypatch.setattr(pl, "MODEL_FILE", str(good))
    assert ident.classify(" ".join(SPANISH))[0][0] == "es"


def test_the_real_identifier_is_deterministic_across_threads():
    ident = detect.IDENTIFIERS["py3langid"]
    text = " ".join(SPANISH)
    want = ident.classify(text)
    out: list[Any] = []
    threads = [threading.Thread(target=lambda: out.append(ident.classify(text))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == [want] * 4


# --------------------------------------------------------------------------- function words

HU_WORDS = frozenset({"és", "hogy", "az", "egy", "illetve", "után", "amely", "szerint", "lesz", "között", "továbbá"})
RIVALS = frozenset({"und", "der", "die", "das", "ve", "bir", "bu"})


def test_function_word_score_counts_words_rivals_and_english():
    read = build.engine_read("rapidocr", ["Hogy ÉS az, és 2024 THE the and", "und der Die bu 12ab"])
    s = function_word_score(read, HU_WORDS, RIVALS)
    assert s == FWScore(hits=4, rival_hits=4, en_hits=3, words=12)   # "2024" is no token, "12ab" gives "ab"


def test_function_word_score_lowercases_the_lists_and_ignores_failed_reads():
    read = build.engine_read("rapidocr", ["hogy hogy"])
    assert function_word_score(read, frozenset({"HOGY"}), frozenset()).hits == 2
    assert function_word_score(build.engine_read("r", ["hogy"], status="failed"), HU_WORDS, RIVALS) == FWScore(0, 0, 0, 0)


def test_function_word_score_normalises_the_text_and_turkish_capitals():
    import unicodedata

    nfc = build.engine_read("r", HUNGARIAN)
    nfd = build.engine_read("r", [unicodedata.normalize("NFD", ln) for ln in HUNGARIAN])
    assert function_word_score(nfd, HU_WORDS, RIVALS) == function_word_score(nfc, HU_WORDS, RIVALS)
    decomposed_list = frozenset(unicodedata.normalize("NFD", w) for w in HU_WORDS)
    assert function_word_score(nfc, decomposed_list, RIVALS) == function_word_score(nfc, HU_WORDS, RIVALS)
    tr = frozenset({"için", "bir", "bu"})
    upper = function_word_score(build.engine_read("r", ["BU KİTAP BİR KÖY İÇİN"]), frozenset(), tr)
    lower = function_word_score(build.engine_read("r", ["bu kitap bir köy için"]), frozenset(), tr)
    assert upper.rival_hits == lower.rival_hits == 3


def test_function_word_score_on_made_up_pages():
    hu = function_word_score(build.engine_read("r", HUNGARIAN), HU_WORDS, RIVALS)
    assert hu.hits >= 10 and hu.rival_hits == 0 and hu.words > 40
    tr = function_word_score(build.engine_read("r", ["Bu kitap ve bir köy tarihi için az ez az daha çok"]),
                             HU_WORDS | {"ez"}, RIVALS)
    assert (tr.hits, tr.rival_hits) == (3, 3)


# --------------------------------------------------------------------------- script_call

@pytest.mark.parametrize("osd,eng,want", [
    (OsdResult("Thai", 5.0, 0, 9.0), None, ("Thai", "osd")),
    (OsdResult("Thai", 4.99, 0, 9.0), 90.0, ("Latin", "osd")),
    (OsdResult("Thai", 4.99, 0, 9.0), 59.9, (None, "none")),
    (OsdResult("Thai", 12.0, 0, 9.0), 95.0, ("Thai", "osd")),          # a confident non-Latin script comes first
    (OsdResult("Latin", 30.0, 0, 9.0), 60.0, ("Latin", "osd")),
    (OsdResult("Latin", 30.0, 0, 9.0), 59.99, (None, "none")),
    (OsdResult("Fraktur", 30.0, 0, 9.0), 80.0, ("Latin", "osd")),      # Fraktur is Latin, never non-Latin
    (OsdResult("Korean", 8.0, 0, 9.0), None, ("Hangul", "osd")),
    (OsdResult("Ethiopic", 8.0, 0, 9.0), None, ("Ethiopic", "osd")),   # a script core doesn't know passes through
    (OsdResult(None, 0.0, 0, 0.0), 75.0, ("Latin", "osd")),
    (None, 60.0, ("Latin", "osd")),
    (None, None, (None, "none")),
])
def test_script_call_thresholds(osd, eng, want):
    sc = script_call(osd, eng)
    assert (sc.name, sc.by) == want
    assert sc.eng_mean_conf == eng
    assert sc.conf == (osd.script_conf if osd is not None else 0.0)


def test_script_call_reads_the_thresholds_from_langplan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(langplan, "OSD_NONLATIN_MIN_CONF", 20.0)
    monkeypatch.setattr(langplan, "LATIN_ENG_MIN_MEANCONF", 90.0)
    assert script_call(OsdResult("Thai", 12.0, 0, 9.0), 85.0).name is None


@pytest.mark.parametrize("bad", [-1.0, 100.5, float("nan"), float("inf")])
def test_script_call_refuses_a_conf_off_the_0_100_scale(bad):
    with pytest.raises(ValueError):
        script_call(None, bad)


def test_script_call_takes_the_read_itself():
    read = build.engine_read("eng", ["a b"], meta={"mean_conf": 0.81})          # meta is on [0, 1]
    sc = script_call(None, read=read)
    assert (sc.name, sc.eng_mean_conf) == ("Latin", pytest.approx(81.0))
    assert script_call(None, read=build.engine_read("eng", ["a"], conf=0.3)).name is None
    with pytest.raises(ValueError):
        script_call(None, 81.0, read=read)


def test_read_mean_conf_is_on_the_0_100_scale():
    assert read_mean_conf(build.engine_read("eng", ["a b"], meta={"mean_conf": 0.81})) == pytest.approx(81.0)
    assert read_mean_conf(build.engine_read("eng", ["a", "b"], conf=0.7)) == pytest.approx(70.0)
    words = build.engine_read("eng", lines=[_words_line([("a", 0.5), ("b", 1.0), ("c", 0.9)])])
    assert read_mean_conf(words) == pytest.approx(80.0)
    assert read_mean_conf(build.engine_read("eng", [])) is None
    assert read_mean_conf(build.engine_read("eng", ["x"], status="failed")) is None
    assert read_mean_conf(None) is None


# --------------------------------------------------------------------------- run_probe

def test_run_probe_counts_its_script_on_confident_lines(tmp_path):
    page = build.page_image(tmp_path)
    lines = [build.line("EXAMPLE", conf=0.99),
             build.line(THAI_LINE, conf=0.95),
             build.line("ก Straße 19", conf=0.97),                 # one Thai character: under min_line_chars
             build.line("ขอบคุณ", conf=0.79),                      # under min_conf
             build.line(THANKS, conf=0.8)]                         # conf exactly at the bar
    sess = StubSession(build.engine_read("probe", lines=lines, role="probe"))
    r = run_probe(page, sess, pack="th", script="Thai", min_conf=0.8, min_line_chars=2)
    assert r == ProbeResult("th", len(THAI_LINE) + len(THANKS), 2)
    assert sess.timeouts == [detect.PROBE_TIMEOUT_S]


def test_low_confidence_lookalike_junk_counts_for_nothing(tmp_path):
    page = build.page_image(tmp_path)
    junk = [build.line("นอpอysอNเzs เนsองอyเe 1oyouyนeq d10", conf=0.66) for _ in range(5)]
    r = run_probe(page, StubSession(build.engine_read("probe", lines=junk)), pack="th", script="Thai",
                  min_conf=0.8, min_line_chars=2)
    assert r == ProbeResult("th", 0, 0)


def test_a_failing_probe_finds_nothing(tmp_path):
    page = build.page_image(tmp_path)
    assert run_probe(page, StubSession(fail=True), pack="th", script="Thai", min_conf=0.8,
                     min_line_chars=2) == ProbeResult("th", 0, 0)
    timed_out = build.engine_read("probe", [THAI_LINE], status="timeout")
    assert run_probe(page, StubSession(timed_out), pack="th", script="Thai", min_conf=0.8,
                     min_line_chars=2) == ProbeResult("th", 0, 0)


def test_run_probe_carries_the_manifests_min_chars(tmp_path):
    page = build.page_image(tmp_path)
    sess = StubSession(build.engine_read("probe", lines=[build.line(THAI_LINE, conf=0.95)]))
    r = run_probe(page, sess, pack="th", script="Thai", min_conf=0.8, min_line_chars=2, min_chars=40)
    assert r == ProbeResult("th", len(THAI_LINE), 1, 40)
    assert run_probe(page, StubSession(fail=True), pack="th", script="Thai", min_conf=0.8, min_line_chars=2,
                     min_chars=40).min_chars == 40
    with pytest.raises(ValueError):
        run_probe(page, sess, pack="th", script="Thai", min_conf=0.8, min_line_chars=2, min_chars=0)


def test_run_probe_checks_its_arguments(tmp_path):
    page = build.page_image(tmp_path)
    sess = StubSession(build.engine_read("probe", [THAI_LINE]))
    with pytest.raises(ValueError):
        run_probe(page, sess, pack="th", script="Klingon", min_conf=0.8, min_line_chars=2)
    with pytest.raises(ValueError):
        run_probe(page, sess, pack="th", script="Thai", min_conf=1.5, min_line_chars=2)
    with pytest.raises(ValueError):
        run_probe(page, sess, pack="th", script="Thai", min_conf=0.5, min_line_chars=0)


# Made-up Vietnamese lines, and the same lines as a Thai probe's recogniser reads them: the
# horned letters (ư, ơ and their tones) come back as Thai lookalikes, 5 to 10 per line, at a
# confidence that clears the bar (as measured on the Vietnamese kit pages, 2026-11-06).
VIETNAMESE = [
    "Mùa mưa năm nay đến sớm hơn mọi năm, nước sông dâng lên từng ngày.",
    "Người dân trong làng dựng lại hàng rào tre trước khi cơn bão đi qua.",
    "Trên cánh đồng, những thửa ruộng mới cấy đã xanh mướt sau hai tuần.",
    "Ở chợ phiên cuối tuần, người ta bán rau, cá và những giỏ trứng tươi.",
    "Thư viện của trường mở cửa đến tối để học sinh có chỗ ôn bài yên tĩnh.",
    "Con đường dẫn lên đồi được lát đá, hai bên trồng hàng cây phượng đỏ.",
]
_AS_THAI = str.maketrans({"ư": "น", "ừ": "น", "ứ": "น", "ữ": "น", "ự": "น", "ử": "น",
                          "ơ": "ซ", "ờ": "ซ", "ớ": "ซ", "ở": "ซ", "ợ": "ซ"})
VIETNAMESE_AS_THAI = [t.translate(_AS_THAI) for t in VIETNAMESE]


def _probe(tmp_path, lines: Sequence[tuple[str, float]], **kw: Any) -> ProbeResult:
    sess = StubSession(build.engine_read("probe", lines=[build.line(t, conf=c) for t, c in lines], role="probe"))
    return run_probe(build.page_image(tmp_path), sess, pack="th", script="Thai", min_conf=0.8, min_line_chars=2,
                     min_chars=20, **kw)


def test_stray_thai_characters_in_vietnamese_lines_count_for_nothing(tmp_path):
    lines = [(t, 0.85 + 0.01 * k) for k, t in enumerate(VIETNAMESE_AS_THAI)]
    for t in VIETNAMESE_AS_THAI:
        n, other = detect._line_script_counts(t, "Thai")
        assert 5 <= n <= 10 and n < 0.25 * (n + other)
    assert _probe(tmp_path, lines) == ProbeResult("th", 0, 0, 20)
    # Counted line by line without the share, they pass the pack's min_chars: the page used to go to th.
    old = _probe(tmp_path, lines, min_line_share=0.0)
    assert old.lines == len(lines) and old.chars >= 20


def test_a_thai_block_among_english_lines_still_counts(tmp_path):
    sky = "ฝนตกหนักตลอดทั้งคืนที่ผ่านมา"                    # made up; every character is Thai
    garden = "สวนหลังบ้านมีดอกไม้หลายสี"
    library = "ห้องสมุดเปิด Monday ถึง Friday"                # Thai with two English words: 15 of 27 letters
    lines = [("Rivers carry water from the hills down to the sea.", 0.97), (sky, 0.93),
             ("The garden behind the school has flowers of many colours.", 0.96), (garden, 0.91),
             (library, 0.9), ("Clouds form when warm air cools high above the land.", 0.95)]
    assert detect._line_script_counts(library, "Thai") == (15, 12)
    r = _probe(tmp_path, lines)
    assert r == ProbeResult("th", len(sky) + len(garden) + 15, 3, 20)
    assert _probe(tmp_path, lines, min_line_share=0.0) == r              # nothing here is stray


def test_the_line_share_bar_is_inclusive_and_counts_only_letters(tmp_path):
    assert detect.PROBE_LINE_MIN_SHARE == 0.5
    at_half = ("กขAB", 0.9)                                              # 2 Thai of 4 letters
    under = ("กขABC", 0.9)                                               # 2 of 5
    assert _probe(tmp_path, [at_half]).chars == 2
    assert _probe(tmp_path, [under]).chars == 0
    # Digits, punctuation, spaces and unattached marks belong to no script and count for neither side.
    assert detect._line_script_counts("กข 12-34, (x)", "Thai") == (2, 1)
    assert detect._line_script_counts("กข é", "Thai") == (2, 1)
    assert _probe(tmp_path, [("กข 12-34 / 56.", 0.9)]).chars == 2
    # Letters of any other script count against the line, not only Latin.
    assert detect._line_script_counts("กขαβγ", "Thai") == (2, 3)
    assert _probe(tmp_path, [("กขαβγ", 0.9)]).chars == 0
    assert _probe(tmp_path, [under], min_line_share=0.4).chars == 2
    assert _probe(tmp_path, [under], min_line_share=0.0).chars == 2


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan, math.inf, True, "0.5", None])
def test_run_probe_checks_its_line_share(tmp_path, bad):
    sess = StubSession(build.engine_read("probe", [THAI_LINE]))
    with pytest.raises(ValueError):
        run_probe(build.page_image(tmp_path), sess, pack="th", script="Thai", min_conf=0.8, min_line_chars=2,
                  min_line_share=bad)


def test_run_probe_with_the_fake_engine(tmp_path):
    img = build.text_like(1200, 800, lines=3)
    page = build.page_image(tmp_path, image=img, dpi=200)
    engine = build.FakeEngine(texts=[THAI_LINE, "Latin words", THAI_LINE], platforms=["fake-os"])
    sess = engine.open("probe", "probe", {"lang": "en"}, {})
    r = run_probe(page, sess, pack="th", script="Thai", min_conf=0.8, min_line_chars=2)
    assert r.pack == "th" and r.lines == 2 and r.chars == 2 * len(THAI_LINE)


# --------------------------------------------------------------------------- mixed_script_check

def _page(tmp_path, dpi: float = 150.0, size: tuple[int, int] = (1240, 1754)):
    return build.page_image(tmp_path, image=build.blank(*size, color=(255, 255, 255)), dpi=dpi)


def _eng(lines: Sequence[tuple[float, float, float, float, float]]) -> EngineRead:
    """An eng read from (x0, y0, x1, y1, conf) boxes."""
    return build.engine_read("tesseract", lines=[build.line(f"line {i}", build.quad(x0, y0, x1, y1), conf=c)
                                                 for i, (x0, y0, x1, y1, c) in enumerate(lines)])


def test_mixed_script_fires_on_a_low_block_and_scales_it_to_300_dpi(tmp_path):
    page = _page(tmp_path, dpi=150)
    eng = _eng([(100, 100, 1100, 140, 0.92), (100, 300, 900, 340, 0.31), (100, 350, 900, 390, 0.28),
                (100, 600, 1100, 640, 0.95)])
    osd = StubOsd(lambda img, dpi: OsdResult("Thai", 12.0, 0, 5.0))
    m = mixed_script_check(page, eng, osd, calib())
    assert m == MixedScript(True, "Thai", 1, 12.0)                 # the two low lines merge into one block
    assert len(osd.calls) == 1
    (w, h), dpi = osd.calls[0]
    assert dpi == pytest.approx(300.0, rel=0.01)
    pad = 2 * max(2.0, 0.25 * 90)
    assert (w, h) == (pytest.approx(2 * (800 + pad), abs=4), pytest.approx(2 * (90 + pad), abs=4))


def test_mixed_script_does_not_fire_on_latin_or_a_weak_call(tmp_path):
    page = _page(tmp_path)
    eng = _eng([(100, 300, 900, 340, 0.2), (100, 800, 900, 840, 0.1)])
    latin = StubOsd(lambda img, dpi: OsdResult("Latin", 40.0, 0, 5.0))
    assert mixed_script_check(page, eng, latin, calib()) == MixedScript(True, None, 2, None)
    weak = StubOsd(lambda img, dpi: OsdResult("Thai", 4.9, 0, 5.0))
    assert mixed_script_check(page, eng, weak, calib()) == MixedScript(True, None, 2, None)
    none = StubOsd(lambda img, dpi: None)
    assert mixed_script_check(page, eng, none, calib()) == MixedScript(True, None, 2, None)

    def boom(img, dpi):
        raise EngineFailure("made_up", "a made-up OSD failure")
    assert mixed_script_check(page, eng, StubOsd(boom), calib()) == MixedScript(True, None, 2, None)


def test_mixed_script_takes_the_most_confident_non_latin_call(tmp_path):
    page = _page(tmp_path)
    eng = _eng([(100, 300, 900, 340, 0.2), (100, 800, 900, 840, 0.1), (100, 1200, 900, 1240, 0.3)])
    answers = iter([OsdResult("Cyrillic", 7.0, 0, 5.0), OsdResult("Thai", 15.0, 0, 5.0), OsdResult("Latin", 50.0, 0, 5.0)])
    m = mixed_script_check(page, eng, StubOsd(lambda img, dpi: next(answers)), calib())
    assert m == MixedScript(True, "Thai", 3, 15.0)


def test_mixed_script_respects_the_calibration(tmp_path):
    page = _page(tmp_path, dpi=150)
    eng = _eng([(100, 100, 1100, 140, 0.45), (100, 300, 900, 305, 0.1),       # 5 px tall: 10 px at 300 dpi
                (100, 500, 900, 540, 0.2), (100, 800, 900, 840, 0.3), (100, 1100, 900, 1140, 0.25)])
    osd = StubOsd(lambda img, dpi: OsdResult("Latin", 40.0, 0, 5.0))
    assert mixed_script_check(page, eng, osd, calib(block_max_conf=0.4, min_block_px=20, max_blocks=2)).blocks == 2
    assert len(osd.calls) == 2
    first_heights = sorted(h for (w, h), _ in osd.calls)
    assert all(h > 60 for h in first_heights)                     # the 5 px block was too small
    osd2 = StubOsd(lambda img, dpi: OsdResult("Latin", 40.0, 0, 5.0))
    assert mixed_script_check(page, eng, osd2, calib(block_max_conf=0.5, min_block_px=5, max_blocks=10)).blocks == 5
    osd3 = StubOsd(lambda img, dpi: OsdResult("Thai", 6.0, 0, 5.0))
    assert mixed_script_check(page, eng, osd3, calib(osd_min_conf=6.5)).script is None


def test_mixed_script_without_calibration_never_calls_osd(tmp_path):
    page = _page(tmp_path)
    eng = _eng([(100, 100, 1100, 140, 0.92), (100, 300, 900, 340, 0.49), (100, 600, 900, 640, 0.5),
                (100, 900, 900, 940, 0.1)])
    osd = StubOsd(lambda img, dpi: OsdResult("Thai", 99.0, 0, 5.0))
    m = mixed_script_check(page, eng, osd, None)
    assert m == MixedScript(False, None, 2, None) and osd.calls == []
    assert mixed_script_check(page, _eng([(100, 100, 900, 140, 0.9)]), osd, None) == MixedScript(False, None, 0, None)


def test_mixed_script_on_a_page_with_nothing_low(tmp_path):
    page = _page(tmp_path)
    osd = StubOsd(lambda img, dpi: OsdResult("Thai", 99.0, 0, 5.0))
    assert mixed_script_check(page, _eng([(100, 100, 900, 140, 0.9)]), osd, calib()) == MixedScript(True, None, 0, None)
    failed = build.engine_read("tesseract", ["x"], status="failed")
    assert mixed_script_check(page, failed, osd, calib()) == MixedScript(True, None, 0, None)
    assert osd.calls == []


def test_mixed_script_at_300_dpi_is_not_rescaled(tmp_path):
    page = _page(tmp_path, dpi=300, size=(2480, 3508))
    eng = _eng([(200, 600, 1800, 680, 0.2)])
    osd = StubOsd(lambda img, dpi: OsdResult("Thai", 12.0, 0, 5.0))
    assert mixed_script_check(page, eng, osd, calib()).script == "Thai"
    (w, h), dpi = osd.calls[0]
    assert dpi == 300.0 and h == pytest.approx(80 + 2 * 20, abs=2)


def test_mixed_script_groups_lines_by_tesseracts_block_ids(tmp_path):
    page = _page(tmp_path)
    lines = [build.line(f"line {i}", build.quad(100, y, 900, y + 40), conf=0.2) for i, y in enumerate((300, 350, 400))]
    osd = StubOsd(lambda img, dpi: OsdResult("Latin", 40.0, 0, 5.0))
    by_geometry = build.engine_read("tesseract", lines=lines)
    assert mixed_script_check(page, by_geometry, osd, calib()).blocks == 1
    by_id = build.engine_read("tesseract", lines=lines, meta={"line_blocks": [0, 0, 1]})
    assert mixed_script_check(page, by_id, osd, calib()).blocks == 2
    misaligned = build.engine_read("tesseract", lines=lines, meta={"line_blocks": [0, 1]})
    assert mixed_script_check(page, misaligned, osd, calib()).blocks == 1


def test_mixed_script_counts_a_nan_confidence_as_low(tmp_path):
    page = _page(tmp_path)
    eng = _eng([(100, 300, 900, 340, float("nan"))])
    assert mixed_script_check(page, eng, StubOsd(lambda i, d: None), None) == MixedScript(False, None, 1, None)
    osd = StubOsd(lambda img, dpi: OsdResult("Thai", 12.0, 0, 5.0))
    assert mixed_script_check(page, eng, osd, calib()) == MixedScript(True, "Thai", 1, 12.0)


def test_mixed_script_counts_only_crops_given_to_osd_and_clips_to_the_page(tmp_path):
    page = _page(tmp_path, size=(1240, 1754))
    osd = StubOsd(lambda img, dpi: OsdResult("Latin", 40.0, 0, 5.0))
    outside = _eng([(2000, 2000, 2600, 2040, 0.1), (-500, -300, -100, -260, 0.1)])
    assert mixed_script_check(page, outside, osd, calib()) == MixedScript(True, None, 0, None) and osd.calls == []
    partly = _eng([(-400, 300, 400, 340, 0.1)])                               # half of it lies left of the page
    assert mixed_script_check(page, partly, osd, calib()).blocks == 1
    (w, h), _ = osd.calls[0]
    pad = max(2.0, 0.25 * 40)
    assert w == pytest.approx(2 * (400 + pad), abs=4)                         # cropped from x = 0, not shifted


@pytest.mark.parametrize("damage", ["truncated", "jpeg", "missing", "size"])
def test_an_unreadable_render_is_a_named_engine_failure(tmp_path, damage):
    page = _page(tmp_path)
    eng = _eng([(100, 300, 900, 340, 0.1)])
    if damage == "truncated":
        page.path.write_bytes(page.path.read_bytes()[:200])
    elif damage == "jpeg":
        Image.new("RGB", (page.width, page.height), "white").save(page.path, format="JPEG")
    elif damage == "missing":
        page.path.unlink()
    else:
        Image.new("RGB", (page.width + 1, page.height), "white").save(page.path, format="PNG")
    with pytest.raises(EngineFailure) as e:
        mixed_script_check(page, eng, StubOsd(lambda i, d: None), calib())
    assert e.value.code == "page_unreadable"


@pytest.mark.parametrize("dpi", [0.0, -72.0, float("nan"), float("inf")])
def test_a_page_dpi_that_is_not_a_positive_number_is_a_named_engine_failure(tmp_path, dpi):
    from dataclasses import replace

    page = replace(_page(tmp_path), dpi=dpi)
    with pytest.raises(EngineFailure) as e:
        mixed_script_check(page, _eng([(100, 300, 900, 340, 0.1)]), StubOsd(lambda i, d: None), calib())
    assert e.value.code == "page_unreadable"
