"""Language planning (``ocr/langplan.py``) on synthetic signals and a fake index. Every text is made up.

The index here is a duck-typed stand-in for ``langpacks.index.Index`` (DESIGN §2.2): ``entries``,
``get``, ``base``, and per entry ``id``, ``base``, ``scripts``, ``langid_codes``, ``detect`` (the
copy of the pack's detect block) and ``dist.bytes``. The function-word lists are pack data in real
use; the ones below are a short sample of common Hungarian and rival words.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fixtures import build
from pdfintake.errors import SetupError, UsageError
from pdfintake.ocr import detect, langplan
from pdfintake.ocr.detect import FWScore, LangWindow, MixedScript, PageSignals, ProbeResult
from pdfintake.ocr.langplan import Calibration, Candidate, LanguagePlan, document_rule, plan
from pdfintake.record import NOTE_CODES, OPTION_ID_RE, WARNING_CODES
from pdfintake.types import LangCall, ScriptCall

HU_WORDS = ["és", "hogy", "az", "egy", "vagy", "csak", "már", "még", "kell", "lesz", "volt", "ezt", "azt", "ez",
            "pedig", "illetve", "valamint", "szerint", "alapján", "között", "részére", "után", "által", "amely",
            "amelyet", "amelyek", "továbbá", "mint", "meg", "nem"]
RIVALS = ["und", "der", "die", "das", "ist", "nicht", "für", "von", "den", "dem", "des", "sie", "wir", "ich", "ein",
          "eine", "einen", "auf", "auch", "wird", "werden", "oder", "sind", "über", "zur", "zum",
          "que", "para", "uma", "dos", "das", "los", "las", "del", "por", "una", "les", "est", "une", "pour", "dans",
          "avec", "sur", "ve", "bir", "bu", "için", "ile", "olan", "olarak", "gibi", "daha", "çok", "sonra", "kadar"]

HU = ["A kertészek tapasztalják tavasszal, hogy a régi fák és a rózsák illata minden reggel egyre erősebben árad,",
      "illetve a nyári zápor után szellő támad, amely a meteorológusok szerint az egyik hétvégén sem lesz erős.",
      "A könyvtár egy régi épületben működik, továbbá a polcok között minden olvasó esetén a térkép segít, ha kell."]
DE = ["Bitte senden Sie uns die unterschriebene Kopie bis Ende Mai zurück, damit wir den Auftrag zügig bearbeiten können.",
      "Mit freundlichen Grüßen aus dem Musterbüro, wir freuen uns auf eine weiterhin gute Zusammenarbeit im neuen Jahr."]
EN_VAN = ["The notice from van der Example to Van Sample and van Placeholder about the delivery of the sample order",
          "says that the parcel will arrive on Friday and that the invoice is attached to this message for your records."]
TR = ["Bu kitap ve bir köy tarihi için yazıldı, ile olan bağı olarak az gibi görünse de daha çok az ez okunan bir eser.",
      "Yazar köyün eski evlerini, okulunu ve çarşısını anlatıyor; her bölüm kısa bir hikâye ile başlıyor ve sade bir dille ilerliyor."]
FR = ["Nous vous remercions de votre commande et nous vous confirmons que le colis sera livré à votre adresse avant vendredi.",
      "La facture est jointe à ce message et reste à votre disposition pour toute question concernant votre compte client."]
ES = ["El ejemplo de factura incluye siete artículos de muestra y un total de cuatro mil unidades para el cliente de prueba.",
      "La empresa enviará el pedido a la dirección indicada en el formulario antes del viernes, según lo acordado por teléfono."]
EN = ["The example invoice lists seven sample items and a total of four thousand units for the test customer account.",
      "The company will send the order to the address given on the form before Friday, as agreed on the phone last week."]
SPARSE = ["2026. 03. 15. Jane Doe s.k."]                           # a signature line: too little text for a window
DE_RICH = ["Die Firma und der Kunde sind sich einig, dass die Ware auf dem Weg zur Filiale ist und von dem Fahrer gebracht wird.",
           "Der Preis für die Ware wird von dem Kunden auf das Konto der Firma gezahlt, und die Rechnung ist auch in der Anlage."]


# --------------------------------------------------------------------------- a fake index

@dataclass(frozen=True)
class FakeDist:
    kind: str
    distribution: str
    url: str | None
    sha256: str
    bytes: int


@dataclass(frozen=True)
class FakeEntry:
    id: str
    version: str
    base: bool
    title: str
    scripts: tuple[str, ...]
    langid_codes: tuple[str, ...]
    detect: Mapping[str, Any]
    manifest_sha256: str
    dist: FakeDist
    installed_bytes: int


@dataclass(frozen=True)
class FakeIndex:
    version: int
    entries: tuple[FakeEntry, ...]

    def get(self, pack_id: str) -> FakeEntry | None:
        return next((e for e in self.entries if e.id == pack_id), None)

    def base(self) -> FakeEntry:
        return next(e for e in self.entries if e.base)

    def for_script(self, script: str) -> list[FakeEntry]:
        return [e for e in self.entries if script in e.scripts]

    def for_language(self, code: str) -> list[FakeEntry]:
        return [e for e in self.entries if code in e.langid_codes]


def entry(pack_id: str, *, scripts=("Latin",), codes=None, detect_block=None, base=False, size=9_000_000) -> FakeEntry:
    d = {"osd_scripts": list(scripts[:1])} if detect_block is None else detect_block
    kind = "installed" if base else "fetch"
    return FakeEntry(pack_id, "1.0.0", base, pack_id.upper(), tuple(scripts), tuple(codes or (pack_id,)), d,
                     "0" * 64, FakeDist(kind, f"pdfintake-lang-{pack_id}", None if base else "https://example.com/x.whl",
                                        "0" * 64, 0 if base else size), size)


EN_ENTRY = entry("en", base=True, detect_block={"osd_scripts": ["Latin"], "probe": False})
# The real index copies a pack's probe as a boolean (langpacks.index.detect_copy).
TH_ENTRY = entry("th", scripts=("Thai", "Latin"), size=11_900_000,
                 detect_block={"osd_scripts": ["Thai"], "langid": {"codes": ["th"]}, "probe": True})
# The exact third, as packs/hu/manifest.json ships it (DESIGN §11.2).
HU_RATIO = 0.3333333333333333
HU_ENTRY = entry("hu", size=12_350_405,
                 detect_block={"osd_scripts": ["Latin"], "langid": {"codes": ["hu"]},
                               "function_words": {"words": HU_WORDS, "rivals": RIVALS, "min_hits": 3,
                                                  "ratio": HU_RATIO}})
DE_ENTRY = entry("de", size=8_628_461, detect_block={"osd_scripts": ["Latin"], "langid": {"codes": ["de"]}})
ES_ENTRY = entry("es", size=13_570_187, detect_block={"osd_scripts": ["Latin"], "langid": {"codes": ["es"]}})
HR_ENTRY = entry("hr", size=9_000_000, detect_block={"osd_scripts": ["Latin"], "langid": {"codes": ["hr"]}})

INDEX = FakeIndex(1, (EN_ENTRY, TH_ENTRY, HU_ENTRY, DE_ENTRY, ES_ENTRY, HR_ENTRY))
ALL = frozenset(e.id for e in INDEX.entries)


# --------------------------------------------------------------------------- signals

def latin_call(eng_conf: float = 85.0) -> ScriptCall:
    return ScriptCall("Latin", 30.0, "osd", eng_conf)


UNDECIDED = ScriptCall(None, 0.0, "none", None)


def sig(i: int, lines: Sequence[str] = (), *, checker: Sequence[str] | None = None, script: ScriptCall | None = None,
        probes: Mapping[str, ProbeResult] | None = None, mixed: MixedScript | None = None,
        windows: list[LangWindow] | None = None) -> PageSignals:
    eng = build.engine_read("tesseract", lines)
    chk = build.engine_read("rapidocr", checker if checker is not None else lines, role="checker")
    return PageSignals(index=i, osd=None, script=script if script is not None else latin_call(), eng_read=eng,
                       checker_read=chk, probes=dict(probes or {}),
                       windows=windows if windows is not None else detect.language_windows(eng),
                       mixed=mixed if mixed is not None else MixedScript(False, None, 0, None))


def thai_sig(i: int, conf: float = 48.2) -> PageSignals:
    return sig(i, ["ข้อความตัวอย่าง"], script=ScriptCall("Thai", conf, "osd", 20.0), windows=[])


def run(signals: Sequence[PageSignals], *, index: FakeIndex = INDEX, lang: Sequence[str] = (),
        force_lang: Sequence[str] = (), installed=ALL, fetch: str = "auto") -> LanguagePlan:
    return plan(signals, index, lang=lang, force_lang=force_lang, installed=installed, fetch=fetch)  # type: ignore[arg-type]


def packs(p: LanguagePlan) -> list[str | None]:
    return [pp.pack for pp in p.pages]


def check_codes(p: LanguagePlan) -> None:
    """Every code the plan emits belongs to the record's closed sets, and option ids fit the pattern."""
    for pp in p.pages:
        assert pp.composition in langplan.COMPOSITIONS
        assert set(pp.warnings) <= WARNING_CODES and set(pp.notes) <= NOTE_CODES
    assert set(p.notes) <= NOTE_CODES
    for c in p.candidates:
        if c.pack is not None:
            assert OPTION_ID_RE.fullmatch(c.tag)
    for pp in p.pages:
        assert pp.unsure in (None, *langplan.UNSURE_KINDS)


class StubIdentifier:
    """Ranks a text by the code its first word starts with ("hr-w001 ..." -> table["hr"]); "*" is the default."""

    name: ClassVar[str] = "py3langid"

    def __init__(self, table: Mapping[str, list[tuple[str, float]]]) -> None:
        self.table = table

    def classify(self, text: str) -> list[tuple[str, float]]:
        if not text.strip():
            return []
        key = text.split()[0].split("-")[0]
        return list(self.table.get(key, self.table.get("*", [])))


@pytest.fixture
def stub_ident(monkeypatch: pytest.MonkeyPatch):
    def install(table: Mapping[str, list[tuple[str, float]]]) -> None:
        monkeypatch.setitem(detect._IDENTIFIERS, "py3langid", StubIdentifier(table))
    return install


def stub_lines(code: str, n_chars: int = 400) -> list[str]:
    words = [f"{code}-w{i:03d}" for i in range(n_chars // 8 + 1)]
    return [" ".join(words[i:i + 8]) for i in range(0, len(words), 8)]


# --------------------------------------------------------------------------- the Hungarian function-word cases

def test_a_hungarian_page_goes_to_hu_and_a_german_page_to_de():
    p = run([sig(0, HU), sig(1, DE)])
    assert packs(p) == ["hu", "de"]
    hu, de = p.pages
    assert hu.languages[0].tag == "hu" and hu.languages[0].by == "function_words" and hu.composition == "language"
    assert de.languages[0].tag == "de" and de.languages[0].by == "langid"
    assert p.confident and p.packs_needed == ["de", "hu"] and p.candidates == []
    check_codes(p)


def test_a_turkish_page_with_az_and_ez_is_not_hungarian():
    s = sig(0, TR)
    fw = detect.function_word_score(s.checker_read, frozenset(HU_WORDS), frozenset(RIVALS))
    assert fw.hits >= 3                                           # az, az, ez: enough words, but outnumbered
    p = run([s])
    assert p.pages[0].pack != "hu"
    assert not p.confident and p.unknown_language == "tr"          # no Turkish pack
    assert [c.tag for c in p.candidates] == ["tr", "en"] and p.candidates[0].pack is None
    check_codes(p)


def test_the_english_van_page_stays_english():
    s = sig(0, EN_VAN)
    assert detect.function_word_score(s.checker_read, frozenset(HU_WORDS), frozenset(RIVALS)).hits == 0
    p = run([s, sig(1, HU)])
    assert packs(p) == ["en", "hu"]
    assert p.pages[0].languages[0] == LangCall("en", p.pages[0].languages[0].prob, "langid", len(s.windows))


def test_the_one_third_hu_de_page_is_hungarian_with_german_as_second_language():
    s = sig(0, HU + DE_RICH + DE)
    fw = detect.function_word_score(s.checker_read, frozenset(HU_WORDS), frozenset(RIVALS))
    assert (fw.hits, fw.rival_hits) == (12, 35)                   # three times as many German function words
    p = run([s])
    pp = p.pages[0]
    assert pp.pack == "hu" and pp.confident
    assert [c.tag for c in pp.languages] == ["hu", "de"]
    assert pp.languages[1].by == "langid" and pp.languages[1].windows >= langplan.SECOND_LANG_MIN_WINDOWS
    assert pp.warnings == ["mixed_language_page"]
    check_codes(p)
    below = sig(0, HU[:2] + DE_RICH + DE)                         # 8 against 35: under a third
    assert run([below]).pages[0].pack != "hu"


@pytest.mark.parametrize("ratio", [None, HU_RATIO])               # the default, and the shipped manifest value
def test_function_word_ratio_boundary(stub_ident, ratio):
    stub_ident({"zz": [("zz", 0.5), ("yy", 0.5)]})
    fw: dict[str, Any] = {"words": ["hx"], "rivals": ["dx"], "min_hits": 3}
    if ratio is not None:
        fw["ratio"] = ratio
    idx = FakeIndex(1, (EN_ENTRY, entry("hu", detect_block={"osd_scripts": ["Latin"], "function_words": fw})))
    at_third = sig(0, ["zz-a hx hx hx " + "dx " * 9 + "x" * 120])
    over = sig(0, ["zz-a hx hx hx " + "dx " * 10 + "x" * 120])
    two_hits = sig(0, ["zz-a hx hx " + "x" * 120])
    assert run([at_third], index=idx).pages[0].pack == "hu"           # 3 against 9: exactly a third
    assert run([over], index=idx).pages[0].pack != "hu"               # 3 against 10
    assert run([two_hits], index=idx).pages[0].pack != "hu"           # under min_hits


def test_the_shipped_ratio_is_the_exact_third():
    """``hits >= ratio * rivals - 1e-9`` with 0.3333333333333333 is exactly ``3 * hits >= rivals``."""
    assert HU_RATIO == langplan.FW_RATIO_DEFAULT == 1 / 3
    fw = langplan._FW(frozenset({"hx"}), frozenset({"dx"}), 1, HU_RATIO)
    for hits in range(1, 61):
        for rivals in range(0, 3 * hits + 4):
            got = langplan._fires(FWScore(hits, rivals, 0, hits + rivals), fw)
            assert got == (3 * hits >= rivals), (hits, rivals)


def test_the_ratio_tolerance_sits_below_the_product():
    """A product a hair above the integer bar still passes; one clearly above it doesn't."""
    fw = langplan._FW(frozenset({"hx"}), frozenset({"dx"}), 1, 0.5 + 1e-12)   # 0.5 * 6 lands just above 3
    assert langplan._fires(FWScore(3, 6, 0, 9), fw)
    assert not langplan._fires(FWScore(3, 7, 0, 10), fw)
    strict = langplan._FW(frozenset({"hx"}), frozenset({"dx"}), 1, 0.5 + 1e-6)
    assert not langplan._fires(FWScore(3, 6, 0, 9), strict)


def test_the_hungarian_pack_manifest_ships_the_exact_third():
    manifest = Path(__file__).resolve().parents[1] / "packs" / "hu" / "manifest.json"
    if not manifest.is_file():
        pytest.skip("packs/hu/manifest.json isn't in this tree")
    fw = json.loads(manifest.read_text(encoding="utf-8"))["detect"]["function_words"]
    assert fw["ratio"] == HU_RATIO


def test_the_function_word_rule_reads_the_checker_read():
    s = sig(0, EN, checker=HU)                                     # eng says English, the checker read says Hungarian
    assert run([s]).pages[0].pack == "hu"
    failed = replace(s, checker_read=build.engine_read("rapidocr", HU, role="checker", status="failed"))
    assert run([failed]).pages[0].pack == "en"                     # a failed checker: the eng read decides


def test_a_sparse_page_in_a_hungarian_document_inherits_hungarian():
    p = run([sig(0, HU), sig(1, SPARSE)])
    assert packs(p) == ["hu", "hu"]
    doc = p.pages[1]
    assert doc.composition == "document" and doc.languages == [LangCall("hu", 1.0, "document", 0)]
    assert p.confident


def test_no_document_rule_when_most_text_pages_are_another_language():
    p = run([sig(0, HU), sig(1, DE + DE), sig(2, DE + DE), sig(3, SPARSE)])
    assert packs(p)[:3] == ["hu", "de", "de"]
    assert p.pages[3].pack != "hu"


def test_a_sparse_page_with_english_words_stays_with_the_base_pack():
    p = run([sig(0, HU), sig(1, ["the and of to the"])])
    assert p.pages[1].pack == "en" and p.pages[1].composition == "base"


# --------------------------------------------------------------------------- document_rule, unit cases

def _fw(hits=0, rivals=0, en=0, words=0) -> FWScore:
    return FWScore(hits, rivals, en, words)


def _sigs(n: int) -> list[PageSignals]:
    return [sig(i, [], windows=[]) for i in range(n)]


def test_document_rule_cases():
    s2 = _sigs(2)
    assert document_rule(s2, {"hu": [_fw(7, 0, 0, 40), _fw(0, 0, 0, 6)]}, {0: "hu"}) == {1: "hu"}
    s4 = _sigs(4)
    fw4 = {"hu": [_fw(7, 0, 0, 40), _fw(0, 8, 0, 40), _fw(0, 8, 0, 40), _fw(0, 0, 0, 6)]}
    assert document_rule(s4, fw4, {0: "hu"}) == {}                 # 1 of 3 text pages: no rule
    assert document_rule(s2, {"hu": [_fw(7, 0, 0, 40), _fw(0, 0, 3, 6)]}, {0: "hu"}) == {}    # English words
    assert document_rule(s2, {"hu": [_fw(7, 0, 0, 40), _fw(0, 1, 1, 6)]}, {0: "hu"}) == {1: "hu"}
    assert document_rule(s2, {"hu": [_fw(7, 0, 0, 40), _fw(0, 0, 0, 0)]}, {0: "hu"}) == {}    # no words at all
    assert document_rule(s2, {"hu": [_fw(7, 0, 0, 40), _fw(0, 0, 0, 6)]}, {}) == {}           # nothing chosen


def test_document_rule_holds_at_half_and_in_the_vacuous_case():
    s4 = _sigs(4)
    half = {"hu": [_fw(7, 0, 0, 40), _fw(0, 9, 0, 40), _fw(0, 0, 0, 5), _fw(0, 0, 0, 5)]}
    assert document_rule(s4, half, {0: "hu", 1: "de"}) == {2: "hu", 3: "hu"}
    sparse_only = {"hu": [_fw(3, 0, 0, 10), _fw(0, 0, 0, 5)]}         # no page has 30 words
    assert document_rule(_sigs(2), sparse_only, {0: "hu"}) == {1: "hu"}


def test_document_rule_prefers_the_pack_most_pages_chose_and_checks_lengths():
    s3 = _sigs(3)
    fw = {"de": [_fw(0, 0, 0, 10), _fw(0, 0, 0, 10), _fw(0, 0, 0, 3)],
          "hu": [_fw(0, 0, 0, 10), _fw(0, 0, 0, 10), _fw(0, 0, 0, 3)]}
    assert document_rule(s3, fw, {0: "hu", 1: "hu"}) == {2: "hu"}
    with pytest.raises(ValueError):
        document_rule(s3, {"hu": [_fw()]}, {0: "hu"})


# --------------------------------------------------------------------------- scripts

def test_a_thai_page_goes_to_the_thai_pack_and_needs_a_fetch_on_a_fresh_install():
    p = run([thai_sig(0)], installed={"en"})
    pp = p.pages[0]
    assert pp.pack == "th" and pp.composition == "script" and pp.languages == [LangCall("th", 1.0, "pack_script", 0)]
    assert p.confident and p.packs_needed == ["th"]
    assert p.unavailable == []                                    # auto: every published pack may be fetched
    never = run([thai_sig(0)], installed={"en"}, fetch="never")
    assert not never.confident and never.pages[0].pack == "th" and never.pages[0].confident
    assert [(c.tag, c.pack, c.download_bytes) for c in never.candidates] == [("th", "th", 11_900_000), ("en", "en", 0)]
    assert never.unavailable == ["de", "es", "hr", "hu", "th"]
    assert run([thai_sig(0)], installed={"en", "th"}, fetch="never").confident


def test_undecided_pages_take_the_document_vote():
    p = run([thai_sig(0), thai_sig(1), sig(2, SPARSE, script=UNDECIDED, windows=[])])
    assert packs(p) == ["th", "th", "th"]
    assert p.pages[2].script == ScriptCall("Thai", 1.0, "vote", None)
    latin = run([sig(0, EN), sig(1, ES, script=UNDECIDED)])
    assert packs(latin) == ["en", "es"]
    assert latin.pages[1].script is not None and latin.pages[1].script.by == "vote"


def test_a_script_vote_tie_goes_to_latin():
    p = run([thai_sig(0), sig(1, EN), sig(2, SPARSE, script=UNDECIDED, windows=[])])
    assert p.pages[2].script is not None and p.pages[2].script.name == "Latin"
    assert p.pages[2].pack == "en"


def test_a_wholly_undecided_document_is_read_by_the_base_pack():
    p = run([sig(0, SPARSE, script=UNDECIDED, windows=[]), sig(1, ["x y"], script=UNDECIDED, windows=[])])
    assert packs(p) == ["en", "en"] and {pp.composition for pp in p.pages} == {"base"} and p.confident


def test_a_non_latin_script_with_no_pack_is_left_undecided_by_plan():
    p = run([sig(0, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])])
    assert p.pages[0].pack is None and not p.pages[0].confident and not p.confident
    assert p.unknown_language == "Cyrillic"
    assert [c.tag for c in p.candidates] == ["en"]
    assert run([sig(0, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])],
               lang=["en"]).pages[0].pack == "en"


# --------------------------------------------------------------------------- probes and the trigger

def test_a_probe_tripped_latin_page_goes_to_the_thai_pack():
    s = sig(0, EN, probes={"th": ProbeResult("th", 45, 3)})
    p = run([s])
    pp = p.pages[0]
    assert pp.pack == "th" and pp.composition == "probe" and pp.warnings == ["mixed_script_page"]
    assert pp.script == ScriptCall("Thai", 45.0, "probe", 85.0)
    assert p.confident
    under = run([sig(0, EN, probes={"th": ProbeResult("th", 19, 3)})])
    assert under.pages[0].pack == "en" and under.pages[0].notes == []       # a probe ran: no unchecked note
    check_codes(p)


def test_a_probe_page_is_not_counted_by_the_function_word_rule():
    s = sig(0, HU, probes={"th": ProbeResult("th", 60, 4)})
    assert run([s]).pages[0].pack == "th"


def test_on_a_fresh_install_a_trigger_hit_for_thai_is_a_candidate():
    hit = MixedScript(True, "Thai", 2, 12.0)
    auto = run([sig(0, EN, mixed=hit)], installed={"en"})
    pp = auto.pages[0]
    assert pp.pack == "th" and pp.composition == "trigger" and pp.warnings == ["mixed_script_page"]
    assert pp.script == ScriptCall("Thai", 12.0, "trigger", 85.0)
    assert auto.confident and auto.packs_needed == ["th"]
    never = run([sig(0, EN, mixed=hit)], installed={"en"}, fetch="never")
    assert not never.confident and never.candidates[0] == Candidate("th", "th", never.candidates[0].evidence, 11_900_000)
    check_codes(never)


def test_without_calibration_a_low_block_gets_the_unchecked_note():
    p = run([sig(0, EN, mixed=MixedScript(False, None, 2, None))], installed={"en"})
    assert p.pages[0].pack == "en" and p.pages[0].notes == ["mixed_script_unchecked"]
    assert p.notes == ["mixed_script_unchecked"] and p.confident
    clean = run([sig(0, EN, mixed=MixedScript(False, None, 0, None))], installed={"en"})
    assert clean.pages[0].notes == [] and clean.notes == []
    probed = run([sig(0, EN, mixed=MixedScript(False, None, 2, None), probes={"th": ProbeResult("th", 0, 0)})])
    assert probed.pages[0].notes == []
    check_codes(p)


def test_a_trigger_hit_for_a_script_with_no_pack_is_a_warning():
    p = run([sig(0, EN, mixed=MixedScript(True, "Cyrillic", 1, 9.0))])
    pp = p.pages[0]
    assert pp.pack == "en" and pp.warnings == ["mixed_script_no_pack"] and p.confident


def test_the_trigger_runs_when_the_packs_own_probe_ran_but_did_not_fire():
    s = sig(0, EN, mixed=MixedScript(True, "Thai", 2, 12.0), probes={"th": ProbeResult("th", 19, 1)})
    pp = run([s]).pages[0]
    assert pp.pack == "th" and pp.composition == "trigger" and pp.warnings == ["mixed_script_page"]
    fired = sig(0, EN, mixed=MixedScript(True, "Thai", 2, 12.0), probes={"th": ProbeResult("th", 45, 3)})
    assert run([fired]).pages[0].composition == "probe"                    # a probe that fired comes first


def test_a_probe_reported_for_a_pack_without_a_probe_is_ignored():
    s = sig(0, EN, probes={"hu": ProbeResult("hu", 500, 20), "zz": ProbeResult("zz", 500, 20)})
    assert run([s]).pages[0].pack == "en"


def test_a_probe_copied_as_true_uses_the_default_threshold():
    at = langplan.PROBE_MIN_CHARS_DEFAULT
    assert run([sig(0, EN, probes={"th": ProbeResult("th", at, 2)})]).pages[0].pack == "th"
    assert run([sig(0, EN, probes={"th": ProbeResult("th", at - 1, 2)})]).pages[0].pack == "en"


def test_the_installed_manifests_min_chars_travels_on_the_probe_result():
    assert run([sig(0, EN, probes={"th": ProbeResult("th", 25, 2, min_chars=30)})]).pages[0].pack == "en"
    assert run([sig(0, EN, probes={"th": ProbeResult("th", 12, 1, min_chars=10)})]).pages[0].pack == "th"


def test_a_full_probe_block_in_the_index_copy_is_read_too():
    th = replace(TH_ENTRY, detect={"osd_scripts": ["Thai"], "probe": {"min_chars": 40}})
    idx = FakeIndex(1, (EN_ENTRY, th))
    assert run([sig(0, EN, probes={"th": ProbeResult("th", 39, 2)})], index=idx).pages[0].pack == "en"
    assert run([sig(0, EN, probes={"th": ProbeResult("th", 40, 2)})], index=idx).pages[0].pack == "th"
    off = replace(TH_ENTRY, detect={"osd_scripts": ["Thai"], "probe": False})
    assert run([sig(0, EN, probes={"th": ProbeResult("th", 500, 9)})],
               index=FakeIndex(1, (EN_ENTRY, off))).pages[0].pack == "en"


def test_probe_routed_pages_count_as_text_pages_for_the_document_rule():
    en_long = EN + EN                                                   # 30+ words, English
    signals = [sig(0, HU), sig(1, en_long, probes={"th": ProbeResult("th", 60, 4)}),
               sig(2, en_long, probes={"th": ProbeResult("th", 60, 4)}), sig(3, SPARSE, windows=[])]
    p = run(signals)
    assert [(pp.pack, pp.composition) for pp in p.pages][:3] == [("hu", "language"), ("th", "probe"), ("th", "probe")]
    assert p.pages[3].pack == "en" and p.pages[3].composition == "base"       # 1 of 3 text pages: no rule


# Made-up Vietnamese lines, and the same lines as the Thai probe's recogniser reads them (the
# horned letters come back as Thai lookalikes, several per line, as on the Vietnamese kit pages).
VI = ["Mùa mưa năm nay đến sớm hơn mọi năm, nước sông dâng lên từng ngày.",
      "Người dân trong làng dựng lại hàng rào tre trước khi cơn bão đi qua.",
      "Trên cánh đồng, những thửa ruộng mới cấy đã xanh mướt sau hai tuần.",
      "Ở chợ phiên cuối tuần, người ta bán rau, cá và những giỏ trứng tươi.",
      "Thư viện của trường mở cửa đến tối để học sinh có chỗ ôn bài yên tĩnh.",
      "Con đường dẫn lên đồi được lát đá, hai bên trồng hàng cây phượng đỏ."]
VI_AS_THAI = [t.translate(str.maketrans({"ư": "น", "ừ": "น", "ứ": "น", "ữ": "น", "ự": "น", "ử": "น", "ơ": "ซ",
                                          "ờ": "ซ", "ớ": "ซ", "ở": "ซ", "ợ": "ซ"})) for t in VI]
TH_BLOCK = ["ฝนตกหนักตลอดทั้งคืนที่ผ่านมา", "สวนหลังบ้านมีดอกไม้หลายสี"]          # made up; all Thai
VI_ENTRY = entry("vi", size=10_000_000, detect_block={"osd_scripts": ["Latin"], "langid": {"codes": ["vi"]}})
VI_INDEX = FakeIndex(1, INDEX.entries + (VI_ENTRY,))


class ProbeSession:
    """The Thai probe's read of a page, as lines of (text, confidence)."""

    def __init__(self, lines: Sequence[tuple[str, float]]) -> None:
        self.read_ = build.engine_read("probe", lines=[build.line(t, conf=c) for t, c in lines], role="probe")

    def read(self, page: Any, *, timeout_s: float) -> Any:
        return self.read_


def th_probe(tmp_path: Path, lines: Sequence[tuple[str, float]], **kw: Any) -> ProbeResult:
    """``detect.run_probe`` with the th pack's shipped probe settings (min_chars 20, min_conf 0.8, 2 per line)."""
    return detect.run_probe(build.page_image(tmp_path), ProbeSession(lines), pack="th", script="Thai", min_conf=0.8,
                            min_line_chars=2, min_chars=20, **kw)


def test_stray_thai_in_a_vietnamese_probe_read_leaves_the_page_vietnamese(tmp_path: Path):
    probe_lines = [(t, 0.86 + 0.01 * k) for k, t in enumerate(VI_AS_THAI)]
    pr = th_probe(tmp_path, probe_lines)
    assert pr.chars == 0
    p = run([sig(0, VI, probes={"th": pr})], index=VI_INDEX)
    pp = p.pages[0]
    assert (pp.pack, pp.composition) == ("vi", "language") and pp.warnings == [] and p.confident
    assert pp.languages[0].tag == "vi" and pp.languages[0].prob >= langplan.LANGID_MIN_PROB
    # The same read counted without the line share: the probe fired and took the page from vi.
    old = th_probe(tmp_path, probe_lines, min_line_share=0.0)
    assert old.chars >= 20
    taken = run([sig(0, VI, probes={"th": old})], index=VI_INDEX).pages[0]
    assert (taken.pack, taken.composition) == ("th", "probe")
    check_codes(p)


def test_a_thai_block_on_a_vietnamese_page_still_goes_to_th(tmp_path: Path):
    probe_lines = [(t, 0.88) for t in VI_AS_THAI[:3]] + [(t, 0.93) for t in TH_BLOCK] + [(VI_AS_THAI[3], 0.9)]
    pr = th_probe(tmp_path, probe_lines)
    assert pr.chars == sum(len(t) for t in TH_BLOCK) and pr.lines == 2
    pp = run([sig(0, VI, probes={"th": pr})], index=VI_INDEX).pages[0]
    assert (pp.pack, pp.composition) == ("th", "probe") and pp.warnings == ["mixed_script_page"]


def test_thai_with_english_lines_goes_to_th(tmp_path: Path):
    english = EN + EN                                                   # a confident English call
    probe_lines = [(english[0], 0.97), (TH_BLOCK[0], 0.92), (english[1], 0.96), (TH_BLOCK[1], 0.9),
                   ("ห้องสมุดเปิด Monday ถึง Friday", 0.9)]               # Thai with two English words
    pr = th_probe(tmp_path, probe_lines)
    assert pr.lines == 3 and pr.chars == sum(len(t) for t in TH_BLOCK) + 15
    s = sig(0, english, probes={"th": pr})
    assert detect.page_language(s.eng_read)[0][0] == "en"
    p = run([s])
    pp = p.pages[0]
    assert (pp.pack, pp.composition) == ("th", "probe") and pp.warnings == ["mixed_script_page"] and p.confident
    # A page of Thai lines only, undecided by OSD (no eng text to speak of), goes to th as well.
    only = th_probe(tmp_path, [(t, 0.95) for t in TH_BLOCK * 2])
    assert run([sig(0, [], script=UNDECIDED, probes={"th": only}, windows=[])]).pages[0].pack == "th"


# --------------------------------------------------------------------------- languages without a confident pack

def test_an_unknown_language_with_no_pack_is_left_undecided_with_english_as_candidate():
    p = run([sig(0, FR)])
    assert not p.confident and p.unknown_language == "fr"
    assert p.pages[0].pack is None and p.pages[0].languages[0].tag == "fr"
    offered = [c for c in p.candidates if c.pack is not None]
    assert [(c.tag, c.pack, c.download_bytes) for c in offered] == [("en", "en", 0)]
    assert p.candidates[-1].tag == "en"
    check_codes(p)


def test_a_close_group_is_left_undecided_by_plan(stub_ident):
    stub_ident({"hr": [("hr", 0.9), ("bs", 0.07), ("sr", 0.03)]})
    p = run([sig(0, stub_lines("hr"))])
    assert not p.confident and p.pages[0].pack is None and p.pages[0].languages[0].tag == "hr"
    assert [(c.tag, c.pack) for c in p.candidates] == [("hr", "hr"), ("bs", None), ("sr", None), ("en", "en")]
    assert p.unknown_language is None
    check_codes(p)


def test_a_close_group_is_confident_when_one_pack_covers_it(stub_ident):
    stub_ident({"bs": [("bs", 0.95), ("hr", 0.05)]})
    whole = replace(HR_ENTRY, detect={"osd_scripts": ["Latin"], "langid": {"codes": ["hr"], "close_group": ["bs", "sr"]}})
    idx = FakeIndex(1, (EN_ENTRY, whole))
    p = run([sig(0, stub_lines("bs"))], index=idx)
    assert p.confident and p.pages[0].pack == "hr" and p.pages[0].languages[0].tag == "bs"
    alone = run([sig(0, stub_lines("bs"))])
    assert not alone.confident and alone.unknown_language == "bs"
    assert ("hr", "hr") in [(c.tag, c.pack) for c in alone.candidates]


def test_the_real_identifier_is_unsure_about_made_up_croatian():
    hr = ["Primjer računa sadrži sedam stavki uzorka i ukupno četiri tisuće jedinica, a tvrtka će poslati narudžbu prije petka.",
          "Molimo vas da potpisanu kopiju vratite do kraja mjeseca kako bismo mogli brzo obraditi vaš zahtjev i poslati robu."]
    p = run([sig(0, hr)])
    assert not p.confident and p.pages[0].pack is None
    assert [c.tag for c in p.candidates][:2] == ["hr", "bs"] and p.candidates[-1].tag == "en"


def test_a_close_group_split_by_the_identifier_is_judged_as_a_group(stub_ident):
    stub_ident({"hr": [("hr", 0.61), ("bs", 0.34), ("sr", 0.05)]})
    whole = replace(HR_ENTRY, detect={"osd_scripts": ["Latin"], "langid": {"codes": ["hr"], "close_group": ["bs", "sr"]}})
    split = [LangWindow(0, "hr", 0.6), LangWindow(120, "bs", 0.6), LangWindow(240, "hr", 0.6)]
    p = run([sig(0, stub_lines("hr", 360), windows=split)], index=FakeIndex(1, (EN_ENTRY, whole)))
    assert p.confident and p.pages[0].pack == "hr" and p.pages[0].languages[0] == LangCall("hr", 0.61, "langid", 3)
    alone = run([sig(0, stub_lines("hr", 360), windows=split)])
    assert not alone.confident and alone.pages[0].languages[0].tag == "hr"
    assert [(c.tag, c.pack) for c in alone.candidates] == [("hr", "hr"), ("bs", None), ("sr", None), ("en", "en")]
    off = [LangWindow(0, "hr", 0.6), LangWindow(120, "sl", 0.6), LangWindow(240, "sl", 0.6)]
    assert run([sig(0, stub_lines("hr", 360), windows=off)], index=FakeIndex(1, (EN_ENTRY, whole))).pages[0].pack is None


def test_the_real_identifier_on_made_up_croatian_with_a_covering_pack():
    hr = ["Primjer računa sadrži sedam stavki uzorka i ukupno četiri tisuće jedinica, a tvrtka će poslati narudžbu prije petka.",
          "Molimo vas da potpisanu kopiju vratite do kraja mjeseca kako bismo mogli brzo obraditi vaš zahtjev i poslati robu."]
    whole = replace(HR_ENTRY, detect={"osd_scripts": ["Latin"], "langid": {"codes": ["hr"], "close_group": ["bs", "sr"]}})
    p = run([sig(0, hr)], index=FakeIndex(1, (EN_ENTRY, whole)))
    assert p.confident and p.pages[0].pack == "hr" and p.pages[0].composition == "language"


@pytest.mark.parametrize("prob,decided", [(0.8, True), (0.79, False)])
def test_the_langid_probability_bar_is_inclusive(stub_ident, prob, decided):
    stub_ident({"es": [("es", prob), ("de", round(1.0 - prob, 2))]})
    pp = run([sig(0, stub_lines("es"))]).pages[0]
    assert (pp.pack == "es") is decided
    assert pp.composition == ("language" if decided else "undecided")


def test_an_unsure_page_offers_the_identifiers_codes(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3), ("it", 0.1), ("pt", 0.04)]})
    p = run([sig(0, stub_lines("mix"))])
    assert p.pages[0].pack is None and p.pages[0].composition == "undecided"
    assert [(c.tag, c.pack) for c in p.candidates] == [("es", "es"), ("de", "de"), ("it", None), ("en", "en")]
    assert p.unknown_language is None


def test_window_agreement_is_needed(stub_ident):
    stub_ident({"es": [("es", 0.95)], "de": [("de", 0.95)]})
    es, de = stub_lines("es", 120), stub_lines("de", 120)
    windows = [LangWindow(0, "es", 0.95), LangWindow(120, "es", 0.95), LangWindow(240, "de", 0.95)]
    assert run([sig(0, es + de, windows=windows)]).pages[0].pack == "es"            # 2 of 3 agree
    windows = [LangWindow(0, "es", 0.95), LangWindow(120, "de", 0.95), LangWindow(240, "de", 0.95),
               LangWindow(360, "fr", 0.95)]
    assert run([sig(0, es + de, windows=windows)]).pages[0].pack is None           # 1 of 4 agree


def test_a_second_language_needs_three_windows_and_a_fifth(stub_ident):
    stub_ident({"de": [("de", 0.95)]})
    def w(codes): return [LangWindow(120 * i, c, 0.9) for i, c in enumerate(codes)]
    lines = stub_lines("de", 1200)
    two = run([sig(0, lines, windows=w(["de"] * 8 + ["hu"] * 2))]).pages[0]
    assert [c.tag for c in two.languages] == ["de"] and two.warnings == []
    three = run([sig(0, lines, windows=w(["de"] * 9 + ["hu"] * 3))]).pages[0]
    assert [c.tag for c in three.languages] == ["de", "hu"] and three.warnings == ["mixed_language_page"]
    thin = run([sig(0, lines, windows=w(["de"] * 13 + ["hu"] * 3))]).pages[0]       # 3 of 16: under a fifth
    assert [c.tag for c in thin.languages] == ["de"]


# --------------------------------------------------------------------------- --lang and --force-lang

def test_lang_sets_only_unsure_pages_and_force_lang_sets_all(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)], "*": [("hu", 0.99)]})
    unsure = stub_lines("mix") + ["the and of to the"]          # English words keep it from inheriting Hungarian
    signals = [sig(0, HU), sig(1, unsure), sig(2, SPARSE, windows=[])]
    detected = run(signals)
    assert packs(detected) == ["hu", None, "hu"] and not detected.confident
    hinted = run(signals, lang=["de"])
    assert packs(hinted) == ["hu", "de", "hu"]          # the sparse page took the document's language, so it is sure
    assert hinted.pages[1].composition == "hint" and hinted.pages[1].languages == [LangCall("de", 1.0, "hint", 0)]
    assert hinted.confident
    forced = run(signals, force_lang=["de"])
    assert packs(forced) == ["de", "de", "de"] and {pp.composition for pp in forced.pages} == {"forced"}
    assert forced.pages[0].languages == [LangCall("de", 1.0, "forced", 0)]


def test_lang_reaches_base_pack_pages_and_none_gives_no_layer():
    alone = [sig(0, SPARSE, windows=[])]
    assert run(alone).pages[0].composition == "base"
    hinted = run(alone, lang=["hu"])
    assert packs(hinted) == ["hu"] and hinted.pages[0].composition == "hint"
    inherited = run([sig(0, EN), sig(1, SPARSE, windows=[])], lang=["hu"])
    assert packs(inherited) == ["en", "en"] and inherited.pages[1].composition == "document"
    none = run([sig(0, FR), sig(1, EN)], lang=["none"])
    assert packs(none) == [None, "en"] and none.pages[0].composition == "none" and none.confident
    assert run([sig(0, EN)], force_lang=["none"]).pages[0].pack is None


def test_lang_never_reaches_a_confident_page_whose_pack_is_blocked():
    p = run([thai_sig(0), sig(1, FR)], installed={"en", "hu"}, fetch="never", lang=["hu"])
    assert [(pp.pack, pp.composition) for pp in p.pages] == [("th", "script"), ("hu", "hint")]
    assert not p.confident and [c.tag for c in p.candidates] == ["th", "en"]           # Thai can't be fetched
    offline = run([thai_sig(0)], installed={"en"}, fetch="never", lang=["en"])
    assert not offline.confident and offline.pages[0].pack == "th"


def test_apply_lang_approves_that_one_download():
    first = run([sig(0, HU)], installed={"en"}, fetch="never")
    assert not first.confident and first.candidates[0].tag == "hu"
    again = plan([sig(0, HU)], INDEX, lang=["hu"], force_lang=(), installed={"en"}, fetch="never", approved={"hu"})
    assert again.confident and again.pages[0].pack == "hu" and again.packs_needed == ["hu"]
    assert "hu" not in again.unavailable
    th = plan([thai_sig(0)], INDEX, lang=["th"], force_lang=(), installed={"en"}, fetch="never", approved={"th"})
    assert th.confident and th.pages[0].composition == "script"
    # the request covers that pack only: a Thai page in the same document stays offline
    both = plan([sig(0, HU), thai_sig(1)], INDEX, lang=["hu"], force_lang=(), installed={"en"}, fetch="never",
                approved={"hu"})
    assert not both.confident and [pp.pack for pp in both.pages] == ["hu", "th"] and both.candidates[0].tag == "th"
    with pytest.raises(ValueError):
        plan([sig(0, HU)], INDEX, lang=(), force_lang=(), installed={"en"}, fetch="never", approved={"zz"})


def test_apply_lang_reaches_the_pages_of_a_pack_it_turned_away_from():
    en = plan([thai_sig(0), sig(1, EN)], INDEX, lang=["en"], force_lang=(), installed={"en"}, fetch="never",
              declined={"th"})
    assert en.confident and [(pp.pack, pp.composition) for pp in en.pages] == [("en", "hint"), ("en", "language")]
    assert "Thai" in en.pages[0].reason                                   # the reader can't read the page's script
    none = plan([thai_sig(0)], INDEX, lang=["none"], force_lang=(), installed={"en"}, fetch="never", declined={"th"})
    assert none.confident and none.pages[0].composition == "none"
    kept = plan([thai_sig(0)], INDEX, lang=["en"], force_lang=(), installed={"en", "th"}, fetch="never", declined={"th"})
    assert kept.pages[0].pack == "th"                                     # installed: there was nothing to decline
    forced = run([sig(0, EN)], installed={"en"}, fetch="never", force_lang=["de"])
    assert not forced.confident and [c.tag for c in forced.candidates] == ["de", "en"]
    later = plan([sig(0, EN)], INDEX, lang=["en"], force_lang=["de"], installed={"en"}, fetch="never",
                    declined={"de"})
    assert later.confident and later.pages[0].pack == "en"


def test_force_lang_wins_over_lang():
    p = run([thai_sig(0), sig(1, FR)], force_lang=["th"], lang=["en"], fetch="never", installed={"en"})
    assert {pp.composition for pp in p.pages} == {"forced"} and packs(p) == ["th", "th"]
    assert not p.confident and p.candidates[0].tag == "th"


def test_several_codes_keep_the_first_as_reader():
    p = run([sig(0, FR)], lang=["hu,de"])
    assert p.pages[0].pack == "hu" and [c.tag for c in p.pages[0].languages] == ["hu", "de"]
    f = run([sig(0, EN)], force_lang=["DE", " hu "])
    assert f.pages[0].pack == "de" and [c.tag for c in f.pages[0].languages] == ["de", "hu"]


def test_every_pack_backed_candidate_is_reachable_through_lang():
    ru = replace(entry("ru", scripts=("Cyrillic",), detect_block={"osd_scripts": ["Cyrillic"], "probe": False}),
                 langid_codes=())                                   # detect.langid is optional in a manifest
    idx = FakeIndex(1, (EN_ENTRY, HU_ENTRY, ru))
    cyr = sig(0, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])
    offline = run([cyr, sig(1, HU)], index=idx, installed={"en"}, fetch="never")
    assert sorted(c.tag for c in offline.candidates if c.pack is not None) == ["en", "hu", "ru"]
    for c in offline.candidates:
        if c.pack is None:
            continue
        p = plan([cyr, sig(1, HU)], idx, lang=[c.tag], force_lang=(), installed={"en"}, fetch="never",
                 approved={c.pack} - {"en"})
        assert all(pp.pack is not None for pp in p.pages)
    hinted = run([sig(0, FR)], index=idx, lang=["ru"])
    assert hinted.pages[0].pack == "ru"


@pytest.mark.parametrize("kw,code", [
    ({"lang": ["xx"]}, "unknown_language"),
    ({"force_lang": ["fr"]}, "unknown_language"),
    ({"lang": ["none", "hu"]}, "lang_none_with_codes"),
    ({"lang": ["english"]}, "bad_language_code"),
    ({"lang": ["../x"]}, "bad_language_code"),
    ({"fetch": "sometimes"}, "bad_fetch"),
    ({"fetch": "ask"}, "bad_fetch"),                              # §12.3: the ask mode is gone
])
def test_bad_hints_are_usage_errors(kw, code):
    with pytest.raises(UsageError) as e:
        run([sig(0, EN)], **kw)
    assert e.value.code == code
    assert "../x" not in str(e.value)


def test_the_forced_reason_names_a_script_the_pack_cannot_read():
    p = run([thai_sig(0)], force_lang=["hu"])
    assert p.pages[0].pack == "hu" and "Thai" in p.pages[0].reason


# --------------------------------------------------------------------------- outcome

def test_no_pages_is_a_confident_empty_plan():
    assert run([]) == LanguagePlan([], [], True, [], None, [])


def test_duplicate_page_indexes_are_refused():
    with pytest.raises(ValueError):
        run([sig(0, EN), sig(0, EN)])


def test_the_plan_is_deterministic_and_holds_no_document_text():
    signals = [sig(0, HU), sig(1, FR), sig(2, SPARSE, windows=[]), thai_sig(3)]
    a = run(signals, installed={"en"}, fetch="never")
    b = run(signals, installed={"en"}, fetch="never")
    assert a == b
    text = json.dumps([pp.reason for pp in a.pages] + [c.evidence for c in a.candidates], ensure_ascii=False)
    for line in HU + FR + SPARSE:
        for word in line.split():
            if len(word) > 6:
                assert word not in text


def test_a_mixed_document_needs_every_pack_it_names():
    p = run([sig(0, HU), thai_sig(1), sig(2, ES), sig(3, EN)])
    assert packs(p) == ["hu", "th", "es", "en"]
    assert p.packs_needed == ["en", "es", "hu", "th"] and p.confident


def test_the_real_identifier_reads_made_up_spanish_end_to_end():
    p = run([sig(0, ES)], installed={"en"})
    assert p.pages[0].pack == "es" and p.pages[0].languages[0].by == "langid" and p.packs_needed == ["es"]


# --------------------------------------------------------------------------- resolve_unsure: nothing is asked (§12.3)

def resolved(signals: Sequence[PageSignals], **kw: Any) -> LanguagePlan:
    index = kw.pop("index", INDEX)
    p = langplan.resolve_unsure(run(signals, index=index, **kw), index)
    check_codes(p)
    return p


def offers(p: LanguagePlan) -> list[tuple[str, list[int], str, str, str | None, list[str]]]:
    return [(o.reason, o.pages, o.pack, o.read_as, o.detected, [c.tag for c in o.options]) for o in p.offers]


def test_an_unsure_page_is_read_with_its_strongest_candidate(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3), ("it", 0.1), ("pt", 0.04)]})
    p = resolved([sig(0, stub_lines("mix"))])
    pp = p.pages[0]
    assert (pp.pack, pp.composition, pp.confident) == ("es", "strongest", False)
    assert pp.languages == [LangCall("es", 0.55, "langid", 0)] and pp.warnings == ["language_uncertain"]
    assert [c.tag for c in pp.candidates] == ["es", "de", "it", "en"]
    assert offers(p) == [("uncertain", [0], "es", "es", None, ["de", "en"])]     # "it" has no pack to offer
    assert p.offers[0].kind == "language" and p.offers[0].apply_flag == "--lang"
    assert p.resolved and not p.confident and p.packs_needed == ["es"] and p.warnings == ["language_uncertain"]
    assert "es" in pp.reason


def test_an_english_leaning_unsure_page_is_read_as_english(stub_ident):
    stub_ident({"mix": [("en", 0.6), ("de", 0.3)]})
    p = resolved([sig(0, stub_lines("mix"))])
    assert [c.tag for c in p.pages[0].candidates] == ["en", "de"]       # the base pack keeps the rank it earned
    assert p.pages[0].pack == "en" and p.pages[0].composition == "strongest"
    assert offers(p) == [("uncertain", [0], "en", "en", None, ["de"])]


def test_a_close_group_takes_its_strongest_member_with_a_pack(stub_ident):
    stub_ident({"hr": [("hr", 0.9), ("bs", 0.07), ("sr", 0.03)], "bs": [("bs", 0.95), ("hr", 0.05)]})
    p = resolved([sig(0, stub_lines("hr"))])
    assert p.pages[0].pack == "hr" and p.pages[0].warnings == ["language_uncertain"]
    bs = resolved([sig(0, stub_lines("bs"))])                       # Bosnian has no pack; Croatian does
    pp = bs.pages[0]
    assert [(c.tag, c.pack) for c in pp.candidates][0] == ("bs", None)
    assert pp.pack == "hr" and pp.composition == "strongest" and pp.languages[0].tag == "bs"
    assert pp.warnings == ["language_uncertain"]
    assert offers(bs) == [("uncertain", [0], "hr", "hr", None, ["en"])]


def test_a_close_group_with_no_member_pack_is_read_by_the_base_pack(stub_ident):
    stub_ident({"bs": [("bs", 0.95), ("hr", 0.05)]})
    idx = FakeIndex(1, (EN_ENTRY, HU_ENTRY))
    p = resolved([sig(0, stub_lines("bs"))], index=idx)
    pp = p.pages[0]
    assert (pp.pack, pp.composition) == ("en", "no_pack")
    assert pp.warnings == ["language_uncertain", "no_pack_for_language"] and p.offers == []
    assert pp.detected == "bs" and "language bs" in pp.reason


def test_a_language_with_no_pack_is_read_by_the_base_pack():
    p = resolved([sig(0, FR)])
    pp = p.pages[0]
    assert (pp.pack, pp.composition, pp.unsure, pp.detected) == ("en", "no_pack", "no_pack", "fr")
    assert pp.languages[0].tag == "fr" and pp.warnings == ["no_pack_for_language"]   # sure of it: not uncertain
    assert p.offers == [] and p.unknown_language == "fr" and p.packs_needed == ["en"]
    assert p.warnings == ["no_pack_for_language"] and "language fr" in pp.reason
    tr = resolved([sig(0, TR)])
    assert tr.pages[0].pack == "en" and tr.pages[0].detected == "tr" and tr.pages[0].pack != "hu"


def test_a_script_with_no_pack_is_read_by_the_base_pack():
    p = resolved([sig(0, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])])
    pp = p.pages[0]
    assert (pp.pack, pp.composition, pp.detected) == ("en", "no_pack", "Cyrillic")
    assert pp.warnings == ["no_pack_for_language"] and "Cyrillic script" in pp.reason
    assert p.unknown_language == "Cyrillic" and p.offers == []


def test_a_strongest_candidate_without_a_pack_is_read_by_the_base_pack_and_offers_the_rest(stub_ident):
    stub_ident({"mix": [("it", 0.5), ("es", 0.3), ("de", 0.15)]})
    p = resolved([sig(0, stub_lines("mix"))])
    pp = p.pages[0]
    assert (pp.pack, pp.composition) == ("en", "no_pack")
    assert pp.warnings == ["language_uncertain", "no_pack_for_language"] and pp.detected == "it"
    assert offers(p) == [("no_pack", [0], "en", "en", "it", ["es", "de"])]


def test_fetch_auto_needs_no_resolving():
    p = resolved([thai_sig(0), sig(1, HU)], installed={"en"})
    assert packs(p) == ["th", "hu"] and p.offers == [] and p.warnings == [] and p.confident and p.resolved
    assert p.packs_needed == ["hu", "th"]                            # both fetched, without asking


def test_offline_a_pack_that_isnt_installed_is_read_by_the_base_pack():
    p = resolved([thai_sig(0), sig(1, EN)], installed={"en"}, fetch="never")
    th = p.pages[0]
    assert (th.pack, th.composition, th.confident) == ("en", "offline", True)
    assert th.languages == [LangCall("th", 1.0, "pack_script", 0)] and th.warnings == ["pack_not_installed"]
    assert offers(p) == [("not_installed", [0], "en", "en", None, ["th"])]
    assert p.offers[0].options[0].download_bytes == 11_900_000
    assert p.packs_needed == ["en"] and p.warnings == ["pack_not_installed"]
    assert resolved([thai_sig(0)], installed={"en", "th"}, fetch="never").pages[0].pack == "th"


def test_offline_hinted_and_forced_pages_fall_back_too():
    hinted = resolved([sig(0, FR)], installed={"en"}, fetch="never", lang=["hu"])
    assert (hinted.pages[0].pack, hinted.pages[0].composition) == ("en", "offline")
    forced = resolved([sig(0, EN)], installed={"en"}, fetch="never", force_lang=["de"])
    assert forced.pages[0].pack == "en" and offers(forced) == [("not_installed", [0], "en", "en", None, ["de"])]


def test_offline_an_unsure_pages_strongest_pack_falls_back_to_the_base_pack(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)]})
    p = resolved([sig(0, stub_lines("mix"))], installed={"en", "de"}, fetch="never")
    pp = p.pages[0]
    assert (pp.pack, pp.composition) == ("en", "offline")
    assert pp.warnings == ["language_uncertain", "pack_not_installed"]
    assert offers(p) == [("not_installed", [0], "en", "en", None, ["es", "de"])]
    assert [c.download_bytes for c in p.offers[0].options] == [13_570_187, 0]


def test_pages_settled_the_same_way_share_one_offer(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)]})
    p = resolved([sig(0, stub_lines("mix")), sig(1, EN), sig(2, stub_lines("mix"))])
    assert packs(p) == ["es", "en", "es"]
    assert offers(p) == [("uncertain", [0, 2], "es", "es", None, ["de", "en"])]
    offline = resolved([thai_sig(0), thai_sig(1)], installed={"en"}, fetch="never")
    assert offers(offline) == [("not_installed", [0, 1], "en", "en", None, ["th"])]


def test_hint_force_and_none_pages_are_left_as_they_are(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)]})
    hinted = resolved([sig(0, stub_lines("mix"))], lang=["de"])
    assert hinted.pages[0].composition == "hint" and hinted.offers == [] and hinted.warnings == []
    none = resolved([sig(0, stub_lines("mix"))], lang=["none"])
    assert packs(none) == [None] and none.pages[0].composition == "none" and none.offers == []
    forced = resolved([sig(0, stub_lines("mix"))], force_lang=["none"])
    assert packs(forced) == [None] and forced.pages[0].composition == "none"


def test_a_lang_none_page_keeps_no_layer_beside_a_resolved_page():
    none = resolved([sig(0, FR), sig(1, EN)], lang=["none"])
    assert packs(none) == [None, "en"] and none.pages[0].composition == "none" and none.offers == []


def test_resolving_twice_changes_nothing(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)]})
    cyr = sig(2, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])
    once = resolved([sig(0, stub_lines("mix")), thai_sig(1), cyr], installed={"en", "es"}, fetch="never")
    assert langplan.resolve_unsure(once, INDEX) == once
    assert [pp.composition for pp in once.pages] == ["strongest", "offline", "no_pack"]


def test_a_run_never_leaves_a_page_undecided(stub_ident):
    stub_ident({"hr": [("hr", 0.9), ("bs", 0.07), ("sr", 0.03)], "mix": [("it", 0.5), ("es", 0.3)],
                "*": [("en", 0.99)]})
    cases = [[sig(0, FR)], [sig(0, TR)], [sig(0, stub_lines("hr"))], [sig(0, stub_lines("mix"))],
             [sig(0, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])],
             [thai_sig(0), sig(1, SPARSE, script=UNDECIDED, windows=[])],
             [sig(0, EN, mixed=MixedScript(True, "Thai", 2, 12.0))]]
    for signals in cases:
        for fetch, installed in (("auto", {"en"}), ("never", {"en"}), ("never", ALL)):
            p = resolved(signals, fetch=fetch, installed=installed)
            for pp in p.pages:
                assert pp.pack is not None and pp.composition != "undecided"
                assert fetch == "auto" or pp.pack in installed
            for o in p.offers:
                assert o.options and all(OPTION_ID_RE.fullmatch(c.tag) and c.pack for c in o.options)
                assert o.reason in langplan.OFFER_REASONS and o.pack not in [c.pack for c in o.options]
        # every pack but the base one failed this run: still nothing left undecided or waiting
        failed = ALL - {"en"}
        p = langplan.resolve_unsure(run(signals, installed={"en"}), INDEX, broken=failed)
        check_codes(p)
        assert all(pp.pack in ("en", None) and pp.composition != "undecided" for pp in p.pages)
        assert langplan.resolve_unsure(resolved(signals, installed={"en"}), INDEX, broken=failed) == p
        for o in p.offers:
            assert o.reason in langplan.OFFER_REASONS and o.pack == "en" and o.options


def test_a_plan_without_page_candidates_uses_the_plans_own():
    page = langplan.PagePlan(0, None, "undecided", [], False, "the language identifier wasn't sure about this page")
    lp = LanguagePlan([page], [], False, [Candidate("de", "de", "made up", 0, 0.5), Candidate("en", "en", "base", 0)],
                      None, [])
    p = langplan.resolve_unsure(lp, INDEX)
    assert p.pages[0].pack == "de" and offers(p) == [("uncertain", [0], "de", "de", None, ["en"])]
    with pytest.raises(ValueError):
        langplan.resolve_unsure(replace(lp, unavailable=["en"]), INDEX)


def test_the_resolve_codes_are_closed_record_codes():
    assert set(langplan.RESOLVE_WARNINGS) <= WARNING_CODES


def test_every_offer_builds_a_record_language_offer(stub_ident):
    from pdfintake import record
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)], "it": [("it", 0.5), ("es", 0.3)]})
    p = resolved([sig(0, stub_lines("mix")), sig(1, stub_lines("it")), thai_sig(2)], installed={"en", "de"},
                 fetch="never")
    assert {o.reason for o in p.offers} == {"not_installed", "no_pack"}
    for o in p.offers:
        built = record.language_offer(used=o.read_as, others=[(c.tag, c.download_bytes) for c in o.options],
                                      pages=[i + 1 for i in o.pages], reason=o.reason, detected=o.detected)
        assert built.apply_flag == o.apply_flag and [x.id for x in built.options] == [c.tag for c in o.options]
        if o.reason != "uncertain":
            assert "likeliest" not in built.summary


def test_every_no_pack_page_builds_a_record_language_warning(stub_ident):
    """N16-R2: what langplan puts in ``detected`` is always something the record can word."""
    from pdfintake import record
    seen: list[str | None] = []

    def check(cases: Sequence[Sequence[PageSignals]]) -> None:
        for signals in cases:
            for index in (INDEX, FakeIndex(1, (EN_ENTRY, HU_ENTRY))):
                p = resolved(signals, index=index)
                for pp in p.pages:
                    if "no_pack_for_language" not in pp.warnings:
                        continue
                    seen.append(pp.detected)
                    msg = record.language_warning("no_pack_for_language", language=pp.detected, page=pp.index + 1)
                    assert msg.code == "no_pack_for_language"
                for o in p.offers:
                    if o.reason == "no_pack":
                        record.language_warning("no_pack_for_language", language=o.detected)

    check([[sig(0, FR)], [sig(0, TR)],                                          # the real identifier
           [sig(0, ["xyz"], script=ScriptCall("Cyrillic", 22.0, "osd", 15.0), windows=[])],
           [sig(0, ["xyz"], script=ScriptCall("cyr-1", 22.0, "osd", 15.0), windows=[])],  # not a plain name
           [sig(0, ["xyz"], script=ScriptCall("Ethiopic", 22.0, "osd", 15.0), windows=[])]])
    stub_ident({"mix": [("it", 0.5), ("es", 0.3)], "bs": [("bs", 0.95), ("hr", 0.05)]})
    check([[sig(0, stub_lines("mix"))], [sig(0, stub_lines("bs"))]])
    assert {"fr", "tr", "it", "bs", "Cyrillic", "Ethiopic", None} <= set(seen)
    unsafe = resolved([sig(0, ["xyz"], script=ScriptCall("cyr-1", 22.0, "osd", 15.0), windows=[])])
    assert unsafe.pages[0].detected is None and "cyr-1" not in unsafe.pages[0].reason


# --------------------------------------------------------------------------- resolve_unsure: packs that failed (N16-R1)

def test_a_broken_pack_is_replaced_by_the_base_pack_on_a_confident_page():
    p = langplan.resolve_unsure(run([thai_sig(0), sig(1, EN)], installed={"en"}), INDEX, broken={"th"})
    check_codes(p)
    th = p.pages[0]
    assert (th.pack, th.composition, th.via, th.confident) == ("en", "broken", "script", True)
    assert th.warnings == ["pack_broken"] and "pack th couldn't be used this run" in th.reason
    assert th.languages == [LangCall("th", 1.0, "pack_script", 0)]                  # what the page is in
    assert offers(p) == [("broken", [0], "en", "en", None, ["th"])]                # read it again later
    assert p.offers[0].options[0].download_bytes == 11_900_000
    assert p.packs_needed == ["en"] and p.warnings == ["pack_broken"] and p.pages[1].via is None


def test_an_unsure_page_whose_strongest_pack_is_broken_is_read_by_the_base_pack(stub_ident):
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)]})
    p = resolved([sig(0, stub_lines("mix"))])
    once = langplan.resolve_unsure(run([sig(0, stub_lines("mix"))]), INDEX, broken={"es"})
    check_codes(once)
    pp = once.pages[0]
    assert (pp.pack, pp.composition, pp.via) == ("en", "broken", "undecided")
    assert pp.warnings == ["language_uncertain", "pack_broken"]
    assert offers(once) == [("broken", [0], "en", "en", None, ["es", "de"])]
    # flow can resolve first, fetch, then resolve again with what failed: the same plan comes out
    assert langplan.resolve_unsure(p, INDEX, broken={"es"}) == once
    assert langplan.resolve_unsure(once, INDEX, broken={"es"}) == once
    # a broken pack the page wasn't planned with changes nothing
    assert langplan.resolve_unsure(p, INDEX, broken={"th"}) == p


def test_a_broken_pack_reaches_hinted_and_forced_pages_and_keeps_where_they_came_from():
    hinted = langplan.resolve_unsure(run([sig(0, FR)], lang=["hu"]), INDEX, broken={"hu"})
    assert (hinted.pages[0].pack, hinted.pages[0].composition, hinted.pages[0].via) == ("en", "broken", "hint")
    forced = langplan.resolve_unsure(run([sig(0, EN), sig(1, EN)], force_lang=["de"]), INDEX, broken={"de"})
    assert [(pp.pack, pp.via) for pp in forced.pages] == [("en", "forced"), ("en", "forced")]
    assert offers(forced) == [("broken", [0, 1], "en", "en", None, ["de"])]


def test_a_pack_both_offline_and_broken_is_not_installed():
    p = langplan.resolve_unsure(run([thai_sig(0)], installed={"en"}, fetch="never"), INDEX, broken={"th"})
    assert p.pages[0].warnings == ["pack_not_installed"] and p.pages[0].composition == "offline"


def test_the_base_pack_cannot_be_broken():
    with pytest.raises(ValueError):
        langplan.resolve_unsure(run([sig(0, EN)]), INDEX, broken={"en"})


def test_a_broken_pack_warning_builds_a_record_warning():
    from pdfintake import record
    p = langplan.resolve_unsure(run([thai_sig(0)]), INDEX, broken={"th"})
    assert record.warning("pack_broken", p.pages[0].index + 1).code == "pack_broken"


# --------------------------------------------------------------------------- resolve_unsure: signature and via

def test_resolve_unsure_takes_the_designs_parameter_names():
    lp = run([sig(0, FR)])
    assert langplan.resolve_unsure(plan=lp, index=INDEX) == langplan.resolve_unsure(lp, INDEX)


def test_a_changed_page_keeps_the_composition_plan_gave_it(stub_ident):
    offline = resolved([sig(0, FR), sig(1, EN), thai_sig(2)], installed={"en"}, fetch="never", lang=["hu"])
    assert [(pp.composition, pp.via) for pp in offline.pages] == [("offline", "hint"), ("language", None),
                                                                  ("offline", "script")]
    forced = resolved([sig(0, EN)], installed={"en"}, fetch="never", force_lang=["de"])
    assert (forced.pages[0].composition, forced.pages[0].via) == ("offline", "forced")
    stub_ident({"mix": [("es", 0.55), ("de", 0.3)]})
    unsure = resolved([sig(0, stub_lines("mix"))])
    assert (unsure.pages[0].composition, unsure.pages[0].via) == ("strongest", "undecided")


def test_resolving_holds_no_document_text():
    p = resolved([sig(0, HU), sig(1, FR), sig(2, SPARSE, windows=[]), thai_sig(3)], installed={"en"}, fetch="never")
    text = json.dumps([pp.reason for pp in p.pages] + [c.evidence for o in p.offers for c in o.options],
                      ensure_ascii=False)
    for line in HU + FR + SPARSE:
        for word in line.split():
            if len(word) > 6:
                assert word not in text


# --------------------------------------------------------------------------- calibration

@pytest.fixture
def calib_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "calibration.json"
    monkeypatch.setattr(langplan, "_calibration_file", lambda: path)
    return path


def test_the_shipped_null_calibration_turns_the_trigger_off(calib_file: Path):
    calib_file.write_text('{"mixed_script": null}', encoding="utf-8")
    assert langplan.load_calibration() is None


def test_the_calibration_that_ships_is_null_and_turns_the_trigger_off():
    # Unpatched: reads the ocr/calibration.json in the package. It ships as {"mixed_script": null}
    # until the kit has measured the trigger (decided 2026-11-05, §11.2).
    from importlib import resources

    shipped = resources.files("pdfintake.ocr").joinpath("calibration.json")
    assert shipped.is_file(), "ocr/calibration.json is missing from the package"
    assert json.loads(shipped.read_text(encoding="utf-8")) == {"mixed_script": None}
    assert langplan._calibration_file() == shipped
    assert langplan.load_calibration() is None


def test_a_measured_calibration_loads(calib_file: Path):
    data = {"mixed_script": {"block_max_conf": 0.45, "osd_min_conf": 6, "min_block_px": 40, "max_blocks": 4,
                             "measured": "kit 1.0; public-v1@" + "ab" * 32}}
    calib_file.write_text(json.dumps(data), encoding="utf-8")
    c = langplan.load_calibration()
    assert c == Calibration(0.45, 6.0, 40, 4, data["mixed_script"]["measured"])
    assert isinstance(c.osd_min_conf, float)


def test_a_missing_calibration_file_is_a_setup_error(calib_file: Path):
    with pytest.raises(SetupError) as e:
        langplan.load_calibration()
    assert e.value.code == "calibration_missing"


@pytest.mark.parametrize("content,needle", [
    ("not json", "valid JSON"),
    ("[]", "JSON object"),
    ("{}", "'mixed_script' is missing"),
    ('{"mixed_script": null, "extra": 1}', "unknown keys"),
    ('{"mixed_script": 3}', "null or an object"),
    ('{"mixed_script": {"block_max_conf": 50, "osd_min_conf": 5, "min_block_px": 40, "max_blocks": 4, "measured": "x"}}',
     "block_max_conf"),
    ('{"mixed_script": {"block_max_conf": 0.5, "osd_min_conf": 5, "min_block_px": 0, "max_blocks": 4, "measured": "x"}}',
     "min_block_px"),
    ('{"mixed_script": {"block_max_conf": 0.5, "osd_min_conf": 5, "min_block_px": 40, "max_blocks": true, "measured": "x"}}',
     "max_blocks"),
    ('{"mixed_script": {"block_max_conf": 0.5, "osd_min_conf": 5, "min_block_px": 40, "max_blocks": 4}}', "lacks measured"),
    ('{"mixed_script": {"block_max_conf": 0.5, "osd_min_conf": 5, "min_block_px": 40, "max_blocks": 4, "measured": "x", "z": 1}}',
     "unknown keys"),
    ('{"mixed_script": {"block_max_conf": 0.5, "osd_min_conf": 5, "min_block_px": 40, "max_blocks": 4, "measured": "<b>"}}',
     "measured"),
])
def test_a_malformed_calibration_is_a_setup_error(calib_file: Path, content: str, needle: str):
    calib_file.write_text(content, encoding="utf-8")
    with pytest.raises(SetupError) as e:
        langplan.load_calibration()
    assert needle in str(e.value)


def test_a_calibration_with_a_bom_loads_and_odd_json_is_a_setup_error(calib_file: Path):
    calib_file.write_bytes(b"\xef\xbb\xbf" + b'{"mixed_script": null}')
    assert langplan.load_calibration() is None
    real = '{"block_max_conf": 0.45, "osd_min_conf": 6, "min_block_px": 40, "max_blocks": 4, "measured": "x"}'
    calib_file.write_text('{"mixed_script": ' + real + ', "mixed_script": null}', encoding="utf-8")
    with pytest.raises(SetupError) as e:
        langplan.load_calibration()                                     # a repeated key never switches it off
    assert e.value.code == "calibration_invalid" and "repeats" in str(e.value)
    calib_file.write_text('{"mixed_script": ' + "[" * 100_000 + "]" * 100_000 + "}", encoding="utf-8")
    with pytest.raises(SetupError) as e:
        langplan.load_calibration()
    assert e.value.code == "calibration_invalid"


def test_the_thresholds_are_the_designs_starting_values():
    assert (langplan.OSD_NONLATIN_MIN_CONF, langplan.LATIN_ENG_MIN_MEANCONF, langplan.LANGID_MIN_PROB) == (5.0, 60.0, 0.8)
    assert (langplan.LANGID_WINDOW, langplan.DOC_AGREE) == (120, 2 / 3)
    assert (langplan.SECOND_LANG_MIN_WINDOWS, langplan.SECOND_LANG_MIN_SHARE) == (3, 0.2)
    assert ("hr", "bs", "sr") in langplan.CLOSE_GROUPS
