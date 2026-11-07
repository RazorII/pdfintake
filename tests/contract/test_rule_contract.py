"""The pick-rule contract, run against every registered rule and a built-in fake.

Rules are registered explicitly, so their contract parameters are too: ``SCENARIOS`` says, for each
registered (name, version), which reader uses it competes over and how to write its params for a
given set of readers, checker and corroborators. A registered rule with no row here fails, so a new
rule can't skip the contract. A rule whose module isn't built yet is skipped with the reason.

What every rule must do:
* be deterministic, whatever order the reads arrive in;
* with no reader working, keep the checker's read (named, ``fallback=True``) under
  ``fallback="checker"``, and give no layer under ``fallback="none"``;
* with exactly one reader working, keep it, uncorroborated;
* never let a corroborator win while a reader works, and never let anything but a reader win
  under ``fallback="none"``;
* break ties by ``tie_order`` (rules that take one);
* give a non-empty ``reason`` every time;
* refuse unknown parameter keys.
"""

from __future__ import annotations

import importlib.util
import sys
import types as pytypes
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from fixtures import build
from pdfintake.errors import EXIT_SETUP, PackError, SetupError
from pdfintake.ocr import rules
from pdfintake.params import Param, ParamTable, ValidationContext
from pdfintake.types import EngineRead, Pick

TEXT = {
    "Latin": [
        "The example invoice lists seven sample items",
        "with a made-up total of 4521 units and a note",
        "that the sample order ships on a quiet Tuesday",
    ],
    "Thai": ["ข้อความตัวอย่างสำหรับการทดสอบ", "รายการสินค้าตัวอย่างเจ็ดรายการ", "ยอดรวมสมมุติ 4521 หน่วย"],
}
JUNK = {"Latin": ["qzx vvkq jjwp xxqz", "pqqz kkxv"], "Thai": ["ฃฅฆ ฌญฎ", "ฏฐฑ ฒณ"]}
NOT_WORKING = ("failed", "timeout", "unavailable")


@dataclass(frozen=True)
class Scenario:
    readers: tuple[str, ...]
    checker: str
    params: Callable[[Sequence[str], str, Sequence[str], str], dict[str, Any]]   # (readers, checker, corroborators, script)


SCENARIOS: dict[tuple[str, int], Scenario] = {
    ("corroborated_trigrams", 1): Scenario(
        ("vision", "tesseract"), "rapidocr",
        lambda readers, checker, corr, script: {"script": script, "tie_order": list(readers), "corroborators": list(corr)}),
    ("primary_with_floor", 1): Scenario(
        ("tesseract",), "rapidocr",
        lambda readers, checker, corr, script: {"primary": readers[0], "checker": checker}),
    ("single", 1): Scenario(
        ("tesseract",), "rapidocr",
        lambda readers, checker, corr, script: {"reader": readers[0]}),
}


class MostTextRule:
    """The fake member: the working reader with the most characters wins; ties by tie_order."""

    name: ClassVar[str] = "fake_most_text"
    version: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("tie_order", "use_list", required=True, min=1),
        Param("corroborators", "use_list", default=()),
    )
    scripts: ClassVar[frozenset[str] | None] = None

    def pick(self, reads: Mapping[str, EngineRead], params: Mapping[str, Any], *, fallback: str,
             checker_id: str | None) -> Pick:
        order = list(params["tie_order"])
        working = [u for u in order if u in reads and reads[u].status == "ok" and u not in params["corroborators"]]
        scores = {u: float(len(reads[u].text())) for u in working}
        if not working:
            if fallback == "checker" and checker_id in reads and reads[checker_id].status == "ok":
                return Pick(self.name, self.version, checker_id, list(reads[checker_id].lines), scores, True,
                            "no reader worked, so the checker's read was kept")
            return Pick(self.name, self.version, None, [], scores, False, "no reader worked")
        best = max(working, key=lambda u: (scores[u], -order.index(u)))
        return Pick(self.name, self.version, best, list(reads[best].lines), scores, False, "most text")


SCENARIOS_FAKE = Scenario(("alpha", "beta"), "gamma",
                          lambda readers, checker, corr, script: {"tie_order": list(readers), "corroborators": list(corr)})


@dataclass
class Member:
    key: tuple[str, int]
    rule: Any
    scenario: Scenario
    script: str

    def params(self, readers: Sequence[str] | None = None, corroborators: Sequence[str] = ()) -> dict[str, Any]:
        readers = list(readers if readers is not None else self.scenario.readers)
        raw = self.scenario.params(readers, self.scenario.checker, list(corroborators), self.script)
        return self.rule.PARAMS.validate(raw, self.context())

    def context(self) -> ValidationContext:
        return ValidationContext(use_ids=frozenset((*self.scenario.readers, self.scenario.checker)),
                                 asset_keys=frozenset(), asset_kinds={}, scripts=_known_scripts(),
                                 where=f"contract.{self.key[0]}.params")

    def read(self, use_id: str, status: str = "ok", texts: Sequence[str] | None = None) -> EngineRead:
        role = "checker" if use_id == self.scenario.checker else "reader"
        if status != "ok":
            return build.engine_read(use_id, (), role=role, status=status,
                                     error=None if status == "unavailable" else f"made-up {status}")
        return build.engine_read(use_id, texts if texts is not None else TEXT[self.script], role=role)

    def pick(self, reads: Mapping[str, EngineRead], fallback: str = "none", params: dict[str, Any] | None = None) -> Pick:
        return self.rule.pick(dict(reads), params if params is not None else self.params(),
                              fallback=fallback, checker_id=self.scenario.checker)


def _known_scripts() -> frozenset[str]:
    try:
        from pdfintake.ocr.scripts import SCRIPTS
    except ImportError:
        return frozenset({"Latin", "Thai", "Cyrillic", "Greek", "Arabic", "Hebrew", "Han", "Hangul", "Devanagari"})
    return frozenset(SCRIPTS)


def _built(key: tuple[str, int]) -> bool:
    module = rules._REGISTRY[key].partition(":")[0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _script_for(rule: Any) -> str:
    allowed = getattr(rule, "scripts", None)
    if not allowed or "Latin" in allowed:
        return "Latin"
    return sorted(allowed)[0]


@pytest.fixture(params=[*rules.names(), ("fake_most_text", 1)], ids=lambda k: f"{k[0]}-v{k[1]}")
def member(request: pytest.FixtureRequest) -> Member:
    key = request.param
    if key == ("fake_most_text", 1):
        rule = MostTextRule()
        return Member(key, rule, SCENARIOS_FAKE, _script_for(rule))
    assert key in SCENARIOS, f"rule {key[0]} v{key[1]} is registered but has no contract scenario in this file"
    if not _built(key):
        pytest.skip(f"{rules._REGISTRY[key]} is not built yet")
    rule = rules.get(*key)
    return Member(key, rule, SCENARIOS[key], _script_for(rule))


def _all_ok(m: Member) -> dict[str, EngineRead]:
    reads = {u: m.read(u) for u in m.scenario.readers}
    reads[m.scenario.checker] = m.read(m.scenario.checker)
    return reads


def _assert_pick(m: Member, pick: Any, reads: Mapping[str, EngineRead]) -> None:
    assert isinstance(pick, Pick)
    assert (pick.rule, pick.version) == m.key
    assert isinstance(pick.reason, str) and pick.reason.strip(), "every pick gives a reason"
    assert isinstance(pick.fallback, bool)
    assert isinstance(pick.warnings, list)
    unknown = [w for w in pick.warnings if w not in rules.PICK_WARNING_CODES]
    assert not unknown, f"Pick.warnings holds codes a rule may not emit: {unknown}"
    assert set(pick.scores) <= set(reads), f"scores name unknown uses: {set(pick.scores) - set(reads)}"
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pick.scores.values())
    if pick.chosen is None:
        assert pick.lines == []
    else:
        assert pick.chosen in reads
        assert pick.lines == reads[pick.chosen].lines


# --------------------------------------------------------------------------- members


def test_member_declares_the_protocol(member: Member):
    rule = member.rule
    assert isinstance(rule, rules.PickRule)
    assert (rule.name, rule.version) == member.key
    assert isinstance(rule.PARAMS, ParamTable)
    assert rule.scripts is None or (isinstance(rule.scripts, frozenset) and all(isinstance(s, str) for s in rule.scripts))


def test_deterministic_in_any_read_order(member: Member):
    reads = _all_ok(member)
    reads[member.scenario.readers[0]] = member.read(member.scenario.readers[0], texts=TEXT[member.script][:2])
    for fallback in ("checker", "none"):
        a = member.pick(reads, fallback)
        b = member.pick(reads, fallback)
        c = member.pick(dict(reversed(list(reads.items()))), fallback)
        _assert_pick(member, a, reads)
        assert a == b == c


@pytest.mark.parametrize("fallback", ["checker", "none"])
def test_no_reader_working(member: Member, fallback: str):
    readers = member.scenario.readers
    reads = {u: member.read(u, NOT_WORKING[i % 3]) for i, u in enumerate(readers)}
    checker = member.scenario.checker
    reads[checker] = member.read(checker)
    pick = member.pick(reads, fallback)
    _assert_pick(member, pick, reads)
    if fallback == "checker":
        assert pick.chosen == checker and pick.fallback is True
        assert pick.lines == reads[checker].lines
    else:
        assert pick.chosen is None and pick.lines == [], "fallback none gives the page no layer"
    reads[checker] = member.read(checker, "failed")
    pick = member.pick(reads, fallback)
    _assert_pick(member, pick, reads)
    assert pick.chosen is None and pick.lines == []


def test_one_reader_working_is_kept_uncorroborated(member: Member):
    readers = member.scenario.readers
    for keep in readers:
        reads = {u: (member.read(u) if u == keep else member.read(u, "failed")) for u in readers}
        reads[member.scenario.checker] = member.read(member.scenario.checker, "failed")
        for fallback in ("checker", "none"):
            pick = member.pick(reads, fallback)
            _assert_pick(member, pick, reads)
            assert pick.chosen == keep and pick.fallback is False
            assert pick.lines == reads[keep].lines


def test_only_a_reader_wins_under_fallback_none(member: Member):
    checker = member.scenario.checker
    reads = {u: member.read(u, texts=JUNK[member.script]) for u in member.scenario.readers}
    reads[checker] = member.read(checker)
    pick = member.pick(reads, "none")
    _assert_pick(member, pick, reads)
    assert pick.chosen in member.scenario.readers


def test_a_corroborator_never_wins(member: Member):
    if "corroborators" not in member.rule.PARAMS:
        pytest.skip("this rule takes no corroborators")
    checker = member.scenario.checker
    reads = {u: member.read(u, texts=JUNK[member.script]) for u in member.scenario.readers}
    reads[checker] = member.read(checker)            # the best text on the page, but only a corroborator
    for fallback in ("checker", "none"):
        pick = member.pick(reads, fallback, member.params(corroborators=[checker]))
        _assert_pick(member, pick, reads)
        assert pick.chosen in member.scenario.readers, f"the corroborator won under fallback {fallback}"


def test_ties_follow_tie_order(member: Member):
    if "tie_order" not in member.rule.PARAMS:
        pytest.skip("this rule takes no tie_order")
    readers = list(member.scenario.readers)
    if len(readers) < 2:
        pytest.skip("this scenario has one reader")
    reads = {u: member.read(u) for u in readers}
    reads[member.scenario.checker] = member.read(member.scenario.checker, "failed")
    for order in (readers, list(reversed(readers))):
        pick = member.pick(reads, "none", member.params(readers=order))
        _assert_pick(member, pick, reads)
        assert pick.chosen == order[0], f"a tie went to {pick.chosen}, tie_order says {order[0]}"


def test_params_refuse_unknown_keys(member: Member):
    raw = member.scenario.params(list(member.scenario.readers), member.scenario.checker, [], member.script)
    member.rule.PARAMS.validate(raw, member.context())
    with pytest.raises(PackError) as info:
        member.rule.PARAMS.validate({**raw, "made_up_key": 1}, member.context())
    assert "made_up_key" in str(info.value)


# --------------------------------------------------------------------------- the registry


READER_WARNINGS = {"reader_failed", "reader_timeout", "fallback_used", "agreement_low", "no_layer_readers_failed",
                   "vision_language_unsupported"}                       # DESIGN §3.2, the readers group


def test_pick_warning_codes_are_reader_warnings():
    assert rules.PICK_WARNING_CODES == {"agreement_low"}
    assert rules.PICK_WARNING_CODES <= READER_WARNINGS
    try:
        from pdfintake.record import WARNING_CODES
    except ImportError:
        return                                                          # record.py (U2) is not built yet
    assert rules.PICK_WARNING_CODES <= set(WARNING_CODES)


def test_every_registered_rule_has_a_scenario():
    assert set(rules.names()) <= set(SCENARIOS), "add a contract scenario for every registered rule"


def test_registry_names_in_order():
    assert rules.names() == (("corroborated_trigrams", 1), ("primary_with_floor", 1), ("single", 1))


@pytest.mark.parametrize("key", rules.names(), ids=lambda k: f"{k[0]}-v{k[1]}")
def test_registered_rule_resolves_to_itself(key: tuple[str, int]):
    if not _built(key):
        pytest.skip(f"{rules._REGISTRY[key]} is not built yet")
    rule = rules.get(*key)
    assert (rule.name, rule.version) == key
    assert rules.get(*key) is rule


def test_unknown_rule_is_named_precisely():
    with pytest.raises(PackError) as info:
        rules.get("most_votes", 1)
    assert info.value.code == "unknown_rule" and info.value.exit_code == EXIT_SETUP
    assert "'most_votes'" in info.value.message
    for name, version in rules.names():
        assert f"{name} v{version}" in info.value.message


def test_unknown_rule_version_is_named_precisely():
    with pytest.raises(PackError) as info:
        rules.get("single", 7)
    assert info.value.code == "unknown_rule_version"
    assert "'single'" in info.value.message and "version 7" in info.value.message
    assert info.value.detail["known_versions"] == [1]
    with pytest.raises(PackError) as info:
        rules.get("single", True)         # a bool is not a version
    assert info.value.code == "unknown_rule_version"
    for bad in (True, "1", 1.0, None):
        with pytest.raises(PackError) as info:
            rules.get("most_votes", bad)  # an unknown name is unknown_rule, whatever the version
        assert info.value.code == "unknown_rule", bad


@pytest.fixture
def clean_rules(monkeypatch: pytest.MonkeyPatch):
    rules._clear_cache()
    yield monkeypatch
    rules._clear_cache()


def test_override_and_failures(clean_rules: pytest.MonkeyPatch):
    fake = MostTextRule()
    clean_rules.setitem(rules._REGISTRY, ("fake_most_text", 1), fake)
    assert rules.get("fake_most_text", 1) is fake
    mod = pytypes.ModuleType("pdfintake_test_fake_rules")
    mod.RULE = fake
    clean_rules.setitem(sys.modules, mod.__name__, mod)
    clean_rules.setitem(rules._REGISTRY, ("fake_most_text", 1), f"{mod.__name__}:RULE")
    assert rules.get("fake_most_text", 1) is fake
    clean_rules.setitem(rules._REGISTRY, ("fake_most_text", 2), f"{mod.__name__}:RULE")
    with pytest.raises(SetupError) as info:
        rules.get("fake_most_text", 2)
    assert info.value.code == "rule_name_mismatch"
    clean_rules.setitem(rules._REGISTRY, ("fake_gone", 1), "pdfintake_test_no_such_rules:RULE")
    with pytest.raises(SetupError) as info:
        rules.get("fake_gone", 1)
    assert info.value.code == "rule_import_failed" and "pdfintake_test_no_such_rules" in info.value.message
