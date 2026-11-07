"""Tests for pdfintake.record (DESIGN §3.2, §7.1, §12): round trip, validation, exit priority, load's
refusal of anything but a record id, that no document text survives to_json, and that a record
never waits for an answer (offers, not decisions; no exit 10).

Every value here is made up, and every path is under /Users/example/ (an invented account).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import pytest

from fixtures import build
from pdfintake import record
from pdfintake.errors import PdfIntakeError, UsageError
from pdfintake.ocr.rules import PICK_WARNING_CODES
from pdfintake.types import Fill, LangCall, Line, PageOCR, Pick, ScriptCall, Word

SPEC_FIELDS = [
    "record_version", "id", "parent", "tool", "created", "status", "exit", "provenance", "options", "input",
    "output", "refusal", "steps", "verify", "language", "pages", "shrink", "metadata", "smaller", "offers",
    "warnings", "notes", "history",
]
SPEC_WARNINGS = {
    "reader_failed", "reader_timeout", "fallback_used", "agreement_low", "no_layer_readers_failed",
    "vision_language_unsupported", "mixed_language_page", "mixed_script_page", "mixed_script_no_pack",
    "pack_broken", "existing_layer_replaced", "layer_shrank", "existing_layer_kept", "image_text_not_read",
    "garbled_text", "clip_text_kept", "layer_dropped_chars", "orientation_disagree", "not_turned_annotations",
    "input_repaired", "contains_jbig2", "revisions_dropped", "objects_dropped", "render_capped",
    "metadata_format_limit",
    "exporting_a_copy",                                 # §11.1, export only
    "language_uncertain", "no_pack_for_language", "pack_not_installed",                # §12.3
}
SPEC_NOTES = {"input_unchanged", "nothing_written", "vision_not_on_platform", "pack_fetched",
              "mixed_script_unchecked", "rotate_kept",
              "signature_not_carried", "conformance_claim_dropped", "copy_unlocked",       # §10.2, §11.1
              "skipped_no_password",                                                       # §12.1
              "render_scaled_down"}                                                        # §11.2
PASSWORD = "made up record phrase"          # a made-up test password
CANARY = "ZQXJ-canary-Wombat-7781"
HEX64 = "9f" * 32
MTIME_NS = 1_762_300_000 * 10**9          # a made-up modification time


@dataclass
class FakeOptions:              # the RunOptions shape (DESIGN §2.2, flow.py)
    provenance: str | None = "ordinary"
    out: Path | None = None
    outdir: Path | None = None
    lang: tuple[str, ...] = ()
    force_lang: tuple[str, ...] = ()
    fetch: str = "auto"
    smaller: bool = False
    overwrite: bool = False
    redo_ocr: bool = False
    keep_existing_layer: bool = False
    separate: bool = False
    plan_only: bool = False
    workers: int = 2


@dataclass
class FakeSeal:                 # the InputSeal shape (DESIGN §2.2, integrity.py)
    paths: list[Path]
    sha256: list[str]
    sizes: list[int]
    mtimes_ns: list[int] = field(default_factory=list)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(record, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(record, "_engine_versions", lambda: {"tesseract": "5.5.3 (tesserocr 2.12.0)",
                                                             "vision": "unavailable: not_on_platform"})
    return tmp_path


def make(**over: Any) -> record.Record:
    seal = FakeSeal([Path("/Users/example/Desktop/Receipt.pdf")], [HEX64], [4813220], [MTIME_NS])
    rec = record.new([Path("/Users/example/Desktop/Receipt.pdf")], opts=FakeOptions(), seal=seal,
                     platform_key="darwin-arm64")
    for k, v in over.items():
        setattr(rec, k, v)
    return rec


def full_record() -> record.Record:
    rec = make()
    rec.status = "done"
    rec.exit = 0
    rec.input.update(kind="pdf", pages=1)
    rec.output = record.output_entry("/Users/example/Desktop/Receipt (searchable).pdf", "c1" * 32, 4302118)
    rec.steps = [{"step": "refusals", "status": "clear", "detail": {"signed_fields": 0, "pdfa": None}},
                 {"step": "shrink", "status": "ok", "detail": {"before": 4813220, "after": 4302118}}]
    rec.verify = [{"name": "text_retention", "ok": True, "detail": "0.996 document", "page": None}]
    rec.language = {"plan": "confident", "packs": [], "canaries": [], "detection": {}}
    rec.pages = [{"page": 1, "class": "image_only", "dpi": 200.0, "warnings": ["agreement_low"],
                  "fills": {"number": 2}}]
    rec.shrink = {"before": 4813220, "after": 4302118}
    rec.metadata = {"kept": ["Info", "XMP"], "changed": [], "dropped": []}
    rec.smaller = {"offered": True, "saving": {"bytes": 3098807, "pct": 72.0}, "bar": 90, "min_score": 92.4,
                   "applied": False, "written": None}
    rec.offers = [record.smaller_offer(output=rec.output["path"], inputs=rec.input["paths"],
                                       before_bytes=4302118, after_bytes=1203311, saving_pct=72.0,
                                       min_score=92.4, bar=90),
                  record.language_offer(used="hr", others=[("bs", 9_000_000), ("sr", None)], pages=[1])]
    rec.warnings = [record.warning("agreement_low", page=1)]
    rec.notes = [record.note("input_unchanged")]
    rec.history = [{"at": "2026-11-05T10:00:00Z", "flag": "--lang", "answer": "bs",
                    "result": 0, "child": "r-20261105-0a0b0c0d"}]
    return rec


# ------------------------------------------------------------------------- interface and fields

def test_interface_constants():
    assert record.RECORD_VERSION == 1
    assert get_args(record.ApplyFlag) == ("--choice", "--lang")                  # §12.5: no --provenance
    assert record.RECORD_ID_RE.pattern == r"^r-[0-9]{8}-[0-9a-f]{8}$"
    assert record.OPTION_ID_RE.pattern == r"^(smaller|none|[a-z]{2,3})$"
    assert get_args(record.OfferKind) == ("language", "smaller")
    assert record.OFFER_KINDS == ("language", "smaller")
    assert dict(record.APPLY_FLAG_FOR) == {"language": "--lang", "smaller": "--choice"}
    assert [f.name for f in dataclasses.fields(record.Offer)] == ["kind", "apply_flag", "summary", "options"]
    assert [f.name for f in dataclasses.fields(record.Option)] == ["id", "label", "description"]


def test_nothing_of_the_blocking_decision_is_left():
    # §12.1, §12.5: no decision, no provenance or password question, no exit 10 (decided 2026-11-05)
    for name in ("Decision", "DecisionKind", "DECISION_KINDS", "provenance_decision", "password_decision",
                 "smaller_decision", "language_decision", "PROVENANCE_QUESTION", "PASSWORD_QUESTION",
                 "DEFAULT_ANSWER", "EXIT_DECISION"):
        assert not hasattr(record, name), name
    assert "decision" not in record.STATUSES and "decision" not in record.FIELDS
    assert "asked" not in record.LANGUAGE_PLANS and "uncertain" in record.LANGUAGE_PLANS
    assert 10 not in record.EXIT_PRIORITY
    assert "password_wrong" not in record.NOTE_CODES
    for oid in ("ordinary", "counterparty", "evidence", "supply", "decline", "keep"):
        assert not record.OPTION_ID_RE.fullmatch(oid), oid


def test_record_fields_are_the_spec_fields_in_order(cache: Path):
    assert list(record.FIELDS) == SPEC_FIELDS
    assert list(record.to_json(make())) == SPEC_FIELDS


def test_closed_codes_match_the_spec():
    assert record.WARNING_CODES == SPEC_WARNINGS
    assert record.NOTE_CODES == SPEC_NOTES
    assert PICK_WARNING_CODES <= record.WARNING_CODES
    assert set(record.WARNING_MESSAGES) == record.WARNING_CODES
    assert set(record.NOTE_MESSAGES) == record.NOTE_CODES
    with pytest.raises(ValueError):
        record.warning("made_up_code")
    with pytest.raises(ValueError):
        record.note("agreement_low")            # a warning code isn't a note


def test_new_fills_tool_options_and_input(cache: Path):
    rec = make()
    assert record.RECORD_ID_RE.fullmatch(rec.id)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec.created)
    assert rec.id[2:10] == rec.created[:10].replace("-", "")
    assert set(rec.tool) == {"version", "platform", "os", "python", "engines"}
    assert rec.tool["platform"] == "darwin-arm64"
    assert rec.status == "failed" and rec.exit == 1         # never claims success before the run says so
    assert rec.provenance == "ordinary"
    assert rec.options["fetch"] == "auto" and rec.options["lang"] == [] and rec.options["out"] is None
    assert rec.input["seal"] == {"sha256": [HEX64], "sizes": [4813220], "mtimes_ns": [MTIME_NS]}
    assert rec.input["paths"] == [str(Path("/Users/example/Desktop/Receipt.pdf").absolute())]
    assert record.validate(record.to_json(rec)) == []
    ids = {make().id for _ in range(50)}
    assert len(ids) == 50


def test_new_without_seal_and_with_parent(cache: Path):
    rec = record.new([Path("/Users/example/a.jpg"), Path("/Users/example/b.jpg")],
                     opts=FakeOptions(provenance=None, outdir=Path("/Users/example/Out")), seal=None,
                     platform_key="linux-x86_64", parent="r-20261105-1a2b3c4d")
    assert rec.input["seal"] is None and rec.provenance is None and rec.parent == "r-20261105-1a2b3c4d"
    assert rec.options["outdir"] == str(Path("/Users/example/Out"))
    with pytest.raises(UsageError):
        record.new([], opts=FakeOptions(), seal=None, platform_key="linux-x86_64", parent="../x")


def test_engine_versions_never_raise():
    versions = record._engine_versions()
    from pdfintake.ocr import engines

    assert set(versions) == set(engines.names())
    assert all(isinstance(v, str) and v for v in versions.values())


def test_records_live_under_the_cache_root():
    platforms = pytest.importorskip("pdfintake.platforms")
    assert record._records_dir() == platforms.cache_root() / "records"


# ------------------------------------------------------------------------- round trip

def test_round_trip(cache: Path):
    rec = full_record()
    path = record.save(rec)
    assert path == cache / "records" / f"{rec.id}.json"
    assert [p.name for p in (cache / "records").iterdir()] == [f"{rec.id}.json"]     # no partial left
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == record.to_json(rec)
    back = record.load(rec.id)
    assert back == rec
    assert record.to_json(back) == record.to_json(rec)
    assert [type(o) for o in back.offers] == [record.Offer, record.Offer]
    assert isinstance(back.offers[0].options[0], record.Option)
    assert isinstance(back.warnings[0], record.Message)


def test_save_refuses_an_invalid_record_and_writes_nothing(cache: Path):
    rec = make(warnings=[record.Message("not_a_code", "x")])
    with pytest.raises(PdfIntakeError) as err:
        record.save(rec)
    assert err.value.code == "record_invalid"
    assert not (cache / "records").exists() or not any((cache / "records").iterdir())


def test_save_failure_leaves_no_partial(cache: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*a: Any, **k: Any) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(record.os, "replace", boom)
    with pytest.raises(OSError):
        record.save(make())
    assert list((cache / "records").iterdir()) == []


def test_spec_example_validates(cache: Path):
    # The §3.2 example (made-up values), with the fields the example leaves out filled in.
    example = record.to_json(full_record())
    example["pages"] = [{
        "page": 1, "class": "image_only", "dpi": 200,
        "orientation": {"turn_cw": 0, "by": "vote", "action": "none"},
        "script": {"name": "Thai", "conf": 48.2, "by": "osd"}, "pack": "th", "composition": "script",
        "readers": [{"use_id": "vision", "status": "ok", "seconds": 0.4, "score": 299},
                    {"use_id": "rapidocr", "role": "checker", "status": "ok", "seconds": 0.5}],
        "pick": {"rule": "corroborated_trigrams", "version": 1, "chosen": "vision", "fallback": False},
        "fills": {"number": 2, "id": 1},
        "must_survive": {"tone-marks": {"vision": 41, "tesseract": 37, "layer": 41}},
        "retention": 0.998, "warnings": []}]
    assert record.validate(example) == []


# ------------------------------------------------------------------------- validate

def _bad(data: dict[str, Any], fragment: str) -> None:
    problems = record.validate(data)
    assert any(fragment in p for p in problems), problems


@pytest.fixture
def good(cache: Path) -> dict[str, Any]:
    data = record.to_json(full_record())
    assert record.validate(data) == []
    return data


def test_validate_rejects_unknown_and_missing_fields(good: dict[str, Any]):
    d = copy.deepcopy(good); d["surprise"] = 1
    _bad(d, "unknown fields: surprise")
    d = copy.deepcopy(good); del d["history"]
    _bad(d, "missing fields: history")
    d = copy.deepcopy(good); d["decision"] = None                  # the retired field
    _bad(d, "unknown fields: decision")
    d = copy.deepcopy(good); d["offers"][0]["extra"] = "x"
    _bad(d, "offers[0] has unknown fields: extra")
    d = copy.deepcopy(good); d["offers"][0]["options"][0]["recommended"] = True
    _bad(d, "offers[0].options[0] has unknown fields: recommended")
    d = copy.deepcopy(good); d["offers"][0]["question"] = "q?"
    _bad(d, "offers[0] has unknown fields: question")
    d = copy.deepcopy(good); d["warnings"][0]["detail"] = {}
    _bad(d, "warnings[0] has unknown fields: detail")


@pytest.mark.parametrize("bad_id", ["r-2026115-1a2b3c4d", "r-20261105-1A2B3C4D", "r-20261105-1a2b3c4", "x",
                                    "r-20261105-1a2b3c4d\n", "../r-20261105-1a2b3c4d", 12, None])
def test_validate_rejects_a_bad_record_id(good: dict[str, Any], bad_id: Any):
    d = copy.deepcopy(good); d["id"] = bad_id
    _bad(d, "id ")
    d = copy.deepcopy(good); d["parent"] = bad_id if bad_id is not None else 5
    _bad(d, "parent ")


@pytest.mark.parametrize("bad_option", ["kee p", "a/b", "../smaller", "smaller ", "Smaller", "smaller\n", "abcd",
                                        "", 3, "keep", "ordinary", "supply", "decline"])
def test_validate_rejects_a_bad_option_id(good: dict[str, Any], bad_option: Any):
    d = copy.deepcopy(good); d["offers"][0]["options"][0]["id"] = bad_option
    _bad(d, "offers[0].options[0].id")
    d = copy.deepcopy(good); d["offers"][1]["options"][0]["id"] = bad_option
    _bad(d, "offers[1].options[0].id")


def test_validate_rejects_unknown_codes(good: dict[str, Any]):
    d = copy.deepcopy(good); d["warnings"][0]["code"] = "made_up"
    _bad(d, "isn't a known warning code")
    d = copy.deepcopy(good); d["notes"][0]["code"] = "agreement_low"
    _bad(d, "isn't a known note code")
    d = copy.deepcopy(good); d["pages"][0]["warnings"] = ["made_up"]
    _bad(d, "pages[0].warnings[0]")
    d = copy.deepcopy(good); d["refusal"] = {"code": "Not a code!", "message": "m"}
    _bad(d, "refusal.code")


def test_validate_offer_kind_flag_and_options(good: dict[str, Any]):
    d = copy.deepcopy(good); d["offers"][0]["apply_flag"] = "--lang"
    _bad(d, "apply_flag for a smaller offer is --choice")
    d = copy.deepcopy(good); d["offers"][1]["apply_flag"] = "--choice"
    _bad(d, "apply_flag for a language offer is --lang")
    for flag in ("--provenance", "--output", "--password-file"):
        d = copy.deepcopy(good); d["offers"][0]["apply_flag"] = flag
        _bad(d, "offers[0].apply_flag")
    for kind in ("provenance", "password", "overwrite", "decision"):
        d = copy.deepcopy(good); d["offers"][0]["kind"] = kind
        _bad(d, "offers[0].kind")
    d = copy.deepcopy(good); d["offers"][0]["options"][0]["id"] = "en"
    _bad(d, "isn't an option of a smaller offer")
    d = copy.deepcopy(good); d["offers"][1]["options"][0]["id"] = "none"
    _bad(d, "isn't a language code")
    d = copy.deepcopy(good); d["offers"][1]["options"][1]["id"] = "bs"
    _bad(d, "is repeated")
    d = copy.deepcopy(good); d["offers"][0]["summary"] = ""
    _bad(d, "offers[0].summary")
    d = copy.deepcopy(good); d["offers"][0]["options"] = []
    _bad(d, "offers[0].options must be a non-empty array")
    d = copy.deepcopy(good); d["offers"] = {"kind": "smaller"}
    _bad(d, "offers must be an array")
    d = copy.deepcopy(good); d["history"][0]["answer"] = "a b"
    _bad(d, "history[0].answer")
    d = copy.deepcopy(good); d["history"][0]["flag"] = "--provenance"
    _bad(d, "history[0].flag")
    d = copy.deepcopy(good); d["offers"] = []                      # a record with nothing on offer
    assert record.validate(d) == []


def test_offers_fit_the_output(good: dict[str, Any]):
    d = copy.deepcopy(good); d["output"] = None
    _bad(d, "offers are about a written output")
    d = copy.deepcopy(good); d["offers"].append(copy.deepcopy(d["offers"][0]))
    _bad(d, "offers a smaller copy at most once")
    d = copy.deepcopy(good)
    d["output"].update(kind="searchable_copy", original={"name": "Lease.pdf", "sha256": HEX64},
                       copy_reasons=["signed"])
    _bad(d, "no smaller copy is offered for a searchable copy")
    d["offers"] = d["offers"][1:]                                   # a language offer on a copy is fine
    assert record.validate(d) == []


def test_a_record_never_waits(good: dict[str, Any]):
    # §12.5: a run that wrote its file exits 0 whatever it offers; exit 10 and status decision are gone
    assert good["status"] == "done" and good["exit"] == 0 and good["offers"]
    d = copy.deepcopy(good); d["exit"] = 10
    _bad(d, "exit 10 isn't a record exit code")
    d = copy.deepcopy(good); d["status"] = "decision"
    _bad(d, "status 'decision' isn't one of")
    d = copy.deepcopy(good); d["language"]["plan"] = "asked"
    _bad(d, "language.plan 'asked'")
    d = copy.deepcopy(good); d["language"]["plan"] = "uncertain"
    assert record.validate(d) == []


def test_validate_other_enums(good: dict[str, Any]):
    for key, value, fragment in [("status", "maybe", "status"), ("exit", 9, "exit"), ("exit", True, "exit"),
                                 ("provenance", "mine", "provenance"), ("record_version", 2, "record_version"),
                                 ("created", "yesterday", "created")]:
        d = copy.deepcopy(good); d[key] = value
        _bad(d, fragment)
    d = copy.deepcopy(good); d["steps"][0]["status"] = "great"
    _bad(d, "steps[0].status")
    d = copy.deepcopy(good); d["input"]["seal"]["sha256"] = ["nothex"]
    _bad(d, "input.seal.sha256")


def test_validate_refuses_text_anywhere(good: dict[str, Any]):
    d = copy.deepcopy(good); d["steps"][1]["detail"]["text"] = "something a page said"
    _bad(d, "steps[1].detail.text: a record never holds document text")
    d = copy.deepcopy(good); d["pages"][0]["pick"] = {"rule": "single", "lines": []}
    _bad(d, "pages[0].pick.lines")
    # LayerReport.expected_text (§2.2 U8) or any other *_text key carries page text
    d = copy.deepcopy(good); d["steps"][1]["detail"]["expected_text"] = {"1": "ZQXJ canary page text"}
    _bad(d, "steps[1].detail.expected_text: a record never holds document text")
    d = copy.deepcopy(good); d["steps"][1]["detail"]["page_text"] = "x"
    _bad(d, "steps[1].detail.page_text")
    # LayerReport.dropped must reach a record as U+XXXX code points, never the raw characters
    d = copy.deepcopy(good); d["steps"][1]["detail"]["dropped"] = {"\u4e2d": 1}
    _bad(d, "steps[1].detail.dropped: dropped characters are named as U+XXXX")
    d = copy.deepcopy(good); d["steps"][1]["detail"]["dropped"] = {"U+4E2D": 1, "U+1F600": 2}
    assert record.validate(d) == []
    # a count keyed by a warning code that happens to end in _text is not text
    d = copy.deepcopy(good); d["steps"][1]["detail"]["by_warning"] = {"garbled_text": 1}
    assert record.validate(d) == []


def test_page_class_counts_may_name_classes_ending_in_text(good: dict[str, Any]):
    # classify's step counts pages by class, and two class names end in "_text"
    d = copy.deepcopy(good)
    d["steps"][1]["detail"]["kinds"] = {"born_digital": 1, "clip_text": 1, "hidden_text": 2}
    assert record.validate(d) == []
    # only as counts, and only there
    d = copy.deepcopy(good); d["steps"][1]["detail"]["kinds"] = {"hidden_text": "words a page said"}
    _bad(d, "steps[1].detail.kinds.hidden_text: a record never holds document text")
    d = copy.deepcopy(good); d["steps"][1]["detail"]["hidden_text"] = 2
    _bad(d, "steps[1].detail.hidden_text: a record never holds document text")
    d = copy.deepcopy(good); d["steps"][1]["detail"]["kinds"] = {"page_text": 1}
    _bad(d, "steps[1].detail.kinds.page_text")


def test_page_kinds_match_classify():
    import typing

    from pdfintake.pdf import classify

    assert record.PAGE_KINDS == frozenset(typing.get_args(classify.PageKind))


_WRONG_TYPES = [[], {}, ["agreement_low"], {"code": "x"}, 3, None]


@pytest.mark.parametrize("bad", _WRONG_TYPES, ids=repr)
@pytest.mark.parametrize("where", ["warnings.code", "notes.code", "pages.warnings", "offer.kind",
                                   "offer.apply_flag", "option.id", "status", "provenance", "steps.status",
                                   "history.flag", "language.plan", "exit"])
def test_validate_reports_wrong_types_instead_of_raising(good: dict[str, Any], where: str, bad: Any):
    d = copy.deepcopy(good)
    if where == "warnings.code":
        d["warnings"][0]["code"] = bad
    elif where == "notes.code":
        d["notes"][0]["code"] = bad
    elif where == "pages.warnings":
        d["pages"][0]["warnings"] = [bad]                # a code that is a list, a dict, a number or null
    elif where == "offer.kind":
        d["offers"][0]["kind"] = bad
    elif where == "offer.apply_flag":
        d["offers"][0]["apply_flag"] = bad
    elif where == "option.id":
        d["offers"][1]["options"][0]["id"] = bad
        d["offers"][1]["options"][1]["id"] = copy.deepcopy(bad)
    elif where == "steps.status":
        d["steps"][0]["status"] = bad
    elif where == "history.flag":
        d["history"][0]["flag"] = bad
    elif where == "language.plan":
        if bad is None:
            pytest.skip("a null plan is allowed")
        d["language"]["plan"] = bad
    else:
        if (bad is None and where == "provenance") or (bad == 3 and where == "exit"):
            pytest.skip("a valid value")
        d[where] = bad
    problems = record.validate(d)                     # never raises
    assert problems and all(isinstance(p, str) for p in problems)


def test_validate_reports_a_page_warnings_value_that_isnt_an_array(good: dict[str, Any]):
    d = copy.deepcopy(good); d["pages"][0]["warnings"] = {"agreement_low": 1}
    _bad(d, "pages[0].warnings must be an array")


def test_load_and_save_report_a_wrongly_typed_record_as_invalid(cache: Path):
    rec = full_record()
    data = record.to_json(rec)
    data["warnings"][0]["code"] = ["agreement_low"]
    data["offers"][0]["kind"] = {"k": 1}
    data["offers"][1]["options"][0]["id"] = []
    data["pages"][0]["warnings"] = [{}]
    (cache / "records").mkdir(exist_ok=True)
    (cache / "records" / f"{rec.id}.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(UsageError) as err:
        record.load(rec.id)
    assert err.value.code == "record_invalid" and err.value.exit_code == 2
    rec.warnings = [record.Message(["agreement_low"], "m")]          # type: ignore[arg-type]
    with pytest.raises(PdfIntakeError) as err2:
        record.save(rec)
    assert err2.value.code == "record_invalid"


def test_validate_searchable_copy_output(good: dict[str, Any]):
    # §10.2: a copy beside an untouched original names it and says why
    good = copy.deepcopy(good)
    good["offers"] = [o for o in good["offers"] if o["kind"] != "smaller"]    # never offered for a copy
    d = copy.deepcopy(good)
    d["output"].update(kind="searchable_copy", original={"name": "Lease.pdf", "sha256": HEX64},
                       copy_reasons=["provenance_counterparty"])
    assert record.validate(d) == []
    d = copy.deepcopy(good); d["output"]["kind"] = "searchable"
    assert record.validate(d) == []
    d = copy.deepcopy(good); del d["output"]["kind"]               # §11.1: every output says which it is
    _bad(d, "output is missing: kind")
    d = copy.deepcopy(good); d["output"]["kind"] = "searchable_copy"
    _bad(d, "names its original")
    _bad(d, "says why it is a copy")
    d = copy.deepcopy(good); d["output"]["kind"] = "smaller"
    _bad(d, "output.kind")
    d = copy.deepcopy(good)
    d["output"].update(kind="searchable_copy", original={"name": "", "sha256": "x"}, copy_reasons=["Bad code"])
    _bad(d, "output.original.name")
    _bad(d, "output.original.sha256")
    _bad(d, "output.copy_reasons[0]")
    d = copy.deepcopy(good); d["output"]["copy_reasons"] = ["signed"]
    _bad(d, "only a searchable copy carries")
    for name in ("/Users/example/Desktop/Lease.pdf", "Desktop/Lease.pdf", "C:\\Users\\example\\Lease.pdf", ".."):
        d = copy.deepcopy(good)                        # §10.2: the original's file name only, never its folder
        d["output"].update(kind="searchable_copy", original={"name": name, "sha256": HEX64}, copy_reasons=["signed"])
        _bad(d, "output.original.name is the original's file name only")


def test_validate_handles_non_objects():
    assert record.validate([]) == ["a record is a JSON object"]          # type: ignore[arg-type]
    assert record.validate({})                                           # missing everything, no crash


# ------------------------------------------------------------------------- overall_exit

@pytest.mark.parametrize("codes,expected", [
    ([0], 0), ([], 0), ([0, 4], 4), ([4, 5], 4), ([5, 0], 5), ([5, 1], 1), ([1, 4], 1), ([1, 3], 3), ([3, 2], 2),
    ([0, 4, 5, 1, 3, 2], 2), ([0, 5, 4], 4), ([4, 0, 4], 4), ([0, 0], 0),
])
def test_overall_exit_priorities(cache: Path, codes: list[int], expected: int):
    # §12.5: 2 > 3 > 1 > 4 > 5 > 0, decided 2026-11-05
    assert record.overall_exit([make(exit=c) for c in codes]) == expected
    assert record.EXIT_PRIORITY == (2, 3, 1, 4, 5, 0)
    assert "2 > 3 > 1 > 4 > 5 > 0" in record.overall_exit.__doc__ and "> 10" not in record.overall_exit.__doc__


@pytest.mark.parametrize("code", [7, 10])
def test_overall_exit_refuses_a_code_records_never_carry(cache: Path, code: int):
    with pytest.raises(ValueError):
        record.overall_exit([make(exit=0), make(exit=code)])


# ------------------------------------------------------------------------- load

@pytest.mark.parametrize("bad", ["../x", "/tmp/r-20261105-1a2b3c4d.json", "r-20261105-1a2b3c4d.json",
                                 "r-20261105-1a2b3c4d/../../x", "records/r-20261105-1a2b3c4d",
                                 "r-20261105-1a2b3c4d\n", " r-20261105-1a2b3c4d", "R-20261105-1A2B3C4D",
                                 "", "..", "~/r-20261105-1a2b3c4d"])
def test_load_refuses_anything_but_a_record_id(cache: Path, bad: str):
    with pytest.raises(UsageError) as err:
        record.load(bad)
    assert err.value.code == "bad_record_id" and err.value.exit_code == 2


def test_load_refuses_a_path_object(cache: Path, tmp_path: Path):
    rec = make(); record.save(rec)
    with pytest.raises(UsageError):
        record.load(cache / "records" / f"{rec.id}.json")              # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["folder", "not_utf8", "unreadable"])
def test_load_turns_a_record_file_it_cannot_read_into_record_invalid(cache: Path, kind: str):
    # C21 (decided 2026-11-06): an OSError or a decode error from reading is exit 2 record_invalid,
    # never a traceback or the CLI's generic internal error.
    import os
    import sys

    folder = cache / "records"
    folder.mkdir(exist_ok=True)
    rid = "r-20261105-1a2b3c4e"
    path = folder / f"{rid}.json"
    if kind == "folder":
        path.mkdir()
    elif kind == "not_utf8":
        path.write_bytes(b"\xff\xfe{\"id\": 1}")
    else:
        if sys.platform == "win32" or os.geteuid() == 0:
            pytest.skip("POSIX file modes, and root reads any file")
        path.write_text("{}", encoding="utf-8")
        path.chmod(0)
    try:
        with pytest.raises(UsageError) as err:
            record.load(rid)
    finally:
        if kind == "unreadable":
            path.chmod(0o600)
    assert err.value.code == "record_invalid" and err.value.exit_code == 2
    assert err.value.__cause__ is None and err.value.__suppress_context__
    assert rid in err.value.message


def test_load_unknown_and_damaged(cache: Path):
    with pytest.raises(UsageError) as err:
        record.load("r-20261105-00000000")
    assert err.value.code == "unknown_record"
    (cache / "records").mkdir(exist_ok=True)
    (cache / "records" / "r-20261105-11111111.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(UsageError) as err:
        record.load("r-20261105-11111111")
    assert err.value.code == "record_invalid"
    rec = make(); data = record.to_json(rec)
    (cache / "records" / "r-20261105-22222222.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(UsageError) as err:                               # holds another record's id
        record.load("r-20261105-22222222")
    assert err.value.code == "record_invalid"


# ------------------------------------------------------------------------- no document text

def _canary_page() -> PageOCR:
    word = Word(CANARY, 0.9, build.quad(0, 0, 10, 10))
    lines = [build.line(f"{CANARY} line one"), Line(f"x {CANARY}", 0.8, build.quad(0, 0, 9, 9), (word,))]
    reads = [build.engine_read("tesseract", lines=lines, seconds=1.25),
             build.engine_read("rapidocr", [f"{CANARY} checker"], role="checker")]
    pick = Pick("single", 1, "tesseract", list(lines), {"tesseract": 12.0}, False, "only reader")
    fills = [Fill("number", f"4521{CANARY}", 0.97, build.quad(1, 1, 5, 5), "rapidocr", near=CANARY),
             Fill("number", "4410", 0.9, build.quad(1, 1, 5, 5), "rapidocr")]
    return PageOCR(index=0, script=ScriptCall("Latin", 0.0, "osd", 0.81),
                   languages=[LangCall("en", 0.97, "langid", 4)], pack="en", composition="script",
                   reads=reads, pick=pick, fills=fills, repairs={"diacritic_word": 1},
                   must_survive={"digits": {"tesseract": 4, "layer": 4}}, warnings=["agreement_low"])


def test_no_document_text_survives_to_json(cache: Path):
    ocr = _canary_page()
    rec = make(pages=[ocr, {"page": 2, "readers": [ocr.reads[0]], "warnings": []}])
    data = record.to_json(rec)
    dumped = json.dumps(data, ensure_ascii=False)
    assert CANARY not in dumped
    problems = record.validate(data)
    assert problems and all(p.startswith("pages[1].readers[0].lines") for p in problems), problems
    page = data["pages"][0]
    assert page["page"] == 1 and page["fills"] == {"number": 2} and page["pick"]["chosen"] == "tesseract"
    assert page["readers"][0] == {"use_id": "tesseract", "engine": "tesseract", "spec": "tesseract:synthetic",
                                  "role": "reader", "status": "ok", "seconds": 1.25, "error": None, "score": 12.0}
    assert "lines" not in page["pick"] and page["warnings"] == ["agreement_low"]


def test_a_saved_record_holds_no_canary(cache: Path):
    rec = make(pages=[_canary_page()])
    path = record.save(rec)
    assert CANARY not in path.read_text(encoding="utf-8")
    assert record.load(rec.id).pages[0]["page"] == 1


def test_a_raw_read_in_a_dict_page_is_refused_not_saved(cache: Path):
    rec = make(pages=[{"page": 1, "readers": [_canary_page().reads[0]]}])
    data = record.to_json(rec)
    assert CANARY not in json.dumps(data, ensure_ascii=False)
    with pytest.raises(PdfIntakeError):                  # its lines (as counts) still aren't record fields
        record.save(rec)


# ------------------------------------------------------------------------- offers and wording

def test_smaller_offer_matches_the_spec_example():
    o = record.smaller_offer(output="/Users/example/Desktop/Receipt (searchable).pdf",
                             inputs=["/Users/example/Desktop/Receipt.pdf"], before_bytes=4302118,
                             after_bytes=1203311, saving_pct=72.0, min_score=92.4, bar=90)
    assert (o.kind, o.apply_flag) == ("smaller", "--choice")
    assert o.summary == ("A smaller copy could be saved as well: 72% smaller (4.3 MB to 1.2 MB). Its most detailed "
                         "part scores 92 out of 100; at 90 or above, a difference shows only when you flick "
                         "between the two.")
    assert [(x.id, x.label) for x in o.options] == [("smaller", "Save the smaller copy too")]
    assert o.options[0].description == ("Saves Receipt (searchable) (smaller).pdf next to it. "
                                        "Receipt (searchable).pdf and Receipt.pdf stay as they are.")
    assert all("/Users/example" not in t for t in [o.summary] + [x.description for x in o.options])
    assert "?" not in o.summary                                      # an offer, not a question


def test_smaller_offer_threshold_is_30_percent_with_no_byte_floor():
    assert record.SMALLER_MIN_SAVING_PCT == 30.0
    assert record.STAGED_DAYS == 14
    tiny = record.smaller_offer(output="Tiny (searchable).pdf", inputs=["a.jpg", "b.jpg"], before_bytes=10_000,
                                after_bytes=7_000, saving_pct=30.0, min_score=90.0, bar=90)
    assert "30% smaller (10 KB to 7 KB)" in tiny.summary
    assert "the original files stay as they are" in tiny.options[0].description
    with pytest.raises(ValueError):
        record.smaller_offer(output="x.pdf", inputs=["y.pdf"], before_bytes=10_000, after_bytes=7_100,
                             saving_pct=29.9, min_score=95, bar=90)
    with pytest.raises(ValueError):
        record.smaller_offer(output="x.pdf", inputs=["y.pdf"], before_bytes=10_000, after_bytes=5_000,
                             saving_pct=50, min_score=89.9, bar=90)
    # the floors never round a near miss up
    near = record.smaller_offer(output="x.pdf", inputs=["y.pdf"], before_bytes=10_000_000, after_bytes=6_000_000,
                                saving_pct=39.99, min_score=90.99, bar=90)
    assert "39% smaller" in near.summary and "scores 90 out of 100" in near.summary


def _with_offers(rec: record.Record, *offers: record.Offer) -> dict[str, Any]:
    rec.status, rec.exit = "done", 0
    rec.output = record.output_entry("/Users/example/Desktop/Receipt (searchable).pdf", "c1" * 32, 4302118)
    rec.offers = list(offers)
    return record.to_json(rec)


def test_language_offer(cache: Path):
    o = record.language_offer(used="hr", others=[("hr", None), ("bs", 9_000_000), ("sr", 0)], pages=[2, 3])
    assert (o.kind, o.apply_flag) == ("language", "--lang")
    assert [(x.id, x.label) for x in o.options] == [("bs", "Bosnian (download 9 MB)"), ("sr", "Serbian")]
    assert o.summary == ("Pages 2 and 3 were read as Croatian, the likeliest language, but pdfintake wasn't sure. "
                         "Pages 2 and 3 can be read again as Bosnian or Serbian if you ask.")
    assert o.options[0].description == ("Reads pages 2 and 3 again with the Bosnian language pack, "
                                        "downloading it once (9 MB).")
    assert o.options[1].description.endswith("already installed.")
    one = record.language_offer(used="de", others=[("en", None)], pages=[4])
    assert one.summary.startswith("Page 4 was read as German") and one.options[0].id == "en"
    whole = record.language_offer(used="es", others=[("pt", 5_000_000), ("en", None)])
    assert whole.summary.startswith("These pages were read as Spanish")
    for offer in (o, one, whole):
        assert "?" not in offer.summary
        assert record.validate(_with_offers(make(), offer)) == []
    with pytest.raises(ValueError):                                  # nothing else to offer
        record.language_offer(used="hr", others=[("hr", None)])
    with pytest.raises(ValueError):
        record.language_offer(used="hr", others=[("not a code", None)])
    with pytest.raises(ValueError):
        record.language_offer(used="hr", others=[("bs", None), ("bs", None)])


def test_every_offer_is_acted_on_with_ids_only(cache: Path):
    offers = [record.smaller_offer(output="a (searchable).pdf", inputs=["a.pdf"], before_bytes=3_000_000,
                                   after_bytes=1_000_000, saving_pct=66.7, min_score=93, bar=90),
              record.language_offer(used="de", others=[("hu", None), ("en", None)])]
    rec = make()
    for o in offers:
        for x in o.options:
            cmd = f"pdfintake apply {rec.id} {o.apply_flag} {x.id} --json"
            assert re.fullmatch(r"pdfintake apply r-\d{8}-[0-9a-f]{8} --(choice|lang) [a-z]+ --json", cmd)


def test_language_warnings_name_the_languages():
    # §12.3: what used to be a question is a warning, worded in core
    w = record.language_warning("language_uncertain", language="hr", others=["hr", "bs", "sr"], page=2)
    assert (w.code, w.page) == ("language_uncertain", 2)
    assert w.message == ("pdfintake wasn't sure which language this page is in, so it read it as Croatian, the "
                         "likeliest. It could also be Bosnian or Serbian. It can read it again as another language "
                         "if you ask.")
    alone = record.language_warning("language_uncertain", language="hr")
    assert "It could also be" not in alone.message
    n = record.language_warning("no_pack_for_language", language="sw", page=1)
    assert "Swahili" in n.message and "docs/add-a-language.md" in n.message
    i = record.language_warning("pack_not_installed", language="th")
    assert i.message.startswith("The Thai language pack isn't installed") and i.page is None
    for code in ("language_uncertain", "no_pack_for_language", "pack_not_installed"):
        assert code in record.WARNING_GROUPS["language"]
        assert record.warning(code).message == record.WARNING_MESSAGES[code]
    with pytest.raises(ValueError):
        record.language_warning("mixed_language_page", language="th")    # type: ignore[arg-type]
    with pytest.raises(ValueError):
        record.language_warning("language_uncertain", language="not a code")


def test_no_pack_for_language_names_a_language_a_script_or_neither():
    # langplan's ``detected`` is a language code or a script name (N2-R1, decided 2026-11-05)
    s = record.language_warning("no_pack_for_language", language="Ethiopic", page=3)
    assert s.message.startswith("This page has text in the Ethiopic writing system, and pdfintake has no "
                                "language pack for it yet") and s.page == 3
    c = record.language_warning("no_pack_for_language", language="sw")
    assert c.message.startswith("This page looks like Swahili, and pdfintake has no language pack for it yet")
    none = record.language_warning("no_pack_for_language", language=None, page=1)
    assert none.message == record.WARNING_MESSAGES["no_pack_for_language"] and none.page == 1
    for bad in ("not a code", "Ethiopic script", "ETHIOPIC!", "x", ""):
        with pytest.raises(ValueError):
            record.language_warning("no_pack_for_language", language=bad)
    for code in ("language_uncertain", "pack_not_installed"):
        with pytest.raises(ValueError):
            record.language_warning(code, language=None)          # type: ignore[arg-type]
        with pytest.raises(ValueError):
            record.language_warning(code, language="Ethiopic")    # these two always name a code


def test_an_uncertain_page_read_as_another_language_says_so():
    # A close group's member with a pack, or the base pack when the likeliest has none
    w = record.language_warning("language_uncertain", language="bs", others=["hr"], read_as="hr", page=1)
    assert w.message == ("pdfintake wasn't sure which language this page is in. The likeliest is Bosnian; it could "
                         "also be Croatian. It was read as Croatian, and it can read it again as another language "
                         "if you ask.")
    b = record.language_warning("pack_broken", language="th", page=2)
    assert b.message.startswith("The Thai language pack couldn't be used this run") and b.code == "pack_broken"
    same = record.language_warning("language_uncertain", language="hr", others=["bs"], read_as="hr")
    assert same.message == record.language_warning("language_uncertain", language="hr", others=["bs"]).message
    with pytest.raises(ValueError):
        record.language_warning("language_uncertain", language="hr", read_as="not a code")


def test_language_offer_words_each_reason(cache: Path):
    unsure = record.language_offer(reason="uncertain", used="de", others=[("hu", 12_350_405)], pages=[1])
    assert unsure.summary == record.language_offer(used="de", others=[("hu", 12_350_405)], pages=[1]).summary
    code = record.language_offer(reason="no_pack", used="en", others=[("de", 0)], pages=[3], detected="sw")
    assert code.summary == ("Page 3 looks like Swahili, and pdfintake has no language pack for it yet, so it was "
                            "read as English. Page 3 can be read again as German if you ask.")
    script = record.language_offer(reason="no_pack", used="en", others=[("de", 0)], pages=[2, 5], detected="Ethiopic")
    assert script.summary == ("Pages 2 and 5 have text in the Ethiopic writing system, and pdfintake has no language "
                              "pack for it yet, so they were read as English. Pages 2 and 5 can be read again as "
                              "German if you ask.")
    plain = record.language_offer(reason="no_pack", used="en", others=[("es", None)], pages=[1])
    assert plain.summary.startswith("Page 1 is in a language pdfintake has no language pack for yet, so it was read "
                                    "as English.")
    several = record.language_offer(reason="no_pack", used="en", others=[("es", None)], pages=[1, 2], detected="it")
    assert several.summary.startswith("Pages 1 and 2 look like Italian, and pdfintake")
    broken = record.language_offer(reason="broken", used="en", others=[("th", 11_900_000)], pages=[4])
    assert broken.summary == ("Page 4 was read as English, because the language pack it needs couldn't be used this "
                              "run (its download or check failed). Page 4 can be read again as Thai if you ask.")
    off = record.language_offer(reason="not_installed", used="en", others=[("th", 11_900_000)], pages=[4])
    assert off.summary == ("Page 4 was read as English, because the language pack it needs isn't installed and this "
                           "run didn't download anything. Page 4 can be read again as Thai if you ask.")
    assert off.options[0].label == "Thai (download 12 MB)"
    offs = record.language_offer(reason="not_installed", used="en", others=[("th", 11_900_000)], pages=[1, 2])
    assert "the language pack they need isn't installed" in offs.summary
    for offer in (unsure, code, script, plain, several, off, offs, broken):
        assert "?" not in offer.summary and (offer.kind, offer.apply_flag) == ("language", "--lang")
        assert record.validate(_with_offers(make(), offer)) == []
    with pytest.raises(ValueError):
        record.language_offer(reason="asked", used="en", others=[("de", 0)])
    with pytest.raises(ValueError):                                  # only a no_pack offer names what was found
        record.language_offer(reason="uncertain", used="de", others=[("en", 0)], detected="fr")
    with pytest.raises(ValueError):
        record.language_offer(reason="no_pack", used="en", others=[("de", 0)], detected="Ethiopic script")


def test_record_follows_langplans_vocabulary():
    from pdfintake.ocr import langplan

    assert record.LANGUAGE_OFFER_REASONS == langplan.OFFER_REASONS
    assert record._SCRIPT_NAME_RE.pattern == langplan._SCRIPT_NAME_RE.pattern
    assert set(langplan.RESOLVE_WARNINGS) <= set(record.WARNING_GROUPS["language"])
    from pdfintake.ocr import scripts

    for name in scripts.SCRIPTS:                                    # every core script can be named
        assert record.language_warning("no_pack_for_language", language=name).message.startswith(
            f"This page has text in the {name} writing system")


@dataclass(frozen=True)
class _Dist:
    bytes: int


@dataclass(frozen=True)
class _Entry:
    id: str
    base: bool
    langid_codes: tuple[str, ...]
    dist: _Dist


@dataclass(frozen=True)
class _Index:
    entries: tuple[_Entry, ...]

    def get(self, pack_id: str) -> _Entry | None:
        return next((e for e in self.entries if e.id == pack_id), None)

    def base(self) -> _Entry:
        return next(e for e in self.entries if e.base)


def test_every_offer_and_warning_resolve_unsure_gives_builds_a_valid_record(cache: Path):
    # N2-R1: feed each kind of LanguageOffer langplan.resolve_unsure returns, and each page's
    # language warnings, through record exactly as flow will, and validate the record.
    from pdfintake.ocr import langplan
    from pdfintake.ocr.langplan import Candidate, LanguagePlan, PagePlan

    index = _Index((_Entry("en", True, ("en",), _Dist(0)), _Entry("de", False, ("de",), _Dist(8_628_461)),
                    _Entry("hu", False, ("hu",), _Dist(12_350_405)), _Entry("hr", False, ("hr",), _Dist(9_000_000))))
    en = Candidate("en", "en", "the base pack", 0)
    de, hu = Candidate("de", "de", "made up", 0, 0.5), Candidate("hu", "hu", "made up", 12_350_405, 0.3)

    def undecided(i: int, unsure: str, cands: list[Candidate], detected: str | None = None) -> PagePlan:
        return PagePlan(i, None, "undecided", [], False, "made up", unsure=unsure, detected=detected,
                        candidates=cands)

    pages = [
        undecided(0, "uncertain", [de, hu, en]),                                     # uncertain
        undecided(1, "no_pack", [Candidate("sw", None, "made up", None, 0.9), de, en], detected="sw"),
        undecided(2, "no_pack", [de, en], detected="Ethiopic"),                      # a script with no pack
        undecided(3, "uncertain", [Candidate("it", None, "made up", None, 0.5), de, en]),
        undecided(4, "uncertain", [hu, de, en]),                                     # offline: hu not installed
        PagePlan(5, "hu", "langid", [LangCall("hu", 0.97, "langid", 4)], True, "made up"),
        undecided(6, "close_group", [Candidate("bs", None, "made up", None, 0.9), Candidate("hr", "hr", "x", 0), en]),
        PagePlan(7, "hr", "langid", [LangCall("hr", 0.97, "langid", 4)], True, "made up"),   # hr broken
    ]
    lp = LanguagePlan(pages, [], False, [de, en], None, [], unavailable=["hu"])
    p = langplan.resolve_unsure(lp, index, broken={"hr"})
    assert {o.reason for o in p.offers} == set(record.LANGUAGE_OFFER_REASONS)
    assert any(o.detected == "Ethiopic" for o in p.offers)

    rec = make()
    rec.status, rec.exit = "done", 0
    rec.output = record.output_entry("/Users/example/Desktop/Letter (searchable).pdf", "c1" * 32, 1000)
    for o in p.offers:
        rec.offers.append(record.language_offer(reason=o.reason, used=o.read_as, detected=o.detected,
                                                others=[(c.tag, c.download_bytes) for c in o.options],
                                                pages=[i + 1 for i in o.pages]))
    tag_of = {e.id: e.langid_codes[0] for e in index.entries}
    seen: set[str] = set()
    for before, pp in zip(pages, p.pages):
        for code in pp.warnings:
            seen.add(code)
            page_no = pp.index + 1
            if code == "language_uncertain":
                likeliest = pp.candidates[0].tag
                rec.warnings.append(record.language_warning(code, language=likeliest, page=page_no,
                                                            others=[c.tag for c in pp.candidates[1:]],
                                                            read_as=tag_of[pp.pack]))
            elif code == "no_pack_for_language":
                rec.warnings.append(record.language_warning(code, language=pp.detected, page=page_no))
            elif code in ("pack_not_installed", "pack_broken"):
                wanted = before.pack or next(c.pack for c in pp.candidates if c.pack)
                rec.warnings.append(record.language_warning(code, language=tag_of[wanted], page=page_no))
    assert seen == set(langplan.RESOLVE_WARNINGS)
    data = record.to_json(rec)
    assert record.validate(data) == []
    words = json.dumps([o["summary"] for o in data["offers"]] + [w["message"] for w in data["warnings"]])
    assert "Ethiopic writing system" in words and "Swahili" in words and "Bosnian" in words


def test_an_uncertain_page_record_validates(cache: Path):
    rec = make()
    data = _with_offers(rec, record.language_offer(used="hr", others=[("bs", None)], pages=[1]))
    data["language"] = {"plan": "uncertain", "packs": [], "canaries": [], "detection": {}}
    data["warnings"] = [record.to_json(make(warnings=[record.language_warning(
        "language_uncertain", language="hr", others=["bs"], page=1)]))["warnings"][0]]
    assert record.validate(data) == []


# ------------------------------------------------------------------------- §10 / §11.1: copies and passwords

def test_output_entry_builds_both_kinds():
    plain = record.output_entry("/Users/example/a (searchable).pdf", "c1" * 32, 10)
    assert plain == {"path": "/Users/example/a (searchable).pdf", "sha256": "c1" * 32, "bytes": 10,
                     "kind": "searchable"}
    cp = record.output_entry("/Users/example/Lease (searchable copy).pdf", "c1" * 32, 10,
                             original=Path("/Users/example/Lease.pdf"), original_sha256=HEX64,
                             copy_reasons=["signed", "provenance_evidence"])
    assert cp["kind"] == "searchable_copy"
    assert cp["original"] == {"name": "Lease.pdf", "sha256": HEX64}          # the name only, never the folder
    assert cp["copy_reasons"] == ["signed", "provenance_evidence"]
    with pytest.raises(ValueError):
        record.output_entry("x.pdf", "c1" * 32, 1, copy_reasons=["signed"])
    with pytest.raises(ValueError):
        record.output_entry("x.pdf", "c1" * 32, 1, original="Lease.pdf", original_sha256=HEX64)
    with pytest.raises(ValueError):
        record.output_entry("x.pdf", "c1" * 32, 1, original="Lease.pdf", original_sha256="nothex",
                            copy_reasons=["signed"])


def _copy(good: dict[str, Any], reasons: Any) -> dict[str, Any]:
    d = copy.deepcopy(good)
    d["offers"] = [o for o in d["offers"] if o["kind"] != "smaller"]     # never offered for a copy
    d["output"].update(kind="searchable_copy", original={"name": "Lease.pdf", "sha256": HEX64}, copy_reasons=reasons)
    return d


def test_copy_reasons_are_checked_against_refusals_copyable(good: dict[str, Any]):
    from pdfintake import refusals

    for code in refusals.COPYABLE:
        assert record.validate(_copy(good, [code])) == [], code
    for code in refusals.HARD:                             # xfa and the rest are refused, never copied
        _bad(_copy(good, [code]), "output.copy_reasons[0]")
    _bad(_copy(good, ["made_up"]), "isn't a copy reason")
    _bad(_copy(good, []), "at least one copy reason")
    _bad(_copy(good, ["signed", "signed"]), "is repeated")
    assert record.validate(_copy(good, ["signed", "encrypted", "provenance_counterparty"])) == []
    _bad(_copy(good, ["provenance_counterparty", "signed"]), "refusals.CODES order")


def test_copy_reasons_follow_whatever_copyable_says(good: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    # The import is lazy, so the check follows refusals.COPYABLE as it stands, not a copy of it.
    from pdfintake import refusals

    monkeypatch.setattr(refusals, "COPYABLE", frozenset({"signed"}))
    assert record.validate(_copy(good, ["signed"])) == []
    _bad(_copy(good, ["pdfa"]), "isn't a copy reason")


def test_copy_mode_codes_and_wording():
    assert "copy_unlocked" in record.NOTE_CODES and "copy_unlocked" not in record.WARNING_CODES
    assert record.note("copy_unlocked").message.lower().startswith(
        "the copy has no password, so anyone with the file can read it")       # the skill's one line, §11.1
    assert record.note("signature_not_carried").message == (
        "The signed original is the one that counts; this copy is for finding it.")       # §10.3
    for code in ("conformance_claim_dropped", "skipped_no_password", "render_scaled_down"):
        assert record.note(code).message
    assert record.warning("exporting_a_copy").message
    assert "exporting_a_copy" in record.WARNING_GROUPS["export"]
    assert "encryption_kept" not in record.NOTE_CODES                       # §10.4: a copy is never encrypted
    assert record.WARNING_MESSAGES["layer_dropped_chars"].startswith(
        "Some characters couldn't be written into the text layer")         # §11.2
    assert "32767" in record.NOTE_MESSAGES["render_scaled_down"]


def test_the_per_file_input_codes_are_core_worded_usage_codes():
    # C20 (decided 2026-11-06): a folder, a pipe or a device given as an input is its own exit-2 record.
    for code in ("input_missing", "input_not_file", "unsupported_input"):
        err = record.usage_error(code, path="/Users/example/Desktop/Scans")
        assert isinstance(err, UsageError) and err.exit_code == 2 and err.code == code
        assert err.message == record.USAGE_MESSAGES[code] and "{" not in err.message
    assert "regular file" in record.USAGE_MESSAGES["input_not_file"]
    for what in ("folder", "pipe", "device"):
        assert what in record.USAGE_MESSAGES["input_not_file"], what


def test_pack_broken_covers_a_failed_download_and_names_the_base_pack():
    # N16 (decided 2026-11-06): resolve_unsure's ``broken`` holds packs whose download failed as well
    # as packs whose check failed, and the page was read with the base pack either way.
    default = record.warning("pack_broken").message
    assert "download" in default and "check" in default and "base English pack" in default
    named = record.language_warning("pack_broken", language="hu", page=4).message
    assert named.startswith("The Hungarian language pack couldn't be used this run")
    assert "download" in named and "check" in named and "base English pack" in named
    assert "again with the Hungarian pack" in named
    with pytest.raises(ValueError):
        record.language_warning("pack_broken", language=None)      # type: ignore[arg-type]


@pytest.mark.parametrize("reason, words", [
    ("uncertain", "the likeliest language, but pdfintake wasn't sure"),
    ("no_pack", "has no language pack for it yet"),
    ("not_installed", "isn't installed and this run didn't download anything"),
    ("broken", "couldn't be used this run"),
])
def test_language_offer_has_a_true_summary_for_every_reason(reason: str, words: str):
    used = "hr" if reason == "uncertain" else "en"
    offer = record.language_offer(reason=reason, used=used, others=[("bs", 9_000_000)], pages=[2],
                                  detected="sw" if reason == "no_pack" else None)
    assert words in offer.summary, offer.summary
    others = set(record.LANGUAGE_OFFER_REASONS) - {reason}
    assert record.LANGUAGE_OFFER_REASONS == ("uncertain", "no_pack", "not_installed", "broken") and others
    if reason != "uncertain":
        assert "likeliest" not in offer.summary                   # English is never called the likeliest


def test_a_damaged_refusal_is_in_the_records_vocabulary(good: dict[str, Any]):
    # C20-A6 (decided 2026-11-06): a file that fails the structure checks is refused as `damaged`.
    from pdfintake import refusals

    assert "damaged" in refusals.HARD and "damaged" in refusals.CODES
    d = copy.deepcopy(good)
    d.update(status="refused", exit=4, output=None, offers=[])
    d["refusal"] = {"code": "damaged", "message": refusals._MESSAGES["damaged"],
                    "detectors": {"walk": ["open_failed"], "raw": ["no_trailer"]}, "repaired": False}
    assert record.validate(d) == []
    _bad(_copy(good, ["damaged"]), "output.copy_reasons[0]")       # refused, never a copy reason
    d["refusal"]["code"] = "no_smaller_for_a_copy"                 # apply's refusal never sits in a record
    _bad(d, "refusal.code")


def test_a_refusal_code_is_a_hard_code(good: dict[str, Any]):
    # Only refusals.HARD refuses a file (decided 2026-11-06); a copyable code makes a searchable copy,
    # so a record whose refusal names one is invalid.
    from pdfintake import refusals

    d = copy.deepcopy(good)
    d.update(status="refused", exit=4, output=None, offers=[])
    for code in sorted(refusals.HARD):
        d["refusal"] = {"code": code, "message": "m", "detectors": {"walk": [], "raw": []}, "repaired": False}
        assert record.validate(d) == [], code
    for code in sorted(refusals.COPYABLE):
        d["refusal"] = {"code": code, "message": "m", "detectors": {"walk": [], "raw": []}, "repaired": False}
        _bad(d, "refusal.code")


def test_copy_never_overwrites_is_a_usage_error():
    err = record.usage_error("copy_never_overwrites")
    assert isinstance(err, UsageError) and err.exit_code == 2 and err.code == "copy_never_overwrites"
    assert "--overwrite" in err.message
    assert record.USAGE_CODES == {"copy_never_overwrites", "passwords_file_unsafe", "output_in_use",
                                  "input_missing", "input_not_file", "unsupported_input"}
    with pytest.raises(ValueError):
        record.usage_error("made_up")


def test_passwords_file_unsafe_is_a_usage_error():
    # §12.1: a saved passwords file anyone else could read or change is never read (exit 2)
    err = record.usage_error("passwords_file_unsafe", mode="0644")
    assert isinstance(err, UsageError) and err.exit_code == 2 and err.code == "passwords_file_unsafe"
    assert "600" in err.message and "{" not in err.message
    assert err.detail == {"mode": "0644"}


def test_the_passwords_file_fallback_wording_fits_every_reason():
    # N2-R4: passwords.py raises this code with its own per-reason message; core's fallback for the
    # code must be true whatever the reason, and agree with passwords.py's limits.
    from pdfintake import passwords

    msg = record.USAGE_MESSAGES[passwords.UNSAFE]
    assert passwords.UNSAFE in record.USAGE_CODES
    assert f"at most {passwords.MAX_ENTRIES} passwords" in msg
    for part in ("regular", "UTF-8", "user folder", "only you can read or change", "chmod 600", "one per line"):
        assert part in msg, part


def test_a_file_no_saved_password_opens_is_skipped_with_exit_0(cache: Path):
    # §12.1: skip, don't stop; nothing written, the original untouched, the batch carries on
    rec = make()
    rec.status, rec.exit, rec.output = "done", 0, None
    rec.notes = [record.note("skipped_no_password"), record.note("input_unchanged")]
    data = record.to_json(rec)
    assert record.validate(data) == []
    assert "saved passwords" in record.NOTE_MESSAGES["skipped_no_password"]
    assert record.overall_exit([rec, make(exit=0)]) == 0
    path = record.save(rec)
    assert record.load(rec.id).notes[0].code == "skipped_no_password"
    assert path.is_file()


@dataclass
class PasswordOptions(FakeOptions):     # RunOptions as U20/U21 may grow it (§12.1)
    passwords_file: Path | None = None
    password: str | None = None


def test_options_never_carry_a_password(cache: Path):
    opts = PasswordOptions(passwords_file=Path("/Users/example/.pw"), password=PASSWORD)
    rec = record.new([Path("/Users/example/Locked.pdf")], opts=opts, seal=None, platform_key="darwin-arm64")
    assert not any("password" in k for k in rec.options)
    dumped = json.dumps(record.to_json(rec))
    assert PASSWORD not in dumped and "/Users/example/.pw" not in dumped
    assert record.validate(record.to_json(rec), forbid=[PASSWORD]) == []


@pytest.mark.parametrize("key", ["password", "passwords_file", "Password", "user_password", "password_stdin"])
def test_validate_refuses_a_password_key_anywhere(good: dict[str, Any], key: str):
    d = copy.deepcopy(good); d["options"][key] = "x"
    _bad(d, f"options.{key}: a record never holds a password")
    d = copy.deepcopy(good); d["steps"][0]["detail"][key] = None
    _bad(d, f"steps[0].detail.{key}: a record never holds a password")
    # a count keyed by a note code is not a password field
    d = copy.deepcopy(good); d["steps"][0]["detail"]["by_note"] = {"skipped_no_password": 1}
    assert record.validate(d) == []
    d = copy.deepcopy(good); d["steps"][0]["detail"]["by_usage"] = {"passwords_file_unsafe": 1}
    assert record.validate(d) == []


def test_a_password_anywhere_is_refused_and_never_repeated(cache: Path, good: dict[str, Any]):
    places = [
        lambda d: d["steps"][0]["detail"].update(error=f"could not open with {PASSWORD}"),
        lambda d: d["warnings"][0].update(message=PASSWORD),
        lambda d: d["steps"][0]["detail"].update({PASSWORD: 1}),
        lambda d: d["history"][0].update(answer=PASSWORD),            # not an option id, so it is scanned
        lambda d: d["verify"][0].update(detail=f"opened with {PASSWORD}"),
        lambda d: d["refusal"].__class__ is type(None) and d.update(
            refusal={"code": "xfa", "message": f"no {PASSWORD}", "detectors": None, "repaired": False}),
    ]
    for put in places:
        d = copy.deepcopy(good); put(d)
        problems = record.validate(d, forbid=[PASSWORD])
        assert problems == ["a record never holds a password (1 place)"], problems
        assert PASSWORD not in json.dumps(problems)
    assert record.validate(good, forbid=[PASSWORD]) == []
    assert record.validate(good, forbid=[PASSWORD.encode()]) == []
    assert record.validate(good, forbid=[""]) == []               # an empty open password isn't a secret
    # a password under four characters can't be told from the record's own words ("ok"), so it isn't
    # scanned for; no field can take it anyway (the key and options tests above)
    assert record.validate(good, forbid=["ok"]) == []
    d = copy.deepcopy(good); d["warnings"][0]["message"] = "abcd"
    assert record.validate(d, forbid=["abcd"]) == ["a record never holds a password (1 place)"]


def _copy_record_for(path: str) -> record.Record:
    """A finished copy-mode record for an encrypted input at ``path`` (made-up values)."""
    rec = record.new([Path(path)], opts=FakeOptions(outdir=Path(path).parent), seal=None,
                     platform_key="darwin-arm64")
    rec.status, rec.exit = "done", 0
    stem = Path(path).stem
    out = str(Path(path).with_name(f"{stem} (searchable copy).pdf"))
    rec.output = record.output_entry(out, "c1" * 32, 10, original=path, original_sha256=HEX64,
                                     copy_reasons=["encrypted"])
    from pdfintake import refusals

    rec.steps = [{"step": "layer", "status": "ok", "detail": {"wrote": out, "chars_written": 120}},
                 {"step": "copy", "status": "ok", "detail": {"why": [refusals.copy_message("encrypted")]}}]
    rec.verify = [{"name": "not_encrypted", "ok": True, "detail": f"{Path(out).name} opens without a password",
                   "page": None}]
    rec.notes = [record.note("copy_unlocked"), record.note("input_unchanged")]
    rec.offers = [record.language_offer(used="en", others=[("de", None)], pages=[1])]
    rec.history = [{"at": rec.created, "flag": "--lang", "answer": "de", "result": 0,
                    "child": "r-20261105-0a0b0c0d"}]
    return rec


@pytest.mark.parametrize("pin", ["4417", "2026", "Desktop", "copy", "searchable", "password"])
def test_a_short_password_in_the_records_own_fields_still_saves(cache: Path, pin: str):
    # A PIN that is also the year (in the id and the created time), account digits that are also in
    # the file name, or a word of core's own wording are not a password in the record: the record
    # saves, and its failure record would too (decided 2026-11-05).
    rec = _copy_record_for("/Users/example/Desktop/eStatement_4417.pdf")
    rec.id = "r-20261105-4417a026"
    rec.created = "2026-11-05T10:20:26Z"
    data = record.to_json(rec)
    assert record.validate(data) == []
    assert record.validate(data, forbid=[pin]) == [], pin
    assert record.save(rec, forbid=[pin]).is_file()
    rec.status, rec.exit, rec.offers = "failed", 1, []
    assert record.validate(record.to_json(rec), forbid=[pin]) == []
    # ... but the same password in free text is still refused, next to the path that holds it. A
    # password that is a whole word of core's own wording ("copy", "password") can't be told apart
    # from that wording, so only a longer string holding it is found there.
    secret = pin if pin.casefold() not in record._core_words() else f"{pin}-77"
    rec.steps.append({"step": "open", "status": "failed",
                      "detail": {"error": f"/Users/example/Desktop/eStatement_4417.pdf: tried {secret}"}})
    assert record.validate(record.to_json(rec), forbid=[secret]) == ["a record never holds a password (1 place)"]
    rec.steps.pop()
    rec.warnings = [record.warning("reader_failed", page=1, message=f"Page 1 failed: {secret}!")]
    assert record.validate(record.to_json(rec), forbid=[secret]) == ["a record never holds a password (1 place)"]


def test_core_words_are_words_of_core_wording():
    words = record._core_words()
    assert {"password", "copy", "searchable", "original"} <= words
    assert not any(w.isdigit() or len(w) < 4 for w in words)
    assert "desktop" not in words and "4417" not in words


def test_a_skipped_field_is_scanned_when_it_breaks_its_pattern(good: dict[str, Any]):
    # The id, times and codes are left out of the scan only while they match their own pattern; a
    # value that doesn't is scanned, so a password there is still found and never repeated.
    for put in (lambda d: d.update(id=f"r-{PASSWORD}"),
                lambda d: d.update(created=PASSWORD),
                lambda d: d["history"][0].update(child=PASSWORD),
                lambda d: d["notes"][0].update(code=PASSWORD),
                lambda d: d["output"].update(kind=PASSWORD)):
        d = copy.deepcopy(good); put(d)
        problems = record.validate(d, forbid=[PASSWORD])
        assert problems == ["a record never holds a password (1 place)"], problems


def test_no_problem_ever_repeats_a_password(good: dict[str, Any]):
    # A plain-name key isn't scanned (it is code's own name), but an unknown one is reported by name:
    # the report hides the password.
    d = copy.deepcopy(good); d["output"]["hunter42pin"] = 1
    problems = record.validate(d, forbid=["hunter42pin"])
    assert problems and all("hunter42pin" not in x for x in problems), problems
    assert any("output has unknown fields" in x for x in problems)


def test_save_refuses_a_record_holding_the_password(cache: Path):
    rec = full_record()
    rec.steps.append({"step": "open", "status": "failed", "detail": {"error": f"bad {PASSWORD}"}})
    with pytest.raises(PdfIntakeError) as err:
        record.save(rec, forbid=[PASSWORD])
    assert err.value.code == "record_invalid"
    assert PASSWORD not in str(err.value) and PASSWORD not in json.dumps(err.value.detail)
    assert not (cache / "records").exists() or not any((cache / "records").iterdir())
    rec.steps.pop()
    path = record.save(rec, forbid=[PASSWORD])
    assert PASSWORD not in path.read_text(encoding="utf-8")


def test_no_message_wording_holds_a_password_slot():
    # Every word core shows is fixed text: none of it has a placeholder a password could fill.
    texts = list(record.NOTE_MESSAGES.values()) + list(record.WARNING_MESSAGES.values())
    texts += list(record.USAGE_MESSAGES.values())
    assert not any("{" in t for t in texts)


def test_decision_record_doc_lists_every_code():
    doc = (Path(__file__).resolve().parent.parent / "docs" / "decision-record.md").read_text(encoding="utf-8")
    for code in sorted(record.WARNING_CODES | record.NOTE_CODES | record.USAGE_CODES):
        assert f"`{code}`" in doc, code
    for kind in record.OFFER_KINDS:
        assert f"| `{kind}` | `{record.APPLY_FLAG_FOR[kind]}` |" in doc, kind
    assert record.OPTION_ID_RE.pattern in doc
    assert "never waits for an answer" in doc                          # §12, said plainly
    for gone in ("`password_wrong`", "`supply`", "`decline`", "`--password-stdin`", "| 10 |", "`decision`",
                 "AskUserQuestion", "Ask the user"):
        assert gone not in doc, gone
    assert "`encryption_kept`" not in doc
    assert "`Info/<Key>`" in doc
    from pdfintake import refusals

    for code in refusals.COPYABLE | refusals.HARD:
        assert f"`{code}`" in doc, code
    assert not any(line.startswith("> ") for line in doc.splitlines())    # no question is quoted any more


def _doc_json_blocks() -> list[Any]:
    doc = (Path(__file__).resolve().parent.parent / "docs" / "decision-record.md").read_text(encoding="utf-8")
    return [json.loads(b) for b in re.findall(r"```json\n(.*?)\n```", doc, flags=re.S)]


def test_the_record_doc_examples_are_valid_as_written():
    # N2-R6: the full example passes validate as it stands (no shortened digests), and the offer and
    # copy-output snippets are valid parts of a record.
    blocks = _doc_json_blocks()
    full = [b for b in blocks if isinstance(b, dict) and "record_version" in b]
    assert len(full) == 1 and record.validate(full[0]) == []
    offers = [b for b in blocks if isinstance(b, dict) and "apply_flag" in b]
    outputs = [b for b in blocks if isinstance(b, dict) and b.get("kind") == "searchable_copy"]
    assert offers and outputs
    for snippet in offers + outputs:
        data = copy.deepcopy(full[0])
        if "apply_flag" in snippet:
            data["offers"] = [snippet]
        else:
            data["output"], data["offers"] = snippet, []
        assert record.validate(data) == [], snippet
    assert "…" not in json.dumps(blocks, ensure_ascii=False)


def test_decision_record_doc_says_how_meta_diff_names_the_parts():
    # docs/decision-record.md -> "metadata names": Info appears whole only when unchanged; XMP is
    # always whole; a changed or dropped Info is listed key by key.
    from pdfintake.pdf.meta import MetaSnapshot, diff

    def snap(info: dict[str, str], xmp: str | None) -> MetaSnapshot:
        return MetaSnapshot(info=info, xmp_sha256=xmp, xmp_bytes=0 if xmp is None else 10, pdf_version="1.7")

    a, b = "ab" * 32, "cd" * 32
    same = diff(snap({"Title": "x"}, a), snap({"Title": "x"}, a))
    assert (same.kept, same.changed, same.dropped) == (["Info", "XMP"], [], [])
    gone = diff(snap({"Title": "x", "Author": "y"}, a), snap({}, None))
    assert "Info" not in gone.kept + gone.changed + gone.dropped
    assert sorted(gone.dropped) == ["Info/Author", "Info/Title", "XMP"]
    changed = diff(snap({"Producer": "x"}, a), snap({"Producer": "z"}, b))
    assert (changed.changed, changed.dropped) == (["Info/Producer", "XMP"], [])
    doc = (Path(__file__).resolve().parent.parent / "docs" / "decision-record.md").read_text(encoding="utf-8")
    assert "`Info` appears whole only when it is unchanged" in doc
    assert "kept, changed or dropped whole" not in doc
