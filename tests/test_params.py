"""ParamTable: definitions refused when they'd let free text in, and every pack problem named with its path."""

from __future__ import annotations

import json
import math

import pytest

from pdfintake.errors import EXIT_SETUP, PackError
from pdfintake.params import (CHAR_TABLE_MAX_KEYS, CHAR_TABLE_MAX_VALUE_LEN, CHAR_TABLE_MAX_VALUES, PARAM_KINDS, Param,
                              ParamTable, ValidationContext)


def ctx(where: str = "readers.default.engines[0].params", **kw) -> ValidationContext:
    base = dict(use_ids=frozenset({"tesseract", "vision", "rapidocr"}),
                asset_keys=frozenset({"tha", "rapid-rec-th", "en:rapid-det", "en:rapid-rec-latin"}),
                asset_kinds={"tha": "traineddata", "rapid-rec-th": "onnx_rec", "en:rapid-det": "onnx_det",
                             "en:rapid-rec-latin": "onnx_rec"},
                scripts=frozenset({"Latin", "Thai"}), where=where)
    base.update(kw)
    return ValidationContext(**base)


def problems(table: ParamTable, raw, context: ValidationContext | None = None) -> list[str]:
    with pytest.raises(PackError) as info:
        table.validate(raw, context or ctx())
    err = info.value
    assert err.code == "invalid_params" and err.exit_code == EXIT_SETUP
    assert err.detail["problems"], "a PackError lists its problems"
    for p in err.detail["problems"]:
        assert p in err.message
    return err.detail["problems"]


TABLE = ParamTable(Param("table", "char_table", required=True))


# --------------------------------------------------------------------------- char_table


def test_char_table_accepts_the_spec_example():
    raw = {"table": {"ä": ["a", "á", "ä"], "Ä": ["A", "Á", "Ä"], "ß": ["ß", "B", "ss", "fs", "8"]}}
    assert TABLE.validate(raw, ctx()) == {"table": {"ä": ["a", "á", "ä"], "Ä": ["A", "Á", "Ä"], "ß": ["ß", "B", "ss", "fs", "8"]}}


def test_char_table_rejects_a_two_character_key():
    found = problems(TABLE, {"table": {"ss": ["ß"]}})
    assert any('table["ss"]' in p and "exactly one character" in p for p in found), found


def test_char_table_rejects_a_four_character_value():
    found = problems(TABLE, {"table": {"ä": ["a", "aeae"]}})
    assert any('["ä"][1]' in p and "1 to 3 characters" in p for p in found), found


def test_char_table_rejects_nine_values():
    found = problems(TABLE, {"table": {"a": [chr(0x61 + i) for i in range(9)]}})
    assert any("9 values" in p and f"at most {CHAR_TABLE_MAX_VALUES}" in p for p in found), found


def test_char_table_rejects_seventeen_keys():
    found = problems(TABLE, {"table": {chr(0x41 + i): ["x"] for i in range(17)}})
    assert any("17 keys" in p and f"at most {CHAR_TABLE_MAX_KEYS}" in p for p in found), found


def test_char_table_limits_are_the_spec_values():
    assert (CHAR_TABLE_MAX_KEYS, CHAR_TABLE_MAX_VALUES, CHAR_TABLE_MAX_VALUE_LEN) == (16, 8, 3)
    TABLE.validate({"table": {chr(0x41 + i): [chr(0x61 + j) for j in range(8)] for i in range(16)}}, ctx())
    TABLE.validate({"table": {"a": ["abc"]}}, ctx())


@pytest.mark.parametrize("bad", [
    {"a": "abc"},              # a string, not an array
    {"a": [""]},               # an empty variant would match everywhere
    {"a": ["a\x00"]},          # a control character
    {"a": [3]},                # not a string
    {"a": ["b", "b"]},         # listed twice
    ["a", "b"],                # not an object
])
def test_char_table_rejects_other_shapes(bad):
    assert problems(TABLE, {"table": bad})


def test_char_table_values_are_literals_not_regex():
    # ".*" is a legal literal value: the table never carries a pattern, consumers re.escape it.
    out = TABLE.validate({"table": {"a": [".*", "a+"]}}, ctx())
    assert out["table"]["a"] == [".*", "a+"]


# --------------------------------------------------------------------------- definitions


def test_str_without_pattern_or_choices_is_refused_at_definition():
    with pytest.raises(ValueError, match="pattern or a choices"):
        ParamTable(Param("lang", "str"))
    with pytest.raises(ValueError, match="pattern or a choices"):
        Param("languages", "str_list")
    ParamTable(Param("lang", "str", pattern=r"[a-z]{3}(\+[a-z]{3})*"), Param("level", "str", choices=("accurate", "fast")))


@pytest.mark.parametrize("kwargs, message", [
    (dict(name="Bad-Name", kind="int"), "parameter name"),
    (dict(name="x", kind="text"), "unknown kind"),
    (dict(name="x", kind="int", pattern="[0-9]+"), "pattern does not apply"),
    (dict(name="x", kind="bool", min=0), "min does not apply"),
    (dict(name="x", kind="use_ref", choices=("a",)), "choices does not apply"),
    (dict(name="x", kind="char_table", max_len=3), "max_len does not apply"),
    (dict(name="x", kind="str", pattern="(", ), "doesn't compile"),
    (dict(name="x", kind="int", min=5, max=1), "min is above max"),
    (dict(name="x", kind="int", required=True, default=3), "no default"),
    (dict(name="x", kind="int", default=700, max=600), "default is invalid"),
    (dict(name="x", kind="int", choices=("3",)), "not an integer"),
    (dict(name="x", kind="use_list", min=0.5), "whole number"),
    (dict(name="x", kind="str", pattern="a", max_len=0), "max_len"),
    (dict(name="x", kind="float", choices=()), "must not be empty"),
])
def test_bad_definitions_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Param(**kwargs)


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="twice"):
        ParamTable(Param("psm", "int"), Param("psm", "int"))


def test_every_kind_is_known():
    assert PARAM_KINDS == ("int", "float", "bool", "str", "str_list", "use_ref", "use_list", "asset_ref", "script", "char_table")


# --------------------------------------------------------------------------- validation

TESS = ParamTable(
    Param("lang", "str", required=True, pattern=r"[a-z]{3}(\+[a-z]{3})*", max_len=40),
    Param("psm", "int", choices=(3, 4, 6), default=3),
    Param("timeout_s", "float", min=10, max=600, default=300.0),
)


def test_defaults_fill_in_and_floats_come_back_as_floats():
    assert TESS.validate({"lang": "tha+eng"}, ctx()) == {"lang": "tha+eng", "psm": 3, "timeout_s": 300.0}
    out = TESS.validate({"lang": "hun", "timeout_s": 60}, ctx())
    assert out["timeout_s"] == 60.0 and isinstance(out["timeout_s"], float)


def test_every_problem_is_reported_at_once_with_its_path():
    found = problems(TESS, {"psm": 5, "timeout_s": 5, "colour": "red"}, ctx("readers.default.engines[1].params"))
    joined = "\n".join(found)
    assert "readers.default.engines[1].params.colour: unknown parameter" in joined
    assert "readers.default.engines[1].params.lang: required parameter is missing" in joined
    assert "readers.default.engines[1].params.psm: 5 isn't one of 3, 4, 6" in joined
    assert "readers.default.engines[1].params.timeout_s: must be at least 10" in joined
    assert len(found) == 4


@pytest.mark.parametrize("raw", [
    {"lang": "tha+eng; rm -rf /"},       # pattern
    {"lang": "x" * 41},                  # max_len and pattern
    {"lang": 3},
    {"lang": None},
    {"lang": "eng", "psm": True},        # a bool is not an int
    {"lang": "eng", "psm": 3.0},         # a float is not an int
    {"lang": "eng", "timeout_s": math.nan},
    {"lang": "eng", "timeout_s": "60"},
])
def test_wrong_values_are_refused(raw):
    assert problems(TESS, raw)


def test_not_an_object():
    assert "expected an object" in problems(TESS, ["lang", "eng"])[0]


def test_bool_float_and_str_list():
    table = ParamTable(
        Param("language_correction", "bool", default=True),
        Param("languages", "str_list", pattern=r"[a-z]{2}-[A-Z]{2}", min=1, max=4),
        Param("min_score", "float", min=0.0, max=1.0, default=0.5),
    )
    assert table.validate({"languages": ["th-TH", "en-US"]}, ctx()) == {
        "language_correction": True, "languages": ["th-TH", "en-US"], "min_score": 0.5}
    found = problems(table, {"language_correction": 1, "languages": ["th-TH", "th-TH", "thai"], "min_score": 1.5})
    joined = "\n".join(found)
    assert "language_correction: expected true or false" in joined
    assert "languages[1]" in joined and "twice" in joined
    assert "languages[2]" in joined and "doesn't match" in joined
    assert "min_score: must be at most 1" in joined
    assert "the number of items must be at least 1" in "\n".join(problems(table, {"languages": []}))
    assert "expected an array" in problems(table, {"languages": "th-TH"})[0]


def test_use_refs_point_into_the_reader_set():
    table = ParamTable(
        Param("primary", "use_ref", required=True),
        Param("tie_order", "use_list", min=1),
        Param("corroborators", "use_list", default=()),
    )
    out = table.validate({"primary": "tesseract", "tie_order": ["vision", "tesseract"]}, ctx())
    assert out == {"primary": "tesseract", "tie_order": ["vision", "tesseract"], "corroborators": []}
    found = "\n".join(problems(table, {"primary": "abbyy", "tie_order": ["vision", "vision", "other"],
                                       "corroborators": "rapidocr"}))
    assert "primary" in found and "abbyy" in found and "names no engine use" in found
    assert "tie_order[1]" in found and "twice" in found
    assert "tie_order[2]" in found and "other" in found
    assert "corroborators: expected an array" in found


def test_asset_refs_check_the_kind():
    table = ParamTable(Param("rec", "asset_ref", required=True, choices=("onnx_rec",)))
    assert table.validate({"rec": "rapid-rec-th"}, ctx()) == {"rec": "rapid-rec-th"}
    assert table.validate({"rec": "en:rapid-rec-latin"}, ctx()) == {"rec": "en:rapid-rec-latin"}
    assert "is a onnx_det" in problems(table, {"rec": "en:rapid-det"})[0]
    assert "names no model" in problems(table, {"rec": "rapid-rec-xx"})[0]


def test_scripts_are_core_names():
    table = ParamTable(Param("script", "script", required=True))
    assert table.validate({"script": "Thai"}, ctx()) == {"script": "Thai"}
    assert "not a script this core knows" in problems(table, {"script": "Klingon"})[0]
    narrow = ParamTable(Param("script", "script", required=True, choices=("Thai",)))
    assert "isn't one of Thai" in problems(narrow, {"script": "Latin"})[0]


def test_defaults_are_fresh_copies():
    table = ParamTable(Param("corroborators", "use_list", default=()),
                       Param("table", "char_table", default={"a": ("b",)}))
    first = table.validate({}, ctx())
    first["corroborators"].append("vision")
    first["table"]["a"].append("c")
    assert table.validate({}, ctx()) == {"corroborators": [], "table": {"a": ["b"]}}


def test_describe_lists_every_param_in_order():
    rows = TESS.describe()
    assert [r["name"] for r in rows] == ["lang", "psm", "timeout_s"]
    assert rows[0] == {"name": "lang", "kind": "str", "required": True, "default": None, "choices": None,
                       "min": None, "max": None, "pattern": r"[a-z]{3}(\+[a-z]{3})*", "max_len": 40}
    assert rows[1]["choices"] == [3, 4, 6] and rows[1]["default"] == 3
    json.dumps(rows)


def test_table_helpers():
    assert "psm" in TESS and "dpi" not in TESS
    assert len(TESS) == 3 and [p.name for p in TESS] == ["lang", "psm", "timeout_s"]
    assert ParamTable().validate({}, ctx()) == {}
    assert "no parameters" in problems(ParamTable(), {"x": 1})[0]


def test_validate_is_idempotent_and_never_returns_null():
    table = ParamTable(
        Param("rec", "asset_ref", required=True, choices=("onnx_rec",)),
        Param("use_cls", "bool", default=False),
        Param("min_score", "float", default=0.5, min=0, max=1),
        Param("note_key", "str", pattern=r"[a-z_]{1,20}"),            # optional, no default
        Param("psm", "int", choices=(3, 4, 6)),                        # optional, no default
        Param("tie_order", "use_list", min=1),                         # optional, no default
        Param("corroborators", "use_list", default=()),
        Param("table", "char_table", default={"a": ("b",)}),
        Param("script", "script"),
    )
    for raw in ({"rec": "en:rapid-rec-latin"},
                {"rec": "rapid-rec-th", "min_score": 1, "note_key": "made_up", "psm": 4,
                 "tie_order": ["vision", "tesseract"], "script": "Thai", "table": {"ß": ["ss", "B"]}}):
        once = table.validate(raw, ctx())
        assert None not in once.values(), "an absent optional param without a default is left out, not null"
        assert table.validate(once, ctx()) == once
        assert table.validate(json.loads(json.dumps(once)), ctx()) == once, "validated params survive a JSON round trip"
    once = table.validate({"rec": "en:rapid-rec-latin"}, ctx())
    assert set(once) == {"rec", "use_cls", "min_score", "corroborators", "table"}
