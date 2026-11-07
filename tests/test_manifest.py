"""The pack manifest loader (langpacks/manifest.py): strict shape, vocabulary, sources, and agreement with jsonschema.

The fixture packs in ``tests/fixtures/packs/`` carry made-up pins. They are checked against the real
core registries; tests that need an engine the registry doesn't have use a fake registered through
the engine registry's test-only override.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake.errors import EXIT_SETUP, PackError
from pdfintake.langpacks import manifest
from pdfintake.langpacks.manifest import SOURCE_PATTERNS, check, parse
from pdfintake.ocr import engines

FX = Path(__file__).resolve().parent / "fixtures" / "packs"
SCHEMA_PATH = Path(manifest.__file__).with_name("manifest.schema.json")


def raw(pack: str) -> dict[str, Any]:
    return json.loads((FX / pack / "manifest.json").read_text(encoding="utf-8"))


def to_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


@pytest.fixture(scope="module")
def en() -> manifest.Manifest:
    return parse((FX / "en" / "manifest.json").read_bytes())


def problems(data: dict[str, Any], en: manifest.Manifest | None, **kw: Any) -> list[str]:
    return check(data, base=en, **kw)


def has(found: list[str], path: str, *words: str) -> bool:
    return any(p.startswith(path + ":") and all(w in p for w in words) for p in found)


# --------------------------------------------------------------------------- the fixture packs

def test_fixture_packs_parse(en: manifest.Manifest) -> None:
    assert en.base and en.id == "en" and en.canary is not None
    th = parse((FX / "th" / "manifest.json").read_bytes(), base=en)
    hu = parse((FX / "hu" / "manifest.json").read_bytes(), base=en)
    assert th.readers_for("win-amd64") is th.readers["default"]
    assert th.readers_for("darwin-arm64").pick.rule == "corroborated_trigrams"
    assert [u.id for u in th.readers_for("darwin-arm64").engines] == ["vision", "tesseract", "rapidocr"]
    assert th.detect.probe is not None and th.detect.probe.use.role == "probe"
    assert hu.detect.function_words >= {"és", "hogy"} and hu.detect.fw_min_hits == 3
    assert hu.detect.probe is None and th.detect.function_words == frozenset()
    assert th.raw_sha256 == __import__("hashlib").sha256((FX / "th" / "manifest.json").read_bytes()).hexdigest()
    # validated params carry the member's defaults and are read-only
    tess = th.readers["default"].engines[0]
    assert tess.params["psm"] == 3 and "timeout_s" in tess.params
    with pytest.raises(TypeError):
        tess.params["psm"] = 6  # type: ignore[index]
    assert th.models[0].notice == "NOTICE-tessdata"
    assert th.must_survive[0].unit == "trigram" and isinstance(th.must_survive[0].floor, float)


def test_source_patterns_are_the_schema_allowlist() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert SOURCE_PATTERNS["model"].pattern == schema["$defs"]["model"]["properties"]["source"]["pattern"]
    assert SOURCE_PATTERNS["font"].pattern == schema["$defs"]["font"]["properties"]["source"]["pattern"]


# --------------------------------------------------------------------------- every problem, with its path

def test_every_invalid_field_reported_at_once(en: manifest.Manifest) -> None:
    d = raw("th")
    d["id"] = "TH"
    d["colour"] = "blue"
    d["normalize"] = ["nfc", "shout"]
    d["models"][0]["sha256"] = "ABC"
    d["readers"]["default"]["engines"][0]["params"]["psm"] = 7
    d["readers"]["default"]["engines"][1]["engine"] = "nosuch"
    d["fills"].append({"name": "glitter", "version": 1})
    with pytest.raises(PackError) as exc:
        parse(to_bytes(d), base=en)
    err = exc.value
    assert err.code == "invalid_manifest" and err.exit_code == EXIT_SETUP
    found = err.detail["problems"]
    assert all(p.startswith("$") for p in found)
    assert has(found, "$.id")
    assert has(found, "$.colour", "unknown field")
    assert has(found, "$.normalize[1]", "shout")
    assert has(found, "$.models[0].sha256")
    assert has(found, "$.readers.default.engines[0].params.psm")
    assert has(found, "$.readers.default.engines[1].engine", "nosuch")
    assert has(found, "$.fills[2].name", "glitter")
    assert len(found) >= 7
    assert str(err).count(";") >= 6          # the message names them all too


def test_check_returns_empty_for_valid(en: manifest.Manifest) -> None:
    assert problems(raw("th"), en) == []
    assert problems(raw("hu"), en) == []
    assert check(raw("en")) == []


# --------------------------------------------------------------------------- registry checks

def test_unknown_rule_and_version(en: manifest.Manifest) -> None:
    d = raw("th")
    d["readers"]["default"]["pick"]["rule"] = "loudest"
    d["readers"]["darwin-arm64"]["pick"]["version"] = 9
    found = problems(d, en)
    assert has(found, "$.readers.default.pick.rule", "loudest")
    assert has(found, '$.readers["darwin-arm64"].pick.rule', "version 9")


def test_unknown_step_names_and_versions(en: manifest.Manifest) -> None:
    d = raw("hu")
    d["fills"] = [{"name": "numbers", "version": 2}]
    d["repairs"] = [{"name": "magic", "version": 1}]
    found = problems(d, en)
    assert has(found, "$.fills[0].version", "no version 2")
    assert has(found, "$.repairs[0].name", "magic")


def test_unknown_script_and_platform(en: manifest.Manifest) -> None:
    d = raw("th")
    d["scripts"] = ["Thai", "Klingon"]
    d["detect"]["osd_scripts"] = ["Elvish"]
    d["fonts"][0]["covers"] = ["Runic"]
    d["readers"]["amiga-m68k"] = copy.deepcopy(d["readers"]["default"])
    d["bench"]["platforms"] = {"beos-x86": {"metrics": {}, "worst_page": 0.5, "seconds_per_page": 1,
                                            "selection": {"policy": "maximin-v1", "margin": 0.1}}}
    found = problems(d, en)
    assert has(found, "$.scripts[1]", "Klingon")
    assert has(found, "$.detect.osd_scripts[0]", "Elvish")
    assert has(found, "$.fonts[0].covers[0]", "Runic")
    assert has(found, '$.readers["amiga-m68k"]', "platform")
    assert has(found, '$.bench.platforms["beos-x86"]', "platform")


def test_models_resolve_to_the_right_engine_and_kind(en: manifest.Manifest) -> None:
    d = raw("th")
    rs = d["readers"]["default"]["engines"]
    rs[0]["models"] = ["tha", "en:rapid-det", "nothere", "en:nothere", "tha"]
    d["models"][1]["kind"] = "traineddata"           # a traineddata file for rapidocr
    found = problems(d, en)
    assert has(found, "$.readers.default.engines[0].models[1]", "rapidocr", "not of 'tesseract'")
    assert has(found, "$.readers.default.engines[0].models[2]", "no model 'nothere'")
    assert has(found, "$.readers.default.engines[0].models[3]", "base pack has no model")
    assert has(found, "$.readers.default.engines[0].models[4]", "twice")
    assert has(found, "$.models[1].kind", "rapidocr")


def test_en_refs_need_the_base_manifest() -> None:
    found = check(raw("th"))
    assert has(found, "$.readers.default.engines[0].models[1]", "no base manifest")


def test_asset_param_must_be_one_of_the_uses_models(en: manifest.Manifest) -> None:
    d = raw("th")
    d["readers"]["default"]["engines"][1]["models"] = ["en:rapid-det", "en:rapid-cls"]
    found = problems(d, en)
    assert has(found, "$.readers.default.engines[1].params.rec", "isn't among this use's models")


def test_asset_param_kind_is_checked(en: manifest.Manifest) -> None:
    d = raw("th")
    use = d["readers"]["default"]["engines"][1]
    use["params"]["rec"] = "en:rapid-det"
    found = problems(d, en)
    assert has(found, "$.readers.default.engines[1].params.rec", "onnx_det")


# --------------------------------------------------------------------------- fills, picks, coverage

def test_words_fill_refused_on_thai(en: manifest.Manifest) -> None:
    d = raw("th")
    d["fills"].append({"name": "words", "version": 1})
    found = problems(d, en)
    assert has(found, "$.fills[2].name", "words", "Latin", "Thai")
    d = raw("hu")
    d["fills"].append({"name": "words", "version": 1})
    assert problems(d, en) == []


def test_corroborator_with_role_reader_refused(en: manifest.Manifest) -> None:
    d = raw("th")
    d["readers"]["darwin-arm64"]["pick"]["params"]["corroborators"] = ["tesseract"]
    found = problems(d, en)
    assert has(found, '$.readers["darwin-arm64"].pick.params.corroborators[0]', "reader")
    d["readers"]["darwin-arm64"]["pick"]["params"]["corroborators"] = ["rapidocr"]
    assert problems(d, en) == []


def test_pick_params_checked_against_reader_set(en: manifest.Manifest) -> None:
    d = raw("th")
    d["readers"]["default"]["pick"]["params"] = {"reader": "vision"}     # vision is only in the Mac set
    found = problems(d, en)
    assert has(found, "$.readers.default.pick.params.reader", "vision")


def test_default_reader_set_covers_every_supported_platform(en: manifest.Manifest) -> None:
    d = raw("th")
    vision = d["readers"]["darwin-arm64"]["engines"][0]
    d["readers"]["default"]["engines"].insert(0, copy.deepcopy(vision))
    d["readers"]["linux-x86_64"] = {"engines": [copy.deepcopy(vision), d["readers"]["default"]["engines"][1]],
                                    "pick": {"rule": "single", "version": 1, "params": {"reader": "vision"}},
                                    "fallback": "none"}
    found = problems(d, en)
    assert has(found, "$.readers.default.engines[0].engine", "vision", "win-amd64", "every supported platform")
    assert has(found, "$.readers.linux_x86_64.engines[0].engine", "linux-x86_64") or \
        has(found, '$.readers["linux-x86_64"].engines[0].engine', "linux-x86_64")


def test_detect_probe_covers_every_supported_platform(en: manifest.Manifest) -> None:
    """A probe runs on every platform during detection, so a Mac-only engine can't be the probe."""
    d = raw("th")
    vision = copy.deepcopy(d["readers"]["darwin-arm64"]["engines"][0])
    vision.update(id="probe", role="probe")
    d["detect"]["probe"]["use"] = vision
    found = problems(d, en)
    assert has(found, "$.detect.probe.use.engine", "vision", "win-amd64", "every supported platform")
    assert problems(raw("th"), en) == []


def test_reader_set_roles(en: manifest.Manifest) -> None:
    d = raw("th")
    eng = d["readers"]["default"]["engines"]
    eng[0]["role"] = "checker"
    d["readers"]["default"]["fallback"] = "checker"
    found = problems(d, en)
    assert has(found, "$.readers.default.engines", "role reader")
    assert has(found, "$.readers.default.engines", "more than one", "checker")
    d = raw("hu")
    d["readers"]["default"]["engines"][1]["role"] = "reader"
    d["readers"]["default"]["fallback"] = "checker"
    assert has(problems(d, en), "$.readers.default.fallback", "no engine use has role checker")
    d = raw("hu")
    d["readers"]["default"]["engines"][1]["role"] = "probe"
    assert has(problems(d, en), "$.readers.default.engines[1].role", "detect.probe")
    d = raw("hu")
    d["readers"]["default"]["engines"][1]["id"] = "tesseract"
    assert has(problems(d, en), "$.readers.default.engines[1].id", "twice")


def test_fake_engine_through_registry_override(monkeypatch: pytest.MonkeyPatch, en: manifest.Manifest) -> None:
    """An engine is vocabulary: a registered fake is accepted, and its platforms are what coverage checks."""
    fake = build.FakeEngine(platforms={"darwin-arm64"})
    monkeypatch.setitem(engines._REGISTRY, "fake", fake)
    d = raw("hu")
    d["readers"]["darwin-arm64"] = {
        "engines": [{"id": "fake", "engine": "fake", "role": "reader", "params": {"lang": "hu"}, "models": []}],
        "pick": {"rule": "single", "version": 1, "params": {"reader": "fake"}}, "fallback": "none"}
    assert problems(d, en) == []
    d["readers"]["default"]["engines"].append(
        {"id": "fake", "engine": "fake", "role": "checker", "params": {"lang": "hu"}, "models": []})
    d["readers"]["default"]["engines"][1]["role"] = "reader"
    assert has(problems(d, en), "$.readers.default.engines[2].engine", "fake")


# --------------------------------------------------------------------------- versions and the base pack

def test_requires_core_gate(en: manifest.Manifest) -> None:
    d = raw("hu")
    d["requires_core"] = ">=0.2.0"
    assert has(problems(d, en), "$.requires_core", "0.2.0", "0.1.0")
    assert problems(d, en, core_version="0.2.0") == []
    assert problems(d, en, core_version="1.0.0") == []
    with pytest.raises(PackError):
        parse(to_bytes(d), base=en)


def test_schema_version(en: manifest.Manifest) -> None:
    d = raw("hu")
    d["schema"] = 2
    assert has(problems(d, en), "$.schema", "must be 1")


def test_only_the_base_pack_is_base(en: manifest.Manifest) -> None:
    d = raw("th")
    d["base"] = True
    found = problems(d, en)
    assert has(found, "$.base", "only the base pack")
    assert has(found, "$.readers.default.engines[0].models[1]", "base pack names its own models")
    e = raw("en")
    del e["base"]
    assert has(check(e), "$.base", "must set base: true")


def test_tessdata_names_unique_against_base(en: manifest.Manifest) -> None:
    d = raw("hu")
    d["models"][0]["file"] = "eng.traineddata"
    assert has(problems(d, en), "$.models[0].file", "eng.traineddata", "unique")
    d = raw("th")
    d["models"].append(dict(d["models"][0], key="tha2"))
    found = problems(d, en)
    assert has(found, "$.models[2].file", "twice")


def test_file_names_compared_ignoring_case(en: manifest.Manifest) -> None:
    """The store and the tessdata directory may sit on a case-insensitive file system (decided 2026-11-05)."""
    d = raw("hu")
    d["models"][0]["file"] = "Eng.TrainedData"
    assert has(problems(d, en), "$.models[0].file", "'Eng.TrainedData'", "base pack", "ignoring case", "unique")
    d = raw("th")
    d["models"].append(dict(d["models"][0], key="tha2", file=d["models"][0]["file"].upper()))
    assert has(problems(d, en), "$.models[2].file", "twice", "ignoring case")
    d = raw("th")
    d["fonts"].append(dict(d["fonts"][0], key="noto-2", file=d["fonts"][0]["file"].swapcase()))
    assert has(problems(d, en), "$.fonts[1].file", "twice", "ignoring case")
    assert problems(raw("th"), en) == [] and problems(raw("hu"), en) == []


@pytest.mark.parametrize("name", [
    "tha.py", "tha.pth", "libtha.so", "libtha.so.1", "libtha.so.1.2", "tha.dylib", "tha.dll", "tha.pyd",
    "tha.PTH", "Tha.Py", "tha.traineddata.py", "tha.sh", "tha.jar",
])
def test_model_file_with_code_suffix_refused(en: manifest.Manifest, name: str) -> None:
    d = raw("hu")
    d["models"][0]["file"] = name
    found = [p for p in problems(d, en) if p.startswith("$.models[0].file:")]
    assert len(found) == 1, (name, found)          # once, in the loader's words, never the schema's regex
    assert "ends in a code suffix" in found[0] and "doesn't match" not in found[0], found
    with pytest.raises(PackError) as err:
        parse(to_bytes(d), base=en)
    assert [p for p in err.value.detail["problems"] if p.startswith("$.models[0].file:")] == found


def test_schema_pattern_still_refuses_what_it_names() -> None:
    """The schema keeps its own noCodeSuffix pattern for the six suffixes §11.2 names; only its
    message gives way to the loader's at the same path."""
    for name in ("tha.py", "tha.PTH", "libtha.so", "tha.dylib", "tha.dll", "tha.pyd"):
        d = raw("hu")
        d["models"][0]["file"] = name
        assert has(manifest._schema_problems(d), "$.models[0].file", "doesn't match"), name


def test_merge_keeps_an_unrelated_schema_problem_at_the_same_path(en: manifest.Manifest) -> None:
    d = raw("hu")
    d["models"][0]["file"] = "-tha.py"              # a bad first character as well as a code suffix
    found = [p for p in problems(d, en) if p.startswith("$.models[0].file:")]
    assert any("ends in a code suffix" in p for p in found)
    assert any("[A-Za-z0-9]" in p for p in found)   # the file-name pattern's own refusal stays
    assert not any("(?!" in p for p in found)       # the noCodeSuffix regex text doesn't


def test_loader_checks_stand_without_the_schema_patterns(monkeypatch: pytest.MonkeyPatch,
                                                          en: manifest.Manifest) -> None:
    """The code-suffix and blank must_find checks are the loader's own; the schema may state them too."""
    monkeypatch.setattr(manifest, "_schema_problems", lambda data: [])
    monkeypatch.setattr(manifest, "_schema_refuses_blank_must_find", lambda: False)
    d = raw("hu")
    d["models"][0]["file"] = "hun.pth"
    assert has(problems(d, en), "$.models[0].file", "code suffix")
    d = raw("th")
    d["canary"]["must_find"] = ["\u3000"]
    assert has(problems(d, en), "$.canary.must_find[0]", "other than whitespace")


# The loader refuses more code suffixes than the schema's noCodeSuffix pattern, which names only the
# six §11.2 lists. That split is deliberate (decided 2026-11-05): the loader is the gate every pack
# passes, so it may be stricter than the schema, never looser. These names are refused by the loader
# whatever the schema says, and stay out of the jsonschema agreement table above for that reason.
LOADER_ONLY_CODE_NAMES = [f"tha{s}" for s in manifest.CODE_SUFFIXES
                          if s not in (".py", ".pyd", ".pth", ".so", ".dylib", ".dll")] + [
    "libtha.so.1", "libtha.so.1.2", "THA.SH", "Tha.Exe"]


@pytest.mark.parametrize("name", LOADER_ONLY_CODE_NAMES)
def test_loader_refuses_code_suffixes_past_the_schemas_six(en: manifest.Manifest, name: str) -> None:
    d = raw("th")
    d["models"][1]["file"] = name
    found = [p for p in check(d, base=en) if p.startswith("$.models[1].file:")]
    assert len(found) == 1 and "ends in a code suffix" in found[0], (name, found)
    with pytest.raises(PackError):
        parse(to_bytes(d), base=en)


@pytest.mark.parametrize("name", ["tha.py", "Tha.Py", "tha.PTH", "libtha.so", "tha.dylib", "tha.dll",
                                  "tha.pyd", *LOADER_ONLY_CODE_NAMES,
                                  "hun.traineddata", "x.onnx", "a.sox", "so.traineddata", "tha.py.onnx"])
def test_schema_never_refuses_a_code_suffix_the_loader_allows(name: str) -> None:
    """The safe direction of the split: whatever noCodeSuffix refuses, is_code_name refuses too, so
    widening the schema later can only close the gap, never open one."""
    d = raw("th")
    d["models"][1]["file"] = name
    if any(p.startswith("$.models[1].file:") for p in manifest._schema_problems(d)):
        assert manifest.is_code_name(name), name


@pytest.mark.parametrize("name", ["hun.traineddata", "x.onnx", "solo.bin", "a.sox", "pyramid.onnx", "so.1"])
def test_model_file_without_code_suffix_passes(name: str) -> None:
    assert not manifest.is_code_name(name)


def test_function_word_defaults_are_langplans() -> None:
    """The function-word rule is the exact third (§11.2): Detect.fw_ratio for a manifest without a
    ratio must be what langplan applies."""
    from pdfintake.ocr import langplan

    assert manifest.FW_RATIO_DEFAULT == langplan.FW_RATIO_DEFAULT == 1 / 3 == 0.3333333333333333
    assert manifest.FW_MIN_HITS_DEFAULT == langplan.FW_MIN_HITS_DEFAULT
    d = raw("hu")
    d["detect"]["function_words"].pop("ratio", None)
    d["detect"]["function_words"].pop("min_hits", None)
    hu = parse(to_bytes(d), base=parse((FX / "en" / "manifest.json").read_bytes()))
    assert hu.detect.fw_ratio == langplan.FW_RATIO_DEFAULT and hu.detect.fw_min_hits == langplan.FW_MIN_HITS_DEFAULT


def test_pack_schema_doc_states_the_exact_third() -> None:
    doc = (Path(__file__).resolve().parents[1] / "docs" / "pack-schema.md").read_text(encoding="utf-8")
    assert "`ratio` 1/3, the exact third" in doc and "0.334" not in doc


def test_code_suffixes_match_the_wheel_builder() -> None:
    from pdfintake import datawheel

    assert set(manifest.CODE_SUFFIXES) == set(datawheel.CODE_SUFFIXES)
    assert {".py", ".pth", ".so", ".dylib", ".dll", ".pyd"} <= set(manifest.CODE_SUFFIXES)


@pytest.mark.parametrize("blank", [" ", "\t", "  \n ", "\u3000", "\u00a0"])
def test_canary_must_find_needs_a_non_whitespace_character(en: manifest.Manifest, blank: str) -> None:
    d = raw("th")
    d["canary"]["must_find"] = ["4521", blank]
    found = problems(d, en)
    assert has(found, "$.canary.must_find[1]"), found
    assert not has(found, "$.canary.must_find[0]")
    assert len([p for p in found if p.startswith("$.canary.must_find[1]:")]) == 1     # reported once
    with pytest.raises(PackError):
        parse(to_bytes(d), base=en)
    d["canary"]["must_find"] = ["4521", " a b "]
    assert problems(d, en) == []


def test_duplicate_keys_in_lists(en: manifest.Manifest) -> None:
    d = raw("th")
    d["languages"].append({"tag": "th", "iso639_3": "tha", "role": "secondary"})
    d["fills"].append({"name": "ids", "version": 1})
    d["must_survive"].append(dict(d["must_survive"][0]))
    d["models"].append(dict(d["models"][1], file="other.onnx"))
    found = problems(d, en)
    assert has(found, "$.languages[2].tag", "twice")
    assert has(found, "$.fills[2].name", "twice")
    assert has(found, "$.must_survive[1].name", "twice")
    assert has(found, "$.models[2].key", "twice")


def test_a_primary_language_is_required(en: manifest.Manifest) -> None:
    d = raw("th")
    d["languages"][0]["role"] = "secondary"
    assert has(problems(d, en), "$.languages", "primary")


# --------------------------------------------------------------------------- the source allowlist

@pytest.mark.parametrize("source", [
    "https://raw.example.com/tesseract-ocr/tessdata_best/raw/0123456789abcdef0123456789abcdef01234567/tha.traineddata",
    "https://github.com/tesseract-ocr/tessdata_best/raw/0123456789abcdef0123456789abcdef01234567/tha.traineddata?x=1",
    "https://github.com/tesseract-ocr/tessdata_best/raw/0123456789abcdef0123456789abcdef01234567/tha.traineddata#x",
    "https://github.com/RapidAI/RapidOCR/raw/0123456789abcdef0123456789abcdef01234567/models/../../evil.onnx",
    "http://github.com/tesseract-ocr/tessdata_best/raw/0123456789abcdef0123456789abcdef01234567/tha.traineddata",
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/tha.traineddata",
    "https://github.com/tesseract-ocr/tessdata_best/raw/0123456789abcdef0123456789abcdef01234567/tha.traineddata\n",
])
def test_model_source_refused(en: manifest.Manifest, source: str) -> None:
    d = raw("th")
    d["models"][0]["source"] = source
    assert has(problems(d, en), "$.models[0].source")


def test_query_and_dotdot_named(en: manifest.Manifest) -> None:
    d = raw("th")
    d["models"][0]["source"] += "?x=1"
    d["models"][1]["source"] = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/../x.onnx"
    found = problems(d, en)
    assert has(found, "$.models[0].source", "query")
    assert has(found, "$.models[1].source", "'..'")


def test_font_source_refused(en: manifest.Manifest) -> None:
    d = raw("th")
    d["fonts"][0]["source"] = "https://github.com/example/thai/releases/download/v1/NotoSansThai-Regular.ttf"
    assert has(problems(d, en), "$.fonts[0].source")
    d["fonts"][0]["source"] = "https://github.com/notofonts/thai/releases/download/v1/../NotoSansThai-Regular.ttf"
    assert has(problems(d, en), "$.fonts[0].source", "'..'")


def test_source_problems_helper() -> None:
    ok = "https://github.com/notofonts/thai/releases/download/fixture-v1/NotoSansThai-Regular.ttf"
    assert manifest.source_problems("font", ok) == []
    assert manifest.source_problems("model", ok) != []


# --------------------------------------------------------------------------- char_table parameters

def test_char_table_value_over_three_characters_refused(en: manifest.Manifest) -> None:
    d = raw("hu")
    table = {"ä": ["a", "á", "ä"], "ß": ["ß", "B", "ss", "fs", "8"]}
    d["repairs"] = [{"name": "diacritic_word", "version": 1, "params": {"source": "rapidocr", "table": table}}]
    assert problems(d, en) == []
    hu = parse(to_bytes(d), base=en)
    assert hu.repairs[0].params["table"]["ß"][2] == "ss"
    d["repairs"][0]["params"]["table"] = {"ß": ["ssss"]}
    assert has(problems(d, en), '$.repairs[0].params.table["ß"][0]', "1 to 3 characters")
    d["repairs"][0]["params"] = {"source": "vision", "table": table}
    assert has(problems(d, en), "$.repairs[0].params.source", "vision")


# --------------------------------------------------------------------------- strict JSON

@pytest.mark.parametrize("data, word", [
    (b'{"schema": 1, "schema": 1}', "twice"),
    (b'{"schema": NaN}', "NaN"),
    (b"\xef\xbb\xbf{}", "byte-order mark"),
    (b"[1, 2]", "JSON object"),
    (b"\xff\xfe", "UTF-8"),
    (b"{", "valid JSON"),
])
def test_strict_json(data: bytes, word: str) -> None:
    with pytest.raises(PackError) as exc:
        parse(data)
    assert exc.value.code == "invalid_manifest" and word in exc.value.message


def test_bytes_only_and_size_cap() -> None:
    with pytest.raises(PackError):
        parse("{}")  # type: ignore[arg-type]
    with pytest.raises(PackError) as exc:
        parse(b" " * (manifest.MAX_MANIFEST_BYTES + 1))
    assert "at most" in exc.value.message


def test_trailing_newline_never_passes_a_pattern(en: manifest.Manifest) -> None:
    """Stricter than jsonschema on purpose: Python's $ matches before a final newline."""
    d = raw("hu")
    d["id"] = "hu\n"
    assert has(problems(d, en), "$.id")


def test_schema_walker_refuses_unknown_keywords() -> None:
    with pytest.raises(ValueError, match="oneOf"):
        manifest._Schema({"type": "object", "properties": {"x": {"oneOf": []}}})


def test_malformed_parts_never_crash(en: manifest.Manifest) -> None:
    for broken in ({}, {"readers": []}, {"readers": {"default": 3}}, {"models": [1, None]},
                   {"detect": {"probe": {"use": 7}}}, {"fills": [{"name": 1}]}, {"repairs": "x"}):
        found = check(broken, base=en)
        assert found and all(isinstance(p, str) and p.startswith("$") for p in found)


# --------------------------------------------------------------------------- agreement with jsonschema

def _set(path: str, value: Any) -> Any:
    def op(d: dict[str, Any]) -> None:
        *head, last = path.split(".")
        node: Any = d
        for k in head:
            node = node[int(k)] if isinstance(node, list) else node[k]
        if isinstance(node, list):
            node[int(last)] = value
        elif value is _DEL:
            del node[last]
        else:
            node[last] = value
    return op


_DEL = object()
_BENCH_ROW = {"metrics": {"words": [0.9, 0.95, 1.0]}, "worst_page": 0.8, "seconds_per_page": 1.2,
              "selection": {"policy": "maximin-v1", "margin": 0.02}}

VALID = [
    lambda d: None,
    _set("title", "Thai (test)"),
    _set("version", "2.10.3"),
    _set("canary", _DEL),
    _set("base", False),
    _set("detect.langid.close_group", ["lo"]),
    _set("must_survive", []),
    _set("fills", []),
    _set("normalize", ["nfc"]),
    _set("bench.engine_versions", {"tesseract": "5.5.3"}),
    _set("bench.platforms", {"linux-x86_64": _BENCH_ROW}),
    _set("bench.private", {"pages": 3, "metrics": {"words": [0.9, 0.95, 1.0]}}),
    _set("models.0.bytes", 104857600),
    _set("canary.dpi", 150),
    _set("requires_core", ">=0.0.1"),
    _set("detect.probe.min_line_chars", _DEL),
    _set("models.0.notice", _DEL),
    _set("fonts", []),
    _set("readers.default.pick.version", 1.0),
    _set("languages", [{"tag": "th", "iso639_3": "tha", "role": "primary"},
                       {"tag": "en", "iso639_3": "eng", "role": "secondary"},
                       {"tag": "lo", "iso639_3": "lao", "role": "secondary"}]),
    # Names that only look like code: a code suffix inside the name, or a near miss (decided 2026-11-05).
    _set("models.1.file", "pyramid.onnx"),
    _set("models.1.file", "tha.py.onnx"),
    _set("models.1.file", "so.1"),
    _set("models.1.file", "a.sox"),
    _set("models.0.file", "so.traineddata"),
    _set("canary.must_find", ["4521", " x "]),
]
# Model file names: the six code suffixes the schema's noCodeSuffix names (any case), and names that
# pass noCodeSuffix but fail the base file pattern. Blank must_find entries. (decided 2026-11-05)
CODE_SUFFIX_NAMES = ["tha.py", "Tha.Py", "tha.PTH", "libtha.so", "tha.dylib", "tha.dll", "tha.pyd"]
BASE_PATTERN_MISSES = ["x/y.onnx", ".hidden", "a b.onnx"]
BLANK_MUST_FIND = [[" "], ["4521", "\t"], ["  \n "], ["\u3000"], ["\u00a0"]]

INVALID = [
    *(_set("models.1.file", name) for name in CODE_SUFFIX_NAMES),
    *(_set("models.1.file", name) for name in BASE_PATTERN_MISSES),
    *(_set("canary.must_find", blank) for blank in BLANK_MUST_FIND),
    _set("schema", 2),
    _set("id", "TH"),
    _set("version", "1.0"),
    _set("requires_core", "0.1.0"),
    _set("title", "thai"),
    _set("extra", True),
    _set("bench", _DEL),
    _set("languages", []),
    _set("scripts", ["Thai", "Thai"]),
    _set("models.0.sha256", "AB" * 32),
    _set("models.0.bytes", 0),
    _set("models.0.kind", "onnx"),
    _set("models.0.source", "https://raw.example.com/tha.traineddata"),
    _set("fonts.0.licence", "MIT"),
    _set("canary.file", "canary.png"),
    _set("must_survive.0.unit", "line"),
    _set("readers.default.engines.0.params.lang", None),
    _set("readers.default.fallback", "both"),
    _set("detect.function_words", {"words": ["x"], "min_hits": 0}),
    _set("readers.Default", {"engines": [], "pick": {}, "fallback": "none"}),
    _set("bench.corpus", "public"),
    _set("detect.probe.min_conf", 1.5),
    _set("readers.default.engines", []),
    _set("models.0.bytes", True),
]


def test_hand_validator_agrees_with_jsonschema(en: manifest.Manifest) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    cases = [(op, True) for op in VALID] + [(op, False) for op in INVALID]
    assert len(cases) >= 40
    for i, (op, expect_valid) in enumerate(cases):
        d = raw("th")
        op(d)
        js_ok = validator.is_valid(d)
        hand = check(d, base=en)
        assert js_ok == expect_valid, f"case {i}: the generator meant {expect_valid}, jsonschema says {js_ok}"
        assert (hand == []) == js_ok, f"case {i}: jsonschema says {js_ok}, the loader says {hand}"
        assert (manifest._schema_problems(d) == []) == js_ok, f"case {i}"
        if js_ok:
            parse(to_bytes(d), base=en)          # builds the frozen manifest too


@pytest.mark.parametrize("name", BASE_PATTERN_MISSES)
def test_names_past_no_code_suffix_still_need_the_base_file_pattern(en: manifest.Manifest, name: str) -> None:
    """noCodeSuffix only rules out code suffixes; a slash, a leading dot or a space is the base
    file-name pattern's to refuse, and the loader and jsonschema both refuse it."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert jsonschema.Draft202012Validator(schema["$defs"]["noCodeSuffix"]).is_valid(name)
    assert not manifest.is_code_name(name)
    d = raw("th")
    d["models"][1]["file"] = name
    assert not jsonschema.Draft202012Validator(schema).is_valid(d)
    assert has(problems(d, en), "$.models[1].file", "doesn't match")
    with pytest.raises(PackError):
        parse(to_bytes(d), base=en)


# --------------------------------------------------------------------------- the docs

def test_pack_schema_doc_lists_every_member_and_parameter() -> None:
    from pdfintake.ocr import fills, rules, textfix

    doc = (Path(__file__).resolve().parents[1] / "docs" / "pack-schema.md").read_text(encoding="utf-8")
    members: list[tuple[str, Any]] = [(n, engines.get(n)) for n in engines.names()]
    members += [(n, rules.get(n, v)) for n, v in rules.names()]
    members += [(n, m) for (n, _), m in fills.FILLS.items()]
    members += [(n, m) for (n, _), m in textfix.REPAIRS.items()]
    for name, member in members:
        assert f"`{name}`" in doc, f"docs/pack-schema.md doesn't list {name}"
        for p in member.PARAMS:
            assert f"`{p.name}`" in doc, f"docs/pack-schema.md doesn't list {name}.{p.name}"
    for name in textfix.NORMALIZERS:
        assert f"`{name}`" in doc
    for host in __import__("pdfintake.langpacks.index", fromlist=["ALLOWED_HOSTS"]).ALLOWED_HOSTS:
        assert f"`{host}`" in doc
