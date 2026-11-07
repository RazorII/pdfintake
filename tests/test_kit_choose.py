"""kit/choose.py and kit/bench.py: the maximin-v2 choice, the manifest it writes, the candidates and
the bench through ``pipeline.read_pages``, and every kit command end to end on a fake-engine pack.

Fake engines (``fixtures.build.FakeEngine``) are registered for each test; the reads are made up.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake import platforms
from pdfintake.errors import UsageError
from pdfintake.kit import bench as B
from pdfintake.kit import choose as CH
from pdfintake.kit import cli
from pdfintake.kit import render as R
from pdfintake.langpacks import manifest as M
from pdfintake.types import OsdResult
from test_kit_render import FONT2_URL, FONT_URL, TEXT, corpus_for, fake_pack, font_files, register_fake
from test_kit_rules import write_canary_env

FIXTURE_PACKS = Path(__file__).resolve().parent / "fixtures" / "packs"
CONDS = ("clean", "scan", "small")
KEY = platforms.current().key


def S(index: int, scores=(0.9, 0.9, 0.9), *, engines: int = 1, secs: float = 1.0, must: dict | None = None,
      ok: bool = True, rs: dict | None = None) -> dict[str, Any]:
    return {"index": index, "ok": ok, "engines": engines, "seconds_per_page": secs, "primary_metric": "words",
            "metrics": {"words": dict(zip(CONDS, scores)), "digits": dict(zip(CONDS, (1.0, 0.9, 0.8)))},
            "must": must or {}, "worst_page": min(scores), "reader_set": rs or {"set": index}}


def MS(name: str = "acute", floor: float = 0.8) -> M.MustSurvive:
    return M.MustSurvive(name, "é", "word", floor)


# --------------------------------------------------------------------------- maximin-v2

def test_maximin_takes_the_best_worst_condition():
    r = B.BenchResult(KEY, {}, [S(0, (0.99, 0.70, 0.95)), S(1, (0.90, 0.88, 0.89)), S(2, (0.80, 0.80, 0.80))])
    sel = CH.choose(r, [])
    assert (sel.chosen, sel.runner_up, sel.margin, sel.policy) == (1, 2, 0.08, "maximin-v2")


def test_ties_go_to_digits_then_fewer_engines_then_speed_then_order():
    three = S(3, (0.947,) * 3, engines=1, secs=1.0)
    three["metrics"]["digits"] = {"clean": 0.5, "scan": 0.5, "small": 0.5}      # reads differently from set 2
    r = B.BenchResult(KEY, {}, [S(0, (0.95,) * 3, engines=2, secs=0.5), S(1, (0.948,) * 3, engines=1, secs=2.0),
                                S(2, (0.947,) * 3, engines=1, secs=1.0), three])
    sel = CH.choose(r, [])
    # Equal digits (0.8 worst) among 0, 1 and 2: fewer engines, then speed. Set 3, as fast, reads
    # digits worse (0.5), so it comes after set 1 under maximin-v2 (maximin-v1 put it next).
    assert sel.chosen == 2 and sel.runner_up == 1 and sel.margin == -0.001
    # Sets 0 and 1 scored higher and lost on the tie-break: the selection says so, and on what.
    assert sel.ahead == [0, 1] and sel.chosen_score == 0.947
    assert [(x["index"], x["decided_by"]) for x in sel.tie_break["lost"]] == [(0, "engines"), (1, "speed")]
    assert sel.tie_break["chosen"]["digits"] == 0.8 and sel.tie_break["chosen"]["digits_cond"] == "small"
    assert CH.choose(B.BenchResult(KEY, {}, [S(0, (0.95,) * 3, engines=2), S(1, (0.94,) * 3)]), []).chosen == 0


def _pages(words: float) -> list[dict[str, Any]]:
    return [{"page": "prose-0-clean", "cond": "clean", "words": words, "trigrams": None, "digits": 1.0,
             "must": {}, "seconds": 1.0}]


def test_the_runner_up_reads_differently_from_the_chosen_set():
    """es set 0 and set 29 were two copies of RapidOCR alone, the same to four decimals: neither is
    the other's runner-up, and the margin between them said nothing (kit trial es, re-run R3)."""
    twin = S(29, (0.9454,) * 3, secs=2.10)
    twin["pages"] = _pages(0.9454)
    first = S(0, (0.9454,) * 3, secs=2.13)
    first["pages"] = [{**_pages(0.9454)[0], "seconds": 7.0}]            # seconds don't make a set read differently
    other = S(52, (0.9461,) * 3, engines=2, secs=2.5)
    third = S(8, (0.90,) * 3, secs=1.0)
    sel = CH.choose(B.BenchResult(KEY, {}, [first, other, third, twin]), [])
    assert sel.chosen == 29 and sel.runner_up == 52 and sel.margin == round(0.9454 - 0.9461, 4) < 0
    assert sel.ahead == [52] and sel.runner_up_score == 0.9461
    # Nothing else reads differently: no runner-up, margin 0.
    alone = CH.choose(B.BenchResult(KEY, {}, [first, twin]), [])
    assert alone.chosen == 29 and alone.runner_up is None and alone.margin == 0.0
    # A twin that read one page differently is a distinct reader.
    near = S(30, (0.9454,) * 3, secs=2.2)
    near["pages"] = _pages(0.9453)
    assert CH.choose(B.BenchResult(KEY, {}, [first, twin, near]), []).runner_up == 30


def test_the_runner_up_follows_the_policys_order_not_the_score():
    """kit trial it: a set scoring higher than the chosen one but with more engines lost the tie; the
    runner-up is the set the policy would have taken next, and ``ahead`` names the higher scorer."""
    chosen = S(29, (0.9361,) * 3, secs=9.9)
    higher = S(43, (0.9377,) * 3, engines=3, secs=12.0)
    two = S(52, (0.9343,) * 3, engines=2, secs=5.0)
    sel = CH.choose(B.BenchResult(KEY, {}, [chosen, higher, two]), [])
    assert (sel.chosen, sel.runner_up, sel.ahead) == (29, 52, [43])
    assert sel.margin == round(0.9361 - 0.9343, 4)


def test_a_floor_drops_a_set_and_no_survivor_is_exit_6():
    must = [MS()]
    r = B.BenchResult(KEY, {}, [S(0, (0.99,) * 3, must={"acute": {"clean": 0.9, "scan": 0.7, "small": None}}),
                                S(1, (0.90,) * 3, must={"acute": {"clean": 0.9, "scan": 0.85, "small": 0.8}}),
                                S(2, (1.0,) * 3, ok=False)])
    assert CH.choose(r, must).chosen == 1
    with pytest.raises(CH.FloorFailed) as exc:
        CH.choose(B.BenchResult(KEY, {}, [r.sets[0]]), must)
    assert exc.value.exit_code == 6


def _rs(lang: str, rule: str = "single") -> dict[str, Any]:
    return {"engines": [{"id": "tesseract", "engine": "tesseract", "role": "reader", "params": {"lang": lang, "psm": 3},
                         "models": []}], "pick": {"rule": rule, "version": 1, "params": {}}, "fallback": "none"}


def test_exit_6_names_the_closest_sets_and_where_each_misses():
    """kit trial pl, item 8: the error listed the eight lowest failures (English-model sets at 0.000
    on clean pages) while the sets that came closest failed only on small pages, by 0.02."""
    must = [M.MustSurvive("ogonek", "ąę", "word", 0.9)]
    eng = [S(i, (0.5,) * 3, must={"ogonek": {"clean": 0.0, "scan": 0.0, "small": 0.0}}, rs=_rs("eng")) for i in (1, 2)]
    pol = S(8, (0.97, 0.97, 0.86), must={"ogonek": {"clean": 0.999, "scan": 0.991, "small": 0.8817}}, rs=_rs("pol"))
    pol_twin = dict(pol, index=9)
    pair = S(44, (0.97, 0.96, 0.88), engines=2, rs=_rs("pol", "corroborated_trigrams"),
             must={"ogonek": {"clean": 0.9965, "scan": 0.9641, "small": 0.8836}})
    two_misses = S(15, (0.92, 0.93, 0.84), rs=_rs("pol"), must={"ogonek": {"clean": 0.95, "scan": 0.89, "small": 0.89}})
    r = B.BenchResult(KEY, {}, [*eng, pol, pol_twin, pair, two_misses, S(3, ok=False)])
    with pytest.raises(CH.FloorFailed) as exc:
        CH.choose(r, must)
    closest = exc.value.detail["closest"]
    assert [c["index"] for c in closest] == [15, 44, 8]          # largest shortfall 0.011, then 0.0164, 0.0183
    assert closest[0]["misses"] == [{"name": "ogonek", "cond": "scan", "value": 0.89, "floor": 0.9},
                                    {"name": "ogonek", "cond": "small", "value": 0.89, "floor": 0.9}]
    assert closest[2]["reads_alike"] == [9] and closest[2]["reader_set"] == "tesseract[pol]"
    assert closest[1]["pick"] == "corroborated_trigrams"
    msg = exc.value.message
    assert "set 44 (tesseract[pol], corroborated_trigrams): ogonek small 0.8836 < 0.9" in msg
    assert "sets 9 read the same" in msg and "set 1 " not in msg and "clean 0.000" not in msg
    assert exc.value.detail["problems"][0].startswith("set 15 (tesseract[pol], single): ogonek scan 0.8900 < 0.9")
    with pytest.raises(CH.FloorFailed) as none:
        CH.choose(B.BenchResult(KEY, {}, [S(0, ok=False)]), must)
    assert "no set ran" in none.value.message and none.value.detail["closest"] == []


def test_an_override_needs_a_reason_and_a_floor():
    must = [MS()]
    r = B.BenchResult(KEY, {}, [S(0, (0.99,) * 3), S(1, (0.90,) * 3),
                                S(2, (0.95,) * 3, must={"acute": {"clean": 0.1, "scan": 0.1, "small": 0.1}})])
    with pytest.raises(UsageError):
        CH.choose(r, must, override=1)
    sel = CH.choose(r, must, override=1, override_reason="Faster on long documents.")
    assert sel.chosen == 1 and sel.override_reason == "Faster on long documents." and sel.margin < 0
    with pytest.raises(CH.FloorFailed):
        CH.choose(r, must, override=2, override_reason="x")


def _hu_raw() -> dict[str, Any]:
    return json.loads((FIXTURE_PACKS / "hu" / "manifest.json").read_text(encoding="utf-8"))


def test_write_manifest_puts_a_mac_only_set_under_its_platform():
    raw = _hu_raw()
    tess_set = raw["readers"]["default"]
    vision_set = {"engines": [{"id": "vision", "engine": "vision", "role": "reader",
                               "params": {"languages": ["hu-HU"], "revision": 3, "level": "accurate",
                                          "language_correction": True}, "models": []},
                              tess_set["engines"][0]],
                  "pick": {"rule": "corroborated_trigrams", "version": 1,
                           "params": {"script": "Latin", "tie_order": ["vision", "tesseract"], "corroborators": []}},
                  "fallback": "none"}
    results = {"linux-x86_64": B.BenchResult("linux-x86_64", {"tesseract": "5.5.3"}, [S(0, rs=tess_set)]),
               "darwin-arm64": B.BenchResult("darwin-arm64", {"vision": "rev3"}, [S(0, rs=tess_set),
                                                                                   S(1, (0.97,) * 3, rs=vision_set)])}
    sels = {p: CH.choose(r, []) for p, r in results.items()}
    out = CH.write_manifest(raw, results, sels, corpus="public-v1@" + "ab" * 32)
    assert out["readers"]["default"] == tess_set and out["readers"]["darwin-arm64"] == vision_set
    claim = out["bench"]["platforms"]["darwin-arm64"]
    assert claim["metrics"]["words"] == [0.97, 0.97, 0.97] and claim["selection"] == {"policy": "maximin-v2", "margin": 0.07}
    assert out["bench"]["engine_versions"] == {"tesseract": "5.5.3", "vision": "rev3"}
    base = M.parse((FIXTURE_PACKS / "en" / "manifest.json").read_bytes())
    assert M.check(out, base=base) == []


# --------------------------------------------------------------------------- candidates and the bench

def test_candidates_combine_every_engine_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    packs, _work, _web = fake_pack(tmp_path, monkeypatch, second=True)
    m, raw, _ = B.load_manifest(packs, "en")
    jsons = B.candidate_jsons(m, KEY, None)
    assert jsons[0] == B.reader_set_json(m.readers_for(KEY))
    rules = {s["pick"]["rule"] for s in jsons}
    assert rules == {"single", "primary_with_floor", "corroborated_trigrams"}
    assert len({json.dumps(s, sort_keys=True) for s in jsons}) == len(jsons)
    sets = B.candidates(m, KEY)
    assert len(sets) == len(jsons)
    assert any(len(s.engines) == 2 and s.pick.rule == "corroborated_trigrams" for s in sets)


def test_bench_reads_through_the_pipeline_and_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=TEXT.split(". ")[:30])
    m, raw, _ = B.load_manifest(packs, "en")
    pack = B.installed_pack(m, B.files_dir(work, "en"), work / "en" / "pack")
    c = corpus_for(pages={"prose": 1, "form": 1, "microprint": 0, "mixed_script": 0})
    (tmp_path / "f").mkdir()
    pages = R.render_pages(c, font_files(tmp_path / "f"), tmp_path / "pages", texts={c.texts[0].sha256: TEXT})
    calls: list[int] = []
    from pdfintake.ocr import pipeline

    real = pipeline.read_pages

    def spy(*a: Any, **k: Any) -> Any:
        calls.append(len(a[0]))
        assert k["signals"] is None
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "read_pages", spy)
    result = B.bench(pages, pack, pack, KEY, B.candidates(m, KEY))
    assert calls == [len(pages)] * len(result.sets)
    s = result.sets[0]
    assert s["ok"] and set(s["metrics"]["words"]) == set(CONDS) and 0.0 <= s["worst_page"] <= 1.0
    assert len(s["pages"]) == len(pages) and result.engine_versions["fake"] == "1.0"
    assert json.loads(json.dumps(B.result_json(result)))["platform"] == KEY


def test_mixed_script_sweep_writes_only_numbers(tmp_path: Path):
    (tmp_path / "f").mkdir()
    c = corpus_for(pages={"prose": 1, "form": 0, "microprint": 0, "mixed_script": 1})
    pages = R.render_pages(c, font_files(tmp_path / "f", thai=True), tmp_path / "pages",
                           texts={c.texts[0].sha256: TEXT})

    class Osd:
        def detect(self, image: Any, dpi: float) -> OsdResult:
            return OsdResult("Thai", 9.0, 0, 5.0)

    reads = []
    for p in pages:
        low = p.kind == "mixed_script"
        reads.append(build.engine_read("eng", ["some line", "another line"], conf=0.3 if low else 0.97))
    out = B.mixed_script_sweep(pages, None, platform_key=KEY, osd=Osd(), eng_reads=reads,  # type: ignore[arg-type]
                               corpus_ref="test-v1@" + "ab" * 32)
    assert out["pages"] == {"mixed": 3, "latin": 3, "not_counted": 0}
    assert out["best"] is not None and out["best"]["recall"] == 1.0 and out["best"]["false_positive_rate"] == 0.0
    # The string calibration.json carries; the loader's pattern accepts it.
    from pdfintake.ocr.langplan import _MEASURED_RE

    assert out.pop("measured") == f"kit {B.KIT_VERSION} test-v1@" + "ab" * 32
    assert _MEASURED_RE.fullmatch(f"kit {B.KIT_VERSION} test-v1@" + "ab" * 32)

    def numbers(v: Any) -> bool:
        if isinstance(v, dict):
            return all(isinstance(k, str) and numbers(x) for k, x in v.items())
        if isinstance(v, list):
            return all(numbers(x) for x in v)
        return v is None or isinstance(v, (int, float))

    assert numbers(out)
    # A non-Latin page that isn't a mixed-script page (a Thai pack's prose) is neither a positive nor
    # a negative: the trigger exists for non-Latin lines on Latin pages (decided 2026-11-05).
    import dataclasses

    thai = next(p for p in pages if p.kind == "mixed_script")
    as_prose = [dataclasses.replace(p, kind="prose") if p.kind == "mixed_script" else p for p in pages]
    out2 = B.mixed_script_sweep(as_prose, None, platform_key=KEY, osd=Osd(), eng_reads=reads)  # type: ignore[arg-type]
    assert out2["pages"] == {"mixed": 0, "latin": 3, "not_counted": 3} and "measured" not in out2
    assert all(g["recall"] is None for g in out2["grid"]) and thai.kind == "mixed_script"


# --------------------------------------------------------------------------- every command, end to end

def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = cli.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


def test_every_kit_command_on_a_fake_engine_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"], second=True)
    write_canary_env(tmp_path)                          # kit wheel compares the canary libraries
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    code, out = _run(capsys, "kit", "corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL,
                     "--font", FONT2_URL, "--pages", "prose=1,form=1,microprint=0,mixed_script=0")
    assert code == 0, out
    code, out = _run(capsys, "kit", "render", "en", *common)
    assert code == 0 and out["pages"] == 6, out
    code, out = _run(capsys, "kit", "candidates", "en", *common)
    assert code == 0 and out["count"] >= 4, out
    # The fake engine's own proposal is the draft's set 0 once the loader fills its defaults: merged.
    assert out["candidates"]["cut"] == 0 and out["candidates"]["total"] == out["count"] + out["candidates"]["merged"]
    assert out["candidates"]["merged"] == 1 and list(out["candidates"]["merged_sets"]) == ["0"]
    code, out = _run(capsys, "kit", "bench", "en", *common, "--offline")
    assert code == 0 and all(s["ok"] for s in out["sets"]), out
    bench = json.loads((packs / "en" / "bench.json").read_text(encoding="utf-8"))
    assert list(bench["platforms"]) == [KEY] and bench["corpus"].startswith("public-v1@")
    code, out = _run(capsys, "kit", "choose", "en", *common)
    assert code == 0 and KEY in out["selections"], out
    manifest = json.loads((packs / "en" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bench"]["platforms"][KEY]["selection"]["policy"] == "maximin-v2"
    assert manifest["bench"]["corpus"] == bench["corpus"]
    code, out = _run(capsys, "kit", "wheel", "en", *common, "--offline", "--must-find", "example", "--must-find", "4521")
    assert code == 0, out
    assert out["canary"]["must_find"] == ["example", "4521"] and all(r["ok"] for r in out["canary_reads"])
    assert Path(out["wheel"]).name == "pdfintake_data_en-1.0.0-py3-none-any.whl"
    wheel = out
    code, out = _run(capsys, "kit", "verify", "en", *common)
    assert code == 0 and out["ok"], out
    assert out["checks"]["manifest"] == "ok" and out["checks"]["layer"] == "ok" and out["checks"]["wheel"] == "ok"
    # The real re-bench, as CI runs it: it re-renders with the kit's seed and matches the claim.
    code, out = _run(capsys, "kit", "verify", "en", *common, "--rebench")
    assert code == 0 and out["ok"], out
    assert out["rebench"]["drift"] == [] and out["rebench"]["floors"] == [] and out["rebench"]["claimed"]
    # Re-running choose after wheel keeps the canary (the reader sets didn't change).
    code, out = _run(capsys, "kit", "choose", "en", *common)
    assert code == 0 and out["canary"] == "kept", out
    code, out = _run(capsys, "kit", "verify", "en", *common)
    assert code == 0 and out["ok"], out
    code, out = _run(capsys, "kit", "report", "en", *common)
    assert code == 0 and "maximin-v2" in out["body"] and "packs/en/**" in out["body"]
    # the index tool takes the built wheel
    from test_kit_verify import load_build_index

    bi = load_build_index()
    index = tmp_path / "index.json"
    dist = work / "en" / "dist"
    assert bi.main(["--packs", str(packs), "--index", str(index), "--wheels", str(dist)]) == 0
    assert bi.main(["--packs", str(packs), "--index", str(index), "--wheels", str(dist), "--check"]) == 0
    entry = json.loads(index.read_text(encoding="utf-8"))["packs"][0]
    assert entry["dist"]["sha256"] == wheel["sha256"] and entry["dist"]["kind"] == "installed"


def test_choose_without_a_bench_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    packs = tmp_path / "packs"
    shutil.copytree(FIXTURE_PACKS, packs)
    code, out = _run(capsys, "kit", "choose", "th", "--packs", str(packs))
    assert code == 2 and out["error"]["code"] == "kit_no_bench"


def test_rebench_drift_is_exit_7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"])
    write_canary_env(tmp_path)                          # kit wheel compares the canary libraries
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    for argv in (["corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL, "--font", FONT2_URL,
                  "--pages", "prose=1,form=1,microprint=0,mixed_script=0"],
                 ["render", "en", *common], ["bench", "en", *common, "--offline", "--current-only"],
                 ["choose", "en", *common],
                 ["wheel", "en", *common, "--offline", "--must-find", "example", "--must-find", "4521"]):
        code, out = _run(capsys, "kit", *argv)
        assert code == 0, out
    mfile = packs / "en" / "manifest.json"
    raw = json.loads(mfile.read_text(encoding="utf-8"))
    claim = raw["bench"]["platforms"][KEY]["metrics"]
    name = sorted(claim)[0]
    claim[name] = [round(v - 0.5 if v >= 0.5 else v + 0.5, 4) for v in claim[name]]
    mfile.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    code, out = _run(capsys, "kit", "verify", "en", *common, "--rebench")
    assert code == 7 and not out["ok"], out
    assert len(out["rebench"]["drift"]) == 3 and all(d.startswith(f"{name} ") for d in out["rebench"]["drift"])
    assert out["rebench"]["floors"] == []


def test_rechoose_drops_the_canary_only_when_the_readers_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"])
    write_canary_env(tmp_path)                          # kit wheel compares the canary libraries
    shutil.copyfile(packs / "en" / "manifest.json", packs / "en" / "manifest.draft.json")    # as kit init leaves it
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    for argv in (["corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL, "--font", FONT2_URL,
                  "--pages", "prose=1,form=0,microprint=0,mixed_script=0"],
                 ["render", "en", *common], ["bench", "en", *common, "--offline"], ["choose", "en", *common],
                 ["wheel", "en", *common, "--offline", "--must-find", "example", "--must-find", "4521"]):
        code, out = _run(capsys, "kit", *argv)
        assert code == 0, out
    mfile = packs / "en" / "manifest.json"
    raw = json.loads(mfile.read_text(encoding="utf-8"))
    canary = raw["canary"]
    private = {"pages": 3, "metrics": {"words": [0.9, 0.8, 1.0]}}
    raw["bench"]["private"] = private                                    # as kit private writes it
    mfile.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    code, out = _run(capsys, "kit", "choose", "en", *common)
    assert code == 0 and out["canary"] == "kept" and out["next"] == [], out
    again = json.loads(mfile.read_text(encoding="utf-8"))
    assert again["canary"] == canary and again["bench"]["private"] == private
    code, out = _run(capsys, "kit", "verify", "en", *common)
    assert code == 0 and out["ok"], out
    # The readers in manifest.json differ from what choose picks now: the canary was read with
    # the old ones, so it goes and wheel must run again; the private block stays.
    again["readers"]["default"]["engines"][0]["params"] = {"lang": "en", "psm": 6}
    mfile.write_text(json.dumps(again, indent=2), encoding="utf-8")
    code, out = _run(capsys, "kit", "choose", "en", *common)
    assert code == 0 and out["canary"] == "dropped" and "kit wheel" in out["next"][0], out
    final = json.loads(mfile.read_text(encoding="utf-8"))
    assert "canary" not in final and final["bench"]["private"] == private


def test_the_candidate_ceiling_applies_per_rule_and_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, second=True)
    register_fake(monkeypatch, "fakec")
    register_fake(monkeypatch, "faked")
    m, _raw, _ = B.load_manifest(packs, "en")
    everything, info = B.candidate_sets(m, KEY, None)
    assert info["cut"] == 0 and info["total"] == len(everything) + info["merged"]
    per_rule: dict[str, int] = {}
    for s in everything:
        per_rule[s["pick"]["rule"]] = per_rule.get(s["pick"]["rule"], 0) + 1
    assert set(per_rule) == {"single", "primary_with_floor", "corroborated_trigrams"}
    cap = 3
    assert all(n > cap for n in per_rule.values())                       # every rule is over the ceiling
    monkeypatch.setattr(B, "MAX_SETS_PER_RULE", cap)
    kept, info = B.candidate_sets(m, KEY, None)
    assert kept[0] == everything[0]                                      # the draft's own set stays first
    rules = [s["pick"]["rule"] for s in kept]
    assert set(rules) == set(per_rule)                                   # no rule is crowded out
    assert all(rules[1:].count(r) <= cap for r in per_rule)
    assert info["total"] == len(everything) + info["merged"] and info["kept"] == len(kept)
    assert info["cut"] == len(everything) - len(kept) > 0
    assert info["cut_by_rule"] == {r: n - cap for r, n in per_rule.items()}      # the draft's set counts in its rule
    code, out = _run(capsys, "kit", "candidates", "en", "--packs", str(packs), "--work", str(work))
    assert code == 0 and out["candidates"]["cut"] == info["cut"] and out["count"] == len(kept)
    listed = json.loads((work / "en" / f"candidates-{KEY}.json").read_text(encoding="utf-8"))
    assert listed["candidates"]["cut_by_rule"] == info["cut_by_rule"]


def test_a_failed_set_keeps_only_its_error_code_in_bench_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pdfintake.errors import PackError
    from pdfintake.kit import verify as V
    from pdfintake.ocr import pipeline

    packs, work, _web = fake_pack(tmp_path, monkeypatch)
    m, _raw, _ = B.load_manifest(packs, "en")
    pack = B.installed_pack(m, B.files_dir(work, "en"), work / "en" / "pack")
    c = corpus_for(pages={"prose": 1, "form": 0, "microprint": 0, "mixed_script": 0})
    (tmp_path / "f").mkdir()
    pages = R.render_pages(c, font_files(tmp_path / "f"), tmp_path / "pages", texts={c.texts[0].sha256: TEXT})

    def broken(*a: Any, **k: Any) -> Any:
        raise PackError("model_hash", "/var/example-cache/pdfintake/models/x.onnx has SHA-256 00")

    monkeypatch.setattr(pipeline, "read_pages", broken)
    result = B.bench(pages, pack, pack, KEY, B.candidates(m, KEY)[:1])
    assert result.sets[0]["ok"] is False and result.sets[0]["error"] == {"code": "model_hash"}
    assert "example-cache" in result.messages[0]
    written = json.dumps(B.result_json(result))
    assert "example-cache" not in written
    assert V._bench_file_problems({"platforms": {KEY: B.result_json(result)}}) == []
    leaky = {"platforms": {KEY: {"sets": [{"error": {"code": "model_hash", "message": "/var/example-cache/x"}}]}}}
    assert V._bench_file_problems(leaky) == [
        "bench.json.platforms.%s.sets[0].error holds more than an error code; bench.json holds numbers only" % KEY]


@pytest.mark.parametrize("command", ["choose", "bench", "report", "internal"])
def test_a_malformed_kit_file_is_one_json_error(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]):
    packs = tmp_path / "packs"
    shutil.copytree(FIXTURE_PACKS, packs)
    work = tmp_path / "work"
    argv = ["kit", command, "th", "--packs", str(packs), "--work", str(work)]
    if command == "choose":
        (packs / "th" / "bench.json").write_text("{not json", encoding="utf-8")
        want = "kit_bench_invalid"
    elif command == "bench":
        packs2, work, _web = fake_pack(tmp_path / "b", monkeypatch)
        (work / "en").mkdir(parents=True, exist_ok=True)
        (work / "en" / f"candidates-{KEY}.json").write_text("{not json", encoding="utf-8")
        (packs2 / "en" / "corpus.json").write_text("{}", encoding="utf-8")
        pages = work / "en" / "pages"
        pages.mkdir(parents=True)
        (pages / R.PAGES_FILE).write_text(json.dumps({"seed": 0, "pages": []}), encoding="utf-8")
        argv = ["kit", "bench", "en", "--packs", str(packs2), "--work", str(work), "--offline"]
        want = "kit_candidates_invalid"
    elif command == "report":
        (packs / "th" / "corpus.json").write_text("[1, 2", encoding="utf-8")
        want = "invalid_corpus"
    else:
        def boom(args: Any) -> Any:
            raise KeyError("platforms")

        monkeypatch.setitem(cli.COMMANDS, "report", boom)
        argv = ["kit", "report", "th", "--packs", str(packs)]
        want = "kit_internal"
    code = cli.main(argv)
    out = json.loads(capsys.readouterr().out)                            # one JSON object, no traceback
    assert code == 1 and out["ok"] is False and out["error"]["code"] == want, out
