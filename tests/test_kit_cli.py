"""The kit polish from the Wave D pack trials (docs/kit-trial-{es,fr,it,pt,pl,vi}.md): the notice
``kit init`` writes and ``kit choose`` keeps in step, ``--must-find``'s limit, labelled uses and
progress lines, ``kit init --from-manifest``, and what ``kit report`` shows.

Every pack, text and number here is made up; downloads are answered by ``FakeWeb``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pdfintake import platforms
from pdfintake.kit import bench as B
from pdfintake.kit import choose as CH
from pdfintake.kit import cli
from pdfintake.kit import corpus as C
from pdfintake.kit import render as R
from pdfintake.kit import report as REP
from pdfintake.kit import verify as V
from pdfintake.langpacks import manifest as M
from test_kit_render import FakeWeb

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PACKS = ROOT / "tests" / "fixtures" / "packs"
REAL_PACKS = ROOT / "packs"
KEY = platforms.current().key
CONDS = ("clean", "scan", "small")
TEMPLATE = (CH.NOTICE_HEADER + "\n"
            "  osd.traineddata  https://example.com/tessdata/raw/0000/osd.traineddata\n"
            "  eng.traineddata  https://example.com/tessdata/raw/0000/eng.traineddata\n"
            "\n"
            "The licence text below is an example licence.\n"
            "\n"
            "Example licence text.\n")
RAPID_ONLY = {"engines": [{"id": "rapidocr", "engine": "rapidocr", "role": "reader", "params": {"rec": "en:rapid-rec-latin"},
                           "models": ["en:rapid-det", "en:rapid-cls", "en:rapid-rec-latin"]}],
              "pick": {"rule": "single", "version": 1, "params": {"reader": "rapidocr"}}, "fallback": "none"}


def _packs(tmp: Path, *, template: str | None = TEMPLATE) -> Path:
    dest = tmp / "packs"
    shutil.copytree(FIXTURE_PACKS, dest)
    if template is not None:
        (dest / "en" / CH.NOTICE_TESSDATA).write_text(template, encoding="utf-8")
    return dest


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    code = cli.main(list(argv))
    got = capsys.readouterr()
    return code, json.loads(got.out), got.err


def _hu_tess() -> dict[str, Any]:
    return json.loads((FIXTURE_PACKS / "hu" / "manifest.json").read_text(encoding="utf-8"))["readers"]["default"]


# --------------------------------------------------------------------------- (1) NOTICE-tessdata

def test_init_writes_a_notice_naming_only_the_packs_own_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]):
    """kit init copied packs/en's notice verbatim, so the new pack's notice named osd and eng (kit
    trial vi shipped like that until it was fixed by hand)."""
    packs = _packs(tmp_path)
    url = cli.TESSDATA_URL.format(commit=cli.TESSDATA_BEST_COMMIT, name="vie")
    monkeypatch.setattr(C, "_http_get", FakeWeb({url: b"made-up vie model"}))
    code, out, _err = _run(capsys, "kit", "init", "vi", "--script", "Latin", "--packs", str(packs),
                           "--must-survive", "letters=ăâđ:word:0.9")
    assert code == 0 and out["notice"] == "written", out
    text = (packs / "vi" / CH.NOTICE_TESSDATA).read_text(encoding="utf-8")
    assert f"  vie.traineddata  {url}\n" in text
    assert "osd.traineddata" not in text and "eng.traineddata" not in text
    assert text.startswith(CH.NOTICE_HEADER) and text.endswith("The licence text below is an example licence.\n\n"
                                                               "Example licence text.\n")
    # the notice holds only the draft's model names: the same check the contract test runs on real packs
    assert set(V._NOTICE_MODEL_RE.findall(text)) == {"vie.traineddata"}


def test_the_notice_follows_the_models_kit_choose_keeps(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """kit choose dropped a model and left its notice behind (es, fr, it and pt got a stray copy)."""
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    shutil.copyfile(pdir / "manifest.json", pdir / B.DRAFT_FILE)
    assert CH.sync_notice(pdir, raw, packs) == "written"
    assert "hun.traineddata" in (pdir / CH.NOTICE_TESSDATA).read_text(encoding="utf-8")
    base = M.parse((packs / "en" / "manifest.json").read_bytes())
    tess = B.parse_reader_set(raw, _hu_tess(), KEY, base)
    rapid = B.parse_reader_set(raw, RAPID_ONLY, KEY, base)

    def bench(best: int) -> None:
        sets = []
        for i, rs in enumerate((tess, rapid)):
            v = 0.95 if i == best else 0.80
            sets.append({"index": i, "ok": True, "engines": len(rs.engines), "primary_metric": "words",
                         "seconds_per_page": 1.0, "worst_page": v, "reader_set": B.reader_set_json(rs),
                         "metrics": {"words": dict.fromkeys(CONDS, v)},
                         "must": {"double-acute": dict.fromkeys(CONDS, 0.95)}})
        (pdir / B.BENCH_FILE).write_text(json.dumps({"kit": B.KIT_VERSION, "corpus": "none", "platforms": {
            KEY: {"platform": KEY, "engine_versions": {}, "sets": sets}}}), encoding="utf-8")

    bench(best=1)
    code, out, _err = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 0 and out["dropped_models"] == ["hun.traineddata"] and out["notice"] == "removed", out
    assert not (pdir / CH.NOTICE_TESSDATA).exists()
    sel = out["selections"][KEY]
    assert sel["chosen_set"] == "rapidocr[rapid-rec-latin]" and sel["runner_up_set"].startswith("tesseract[hun] + ")
    assert sel["chosen_score"] == 0.95 and sel["runner_up_score"] == 0.80 and sel["ahead"] == []
    bench(best=0)                                        # the Tesseract set wins again: the notice comes back
    code, out, _err = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 0 and out["dropped_models"] == [] and out["notice"] == "written", out
    assert set(V._NOTICE_MODEL_RE.findall((pdir / CH.NOTICE_TESSDATA).read_text(encoding="utf-8"))) == {"hun.traineddata"}
    code, out, _err = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 0 and out["notice"] == "unchanged", out


def test_no_notice_without_a_tesseract_model_and_never_for_the_base_pack(tmp_path: Path):
    pdir = tmp_path / "xx"
    pdir.mkdir()
    (pdir / CH.NOTICE_TESSDATA).write_text("a stray copy\n", encoding="utf-8")
    assert CH.tessdata_notice([], TEMPLATE) is None
    assert CH.sync_notice(pdir, {"models": []}, tmp_path) == "removed" and not (pdir / CH.NOTICE_TESSDATA).exists()
    assert CH.sync_notice(pdir, {"models": []}, tmp_path) == "none"
    (pdir / CH.NOTICE_TESSDATA).write_text("the base pack's own\n", encoding="utf-8")
    assert CH.sync_notice(pdir, {"base": True, "models": []}, tmp_path) == "unchanged"
    assert (pdir / CH.NOTICE_TESSDATA).read_text(encoding="utf-8") == "the base pack's own\n"
    # Without the base pack's notice to copy the licence from, the notice still names the model.
    text = CH.tessdata_notice([{"file": "xxx.traineddata", "source": "https://example.com/xxx", "notice": CH.NOTICE_TESSDATA}],
                              None)
    assert text is not None and "  xxx.traineddata  https://example.com/xxx\n" in text and "Apache" in text


@pytest.mark.parametrize("pack", sorted(p.parent.name for p in REAL_PACKS.glob(f"*/{CH.NOTICE_TESSDATA}")
                                        if p.parent.name != "en" and (p.parent / "manifest.json").is_file()))
def test_the_notice_kit_choose_writes_is_the_one_each_pack_has(pack: str):     # a held-back pack has no manifest
    """The generator reproduces every launch pack's NOTICE-tessdata byte for byte from packs/en's,
    so kit choose rewriting it changes nothing that was right."""
    raw = json.loads((REAL_PACKS / pack / "manifest.json").read_text(encoding="utf-8"))
    want = (REAL_PACKS / pack / CH.NOTICE_TESSDATA).read_bytes()
    got = CH.tessdata_notice(raw["models"], CH.notice_template(REAL_PACKS))
    assert got is not None and got.encode("utf-8") == want


def test_verify_fails_a_notice_that_names_another_packs_models(tmp_path: Path):
    pdir = tmp_path / "vi"
    pdir.mkdir()
    model = {"file": "vie.traineddata", "source": "https://example.com/raw/vie.traineddata", "notice": CH.NOTICE_TESSDATA}
    (pdir / CH.NOTICE_TESSDATA).write_text(TEMPLATE, encoding="utf-8")            # the copied base notice
    (pdir / "NOTICE-unused").write_text("x.onnx\n", encoding="utf-8")
    problems, warnings = V.notice_findings(pdir, {"models": [model]})
    assert problems and "names ['eng.traineddata', 'osd.traineddata']" in problems[0]
    assert warnings == ["NOTICE-unused: no model in the manifest uses it; don't commit it"]
    (pdir / CH.NOTICE_TESSDATA).write_text(CH.tessdata_notice([model], TEMPLATE) or "", encoding="utf-8")
    assert V.notice_findings(pdir, {"models": [model]})[0] == []


def test_kit_verify_fails_the_manifest_check_on_a_copied_notice(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The notice check is wired into kit verify, not only callable: a pack carrying packs/en's
    notice fails the manifest check, and the pack's own notice passes it."""
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    (pdir / CH.NOTICE_TESSDATA).write_text(TEMPLATE, encoding="utf-8")
    code, out, _ = _run(capsys, "kit", "verify", "hu", "--packs", str(packs), "--work", str(tmp_path / "w"))
    assert code == 1 and not out["ok"] and out["checks"]["manifest"] == "failed", out
    notice = [p for p in out["problems"] if p.startswith("notice: ")]
    assert notice and "names ['eng.traineddata', 'osd.traineddata']" in notice[0], out["problems"]
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    assert CH.sync_notice(pdir, raw, packs) == "written"
    code, out, _ = _run(capsys, "kit", "verify", "hu", "--packs", str(packs), "--work", str(tmp_path / "w"))
    assert code == 0 and out["checks"]["manifest"] == "ok", out


# --------------------------------------------------------------------------- (2) --must-find

def test_more_must_find_entries_than_a_canary_holds_is_exit_2_before_anything_renders(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """kit trials fr, it, pt and vi: five or six words rendered, then failed with kit_invalid_manifest."""
    packs = _packs(tmp_path)

    def no_render(*a: Any, **k: Any) -> Any:
        raise AssertionError("rendered before the arguments were checked")

    monkeypatch.setattr(R, "render_canary", no_render)
    monkeypatch.setattr(B, "ensure_files", no_render)
    words = ["città", "perché", "così", "È", "4521"]
    argv = ["kit", "wheel", "hu", "--packs", str(packs), "--work", str(tmp_path / "w"), "--offline"]
    code, out, _err = _run(capsys, *argv, *(x for w in words for x in ("--must-find", w)))
    assert code == 2 and out["error"]["code"] == "kit_usage", out
    assert "at most 4 entries, 4521 included (5 given)" in out["error"]["message"]
    code, out, _err = _run(capsys, *argv, "--must-find", "x" * 41, "--must-find", "4521")
    assert code == 2 and "1 to 40 characters" in out["error"]["message"], out
    assert cli._must_find_limits() == (4, 40)            # read from the canary schema


# --------------------------------------------------------------------------- (3) labels and progress

def test_a_use_is_named_with_its_model():
    assert B.use_label({"id": "tesseract", "engine": "tesseract", "params": {"lang": "vie", "psm": 3}}) == "tesseract[vie]"
    assert B.use_label({"id": "tesseract", "params": {"lang": "tha+eng"}}) == "tesseract[tha+eng]"
    assert B.use_label({"id": "rapidocr", "params": {"rec": "en:rapid-rec-latin"}}) == "rapidocr[rapid-rec-latin]"
    assert B.use_label({"id": "vision2", "params": {"languages": ["vi-VT", "en-US"], "language_correction": False}}) \
        == "vision2[vi-VT, no correction]"
    assert B.use_label({"id": "vision", "params": {"languages": ["vi-VT"], "language_correction": True}}) == "vision[vi-VT]"
    assert B.use_label(SimpleNamespace(id="fake", params={})) == "fake"
    assert B.set_label({"engines": [{"id": "tesseract", "params": {"lang": "eng"}},
                                    {"id": "rapidocr", "params": {"rec": "en:rapid-rec-latin"}}]}) \
        == "tesseract[eng] + rapidocr[rapid-rec-latin]"


def test_a_progress_line_counts_sets_as_the_json_does():
    """kit trial pl, item 5: ``set 1/54`` was the set with ``"index": 0``."""
    uses = (SimpleNamespace(id="tesseract", params={"lang": "pol", "psm": 3}),
            SimpleNamespace(id="rapidocr", params={"rec": "en:rapid-rec-latin"}))
    rs = SimpleNamespace(engines=uses, pick=SimpleNamespace(rule="primary_with_floor"))
    line = cli.progress_line(0, 54, rs, {"ok": True}, 21.4, 24, 18 * 60)
    assert line == ("kit bench: set 0 (1 of 54): tesseract[pol] + rapidocr[rapid-rec-latin], primary_with_floor; "
                    "ok, 21 s for 24 pages; about 18 min left")
    failed = cli.progress_line(53, 54, rs, {"ok": False, "error": {"code": "engine_missing"}}, 1.0, 24, 0.0)
    assert failed.startswith("kit bench: set 53 (54 of 54): ") and "failed (engine_missing)" in failed


# --------------------------------------------------------------------------- (8) init --from-manifest

def test_init_from_manifest_rebuilds_the_draft_and_repins_the_dropped_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                            capsys: pytest.CaptureFixture[str]):
    """kit trial es, re-run R2: with the draft gone and the model dropped from manifest.json, a later
    bench would never propose the language's Tesseract model again."""
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    raw["models"] = []
    raw["readers"] = {"default": RAPID_ONLY}
    raw["bench"] = {"kit": "1.0", "corpus": "none", "platforms": {KEY: {"metrics": {}, "worst_page": 0.9,
                                                                         "seconds_per_page": 1.0,
                                                                         "selection": {"policy": CH.POLICY, "margin": 0.0}}}}
    before = json.dumps(raw, indent=2, ensure_ascii=False)
    (pdir / "manifest.json").write_text(before, encoding="utf-8")
    url = cli.TESSDATA_URL.format(commit=cli.TESSDATA_BEST_COMMIT, name="hun")
    monkeypatch.setattr(C, "_http_get", FakeWeb({url: b"made-up hun model"}))
    code, out, _err = _run(capsys, "kit", "init", "hu", "--from-manifest", "--packs", str(packs))
    assert code == 0 and out["model"]["file"] == "hun.traineddata" and out["model"]["source"] == url, out
    draft = json.loads((pdir / B.DRAFT_FILE).read_text(encoding="utf-8"))
    assert [m["file"] for m in draft["models"]] == ["hun.traineddata"] and "canary" not in draft
    assert draft["readers"] == raw["readers"] and draft["must_survive"] == raw["must_survive"]
    assert draft["bench"] == {"kit": B.KIT_VERSION, "corpus": "none", "platforms": {}}
    assert (pdir / "manifest.json").read_text(encoding="utf-8") == before        # manifest.json is untouched
    assert out["notice"] == "none" and not (pdir / CH.NOTICE_TESSDATA).exists()   # it follows manifest.json
    assert M.check(draft, base=M.parse((packs / "en" / "manifest.json").read_bytes())) == []
    # the draft exists now: again needs --force; --must-survive replaces the groups (a new floor)
    assert _run(capsys, "kit", "init", "hu", "--from-manifest", "--packs", str(packs))[0] == 2
    code, out, _err = _run(capsys, "kit", "init", "hu", "--from-manifest", "--force", "--packs", str(packs),
                           "--must-survive", "double-acute=őűŐŰ:word:0.85")
    assert code == 0 and out["model"]["file"] == "hun.traineddata", out          # rebuilt from manifest.json again
    draft = json.loads((pdir / B.DRAFT_FILE).read_text(encoding="utf-8"))
    assert draft["must_survive"] == [{"name": "double-acute", "chars": "őűŐŰ", "unit": "word", "floor": 0.85}]


def test_init_from_manifest_needs_a_manifest_and_no_script(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    code, out, _err = _run(capsys, "kit", "init", "es", "--from-manifest", "--packs", str(packs))
    assert code == 2 and out["error"]["code"] == "kit_no_manifest", out
    code, out, _err = _run(capsys, "kit", "init", "hu", "--from-manifest", "--script", "Latin", "--packs", str(packs))
    assert code == 2 and "--from-manifest takes the script" in out["error"]["message"], out
    code, out, _err = _run(capsys, "kit", "init", "es", "--packs", str(packs))
    assert code == 2 and "needs --script" in out["error"]["message"], out


def test_init_names_kit_corpus_while_the_corpus_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]):
    """kit trials fr, pt and pl: ``next`` was empty right after init, with nothing pinned."""
    packs = _packs(tmp_path)
    url = cli.TESSDATA_URL.format(commit=cli.TESSDATA_BEST_COMMIT, name="pol")
    monkeypatch.setattr(C, "_http_get", FakeWeb({url: b"made-up pol model"}))
    code, out, _err = _run(capsys, "kit", "init", "pl", "--script", "Latin", "--packs", str(packs),
                           "--must-survive", "ogonek=ąę:word:0.9")
    assert code == 0, out
    assert len(out["next"]) == 1 and "pdfintake kit corpus pl" in out["next"][0]


def test_a_floor_added_after_the_bench_makes_the_bench_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A must-survive group the bench never measured would pass every floor unmeasured."""
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    base = M.parse((packs / "en" / "manifest.json").read_bytes())
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    rs = B.reader_set_json(B.parse_reader_set(raw, RAPID_ONLY, KEY, base))
    entry = {"index": 0, "ok": True, "engines": 1, "primary_metric": "words", "seconds_per_page": 1.0,
             "worst_page": 0.9, "reader_set": rs, "metrics": {"words": dict.fromkeys(CONDS, 0.9)}, "must": {}}
    (pdir / B.BENCH_FILE).write_text(json.dumps({"kit": B.KIT_VERSION, "corpus": "none", "platforms": {
        KEY: {"platform": KEY, "engine_versions": {}, "sets": [entry]}}}), encoding="utf-8")
    code, out, _err = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 2 and out["error"]["code"] == "kit_bench_stale" and "double-acute" in out["error"]["message"], out
    entry["must"] = {"double-acute": dict.fromkeys(CONDS)}               # measured: the corpus had none of them
    (pdir / B.BENCH_FILE).write_text(json.dumps({"kit": B.KIT_VERSION, "corpus": "none", "platforms": {
        KEY: {"platform": KEY, "engine_versions": {}, "sets": [entry]}}}), encoding="utf-8")
    assert _run(capsys, "kit", "choose", "hu", "--packs", str(packs))[0] == 0


# --------------------------------------------------------------------------- (5) exit 6 through the CLI

def test_kit_choose_exit_6_prints_the_closest_sets(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    base = M.parse((packs / "en" / "manifest.json").read_bytes())
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    sets = []
    for i, (rs, small) in enumerate(((_hu_tess(), 0.88), (RAPID_ONLY, 0.40))):
        sets.append({"index": i, "ok": True, "engines": 1, "primary_metric": "words", "seconds_per_page": 1.0,
                     "worst_page": 0.9, "reader_set": B.reader_set_json(B.parse_reader_set(raw, rs, KEY, base)),
                     "metrics": {"words": dict.fromkeys(CONDS, 0.9)},
                     "must": {"double-acute": {"clean": 0.99, "scan": 0.97, "small": small}}})
    (pdir / B.BENCH_FILE).write_text(json.dumps({"kit": B.KIT_VERSION, "corpus": "none", "platforms": {
        KEY: {"platform": KEY, "engine_versions": {}, "sets": sets}}}), encoding="utf-8")
    code, out, _err = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 6 and out["error"]["code"] == "kit_floor", out
    closest = out["error"]["closest"]
    assert [c["index"] for c in closest] == [0, 1] and closest[0]["reader_set"].startswith("tesseract[hun]")
    assert closest[0]["misses"] == [{"name": "double-acute", "cond": "small", "value": 0.88, "floor": 0.9}]
    assert out["error"]["problems"][0].startswith("set 0 (tesseract[hun] + rapidocr[rapid-rec-latin], "
                                                  "primary_with_floor): double-acute small 0.8800 < 0.9")


# --------------------------------------------------------------------------- (6) kit report

def _report_pack(tmp: Path, *, readers: dict[str, Any], models: list[Any] | None = None) -> Path:
    packs = _packs(tmp)
    pdir = packs / "hu"
    base = M.parse((packs / "en" / "manifest.json").read_bytes())
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    as_benched = json.loads(json.dumps(raw))            # the draft the bench read with still pinned hun
    raw["readers"] = readers
    if models is not None:
        raw["models"] = models
    claim = {"metrics": {"words": [0.95, 0.94, 0.9]}, "worst_page": 0.5, "seconds_per_page": 2.0,
             "selection": {"policy": CH.POLICY, "margin": -0.0018}}
    raw["bench"] = {"kit": "1.0", "corpus": "none", "platforms": {KEY: claim}}
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    sets = []
    for i, rs in enumerate((_hu_tess(), RAPID_ONLY)):
        sets.append({"index": i, "ok": True, "engines": len(rs["engines"]), "primary_metric": "words",
                     "seconds_per_page": 1.5, "worst_page": 0.42,
                     "reader_set": B.reader_set_json(B.parse_reader_set(as_benched, rs, KEY, base)),
                     "metrics": {"words": {"clean": 0.979, "scan": 0.975, "small": 0.875}},
                     "must": {"double-acute": {"clean": 0.999, "scan": 0.996, "small": 0.904}}})
    (pdir / B.BENCH_FILE).write_text(json.dumps({"kit": B.KIT_VERSION, "corpus": "none", "platforms": {
        KEY: {"platform": KEY, "engine_versions": {}, "sets": sets}}}), encoding="utf-8")
    return pdir


def test_report_never_prints_an_empty_models_table(tmp_path: Path):
    """kit trials es (R7), fr, it and pt: with the model dropped, the table was a header and no rows."""
    pdir = _report_pack(tmp_path, readers={"default": RAPID_ONLY}, models=[])
    body = REP.pr_body(pdir)
    assert "| File | Kind |" not in body and "None of its own" in body
    assert ("The base pack's models the reader sets read with: `en:rapid-det` (`det_mobile.onnx`), "
            "`en:rapid-cls` (`cls_mobile.onnx`), `en:rapid-rec-latin` (`latin_rec_mobile.onnx`)") in body
    assert "`rapidocr[rapid-rec-latin]` (rapidocr, reader)" in body


def test_report_shows_the_characters_the_unbenched_platforms_and_the_margin(tmp_path: Path):
    """kit trial it, items 9 and 10, and vi, item 4."""
    pdir = _report_pack(tmp_path, readers={"default": _hu_tess()})
    body = REP.pr_body(pdir)
    assert "`double-acute` words holding `őűŐŰ`, floor 0.9" in body
    unbenched = [p.key for p in platforms.PLATFORMS if p.supported and p.key != KEY]
    assert "Not benched yet: " + ", ".join(f"`{k}`" for k in unbenched) in body
    assert "margin -0.0018" in body and REP.MARGIN_NOTE in body           # four places: 0.0001 isn't 0
    assert "| `hun.traineddata` | traineddata |" in body
    assert "words holding" in body and "must find `őrlő`, `fűzfa`" in body
    assert f"its numbers are the `{KEY}` table above" in body


def test_report_shows_the_fills_and_repairs_the_bench_read_with(tmp_path: Path):
    """Wave E review: a pack-level repair changes what every set reads, but the body never named it."""
    pdir = _report_pack(tmp_path, readers={"default": _hu_tess()})
    body = REP.pr_body(pdir)
    assert "- Fills: `numbers` v1, `ids` v1" in body and "- Repairs: none" in body
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    raw["repairs"] = [{"name": "diacritic_word", "version": 1,
                       "params": {"source": "rapidocr", "table": {"ä": ["a", "á"]}}}]
    raw["fills"] = []
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    body = REP.pr_body(pdir)
    assert '- Repairs: `diacritic_word` v1 `` {"source": "rapidocr", "table": {"ä": ["a", "á"]}} ``' in body
    assert "- Fills: none" in body


def test_report_shows_the_default_sets_numbers_when_no_benched_platform_reads_it(tmp_path: Path):
    """kit trial vi, item 5: the Mac chose its own set; Linux and Windows read with ``default``, whose
    numbers were in bench.json but not in the report."""
    pdir = _report_pack(tmp_path, readers={"default": _hu_tess(), KEY: RAPID_ONLY})
    body = REP.pr_body(pdir)
    assert f"measured as set 0 of the `{KEY}` bench (worst page 0.420, 1.500 s/page)" in body
    assert "| must-double-acute | 0.999 | 0.996 | 0.904 |" in body and "| words | 0.979 | 0.975 | 0.875 |" in body
    (pdir / B.BENCH_FILE).unlink()
    assert "no platform's bench has measured it yet" in REP.pr_body(pdir)


def test_report_takes_the_benched_platforms_from_bench_json(tmp_path: Path):
    """Review F3: with bench.json holding numbers the manifest doesn't claim yet, the report listed the
    platform as not benched and quoted its bench a few lines earlier. It now agrees with kit choose's
    ``unbenched`` and says the claim is what's missing."""
    pdir = _report_pack(tmp_path, readers={"default": _hu_tess()})
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    raw["bench"]["platforms"] = {}
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    body = REP.pr_body(pdir)
    unbenched = [p.key for p in platforms.PLATFORMS if p.supported and p.key != KEY]
    want = "Not benched yet: " + ", ".join(f"`{k}`" for k in unbenched)
    assert (want + "." in body) if unbenched else ("Not benched yet" not in body)
    assert f"`{KEY}`" not in body.split("Not benched yet", 1)[-1].split("\n", 1)[0]
    assert f"Benched but no numbers claimed in the manifest yet: `{KEY}`. Run `pdfintake kit choose hu` again." in body
    (pdir / B.BENCH_FILE).unlink()                     # no bench.json: the manifest's claims decide, as before
    body = REP.pr_body(pdir)
    assert "Benched but no numbers claimed" not in body and f"`{KEY}`" in body.split("Not benched yet", 1)[-1].split("\n", 1)[0]
