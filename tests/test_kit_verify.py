"""kit/verify.py, ``kit verify`` as pack-bench.yml calls it, and tools/build_index.py on the fixture data."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from pdfintake.kit import cli
from pdfintake.kit import verify as V
from pdfintake.langpacks import manifest as M
from pdfintake.pdf.textlayer import FontSpec
from test_kit_render import latin_chars

from fixtures import build

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PACKS = ROOT / "tests" / "fixtures" / "packs"
FIXTURE_INDEX = ROOT / "tests" / "fixtures" / "index.test.json"
WORKFLOW = ROOT / ".github" / "workflows" / "pack-bench.yml"


def load_build_index() -> Any:
    spec = importlib.util.spec_from_file_location("pdfintake_build_index_tool", ROOT / "tools" / "build_index.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _packs(tmp: Path) -> Path:
    dest = tmp / "packs"
    shutil.copytree(FIXTURE_PACKS, dest)
    return dest


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = cli.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------- build_index.py

def test_build_index_check_passes_on_the_fixture_data(capsys: pytest.CaptureFixture[str]):
    bi = load_build_index()
    assert bi.main(["--packs", str(FIXTURE_PACKS), "--index", str(FIXTURE_INDEX), "--wheels", "no-such-folder",
                    "--check"]) == 0
    current = json.loads(FIXTURE_INDEX.read_text(encoding="utf-8"))
    again = bi.build(FIXTURE_PACKS, current, ROOT / "no-such-folder")
    assert bi.render(again) == FIXTURE_INDEX.read_text(encoding="utf-8")
    assert [e["id"] for e in again["packs"]] == ["en", "th", "hu"]          # the base first, then the index's order


# A manifest changed under a published version, and a same-version wheel with another manifest:
# "published" now comes from tools/published-packs.json, not from the index, so these are
# tests/test_build_index.py's (test_a_released_manifest_edit_is_refused and
# test_a_released_wheel_never_changes).


def test_build_index_refuses_what_it_cannot_pin(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bi = load_build_index()
    packs = _packs(tmp_path)
    index = tmp_path / "index.json"
    shutil.copyfile(FIXTURE_INDEX, index)
    raw = json.loads((packs / "th" / "manifest.json").read_text(encoding="utf-8"))
    raw["version"] = "1.1.0"
    (packs / "th" / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    raw = json.loads((packs / "hu" / "manifest.json").read_text(encoding="utf-8"))
    del raw["canary"]
    (packs / "hu" / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    assert bi.main(["--packs", str(packs), "--index", str(index), "--wheels", str(tmp_path / "dist")]) == 1
    out = capsys.readouterr().out
    assert "th: version 1.1.0 has no wheel" in out and "hu: a published pack needs a canary" in out
    assert index.read_bytes() == FIXTURE_INDEX.read_bytes()


# --------------------------------------------------------------------------- kit verify

def test_pack_bench_calls_kit_verify_with_flags_the_kit_takes(monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]):
    text = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r'pdfintake (kit verify "\$PACK"[^>]*?)\s*>', text)
    assert m, "pack-bench.yml no longer calls kit verify"
    argv = m.group(1).replace('"$PACK"', "th").split()
    seen: dict[str, Any] = {}

    def fake(pack_dir: Path, **kw: Any) -> V.PackVerify:
        seen.update(kw, pack=Path(pack_dir).name)
        return V.PackVerify(True, [], {"platform": "x", "drift": [], "floors": []})

    monkeypatch.setattr(V, "verify_pack", fake)
    code, out = _run(capsys, *argv, "--packs", str(FIXTURE_PACKS))
    assert code == 0 and seen["download"] is True and seen["rebench"] is True and seen["pack"] == "th"
    assert set(out) >= {"ok", "problems", "rebench"}            # what the workflow's summary step reads


@pytest.mark.parametrize("pack", ["en", "th", "hu"])
def test_kit_verify_passes_on_the_fixture_packs(pack: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _run(capsys, "kit", "verify", pack, "--packs", str(FIXTURE_PACKS), "--work", str(tmp_path), "--json")
    assert code == 0 and out["ok"] and out["problems"] == [] and out["rebench"] is None, out
    assert out["checks"]["manifest"] == "ok" and out["checks"]["download"].startswith("skipped")


def test_kit_verify_reports_manifest_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    raw = json.loads((packs / "hu" / "manifest.json").read_text(encoding="utf-8"))
    raw["models"][0]["source"] = "https://example.com/hun.traineddata"
    (packs / "hu" / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    (packs / "hu" / "bench.json").write_text(json.dumps({"platforms": {"linux-x86_64": {"note": "a sentence"}}}),
                                             encoding="utf-8")
    code, out = _run(capsys, "kit", "verify", "hu", "--packs", str(packs), "--work", str(tmp_path / "w"))
    assert code == 1 and not out["ok"]
    joined = " ".join(out["problems"])
    assert "$.models[0].source" in joined and "bench.json" in joined


@pytest.mark.parametrize(("finding", "code"), [({"drift": ["words clean: claimed 0.9, measured 0.8"], "floors": []}, 7),
                                               ({"drift": [], "floors": ["acute scan: 0.5 is below the floor 0.9"]}, 6)])
def test_rebench_findings_set_the_kit_exit_codes(finding: dict, code: int, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(V, "_rebench", lambda *a, **k: {"platform": "linux-x86_64", **finding})
    got, out = _run(capsys, "kit", "verify", "hu", "--packs", str(FIXTURE_PACKS), "--work", str(tmp_path), "--rebench")
    assert got == code and not out["ok"] and out["rebench"]["platform"] == "linux-x86_64"


def test_compare_uses_the_tolerance_and_absolute_floors():
    must = [M.MustSurvive("acute", "é", "word", 0.9)]
    claim = {"metrics": {"words": [0.95, 0.90, 0.80], "must-acute": [1.0, 0.95, 0.92]}}
    measured = {"metrics": {"words": {"clean": 0.955, "scan": 0.889, "small": 0.80}},
                "must": {"acute": {"clean": 1.0, "scan": 0.95, "small": 0.92}}}
    assert V.compare(claim, measured, must, tolerance=0.02) == {"drift": [], "floors": []}
    got = V.compare(claim, measured, must, tolerance=0.01)
    assert got["floors"] == [] and len(got["drift"]) == 1 and "words scan" in got["drift"][0]
    low = {"metrics": measured["metrics"], "must": {"acute": {"clean": 0.85, "scan": None, "small": 0.95}}}
    got = V.compare(None, low, must, tolerance=0.01)
    assert got["drift"] == [] and got["floors"] == ["acute clean: 0.8500 is below the floor 0.9"]


def test_layer_round_trip_finds_what_the_fonts_can_write(tmp_path: Path):
    import hashlib

    font = build.write_box_font(tmp_path / "ExampleSans-Regular.ttf", latin_chars(), family="Example Sans")
    spec = FontSpec("example-sans", font, hashlib.sha256(font.read_bytes()).hexdigest(), ("Latin",), 0)
    assert V.layer_round_trip(["example", "4521", "aéb", "Őrlő"], [spec], tmp_path / "a.pdf") == []
    assert V.layer_round_trip(["example", "abc中d"], [spec], tmp_path / "b.pdf") == ["abc中d"]


@pytest.mark.parametrize(("checks", "rebench", "code"), [
    ({"manifest": "ok", "download": "failed"}, {"drift": ["words clean: claimed 0.9, measured 0.8"], "floors": []}, 1),
    ({"manifest": "ok", "wheel": "failed"}, {"drift": [], "floors": ["acute scan: 0.5 is below the floor 0.9"]}, 1),
    ({"manifest": "ok", "download": "ok"}, {"error": {"code": "kit_download", "message": "x"}}, 1),
    ({"manifest": "ok", "download": "ok"}, {"drift": ["d"], "floors": ["f"]}, 6),
    ({"manifest": "ok", "download": "ok"}, {"drift": ["d"], "floors": []}, 7),
])
def test_a_failed_check_wins_over_rebench_findings(checks: dict, rebench: dict, code: int,
                                                   monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    # A drift must never hide a broken download in CI's step summary (decided 2026-11-05).
    def fake(pack_dir: Path, **kw: Any) -> V.PackVerify:
        return V.PackVerify(False, ["something"], {"platform": "x", **rebench}, dict(checks),
                            floor_failed=bool(rebench.get("floors")), drifted=bool(rebench.get("drift")))

    monkeypatch.setattr(V, "verify_pack", fake)
    got, out = _run(capsys, "kit", "verify", "th", "--packs", str(FIXTURE_PACKS), "--rebench")
    assert got == code and out["ok"] is False
