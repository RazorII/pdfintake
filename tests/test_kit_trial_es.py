"""The kit fixes from its first trial (docs/kit-trial-es.md): one test per kit bug the trial hit.

Every pack, text and font here is made up; the packs are read by fake engines (test_kit_render).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake import datawheel, platforms
from pdfintake.errors import PackError
from pdfintake.kit import bench as B
from pdfintake.kit import choose as CH
from pdfintake.kit import cli
from pdfintake.kit import render as R
from pdfintake.langpacks import manifest as M
from test_kit_render import (FONT2_URL, FONT_URL, TEXT, corpus_for, fake_manifest, fake_pack, latin_chars,
                             latin_font, register_fake, thai_font)
from test_kit_rules import write_canary_env

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PACKS = ROOT / "tests" / "fixtures" / "packs"
KEY = platforms.current().key
OTHER = next(p.key for p in platforms.PLATFORMS if p.supported and p.key != KEY)
posix = pytest.mark.skipif(sys.platform == "win32", reason="dev-install.sh is bash")


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    code = cli.main(list(argv))
    got = capsys.readouterr()
    return code, json.loads(got.out), got.err


def _flow(packs: Path, work: Path, capsys: pytest.CaptureFixture[str], *, wheel: bool = True) -> dict[str, Any]:
    """corpus, render, bench (the draft's own set), choose and, with ``wheel``, wheel; the choose output."""
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    steps = [["corpus", "en", *common, "--wikipedia", "Q1,Q2", "--font", FONT_URL, "--font", FONT2_URL,
              "--pages", "prose=1,form=0,microprint=0,mixed_script=0"],
             ["render", "en", *common], ["bench", "en", *common, "--offline", "--current-only"],
             ["choose", "en", *common]]
    if wheel:
        write_canary_env(packs.parent)                   # kit wheel compares the canary libraries
        steps.append(["wheel", "en", *common, "--offline", "--must-find", "example", "--must-find", "4521"])
    chosen: dict[str, Any] = {}
    for argv in steps:
        code, out, _err = _run(capsys, "kit", *argv)
        assert code == 0, out
        if argv[0] == "choose":
            chosen = out
    return chosen


# --------------------------------------------------------------------------- item 10: candidates

def test_a_tesseract_lang_is_never_prefixed_so_every_candidate_loads():
    """``lang: eng`` named the base pack's model key, so the kit wrote ``en:eng`` and the loader refused
    every set with Tesseract eng (21 of 54 for es): they were never benched."""
    base = M.parse((FIXTURE_PACKS / "en" / "manifest.json").read_bytes())
    raw = json.loads((FIXTURE_PACKS / "hu" / "manifest.json").read_text(encoding="utf-8"))
    m = M.parse(json.dumps(raw).encode("utf-8"), base=base)
    key = "linux-x86_64"
    sets, info = B.candidate_sets(m, key, base, raw=raw)
    assert info["refused_by_loader"] == 0
    assert info["kept"] == len(sets) == info["total"] - info["merged"] - info["cut"]
    uses = [u for s in sets for u in s["engines"]]
    langs = {u["params"]["lang"] for u in uses if u["engine"] == "tesseract"}
    assert langs == {"eng", "hun"}
    eng = next(u for u in uses if u["engine"] == "tesseract" and u["params"]["lang"] == "eng")
    assert eng["models"] == ["en:eng"]
    rec = {u["params"]["rec"] for u in uses if u["engine"] == "rapidocr"}
    assert rec == {"en:rapid-rec-latin"}                       # an asset reference still takes the prefix
    for s in sets:
        B.parse_reader_set(raw, s, key, base)


def test_refused_sets_are_counted_before_the_ceiling_and_kept_is_the_list(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch,
                                                                           capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, second=True)
    m, raw, _ = B.load_manifest(packs, "en")
    everything, _info = B.candidate_sets(m, KEY, None)
    real = B.parse_reader_set

    def refuse_floor_sets(raw_: Any, s: Any, key: str, base: Any) -> Any:
        if s["pick"]["rule"] == "primary_with_floor":
            raise PackError("invalid_manifest", "refused for the test")
        return real(raw_, s, key, base)

    monkeypatch.setattr(B, "parse_reader_set", refuse_floor_sets)
    floor_sets = sum(1 for s in everything[1:] if s["pick"]["rule"] == "primary_with_floor")
    assert floor_sets > 0
    sets, info = B.candidate_sets(m, KEY, None, raw=raw)
    assert info["refused_by_loader"] == floor_sets
    assert info["kept"] == len(sets) == info["total"] - info["refused_by_loader"] - info["merged"] - info["cut"]
    code, out, _err = _run(capsys, "kit", "candidates", "en", "--packs", str(packs), "--work", str(work))
    assert code == 0 and out["count"] == out["candidates"]["kept"] == len(sets), out
    assert out["candidates"]["refused_by_loader"] == floor_sets
    # each use's parameters, so tesseract2 or fakeb can be told apart without opening the file (item 11)
    assert all(set(s["params"]) == set(s["engines"]) and all(isinstance(p, dict) for p in s["params"].values())
               for s in out["sets"])


# --------------------------------------------------------------------------- items 12, 13: the bench

def test_bench_reports_progress_on_stderr_and_floors_per_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"], second=True)
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    for argv in (["corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL, "--font", FONT2_URL,
                  "--pages", "prose=1,form=0,microprint=0,mixed_script=0"], ["render", "en", *common]):
        assert _run(capsys, "kit", *argv)[0] == 0
    code, out, err = _run(capsys, "kit", "bench", "en", *common, "--offline")
    assert code == 0, out
    lines = [x for x in err.splitlines() if x.startswith("kit bench: set ")]
    n = len(out["sets"])
    assert len(lines) == n and lines[0].startswith(f"kit bench: set 0 (1 of {n}): ") and "min left" in lines[-1]
    assert all(s["floors_ok"] is True and "below_floor" not in s for s in out["sets"])


def test_a_set_below_a_floor_is_ok_but_not_floors_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch)
    raw = json.loads((packs / "en" / "manifest.json").read_text(encoding="utf-8"))
    raw["must_survive"] = [{"name": "accents", "chars": "éá", "unit": "word", "floor": 0.9}]
    (packs / "en" / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    for argv in (["corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL, "--font", FONT2_URL,
                  "--pages", "prose=1,form=0,microprint=0,mixed_script=0"], ["render", "en", *common]):
        assert _run(capsys, "kit", *argv)[0] == 0
    below = {"accents": {"clean": 1.0, "scan": 0.5, "small": 0.95}}

    def fake_bench(pages: Any, pack: Any, base: Any, key: str, sets: Any, **kw: Any) -> B.BenchResult:
        entries = [{"index": i, "ok": True, "engines": 1, "primary_metric": "words", "seconds_per_page": 1.0,
                    "metrics": {"words": {"clean": 0.9, "scan": 0.9, "small": 0.9}}, "worst_page": 0.9,
                    "must": below if i == 0 else {"accents": {"clean": 1.0, "scan": 1.0, "small": 1.0}},
                    "reader_set": B.reader_set_json(rs)} for i, rs in enumerate(sets)]
        return B.BenchResult(key, {"fake": "1.0"}, entries)

    monkeypatch.setattr(B, "bench", fake_bench)
    code, out, _err = _run(capsys, "kit", "bench", "en", *common, "--offline")
    assert code == 0, out
    first = out["sets"][0]
    assert first["ok"] is True and first["floors_ok"] is False and first["below_floor"] == ["accents scan 0.500 < 0.9"]


# --------------------------------------------------------------------------- items 14-18: choose

def test_choose_names_kit_wheel_and_the_platforms_without_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"])
    out = _flow(packs, work, capsys, wheel=False)
    assert out["canary"] == "none" and len(out["next"]) == 1 and "kit wheel en" in out["next"][0], out
    supported = [p.key for p in platforms.PLATFORMS if p.supported]
    assert out["unbenched"] == [k for k in supported if k != KEY] and out["dropped_models"] == []


def _hu_out(readers: dict[str, Any]) -> dict[str, Any]:
    raw = json.loads((FIXTURE_PACKS / "hu" / "manifest.json").read_text(encoding="utf-8"))
    raw["readers"] = readers
    return raw


def test_choose_drops_the_models_no_reader_set_uses():
    base = M.parse((FIXTURE_PACKS / "en" / "manifest.json").read_bytes())
    rapid_only = {"engines": [{"id": "rapidocr", "engine": "rapidocr", "role": "reader",
                               "params": {"rec": "en:rapid-rec-latin"},
                               "models": ["en:rapid-det", "en:rapid-cls", "en:rapid-rec-latin"]}],
                  "pick": {"rule": "single", "version": 1, "params": {"reader": "rapidocr"}}, "fallback": "none"}
    out = _hu_out({"default": rapid_only})
    assert CH.prune_models(out) == ["hun.traineddata"] and out["models"] == []
    assert M.check(out, base=base) == []
    # A model any platform's set uses stays, and so does one a Tesseract use names only by its lang.
    tess = json.loads((FIXTURE_PACKS / "hu" / "manifest.json").read_text(encoding="utf-8"))["readers"]["default"]
    kept = _hu_out({"default": rapid_only, "linux-x86_64": tess})
    assert CH.prune_models(kept) == [] and [m["file"] for m in kept["models"]] == ["hun.traineddata"]
    by_lang = _hu_out({"default": json.loads(json.dumps(tess))})
    by_lang["readers"]["default"]["engines"][0]["models"] = []
    assert CH.prune_models(by_lang) == []
    # The base pack is never pruned: a run reads its OSD model and voter outside any reader set.
    en = json.loads((FIXTURE_PACKS / "en" / "manifest.json").read_text(encoding="utf-8"))
    before = list(en["models"])
    assert CH.prune_models(en) == [] and en["models"] == before


# --------------------------------------------------------------------------- items 18-20: verify

def test_rebench_on_a_platform_without_numbers_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"])
    _flow(packs, work, capsys)
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    code, out, _err = _run(capsys, "kit", "verify", "en", *common, "--rebench", "--platform", OTHER)
    assert code == 0 and out["ok"], out
    assert out["rebench"]["claimed"] is None and f"no numbers for {OTHER}" in out["rebench"]["warning"]
    assert any(w.startswith("rebench: the pack claims no numbers") for w in out["warnings"])
    code, out, _err = _run(capsys, "kit", "verify", "en", *common, "--rebench")
    assert code == 0 and "warning" not in out["rebench"], out
    assert not any(w.startswith("rebench") for w in out["warnings"])


def _es_pack(tmp: Path, monkeypatch: pytest.MonkeyPatch, *, canary_text: str = "example 4521") -> tuple[Path, dict[str, bytes]]:
    """``packs/en`` (fake base, one font) and ``packs/es``, a Latin pack with no font of its own whose
    canary was rendered with the base font; and the bytes the fake upstream serves."""
    register_fake(monkeypatch, "fake")
    font = latin_font()
    packs = tmp / "packs"
    (packs / "en").mkdir(parents=True)
    (packs / "en" / "manifest.json").write_text(json.dumps(fake_manifest(font)), encoding="utf-8")
    (packs / "en" / "OFL.txt").write_text("Example licence text for the test font.\n", encoding="utf-8")
    fdir = tmp / "font"
    fdir.mkdir()
    (fdir / "ExampleSans-Regular.ttf").write_bytes(font)
    png = R.render_canary(canary_text, [fdir / "ExampleSans-Regular.ttf"], tmp / "canary-es.png")
    data = png.read_bytes()
    raw = fake_manifest(font, must=[{"name": "tilde", "chars": "ñ", "unit": "word", "floor": 0.9}])
    del raw["base"]
    raw.update(id="es", title="Spanish", languages=[{"tag": "es", "iso639_3": "spa", "role": "primary"}],
               detect={"osd_scripts": ["Latin"], "langid": {"codes": ["es"]}}, fonts=[],
               canary={"file": "canary-es.png", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                       "dpi": 300, "must_find": ["example", "4521"]})
    (packs / "es").mkdir()
    (packs / "es" / "manifest.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    base = M.parse((packs / "en" / "manifest.json").read_bytes())
    assert M.check(raw, base=base) == []
    upstream = {FONT_URL: font}

    def fake_download(url: str, dest: Path, size: int, sha: str) -> None:
        body = upstream[url]
        assert len(body) == size and hashlib.sha256(body).hexdigest() == sha
        Path(dest).write_bytes(body)

    monkeypatch.setattr(datawheel, "_download", fake_download)

    def no_installed_base() -> Any:
        raise PackError("index_missing", "no pack index in this test")

    from pdfintake.langpacks import store

    monkeypatch.setattr(store, "base", no_installed_base)
    return packs, upstream


def test_verify_download_on_a_fresh_work_folder_runs_all_four_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                     capsys: pytest.CaptureFixture[str]):
    """CI's case: a fresh checkout, no kit wheel. The base fonts were never fetched (a bare
    {"fonts": ...} manifest that datawheel refuses) and the canary was never there, so the layer and
    wheel checks were skipped and verify still said ok."""
    packs, _up = _es_pack(tmp_path, monkeypatch)
    work = tmp_path / "work"
    code, out, _err = _run(capsys, "kit", "verify", "es", "--packs", str(packs), "--work", str(work), "--download")
    assert code == 0 and out["ok"], out
    assert out["checks"] == {"manifest": "ok", "download": "ok", "layer": "ok", "wheel": "ok"}
    assert out["warnings"] == []
    assert (work / "en" / "files" / "ExampleSans-Regular.ttf").is_file()
    assert (work / "es" / "files" / "canary-es.png").is_file()


def test_verify_without_download_lists_what_it_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]):
    packs, _up = _es_pack(tmp_path, monkeypatch)
    code, out, _err = _run(capsys, "kit", "verify", "es", "--packs", str(packs), "--work", str(tmp_path / "w"))
    assert code == 0 and out["ok"], out
    assert {k for k, v in out["checks"].items() if v.startswith("skipped")} == {"download", "layer", "wheel"}
    assert len(out["warnings"]) == 3 and all("pass --download" in w for w in out["warnings"])


def test_verify_download_fails_a_canary_that_doesnt_render_to_its_pin(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      capsys: pytest.CaptureFixture[str]):
    packs, _up = _es_pack(tmp_path, monkeypatch, canary_text="an example 4521")   # pinned from other text
    work = tmp_path / "work"
    code, out, _err = _run(capsys, "kit", "verify", "es", "--packs", str(packs), "--work", str(work), "--download")
    assert code == 1 and not out["ok"] and out["checks"]["wheel"] == "failed", out
    assert any("doesn't re-render to its pinned bytes" in p for p in out["problems"])
    assert not (work / "es" / "files" / "canary-es.png").exists()


# --------------------------------------------------------------------------- items 5, 9: corpus and report

def test_corpus_and_report_name_every_text_and_report_lists_what_to_commit(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch,
                                                                          capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"])
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    code, out, _err = _run(capsys, "kit", "corpus", "en", *common, "--wikipedia", "Q1,Q2", "--font", FONT_URL)
    assert code == 0 and [t["title"] for t in out["pinned"]] == ["Example Q1", "Example Q2"], out
    assert all(t["licence"] == "CC-BY-SA-4.0" and t["url"].startswith("https://en.wikipedia.org/") for t in out["pinned"])
    shutil.copyfile(packs / "en" / "manifest.json", packs / "en" / "manifest.draft.json")    # as kit init leaves it
    _flow(packs, work, capsys)
    code, out, _err = _run(capsys, "kit", "report", "en", *common)
    assert code == 0, out
    assert "Example Q1: wikipedia `en` revision" in out["body"] and "Example Q2: wikipedia" in out["body"]
    assert out["commit"] == ["packs/en/OFL.txt", "packs/en/bench.json", "packs/en/corpus.json", "packs/en/manifest.json"]
    assert out["leave_out"] == ["packs/en/manifest.draft.json"] and "missing" not in out


# --------------------------------------------------------------------------- item 8: the mixed-script page

def test_a_latin_packs_second_script_is_greek_before_thai(tmp_path: Path):
    greek = "".join(R.MIXED_SAMPLES["Greek"])
    lg = build.write_box_font(tmp_path / "ExampleLG-Regular.ttf", latin_chars() + greek, family="Example LG")
    th = tmp_path / "ExampleThai-Regular.ttf"
    th.write_bytes(thai_font())
    c = corpus_for(pages={"prose": 0, "form": 0, "microprint": 0, "mixed_script": 1})
    pages = R.render_pages(c, [th, lg], tmp_path / "pages", texts={c.texts[0].sha256: TEXT})
    truth = pages[0].truth
    assert pages[0].kind == "mixed_script" and R.MIXED_SAMPLES["Greek"][0] in truth
    assert not any(0x0E00 <= ord(ch) <= 0x0E7F for ch in truth)


# --------------------------------------------------------------------------- items 1, 2: dev-install.sh

FAKE_UV = r'''
import os, sys
from pathlib import Path
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("uv " + " ".join(a) + "\n")
if a[0] == "venv":
    (Path(a[-1]) / "bin").mkdir(parents=True)
elif a[0] != "pip":
    sys.exit(99)
'''

FAKE_DATAWHEEL = r'''
import json, sys
from pathlib import Path


def wheel_problems(path, manifest):
    return [] if Path(path).read_bytes() == b"fake wheel" else ["not the fake wheel"]


if __name__ == "__main__":
    a = sys.argv[1:]
    m = json.loads(Path(a[1], "manifest.json").read_text(encoding="utf-8"))
    Path(a[3], f"pdfintake_data_{m['id']}-{m['version']}-py3-none-any.whl").write_bytes(b"fake wheel")
'''


def _bash() -> str:
    return "/bin/bash" if sys.platform == "darwin" and Path("/bin/bash").exists() else (shutil.which("bash") or "bash")


def _install(tmp: Path, **env: str) -> subprocess.CompletedProcess[str]:
    repo = tmp / "repo"
    if not repo.exists():
        (repo / "scripts").mkdir(parents=True)
        (repo / "src" / "pdfintake").mkdir(parents=True)
        (repo / "packs" / "en").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "dev-install.sh", repo / "scripts" / "dev-install.sh")
        (repo / "src" / "pdfintake" / "datawheel.py").write_text(FAKE_DATAWHEEL, encoding="utf-8")
        (repo / "packs" / "en" / "manifest.json").write_text(json.dumps({"id": "en", "version": "1.0.0", "base": True}),
                                                            encoding="utf-8")
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "uv_fake.py").write_text(FAKE_UV, encoding="utf-8")
        (bin_dir / "uv").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "uv_fake.py"}" "$@"\n', encoding="utf-8")
        (bin_dir / "uv").chmod(0o755)
    e = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PDFINTAKE_DIST", "UV", "GITHUB_PATH"}}
    e.update(PATH=f"{tmp / 'bin'}{os.pathsep}{e['PATH']}", FAKE_LOG=str(tmp / "calls.log"), PYTHONDONTWRITEBYTECODE="1",
             **env)
    return subprocess.run([_bash(), str(repo / "scripts" / "dev-install.sh")], cwd=tmp, env=e, capture_output=True,
                          text=True, timeout=120)


@posix
def test_dev_install_says_how_to_reach_pdfintake(tmp_path: Path):
    proc = _install(tmp_path)
    assert proc.returncode == 0, proc.stderr
    activate = tmp_path / "repo" / ".venv" / "bin" / "activate"
    assert f'next: source "{activate}"' in proc.stdout and "active virtual environment" not in proc.stderr


@posix
def test_dev_install_says_when_another_environment_takes_the_install(tmp_path: Path):
    other = tmp_path / "another project env"
    (other / "bin").mkdir(parents=True)
    proc = _install(tmp_path, VIRTUAL_ENV=str(other))
    assert proc.returncode == 0, proc.stderr
    assert f"installing into the active virtual environment {other}" in proc.stderr
    assert "next: source" not in proc.stdout                       # that environment is already active
    own = tmp_path / "repo" / ".venv"
    (own / "bin").mkdir(parents=True)
    proc = _install(tmp_path, VIRTUAL_ENV=str(own))
    assert proc.returncode == 0 and "active virtual environment" not in proc.stderr, proc.stderr


@posix
def test_dev_install_survives_a_python_that_ends_lines_with_crlf(tmp_path: Path):
    """Windows CI: Git Bash keeps the \\r of python's \\r\\n line ends, which made the pack id `en\\r`."""
    _install(tmp_path)                                       # builds the fake repo and bin folder
    (tmp_path / "repo" / ".venv").exists() and shutil.rmtree(tmp_path / "repo" / ".venv")
    crlf = tmp_path / "bin" / "python3"
    crlf.write_text(
        f'#!/bin/sh\n"{sys.executable}" "$@" | "{sys.executable}" -c '
        '"import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().replace(b\'\\\\n\', b\'\\\\r\\\\n\'))"\n',
        encoding="utf-8")
    crlf.chmod(0o755)
    proc = _install(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "\r" not in proc.stdout and "data wheel for en: " in proc.stdout
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "\r" not in calls and "--overrides" in calls


# --------------------------------------------------------------------------- dev-install.sh draws the canaries
#
# A pack's canary has no upstream source, so a fresh clone's dev-install.sh must draw it before
# datawheel.py builds the wheel (without it, the install failed on every fresh clone). These tests run
# the real kit/render.py, through the script's own canary step, in this test environment's Python;
# uv and datawheel.py are fakes, and every pack, text and font is made up.

CANARY_LIBS = ("fpdf2", "uharfbuzz", "pillow", "pypdfium2", "pikepdf", "fonttools")

CANARY_UV = r"""
import os, sys
from pathlib import Path
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("uv " + " ".join(a) + "\n")
if a[0] == "venv":
    target = Path(a[-1])
    (target / "bin").mkdir(parents=True)
    if target.name == "render-env":           # its python logs the job, then runs the real step here
        py = target / "bin" / "python"
        py.write_text(f'#!/bin/sh\n"{sys.executable}" "{os.environ["FAKE_JOBLOG"]}" "$@"\n'
                      f'exec "{sys.executable}" "$@"\n', encoding="utf-8")
        py.chmod(0o755)
elif a[0] != "pip":
    sys.exit(99)
"""

JOB_LOGGER = r"""
import json, os, sys
from pathlib import Path
assert sys.argv[1] == "-I", sys.argv
job = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("render " + json.dumps(job, ensure_ascii=False) + "\n")
"""

# Stands in for datawheel.py: fonts come from FAKE_FONTS instead of upstream, and the build refuses a
# canary whose bytes differ from the manifest's pin, as the real build does.
CANARY_DATAWHEEL = r"""
import sys
if __name__ == "__main__":
    del sys.path[0]            # as the real one does: src/pdfintake/types.py would shadow the stdlib's
import hashlib, json, os, shutil
from pathlib import Path


def _log(line):
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _bytes(m):
    return f"fake wheel {m['id']}".encode()


def wheel_problems(path, manifest):
    return [] if Path(path).read_bytes() == _bytes(manifest) else ["not the fake wheel"]


def download_sources(manifest, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in manifest.get("fonts") or ():
        if not (dest / f["file"]).exists():
            shutil.copyfile(Path(os.environ["FAKE_FONTS"], f["file"]), dest / f["file"])
    _log(f"download {manifest['id']} {len(manifest.get('models') or ())} models")
    return []


if __name__ == "__main__":
    a = sys.argv[1:]
    m = json.loads(Path(a[1], "manifest.json").read_text(encoding="utf-8"))
    files = Path(a[a.index("--files") + 1])
    c = m.get("canary")
    if c:
        png = files / c["file"]
        got = hashlib.sha256(png.read_bytes()).hexdigest() if png.is_file() else None
        if got != c["sha256"]:
            print(json.dumps({"ok": False, "error": f"{png} has SHA-256 {got}; the manifest pins {c['sha256']}"}))
            sys.exit(1)
    kind = "data" if m.get("base") else "lang"
    Path(a[3], f"pdfintake_{kind}_{m['id']}-{m['version']}-py3-none-any.whl").write_bytes(_bytes(m))
    _log(f"build {a[1]}")
"""

# (pack, base, own font, canary text, dpi): en is the base pack; es has no font of its own, so its
# canary is drawn with the base font; eo has its own, wider font, drawn first, with the base font after.
CANARY_PACKS = (("en", True, None, ["example", "4521"], 300),
                ("es", False, None, ["sample", "4521"], 300),
                ("eo", False, "ExampleWide-Regular.ttf", ["test", "4521"], 150))


class CanaryRepo:
    """A fresh clone's shape: the real package (fake datawheel.py), the real pyproject.toml, three packs."""

    def __init__(self, tmp: Path) -> None:
        pytest.importorskip("uharfbuzz")
        self.tmp, self.repo, self.bin, self.log = tmp, tmp / "repo", tmp / "bin", tmp / "calls.log"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "dev-install.sh", self.repo / "scripts" / "dev-install.sh")
        shutil.copy(ROOT / "pyproject.toml", self.repo / "pyproject.toml")
        (self.repo / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "canary-env.txt", self.repo / "tools" / "canary-env.txt")
        shutil.copytree(ROOT / "src" / "pdfintake", self.repo / "src" / "pdfintake",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.repo / "src" / "pdfintake" / "datawheel.py").write_text(CANARY_DATAWHEEL, encoding="utf-8")
        self.fonts = tmp / "upstream fonts"
        self.fonts.mkdir()
        latin = build.box_font(latin_chars(), family="Example Sans")
        (self.fonts / "ExampleSans-Regular.ttf").write_bytes(latin)
        (self.fonts / "ExampleWide-Regular.ttf").write_bytes(build.box_font(latin_chars(), family="Example Wide",
                                                                             advance=760))
        self.manifests: dict[str, dict[str, Any]] = {}
        for pid, base, own, must_find, dpi in CANARY_PACKS:
            fonts = [own] if own else (["ExampleSans-Regular.ttf"] if base else [])
            draw = [self.fonts / f for f in fonts] + ([] if base else [self.fonts / "ExampleSans-Regular.ttf"])
            png = R.render_canary(" ".join(must_find), draw, tmp / "pins" / f"canary-{pid}.png", dpi=dpi)
            data = png.read_bytes()
            m: dict[str, Any] = {"id": pid, "version": "1.0.0", "fonts": [{"file": f} for f in fonts],
                                 "canary": {"file": f"canary-{pid}.png", "bytes": len(data),
                                            "sha256": hashlib.sha256(data).hexdigest(), "dpi": dpi,
                                            "must_find": must_find}}
            if base:
                m["base"] = True
            self.manifests[pid] = m
            self.write(pid)
        self.bin.mkdir()
        (self.bin / "uv_fake.py").write_text(CANARY_UV, encoding="utf-8")
        (self.bin / "uv").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.bin / "uv_fake.py"}" "$@"\n',
                                     encoding="utf-8")
        (self.bin / "uv").chmod(0o755)
        (self.bin / "joblog.py").write_text(JOB_LOGGER, encoding="utf-8")

    def write(self, pid: str) -> None:
        (self.repo / "packs" / pid).mkdir(parents=True, exist_ok=True)
        (self.repo / "packs" / pid / "manifest.json").write_text(json.dumps(self.manifests[pid], indent=2),
                                                                 encoding="utf-8")

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        self.log.unlink(missing_ok=True)
        e = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PDFINTAKE_DIST", "UV", "GITHUB_PATH"}}
        e.update(PATH=f"{self.bin}{os.pathsep}{e['PATH']}", FAKE_LOG=str(self.log), FAKE_FONTS=str(self.fonts),
                 FAKE_JOBLOG=str(self.bin / "joblog.py"), PYTHONDONTWRITEBYTECODE="1", **env)
        return subprocess.run([_bash(), str(self.repo / "scripts" / "dev-install.sh"), *args], cwd=self.tmp, env=e,
                              capture_output=True, text=True, timeout=300)

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def renders(self) -> dict[str, dict[str, Any]]:
        jobs = [json.loads(c[len("render "):]) for c in self.calls() if c.startswith("render ")]
        return {Path(j["dest"]).name: j for j in jobs}

    def wheel(self, pid: str) -> Path:
        kind = "data" if self.manifests[pid].get("base") else "lang"
        return self.repo / "dist" / f"pdfintake_{kind}_{pid}-1.0.0-py3-none-any.whl"


def _canary_env_pins() -> list[str]:
    """What tools/canary-env.txt pins the canary renderer's libraries to, as dev-install.sh installs them."""
    found: dict[str, str] = {}
    for line in (ROOT / "tools" / "canary-env.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            found[re.sub(r"[-_.]+", "-", name.strip()).lower()] = version.strip()
    assert set(found) == set(CANARY_LIBS), found
    return [f"{n}=={found[n]}" for n in CANARY_LIBS]


@pytest.fixture
def canary_repo(tmp_path: Path) -> CanaryRepo:
    return CanaryRepo(tmp_path)


@posix
def test_dev_install_draws_each_canary_before_the_build(canary_repo: CanaryRepo):
    proc = canary_repo.run("--packs", "all")
    assert proc.returncode == 0, proc.stderr
    calls = canary_repo.calls()
    # One throwaway environment for every pack, holding the libraries at tools/canary-env.txt's pins.
    made = [c for c in calls if c.startswith("uv venv") and c.endswith("render-env")]
    assert len(made) == 1, calls
    env_python = made[0].split()[-1] + "/bin/python"
    installs = [c for c in calls if c.startswith(f"uv pip install --python {env_python} ")]
    assert [c.split()[5:] for c in installs] == [_canary_env_pins()]
    # Each canary is drawn from the manifest's text, dpi and fonts (the pack's own first, then the
    # base pack's), into the folder the build reads, before that build.
    jobs = canary_repo.renders()
    assert sorted(jobs) == ["canary-en.png", "canary-eo.png", "canary-es.png"]
    want_fonts = {"en": ["ExampleSans-Regular.ttf"], "es": ["ExampleSans-Regular.ttf"],
                  "eo": ["ExampleWide-Regular.ttf", "ExampleSans-Regular.ttf"]}
    for pid, _base, _own, must_find, dpi in CANARY_PACKS:
        job = jobs[f"canary-{pid}.png"]
        assert job["text"] == " ".join(must_find) and job["dpi"] == dpi, job
        assert [Path(f).name for f in job["fonts"]] == want_fonts[pid], job
        assert Path(job["src"]) == canary_repo.repo / "src"
        assert calls.index("render " + json.dumps(job, ensure_ascii=False)) < calls.index(f"build packs/{pid}")
        assert canary_repo.wheel(pid).read_bytes() == f"fake wheel {pid}".encode()
    assert any(c.startswith("uv pip install --python") and "--overrides" in c for c in calls)
    # A wheel already sound is reused: nothing is drawn, built or installed for the canaries again.
    proc = canary_repo.run("--packs", "all")
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith(("render", "build", "download")) or c.endswith("render-env")
                   for c in canary_repo.calls()), canary_repo.calls()


@posix
def test_dev_install_stops_on_a_canary_that_differs_from_its_pin(canary_repo: CanaryRepo):
    canary_repo.manifests["es"]["canary"]["sha256"] = "0" * 64
    canary_repo.write("es")
    proc = canary_repo.run("--packs", "es")
    assert proc.returncode == 1
    assert "the manifest pins " + "0" * 64 in proc.stderr and "pdfintake kit wheel es" in proc.stderr, proc.stderr
    assert not canary_repo.wheel("es").exists()
    assert not any("--overrides" in c for c in canary_repo.calls()), "nothing is installed after a failed build"


@posix
def test_dev_install_wheels_only_builds_the_named_packs_and_installs_nothing(canary_repo: CanaryRepo):
    proc = canary_repo.run("--wheels-only", "--packs", "es", "--python", "3.12")
    assert proc.returncode == 0, proc.stderr
    calls = canary_repo.calls()
    assert [c for c in calls if c.startswith("build")] == ["build packs/es"]
    assert any(c.startswith("uv venv --python 3.12 ") and c.endswith("render-env") for c in calls), calls
    assert not any(c.startswith("uv venv") and c.endswith(".venv") for c in calls)
    assert not any("--overrides" in c for c in calls)
    assert sorted(p.name for p in (canary_repo.repo / "dist").iterdir()) == [canary_repo.wheel("es").name]
    assert "nothing installed" in proc.stdout
    proc = canary_repo.run("--wheels-only", "--tool")
    assert proc.returncode == 2 and "--wheels-only" in proc.stderr


@posix
def test_dev_install_needs_every_renderer_pin_in_canary_env(canary_repo: CanaryRepo):
    env = canary_repo.repo / "tools" / "canary-env.txt"
    env.write_text("".join(ln for ln in env.read_text(encoding="utf-8").splitlines(keepends=True)
                           if not ln.startswith("fpdf2")), encoding="utf-8")
    proc = canary_repo.run()
    assert proc.returncode == 1 and "tools/canary-env.txt has no pin for fpdf2" in proc.stderr, proc.stderr


@posix
def test_dev_install_draws_canaries_under_a_python_that_ends_lines_with_crlf(canary_repo: CanaryRepo):
    """The Windows fix holds on the canary path too: every value read back still loses its \\r."""
    crlf = canary_repo.bin / "python3"
    crlf.write_text(
        f'#!/bin/sh\n"{sys.executable}" "$@" | "{sys.executable}" -c '
        '"import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().replace(b\'\\\\n\', b\'\\\\r\\\\n\'))"\n',
        encoding="utf-8")
    crlf.chmod(0o755)
    proc = canary_repo.run("--packs", "all")
    assert proc.returncode == 0, proc.stderr
    assert "\r" not in proc.stdout and "data wheel for eo: " in proc.stdout
    assert sorted(canary_repo.renders()) == ["canary-en.png", "canary-eo.png", "canary-es.png"]
