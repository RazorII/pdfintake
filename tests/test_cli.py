"""cli.py: arguments, exit codes, the JSON a skill reads, and that no command ever asks.

The run and apply tests replace ``flow.run`` and ``flow.apply`` with recorders, so they test what
the CLI hands to flow and prints, not the pipeline (tests/test_flow.py covers that). ``export`` and
``passwords add`` run for real on files built here. Two tests start ``python -m pdfintake`` in a
subprocess: one with stdin closed and a cp1252 console, one with stdin a pipe nobody writes to.
Every file name, text and password is made up.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake import __version__, cli, flow, record
from pdfintake.errors import EXIT_FAILED, EXIT_OK, EXIT_REFUSED, EXIT_SETUP, EXIT_USAGE, Refused, SetupError, UsageError

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PASSWORD = "made up cli phrase"              # a made-up password


@pytest.fixture(autouse=True)
def _private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("PDFINTAKE_CACHE", str(cache))
    return cache


def call(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def call_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code, out, _ = call(capsys, *argv)
    data = json.loads(out)
    assert data["exit"] == code
    return code, data


def fingerprint(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns


def make_record(inputs: list[Path], *, exit: int = 0, status: str = "done", out: Path | None = None) -> record.Record:
    rec = record.new(inputs, opts=flow.RunOptions(), seal=None, platform_key="darwin-arm64")
    rec.status, rec.exit = status, exit  # type: ignore[assignment]
    rec.input["kind"] = "pdf"
    if out is not None:
        rec.output = record.output_entry(out, "0" * 64, 1234)
    rec.notes.append(record.note("input_unchanged"))
    return rec


class Recorder:
    """Stands in for flow.run / flow.apply: remembers the call, returns the records it was given."""

    def __init__(self, records: list[record.Record] | None = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.records = records or []
        self.raises = raises

    def run(self, inputs: Any, opts: flow.RunOptions) -> list[record.Record]:
        self.calls.append((list(inputs), opts))
        if self.raises is not None:
            raise self.raises
        return self.records

    def apply(self, record_id: str, **kw: Any) -> list[record.Record]:
        self.calls.append((record_id, kw))
        if self.raises is not None:
            raise self.raises
        return self.records


# --------------------------------------------------------------------------- version, help, arguments

def test_version(capsys: pytest.CaptureFixture[str]):
    code, out, _ = call(capsys, "--version")
    assert code == 0 and out.strip() == f"pdfintake {__version__}"


def test_help_exits_0(capsys: pytest.CaptureFixture[str]):
    code, out, _ = call(capsys, "run", "--help")
    assert code == 0 and "--paths-from" in out and "--passwords-file" in out


@pytest.mark.parametrize("argv", [
    ["frobnicate"],
    [],
    ["run", "a.pdf", "--bogus"],
    ["run", "a.pdf", "--fetch", "ask"],                         # §12.3: ask is gone
    ["run", "a.pdf", "--smaller", "ask"],                       # §12.4: a plain flag; "ask" isn't an input here...
    ["run", "a.pdf", "--provenance", "mine"],
    ["run", "a.pdf", "--workers", "two"],
    ["run", "a.pdf", "-o", "x.pdf", "--outdir", "d"],
    ["run", "a.pdf", "--password-file", "pw.txt"],              # §12.1: only --passwords-file exists
    ["run", "a.pdf", "--password-stdin"],
    ["run", "a.pdf", "--passwords", "pw.txt"],                  # no abbreviations
    ["run", "a.pdf", "--lang", ""],                             # an empty variable in a caller's script
    ["run", "a.pdf", "--lang", ","],
    ["run", "a.pdf", "--force-lang", " , "],
    ["run", "a.pdf", "--force-lang="],
    ["apply", "r-20261105-1a2b3c4d"],                           # an answer is required
    ["apply", "r-20261105-1a2b3c4d", "--choice", "smaller", "--lang", "hu"],
    ["apply", "r-20261105-1a2b3c4d", "--choice", "keep"],       # §12.5: only smaller
    ["apply", "r-20261105-1a2b3c4d", "--provenance", "ordinary"],
    ["export", "clean", "a.pdf", "--provenance", "ordinary"],   # -o is required
    ["export", "flatten", "a.pdf", "-o", "b.pdf"],              # --provenance is required
    ["export", "flatten", "a.pdf", "-o", "b.pdf", "--provenance", "ordinary", "--strip-dates"],
    ["passwords", "add"],
    ["passwords", "add", "--from-file", "x", "--password", PASSWORD],
    ["packs"],
    ["record", "show"],
])
def test_argument_errors_exit_2(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    rec = Recorder([make_record([Path("/tmp/a.pdf")])])
    monkeypatch.setattr(flow, "run", rec.run)
    code, out, err = call(capsys, *argv)
    if argv[:3] == ["run", "a.pdf", "--smaller"]:
        # "--smaller ask" is the flag plus a second input (inputs may follow options); flow then
        # reports the inputs.
        assert code == 0 and [p.name for p in rec.calls[0][0]] == ["a.pdf", "ask"]
        return
    assert code == EXIT_USAGE
    assert PASSWORD not in out + err
    assert not rec.calls


def test_an_argument_error_with_json_is_one_json_object(capsys: pytest.CaptureFixture[str]):
    code, data = call_json(capsys, "run", "a.pdf", "--fetch", "ask", "--json")
    assert code == EXIT_USAGE
    assert data["records"] == [] and data["error"]["code"] == "bad_arguments"
    code, data = call_json(capsys, "doctor", "--bogus", "--json")
    assert code == EXIT_USAGE and "records" not in data


def test_an_empty_language_is_a_usage_error_naming_the_flag(capsys: pytest.CaptureFixture[str],
                                                            monkeypatch: pytest.MonkeyPatch):
    rec = Recorder([make_record([Path("/tmp/a.pdf")])])
    monkeypatch.setattr(flow, "run", rec.run)
    code, data = call_json(capsys, "run", "a.pdf", "--force-lang", ",", "--json")
    assert code == EXIT_USAGE and data["records"] == []
    assert data["error"]["code"] == "bad_language_code" and "--force-lang" in data["error"]["message"]
    assert not rec.calls


SECRET = "Invented-Secret-55"                         # a made-up value that must not be echoed


@pytest.mark.parametrize("argv", [
    ["run", "a.pdf", "--provenance", SECRET],                                   # invalid choice
    ["run", "a.pdf", "--provenance", f"it's {SECRET} (choose from x"],          # a value that mimics argparse
    ["run", "a.pdf", "--fetch", SECRET],
    ["run", "a.pdf", "--workers", SECRET],                                      # invalid int value
    ["apply", "r-20261105-1a2b3c4d", "--choice", SECRET],
    ["export", "clean", "a.pdf", "-o", "b.pdf", "--provenance", SECRET],
    [SECRET],                                                                   # invalid command
    ["doctor", f"--{SECRET}", SECRET],                                          # unrecognized
])
@pytest.mark.parametrize("as_json", [True, False])
def test_an_argument_error_never_echoes_a_value(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
                                                argv: list[str], as_json: bool):
    monkeypatch.setattr(flow, "run", Recorder([make_record([Path("/tmp/a.pdf")])]).run)
    code, out, err = call(capsys, *argv, *(["--json"] if as_json else []))
    assert code == EXIT_USAGE
    shown = out + err
    if argv[-2:] == [f"--{SECRET}", SECRET]:
        # An unknown option is named (it is an option, not a value); the value after it is not.
        assert shown.count(SECRET) == 1 and "1 other value" in shown
    else:
        assert SECRET not in shown and cli.VALUE_PLACEHOLDER in shown
    if as_json:
        assert json.loads(out)["error"]["code"] == "bad_arguments"


def test_argparse_lists_survive_the_scrub(capsys: pytest.CaptureFixture[str]):
    code, data = call_json(capsys, "run", "a.pdf", "--provenance", SECRET, "--json")
    message = data["error"]["message"]
    assert code == EXIT_USAGE and "--provenance" in message
    assert "ordinary" in message and "counterparty" in message and "evidence" in message


# --------------------------------------------------------------------------- run

def test_run_hands_every_flag_to_flow(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Path):
    out = tmp_path / "Scan (searchable).pdf"
    rec = Recorder([make_record([tmp_path / "Scan.pdf"], out=out)])
    monkeypatch.setattr(flow, "run", rec.run)
    pw = tmp_path / "pw"
    code, data = call_json(capsys, "run", str(tmp_path / "Scan.pdf"), "--provenance", "evidence",
                           "--outdir", str(tmp_path), "--lang", "th,en", "--force-lang", "hu", "--fetch", "never",
                           "--smaller", "--separate", "--redo-ocr", "--keep-existing-layer", "--overwrite", "--plan",
                           "--workers", "3", "--passwords-file", str(pw), "--json")
    assert code == 0
    inputs, opts = rec.calls[0]
    assert inputs == [tmp_path / "Scan.pdf"]
    assert opts == flow.RunOptions(provenance="evidence", out=None, outdir=tmp_path, lang=("th", "en"),
                                   force_lang=("hu",), fetch="never", smaller=True, overwrite=True, redo_ocr=True,
                                   keep_existing_layer=True, separate=True, plan_only=True, workers=3,
                                   passwords_file=pw)
    assert [r["id"] for r in data["records"]] == [rec.records[0].id]
    assert data["records"][0]["output"]["path"] == str(out)


def test_inputs_may_follow_options(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    rec = Recorder([make_record([Path("/tmp/a.pdf")])])
    monkeypatch.setattr(flow, "run", rec.run)
    assert call_json(capsys, "run", "a.pdf", "--smaller", "b.pdf", "--json", "c.pdf")[0] == 0
    assert [p.name for p in rec.calls[0][0]] == ["a.pdf", "b.pdf", "c.pdf"] and rec.calls[0][1].smaller


def test_run_defaults_ask_nothing(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rec = Recorder([make_record([tmp_path / "a.pdf"])])
    monkeypatch.setattr(flow, "run", rec.run)
    code, _ = call_json(capsys, "run", str(tmp_path / "a.pdf"), "-o", str(tmp_path / "b.pdf"), "--json")
    assert code == 0
    opts = rec.calls[0][1]
    assert opts.provenance == "ordinary" and opts.fetch == "auto" and opts.smaller is False
    assert opts.out == tmp_path / "b.pdf" and opts.passwords_file is None and opts.workers == flow.DEFAULT_WORKERS


def test_the_process_exit_is_the_highest_priority_record(capsys: pytest.CaptureFixture[str],
                                                        monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    recs = [make_record([tmp_path / "a.pdf"]),
            make_record([tmp_path / "b.pdf"], exit=EXIT_REFUSED, status="refused"),
            make_record([tmp_path / "c.pdf"], exit=EXIT_FAILED, status="failed")]
    monkeypatch.setattr(flow, "run", Recorder(recs).run)
    code, data = call_json(capsys, "run", "a.pdf", "b.pdf", "c.pdf", "--json")
    assert code == EXIT_FAILED                                    # 2 > 3 > 1 > 4 > 5 > 0
    assert [r["exit"] for r in data["records"]] == [0, 4, 1]


def test_an_invocation_that_cant_run_is_one_json_error(capsys: pytest.CaptureFixture[str],
                                                       monkeypatch: pytest.MonkeyPatch):
    err = UsageError("mixed_inputs", "PDFs and images can't be mixed in one run; run them separately.")
    monkeypatch.setattr(flow, "run", Recorder(raises=err).run)
    code, data = call_json(capsys, "run", "a.pdf", "b.jpg", "--json")
    assert code == EXIT_USAGE and data["records"] == []
    assert data["error"] == {"code": "mixed_inputs", "message": err.message}
    monkeypatch.setattr(flow, "run", Recorder(raises=SetupError("index_missing", "no index (made up)")).run)
    code, out, err_text = call(capsys, "run", "a.pdf")
    assert code == EXIT_SETUP and out == "" and "no index (made up)" in err_text


def test_an_unexpected_error_names_only_its_kind(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flow, "run", Recorder(raises=ValueError(f"library text quoting {PASSWORD}")).run)
    code, data = call_json(capsys, "run", "a.pdf", "--json")
    assert code == EXIT_FAILED and data["error"]["code"] == "internal_error"
    assert "ValueError" in data["error"]["message"] and PASSWORD not in json.dumps(data)


def test_a_record_that_cant_be_reported_is_still_one_json_object(capsys: pytest.CaptureFixture[str],
                                                                 monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """record.overall_exit raises on an exit code no record may carry; the CLI still prints one object."""
    bad = make_record([tmp_path / "a.pdf"], exit=99, status="failed")
    monkeypatch.setattr(flow, "run", Recorder([bad]).run)
    monkeypatch.setattr(flow, "apply", Recorder([bad]).apply)
    for argv in (["run", "a.pdf", "--json"], ["apply", "r-20261105-1a2b3c4d", "--choice", "smaller", "--json"]):
        code, data = call_json(capsys, *argv)
        assert code == EXIT_FAILED and data["records"] == [] and data["error"]["code"] == "internal_error"
    code, out, err = call(capsys, "run", "a.pdf")
    assert code == EXIT_FAILED and out == "" and "unexpected ValueError" in err


def test_an_interrupt_is_130(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flow, "run", Recorder(raises=KeyboardInterrupt()).run)
    code, _, err = call(capsys, "run", "a.pdf")
    assert code == 130 and "interrupted" in err


def test_human_output_escapes_control_characters(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path):
    src = tmp_path / "bad\x1b[2Jname\nnext.pdf"
    rec = make_record([src], out=tmp_path / "out.pdf")
    rec.warnings.append(record.warning("agreement_low", page=2))
    rec.offers.append(record.Offer("smaller", "--choice", "A smaller copy would save 40% (made up).",
                                   [record.Option("smaller", "Save it", "Saves it beside.")]))
    monkeypatch.setattr(flow, "run", Recorder([rec]).run)
    code, out, _ = call(capsys, "run", str(src))
    assert code == 0
    assert "\x1b" not in out and "bad\\x1b[2Jname\\nnext.pdf" in out
    assert "warning agreement_low (page 2)" in out
    assert f"pdfintake apply {rec.id} --choice smaller" in out


# --------------------------------------------------------------------------- --paths-from

WEIRD = ["$(touch pwned).pdf", "it's \"quoted\".pdf", "  spaced  .pdf", "-leading-dash.pdf", "ตัวอย่าง 📎.pdf",
         "semi;colon`back`.pdf"]


def test_paths_from_keeps_every_name_exactly(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path):
    rec = Recorder([make_record([tmp_path / "x.pdf"])])
    monkeypatch.setattr(flow, "run", rec.run)
    names = [str(tmp_path / n) for n in WEIRD]
    listing = tmp_path / "list.txt"
    listing.write_bytes(("﻿" + "\r\n".join(names[:3]) + "\r\n\n" + "\n".join(names[3:]) + "\n").encode("utf-8"))
    code, _ = call_json(capsys, "run", str(tmp_path / "first.pdf"), "--paths-from", str(listing), "--json")
    assert code == 0
    assert [str(p) for p in rec.calls[0][0]] == [str(tmp_path / "first.pdf"), *names]
    assert not (Path.cwd() / "pwned").exists() and not (tmp_path / "pwned").exists()


def test_paths_from_relative_names_are_taken_from_the_current_folder(capsys: pytest.CaptureFixture[str],
                                                                     monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rec = Recorder([make_record([tmp_path / "x.pdf"])])
    monkeypatch.setattr(flow, "run", rec.run)
    (tmp_path / "l.txt").write_text("sub/one.pdf\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "--paths-from", "l.txt", "--json"]) == 0
    capsys.readouterr()
    assert rec.calls[0][0] == [Path("sub/one.pdf")]


@pytest.mark.parametrize("problem", ["missing", "folder", "not_utf8", "stdin_dash", "fifo"])
def test_a_paths_from_list_that_cant_be_read_is_exit_2_and_never_waits(capsys: pytest.CaptureFixture[str],
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      tmp_path: Path, problem: str):
    rec = Recorder([make_record([tmp_path / "x.pdf"])])
    monkeypatch.setattr(flow, "run", rec.run)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "list.txt"
    if problem == "folder":
        target.mkdir()
    elif problem == "not_utf8":
        target.write_bytes(b"caf\xe9.pdf\n")
    elif problem == "stdin_dash":
        target = Path("-")
        monkeypatch.setattr(sys, "stdin", _NoReads())
    elif problem == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs on this platform")
        os.mkfifo(target)                                   # opening it to read would wait for a writer
    code, data = call_json(capsys, "run", "--paths-from", str(target), "--json")
    assert code == EXIT_USAGE and data["error"]["code"].startswith("paths_from_")
    assert not rec.calls


# --------------------------------------------------------------------------- apply, record show

def test_apply_hands_the_answer_to_flow(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path):
    child = make_record([tmp_path / "a.pdf"], out=tmp_path / "a (searchable).pdf")
    rec = Recorder([child])
    monkeypatch.setattr(flow, "apply", rec.apply)
    code, data = call_json(capsys, "apply", "r-20261105-1a2b3c4d", "--lang", "hu", "--passwords-file",
                           str(tmp_path / "pw"), "--json")
    assert code == 0 and data["records"][0]["id"] == child.id
    assert rec.calls[0] == ("r-20261105-1a2b3c4d", {"choice": None, "lang": "hu",
                                                    "passwords_file": tmp_path / "pw"})
    code, _ = call_json(capsys, "apply", "r-20261105-1a2b3c4d", "--choice", "smaller", "--json")
    assert rec.calls[1] == ("r-20261105-1a2b3c4d", {"choice": "smaller", "lang": None, "passwords_file": None})


def test_apply_with_a_path_for_a_record_id_is_exit_2(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    code, data = call_json(capsys, "apply", str(tmp_path / "r-20261105-1a2b3c4d.json"), "--choice", "smaller",
                           "--json")
    assert code == EXIT_USAGE and data["error"]["code"] == "bad_record_id" and data["records"] == []
    code, data = call_json(capsys, "apply", "r-20261105-1a2b3c4d", "--choice", "smaller", "--json")
    assert code == EXIT_USAGE and data["error"]["code"] == "unknown_record"


def test_record_show(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    rec = make_record([tmp_path / "Letter.pdf"], out=tmp_path / "Letter (searchable).pdf")
    record.save(rec)
    code, data = call_json(capsys, "record", "show", rec.id, "--json")
    assert code == 0 and data["record"]["id"] == rec.id
    code, out, _ = call(capsys, "record", "show", rec.id)
    assert code == 0 and "Letter (searchable).pdf" in out
    code, data = call_json(capsys, "record", "show", "../../etc/passwd", "--json")
    assert code == EXIT_USAGE and data["error"]["code"] == "bad_record_id"


@pytest.mark.parametrize("damage", ["not_utf8", "a_folder"])
def test_record_show_of_a_record_file_that_cant_be_read_is_one_json_object(capsys: pytest.CaptureFixture[str],
                                                                           damage: str):
    records = Path(os.environ["PDFINTAKE_CACHE"]) / "records"
    records.mkdir(parents=True, exist_ok=True)
    rid = "r-20261105-1a2b3c4d"
    if damage == "not_utf8":
        (records / f"{rid}.json").write_bytes(b"\xff\xfe")
    else:
        (records / f"{rid}.json").mkdir()
    code, out, err = call(capsys, "record", "show", rid, "--json")
    data = json.loads(out)                                   # exactly one JSON object, nothing else
    assert code == data["exit"] and code in (EXIT_FAILED, EXIT_USAGE)
    assert data["error"]["code"] in ("internal_error", "record_invalid") and "record" not in data
    code, out, err = call(capsys, "record", "show", rid)
    assert code in (EXIT_FAILED, EXIT_USAGE) and out == "" and err


# --------------------------------------------------------------------------- doctor, packs

def test_doctor_exit_follows_blocking(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    from pdfintake import doctor

    rep = {"pdfintake": "0.1.0", "platform": {"key": "darwin-arm64"}, "engines": {}, "base_pack": {"ok": False,
           "message": "the base pack is missing (made up)"}, "packs": [], "problems": ["made up"], "blocking": True,
           "exit": 3, "cache": "/tmp/x"}
    monkeypatch.setattr(doctor, "report", lambda **kw: dict(rep))
    code, data = call_json(capsys, "doctor", "--json")
    assert code == EXIT_SETUP and data["blocking"] is True
    code, out, _ = call(capsys, "doctor")
    assert code == EXIT_SETUP and "not ready (exit 3)" in out
    monkeypatch.setattr(doctor, "report", lambda **kw: {**rep, "blocking": False, "problems": [], "exit": 0})
    assert call_json(capsys, "doctor", "--canary", "--json")[0] == EXIT_OK


def test_packs_list_show_and_unknown(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    from pdfintake.langpacks import index as I
    from pdfintake.langpacks import store

    idx = I.load(REPO / "tests" / "fixtures" / "index.test.json")
    monkeypatch.setattr(I, "load", lambda path=None: idx)
    monkeypatch.setattr(store, "verify", lambda pack_id=None: [store.VerifyResult(pack_id or "en", pack_id in (None, "en"),
                                                                                   [] if pack_id in (None, "en") else ["not installed (made up)"])])
    code, data = call_json(capsys, "packs", "list", "--json")
    assert code == 0 and {"en", "th"} <= {p["id"] for p in data["packs"]}
    assert next(p for p in data["packs"] if p["id"] == "en")["installed"] is True
    code, data = call_json(capsys, "packs", "show", "th", "--json")
    assert code == 0 and data["pack"]["installed"] is False and data["pack"]["problems"]
    code, data = call_json(capsys, "packs", "show", "xx", "--json")
    assert code == EXIT_USAGE and data["error"]["code"] == "unknown_pack"
    code, data = call_json(capsys, "packs", "path", "--json")
    assert code == 0 and data["cache"] == os.environ["PDFINTAKE_CACHE"]


# --------------------------------------------------------------------------- export

def _scan_pdf(path: Path) -> Path:
    return build.write_pdf(path, [build.PageSpec(images=(build.ImageSpec(build.text_like(600, 800, lines=4)),))])


def test_export_clean_writes_through_commit_and_never_touches_the_input(capsys: pytest.CaptureFixture[str],
                                                                       monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    src = _scan_pdf(tmp_path / "Scan.pdf")
    before = fingerprint(src)
    commits: list[Path] = []
    real = flow.commit
    monkeypatch.setattr(flow, "commit", lambda built, dest, *a, **k: (commits.append(Path(dest)),
                                                                      real(built, dest, *a, **k))[1])
    dest = tmp_path / "Share.pdf"
    code, data = call_json(capsys, "export", "clean", str(src), "-o", str(dest), "--provenance", "ordinary",
                           "--strip-dates", "--json")
    assert code == 0, data
    assert commits == [dest] and dest.is_file()
    assert data["output"]["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert fingerprint(src) == before
    staging = Path(os.environ["PDFINTAKE_CACHE"]) / "staging"
    assert not [p for p in staging.rglob("*") if p.is_file()]          # the built file doesn't linger
    # An existing output needs --overwrite; with it, it is replaced.
    code, data = call_json(capsys, "export", "flatten", str(src), "-o", str(dest), "--provenance", "ordinary", "--json")
    assert code == EXIT_USAGE and data["error"]["code"] == "output_exists"
    code, data = call_json(capsys, "export", "flatten", str(src), "-o", str(dest), "--provenance", "ordinary",
                           "--overwrite", "--json")
    assert code == 0, data
    assert fingerprint(src) == before


def test_export_refuses_evidence_and_never_writes(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    src = _scan_pdf(tmp_path / "Scan.pdf")
    before = fingerprint(src)
    dest = tmp_path / "Share.pdf"
    code, data = call_json(capsys, "export", "clean", str(src), "-o", str(dest), "--provenance", "evidence", "--json")
    assert code == EXIT_REFUSED and data["refusal"]["code"] == "provenance_evidence"
    assert not dest.exists() and fingerprint(src) == before
    code, data = call_json(capsys, "export", "clean", str(src), "-o", str(src), "--provenance", "ordinary", "--json")
    assert code == EXIT_USAGE and data["error"]["code"] == "output_is_input"
    assert fingerprint(src) == before


# --------------------------------------------------------------------------- passwords add

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_passwords_add_creates_a_private_list_and_never_prints_a_password(capsys: pytest.CaptureFixture[str],
                                                                          tmp_path: Path):
    source = tmp_path / "one-off.txt"
    source.write_text(PASSWORD + "\n", encoding="utf-8")
    target = tmp_path / "config" / "passwords"
    code, out, err = call(capsys, "passwords", "add", "--from-file", str(source), "--passwords-file", str(target),
                          "--json")
    assert code == 0 and json.loads(out) == {"exit": 0, "passwords_file": str(target)}
    assert PASSWORD not in out + err
    assert target.read_text(encoding="utf-8") == PASSWORD + "\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert source.read_text(encoding="utf-8") == PASSWORD + "\n"          # left for the caller to delete
    assert call(capsys, "passwords", "add", "--from-file", str(source), "--passwords-file", str(target))[0] == 0
    assert target.read_text(encoding="utf-8") == PASSWORD + "\n"          # a duplicate is skipped
    os.chmod(target, 0o644)
    code, out, err = call(capsys, "passwords", "add", "--from-file", str(source), "--passwords-file", str(target),
                          "--json")
    assert code == EXIT_USAGE and json.loads(out)["error"]["code"] == "passwords_file_unsafe"
    assert PASSWORD not in out + err


# --------------------------------------------------------------------------- kit

def test_kit_names_the_extra_when_it_is_missing(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    from pdfintake.kit import cli as kit_cli

    called: list[str] = []
    monkeypatch.setitem(kit_cli.COMMANDS, "report", lambda args: (called.append("report"), ({}, 0))[1])
    monkeypatch.setattr(cli, "_kit_extra_present", lambda: False)
    code, out, _ = call(capsys, "kit", "report", "en", "--json")
    data = json.loads(out)
    assert code == EXIT_SETUP and data["error"]["code"] == "kit_extra_missing"
    assert "install pdfintake[kit]" in data["error"]["message"] and not called
    monkeypatch.setattr(cli, "_kit_extra_present", lambda: True)
    code, out, _ = call(capsys, "kit", "report", "en", "--json")
    assert code == 0 and called == ["report"] and json.loads(out)["ok"] is True


def test_the_kit_group_stands_in_when_its_module_cant_load(capsys: pytest.CaptureFixture[str],
                                                           monkeypatch: pytest.MonkeyPatch):
    import pdfintake.kit

    monkeypatch.setitem(sys.modules, "pdfintake.kit.cli", None)       # import fails
    monkeypatch.delattr(pdfintake.kit, "cli", raising=False)
    code, out, _ = call(capsys, "kit", "init", "xx", "--script", "Latin")
    assert code == EXIT_SETUP and "install pdfintake[kit]" in json.loads(out)["error"]["message"]


# --------------------------------------------------------------------------- nothing is ever asked

class _NoReads(io.StringIO):
    def read(self, *a: Any) -> str:
        raise AssertionError("the CLI read standard input")

    readline = read

    def __iter__(self):
        raise AssertionError("the CLI read standard input")


def test_no_command_reads_stdin(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sys, "stdin", _NoReads())
    monkeypatch.setattr(flow, "run", Recorder([make_record([tmp_path / "a.pdf"])]).run)
    for argv in (["--version"], ["run", "a.pdf", "--json"], ["run", "--json"],
                 ["apply", "r-20261105-1a2b3c4d", "--choice", "smaller", "--json"], ["packs", "path"]):
        cli.main(argv)
    capsys.readouterr()


def test_cli_source_has_no_prompt_and_no_exit_10():
    tree = ast.parse((SRC / "pdfintake" / "cli.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("input",), "a prompt"
        if isinstance(node, ast.Attribute):
            assert not (node.attr == "stdin"), "reads stdin"
        if isinstance(node, ast.Name):
            assert node.id not in ("getpass", "EXIT_DECISION")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            assert node.value != 10, "exit 10 is retired"


# --------------------------------------------------------------------------- python -m pdfintake

def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHONIOENCODING")}
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *[p for p in [env.get("PYTHONPATH")] if p]])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


@pytest.mark.cli_subprocess
def test_python_m_with_stdin_closed_and_a_cp1252_console(tmp_path: Path):
    rec = make_record([tmp_path / "ตัวอย่าง 📎.pdf"], out=tmp_path / "ตัวอย่าง 📎 (searchable).pdf")
    record.save(rec)
    env = _env(PYTHONIOENCODING="cp1252")
    done = subprocess.run([sys.executable, "-m", "pdfintake", "--version"], stdin=subprocess.DEVNULL,
                          capture_output=True, env=env, timeout=120, close_fds=True)
    assert done.returncode == 0 and done.stdout.decode("utf-8").strip() == f"pdfintake {__version__}"
    for argv in (["record", "show", rec.id, "--json"], ["record", "show", rec.id]):
        # stdin is closed outright: fd 0 doesn't exist in the child.
        done = subprocess.run([sys.executable, "-c",
                               "import os, sys; os.close(0); from pdfintake.cli import main; sys.exit(main(sys.argv[1:]))",
                               *argv], capture_output=True, env=env, timeout=120)
        assert done.returncode == 0, done.stderr
        assert "ตัวอย่าง 📎 (searchable).pdf" in done.stdout.decode("utf-8")


@pytest.mark.cli_subprocess
def test_a_run_never_waits_on_an_open_stdin(tmp_path: Path):
    """stdin is a pipe nobody writes to: a command that read it would hang until the timeout."""
    env = _env()
    for argv in (["run", "--json"], ["run", str(tmp_path / "missing.pdf"), "--fetch", "never", "--json"],
                 ["apply", "r-20261105-1a2b3c4d", "--choice", "smaller", "--json"]):
        out_file = tmp_path / "out.json"
        with open(out_file, "wb") as out_fh:
            proc = subprocess.Popen([sys.executable, "-m", "pdfintake", *argv], stdin=subprocess.PIPE,
                                    stdout=out_fh, stderr=subprocess.DEVNULL, env=env)
            try:
                proc.wait(timeout=180)                     # stdin stays open and empty the whole time
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail(f"{argv} waited on stdin")
            finally:
                assert proc.stdin is not None
                proc.stdin.close()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert proc.returncode == data["exit"] and proc.returncode != 10
