"""``pdfintake text`` (pdf/layertext.py, cli._cmd_text; DESIGN §16).

Every PDF is built here from made-up text. The command reads the text layer a PDF already has and
never reads a page with OCR, never changes or writes a PDF, and writes no record; a record it is
given (``--record``) is built and saved here too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pikepdf
import pytest

from fixtures import build
from fixtures import garbled as G
from pdfintake import __version__, cli, flow, record
from pdfintake.errors import EXIT_FAILED, EXIT_OK, EXIT_REFUSED, EXIT_STALE, EXIT_USAGE, InputChanged, UsageError
from pdfintake.pdf import classify, layertext, verify

FIRST = "Hello Jane Doe exam-"
SECOND = "ple second line"
OURS = "Invoice 555-0123 for Jane Doe"
OTHER = "Receipt from example.com"
SECRET_WORDS = ("Jane Doe", "Invoice", "Receipt", "example.com", "555-0123")


@pytest.fixture(autouse=True)
def _private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("PDFINTAKE_CACHE", str(cache))
    return cache


def _image() -> Any:
    return build.text_like(400, 300)


def _ours(path: Path, index: int) -> None:
    """Give page ``index`` pdfintake's own layer marker (what pdf/textlayer.write_layer adds)."""
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.pages[index].obj.PieceInfo = pikepdf.Dictionary(
            {classify.MARKER_KEY: pikepdf.Dictionary(LastModified=pikepdf.String("D:20261117000000Z"))})
        pdf.save(path, deterministic_id=True)


def mixed_pdf(path: Path) -> Path:
    """Page 1 born-digital text, 2 a scan with no layer, 3 a scan with pdfintake's layer, 4 empty,
    5 a scan with another tool's invisible layer."""
    img = _image()
    build.write_pdf(path, [
        build.PageSpec(texts=(build.TextSpec(FIRST, y=720), build.TextSpec(SECOND, y=700))),
        build.PageSpec(images=(build.ImageSpec(img),)),
        build.PageSpec(images=(build.ImageSpec(img),), texts=(build.TextSpec(OURS, mode=3),)),
        build.PageSpec(),
        build.PageSpec(images=(build.ImageSpec(img),), texts=(build.TextSpec(OTHER, mode=3),)),
    ])
    _ours(path, 2)
    return path


def scan_pdf(path: Path, pages: int = 2) -> Path:
    img = _image()
    return build.write_pdf(path, [build.PageSpec(images=(build.ImageSpec(img),)) for _ in range(pages)])


def call(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def jsonl(out: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in out.splitlines()]
    return rows[0], rows[1:]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- the export

def test_jsonl_is_a_header_then_one_line_per_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    code, out, err = call(capsys, "text", str(pdf))
    assert code == EXIT_OK and err == ""
    header, pages = jsonl(out)
    assert header["format"] == "pdfintake-text" and header["format_version"] == 1
    assert header["source_sha256"] == sha(pdf)
    assert header["tool_version"] == __version__ and header["pages"] == 5 and header["range"] == [1, 5]
    assert header["producer"]["tool"] == "pdfintake" and header["producer"]["route"] == "layer"
    assert header["producer"]["tool_version"] == __version__ and "pdfium" in header["producer"]["engines"]
    assert header["record"] is None and header["font_maps"] == "checked" and header["warnings"] == []
    assert header["layers"] == {"pdfintake": 1, "other": 2, "none": 2}
    assert [p["page"] for p in pages] == [1, 2, 3, 4, 5]
    assert [p["layer"] for p in pages] == ["other", "none", "pdfintake", "none", "other"]
    assert pages[0]["text"] == f"{FIRST}\n{SECOND}"            # pdfium's line-end hyphen mark undone
    assert pages[2]["text"] == OURS and pages[4]["text"] == OTHER
    assert pages[1]["text"] == "" and pages[1]["flags"] == ["no_text_layer"]
    assert pages[3]["text"] == "" and pages[3]["flags"] == []  # an empty page has nothing to read
    for p in pages:
        assert p["chars"] == len(p["text"])
        assert set(p) == {"page", "text", "chars", "layer", "flags", "lang", "lang_basis", "pack"}
        assert p["lang"] is None and p["pack"] is None


def test_the_text_is_the_layer_verify_reads(tmp_path: Path):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    count, raw = verify.layer_text(pdf)
    exp = layertext.read(pdf)
    assert count == 5
    assert [p.text for p in exp.lines] == [layertext.clean(raw[i]) for i in range(5)]


def test_txt_ends_every_page_with_a_form_feed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    code, out, _ = call(capsys, "text", str(pdf), "--format", "txt")
    assert code == EXIT_OK
    pieces = out.split("\f")
    assert len(pieces) == 6 and pieces[-1] == ""
    assert pieces[0] == f"{FIRST}\n{SECOND}\n" and pieces[1] == "" and pieces[2] == OURS + "\n" and pieces[3] == ""


def test_pages_limits_the_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    code, out, _ = call(capsys, "text", str(pdf), "--pages", "2-3")
    header, pages = jsonl(out)
    assert code == EXIT_OK and header["pages"] == 5 and header["range"] == [2, 3]
    assert [p["page"] for p in pages] == [2, 3]
    code, out, _ = call(capsys, "text", str(pdf), "--pages", "5")
    assert code == EXIT_OK and [p["page"] for p in jsonl(out)[1]] == [5]


@pytest.mark.parametrize("spec", ["0", "3-2", "two", "1-", "-1", "1,2", "6", "4-6"])
def test_a_bad_page_range_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str], spec: str):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    code, out, _ = call(capsys, "text", str(pdf), "--pages", spec, "--json")
    data = json.loads(out)
    assert code == EXIT_USAGE and data["exit"] == EXIT_USAGE and data["error"]["code"] in ("bad_pages", "bad_arguments")


def test_a_pdf_without_a_layer_gives_none_pages_and_points_at_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = scan_pdf(tmp_path / "Scan.pdf", pages=3)
    code, out, err = call(capsys, "text", str(pdf))
    assert code == EXIT_OK
    header, pages = jsonl(out)
    assert [p["layer"] for p in pages] == ["none"] * 3 and all(p["text"] == "" for p in pages)
    assert all(p["flags"] == ["no_text_layer"] for p in pages)
    assert [w["code"] for w in header["warnings"]] == ["no_text_layer"]
    assert "pdfintake run" in header["warnings"][0]["message"]
    assert "warning no_text_layer" in err and "pdfintake run" in err


def test_a_blank_pdf_has_nothing_for_run_to_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = build.write_pdf(tmp_path / "Blank.pdf", 2)
    code, out, err = call(capsys, "text", str(pdf))
    header, pages = jsonl(out)
    assert code == EXIT_OK and err == "" and header["warnings"] == []
    assert [(p["layer"], p["flags"]) for p in pages] == [("none", []), ("none", [])]


def test_text_never_reads_a_page_with_ocr(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                          monkeypatch: pytest.MonkeyPatch):
    from pdfintake.ocr import pipeline

    def no_ocr(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("text read a page with OCR")

    monkeypatch.setattr(pipeline, "read_pages", no_ocr)
    monkeypatch.setattr(flow, "run", no_ocr)
    for pdf in (mixed_pdf(tmp_path / "Mixed.pdf"), scan_pdf(tmp_path / "Scan.pdf")):
        code, out, _ = call(capsys, "text", str(pdf))
        assert code == EXIT_OK
    assert list((tmp_path / "cache").rglob("r-*.json")) == []           # and no record was written


def test_the_input_is_never_changed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    before = (sha(pdf), pdf.stat().st_mtime_ns)
    call(capsys, "text", str(pdf))
    call(capsys, "text", str(pdf), "-o", str(tmp_path / "out.jsonl"), "--json")
    assert (sha(pdf), pdf.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Mixed.pdf", "cache", "out.jsonl"]


def test_no_work_or_staging_files_are_left(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = _garbled(tmp_path)
    assert call(capsys, "text", str(pdf), "-o", str(tmp_path / "out.jsonl"))[0] == EXIT_OK
    cache = tmp_path / "cache"
    left = [p for kind in ("work", "staging") if (cache / kind).exists() for p in (cache / kind).rglob("*") if p.is_file()]
    assert left == []


# --------------------------------------------------------------------------- clean

@pytest.mark.parametrize("raw,want", [
    ("one\r\ntwo\rthree", "one\ntwo\nthree"),
    ("exam\x02ple", "exam-\nple"),
    ("exam\ufffeple", "exam-\nple"),
    ("exam\x02\r\nple", "exam-\nple"),
    ("cafe\u0301", "caf\u00e9"),                     # NFC
    ("bad \ud800 half", "bad \ufffd half"),          # a lone surrogate can't be written as UTF-8
    (" \r\n\t ", ""),
    ("", ""),
])
def test_clean(raw: str, want: str):
    assert layertext.clean(raw) == want


def test_a_lone_surrogate_reaches_stdout_as_utf8(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    real = verify.layer_text
    monkeypatch.setattr(verify, "layer_text", lambda p, pages=None: (real(p, pages)[0], {i: "x\udc80y" for i in pages}))
    code, out, _ = call(capsys, "text", str(pdf))
    assert code == EXIT_OK and jsonl(out)[1][0]["text"] == "x\ufffdy"


# --------------------------------------------------------------------------- flags

def _garbled(tmp_path: Path) -> Path:
    chars = "abcdefghij"
    prog = G.shape_font(chars, order=chars[::-1], cmap=False, names=False)
    return G.garbled_pdf(tmp_path / "Garbled.pdf", [G.Page(G.lines_of(["badge", "chief", "jade"]))], [G.FontSpec(prog)])


def test_a_page_whose_font_has_no_map_is_flagged_garbled(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = _garbled(tmp_path)
    code, out, _ = call(capsys, "text", str(pdf))
    header, pages = jsonl(out)
    assert code == EXIT_OK and header["font_maps"] == "checked"
    assert pages[0]["layer"] == "other" and "garbled_text" in pages[0]["flags"]


def test_a_font_check_that_cant_run_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                           monkeypatch: pytest.MonkeyPatch):
    from pdfintake.pdf import fontmap

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("a font program pdfintake can't parse")

    monkeypatch.setattr(fontmap, "analyse", broken)
    code, out, _ = call(capsys, "text", str(mixed_pdf(tmp_path / "Mixed.pdf")))
    assert code == EXIT_OK and jsonl(out)[0]["font_maps"] == "failed"


def test_a_cache_that_cant_hold_the_font_check_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                      monkeypatch: pytest.MonkeyPatch):
    from pdfintake import fsutil

    def no_room(*_a: Any, **_k: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fsutil, "work_dir", no_room)
    code, out, _ = call(capsys, "text", str(mixed_pdf(tmp_path / "Mixed.pdf")))
    assert code == EXIT_OK and jsonl(out)[0]["font_maps"] == "failed"


def test_a_born_digital_page_mostly_pictures_is_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = build.write_pdf(tmp_path / "Slide.pdf", [build.PageSpec(images=(build.ImageSpec(_image()),),
                                                                  texts=(build.TextSpec("Title"),))])
    code, out, _ = call(capsys, "text", str(pdf))
    assert code == EXIT_OK and jsonl(out)[1][0]["flags"] == ["image_text_not_read"]


def test_text_flags_are_closed_warning_codes():
    assert set(record.TEXT_FLAGS) <= record.WARNING_CODES
    assert len(set(record.TEXT_FLAGS)) == len(record.TEXT_FLAGS)
    assert "no_text_layer" in record.WARNING_GROUPS["text"]


# --------------------------------------------------------------------------- --record

def saved_record(pdf: Path, *, sha256: str | None = None, fontmap: dict[str, Any] | None = None) -> str:
    """A done record whose output is ``pdf``: page 3 OCR'd with the de pack, warnings on pages 1 and 3."""
    rec = record.new([pdf.with_name("Scan.pdf")], opts=flow.RunOptions(), seal=None, platform_key="darwin-arm64")
    rec.status, rec.exit = "done", 0                   # type: ignore[assignment]
    rec.input["kind"] = "pdf"
    rec.output = record.output_entry(pdf, sha256 or sha(pdf), pdf.stat().st_size)
    rec.language = {"plan": "confident", "packs": [{"id": "de", "version": "1.0.0", "fetched": False}],
                    "canaries": [], "detection": {}}
    rec.pages = [{"page": 3, "class": "image_only", "pack": "de", "warnings": ["layer_dropped_chars", "layer_shrank"],
                  "languages": [{"tag": "de", "prob": 0.97, "by": "langid", "windows": 3},
                                {"tag": "en", "prob": 0.4, "by": "langid", "windows": 1}]},
                 {"page": 5, "class": "image_only", "pack": "en", "warnings": [],
                  "languages": [{"tag": "fr", "prob": 1.0, "by": "forced", "windows": 0}]}]
    rec.warnings = [record.warning("agreement_low", 3), record.warning("existing_layer_replaced", 5),
                    record.warning("font_map_partial", 1)]
    rec.steps = [{"step": "fontmap", "status": "changed",
                  "detail": fontmap if fontmap is not None else {"pages": [], "partial": [1], "unrepaired": []}}]
    rec.notes.append(record.note("input_unchanged"))
    record.save(rec)
    return rec.id


def test_a_matching_record_adds_languages_packs_and_its_page_warnings(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    rid = saved_record(pdf)
    code, out, _ = call(capsys, "text", str(pdf), "--record", rid)
    assert code == EXIT_OK
    header, pages = jsonl(out)
    assert header["record"]["id"] == rid and header["font_maps"] == "record"
    assert header["record"]["language"] == {"plan": "confident", "packs": [{"id": "de", "version": "1.0.0"}]}
    assert header["record"]["tool"]["version"] == __version__
    assert (pages[2]["lang"], pages[2]["lang_basis"], pages[2]["pack"]) == (["de", "en"], "detected", "de")
    assert pages[2]["flags"] == ["layer_dropped_chars", "agreement_low"]     # layer_shrank isn't about the text
    assert (pages[4]["lang"], pages[4]["lang_basis"], pages[4]["pack"]) == (["fr"], "forced", "en")
    assert pages[4]["flags"] == []                                           # existing_layer_replaced isn't either
    assert pages[0]["flags"] == ["font_map_partial"]                          # from the fontmap step and its warning
    assert pages[0]["lang"] is None and pages[1]["flags"] == ["no_text_layer"]


def test_a_record_for_another_file_is_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    rid = saved_record(pdf, sha256="0" * 64)
    code, out, _ = call(capsys, "text", str(pdf), "--record", rid, "--json")
    data = json.loads(out)
    assert code == EXIT_STALE and data["error"]["code"] == "record_mismatch"


@pytest.mark.parametrize("rid,want", [("r-20261117-00000000", "unknown_record"), ("../x", "bad_record_id")])
def test_an_unknown_or_bad_record_id_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str], rid: str, want: str):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    code, out, _ = call(capsys, "text", str(pdf), "--record", rid, "--json")
    assert code == EXIT_USAGE and json.loads(out)["error"]["code"] == want


def test_a_record_whose_font_check_failed_leaves_it_to_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = _garbled(tmp_path)
    rid = saved_record(pdf)
    rec = record.load(rid)
    rec.steps = [{"step": "fontmap", "status": "failed", "detail": {"error": "RuntimeError"}}]
    rec.warnings = []
    record.save(rec)
    code, out, _ = call(capsys, "text", str(pdf), "--record", rid)
    header, pages = jsonl(out)
    assert code == EXIT_OK and header["font_maps"] == "checked" and "garbled_text" in pages[0]["flags"]


# --------------------------------------------------------------------------- inputs that aren't text PDFs

def test_an_image_is_refused_naming_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    png = build.write_png(_image(), tmp_path / "Photo.png")
    code, out, _ = call(capsys, "text", str(png), "--json")
    data = json.loads(out)
    assert code == EXIT_USAGE and data["error"]["code"] == "unsupported_input"
    assert "pdfintake run" in data["error"]["message"]


@pytest.mark.parametrize("kind,want", [("missing", "input_missing"), ("folder", "input_not_file"),
                                       ("words", "unsupported_input")])
def test_inputs_that_cant_be_read_are_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str, want: str):
    target = {"missing": tmp_path / "missing.pdf", "folder": tmp_path, "words": tmp_path / "notes.txt"}[kind]
    if kind == "words":
        target.write_text("plain made-up words", encoding="utf-8")
    code, out, _ = call(capsys, "text", str(target), "--json")
    assert code == EXIT_USAGE and json.loads(out)["error"]["code"] == want


def test_a_pdf_that_needs_a_password_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    plain = mixed_pdf(tmp_path / "plain.pdf")
    locked = tmp_path / "Locked.pdf"
    with pikepdf.open(plain) as pdf:
        pdf.save(locked, encryption=pikepdf.Encryption(user="made up open phrase", owner="made up owner phrase"))
    code, out, _ = call(capsys, "text", str(locked), "--json")
    data = json.loads(out)
    assert code == EXIT_REFUSED and data["error"]["code"] == "encrypted" and data["refusal"]["code"] == "encrypted"
    assert "made up" not in out


def test_an_edit_lock_alone_is_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    plain = mixed_pdf(tmp_path / "plain.pdf")
    locked = tmp_path / "EditLocked.pdf"
    with pikepdf.open(plain) as pdf:
        pdf.save(locked, encryption=pikepdf.Encryption(user="", owner="made up owner phrase",
                                                        allow=pikepdf.Permissions(modify_other=False)))
    code, out, _ = call(capsys, "text", str(locked))
    assert code == EXIT_OK and jsonl(out)[1][0]["text"] == f"{FIRST}\n{SECOND}"


def test_a_file_pdfium_cant_open_is_damaged(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bad = tmp_path / "Broken.pdf"
    bad.write_bytes(b"%PDF-1.7\n" + b"\x00" * 200)
    code, out, _ = call(capsys, "text", str(bad), "--json")
    assert code == EXIT_REFUSED and json.loads(out)["error"]["code"] == "damaged"


def test_a_file_that_changes_while_it_is_read_is_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                       monkeypatch: pytest.MonkeyPatch):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    real = verify.layer_text

    def edited(path: Path, pages: Any = None) -> Any:
        got = real(path, pages)
        with open(pdf, "ab") as fh:
            fh.write(b"\n% appended\n")
        return got

    monkeypatch.setattr(verify, "layer_text", edited)
    code, out, _ = call(capsys, "text", str(pdf), "--json")
    assert code == EXIT_STALE and json.loads(out)["error"]["code"] == "input_changed"


# --------------------------------------------------------------------------- -o FILE

def test_out_file_holds_the_export_and_the_summary_holds_no_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    dest = tmp_path / "Mixed.jsonl"
    code, out, err = call(capsys, "text", str(pdf), "-o", str(dest), "--json")
    data = json.loads(out)
    assert code == EXIT_OK and err == ""
    assert data["output"] == {"path": str(dest), "sha256": sha(dest), "bytes": dest.stat().st_size}
    assert data["input"]["sha256"] == sha(pdf) and data["layers"] == {"pdfintake": 1, "other": 2, "none": 2}
    assert data["flags"] == {"no_text_layer": [2]} and data["font_maps"] == "checked"
    for word in SECRET_WORDS:
        assert word not in out
    header, pages = jsonl(dest.read_text(encoding="utf-8"))
    assert header["source_sha256"] == sha(pdf) and pages[2]["text"] == OURS


def test_out_dash_is_standard_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    code, out, _ = call(capsys, "text", str(pdf), "-o", "-", "--format", "txt")
    assert code == EXIT_OK and out.startswith(FIRST)
    assert not (tmp_path / "-").exists()


def test_out_file_without_overwrite_is_exit_2_and_with_it_is_replaced(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    dest = tmp_path / "Mixed.txt"
    dest.write_text("older", encoding="utf-8")
    code, out, _ = call(capsys, "text", str(pdf), "-o", str(dest), "--format", "txt", "--json")
    assert code == EXIT_USAGE and json.loads(out)["error"]["code"] == "output_exists"
    assert dest.read_text(encoding="utf-8") == "older"
    code, out, _ = call(capsys, "text", str(pdf), "-o", str(dest), "--format", "txt", "--overwrite")
    assert code == EXIT_OK and dest.read_text(encoding="utf-8").startswith(FIRST)
    assert "pages 1-5: 1 pdfintake, 2 other, 2 none" in out


def test_the_output_is_never_the_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    before = sha(pdf)
    code, out, _ = call(capsys, "text", str(pdf), "-o", str(pdf), "--overwrite", "--json")
    assert code == EXIT_USAGE and json.loads(out)["error"]["code"] == "output_is_input"
    assert sha(pdf) == before


def test_out_file_goes_through_flow_commit(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                           monkeypatch: pytest.MonkeyPatch):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")
    seen: list[Path] = []
    real = flow.commit

    def spy(built: Path, dest: Path, *a: Any, **k: Any) -> Any:
        seen.append(Path(dest))
        assert Path(built).is_relative_to(tmp_path / "cache" / "staging")
        return real(built, dest, *a, **k)

    monkeypatch.setattr(flow, "commit", spy)
    assert call(capsys, "text", str(pdf), "-o", str(tmp_path / "out.jsonl"))[0] == EXIT_OK
    assert seen == [tmp_path / "out.jsonl"]


# --------------------------------------------------------------------------- nothing leaks

def test_an_unexpected_error_never_echoes_page_text(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                   monkeypatch: pytest.MonkeyPatch):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError(f"failed near {OURS}")

    monkeypatch.setattr(classify, "classify", boom)
    for extra in ((), ("--json",)):
        code, out, err = call(capsys, "text", str(pdf), *extra)
        assert code == EXIT_FAILED
        for word in SECRET_WORDS:
            assert word not in out and word not in err


def test_a_failure_prints_no_text_before_the_error(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                   monkeypatch: pytest.MonkeyPatch):
    pdf = mixed_pdf(tmp_path / "Mixed.pdf")

    def stale(self: Any) -> None:
        raise InputChanged("input_changed", "The input changed; run again.")

    from pdfintake import integrity

    monkeypatch.setattr(integrity.InputSeal, "check", stale)
    code, out, _ = call(capsys, "text", str(pdf), "--json")
    assert code == EXIT_STALE and list(json.loads(out)) == ["exit", "error"]
    assert FIRST not in out


def test_help_and_parse_pages():
    assert layertext.parse_pages(None) is None
    assert layertext.parse_pages("4") == (4, 4) and layertext.parse_pages(" 2 - 9 ") == (2, 9)
    with pytest.raises(UsageError) as info:
        layertext.parse_pages("9-2")
    assert info.value.code == "bad_pages"
    assert cli.main(["text", "--help"]) == EXIT_OK
