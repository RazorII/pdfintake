"""Refusals (U3): both detectors, every code, the damaged-file variants (DESIGN §2.2, §7.1).

Since §10.1 and §11.1 (decided 2026-11-05) a code either refuses (``HARD``) or puts the file in
copy mode (``COPYABLE``); the "§10.1 rows" section tests every row of that table.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import io
import json
import sys
import threading
import time
import zlib
from pathlib import Path

import pytest

from fixtures import build
from fixtures import refusal_shells as rs
from pdfintake import refusals
from pdfintake.errors import EXIT_USAGE, UsageError
from pdfintake.types import to_jsonable

SRC = Path(__file__).resolve().parents[1] / "src" / "pdfintake"


@functools.lru_cache(maxsize=None)
def _bytes(name: str) -> bytes:
    if name in rs.SHELLS:
        return rs.SHELLS[name].build()
    return {
        "plain": rs.base_pdf,
        "orphan": rs.orphan_byterange,
        "removed_later": rs.signature_removed_later,
        "widget_not_in_fields": rs.signed_widget_not_in_fields,
        "ur3_objstm": lambda: rs.ur3(object_streams=True),
        "xfa_objstm": lambda: rs.xfa(object_streams=True),
        "signed_objstm": lambda: rs.signed_visible(object_streams=True),
        "certified_objstm": lambda: rs.certified(1, object_streams=True),
        "pdfua_flate": lambda: rs.pdfua(compress=True),
        "jbig2": rs.jbig2_image,
        "inherited_ft": rs.inherited_sig_field,
        "direct_sig": rs.direct_sig_value,
    }[name]()


def _needs_pyhanko(name: str) -> None:
    if (name in rs.SHELLS and rs.SHELLS[name].pyhanko) or name in (
            "removed_later", "widget_not_in_fields", "signed_objstm", "certified_objstm"):
        pytest.importorskip("pyhanko")


def _write(tmp_path: Path, name: str, damage: str = "intact") -> Path:
    _needs_pyhanko(name)
    path = tmp_path / f"{name}-{damage}.pdf"
    path.write_bytes(rs.DAMAGES[damage](_bytes(name)))
    return path


def _first_code(report: refusals.RefusalReport, *, adds_layer: bool = True) -> str | None:
    """The most important code the file carries, hard or copyable (None: an ordinary file)."""
    got = refusals._codes(report, adds_layer=adds_layer)
    return got[0] if got else None


def _hard(report: refusals.RefusalReport, *, adds_layer: bool = True) -> str | None:
    got = refusals.refusal(report, adds_layer=adds_layer)
    return got[0] if got else None


def _split(codes: list[str]) -> tuple[str | None, list[str]]:
    """(the hard code refusal() must name, the list copy_reasons() must give) for a list from _codes."""
    hard = [c for c in codes if c in refusals.HARD]
    return (hard[0] if hard else None), [c for c in refusals.CODES if c in codes and c in refusals.COPYABLE]


# --------------------------------------------------------------------------- every fixture, every damage


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
@pytest.mark.parametrize("name", list(rs.SHELLS))
def test_every_shell_gives_its_code(tmp_path: Path, name: str, damage: str):
    shell = rs.SHELLS[name]
    report = refusals.read(_write(tmp_path, name, damage))
    no_trailer = refusals.NO_TRAILER in report.raw_markers      # a one-revision file cut before its trailer
    assert no_trailer is (damage == "no_trailer" and not shell.pyhanko)
    for adds_layer, expected in ((True, shell.code), (False, shell.code_no_layer)):
        got = refusals._codes(report, adds_layer=adds_layer)
        assert (got[0] if got else None) == (expected or ("damaged" if no_trailer else None)), report
        if no_trailer:
            # A file cut before its trailer is damaged, not a disagreement (decided 2026-11-06), and
            # copy mode still finds the hard code.
            assert "damaged" in got and "detectors_disagree" not in got, got
        # refusal() names hard codes only; copy_reasons() the copyable ones, in CODES order.
        hard, copyable = _split(got)
        assert _hard(report, adds_layer=adds_layer) == hard
        assert refusals.copy_reasons(report, "ordinary", adds_layer=adds_layer) == copyable
        if expected is not None:
            assert expected in (refusals.HARD if hard == expected else copyable)
        if damage == "intact":
            assert hard == (expected if expected in refusals.HARD else None)


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
@pytest.mark.parametrize("name, code", [("signed_objstm", "signed"), ("certified_objstm", "certified"),
                                        ("ur3_objstm", "usage_rights"), ("widget_not_in_fields", "signed")])
def test_xref_stream_files_and_orphan_widgets_give_their_code(tmp_path: Path, name: str, code: str, damage: str):
    report = refusals.read(_write(tmp_path, name, damage))
    assert _first_code(report) == code, report


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
def test_damage_is_repaired_and_reported(tmp_path: Path, damage: str):
    report = refusals.read(_write(tmp_path, "plain", damage))
    # A broken startxref is repaired and passes; with no trailer left at all, the file is damaged.
    assert _first_code(report) == ("damaged" if damage == "no_trailer" else None)
    assert report.repaired is (damage != "intact")
    if damage == "startxref":
        assert report.qpdf_warnings > 0


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
def test_an_empty_signature_field_gives_no_refusal(tmp_path: Path, damage: str):
    report = refusals.read(_write(tmp_path, "empty_field", damage))
    assert refusals.refusal(report, adds_layer=True) is None
    assert report.signed_fields == [] and report.sig_dicts == 0
    if damage != "no_trailer":            # without its trailer qpdf reads the revision before the field
        assert report.empty_sig_fields == 1


# --------------------------------------------------------------------------- what each detector found


def test_both_detectors_see_a_signature(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "signed_visible"))
    assert report.walk_codes == ["signed"]
    assert report.raw_markers == ["/ByteRange ["]
    assert report.signed_fields == ["field 1, page 1"]
    assert report.sig_dicts == 1 and report.incremental_updates == 1
    assert report.repaired is False and report.qpdf_warnings == 0


@pytest.mark.parametrize("p", [1, 2])
def test_certification_reports_its_p_value(tmp_path: Path, p: int):
    report = refusals.read(_write(tmp_path, f"certified_p{p}"))
    assert report.certified_p == p
    assert "certified" in report.walk_codes and "/DocMDP" in report.raw_markers


def test_a_signed_field_without_a_widget_is_caught(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "signed_no_widget"))
    assert report.signed_fields == ["field 1, no widget"]
    assert _first_code(report) == "signed"


def test_a_signed_widget_missing_from_the_form_is_caught(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "widget_not_in_fields"))
    assert report.signed_fields == ["field 1, page 1"]


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
def test_a_kid_inherits_the_signature_field_type(tmp_path: Path, damage: str):
    report = refusals.read(_write(tmp_path, "inherited_ft", damage))
    assert report.signed_fields == ["field 1, page 1"] and report.sig_dicts == 1
    assert _first_code(report) == "signed"


def test_a_direct_signature_dictionary_is_counted_once(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "direct_sig"))
    assert report.signed_fields == ["field 1, no widget"] and report.sig_dicts == 1
    assert _first_code(report) == "signed"


def test_a_document_timestamp_is_not_a_signed_field(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "doc_timestamp"))
    assert report.doc_timestamps == 1 and report.signed_fields == []
    assert report.raw_markers == ["/ByteRange [", "/DocTimeStamp"]


def test_usage_rights_are_not_a_signed_field(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "ur3"))
    assert report.usage_rights is True and report.signed_fields == [] and report.sig_dicts == 1
    assert report.raw_markers == ["/ByteRange [", "/UR3"]


@pytest.mark.parametrize("damage", ["intact", "startxref"])
def test_the_raw_scan_reads_inside_object_streams(tmp_path: Path, damage: str):
    assert b"/XFA" not in _bytes("xfa_objstm"), "the fixture should hide /XFA inside a compressed object stream"
    report = refusals.read(_write(tmp_path, "xfa_objstm", damage))
    assert report.xfa is True and report.raw_markers == ["/XFA"]
    assert refusals._codes(report, adds_layer=True) == ["xfa"]


def test_a_compressed_object_stream_alone_still_refuses(tmp_path: Path):
    """Without its xref stream qpdf can't reach the compressed catalog; the raw scan's /XFA stands."""
    report = refusals.read(_write(tmp_path, "xfa_objstm", "no_trailer"))
    assert report.walk_codes == ["open_failed"] and report.raw_markers == ["/XFA", "no_trailer"]
    assert refusals._codes(report, adds_layer=True) == ["xfa", "damaged"]


def test_the_raw_scan_reads_compressed_xmp(tmp_path: Path):
    assert b"pdfuaid" not in _bytes("pdfua_flate")
    report = refusals.read(_write(tmp_path, "pdfua_flate"))
    assert report.pdfua == "1" and report.raw_markers == ["pdfuaid:part"]
    assert refusals.refusal(report, adds_layer=False) is None


def test_encryption_kinds(tmp_path: Path):
    owner = refusals.read(_write(tmp_path, "encrypted_owner"))
    user = refusals.read(_write(tmp_path, "encrypted_user"))
    assert owner.encrypted == "owner" and user.encrypted == "user"
    assert "/Encrypt" in owner.raw_markers and "/Encrypt" in user.raw_markers
    assert "password to open" in refusals.copy_message("encrypted", user)
    assert "owner password" in refusals.copy_message("encrypted", owner)
    for report in (owner, user):
        assert refusals.refusal(report, adds_layer=False) is None
        assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]


def test_encryption_without_a_trailer_is_still_seen_by_the_raw_scan(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "encrypted_user", "no_trailer"))
    assert report.encrypted is False                  # qpdf lost the /Encrypt entry with the trailer
    assert "/Encrypt" in report.raw_markers and report.repaired
    # The raw scan's /Encrypt is read as encryption, and the lost trailer is damage, not a disagreement.
    assert refusals._codes(report, adds_layer=True) == ["encrypted", "damaged"]


@pytest.mark.parametrize("name", ["encrypted_owner", "encrypted_user"])
def test_a_truncated_encrypted_file_is_never_accepted(tmp_path: Path, name: str):
    """Cut anywhere, even before the /Encrypt dictionary: qpdf rebuilds ciphertext as plain, so refuse."""
    data = _bytes(name)
    path = tmp_path / "cut.pdf"
    for cut in range(200, len(data), 40):
        path.write_bytes(data[:cut])
        report = refusals.read(path)
        # Never an ordinary output: refused, or at least a copy beside the untouched original.
        assert (refusals.refusal(report, adds_layer=True) is not None
                or "encrypted" in refusals.copy_reasons(report, "ordinary", adds_layer=True)), (cut, report)
        if not report.encrypted:                       # the walk lost the encryption: the hard code refuses
            assert _hard(report) in ("damaged", "detectors_disagree"), (cut, report)
            if refusals.NO_TRAILER in report.raw_markers or "open_failed" in report.walk_codes:
                assert _hard(report) == "damaged", (cut, report)


def test_an_owner_encrypted_file_still_compares_the_detectors(tmp_path: Path):
    path = tmp_path / "owner-orphan.pdf"
    path.write_bytes(rs.owner_encrypted_plus_orphan())
    report = refusals.read(path)
    assert report.encrypted == "owner" and report.sig_dicts == 0
    assert "/ByteRange [" in report.raw_markers
    assert refusals._codes(report, adds_layer=True) == ["encrypted", "detectors_disagree"]


def test_raw_encryption_bytes_in_a_file_that_opens_plain_disagree(tmp_path: Path):
    """An unreferenced security-handler dictionary: not encrypted (the walk opened it cleanly), but suspicious."""
    path = tmp_path / "unref-encrypt.pdf"
    path.write_bytes(rs.unreferenced_encrypt_dict())
    report = refusals.read(path)
    assert report.encrypted is False and report.repaired is False and "/Encrypt" in report.raw_markers
    assert refusals._codes(report, adds_layer=True) == ["detectors_disagree"]


def test_jbig2_streams_are_counted_not_refused(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "jbig2"))
    assert report.jbig2_streams == 1
    assert refusals.refusal(report, adds_layer=True) is None


def test_a_clean_pdf_reports_nothing(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "plain"))
    assert report == refusals.RefusalReport(
        signed_fields=[], empty_sig_fields=0, sig_dicts=0, doc_timestamps=0, certified_p=None,
        usage_rights=False, pdfa=None, pdfua=None, encrypted=False, xfa=False, jbig2_streams=0,
        incremental_updates=0, xmp_readable=True, walk_codes=[], raw_markers=[], repaired=False,
        qpdf_warnings=0)


# --------------------------------------------------------------------------- disagreement


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
def test_an_unreferenced_byterange_refuses_with_detectors_disagree(tmp_path: Path, damage: str):
    report = refusals.read(_write(tmp_path, "orphan", damage))
    assert "/ByteRange [" in report.raw_markers
    assert report.walk_codes == [] and report.sig_dicts == 0
    assert _first_code(report) == "detectors_disagree"
    assert _first_code(report, adds_layer=False) == "detectors_disagree"


@pytest.mark.parametrize("damage", list(rs.DAMAGES))
def test_a_signature_an_update_removed_still_refuses(tmp_path: Path, damage: str):
    report = refusals.read(_write(tmp_path, "removed_later", damage))
    if damage == "no_trailer":
        # Without the last trailer qpdf silently reads the signed revision; either answer refuses.
        assert _first_code(report) in ("signed", "detectors_disagree")
        assert report.repaired is True                # objects after the last startxref
    else:
        assert report.signed_fields == [] and report.incremental_updates == 2
        assert _first_code(report) == "detectors_disagree"


@pytest.mark.parametrize("kind", ["name_escaped", "comment", "indirect", "fake_stream_in_string",
                                  "paren_in_comment", "unclosed_string", "broken_endstream"])
def test_the_raw_scan_finds_a_disguised_unreferenced_signature(tmp_path: Path, kind: str):
    path = tmp_path / f"{kind}.pdf"
    path.write_bytes(rs.orphan_variant(kind))
    report = refusals.read(path)
    assert report.raw_markers == ["/ByteRange ["] and report.sig_dicts == 0, report
    assert _first_code(report) == "detectors_disagree"


@pytest.mark.parametrize("encoding, keyword", [("flate", b"stream\n"), ("predictor", b"stream\n"),
                                               ("ahx_flate", b"stream\n"), ("a85", b"stream\n"),
                                               ("none", b"stream\n"), ("flate", b"stream \r\n")])
def test_the_raw_scan_decodes_object_streams_through_their_filters(tmp_path: Path, encoding: str, keyword: bytes):
    data = rs.orphan_in_objstm(encoding, keyword=keyword)
    if encoding != "none":
        assert b"/ByteRange" not in data
    path = tmp_path / "objstm.pdf"
    path.write_bytes(data)
    report = refusals.read(path)
    assert report.raw_markers == ["/ByteRange ["], report
    assert _first_code(report) == "detectors_disagree"


def test_an_object_stream_the_scan_cannot_decode_fails_closed(tmp_path: Path):
    path = tmp_path / "lzw.pdf"
    path.write_bytes(rs.orphan_in_objstm("lzw"))
    report = refusals.read(path)
    assert report.raw_markers == ["unscanned"] and _first_code(report) == "detectors_disagree"
    # Decoded with the predictor left out, the stream is garbage: its header doesn't check out.
    path.write_bytes(rs.orphan_in_objstm("predictor").replace(b" /DecodeParms << /Predictor 12 /Columns 8 >>", b""))
    report = refusals.read(path)
    assert report.raw_markers == ["unscanned"] and _first_code(report) == "detectors_disagree"


def test_an_object_stream_past_the_inflate_cap_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "padded.pdf"
    path.write_bytes(rs.orphan_in_objstm(pad=200_000))
    assert refusals.read(path).raw_markers == ["/ByteRange ["]
    monkeypatch.setattr(refusals, "MAX_INFLATE_STREAM", 4096)
    report = refusals.read(path)
    assert report.raw_markers == ["unscanned"] and _first_code(report) == "detectors_disagree"


def test_a_broken_endstream_does_not_hide_a_removed_signature(tmp_path: Path):
    pytest.importorskip("pyhanko")
    path = tmp_path / "removed-broken.pdf"
    path.write_bytes(rs.break_endstream_before(_bytes("removed_later"), b"/ByteRange"))
    report = refusals.read(path)
    assert "/ByteRange [" in report.raw_markers and report.signed_fields == []
    assert _first_code(report) == "detectors_disagree"


@pytest.mark.parametrize("title", ["Chapter 12: the /ByteRange [ array", "The /Encrypt << dictionary",
                                   "Converting /XFA forms", "About pdfaid:part", "A (nested) /DocMDP note",
                                   "An escaped \\) /UR3 paren"])
def test_marker_text_in_a_string_is_not_a_refusal(tmp_path: Path, title: str):
    path = tmp_path / "title.pdf"
    path.write_bytes(rs.with_title(title))
    report = refusals.read(path)
    assert report.raw_markers == [] and _first_code(report) is None, report


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("bpp", [1, 3])
def test_png_predictors_decode(kind: int, bpp: int):
    payload = bytes(range(256)) * 3 + b"example object stream text"
    columns = 12 * bpp
    encoded = rs.png_predict(payload, columns=columns, kind=kind, bpp=bpp)
    parms = f"<< /Predictor 12 /Colors {bpp} /Columns {12} >>".encode()
    decoded = refusals._predict(encoded, parms)
    assert decoded is not None and decoded[:len(payload)] == payload


def test_tiff_predictor_decodes():
    import numpy as np

    rows = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    diff = rows.copy()
    diff[:, 1:, :] = (rows[:, 1:, :].astype(int) - rows[:, :-1, :].astype(int)) % 256
    got = refusals._predict(diff.tobytes(), b"<< /Predictor 2 /Colors 3 /Columns 4 >>")
    assert got == rows.tobytes()


def test_a_digit_run_after_startxref_does_not_hang(tmp_path: Path):
    path = tmp_path / "digits.pdf"
    path.write_bytes(rs.base_pdf() + b"\n" + b"7" * 1_000_000 + b"\n")
    start = time.monotonic()
    refusals.read(path)
    assert time.monotonic() - start < 5


def test_a_file_truncated_while_it_is_read_returns_normally(tmp_path: Path):
    path = tmp_path / "big.pdf"
    path.write_bytes(rs.big_pdf(48))

    def truncate() -> None:
        time.sleep(0.02)
        with open(path, "r+b") as fh:
            fh.truncate(4096)

    worker = threading.Thread(target=truncate)
    worker.start()
    try:
        report = refusals.read(path)
    finally:
        worker.join()
    assert isinstance(report, refusals.RefusalReport)
    tree = ast.parse((SRC / "refusals.py").read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "mmap" not in imported                       # a mapped file that shrinks raises SIGBUS


def test_a_file_too_large_to_scan_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = _write(tmp_path, "plain")
    monkeypatch.setattr(refusals, "MAX_RAW_BYTES", path.stat().st_size - 1)
    report = refusals.read(path)
    assert report.raw_markers == ["unscanned"] and _first_code(report) == "detectors_disagree"


def test_a_walk_that_fails_part_way_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise RuntimeError("planted failure")

    monkeypatch.setattr(refusals, "_walk_open", boom)
    report = refusals.read(_write(tmp_path, "plain"))
    assert report.walk_codes == ["walk_failed"]
    assert _first_code(report) == "detectors_disagree"      # an intact file: not damage, a detector stopped


def test_a_walk_that_fails_on_a_repaired_file_is_damaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise RuntimeError("planted failure")

    monkeypatch.setattr(refusals, "_walk_open", boom)
    report = refusals.read(_write(tmp_path, "plain", "startxref"))
    assert report.walk_codes == ["walk_failed"] and report.repaired
    assert refusals._codes(report, adds_layer=True) == ["damaged"]
    assert _hard(report) == "damaged"
    # The message covers this case too: the file opened and wasn't cut short (decided 2026-11-06).
    message = refusals.refusal(report)[1]
    assert "couldn't be read through after a repair" in message


def test_a_file_cut_before_its_trailer_is_damaged_even_when_qpdf_reads_all_of_it(tmp_path: Path):
    # qpdf rebuilds a file cut before its trailer blind, and the walk reads everything it finds;
    # with no trailer and no xref stream left, the file is still damaged (decided 2026-11-06).
    good = rs.base_pdf()
    report = refusals.read(_file(tmp_path, "cut", good[: len(good) // 2]))
    assert report.walk_codes == [] and report.raw_markers == [refusals.NO_TRAILER]
    assert _hard(report) == "damaged"


# --------------------------------------------------------------------------- damaged files (decided 2026-11-06)

def _damaged_files(tmp_path: Path) -> dict[str, Path]:
    """The kinds of damage a batch meets (Wave C review C20-A6), built in code."""
    good = rs.base_pdf()
    data = {
        "truncated": good[: len(good) // 2],
        "garbage_body": b"%PDF-1.7\n" + bytes((i * 37 + 11) % 256 for i in range(4000)),
        "header_only": b"%PDF-1.7\n",
        "zero_bytes_after_header": b"%PDF-1.4\n" + b"\x00" * 10000 + b"\n%%EOF\n",
        "empty_pages": (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"),
    }
    return {name: _file(tmp_path, name, body) for name, body in data.items()}


def test_a_damaged_file_is_refused_as_damaged_not_as_a_disagreement(tmp_path: Path):
    for name, path in _damaged_files(tmp_path).items():
        report = refusals.read(path)
        got = refusals.refusal(report, adds_layer=True)
        assert got is not None, name
        code, message = got
        assert code == "damaged", (name, report)
        assert "detectors_disagree" not in refusals._codes(report, adds_layer=True), name
        assert message == refusals._MESSAGES["damaged"]
        assert "damaged" in message and "makes no copy" in message and "{" not in message
        assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == []


def test_a_file_qpdf_repairs_and_reads_in_full_is_not_damaged(tmp_path: Path):
    # No xref table at all: qpdf rebuilds it, the walk reads everything, and the file is processed
    # with the warning input_repaired rather than refused.
    body = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 100 100]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")
    report = refusals.read(_file(tmp_path, "no-xref", body))
    assert report.repaired and report.walk_codes == [] and report.raw_markers == []
    assert refusals._codes(report, adds_layer=True) == []


def _report(**changes) -> refusals.RefusalReport:
    base = dict(signed_fields=[], empty_sig_fields=0, sig_dicts=0, doc_timestamps=0, certified_p=None,
                usage_rights=False, pdfa=None, pdfua=None, encrypted=False, xfa=False, jbig2_streams=0,
                incremental_updates=0, xmp_readable=True, walk_codes=[], raw_markers=[], repaired=False,
                qpdf_warnings=0)
    base.update(changes)
    return refusals.RefusalReport(**base)


def test_a_damaged_file_the_detectors_also_disagree_about_names_the_damage_first():
    # A one-revision file cut before its trailer, with a signature dictionary the walk can't reach.
    report = _report(raw_markers=["/ByteRange [", "no_trailer"], repaired=True, qpdf_warnings=3)
    assert refusals._codes(report, adds_layer=True) == ["damaged", "detectors_disagree"]
    assert _hard(report) == "damaged"


@pytest.mark.parametrize("changes, expected", [
    ({"walk_codes": ["open_failed"]}, ["damaged"]),
    ({"raw_markers": ["no_trailer"], "repaired": True}, ["damaged"]),
    ({"walk_codes": ["walk_failed"], "repaired": True}, ["damaged"]),
    ({"walk_codes": ["walk_failed"]}, ["detectors_disagree"]),               # e.g. the object cap
    ({"raw_markers": ["unscanned"]}, ["detectors_disagree"]),                # fails closed, not damage
    ({"repaired": True, "qpdf_warnings": 2}, []),                             # repaired and read in full
])
def test_what_counts_as_damaged(changes: dict, expected: list[str]):
    assert refusals._codes(_report(**changes), adds_layer=True) == expected


def test_markers_in_page_content_are_not_refusals(tmp_path: Path):
    """The raw scan skips stream data other than XMP and object streams, so printed text can't trip it."""
    path = tmp_path / "words.pdf"
    data = build.pdf_bytes([build.PageSpec(texts=(build.TextSpec("Example /ByteRange [ /DocMDP /UR3 /XFA pdfaid:part"),))])
    import pikepdf

    path.write_bytes(rs._with(data, lambda pdf: None, compress_streams=False,
                              stream_decode_level=pikepdf.StreamDecodeLevel.generalized))
    assert b"/ByteRange [" in path.read_bytes()       # uncompressed in the content stream
    report = refusals.read(path)
    assert report.raw_markers == [] and _first_code(report) is None


# --------------------------------------------------------------------------- PDF/UA, PDF/A and XMP


def test_pdfua_is_a_copy_reason_only_when_a_layer_is_added(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "pdfua"))
    assert report.pdfua == "1" and "pdfua" in report.walk_codes
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["pdfua_needs_layer"]
    with pytest.raises(TypeError):                    # adds_layer is required: no default decides the mode
        refusals.copy_reasons(report, "ordinary")  # type: ignore[call-arg]
    assert inspect.signature(refusals.copy_reasons).parameters["adds_layer"].default is inspect.Parameter.empty
    assert refusals.refusal(report) is None           # refusal() needs no layer answer: no hard code uses it
    assert "PDF/UA-1" in refusals.copy_message("pdfua_needs_layer", report)
    assert refusals.copy_reasons(report, "ordinary", adds_layer=False) == []
    for adds_layer in (True, False):
        assert refusals.refusal(report, adds_layer=adds_layer) is None


def _xmp_pdf(tmp_path: Path, name: str, xmp: str | bytes, *, compress: bool = False) -> Path:
    import pikepdf

    path = tmp_path / f"{name}.pdf"
    path.write_bytes(rs.with_xmp(xmp, compress=compress))
    with pikepdf.open(path) as check:                 # the XMP must arrive byte for byte
        raw = xmp.encode("utf-8") if isinstance(xmp, str) else xmp
        assert check.Root.Metadata.read_bytes() == raw
    return path


def test_pdfa_from_element_and_attribute_forms(tmp_path: Path):
    a = refusals.read(_xmp_pdf(tmp_path, "a2b", rs.XMP_PDFA_2B))
    b = refusals.read(_xmp_pdf(tmp_path, "a3u", rs.XMP_PDFA_ATTR))
    assert (a.pdfa, b.pdfa) == ("2B", "3U")
    assert refusals.refusal(a, adds_layer=False) is None
    for report in (a, b):
        assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["pdfa"]
    assert "PDF/A-2B" in refusals.copy_message("pdfa", a)


@pytest.mark.parametrize("xmp", [rs.XMP_PDFA_OTHER_PREFIX, rs.XMP_PDFA_OTHER_PREFIX_ELEMENT],
                         ids=["attribute", "element"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-be"])
def test_pdfa_under_another_prefix_is_still_pdfa(tmp_path: Path, xmp: str, encoding: str):
    """§10.1: a valid PDF/A file gets a copy whatever prefix its XMP binds the namespace to."""
    data = {"utf-8": xmp, "utf-16": xmp.encode("utf-16"), "utf-16-be": ("﻿" + xmp).encode("utf-16-be")}[encoding]
    report = refusals.read(_xmp_pdf(tmp_path, "prefix", data))
    assert report.pdfa == "1B" and report.raw_markers == ["pdfaid:part"]   # the raw scan follows the binding
    assert refusals._codes(report, adds_layer=True) == ["pdfa"]
    assert refusals.refusal(report) is None
    assert refusals.copy_reasons(report, "ordinary", adds_layer=False) == ["pdfa"]


def test_pdfua_under_another_prefix_is_still_pdfua(tmp_path: Path):
    report = refusals.read(_xmp_pdf(tmp_path, "ua-prefix", rs.XMP_PDFUA_OTHER_PREFIX))
    assert report.pdfua == "1" and report.raw_markers == ["pdfuaid:part"]
    assert refusals.refusal(report) is None
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["pdfua_needs_layer"]
    assert refusals.copy_reasons(report, "ordinary", adds_layer=False) == []


def test_a_prefix_bound_elsewhere_is_not_a_claim(tmp_path: Path):
    """The raw scan follows only a binding to the id namespace: ``x:part`` under another URI says nothing."""
    report = refusals.read(_xmp_pdf(tmp_path, "unrelated", rs.XMP_UNRELATED_PART))
    assert report.pdfa is None and report.raw_markers == [] and refusals._codes(report, adds_layer=True) == []



def test_utf16_xmp_is_read_by_both_detectors(tmp_path: Path):
    report = refusals.read(_xmp_pdf(tmp_path, "utf16", rs.XMP_PDFA_ATTR.encode("utf-16")))
    assert report.pdfa == "3U" and report.raw_markers == ["pdfaid:part"]
    assert refusals._codes(report, adds_layer=True) == ["pdfa"]


@pytest.mark.parametrize("xmp, readable", [
    (rs.XMP_PLAIN, True), ("", True), ("   \n\x00", True),
    (rs.XMP_BROKEN, False), (rs.XMP_DOCTYPE, False), ("not xml at all", False),
])
def test_xmp_readability(tmp_path: Path, xmp: str, readable: bool):
    report = refusals.read(_xmp_pdf(tmp_path, "x", xmp))
    assert report.xmp_readable is readable
    assert _first_code(report) == (None if readable else "xmp_unreadable")


def test_a_metadata_entry_that_is_not_a_stream_is_unreadable(tmp_path: Path):
    import pikepdf

    path = tmp_path / "meta-dict.pdf"
    path.write_bytes(rs._with(rs.base_pdf(), lambda pdf: setattr(pdf.Root, "Metadata", pikepdf.Dictionary(A=1))))
    assert _first_code(refusals.read(path)) == "xmp_unreadable"


@pytest.mark.parametrize("filter_text", ["/FlateDecode", "[/FlateDecode]", "/Fl"])
def test_an_xmp_bomb_is_unreadable_not_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filter_text: str):
    calls: list[int] = []
    capped = refusals._inflate_checked

    def spy(body: bytes, cap: int):
        calls.append(cap)
        return capped(body, cap)

    path = tmp_path / "bomb.pdf"
    path.write_bytes(rs.with_xmp(" " * 100_000 + rs.XMP_PLAIN, compress=True, filter_text=filter_text))
    monkeypatch.setattr(refusals, "MAX_XMP_BYTES", 4096)
    monkeypatch.setattr(refusals, "_inflate_checked", spy)
    report = refusals.read(path)
    assert report.xmp_readable is False and 4096 in calls       # the capped inflate, never read_bytes


@pytest.mark.parametrize("filter_text", ["[/FlateDecode]", "/Fl"])
def test_flate_xmp_in_its_other_spellings_is_read(tmp_path: Path, filter_text: str):
    path = tmp_path / "a2b.pdf"
    path.write_bytes(rs.with_xmp(rs.XMP_PDFA_2B, compress=True, filter_text=filter_text))
    report = refusals.read(path)
    assert report.pdfa == "2B" and report.raw_markers == ["pdfaid:part"]
    assert refusals._codes(report, adds_layer=True) == ["pdfa"]


@pytest.mark.parametrize("filter_text", ["/FlateDecode /DecodeParms << /Predictor 1 >>", "/LZWDecode",
                                         "[/ASCIIHexDecode /FlateDecode]"])
def test_xmp_with_any_other_filter_is_unreadable(tmp_path: Path, filter_text: str):
    path = tmp_path / "other.pdf"
    path.write_bytes(rs.with_xmp(rs.XMP_PLAIN, compress=True, filter_text=filter_text))
    report = refusals.read(path)
    assert report.xmp_readable is False and _first_code(report) == "xmp_unreadable"


@pytest.mark.parametrize("where, compress", [("page", False), ("image", False), ("image", True)])
def test_a_page_or_image_xmp_claim_is_not_the_documents(tmp_path: Path, where: str, compress: bool):
    path = tmp_path / f"{where}.pdf"
    path.write_bytes(rs.object_xmp(where, compress_streams=compress))
    report = refusals.read(path)
    assert report.pdfa is None and report.walk_codes == ["object_pdfa"] and report.raw_markers == ["pdfaid:part"]
    assert refusals._codes(report, adds_layer=True) == []


def test_a_page_pdfua_claim_never_needs_a_layer_refusal(tmp_path: Path):
    path = tmp_path / "page-ua.pdf"
    path.write_bytes(rs.object_xmp("page", rs.XMP_PDFUA_1))
    report = refusals.read(path)
    assert report.walk_codes == ["object_pdfua"] and report.raw_markers == ["pdfuaid:part"]
    assert refusals._codes(report, adds_layer=True) == []


@pytest.mark.parametrize("xmp", [rs.XMP_PDFA_2B, rs.XMP_PDFA_OTHER_PREFIX], ids=["pdfaid", "other-prefix"])
def test_an_unreferenced_xmp_claim_still_disagrees(tmp_path: Path, xmp: str):
    path = tmp_path / "unref-xmp.pdf"
    path.write_bytes(rs.object_xmp("unreferenced", xmp))
    report = refusals.read(path)
    assert report.walk_codes == [] and report.raw_markers == ["pdfaid:part"]
    assert _first_code(report) == "detectors_disagree"


def test_inflate_is_capped():
    bomb = zlib.compress(b"\x00" * 5_000_000, 9)
    assert len(refusals._inflate(bomb, 1000)) == 1000
    assert refusals._inflate(b"not zlib at all", 1000) == b""
    good = zlib.compress(b"pdfaid:part")
    assert refusals._inflate(good + b"trailing junk", 1000) == b"pdfaid:part"


# --------------------------------------------------------------------------- robustness


@pytest.mark.parametrize("content", [b"", b"%PDF-1.7\n", b"not a pdf", b"%PDF-1.4\n1 0 obj << /Type /Catalog"],
                         ids=["empty", "header", "text", "truncated"])
def test_read_never_raises_for_content(tmp_path: Path, content: bytes):
    """A file the walk can't open is refused as damaged (§3.3, §10.1; decided 2026-11-06)."""
    path = tmp_path / "junk.pdf"
    path.write_bytes(content)
    report = refusals.read(path)
    assert report.walk_codes == ["open_failed"]
    assert refusals._codes(report, adds_layer=True) == ["damaged"]


def test_an_unopenable_file_with_signature_bytes_refuses(tmp_path: Path):
    path = tmp_path / "junk.pdf"
    path.write_bytes(b"%PDF-1.7\n5 0 obj << /Type /Sig /ByteRange [0 1 2 3] /Contents <00> >> endobj\n")
    report = refusals.read(path)
    assert report.walk_codes == ["open_failed"] and report.raw_markers == ["/ByteRange [", "no_trailer"]
    assert refusals._codes(report, adds_layer=True) == ["signed", "damaged"]
    assert _hard(report) == "damaged"


def test_read_leaves_the_input_untouched(tmp_path: Path):
    path = _write(tmp_path, "certified_p1")
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    refusals.read(path)
    refusals.output_signature_material(path)
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) == before


def test_a_missing_path_is_not_content():
    with pytest.raises(OSError):
        refusals.read(Path("/nonexistent/example/input.pdf"))


def test_the_report_is_plain_json(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "certified_p2"))
    data = json.loads(json.dumps(to_jsonable(report)))
    assert data["certified_p"] == 2 and data["walk_codes"] == ["certified", "signed"]


# --------------------------------------------------------------------------- codes and messages


def test_codes_are_the_spec_tuple():
    assert refusals.CODES == ("signed", "certified", "usage_rights", "timestamped", "pdfa", "pdfua_needs_layer",
                              "encrypted", "xfa", "xmp_unreadable", "detectors_disagree", "damaged",
                              "provenance_counterparty", "provenance_evidence")
    assert set(refusals._MESSAGES) == set(refusals.CODES)


def test_every_refusal_is_a_known_code_with_a_core_message(tmp_path: Path):
    pytest.importorskip("pyhanko")
    seen: set[str] = set()
    copied: set[str] = set()
    for name in list(rs.SHELLS) + ["orphan", "removed_later"]:
        for damage in rs.DAMAGES:
            report = refusals.read(_write(tmp_path, name, damage))
            for adds_layer in (True, False):
                reasons = refusals.copy_reasons(report, "ordinary", adds_layer=adds_layer)
                assert set(reasons) <= refusals.COPYABLE
                copied |= set(reasons)
                for code in reasons:
                    message = refusals.copy_message(code, report)
                    assert "{" not in message and message.endswith(".") and "searchable copy" in message
                got = refusals.refusal(report, adds_layer=adds_layer)
                if got is None:
                    continue
                code, message = got
                assert code in refusals.HARD
                assert "{" not in message and message.endswith(".") and "makes no copy" in message
                seen.add(code)
    assert seen == refusals.HARD
    assert copied == refusals.COPYABLE - {"provenance_counterparty", "provenance_evidence"}


def test_no_document_text_reaches_a_message_or_a_label(tmp_path: Path):
    pytest.importorskip("pyhanko")
    path = tmp_path / "canary.pdf"
    path.write_bytes(rs.signed_visible(field="ExampleCanaryField"))
    assert b"ExampleCanaryField" in path.read_bytes()
    report = refusals.read(path)
    reasons = refusals.copy_reasons(report, "ordinary", adds_layer=True)
    assert reasons == ["signed"]
    dumped = json.dumps(to_jsonable(report)) + "".join(refusals.copy_message(c, report) for c in reasons)
    assert "ExampleCanaryField" not in dumped


def test_copy_message_words_every_copy_reason(tmp_path: Path):
    """The copy wording is public (decided 2026-11-05), so copy mode can say why, in core words."""
    for code in refusals.COPYABLE:
        message = refusals.copy_message(code)
        assert "{" not in message and message.endswith(".") and "searchable copy" in message, code
        assert "makes no copy" not in message
    for code in refusals.HARD:
        with pytest.raises(ValueError):
            refusals.copy_message(code)
    assert refusals.copy_message("provenance_evidence") == refusals.check_provenance("evidence")[1]
    # PDF/A is a copy reason with or without a layer, so its wording doesn't depend on one.
    pdfa = refusals.read(_xmp_pdf(tmp_path, "a2b-msg", rs.XMP_PDFA_2B))
    assert refusals.copy_reasons(pdfa, "ordinary", adds_layer=False) == ["pdfa"]
    message = refusals.copy_message("pdfa", pdfa)
    assert "PDF/A-2B" in message and "layer" not in message


def test_check_provenance():
    assert refusals.check_provenance("ordinary") is None
    assert refusals.check_provenance("counterparty")[0] == "provenance_counterparty"
    assert refusals.check_provenance("evidence")[0] == "provenance_evidence"
    for value in ("counterparty", "evidence"):
        message = refusals.check_provenance(value)[1]
        assert "searchable copy" in message and "makes no copy" not in message
    for bad in ("", "Ordinary", "other", "ordinary "):
        with pytest.raises(UsageError) as info:
            refusals.check_provenance(bad)
        assert info.value.exit_code == EXIT_USAGE


# --------------------------------------------------------------------------- §10.1 rows: copy or refuse


def _file(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(data)
    return path


def test_the_code_sets_are_the_spec_split():
    assert refusals.COPYABLE == {"provenance_counterparty", "provenance_evidence", "signed", "certified",
                                 "usage_rights", "timestamped", "pdfa", "pdfua_needs_layer", "encrypted"}
    assert refusals.HARD == {"xfa", "xmp_unreadable", "detectors_disagree", "damaged"}
    assert not refusals.COPYABLE & refusals.HARD
    assert refusals.COPYABLE | refusals.HARD == set(refusals.CODES)


@pytest.mark.parametrize("provenance, code", [("counterparty", "provenance_counterparty"),
                                              ("evidence", "provenance_evidence")])
def test_row_provenance_gets_a_copy(tmp_path: Path, provenance: str, code: str):
    report = refusals.read(_write(tmp_path, "plain"))
    assert refusals.refusal(report, adds_layer=True) is None
    assert refusals.copy_reasons(report, provenance, adds_layer=True) == [code]
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == []
    # With another reason the list keeps CODES order, provenance last.
    ur3 = refusals.read(_write(tmp_path, "ur3"))
    assert refusals.copy_reasons(ur3, provenance, adds_layer=True) == ["usage_rights", code]


@pytest.mark.parametrize("bad", ["", "Ordinary", "other", "evidence "])
def test_copy_reasons_checks_the_provenance(tmp_path: Path, bad: str):
    report = refusals.read(_write(tmp_path, "plain"))
    with pytest.raises(UsageError) as info:
        refusals.copy_reasons(report, bad, adds_layer=True)
    assert info.value.exit_code == EXIT_USAGE


@pytest.mark.parametrize("name, reasons", [
    ("signed_visible", ["signed"]), ("signed_invisible", ["signed"]), ("signed_no_widget", ["signed"]),
    ("certified_p1", ["signed", "certified"]), ("certified_p2", ["signed", "certified"]),
    ("ur3", ["usage_rights"]), ("doc_timestamp", ["timestamped"]),
])
def test_row_signatures_certification_usage_rights_and_timestamps_get_a_copy(tmp_path: Path, name: str,
                                                                             reasons: list[str]):
    report = refusals.read(_write(tmp_path, name))
    for adds_layer in (True, False):
        assert refusals.refusal(report, adds_layer=adds_layer) is None
        assert refusals.copy_reasons(report, "ordinary", adds_layer=adds_layer) == reasons


def test_row_pdfa_and_pdfua_that_needs_a_layer_get_a_copy(tmp_path: Path):
    pdfa = refusals.read(_write(tmp_path, "pdfa_2b"))
    pdfua = refusals.read(_write(tmp_path, "pdfua"))
    for report in (pdfa, pdfua):
        assert refusals.refusal(report, adds_layer=True) is None
    assert refusals.copy_reasons(pdfa, "ordinary", adds_layer=False) == ["pdfa"]
    assert refusals.copy_reasons(pdfua, "ordinary", adds_layer=True) == ["pdfua_needs_layer"]
    assert refusals.copy_reasons(pdfua, "ordinary", adds_layer=False) == []     # shrunk only, not a copy


def test_row_permissions_only_encryption_gets_a_copy(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "encrypted_owner"))
    assert report.encrypted == "owner"
    assert refusals.refusal(report, adds_layer=True) is None
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]
    assert "saved without a password" in refusals.copy_message("encrypted", report)


def test_refusals_never_takes_a_password():
    """§11.1: the flow decrypts once; nothing here has a password parameter."""
    for fn in (refusals.read, refusals.refusal, refusals.copy_reasons, refusals.copy_message,
               refusals.check_provenance,
               refusals.output_signature_material):
        assert not [p for p in inspect.signature(fn).parameters if "pass" in p.lower()], fn
    tree = ast.parse((SRC / "refusals.py").read_text(encoding="utf-8"))
    keywords = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    assert "password" not in keywords


def test_row_open_password_reads_as_user_encrypted_without_one(tmp_path: Path):
    path = _write(tmp_path, "encrypted_user")
    report = refusals.read(path)
    assert report.encrypted == "user" and report.walk_codes == ["encrypted"]
    assert "password to open" in refusals.copy_message("encrypted", report)
    assert refusals.refusal(report, adds_layer=True) is None
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]
    # The flow's unencrypted work copy reads as an ordinary file.
    work = refusals.read(_file(tmp_path, "work", rs.decrypt(path.read_bytes())))
    assert work.encrypted is False and "/Encrypt" not in work.raw_markers
    assert refusals.refusal(work, adds_layer=True) is None
    assert refusals.copy_reasons(work, "ordinary", adds_layer=True) == []


def test_a_hard_code_hidden_by_an_open_password_shows_on_the_work_copy(tmp_path: Path):
    """An XFA form inside an encrypted object stream: provisional on the original, refused on the work copy."""
    data = rs.encrypt(rs.xfa(object_streams=True), object_streams=True)
    assert b"/XFA" not in data
    report = refusals.read(_file(tmp_path, "hidden-xfa", data))
    assert report.encrypted == "user" and "/XFA" not in report.raw_markers
    assert refusals.refusal(report, adds_layer=True) is None             # provisional: the walk saw nothing
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]
    work = refusals.read(_file(tmp_path, "hidden-xfa-work", rs.decrypt(data)))
    assert _hard(work) == "xfa" and refusals.copy_reasons(work, "ordinary", adds_layer=True) == []


def test_a_copy_reason_hidden_by_an_open_password_shows_on_the_work_copy(tmp_path: Path):
    data = rs.encrypt(rs.pdfa(compress=True), object_streams=True)
    report = refusals.read(_file(tmp_path, "hidden-pdfa", data))
    assert report.encrypted == "user" and refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]
    work = refusals.read(_file(tmp_path, "hidden-pdfa-work", rs.decrypt(data)))
    assert work.pdfa == "2B" and refusals.refusal(work, adds_layer=True) is None
    assert refusals.copy_reasons(work, "ordinary", adds_layer=True) == ["pdfa"]


@pytest.mark.parametrize("user", ["", rs.PASSWORD_USER], ids=["owner", "user"])
def test_unscanned_encrypted_object_streams_are_not_a_disagreement(tmp_path: Path, user: str):
    """Decided 2026-11-05: an encrypted input is always a copy, so what the raw scan can't decrypt doesn't refuse.

    The same marker on an unencrypted file still refuses as ``detectors_disagree``.
    """
    data = rs.encrypt(rs.base_pdf(object_streams=True), user=user, object_streams=True)
    report = refusals.read(_file(tmp_path, "enc-objstm", data))
    assert report.encrypted == ("user" if user else "owner")
    assert refusals.UNSCANNED in report.raw_markers
    assert refusals.refusal(report) is None
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]
    # Unencrypted, a stream the scan can't read fails closed.
    plain = refusals.read(_write(tmp_path, "plain"))
    plain.raw_markers = [refusals.UNSCANNED]
    assert _hard(plain) == "detectors_disagree"


def test_an_open_password_does_not_hide_what_the_raw_scan_sees(tmp_path: Path):
    """Dictionary keys aren't encrypted: an XFA entry outside an object stream refuses on the original."""
    report = refusals.read(_file(tmp_path, "xfa-user", rs.encrypt(rs.xfa())))
    assert report.encrypted == "user" and "/XFA" in report.raw_markers
    assert _hard(report) == "xfa" and refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["encrypted"]
    report = refusals.read(_file(tmp_path, "ur3-user", rs.encrypt(rs.ur3())))
    assert "usage_rights" in refusals.copy_reasons(report, "ordinary", adds_layer=True)
    assert refusals.refusal(report, adds_layer=True) is None


@pytest.mark.parametrize("encrypted", [False, True])
def test_row_xfa_is_still_refused(tmp_path: Path, encrypted: bool):
    def add_xfa(pdf):
        import pikepdf

        pdf.Root.AcroForm = pdf.make_indirect(pikepdf.Dictionary(
            Fields=pikepdf.Array([]), XFA=pdf.make_stream(b'<xdp:xdp xmlns:xdp="http://ns.adobe.com/xdp/"/>')))

    data = rs._with(rs.ur3(), add_xfa)               # usage rights and XFA together: the hard code wins
    if encrypted:
        data = rs.encrypt(data, user="")
    report = refusals.read(_file(tmp_path, "ur3-xfa", data))
    code, message = refusals.refusal(report, adds_layer=True)
    assert code == "xfa" and "makes no copy" in message
    reasons = refusals.copy_reasons(report, "evidence", adds_layer=True)
    assert "xfa" not in reasons and "usage_rights" in reasons and reasons[-1] == "provenance_evidence"
    assert ("encrypted" in reasons) is encrypted


def test_row_disagreement_unreadable_xmp_and_damage_are_still_refused(tmp_path: Path):
    orphan = refusals.read(_write(tmp_path, "orphan"))
    assert _hard(orphan) == "detectors_disagree" and refusals.copy_reasons(orphan, "ordinary", adds_layer=True) == []
    xmp = refusals.read(_write(tmp_path, "xmp_unreadable"))
    assert _hard(xmp) == "xmp_unreadable" and refusals.copy_reasons(xmp, "ordinary", adds_layer=True) == []
    junk = refusals.read(_file(tmp_path, "junk", b"%PDF-1.7\nnot a document"))
    assert _hard(junk) == "damaged"
    # A damaged encrypted file is refused even though encryption alone would only mean a copy.
    cut = refusals.read(_write(tmp_path, "encrypted_user", "no_trailer"))
    assert _hard(cut) == "damaged" and refusals.copy_reasons(cut, "counterparty", adds_layer=True) == [
        "encrypted", "provenance_counterparty"]


def test_refusal_never_names_a_copyable_code(tmp_path: Path):
    report = refusals.read(_write(tmp_path, "certified_p1"))
    assert refusals._codes(report, adds_layer=True) == ["certified", "signed"]
    assert refusals.refusal(report, adds_layer=True) is None
    assert refusals.copy_reasons(report, "ordinary", adds_layer=True) == ["signed", "certified"]       # CODES order


# --------------------------------------------------------------------------- output_signature_material


@pytest.mark.parametrize("name, expected", [
    ("plain", []),
    ("empty_field", []),
    ("signed_visible", ["/ByteRange", "/Sig field"]),
    ("certified_p1", ["/ByteRange", "/Perms", "/Sig field"]),
    ("ur3", ["/ByteRange", "/Perms"]),
    ("orphan", ["/ByteRange"]),
])
def test_output_signature_material(tmp_path: Path, name: str, expected: list[str]):
    assert refusals.output_signature_material(_write(tmp_path, name)) == expected


def test_output_signature_material_of_a_rewritten_signed_file(tmp_path: Path):
    """A full pikepdf rewrite keeps the dead signature dictionary: still material."""
    import pikepdf

    pytest.importorskip("pyhanko")
    path = tmp_path / "rewritten.pdf"
    with pikepdf.open(io.BytesIO(_bytes("signed_invisible"))) as pdf:
        pdf.save(path)
    assert refusals.output_signature_material(path) == ["/ByteRange", "/Sig field"]


def test_an_empty_signature_value_is_not_output_material(tmp_path: Path):
    """The same test as read(): a /Sig field whose value holds neither /ByteRange nor /Contents is empty."""
    import pikepdf

    def edit(pdf):
        field = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name.Sig, T=pikepdf.String("Example"),
                                                     V=pikepdf.Dictionary(Type=pikepdf.Name.Sig)))
        pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([field]))

    path = tmp_path / "v-only.pdf"
    path.write_bytes(rs._with(rs.base_pdf(), edit))
    report = refusals.read(path)
    assert refusals.refusal(report, adds_layer=True) is None and report.empty_sig_fields == 1
    assert refusals.output_signature_material(path) == []


def test_marker_text_in_an_output_string_is_not_material(tmp_path: Path):
    import pikepdf

    path = tmp_path / "subject.pdf"
    path.write_bytes(rs._with(rs.base_pdf(), lambda pdf: pdf.docinfo.__setitem__(
        "/Subject", pikepdf.String("Example /DocMDP and /ByteRange [ note"))))
    assert refusals.output_signature_material(path) == []


def test_output_signature_material_fails_closed(tmp_path: Path):
    path = tmp_path / "junk.pdf"
    path.write_bytes(b"not a pdf")
    assert refusals.output_signature_material(path) == ["unreadable"]


# --------------------------------------------------------------------------- packaging


def test_pyhanko_is_a_test_extra_only():
    import tomllib

    project = tomllib.loads((SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert not [d for d in project["dependencies"] if d.lower().startswith("pyhanko")]
    assert [d for d in project["optional-dependencies"]["test"] if d.lower().startswith("pyhanko")]
    tree = ast.parse((SRC / "refusals.py").read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level}
    assert "pyhanko" not in imported and "pyhanko" not in sys.modules.get("pdfintake.refusals").__dict__
