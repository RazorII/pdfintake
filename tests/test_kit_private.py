"""kit/private.py: the private bench keeps only numbers, and its guard trips (exit 8) on anything else.

The "private" folder here holds made-up PDFs built in the test; fake engines read them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake import platforms
from pdfintake.kit import bench as B
from pdfintake.kit import cli
from pdfintake.kit import private as P
from pdfintake.langpacks import manifest as M
from test_kit_render import fake_pack, latin_chars

KEY = platforms.current().key
LINES = ["The example invoice lists seven sample items.", "A total of 4521 units for the test account.",
         "Jane Doe will send the form before Friday."]
PLANTED = "Jane Doe, 12 Example Street"


def _draft(tmp: Path, monkeypatch: pytest.MonkeyPatch, texts: list[str] | None = None) -> tuple[Path, Path, M.Manifest]:
    packs, work, _web = fake_pack(tmp, monkeypatch, texts=texts)
    m, _raw, _ = B.load_manifest(packs, "en")
    return packs, work, m


def test_the_guard_refuses_text_and_names_none_of_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    packs, _work, m = _draft(tmp_path, monkeypatch)
    allowed = P.allowed_metrics(m)
    assert allowed == {"words", "trigrams", "digits", "must-accents"}
    for agg in (P.PrivateAggregate(3, {PLANTED: [0.9, 0.8, 1.0]}),
                P.PrivateAggregate(3, {"words": [0.9, PLANTED, 1.0]}),
                P.PrivateAggregate(3, {"words": [0.9, 0.8, 1.5]}),
                P.PrivateAggregate(0, {"words": [0.9, 0.8, 1.0]})):
        with pytest.raises(P.PrivacyGuard) as exc:
            P.write_private(packs / "en" / "bench.private.json", agg, m)
        assert exc.value.exit_code == 8 and PLANTED not in str(exc.value) and PLANTED not in repr(exc.value.detail)
        assert not (packs / "en" / "bench.private.json").exists()
    data = P.write_private(packs / "en" / "bench.private.json", P.PrivateAggregate(3, {"words": [0.9, 0.8, 1.0]}), m)
    assert data == {"pages": 3, "metrics": {"words": [0.9, 0.8, 1.0]}}
    assert json.loads((packs / "en" / "bench.private.json").read_text(encoding="utf-8")) == data


def test_kit_private_exits_8_on_a_planted_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]):
    packs, work, _m = _draft(tmp_path, monkeypatch)
    folder = tmp_path / "mine"
    folder.mkdir()
    monkeypatch.setattr(P, "private_bench", lambda *a, **k: P.PrivateAggregate(2, {"words": [0.9, 0.8, 1.0],
                                                                                   PLANTED: [0.5, 0.5, 0.5]}))
    code = cli.main(["kit", "private", "en", "--folder", str(folder), "--packs", str(packs), "--work", str(work),
                     "--offline", "--json"])
    out = capsys.readouterr().out
    assert code == 8 and PLANTED not in out and json.loads(out)["error"]["code"] == "kit_privacy"
    assert not (packs / "en" / "bench.private.json").exists()
    monkeypatch.setattr(P, "private_bench", lambda *a, **k: P.PrivateAggregate(2, {"words": [0.9, 0.8, 1.0]}))
    assert cli.main(["kit", "private", "en", "--folder", str(folder), "--packs", str(packs), "--work", str(work),
                     "--offline"]) == 0
    raw = json.loads((packs / "en" / "manifest.json").read_text(encoding="utf-8"))
    assert raw["bench"]["private"] == {"pages": 2, "metrics": {"words": [0.9, 0.8, 1.0]}}
    assert M.check(raw) == []


def test_private_bench_reads_only_born_digital_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _packs, work, m = _draft(tmp_path, monkeypatch, texts=LINES)
    pack = B.installed_pack(m, B.files_dir(work, "en"), work / "en" / "pack")
    folder = tmp_path / "mine"
    folder.mkdir()
    font = build.write_box_font(tmp_path / "ExampleSans-Regular.ttf", latin_chars(), family="Example Sans")
    build.fpdf_text_pdf(folder / "letter.pdf", LINES, font, size=14.0)
    build.write_pdf(folder / "scan.pdf", 1)                              # no text: not a bench page
    (folder / "notes.txt").write_text("not a PDF", encoding="utf-8")
    agg = P.private_bench(folder, pack, pack, KEY)
    assert agg.pages == 1
    assert set(agg.metrics) <= P.allowed_metrics(m) and agg.metrics["words"] == [1.0, 1.0, 1.0]
    assert agg.metrics["digits"] == [1.0, 1.0, 1.0]
    P.guard(agg, P.allowed_metrics(m))
    assert not list(Path(work).rglob("*.png"))                           # the renders were temporary


def test_a_private_page_with_few_group_characters_gives_no_must_triple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two of the group's characters (é) on the only page: too few to measure, where each page's own
    figure used to give 1.0."""
    lines = ["The café menu lists a crème dish.", "Jane Doe paid at the counter.", "A pâté plate for the test account."]
    _packs, work, m = _draft(tmp_path, monkeypatch, texts=lines)
    pack = B.installed_pack(m, B.files_dir(work, "en"), work / "en" / "pack")
    folder = tmp_path / "mine"
    folder.mkdir()
    font = build.write_box_font(tmp_path / "ExampleSans-Regular.ttf", latin_chars(), family="Example Sans")
    build.fpdf_text_pdf(folder / "letter.pdf", lines, font, size=14.0)
    agg = P.private_bench(folder, pack, pack, KEY)
    assert agg.pages == 1 and agg.metrics["words"] == [1.0, 1.0, 1.0]
    assert "must-accents" not in agg.metrics


def test_the_private_must_triple_counts_pages_as_the_bench_counts_conditions():
    """A page with one or two of a group's characters inside read words would score 0 or 1 and set
    the worst or best; like a condition in the bench, a page counts only with 20 read characters."""
    from pdfintake.kit.score import PageScore

    def page(read: int, survived: int) -> PageScore:
        return PageScore("p", "clean", None, None, None, must_counts={"g": [read + 5, read, survived]})

    assert P.must_triple([page(40, 40), page(30, 27), page(2, 0)], "g") == [0.95, 0.9, 1.0]
    # No page has 20 of its own: the pages add up into one figure, or none when even that is short.
    assert P.must_triple([page(12, 12), page(10, 8)], "g") == [0.9091, 0.9091, 0.9091]
    assert P.must_triple([page(12, 12), page(2, 0)], "g") is None
    assert P.must_triple([PageScore("p", "clean", None, None, None)], "g") is None
