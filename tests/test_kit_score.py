"""kit/score.py: multiset recalls of words, trigrams, digit runs and must-survive units (made-up text)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import build
from pdfintake.kit import score as S
from pdfintake.kit.render import BenchPage
from pdfintake.langpacks.manifest import MustSurvive
from pdfintake.types import EngineRead, Fill, LangCall, PageOCR, Pick, ScriptCall


def test_word_recall_is_a_multiset_recall():
    assert S.word_recall("the cat, the dog.", "The cat the DOG", "Latin") == 1.0
    assert S.word_recall("the the the cat", "the cat", "Latin") == pytest.approx(0.5)
    assert S.word_recall("", "anything", "Latin") == 1.0


def test_trigram_recall_ignores_spacing():
    truth = "บริษัทตัวอย่าง จำกัด"
    assert S.trigram_recall(truth, "บริษัท ตัวอย่างจำกัด", "Thai") == 1.0
    assert 0.0 < S.trigram_recall(truth, "บริษัท", "Thai") < 1.0


def test_digit_runs_count_any_script_digits():
    assert S.digit_run_recall("total 4521 and 17", "total ๔๕๒๑ and 17") == 1.0
    assert S.digit_run_recall("4521 4521", "4521") == pytest.approx(0.5)


def test_must_survive_keeps_case_and_needs_units():
    ms = MustSurvive("sharp-s", "ßäÄ", "word", 0.9)
    assert S.must_survive_recall("Straße und Äpfel", "Straße und Apfel", ms, "Latin") == pytest.approx(0.5)
    assert S.must_survive_recall("no such letters", "no such letters", ms, "Latin") is None
    tri = MustSurvive("tone-marks", "่้", "trigram", 0.75)
    assert S.must_survive_recall("ข้อความตัวอย่าง", "ขอความตัวอยาง", tri, "Thai") == 0.0
    # char groups count inside spans whose skeleton was read: "cafee" matches, the last é wasn't read.
    ch = MustSurvive("acute", "é", "char", 0.9)
    assert S.must_survive_counts("café éé", "cafe é", ch, "Latin") == (3, 2, 1)
    assert S.must_survive_recall("café éé", "cafe é", ch, "Latin") == pytest.approx(1 / 2)


def test_primary_metric():
    assert S.primary_metric("Latin") == "words"
    assert S.primary_metric("Thai") == "trigrams"


def _ocr(lines: list[str], fills: list[str] = (), seconds: float = 0.5) -> PageOCR:
    read = build.engine_read("tess", lines)
    read.seconds = seconds
    pick = Pick("single", 1, "tess", read.lines, {}, False, "made up")
    fs = [Fill("number", t, 0.9, build.quad(0, 0, 10, 10), "checker") for t in fills]
    return PageOCR(0, ScriptCall("Latin", 1.0, "hint", None), [LangCall("en", 1.0, "forced", 0)], "en", "forced",
                   [read], pick, fs, {}, {}, [])


def test_score_page_reads_the_pick_and_the_fills(tmp_path: Path):
    page = BenchPage("p-clean", "clean", "form", tmp_path / "p.png", "Total Café: 4521", 300.0)
    must = [MustSurvive("acute", "é", "word", 0.9)]
    without = S.score_page(page, _ocr(["Total Cafe:"]), False, must, "Latin")
    assert without.digits == 0.0 and without.must == {"acute": 0.0} and without.words == pytest.approx(1 / 3)
    assert without.must_counts == {"acute": [1, 1, 0]}
    with_fills = S.score_page(page, _ocr(["Total Café:"], ["4521"]), True, must, "Latin")
    assert with_fills.digits == 1.0 and with_fills.must == {"acute": 1.0} and with_fills.words == 1.0
    assert with_fills.seconds == 0.5 and with_fills.page == "p-clean" and with_fills.cond == "clean"
    thai = S.score_page(BenchPage("t", "scan", "prose", tmp_path / "t.png", "ตัวอย่าง", 200.0),
                        _ocr(["ตัวอย่าง"]), True, [], "Thai")
    assert thai.words is None and thai.trigrams == 1.0 and thai.digits is None
