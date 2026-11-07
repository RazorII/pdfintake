"""``primary_with_floor`` v1 and ``single`` v1 on synthetic reads (every text is made up)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from fixtures import build
from pdfintake.errors import PackError
from pdfintake.ocr import rules
from pdfintake.ocr.rules.primary import FLOOR_V1, SINGLE_V1
from pdfintake.params import ValidationContext
from pdfintake.types import EngineRead

CTX = ValidationContext(use_ids=frozenset({"tesseract", "rapidocr", "vision"}), asset_keys=frozenset(), asset_kinds={},
                        scripts=frozenset({"Latin"}), where="test.pick.params")


# --------------------------------------------------------------------------- helpers


def words(n: int, start: int = 0) -> list[str]:
    """n distinct made-up words of six letters."""
    return [f"word{chr(97 + (i // 26) % 26)}{chr(97 + i % 26)}" for i in range(start, start + n)]


def as_lines(ws: Sequence[str], per: int = 8) -> list[str]:
    return [" ".join(ws[i:i + per]) for i in range(0, len(ws), per)]


def floor_params(**extra) -> dict:
    return FLOOR_V1.PARAMS.validate({"primary": "tesseract", "checker": "rapidocr", **extra}, CTX)


def tess(texts: Sequence[str] = (), status: str | None = None, conf: float = 0.9) -> EngineRead:
    return build.engine_read("tesseract", texts, role="reader", status=status, conf=conf,
                             error=f"made-up {status}" if status in ("failed", "timeout") else None)


def rapid(texts: Sequence[str] = (), status: str | None = None, conf: float = 0.9) -> EngineRead:
    return build.engine_read("rapidocr", texts, role="checker", status=status, conf=conf)


def floor_pick(reads: dict[str, EngineRead], fallback: str, p: dict | None = None):
    return FLOOR_V1.pick(reads, p if p is not None else floor_params(), fallback=fallback, checker_id="rapidocr")


# --------------------------------------------------------------------------- primary_with_floor (§7.1)


def test_agreement_005_falls_back_with_checker_and_warns_with_none():
    theirs = words(40)
    reads = {"tesseract": tess(["page " + " ".join(theirs[:2]) + " 4521"]), "rapidocr": rapid(as_lines(theirs))}
    back = floor_pick(reads, "checker")
    assert back.chosen == "rapidocr" and back.fallback is True
    assert back.lines == reads["rapidocr"].lines
    assert back.scores == {"tesseract": 0.05}
    assert back.warnings == []                         # the pipeline adds fallback_used from pick.fallback
    warned = floor_pick(reads, "none")
    assert warned.chosen == "tesseract" and warned.fallback is False
    assert warned.lines == reads["tesseract"].lines
    assert warned.scores == {"tesseract": 0.05}
    assert warned.warnings == ["agreement_low"]
    for p in (back, warned):
        assert "0.05" in p.reason and "floor" in p.reason
        assert not any(w in p.reason for w in theirs), "a reason never quotes document text"


def test_under_20_confident_words_the_floor_is_skipped():
    reads = {"tesseract": tess(["something else entirely here"]), "rapidocr": rapid(as_lines(words(19)))}
    for fallback in ("checker", "none"):
        p = floor_pick(reads, fallback)
        assert p.chosen == "tesseract" and p.fallback is False and p.warnings == []
        assert p.scores == {"tesseract": 0.0}
        assert "19 confident words" in p.reason
    # Plenty of words, but on lines under word_conf: still not enough to judge the primary by.
    reads["rapidocr"] = rapid(as_lines(words(60)), conf=0.7)
    p = floor_pick(reads, "checker")
    assert p.chosen == "tesseract" and p.fallback is False and p.scores == {}
    # Words shorter than min_len don't count either.
    reads["rapidocr"] = rapid(as_lines([w[:3] for w in words(60)]))
    assert floor_pick(reads, "checker").chosen == "tesseract"
    # Twenty is enough.
    reads["rapidocr"] = rapid(as_lines(words(20)))
    p = floor_pick(reads, "checker")
    assert p.chosen == "rapidocr" and p.fallback is True


def test_at_the_floor_the_primary_is_kept():
    theirs = words(40)
    reads = {"tesseract": tess(as_lines(theirs[:20])), "rapidocr": rapid(as_lines(theirs))}
    p = floor_pick(reads, "checker")
    assert p.chosen == "tesseract" and p.scores == {"tesseract": 0.5} and p.warnings == []


def test_the_primarys_words_count_from_every_line_whatever_its_confidence():
    # Tesseract's line conf (mean word conf / 100) often sits under word_conf on a page it read well;
    # word_conf filters the checker's words only, so a full agreement stays a full agreement.
    theirs = words(25)
    reads = {"tesseract": tess(as_lines(theirs), conf=0.55), "rapidocr": rapid(as_lines(theirs), conf=0.9)}
    for fallback in ("checker", "none"):
        p = floor_pick(reads, fallback)
        assert p.chosen == "tesseract" and p.fallback is False and p.warnings == []
        assert p.scores == {"tesseract": 1.0}
        assert p.lines == reads["tesseract"].lines


def test_agreement_counts_the_checkers_words_with_multiplicity():
    theirs = words(10) * 3 + words(10, start=10)                # 40 words, 30 of them repeats of 10
    reads = {"tesseract": tess(as_lines(words(10))), "rapidocr": rapid(as_lines(theirs))}
    assert floor_pick(reads, "none").scores == {"tesseract": 0.75}


def test_folding_lets_double_acute_and_umlaut_agree():
    stems = [f"{w}szőlő" for w in words(25)]
    seen_as = [s.replace("ő", "ö") for s in stems]               # the checker's model has no ő
    reads = {"tesseract": tess(as_lines(stems)), "rapidocr": rapid(as_lines(seen_as))}
    p = floor_pick(reads, "none")
    assert p.scores == {"tesseract": 1.0} and p.chosen == "tesseract" and p.warnings == []


def test_params_change_the_floor():
    theirs = words(30)
    reads = {"tesseract": tess(as_lines(theirs[:12])), "rapidocr": rapid(as_lines(theirs))}   # 0.4
    assert floor_pick(reads, "checker").chosen == "rapidocr"
    assert floor_pick(reads, "checker", floor_params(floor=0.3)).chosen == "tesseract"
    assert floor_pick(reads, "checker", floor_params(min_words=31)).chosen == "tesseract"
    assert floor_pick(reads, "checker", floor_params(min_len=7)).chosen == "tesseract"   # six-letter words drop out
    assert floor_pick(reads, "checker", floor_params(word_conf=0.95)).chosen == "tesseract"


def test_an_empty_primary_is_judged_by_the_floor():
    reads = {"tesseract": tess(), "rapidocr": rapid(as_lines(words(25)))}
    assert reads["tesseract"].status == "empty"
    back = floor_pick(reads, "checker")
    assert back.chosen == "rapidocr" and back.fallback is True and back.scores == {"tesseract": 0.0}
    kept = floor_pick(reads, "none")
    assert kept.chosen == "tesseract" and kept.lines == [] and kept.warnings == ["agreement_low"]


@pytest.mark.parametrize("status", ["failed", "timeout", "unavailable", "empty", None])
def test_without_a_checker_read_the_primary_is_kept(status: str | None):
    reads = {"tesseract": tess(as_lines(words(3)))}
    if status is not None:
        reads["rapidocr"] = rapid(status=status)
    for fallback in ("checker", "none"):
        p = floor_pick(reads, fallback)
        assert p.chosen == "tesseract" and p.fallback is False and p.warnings == [] and p.scores == {}


@pytest.mark.parametrize("status", ["failed", "timeout", "unavailable"])
def test_primary_not_working(status: str):
    reads = {"tesseract": tess(status=status), "rapidocr": rapid(as_lines(words(30)))}
    back = floor_pick(reads, "checker")
    assert back.chosen == "rapidocr" and back.fallback is True and back.warnings == []
    gone = floor_pick(reads, "none")
    assert gone.chosen is None and gone.lines == [] and gone.warnings == []
    assert "no layer" in gone.reason


def test_floor_params():
    assert floor_params() == {"primary": "tesseract", "checker": "rapidocr", "floor": 0.5, "min_words": 20,
                              "word_conf": 0.8, "min_len": 4}
    for bad in ({"floor": 1.5}, {"word_conf": -0.1}, {"min_len": 0}, {"min_words": 2.5}, {"primary": "nobody"}):
        with pytest.raises(PackError):
            FLOOR_V1.PARAMS.validate({"primary": "tesseract", "checker": "rapidocr", **bad}, CTX)
    with pytest.raises(PackError) as info:
        FLOOR_V1.PARAMS.validate({"primary": "tesseract"}, CTX)
    assert "checker" in str(info.value)


# --------------------------------------------------------------------------- single


def single_pick(reads: dict[str, EngineRead], fallback: str, reader: str = "tesseract"):
    return SINGLE_V1.pick(reads, SINGLE_V1.PARAMS.validate({"reader": reader}, CTX), fallback=fallback,
                          checker_id="rapidocr")


def test_single_keeps_its_reader():
    reads = {"tesseract": tess(["one made-up line"]), "rapidocr": rapid(as_lines(words(40))),
             "vision": build.engine_read("vision", as_lines(words(40)), role="reader")}
    for fallback in ("checker", "none"):
        p = single_pick(reads, fallback)
        assert p.chosen == "tesseract" and p.fallback is False and p.warnings == [] and p.scores == {}
        assert p.lines == reads["tesseract"].lines and p.reason
    assert single_pick(reads, "none", reader="vision").chosen == "vision"


def test_single_reader_failing():
    reads = {"tesseract": tess(status="failed"), "rapidocr": rapid(["a made-up checker line"])}
    back = single_pick(reads, "checker")
    assert back.chosen == "rapidocr" and back.fallback is True and "tesseract failed" in back.reason
    gone = single_pick(reads, "none")
    assert gone.chosen is None and gone.lines == [] and gone.fallback is False
    missing = single_pick({"rapidocr": rapid(["a made-up checker line"])}, "checker")
    assert missing.chosen == "rapidocr" and "gave no read" in missing.reason


def test_single_empty_read_is_kept():
    p = single_pick({"tesseract": tess(), "rapidocr": rapid(["made-up"])}, "checker")
    assert p.chosen == "tesseract" and p.lines == [] and p.fallback is False


def test_single_params():
    with pytest.raises(PackError):
        SINGLE_V1.PARAMS.validate({}, CTX)
    with pytest.raises(PackError):
        SINGLE_V1.PARAMS.validate({"reader": "tesseract", "floor": 0.5}, CTX)


def test_floor_serves_spaced_word_scripts_only():
    assert FLOOR_V1.scripts == frozenset({"Latin", "Cyrillic", "Greek"})
    assert "Thai" not in FLOOR_V1.scripts
    assert SINGLE_V1.scripts is None


def test_registered():
    assert rules.get("primary_with_floor", 1) is FLOOR_V1
    assert rules.get("single", 1) is SINGLE_V1
    assert isinstance(FLOOR_V1, rules.PickRule) and isinstance(SINGLE_V1, rules.PickRule)
