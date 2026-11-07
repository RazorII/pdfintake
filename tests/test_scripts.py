"""ocr/scripts.py: script lookup, runs, per-script trigrams and the corroborated count."""

from __future__ import annotations

import unicodedata
from collections import Counter

import pytest

from pdfintake.ocr import scripts
from pdfintake.ocr.scripts import SCRIPTS, corroborated, script_of, script_runs, script_text, trigrams


@pytest.mark.parametrize("cp, expected", [
    (0x0040, None), (0x0041, "Latin"), (0x005A, "Latin"), (0x005B, None),
    (0x0060, None), (0x0061, "Latin"), (0x007A, "Latin"), (0x007B, None),
    (0x00BF, None), (0x00C0, "Latin"), (0x00D7, None), (0x00F7, None), (0x0151, "Latin"), (0x0171, "Latin"),
    (0x0300, "Inherited"), (0x036F, "Inherited"),
    (0x0E00, None), (0x0E01, "Thai"), (0x0E31, "Thai"), (0x0E3A, "Thai"), (0x0E3B, None),
    (0x0E3F, None),                                    # the baht sign is Common
    (0x0E40, "Thai"), (0x0E50, "Thai"), (0x0E5B, "Thai"), (0x0E5C, None),
    (0x0391, "Greek"), (0x0410, "Cyrillic"), (0x05D0, "Hebrew"), (0x0627, "Arabic"), (0x0915, "Devanagari"),
    (0x4E00, "Han"), (0x9FFF, "Han"), (0x3042, "Hiragana"), (0x30A2, "Katakana"), (0xAC00, "Hangul"),
    (0xD7A3, "Hangul"), (0xD7A4, None),
    (0x0030, None), (0x0020, None), (0x002E, None),
])
def test_script_of_boundaries(cp: int, expected: str | None) -> None:
    assert script_of(chr(cp)) == expected


def test_script_of_takes_one_character() -> None:
    with pytest.raises(ValueError):
        script_of("ab")


def test_table_is_consistent() -> None:
    for name, sc in SCRIPTS.items():
        assert sc.name == name
        assert sc.required, name
        for cp in sc.required:
            assert script_of(chr(cp)) == name, f"{name} U+{cp:04X}"
        assert sc.osd_names
    assert SCRIPTS["Latin"].spaced and not SCRIPTS["Thai"].spaced
    assert "Fraktur" in SCRIPTS["Latin"].osd_names
    assert ord("ő") in SCRIPTS["Latin"].required and ord("ű") in SCRIPTS["Latin"].required


def test_thai_trigram_ranges_exclude_digits() -> None:
    thai = SCRIPTS["Thai"]
    assert thai.trigram_ranges[0][0] == 0x0E01 and thai.trigram_ranges[-1][1] == 0x0E4F
    for cp in range(0x0E50, 0x0E5A):
        assert not thai.counts_for_trigrams(chr(cp))


def test_thai_trigrams_count_the_baht_sign() -> None:
    # U+0E01-U+0E4F count, the baht sign included, though script_of calls it Common
    assert script_of("฿") is None
    assert SCRIPTS["Thai"].counts_for_trigrams("฿")
    assert trigrams("ราคา฿100", "Thai") == Counter({"ราค": 1, "าคา": 1, "คา฿": 1})
    assert trigrams("฿฿฿", "Thai") == Counter({"฿฿฿": 1})
    assert trigrams("กข฿คง", "Thai") == Counter({"กข฿": 1, "ข฿ค": 1, "฿คง": 1})


def test_thai_digits_are_excluded_from_trigrams() -> None:
    assert trigrams("๑๒๓๔๕", "Thai") == Counter()
    assert trigrams("กขค๑๒๓", "Thai") == Counter({"กขค": 1})
    # digits between letters don't break the run: runs are joined
    assert trigrams("กข๑ค", "Thai") == Counter({"กขค": 1})


def test_trigrams_join_runs_and_normalise() -> None:
    # spacing and line breaks differ between readers; runs are joined
    assert trigrams("กข ค\nง", "Thai") == Counter({"กขค": 1, "ขคง": 1})
    # sara am composed and decomposed compare the same (NFKC)
    assert trigrams("กำก", "Thai") == trigrams("กําก", "Thai")
    # Latin letters never count for Thai, and Thai never for Latin
    assert trigrams("ABC", "Thai") == Counter()
    assert trigrams("ขคง", "Latin") == Counter()
    assert trigrams("ab cd", "Latin") == Counter({"abc": 1, "bcd": 1})


def test_trigrams_unknown_script() -> None:
    with pytest.raises(KeyError):
        trigrams("abc", "Klingon")


def test_corroborated_count_is_asymmetric() -> None:
    mine = trigrams("กขคกขคกขค", "Thai")              # กขค three times
    theirs = trigrams("กขค", "Thai")
    assert corroborated(mine, set(theirs)) == 3          # multiplicity on mine
    assert corroborated(theirs, set(mine)) == 1          # a set on the others
    assert corroborated(mine, set()) == 0


def test_corroborated_background_junk() -> None:
    real = "ข้อความตัวอย่าง"
    junk = "ขฃคฅฆงจฉชซฌญ" * 5
    a = trigrams(real, "Thai")
    b = trigrams(real + junk, "Thai")
    others = set(trigrams(real, "Thai"))
    assert sum(b.values()) > sum(a.values())
    assert corroborated(b, others) == corroborated(a, others)


def test_script_runs() -> None:
    assert script_runs("") == []
    assert script_runs("12 - 34") == [("Common", "12 - 34")]
    assert script_runs("ข้อความ ABC 12") == [("Thai", "ข้อความ "), ("Latin", "ABC 12")]
    assert script_runs("(ABC) กข") == [("Latin", "(ABC) "), ("Thai", "กข")]
    # a generic combining mark stays with its base
    text = "éx กข"
    assert script_runs(text) == [("Latin", "éx "), ("Thai", "กข")]
    for t in ("ข้อความ ABC 12", "á ๑ xyz", "  กข"):
        assert "".join(r for _, r in script_runs(t)) == t


def test_script_text() -> None:
    assert script_text("ข้อความ ABC กข 12", "Thai") == "ข้อความ กข"
    assert script_text("ข้อความ ABC def", "Latin") == "ABC def"
    assert script_text("é ก", "Latin") == "é"
    assert script_text("", "Latin") == ""
    with pytest.raises(KeyError):
        script_text("x", "Klingon")


def test_script_dataclass_refuses_bad_tables() -> None:
    with pytest.raises(ValueError):
        scripts.Script("X", ((10, 5),), (), True, ("X",), frozenset({10}))
    with pytest.raises(ValueError):
        scripts.Script("X", ((10, 20),), ((30, 40),), True, ("X",), frozenset({10}))
    with pytest.raises(ValueError):
        scripts.Script("X", ((10, 20),), ((10, 20),), True, ("X",), frozenset({99}))
    # a trigram range may reach a Common symbol outside the ranges, never a letter or control
    scripts.Script("X", ((0x41, 0x5A),), ((0x24, 0x24), (0x41, 0x5A)), True, ("X",), frozenset({0x41}))
    with pytest.raises(ValueError):
        scripts.Script("X", ((0x41, 0x5A),), ((0x41, 0x5B),), True, ("X",), frozenset({0x41}))
    with pytest.raises(ValueError):
        scripts.Script("X", ((0x41, 0x5A),), ((0x41, 0x7A),), True, ("X",), frozenset({0x41}))


def test_inherited_includes_every_unscripted_mark() -> None:
    for cp in range(0x0300, 0x0370):
        ch = chr(cp)
        if unicodedata.category(ch).startswith("M"):
            assert script_of(ch) == "Inherited"
