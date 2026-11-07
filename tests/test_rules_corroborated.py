"""``corroborated_trigrams`` v1 on synthetic reads (every text is made up).

The Thai texts are strings of Thai consonants built by ``_Gen``: deterministic (SHA-256 counter
stream), and no trigram repeats inside or across the pieces it builds, so every corroboration
below is one the test put there on purpose.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import pytest

from fixtures import build
from pdfintake.errors import PackError
from pdfintake.ocr import rules
from pdfintake.ocr.rules.corroborated import RULE_V1
from pdfintake.params import ValidationContext
from pdfintake.types import EngineRead

THAI_CONSONANTS = [chr(c) for c in range(0x0E01, 0x0E2F)]
CTX = ValidationContext(use_ids=frozenset({"vision", "tesseract", "rapidocr"}), asset_keys=frozenset(), asset_kinds={},
                        scripts=frozenset({"Thai", "Latin"}), where="test.pick.params")


# --------------------------------------------------------------------------- helpers


class _Gen:
    """Thai-consonant strings with no trigram used twice across everything one generator builds."""

    def __init__(self, seed: int) -> None:
        self.seed, self.n, self.used = seed, 0, set()

    def fresh(self, length: int) -> str:
        out = ""
        while len(out) < length:
            self.n += 1
            for b in hashlib.sha256(f"{self.seed}:{self.n}".encode()).digest():
                ch = THAI_CONSONANTS[b % len(THAI_CONSONANTS)]
                gram = (out + ch)[-3:]
                if len(out) < 2 or gram not in self.used:
                    if len(out) >= 2:
                        self.used.add(gram)
                    out += ch
                    break
        return out


def thai_chars(read: EngineRead) -> int:
    return sum("ก" <= c <= "๏" for c in read.text())


def params(tie_order: Sequence[str] = ("vision", "tesseract"), corroborators: Sequence[str] = (),
           script: str = "Thai") -> dict:
    return RULE_V1.PARAMS.validate({"script": script, "tie_order": list(tie_order),
                                    "corroborators": list(corroborators)}, CTX)


def reader(use_id: str, texts: Sequence[str] = (), status: str | None = None) -> EngineRead:
    return build.engine_read(use_id, texts, role="reader", status=status,
                             error=f"made-up {status}" if status in ("failed", "timeout") else None)


def checker(texts: Sequence[str] = (), status: str | None = None) -> EngineRead:
    return build.engine_read("rapidocr", texts, role="checker", status=status)


def pick(reads: dict[str, EngineRead], p: dict | None = None, fallback: str = "none"):
    return RULE_V1.pick(reads, p if p is not None else params(), fallback=fallback, checker_id="rapidocr")


def no_thai(reason: str) -> bool:
    return not re.search("[฀-๿]", reason)


# --------------------------------------------------------------------------- §7.1


def test_background_junk_does_not_win_corroborated_thai_does():
    g = _Gen(7)
    real_a, real_b, junk = g.fresh(150), g.fresh(250), g.fresh(1500)
    vision = reader("vision", [real_a, real_b])
    tesseract = reader("tesseract", [real_a[:60], junk])          # the background pattern, read as Thai
    rapid = checker([real_a, real_b])
    assert thai_chars(tesseract) > 3 * thai_chars(vision)
    reads = {"tesseract": tesseract, "vision": vision, "rapidocr": rapid}
    p = pick(reads, params(tie_order=("tesseract", "vision"), corroborators=("rapidocr",)))
    assert p.chosen == "vision" and p.fallback is False and p.warnings == []
    assert p.scores["vision"] > p.scores["tesseract"]
    assert p.lines == vision.lines
    assert "rapidocr" not in p.scores
    # Without a corroborator the two readers share only real_a: 1,500 more characters of junk add nothing.
    memo = pick(reads, params(tie_order=("vision", "tesseract")))
    assert memo.scores["vision"] == memo.scores["tesseract"] == 58.0
    assert memo.chosen == "vision" and "tie_order" in memo.reason


def test_a_tie_goes_to_vision():
    text = _Gen(3).fresh(200)
    for order in (("tesseract", "vision"), ("vision", "tesseract")):
        reads = {u: reader(u, [text]) for u in order}
        reads["rapidocr"] = checker([text])
        p = pick(reads)
        assert p.chosen == "vision"
        assert p.scores == {"vision": 198.0, "tesseract": 198.0}
        assert "tied with tesseract" in p.reason


@pytest.mark.parametrize("status", ["failed", "timeout", "unavailable"])
def test_vision_failing_leaves_tesseract_uncontested(status: str):
    text = _Gen(4).fresh(120)
    reads = {"vision": reader("vision", status=status), "tesseract": reader("tesseract", [text]),
             "rapidocr": checker([_Gen(5).fresh(300)])}                     # the checker disagrees entirely
    for fallback in ("none", "checker"):
        for corr in ((), ("rapidocr",)):
            p = pick(reads, params(corroborators=corr), fallback)
            assert p.chosen == "tesseract" and p.fallback is False and p.warnings == []
            assert p.lines == reads["tesseract"].lines
            assert p.scores == {}
            assert "uncorroborated" in p.reason


def test_neither_reader_checker_fallback_or_no_layer():
    reads = {"vision": reader("vision", status="failed"), "tesseract": reader("tesseract", status="timeout"),
             "rapidocr": checker([_Gen(6).fresh(90)])}
    kept = pick(reads, fallback="checker")
    assert kept.chosen == "rapidocr" and kept.fallback is True
    assert kept.lines == reads["rapidocr"].lines
    assert "vision failed" in kept.reason and "tesseract timed out" in kept.reason
    none = pick(reads, fallback="none")
    assert none.chosen is None and none.lines == [] and none.fallback is False
    assert "no layer" in none.reason
    reads["rapidocr"] = checker(status="failed")
    gone = pick(reads, fallback="checker")
    assert gone.chosen is None and gone.lines == [] and gone.fallback is False
    # A checker that ran and read nothing has no read to fall back on either: no layer, not fallback_used.
    reads["rapidocr"] = checker()
    assert reads["rapidocr"].status == "empty"
    blank = pick(reads, fallback="checker")
    assert blank.chosen is None and blank.lines == [] and blank.fallback is False
    assert "no layer" in blank.reason


def _microprint() -> dict[str, EngineRead]:
    """A Tesseract read of 1,900 Thai characters, mostly background microprint, against a Vision
    read of 600 real ones (all made up). Pieces:
      a: real text both readers read;  b: real text only Vision read;  c: a real block Vision dropped;
      m: a printed line Vision read once; Tesseract's microprint junk repeats fragments of it.
    RapidOCR, the checker, read a, part of b, part of c and m.
    """
    g = _Gen(0)
    a, b, c, m = g.fresh(166), g.fresh(394), g.fresh(120), g.fresh(40)
    fragments = [m[0:14], m[12:26], m[24:38], m[36:40], m[0:5]]
    junk_total = 1900 - len(a) - len(c) - sum(map(len, fragments))
    per, left, junk = junk_total // len(fragments), junk_total, []
    for i, frag in enumerate(fragments):
        n = per if i < len(fragments) - 1 else left
        left -= n
        junk += [frag, g.fresh(n)]
    return {"vision": reader("vision", [a, b, m]), "tesseract": reader("tesseract", [a, c, *junk]),
            "rapidocr": checker([a, b[:97], c[:28], m])}


def test_microprint_regression_both_forms():
    reads = _microprint()
    assert thai_chars(reads["vision"]) == 600 and thai_chars(reads["tesseract"]) == 1900
    # Memo §1's form: Vision and Tesseract only corroborate each other, and the junk wins.
    memo = pick(reads, params(corroborators=()))
    assert memo.scores == {"vision": 202.0, "tesseract": 205.0}
    assert memo.chosen == "tesseract"
    # With RapidOCR as a non-competing corroborator, Vision's real text is backed and it wins.
    fixed = pick(reads, params(corroborators=("rapidocr",)))
    assert fixed.scores == {"vision": 299.0, "tesseract": 231.0}
    assert fixed.chosen == "vision" and fixed.lines == reads["vision"].lines
    for p in (memo, fixed):
        assert p.fallback is False and p.warnings == [] and no_thai(p.reason)


# --------------------------------------------------------------------------- edges


def test_an_empty_reader_does_not_take_the_page_from_one_with_text():
    text = _Gen(8).fresh(80)
    reads = {"vision": reader("vision"), "tesseract": reader("tesseract", [text]), "rapidocr": checker([text])}
    assert reads["vision"].status == "empty"
    p = pick(reads, params(corroborators=("rapidocr",)))
    assert p.chosen == "tesseract" and p.fallback is False


def test_every_reader_empty_keeps_an_empty_read_not_a_failure():
    reads = {"vision": reader("vision"), "tesseract": reader("tesseract"), "rapidocr": checker([_Gen(9).fresh(50)])}
    for fallback in ("none", "checker"):
        p = pick(reads, fallback=fallback)
        assert p.chosen == "vision" and p.lines == [] and p.fallback is False


def test_deterministic_and_order_free():
    reads = _microprint()
    shuffled = dict(reversed(list(reads.items())))
    for corr in ((), ("rapidocr",)):
        assert pick(reads, params(corroborators=corr)) == pick(shuffled, params(corroborators=corr))


def test_readers_outside_tie_order_rank_after_it_by_use_id():
    text = _Gen(10).fresh(60)
    reads = {u: reader(u, [text]) for u in ("zeta", "beta", "vision")}
    p = RULE_V1.pick(reads, RULE_V1.PARAMS.validate({"script": "Thai", "tie_order": ["vision"]},
                                                    ValidationContext(frozenset(reads), frozenset(), {},
                                                                      frozenset({"Thai"}), "t")),
                     fallback="none", checker_id=None)
    assert p.chosen == "vision"
    p = RULE_V1.pick(reads, {"script": "Thai", "corroborators": []}, fallback="none", checker_id=None)
    assert p.chosen == "beta"


def test_latin_script_counts_latin_trigrams_only():
    reads = {"vision": reader("vision", ["sample text 4521 example", "ฃฅฆฌญฎ"]),
             "tesseract": reader("tesseract", ["sample text 9999 example", "ฏฐฑฒณ"])}
    p = RULE_V1.pick(reads, params(script="Latin"), fallback="none", checker_id=None)
    assert p.scores["vision"] == p.scores["tesseract"] > 0
    assert p.chosen == "vision"


def test_params():
    out = RULE_V1.PARAMS.validate({"script": "Thai"}, CTX)
    assert out == {"script": "Thai", "corroborators": []}
    with pytest.raises(PackError) as info:
        RULE_V1.PARAMS.validate({"tie_order": ["vision"]}, CTX)
    assert "script" in str(info.value)
    with pytest.raises(PackError):
        RULE_V1.PARAMS.validate({"script": "Klingon"}, CTX)
    with pytest.raises(PackError):
        RULE_V1.PARAMS.validate({"script": "Thai", "corroborators": ["nobody"]}, CTX)


def test_registered():
    assert rules.get("corroborated_trigrams", 1) is RULE_V1
    assert isinstance(RULE_V1, rules.PickRule)
