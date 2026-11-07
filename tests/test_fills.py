"""ocr/fills.py: the checker fills, on synthetic cases (made-up text and values only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake.errors import PackError
from pdfintake.ocr.fills import (FILLS, ORDER, _sellers, _substring_edits, digit_groups, flatten, fold, lev, run_fills,
                                 substring_edits)
from pdfintake.params import ValidationContext
from pdfintake.protocols import FillStep
from pdfintake.types import Fill, Line, PageImage

CTX = ValidationContext(frozenset(), frozenset(), {}, frozenset(), "fills")


@dataclass(frozen=True)
class Use:                      # the shape of langpacks.manifest.StepUse
    name: str
    version: int = 1
    params: dict[str, Any] = field(default_factory=dict)


def box(x: float, y: float, w: float = 400, h: float = 30):
    return build.quad(x, y, x + w, y + h)


def L(text: str, y: float = 10, conf: float = 1.0, x: float = 10, w: float = 400) -> Line:
    return Line(text, conf, box(x, y, w))


def checker(*lines: Line, status: str | None = None):
    return build.engine_read("rapidocr", role="checker", lines=list(lines), status=status)


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> PageImage:
    return build.page_image(tmp_path_factory.mktemp("page"), image=build.blank(1000, 1400), dpi=200)


def step(name: str, chosen, chk, page, already=(), **params) -> list[Fill]:
    s = FILLS[(name, 1)]
    return s.fill(chosen, chk, page, s.PARAMS.validate(params, CTX), tuple(already))


# --------------------------------------------------------------------------- the registry

def test_registry() -> None:
    assert set(FILLS) == {(n, 1) for n in ORDER}
    assert ORDER == ("lines", "words", "numbers", "ids", "links")
    for (name, version), s in FILLS.items():
        assert isinstance(s, FillStep) and (s.name, s.version) == (name, version)
    assert FILLS[("words", 1)].scripts_ok == frozenset({"Latin"})
    assert FILLS[("numbers", 1)].scripts_ok is None
    defaults = {n: FILLS[(n, 1)].PARAMS.validate({}, CTX) for n in ORDER}
    assert defaults["numbers"] == {"min_digits": 3, "min_score": 0.5}
    assert defaults["ids"] == {"min_len": 6, "max_len": 16, "min_score": 0.9, "near_edits": 3}
    assert defaults["lines"] == {"min_score": 0.9, "min_alnum": 8, "ratio": 0.45}


# --------------------------------------------------------------------------- helpers

def test_digit_groups() -> None:
    g = digit_groups("date 28/06/2026 total 1,350.00 tel ๐๘๑")
    assert "28062026" in g and "2026" in g                  # a date is one number
    assert "135000" in g                                    # an amount keeps its groups
    assert "081" in g                                       # Thai digits count
    assert "2500" not in digit_groups("250.00")             # 250.00 never counts as 2500
    assert {"250", "25000"} <= digit_groups("250.00")
    assert "350" not in digit_groups("1350")                # no match inside a longer group
    assert "123" in digit_groups("١٢٣")                     # any decimal digits


def test_digit_groups_cap() -> None:
    g = digit_groups(" ".join(["1234567890"] * 6))
    assert max(len(x) for x in g) <= 40
    assert "1234567890" * 3 in g and "1234567890" * 5 not in g


def test_lev_and_substring_edits() -> None:
    assert lev("kitten", "sitting") == 3
    assert lev("", "abc") == 3 and lev("abc", "abc") == 0
    assert lev("kitten", "sitting", cap=2) == 3
    assert lev("abcdef", "a", cap=2) == 3
    assert lev("kitten", "sitting", cap=3) == 3
    assert substring_edits("needle", "a haystack with a needle in it") == 0
    assert substring_edits("neadle", "a haystack with a needle in it") == 1
    assert substring_edits("abc", "") == 3


def test_bit_vector_matches_the_dynamic_programme() -> None:
    import random

    rng = random.Random(7)
    for _ in range(3000):
        a = "".join(rng.choice("abcé") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("abcé") for _ in range(rng.randint(0, 30)))
        assert _substring_edits(a, b) == _sellers(a, b), (a, b)
        stop = rng.choice([1, 2])
        assert (_substring_edits(a, b, stop) <= stop) == (_sellers(a, b) <= stop), (a, b, stop)
    for _ in range(40):                                  # patterns longer than a machine word
        a = "".join(rng.choice("abcd") for _ in range(rng.randint(60, 140)))
        b = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 300)))
        assert _substring_edits(a, b) == _sellers(a, b)


def test_fold_and_flatten() -> None:
    assert fold("Ünïcödé ő") == "unicode o"
    assert flatten("<b>Total</b> &amp; **sum**  \n x") == "Total & sum x"


# --------------------------------------------------------------------------- numbers

def test_numbers(page) -> None:
    chosen = [L("Total 1,350.00", 10), L("Number 4242", 40)]
    chk = checker(L("Total 1,350.00", 10, 0.95),                 # present: nothing added
                  L("Number 4242-42424-24242", 40, 0.93),         # the later groups missing
                  L("REF:4111111111-111", 70, 0.91),              # a long number with a separator
                  L("Tel 202-555-0143", 100, 0.30),               # under the score floor
                  L("Room 12", 130, 0.99))                        # under 3 digits
    fills = step("numbers", chosen, chk, page)
    texts = [f.text for f in fills]
    assert texts == ["42424242424242 4242-42424-24242", "4111111111111 4111111111-111"]
    assert all(f.kind == "number" and f.source == "rapidocr" and f.near is None for f in fills)
    assert fills[0].box == chk.lines[1].box and fills[0].conf == 0.93


def test_numbers_whole_groups(page) -> None:
    # 250.00 never counts as 2500, and 350 doesn't match inside 1350
    fills = step("numbers", [L("Amount 250.00 and 1350")], checker(L("2500 350 25000", 10, 0.9)), page)
    assert [f.text for f in fills] == ["2500", "350"]
    # the same number read twice goes in once
    fills = step("numbers", [L("x")], checker(L("4521", 10), L("4521", 40)), page)
    assert [f.text for f in fills] == ["4521"]
    # a number already in an earlier fill isn't added again
    earlier = [Fill("line", "Invoice 4521", 0.95, box(10, 10), "rapidocr")]
    assert step("numbers", [L("x")], checker(L("4521", 10)), page, already=earlier) == []
    # Thai digits in the checker read compare as ASCII digits
    assert step("numbers", [L("ยอด 4521")], checker(L("ยอด ๔๕๒๑", 10)), page) == []


def test_numbers_params(page) -> None:
    fills = step("numbers", [L("x")], checker(L("12 345", 10, 0.6)), page, min_digits=2, min_score=0.7)
    assert fills == []
    fills = step("numbers", [L("x")], checker(L("12 345", 10, 0.8)), page, min_digits=2, min_score=0.7)
    assert [f.text for f in fills] == ["12", "345"]


# --------------------------------------------------------------------------- ids

def test_ids(page) -> None:
    chosen = [L("Serial No. ZRT8PPLKNV", 10), L("APP.CODE : KOLT57", 40),
              L("ตัวอย่าง 4402-QM581-00771", 70), L("REF NO 1234567890", 100)]
    chk = checker(L("Serial No. ZRT88PLKMW", 10, 0.95),       # three edits off: added, near recorded
                  L("APP.CODE:HDRT57", 40, 0.93),              # three edits off
                  L("a4402-QM581-00771", 70, 0.97),             # a glyph glued on at the edge: present
                  L("REF NO 1234567890", 100, 0.99),          # present
                  L("Lot AB12", 130, 0.98),                     # under 6 characters
                  L("Model X9Y8Z7W6", 160, 0.62),               # under the score floor
                  L("Customer Service", 190, 0.99),             # no digit: not an ID
                  L("Code ZZZZZZZZZZZZZZZZ9", 220, 0.99))       # over 16 characters
    fills = step("ids", chosen, chk, page)
    assert [(f.text, f.near) for f in fills] == [("ZRT88PLKMW", "zrt8pplknv"), ("HDRT57", "kolt57")]
    assert all(f.kind == "id" and f.source == "rapidocr" for f in fills)


def test_ids_far_and_separated(page) -> None:
    # absent at any distance is still added, with no near token; a separated form goes in plain first
    fills = step("ids", [L("nothing similar here")], checker(L("Order ZX-4417-QP", 10)), page)
    assert [(f.text, f.near) for f in fills] == [("ZX4417QP ZX-4417-QP", None)]
    # a near miss recorded within near_edits only
    fills = step("ids", [L("Code AB12CD34")], checker(L("Code AB12CD99", 10)), page, near_edits=1)
    assert [(f.text, f.near) for f in fills] == [("AB12CD99", None)]
    fills = step("ids", [L("Code AB12CD34")], checker(L("Code AB12CD99", 10)), page)
    assert [(f.text, f.near) for f in fills] == [("AB12CD99", "ab12cd34")]
    # length bounds: 6 and 16 folded characters are in, 5 and 17 are out
    for token, n in (("AB123", 0), ("AB1234", 1), ("A" * 15 + "1", 1), ("A" * 16 + "1", 0)):
        assert len(step("ids", [L("x")], checker(L(token, 10)), page)) == n, token
    # an ID with other case or separators than the chosen read's is present
    assert step("ids", [L("ref kv-5904127")], checker(L("KV5904127", 10)), page) == []
    # an ID already in an earlier fill isn't added again
    earlier = [Fill("number", "KV5904127", 0.95, box(10, 10), "rapidocr")]
    assert step("ids", [L("x")], checker(L("KV5904127", 10)), page, already=earlier) == []


# --------------------------------------------------------------------------- lines

def test_lines(page) -> None:
    chosen = [L("ข้อความตัวอย่าง", 10), L("Specimen collection timestamp", 40)]
    chk = checker(L("ข้อความตัวอย่าง", 10, 0.95),                           # no Latin: never a line fill
                  L("Spxcimxn cxllxctixn timxstxmp", 40, 0.95),             # a misread of a line we have
                  L("Sample received at the example laboratory", 70, 0.96),  # a line the chosen read dropped
                  L("Options", 100, 0.99),                                   # under min_alnum
                  L("Another dropped line of example text", 130, 0.85))      # under min_score
    fills = step("lines", chosen, chk, page)
    assert [f.text for f in fills] == ["Sample received at the example laboratory"]
    assert fills[0].kind == "line" and fills[0].box == chk.lines[2].box


def test_lines_ratio(page) -> None:
    chosen = [L("Specimen collection timestamp", 40)]
    misread = L("Spxcimxn cxllxctixn timxstxmp", 40, 0.95)   # 7 edits over 27 characters
    assert step("lines", chosen, checker(misread), page) == []                 # 7 <= 0.45 * 27
    assert [f.text for f in step("lines", chosen, checker(misread), page, ratio=0.2)] == [misread.text]  # 7 > 5.4


def test_lines_half_the_words_present(page) -> None:
    chosen = [L("alpha bravo charlie delta", 10)]
    # two of four 4+ words are present: not more than half missing, so no fill
    assert step("lines", chosen, checker(L("alpha bravo kilos lemur", 10)), page) == []
    # accent-folded copy is present
    assert step("lines", [L("Szőlőszem árnyéka hideg")], checker(L("Szölöszem arnyéka hideg", 10)), page) == []


def test_lines_skip_boxes_already_filled(page) -> None:
    line = L("Sample received at the example laboratory", 70, 0.96)
    earlier = [Fill("line", line.text, line.conf, line.box, "rapidocr")]
    assert step("lines", [L("x")], checker(line), page, already=earlier) == []


# --------------------------------------------------------------------------- words

def test_words(page) -> None:
    chosen = [L("Könyv: Jane Ooe, kiadó: Example, fejezet", 10),
              L("A régi szőlő mellett az erdő szélén a kertésznek", 40)]
    chk = checker(L("Könyv: Jane Doe, kiadó: Example, fejezet", 10, 0.95),
                  L("A régi szölö mellett az erdö szélén a kertésznek", 40, 0.95),   # ö copies fold the same
                  L("Kerekparut Konyvtarepulet Vasutallomas", 70, 0.6))                # a low-score line
    fills = step("words", chosen, chk, page)
    # "Doe" is three letters: under min_letters
    assert fills == []
    chk = checker(L("Könyv: Jane Doeson, kiadó: Example, fejezet", 10, 0.95))
    chosen2 = [L("Könyv: Jane Ooeson, kiadó: Example, fejezet", 10)]
    fills = step("words", chosen2, chk, page)
    assert [(f.kind, f.text) for f in fills] == [("word", "Doeson")]


def test_words_german_folding(page) -> None:
    # á written for ä doesn't count as present: ä folds to ae
    fills = step("words", [L("die Gártnerin")], checker(L("die Gärtnerin", 10, 0.98)), page)
    assert [f.text for f in fills] == ["Gärtnerin"]
    # a word in an earlier fill is present
    earlier = [Fill("line", "Gemäß Fahrplan", 0.92, box(10, 10), "rapidocr")]
    assert step("words", [L("x")], checker(L("Gemäß Fahrplan", 10, 0.92)), page, already=earlier) == []


def test_words_latin_only(page) -> None:
    chk = checker(L("Exampleword missing", 10, 0.95))
    assert run_fills([Use("words")], [L("x")], chk, page, "Thai") == []
    assert [f.text for f in run_fills([Use("words")], [L("x")], chk, page, "Latin")] == ["Exampleword missing"]


# --------------------------------------------------------------------------- links

def test_links(page) -> None:
    chosen = [L("jane(Vexample.com", 10), L("see: https://docs.example.com/setup.", 40)]
    chk = checker(L("jane@example.com", 10, 0.93),
                  L("see: https://docs.example.com/setup.", 40, 0.9),     # present
                  L("www.example.org", 70, 0.3),                           # under the score floor
                  L("Example Street 21.", 100, 0.95))                      # not an address
    fills = step("links", chosen, chk, page)
    assert [(f.kind, f.text) for f in fills] == [("link", "jane@example.com")]
    # trailing punctuation is trimmed
    fills = step("links", [L("x")], checker(L("(www.example.net).", 10)), page)
    assert [f.text for f in fills] == ["www.example.net"]
    fills = step("links", [L("see the site")], checker(L("see www.example.net.", 10)), page)
    assert [f.text for f in fills] == ["www.example.net"]


# --------------------------------------------------------------------------- run_fills

def test_run_fills_order_and_already(page) -> None:
    chosen = [L("Invoice heading only", 10)]
    chk = checker(L("Sample received at laboratory 88451", 40, 0.96),
                  L("Contact jane@example.com ref AB12CD34", 70, 0.95))
    # listed out of order: they run in ORDER, and later steps see earlier fills
    uses = [Use("links"), Use("ids"), Use("numbers"), Use("lines")]
    fills = run_fills(uses, chosen, chk, page, "Latin")
    assert [f.kind for f in fills] == ["line", "line"]
    # without the line fill each value goes in once, by its own step
    uses = [Use("numbers"), Use("ids"), Use("links")]
    fills = run_fills(uses, chosen, chk, page, "Latin")
    assert [(f.kind, f.text) for f in fills] == [("number", "88451"), ("id", "AB12CD34"), ("link", "jane@example.com")]


def test_run_fills_link_not_absorbed_by_an_id_fill(page) -> None:
    # www.example24.com is a letters-and-digits token: the ids step fills it, and links still does
    chk = checker(L("Visit www.example24.com today", 10, 0.95))
    fills = run_fills([Use("ids"), Use("links")], [L("Visit our shop", 10)], chk, page, "Latin")
    assert [(f.kind, f.text) for f in fills] == [("id", "wwwexample24com www.example24.com"),
                                                 ("link", "www.example24.com")]


def test_run_fills_without_a_checker(page) -> None:
    uses = [Use("numbers")]
    assert run_fills(uses, [L("x")], None, page, "Latin") == []
    assert run_fills(uses, [L("x")], checker(L("4521", 10), status="failed"), page, "Latin") == []
    assert run_fills(uses, [L("x")], checker(), page, "Latin") == []
    assert run_fills([], [L("x")], checker(L("4521", 10)), page, "Latin") == []


def test_run_fills_params_and_unknown_steps(page) -> None:
    chk = checker(L("12 345", 10, 0.8))
    assert [f.text for f in run_fills([Use("numbers", params={"min_digits": 2})], [L("x")], chk, page, "Thai")] \
        == ["12", "345"]
    with pytest.raises(PackError):
        run_fills([Use("numbers", params={"min_digits": 0})], [L("x")], chk, page, "Thai")
    with pytest.raises(PackError) as err:
        run_fills([Use("numbers", version=2)], [L("x")], chk, page, "Thai")
    assert err.value.code == "unknown_fill"
    with pytest.raises(PackError):
        run_fills([Use("dots")], [L("x")], None, page, "Thai")


def test_fills_are_deterministic(page) -> None:
    chosen = [L("Total 1,350.00", 10), L("Serial ZRT8PPLKNV", 40)]
    chk = checker(L("Total 1,350.00 and 7713-40562", 10, 0.95), L("Serial ZRT88PLKMW jane@example.com", 40, 0.95),
                  L("Sample received at the example laboratory", 70, 0.96))
    uses = [Use(n) for n in ORDER]
    a = run_fills(uses, chosen, chk, page, "Latin")
    b = run_fills(uses, chosen, chk, page, "Latin")
    assert a == b and a


def test_fills_perf(page) -> None:
    """A dense page: 60 checker lines against a 4,000-character chosen read stays well under a second."""
    import time

    words = [f"word{i:03d}x" for i in range(400)]
    chosen = [L(" ".join(words[i:i + 8]), 10 + i) for i in range(0, 400, 8)]
    chk = checker(*[L(f"Unrelated example line number {i} with token AB{i:02d}CD{i:02d} and 4{i:03d}5", 10 + 30 * i, 0.95)
                    for i in range(60)])
    t = time.perf_counter()
    fills = run_fills([Use(n) for n in ORDER], chosen, chk, page, "Latin")
    elapsed = time.perf_counter() - t
    assert fills and elapsed < 5.0, elapsed
