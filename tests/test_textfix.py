"""ocr/textfix.py: the normalisers and the diacritic_word repair (synthetic reads only)."""

from __future__ import annotations

import pytest

from fixtures import build
from pdfintake.errors import PackError
from pdfintake.ocr.textfix import NORMALIZERS, REPAIRS, apply_normalizers, compile_word_pattern, fix_sara_am
from pdfintake.params import ValidationContext
from pdfintake.protocols import Normalizer, Repair
from pdfintake.types import Line, Word

DE_TABLE = {"ä": ["a", "á", "ä"], "Ä": ["A", "Á", "Ä"], "ß": ["ß", "B", "ss", "fs", "8"]}


def box(x: float, y: float, w: float = 600, h: float = 30):
    return build.quad(x, y, x + w, y + h)


def ln(text: str, b, conf: float = 0.9) -> Line:
    return Line(text, conf, b)


HU_OWN = "áéíóúőűÁÉÍÓÚŐŰ"           # letters Hungarian writes and German doesn't


def repair(lines, checker_lines, table=DE_TABLE, status="ok"):
    rep = REPAIRS[("diacritic_word", 1)]
    read = build.engine_read("rapidocr", role="checker", lines=checker_lines, status=status)
    ctx = ValidationContext(frozenset({"tesseract", "rapidocr"}), frozenset(), {}, frozenset(), "repairs[0].params")
    params = rep.PARAMS.validate({"source": "rapidocr", "table": table}, ctx)
    return rep.repair(list(lines), read, params)


DE_CONTEXT = ["und", "der", "die", "das", "für", "von", "ist", "nicht"]


def repair_v2(lines, checker_lines, *, page_languages, never=HU_OWN, languages=("de",), table=DE_TABLE,
              context=None):
    rep = REPAIRS[("diacritic_word", 2)]
    read = build.engine_read("rapidocr", role="checker", lines=checker_lines)
    ctx = ValidationContext(frozenset({"tesseract", "rapidocr"}), frozenset(), {}, frozenset(), "repairs[0].params")
    raw = {"source": "rapidocr", "table": table, "languages": list(languages)}
    if never is not None:
        raw["never"] = never
    if context is not None:
        raw["context"] = list(context)
    params = rep.PARAMS.validate(raw, ctx)
    return rep.repair(list(lines), read, params, languages=list(page_languages))


# --------------------------------------------------------------------------- normalisers

def test_registry_members_follow_the_protocols() -> None:
    assert set(NORMALIZERS) == {"nfc", "thai_sara_am"}
    for name, n in NORMALIZERS.items():
        assert isinstance(n, Normalizer) and n.name == name
    assert set(REPAIRS) == {("diacritic_word", 1), ("diacritic_word", 2)}
    for (name, version), r in REPAIRS.items():
        assert isinstance(r, Repair) and (r.name, r.version) == (name, version)


def test_sara_am_both_orders() -> None:
    assert fix_sara_am("จํากัด") == "จำกัด"                      # nikhahit, sara aa
    assert fix_sara_am("นํ้า") == "น้ำ"           # nikhahit, tone, sara aa
    assert fix_sara_am("น้ํา") == "น้ำ"           # tone, nikhahit, sara aa
    assert fix_sara_am("สำนักงาน ํ") == "สำนักงาน ํ"             # composed ำ and a lone nikhahit stay
    assert fix_sara_am("") == ""


def test_apply_normalizers_in_order_and_keeps_boxes() -> None:
    b = box(10, 10)
    lines = [Line("é จํา", 0.8, b, (Word("é", 0.8, b), Word("จํา", 0.7, b)))]
    out = apply_normalizers(["nfc", "thai_sara_am"], lines)
    assert out[0].text == "é จำ"
    assert [w.text for w in out[0].words] == ["é", "จำ"]
    assert out[0].box == b and out[0].conf == 0.8 and out[0].words[1].conf == 0.7
    assert apply_normalizers([], lines) == lines
    with pytest.raises(PackError) as err:
        apply_normalizers(["nfkd"], lines)
    assert err.value.code == "unknown_normalizer"


# --------------------------------------------------------------------------- diacritic_word

def test_diacritic_word_repairs() -> None:
    reader = [ln("Wir pflegen das Gewáchshaus auBerhalb der Saison", box(10, 10)),
              ln("Bäckerin Doe und Backerin Roe", box(10, 40)),
              ln("Hinweis gemáß Seite 4, Szőlőszemek", box(10, 70)),
              ln("a feltételeket megtalálja az oldalon", box(10, 100))]
    checker = [ln("Wir pflegen das Gewächshaus außerhalb der Saison", box(12, 12), 0.98),
               ln("Bäckerin Doe und Bäckerin Roe", box(12, 42), 0.98),
               ln("Hinweis gemäß Seite 4, Szölöszemek", build.quad(900, 72, 1500, 102), 0.98),   # another column
               ln("a feltételeket megtaláßa az oldalon", box(12, 102), 0.95)]
    out, counts = repair(reader, checker)
    assert "Gewächshaus" in out[0].text and "außerhalb" in out[0].text
    # a repeated word repairs a different occurrence each time: the right one claims its own partner
    assert out[1].text == "Bäckerin Doe und Bäckerin Roe"
    # a word on another row is not repaired
    assert "gemáß" in out[2].text
    # the checker's ß in a word of another language doesn't replace the reader's word
    assert "megtalálja" in out[3].text
    assert "Szőlőszemek" in out[2].text                     # the reader's own diacritics are untouched
    assert counts == {"fixed": 3, "unpartnered": 2}
    assert [o.box for o in out] == [r.box for r in reader]


def test_diacritic_word_repeated_word_fixed_at_both_places() -> None:
    reader = [ln("Doe Gártnerin und Roe Gártnerin", box(10, 10))]
    checker = [ln("Doe Gärtnerin und Roe Gärtnerin", box(10, 10))]
    out, counts = repair(reader, checker)
    assert out[0].text.count("Gärtnerin") == 2 and counts["fixed"] == 2


def test_diacritic_word_exact_match_claims_its_partner_first() -> None:
    # the correctly read occurrence takes the checker's one word, so the misread one stays
    reader = [ln("Backerin und Bäckerin", box(10, 10))]
    checker = [ln("Bäckerin", box(10, 10))]
    out, counts = repair(reader, checker)
    assert out[0].text == "Backerin und Bäckerin"
    assert counts == {"fixed": 0, "unpartnered": 0}


def test_diacritic_word_skips_a_word_with_an_extra_letter() -> None:
    reader = [ln("die Gaártnerin Doe", box(10, 10))]
    checker = [ln("die Gärtnerin Doe", box(10, 10))]
    out, counts = repair(reader, checker)
    assert out[0].text == "die Gaártnerin Doe"
    assert counts == {"fixed": 0, "unpartnered": 1}


def test_diacritic_word_rejects_hungarian_words_with_eszett() -> None:
    reader = [ln("az ügyfél megtalálja", box(10, 10))]
    checker = [ln("az ügyfél megtaláßa", box(10, 10))]
    out, counts = repair(reader, checker)
    assert out[0].text == "az ügyfél megtalálja" and counts["unpartnered"] == 1


def test_table_values_are_matched_literally() -> None:
    table = {"ä": [".*"]}
    pat = compile_word_pattern("bär", table)
    assert pat.fullmatch("b.*r") and pat.fullmatch("bär")
    assert not pat.fullmatch("bxyzr") and not pat.fullmatch("br")
    reader = [ln("eins bxyzr zwei b.*r", box(10, 10))]
    checker = [ln("bär", box(10, 10))]
    out, counts = repair(reader, checker, table=table)
    assert out[0].text == "eins bxyzr zwei bär" and counts["fixed"] == 1


def test_table_with_regex_metacharacters_everywhere() -> None:
    table = {"ä": ["(", "[a", "\\"], "+": ["?"]}
    pat = compile_word_pattern("ä+", table)
    assert pat.fullmatch("(?") and pat.fullmatch("[a+") and pat.fullmatch("\\?")
    assert not pat.fullmatch("a+")


def test_diacritic_word_needs_a_working_checker() -> None:
    reader = [ln("Gewáchshaus", box(10, 10))]
    out, counts = repair(reader, [], status="empty")
    assert out == reader and counts == {"fixed": 0, "unpartnered": 0}
    rep = REPAIRS[("diacritic_word", 1)]
    out, counts = rep.repair(reader, None, {"source": "rapidocr", "table": DE_TABLE})
    assert out == reader


def test_diacritic_word_keeps_words_aligned_or_drops_them() -> None:
    b = box(10, 10)
    aligned = build.line("das Gewáchshaus Doe", b, with_words=True)
    out, _ = repair([aligned], [ln("Gewächshaus", b)])
    assert [w.text for w in out[0].words] == ["das", "Gewächshaus", "Doe"]
    assert [w.box for w in out[0].words] == [w.box for w in aligned.words]
    unaligned = Line("das Gewáchshaus Doe", 0.9, b, (Word("dasGewáchshaus", 0.9, b),))
    out, _ = repair([unaligned], [ln("Gewächshaus", b)])
    assert out[0].text == "das Gewächshaus Doe" and out[0].words == ()


# --------------------------------------------------------------------------- diacritic_word v2

# A Hungarian page in small print: the checker reads á as ä in Hungarian words, and German words
# the reader wrote with Hungarian letters sit beside them.
HU_READER = [ln("Doe a pályája mellett, az apály idején", box(10, 10)),
             ln("Gártnerin Jane Doe, gemáB Plan", box(10, 40))]
HU_CHECKER = [ln("Doe a pälyája mellett, az apäly idején", box(12, 12)),
              ln("Gärtnerin Jane Doe, gemäß Plan", box(12, 42))]


def test_v2_repairs_german_words_on_a_page_detected_as_german_too() -> None:
    out, counts = repair_v2(HU_READER, HU_CHECKER, page_languages=["hu", "de"])
    assert out[1].text == "Gärtnerin Jane Doe, gemäß Plan"
    # pälyája holds á, a letter German never writes, so it's refused; apäly holds no such letter
    # and is taken: what's left of the risk on a bilingual page in small print
    assert "pályája" in out[0].text and "apäly" in out[0].text
    assert counts == {"fixed": 3, "unpartnered": 0, "refused": 1}


def test_v2_leaves_a_page_detected_as_one_language_alone() -> None:
    for page_languages in (["hu"], [], ["en"]):
        out, counts = repair_v2(HU_READER, HU_CHECKER, page_languages=page_languages)
        assert [o.text for o in out] == [r.text for r in HU_READER]
        assert counts == {"fixed": 0, "unpartnered": 0, "refused": 0}


def test_v2_without_never_letters_is_v1_on_its_pages() -> None:
    v2, c2 = repair_v2(HU_READER, HU_CHECKER, page_languages=["hu", "de"], never=None)
    v1, c1 = repair(HU_READER, HU_CHECKER)
    assert [o.text for o in v2] == [o.text for o in v1] and c2 == c1 and "refused" not in c2


def test_v2_takes_words_only_from_a_line_of_the_other_language() -> None:
    reader = [ln("Doe a pályája mellett, az apály idején", box(10, 10)),
              ln("Die Gártnerin Jane Doe ist gemáB Plan", box(10, 40)),
              ln("Gártnerin", box(10, 70))]
    checker = [ln("Doe a pälyája mellett, az apäly idején", box(12, 12)),
               ln("Die Gärtnerin Jane Doe ist gemäß Plan", box(12, 42)),
               ln("Gärtnerin", box(12, 72))]
    out, counts = repair_v2(reader, checker, page_languages=["hu", "de"], context=DE_CONTEXT)
    assert out[0].text == reader[0].text                    # a Hungarian line: apäly isn't taken
    assert out[1].text == "Die Gärtnerin Jane Doe ist gemäß Plan"
    assert out[2].text == "Gártnerin"                  # a line of one word has no context: left as read
    assert counts == {"fixed": 2, "unpartnered": 0, "refused": 0, "no_context": 3}
    # context words match ignoring case; off the page's languages nothing counts at all
    upper = [ln("DIE Gärtnerin", box(12, 42))]
    out, counts = repair_v2([ln("DIE Gártnerin", box(10, 40))], upper, page_languages=["de"], context=["die"])
    assert out[0].text == "DIE Gärtnerin" and counts["fixed"] == 1
    out, counts = repair_v2(reader, checker, page_languages=["hu"], context=DE_CONTEXT)
    assert counts == {"fixed": 0, "unpartnered": 0, "refused": 0, "no_context": 0}


def test_v2_params() -> None:
    rep = REPAIRS[("diacritic_word", 2)]
    ctx = ValidationContext(frozenset({"rapidocr"}), frozenset(), {}, frozenset(), "repairs[0].params")
    ok = {"source": "rapidocr", "table": DE_TABLE, "languages": ["de"], "never": HU_OWN}
    assert rep.PARAMS.validate(ok, ctx)["never"] == HU_OWN
    assert rep.PARAMS.validate({k: v for k, v in ok.items() if k != "never"}, ctx)["never"] == ""
    assert rep.PARAMS.validate({**ok, "context": ["und", "Die"]}, ctx)["context"] == ["und", "Die"]
    for bad in ({**ok, "languages": []}, {**ok, "languages": ["DE"]}, {**ok, "languages": ["de", "de"]},
                {k: v for k, v in ok.items() if k != "languages"}, {**ok, "never": "a.*"}, {**ok, "never": "a b"},
                {**ok, "never": "1"}, {**ok, "never": "á" * 41}, {**ok, "context": []}, {**ok, "context": ["a b"]},
                {**ok, "context": ["1"]}, {**ok, "context": ["und", "und"]}, {**ok, "context": [f"w{'a' * i}" for i in range(65)]}):
        with pytest.raises(PackError):
            rep.PARAMS.validate(bad, ctx)


def test_diacritic_word_params() -> None:
    rep = REPAIRS[("diacritic_word", 1)]
    ctx = ValidationContext(frozenset({"rapidocr"}), frozenset(), {}, frozenset(), "repairs[0].params")
    with pytest.raises(PackError):
        rep.PARAMS.validate({"source": "nobody", "table": DE_TABLE}, ctx)
    with pytest.raises(PackError):
        rep.PARAMS.validate({"source": "rapidocr", "table": {"ab": ["x"]}}, ctx)
    with pytest.raises(PackError):
        rep.PARAMS.validate({"source": "rapidocr"}, ctx)
