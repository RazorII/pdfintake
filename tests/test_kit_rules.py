"""The kit rules decided after the launch-pack trials (DESIGN questions, item 6), and the kit items
that came with them: must-survive counts only what an engine read, the maximin-v2 tie-break, merged
duplicate candidates, ``kit wheel``'s canary-environment check, and what ``kit report`` shows.

Every text, number and pack here is made up; downloads are answered by ``FakeWeb``.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pdfintake import platforms
from pdfintake.kit import bench as B
from pdfintake.kit import choose as CH
from pdfintake.kit import cli
from pdfintake.kit import render as R
from pdfintake.kit import report as REP
from pdfintake.kit import score as S
from pdfintake.kit.render import BenchPage
from pdfintake.langpacks import manifest as M
from pdfintake.langpacks.manifest import MustSurvive
from pdfintake.ocr import engines
from test_kit_cli import RAPID_ONLY, _hu_tess, _packs
from test_kit_render import FONT2_URL, FONT_URL, fake_pack, register_fake

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PACKS = ROOT / "tests" / "fixtures" / "packs"
KEY = platforms.current().key
LINUX = "linux-x86_64"
CONDS = ("clean", "scan", "small")
DOUBLE_ACUTE = MustSurvive("double-acute", "őűŐŰ", "word", 0.9)
CANARY_LIBS = ("fpdf2", "uharfbuzz", "Pillow", "pypdfium2", "pikepdf", "fonttools")     # what kit/render.py draws with


def write_canary_env(root: Path, **override: str) -> Path:
    """``root/tools/canary-env.txt`` pinning the canary libraries at the versions this environment
    runs (the repository's own file pins the maintainer's), so ``kit wheel`` draws in a test;
    ``override`` pins one library (by its name in ``CANARY_LIBS``) at another version."""
    from importlib import metadata

    lines = ["# made up for a test: the versions this test environment runs"]
    for lib in CANARY_LIBS:
        lines.append(f"{lib}=={override.get(lib, metadata.version(lib))}")
    f = Path(root) / "tools" / "canary-env.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = cli.main(list(argv))
    return code, json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------- the must-survive measure

# Made-up Hungarian-looking prose: 12 double-acute characters in 9 words per sentence.
SENTENCE = ("Az őrlő a kőfal mellett áll, és a fűzfa alatt hűvös az erdő. "
            "A szőlő őszi, a gyűrű régi, a fűszer friss.")


def _ocr(text: str) -> Any:
    """A page's OCR as score_page reads it: the picked lines, no fills, no reads to time."""
    return SimpleNamespace(pick=SimpleNamespace(lines=[SimpleNamespace(text=ln) for ln in text.split("\n")]),
                           fills=[], reads=[])


def _set(index: int, pages: list[BenchPage], reads: list[str], *, engines_n: int = 1,
         digits: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict[str, Any]:
    scores = [S.score_page(p, _ocr(r), False, [DOUBLE_ACUTE], "Latin") for p, r in zip(pages, reads)]
    entry = {"index": index, "ok": True, "engines": engines_n, "primary_metric": "words",
             "reader_set": {"set": index}, **B._aggregate(scores, pages, "words", [DOUBLE_ACUTE.name])}
    entry["metrics"]["digits"] = dict(zip(CONDS, digits))
    return entry


def _hu_like_pages(tmp: Path) -> list[BenchPage]:
    """Eight pages per condition, as a launch pack renders them; the small microprint page is the
    5 pt page at 110 dpi that no engine reads."""
    out = []
    for cond in CONDS:
        for k in range(7):
            out.append(BenchPage(f"prose-{k}-{cond}", cond, "prose", tmp / "p.png", " ".join([SENTENCE] * 3), 300.0))
        out.append(BenchPage(f"microprint-0-{cond}", cond, "microprint", tmp / "p.png", " ".join([SENTENCE] * 6), 300.0))
    return out


def _old_word_recall(truth: str, read: str) -> float | None:
    """What kit 1.0 measured on one page: multiset recall of the truth's words holding a group
    character, case kept, whether or not anything on the page was read."""
    def words(t: str) -> Counter[str]:
        return Counter(w for w in (S._EDGE.sub("", x) for x in t.split()) if any(c in "őűŐŰ" for c in w))

    want = words(truth)
    return S._recall(want, words(read)) if want else None


def test_an_unread_small_page_no_longer_sinks_the_group_but_lost_accents_still_fail(tmp_path: Path):
    """hu and pl missed the 0.9 floor on small pages (0.894, 0.884) only because of one 5 pt page no
    engine reads. Now only the words an engine read count: that set passes, and a set that reads
    the words but turns ő into ö still fails."""
    pages = _hu_like_pages(tmp_path)
    # The good set reads every page but the small microprint one, of which it gets one sentence.
    good_reads = [SENTENCE if p.id == "microprint-0-small" else p.truth for p in pages]
    good = _set(0, pages, good_reads)
    old = statistics.fmean(_old_word_recall(p.truth, r) for p, r in zip(pages, good_reads) if p.cond == "small")
    assert round(old, 3) == 0.896                                    # kit 1.0: below the floor
    assert good["must"]["double-acute"] == {"clean": 1.0, "scan": 1.0, "small": 1.0}
    cov = good["must_coverage"]["double-acute"]["small"]
    assert cov == {"truth": 7 * 36 + 72, "read": 7 * 36 + 12, "survived": 7 * 36 + 12, "measurable": True}
    # The accent loser reads every word, small pages too, but writes ö for ő there (ű survives).
    lossy = [r.replace("ő", "ö") if p.cond == "small" else r for p, r in zip(pages, good_reads)]
    loser = _set(1, pages, lossy, engines_n=1)
    assert loser["must"]["double-acute"]["small"] == round(5 / 12, 4)
    assert loser["must_coverage"]["double-acute"]["small"]["read"] == 7 * 36 + 12   # the words were read
    loser["metrics"]["words"] = {c: 0.99 for c in CONDS}               # it would win on the score
    sel = CH.choose(B.BenchResult(KEY, {}, [good, loser]), [DOUBLE_ACUTE])
    assert sel.chosen == 0 and sel.runner_up is None
    assert CH._floor_misses(loser, [DOUBLE_ACUTE]) == [{"name": "double-acute", "cond": "small",
                                                        "value": round(5 / 12, 4), "floor": 0.9}]
    with pytest.raises(CH.FloorFailed):
        CH.choose(B.BenchResult(KEY, {}, [loser]), [DOUBLE_ACUTE])


def test_a_condition_with_too_few_read_characters_is_not_measurable(tmp_path: Path):
    """Fewer than 20 of the group's characters inside read words: no figure, no floor there; the
    counts are kept. A group measurable in no condition passes, and choose names it."""
    pages = _hu_like_pages(tmp_path)
    blind = [("" if p.cond == "small" else p.truth) for p in pages]
    blind[[p.id for p in pages].index("prose-0-small")] = "Az őrlő a kőfal mellett"    # 3 characters read
    s = _set(0, pages, blind)
    assert s["must"]["double-acute"]["small"] is None
    assert s["must_coverage"]["double-acute"]["small"] == {"truth": 7 * 36 + 72, "read": 3, "survived": 3,
                                                           "measurable": False}
    assert CH._floor_misses(s, [DOUBLE_ACUTE]) == [] and CH.unmeasured(s, [DOUBLE_ACUTE]) == []
    nowhere = _set(1, pages, [""] * len(pages))
    assert all(v is None for v in nowhere["must"]["double-acute"].values())
    assert CH.unmeasured(nowhere, [DOUBLE_ACUTE]) == ["double-acute"]


def test_word_matching_folds_marks_and_keeps_the_exact_character():
    def counts(truth: str, read: str, chars: str = "őűŐŰ") -> tuple[int, int, int]:
        return S.must_survive_counts(truth, read, MustSurvive("g", chars, "word", 0.9), "Latin")

    assert counts("Az őrlő és a fűzfa.", "Az orlo es a fűzfa") == (3, 3, 1)      # read, accents lost
    assert counts("Az őrlő és a fűzfa.", "Az örlö es a füzfa") == (3, 3, 0)
    assert counts("Az őrlő és a fűzfa.", "xx yy") == (3, 0, 0)                  # nothing read: nothing counted
    assert counts("Az őrlő és a fűzfa.", "Az ORLŐ és a fűzfa") == (3, 3, 1)     # matched case-folded; Ő isn't ő
    assert counts("Az Őrlő", "az őrlő") == (2, 2, 1)                            # Ő read as ő is lost
    assert counts("Az őrlő", "Az  őrlő") == (2, 2, 2)                # decomposed marks: NFC first
    sharp = MustSurvive("s", "ßäÄ", "word", 0.9)
    assert S.must_survive_counts("Straße und Äpfel", "STRASSE und Apfel", sharp, "Latin") == (2, 2, 0)
    assert S.must_survive_counts("Straße und Äpfel", "Straße und Äpfel", sharp, "Latin") == (2, 2, 2)
    assert S.fold("Łódź Đà Nẵng œuvre Øre ıi Straße") == "lodz da nang oeuvre ore ii strasse"


def test_thai_tone_marks_count_inside_spans_whose_skeleton_was_read():
    tones = MustSurvive("tone-marks", "่้๊๋ำ", "trigram", 0.75)
    truth = "ข้อความตัวอย่าง น้ำ"
    assert S.must_survive_counts(truth, truth, tones, "Thai") == (4, 4, 4)
    # Spacing differs, sara am comes as nikhahit + sara aa, two tone marks are lost: the skeleton holds.
    assert S.must_survive_counts(truth, "ขอความตัว อยาง น้ํา", tones, "Thai") == (4, 4, 2)
    # Too short to be a span an engine read (fewer than MIN_SPAN clusters match in a row).
    assert S.must_survive_counts(truth, "ขอ", tones, "Thai") == (4, 0, 0)
    assert S.must_survive_recall(truth, "", tones, "Thai") is None


def test_twenty_read_characters_make_a_condition_measurable_nineteen_do_not():
    def page(cond: str, read: int, survived: int) -> S.PageScore:
        return S.PageScore("p", cond, None, None, None, must_counts={"g": [40, read, survived]})

    assert B.must_coverage([page("clean", 19, 19)], "g", "clean") == {"truth": 40, "read": 19, "survived": 19,
                                                                     "measurable": False}
    assert B.must_coverage([page("clean", 20, 19)], "g", "clean") == {"truth": 40, "read": 20, "survived": 19,
                                                                     "measurable": True}
    # The pages of a condition add up; another condition's pages don't count.
    pages = [page("clean", 10, 10), page("clean", 9, 9), page("scan", 30, 30)]
    assert B.must_coverage(pages, "g", "clean")["measurable"] is False
    assert B.must_coverage(pages + [page("clean", 1, 0)], "g", "clean")["measurable"] is True


def test_a_conditions_figure_pools_its_pages_rather_than_averaging_them():
    """30 of 30 read characters kept on one page and 0 of 10 on another is 30 / 40 = 0.75 for the
    condition, not the mean of the two page figures (0.5)."""
    scores = [S.PageScore("a", "clean", 1.0, None, None, seconds=1.0, must_counts={"g": [30, 30, 30]}),
              S.PageScore("b", "clean", 1.0, None, None, seconds=1.0, must_counts={"g": [12, 10, 0]})]
    agg = B._aggregate(scores, [], "words", ["g"])
    assert agg["must"]["g"] == {"clean": 0.75, "scan": None, "small": None}
    assert agg["must_coverage"]["g"]["clean"] == {"truth": 42, "read": 40, "survived": 30, "measurable": True}


def test_sara_am_written_with_a_tone_mark_inside_still_counts_as_kept():
    """A reader that writes น้ำ as nikhahit, tone mark, sara aa wrote the ำ: the thai_sara_am
    normaliser takes that spelling back, and so does the measure."""
    tones = MustSurvive("tone-marks", "่้๊๋ำ", "trigram", 0.75)
    truth = "น้ำ ค่ำ ข้อความ"
    assert S.must_survive_counts(truth, truth, tones, "Thai") == (5, 5, 5)
    assert S.must_survive_counts(truth, "นํ้า คํ่า ข้อความ", tones, "Thai") == (5, 5, 5)
    assert S.must_survive_counts(truth, "นํา คํา ขอความ", tones, "Thai") == (5, 5, 2)     # tone marks lost


def test_every_launch_pack_group_character_folds_away():
    """The fold must take every protected character back to its base letter, or a word whose accent
    was lost would never match and the loss would go uncounted (ł, đ, œ and ß don't decompose)."""
    seen = set()
    for f in sorted((ROOT / "packs").glob("*/manifest*.json")):
        for g in json.loads(f.read_text(encoding="utf-8")).get("must_survive") or ():
            for ch in g["chars"]:
                seen.add(ch)
                folded = S.fold(ch)
                assert ch not in folded, (f.parent.name, g["name"], ch)
                assert all(c.isascii() and c.isalpha() or "ก" <= c <= "ฮ" or c == "า"
                           for c in folded), (f.parent.name, ch, folded)
    assert {"ł", "đ", "ß", "œ", "æ", "ő", "ำ"} <= seen


# --------------------------------------------------------------------------- maximin-v2

def _vi_set(index: int, worst: float, digits: tuple[float, float, float], secs: float) -> dict[str, Any]:
    return {"index": index, "ok": True, "engines": 2, "seconds_per_page": secs, "primary_metric": "words",
            "metrics": {"words": {"clean": 0.979, "scan": 0.975, "small": worst}, "digits": dict(zip(CONDS, digits))},
            "must": {}, "must_coverage": {}, "worst_page": 0.3, "reader_set": {"set": index}}


def test_a_near_tie_goes_to_the_better_digits_before_speed():
    """kit trial vi: set 46 won a near-tie (margin 0.0018) on speed while reading digits on scans at
    0.82 against 0.98. maximin-v2 breaks the tie by worst-condition digit recall first."""
    slow = _vi_set(0, 0.8752, (0.9899, 0.9825, 0.9677), 2.9)
    fast = _vi_set(46, 0.8770, (0.9963, 0.8159, 0.8984), 2.4)
    sel = CH.choose(B.BenchResult(KEY, {}, [slow, fast]), [])
    assert (sel.chosen, sel.runner_up, sel.margin, sel.policy) == (0, 46, -0.0018, "maximin-v2")
    assert sel.tie_break == {"window": B.TIE,
                             "chosen": {"index": 0, "score": 0.8752, "digits": 0.9677, "digits_cond": "small",
                                        "engines": 2, "seconds_per_page": 2.9},
                             "lost": [{"index": 46, "score": 0.877, "digits": 0.8159, "digits_cond": "scan",
                                       "engines": 2, "seconds_per_page": 2.4, "decided_by": "digits"}]}
    line = CH.tie_break_line(sel.tie_break)
    assert ("set 0 scored 0.8752 with 2 engines at 2.900 s/page and worst-condition digit recall 0.968 (small)"
            in line)
    assert ("set 46 scored 0.8770 with 2 engines at 2.400 s/page, digits 0.816 (scan), and lost on higher "
            "worst-condition digit recall") in line
    # Outside the window the score still decides, whatever the digits.
    far = _vi_set(46, 0.8900, (0.5, 0.5, 0.5), 2.4)
    assert CH.choose(B.BenchResult(KEY, {}, [slow, far]), []).chosen == 46
    # Equal digits: fewer engines, then speed. The faster set also scored higher, so no order would
    # have picked the slower one: nothing to explain.
    twin_digits = _vi_set(46, 0.8770, (0.9899, 0.9825, 0.9677), 2.4)
    sel = CH.choose(B.BenchResult(KEY, {}, [slow, twin_digits]), [])
    assert sel.chosen == 46 and sel.tie_break is None


def test_the_tie_break_is_reported_when_digits_beat_a_faster_set_that_scored_the_same_or_less():
    """Review of wave E: maximin-v2 picked vi set 45 over set 44 (same score, fewer engines, faster,
    digits on scans 0.82) and set 46 (0.0018 lower, maximin-v1's pick), and nothing said so, because
    only a set that scored higher was reported."""
    chosen = {**_vi_set(0, 0.900, (0.99, 0.99, 0.99), 4.0), "engines": 2}
    lower = {**_vi_set(1, 0.898, (0.99, 0.80, 0.95), 1.0), "engines": 1}
    same = {**_vi_set(2, 0.900, (0.99, 0.81, 0.95), 3.0), "engines": 2}
    beaten = {**_vi_set(3, 0.899, (0.99, 0.70, 0.95), 5.0), "engines": 3}      # loses on every count
    far = {**_vi_set(4, 0.880, (0.99, 0.60, 0.95), 0.5), "engines": 1}         # outside the window
    sel = CH.choose(B.BenchResult(KEY, {}, [chosen, lower, same, beaten, far]), [])
    assert sel.chosen == 0 and sel.ahead == [] and sel.margin >= 0
    assert [(x["index"], x["decided_by"], x["digits"]) for x in sel.tie_break["lost"]] == [(1, "digits", 0.80),
                                                                                         (2, "digits", 0.81)]
    line = CH.tie_break_line(sel.tie_break)
    assert ("set 1 scored 0.8980 with 1 engine at 1.000 s/page, digits 0.800 (scan), and lost on higher "
            "worst-condition digit recall") in line
    assert "set 3 " not in line and "set 4 " not in line
    # The same faster set, now with the same digits: it wins on fewer engines over the higher scorer.
    sel = CH.choose(B.BenchResult(KEY, {}, [chosen, {**lower, "metrics": {**lower["metrics"],
                                                                         "digits": chosen["metrics"]["digits"]}}]), [])
    assert sel.chosen == 1 and [(x["index"], x["decided_by"]) for x in sel.tie_break["lost"]] == [(0, "engines")]


def test_the_digit_floor_uses_only_the_conditions_with_digits():
    """A condition whose pages hold no digits has no digit figure; it neither sinks nor lifts a set."""
    no_small = _vi_set(0, 0.900, (0.99, 0.98, 0.0), 3.0)
    del no_small["metrics"]["digits"]["small"]
    other = _vi_set(1, 0.900, (0.99, 0.90, 0.95), 1.0)
    assert CH.digit_floor(no_small) == (0.98, "scan") and CH.digit_floor(other) == (0.90, "scan")
    sel = CH.choose(B.BenchResult(KEY, {}, [no_small, other]), [])
    assert sel.chosen == 0 and sel.tie_break["lost"][0]["decided_by"] == "digits"
    no_digits = _vi_set(2, 0.900, (0.0, 0.0, 0.0), 1.0)
    no_digits["metrics"]["digits"] = {}
    assert CH.digit_floor(no_digits) == (None, None)


def test_only_maximin_v2_is_known():
    with pytest.raises(Exception) as exc:
        CH.choose(B.BenchResult(KEY, {}, [_vi_set(0, 0.9, (1, 1, 1), 1.0)]), [], policy="maximin-v1")
    assert exc.value.code == "kit_policy" and "kit choose" in exc.value.message and exc.value.exit_code == 2


# --------------------------------------------------------------------------- merged duplicate candidates

def test_candidates_merge_sets_the_loader_reads_as_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """es set 29 was set 0 again: the draft's own set, written with its defaults filled in, and the
    generated one without them (vi: a defaulted timeout_s). They are benched once, under the lower
    index, and the merged index is named."""
    monkeypatch.setattr(engines, "_REGISTRY", {})                    # the fake engines only
    packs, _work, _web = fake_pack(tmp_path, monkeypatch, second=True)
    m, raw, _ = B.load_manifest(packs, "en")
    sets, info = B.candidate_sets(m, KEY, None, raw=raw)
    own = sets[0]["engines"][0]
    assert own["params"] == {"lang": "en", "delay_s": 0.0}             # the loader's defaults, filled in
    # Proposed: 0 own set, 1 fake alone ({"lang": "en"}: set 0 once the loader fills delay_s), 2-3 fake
    # with a fakeb checker, 4 fakeb alone, 5-6 fakeb with a fake checker, 7 the pair.
    assert info["total"] == 8 and info["merged"] == 1 and info["merged_sets"] == {"0": [1]}
    assert info["kept"] == len(sets) == 7 == info["total"] - info["refused_by_loader"] - info["merged"] - info["cut"]
    keys = {B._same_key(s, raw, KEY, None) for s in sets}
    assert len(keys) == len(sets)
    assert sets[1]["engines"][0]["id"] == "fake" and len(sets[1]["engines"]) == 2


def test_a_set_that_differs_only_in_its_use_ids_is_merged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """vi, fr, it, pl, pt and hu: the draft's own Tesseract set (id ``tesseract``) came back as a
    candidate under ``tesseract2``. Ids are names; the sets read alike, so they are benched once."""
    monkeypatch.setattr(engines, "_REGISTRY", {})
    packs, _work, _web = fake_pack(tmp_path, monkeypatch, second=True)
    mfile = packs / "en" / "manifest.json"
    raw = json.loads(mfile.read_text(encoding="utf-8"))
    own = raw["readers"]["default"]
    own["engines"][0]["id"] = "mine"
    own["pick"]["params"]["reader"] = "mine"
    mfile.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    m, raw, _ = B.load_manifest(packs, "en")
    sets, info = B.candidate_sets(m, KEY, None, raw=raw)
    assert info["merged_sets"] == {"0": [1]} and sets[0]["engines"][0]["id"] == "mine"   # the draft's ids stay
    assert all(s["engines"][0]["id"] != "fake" or len(s["engines"]) > 1 or s["pick"]["rule"] != "single"
               for s in sets)                                             # no lone "fake" single set is left
    # Order still counts: fakeb reading with a fake checker isn't fake reading with a fakeb checker.
    pairs = [s for s in sets if len(s["engines"]) == 2 and s["pick"]["rule"] == "single"]
    assert len(pairs) == 2


def _hu_draft(readers: dict[str, Any] | None = None) -> tuple[M.Manifest, dict[str, Any], M.Manifest]:
    """The hu fixture draft (optionally with another default reader set), its JSON and the base pack."""
    base = M.parse((FIXTURE_PACKS / "en" / "manifest.json").read_bytes())
    raw = json.loads((FIXTURE_PACKS / "hu" / "manifest.json").read_text(encoding="utf-8"))
    if readers is not None:
        raw["readers"]["default"] = readers
    return M.parse(json.dumps(raw).encode("utf-8"), base=base), raw, base


def test_a_set_listing_its_models_in_another_order_is_merged():
    """vi set 0 and set 14 listed the RapidOCR models in another order. A use's model list is a set
    as far as a run goes, so they are one set; without the sort, eight of the nine launch packs'
    merges disappear."""
    listed = RAPID_ONLY["engines"][0]["models"]
    for models in (list(listed), list(reversed(listed))):           # one of the two isn't the kit's order
        own = json.loads(json.dumps(RAPID_ONLY))
        own["engines"][0]["models"] = models
        m, raw, base = _hu_draft(own)
        sets, info = B.candidate_sets(m, LINUX, base, raw=raw)
        assert info["merged"] == 1 and list(info["merged_sets"]) == ["0"], (models, info)
        assert sets[0]["engines"][0]["models"] == models               # the draft's set as written


def _two_uses(t_id: str = "tesseract", r_id: str = "rapidocr", *, rule: str, role: str = "reader",
              **params: Any) -> dict[str, Any]:
    tess = {**_hu_tess()["engines"][0], "id": t_id}
    rapid = {**RAPID_ONLY["engines"][0], "id": r_id, "role": role}
    return {"engines": [tess, rapid], "pick": {"rule": rule, "version": 1, "params": params}, "fallback": "none"}


def test_sets_that_differ_only_in_a_pick_parameter_stay_apart():
    _m, raw, base = _hu_draft()

    def pwf(**extra: Any) -> dict[str, Any]:
        return _two_uses(rule="primary_with_floor", role="checker", primary="tesseract", checker="rapidocr", **extra)

    def key(rs: dict[str, Any]) -> str:
        B.parse_reader_set(raw, rs, LINUX, base)                       # loads: the key is the loader's reading
        return B._same_key(rs, raw, LINUX, base)

    assert key(pwf()) == key(pwf(floor=0.5))                          # the default, written out or not
    assert key(pwf()) != key(pwf(floor=0.6))
    assert key(_two_uses("a", "b", rule="primary_with_floor", role="checker", primary="a", checker="b")) == key(pwf())


def test_corroborated_sets_whose_ids_sort_differently_are_not_merged():
    """corroborated_trigrams ranks the readers its tie_order leaves out by use id. Renaming the ids by
    place must not hide that one set puts Tesseract first in a tie and the other RapidOCR."""
    _m, raw, base = _hu_draft()

    def corr(t_id: str, r_id: str, tie: list[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"script": "Latin", "corroborators": []}
        if tie is not None:
            params["tie_order"] = tie
        return _two_uses(t_id, r_id, rule="corroborated_trigrams", **params)

    def key(rs: dict[str, Any]) -> str:
        B.parse_reader_set(raw, rs, LINUX, base)
        return B._same_key(rs, raw, LINUX, base)

    assert key(corr("ta", "tb")) != key(corr("zz", "aa"))             # "ta" < "tb", but "aa" < "zz"
    assert key(corr("ta", "tb")) == key(corr("zz", "aa", ["zz", "aa"]))
    assert key(corr("ta", "tb")) == key(corr("tess", "rapid", ["tess"]))
    assert key(corr("zz", "aa")) == key(corr("ta", "tb", ["tb", "ta"]))


def test_bench_choose_and_report_name_the_merged_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521", "another example line"], second=True)
    common = ["--packs", str(packs), "--work", str(work), "--json"]
    for argv in (["corpus", "en", *common, "--wikipedia", "Q1", "--font", FONT_URL, "--font", FONT2_URL,
                  "--pages", "prose=1,form=0,microprint=0,mixed_script=0"], ["render", "en", *common]):
        assert _run(capsys, "kit", *argv)[0] == 0
    code, out = _run(capsys, "kit", "candidates", "en", *common)
    assert code == 0 and out["candidates"]["merged"] >= 1, out
    merged = out["candidates"]["merged_sets"]["0"]
    assert out["sets"][0]["merged"] == merged and all("merged" not in s for s in out["sets"][1:] if
                                                       str(s["index"]) not in out["candidates"]["merged_sets"])
    code, out = _run(capsys, "kit", "bench", "en", *common, "--offline")
    assert code == 0 and out["sets"][0]["merged"] == merged, out
    bench = json.loads((packs / "en" / "bench.json").read_text(encoding="utf-8"))
    assert bench["kit"] == B.KIT_VERSION and bench["platforms"][KEY]["sets"][0]["merged"] == merged
    code, out = _run(capsys, "kit", "choose", "en", *common, "--override", f"{KEY}=0", "--override-reason",
                     "Made-up reason for the test.")
    assert code == 0 and out["selections"][KEY]["chosen_merged"] == merged, out
    body = REP.pr_body(packs / "en")
    assert f"reads with set 0 of its bench (also proposed as {', '.join(f'set {i}' for i in merged)} before merging)" in body


def test_a_candidates_list_from_the_old_kit_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]):
    packs, work, _web = fake_pack(tmp_path, monkeypatch)
    (work / "en").mkdir(parents=True, exist_ok=True)
    m, _raw, _ = B.load_manifest(packs, "en")
    old = {"platform": KEY, "sets": [B.reader_set_json(m.readers_for(KEY))],
           "candidates": {"total": 1, "refused_by_loader": 0, "cut": 0, "cut_by_rule": {}, "kept": 1, "max_per_rule": 40}}
    (work / "en" / f"candidates-{KEY}.json").write_text(json.dumps(old), encoding="utf-8")
    pages = work / "en" / "pages"
    pages.mkdir(parents=True)
    (pages / R.PAGES_FILE).write_text(json.dumps({"seed": 0, "pages": []}), encoding="utf-8")
    code, out = _run(capsys, "kit", "bench", "en", "--packs", str(packs), "--work", str(work), "--offline")
    assert code == 2 and out["error"]["code"] == "kit_candidates_stale", out
    assert "pdfintake kit candidates en" in out["error"]["message"]


# --------------------------------------------------------------------------- the policy in choose and verify

def _bench_file(pdir: Path, sets: list[dict[str, Any]], *, kit: str = B.KIT_VERSION) -> None:
    (pdir / B.BENCH_FILE).write_text(json.dumps({"kit": kit, "corpus": "none", "platforms": {
        KEY: {"platform": KEY, "engine_versions": {}, "sets": sets}}}), encoding="utf-8")


def _hu_set(pdir: Path, index: int, rs: dict[str, Any], *, worst: float = 0.9, digits=(1.0, 0.99, 0.98),
            coverage: dict[str, Any] | None = None, merged: list[int] | None = None) -> dict[str, Any]:
    base = M.parse((pdir.parent / "en" / "manifest.json").read_bytes())
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    cov = coverage or {"double-acute": {c: {"truth": 120, "read": 118, "survived": 118, "measurable": True}
                                        for c in CONDS}}
    out = {"index": index, "ok": True, "engines": len(rs["engines"]), "primary_metric": "words",
           "seconds_per_page": 1.0 + index, "worst_page": 0.4,
           "reader_set": B.reader_set_json(B.parse_reader_set(raw, rs, KEY, base)),
           "metrics": {"words": {"clean": 0.97, "scan": 0.96, "small": worst}, "digits": dict(zip(CONDS, digits))},
           "must": {name: {c: (round(x["survived"] / x["read"], 4) if x["measurable"] else None) for c, x in row.items()}
                    for name, row in cov.items()},
           "must_coverage": cov}
    if merged:
        out["merged"] = merged
    return out


def test_choose_refuses_a_kit_1_0_bench_and_a_claim_from_another_policy(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    _bench_file(pdir, [_hu_set(pdir, 0, _hu_tess())], kit="1.0")
    code, out = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 2 and out["error"]["code"] == "kit_bench_stale" and "kit bench hu" in out["error"]["message"], out
    # A platform the draft claims under maximin-v1 that this bench.json doesn't cover can't stay.
    _bench_file(pdir, [_hu_set(pdir, 0, _hu_tess())])
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    other = next(p.key for p in platforms.PLATFORMS if p.key != KEY)
    raw["bench"]["platforms"] = {other: {"metrics": {}, "worst_page": 0.5, "seconds_per_page": 1.0,
                                         "selection": {"policy": "maximin-v1", "margin": 0.0}}}
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    code, out = _run(capsys, "kit", "choose", "hu", "--packs", str(packs))
    assert code == 2 and out["error"]["code"] == "kit_policy", out
    assert "maximin-v1" in out["error"]["message"] and "pdfintake kit choose hu" in out["error"]["message"]


def test_verify_refuses_a_manifest_chosen_under_another_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    pdir = packs / "hu"
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    from pdfintake.kit import choose as CHM

    assert CHM.sync_notice(pdir, raw, packs) == "written"
    claim = {"metrics": {"words": [0.97, 0.96, 0.9]}, "worst_page": 0.4, "seconds_per_page": 1.0,
             "selection": {"policy": "maximin-v1", "margin": 0.0}}
    raw["bench"]["platforms"] = {KEY: claim}
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    code, out = _run(capsys, "kit", "verify", "hu", "--packs", str(packs), "--work", str(tmp_path / "w"))
    assert code == 1 and out["checks"]["manifest"] == "failed", out
    want = f"manifest: {KEY}: the selection was made under policy 'maximin-v1', and this kit chooses with maximin-v2 only"
    assert any(p.startswith(want) and "pdfintake kit choose hu" in p for p in out["problems"]), out["problems"]
    claim["selection"]["policy"] = CH.POLICY
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    code, out = _run(capsys, "kit", "verify", "hu", "--packs", str(packs), "--work", str(tmp_path / "w"))
    assert code == 0 and out["checks"]["manifest"] == "ok", out


def test_a_group_measurable_in_only_some_conditions_gets_no_claimed_triple():
    """The manifest claims ``[clean, scan, small]`` or nothing: a group not measurable in some
    condition is checked against its floors only when CI re-benches."""
    s = {"metrics": {"words": {"clean": 0.97, "scan": 0.96, "small": 0.9}},
         "must": {"part": {"clean": 0.95, "scan": 0.97, "small": None},
                  "full": {"clean": 1.0, "scan": 0.99, "small": 0.95}}}
    claimed = CH._triples(s)
    assert claimed == {"words": [0.97, 0.96, 0.9], "must-full": [1.0, 0.99, 0.95]}


# --------------------------------------------------------------------------- choose and report: coverage, tie-break

def _report_pack(tmp: Path) -> Path:
    """The hu fixture with a second group the corpus never reached, chosen by a tie-break on digits:
    set 0 (Tesseract, digits 0.982 on scans) over set 1 (RapidOCR, 0.002 higher, digits 0.816)."""
    packs = _packs(tmp)
    pdir = packs / "hu"
    raw = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    raw["must_survive"].append({"name": "acute", "chars": "áé", "unit": "word", "floor": 0.9})
    (pdir / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    nothing = {c: {"truth": 0, "read": 0, "survived": 0, "measurable": False} for c in CONDS}
    cov0 = {"double-acute": {"clean": {"truth": 120, "read": 120, "survived": 120, "measurable": True},
                             "scan": {"truth": 120, "read": 110, "survived": 109, "measurable": True},
                             "small": {"truth": 150, "read": 12, "survived": 12, "measurable": False}},
            "acute": nothing}
    cov1 = {"double-acute": {c: {"truth": 120, "read": 100, "survived": 99, "measurable": True} for c in CONDS},
            "acute": nothing}
    _bench_file(pdir, [_hu_set(pdir, 0, _hu_tess(), worst=0.875, digits=(0.99, 0.982, 0.985), coverage=cov0,
                               merged=[3]),
                       _hu_set(pdir, 1, RAPID_ONLY, worst=0.877, digits=(0.996, 0.816, 0.898), coverage=cov1)])
    return pdir


def test_choose_reports_coverage_the_tie_break_and_a_group_measured_nowhere(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture[str]):
    pdir = _report_pack(tmp_path)
    code, out = _run(capsys, "kit", "choose", "hu", "--packs", str(pdir.parent))
    assert code == 0, out
    sel = out["selections"][KEY]
    assert sel["chosen"] == 0 and sel["chosen_merged"] == [3] and sel["policy"] == "maximin-v2"
    assert sel["must_coverage"]["double-acute"]["small"] == {"truth": 150, "read": 12, "survived": 12,
                                                             "measurable": False}
    assert sel["tie_break"]["lost"][0]["decided_by"] == "digits"
    assert "lost on higher worst-condition digit recall" in sel["tie_break"]["summary"]
    assert "set 1 (rapidocr[rapid-rec-latin])" in sel["tie_break"]["summary"]
    assert out["warnings"] == [f"{KEY}: the must-survive group acute couldn't be measured in any condition (no "
                               "condition had 20 of its characters inside words or spans the chosen set read), so "
                               "its floor wasn't checked"]
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    claim = manifest["bench"]["platforms"][KEY]
    assert claim["selection"]["policy"] == "maximin-v2" and "must-acute" not in claim["metrics"]


def test_report_shows_coverage_per_condition_and_the_digits_that_decided(tmp_path: Path,
                                                                         capsys: pytest.CaptureFixture[str]):
    pdir = _report_pack(tmp_path)
    assert _run(capsys, "kit", "choose", "hu", "--packs", str(pdir.parent))[0] == 0
    body = REP.pr_body(pdir)
    assert f"- `{KEY}` reads with set 0 of its bench (also proposed as set 3 before merging)." in body
    assert ("- Must-survive coverage, `double-acute`: clean 120/120, scan 110/120, small 12/150 (not measurable)"
            in body)
    assert ("- Must-survive coverage, `acute`: clean 0/0 (not measurable), scan 0/0 (not measurable), small 0/0 "
            "(not measurable). **Warning:** `acute` couldn't be measured in any condition, so its floor wasn't "
            "checked on this platform") in body
    assert REP.COVERAGE_NOTE.format(n=S.MIN_MEASURABLE) in body
    tie = next(ln for ln in body.splitlines() if ln.startswith("- Tie-break"))
    assert ("set 0 (tesseract[hun] + rapidocr[rapid-rec-latin]) scored 0.8750 with 2 engines at 1.000 s/page and "
            "worst-condition digit recall 0.982 (scan)") in tie
    assert "set 1 (rapidocr[rapid-rec-latin]) scored 0.8770 with 1 engine at 2.000 s/page, digits 0.816 (scan)" in tie
    assert "digit recall, then fewer engines, then speed" in body                 # the margin note
    # Set 1 a little lower (still within the window): it lost on digits all the same, having fewer
    # engines, and the PR body still says so.
    raw = json.loads((pdir / B.BENCH_FILE).read_text(encoding="utf-8"))
    raw["platforms"][KEY]["sets"][1]["metrics"]["words"]["small"] = 0.873
    (pdir / B.BENCH_FILE).write_text(json.dumps(raw), encoding="utf-8")
    tie = next(ln for ln in REP.pr_body(pdir).splitlines() if ln.startswith("- Tie-break"))
    assert ("set 1 (rapidocr[rapid-rec-latin]) scored 0.8730 with 1 engine at 2.000 s/page, digits 0.816 (scan), "
            "and lost on higher worst-condition digit recall") in tie
    # The score alone decided: no tie-break line.
    raw = json.loads((pdir / B.BENCH_FILE).read_text(encoding="utf-8"))
    raw["platforms"][KEY]["sets"][1]["metrics"]["words"]["small"] = 0.80
    (pdir / B.BENCH_FILE).write_text(json.dumps(raw), encoding="utf-8")
    assert "- Tie-break" not in REP.pr_body(pdir)


# --------------------------------------------------------------------------- kit wheel and tools/canary-env.txt

def test_wheel_refuses_canary_libraries_at_other_versions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]):
    """dev-install.sh and pack-publish draw every canary in an environment built from
    tools/canary-env.txt, so a pin drawn under other versions wouldn't reproduce there."""
    packs, work, _web = fake_pack(tmp_path, monkeypatch)

    def no_draw(*a: Any, **k: Any) -> Any:
        raise AssertionError("drew or downloaded before the canary environment was checked")

    monkeypatch.setattr(R, "render_canary", no_draw)
    monkeypatch.setattr(B, "ensure_files", no_draw)
    argv = ["kit", "wheel", "en", "--packs", str(packs), "--work", str(work), "--offline",
            "--must-find", "example", "--must-find", "4521"]
    write_canary_env(tmp_path, fpdf2="0.0.1")
    code, out = _run(capsys, *argv)
    assert code == 2 and out["error"]["code"] == "kit_canary_env", out
    msg = out["error"]["message"]
    assert "fpdf2" in msg and "0.0.1 pinned" in msg and "tools/canary-env.txt" in msg and "uv pip install -r" in msg
    (tmp_path / "tools" / "canary-env.txt").unlink()
    code, out = _run(capsys, *argv)
    assert code == 2 and out["error"]["code"] == "kit_canary_env" and "is missing" in out["error"]["message"], out
    assert cli.repo_root(packs) == tmp_path.resolve()                 # the file is read beside --packs


def test_a_canary_a_use_misses_leaves_the_manifest_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]):
    """kit wheel pins the canary only when every use reads it. A miss leaves manifest.json byte for
    byte as it was and writes no wheel, and the error names what each use missed."""
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521"])
    write_canary_env(tmp_path)
    mfile, out_dir = packs / "en" / "manifest.json", tmp_path / "dist"
    before = mfile.read_bytes()
    argv = ["kit", "wheel", "en", "--packs", str(packs), "--work", str(work), "--offline", "--out", str(out_dir)]
    code, out = _run(capsys, *argv, "--must-find", "sample", "--must-find", "4521")
    assert code == 1 and out["error"]["code"] == "kit_canary_failed", out
    assert mfile.read_bytes() == before
    assert not out_dir.exists() or list(out_dir.iterdir()) == []
    reads = out["error"]["canary_reads"]
    assert [(r["use"], r["ok"], r["found"], r["missing"]) for r in reads] == [("fake", False, ["4521"], ["sample"])]
    code, out = _run(capsys, *argv, "--must-find", "example", "--must-find", "4521")
    assert code == 0 and all(r["ok"] for r in out["canary_reads"]), out
    assert json.loads(mfile.read_text(encoding="utf-8"))["canary"]["must_find"] == ["example", "4521"]
    assert Path(out["wheel"]).parent == out_dir


def test_kit_wheel_reads_the_canary_with_every_reader_set_the_pack_names(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch,
                                                                        capsys: pytest.CaptureFixture[str]):
    """A run reads the canary with its own platform's set, so a canary the ``default`` set can't read
    breaks the pack on every platform without an entry of its own. vi shipped that way in Wave D: its
    Mac set read the canary, its Linux and Windows set (with a RapidOCR checker) didn't, and nothing
    noticed until a manual probe. kit wheel now reads the canary with each set the pack names."""
    monkeypatch.setattr(engines, "_REGISTRY", {})
    packs, work, _web = fake_pack(tmp_path, monkeypatch, texts=["example 4521"], second=True)
    register_fake(monkeypatch, "fakeb", ["exampel 4521"])              # the other set's engine misspells it
    write_canary_env(tmp_path)
    mfile, out_dir = packs / "en" / "manifest.json", tmp_path / "dist"
    raw = json.loads(mfile.read_text(encoding="utf-8"))
    other = json.loads(json.dumps(raw["readers"]["default"]).replace('"fake"', '"fakeb"'))
    raw["readers"] = {"default": other, KEY: raw["readers"]["default"]}  # this computer reads with fake
    mfile.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    before = mfile.read_bytes()
    argv = ["kit", "wheel", "en", "--packs", str(packs), "--work", str(work), "--offline", "--out", str(out_dir),
            "--must-find", "example", "--must-find", "4521"]
    code, out = _run(capsys, *argv)
    assert code == 1 and out["error"]["code"] == "kit_canary_failed", out
    assert mfile.read_bytes() == before and (not out_dir.exists() or list(out_dir.iterdir()) == [])
    reads = out["error"]["canary_reads"]
    assert reads[0] == {"platform": KEY, "use": "fake", "label": reads[0]["label"], "ok": True,
                        "found": ["example", "4521"], "missing": []}
    assert [(r["use"], r["ok"], r["missing"]) for r in reads[1:]] == [("fakeb", False, ["example"])]
    assert reads[1]["platform"] != KEY and "fakeb" in out["error"]["message"]


def test_the_canary_env_file_compares_names_loosely_and_skips_comments(tmp_path: Path):
    from importlib import metadata

    v = {lib: metadata.version(lib) for lib in CANARY_LIBS}
    f = tmp_path / "tools" / "canary-env.txt"
    f.parent.mkdir()
    f.write_text("# a comment\n\n"
                 f"FPDF2=={v['fpdf2']}  # trailing comment\nUHarfBuzz=={v['uharfbuzz']}\nPILLOW=={v['Pillow']}\n"
                 f"PyPdfium2=={v['pypdfium2']}\npikepdf=={v['pikepdf']}\nFontTools=={v['fonttools']}\n", encoding="utf-8")
    cli.check_canary_env(tmp_path)                                    # passes: nothing raised
    assert cli.canary_env_pins(f)["pillow"] == v["Pillow"]
    with pytest.raises(Exception) as exc:
        cli.check_canary_env(tmp_path, versions=lambda name: None if name == "pikepdf" else v[name])
    assert exc.value.exit_code == 2 and "pikepdf isn't installed" in exc.value.message
    f.write_text(f"fpdf2=={v['fpdf2']}\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:
        cli.check_canary_env(tmp_path)
    assert "uharfbuzz isn't pinned there" in exc.value.message
    f.write_text("fpdf2>=2.0\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:
        cli.check_canary_env(tmp_path)
    assert "line 1 isn't name==version" in exc.value.message
    # What scripts/dev-install.sh refuses, kit wheel refuses too: a second pin, an unknown name.
    f.write_text(f"fpdf2=={v['fpdf2']}\nFPDF2==0.0.1\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:
        cli.canary_env_pins(f)
    assert exc.value.code == "kit_canary_env" and "pins fpdf2 twice (line 2 again)" in exc.value.message
    f.write_text("examplelib==1.0\n", encoding="utf-8")
    with pytest.raises(Exception) as exc:
        cli.canary_env_pins(f)
    assert exc.value.exit_code == 2 and "examplelib isn't a library the canary is drawn with" in exc.value.message


def test_the_repository_pins_every_canary_library():
    """tools/canary-env.txt itself (the maintainer's): every library kit wheel compares is pinned."""
    f = ROOT / cli.CANARY_ENV
    if not f.is_file():
        pytest.skip("tools/canary-env.txt isn't in this checkout yet")
    pins = cli.canary_env_pins(f)
    assert {cli._dist_key(lib) for lib in cli.CANARY_LIBS} <= set(pins)
    assert {cli._dist_key(lib) for lib in CANARY_LIBS} == {cli._dist_key(lib) for lib in cli.CANARY_LIBS}
