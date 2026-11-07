"""How well a read kept a bench page's text: recall of words, trigrams, digit runs and must-survive units.

Every metric is a multiset recall in [0, 1]: of the truth's units, the share the read holds, each
unit counted at most as often as the read has it. Text is NFC-normalised first, and words are
compared case-folded with their edge punctuation removed. Digits in any script count as their ASCII
value. A metric the truth gives nothing to measure (a page with no digits, no must-survive unit) is
None in a ``PageScore``, never a free 1.0.

* **words** for scripts written with spaces between words (``WORD_SCRIPTS``: Latin, Cyrillic, Greek);
* **trigrams** of the page's script for every script (``ocr.scripts.trigrams``, runs joined, so a
  reader's spacing doesn't count against it);
* **digits**: runs of digits;
* **must-survive**: the characters of each ``must_survive`` group that came through, counted only
  where an engine read the text around them (below).

The primary metric of a pack (what ``choose`` maximises) is word recall for spaced scripts and
trigram recall otherwise (DESIGN §4.5 step 6).

**Must-survive counts only what an engine read** (decided 2026-11-06, DESIGN questions item 6). A
page no engine reads loses every character on it, but it lost none of them to OCR: the group's
floor exists to catch an engine that reads the word and drops the accent. So:

* ``word`` groups: the truth's words and the read's words are folded (``fold``: NFD, combining marks
  dropped, case-folded, then ``FOLD_TABLE`` for the letters NFD doesn't take apart, such as ł, đ, ø
  and ß) and aligned (``difflib``). A truth word holding one of the group's characters is *read*
  when it is aligned with a read word of the same folded form; only read words count. Within a
  read word, each of the group's characters *survived* when the read word has exactly that
  character in the same place (the same code point after NFC: ő read as ö, or Ő as ő, is lost).
* ``trigram`` and ``char`` groups (scripts written without spaces, such as Thai): the text is cut
  into clusters (a base character and the marks on it), keeping the characters of the group's
  script, as the trigram metric does. The truth's and the read's *skeletons* (each cluster folded,
  so a tone mark or a dropped vowel sign doesn't break it) are aligned, and a run of at least
  ``MIN_SPAN`` clusters whose skeletons match is a *span the engine read*. Only the group's
  characters inside such spans count; one survived when the read's cluster in that place holds it.

``must_survive_counts`` gives, per page and group, ``(truth, read, survived)``: the group's
characters in the truth, those inside read words or spans, and those that survived. The bench adds
them up per condition (``bench._aggregate``): the condition's figure is survived / read, and a
condition with fewer than ``MIN_MEASURABLE`` read characters is *not measurable* there (None): its
floor doesn't apply there, and ``bench.json`` keeps its counts.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..ocr.rules.primary import WORD_SCRIPTS
from ..ocr.textfix import fix_sara_am

if TYPE_CHECKING:
    from ..langpacks.manifest import MustSurvive
    from ..types import PageOCR
    from .render import BenchPage

_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_DIGITS = re.compile(r"[0-9]+")

MIN_MEASURABLE = 20     # read characters a condition needs before a group's figure is measured there
MIN_SPAN = 4            # matching skeleton clusters in a row that make a span an engine read (trigram, char)
# Letters NFD doesn't take apart into a base letter and a mark, folded by hand (after case-folding).
# The launch packs' groups need ł, đ, ß, æ and œ; Thai sara am folds to sara aa, as if its nikhahit
# were a mark. Every launch pack's group characters fold away (tests/test_kit_rules.py checks it).
FOLD_TABLE: Mapping[str, str] = {"ø": "o", "ł": "l", "đ": "d", "ð": "d", "ß": "ss", "æ": "ae", "œ": "oe",
                                 "ı": "i", "ħ": "h", "ŧ": "t", "ŀ": "l", "ำ": "า"}


@dataclass
class PageScore:
    page: str
    cond: str
    words: float | None
    trigrams: float | None
    digits: float | None
    must: dict[str, float | None] = field(default_factory=dict)
    seconds: float = 0.0
    # Per group: [characters in the truth, inside words or spans an engine read, survived].
    must_counts: dict[str, list[int]] = field(default_factory=dict)


def primary_metric(script: str) -> str:
    """``words`` for a spaced script, ``trigrams`` otherwise."""
    return "words" if script in WORD_SCRIPTS else "trigrams"


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _recall(truth: Counter[str], read: Counter[str]) -> float:
    total = sum(truth.values())
    if total == 0:
        return 1.0
    return sum(min(n, read.get(k, 0)) for k, n in truth.items()) / total


def _word_counts(text: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for tok in _norm(text).split():
        w = _EDGE.sub("", tok).casefold()
        if w:
            out[w] += 1
    return out


def _ascii_digits(text: str) -> str:
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() and not ("0" <= ch <= "9") else ch for ch in text)


def _digit_runs(text: str) -> Counter[str]:
    return Counter(_DIGITS.findall(_ascii_digits(_norm(text))))


def _trigram_counts(text: str, script: str) -> Counter[str]:
    from ..ocr import scripts

    return scripts.trigrams(_norm(text), script)


def word_recall(truth: str, read: str, script: str) -> float:
    """Multiset recall of the truth's words (1.0 when it has none)."""
    return _recall(_word_counts(truth), _word_counts(read))


def trigram_recall(truth: str, read: str, script: str) -> float:
    """Multiset recall of the truth's character trigrams in ``script`` (1.0 when it has none)."""
    return _recall(_trigram_counts(truth, script), _trigram_counts(read, script))


def digit_run_recall(truth: str, read: str) -> float:
    """Multiset recall of the truth's digit runs (1.0 when it has none)."""
    return _recall(_digit_runs(truth), _digit_runs(read))


def chars_script(chars: str) -> str | None:
    """The script most of ``chars`` belong to (None when none is a script this core knows)."""
    from ..ocr import scripts

    counts: Counter[str] = Counter()
    for ch in chars:
        s = scripts.script_of(ch)
        if s is not None and s in scripts.SCRIPTS:
            counts[s] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda t: (-t[1], t[0]))[0][0]


def fold(text: str) -> str:
    """``text`` with its marks gone: NFD, combining marks (Mn, Mc, Me) dropped, case-folded, then
    ``FOLD_TABLE`` for the letters NFD leaves whole. ``fold("Őrlő") == fold("orlo")``."""
    kept = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.category(ch).startswith("M"))
    return "".join(FOLD_TABLE.get(ch, ch) for ch in kept.casefold())


def _prep(text: str) -> str:
    """NFC, with Thai sara am written as nikhahit + sara aa (a tone mark between them or not) put
    back together as the ``thai_sara_am`` normaliser does (NFC doesn't: the trigram metric's NFKC goes
    the other way, which would make sara am two characters here)."""
    return fix_sara_am(_norm(text))


def _clusters(text: str) -> list[str]:
    """``text`` cut into clusters: a base character and the marks after it."""
    out: list[str] = []
    for ch in text:
        if out and unicodedata.category(ch).startswith("M"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _kept(cluster: str, chars: frozenset[str]) -> Counter[str]:
    return Counter(ch for ch in cluster if ch in chars)


def _survived(truth: str, read: str | None, chars: frozenset[str]) -> int:
    """The group's characters in the truth's cluster that the read's cluster in its place holds."""
    if read is None:
        return 0
    have = Counter(read)
    return sum(min(n, have[ch]) for ch, n in _kept(truth, chars).items())


def _placed(word: str) -> list[tuple[int, int, str]]:
    """A word's clusters, each with the span of the folded word it folds to."""
    out = []
    pos = 0
    for c in _clusters(word):
        n = len(fold(c))
        out.append((pos, pos + n, c))
        pos += n
    return out


def _word_counts_for(truth: str, read: str, chars: frozenset[str]) -> tuple[int, int, int]:
    def words(text: str) -> list[str]:
        return [w for w in (_EDGE.sub("", tok) for tok in _prep(text).split()) if w]

    tw, rw = words(truth), words(read)
    total = sum(sum(_kept(w, chars).values()) for w in tw)
    if not total:
        return 0, 0, 0
    got = done = 0
    sm = difflib.SequenceMatcher(None, [fold(w) for w in tw], [fold(w) for w in rw], autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            t = tw[a + k]
            n = sum(_kept(t, chars).values())
            if not n:
                continue
            got += n
            # The folded words are equal, so each cluster of the truth word has its place in the
            # read word: the same span of the folded form (ß and a read "ss" have no common place).
            places = {(lo, hi): c for lo, hi, c in _placed(rw[b + k])}
            done += sum(_survived(c, places.get((lo, hi)), chars) for lo, hi, c in _placed(t))
    return total, got, done


def _span_counts_for(truth: str, read: str, chars: frozenset[str], script: str) -> tuple[int, int, int]:
    from ..ocr import scripts

    sc = scripts.SCRIPTS.get(script)

    def clusters(text: str) -> list[str]:
        return _clusters("".join(ch for ch in _prep(text) if sc is None or sc.counts_for_trigrams(ch)))

    tc, rc = clusters(truth), clusters(read)
    total = sum(sum(_kept(c, chars).values()) for c in tc)
    if not total:
        return 0, 0, 0
    got = done = 0
    sm = difflib.SequenceMatcher(None, [fold(c) for c in tc], [fold(c) for c in rc], autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        if size < MIN_SPAN:
            continue
        for k in range(size):
            n = sum(_kept(tc[a + k], chars).values())
            if n:
                got += n
                done += _survived(tc[a + k], rc[b + k], chars)
    return total, got, done


def must_survive_counts(truth: str, read: str, ms: "MustSurvive", script: str) -> tuple[int, int, int]:
    """``(truth, read, survived)`` for ``ms`` on one page: the group's characters in the truth, those
    inside words (``word``) or spans (``trigram``, ``char``) the engine read, and those that survived.
    See the module docstring."""
    chars = frozenset(_norm(ms.chars))
    if ms.unit == "word":
        return _word_counts_for(truth, read, chars)
    return _span_counts_for(truth, read, chars, chars_script(ms.chars) or script)


def must_survive_recall(truth: str, read: str, ms: "MustSurvive", script: str) -> float | None:
    """The share of the group's read characters that survived; None when the read holds none of
    them (the truth has none, or nothing around them was read)."""
    _total, got, done = must_survive_counts(truth, read, ms, script)
    return done / got if got else None


def read_text(ocr: "PageOCR", fills: bool) -> str:
    """The page's final text: the chosen lines, then the fills when ``fills``."""
    parts = [ln.text for ln in ocr.pick.lines]
    if fills:
        parts += [f.text for f in ocr.fills]
    return "\n".join(parts)


def score_page(page: "BenchPage", ocr: "PageOCR", fills: bool, must: Sequence["MustSurvive"], script: str) -> PageScore:
    """The page's scores against its truth, and the seconds its reads took."""
    text = read_text(ocr, fills)
    truth = page.truth
    words = word_recall(truth, text, script) if script in WORD_SCRIPTS and _word_counts(truth) else None
    tri = trigram_recall(truth, text, script) if _trigram_counts(truth, script) else None
    digits = digit_run_recall(truth, text) if _digit_runs(truth) else None
    seconds = sum(float(r.seconds or 0.0) for r in ocr.reads if not (r.meta or {}).get("pass1"))
    counts = {ms.name: list(must_survive_counts(truth, text, ms, script)) for ms in must}
    figures = {name: (round(c[2] / c[1], 4) if c[1] else None) for name, c in counts.items()}
    return PageScore(page.id, page.cond, words, tri, digits, figures, round(seconds, 4), counts)
