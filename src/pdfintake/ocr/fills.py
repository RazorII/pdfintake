"""Checker fills: what the checker read confidently that the chosen read lacks, added to the layer.

Every fill compares the chosen lines (the pick, after repairs) with the checker's read and returns
``Fill`` objects with the checker line's box (px of the shared render) and ``source`` set to the
checker's use id. The pipeline adds them to the layer after the chosen lines, so a search finds a
value whichever engine read it right. A wrong fill only matches a search for that wrong string.

The steps run in ``ORDER`` whatever order a pack lists them in. Each step sees, as ``already``,
the earlier fills of the kinds in ``SEES``, so a value a line fill already carries doesn't go in
twice. ``links`` sees only the line fills: an ``ids`` fill can hold a web address
(www.example24.com is a letters-and-digits token), and the address still gets its own link fill.

* ``lines`` v1: a confident checker line the chosen read dropped. More than half its words of 4+
  characters are missing (a word of 6+ characters tolerates 1 edit, 9+ tolerates 2), and the
  whole line is more than ``ratio`` edits per character from anything in the chosen text, so a
  misread copy of a line the chosen read has stays out.
* ``words`` v1 (spaced Latin only): confident checker words of 4+ letters the chosen read lacks
  once both are folded (accents off), one fill per checker line. ä folds to ae and ß to ss, so a
  word written with á for ä doesn't count as present.
* ``numbers`` v1 (any script): a checker number of ``min_digits``+ digits that is not one of the
  chosen read's digit groups. Digit groups compare whole: "250.00" stands for 250 and 25000 but
  never 2500, and 350 is not present inside 1350. A long number (8+ digits) goes in as plain
  digits followed by the checker's own form.
* ``ids`` v1: a checker token that mixes letters and digits (``min_len`` to ``max_len`` once
  folded) and isn't anywhere in the chosen text, at any distance. An ID that is the chosen text
  plus one character at either end is present (a reader glues a neighbouring glyph to a code).
  The closest chosen token within ``near_edits`` is recorded as ``near``.
* ``links`` v1: a checker email or web address the chosen read lacks character for character.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import PackError
from ..params import Param, ParamTable, ValidationContext
from ..protocols import FillStep
from ..types import EngineRead, Fill, Line, PageImage

if TYPE_CHECKING:
    from ..langpacks.manifest import StepUse

ORDER: tuple[str, ...] = ("lines", "words", "numbers", "ids", "links")

# The fill kinds each step is handed as ``already``.
SEES: Mapping[str, frozenset[str]] = MappingProxyType({
    "lines": frozenset(),
    "words": frozenset({"line"}),
    "numbers": frozenset({"line", "word"}),
    "ids": frozenset({"line", "word", "number"}),
    "links": frozenset({"line"}),
})

# Every decimal digit (Thai, Arabic-Indic, Devanagari, ...) read as its ASCII digit.
_DIGITS: dict[int, int] = {}
for _cp in range(0x10000):
    _ch = chr(_cp)
    if unicodedata.category(_ch) == "Nd" and not ("0" <= _ch <= "9"):
        _DIGITS[_cp] = ord("0") + unicodedata.decimal(_ch)
del _cp, _ch

_NUM_RUN = re.compile(r"[0-9]+(?:[ ,./:\-]{1,3}[0-9]+)*")   # a number in the chosen read (spaced groups too)
_NUM = re.compile(r"[0-9]+(?:[-./,:][0-9]+)*")               # a number in the checker's read
_GROUP_CAP = 32
_ID = re.compile(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*")
_WORD = re.compile(r"[^\W\d_]+")
_LINK = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?:https?://|www\.)[\w./?=&%#~+-]+")


# --------------------------------------------------------------------------- text helpers

def _ascii_digits(text: str) -> str:
    return (text or "").translate(_DIGITS)


def digit_groups(text: str) -> set[str]:
    """Every digit string a number in ``text`` can stand for: consecutive digit groups of one
    number, run together. "28/06/2026" gives 28062026, 062026 and 2026; "250.00" gives 250, 25000
    and 00 but never 2500; a run stops growing past 32 digits."""
    out: set[str] = set()
    for m in _NUM_RUN.finditer(_ascii_digits(text)):
        groups = re.findall(r"[0-9]+", m.group())
        for i in range(len(groups)):
            acc = ""
            for g in groups[i:]:
                acc += g
                out.add(acc)
                if len(acc) > _GROUP_CAP:
                    break
    return out


def fold(s: str) -> str:
    """Lowercase, accents and combining marks off: "Ünïcödé" -> "unicode"."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _alnum(s: str) -> str:
    """Latin letters and digits only, folded and run together: "Ref No: A-12" -> "refnoa12"."""
    return re.sub(r"[^a-z0-9]+", "", fold(s))


def _is_id(a: str) -> bool:
    return bool(re.search(r"[a-z]", a)) and bool(re.search(r"[0-9]", a))


def lev(a: str, b: str, cap: int | None = None) -> int:
    """Levenshtein distance; with ``cap``, any distance above it is returned as ``cap + 1``."""
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    row = list(range(len(b) + 1))
    for ca in a:
        diag, row[0] = row[0], row[0] + 1
        for k in range(1, len(b) + 1):
            cur = row[k]
            row[k] = min(cur + 1, row[k - 1] + 1, diag + (ca != b[k - 1]))
            diag = cur
        if cap is not None and min(row) > cap:
            return cap + 1
    d = row[-1]
    return d if cap is None or d <= cap else cap + 1


def _sellers(p: str, t: str, stop: int = 0) -> int:
    """The reference dynamic programme for ``_substring_edits`` (Sellers); kept for the tests."""
    m = len(p)
    col, best = list(range(m + 1)), m
    if best <= stop:
        return best
    for ch in t:
        diag, col[0] = col[0], 0
        for i in range(1, m + 1):
            cur = col[i]
            col[i] = min(cur + 1, col[i - 1] + 1, diag + (p[i - 1] != ch))
            diag = cur
        if col[m] < best:
            best = col[m]
            if best <= stop:
                break
    return best


def _substring_edits(p: str, t: str, stop: int = 0) -> int:
    """Fewest edits that turn ``p`` into some substring of ``t``; returns as soon as the best is at
    most ``stop``.

    The same numbers as Sellers' dynamic programme (``_sellers``), computed with Myers'
    bit-vector recurrence (Hyyro's formulation for searching): one pass over ``t`` with the
    pattern's column held in two integers, so a long chosen text costs a few operations per
    character instead of one per character and pattern position.
    """
    m = len(p)
    best = m
    if best <= stop or not t:
        return best
    peq: dict[str, int] = {}
    for i, c in enumerate(p):
        peq[c] = peq.get(c, 0) | (1 << i)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for c in t:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & mask) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = (ph << 1) & mask                # searching: the top row is 0 everywhere, no carry in
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
        if score < best:
            best = score
            if best <= stop:
                break
    return best


def substring_edits(needle: str, hay: str) -> int:
    """Fewest edits that turn ``needle`` into some substring of ``hay``."""
    return _substring_edits(needle, hay)


def flatten(text: str) -> str:
    """Markup-free searchable text: tags and light markdown stripped, entities decoded, whitespace collapsed."""
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    t = re.sub(r"[#*`|>]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _confident(checker: EngineRead, min_score: float) -> Iterable[Line]:
    return (ln for ln in checker.lines if ln.box is not None and ln.conf >= min_score and ln.text)


# --------------------------------------------------------------------------- the steps

class _LinesV1:
    name: ClassVar[str] = "lines"
    version: ClassVar[int] = 1
    scripts_ok: ClassVar[frozenset[str] | None] = None
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("min_score", "float", default=0.9, min=0.0, max=1.0),
        Param("min_alnum", "int", default=8, min=1, max=200),
        Param("ratio", "float", default=0.45, min=0.0, max=1.0),
    )

    def fill(self, chosen: Sequence[Line], checker: EngineRead, page: PageImage,
             params: Mapping[str, Any], already: Sequence[Fill]) -> list[Fill]:
        min_score, min_alnum, ratio = params["min_score"], params["min_alnum"], params["ratio"]
        hay = _alnum(" ".join(flatten(ln.text) for ln in chosen))
        taken = {f.box for f in already}
        out: list[Fill] = []
        for ln in _confident(checker, min_score):
            line = _alnum(ln.text)
            if len(line) < min_alnum:
                continue
            toks = [w for w in re.findall(r"[a-z0-9]+", fold(ln.text)) if len(w) >= 4]
            if not toks:
                continue

            def tol(w: str) -> int:
                return 0 if len(w) < 6 else 1 if len(w) < 9 else 2

            miss = [w for w in toks if w not in hay
                    and not (tol(w) and _substring_edits(w, hay, tol(w)) <= tol(w))]
            if 2 * len(miss) <= len(toks):
                continue
            if _substring_edits(line, hay) <= ratio * len(line):
                continue                               # a misread of a line the chosen read has
            if ln.box in taken:
                continue
            taken.add(ln.box)
            out.append(Fill("line", ln.text, ln.conf, ln.box, checker.use_id))
        return out


class _WordsV1:
    name: ClassVar[str] = "words"
    version: ClassVar[int] = 1
    scripts_ok: ClassVar[frozenset[str] | None] = frozenset({"Latin"})
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("min_score", "float", default=0.8, min=0.0, max=1.0),
        Param("min_letters", "int", default=4, min=2, max=24),
    )

    @staticmethod
    def _fold(x: str) -> str:
        return fold(x.lower().replace("ä", "ae").replace("ß", "ss"))

    def fill(self, chosen: Sequence[Line], checker: EngineRead, page: PageImage,
             params: Mapping[str, Any], already: Sequence[Fill]) -> list[Fill]:
        min_score, min_letters = params["min_score"], params["min_letters"]
        have = {self._fold(x) for ln in chosen for x in _WORD.findall(ln.text or "")}
        have |= {self._fold(x) for f in already for x in _WORD.findall(f.text)}
        word_re = re.compile(r"[^\W\d_]{%d,}" % min_letters)
        out: list[Fill] = []
        seen: set[str] = set()
        for ln in _confident(checker, min_score):
            miss = []
            for x in word_re.findall(ln.text):
                fx = self._fold(x)
                if fx not in have and fx not in seen:
                    seen.add(fx)
                    miss.append(x)
            if miss:
                out.append(Fill("word", " ".join(miss), ln.conf, ln.box, checker.use_id))
        return out


class _NumbersV1:
    name: ClassVar[str] = "numbers"
    version: ClassVar[int] = 1
    scripts_ok: ClassVar[frozenset[str] | None] = None
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("min_digits", "int", default=3, min=1, max=_GROUP_CAP),
        Param("min_score", "float", default=0.5, min=0.0, max=1.0),
    )

    def fill(self, chosen: Sequence[Line], checker: EngineRead, page: PageImage,
             params: Mapping[str, Any], already: Sequence[Fill]) -> list[Fill]:
        min_digits, min_score = params["min_digits"], params["min_score"]
        have = digit_groups("\n".join([ln.text for ln in chosen] + [f.text for f in already]))
        out: list[Fill] = []
        seen: set[str] = set()
        for ln in _confident(checker, min_score):
            for m in _NUM.finditer(_ascii_digits(ln.text)):
                raw = m.group(0)
                d = re.sub(r"[^0-9]", "", raw)
                if len(d) < min_digits or d in have or d in seen:
                    continue
                seen.add(d)
                text = f"{d} {raw}" if len(d) >= 8 and raw != d else raw
                out.append(Fill("number", text, ln.conf, ln.box, checker.use_id))
        return out


class _IdsV1:
    name: ClassVar[str] = "ids"
    version: ClassVar[int] = 1
    scripts_ok: ClassVar[frozenset[str] | None] = None
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("min_len", "int", default=6, min=2, max=64),
        Param("max_len", "int", default=16, min=2, max=64),
        Param("min_score", "float", default=0.9, min=0.0, max=1.0),
        Param("near_edits", "int", default=3, min=0, max=8),
    )

    def fill(self, chosen: Sequence[Line], checker: EngineRead, page: PageImage,
             params: Mapping[str, Any], already: Sequence[Fill]) -> list[Fill]:
        min_len, max_len = params["min_len"], params["max_len"]
        min_score, near_edits = params["min_score"], params["near_edits"]
        text = _ascii_digits("\n".join(ln.text for ln in chosen))
        have = _alnum(text) + "|" + _alnum(" ".join(f.text for f in already))
        toks = [_alnum(m.group(0)) for m in _ID.finditer(text)]
        out: list[Fill] = []
        seen: set[str] = set()
        for ln in _confident(checker, min_score):
            for m in _ID.finditer(_ascii_digits(ln.text)):
                raw = m.group(0)
                a = _alnum(raw)
                if not (min_len <= len(a) <= max_len) or not _is_id(a):
                    continue
                if a in seen or a in have or a[1:] in have or a[:-1] in have:
                    continue
                seen.add(a)
                k, near = min(((lev(a, b, cap=near_edits), b) for b in toks if abs(len(a) - len(b)) <= near_edits),
                              default=(len(a), None))
                plain = re.sub(r"[^A-Za-z0-9]", "", raw)
                out.append(Fill("id", plain if raw == plain else f"{plain} {raw}", ln.conf, ln.box,
                                checker.use_id, near if near is not None and k <= near_edits else None))
        return out


class _LinksV1:
    name: ClassVar[str] = "links"
    version: ClassVar[int] = 1
    scripts_ok: ClassVar[frozenset[str] | None] = None
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("min_score", "float", default=0.5, min=0.0, max=1.0),
    )

    def fill(self, chosen: Sequence[Line], checker: EngineRead, page: PageImage,
             params: Mapping[str, Any], already: Sequence[Fill]) -> list[Fill]:
        have = "\n".join([ln.text for ln in chosen] + [f.text for f in already])
        out: list[Fill] = []
        seen: set[str] = set()
        for ln in _confident(checker, params["min_score"]):
            for m in _LINK.finditer(ln.text):
                link = m.group(0).rstrip(".,;:)")
                if link in seen or link in have:
                    continue
                seen.add(link)
                out.append(Fill("link", link, ln.conf, ln.box, checker.use_id))
        return out


FILLS: Mapping[tuple[str, int], FillStep] = MappingProxyType({
    (s.name, s.version): s for s in (_LinesV1(), _WordsV1(), _NumbersV1(), _IdsV1(), _LinksV1())
})


def _step(name: str, version: int) -> FillStep:
    try:
        return FILLS[(name, version)]
    except KeyError:
        known = ", ".join(f"{n} v{v}" for n, v in FILLS)
        raise PackError("unknown_fill", f"unknown fill {name!r} v{version!r}; this version of pdfintake knows {known}",
                        fill=name, version=version) from None


_SCRIPT_WORD = re.compile(r"[^\W\d_]{2,}")


def script_words(read: EngineRead, chosen: Sequence[Line], covered: frozenset[str]) -> list[Fill]:
    """Thorough mode's word fill (DESIGN §14.1): every occurrence in ``read`` of a word whose letters
    are all in a script outside ``covered`` (the pack's scripts) beyond the occurrences the chosen text
    already has, one fill per line of ``read`` with that line's box. So a Greek sentence printed four
    times on a German page goes in four times, each where it is printed, as it was measured (the
    words dots read, less the words the layer has, counted). A German word the pack read differently
    doesn't go in: on text the pack reads, the extra words were half junk when measured. Not a pack
    step: thorough mode adds it, whatever the pack lists."""
    from .scripts import INHERITED, script_of

    if read.status != "ok" or not read.lines:
        return []
    have = Counter(w.casefold() for ln in chosen for w in _SCRIPT_WORD.findall(ln.text))
    out: list[Fill] = []
    for ln in read.lines:
        missing: list[str] = []
        for w in _SCRIPT_WORD.findall(ln.text):
            names = {script_of(ch) for ch in w if ch.isalpha()} - {None, INHERITED}
            if not names or names & covered:
                continue
            key = w.casefold()
            if have[key] > 0:                     # the chosen text has this occurrence already
                have[key] -= 1
                continue
            missing.append(w)
        if missing:
            out.append(Fill("word", " ".join(missing), ln.conf, ln.box, read.use_id))
    return out


def missing_numbers(read: EngineRead, chosen: Sequence[Line]) -> list[Fill]:
    """Thorough mode's number fill (DESIGN §14.1): every number in ``read`` with a digit run the chosen
    text lacks, counted (a run the chosen text has twice covers two occurrences), one fill per line of
    ``read`` with that line's box, each number as ``read`` wrote it. Runs of every length count, as
    they did when it was measured: most of what dots gives back is a one- or two-digit run the layer
    misread ("4,5" read as "45"). Not a pack step: thorough mode adds it, whatever the pack lists."""
    if read.status != "ok" or not read.lines:
        return []
    have = Counter(re.findall(r"[0-9]+", _ascii_digits("\n".join(ln.text for ln in chosen))))
    out: list[Fill] = []
    for ln in read.lines:
        missing: list[str] = []
        for m in _NUM.finditer(_ascii_digits(ln.text)):
            lacks = False
            for run in re.findall(r"[0-9]+", m.group()):
                if have[run] > 0:                  # the chosen text has this occurrence already
                    have[run] -= 1
                else:
                    lacks = True
            if lacks:
                missing.append(m.group())
        if missing:
            out.append(Fill("number", " ".join(missing), ln.conf, ln.box, read.use_id))
    return out


def run_fills(steps: Sequence["StepUse"], chosen: Sequence[Line], checker: EngineRead | None,
              page: PageImage, script: str) -> list[Fill]:
    """Every listed fill step, in ``ORDER``, against the checker's read; [] without a usable checker.

    A step whose ``scripts_ok`` excludes the page's ``script`` is skipped (the loader already
    refuses it on a pack whose scripts it can't serve). Each step's params are checked against its
    table, so defaults apply even to params a caller passes raw.
    """
    resolved = []
    for use in steps:
        step = _step(use.name, use.version)
        if use.name not in ORDER:
            raise PackError("unknown_fill", f"fill {use.name!r} has no place in the fill order", fill=use.name)
        resolved.append((ORDER.index(use.name), step, use))
    if checker is None or checker.status != "ok" or not checker.lines:
        return []
    resolved.sort(key=lambda t: t[0])
    out: list[Fill] = []
    for _, step, use in resolved:
        if step.scripts_ok is not None and script not in step.scripts_ok:
            continue
        ctx = ValidationContext(frozenset(), frozenset(), {}, frozenset(), f"fills.{use.name}")
        params = step.PARAMS.validate(dict(getattr(use, "params", None) or {}), ctx)
        sees = SEES[use.name]
        out.extend(step.fill(chosen, checker, page, params, tuple(f for f in out if f.kind in sees)))
    return out
