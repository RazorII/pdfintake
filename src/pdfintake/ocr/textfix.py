"""Text normalisers and repairs: registered core code that packs name, never import.

Normalisers (``NORMALIZERS``) run on every engine's read in the pipeline, in the order the pack
lists them, so every engine gets the same treatment:

* ``nfc``: Unicode NFC.
* ``thai_sara_am``: Thai sara am put back together (``fix_sara_am``). Some engines write ำ
  (U+0E33) as nikhahit + sara aa (U+0E4D U+0E32), sometimes with a tone mark between them; a
  search for a word with ำ misses that spelling, so ``ํา`` becomes ``ำ`` and ``ํ่า`` becomes ``่ำ``.

Repairs (``REPAIRS``) run on the picked lines with the checker's read:

* ``diacritic_word`` v1: a reader that writes some letters of a word wrongly (a reader trained on
  one language reading a word of another, ä as á, ß as B) gets each such word back from the
  checker, which read it right. The pack's ``table`` maps each letter to the forms the reader
  writes it as. Every value is a literal, compiled with ``re.escape`` only. A checker word holding
  a table letter replaces its partner: a word on a picked line that shares the checker line's text
  row and differs from it only in those letters. Word by word, each picked word claimed once, so a
  repeated word repairs a different occurrence each time; whole lines are never swapped.
* ``diacritic_word`` v2: v1 limited to the pages, lines and words where the other language is
  really there (decided 2026-11-07). It runs only on a page whose detected languages include one
  of ``languages`` (the language the table's letters belong to); it takes words only from a
  checker line holding one of the ``context`` words (that language's function words, so the line
  is a sentence of it); and it never takes a checker word holding any of the ``never`` letters
  (the reader language's own letters, which the other language doesn't write). Each limit stops a
  word of the reader's language that the checker misread with a table letter, as RapidOCR reads
  Hungarian á as ä in tiny print: the page limit on a page of one language, the line limit on a
  line of the reader's language, the letter limit on a word that carries another of its letters.
  Counts: ``fixed``, ``unpartnered``, ``refused`` (words the ``never`` letters kept out) and
  ``no_context`` (words on a line without a ``context`` word).

Repairs take the page's detected languages (ISO 639-1 tags, primary first) as ``languages``;
v1 ignores them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar

from ..errors import PackError
from ..params import Param, ParamTable
from ..protocols import Normalizer, Repair
from ..types import EngineRead, Line, Quad, Word

_SARA_AM = re.compile("\u0E4D([\u0E48-\u0E4B]?)\u0E32")


def fix_sara_am(text: str) -> str:
    """Thai nikhahit + sara aa (with an optional tone mark between them) as the composed sara am."""
    return _SARA_AM.sub("\\1\u0E33", text or "")


class _Nfc:
    name: ClassVar[str] = "nfc"

    def __call__(self, text: str) -> str:
        return unicodedata.normalize("NFC", text or "")


class _ThaiSaraAm:
    name: ClassVar[str] = "thai_sara_am"

    def __call__(self, text: str) -> str:
        return fix_sara_am(text)


NORMALIZERS: Mapping[str, Normalizer] = MappingProxyType({n.name: n for n in (_Nfc(), _ThaiSaraAm())})


def _normalizer(name: str) -> Normalizer:
    try:
        return NORMALIZERS[name]
    except KeyError:
        raise PackError("unknown_normalizer",
                        f"unknown normaliser {name!r}; this version of pdfintake knows {', '.join(NORMALIZERS)}",
                        normalizer=name, known=list(NORMALIZERS)) from None


def apply_normalizers(names: Sequence[str], lines: Sequence[Line]) -> list[Line]:
    """``lines`` with each normaliser applied in order to every line's and word's text; boxes untouched."""
    funcs = [_normalizer(n) for n in names]
    if not funcs:
        return list(lines)

    def norm(text: str) -> str:
        for f in funcs:
            text = f(text)
        return text

    out = []
    for ln in lines:
        words = tuple(Word(norm(w.text), w.conf, w.box) for w in ln.words)
        out.append(Line(norm(ln.text), ln.conf, ln.box, words))
    return out


# --------------------------------------------------------------------------- repairs

_EDGE = re.compile(r"^\W+|\W+$")


def _core(tok: str) -> str:
    """A token without its leading and trailing punctuation."""
    return _EDGE.sub("", tok)


def _rect(box: Quad) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _same_row(a: Quad, b: Quad) -> bool:
    """Boxes that share a text row: they overlap across, and down by half the smaller height."""
    ax0, ay0, ax1, ay1 = _rect(a)
    bx0, by0, bx1, by1 = _rect(b)
    return (min(ax1, bx1) > max(ax0, bx0)
            and min(ay1, by1) - max(ay0, by0) > 0.5 * min(ay1 - ay0, by1 - by0))


def compile_word_pattern(word: str, table: Mapping[str, Sequence[str]]) -> re.Pattern[str]:
    """The forms a reader may have written ``word`` in: each table letter as itself or any of its
    variants, every other character as itself. Every piece goes through ``re.escape``."""
    out = []
    for ch in word:
        variants = table.get(ch)
        if variants:
            alts: list[str] = []
            for v in (ch, *variants):
                if v not in alts:
                    alts.append(v)
            out.append("(?:" + "|".join(re.escape(v) for v in alts) + ")")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out), re.IGNORECASE)


def _diacritic_words(lines: list[Line], checker: EngineRead | None, table: Mapping[str, Sequence[str]],
                     never: str = "", context: frozenset[str] | None = None) -> tuple[list[Line], dict[str, int]]:
    """The word repair both versions share: ``lines`` with partnered words replaced, and counts.

    A checker word holding any character of ``never`` is never taken (counted ``refused``). With
    ``context`` (lower-case words), only a checker line holding one of them gives words; the words
    of any other line are counted ``no_context``."""
    counts = {"fixed": 0, "unpartnered": 0}
    if never:
        counts["refused"] = 0
    if context is not None:
        counts["no_context"] = 0
    if checker is None or checker.status != "ok" or not table:
        return list(lines), counts
    keys = frozenset(table)
    banned = frozenset(never)
    texts = [ln.text for ln in lines]
    claimed: set[tuple[int, int]] = set()
    changed: set[int] = set()
    for cl in checker.lines:
        if not any(ch in keys for ch in cl.text or ""):
            continue
        in_context = context is None or any(_core(w).lower() in context for w in (cl.text or "").split())
        rows = [i for i, ln in enumerate(lines) if _same_row(cl.box, ln.box)]
        for w in (cl.text or "").split():
            core = _core(w)
            if not any(ch in keys for ch in core):
                continue
            if not in_context:
                counts["no_context"] += 1
                continue
            if banned and any(ch in banned for ch in core):
                counts["refused"] += 1
                continue
            pat = compile_word_pattern(core, table)
            hit: tuple[int, int, str, str] | None = None
            for exact in (True, False):
                for li in rows:
                    for k, tok in enumerate(texts[li].split(" ")):
                        c2 = _core(tok)
                        if (li, k) in claimed or not c2:
                            continue
                        if (c2 == core) if exact else bool(pat.fullmatch(c2)):
                            hit = (li, k, tok, c2)
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit is None:
                counts["unpartnered"] += 1
                continue
            li, k, tok, c2 = hit
            claimed.add((li, k))
            if c2 != core:
                toks = texts[li].split(" ")
                toks[k] = tok.replace(c2, core, 1)
                texts[li] = " ".join(toks)
                changed.add(li)
                counts["fixed"] += 1
    out: list[Line] = []
    for i, ln in enumerate(lines):
        if i not in changed:
            out.append(ln)
            continue
        toks = texts[i].split(" ")
        words: tuple[Word, ...] = ()
        if ln.words and len(ln.words) == len(toks):
            words = tuple(Word(t, w.conf, w.box) for t, w in zip(toks, ln.words))
        # Words that can't be aligned with the repaired text are dropped rather than left stale.
        out.append(Line(texts[i], ln.conf, ln.box, words))
    return out, counts


class _DiacriticWordV1:
    name: ClassVar[str] = "diacritic_word"
    version: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("source", "use_ref", required=True),       # the use whose read supplies the right words
        Param("table", "char_table", required=True),     # letter -> the forms the reader writes it as
    )

    def repair(self, lines: list[Line], checker: EngineRead | None, params: Mapping[str, Any], *,
               languages: Sequence[str] = ()) -> tuple[list[Line], dict[str, int]]:
        """The picked lines with partnered words replaced, and counts ``fixed`` and ``unpartnered``."""
        return _diacritic_words(lines, checker, params.get("table") or {})


class _DiacriticWordV2:
    name: ClassVar[str] = "diacritic_word"
    version: ClassVar[int] = 2
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("source", "use_ref", required=True),       # the use whose read supplies the right words
        Param("table", "char_table", required=True),     # letter -> the forms the reader writes it as
        # run only on a page detected as one of these languages (the table letters' language)
        Param("languages", "str_list", required=True, pattern=r"[a-z]{2,3}", min=1, max=4),
        # a checker word holding any of these letters (the reader language's own) is never taken
        Param("never", "str", default="", pattern=r"[^\W\d_]{0,40}"),
        # words come only from a checker line holding one of these (matched ignoring case)
        Param("context", "str_list", pattern=r"[^\W\d_]{1,20}", min=1, max=64),
    )

    def repair(self, lines: list[Line], checker: EngineRead | None, params: Mapping[str, Any], *,
               languages: Sequence[str] = ()) -> tuple[list[Line], dict[str, int]]:
        """v1 on a page detected as one of ``params["languages"]``, from checker lines holding a
        ``context`` word, without checker words that hold a ``never`` letter; on any other page
        the lines come back as they were."""
        never = str(params.get("never") or "")
        words = params.get("context")
        context = frozenset(str(w).lower() for w in words) if words else None
        if not set(params.get("languages") or ()) & set(languages):
            return list(lines), {"fixed": 0, "unpartnered": 0, **({"refused": 0} if never else {}),
                                 **({"no_context": 0} if context is not None else {})}
        return _diacritic_words(lines, checker, params.get("table") or {}, never, context)


REPAIRS: Mapping[tuple[str, int], Repair] = MappingProxyType({
    (r.name, r.version): r for r in (_DiacriticWordV1(), _DiacriticWordV2())
})
