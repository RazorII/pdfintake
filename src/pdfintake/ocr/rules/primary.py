"""Two pick rules that trust one named reader: ``primary_with_floor`` v1 and ``single`` v1.

``primary_with_floor`` keeps the primary reader's read unless the checker shows that the primary
failed to read the page. The test is agreement: the share of the checker's confident words
(``min_len`` letters or more, on lines of confidence ``word_conf`` or more, folded so ``ö`` and
``ő`` compare equal) that the primary read too, counted with multiplicity on the checker's side.
The test only runs once the checker has ``min_words`` such words; on a sparser page the primary is
kept as it is. Below the ``floor`` the reader set's ``fallback`` decides:

* ``"checker"`` keeps the checker's read, named, with ``Pick.fallback`` set;
* ``"none"`` keeps the primary's read and adds ``agreement_low`` to ``Pick.warnings``.

The motivating case is a page whose layout the primary's page segmentation mistakes for a picture:
it reads a handful of words where the checker reads the whole text block.

``single`` keeps its one ``reader``.

Failure semantics, shared by every rule (``_no_reader``): with no reader working,
``fallback="checker"`` keeps the checker's read (``Pick.fallback`` set, so the pipeline adds
``fallback_used``) and ``fallback="none"`` gives the page no layer (``chosen`` is None, so the
pipeline adds ``no_layer_readers_failed``). A read counts as working when its status is ``ok`` or
``empty``: an engine that ran and found no text read the page, it didn't fail.

No document text goes into ``reason``; it names use ids, counts and shares only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from ...params import Param, ParamTable
from ...types import EngineRead, Line, Pick
from ..fills import fold

WORKING: frozenset[str] = frozenset({"ok", "empty"})
WORD_SCRIPTS: frozenset[str] = frozenset({"Latin", "Cyrillic", "Greek"})
_STATUS_WORDS = {"failed": "failed", "timeout": "timed out", "unavailable": "is unavailable"}


def _working(read: EngineRead | None) -> bool:
    return read is not None and read.status in WORKING


def _status_summary(reads: Mapping[str, EngineRead], uses: Sequence[str]) -> str:
    parts = []
    for u in uses:
        read = reads.get(u)
        parts.append(f"{u} {'gave no read' if read is None else _STATUS_WORDS.get(read.status, read.status)}")
    return ", ".join(parts)


def _no_reader(rule: Any, reads: Mapping[str, EngineRead], readers: Sequence[str], *,
               fallback: Literal["checker", "none"], checker_id: str | None,
               scores: dict[str, float] | None = None) -> Pick:
    """The shared answer when none of ``readers`` worked."""
    why = _status_summary(reads, readers) if readers else "the reader set has no reader"
    scores = dict(scores or {})
    if fallback == "checker":
        checker = reads.get(checker_id) if checker_id is not None else None
        if checker is not None and checker.status == "ok":
            return Pick(rule.name, rule.version, checker_id, list(checker.lines), scores, True,
                        f"no reader worked ({why}), so the checker {checker_id}'s read was kept")
        what = "no checker" if checker is None else f"the checker {checker_id} " + \
            ("read nothing" if checker.status == "empty" else _STATUS_WORDS.get(checker.status, checker.status))
        return Pick(rule.name, rule.version, None, [], scores, False,
                    f"no reader worked ({why}) and {what}, so the page gets no layer")
    return Pick(rule.name, rule.version, None, [], scores, False,
                f"no reader worked ({why}), so the page gets no layer")


def _kept(rule: Any, reads: Mapping[str, EngineRead], use_id: str, scores: dict[str, float], reason: str,
          warnings: Sequence[str] = ()) -> Pick:
    return Pick(rule.name, rule.version, use_id, list(reads[use_id].lines), scores, False, reason, list(warnings))


def _words(lines: Sequence[Line], min_len: int, min_conf: float | None) -> list[str]:
    """Folded words of ``min_len`` letters or more, from lines at or above ``min_conf`` (None: every line)."""
    pattern = re.compile(r"[^\W\d_]{%d,}" % min_len)
    out = []
    for ln in lines:
        if min_conf is not None and ln.conf < min_conf:
            continue
        out.extend(fold(w) for w in pattern.findall(ln.text or ""))
    return out


class PrimaryWithFloor:
    """``primary_with_floor`` v1: the primary's read, unless it holds under ``floor`` of the checker's words."""

    name: ClassVar[str] = "primary_with_floor"
    version: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("primary", "use_ref", required=True),
        Param("checker", "use_ref", required=True),
        Param("floor", "float", default=0.5, min=0.0, max=1.0),
        Param("min_words", "int", default=20, min=0, max=100000),
        Param("word_conf", "float", default=0.8, min=0.0, max=1.0),
        Param("min_len", "int", default=4, min=1, max=40),
    )
    # The word test needs spaced words of precomposed letters: ``[^\W\d_]`` cuts unspaced scripts and
    # scripts written with combining marks (Thai, Devanagari) at every mark, so the rule serves only these.
    scripts: ClassVar[frozenset[str] | None] = WORD_SCRIPTS

    def pick(self, reads: Mapping[str, EngineRead], params: Mapping[str, Any], *, fallback: Literal["checker", "none"],
             checker_id: str | None) -> Pick:
        primary, checker = params["primary"], params["checker"]
        floor = float(params.get("floor", 0.5))
        min_words = int(params.get("min_words", 20))
        word_conf = float(params.get("word_conf", 0.8))
        min_len = int(params.get("min_len", 4))
        if not _working(reads.get(primary)):
            return _no_reader(self, reads, [primary], fallback=fallback, checker_id=checker_id)
        check = reads.get(checker)
        if check is None or check.status != "ok" or checker == primary:
            what = "gave no read" if check is None else ("is the primary" if checker == primary
                                                         else f"has status {check.status}")
            return _kept(self, reads, primary, {},
                         f"the checker {checker} {what}, so {primary}'s read was kept without the agreement test")
        theirs = _words(check.lines, min_len, word_conf)
        if not theirs:
            return _kept(self, reads, primary, {},
                         f"the checker {checker} read no confident words, so {primary}'s read was kept")
        mine = set(_words(reads[primary].lines, min_len, None))
        agree = round(sum(w in mine for w in theirs) / len(theirs), 3)
        scores = {primary: agree}
        if len(theirs) < min_words:
            return _kept(self, reads, primary, scores,
                         f"the checker {checker} read {len(theirs)} confident words, under the {min_words} the floor "
                         f"needs, so {primary}'s read was kept (agreement {agree:.2f})")
        if agree >= floor:
            return _kept(self, reads, primary, scores,
                         f"{primary} holds {agree:.2f} of the checker {checker}'s {len(theirs)} confident words, "
                         f"at or above the floor of {floor:.2f}")
        below = (f"{primary} holds only {agree:.2f} of the checker {checker}'s {len(theirs)} confident words, "
                 f"under the floor of {floor:.2f}")
        if fallback == "checker":
            return Pick(self.name, self.version, checker, list(check.lines), scores, True,
                        f"{below}, so {checker}'s read was kept")
        return _kept(self, reads, primary, scores, f"{below}; with no fallback {primary}'s read was kept",
                     ["agreement_low"])


class Single:
    """``single`` v1: the one ``reader``'s read."""

    name: ClassVar[str] = "single"
    version: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("reader", "use_ref", required=True),
    )
    scripts: ClassVar[frozenset[str] | None] = None

    def pick(self, reads: Mapping[str, EngineRead], params: Mapping[str, Any], *, fallback: Literal["checker", "none"],
             checker_id: str | None) -> Pick:
        reader = params["reader"]
        if not _working(reads.get(reader)):
            return _no_reader(self, reads, [reader], fallback=fallback, checker_id=checker_id)
        what = "read nothing" if reads[reader].status == "empty" else "is the only reader"
        return _kept(self, reads, reader, {}, f"{reader} {what}; its read was kept")


FLOOR_V1 = PrimaryWithFloor()
SINGLE_V1 = Single()
