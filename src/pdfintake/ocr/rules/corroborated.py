"""The ``corroborated_trigrams`` v1 pick rule: keep the read with the most corroborated text.

Every working reader competes. A reader's score is the number of its character trigrams in the
pack's ``script`` (``scripts.trigrams``: the script's runs joined, so spacing and line breaks
between readers don't matter) that at least one other source also read, counted with multiplicity
on the reader's side and as a set on the others (``scripts.corroborated``). The other sources are
the other competing readers plus every ``corroborators`` use. A corroborator only lends its read:
it never competes and never wins, however good its read is.

Why corroboration and not the amount of text: a reader that turns a page's background pattern
into text produces many characters of junk, and junk off a pattern is rarely read the same way
twice. Text that a second engine also read is very likely really on the page.

Ties go to the first competitor in ``tie_order``; readers missing from it rank after it, by use id.

Failure semantics (shared with every rule, see ``primary._no_reader``):

* no reader working: ``fallback="checker"`` keeps the checker's read, named; ``"none"`` gives
  the page no layer;
* exactly one reader working: its read is kept as it is, uncorroborated, since there is nothing
  to corroborate it against.

A read is working when its status is ``ok`` or ``empty``. When some working readers read text and
others read nothing, only those with text compete; when every working reader read nothing, the
first in ``tie_order`` is kept (the page holds no text, which is not a failure).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from ...params import Param, ParamTable
from ...types import EngineRead, Pick
from ..scripts import corroborated, trigrams
from .primary import _no_reader, _working


class CorroboratedTrigrams:
    """``corroborated_trigrams`` v1."""

    name: ClassVar[str] = "corroborated_trigrams"
    version: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("script", "script", required=True),
        Param("tie_order", "use_list"),
        Param("corroborators", "use_list", default=()),
    )
    scripts: ClassVar[frozenset[str] | None] = None

    def pick(self, reads: Mapping[str, EngineRead], params: Mapping[str, Any], *, fallback: Literal["checker", "none"],
             checker_id: str | None) -> Pick:
        script = params["script"]
        tie_order = list(params.get("tie_order") or ())
        corroborators = list(params.get("corroborators") or ())
        lenders = set(corroborators)
        competitors = [u for u in reads if reads[u].role == "reader" and u not in lenders]
        ranked = [u for u in tie_order if u in competitors] + sorted(u for u in competitors if u not in tie_order)
        rank = {u: i for i, u in enumerate(ranked)}
        working = [u for u in ranked if _working(reads[u])]
        if not working:
            named = ranked or [u for u in tie_order if u not in lenders]
            return _no_reader(self, reads, named, fallback=fallback, checker_id=checker_id)
        with_text = [u for u in working if reads[u].text().strip()]
        if not with_text:
            first = working[0]
            return Pick(self.name, self.version, first, list(reads[first].lines), {}, False,
                        f"no working reader read any text; {first}'s empty read was kept")
        if len(with_text) == 1:
            only = with_text[0]
            others = [u for u in ranked if u != only]
            why = f" ({', '.join(others)} read nothing or did not work)" if others else ""
            return Pick(self.name, self.version, only, list(reads[only].lines), {}, False,
                        f"{only} was the only reader with a read{why}, so it was kept uncorroborated")
        grams = {u: trigrams(reads[u].text(), script) for u in with_text}
        lent = [u for u in corroborators if u in reads and reads[u].status == "ok"]
        lent_grams: set[str] = set()
        for u in lent:
            lent_grams |= set(trigrams(reads[u].text(), script))
        scores: dict[str, float] = {}
        for u in with_text:
            seen = set(lent_grams)
            for o in with_text:
                if o != u:
                    seen |= set(grams[o])
            scores[u] = float(corroborated(grams[u], seen))
        best = max(with_text, key=lambda u: (scores[u], -rank[u]))
        tied = [u for u in with_text if scores[u] == scores[best]]
        sources = "the other readers" + (f" and {', '.join(lent)}" if lent else "")
        reason = f"most {script} trigrams corroborated by {sources} ({int(scores[best])})"
        if len(tied) > 1:
            reason += f"; tied with {', '.join(u for u in tied if u != best)}, so tie_order decided"
        return Pick(self.name, self.version, best, list(reads[best].lines), scores, False, reason)


RULE_V1 = CorroboratedTrigrams()
