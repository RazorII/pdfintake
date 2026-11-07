"""Choosing a pack's reader set from its bench, by a fixed policy, and writing the manifest.

**maximin-v2** (DESIGN §4.5 step 6, with the tie-break decided on 2026-11-06, DESIGN questions
item 6):

1. Drop every set that didn't run, and every set with a ``must_survive`` figure below its floor in
   any condition where the group is measurable (the floor is absolute; ``score`` says what a figure
   counts and when a condition isn't measurable).
2. Score each remaining set by its *worst* condition: the minimum over clean, scan and small of the
   pack's primary metric (word recall for spaced scripts, trigram recall otherwise).
3. Keep the highest score. Sets within ``TIE`` (0.005) of it are a tie, broken by the higher
   worst-condition digit recall (``digit_floor``: the minimum over the conditions whose pages hold
   digits), then by fewer engines, then by fewer seconds per page, then by the earlier set. The
   digits come first because the near-ties the kit trials met were decided by speed in favour of a
   set that read numbers worse (vi: margin 0.0018, digits on scans 0.82 against 0.98).

``maximin-v1`` broke a tie by engines and speed alone. The kit knows v2 only: a manifest chosen
under another policy is refused by ``kit verify`` and ``kit choose`` until ``kit choose`` runs again.

**When the tie-break decided**, ``Selection.tie_break`` says on what, with each set's
worst-condition digit recall; ``kit choose`` and ``kit report`` show it. It decided whenever a set
in the tie window that reads differently lost to the chosen set although the score alone, or fewer
engines and speed alone, would have put it first: it scored higher, or it had fewer engines, or as
many and was faster (or as fast and earlier). Those sets are listed under ``lost``. A set in the
window that the chosen set beats on every count isn't listed: no order would have picked it. (The
vi bench of kit 1.0, chosen again under v2: set 45 over set 44, which scored the same with fewer
engines and faster but read digits on scans at 0.82, and over set 46, maximin-v1's pick.)

**The runner-up** is the set the policy would have chosen without the chosen set: the next set in
the policy's order (score, then the tie-break) that reads differently from the chosen one. Two sets
*read alike* when every number the bench measured for them is the same, page by page (seconds
aside): a second copy of one reader (es set 0 and set 29), or a reader with a checker that changed
nothing. They are one reader as far as the bench can tell, so neither is the other's runner-up.

**The margin** is the chosen set's score minus the runner-up's. It is negative when the runner-up
scored higher but lost on the tie-break (within ``TIE`` of the best, with worse digits, more engines
or slower),
or when an override chose a lower-scoring set (the runner-up is then the policy's next set, its own
pick first), and 0.0 when there is no runner-up. ``ahead`` lists every passing set that scored higher than the
chosen set, so a reviewer sees them whatever the runner-up is (kit trials it and vi): under the
policy those lost on the tie-break; under an override they also include any set the override passed
over, the policy's own pick among them.

An override names another set and needs a reason; the reason goes into the manifest, and CI flags
it for review. Nothing here asks: an override is a flag the contributor passes.

**No set passes the floors** (``FloorFailed``, exit 6): the error names the sets that came closest,
each with its reader set and exactly where it misses (must-survive group, condition, value and
floor), closest first, a set that reads alike with one already named counted under it rather than
listed (kit trial pl, item 8).

``write_manifest`` turns the selections into the manifest: the readers (``default`` from a platform
whose chosen set runs everywhere, a platform's own entry only where its choice differs) and the
``bench`` block, which holds numbers only (metric triples are ``[clean, scan, small]``; a must-survive
group not measurable in some condition gets no triple, so CI checks it against its floors only).

``carry_over`` keeps what later commands wrote into an existing ``manifest.json`` when ``kit choose``
runs again (re-bench, re-choose is the usual loop after a CI drift): the ``bench.private`` block
``kit private`` wrote, and the canary ``kit wheel`` pinned, unless the reader sets changed, because
the canary was read with the old ones; then ``kit wheel`` has to run again. Whenever the manifest is
left without a canary, its ``next`` names ``kit wheel``.

``prune_models`` drops the models no chosen reader set uses (never the base pack's), so the wheel
carries only what a run reads, and ``sync_notice`` keeps ``NOTICE-tessdata`` naming exactly the
Tesseract models the manifest keeps (removed when none is left); ``unbenched`` names the supported platforms with no numbers yet,
which CI checks against the must-survive floors only.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import EXIT_KIT_FLOOR, PdfIntakeError, UsageError
from .bench import KIT_VERSION, TIE, BenchResult, set_label

if TYPE_CHECKING:
    from ..langpacks.manifest import MustSurvive

POLICY = "maximin-v2"
CONDS = ("clean", "scan", "small")
DEFAULT_ORDER = ("linux-x86_64", "win-amd64", "linux-aarch64", "darwin-arm64")
MAX_REASON = 300
CLOSEST = 3                     # sets FloorFailed names


class FloorFailed(PdfIntakeError):
    """No reader set keeps every must-survive unit above its floor (kit exit 6)."""

    exit_code = EXIT_KIT_FLOOR


@dataclass
class Selection:
    policy: str
    platform: str
    chosen: int
    runner_up: int | None
    margin: float
    override_reason: str | None
    # For the command's output, never the manifest: the two scores the margin is made of, and the
    # passing sets that scored higher than the chosen one (lost on the tie-break, or passed over by an
    # override).
    chosen_score: float | None = None
    runner_up_score: float | None = None
    ahead: list[int] = field(default_factory=list)
    # When the tie-break decided (a set in the tie window that reads differently lost although it
    # scored higher, or had fewer engines or was faster), what decided it:
    # ``{"window": TIE, "chosen": {...}, "lost": [{..., "decided_by": ...}]}``; None otherwise.
    tie_break: dict[str, Any] | None = None


def policy_message(found: Any, pack_id: str = "<id>") -> str:
    """Why a selection made under another policy is refused, and what to run."""
    return (f"the selection was made under policy {found!r}, and this kit chooses with {POLICY} only: "
            f"run `pdfintake kit choose {pack_id}` again (it says when `kit bench` has to run first)")


def _floor_misses(s: Mapping[str, Any], must: Sequence["MustSurvive"]) -> list[dict[str, Any]]:
    """Where ``s`` misses a must-survive floor: ``{name, cond, value, floor}`` per miss."""
    out = []
    for ms in must:
        row = (s.get("must") or {}).get(ms.name) or {}
        for cond in CONDS:
            v = row.get(cond)
            if v is not None and v < ms.floor - 1e-9:
                out.append({"name": ms.name, "cond": cond, "value": round(float(v), 4), "floor": ms.floor})
    return out


def _floor_problems(s: Mapping[str, Any], must: Sequence["MustSurvive"]) -> list[str]:
    return [f"{x['name']} {x['cond']} {x['value']:.3f} < {x['floor']}" for x in _floor_misses(s, must)]


def reading(s: Mapping[str, Any]) -> str:
    """What the bench measured for ``s``, seconds aside: two sets with the same reading read alike."""
    pages = [{k: v for k, v in p.items() if k != "seconds"} for p in s.get("pages") or () if isinstance(p, Mapping)]
    return json.dumps({"ok": bool(s.get("ok")), "metrics": s.get("metrics"), "must": s.get("must"),
                       "worst_page": s.get("worst_page"), "pages": pages}, sort_keys=True, default=str)


def reads_alike(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether ``a`` and ``b`` read alike: every number the bench measured is the same (seconds aside)."""
    return bool(a.get("ok")) and bool(b.get("ok")) and reading(a) == reading(b)


def set_score(s: Mapping[str, Any]) -> float | None:
    """A set's maximin score: its primary metric's worst condition (None when it has none)."""
    if not s.get("ok"):
        return None
    row = (s.get("metrics") or {}).get(s.get("primary_metric") or "") or {}
    vals = [row.get(c) for c in CONDS if row.get(c) is not None]
    return min(vals) if vals else None


def digit_floor(s: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """A set's worst-condition digit recall and its condition, over the conditions whose pages hold
    digits ((None, None) when no page does)."""
    row = (s.get("metrics") or {}).get("digits") or {}
    vals = [(float(row[c]), c) for c in CONDS if row.get(c) is not None]
    if not vals:
        return None, None
    v, c = min(vals, key=lambda t: t[0])
    return v, c


def _tie_key(s: Mapping[str, Any]) -> tuple[float, int, float, int]:
    """Inside the tie window: higher worst-condition digit recall, fewer engines, faster, earlier."""
    d, _c = digit_floor(s)
    return (-(d if d is not None else -1.0), int(s.get("engines") or 0), float(s.get("seconds_per_page") or 0.0),
            int(s["index"]))


_TIE_STEPS = ("digits", "engines", "speed", "order")


def _decided_by(winner: Mapping[str, Any], loser: Mapping[str, Any]) -> str:
    """The first tie-break step on which ``winner`` beat ``loser``."""
    for step, w, l in zip(_TIE_STEPS, _tie_key(winner), _tie_key(loser)):
        if w != l:
            return step
    return "order"


def _brief(s: Mapping[str, Any]) -> dict[str, Any]:
    d, c = digit_floor(s)
    return {"index": int(s["index"]), "score": set_score(s), "digits": d, "digits_cond": c,
            "engines": int(s.get("engines") or 0), "seconds_per_page": s.get("seconds_per_page")}


def _lost_on_tie_break(passing: Sequence[Mapping[str, Any]], chosen: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The sets the tie-break ordered behind ``chosen`` (the policy's pick) although the score alone,
    or fewer engines and speed alone, would have put them first: in the tie window, reading
    differently, and either scoring higher or ahead on (engines, seconds per page, index)."""
    if not passing:
        return []
    top = max(set_score(s) or 0.0 for s in passing)
    score = set_score(chosen) or 0.0
    plain = _tie_key(chosen)[1:]
    return [s for s in passing if s is not chosen and (set_score(s) or 0.0) >= top - TIE
            and not reads_alike(s, chosen)
            and ((set_score(s) or 0.0) > score + 1e-12 or _tie_key(s)[1:] < plain)]


def _order(cands: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Best first: score, then within TIE of the best remaining, the tie-break (``_tie_key``)."""
    left = sorted(cands, key=lambda s: -(set_score(s) or 0.0))
    out: list[Mapping[str, Any]] = []
    while left:
        top = set_score(left[0]) or 0.0
        tied = [s for s in left if (set_score(s) or 0.0) >= top - TIE]
        pick = min(tied, key=_tie_key)
        out.append(pick)
        left = [s for s in left if s is not pick]
    return out


def choose(result: BenchResult, must: Sequence["MustSurvive"], *, policy: str = POLICY,
           override: int | None = None, override_reason: str | None = None) -> Selection:
    """The ``maximin-v2`` selection for one platform's bench (FloorFailed when no set passes the floors)."""
    if policy != POLICY:
        raise UsageError("kit_policy", policy_message(policy))
    passing = [s for s in result.sets if set_score(s) is not None and not _floor_problems(s, must)]
    if not passing:
        raise _floor_failed(result, must)
    ranked = _order(passing)
    chosen = ranked[0]
    reason = None
    if override is not None:
        reason = (override_reason or "").strip()
        if not reason:
            raise UsageError("kit_override_reason", "an override needs --override-reason")
        if len(reason) > MAX_REASON:
            raise UsageError("kit_override_reason", f"the override reason is longer than {MAX_REASON} characters")
        found = [s for s in result.sets if int(s["index"]) == int(override)]
        if not found:
            raise UsageError("kit_usage", f"there is no set {override} in the {result.platform} bench")
        if found[0] not in passing:
            raise FloorFailed("kit_floor", f"set {override} didn't run or misses a must-survive floor; "
                                           "an override can't go below a floor", platform=result.platform)
        chosen = found[0]
        ranked = [chosen] + [s for s in ranked if s is not chosen]
    runner = next((s for s in ranked[1:] if not reads_alike(s, chosen)), None)
    score = set_score(chosen)
    margin = (score or 0.0) - (set_score(runner) or 0.0) if runner is not None else 0.0
    higher = [s for s in passing if s is not chosen and (set_score(s) or 0.0) > (score or 0.0) + 1e-12]
    ahead = sorted(int(s["index"]) for s in higher)
    tie_break = None
    lost = _lost_on_tie_break(passing, chosen)
    if override is None and lost:
        tie_break = {"window": TIE, "chosen": _brief(chosen),
                     "lost": [{**_brief(s), "decided_by": _decided_by(chosen, s)}
                              for s in sorted(lost, key=lambda s: int(s["index"]))]}
    return Selection(policy, result.platform, int(chosen["index"]),
                     int(runner["index"]) if runner is not None else None, round(margin, 4), reason,
                     chosen_score=score, runner_up_score=set_score(runner) if runner is not None else None,
                     ahead=ahead, tie_break=tie_break)


def tie_break_line(tb: Mapping[str, Any], label: Any = None) -> str:
    """One sentence on what the tie-break decided, with the digit figures; ``label(index)`` may add
    each set's labelled uses."""
    def name(i: int) -> str:
        extra = label(i) if label is not None else None
        return f"set {i} ({extra})" if extra else f"set {i}"

    def digits(b: Mapping[str, Any]) -> str:
        d = b.get("digits")
        return f"{d:.3f} ({b.get('digits_cond')})" if isinstance(d, (int, float)) else "none (no digits on the pages)"

    def cost(b: Mapping[str, Any]) -> str:
        n, secs = b.get("engines"), b.get("seconds_per_page")
        at = f" at {secs:.3f} s/page" if isinstance(secs, (int, float)) else ""
        return f"{n} engine{'' if n == 1 else 's'}{at}"

    c = tb["chosen"]
    parts = []
    for b in tb.get("lost") or ():
        step = b.get("decided_by")
        why = {"digits": "higher worst-condition digit recall", "engines": "fewer engines", "speed": "speed",
               "order": "the earlier set"}.get(str(step), str(step))
        parts.append(f"{name(int(b['index']))} scored {b.get('score'):.4f} with {cost(b)}, digits {digits(b)}, "
                     f"and lost on {why}")
    return (f"Tie-break (within {tb.get('window', TIE)} of the best score): {name(int(c['index']))} scored "
            f"{c.get('score'):.4f} with {cost(c)} and worst-condition digit recall {digits(c)}; "
            + "; ".join(parts) + ".")


def unmeasured(s: Mapping[str, Any], must: Sequence["MustSurvive"]) -> list[str]:
    """The must-survive groups ``s`` couldn't measure in any condition: their floors weren't checked."""
    out = []
    for ms in must:
        cov = (s.get("must_coverage") or {}).get(ms.name) or {}
        if not any(isinstance(c, Mapping) and c.get("measurable") for c in cov.values()):
            out.append(ms.name)
    return out


def closest_sets(result: BenchResult, must: Sequence["MustSurvive"], n: int = CLOSEST) -> list[dict[str, Any]]:
    """The ``n`` sets that came closest to every must-survive floor, closest first: the smallest
    largest shortfall, then the fewest misses, then the higher score, then the earlier set. Each is
    ``{index, reader_set (labels), pick, score, misses: [{name, cond, value, floor}], reads_alike: [indices]}``;
    a set that reads alike with one already named goes under its ``reads_alike``, not into the list."""
    ran = [s for s in result.sets if s.get("ok")]

    def key(s: Mapping[str, Any]) -> tuple[float, int, float, int]:
        misses = _floor_misses(s, must)
        short = max((x["floor"] - x["value"] for x in misses), default=0.0)
        return (round(short, 6), len(misses), -(set_score(s) or 0.0), int(s["index"]))

    out: list[dict[str, Any]] = []
    named: list[Mapping[str, Any]] = []
    for s in sorted(ran, key=key):
        twin = next((i for i, t in enumerate(named) if reads_alike(s, t)), None)
        if twin is not None:
            out[twin]["reads_alike"].append(int(s["index"]))
            continue
        if len(out) >= n:
            continue
        rs = s.get("reader_set") or {}
        named.append(s)
        out.append({"index": int(s["index"]), "reader_set": set_label(rs),
                    "pick": ((rs.get("pick") or {}).get("rule") if isinstance(rs, Mapping) else None),
                    "score": set_score(s), "misses": _floor_misses(s, must), "reads_alike": []})
    return out


def _floor_failed(result: BenchResult, must: Sequence["MustSurvive"]) -> FloorFailed:
    closest = closest_sets(result, must)
    if not closest:
        return FloorFailed("kit_floor", f"no reader set on {result.platform} keeps every must-survive unit above "
                                        "its floor (no set ran)", platform=result.platform, problems=[], closest=[])
    lines = []
    for c in closest:
        where = "; ".join(f"{x['name']} {x['cond']} {x['value']:.4f} < {x['floor']}" for x in c["misses"])
        alike = f" (sets {', '.join(str(i) for i in c['reads_alike'])} read the same)" if c["reads_alike"] else ""
        lines.append(f"set {c['index']} ({c['reader_set']}, {c['pick']}): {where}{alike}")
    return FloorFailed("kit_floor", f"no reader set on {result.platform} keeps every must-survive unit above its "
                                    f"floor; the closest: {' | '.join(lines)}",
                       platform=result.platform, problems=lines, closest=closest)


# --------------------------------------------------------------------------- the manifest

def _set_of(result: BenchResult, index: int) -> Mapping[str, Any]:
    for s in result.sets:
        if int(s["index"]) == index:
            return s
    raise UsageError("kit_usage", f"set {index} isn't in the {result.platform} bench")


def _runs_everywhere(rs: Mapping[str, Any]) -> bool:
    from .. import platforms
    from ..ocr import engines

    supported = {p.key for p in platforms.PLATFORMS if p.supported}
    for u in rs.get("engines") or ():
        try:
            eng = engines.get(u["engine"])
        except PdfIntakeError:
            return False
        if not supported <= set(getattr(eng, "platforms", ())):
            return False
    return True


def _clip(v: Any) -> float:
    try:
        return round(min(1.0, max(0.0, float(v))), 4)
    except (TypeError, ValueError):
        return 0.0


def _triples(s: Mapping[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    rows = dict(s.get("metrics") or {})
    rows.update({f"must-{k}": v for k, v in (s.get("must") or {}).items()})
    for name, row in rows.items():
        vals = [row.get(c) for c in CONDS]
        if all(v is not None for v in vals) and len(out) < 12:
            out[name] = [_clip(v) for v in vals]
    return out


def platform_claim(result: BenchResult, sel: Selection) -> dict[str, Any]:
    """The manifest's ``bench.platforms`` entry for one platform: numbers only."""
    s = _set_of(result, sel.chosen)
    selection: dict[str, Any] = {"policy": sel.policy, "margin": sel.margin}
    if sel.override_reason:
        selection["override_reason"] = sel.override_reason
    return {"metrics": _triples(s), "worst_page": _clip(s.get("worst_page") if s.get("worst_page") is not None else 0.0),
            "seconds_per_page": round(max(0.0, float(s.get("seconds_per_page") or 0.0)), 4), "selection": selection}


def write_manifest(draft: Mapping[str, Any], results: Mapping[str, BenchResult], selections: Mapping[str, Selection],
                   *, corpus: str = "none") -> dict[str, Any]:
    """The manifest for ``draft`` (raw JSON) with the chosen readers and the bench block."""
    chosen = {p: dict(_set_of(results[p], sel.chosen)["reader_set"]) for p, sel in selections.items()}
    order = [p for p in DEFAULT_ORDER if p in chosen] + sorted(p for p in chosen if p not in DEFAULT_ORDER)
    default = next((chosen[p] for p in order if _runs_everywhere(chosen[p])), None)
    old = dict(draft.get("readers") or {})
    if default is None:
        default = old.get("default")
    readers: dict[str, Any] = {"default": default}
    for p in sorted(chosen):
        if chosen[p] != default:
            readers[p] = chosen[p]
    for p, rs in old.items():                    # a platform this bench didn't cover keeps its entry
        if p != "default" and p not in chosen:
            readers[p] = rs
    versions: dict[str, str] = {}
    for r in results.values():
        for k, v in r.engine_versions.items():
            versions.setdefault(k, str(v)[:60])
    bench: dict[str, Any] = {"kit": KIT_VERSION, "corpus": corpus}
    if versions:
        bench["engine_versions"] = dict(list(sorted(versions.items()))[:8])
    old_bench = dict(draft.get("bench") or {})
    plats = dict(old_bench.get("platforms") or {})
    for p, sel in selections.items():
        plats[p] = platform_claim(results[p], sel)
    bench["platforms"] = {p: plats[p] for p in sorted(plats)}
    if "private" in old_bench:
        bench["private"] = old_bench["private"]
    out = dict(draft)
    out["readers"] = readers
    out["bench"] = bench
    return out


WHEEL_NEXT = ("the manifest has no canary yet: run `pdfintake kit wheel {id} --must-find <a word with the "
              "language's special characters> --must-find 4521`")


def carry_over(out: dict[str, Any], existing: Mapping[str, Any] | None) -> dict[str, Any]:
    """Carry ``bench.private`` and the canary from the existing ``manifest.json`` into ``out`` (in
    place); the canary only while the reader sets are unchanged. Returns what happened, for the
    command's output: ``canary`` (``kept``, ``dropped`` or ``none``) and ``next`` steps, which name
    ``kit wheel`` whenever the manifest is left without a canary."""
    pid = out.get("id", "<id>")
    if existing:
        private = (existing.get("bench") or {}).get("private") if isinstance(existing.get("bench"), Mapping) else None
        if private is not None:                   # kit private writes manifest.json, so its copy is the newer
            out["bench"]["private"] = private
    canary = existing.get("canary") if existing else None
    if canary is None:
        if out.get("canary"):
            return {"canary": "kept", "next": []}
        return {"canary": "none", "next": [WHEEL_NEXT.format(id=pid)]}
    if existing.get("readers") == out.get("readers"):      # type: ignore[union-attr]
        out["canary"] = canary
        return {"canary": "kept", "next": []}
    out.pop("canary", None)
    return {"canary": "dropped",
            "next": [f"the reader sets changed, so the canary must be read again: run `pdfintake kit wheel {pid}`"]}


def _tess_langs(use: Mapping[str, Any]) -> set[str]:
    lang = (use.get("params") or {}).get("lang") if use.get("engine") == "tesseract" else None
    return set(lang.split("+")) if isinstance(lang, str) else set()


def prune_models(out: dict[str, Any]) -> list[str]:
    """Drop from ``out`` (in place) every model of the pack that no reader set, on any platform, and
    no detect probe uses; return the dropped files. A model is used when a use lists its key in
    ``models``, or, for a Tesseract model, when a Tesseract use's ``lang`` names its file. The pack's
    wheel carries every model the manifest lists, so an unused one is only a download (kit trial
    es, item 16: a 13.6 MB ``spa.traineddata`` next to a RapidOCR-only reader set).

    The base pack is never pruned: a run uses its OSD model and its orientation voter by key, outside
    any reader set."""
    if out.get("base"):
        return []
    uses: list[Mapping[str, Any]] = []
    for rs in (out.get("readers") or {}).values():
        if isinstance(rs, Mapping):
            uses += [u for u in rs.get("engines") or () if isinstance(u, Mapping)]
    probe = ((out.get("detect") or {}).get("probe") or {}).get("use") if isinstance(out.get("detect"), Mapping) else None
    if isinstance(probe, Mapping):
        uses.append(probe)
    keys = {r for u in uses for r in (u.get("models") or ()) if isinstance(r, str)}
    langs = set().union(*(_tess_langs(u) for u in uses)) if uses else set()
    kept, dropped = [], []
    for m in out.get("models") or ():
        stem = str(m.get("file", "")).rsplit(".", 1)[0]
        used = m.get("key") in keys or (m.get("kind") == "traineddata" and stem in langs)
        (kept if used else dropped).append(m)
    out["models"] = kept
    return [str(m.get("file")) for m in dropped]


def unbenched(results: Mapping[str, BenchResult]) -> list[str]:
    """The supported platform keys ``bench.json`` holds no numbers for."""
    from .. import platforms

    return [p.key for p in platforms.PLATFORMS if p.supported and p.key not in results]


# --------------------------------------------------------------------------- NOTICE-tessdata

NOTICE_TESSDATA = "NOTICE-tessdata"
NOTICE_HEADER = ("Tesseract OCR models from the tesseract-ocr project (https://github.com/tesseract-ocr),\n"
                 "distributed unmodified under the Apache License, Version 2.0:\n")
_NOTICE_FALLBACK = ("The licence text is the Apache License, Version 2.0: "
                    "https://www.apache.org/licenses/LICENSE-2.0\n")
_NOTICE_MODEL_LINE = re.compile(r"  \S+\.traineddata  \S+")


def notice_template(packs_root: Path) -> str | None:
    """The base pack's ``NOTICE-tessdata``: ``packs/en``'s, else the installed English data wheel's."""
    from .bench import BASE_ID

    candidates = [Path(packs_root) / BASE_ID / NOTICE_TESSDATA]
    try:
        from ..langpacks import store

        candidates.append(Path(store.base().root) / NOTICE_TESSDATA)
    except Exception:                                    # no installed base pack: packs/en only
        pass
    for f in candidates:
        try:
            return f.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return None


def tessdata_notice(models: Sequence[Mapping[str, Any]], template: str | None) -> str | None:
    """The pack's ``NOTICE-tessdata``: every model the manifest pins with that notice, one line each
    beside its pinned source, in the format of the base pack's notice (``template``), whose own model
    lines are replaced and whose licence text is kept. None when no model uses the notice, so a pack
    never ships a notice naming another pack's models (kit trial vi: ``osd`` and ``eng``)."""
    mine = [m for m in models if isinstance(m, Mapping) and m.get("notice") == NOTICE_TESSDATA]
    if not mine:
        return None
    lines = "".join(f"  {m.get('file')}  {m.get('source')}\n" for m in mine)
    if template:
        parts = template.splitlines(keepends=True)
        at = [i for i, ln in enumerate(parts) if _NOTICE_MODEL_LINE.fullmatch(ln.rstrip("\r\n"))]
        if at:
            return "".join(parts[:at[0]]) + lines + "".join(parts[at[-1] + 1:])
        return NOTICE_HEADER + "\n" + lines + "\n" + template
    return NOTICE_HEADER + "\n" + lines + "\n" + _NOTICE_FALLBACK


def sync_notice(pack_dir: Path, raw: Mapping[str, Any], packs_root: Path) -> str:
    """Make ``pack_dir/NOTICE-tessdata`` name exactly the Tesseract models ``raw`` pins with it:
    ``written`` (new or changed), ``removed`` (no model uses it any more), ``unchanged``, or ``none``
    (none needed, none there). The base pack's notice is never touched."""
    if raw.get("base"):
        return "unchanged"
    f = Path(pack_dir) / NOTICE_TESSDATA
    text = tessdata_notice(list(raw.get("models") or ()), notice_template(packs_root))
    if text is None:
        if f.is_file():
            f.unlink()
            return "removed"
        return "none"
    data = text.encode("utf-8")
    if f.is_file() and f.read_bytes() == data:
        return "unchanged"
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, f)
    return "written"
