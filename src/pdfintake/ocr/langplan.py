"""Language planning: one pack per OCR page, decided in core (DESIGN §4.3).

``plan`` takes every OCR page's pass-1 signals (``detect.PageSignals``) and the packaged index,
and returns a ``LanguagePlan``: each page's pack, how it was chosen, and whether detection was sure
of every page. ``resolve_unsure`` then settles the pages it wasn't sure of. The rules, in order:

1. **Script per page.** ``detect.script_call`` already named a non-Latin OSD script (confidence of
   at least ``OSD_NONLATIN_MIN_CONF``) or Latin (``eng`` mean confidence of at least
   ``LATIN_ENG_MIN_MEANCONF``). Undecided pages take the document's vote (the most common decided
   script; a tie goes to Latin). When no page is decided, the base pack reads every page that
   nothing else claims.
2. **Non-Latin script.** The pack whose ``detect.osd_scripts`` holds the script reads the page.
3. **Probes and the mixed-script trigger** (on Latin and undecided pages). A probe of an installed
   pack that found at least the pack's ``min_chars`` characters sends the page to that pack
   (``detect.run_probe`` counts them only on confident lines written mostly in the pack's script,
   so the stray characters a probe reads into accented Latin text don't add up to a call). When
   no probe fired (a probe that ran but found fewer characters didn't fire), a calibrated trigger
   hit for a script with a published pack sends the page there (fetched without asking); a hit
   for a script with no pack is the warning
   ``mixed_script_no_pack``; an uncalibrated page with low-confidence ``eng`` blocks and no probe
   gets the note ``mixed_script_unchecked``. Lines are never swapped between reads.
4. **Latin language.** The function-word rule (on the RapidOCR Latin checker read) picks a
   pack that has word lists when its words reach ``min_hits`` and ``ratio`` times the rival words
   (``hits >= ratio * rival_hits - 1e-9``; the exact third ships as 0.3333333333333333).
   Otherwise py3langid decides: probability of at least ``LANGID_MIN_PROB`` on the page and
   ``DOC_AGREE`` of its 120-character windows agreeing. When the top language belongs to a close
   group (``CLOSE_GROUPS``), the group is judged as one: its members' summed probability and the
   windows naming any member. A close-group language is confident only when one published pack
   covers the whole group. The document rule (``document_rule``) gives the remaining pages the
   document's language (pages a probe or the trigger routed count as text pages that didn't
   choose it); pages with too little text fall back to the base pack.
5. **A second Latin language** named by at least ``SECOND_LANG_MIN_WINDOWS`` windows and
   ``SECOND_LANG_MIN_SHARE`` of the page keeps the primary pack as reader; it is added to the
   page's ``languages`` with the warning ``mixed_language_page``.
6. **Outcome.** The plan is confident when every page is and every pack it needs is installed or
   may be fetched (``fetch="auto"``). Otherwise ``candidates`` lists what detection weighed, and
   ``unknown_language`` names a detected language (or script) that has no pack. Each page left
   undecided also carries its own ``candidates``, strongest first.

**Nothing is asked during a run (DESIGN §12.3, decided 2026-11-05).** ``plan`` describes what
detection found; ``resolve_unsure`` then settles every page without a question, and ``flow``
always calls it:

* An undecided page is read with its strongest candidate's pack (``candidates[0]``), with the
  warning ``language_uncertain`` and an offer to read it again with one of the others. A close
  group (Croatian, Bosnian, Serbian) takes its strongest member that has a published pack.
* A language (or script) with no pack is read with the base pack, with the warning
  ``no_pack_for_language`` naming it (``LanguageOffer.detected``).
* Published packs are fetched without asking (``fetch="auto"``, the default), whatever the
  confidence. ``fetch="never"`` (offline) reads the pages of a pack that isn't installed with the
  base pack, with the warning ``pack_not_installed`` and an offer to read them again with that
  pack later. There is no ``ask`` mode any more.
* A pack ``flow`` couldn't use this run (its download, hash check or canary failed; the
  ``broken`` keyword) is replaced by the base pack on every page it would have read, with the
  warning ``pack_broken`` and an offer to read those pages again with it later.

Offers are never waited on: each is a ``LanguageOffer`` a person may act on later with
``pdfintake apply <record-id> --lang <tag>``. A page ``resolve_unsure`` changed keeps the
composition ``plan`` gave it in ``via`` (``hint``, ``forced``, ``undecided`` …), so the record can
still say the page came from ``--lang`` or ``--force-lang``.

``--force-lang`` sets every page (detection only confirms the script), and ``--lang`` never
overrides it. ``--lang`` sets the pages detection was unsure about: undecided pages and pages read
by the base pack for lack of text; confident pages keep their pack, even one that can't be fetched
under ``--fetch never``. ``none`` gives those pages no layer. With several codes the first names
the reader pack and the rest are listed as the page's languages.

A later ``pdfintake apply <record-id> --lang <tag>`` (a person acting on an offer, never a question
the run asked) re-plans with two additive keywords of ``plan``: ``approved`` holds the pack that
request names, which may then be fetched even under ``--fetch never`` (the request is the person's
yes to that one download); ``declined`` holds packs the request turned away from, whose pages
``--lang`` then reaches as if detection had been unsure of them.

Every word in a reason or evidence string is core's; pack data reaches them only as ids and codes
the index has already pattern-checked.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from ..errors import SetupError, UsageError
from ..types import LangCall, ScriptCall
from . import detect
from .detect import FWScore, LangWindow, PageSignals, function_word_score

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from ..langpacks.index import Index, IndexEntry

# --------------------------------------------------------------------------- thresholds (DESIGN Decisions 8)

OSD_NONLATIN_MIN_CONF = 5.0; LATIN_ENG_MIN_MEANCONF = 60.0; LANGID_MIN_PROB = 0.8
LANGID_WINDOW = 120; DOC_AGREE = 2 / 3; SECOND_LANG_MIN_WINDOWS = 3; SECOND_LANG_MIN_SHARE = 0.2
CLOSE_GROUPS = (("hr", "bs", "sr"), ("cs", "sk"), ("id", "ms"), ("da", "no", "nb", "nn", "sv"))

# The function-word page rule (the word lists themselves are pack data).
FW_MIN_HITS_DEFAULT = 3            # a pack's function_words.min_hits when it gives none
FW_RATIO_DEFAULT = 1 / 3           # ... and its ratio: hits >= ratio * rival hits - _EPS
# The rule is an exact third (``3 * hits >= rival_hits``). Manifests carry it as 0.3333333333333333
# (the double nearest 1/3, equal to ``1 / 3``), and the comparison allows ``_EPS`` below the product
# so a float product a hair above the integer bar can't turn 3 hits against 9 rivals away
# (decided 2026-11-05).
DOC_TEXT_PAGE_WORDS = 30           # a "text page" for the document rule
DOC_MAX_OTHER = 2                  # a page inherits only with rival + English words at most this
PROBE_MIN_CHARS_DEFAULT = 20       # when the index's detect copy says only "probe": true

CANDIDATE_MIN_PROB = 0.05          # an unsure page offers identifier codes at this probability or more
CANDIDATE_MAX_PER_PAGE = 3
# §12.3 (decided 2026-11-05): published packs are always fetched without asking; "never" is the
# offline mode. The old "ask" mode is gone, and passing it is a UsageError.
FETCH_POLICIES = ("auto", "never")
# The last four are resolve_unsure's: the strongest candidate's pack, the base pack for a language
# with no pack, the base pack for a pack this offline run may not fetch, and the base pack for a
# pack that failed this run (its download, hash check or canary; decided 2026-11-05).
COMPOSITIONS = ("script", "language", "document", "probe", "trigger", "base", "hint", "forced", "none", "undecided",
                "strongest", "no_pack", "offline", "broken")
# Why a page was left undecided (PagePlan.unsure).
UNSURE_KINDS = ("uncertain", "close_group", "no_pack")
# The warning codes resolve_unsure adds to pages (record.WARNING_CODES, group "language").
RESOLVE_WARNINGS = ("language_uncertain", "no_pack_for_language", "pack_not_installed", "pack_broken")
OFFER_REASONS = ("uncertain", "no_pack", "not_installed", "broken")
OFFER_APPLY_FLAG = "--lang"

_CODE_RE = re.compile(r"[a-z]{2,3}")
_SCRIPT_NAME_RE = re.compile(r"[A-Z][A-Za-z]{1,19}")
_EPS = 1e-9


# --------------------------------------------------------------------------- calibration

@dataclass(frozen=True)
class Calibration:
    block_max_conf: float      # a block is a trigger candidate below this mean line confidence, in [0, 1]
    osd_min_conf: float        # OSD's script confidence a non-Latin call needs (OSD's own scale)
    min_block_px: int          # a block's shorter side, in px at about 300 dpi
    max_blocks: int            # blocks given to OSD per page
    measured: str              # the kit version and corpus digest it was measured on


_CALIBRATION_KEYS = ("block_max_conf", "osd_min_conf", "min_block_px", "max_blocks", "measured")
_MEASURED_RE = re.compile(r"[A-Za-z0-9 .,:;@/_+()=-]{1,200}")


class _DuplicateKey(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key[:40]


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def _calibration_file() -> "Traversable":
    from importlib import resources

    return resources.files("pdfintake.ocr").joinpath("calibration.json")


def load_calibration() -> Calibration | None:
    """The mixed-script trigger's measured thresholds from ``ocr/calibration.json``.

    ``{"mixed_script": null}`` (as it ships until the kit has measured the trigger) gives None,
    which turns the trigger off. A missing, unreadable or malformed file is a SetupError naming it:
    package data that didn't install must fail loudly, not switch the trigger off.
    """
    path = _calibration_file()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise SetupError("calibration_missing", "pdfintake's ocr/calibration.json is missing from the install",
                         file=str(path)) from None
    except (OSError, UnicodeDecodeError) as e:
        raise SetupError("calibration_unreadable", f"pdfintake can't read ocr/calibration.json ({type(e).__name__})",
                         file=str(path)) from None
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except _DuplicateKey as e:
        raise SetupError("calibration_invalid", f"ocr/calibration.json repeats the key {e.key!r}",
                         file=str(path)) from None
    except (ValueError, RecursionError):
        raise SetupError("calibration_invalid", "ocr/calibration.json isn't valid JSON", file=str(path)) from None
    problems: list[str] = []
    if not isinstance(data, dict):
        raise SetupError("calibration_invalid", "ocr/calibration.json must be a JSON object", file=str(path))
    extra = sorted(set(data) - {"mixed_script"})
    if extra:
        problems.append("unknown keys " + ", ".join(repr(k) for k in extra))
    if "mixed_script" not in data:
        problems.append("the key 'mixed_script' is missing")
    ms = data.get("mixed_script")
    if ms is None:
        if problems:
            raise SetupError("calibration_invalid", "ocr/calibration.json: " + "; ".join(problems), file=str(path))
        return None
    if not isinstance(ms, dict):
        raise SetupError("calibration_invalid", "ocr/calibration.json: mixed_script must be null or an object",
                         file=str(path))
    extra = sorted(set(ms) - set(_CALIBRATION_KEYS))
    if extra:
        problems.append("mixed_script has unknown keys " + ", ".join(repr(k) for k in extra))
    missing = [k for k in _CALIBRATION_KEYS if k not in ms]
    if missing:
        problems.append("mixed_script lacks " + ", ".join(missing))

    def num(key: str, lo: float, hi: float, *, integer: bool = False, lo_open: bool = False) -> Any:
        v = ms.get(key)
        ok_type = isinstance(v, int) if integer else isinstance(v, (int, float))
        if isinstance(v, bool) or not ok_type or not math.isfinite(v) or v > hi or v < lo or (lo_open and v <= lo):
            if key in ms:
                problems.append(f"mixed_script.{key} must be {'an integer' if integer else 'a number'} "
                                f"{'above' if lo_open else 'from'} {lo:g} {'up ' if lo_open else ''}to {hi:g}")
            return None
        return int(v) if integer else float(v)

    block_max_conf = num("block_max_conf", 0.0, 1.0, lo_open=True)
    osd_min_conf = num("osd_min_conf", 0.0, 1000.0)
    min_block_px = num("min_block_px", 1, 10000, integer=True)
    max_blocks = num("max_blocks", 1, 64, integer=True)
    measured = ms.get("measured")
    if "measured" in ms and not (isinstance(measured, str) and _MEASURED_RE.fullmatch(measured)):
        problems.append("mixed_script.measured must be a short plain string (the kit version and corpus digest)")
    if problems:
        raise SetupError("calibration_invalid", "ocr/calibration.json: " + "; ".join(problems), file=str(path))
    return Calibration(block_max_conf, osd_min_conf, min_block_px, max_blocks, measured)


# --------------------------------------------------------------------------- plan types

@dataclass
class Candidate:
    tag: str                         # a language code (the ``apply --lang`` answer when ``pack`` is set)
    pack: str | None                 # the pack that reads it, or None when there is none yet
    evidence: str                    # core wording, no document text
    download_bytes: int | None       # 0 when installed; None without a pack
    # Beyond DESIGN §2.2 (additive): the identifier's probability for ``tag`` when that is what
    # ranked it, else None (a pack a page needs, or a script call).
    prob: float | None = None


@dataclass
class PagePlan:
    index: int
    pack: str | None                 # None: no layer (``none``), or undecided (confident False)
    composition: str                 # how the pack was chosen: one of COMPOSITIONS
    languages: list[LangCall]        # primary first; a second Latin language after it
    confident: bool                  # detection was sure of this page (resolve_unsure never changes it)
    reason: str                      # core wording, no document text
    # Beyond DESIGN §2.2 (additive): the script the page was planned with (the document vote, a
    # probe or the trigger may change ``signals.script``), and the page's warning and note codes
    # (record.WARNING_CODES / NOTE_CODES) for the pipeline to copy onto the page and the record.
    script: ScriptCall | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # §12.3 (additive): on a page detection left undecided, why (one of UNSURE_KINDS), the language
    # code or script name it found that has no pack, and its own candidates, strongest first (the
    # base pack where its strength puts it, else last). resolve_unsure reads these, and on a page
    # it gives no_pack_for_language, ``detected`` names the language (or script) the warning is about.
    # ``detected`` is always what record.language_warning("no_pack_for_language", language=)
    # accepts: a language code, a script name matching _SCRIPT_NAME_RE, or None when detection named
    # neither.
    unsure: str | None = None
    detected: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    # §12.3 (additive): on a page resolve_unsure changed, the composition ``plan`` gave it ("hint",
    # "forced", "undecided", "language" …); None on a page it left alone. Flow reads the record's
    # ``language.plan`` (hinted, forced, uncertain) from it (decided 2026-11-05).
    via: str | None = None


@dataclass
class LanguageOffer:
    """Something a person could do later about a language choice the run made for them (§12.5).

    Never waited on. Each option's ``tag`` is the answer to ``pdfintake apply <record-id> --lang``.
    ``pages`` are ``PageSignals.index`` values (as the plan's pages carry them), ascending.
    """
    reason: str                      # one of OFFER_REASONS
    pages: list[int]
    pack: str                        # the pack that read these pages in this run
    read_as: str                     # the language tag that pack was used for
    detected: str | None             # "no_pack": the language code (or script name) with no pack yet
    options: list[Candidate]         # other packs to read them again with, strongest first (never empty)
    kind: str = "language"
    apply_flag: str = OFFER_APPLY_FLAG


@dataclass
class LanguagePlan:
    pages: list[PagePlan]
    packs_needed: list[str]          # every pack a page is planned with, sorted (the base pack included)
    confident: bool                  # detection's own verdict (resolve_unsure never changes it)
    candidates: list[Candidate]      # empty when confident; otherwise strongest first, the base pack last
    unknown_language: str | None     # the strongest detected language code (or a script name) with no pack
    notes: list[str]                 # every page's note codes, each once
    # §12.3 (additive): the packs this run may not fetch (``fetch="never"``: every pack in the index
    # that isn't installed or approved), resolve_unsure's offers and the warning codes it added
    # (each once, page order), and whether it has run.
    unavailable: list[str] = field(default_factory=list)
    offers: list[LanguageOffer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved: bool = False


# --------------------------------------------------------------------------- index helpers

@dataclass(frozen=True)
class _FW:
    words: frozenset[str]
    rivals: frozenset[str]
    min_hits: int
    ratio: float


def _detect_block(entry: "IndexEntry") -> Mapping[str, Any]:
    d = getattr(entry, "detect", None)
    return d if isinstance(d, Mapping) else {}


def _osd_scripts(entry: "IndexEntry") -> tuple[str, ...]:
    v = _detect_block(entry).get("osd_scripts") or ()
    return tuple(str(s) for s in v) if isinstance(v, (list, tuple)) else ()


def _close_group_of(entry: "IndexEntry") -> tuple[str, ...]:
    lid = _detect_block(entry).get("langid")
    if not isinstance(lid, Mapping):
        return ()
    v = lid.get("close_group") or ()
    return tuple(str(s) for s in v) if isinstance(v, (list, tuple)) else ()


def _function_words(entry: "IndexEntry") -> _FW | None:
    fw = _detect_block(entry).get("function_words")
    if not isinstance(fw, Mapping):
        return None
    words = fw.get("words") or ()
    if not isinstance(words, (list, tuple, set, frozenset)) or not words:
        return None
    rivals = fw.get("rivals") or ()
    if not isinstance(rivals, (list, tuple, set, frozenset)):
        rivals = ()
    min_hits = fw.get("min_hits", FW_MIN_HITS_DEFAULT)
    ratio = fw.get("ratio", FW_RATIO_DEFAULT)
    if isinstance(min_hits, bool) or not isinstance(min_hits, int) or min_hits < 1:
        min_hits = FW_MIN_HITS_DEFAULT
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not (0.0 <= ratio <= 1.0):
        ratio = FW_RATIO_DEFAULT
    return _FW(frozenset(str(w).lower() for w in words), frozenset(str(w).lower() for w in rivals),
               int(min_hits), float(ratio))


def _probe_min_chars(entry: "IndexEntry", pr: detect.ProbeResult | None = None) -> int | None:
    """The pack's probe threshold; None when the pack has no probe.

    The real index copies ``"probe": true`` only, so the installed manifest's ``min_chars`` reaches
    here on the ``ProbeResult`` (``run_probe(..., min_chars=)``); a full probe block in the index
    copy is read too; otherwise ``PROBE_MIN_CHARS_DEFAULT``.
    """
    p = _detect_block(entry).get("probe")
    if p is None or p is False:
        return None
    v = getattr(pr, "min_chars", None)
    if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
        return v
    if isinstance(p, Mapping):
        v = p.get("min_chars")
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            return v
        return PROBE_MIN_CHARS_DEFAULT
    if p is True:
        return PROBE_MIN_CHARS_DEFAULT
    return None


def _lang_of(entry: "IndexEntry") -> str:
    codes = tuple(getattr(entry, "langid_codes", ()) or ())
    return str(codes[0]) if codes else str(entry.id)


def _entries(index: "Index") -> tuple["IndexEntry", ...]:
    return tuple(index.entries)


def _pack_for_language(index: "Index", code: str) -> "IndexEntry | None":
    """The pack that reads ``code``: the entry listing it first among its langid codes wins."""
    best: tuple[int, int] | None = None
    found = None
    for pos, e in enumerate(_entries(index)):
        codes = tuple(getattr(e, "langid_codes", ()) or ())
        if code in codes:
            key = (codes.index(code), pos)
            if best is None or key < best:
                best, found = key, e
    return found


def _pack_for_script(index: "Index", script: str) -> "IndexEntry | None":
    for e in _entries(index):
        if script in _osd_scripts(e):
            return e
    return None


def _nonlatin_script(entry: "IndexEntry") -> str | None:
    for s in _osd_scripts(entry) + tuple(getattr(entry, "scripts", ()) or ()):
        if s != "Latin":
            return str(s)
    return None


def _group_of(code: str) -> tuple[str, ...] | None:
    for g in CLOSE_GROUPS:
        if code in g:
            return g
    return None


def _covering_pack(index: "Index", group: Sequence[str]) -> "IndexEntry | None":
    """One published pack whose langid codes and close group together cover every member of ``group``."""
    need = set(group)
    for e in _entries(index):
        have = set(getattr(e, "langid_codes", ()) or ()) | set(_close_group_of(e))
        if need <= have:
            return e
    return None


def _download_bytes(entry: "IndexEntry", have: Set[str]) -> int | None:
    if entry.id in have:
        return 0
    dist = getattr(entry, "dist", None)
    b = getattr(dist, "bytes", None)
    return int(b) if isinstance(b, int) and not isinstance(b, bool) else None


def _safe_script(name: str | None) -> str | None:
    """A script name fit for core wording: a core script, or a plain capitalised word."""
    if name is None:
        return None
    from . import scripts

    if name in scripts.SCRIPTS or _SCRIPT_NAME_RE.fullmatch(name):
        return name
    return None


# --------------------------------------------------------------------------- the document rule

def document_rule(signals: Sequence[PageSignals], fw: Mapping[str, Sequence[FWScore]],
                  chosen: Mapping[int, str]) -> dict[int, str]:
    """Pages that inherit the document's language.

    ``fw`` holds, per pack id, one ``FWScore`` per page of ``signals`` (in that order); ``chosen``
    maps a page index to the pack its own evidence already chose. For each pack, most chosen first:
    when at least half of the text pages (``DOC_TEXT_PAGE_WORDS`` words or more) chose it, including
    the vacuous case of no text page at all, every page not yet chosen that has words, and at most
    ``DOC_MAX_OTHER`` rival plus English words, inherits it. Returns {page index: pack id}.
    """
    n = len(signals)
    for p, scores in fw.items():
        if len(scores) != n:
            raise ValueError(f"fw[{p!r}] has {len(scores)} scores for {n} pages")
    idx = [s.index for s in signals]
    present = set(idx)
    counts = Counter(p for i, p in chosen.items() if i in present)
    out: dict[int, str] = {}
    for p in sorted(fw, key=lambda q: (-counts.get(q, 0), q)):
        if not counts.get(p):
            continue
        scores = fw[p]
        textp = [k for k in range(n) if scores[k].words >= DOC_TEXT_PAGE_WORDS]
        if 2 * sum(1 for k in textp if chosen.get(idx[k]) == p) < len(textp):
            continue
        for k in range(n):
            i = idx[k]
            if i in chosen or i in out:
                continue
            s = scores[k]
            if s.words and s.rival_hits + s.en_hits <= DOC_MAX_OTHER:
                out[i] = p
    return out


def _doc_share(signals: Sequence[PageSignals], scores: Sequence[FWScore], chosen: Mapping[int, str], p: str) -> float:
    textp = [k for k in range(len(signals)) if scores[k].words >= DOC_TEXT_PAGE_WORDS]
    if not textp:
        return 1.0
    return sum(1 for k in textp if chosen.get(signals[k].index) == p) / len(textp)


# --------------------------------------------------------------------------- per-page helpers

@dataclass
class _Ev:
    tag: str
    pack: str | None
    evidence: str
    strength: float


def _agree(windows: Sequence[LangWindow], code: str) -> int:
    return sum(1 for w in windows if w.code == code)


def _fires(s: FWScore, fw: _FW) -> bool:
    return s.hits >= fw.min_hits and s.hits >= fw.ratio * s.rival_hits - _EPS


def _fw_read(sig: PageSignals):
    r = sig.checker_read
    if r is not None and r.status == "ok":
        return r
    return sig.eng_read


def _second_language(windows: Sequence[LangWindow], primary: str) -> LangCall | None:
    if len(windows) < SECOND_LANG_MIN_WINDOWS:
        return None
    group = _group_of(primary) or ()
    counts = Counter(w.code for w in windows if _CODE_RE.fullmatch(w.code))
    for code, k in sorted(counts.items(), key=lambda t: (-t[1], t[0])):
        if code == primary or code in group:
            continue
        if k >= SECOND_LANG_MIN_WINDOWS and k + _EPS >= SECOND_LANG_MIN_SHARE * len(windows):
            probs = [w.prob for w in windows if w.code == code]
            return LangCall(code, sum(probs) / len(probs), "langid", k)
        return None
    return None


def _resolve_scripts(sigs: Sequence[PageSignals]) -> tuple[list[str | None], list[ScriptCall]]:
    decided = [s.script.name for s in sigs if s.script.name is not None]
    winner: str | None = None
    share = 0.0
    if decided:
        counts = Counter(decided)
        top = max(counts.values())
        tied = sorted(n for n, c in counts.items() if c == top)
        winner = "Latin" if "Latin" in tied else tied[0]
        share = top / len(decided)
    names: list[str | None] = []
    calls: list[ScriptCall] = []
    for s in sigs:
        if s.script.name is not None:
            names.append(s.script.name)
            calls.append(s.script)
        elif winner is not None:
            names.append(winner)
            calls.append(ScriptCall(winner, share, "vote", s.script.eng_mean_conf))
        else:
            names.append(None)
            calls.append(s.script)
    return names, calls


def _route_mixed(sig: PageSignals, sc: ScriptCall, index: "Index") -> tuple[PagePlan | None, list[str], list[str]]:
    """A probe or the trigger sending a Latin (or undecided) page to a non-Latin pack."""
    fired: list[tuple[int, str, Any]] = []
    for pack_id in sorted(sig.probes):
        pr = sig.probes[pack_id]
        entry = index.get(pack_id)
        need = _probe_min_chars(entry, pr) if entry is not None else None
        if need is not None and pr.chars >= need:
            fired.append((pr.chars, pack_id, entry))
    if fired:
        fired.sort(key=lambda t: (-t[0], t[1]))
        chars, pack_id, entry = fired[0]
        script = _nonlatin_script(entry)
        return (PagePlan(sig.index, pack_id, "probe", [LangCall(_lang_of(entry), 1.0, "pack_script", 0)], True,
                         f"pack {pack_id}'s probe found {chars} characters of its script on this page",
                         script=ScriptCall(script, float(chars), "probe", sc.eng_mean_conf),
                         warnings=["mixed_script_page"]), [], [])
    m = sig.mixed
    if m.checked and m.script is not None and m.script != "Latin":
        entry = _pack_for_script(index, m.script)
        if entry is None:
            return None, ["mixed_script_no_pack"], []
        # No probe fired (DESIGN §4.3 step 4), so the trigger decides, even when the pack's own
        # probe ran and found fewer than its min_chars characters.
        name = _safe_script(m.script) or "a non-Latin"
        return (PagePlan(sig.index, entry.id, "trigger", [LangCall(_lang_of(entry), 1.0, "pack_script", 0)], True,
                         f"the mixed-script check found {name} script on this page",
                         script=ScriptCall(m.script, float(m.conf or 0.0), "trigger", sc.eng_mean_conf),
                         warnings=["mixed_script_page"]), [], [])
    if not m.checked and m.blocks > 0 and not sig.probes:
        return None, [], ["mixed_script_unchecked"]
    return None, [], []


# --------------------------------------------------------------------------- detection

@dataclass
class _State:
    pages: list[PagePlan | None]
    hint_ok: list[bool]
    ev: list[list[_Ev]]
    unknown: list[tuple[float, str] | None]
    unsure: list[str | None]


def _detect_pages(sigs: Sequence[PageSignals], index: "Index", base: "IndexEntry") -> _State:
    n = len(sigs)
    st = _State([None] * n, [False] * n, [[] for _ in range(n)], [None] * n, [None] * n)
    names, calls = _resolve_scripts(sigs)
    latin: list[int] = []
    routed: list[int] = []
    extra_w: list[list[str]] = [[] for _ in range(n)]
    extra_n: list[list[str]] = [[] for _ in range(n)]
    for pos, sig in enumerate(sigs):
        name, sc = names[pos], calls[pos]
        if name is not None and name != "Latin":
            entry = _pack_for_script(index, name)
            how = "the document's vote" if sc.by == "vote" else f"OSD {sc.conf:.1f}"
            shown = _safe_script(name) or "an unknown"
            if entry is not None:
                st.pages[pos] = PagePlan(sig.index, entry.id, "script",
                                         [LangCall(_lang_of(entry), 1.0, "pack_script", 0)], True,
                                         f"{shown} script ({how})", script=sc)
            else:
                st.pages[pos] = PagePlan(sig.index, None, "undecided", [], False,
                                         f"{shown} script ({how}), and no language pack reads it yet", script=sc)
                st.hint_ok[pos] = True
                st.unsure[pos] = "no_pack"
                if _safe_script(name):
                    st.unknown[pos] = (1.0, name)
            continue
        routed_plan, warns, notes = _route_mixed(sig, sc, index)
        if routed_plan is not None:
            st.pages[pos] = routed_plan
            routed.append(pos)
            continue
        extra_w[pos] += warns
        extra_n[pos] += notes
        latin.append(pos)
    if latin:
        _latin_languages(sigs, latin, routed, names, calls, index, base, st, extra_w, extra_n)
    return st


def _latin_languages(sigs: Sequence[PageSignals], latin: Sequence[int], routed: Sequence[int],
                     names: Sequence[str | None], calls: Sequence[ScriptCall], index: "Index", base: "IndexEntry",
                     st: _State, extra_w: Sequence[list[str]], extra_n: Sequence[list[str]]) -> None:
    fw_packs = [(e, fwl) for e in _entries(index) if (fwl := _function_words(e)) is not None]
    # The document rule sees the Latin pages and the pages a probe or the trigger routed (as text
    # pages that chose another pack).
    doc = sorted(set(latin) | set(routed))
    kpos = {pos: k for k, pos in enumerate(doc)}
    reads = [_fw_read(sigs[pos]) for pos in doc]
    fw: dict[str, list[FWScore]] = {e.id: [function_word_score(r, fwl.words, fwl.rivals) for r in reads]
                                    for e, fwl in fw_packs}
    decided: dict[int, tuple[Any, LangCall, str]] = {}
    held: dict[int, tuple[LangCall, str]] = {}

    for pos in latin:
        k = kpos[pos]
        sig = sigs[pos]
        windows = [w for w in sig.windows if _CODE_RE.fullmatch(w.code)]
        fired = [(fw[e.id][k], e) for e, fwl in fw_packs if _fires(fw[e.id][k], fwl)]
        if fired:
            fired.sort(key=lambda t: (-t[0].hits, t[1].id))
            s, e = fired[0]
            code = _lang_of(e)
            prob = s.hits / (s.hits + s.rival_hits) if s.hits + s.rival_hits else 0.0
            decided[pos] = (e, LangCall(code, prob, "function_words", _agree(windows, code)),
                            f"language {code} from its function words ({s.hits} against {s.rival_hits} rival words)")
            continue
        ranked = [(c, p) for c, p in (detect.page_language(sig.eng_read) if windows else [])
                  if _CODE_RE.fullmatch(c)]
        if not ranked:
            continue                                  # too little text: the document rule or the base pack
        code, prob = ranked[0]
        group = _group_of(code)
        if group is not None:
            # A close group is judged as one: the identifier splits a page across its members.
            gprob = min(1.0, sum(p for c, p in ranked if c in group))
            agree = sum(1 for w in windows if w.code in group)
            sure = gprob + _EPS >= LANGID_MIN_PROB and agree + _EPS >= DOC_AGREE * len(windows)
        else:
            gprob = prob
            agree = _agree(windows, code)
            sure = prob >= LANGID_MIN_PROB and agree + _EPS >= DOC_AGREE * len(windows)
        if sure:
            call = LangCall(code, prob, "langid", agree)
            if group is not None:
                cover = _covering_pack(index, group)
                if cover is not None:
                    decided[pos] = (cover, call, f"language {code} ({prob:.2f}; its close group {gprob:.2f}, "
                                                 f"{agree} of {len(windows)} windows), "
                                                 f"a close group that pack {cover.id} reads whole")
                    continue
                held[pos] = (call, f"language {code} is one of a close group ({', '.join(group)}) "
                                   f"and no single pack reads them all")
                st.unsure[pos] = "close_group"
                probs = dict(ranked)
                for member in group:
                    e = _pack_for_language(index, member)
                    st.ev[pos].append(_Ev(member, e.id if e else None, "a close language group",
                                          probs.get(member, 0.0)))
                if _pack_for_language(index, code) is None:
                    st.unknown[pos] = (prob, code)
                continue
            entry = _pack_for_language(index, code)
            if entry is not None:
                decided[pos] = (entry, call, f"language {code} ({prob:.2f}, {agree} of {len(windows)} windows)")
                continue
            held[pos] = (call, f"language {code} ({prob:.2f}) has no language pack yet")
            st.unsure[pos] = "no_pack"
            st.ev[pos].append(_Ev(code, None, "detected, but there is no pack for it yet", prob))
            st.unknown[pos] = (prob, code)
            continue
        for c, p in ranked[:CANDIDATE_MAX_PER_PAGE]:
            if p < CANDIDATE_MIN_PROB:
                break
            e = _pack_for_language(index, c)
            st.ev[pos].append(_Ev(c, e.id if e else None, f"the language identifier gave it {p:.2f}", p))

    chosen = {sigs[pos].index: decided[pos][0].id for pos in decided}
    doc_sigs = [sigs[pos] for pos in doc]
    fw_all: dict[str, Sequence[FWScore]] = dict(fw)
    for pid in sorted(set(chosen.values()) - set(fw_all)):
        fw_all[pid] = [function_word_score(r, frozenset(), frozenset()) for r in reads]
    for pos in routed:                       # counted, never inherited
        pp = st.pages[pos]
        if pp is not None and pp.pack is not None:
            chosen[sigs[pos].index] = pp.pack
    inherit = document_rule(doc_sigs, fw_all, chosen)

    for pos in latin:
        sig = sigs[pos]
        sc = calls[pos]
        warns = list(extra_w[pos])
        notes = list(extra_n[pos])
        windows = [w for w in sig.windows if _CODE_RE.fullmatch(w.code)]
        if pos in decided:
            entry, call, why = decided[pos]
            langs = [call]
            second = _second_language(windows, call.tag)
            if second is not None:
                langs.append(second)
                warns.append("mixed_language_page")
            st.pages[pos] = PagePlan(sig.index, entry.id, "language", langs, True, why, script=sc,
                                     warnings=warns, notes=notes)
        elif pos in held:
            call, why = held[pos]
            st.pages[pos] = PagePlan(sig.index, None, "undecided", [call], False, why, script=sc,
                                     warnings=warns, notes=notes)
            st.hint_ok[pos] = True
        elif sig.index in inherit:
            pid = inherit[sig.index]
            entry = index.get(pid)
            code = _lang_of(entry) if entry is not None else pid
            share = _doc_share(doc_sigs, fw_all[pid], chosen, pid)
            st.pages[pos] = PagePlan(sig.index, pid, "document", [LangCall(code, share, "document", len(windows))],
                                     True, f"took the document's language ({code})", script=sc,
                                     warnings=warns, notes=notes)
        elif not windows or names[pos] is None:
            why = ("too little text to tell the language, so the base pack reads it" if not windows
                   else "neither the script nor the language is clear, so the base pack reads it")
            st.pages[pos] = PagePlan(sig.index, base.id, "base", [], True, why, script=sc,
                                     warnings=warns, notes=notes)
            st.hint_ok[pos] = True
        else:
            st.pages[pos] = PagePlan(sig.index, None, "undecided", [], False,
                                     "the language identifier wasn't sure about this page", script=sc,
                                     warnings=warns, notes=notes)
            st.hint_ok[pos] = True
            st.unsure[pos] = "uncertain"


# --------------------------------------------------------------------------- hints

def _parse_codes(codes: Sequence[str] | None, index: "Index", flag: str) -> Any:
    """None (no codes), "none", or [(code, index entry), ...] in the order given; UsageError otherwise."""
    if not codes:
        return None
    if isinstance(codes, str):
        codes = (codes,)
    out: list[str] = []
    for raw in codes:
        if not isinstance(raw, str):
            raise UsageError("bad_language_code", f"{flag} takes language codes such as en or fr", flag=flag)
        for part in raw.split(","):
            c = part.strip().lower()
            if c and c not in out:
                out.append(c)
    if not out:
        return None
    if "none" in out:
        if len(out) > 1:
            raise UsageError("lang_none_with_codes", f"{flag} none can't be combined with a language", flag=flag)
        return "none"
    pairs: list[tuple[str, Any]] = []
    for c in out:
        if not _CODE_RE.fullmatch(c):
            raise UsageError("bad_language_code", f"{flag} takes ISO 639 language codes such as en or fr", flag=flag)
        e = _pack_for_language(index, c)
        if e is None:
            e = index.get(c)                # a pack without langid codes answers to its id (its candidate tag)
        if e is None:
            raise UsageError("unknown_language", f"pdfintake has no language pack for '{c}' yet", flag=flag, language=c)
        pairs.append((c, e))
    return pairs


def _forced_page(sig: PageSignals, forced: Any) -> PagePlan:
    if forced == "none":
        return PagePlan(sig.index, None, "none", [], True, "no text layer (--force-lang none)", script=sig.script)
    assert isinstance(forced, list)
    code, e = forced[0]
    langs = [LangCall(c, 1.0, "forced", 0) for c, _ in forced]
    why = f"language {code} set with --force-lang"
    s = sig.script.name
    if s is not None and s != "Latin" and s not in tuple(getattr(e, "scripts", ()) or ()):
        why += f"; the page looks like {_safe_script(s) or 'another'} script, which pack {e.id} doesn't read"
    return PagePlan(sig.index, e.id, "forced", langs, True, why, script=sig.script)


def _hinted_page(pp: PagePlan, hint: Any) -> PagePlan:
    keep_w = [w for w in pp.warnings if w != "mixed_language_page"]
    if hint == "none":
        return PagePlan(pp.index, None, "none", [], True, "no text layer (--lang none)", script=pp.script,
                        warnings=keep_w, notes=list(pp.notes))
    assert isinstance(hint, list)
    code, e = hint[0]
    langs = [LangCall(c, 1.0, "hint", 0) for c, _ in hint]
    why = f"language {code} set with --lang"
    s = pp.script.name if pp.script is not None else None
    if s is not None and s != "Latin" and s not in tuple(getattr(e, "scripts", ()) or ()):
        why += f"; the page looks like {_safe_script(s) or 'another'} script, which pack {e.id} doesn't read"
    return PagePlan(pp.index, e.id, "hint", langs, True, why, script=pp.script,
                    warnings=keep_w, notes=list(pp.notes))


# --------------------------------------------------------------------------- plan

def _pooled(evs: Sequence[_Ev]) -> list[tuple[float, _Ev]]:
    """One entry per language tag, strongest first: the strongest evidence for a tag decides its
    rank, and evidence naming a pack is kept over evidence that names none."""
    pool: dict[str, list[Any]] = {}          # tag -> [strength, first seen, the evidence kept]
    for ev in evs:
        if not _CODE_RE.fullmatch(ev.tag):
            continue
        cur = pool.get(ev.tag)
        if cur is None:
            pool[ev.tag] = [ev.strength, len(pool), ev]
            continue
        kept: _Ev = cur[2]
        if (kept.pack is None and ev.pack is not None) or (
                (kept.pack is None) == (ev.pack is None) and ev.strength > cur[0]):
            cur[2] = ev
        cur[0] = max(cur[0], ev.strength)
    return [(t[0], t[2]) for t in sorted(pool.values(), key=lambda t: (-t[0], t[1], t[2].tag))]


def _candidate(strength: float, ev: _Ev, index: "Index", have: Set[str]) -> Candidate:
    e = index.get(ev.pack) if ev.pack else None
    prob = float(strength) if 0.0 <= strength <= 1.0 else None      # 2.0 marks a pack a page needs
    return Candidate(ev.tag, ev.pack if e is not None else None, ev.evidence,
                     _download_bytes(e, have) if e is not None else None, prob)


def _base_candidate(base: "IndexEntry") -> Candidate:
    return Candidate(_lang_of(base), base.id, "the base pack, always installed", 0)


def _page_candidates(evs: Sequence[_Ev], index: "Index", have: Set[str], base: "IndexEntry") -> list[Candidate]:
    """One undecided page's candidates, strongest first. Unlike the document's list, the base pack
    keeps the rank its own evidence earns (an English-leaning page offers English first); without
    evidence for it, it comes last."""
    out = [_candidate(s, ev, index, have) for s, ev in _pooled(evs)]
    base_tag = _lang_of(base)
    for k, c in enumerate(out):
        if c.tag == base_tag:
            out[k] = Candidate(base_tag, base.id, c.evidence, 0, c.prob)
            return out
    return out + [_base_candidate(base)]


def plan(signals: Sequence[PageSignals], index: "Index", *, lang: Sequence[str], force_lang: Sequence[str],
         installed: Set[str], fetch: Literal["auto", "never"],
         approved: Set[str] = frozenset(), declined: Set[str] = frozenset()) -> LanguagePlan:
    """Each OCR page's pack, by the rules in this module's docstring (DESIGN §4.3).

    ``installed`` holds the ids of the packs installed now (the base pack always counts as
    installed). ``fetch="auto"`` (the default everywhere) fetches every published pack the plan
    needs, without asking (§12.3); ``"never"`` works offline: a pack that isn't installed is listed
    in ``unavailable``, and ``resolve_unsure`` reads its pages with the base pack. ``"ask"`` no
    longer exists. A ``--lang`` or ``--force-lang`` code with no pack in the index is a UsageError,
    as is an unknown fetch policy.

    The plan describes what detection found and leaves undecided pages undecided; ``flow`` always
    passes it through ``resolve_unsure``, which settles them without asking.

    Beyond DESIGN §2.2 (additive, for a later ``pdfintake apply <record-id> --lang``, which only a
    person's own request starts): ``approved`` holds the pack ids that request reads pages with;
    each may be fetched even under ``fetch="never"``. ``declined`` holds pack ids the request
    turned away from; pages planned with one that may not be fetched take ``--lang`` as unsure
    pages do. Ids not in the index are a ValueError.
    """
    if fetch not in FETCH_POLICIES:
        raise UsageError("bad_fetch", "--fetch takes auto or never", value=str(fetch)[:16])
    approved = frozenset(approved)
    declined = frozenset(declined)
    for name, ids in (("approved", approved), ("declined", declined)):
        unknown_ids = sorted(str(i)[:16] for i in ids if not isinstance(i, str) or index.get(i) is None)
        if unknown_ids:
            raise ValueError(f"{name} names packs the index doesn't list: {', '.join(unknown_ids)}")
    sigs = list(signals)
    seen: set[int] = set()
    for s in sigs:
        if s.index in seen:
            raise ValueError(f"page {s.index} appears twice in the signals")
        seen.add(s.index)
    base = index.base()
    have = frozenset(installed) | {base.id}
    forced = _parse_codes(force_lang, index, "--force-lang")
    hint = _parse_codes(lang, index, "--lang")
    unavailable = (sorted(e.id for e in _entries(index) if e.id not in have and e.id not in approved)
                   if fetch == "never" else [])

    if forced is not None:
        st = _State([_forced_page(s, forced) for s in sigs], [False] * len(sigs),
                    [[] for _ in sigs], [None] * len(sigs), [None] * len(sigs))
    else:
        st = _detect_pages(sigs, index, base)
    pages: list[PagePlan] = [p for p in st.pages if p is not None]
    assert len(pages) == len(sigs)

    def blocked(pack: str | None) -> bool:
        return pack is not None and pack in unavailable

    if hint is not None:
        for pos, pp in enumerate(pages):
            # --lang reaches only pages detection was unsure about (DESIGN §4.1), and pages whose
            # pack a person declined; a confident page keeps its pack.
            if st.hint_ok[pos] or (pp.pack in declined and blocked(pp.pack)):
                pages[pos] = _hinted_page(pp, hint)

    for pos, pp in enumerate(pages):
        if pp.composition != "undecided":
            continue
        pp.unsure = st.unsure[pos] or "uncertain"
        u = st.unknown[pos]
        pp.detected = u[1] if u is not None else None
        pp.candidates = _page_candidates(st.ev[pos], index, have, base)

    needed = sorted({pp.pack for pp in pages if pp.pack is not None})
    missing = [p for p in needed if blocked(p)]
    notes: list[str] = []
    for pp in pages:
        for code in pp.notes:
            if code not in notes:
                notes.append(code)
    confident = all(pp.confident for pp in pages) and not missing
    if confident:
        return LanguagePlan(pages, needed, True, [], None, notes, unavailable)

    evs: list[_Ev] = []
    for pid in missing:
        e = index.get(pid)
        if e is not None:
            evs.append(_Ev(_lang_of(e), pid, "a page needs this pack, which isn't installed", 2.0))
    unknown: tuple[float, int, str] | None = None
    for pos, pp in enumerate(pages):
        if pp.confident:
            continue
        evs.extend(st.ev[pos])
        u = st.unknown[pos]
        if u is not None and (unknown is None or u[0] > unknown[0]):
            unknown = (u[0], pos, u[1])
    base_tag = _lang_of(base)
    cands = [_candidate(s, ev, index, have) for s, ev in _pooled(evs) if ev.tag != base_tag]
    cands.append(_base_candidate(base))
    return LanguagePlan(pages, needed, False, cands, unknown[2] if unknown else None, notes, unavailable)


# --------------------------------------------------------------------------- resolve_unsure (§12.3)

def _shown(detected: str | None) -> str:
    if detected and _CODE_RE.fullmatch(detected):
        return f"language {detected}"
    s = _safe_script(detected)
    return f"{s} script" if s else "this language"


def _record_detected(detected: str | None) -> str | None:
    """``detected`` as record.language_warning takes it: a language code, a plain script name, or None.

    Anything else (a script name pack data or OSD gave in another shape, the word "none") becomes
    None, which the record words as a language it couldn't name (decided 2026-11-05).
    """
    if detected is None or detected == "none":
        return None
    if _CODE_RE.fullmatch(detected) or _SCRIPT_NAME_RE.fullmatch(detected):
        return detected
    return None


def _add(codes: list[str], *new: str) -> list[str]:
    out = list(codes)
    for c in new:
        if c not in out:
            out.append(c)
    return out


def _fallback(pack: str, unavailable: Set[str], broken: Set[str]) -> tuple[str, str, str] | None:
    """Why ``pack`` can't read pages this run, as (warning, composition, offer reason); None if it can.

    A pack the offline run may not fetch is ``pack_not_installed``; one ``flow`` tried and couldn't
    use (its download, hash check or canary failed) is ``pack_broken``. A pack in both was never
    tried, so it is the first.
    """
    if pack in unavailable:
        return "pack_not_installed", "offline", "not_installed"
    if pack in broken:
        return "pack_broken", "broken", "broken"
    return None


def _fallback_why(warning: str, pack: str) -> str:
    if warning == "pack_not_installed":
        return f"pack {pack} isn't installed and this run doesn't download, so the base pack reads it"
    return f"pack {pack} couldn't be used this run (its download or check failed), so the base pack reads it"


def _resolve_page(pp: PagePlan, lp: LanguagePlan, index: "Index", base: "IndexEntry",
                  unavailable: Set[str], broken: Set[str]) -> tuple[PagePlan, LanguageOffer | None]:
    base_tag = _lang_of(base)
    cands = list(pp.candidates) or list(lp.candidates) or [_base_candidate(base)]
    usable = [c for c in cands if c.pack is not None and index.get(c.pack) is not None]
    warns: list[str] = []
    pick: Candidate | None
    if pp.unsure == "no_pack":
        pick = None                                       # detection was sure; the language has no pack
        detected = pp.detected or (cands[0].tag if cands[0].pack is None else None)
    else:
        warns.append("language_uncertain")
        pick = cands[0]
        group = _group_of(pick.tag)
        if group is None and pp.unsure == "close_group" and pp.languages:
            group = _group_of(pp.languages[0].tag)
        if (pick.pack is None or index.get(pick.pack) is None) and group is not None:
            # A close group takes its strongest member that has a published pack (§12.3).
            pick = next((c for c in usable if c.tag in group), pick)
        detected = None
        if pick.pack is None or index.get(pick.pack) is None:
            detected = pick.tag if _CODE_RE.fullmatch(pick.tag) else pp.detected
            pick = None
    detected = _record_detected(detected)

    if pick is None:
        warns.append("no_pack_for_language")
        pack, read_as, composition, reason_kind = base.id, base_tag, "no_pack", "no_pack"
        why = f"no language pack reads {_shown(detected)} yet, so the base pack reads it"
        langs = list(pp.languages)
    else:
        pack, read_as, composition, reason_kind = pick.pack, pick.tag, "strongest", "uncertain"
        why = (f"detection wasn't sure of the language, so the page is read as its strongest candidate, "
               f"{pick.tag} (pack {pick.pack})")
        langs = list(pp.languages) or [LangCall(pick.tag, float(pick.prob or 0.0), "langid", 0)]
        fb = _fallback(pack, unavailable, broken) if pack is not None else None
        if fb is not None:
            warning, composition, reason_kind = fb
            warns.append(warning)
            why += "; " + _fallback_why(warning, pack)
            pack, read_as = base.id, base_tag

    new = replace(pp, pack=pack, composition=composition, languages=langs, reason=why,
                  warnings=_add(pp.warnings, *warns), notes=list(pp.notes), candidates=list(cands),
                  detected=detected if pick is None else _record_detected(pp.detected),
                  via=pp.via or pp.composition)
    # The pack that failed or wasn't installed stays among the options: a later apply --lang fetches
    # and checks it again.
    options = [c for c in usable if c.pack != pack]
    if not options:
        return new, None
    return new, LanguageOffer(reason_kind, [pp.index], pack, read_as, detected if reason_kind == "no_pack" else None,
                              options)


def _fallback_page(pp: PagePlan, index: "Index", base: "IndexEntry",
                   fb: tuple[str, str, str]) -> tuple[PagePlan, LanguageOffer]:
    """A page with a pack that can't read it this run (offline, or broken): the base pack reads it."""
    entry = index.get(pp.pack) if pp.pack else None
    assert entry is not None and pp.pack is not None
    warning, composition, reason_kind = fb
    evidence = ("the pack this page was planned with, not installed" if warning == "pack_not_installed"
                else "the pack this page was planned with, which couldn't be used this run")
    # A page resolve_unsure already read as its strongest candidate keeps offering its candidates,
    # so settling a broken pack in a second call offers what a single call would have.
    usable = [c for c in pp.candidates if c.pack is not None and index.get(c.pack) is not None and c.pack != base.id]
    if any(c.pack == entry.id for c in usable):
        options = usable
    else:
        want = Candidate(_lang_of(entry), entry.id, evidence, _download_bytes(entry, frozenset()))
        options = [want] + [c for c in usable if c.pack != entry.id]
    new = replace(pp, pack=base.id, composition=composition, reason=f"{pp.reason}; {_fallback_why(warning, entry.id)}",
                  warnings=_add(pp.warnings, warning), notes=list(pp.notes), via=pp.via or pp.composition)
    return new, LanguageOffer(reason_kind, [pp.index], base.id, _lang_of(base), None, options)


def _offer_key(o: LanguageOffer) -> tuple[Any, ...]:
    return (o.reason, o.pack, o.read_as, o.detected, tuple((c.tag, c.pack) for c in o.options))


def resolve_unsure(plan: LanguagePlan, index: "Index", *, broken: Set[str] = frozenset()) -> LanguagePlan:
    """Settle every page without asking (DESIGN §12.3, decided 2026-11-05); ``flow`` always calls it.

    * An undecided page is read with ``candidates[0]``'s pack (its own candidates, else the
      plan's), with the warning ``language_uncertain``. A close group takes its strongest member
      that has a published pack.
    * A language (or script) with no pack, or a strongest candidate without one, is read with the
      base pack, with the warning ``no_pack_for_language``; ``PagePlan.detected`` (and the offer's
      ``detected``) names it as ``record.language_warning`` accepts it: a language code, a script
      name, or None when detection named neither.
    * A page whose pack this run may not fetch (``fetch="never"``, ``plan.unavailable``) is read
      with the base pack, with the warning ``pack_not_installed``.
    * A page whose pack is in ``broken`` (pack ids ``flow`` fetched or loaded and couldn't use this
      run: a failed download, hash check or canary) is read with the base pack, with the warning
      ``pack_broken``. This holds for every page with that pack, ``--lang`` and ``--force-lang``
      pages included, and for an unsure page whose strongest candidate's pack it is.

    Each settled page that could be read another way gets an offer, and pages settled the same
    way share one. Pages detection was sure of (with a usable pack), ``--lang`` and ``--force-lang``
    pages (with a usable pack), and ``none`` pages are left as they are, so a second call with the
    same ``broken`` changes nothing. Flow may call it again on a resolved plan with the packs that
    failed: their pages move to the base pack, and earlier offers drop those pages. A changed page
    keeps the composition the plan gave it in ``via``. ``confident`` keeps detection's verdict;
    ``resolved`` is True and ``packs_needed`` lists what will actually read the pages.

    ValueError when the base pack is unavailable or broken: it ships with the install, and a base
    pack that fails its check is a SetupError for the run before this is called.
    """
    base = index.base()
    unavailable = frozenset(plan.unavailable)
    broken = frozenset(broken)
    if base.id in unavailable:
        raise ValueError("the base pack is always installed, so it can't be unavailable")
    if base.id in broken:
        raise ValueError("the base pack can't be passed as broken: that is a SetupError for the run")
    pages: list[PagePlan] = []
    settled: set[int] = set()
    new_offers: list[LanguageOffer] = []
    for pp in plan.pages:
        fb = _fallback(pp.pack, unavailable, broken) if pp.pack is not None else None
        if pp.pack is None and pp.composition == "undecided":
            new, offer = _resolve_page(pp, plan, index, base, unavailable, broken)
        elif fb is not None:
            new, offer = _fallback_page(pp, index, base, fb)
        else:
            pages.append(pp)
            continue
        pages.append(new)
        settled.add(pp.index)
        if offer is not None:
            new_offers.append(offer)
    # Earlier offers (a plan resolved before) lose the pages this call settled again; an offer
    # left with no page goes. New offers settled the same way as one of them join it.
    offers: list[LanguageOffer] = []
    grouped: dict[tuple[Any, ...], LanguageOffer] = {}
    for o in plan.offers:
        kept = [i for i in o.pages if i not in settled]
        if not kept:
            continue
        o = replace(o, pages=kept, options=list(o.options))
        grouped.setdefault(_offer_key(o), o)
        offers.append(o)
    for o in new_offers:
        key = _offer_key(o)
        if key in grouped:
            grouped[key].pages.extend(i for i in o.pages if i not in grouped[key].pages)
        else:
            grouped[key] = o
            offers.append(o)
    for o in offers:
        o.pages.sort()
    warnings = list(plan.warnings)
    for pp in pages:
        warnings = _add(warnings, *(w for w in pp.warnings if w in RESOLVE_WARNINGS))
    needed = sorted({pp.pack for pp in pages if pp.pack is not None})
    return replace(plan, pages=pages, packs_needed=needed, offers=offers, warnings=warnings, resolved=True)
