"""The decision record: the only interface between the CLI and the skill (record_version 1).

Every ``run`` and ``apply`` writes one record per output document to ``<cache>/records/<id>.json``
and prints the same records as ``{"exit": N, "records": [...]}``. The fields are the ones in
``docs/decision-record.md`` (DESIGN §3.2), in that order, and nothing else.

A run never asks (DESIGN §12, decided 2026-11-05). A record carries no question and nothing waits
on it: what a person *could* do next travels as ``offers`` (read a page again with another
language, save a smaller copy), which a person may act on later with ``pdfintake apply``. A run
never exits 10.

Five rules hold for every record:

* **No document text.** A record holds counts, hashes, scores, codes and paths, never what a
  page says. ``to_json`` turns OCR objects (``PageOCR``, ``Line``, ``Fill``) into counts, and
  ``validate`` refuses any object key that carries text (``text``, ``lines``, ``words``, ``near``,
  or any ``*_text`` key), and a ``dropped`` map keyed by anything but ``U+XXXX`` code points.
* **Closed vocabularies.** Warning, note and usage codes come from ``WARNING_CODES``,
  ``NOTE_CODES`` and ``USAGE_CODES``, offer kinds and apply flags from fixed tables, and every
  option id matches ``OPTION_ID_RE``.
* **Offers are acted on with ids only.** ``pdfintake apply <id> <offer.apply_flag> <option.id> --json``.
  The record id matches ``RECORD_ID_RE`` and the option id ``OPTION_ID_RE``, so no file name or
  other record text ever enters a command built from a record. ``load`` accepts a record id and
  nothing else, never a path.
* **Nothing waits.** ``run`` exits 0 whatever offers its records carry; the exit priority over
  several records is 2 > 3 > 1 > 4 > 5 > 0.
* **No password.** No field can hold one: ``options`` drops every ``*password*`` run option (the
  passwords file's path included), ``validate`` refuses any key that names a password, every
  offer and message is fixed core text, and ``save(rec, forbid=passwords)`` refuses a record in
  which a saved password appears in any value that could carry it (not in the record's own id,
  times, paths or fixed wording), without repeating it.

Every word a user sees in an offer (summaries, labels, descriptions) and the default wording of
every warning, note and usage error are written here, in core.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform as _platform
import re
import secrets
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, get_args

from . import __version__
from .errors import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_SETUP,
    EXIT_STALE,
    EXIT_USAGE,
    PdfIntakeError,
    UsageError,
)
from .types import PageOCR, to_jsonable

if TYPE_CHECKING:
    from .flow import RunOptions
    from .integrity import InputSeal

RECORD_VERSION = 1
RECORD_ID_RE = re.compile(r"^r-[0-9]{8}-[0-9a-f]{8}$")
# An option id is "smaller" or a language code; "none" stays valid for an `apply --lang none` in
# history. The provenance and password answers are gone with their decisions (§12.5, decided 2026-11-05).
OPTION_ID_RE = re.compile(r"^(smaller|thorough|none|[a-z]{2,3})$")
ApplyFlag = Literal["--choice", "--lang"]
APPLY_FLAGS: tuple[str, ...] = get_args(ApplyFlag)

# §12.5: the blocking decision became offers, each acted on later (or never) with one flag. §14:
# "thorough" offers the opt-in dots.mocr mode on Apple silicon, acted on with --choice thorough.
OfferKind = Literal["language", "smaller", "thorough"]
OFFER_KINDS: tuple[str, ...] = get_args(OfferKind)
APPLY_FLAG_FOR: Mapping[str, str] = {"language": "--lang", "smaller": "--choice", "thorough": "--choice"}

# "decision" is gone: no record waits for an answer (§12.5). "resolved" is a record an offer on
# which was acted on with apply; its history says what was done.
Status = Literal["done", "refused", "failed", "plan", "resolved"]
STATUSES: tuple[str, ...] = get_args(Status)
STEP_STATUSES: tuple[str, ...] = ("ok", "skipped", "changed", "clear", "failed")
PROVENANCES: tuple[str, ...] = ("ordinary", "counterparty", "evidence")
# "uncertain" replaces "asked" (§12.3): detection wasn't sure, so the strongest candidate read the
# page, with warning language_uncertain and a language offer.
LANGUAGE_PLANS: tuple[str, ...] = ("confident", "uncertain", "hinted", "forced", "not_needed")
# output.kind (§10.2, §11.1): "searchable" for an ordinary output, "searchable_copy" for a copy written
# beside an original that pdfintake never changes, which then carries ``original`` {name, sha256} and
# ``copy_reasons`` (codes from refusals.COPYABLE). Every output says which it is (decided 2026-11-05).
OUTPUT_KINDS: tuple[str, ...] = ("searchable", "searchable_copy")

# Every exit code a record can carry, and the order in which several records decide the process
# exit: 2 > 3 > 1 > 4 > 5 > 0 (§12.5, decided 2026-11-05). Exit 10 is retired: no record waits.
EXIT_PRIORITY: tuple[int, ...] = (EXIT_USAGE, EXIT_SETUP, EXIT_FAILED, EXIT_REFUSED, EXIT_STALE, EXIT_OK)

# ---------------------------------------------------------------------------------- closed codes

WARNING_GROUPS: Mapping[str, tuple[str, ...]] = {
    "readers": ("reader_failed", "reader_timeout", "fallback_used", "agreement_low",
                "no_layer_readers_failed", "vision_language_unsupported",
                # §14: thorough mode (dots.mocr) failed on a page, or couldn't run at all
                "thorough_failed", "thorough_unavailable"),
    "language": ("mixed_language_page", "mixed_script_page", "mixed_script_no_pack", "pack_broken",
                 # §12.3: nothing is asked, so what used to be a question is now a warning
                 "language_uncertain", "no_pack_for_language", "pack_not_installed"),
    "pages": ("existing_layer_replaced", "layer_shrank", "existing_layer_kept", "image_text_not_read",
              "garbled_text", "clip_text_kept"),
    "layer": ("layer_dropped_chars",),
    "orientation": ("orientation_disagree", "not_turned_annotations"),
    "input": ("input_repaired", "contains_jbig2", "revisions_dropped", "objects_dropped", "render_capped",
              "metadata_format_limit"),
    "export": ("exporting_a_copy",),                    # §11.1: export only, an export made from a searchable copy
}
WARNING_CODES: frozenset[str] = frozenset(code for group in WARNING_GROUPS.values() for code in group)
NOTE_CODES: frozenset[str] = frozenset({"input_unchanged", "nothing_written", "vision_not_on_platform",
                                         "pack_fetched", "mixed_script_unchecked", "rotate_kept",
                                         # §10.2, §11.1: copy mode; §12.1: no saved password opened it
                                         "signature_not_carried", "conformance_claim_dropped", "copy_unlocked",
                                         "skipped_no_password",
                                         # §11.2: a page rendered smaller to stay within 32767 px a side
                                         "render_scaled_down"})

# Core's default wording. A caller may pass a more specific core-worded message, never pack text.
WARNING_MESSAGES: Mapping[str, str] = {
    "reader_failed": "One of the readers failed on this page, so its text layer comes from the others.",
    "reader_timeout": "One of the readers took too long on this page and was stopped, so its text layer comes from the others.",
    "fallback_used": "The readers couldn't settle this page, so pdfintake used its fallback reader.",
    "agreement_low": "The readers agreed less than usual on this page, so some words may be wrong.",
    "thorough_failed": "Thorough mode's dots.mocr read failed on this page, so the regular readers read it alone.",
    "thorough_unavailable": "Thorough mode was asked for but can't run here, so the regular readers read every page alone.",
    "no_layer_readers_failed": "Every reader failed on this page, so it has no text layer.",
    "vision_language_unsupported": "Apple Vision can't read this language on this Mac, so the other readers read it alone.",
    "mixed_language_page": "This page mixes two languages. It was read as the main one, so letters only the other uses may be lost.",
    "mixed_script_page": "This page mixes two writing systems, and each was read with its own language pack.",
    "mixed_script_no_pack": "This page has text in a writing system pdfintake has no language pack for yet, so that text may not be searchable.",
    "pack_broken": ("A language pack couldn't be used this run (its download or its check failed), so this page was "
                    "read with the base English pack; letters English doesn't use may be lost."),
    "language_uncertain": ("pdfintake wasn't sure which language this page is in, so it read it as the likeliest one. "
                           "It can read it again as another language if you ask."),
    "no_pack_for_language": ("This page is in a language pdfintake has no language pack for yet, so it was read with "
                             "English; letters English doesn't use may be lost. docs/add-a-language.md says how to add one."),
    "pack_not_installed": ("The language pack for this page isn't installed and this run didn't download anything, so it "
                           "was read with English; letters English doesn't use may be lost."),
    "existing_layer_replaced": "This page had an invisible text layer from another tool, and pdfintake replaced it with its own reading.",
    "layer_shrank": "The new text layer on this page holds less text than the one it replaced.",
    "existing_layer_kept": "This page already has hidden text that can't be removed safely, so pdfintake left the page as it was.",
    "image_text_not_read": "This page has printed text, so pdfintake didn't read it again. Text inside its pictures isn't searchable.",
    "garbled_text": "This page's text can't be read back correctly (its font map is broken), so pdfintake left it as it was. Searching it may not work.",
    "clip_text_kept": "This page uses text as a clipping shape, and pdfintake left it as it was.",
    "layer_dropped_chars": "Some characters couldn't be written into the text layer.",
    "orientation_disagree": "The two orientation checks disagreed on this page, so pdfintake didn't turn it.",
    "not_turned_annotations": "This page looks sideways, but it has annotations, so pdfintake didn't turn it.",
    "input_repaired": "The input was damaged and was repaired while it was read.",
    "contains_jbig2": "The input contains JBIG2 images, which were kept exactly as they are.",
    "revisions_dropped": "The input had earlier saved revisions; the new file keeps only the latest one.",
    "objects_dropped": "Objects the input no longer used were left out of the new file.",
    "render_capped": "This page is very large, so it was read at a lower resolution.",
    "metadata_format_limit": "Some metadata couldn't be carried over exactly in the new file's format.",
    "exporting_a_copy": "This export was made from a searchable copy, not from the original. For evidence, the original is what goes out.",
}
NOTE_MESSAGES: Mapping[str, str] = {
    "input_unchanged": "The input file was not modified.",
    "nothing_written": "Nothing was written, because a new file wouldn't have added a text layer or saved at least 1%.",
    "vision_not_on_platform": "Apple Vision isn't available on this platform.",
    "pack_fetched": "A language pack was downloaded.",
    "mixed_script_unchecked": "pdfintake couldn't yet check this page for text in a second writing system.",
    "rotate_kept": "This page's rotation flag was kept as it was.",
    "signature_not_carried": "The signed original is the one that counts; this copy is for finding it.",
    "conformance_claim_dropped": "The copy no longer claims PDF/A or PDF/UA conformance, because its added text layer would break that claim. The original still does.",
    "copy_unlocked": "The copy has no password, so anyone with the file can read it. The original keeps its protection.",
    "skipped_no_password": ("This file needs a password to open, and none of the saved passwords opened it, so nothing "
                            "was written and the file was left exactly as it is."),
    "render_scaled_down": "This page is too large to read at full size (no side may pass 32767 pixels), so it was read scaled down to fit.",
}

# The usage codes (exit 2) a record can carry that the docs and the skill explain, with core's wording.
# Raise them with ``usage_error(code)``. Some are raised elsewhere with a message of their own that
# names the file (the per-file input codes by flow and integrity, passwords_file_unsafe by
# passwords.py); for those this is core's fallback, true whatever the case.
USAGE_MESSAGES: Mapping[str, str] = {
    "copy_never_overwrites": ("A searchable copy never replaces an existing file, so --overwrite can't be used for it. "
                              "Move or rename the file in the way, or choose another --outdir."),
    # §12.1. passwords.py raises this code with its own message naming the file and the one reason
    # it failed (not private, not a regular file, too large, too many passwords, outside the user
    # folder, …), so that is the wording a user sees. This is core's fallback for the code, and it
    # is true whatever the reason (decided 2026-11-05).
    "passwords_file_unsafe": ("pdfintake won't use the saved passwords file as it is. It must be a regular UTF-8 text "
                              "file in your user folder that only you can read or change (chmod 600), with at most "
                              "100 passwords, one per line."),
    # Per-file input codes (decided 2026-11-06): one input's own record says exit 2, and the rest of
    # the batch carries on. A FIFO, a device such as /dev/stdin or a folder is never opened, because
    # reading one would make the run wait, and a run never waits (DESIGN §12).
    # Windows only: --overwrite (or apply replacing its own output) onto a file another program keeps
    # open, still open after fsutil.retry_busy's patience. fsutil raises it with the file's name.
    "output_in_use": ("The output is open in another program, so pdfintake couldn't replace it. Close it there "
                      "and run again; nothing was written."),
    "input_missing": "This input doesn't exist or can't be read, so nothing was done with it.",
    "input_not_file": ("This input isn't a regular file (it's a folder, a pipe or a device), so pdfintake didn't "
                       "open it."),
    "unsupported_input": "This input isn't a PDF or an image pdfintake can read, so nothing was done with it.",
}
USAGE_CODES: frozenset[str] = frozenset(USAGE_MESSAGES)

# ---------------------------------------------------------------------------------- the dataclasses


@dataclass
class Option:
    id: str
    label: str
    description: str


@dataclass
class Offer:
    """Something a person could do next with a written output (§12.5). Never waited on.

    ``summary`` says what is on offer, in core's words; each option is acted on later, if at all,
    with ``pdfintake apply <record id> <apply_flag> <option id>``.
    """

    kind: OfferKind
    apply_flag: ApplyFlag
    summary: str
    options: list[Option]


@dataclass
class Message:
    code: str
    message: str
    page: int | None = None


@dataclass(kw_only=True)
class Record:
    """One record, field for field as DESIGN §3.2 lists them (and in that order).

    ``pages`` entries may be ``PageOCR`` objects while a run builds the record; ``to_json`` turns
    each into its summary (``page_entry``) so no read's text can reach JSON. A fresh record says
    ``failed`` (exit 1) until the run sets its outcome, so a record left behind by a crash never
    claims success.
    """

    record_version: int = RECORD_VERSION
    id: str
    parent: str | None = None
    tool: dict[str, Any] = field(default_factory=dict)
    created: str
    status: Status = "failed"
    exit: int = EXIT_FAILED
    provenance: Literal["ordinary", "counterparty", "evidence"] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    refusal: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    verify: list[Any] = field(default_factory=list)
    language: dict[str, Any] = field(default_factory=dict)
    pages: list[Any] = field(default_factory=list)
    shrink: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    smaller: dict[str, Any] | None = None
    offers: list[Offer] = field(default_factory=list)
    warnings: list[Message] = field(default_factory=list)
    notes: list[Message] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)


FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Record))

# ---------------------------------------------------------------------------------- building


def new(inputs: Sequence[Path], *, opts: "RunOptions", seal: "InputSeal | None", platform_key: str,
        parent: str | None = None) -> Record:
    """A fresh record for one output document, with ``tool``, ``options`` and ``input`` filled in.

    The run sets ``status``, ``exit`` and the rest. ``input.kind`` and ``input.pages`` start as
    null for the run to fill. ``options`` is ``opts`` as JSON (``out`` and ``outdir`` included as
    strings, so ``apply`` can re-run to the same place).
    """
    if parent is not None and not _is_record_id(parent):
        raise UsageError("bad_record_id", f"{parent!r} isn't a record id", record_id=str(parent))
    if not isinstance(platform_key, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", platform_key):
        raise ValueError(f"bad platform key {platform_key!r}")
    now = datetime.now(timezone.utc)
    record_id = f"r-{now:%Y%m%d}-{secrets.token_hex(4)}"
    options = _options_json(opts)
    provenance = options.get("provenance")
    tool = {
        "version": __version__,
        "platform": platform_key,
        "os": _os_name(),
        "python": _platform.python_version(),
        "engines": _engine_versions(),
    }
    return Record(
        id=record_id,
        parent=parent,
        tool=tool,
        created=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        provenance=provenance,
        options=options,
        input={"paths": [str(Path(p).absolute()) for p in inputs], "kind": None, "pages": None,
               "seal": _seal_json(seal)},
    )


def warning(code: str, page: int | None = None, message: str | None = None) -> Message:
    """A warning with core's default wording (or ``message``); ValueError for an unknown code."""
    if code not in WARNING_CODES:
        raise ValueError(f"unknown warning code {code!r}")
    return Message(code, message if message is not None else WARNING_MESSAGES[code], page)


def note(code: str, page: int | None = None, message: str | None = None) -> Message:
    """A note with core's default wording (or ``message``); ValueError for an unknown code."""
    if code not in NOTE_CODES:
        raise ValueError(f"unknown note code {code!r}")
    return Message(code, message if message is not None else NOTE_MESSAGES[code], page)


def language_warning(code: Literal["language_uncertain", "no_pack_for_language", "pack_not_installed",
                                   "pack_broken"], *,
                     language: str | None, page: int | None = None, others: Sequence[str] = (),
                     read_as: str | None = None) -> Message:
    """One of the §12.3 language warnings, naming the language in core's words.

    ``language`` is, by code:

    * ``language_uncertain``: the likeliest language, a code (``candidates[0]``). When the page
      was read as another language (``read_as``: a close group's member with a pack, or the base
      pack because the likeliest has no pack or this run didn't download it), the message says
      so; ``langplan.resolve_unsure`` then adds ``no_pack_for_language`` or ``pack_not_installed``
      for the reason. ``others`` are the other candidates, strongest first.
    * ``no_pack_for_language``: what detection found with no pack, as ``PagePlan.detected`` and
      ``LanguageOffer.detected`` carry it: a language code, a script name (``"Ethiopic"``: "text in
      the Ethiopic writing system"), or None when detection named neither (core's default wording).
    * ``pack_not_installed``: the language code of the pack that wasn't installed.
    * ``pack_broken``: the language code of the pack that couldn't be used this run (its download,
      hash check or canary failed), when the base pack read the page instead (``resolve_unsure``'s
      ``broken``); ``warning("pack_broken")`` is the general wording, which covers a failed download
      as well as a failed check (decided 2026-11-06).

    ``others`` and ``read_as`` are used only for ``language_uncertain``. ValueError for a bad code,
    language or script name: a value that is neither never reaches core's wording.
    """
    if code == "no_pack_for_language":
        if language is None:
            return warning(code, page)
        what = _detected_phrase(language)
        message = (f"This page {what}, and pdfintake has no language pack for it yet, so it was read with English; "
                   f"letters English doesn't use may be lost. docs/add-a-language.md says how to add one.")
        return warning(code, page, message)
    if language is None:
        raise ValueError(f"{code!r} names the language")
    name = language_name(language)
    if code == "language_uncertain":
        names = [language_name(c) for c in others if c != language]
        if read_as is None or read_as == language:
            also = f" It could also be {_or_list(names)}." if names else ""
            message = (f"pdfintake wasn't sure which language this page is in, so it read it as {name}, the likeliest."
                       f"{also} It can read it again as another language if you ask.")
        else:
            also = f"; it could also be {_or_list(names)}" if names else ""
            message = (f"pdfintake wasn't sure which language this page is in. The likeliest is {name}{also}. "
                       f"It was read as {language_name(read_as)}, and it can read it again as another language "
                       f"if you ask.")
    elif code == "pack_not_installed":
        message = (f"The {name} language pack isn't installed and this run didn't download anything, so this page "
                   f"was read with English; letters English doesn't use may be lost.")
    elif code == "pack_broken":
        message = (f"The {name} language pack couldn't be used this run (its download or its check failed), so this "
                   f"page was read with the base English pack; letters English doesn't use may be lost. It can read "
                   f"it again with the {name} pack if you ask.")
    else:
        raise ValueError(f"{code!r} isn't a language warning")
    return warning(code, page, message)


# A script name as langplan words it (``langplan._SCRIPT_NAME_RE``; test_record keeps the two equal):
# a core script or a plain capitalised word such as "Ethiopic", never pack text with spaces or marks.
_SCRIPT_NAME_RE = re.compile(r"[A-Z][A-Za-z]{1,19}")


def _detected_phrase(detected: str) -> str:
    """What detection found with no pack, as the rest of "This page …": a language or a writing system."""
    if isinstance(detected, str) and _SCRIPT_NAME_RE.fullmatch(detected):
        return f"has text in the {detected} writing system"
    return f"looks like {language_name(detected)}"


def usage_error(code: str, **detail: Any) -> UsageError:
    """A ``UsageError`` (exit 2) with core's wording for ``code``; ValueError for an unknown code."""
    if code not in USAGE_CODES:
        raise ValueError(f"unknown usage code {code!r}")
    return UsageError(code, USAGE_MESSAGES[code], **detail)


def output_entry(path: Path | str, sha256: str, nbytes: int, *, original: Path | str | None = None,
                 original_sha256: str | None = None, copy_reasons: Sequence[str] = ()) -> dict[str, Any]:
    """The §3.2 ``output`` object. With ``original``, a searchable copy (§10.2, §11.1).

    A copy names its original by file name only (never its folder) and its SHA-256, and says why
    it is a copy with codes from ``refusals.COPYABLE``. Without ``original`` the output is an
    ordinary ``searchable`` file and ``copy_reasons`` must be empty. ValueError otherwise.
    """
    out: dict[str, Any] = {"path": str(path), "sha256": str(sha256), "bytes": int(nbytes)}
    if original is None:
        if copy_reasons or original_sha256 is not None:
            raise ValueError("only a searchable copy has an original and copy reasons")
        out["kind"] = "searchable"
        return out
    if not copy_reasons:
        raise ValueError("a searchable copy says why it is a copy")
    if not (isinstance(original_sha256, str) and _SHA256_RE.match(original_sha256)):
        raise ValueError("a searchable copy names its original's SHA-256")
    out["kind"] = "searchable_copy"
    out["original"] = {"name": PurePath(str(original)).name, "sha256": original_sha256}
    out["copy_reasons"] = list(copy_reasons)
    return out


def page_entry(ocr: PageOCR, *, page_class: str | None = None, dpi: float | None = None,
               orientation: Mapping[str, Any] | None = None, retention: float | None = None) -> dict[str, Any]:
    """The §3.2 ``pages[]`` summary of one OCR'd page: counts, scores and codes, never its text.

    ``page`` counts from 1 (``PageOCR.index`` counts from 0). The pick loses its lines, fills
    become counts by kind, and each read keeps only its id, engine, spec, role, status, time,
    error and the pick's score for it.
    """
    scores = ocr.pick.scores or {}
    readers = []
    for read in ocr.reads:
        readers.append({
            "use_id": read.use_id, "engine": read.engine, "spec": read.spec, "role": read.role,
            "status": read.status, "seconds": round(float(read.seconds), 3),
            "error": None if read.error is None else str(read.error)[:200],
            "score": scores.get(read.use_id),
        })
    s = ocr.script
    return {
        "page": ocr.index + 1,
        "class": page_class,
        "dpi": dpi,
        "orientation": dict(orientation) if orientation is not None else None,
        "script": {"name": s.name, "conf": s.conf, "by": s.by, "eng_mean_conf": s.eng_mean_conf},
        "languages": [{"tag": c.tag, "prob": c.prob, "by": c.by, "windows": c.windows} for c in ocr.languages],
        "pack": ocr.pack,
        "composition": ocr.composition,
        "readers": readers,
        "pick": {"rule": ocr.pick.rule, "version": ocr.pick.version, "chosen": ocr.pick.chosen,
                 "fallback": ocr.pick.fallback},
        "fills": dict(sorted(Counter(f.kind for f in ocr.fills).items())),
        "repairs": dict(ocr.repairs),
        "must_survive": {k: dict(v) for k, v in ocr.must_survive.items()},
        "retention": retention,
        "warnings": list(ocr.warnings),
    }


# ---------------------------------------------------------------------------------- offers

# A smaller copy is offered only at this saving or more, with no byte floor (DESIGN Decisions, Q3).
SMALLER_MIN_SAVING_PCT = 30.0
# The staged candidate is kept this long; after that, acting on the offer rebuilds it (§12.4).
STAGED_DAYS = 14


def smaller_offer(*, output: Path | str, inputs: Sequence[Path | str], before_bytes: int, after_bytes: int,
                  saving_pct: float, min_score: float, bar: float, beside_name: str | None = None) -> Offer:
    """The ``smaller`` offer: a smaller copy could be saved beside the lossless output (§12.4).

    Nothing is written for it during a run; ``--smaller`` on ``run`` writes the copy without an
    offer instead. ``beside_name`` defaults to ``"<output stem> (smaller).pdf"``. ValueError when
    the saving is under ``SMALLER_MIN_SAVING_PCT`` or the worst score is under the bar: such a
    copy is never offered.
    """
    if not (saving_pct >= SMALLER_MIN_SAVING_PCT):
        raise ValueError(f"a smaller copy is offered only at a saving of {SMALLER_MIN_SAVING_PCT:g}% or more")
    if not (min_score >= bar):
        raise ValueError("a smaller copy is offered only when every changed page scores at or above the bar")
    out_name = PurePath(str(output)).name
    beside = beside_name or f"{PurePath(out_name).stem} (smaller).pdf"
    in_names = [PurePath(str(p)).name for p in inputs]
    summary = (
        f"A smaller copy could be saved as well: {math.floor(saving_pct):d}% smaller "
        f"({_size(before_bytes)} to {_size(after_bytes)}). Its most detailed part scores "
        f"{math.floor(min_score):d} out of 100; at {math.floor(bar):d} or above, a difference shows only "
        f"when you flick between the two."
    )
    if len(in_names) == 1 and in_names[0] != out_name:
        stays = f"{out_name} and {in_names[0]} stay as they are."
    elif in_names and in_names != [out_name]:
        stays = f"{out_name} and the original files stay as they are."
    else:
        stays = f"{out_name} stays as it is."
    return Offer("smaller", "--choice", summary, [
        Option("smaller", "Save the smaller copy too", f"Saves {beside} next to it. {stays}"),
    ])


def thorough_offer() -> Offer:
    """The ``thorough`` offer (§14): this Mac could read the document again in thorough mode (dots.mocr
    beside the regular readers). Its words come from ``thorough.offer_text``: what the mode requires,
    what it costs and what it gives, so the skill can put it to the user as it is."""
    from . import thorough

    summary, label, description = thorough.offer_text()
    return Offer("thorough", "--choice", summary, [Option("thorough", label, description)])


# Why a language offer exists: langplan.OFFER_REASONS, the three ways resolve_unsure settles a page
# without asking (§12.3, decided 2026-11-05). test_record keeps the two tuples equal.
LANGUAGE_OFFER_REASONS: tuple[str, ...] = ("uncertain", "no_pack", "not_installed", "broken")


def language_offer(*, used: str, others: Sequence[tuple[str, int | None]], pages: Sequence[int] = (),
                   reason: str = "uncertain", detected: str | None = None) -> Offer:
    """The ``language`` offer: pages could be read again with another language pack (§12.3).

    Built from a ``langplan.LanguageOffer``: ``reason`` is its ``reason``, ``used`` its
    ``read_as``, ``others`` its options as (``tag``, ``download_bytes``), ``detected`` its
    ``detected`` and ``pages`` its pages counted from 1. By ``reason``:

    * ``uncertain``: detection wasn't sure, so the strongest candidate ``used`` read the pages.
    * ``no_pack``: the language detection found (``detected``: a code, a script name, or None
      when it named neither) has no pack yet, so the base pack's language ``used`` read them.
    * ``not_installed``: the pack the pages need isn't installed and the run didn't download
      anything (``--fetch never``), so the base pack's language ``used`` read them.
    * ``broken``: the pack the pages need couldn't be used this run (its download, hash check or
      canary failed), so the base pack's language ``used`` read them.

    ``others`` are the packs to read them again with, strongest first, as (language code, download
    bytes, or None or 0 when the pack is installed); ``used`` among them is skipped. Empty
    ``pages`` means the whole document. ValueError for an unknown reason, ``detected`` with a
    reason other than ``no_pack``, a bad or repeated code, or when there is no other language
    (then there is nothing to offer).
    """
    if reason not in LANGUAGE_OFFER_REASONS:
        raise ValueError(f"{reason!r} isn't one of {', '.join(LANGUAGE_OFFER_REASONS)}")
    if detected is not None and reason != "no_pack":
        raise ValueError("only a no_pack offer names what detection found")
    what = _detected_phrase(detected) if detected is not None else None   # checks it before any wording
    where = _pages_phrase(pages)
    one = len(set(int(x) for x in pages)) == 1
    used_name = language_name(used)
    options: list[Option] = []
    names: list[str] = []
    seen: set[str] = {used}
    for code, size in others:
        if not _is_lang_code(code):
            raise ValueError(f"bad language code {code!r}")
        if code == used:
            continue
        if code in seen:
            raise ValueError(f"repeated language code {code!r}")
        seen.add(code)
        name = language_name(code)
        if size:
            label = f"{name} (download {_size(size)})"
            desc = f"Reads {where.lower()} again with the {name} language pack, downloading it once ({_size(size)})."
        else:
            label = name
            desc = f"Reads {where.lower()} again with the {name} language pack, already installed."
        options.append(Option(code, label, desc))
        names.append(name)
    if not options:
        raise ValueError("a language offer needs another language to offer")
    was, they = ("was", "it") if one else ("were", "they")
    again = f"{where} can be read again as {_or_list(names)} if you ask."
    if reason == "uncertain":
        summary = (f"{where} {was} read as {used_name}, the likeliest language, but pdfintake wasn't sure. "
                   f"{again}")
    elif reason == "no_pack":
        if what is None:
            found = f"{'is' if one else 'are'} in a language pdfintake has no language pack for yet"
        else:
            # "looks like Swahili" / "has text in the Ethiopic writing system", for one page or several
            plural = what.replace("looks like", "look like", 1).replace("has text", "have text", 1)
            found = f"{what if one else plural}, and pdfintake has no language pack for it yet"
        summary = f"{where} {found}, so {they} {was} read as {used_name}. {again}"
    else:                                               # not_installed, broken
        why = ("isn't installed and this run didn't download anything" if reason == "not_installed"
               else "couldn't be used this run (its download or check failed)")
        summary = (f"{where} {was} read as {used_name}, because the language pack {they} "
                   f"need{'s' if one else ''} {why}. {again}")
    return Offer("language", "--lang", summary, options)


LANGUAGE_NAMES: Mapping[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian",
    "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fil": "Filipino", "fr": "French", "ga": "Irish",
    "gl": "Galician", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "lo": "Lao",
    "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese", "nb": "Norwegian Bokmål",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "pa": "Punjabi",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "zh": "Chinese",
}


def language_name(code: str) -> str:
    """The English name of an ISO 639 code, or the code itself in quotes when core has no name for it."""
    if code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]
    if isinstance(code, str) and _is_lang_code(code):
        return f"the language with code '{code}'"
    raise ValueError(f"bad language code {code!r}")


# ---------------------------------------------------------------------------------- JSON


def to_json(rec: Record) -> dict[str, Any]:
    """The record as plain JSON values, fields in §3.2 order. Document text never survives."""
    out: dict[str, Any] = {}
    for name in FIELDS:
        value = getattr(rec, name)
        if name == "pages":
            value = [page_entry(p) if isinstance(p, PageOCR) else p for p in value]
        out[name] = to_jsonable(value)
    return out


def save(rec: Record, *, forbid: Sequence[str | bytes] = ()) -> Path:
    """Write ``<cache>/records/<id>.json`` atomically (temp file, fsync, replace) and return its path.

    The record is validated first; an invalid record is never written (PdfIntakeError ``record_invalid``).
    ``forbid`` takes the saved passwords the run tried (§12.1): a record in which one appears in a
    value that could carry it is refused, and the error never repeats it (see ``validate``).
    """
    data = to_json(rec)
    problems = validate(data, forbid=forbid)
    if problems:
        raise PdfIntakeError("record_invalid", "the record is invalid: " + "; ".join(problems[:8]), problems=problems)
    folder = _records_dir()
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{rec.id}.json"
    fd, tmp = tempfile.mkstemp(prefix=f".{rec.id}.", suffix=".partial", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        from . import fsutil

        fsutil.replace_retrying(tmp, dest)               # a reader on Windows holds it only briefly
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def load(record_id: str) -> Record:
    """The saved record ``record_id``. Only a record id is accepted, never a path (UsageError, exit 2).

    ``unknown_record`` when there is none; ``record_invalid`` when its file can't be read (a folder
    in its place, no permission, bytes that aren't UTF-8), isn't JSON, or doesn't validate.
    """
    if not _is_record_id(record_id):
        shown = repr(record_id)[:80]
        raise UsageError("bad_record_id", f"{shown} isn't a record id (they look like r-20261105-1a2b3c4d)",
                         record_id=str(record_id)[:80])
    path = _records_dir() / f"{record_id}.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UsageError("unknown_record", f"there is no record {record_id}; it may have been pruned, so run again",
                         record_id=record_id) from None
    except (OSError, UnicodeDecodeError):
        # A folder in its place, a file that can't be opened, or bytes that aren't UTF-8 (decided 2026-11-06).
        raise UsageError("record_invalid", f"record {record_id} can't be read; run again", record_id=record_id) from None
    try:
        data = json.loads(text)
    except ValueError:
        raise UsageError("record_invalid", f"record {record_id} can't be read; run again", record_id=record_id) from None
    problems = validate(data)
    if not problems and data.get("id") != record_id:
        problems = [f"the file holds record {data.get('id')!r}"]
    if problems:
        raise UsageError("record_invalid", f"record {record_id} is invalid ({problems[0]}); run again",
                         record_id=record_id, problems=problems)
    return _from_json(data)


def overall_exit(records: Sequence[Record]) -> int:
    """The process exit for several records: the highest-priority code, 2 > 3 > 1 > 4 > 5 > 0 (§12.5); a record never carries 10."""
    codes = [r.exit for r in records]
    for code in codes:
        if code not in EXIT_PRIORITY:
            raise ValueError(f"a record can't carry exit code {code!r}")
    for code in EXIT_PRIORITY:
        if code in codes:
            return code
    return EXIT_OK


# ---------------------------------------------------------------------------------- validation

_TEXT_KEYS = frozenset({"text", "lines", "words", "near", "expected_text"})
# The classify step counts pages by class (pdf/classify.py PageKind). Two class names end in
# "_text", so they are allowed as keys of those counts, and nowhere else.
PAGE_KINDS: frozenset[str] = frozenset({"image_only", "invisible_layer", "our_layer", "hidden_text", "born_digital",
                                        "garbled", "clip_text", "empty"})
_CODEPOINT_RE = re.compile(r"^U\+[0-9A-F]{4,6}$")
_KNOWN: Mapping[str, frozenset[str]] = {
    "tool": frozenset({"version", "platform", "os", "python", "engines"}),
    "input": frozenset({"paths", "kind", "pages", "seal"}),
    "seal": frozenset({"sha256", "sizes", "mtimes_ns"}),
    "output": frozenset({"path", "sha256", "bytes", "kind", "original", "copy_reasons"}),
    "original": frozenset({"name", "sha256"}),
    "refusal": frozenset({"code", "message", "detectors", "repaired"}),
    "step": frozenset({"step", "status", "detail"}),
    "check": frozenset({"name", "ok", "detail", "page"}),
    "language": frozenset({"plan", "packs", "canaries", "detection"}),
    "page": frozenset({"page", "class", "dpi", "orientation", "script", "languages", "pack", "composition",
                       "readers", "pick", "fills", "repairs", "must_survive", "retention", "warnings"}),
    "smaller": frozenset({"offered", "candidate", "saving", "metric", "bar", "min_score", "pages", "kept_lossless",
                          "text_layer_identical", "metadata", "applied", "written"}),
    "offer": frozenset({"kind", "apply_flag", "summary", "options"}),
    "option": frozenset({"id", "label", "description"}),
    "message": frozenset({"code", "message", "page"}),
    "history": frozenset({"at", "flag", "answer", "result", "child"}),
}
_KIND_OPTIONS: Mapping[str, frozenset[str] | None] = {
    "smaller": frozenset({"smaller"}),
    "thorough": frozenset({"thorough"}),
    "language": None,                                   # a language code
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate(data: Mapping[str, Any], *, forbid: Sequence[str | bytes] = ()) -> list[str]:
    """Every problem with a record's JSON, as short sentences; empty when it is valid.

    Checks the field set (missing and unknown fields), the id patterns, the closed vocabularies
    (status, provenance, offer kinds, apply flags, option ids, warning, note and refusal codes),
    the shape of the structured objects, that no object key anywhere carries document text, and
    that no key anywhere names a password.

    ``forbid`` lists secrets (the saved passwords the run tried, §12.1) that must not appear in any
    value that could carry one: step, verify and refusal text, custom messages, reader errors, and keys that aren't
    plain names, wherever they occur inside the string. Fields that existed before the password
    was read or hold only fixed values (the id and times, the paths, file names and hashes, closed
    codes, core's default wording; the list is above ``_secret_hits``) aren't scanned, and an
    occurrence inside one of the record's own paths or file names doesn't count, so a PIN that is
    also the year, or account digits in the file's name, never make a record unsaveable
    (decided 2026-11-05). A secret under four characters isn't scanned for at all: it can't be told
    apart from the record's own words ("ok", "pdf", "en"). Such a password still can't reach a
    record, because no field takes one (no ``*password*`` key, no password in ``options``, and every
    offer and message is fixed core text). When a secret is found, that is the only problem
    reported, and no problem ever repeats a secret or the text around it.
    """
    if not isinstance(data, Mapping):
        return ["a record is a JSON object"]
    forbidden = _forbidden(forbid)
    secrets_found = _secret_hits(data, forbidden)
    if secrets_found:
        return [f"a record never holds a password ({secrets_found} place{'s' if secrets_found != 1 else ''})"]
    p: list[str] = []
    keys = set(data)
    missing = [f for f in FIELDS if f not in keys]
    unknown = sorted(str(k) for k in keys - set(FIELDS))
    if missing:
        p.append("missing fields: " + ", ".join(missing))
    if unknown:
        p.append("unknown fields: " + ", ".join(unknown))
    g = data.get
    if g("record_version") != RECORD_VERSION or isinstance(g("record_version"), bool):
        p.append(f"record_version must be {RECORD_VERSION}")
    if not _is_record_id(g("id")):
        p.append(f"id {_short(g('id'))} doesn't match {RECORD_ID_RE.pattern}")
    if g("parent") is not None and not _is_record_id(g("parent")):
        p.append(f"parent {_short(g('parent'))} doesn't match {RECORD_ID_RE.pattern}")
    if "tool" in keys:
        _check_object(p, "tool", g("tool"), "tool", required=_KNOWN["tool"])
        if isinstance(g("tool"), Mapping) and not isinstance(g("tool").get("engines", {}), Mapping):
            p.append("tool.engines must be an object")
    if "created" in keys and not (isinstance(g("created"), str) and _is_iso_utc(g("created"))):
        p.append("created must be a UTC ISO 8601 time")
    if "status" in keys and not (isinstance(g("status"), str) and g("status") in STATUSES):
        p.append(f"status {_short(g('status'))} isn't one of {', '.join(STATUSES)}")
    if "exit" in keys and (isinstance(g("exit"), bool) or not isinstance(g("exit"), int)
                           or g("exit") not in EXIT_PRIORITY):
        p.append(f"exit {_short(g('exit'))} isn't a record exit code")
    if "provenance" in keys and g("provenance") is not None and not (isinstance(g("provenance"), str)
                                                                     and g("provenance") in PROVENANCES):
        p.append(f"provenance {_short(g('provenance'))} isn't one of {', '.join(PROVENANCES)}")
    if "options" in keys and not isinstance(g("options"), Mapping):
        p.append("options must be an object")
    if "input" in keys:
        _check_input(p, g("input"))
    if g("output") is not None:
        _check_output(p, g("output"))
    if g("refusal") is not None:
        _check_refusal(p, g("refusal"))
    for i, step in enumerate(_list(p, "steps", g("steps"), "steps" in keys)):
        where = f"steps[{i}]"
        if _check_object(p, where, step, "step", required=frozenset({"step", "status"})):
            if not isinstance(step.get("status"), str) or step.get("status") not in STEP_STATUSES:
                p.append(f"{where}.status {_short(step.get('status'))} isn't one of {', '.join(STEP_STATUSES)}")
    for i, check in enumerate(_list(p, "verify", g("verify"), "verify" in keys)):
        where = f"verify[{i}]"
        if _check_object(p, where, check, "check", required=frozenset({"name", "ok"})):
            if not isinstance(check.get("ok"), bool):
                p.append(f"{where}.ok must be true or false")
    if "language" in keys:
        lang = g("language")
        if _check_object(p, "language", lang, "language"):
            if "plan" in lang and lang["plan"] is not None and not (isinstance(lang["plan"], str)
                                                                and lang["plan"] in LANGUAGE_PLANS):
                p.append(f"language.plan {_short(lang['plan'])} isn't one of {', '.join(LANGUAGE_PLANS)}")
    for i, page in enumerate(_list(p, "pages", g("pages"), "pages" in keys)):
        _check_page(p, f"pages[{i}]", page)
    for name in ("shrink", "metadata"):
        if name in keys and not isinstance(g(name), Mapping):
            p.append(f"{name} must be an object")
    if g("smaller") is not None:
        _check_object(p, "smaller", g("smaller"), "smaller")
    offers = _list(p, "offers", g("offers"), "offers" in keys)
    for i, offer in enumerate(offers):
        _check_offer(p, f"offers[{i}]", offer)
    if offers:
        _check_offers_fit(p, offers, g("output"))
    for name, codes in (("warnings", WARNING_CODES), ("notes", NOTE_CODES)):
        for i, msg in enumerate(_list(p, name, g(name), name in keys)):
            where = f"{name}[{i}]"
            if _check_object(p, where, msg, "message", required=frozenset({"code", "message"})):
                if not isinstance(msg.get("code"), str) or msg.get("code") not in codes:
                    p.append(f"{where}.code {_short(msg.get('code'))} isn't a known {name[:-1]} code")
                if not isinstance(msg.get("message"), str):
                    p.append(f"{where}.message must be a string")
                _check_page_no(p, f"{where}.page", msg.get("page"))
    for i, entry in enumerate(_list(p, "history", g("history"), "history" in keys)):
        where = f"history[{i}]"
        if _check_object(p, where, entry, "history", required=_KNOWN["history"]):
            if not isinstance(entry.get("flag"), str) or entry.get("flag") not in APPLY_FLAGS:
                p.append(f"{where}.flag {_short(entry.get('flag'))} isn't one of {', '.join(APPLY_FLAGS)}")
            if not _is_option_id(entry.get("answer")):
                p.append(f"{where}.answer {_short(entry.get('answer'))} doesn't match {OPTION_ID_RE.pattern}")
            child = entry.get("child")
            if child is not None and not _is_record_id(child):
                p.append(f"{where}.child {_short(child)} doesn't match {RECORD_ID_RE.pattern}")
    _find_text_keys(p, data, "")
    # A field the scan leaves out can still be echoed by a problem above (an unknown key, say), so a
    # problem never repeats a secret either.
    return [_redact(x, forbidden) for x in p] if forbidden else p


_MIN_SCANNED_SECRET = 4


def _is_password_key(key: Any) -> bool:
    return (isinstance(key, str) and "password" in key.lower() and key not in NOTE_CODES
            and key not in WARNING_CODES and key not in USAGE_CODES)


def _forbidden(forbid: Sequence[str | bytes]) -> list[str]:
    if isinstance(forbid, (str, bytes)):
        forbid = [forbid]
    out = []
    for secret in forbid:
        if isinstance(secret, bytes):
            secret = secret.decode("utf-8", "replace")
        if isinstance(secret, str) and len(secret) >= _MIN_SCANNED_SECRET:   # "" is a permissions-only file
            out.append(secret)
    return out


# The password scan looks only where a value could carry one (decided 2026-11-05). A short password
# such as a 4-digit PIN is often also the year, or the account digits a statement's file name uses,
# so a scan of every string would match the record's own id, time, paths or wording, and the file
# could never be processed. Left out of the scan, because they existed before the password was read
# or can only hold a fixed value:
#   * the id, parent and created time, and every history entry's time, flag, answer and child,
#     each only while it matches its own pattern (a value that doesn't is scanned, and reported);
#   * tool and options (written from the platform and argv, which never carries the password),
#     input (the paths as given, the kind, the seal) and output (path, hash, kind, the original's
#     name and hash, the copy reasons), language.packs and smaller's candidate, written and metric;
#   * closed codes (status, provenance, step status, refusal, warning and note codes), any SHA-256
#     digest or record id, and an offer of a known kind, whose words core writes from counts,
#     language names and the record's own file names;
#   * a warning or note whose message is core's default wording for its code;
#   * keys that are code's own names (snake_case identifiers, U+XXXX code points).
# Everything else is scanned: step, verify and refusal text, custom messages, reader errors, any
# key that isn't a plain name. Even there, an occurrence that lies inside one of the record's own
# paths, file names, id or time doesn't count: it is that path, not the password. Nor does a whole
# word that core's own wording uses (``_core_words``).

_IDENT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ISO_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|\+00:00)$")


def _secret_hits(data: Mapping[str, Any], forbidden: list[str]) -> int:
    if not forbidden:
        return 0
    covers = _own_strings(data)

    def hit(text: str) -> bool:
        return any(_uncovered(text, s, covers) for s in forbidden)

    def count(value: Any) -> int:
        if isinstance(value, Mapping):
            n = 0
            for k, v in value.items():
                if isinstance(k, str) and not _plain_key(k) and hit(k):
                    n += 1
                n += count(v)
            return n
        if isinstance(value, (list, tuple)):
            return sum(count(v) for v in value)
        if isinstance(value, str) and not (_SHA256_RE.match(value) or _is_record_id(value)):
            return int(hit(value))
        return 0

    return count(_scan_view(data))


def _plain_key(key: str) -> bool:
    return bool(_IDENT_KEY_RE.match(key) or _CODEPOINT_RE.match(key))


def _uncovered(text: str, secret: str, covers: Sequence[str]) -> bool:
    """True when ``secret`` occurs in ``text`` somewhere that isn't inside one of ``covers``."""
    start = text.find(secret)
    while start != -1:
        end = start + len(secret)
        inside = False
        for c in covers:
            if len(c) < len(secret):
                continue
            j = text.find(c, max(0, end - len(c)))
            while j != -1 and j <= start:
                if j + len(c) >= end:
                    inside = True
                    break
                j = text.find(c, j + 1)
            if inside:
                break
        if not inside and _is_core_word(text, start, end):
            inside = True
        if not inside:
            return True
        start = text.find(secret, start + 1)
    return False


def _is_core_word(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is a whole word that core's own wording uses ("password", "copy")."""
    if (start > 0 and text[start - 1].isalnum()) or (end < len(text) and text[end].isalnum()):
        return False
    return text[start:end].casefold() in _core_words()


_CORE_WORDS: frozenset[str] | None = None


def _core_words() -> frozenset[str]:
    """Every word of four letters or more in core's fixed wording, casefolded.

    A password that is one of these words can't be told apart from core's own sentences in step,
    verify and refusal text ("saved without a password", "a searchable copy"), so a whole-word
    occurrence of it isn't counted, as a password under four characters isn't scanned for at all
    (decided 2026-11-05). A password with anything more in it ("password1") is still found.
    """
    global _CORE_WORDS
    if _CORE_WORDS is None:
        texts: list[str] = [*WARNING_MESSAGES.values(), *NOTE_MESSAGES.values(), *USAGE_MESSAGES.values(),
                            *LANGUAGE_NAMES.values()]
        try:
            from .refusals import COPYABLE, copy_message
        except ImportError:                             # pragma: no cover - refusals is part of core
            pass
        else:
            for code in sorted(COPYABLE):
                texts.append(copy_message(code))
        _CORE_WORDS = frozenset(w.casefold() for t in texts for w in re.findall(r"[^\W\d_]{4,}", t))
    return _CORE_WORDS


def _own_strings(data: Mapping[str, Any]) -> list[str]:
    """The record's own id, times, paths and file names (each path also by its name, stem and folder)."""
    out: set[str] = set()

    def add_path(value: Any, *, folder: bool = False) -> None:
        if isinstance(value, str) and value:
            out.add(value)
            if folder:                                  # a folder by its whole path only, never its bare name
                return
            pp = PurePath(value)
            for part in (pp.name, pp.stem, str(pp.parent)):
                if part and part != ".":
                    out.add(part)

    def add_valid(value: Any, ok: Any) -> None:       # an id or a time covers only while it is one
        if isinstance(value, str) and ok(value):
            out.add(value)

    add_valid(data.get("id"), _is_record_id)
    add_valid(data.get("parent"), _is_record_id)
    add_valid(data.get("created"), _ISO_UTC_RE.match)
    inp = data.get("input")
    if isinstance(inp, Mapping) and isinstance(inp.get("paths"), list):
        for p in inp["paths"]:
            add_path(p)
    outp = data.get("output")
    if isinstance(outp, Mapping):
        add_path(outp.get("path"))
        if isinstance(outp.get("original"), Mapping):
            add_path(outp["original"].get("name"))
    opts = data.get("options")
    if isinstance(opts, Mapping):
        add_path(opts.get("out"))
        add_path(opts.get("outdir"), folder=True)
    sm = data.get("smaller")
    if isinstance(sm, Mapping):
        add_path(sm.get("written"))
        if isinstance(sm.get("candidate"), Mapping):
            add_path(sm["candidate"].get("path"))
    hist = data.get("history")
    if isinstance(hist, list):
        for entry in hist:
            if isinstance(entry, Mapping):
                add_valid(entry.get("at"), _ISO_UTC_RE.match)
                add_valid(entry.get("child"), _is_record_id)
    return sorted(out, key=len, reverse=True)


def _blank(value: Any) -> Any:
    """``value`` with its keys kept and every leaf value gone."""
    if isinstance(value, Mapping):
        return {k: _blank(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_blank(v) for v in value]
    return None


def _clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone(v) for v in value]
    return value


def _scan_view(data: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the record with every field the password scan leaves out emptied (see above)."""
    view: dict[str, Any] = _clone(data)

    def drop(obj: Any, key: str, ok: Any = None) -> None:
        if isinstance(obj, dict) and key in obj and (ok is None or ok(obj[key])):
            obj[key] = None

    def is_iso(v: Any) -> bool:
        return isinstance(v, str) and _ISO_UTC_RE.match(v) is not None

    def among(values: Any) -> Any:
        return lambda v: isinstance(v, str) and v in values

    drop(view, "id", _is_record_id)
    drop(view, "parent", _is_record_id)
    drop(view, "created", is_iso)
    drop(view, "status", among(STATUSES))
    drop(view, "provenance", among(PROVENANCES))
    for name in ("tool", "options"):
        if isinstance(view.get(name), dict):
            view[name] = _blank(view[name])
    inp = view.get("input")
    if isinstance(inp, dict):
        if isinstance(inp.get("paths"), list):
            inp["paths"] = _blank(inp["paths"])
        drop(inp, "kind", lambda v: isinstance(v, str) and re.fullmatch(r"[a-z_]{1,16}", v) is not None)
        if isinstance(inp.get("seal"), dict):
            inp["seal"] = _blank(inp["seal"])
    out = view.get("output")
    if isinstance(out, dict):
        drop(out, "path", lambda v: isinstance(v, str))
        drop(out, "kind", among(OUTPUT_KINDS))
        orig = out.get("original")
        if isinstance(orig, dict):
            drop(orig, "name", _is_file_name)
        reasons = out.get("copy_reasons")
        if isinstance(reasons, list):
            known = _copyable_codes()
            out["copy_reasons"] = [None if isinstance(c, str) and c in known else c for c in reasons]
    ref = view.get("refusal")
    if isinstance(ref, dict):
        drop(ref, "code", among(_refusal_codes()))
    for step in view.get("steps") if isinstance(view.get("steps"), list) else ():
        if isinstance(step, dict):
            drop(step, "step", lambda v: isinstance(v, str) and _IDENT_KEY_RE.match(v) is not None)
            drop(step, "status", among(STEP_STATUSES))
    for check in view.get("verify") if isinstance(view.get("verify"), list) else ():
        if isinstance(check, dict):
            drop(check, "name", lambda v: isinstance(v, str) and _IDENT_KEY_RE.match(v) is not None)
    lang = view.get("language")
    if isinstance(lang, dict):
        drop(lang, "plan", among(LANGUAGE_PLANS))
        if isinstance(lang.get("packs"), list):
            lang["packs"] = _blank(lang["packs"])
    sm = view.get("smaller")
    if isinstance(sm, dict):
        drop(sm, "written", lambda v: isinstance(v, str))
        drop(sm, "metric", lambda v: isinstance(v, str))
        if isinstance(sm.get("candidate"), dict):
            sm["candidate"] = _blank(sm["candidate"])
    offers = view.get("offers")
    for n, offer in enumerate(offers if isinstance(offers, list) else ()):
        if not (isinstance(offer, dict) and isinstance(offer.get("kind"), str) and offer["kind"] in OFFER_KINDS):
            continue
        options = offer.get("options")
        kept_ids = [o.get("id") if isinstance(o, dict) and not _is_option_id(o.get("id")) else None
                    for o in options] if isinstance(options, list) else None
        blanked = _blank(offer)
        if kept_ids is not None:                        # an id that fails its pattern is still scanned
            for o, oid in zip(blanked["options"], kept_ids):
                if isinstance(o, dict) and oid is not None:
                    o["id"] = oid
        if not (isinstance(offer.get("apply_flag"), str) and offer["apply_flag"] in APPLY_FLAGS):
            blanked["apply_flag"] = offer.get("apply_flag")
        offers[n] = blanked
    for name, codes, defaults in (("warnings", WARNING_CODES, WARNING_MESSAGES), ("notes", NOTE_CODES, NOTE_MESSAGES)):
        for msg in view.get(name) if isinstance(view.get(name), list) else ():
            if isinstance(msg, dict) and isinstance(msg.get("code"), str) and msg["code"] in codes:
                if msg.get("message") == defaults.get(msg["code"]):
                    msg["message"] = None
                msg["code"] = None
    for entry in view.get("history") if isinstance(view.get("history"), list) else ():
        if isinstance(entry, dict):
            drop(entry, "at", is_iso)
            drop(entry, "flag", among(APPLY_FLAGS))
            drop(entry, "answer", _is_option_id)
            drop(entry, "child", _is_record_id)
    return view


def _is_file_name(value: Any) -> bool:
    """A bare file name: non-empty, no folder, no separator of either platform (§10.2)."""
    return (isinstance(value, str) and bool(value) and "/" not in value and "\\" not in value
            and value not in (".", "..") and PurePath(value).name == value)


def _redact(problem: str, forbidden: Sequence[str]) -> str:
    for secret in forbidden:
        problem = problem.replace(secret, "…")
    return problem


def _check_object(p: list[str], where: str, value: Any, kind: str, *, required: frozenset[str] = frozenset()) -> bool:
    if not isinstance(value, Mapping):
        p.append(f"{where} must be an object")
        return False
    extra = sorted(str(k) for k in set(value) - _KNOWN[kind])
    if extra:
        p.append(f"{where} has unknown fields: {', '.join(extra)}")
    absent = sorted(required - set(value))
    if absent:
        p.append(f"{where} is missing: {', '.join(absent)}")
    return True


def _list(p: list[str], where: str, value: Any, present: bool) -> list[Any]:
    if not present:
        return []
    if not isinstance(value, list):
        p.append(f"{where} must be an array")
        return []
    return value


def _check_input(p: list[str], value: Any) -> None:
    if not _check_object(p, "input", value, "input", required=frozenset({"paths"})):
        return
    paths = value.get("paths")
    if not isinstance(paths, list) or not all(isinstance(x, str) for x in paths):
        p.append("input.paths must be an array of strings")
    kind = value.get("kind")
    if kind is not None and not (isinstance(kind, str) and re.fullmatch(r"[a-z_]{1,16}", kind)):
        p.append(f"input.kind {_short(kind)} isn't a known kind")
    seal = value.get("seal")
    if seal is not None and _check_object(p, "input.seal", seal, "seal", required=_KNOWN["seal"]):
        lists = [seal.get(k) for k in ("sha256", "sizes", "mtimes_ns")]
        if not all(isinstance(x, list) for x in lists):
            p.append("input.seal fields must be arrays")
        else:
            if not all(isinstance(h, str) and _SHA256_RE.match(h) for h in lists[0]):
                p.append("input.seal.sha256 must hold SHA-256 hex digests")
            if not all(isinstance(n, int) and not isinstance(n, bool) for n in lists[1] + lists[2]):
                p.append("input.seal sizes and mtimes must be integers")
            if len({len(x) for x in lists}) != 1 or (isinstance(paths, list) and len(lists[0]) != len(paths)):
                p.append("input.seal must have one entry per input path")


def _check_output(p: list[str], value: Any) -> None:
    if not _check_object(p, "output", value, "output", required=frozenset({"path", "sha256", "bytes", "kind"})):
        return
    kind = value.get("kind")
    if "kind" in value and not (isinstance(kind, str) and kind in OUTPUT_KINDS):
        p.append(f"output.kind {_short(kind)} isn't one of {', '.join(OUTPUT_KINDS)}")
    is_copy = kind == "searchable_copy"
    original = value.get("original")
    if original is not None:
        if _check_object(p, "output.original", original, "original", required=_KNOWN["original"]):
            if not isinstance(original.get("name"), str) or not original.get("name"):
                p.append("output.original.name must be a non-empty string")
            elif not _is_file_name(original["name"]):
                p.append("output.original.name is the original's file name only, never its folder")
            sha = original.get("sha256")
            if not (isinstance(sha, str) and _SHA256_RE.match(sha)):
                p.append("output.original.sha256 must be a SHA-256 hex digest")
    elif is_copy:
        p.append("a searchable copy names its original (output.original)")
    reasons = value.get("copy_reasons")
    if reasons is not None:
        known = _copyable_codes()
        hard = _hard_codes()
        if not isinstance(reasons, list):
            p.append("output.copy_reasons must be an array")
        elif not reasons:
            p.append("output.copy_reasons must name at least one copy reason")
        else:
            good: list[str] = []
            for i, code in enumerate(reasons):
                if (not (isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,40}", code))
                        or (known and code not in known) or code in hard):
                    p.append(f"output.copy_reasons[{i}] {_short(code)} isn't a copy reason")
                elif code in good:
                    p.append(f"output.copy_reasons[{i}] {_short(code)} is repeated")
                else:
                    good.append(code)
            order = {c: n for n, c in enumerate(_refusal_codes_ordered())}
            if order and all(c in order for c in good) and good != sorted(good, key=order.__getitem__):
                p.append("output.copy_reasons must be in refusals.CODES order")
    elif is_copy:
        p.append("a searchable copy says why it is a copy (output.copy_reasons)")
    if not is_copy and (original is not None or reasons is not None):
        p.append("only a searchable copy carries output.original and output.copy_reasons")


def _check_refusal(p: list[str], value: Any) -> None:
    if not _check_object(p, "refusal", value, "refusal", required=frozenset({"code", "message"})):
        return
    code = value.get("code")
    # Only a hard code refuses a file (§11.1, decided 2026-11-06): a copyable code such as ``signed``
    # makes a searchable copy instead, so it never sits in a record's ``refusal``.
    known = _hard_codes()
    if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,40}", code) or (known and code not in known):
        p.append(f"refusal.code {_short(code)} isn't a refusal code (one of refusals.HARD)")


def _check_page(p: list[str], where: str, page: Any) -> None:
    if not _check_object(p, where, page, "page", required=frozenset({"page"})):
        return
    _check_page_no(p, f"{where}.page", page.get("page"), required=True)
    page_warnings = page.get("warnings")
    if page_warnings is not None and not isinstance(page_warnings, list):
        p.append(f"{where}.warnings must be an array of warning codes")
        page_warnings = []
    for i, code in enumerate(page_warnings or []):
        if not isinstance(code, str) or code not in WARNING_CODES:
            p.append(f"{where}.warnings[{i}] {_short(code)} isn't a known warning code")
    pick = page.get("pick")
    if pick is not None and not isinstance(pick, Mapping):
        p.append(f"{where}.pick must be an object")
    fills = page.get("fills")
    if fills is not None and not (isinstance(fills, Mapping) and all(isinstance(v, int) for v in fills.values())):
        p.append(f"{where}.fills must be counts by kind")


def _check_page_no(p: list[str], where: str, value: Any, *, required: bool = False) -> None:
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        p.append(f"{where} must be a page number from 1")


def _check_offer(p: list[str], where: str, value: Any) -> None:
    if not _check_object(p, where, value, "offer", required=_KNOWN["offer"]):
        return
    kind = value.get("kind")
    flag = value.get("apply_flag")
    if not isinstance(kind, str) or kind not in OFFER_KINDS:
        p.append(f"{where}.kind {_short(kind)} isn't one of {', '.join(OFFER_KINDS)}")
        kind = None
    if not isinstance(flag, str) or flag not in APPLY_FLAGS:
        p.append(f"{where}.apply_flag {_short(flag)} isn't one of {', '.join(APPLY_FLAGS)}")
    elif kind is not None and APPLY_FLAG_FOR[kind] != flag:
        p.append(f"{where}.apply_flag for a {kind} offer is {APPLY_FLAG_FOR[kind]}")
    if not isinstance(value.get("summary"), str) or not value.get("summary"):
        p.append(f"{where}.summary must be a non-empty string")
    options = value.get("options")
    if not isinstance(options, list) or not options:
        p.append(f"{where}.options must be a non-empty array")
        return
    seen: set[str] = set()
    allowed = _KIND_OPTIONS.get(kind) if kind is not None else None
    for i, opt in enumerate(options):
        here = f"{where}.options[{i}]"
        if not _check_object(p, here, opt, "option", required=_KNOWN["option"]):
            continue
        oid = opt.get("id")
        if not _is_option_id(oid):
            p.append(f"{here}.id {_short(oid)} doesn't match {OPTION_ID_RE.pattern}")
        elif kind == "language" and not _is_lang_code(oid):
            p.append(f"{here}.id {_short(oid)} isn't a language code")
        elif allowed is not None and oid not in allowed:
            p.append(f"{here}.id {_short(oid)} isn't an option of a {kind} offer")
        if isinstance(oid, str):                        # only a valid-typed id can repeat; others are reported above
            if oid in seen:
                p.append(f"{here}.id {_short(oid)} is repeated")
            seen.add(oid)
        for name in ("label", "description"):
            if not isinstance(opt.get(name), str) or not opt.get(name):
                p.append(f"{here}.{name} must be a non-empty string")


def _check_offers_fit(p: list[str], offers: list[Any], output: Any) -> None:
    """An offer is about a written output: none without one, one smaller offer at most, none for a copy."""
    if output is None:
        p.append("offers are about a written output, and this record has none")
        return
    kinds = [o.get("kind") for o in offers if isinstance(o, Mapping)]
    if kinds.count("smaller") > 1:
        p.append("a record offers a smaller copy at most once")
    if kinds.count("thorough") > 1:
        p.append("a record offers thorough mode at most once")
    if "smaller" in kinds and isinstance(output, Mapping) and output.get("kind") == "searchable_copy":
        p.append("no smaller copy is offered for a searchable copy")


def _find_text_keys(p: list[str], value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            here = f"{where}.{k}" if where else str(k)
            page_count = where.endswith(".kinds") and k in PAGE_KINDS and type(v) is int
            if not isinstance(k, str) or k in _TEXT_KEYS or (k.endswith("_text") and k not in WARNING_CODES
                                                            and not page_count):
                p.append(f"{here}: a record never holds document text")
                continue
            if _is_password_key(k):
                p.append(f"{here}: a record never holds a password")
                continue
            if k == "dropped" and isinstance(v, Mapping):
                raw = [c for c in v if not (isinstance(c, str) and _CODEPOINT_RE.match(c))]
                if raw:
                    p.append(f"{here}: dropped characters are named as U+XXXX code points, never as the characters")
                    continue
            _find_text_keys(p, v, here)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _find_text_keys(p, v, f"{where}[{i}]")


# ---------------------------------------------------------------------------------- private helpers


def _from_json(data: Mapping[str, Any]) -> Record:
    values: dict[str, Any] = {name: data[name] for name in FIELDS}
    values["offers"] = [Offer(o["kind"], o["apply_flag"], o["summary"],
                              [Option(x["id"], x["label"], x["description"]) for x in o["options"]])
                        for o in values["offers"]]
    for name in ("warnings", "notes"):
        values[name] = [Message(m["code"], m["message"], m.get("page")) for m in values[name]]
    return Record(**values)


def _is_record_id(value: Any) -> bool:
    return isinstance(value, str) and RECORD_ID_RE.fullmatch(value) is not None


def _is_option_id(value: Any) -> bool:
    return isinstance(value, str) and OPTION_ID_RE.fullmatch(value) is not None


def _is_lang_code(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z]{2,3}", value) is not None and value not in {"none"}


def _is_iso_utc(value: str) -> bool:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return moment.utcoffset() is not None and moment.utcoffset().total_seconds() == 0


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _size(n: int) -> str:
    """Decimal sizes as people read them: 380 KB, 4.3 MB, 12 MB."""
    n = int(n)
    if n < 1_000_000:
        return f"{max(1, round(n / 1000)):d} KB"
    mb = n / 1_000_000
    if mb < 10:
        text = f"{mb:.1f}"
        return f"{text[:-2] if text.endswith('.0') else text} MB"
    return f"{round(mb):d} MB"


def _or_list(items: Sequence[str]) -> str:
    """"A", "A or B", "A, B or C"."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" or {items[-1]}"


def _pages_phrase(pages: Sequence[int]) -> str:
    nums = sorted(set(int(x) for x in pages))
    if not nums:
        return "These pages"
    if len(nums) == 1:
        return f"Page {nums[0]}"
    return "Pages " + ", ".join(str(n) for n in nums[:-1]) + f" and {nums[-1]}"


def _options_json(opts: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(opts) and not isinstance(opts, type):
        raw = {f.name: getattr(opts, f.name) for f in dataclasses.fields(opts)}
    elif isinstance(opts, Mapping):
        raw = dict(opts)
    else:
        raise TypeError("opts must be RunOptions")
    # A password or the saved passwords file's path never reaches a record; a re-run reads the
    # saved passwords again itself (§12.1, decided 2026-11-05).
    return to_jsonable({k: v for k, v in raw.items() if not _is_password_key(k)})


def _seal_json(seal: Any) -> dict[str, Any] | None:
    if seal is None:
        return None
    try:
        values = {"sha256": list(seal.sha256), "sizes": list(seal.sizes), "mtimes_ns": list(seal.mtimes_ns)}
    except AttributeError:
        data = seal.to_json()
        values = {k: list(data[k]) for k in ("sha256", "sizes", "mtimes_ns")}
    return {"sha256": [str(h) for h in values["sha256"]], "sizes": [int(n) for n in values["sizes"]],
            "mtimes_ns": [int(n) for n in values["mtimes_ns"]]}


def _cache_root() -> Path:
    from .platforms import cache_root

    return cache_root()


def _records_dir() -> Path:
    return _cache_root() / "records"


def _refusal_codes() -> frozenset[str]:
    """The record's refusal vocabulary: ``refusals.CODES``, read when it runs, so a hard code such as
    ``damaged`` (decided 2026-11-06) is accepted as soon as refusals has it. ``apply``'s own
    ``no_smaller_for_a_copy`` stops the command before any record is written, so it is never one."""
    try:
        from .refusals import CODES
    except ImportError:
        return frozenset()
    return frozenset(CODES)


def _refusal_codes_ordered() -> tuple[str, ...]:
    try:
        from .refusals import CODES
    except ImportError:
        return ()
    return tuple(CODES)


def _copyable_codes() -> frozenset[str]:
    """``refusals.COPYABLE`` (§11.1), imported lazily; empty (pattern check only) where refusals lacks it."""
    try:
        from .refusals import COPYABLE  # type: ignore[attr-defined]
    except ImportError:
        return frozenset()
    return frozenset(COPYABLE)


def _hard_codes() -> frozenset[str]:
    """``refusals.HARD`` (§11.1): codes that refuse a file and are never a copy reason."""
    try:
        from .refusals import HARD  # type: ignore[attr-defined]
    except ImportError:
        return frozenset()
    return frozenset(HARD)


def _engine_versions() -> dict[str, str]:
    """Each registered engine's version, or ``"unavailable: <reason>"``. Never raises."""
    try:
        from .ocr import engines
    except Exception as exc:                          # pragma: no cover - the registry is part of core
        return {"registry": f"unavailable: {type(exc).__name__}"}
    out: dict[str, str] = {}
    for name in engines.names():
        try:
            version = engines.get(name).version()
        except PdfIntakeError as exc:
            out[name] = f"unavailable: {exc.code}"
            continue
        except Exception as exc:
            out[name] = f"unavailable: {type(exc).__name__}"
            continue
        out[name] = str(version) if version else "unavailable: not installed or not on this platform"
    return out


def _os_name() -> str:
    """A short OS name and version, read without starting a process (macOS 15.4, Windows 10.0.19045, Ubuntu 24.04)."""
    if sys.platform == "darwin":
        release = _platform.mac_ver()[0]
        return f"macOS {release}" if release else "macOS"
    if sys.platform == "win32":
        try:
            v = sys.getwindowsversion()
            return f"Windows {v.major}.{v.minor}.{v.build}"
        except AttributeError:                        # pragma: no cover
            return "Windows"
    if sys.platform.startswith("linux"):
        try:
            info = _platform.freedesktop_os_release()
        except OSError:
            return "Linux"
        return str(info.get("PRETTY_NAME") or info.get("NAME") or "Linux")[:60]
    return sys.platform
