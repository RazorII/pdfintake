"""Orchestration: ``run`` and ``apply``, and the one place that writes into a user's folders.

``run`` is a straight pipeline over a work copy in the cache (DESIGN §1). For each output document:

1. **Seal.** Every input's SHA-256, size and mtime are recorded before any work, and checked again
   at commit and at the end, so a file that changes during the run is caught (exit 5).
2. **Read the PDF from a work copy.** The input is copied into the run's work folder and the copy
   is checked against the seal; nothing after that opens the user's file again. The refusal reader
   runs on the copy. A hard code (``refusals.HARD``: XFA, unreadable XMP, detectors that disagree)
   refuses the file (exit 4). An encrypted file is opened once with the saved passwords
   (``passwords.open_with``; the passwords file is read only when a file needs it) and saved
   unencrypted into the work folder (mode 0700); every later step reads that work copy. When no
   saved password opens it, the file is skipped: nothing is written, note ``skipped_no_password``,
   exit 0, and the batch carries on (DESIGN §12.1). The work folder is deleted when the file is done,
   whatever happened: SIGTERM and SIGHUP end the call through SystemExit so the cleanup runs, and a
   folder a killed run left behind (its lock no longer held) goes at the start of the next run,
   together with any partial file that run left beside an output it was writing (each output is
   noted in the folder's lock file before it is written).
3. **Images** are copied into the work folder and checked against the seal, then become one upright
   PDF (or one each with ``separate``) with ``intake.assemble``.
4. **Classify** the pages; only pages that are pictures of text (or carry only an old invisible
   layer) are read. Then the copy reasons (``refusals.copy_reasons``) decide the mode: an ordinary
   output, or a searchable copy beside an original pdfintake never changes (DESIGN §10, §11.1).
5. **OCR** (pages that need it): old invisible layers are stripped, the pages are turned upright
   (a RapidOCR vote with OSD as the second opinion), pass 1 gathers the detection signals (or
   takes them from the input's detection cache, below), ``langplan.plan`` and
   ``langplan.resolve_unsure`` give every page a pack without asking, the
   packs are ensured (fetched without asking unless ``fetch="never"``; a pack that fails its
   download, hash check or canary is replaced by the base pack, warning ``pack_broken``), pass 2
   reads, and the layer is written.
6. **Copies only:** signature widgets are flattened into their pages after the layer, the PDF/A and
   PDF/UA claims are dropped and the XMP gains the ``DerivedFrom`` block.
7. **Lossless shrink**, the **verify** suite (any failed check writes nothing, exit 1), then
   ``commit``: the refusal and seal re-check and the one atomic write into the user's folder.
8. **The smaller copy** (ordinary outputs only): built in staging and offered, or written beside
   the lossless file when ``smaller=True``.

**Nothing is asked during a run** (DESIGN §12, decided 2026-11-05). ``run`` never waits and never
exits 10: provenance comes from the options (default ``ordinary``), the language from detection or
the options, the smaller copy from ``smaller``. What a person could still act on is listed as
offers in the record, and ``apply`` acts on one later, only when a person asks.

**One file failing never stops the batch.** Every document gets its own record; an error becomes
that record's status and exit. Only an invocation that can't be run at all (no inputs, PDFs mixed
with images, ``-o`` for several documents, an option outside its values) raises ``UsageError``. An
input that isn't a regular file (a FIFO, a device, a folder) or whose path no file can have is that
input's own usage error, and is never opened.

**Names.** No output lands on any input of the invocation. Documents that would share a name are
numbered in input order before the batch runs, so the same batch maps to the same names whichever
files fail, and a stem too long for a file name is shortened with "…".

**Writes.** ``commit`` and ``apply`` are the only callers of ``fsutil.atomic_write``
(tests/test_write_paths.py checks it), and only ``--overwrite`` (or ``apply --lang`` replacing the
output it re-reads, and only while that output is unchanged at the moment of the rename) lets it
replace a file: otherwise a file that took the name after the guard is kept and the record says
``output_exists``. Every write's destination is noted in a work folder's lock file first, so the
next run can sweep what a SIGKILL leaves. Everything else this module writes goes into the cache,
through ``_save_work``, ``_copy_into_work``, ``_lock_work`` and ``_journal_dest`` (which writes only
to the lock file ``_lock_work`` opened), which refuse any path outside it, and the detection cache's
file through the pack store's atomic JSON writer, into a folder ``fsutil.work_dir`` made in the cache.
The input is never written, renamed or touched.

**No subprocess.** ``run``, ``apply`` and ``doctor`` import every engine first (``engines.warm_up``)
and then hold ``integrity.no_subprocess()`` for the whole call.

**The detection cache** (DESIGN §9.1). Pass 1's result for an input is kept in
``<cache>/work/<input sha>-<version>/``, so a re-run and ``apply --lang`` don't render and detect
the same pages again. ``pipeline.gather`` decides what the key covers and checks every entry
(``pipeline.read_cache``); ``_DetectionCache`` only keeps the entries on disk: one folder per stored
result, its renders and ``detection.json``, written last. A hit copies the renders into the run's own
work folder before pass 2 reads them, so an entry another run replaces or the prune removes never
breaks a run that is using it; a copy that doesn't match its recorded hash is a fresh detection.
Storing a result removes the input's older entries. The folder is not a run's work folder: the
orphan sweep leaves it alone (``CACHE_DIR_RE``), and the 14-day prune removes it. **An encrypted
input is never cached**: its renders and reads would be the decrypted pages, which go with the run's
work folder when it ends (DESIGN §11.1).

The same folder keeps the orientation step's votes and OSD results (``orientation.json``, checked by
``pipeline.read_orientation``), so a re-run doesn't render and read every page for orientation
either, and, for an images input, the date its PDF was first built (``assembled.json``): the work PDF
carries that date, so it is byte-identical on a re-run and every key above matches. A result that met
a failure of the moment (a read that failed or timed out, a session that couldn't open) is never kept.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import secrets
import shutil
import stat
import sys
import threading
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from . import __version__, fsutil, integrity, passwords, platforms, record, refusals
from .errors import (
    EngineFailure,
    InputChanged,
    IntegrityError,
    PackError,
    PdfIntakeError,
    Refused,
    SetupError,
    StaleRecord,
    UsageError,
    VerifyFailed,
)
from .integrity import InputSeal
from .record import Record
from .types import to_jsonable

if TYPE_CHECKING:
    import pikepdf

    from .langpacks.index import Index, IndexEntry
    from .langpacks.store import InstalledPack
    from .ocr.langplan import LanguagePlan, PagePlan
    from .pdf.classify import PageClass
    from .refusals import RefusalReport

__all__ = ["RunOptions", "FileRef", "run", "apply", "commit", "DEFAULT_WORKERS"]

DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
MAX_WORKERS = 16
FETCH_MODES = ("auto", "never")
SUFFIX_SEARCHABLE = " (searchable)"
SUFFIX_COMPACT = " (compact)"
SUFFIX_COPY = " (searchable copy)"
SMALLER_BAR = 90.0                  # the worst changed tile must score at least this (DESIGN Decisions 4)
SMALLER_MIN_PCT = record.SMALLER_MIN_SAVING_PCT
MIN_SAVING = 0.01                   # a compact file must be at least 1% smaller (DESIGN §4.1)
LAYER_SHRANK_RATIO = 0.8            # a new layer under 80% of the old one's characters (warning layer_shrank)
VOTE_DPI = 150.0                    # the orientation vote's render (the vote scales it to 1100 px anyway)
OSD_DPI = 300.0                     # OSD is unreliable below about 300 dpi (measured, DESIGN §1)
PRODUCER = f"pdfintake {__version__}"
LANG_ANSWER_RE = re.compile(r"^(none|[a-z]{2,3})$")      # run's --lang and --force-lang (DESIGN §4.1)
LOCK_NAME = ".pdfintake-run.lock"   # held by the run that owns a work folder, for as long as it runs
ORPHAN_GRACE_S = 600                # a work folder with no lock file is left alone this long (it may be starting)
JOURNAL_MAX = 1 << 20               # the most of a lock file's noted output paths the sweep reads
# An input's detection cache: work/<input sha>-<version> (a run's own folder has a 32-hex prefix and
# a random tail, so it never matches). The orphan sweep leaves these alone; the 14-day prune doesn't.
CACHE_DIR_RE = re.compile(r"^[0-9a-f]{64}-[0-9A-Za-z.+]+$")
CACHE_ENTRY_PREFIX = "e-"           # one stored result in a detection cache folder
ASSEMBLED_FILE = "assembled.json"   # in an images input's cache folder: the date its PDF was first built
ASSEMBLED_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SLOT_LABEL_ROOM = " (searchable copy 9999).pdf"   # the longest name tail a stem must leave room for
ELLIPSIS = "\u2026"
_SIGNATURE_REASONS = frozenset({"signed", "certified", "usage_rights", "timestamped"})
_LANGUAGE_WARNINGS = frozenset({"language_uncertain", "no_pack_for_language", "pack_not_installed", "pack_broken"})


# --------------------------------------------------------------------------- options and results

@dataclass
class RunOptions:
    """Every choice a run makes, from flags or defaults; nothing is asked (DESIGN §12).

    ``provenance`` comes from the user's own words (``--provenance``), otherwise ``ordinary``.
    ``fetch`` is ``auto`` (fetch every published pack the plan needs, without asking) or ``never``
    (offline). ``smaller=True`` writes the smaller copy beside the lossless file wherever it clears
    the bar; otherwise the record offers it. ``passwords_file`` is the saved passwords list
    (``--passwords-file``); None reads the default one when it exists. A record never holds it.
    ``thorough`` is ``--thorough`` (True), ``--no-thorough`` (False) or neither (None: the saved
    ``pdfintake thorough`` setting decides, DESIGN §14).
    """

    provenance: Literal["ordinary", "counterparty", "evidence"] = "ordinary"
    out: Path | None = None
    outdir: Path | None = None
    lang: tuple[str, ...] = ()
    force_lang: tuple[str, ...] = ()
    fetch: Literal["auto", "never"] = "auto"
    smaller: bool = False
    overwrite: bool = False
    redo_ocr: bool = False
    keep_existing_layer: bool = False
    separate: bool = False
    plan_only: bool = False
    workers: int = DEFAULT_WORKERS
    passwords_file: Path | None = None
    thorough: bool | None = None


@dataclass(frozen=True)
class FileRef:
    """A file ``commit`` or ``apply`` wrote: where, its SHA-256 and its size."""

    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class _LangAnswer:
    """``apply --lang``: the pack the person's request reads pages with, and the ones it turned away from."""

    approved: frozenset[str] = frozenset()
    declined: frozenset[str] = frozenset()


@dataclass
class _DocInput:
    paths: list[Path]
    kind: str                         # "pdf", "images", or "bad" (``error`` says why)
    error: PdfIntakeError | None = None
    slot: int = 1                     # its number among the batch's documents that share an output name


# --------------------------------------------------------------------------- seams (tests replace these)

def _platform_key() -> str:
    """The supported platform's key; SetupError naming why this platform isn't supported."""
    return platforms.require_supported().key


def _record_platform() -> str:
    return platforms.current().key


def _load_index() -> Index:
    from .langpacks import index

    return index.load()


def _ensure(entry: IndexEntry, *, allow_network: bool) -> InstalledPack:
    from .langpacks import store

    return store.ensure(entry, allow_network=allow_network)


def _installed() -> dict[str, InstalledPack]:
    from .langpacks import store

    return store.installed()


def _open_voter(base: InstalledPack) -> Any:
    """The orientation voter on the base pack's RapidOCR models, or None when it can't run here.

    A model that fails its hash check stops the file (IntegrityError): an engine never loads a file
    that wasn't just checked (DESIGN §11.2).
    """
    from .ocr.engines import rapid

    try:
        models = {key: base.asset_path(key) for key in (rapid.VOTER_DET_KEY, rapid.VOTER_REC_KEY)}
        return rapid.open_voter(models)
    except IntegrityError:
        raise
    except (EngineFailure, PackError, SetupError, ImportError):
        return None


class _ThreadVoters:
    """The orientation voter as one RapidOCR session per thread, so the vote's four turns can be read
    at once. ``first`` (the session whose open told the run the voter works) goes to the first thread
    that reads; every other thread opens its own on its first read, through ``_open_voter``, so its
    models are hash-checked right before that open (DESIGN §11.2). A thread whose open fails reads
    nothing (EngineFailure, which the vote scores as no text); a tampered model raises IntegrityError
    and stops the file, as it would on the first thread."""

    def __init__(self, first: Any, base: InstalledPack) -> None:
        self._base = base
        self._spare: Any = first
        self._local = threading.local()
        self._lock = threading.Lock()
        self._opened: list[Any] = [first]

    def _session(self) -> Any:
        session = getattr(self._local, "session", None)
        if session is not None:
            return session
        with self._lock:
            session, self._spare = self._spare, None
        if session is None:
            session = _open_voter(self._base)
            if session is None:
                raise EngineFailure("rapidocr_unavailable", "the orientation voter couldn't open on this thread")
            with self._lock:
                self._opened.append(session)
        self._local.session = session
        return session

    def read(self, page: Any, *, timeout_s: float) -> Any:
        return self._session().read(page, timeout_s=timeout_s)

    def close(self) -> None:
        with self._lock:
            sessions, self._opened = self._opened, []
        for session in sessions:
            close = getattr(session, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:                      # noqa: BLE001 - closing never fails a file
                    pass


def _thorough_offered() -> bool:
    """Whether a written output's record offers thorough mode: this Mac can run it and the user hasn't
    decided yet (DESIGN §14.2). A seam the tests replace."""
    from . import thorough

    return thorough.should_offer()


def _thorough_use(d: "_Doc", platform_key: str) -> Any:
    """The dots use for this document's pass 2 when thorough mode reads it, else None. Asked for (the
    flag or the saved setting) but not able to run here: warning ``thorough_unavailable``, and the
    regular readers read alone."""
    from . import thorough

    on, why = thorough.resolve(d.opts.thorough)
    if not on:
        if why is not None:
            _warn(d, "thorough_unavailable")
        return None
    from .ocr import pipeline

    use = pipeline.thorough_use(platform_key)
    if use is None or not use.avail.ok:
        _warn(d, "thorough_unavailable")
        return None
    d.thorough = True
    return use


def _open_osd(base: InstalledPack) -> Any:
    """Tesseract OSD on the base pack's ``osd.traineddata``, or None when it can't run here."""
    key = next((a.key for a in base.manifest.models if a.file == "osd.traineddata"), None)
    if key is None:
        return None
    try:
        path = base.asset_path(key)
        from .ocr.engines import tesseract

        return tesseract.open_osd({key: path})
    except IntegrityError:
        raise
    except (EngineFailure, PackError, SetupError, ImportError):
        return None


# --------------------------------------------------------------------------- the batch

class _LazyPasswords:
    """The saved passwords, read only when a file needs one (``passwords.open_with`` tries "" first)."""

    def __init__(self, batch: _Batch) -> None:
        self._batch = batch

    def __iter__(self) -> Iterator[str]:
        return iter(self._batch.passwords())

    def __repr__(self) -> str:
        return "<saved passwords>"


class _Batch:
    """What a run shares between its documents: names taken, the passwords, the index and packs.

    Each is loaded on first need and kept, errors included, so a problem that hits every file (a
    passwords file that isn't private, a broken base pack) is found once and named in each record.
    """

    def __init__(self, opts: RunOptions, inputs: Sequence[Path] = ()) -> None:
        self.opts = opts
        self.reserved: set[str] = set()
        # Every input of the invocation: no document's output may land on any of them (decided
        # 2026-11-05). A path no file can have (a NUL in it) can't be hit, and can't be compared.
        self.all_inputs: list[Path] = [Path(p) for p in inputs if "\x00" not in str(p)]
        self.record_platform = _record_platform()
        self._passwords: list[str] | None = None
        self._loaded: dict[str, Any] = {}
        self._errors: dict[str, PdfIntakeError] = {}

    def _get(self, name: str, load: Any) -> Any:
        if name in self._errors:
            raise _fresh(self._errors[name])
        if name not in self._loaded:
            failure: PdfIntakeError | None = None
            try:
                self._loaded[name] = load()
            except PdfIntakeError as exc:
                failure = _fresh(exc)
            if failure is not None:
                # Kept for the whole batch: a copy with no traceback and no chained error, so it holds
                # none of the loader's locals (a passwords file's lines among them; decided 2026-11-06).
                self._errors[name] = failure
                raise _fresh(failure)
        return self._loaded[name]

    def passwords(self) -> list[str]:
        if self._passwords is None:
            self._passwords = self._get("passwords", lambda: passwords.candidates(self.opts.passwords_file))
        return self._passwords

    @property
    def forbid(self) -> Sequence[str]:
        """The saved passwords this run has read, for ``record.save``'s check; empty when none were read."""
        return self._passwords if self._passwords is not None else ()

    def platform_key(self) -> str:
        return self._get("platform", _platform_key)

    def index(self) -> Index:
        return self._get("index", _load_index)

    def base(self) -> InstalledPack:
        """The English base pack, ensured once (every hash, and its canary): DESIGN §1 node Q."""
        return self._get("base", lambda: _ensure(self.index().base(), allow_network=False))

    def installed(self) -> dict[str, InstalledPack]:
        packs = self._get("installed", _installed)
        base = self.base()
        packs.setdefault(base.manifest.id, base)
        return packs


def _fresh(exc: PdfIntakeError) -> PdfIntakeError:
    """The same error, built again: same class, code, message and detail, nothing chained to it."""
    try:
        return type(exc)(exc.code, exc.message, **exc.detail)
    except TypeError:                                  # a subclass with another signature
        exc.__traceback__ = exc.__context__ = exc.__cause__ = None
        return exc


# --------------------------------------------------------------------------- run

def run(inputs: Sequence[Path], opts: RunOptions | None = None) -> list[Record]:
    """Process ``inputs`` and return one record per output document, each already saved.

    PDFs are never merged; images become one PDF (one each with ``separate``). Nothing is asked,
    and no record ever carries exit 10 (DESIGN §12). UsageError only for an invocation that can't
    run at all; every per-file problem is that file's record.
    """
    opts = opts if opts is not None else RunOptions()
    _check_options(opts)
    paths = [Path(os.path.abspath(Path(p).expanduser())) for p in inputs]
    if not paths:
        raise UsageError("no_inputs", "No input files were given.")
    from .ocr import engines

    engines.warm_up()
    with integrity.no_subprocess(), _stop_signals_unwind():
        docs = _documents(paths, opts)
        _assign_slots(docs, opts)
        _prune_cache(paths)
        batch = _Batch(opts, paths)
        return [_run_document(doc, opts, batch) for doc in docs]


class _stop_signals_unwind(contextlib.AbstractContextManager):
    """While a run or apply holds it, SIGTERM and SIGHUP end the call by raising SystemExit.

    A launchd stop, a closed terminal or a dropped remote session sends one of them. By default
    the process would end at once and skip every ``finally``, leaving a decrypted work copy in the
    cache and a partial file in the user's folder (DESIGN §11.1). As SystemExit, the work folder
    and the partial file are removed on the way out, and the process still ends with 128 + the
    signal number (decided 2026-11-05). Only installed in the main thread, only over the default
    action (an ignored signal or someone else's handler is left alone), and restored afterwards.
    """

    def __enter__(self) -> _stop_signals_unwind:
        import signal
        import threading

        self._old: dict[int, Any] = {}
        if threading.current_thread() is not threading.main_thread():
            return self
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                if signal.getsignal(sig) is signal.SIG_DFL:
                    self._old[sig] = signal.signal(sig, _end_the_call)
            except (ValueError, OSError):
                continue
        return self

    def __exit__(self, *exc: Any) -> None:
        import signal

        for sig, old in self._old.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError):
                pass
        self._old = {}
        return None


def _end_the_call(signum: int, frame: Any) -> None:
    raise SystemExit(128 + int(signum))


def _check_options(opts: RunOptions) -> None:
    refusals.check_provenance(opts.provenance)        # UsageError for a value outside the three
    if opts.fetch not in FETCH_MODES:
        raise UsageError("bad_fetch", "--fetch takes auto or never.", value=str(opts.fetch)[:16])
    if isinstance(opts.workers, bool) or not isinstance(opts.workers, int) or not 1 <= opts.workers <= MAX_WORKERS:
        raise UsageError("bad_workers", f"--workers takes a whole number from 1 to {MAX_WORKERS}.")
    if opts.out is not None and opts.outdir is not None:
        raise UsageError("out_and_outdir", "Give either -o or --outdir, not both.")
    for flag, codes in (("--lang", opts.lang), ("--force-lang", opts.force_lang)):
        if isinstance(codes, str):
            raise UsageError("bad_language_code", f"{flag} takes language codes such as en or fr.", flag=flag)
        for raw in codes:
            for part in str(raw).split(","):
                if part.strip() and not LANG_ANSWER_RE.match(part.strip().lower()):
                    raise UsageError("bad_language_code", f"{flag} takes ISO 639 language codes such as en or fr.",
                                     flag=flag)


def _documents(paths: Sequence[Path], opts: RunOptions) -> list[_DocInput]:
    """The output documents: each PDF on its own; the images together, or one each with ``separate``."""
    from .intake import sniff

    pdfs: list[Path] = []
    images: list[Path] = []
    bad: list[_DocInput] = []
    order: list[tuple[str, Any]] = []
    for p in paths:
        problem = _input_problem(p)
        if problem is None:
            try:
                kind = sniff.sniff(p)
            except UsageError as exc:                  # swapped for a FIFO or device since the stat
                problem = exc
            except (OSError, ValueError):
                problem = UsageError("input_missing", f"The input {p} doesn't exist or can't be read.", path=str(p))
        if problem is not None:
            doc = _DocInput([p], "bad", problem)
            bad.append(doc)
            order.append(("bad", doc))
            continue
        if kind == "pdf":
            pdfs.append(p)
            order.append(("pdf", p))
        elif kind == "unknown":
            err = UsageError("unsupported_input", f"The input {p} isn't a PDF or an image pdfintake can read.",
                             path=str(p))
            doc = _DocInput([p], "bad", err)
            bad.append(doc)
            order.append(("bad", doc))
        else:
            images.append(p)
            order.append(("image", p))
    if pdfs and images:
        raise UsageError("mixed_inputs", "PDFs and images can't be mixed in one run; run them separately.")
    docs: list[_DocInput] = []
    images_done = False
    for what, item in order:
        if what == "bad":
            docs.append(item)
        elif what == "pdf":
            docs.append(_DocInput([item], "pdf"))
        elif opts.separate:
            docs.append(_DocInput([item], "images"))
        elif not images_done:
            docs.append(_DocInput(list(images), "images"))
            images_done = True
    if opts.out is not None and len(docs) > 1:
        raise UsageError("out_needs_one_output", "-o names one output file, and these inputs make several; "
                                                 "use --outdir instead.")
    return docs


def _input_problem(p: Path) -> UsageError | None:
    """Why ``p`` can't be an input before anything opens it, or None.

    Only a regular file (or a link to one) is opened: a FIFO, ``/dev/stdin`` or another device
    would make the run wait for a writer or read standard input, and a run never waits (DESIGN §12).
    A NUL can't be in a file name, and ``os`` raises ValueError for one rather than OSError.
    """
    text = str(p)
    if "\x00" in text:
        shown = text.replace("\x00", "\\x00")
        return UsageError("input_missing", f"The input {shown} can't exist: its name holds a NUL character.",
                          path=shown)
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return UsageError("input_missing", f"The input {p} doesn't exist or can't be read.", path=text)
    if not stat.S_ISREG(st.st_mode):
        return UsageError("input_not_file", f"The input {p} isn't a regular file.", path=text)
    return None


def _assign_slots(docs: Sequence[_DocInput], opts: RunOptions) -> None:
    """Number the documents that would share an output name, in input order (DESIGN §11.2).

    Two inputs with the same stem going to one folder (``a/Scan.pdf`` and ``b/Scan.pdf`` with
    ``--outdir``, ``Card.png`` and ``Card.jpg`` with ``--separate``) get ``(searchable)`` and
    ``(searchable 2)``. The number is fixed here, before any document runs, and failed or skipped
    documents keep theirs, so a batch run again maps every input to the same output name whichever
    files succeed (decided 2026-11-05).
    """
    if opts.out is not None:
        return
    seen: dict[str, int] = {}
    for doc in docs:
        key = _slot_key(doc.paths[0], opts)
        if key is None:
            continue
        seen[key] = seen.get(key, 0) + 1
        doc.slot = seen[key]


def _slot_key(first: Path, opts: RunOptions) -> str | None:
    if "\x00" in str(first):
        return None
    folder = _out_folder(first, opts)
    return _fold(folder / _fit_stem(first.stem))


def _prune_cache(keep: Sequence[Path] = ()) -> None:
    """Remove work and staging folders untouched for 14 days (DESIGN §3.5), and every work folder
    a run that was killed left behind, whatever its age; never fails a run. ``keep`` is this
    invocation's inputs, which the orphan sweep never touches."""
    _sweep_orphans(keep)
    root = platforms.cache_root()
    for kind in fsutil.PRUNABLE_KINDS:
        folder = root / kind
        if folder.is_dir():
            try:
                fsutil.prune(folder, 14)
            except (OSError, PdfIntakeError):
                pass


def _sweep_orphans(keep: Sequence[Path] = ()) -> None:
    """Remove each work folder no live run holds (DESIGN §11.1, decided 2026-11-05).

    A run holds a lock on ``LOCK_NAME`` in its work folder from the moment it makes the folder until
    it removes it. A process that was killed (SIGKILL, a crash, a power cut) holds nothing, so its
    folder, a decrypted work copy included, goes at the next run instead of after 14 days. A folder
    with no lock file at all is only removed once it is ``ORPHAN_GRACE_S`` old, since a run may be
    between making the folder and locking it. Windows has no such lock here; there a file in use
    can't be removed anyway, and the 14-day prune applies.

    Before an orphaned folder goes, each output path its run noted in the lock file
    (``_journal_dest``) is handed to ``fsutil.sweep_partial``, which removes the partial file a run
    killed in mid-write left beside it in the user's folder, and nothing else: only that output's
    own partial name, only a regular file of this user's, and never one a live writer holds
    (decided 2026-11-06). A partial file that can't be removed then stays until something writes
    that name again. ``keep`` is the calling run's own inputs: a file at a noted partial name that
    is one of them is that run's input, not a leftover, and is never touched (decided 2026-11-06).

    An input's detection cache (``CACHE_DIR_RE``) isn't a run's folder and no run locks it, so the
    sweep leaves it alone; it never holds a decrypted page (an encrypted input isn't cached), and
    the 14-day prune removes it.
    """
    if sys.platform == "win32":
        return
    folder = platforms.cache_root() / "work"
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    now = time.time()
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False) or CACHE_DIR_RE.match(entry.name):
                continue
            work = Path(entry.path)
            try:
                fd = _lock_work(work / LOCK_NAME, create=False)
            except FileNotFoundError:
                if now - entry.stat(follow_symlinks=False).st_mtime > ORPHAN_GRACE_S:
                    _remove_work(work)
                continue
            if fd is None:
                continue                               # a live run holds it
            try:
                for dest in _journaled(fd):
                    fsutil.sweep_partial(dest, keep=keep)
                _remove_work(work)
            finally:
                os.close(fd)
        except (OSError, PdfIntakeError):
            continue


def _journaled(fd: int) -> list[Path]:
    """The output paths a run noted in its lock file (``_journal_dest``); [] when it can't be read.

    Each entry ends with a NUL, so an entry a kill cut short is ignored. Only absolute paths count.
    """
    chunks: list[bytes] = []
    total = 0
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        while total < JOURNAL_MAX:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        return []
    out: list[Path] = []
    for raw in b"".join(chunks)[:JOURNAL_MAX].split(b"\0")[:-1]:
        if raw:
            path = Path(os.fsdecode(raw))
            if path.is_absolute():
                out.append(path)
    return out


# --------------------------------------------------------------------------- one document

@dataclass
class _Doc:
    inputs: list[Path]
    kind: str
    opts: RunOptions
    batch: _Batch
    rec: Record
    answer: _LangAnswer | None = None
    replaces: FileRef | None = None
    step: str = "start"
    work: Path | None = None
    seal: InputSeal | None = None
    lock: int | None = None                 # the open lock file that marks the work folder as live
    slot: int = 1
    reports: list[RefusalReport] = field(default_factory=list)
    source: Path | None = None              # the PDF the build starts from
    reference: Path | None = None           # what the verify suite compares against
    assembled: list[Any] | None = None
    encrypted: bool = False
    skipped: bool = False
    copy_reasons: list[str] = field(default_factory=list)
    thorough: bool = False                  # thorough mode read this document's OCR pages (§14)


def _run_document(doc: _DocInput, opts: RunOptions, batch: _Batch, *, parent: str | None = None,
                  answer: _LangAnswer | None = None, replaces: FileRef | None = None) -> Record:
    try:
        rec = record.new(doc.paths, opts=opts, seal=None, platform_key=batch.record_platform, parent=parent)
    except Exception:                                  # noqa: BLE001 - a path record can't hold
        rec = record.new([Path(_printable(str(p))) for p in doc.paths], opts=opts, seal=None,
                         platform_key=batch.record_platform, parent=parent)
    rec.provenance = opts.provenance
    d = _Doc(list(doc.paths), doc.kind, opts, batch, rec, answer=answer, replaces=replaces, slot=doc.slot)
    try:
        if doc.error is not None:
            raise doc.error
        _process(d)
    except PdfIntakeError as exc:
        _fail(d, exc)
    except PermissionError as exc:
        if str(exc).startswith("pdfintake starts no processes"):
            _fail(d, PdfIntakeError("process_refused", str(exc)))
        else:
            _fail(d, _unexpected(exc))
    except Exception as exc:                           # noqa: BLE001 - one file never stops the batch
        _fail(d, _unexpected(exc))
    finally:
        if d.work is not None:
            _remove_work(d.work)
        if d.lock is not None:
            os.close(d.lock)
            d.lock = None
    _save_record(rec, batch.forbid)
    return rec


def _printable(text: str) -> str:
    """``text`` with what UTF-8 JSON can't hold escaped: a lone surrogate (how Python hands over a
    file name that isn't UTF-8) becomes ``\\udcff``, a NUL ``\\x00``."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8").replace("\x00", "\\x00")


def _scrub(value: Any) -> Any:
    """``value`` with every string made ``_printable``, through dicts, lists, tuples, paths and dataclasses."""
    if isinstance(value, str):
        return _printable(value)
    if isinstance(value, Path):
        return Path(_printable(str(value)))
    if isinstance(value, dict):
        return {_scrub(k): _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _scrub(getattr(value, f.name))
                                             for f in dataclasses.fields(value) if f.init})
    return value


def _unexpected(exc: BaseException) -> PdfIntakeError:
    name = type(exc).__name__
    name = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) else "error"
    return PdfIntakeError("internal_error", f"pdfintake hit an unexpected {name} while processing this file.",
                          kind=name)


def _fail(d: _Doc, exc: PdfIntakeError) -> None:
    rec = d.rec
    rec.status = "refused" if isinstance(exc, Refused) else "failed"
    rec.exit = exc.exit_code
    rec.steps.append({"step": d.step, "status": "failed", "detail": {"code": exc.code, "message": exc.message}})
    if rec.output is None:
        rec.offers = []


def _save_record(rec: Record, forbid: Sequence[str]) -> None:
    """Save the record; a record that won't validate is saved again with only its error codes.

    Never raises: one file's record failing to save never stops the batch (decided 2026-11-05). A
    path the record's JSON can't hold (a name that isn't UTF-8) is saved escaped, and when even the
    bare record can't be saved, the record is returned unsaved with status failed.
    """
    try:
        record.save(rec, forbid=forbid)
        return
    except PdfIntakeError:
        pass
    except Exception:                                  # noqa: BLE001 - e.g. a lone surrogate in a path
        for f in dataclasses.fields(rec):
            if f.name not in ("id", "parent"):
                setattr(rec, f.name, _scrub(getattr(rec, f.name)))
        try:
            record.save(rec, forbid=forbid)
            return
        except Exception:                              # noqa: BLE001
            pass
    for step in rec.steps:                              # messages can quote anything a library said
        detail = step.get("detail")
        if isinstance(detail, dict) and "message" in detail:
            detail.pop("message")
    try:
        record.save(rec, forbid=forbid)
        return
    except Exception:                                  # noqa: BLE001
        pass
    rec.status, rec.exit = "failed", 1
    rec.steps = [{"step": "record", "status": "failed", "detail": {"code": "record_invalid"}}]
    rec.verify, rec.pages, rec.offers, rec.language, rec.smaller = [], [], [], {}, None
    rec.warnings, rec.notes, rec.shrink, rec.metadata, rec.refusal = [], [], {}, {}, None
    rec.input = _scrub(rec.input)
    if rec.output is not None:
        rec.output = _scrub(rec.output)
    try:
        record.save(rec, forbid=forbid)
    except Exception:                                  # noqa: BLE001 - returned unsaved; the batch goes on
        pass


def _step(d: _Doc, name: str, status: str, **detail: Any) -> None:
    d.rec.steps.append({"step": name, "status": status, "detail": to_jsonable(detail)})


def _warn(d: _Doc, code: str, page: int | None = None, message: str | None = None) -> None:
    if code not in record.WARNING_CODES:
        return
    if any(w.code == code and w.page == page for w in d.rec.warnings):
        return
    d.rec.warnings.append(record.warning(code, page, message))


def _add_warning(d: _Doc, msg: record.Message) -> None:
    if any(w.code == msg.code and w.page == msg.page for w in d.rec.warnings):
        return
    d.rec.warnings.append(msg)


def _note(d: _Doc, code: str, page: int | None = None) -> None:
    if code not in record.NOTE_CODES:
        return
    if any(n.code == code and n.page == page for n in d.rec.notes):
        return
    d.rec.notes.append(record.note(code, page))


def _process(d: _Doc) -> None:
    opts, rec = d.opts, d.rec
    d.step = "seal"
    d.seal = InputSeal.take(d.inputs)
    rec.input["seal"] = {"sha256": list(d.seal.sha256), "sizes": list(d.seal.sizes),
                         "mtimes_ns": list(d.seal.mtimes_ns)}
    d.work = fsutil.work_dir("work", f"{d.seal.sha256[0][:32]}-{__version__}-{secrets.token_hex(4)}")
    d.lock = _lock_work(d.work / LOCK_NAME, create=True)     # live until the folder is removed
    if d.kind == "pdf":
        rec.input["kind"] = "pdf"
        _prepare_pdf(d)
        if d.skipped:
            _done_without_output(d, "skipped_no_password")
            return
    else:
        rec.input["kind"] = "images"
        _prepare_images(d)
    assert d.source is not None and d.reference is not None

    from .pdf import classify

    fm = _fontmap(d) if d.kind == "pdf" else None       # §15: proven maps for fonts that have none
    mapped = bool(fm is not None and fm.changed and not opts.plan_only)
    d.step = "classify"
    classes = classify.classify(d.source, redo_ocr=opts.redo_ocr, keep_existing_layer=opts.keep_existing_layer,
                                font_pages=_font_pages(fm))
    rec.input["pages"] = len(classes)
    ocr_pages = [c.index for c in classes if c.needs_ocr]
    adds_layer = bool(ocr_pages)
    for c in classes:
        for code in c.warnings:
            _warn(d, code, c.index + 1)
    if mapped and fm is not None:
        for i in fm.pages_repaired:
            _note(d, "font_map_repaired", i + 1)
        for i in fm.pages_partial:
            _warn(d, "font_map_partial", i + 1)
    _step(d, "classify", "ok", pages=len(classes), ocr=len(ocr_pages),
          kinds=dict(sorted(_count(c.kind for c in classes).items())))

    d.step = "mode"
    d.copy_reasons = _copy_reasons(d.reports, opts.provenance, adds_layer or mapped)
    copy = bool(d.copy_reasons)
    if copy and opts.overwrite and d.replaces is None:
        raise record.usage_error("copy_never_overwrites")
    _step(d, "mode", "changed" if copy else "ok", kind="searchable_copy" if copy else "searchable",
          copy_reasons=d.copy_reasons)
    if opts.plan_only and not adds_layer:
        _plan_record(d, None, None, classes)
        return
    if copy and not adds_layer and not mapped and not d.encrypted:
        # The original already reads as it should, and a rewrite would only lose what it carries.
        _done_without_output(d, "nothing_written")
        return
    if (adds_layer or mapped) and not opts.plan_only:
        _guard_dest(d, _dest_for(d, _suffix(d, copy, layered=True)), copy)   # fail early, before any OCR

    plat = None
    lp = None
    packs: dict[str, InstalledPack] = {}
    fetched: list[str] = []
    broken: set[str] = set()
    src = d.source
    strip_rep = None
    stripped: list[int] = []
    orient_rep = None
    decisions: dict[int, Any] = {}
    layer_input: list[Any] = []
    ocr: list[Any] = []
    signals = None
    base = None
    if adds_layer:
        d.step = "setup"
        plat = d.batch.platform_key()
        index = d.batch.index()
        base = d.batch.base()
        installed = d.batch.installed()
        stripped = [c.index for c in classes if c.strip_old_layer]
        if stripped:
            d.step = "strip"
            src, strip_rep = _strip(d, src, stripped, classes)
            _step(d, "strip", "changed", pages=[i + 1 for i in stripped], ops_dropped=strip_rep.ops_dropped)
        d.step = "orientation"
        src, orient_rep, decisions, osd_map = _orient(d, src, ocr_pages, base)
        d.step = "detect"
        from .ocr import pipeline

        signals = pipeline.gather(src, classes, base, installed, plat, d.work / "ocr", workers=opts.workers,
                                  osd=osd_map, cache=_detection_cache(d))
        _step(d, "detect", "ok", pages=len(signals.pages), cached=signals.cached)
        for i, codes in signals.notes.items():
            for code in codes:
                _note(d, code, i + 1)
        lp = _language_plan(d, signals, index, installed)
        if opts.plan_only:
            _plan_record(d, lp, index, classes)
            return
        d.step = "packs"
        lp, packs, fetched, broken = _ensure_packs(d, lp, index, base)
        d.step = "ocr"
        dots = _thorough_use(d, plat)
        ocr = pipeline.read_pages(signals.pages, lp.pages, packs, base, plat, signals=signals, workers=opts.workers,
                                  thorough=dots)
        layer_input = pipeline.layer_pages(ocr, signals.pages, classes)
        _step(d, "ocr", "ok", pages=len(ocr), packs=list(lp.packs_needed), thorough=d.thorough)
    elif opts.plan_only:
        _plan_record(d, None, None, classes)
        return

    # ---- the build
    import pikepdf

    from .pdf import meta, shrink, textlayer

    d.step = "layer"
    layer_rep = None
    flattened: list[str] = []
    claims: list[str] = []
    changed = mapped or bool(stripped) or bool(orient_rep is not None and orient_rep.changed)
    built = src
    if layer_input or copy:
        built = d.work / "built.pdf"
        with pikepdf.open(src) as pdf:
            if layer_input:
                fonts = _fontset(base, packs)
                layer_rep = textlayer.write_layer(pdf, layer_input, fonts, marker={"Version": __version__})
            if copy:
                d.step = "signatures"
                from .pdf import strip

                flattened = strip.flatten_signatures(pdf)
                d.step = "metadata"
                claims = meta.drop_conformance_claims(pdf)
                meta.mark_derived(pdf, d.inputs[0].name, d.seal.sha256[0])
            _save_work(pdf, built)
        changed = True
    expected = {i: t for i, t in (layer_rep.expected_text if layer_rep else {}).items() if t.strip()}
    layered = bool(expected)
    if layer_rep is not None:
        _step(d, "layer", "ok" if layered else "skipped", pages=len(expected),
              chars_written=sum(layer_rep.chars_written.values()), runs=layer_rep.runs,
              dropped=dict(sorted(layer_rep.dropped.items())))
        if layer_rep.dropped:
            _warn(d, "layer_dropped_chars")
    if copy:
        _step(d, "signatures", "changed" if flattened else "ok", fields_removed=len(flattened))
        _step(d, "metadata", "changed", derived_from=True, claims_dropped=claims)

    d.step = "shrink"
    final = d.work / "final.pdf"
    shrink_rep = shrink.lossless_shrink(built, final, content_changed=changed)
    d.rec.shrink = to_jsonable(shrink_rep)
    _step(d, "shrink", "ok", before=shrink_rep.before, after=shrink_rep.after, byte_copy=shrink_rep.byte_copy)
    if shrink_rep.revisions_dropped:
        _warn(d, "revisions_dropped")
    if shrink_rep.objects_dropped and d.kind == "pdf":
        _warn(d, "objects_dropped")

    # ---- what the run produced, page by page
    if lp is not None and signals is not None:
        _page_records(d, classes, signals, lp, ocr, decisions, orient_rep, strip_rep, stripped, layer_rep, index_of=d.batch.index())
        _language_record(d, lp, d.batch.index(), fetched, broken)
    else:
        d.rec.language = {"plan": "not_needed", "packs": [], "canaries": [], "detection": {}}

    # ---- write or not
    d.step = "decide"
    suffix = _suffix(d, copy, layered or mapped)
    if d.kind == "pdf" and copy and not layered and not mapped and not d.encrypted:
        # No text came out of the pages, so a copy would add nothing searchable to the original.
        _done_without_output(d, "nothing_written")
        return
    if d.kind == "pdf" and not copy and not layered and not mapped:
        if stripped:
            # An old layer was removed and nothing replaced it: the original keeps more text.
            _done_without_output(d, "nothing_written")
            return
        if shrink_rep.byte_copy or shrink_rep.after > shrink_rep.before * (1.0 - MIN_SAVING):
            _done_without_output(d, "nothing_written")
            return

    d.step = "verify"
    from .pdf import verify

    turns = _turns(decisions, orient_rep)
    touched = _touched(orient_rep, expected)
    ctx = verify.BuildContext(
        reference=d.reference, built=final, turns=turns, touched=frozenset(touched), expected=expected,
        assembled=d.assembled, allowed_meta_changes=frozenset({"XMP"}) if copy else frozenset(),
        original_sha256=d.seal.sha256[0] if copy else None, original_name=d.inputs[0].name if copy else None,
        font_expected=dict(fm.expected) if mapped and fm is not None else {})
    checks = verify.verify_build(ctx)
    d.rec.verify = list(checks)
    failed = [c for c in checks if not c.ok]
    if failed:
        raise VerifyFailed("verify_failed", "A check of the new file failed, so nothing was written.",
                           checks=failed)
    _step(d, "verify", "ok", checks=len(checks))

    d.step = "commit"
    dest = _dest_for(d, suffix)
    ref = commit(final, dest, d.seal, d.reports or None, d.opts.provenance, adds_layer=adds_layer or mapped,
                 inputs=_all_inputs(d), overwrite=d.opts.overwrite, replaces=d.replaces, journal=d.lock)
    d.batch.reserved.add(_fold(ref.path))
    if copy:
        d.rec.output = record.output_entry(ref.path, ref.sha256, ref.bytes, original=d.inputs[0],
                                           original_sha256=d.seal.sha256[0], copy_reasons=d.copy_reasons)
    else:
        d.rec.output = record.output_entry(ref.path, ref.sha256, ref.bytes)
    _step(d, "commit", "ok", bytes=ref.bytes)
    if adds_layer and not d.thorough and _thorough_offered():
        d.rec.offers.append(record.thorough_offer())
    d.rec.metadata = to_jsonable(meta.diff(meta.snapshot(d.reference), meta.snapshot(final)))
    if copy:
        if flattened or set(d.copy_reasons) & _SIGNATURE_REASONS:
            _note(d, "signature_not_carried")
        if claims:
            _note(d, "conformance_claim_dropped")
        if d.encrypted:
            _note(d, "copy_unlocked")
    if lp is not None:
        _language_offers(d, lp)

    if not copy:
        d.step = "smaller"
        _smaller(d, final, ref, adds_layer)

    d.step = "seal"
    d.seal.check()
    _note(d, "input_unchanged")
    d.rec.status, d.rec.exit = "done", 0


def _done_without_output(d: _Doc, code: str) -> None:
    """A finished file with nothing written: the input untouched, the reason noted, exit 0."""
    assert d.seal is not None
    if not d.rec.language:
        d.rec.language = {"plan": "not_needed", "packs": [], "canaries": [], "detection": {}}
    _note(d, code)
    d.step = "seal"
    d.seal.check()
    _note(d, "input_unchanged")
    d.rec.output = None
    d.rec.offers = []
    d.rec.status, d.rec.exit = "done", 0


def _font_pages(fm: Any) -> dict[int, str]:
    """page -> ``repaired``, ``partial`` or ``unrepaired`` for classify (§15.1)."""
    if fm is None:
        return {}
    out = {i: "unrepaired" for i in fm.pages_unrepaired}
    if fm.changed:
        out.update({i: "partial" for i in fm.pages_partial})
        out.update({i: "repaired" for i in fm.pages_repaired})
    else:
        out.update({i: "unrepaired" for i in (*fm.pages_partial, *fm.pages_repaired)})
    return out


def _fontmap(d: _Doc) -> Any:
    """The ``fontmap`` step (DESIGN §15): find the fonts whose codes have no map, prove what each glyph
    draws, and write the maps into a new work copy, which becomes ``d.source``. The reference stays the
    input, so verify proves no page renders differently."""
    import pikepdf

    from .pdf import fontmap

    assert d.work is not None and d.source is not None
    d.step = "fontmap"
    work = d.work / "fontmap"
    opts = d.opts
    try:
        res, a = fontmap.repair(d.source, work, references=_font_index(), plan_only=opts.plan_only,
                                read=None if opts.plan_only else (lambda pages: _fontmap_read(d, pages)))
        if res.changed and not opts.plan_only:
            dest = d.work / "fontmapped.pdf"
            with pikepdf.open(d.source) as pdf:
                fontmap.write(pdf, res)
                _save_work(pdf, dest)
            d.source = dest
    except PdfIntakeError:
        raise
    except Exception as exc:  # noqa: BLE001 - a repair that can't run leaves the file as 0.2.0 handled it
        _step(d, "fontmap", "failed", error=_exc_kind(exc))
        return None
    finally:
        fontmap.cleanup(work)
    if not a.glyphs:
        _step(d, "fontmap", "ok", fonts=0)
        return res
    _step(d, "fontmap", "changed" if res.changed and not opts.plan_only else "ok",
          fonts=res.fonts_repaired, fonts_left=res.fonts_left,
          codes={k: res.codes.get(k, 0) for k in (*fontmap.EVIDENCE_COUNTED, "unproven")},
          pages=[i + 1 for i in res.pages_repaired], partial=[i + 1 for i in res.pages_partial],
          unrepaired=[i + 1 for i in res.pages_unrepaired], ocr_pages=[i + 1 for i in res.ocr_pages],
          planned=bool(opts.plan_only))
    return res


def _exc_kind(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if name.isidentifier() and len(name) <= 64 else "error"


_FONT_INDEX: list[Any] = []


def _font_index() -> Any:
    """The installed fonts, indexed once per process (§15.2). A seam the tests replace."""
    from .pdf import fontprog

    if not _FONT_INDEX:
        _FONT_INDEX.append(fontprog.ReferenceIndex())
    return _FONT_INDEX[0]


def _fontmap_read(d: _Doc, pages: Sequence[int]) -> dict[int, tuple[Any, list[Any]]]:
    """OCR evidence for the font-map repair: ``pages`` read as ``run`` reads an OCR page (render, pass 1,
    the language plan, the pack's readers), without thorough mode. Nothing is written to the record but
    the packs fetched for them."""
    from .ocr import pipeline
    from .pdf.classify import PageClass

    assert d.work is not None and d.source is not None
    plat = d.batch.platform_key()
    index = d.batch.index()
    base = d.batch.base()
    installed = d.batch.installed()
    classes = [PageClass(i, "image_only", 0, 0, 0.0, 0.0, True, False) for i in sorted(pages)]
    signals = pipeline.gather(d.source, classes, base, installed, plat, d.work / "fontmap-ocr",
                              workers=d.opts.workers, osd={}, cache=None)
    lp = _language_plan(d, signals, index, installed)
    lp, packs, _fetched, _broken = _ensure_packs(d, lp, index, base, record_step=False)
    ocr = pipeline.read_pages(signals.pages, lp.pages, packs, base, plat, signals=signals, workers=d.opts.workers)
    images = {p.index: p for p in signals.pages}
    return {o.index: (images[o.index], list(o.reads)) for o in ocr if o.index in images}


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


# --------------------------------------------------------------------------- inputs

def _prepare_pdf(d: _Doc) -> None:
    assert d.work is not None and d.seal is not None
    d.step = "refusals"
    copy = d.work / "input.pdf"
    _copy_into_work(d.inputs[0], copy)
    if integrity.sha256_file(copy) != d.seal.sha256[0]:
        raise InputChanged("input_changed", f"The input {d.inputs[0]} changed while it was being read; run again.",
                           path=str(d.inputs[0]))
    report = refusals.read(copy)
    d.reports.append(report)
    _refusal_check(d, report)
    if report.repaired:
        _warn(d, "input_repaired")
    if report.jbig2_streams:
        _warn(d, "contains_jbig2")
    d.source = d.reference = copy
    if not report.encrypted:
        _step(d, "refusals", "clear", walk=list(report.walk_codes), raw=list(report.raw_markers))
        return
    # Decrypt once (DESIGN §11.1): the copy is saved without encryption, and every later step reads it.
    d.step = "decrypt"
    d.encrypted = True
    opened = passwords.open_with(copy, _LazyPasswords(d.batch))
    if opened is None:
        d.skipped = True
        _step(d, "decrypt", "skipped", reason="no_saved_password_opened_it")
        return
    plain = d.work / "decrypted.pdf"
    try:
        _save_work(opened, plain)
    finally:
        opened.close()
        opened = None
    _remove_work(copy)                                 # the encrypted copy isn't needed any more
    again = refusals.read(plain)
    d.reports.append(again)
    _refusal_check(d, again)
    d.source = d.reference = plain
    _step(d, "decrypt", "changed", encrypted=str(report.encrypted))
    _step(d, "refusals", "clear", walk=list(report.walk_codes) + list(again.walk_codes),
          raw=list(report.raw_markers) + list(again.raw_markers))


def _refusal_check(d: _Doc, report: RefusalReport) -> None:
    hard = refusals.refusal(report)
    if hard is None:
        return
    code, message = hard
    d.rec.refusal = {"code": code, "message": message,
                     "detectors": {"walk": list(report.walk_codes), "raw": list(report.raw_markers)},
                     "repaired": bool(report.repaired)}
    raise Refused(code, message)


def _prepare_images(d: _Doc) -> None:
    import pikepdf

    from .intake import assemble
    from .pdf import meta

    assert d.work is not None and d.seal is not None
    d.step = "assemble"
    # Assemble from sealed copies, never from the user's files again (decided 2026-11-05): an
    # image saved over and then put back between the seal and here can't reach the output.
    copies: list[Path] = []
    for i, src in enumerate(d.inputs):
        copy = d.work / f"image-{i:04d}"
        _copy_into_work(src, copy)
        if integrity.sha256_file(copy) != d.seal.sha256[i]:
            raise InputChanged("input_changed", f"The input {src} changed while it was being read; run again.",
                               path=str(src))
        copies.append(copy)
    raw = d.work / "assembled-raw.pdf"
    try:
        pages = assemble.assemble(copies, raw)
    except PdfIntakeError as exc:
        raise _named_for_user(exc, copies, d.inputs) from None
    built = d.work / "assembled.pdf"
    with pikepdf.open(raw) as pdf:
        meta.set_built_info(pdf, producer=PRODUCER, created=_assembly_date(d))
        _save_work(pdf, built)
    d.assembled = pages
    d.source = d.reference = built
    for i, page in enumerate(pages):
        for code in page.warnings:
            _warn(d, code, i + 1)
    _step(d, "assemble", "ok", pages=len(pages), routes=[p.route for p in pages])


def _named_for_user(exc: PdfIntakeError, copies: Sequence[Path], originals: Sequence[Path]) -> PdfIntakeError:
    """``exc`` with each work copy's path in its message and detail replaced by the user's file."""
    def swap(text: str) -> str:
        for copy, original in sorted(zip(copies, originals), key=lambda pair: -len(str(pair[0]))):
            text = text.replace(str(copy), str(original))
        return text

    exc.message = swap(exc.message)
    exc.args = (exc.message,)
    exc.detail = {k: swap(v) if isinstance(v, str) else v for k, v in exc.detail.items()}
    return exc


def _copy_reasons(reports: Sequence[RefusalReport], provenance: str, adds_layer: bool) -> list[str]:
    found: set[str] = set()
    for report in reports:
        found.update(refusals.copy_reasons(report, provenance, adds_layer=adds_layer))
    given = refusals.check_provenance(provenance)
    if given is not None:
        found.add(given[0])
    return [c for c in refusals.CODES if c in found]


# --------------------------------------------------------------------------- the work folder

def _cache_key(path: Path) -> str:
    return unicodedata.normalize("NFC", os.path.normcase(str(Path(path).resolve()))).casefold()


def _in_cache(path: Path) -> bool:
    child, root = _cache_key(path), _cache_key(platforms.cache_root())
    return child.startswith(root.rstrip(os.sep) + os.sep)


def _work_path(dest: Path) -> Path:
    dest = Path(dest)
    if not _in_cache(dest):
        raise PdfIntakeError("not_a_work_path", f"{dest} isn't inside pdfintake's cache, so nothing was written there.")
    return dest


def _save_work(pdf: pikepdf.Pdf, dest: Path) -> None:
    """Save a work file inside the cache: never encrypted, the XMP never rewritten (DESIGN §11.2),
    and readable only by its owner, like the 0700 folder it is in (a decrypted copy included)."""
    target = _work_path(dest)
    pdf.save(target, encryption=False, deterministic_id=True, fix_metadata_version=False)
    if sys.platform != "win32":
        os.chmod(target, 0o600)


def _lock_work(lock: Path, *, create: bool) -> int | None:
    """An exclusive lock on a work folder's ``LOCK_NAME`` file, held as long as the returned descriptor is open.

    ``create`` makes the file (0600) for a run's own new folder; a held lock there is an error.
    Without it, the sweep asks whether a folder is orphaned: None means a live run holds it, and a
    missing file raises FileNotFoundError. Only ever inside the cache; None on Windows, which has no
    such lock here.
    """
    path = _work_path(lock)
    if sys.platform == "win32":
        if create:
            return None
        raise FileNotFoundError(str(path))
    import fcntl

    flags = os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK
    flags |= (os.O_RDWR | os.O_CREAT | os.O_APPEND) if create else os.O_RDONLY
    fd = os.open(path, flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        if create:
            raise
        return None
    return fd


def _journal_dest(lock: int | None, dest: Path) -> None:
    """Note ``dest`` in the work folder's lock file before it is written (decided 2026-11-06).

    A run killed with SIGKILL between writing the partial file beside ``dest`` and renaming it can
    clean nothing up, so the partial file would stay in the user's folder. The next run's
    ``_sweep_orphans`` reads what is noted here and hands each path to ``fsutil.sweep_partial``.
    Each entry is the absolute path as file-system bytes, ended by a NUL (no path can hold one),
    written and flushed before ``atomic_write`` creates the partial file. ``lock`` is the descriptor
    ``_lock_work(create=True)`` returned, so this writes only into a work folder's lock file; None
    (Windows, where nothing sweeps) notes nothing.
    """
    if lock is None:
        return
    view = memoryview(os.fsencode(os.path.abspath(dest)) + b"\0")
    while view:
        view = view[os.write(lock, view):]
    try:
        os.fsync(lock)
    except OSError:
        pass


@contextlib.contextmanager
def _write_journal(lock: int | None) -> Iterator[int | None]:
    """The lock file a write's destination is noted in: ``lock`` when the caller has one, else a
    short-lived work folder of its own (decided 2026-11-06).

    A run and ``apply --lang`` pass their work folder's lock. ``export`` and ``apply --choice`` have
    none, and their partial file would otherwise never be swept after a SIGKILL: export builds in
    staging, which no sweep reads. The folder lives in ``<cache>/work`` for the one write, its lock
    held, and is removed afterwards; a kill leaves it unlocked, so the next run's ``_sweep_orphans``
    reads the note. None on Windows, where nothing sweeps.
    """
    if lock is not None or sys.platform == "win32":
        yield lock
        return
    work = fsutil.work_dir("work", f"write-{__version__}-{secrets.token_hex(6)}")
    own = _lock_work(work / LOCK_NAME, create=True)
    try:
        yield own
    finally:
        _remove_work(work)
        if own is not None:
            os.close(own)


def _copy_into_work(src: Path, dest: Path) -> None:
    """Copy a file into the cache (an input, or a render to or from the detection cache); the source
    is only read."""
    shutil.copyfile(src, _work_path(dest))


def _remove_work(path: Path) -> None:
    """Delete a work or staging file or folder (renders, the decrypted copy, a candidate).

    Only ever below ``<cache>/work`` or ``<cache>/staging``; any other path is left alone.
    """
    key = _cache_key(path)
    for kind in fsutil.PRUNABLE_KINDS:
        root = _cache_key(platforms.cache_root() / kind)
        if key.startswith(root.rstrip(os.sep) + os.sep):
            if Path(path).is_dir() and not Path(path).is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            return


# --------------------------------------------------------------------------- the detection cache (DESIGN §9.1)

def _cache_folder_name(shas: Sequence[str]) -> str:
    """``<input sha>-<version>``: the input's SHA-256, or one SHA-256 over several images' in order."""
    digest = shas[0] if len(shas) == 1 else integrity.sha256_bytes("\n".join(shas).encode("ascii"))
    return f"{digest}-{__version__}"


def _cache_folder(d: _Doc) -> Path | None:
    """The input's cache folder, ``<cache>/work/<input sha>-<version>/``; None for an encrypted input,
    whose renders, votes and reads would be its decrypted pages and go with the run's work folder
    (DESIGN §11.1), or when it can't be made."""
    if d.encrypted or d.seal is None:
        return None
    try:
        return fsutil.work_dir("work", _cache_folder_name(list(d.seal.sha256)))
    except (OSError, PdfIntakeError):
        return None


def _detection_cache(d: _Doc) -> _DetectionCache | None:
    """The input's detection cache (``_cache_folder``), or None."""
    folder = _cache_folder(d)
    return None if folder is None else _DetectionCache(folder)


def _orientation_load(folder: Path, key: str, base: InstalledPack) -> tuple[dict[int, Any], dict[int, Any]] | None:
    """The votes and OSD results stored in the input's cache under ``key``, or None. The voter's and
    OSD's models are hash-checked before a stored result is used, as an open would check them: a
    tampered model stops the file (IntegrityError), cached or not; a missing one is a fresh orientation."""
    from .ocr import pipeline

    raw = _read_small(folder / pipeline.ORIENT_CACHE_FILE, pipeline.ORIENT_MAX_BYTES)
    found = pipeline.read_orientation(raw, key) if raw is not None else None
    if found is None:
        return None
    try:
        pipeline.check_orientation_models(base)
    except IntegrityError:                             # a PackError too: it must stop the file, not miss
        raise
    except PackError:
        return None
    return found


def _orientation_keep(folder: Path, key: str, votes: Mapping[int, Any], osd: Mapping[int, Any]) -> None:
    """Store the votes and OSD results in the input's cache under ``key``, replacing what was there.
    A result that can't be stored faithfully, or a cache that can't be written, only costs the next run."""
    from .langpacks import store
    from .ocr import pipeline

    data = pipeline.orientation_entry(votes, osd, key)
    if data is None:
        return
    try:
        store._write_json_atomic(_work_path(folder / pipeline.ORIENT_CACHE_FILE), data)
    except (OSError, PdfIntakeError, ValueError, TypeError):
        pass


def _assembly_date(d: _Doc) -> datetime:
    """The ``/CreationDate`` of a PDF built from images: when pdfintake first built it from these
    exact images (DESIGN §9.1, decided 2026-11-06). It is kept in the input's cache folder, so a
    re-run and ``apply --lang`` build a byte-identical work PDF and reuse the detection; with no
    stored date (a first run, a pruned cache) it is now, to the second."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    folder = _cache_folder(d)
    if folder is None:
        return now
    stored = _read_assembly_date(_read_small(folder / ASSEMBLED_FILE, 4096))
    if stored is not None and stored <= now:
        return stored
    from .langpacks import store

    try:
        store._write_json_atomic(_work_path(folder / ASSEMBLED_FILE),
                                 {"v": 1, "created": now.strftime(ASSEMBLED_DATE_FORMAT)})
    except (OSError, PdfIntakeError, ValueError, TypeError):
        pass
    return now


def _read_assembly_date(raw: bytes | None) -> datetime | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict) or set(data) != {"v", "created"} or data["v"] != 1:
            return None
        text = data["created"]
        if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
            return None
        return datetime.strptime(text, ASSEMBLED_DATE_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return None


def _read_small(path: Path, cap: int) -> bytes | None:
    """A cache file's bytes; None when it isn't a regular file (a link or a FIFO is never followed or
    waited on), holds more than ``cap`` bytes, or can't be read."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                     | getattr(os, "O_BINARY", 0))
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
            return None
        chunks: list[bytes] = []
        total = 0
        while total <= cap:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        return None if total > cap else b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)


class _DetectionCache:
    """An input's detection cache on disk: ``<cache>/work/<input sha>-<version>/`` (DESIGN §9.1).

    Each stored result is a folder of its own, ``e-<random>``, holding the renders and
    ``pipeline.CACHE_FILE``, which is written last, so a folder without it is never read. What is
    stored and what counts as a match are ``pipeline``'s (``cache_entry``, ``read_cache``); this class
    only moves files inside the cache, through ``_copy_into_work`` and ``_remove_work``. Neither
    ``load`` nor ``keep`` raises for a cache problem.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def _entries(self, *, complete: bool = True) -> list[Path]:
        """The stored results, newest first; with ``complete=False``, every entry folder."""
        from .ocr import pipeline

        try:
            listing = list(os.scandir(self.folder))
        except OSError:
            return []
        found: list[tuple[int, str, Path]] = []
        for e in listing:
            if not e.name.startswith(CACHE_ENTRY_PREFIX):
                continue
            try:
                if not e.is_dir(follow_symlinks=False):
                    continue
                st = os.lstat(Path(e.path) / pipeline.CACHE_FILE) if complete else os.lstat(e.path)
            except OSError:
                continue
            if not complete or stat.S_ISREG(st.st_mode):
                found.append((st.st_mtime_ns, e.name, Path(e.path)))
        return [p for _t, _n, p in sorted(found, reverse=True)]

    def load(self, key: str, work: Path) -> Any:
        """The result stored under ``key`` with its renders copied into ``work``, or None.

        The copies are what pass 2 reads, so an entry removed or replaced later never touches this
        run; each copy must have its recorded size and SHA-256 (``CachedDetection.signals``)."""
        from .ocr import pipeline

        work = Path(work)
        for entry in self._entries():
            raw = _read_small(entry / pipeline.CACHE_FILE, pipeline.MAX_CACHE_BYTES)
            found = pipeline.read_cache(raw, key) if raw is not None else None
            if found is None:
                continue
            try:
                for name in found.renders:
                    src = entry / name
                    if not stat.S_ISREG(os.lstat(src).st_mode):
                        raise FileNotFoundError(str(src))
                    _copy_into_work(src, work / name)
            except (OSError, PdfIntakeError):
                continue
            signals = found.signals(work)
            if signals is not None:
                return signals
        return None

    def keep(self, key: str, signals: Any) -> None:
        """Store ``signals`` under ``key`` as a new entry, then remove the input's other entries."""
        from .langpacks import store
        from .ocr import pipeline

        try:
            entry = _work_path(self.folder / f"{CACHE_ENTRY_PREFIX}{secrets.token_hex(8)}")   # only ever in the cache
        except PdfIntakeError:
            return
        try:
            entry.mkdir(mode=0o700)
            stored = []
            for p in signals.pages:
                dest = entry / Path(p.path).name
                _copy_into_work(p.path, dest)
                stored.append(dataclasses.replace(p, path=dest))
            data = pipeline.cache_entry(dataclasses.replace(signals, pages=stored), key)
            if data is None:
                raise ValueError("this result can't be stored faithfully")
            store._write_json_atomic(entry / pipeline.CACHE_FILE, data)
        except (OSError, PdfIntakeError, ValueError, TypeError):
            _remove_work(entry)
            return
        for old in self._entries(complete=False):
            if old != entry:
                _remove_work(old)


def _fold(path: Path) -> str:
    return unicodedata.normalize("NFC", os.path.normcase(os.path.abspath(str(path)))).casefold()


# --------------------------------------------------------------------------- strip and orientation

def _strip(d: _Doc, src: Path, pages: Sequence[int], classes: Sequence[PageClass]) -> tuple[Path, Any]:
    import pikepdf

    from .pdf import strip

    assert d.work is not None
    dest = d.work / "stripped.pdf"
    with pikepdf.open(src) as pdf:
        rep = strip.strip_text(pdf, pages, {c.index: c for c in classes})
        strip.drop_unused(pdf, pages)                  # only the stripped pages (DESIGN §11.2)
        _save_work(pdf, dest)
    for c in classes:
        if c.index in pages and c.kind == "invisible_layer":
            _warn(d, "existing_layer_replaced", c.index + 1)
    return dest, rep


def _rotate_flags(src: Path, pages: Sequence[int]) -> dict[int, int]:
    import pikepdf

    from .pdf import orient

    out: dict[int, int] = {}
    with pikepdf.open(src) as pdf:
        for i in pages:
            node: Any = pdf.pages[i].obj
            value: Any = None
            for _ in range(64):
                if node is None:
                    break
                if "/Rotate" in node:
                    value = node.get("/Rotate")
                    break
                node = node.get("/Parent")
            try:
                out[i] = orient.normalize_rotate(int(float(value))) if value is not None else 0
            except (TypeError, ValueError, OverflowError):
                out[i] = 0
    return out


def _orient(d: _Doc, src: Path, pages: Sequence[int], base: InstalledPack) -> tuple[Path, Any, dict[int, Any], dict[int, Any]]:
    """Turn the OCR pages upright: the RapidOCR vote, OSD as the second opinion (DESIGN §2.2 U6).

    The votes and OSD results come from the input's orientation cache when it holds them for this
    exact work PDF and these models (DESIGN §9.1); otherwise every page is rendered and read, and the
    result is kept when the voter and OSD both opened and nothing failed along the way. The decisions
    are made from the votes every time."""
    from .ocr import pipeline
    from .pdf import orient

    assert d.work is not None
    flags = _rotate_flags(src, pages)
    folder = _cache_folder(d)
    key = None
    if folder is not None and pages:
        key = pipeline.orientation_key(src, pages, base, d.batch.platform_key(),
                                       {"vote_dpi": VOTE_DPI, "osd_dpi": OSD_DPI})
    found = _orientation_load(folder, key, base) if folder is not None and key is not None else None
    cached = found is not None and sorted(found[0]) == sorted(pages)
    if cached:
        assert found is not None
        votes, osd_map = found
        voter_ok = osd_ok = True                       # only a pass with both open is ever stored
    else:
        votes, osd_map, voter_ok, osd_ok, failed = _orient_read(d, src, pages, base)
        if folder is not None and key is not None and voter_ok and osd_ok and not failed:
            _orientation_keep(folder, key, votes, osd_map)
    decisions = {i: orient.decide(votes[i], osd_map[i], flags.get(i, 0)) for i in pages}
    dest = d.work / "upright.pdf"
    rep = orient.normalize(src, dest, list(decisions.values()))
    for i, dec in decisions.items():
        for code in orient.decision_warnings(dec):
            _warn(d, code, i + 1)
    for i in rep.kept_annotated:
        _warn(d, "not_turned_annotations", i + 1)
    for i in rep.kept_flag:
        _note(d, "rotate_kept", i + 1)
    _step(d, "orientation", "changed" if rep.changed else "ok", baked=[i + 1 for i in rep.baked],
          cleared=[i + 1 for i in rep.cleared], kept=[i + 1 for i in rep.kept_annotated],
          voter=voter_ok, osd=osd_ok, cached=cached)
    return (dest if rep.changed else src), rep, decisions, osd_map


def _orient_read(d: _Doc, src: Path, pages: Sequence[int], base: InstalledPack
                 ) -> tuple[dict[int, Any], dict[int, Any], bool, bool, bool]:
    """Render and read every page for orientation: (votes, OSD results, voter opened, OSD opened,
    something failed in a way the next run may not: a voter read or an OSD call)."""
    from .ocr import pipeline
    from .pdf import orient
    from .pdf.render import Renderer

    assert d.work is not None
    votes: dict[int, Any] = {}
    osd_map: dict[int, Any] = {}
    watch = pipeline.FailureWatch()
    voter = _open_voter(base)
    osd = _open_osd(base)
    n = min(len(orient.TURNS), d.opts.workers)
    voters = _ThreadVoters(voter, base) if voter is not None and n > 1 else voter
    watched = pipeline.watch_session(voters, watch)   # orient.vote swallows a failed read; the watch sees it
    pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="pdfintake-orient") if voters is not voter else None
    try:
        with Renderer(src) as r:
            for i in pages:
                if watched is not None:
                    img = r.render_scaled(i, VOTE_DPI)
                    try:
                        v = orient.vote(img, i, watched, dpi=VOTE_DPI, workdir=d.work / "orient", pool=pool)
                    finally:
                        img.close()
                else:
                    v = orient.OrientVote(i, 0, 0.0, 0)
                o = None
                if osd is not None:
                    big = r.render_scaled(i, OSD_DPI, gray=True)
                    try:
                        o = osd.detect(big, OSD_DPI)
                    except (EngineFailure, OSError, ValueError, RuntimeError):
                        o = None
                        watch.note()
                    finally:
                        big.close()
                votes[i] = v
                osd_map[i] = o
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for session in (voters, osd):
            close = getattr(session, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:                      # noqa: BLE001 - closing never fails a file
                    pass
    return votes, osd_map, voter is not None, osd is not None, watch.failed


def _turns(decisions: Mapping[int, Any], rep: Any) -> dict[int, int]:
    """Page -> the clockwise turn the build applied to what the page displays (0 when kept)."""
    if rep is None:
        return {}
    moved = set(rep.baked) | set(rep.cleared)
    return {i: int(dec.turn_cw) if (dec.confident and i in moved) else 0 for i, dec in decisions.items()}


def _touched(rep: Any, expected: Mapping[int, str]) -> set[int]:
    """Pages baked, cleared or layered; a page whose /Rotate was kept on purpose isn't asked to be upright."""
    touched = set(expected)
    kept: set[int] = set()
    if rep is not None:
        touched |= set(rep.baked) | set(rep.cleared)
        kept = set(rep.kept_annotated) | set(rep.kept_flag)
    return touched - kept


# --------------------------------------------------------------------------- language

def _language_plan(d: _Doc, signals: Any, index: Index, installed: Mapping[str, InstalledPack]) -> LanguagePlan:
    from .ocr import langplan

    answer = d.answer or _LangAnswer()
    lp = langplan.plan(signals.signals, index, lang=d.opts.lang, force_lang=d.opts.force_lang,
                       installed=set(installed), fetch=d.opts.fetch, approved=answer.approved,
                       declined=answer.declined)
    return langplan.resolve_unsure(lp, index)


def _ensure_packs(d: _Doc, lp: LanguagePlan, index: Index, base: InstalledPack, *, record_step: bool = True
                  ) -> tuple[LanguagePlan, dict[str, InstalledPack], list[str], set[str]]:
    """Every pack the plan reads with, ensured; a pack that can't be used this run is broken (§12.3)."""
    from .ocr import langplan

    answer = d.answer or _LangAnswer()
    installed = d.batch.installed()
    packs: dict[str, InstalledPack] = {}
    fetched: list[str] = []
    broken: set[str] = set()
    base_id = base.manifest.id
    while True:
        newly: set[str] = set()
        for pid in lp.packs_needed:
            if pid == base_id or pid in packs or pid in broken:
                continue
            entry = index.get(pid)
            if entry is None:
                newly.add(pid)
                continue
            was = pid in installed
            try:
                pack = _ensure(entry, allow_network=d.opts.fetch == "auto" or pid in answer.approved)
            except PdfIntakeError:
                newly.add(pid)
                continue
            packs[pid] = pack
            installed[pid] = pack
            if not was:
                fetched.append(pid)
        if not newly:
            break
        broken |= newly
        lp = langplan.resolve_unsure(lp, index, broken=broken)
    for _pid in fetched:
        _note(d, "pack_fetched")
    if record_step:
        _step(d, "packs", "changed" if fetched else "ok", used=list(lp.packs_needed), fetched=fetched,
              broken=sorted(broken))
    return lp, packs, fetched, broken


def _pack_lang(index: Index, pack_id: str | None) -> str | None:
    entry = index.get(pack_id) if pack_id else None
    if entry is None:
        return None
    return entry.langid_codes[0] if entry.langid_codes else entry.id


def _language_message(d: _Doc, code: str, page: int, pp: PagePlan | None, lp: LanguagePlan,
                      index: Index) -> record.Message:
    """A §12.3 language warning in core's words, naming the language; the plain wording when it can't."""
    try:
        if pp is None:
            raise ValueError("no plan for the page")
        offer = next((o for o in lp.offers if pp.index in o.pages), None)
        if code == "language_uncertain":
            tags = [c.tag for c in pp.candidates] or [lc.tag for lc in pp.languages]
            if not tags:
                raise ValueError("no candidate")
            read_as = offer.read_as if offer is not None else _pack_lang(index, pp.pack)
            return record.language_warning(code, language=tags[0], page=page, others=tags[1:], read_as=read_as)
        if code == "no_pack_for_language":
            return record.language_warning(code, language=pp.detected, page=page)
        wanted = offer.options[0].tag if offer is not None and offer.options else None
        return record.language_warning(code, language=wanted, page=page)
    except ValueError:
        return record.warning(code, page)


def _page_records(d: _Doc, classes: Sequence[PageClass], signals: Any, lp: LanguagePlan, ocr: Sequence[Any],
                  decisions: Mapping[int, Any], orient_rep: Any, strip_rep: Any, stripped: Sequence[int],
                  layer_rep: Any, *, index_of: Index) -> None:
    from .ocr import pipeline
    from .pdf import orient

    cls = {c.index: c for c in classes}
    rendered = {p.index: p for p in signals.pages}
    plans = {pp.index: pp for pp in lp.pages}
    for o in ocr:
        n = o.index + 1
        dec = decisions.get(o.index)
        orientation = None
        if dec is not None:
            orientation = {"turn_cw": int(dec.turn_cw) if dec.confident else 0, "by": dec.by,
                           "action": orient.page_action(orient_rep, o.index) if orient_rep is not None else "none"}
        page = rendered.get(o.index)
        d.rec.pages.append(record.page_entry(o, page_class=cls[o.index].kind if o.index in cls else None,
                                             dpi=page.dpi if page is not None else None, orientation=orientation))
        for code in o.warnings:
            if code in _LANGUAGE_WARNINGS:
                _add_warning(d, _language_message(d, code, n, plans.get(o.index), lp, index_of))
            else:
                _warn(d, code, n)
        for code in pipeline.notes_for(o):
            _note(d, code, n)
        pp = plans.get(o.index)
        if pp is not None:
            for code in pp.notes:
                _note(d, code, n)
    for code in lp.notes:
        if not any(m.code == code for m in d.rec.notes):
            _note(d, code)
    if strip_rep is not None and layer_rep is not None:
        for i in stripped:
            before = strip_rep.chars_before.get(i, 0)
            after = layer_rep.chars_written.get(i, 0)
            if before and after < before * LAYER_SHRANK_RATIO:
                _warn(d, "layer_shrank", i + 1)


def _plan_label(d: _Doc, lp: LanguagePlan) -> str:
    if d.opts.force_lang:
        return "forced"
    if any(pp.composition == "hint" or pp.via == "hint" for pp in lp.pages):
        return "hinted"
    if not lp.confident or lp.offers or any(pp.via is not None for pp in lp.pages):
        return "uncertain"
    return "confident"


def _language_record(d: _Doc, lp: LanguagePlan, index: Index, fetched: Sequence[str], broken: Sequence[str]) -> None:
    packs = []
    for pid in lp.packs_needed:
        entry = index.get(pid)
        if entry is None:
            continue
        url = entry.dist.url
        packs.append({"id": entry.id, "version": entry.version, "fetched": pid in fetched,
                      "bytes": entry.dist.bytes, "host": urlparse(url).hostname if url else None,
                      "sha256": entry.dist.sha256})
    pages = []
    for pp in lp.pages:
        pages.append({"page": pp.index + 1, "pack": pp.pack, "composition": pp.composition, "via": pp.via,
                      "confident": pp.confident, "script": pp.script.name if pp.script is not None else None,
                      "languages": [lc.tag for lc in pp.languages]})
    d.rec.language = {"plan": _plan_label(d, lp), "packs": packs, "canaries": [],
                      "detection": {"pages": pages, "unknown_language": lp.unknown_language,
                                    "broken": sorted(broken)}}


def _language_offers(d: _Doc, lp: LanguagePlan) -> None:
    for o in lp.offers:
        try:
            offer = record.language_offer(used=o.read_as,
                                          others=[(c.tag, c.download_bytes) for c in o.options],
                                          pages=[i + 1 for i in o.pages], reason=o.reason, detected=o.detected)
        except ValueError:
            continue
        d.rec.offers.append(offer)


def _plan_record(d: _Doc, lp: LanguagePlan | None, index: Index | None, classes: Sequence[PageClass]) -> None:
    """``--plan``: what the run would read and download; nothing is written (DESIGN §4.1)."""
    assert d.seal is not None
    if lp is None or index is None:
        d.rec.language = {"plan": "not_needed", "packs": [], "canaries": [], "detection": {}}
    else:
        _language_record(d, lp, index, [], [])
        installed = d.batch.installed()
        for entry in d.rec.language["packs"]:
            entry["fetched"] = False
            entry["download"] = 0 if entry["id"] in installed else entry["bytes"]
        for pp in lp.pages:
            for code in pp.warnings:
                if code in _LANGUAGE_WARNINGS:
                    _add_warning(d, _language_message(d, code, pp.index + 1, pp, lp, index))
                else:
                    _warn(d, code, pp.index + 1)
    d.step = "seal"
    d.seal.check()
    _note(d, "input_unchanged")
    d.rec.status, d.rec.exit = "plan", 0


def _fontset(base: InstalledPack | None, packs: Mapping[str, InstalledPack]) -> Any:
    """The layer fonts: the base pack's first, then each other pack's, keys kept distinct."""
    from .pdf import textlayer

    assert base is not None
    specs = []
    seen: set[str] = set()
    order = [base] + [packs[k] for k in sorted(packs) if packs[k] is not base]
    for pack in order:
        for spec in pack.font_specs():
            key = spec.key if spec.key not in seen else f"{pack.manifest.id}-{spec.key}"
            seen.add(key)
            specs.append(dataclasses.replace(spec, key=key, priority=len(specs)))
    return textlayer.FontSet.from_specs(specs)


# --------------------------------------------------------------------------- names

def _suffix(d: _Doc, copy: bool, layered: bool) -> str:
    if copy:
        return SUFFIX_COPY
    if layered:
        return SUFFIX_SEARCHABLE
    return SUFFIX_COMPACT if d.kind == "pdf" else ""


def _numbered(suffix: str, n: int) -> str:
    if suffix.endswith(")"):
        return f"{suffix[:-1]} {n})"
    return f"{suffix} {n}"


def _out_folder(first: Path, opts: RunOptions) -> Path:
    return Path(os.path.abspath(Path(opts.outdir).expanduser())) if opts.outdir is not None else first.parent


_JOINERS = frozenset({0x200C, 0x200D, 0x034F})     # ZWNJ, ZWJ, the combining grapheme joiner


def _extends(ch: str) -> bool:
    """Whether ``ch`` belongs to the character before it (a mark, a joiner, a variation selector,
    an emoji modifier or tag), so a name is never cut between them."""
    cp = ord(ch)
    return (unicodedata.category(ch) in ("Mn", "Me", "Mc") or cp in _JOINERS
            or 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF
            or 0x1F3FB <= cp <= 0x1F3FF or 0xE0020 <= cp <= 0xE007F)


def _fit_stem(stem: str) -> str:
    """``stem``, or its start plus "…" when ``stem`` and the longest output tail wouldn't fit a file name.

    A legal input name can be too long once `` (searchable).pdf`` is added. The stem is cut between
    whole characters as people see them (never inside a combining sequence, an emoji sequence or a
    flag), so every input still gets an output (decided 2026-11-05). The same cut is used for every
    suffix, so the numbering in ``_assign_slots`` sees the names the outputs get.
    """
    room = fsutil.NAME_MAX - fsutil._name_length(SLOT_LABEL_ROOM)
    if fsutil._name_length(stem) <= room:
        return stem
    k = len(stem)
    while k > 0:
        k -= 1
        while k > 0 and (_extends(stem[k]) or ord(stem[k - 1]) in _JOINERS):
            k -= 1
        cut = stem[:k]
        flags = len(cut) - len(cut.rstrip("".join(chr(c) for c in range(0x1F1E6, 0x1F200))))
        if flags % 2:
            cut = cut[:-1]                             # never half a flag
        if fsutil._name_length(cut.rstrip() + ELLIPSIS) <= room:
            return cut.rstrip() + ELLIPSIS
    return ELLIPSIS


def _dest_for(d: _Doc, suffix: str) -> Path:
    """The output path: ``-o`` as given, else ``<first stem><suffix>.pdf`` beside it or in ``outdir``.

    Documents of one batch that would share the name are numbered (``(searchable 2)``) by their
    slot, fixed in input order before the batch ran (``_assign_slots``), so a failed document keeps
    its number. A name another document already wrote is skipped as well. A stem too long for a file
    name is shortened (``_fit_stem``).
    """
    if d.replaces is not None:
        return d.replaces.path
    if d.opts.out is not None:
        return Path(os.path.abspath(Path(d.opts.out).expanduser()))
    first = d.inputs[0]
    folder = _out_folder(first, d.opts)
    stem = _fit_stem(first.stem)
    n = max(1, d.slot)
    while True:
        label = suffix if n == 1 else _numbered(suffix, n)
        dest = folder / f"{stem}{label}.pdf"
        if _fold(dest) not in d.batch.reserved:
            return dest
        n += 1


def _all_inputs(d: _Doc) -> list[Path]:
    """Every input of the invocation, this document's first: an output may replace none of them."""
    return [*d.inputs, *(p for p in d.batch.all_inputs if p not in d.inputs)]


def _guard_dest(d: _Doc, dest: Path, copy: bool) -> None:
    if d.replaces is not None:
        return
    fsutil.guard_output(_all_inputs(d), dest, overwrite=d.opts.overwrite and not copy)


# --------------------------------------------------------------------------- commit: the one writer

def commit(built: Path, dest: Path, seal: InputSeal, report: RefusalReport | Sequence[RefusalReport] | None,
           provenance: str, *, adds_layer: bool, inputs: Sequence[Path], overwrite: bool,
           replaces: FileRef | None = None, journal: int | None = None) -> FileRef:
    """Write ``built`` to ``dest`` in the user's folder: the only writer besides ``apply``.

    In order: the refusal re-check (a hard code in any report refuses, exit 4; with copy reasons,
    ``overwrite`` is the usage error ``copy_never_overwrites``), the seal check (InputChanged, exit
    5), ``fsutil.guard_output`` (the output is never an input, never in the cache, never an existing
    file without ``overwrite``), then ``fsutil.atomic_write`` and a hash check of what landed.
    ``atomic_write`` gets the same ``overwrite``, so without it a file that takes the name after the
    guard is kept and this raises UsageError ``output_exists`` (exit 2; decided 2026-11-06).

    ``report`` is the run's refusal report (several for a decrypted input), or None for images and
    for an export, whose own two detectors already decided (the provenance still counts).
    ``replaces`` is ``apply --lang`` replacing the output an earlier run wrote: ``dest`` must be
    that file, unchanged since (StaleRecord, exit 5), and it may be replaced even in copy mode,
    since it is pdfintake's own copy of the same original; the replace itself is a compare-and-swap
    (``fsutil.atomic_write(expect_sha256=...)``), so a save that lands on ``dest`` while the new file
    is copied is kept and this is StaleRecord ``output_changed``. ``journal`` is the run's
    work-folder lock (``_lock_work``): ``dest`` is noted there before the write, so a partial file a
    SIGKILL leaves behind is swept by the next run. Without one (``export``, a direct call) commit
    makes a short-lived work folder of its own for the note (``_write_journal``, decided
    2026-11-06), so every write is noted, whoever calls.
    """
    reports: list[RefusalReport] = []
    if report is not None:
        reports = list(report) if isinstance(report, (list, tuple)) else [report]  # type: ignore[list-item]
    for r in reports:
        hard = refusals.refusal(r, adds_layer=adds_layer)
        if hard is not None:
            raise Refused(hard[0], hard[1])
    reasons = _copy_reasons(reports, provenance, adds_layer)
    if reasons and overwrite and replaces is None:
        raise record.usage_error("copy_never_overwrites")
    seal.check()
    dest = Path(dest)
    if replaces is not None:
        if _fold(dest) != _fold(replaces.path):
            raise StaleRecord("output_moved", "The output to replace isn't where the record says; run again.")
        try:
            current = integrity.sha256_file(dest)
        except IntegrityError:
            raise StaleRecord("output_changed", f"The output {dest} is gone since the record was written; run again.",
                              path=str(dest)) from None
        if current != replaces.sha256:
            raise StaleRecord("output_changed", f"The output {dest} changed since the record was written; run again.",
                              path=str(dest))
        fsutil.guard_output(inputs, dest, overwrite=True)
    else:
        fsutil.guard_output(inputs, dest, overwrite=overwrite)
    digest = integrity.sha256_file(built)
    size = os.stat(built).st_size
    with _write_journal(journal) as lock:
        _journal_dest(lock, dest)
        if replaces is not None:
            # A compare-and-swap at the rename: a save that lands on dest after the hash check above
            # is kept, and this is StaleRecord output_changed (decided 2026-11-06).
            fsutil.atomic_write(built, dest, expect_sha256=replaces.sha256)
        else:
            fsutil.atomic_write(built, dest, overwrite=overwrite)
    if integrity.sha256_file(dest) != digest:
        raise PdfIntakeError("write_mismatch", f"{dest} doesn't hold what was built; check the folder and run again.",
                             path=str(dest))
    return FileRef(dest, digest, size)


# --------------------------------------------------------------------------- the smaller copy

def _smaller(d: _Doc, final: Path, ref: FileRef, adds_layer: bool) -> None:
    """Build the smaller candidate in staging; write it beside (``smaller=True``) or offer it (§12.4)."""
    from . import quality
    from .pdf import smaller

    try:
        cands = smaller.find(final)
    except Exception:                                  # noqa: BLE001 - the lossless file is already written
        _step(d, "smaller", "failed", code="smaller_failed")
        return
    if not cands:
        _step(d, "smaller", "skipped", candidates=0)
        return
    stage = fsutil.work_dir("staging", d.rec.id)
    cand = stage / "candidate.pdf"
    keep = False
    try:
        rep = smaller.build(final, cand, cands, bar=SMALLER_BAR)
        worth = smaller.worth_offering(rep, min_pct=SMALLER_MIN_PCT)
        saved = rep.bytes_before - rep.bytes_after
        pct = round(saved * 100.0 / rep.bytes_before, 1) if rep.bytes_before else 0.0
        d.rec.smaller = {
            "offered": False,
            "candidate": {"path": str(cand), "sha256": integrity.sha256_file(cand), "bytes": rep.bytes_after},
            "saving": {"bytes": saved, "pct": pct},
            "metric": quality.METRIC_ID, "bar": SMALLER_BAR, "min_score": rep.min_score,
            "pages": to_jsonable(rep.pages), "kept_lossless": to_jsonable(rep.kept_lossless),
            "text_layer_identical": bool(rep.text_layer_identical), "metadata": to_jsonable(rep.metadata),
            "applied": False, "written": None,
        }
        if not worth:
            _step(d, "smaller", "skipped", candidates=len(cands), saving_pct=pct)
            return
        if d.opts.smaller:
            beside = fsutil.beside_path(ref.path)
            written = commit(cand, beside, d.seal, d.reports or None, d.opts.provenance, adds_layer=adds_layer,
                             inputs=_all_inputs(d), overwrite=False, journal=d.lock)
            d.rec.smaller.update({"applied": True, "written": str(written.path)})
            _step(d, "smaller", "changed", saving_pct=pct, written=True)
            return
        d.rec.smaller["offered"] = True
        d.rec.offers.append(record.smaller_offer(output=ref.path, inputs=d.inputs, before_bytes=rep.bytes_before,
                                                 after_bytes=rep.bytes_after, saving_pct=pct,
                                                 min_score=float(rep.min_score or 0.0), bar=SMALLER_BAR))
        keep = True
        _step(d, "smaller", "ok", saving_pct=pct, offered=True)
    except (InputChanged, Refused):
        raise
    except UsageError as exc:
        # The beside name was taken after beside_path chose it (output_exists: that file is kept),
        # or can't be written. The lossless file is already written; the smaller copy isn't, and its
        # staged candidate goes below, so the record keeps no candidate (decided 2026-11-06).
        d.rec.smaller = None
        _step(d, "smaller", "failed", code=exc.code)
    except Exception:                                  # noqa: BLE001 - the lossless file is already written
        d.rec.smaller = None
        _step(d, "smaller", "failed", code="smaller_failed")
    finally:
        if not keep:
            _remove_work(stage)


# --------------------------------------------------------------------------- apply

def apply(record_id: str, *, choice: Literal["smaller", "thorough"] | None = None, lang: str | None = None,
          passwords_file: Path | None = None) -> list[Record]:
    """Act on an offer of a finished record, when a person asks for it (DESIGN §12.5).

    Exactly one answer. ``choice="smaller"`` writes the staged smaller copy beside the output
    (rebuilt deterministically when staging was pruned) and returns the updated record.
    ``lang=CODE`` reads the record's inputs again with one of the languages its offer lists and
    replaces the output the earlier run wrote; it returns the new record, whose ``parent`` is
    ``record_id``. ``choice="thorough"`` does the same in thorough mode (DESIGN §14.2), once it is
    set up (exit 3 naming ``pdfintake thorough enable`` otherwise). Exit 5 when the input, the
    output or the candidate changed since the record; exit 2 for an unknown record or an answer no
    offer gives.
    """
    if (choice is None) == (lang is None):
        raise UsageError("one_answer",
                         "apply takes exactly one answer: --choice smaller, --choice thorough or --lang CODE.")
    from .ocr import engines

    engines.warm_up()
    with integrity.no_subprocess(), _stop_signals_unwind():
        rec = record.load(record_id)
        if lang is not None:
            return _apply_lang(rec, lang, passwords_file)
        if choice == "thorough":
            return _apply_thorough(rec, passwords_file)
        job = _smaller_job(rec, str(choice))           # every check; nothing written yet
        _smaller_refusals(rec, job)                    # the refusal re-check (DESIGN §0.7)
        job.seal.check()
        # A work folder whose lock file notes the destination, so a SIGKILL during the write leaves
        # nothing the next run can't sweep (decided 2026-11-06).
        with _write_journal(None) as lock:
            _journal_dest(lock, job.dest)
            # The one write of apply --choice. Never a replace: a file that took the beside name
            # since the guard is kept, and this is UsageError output_exists.
            fsutil.atomic_write(job.candidate, job.dest, overwrite=False)
        return [_smaller_done(rec, job)]


def _seal_of(rec: Record) -> InputSeal:
    seal = rec.input.get("seal") if isinstance(rec.input, Mapping) else None
    if not isinstance(seal, Mapping):
        raise StaleRecord("seal_unreadable", "The record has no input seal; run again.")
    return InputSeal.from_json({"paths": rec.input.get("paths", []), **seal})


def _output_of(rec: Record) -> FileRef:
    out = rec.output
    if not isinstance(out, Mapping):
        raise UsageError("no_output", f"Record {rec.id} has no output to act on.", record_id=rec.id)
    path = Path(str(out.get("path")))
    try:
        current = integrity.sha256_file(path)
    except IntegrityError:
        raise StaleRecord("output_changed", f"The output {path} is gone since the record was written; run again.",
                          path=str(path)) from None
    if current != out.get("sha256"):
        raise StaleRecord("output_changed", f"The output {path} changed since the record was written; run again.",
                          path=str(path))
    return FileRef(path, current, int(out.get("bytes", 0)))


def _history(rec: Record, flag: str, answer: str, result: str, child: str | None) -> None:
    rec.history.append({"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "flag": flag,
                        "answer": answer, "result": result, "child": child})


@dataclass
class _SmallerJob:
    seal: InputSeal
    output: FileRef
    candidate: Path
    sha256: str
    dest: Path


def _smaller_job(rec: Record, choice: str) -> _SmallerJob:
    """Every check before ``apply --choice smaller`` writes: the offer, the seal, the output, the candidate.

    A pruned candidate is rebuilt from the lossless output (the same input gives the same bytes)
    and must hash to what the record measured; any mismatch is exit 5 and nothing is written.
    """
    from .pdf import smaller, verify

    if choice != "smaller":
        raise UsageError("bad_choice", "--choice takes smaller or thorough.", value=str(choice)[:16])
    if not any(o.kind == "smaller" for o in rec.offers):
        if any(h.get("flag") == "--choice" and h.get("answer") == "smaller" for h in rec.history):
            raise UsageError("already_applied", f"The smaller copy of record {rec.id} was already saved.",
                             record_id=rec.id)
        raise UsageError("no_offer", f"Record {rec.id} doesn't offer a smaller copy.", record_id=rec.id)
    sm = rec.smaller if isinstance(rec.smaller, dict) else {}
    cand_info = sm.get("candidate")
    if not isinstance(cand_info, Mapping) or not isinstance(cand_info.get("sha256"), str):
        raise StaleRecord("candidate_unknown", f"Record {rec.id} doesn't say which smaller copy it measured; run again.")
    seal = _seal_of(rec)
    seal.check()                                       # InputChanged, exit 5
    out = _output_of(rec)                              # StaleRecord, exit 5
    stage = fsutil.work_dir("staging", rec.id)
    cand = stage / "candidate.pdf"
    if not cand.is_file():
        smaller.build(out.path, cand, smaller.find(out.path), bar=float(sm.get("bar", SMALLER_BAR)))
    if integrity.sha256_file(cand) != cand_info["sha256"]:
        _remove_work(stage)
        raise StaleRecord("candidate_changed", "The smaller copy doesn't match what the record measured; run again.")
    check = verify.text_layer_identical(out.path, cand)
    if not check.ok:
        raise VerifyFailed("verify_failed", "The smaller copy's text layer differs, so nothing was written.",
                           checks=[check])
    dest = fsutil.beside_path(out.path)
    fsutil.guard_output([*seal.paths, out.path], dest, overwrite=False)
    return _SmallerJob(seal, out, cand, str(cand_info["sha256"]), dest)


def _smaller_refusals(rec: Record, job: _SmallerJob) -> None:
    """The refusal re-check before ``apply --choice smaller`` writes (DESIGN §0.7, decided 2026-11-05).

    The original is read again, from a sealed work copy, as ``run`` reads it: a hard code refuses
    (exit 4), and a copy reason means a smaller copy was never allowed (a copy gets none, §10.2),
    so nothing is written (exit 4). Images have no refusal report; only the provenance counts there.
    """
    reports: list[RefusalReport] = []
    if rec.input.get("kind") == "pdf":
        work = fsutil.work_dir("work", f"{job.seal.sha256[0][:32]}-{__version__}-{secrets.token_hex(4)}")
        lock = _lock_work(work / LOCK_NAME, create=True)
        try:
            copy = work / "input.pdf"
            _copy_into_work(job.seal.paths[0], copy)
            if integrity.sha256_file(copy) != job.seal.sha256[0]:
                raise InputChanged("input_changed", f"The input {job.seal.paths[0]} changed since the record; "
                                   "run again.", path=str(job.seal.paths[0]))
            reports.append(refusals.read(copy))
        finally:
            _remove_work(work)
            if lock is not None:
                os.close(lock)
    adds_layer = bool(rec.pages)
    for report in reports:
        hard = refusals.refusal(report)
        if hard is not None:
            raise Refused(hard[0], hard[1])
    provenance = rec.provenance if isinstance(rec.provenance, str) else "ordinary"
    if _copy_reasons(reports, provenance, adds_layer):
        raise Refused("no_smaller_for_a_copy",
                      "This file is kept as a searchable copy, which never gets a smaller version; nothing was written.")


def _smaller_done(rec: Record, job: _SmallerJob) -> Record:
    if integrity.sha256_file(job.dest) != job.sha256:
        raise PdfIntakeError("write_mismatch", f"{job.dest} doesn't hold the smaller copy; check the folder.",
                             path=str(job.dest))
    _remove_work(job.candidate.parent)
    sm = rec.smaller if isinstance(rec.smaller, dict) else {}
    sm.update({"applied": True, "written": str(job.dest)})
    rec.smaller = sm
    rec.offers = [o for o in rec.offers if o.kind != "smaller"]
    rec.status, rec.exit = "resolved", 0
    _history(rec, "--choice", "smaller", "done", None)
    record.save(rec)
    return rec


def _options_from_json(data: Mapping[str, Any]) -> RunOptions:
    names = {f.name for f in dataclasses.fields(RunOptions)}
    values: dict[str, Any] = {}
    for key, value in data.items():
        if key not in names or key == "passwords_file":
            continue
        if key in ("out", "outdir"):
            value = Path(value) if isinstance(value, str) else None
        elif key in ("lang", "force_lang"):
            value = tuple(str(v) for v in (value or ()))
        values[key] = value
    try:
        return RunOptions(**values)
    except TypeError:
        raise StaleRecord("options_unreadable", "The record's options can't be read; run again.") from None


def _apply_thorough(rec: Record, passwords_file: Path | None) -> list[Record]:
    """``apply --choice thorough``: read the record's inputs again in thorough mode and replace the
    output the earlier run wrote, as ``--lang`` does (DESIGN §14.2)."""
    if not any(o.kind == "thorough" for o in rec.offers):
        if any(h.get("flag") == "--choice" and h.get("answer") == "thorough" for h in rec.history):
            raise UsageError("already_applied", f"Record {rec.id} was already read again in thorough mode.",
                             record_id=rec.id)
        raise UsageError("no_offer", f"Record {rec.id} doesn't offer a thorough read.", record_id=rec.id)
    from . import thorough

    on, why = thorough.resolve(True)
    if not on:
        raise SetupError("thorough_not_ready",
                         f"Thorough mode isn't set up here ({why}). Set it up with 'pdfintake thorough enable', "
                         "then apply again.", record_id=rec.id)
    seal = _seal_of(rec)
    seal.check()                                       # InputChanged, exit 5
    out = _output_of(rec)                              # StaleRecord, exit 5
    opts = _options_from_json(rec.options)
    opts = dataclasses.replace(opts, thorough=True, overwrite=False, plan_only=False, smaller=False, out=None,
                               outdir=None, passwords_file=passwords_file)
    _check_options(opts)
    _sweep_orphans(seal.paths)
    batch = _Batch(opts, seal.paths)
    kind = "pdf" if rec.input.get("kind") == "pdf" else "images"
    doc = _DocInput([Path(p) for p in seal.paths], kind)
    child = _run_document(doc, opts, batch, parent=rec.id, replaces=out)
    _history(rec, "--choice", "thorough", child.status, child.id)
    if child.exit == 0 and child.output is not None:
        rec.status, rec.exit = "resolved", 0
        rec.offers = []
        _remove_work(platforms.cache_root() / "staging" / rec.id)
    record.save(rec, forbid=batch.forbid)
    return [child]


def _apply_lang(rec: Record, lang: str, passwords_file: Path | None) -> list[Record]:
    code = str(lang).strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}", code):
        raise UsageError("bad_language_code", "--lang takes a language code such as en or fr.")
    offers = [o for o in rec.offers if o.kind == "language"]
    if not offers:
        raise UsageError("no_offer", f"Record {rec.id} doesn't offer to read pages with another language.",
                         record_id=rec.id)
    # Only a language the offer lists (DESIGN §12.5). "none" was never one: a re-read with no layer
    # would write nothing and leave the earlier output as it was, while reporting success
    # (decided 2026-11-05).
    options = {opt.id for o in offers for opt in o.options}
    if code not in options:
        raise UsageError("not_an_option", f"{code} isn't one of the languages record {rec.id} offers.",
                         record_id=rec.id, offered=sorted(options))
    seal = _seal_of(rec)
    seal.check()                                       # InputChanged, exit 5
    out = _output_of(rec)                              # StaleRecord, exit 5
    opts = _options_from_json(rec.options)
    opts = dataclasses.replace(opts, lang=(code,), overwrite=False, plan_only=False, smaller=False, out=None,
                               outdir=None, passwords_file=passwords_file)
    _check_options(opts)
    _sweep_orphans(seal.paths)
    batch = _Batch(opts, seal.paths)
    index = batch.index()
    entries = index.for_language(code) or ([index.get(code)] if index.get(code) else [])
    approved: set[str] = {entries[0].id} if entries else set()
    offered = set()
    for o in offers:
        for opt in o.options:
            entries = index.for_language(opt.id) or ([index.get(opt.id)] if index.get(opt.id) else [])
            offered.update(e.id for e in entries[:1])
    declined = {p for p in offered - approved if index.get(p) is not None}
    kind = "pdf" if rec.input.get("kind") == "pdf" else "images"
    doc = _DocInput([Path(p) for p in seal.paths], kind)
    with_answer = _LangAnswer(frozenset(approved), frozenset(declined))
    child = _run_document(doc, opts, batch, parent=rec.id, answer=with_answer, replaces=out)
    _history(rec, "--lang", code, child.status, child.id)
    if child.exit == 0 and child.output is not None:
        rec.status, rec.exit = "resolved", 0
        rec.offers = []
        _remove_work(platforms.cache_root() / "staging" / rec.id)
    record.save(rec, forbid=batch.forbid)
    return [child]
