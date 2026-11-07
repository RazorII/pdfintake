"""The OCR pipeline: render each OCR page once, gather the detection signals, then read every page
with the reader set of the pack ``langplan`` gave it (DESIGN §1, §2.2, §4.3).

**Two passes.** The Hungarian document rule needs every page's first read before any page's
language is settled, so OCR runs in two passes with language planning between them:

1. ``gather`` (pass 1). Each OCR page is rendered once, at its OCR dpi, to a PNG in the run's work
   folder. Then, per page: OSD's script (the result orientation already took, or OSD on the PNG
   scaled to about 300 dpi), the base pack's first ``eng`` read (its first ``reader`` use, which
   doubles as the English reader), the base pack's checker read (RapidOCR's Latin read, which the
   function-word rule reads), and, on pages that aren't a non-Latin script, each installed pack's
   probe and, when no probe fired, the mixed-script trigger and the 120-character language windows.
   ``detect.script_call(osd, read=eng_read)`` names the page's script.
2. ``flow`` plans the language (``langplan.plan`` and ``langplan.resolve_unsure``; nothing is
   asked, DESIGN §12). Nothing has been read with any pack's reader set yet.
3. ``read_pages`` (pass 2) reads each page with its pack's reader set for this platform: every
   engine use reads the same PNG, the pack's normalisers run on every read (the checker's too),
   the pick rule chooses, the pack's repairs and fills run, and the must-survive characters are
   counted per engine and for the layer. A pass-1 read whose engine, parameters and model files
   equal a pass-2 use's, on the same render, is reused instead of read again (the base pack's ``eng`` read is the
   English reader; the Latin checker read is the Hungarian pack's checker). ``layer_pages`` then
   builds the layer writer's input from the chosen lines and the fills, in that order.

``read_pages`` needs no index, store or signals: the kit calls it with a draft manifest wrapped in
an ``InstalledPack`` (``signals=None``), so a bench measures the code ``run`` uses (DESIGN §0.5).
To bench another candidate reader set, the kit passes ``dataclasses.replace(manifest,
readers={platform_key: reader_set})``.

**Threading** (DESIGN §2.2):

* Every render happens in the calling thread, under ``PDFIUM_LOCK``, written to a PNG. At most
  ``2 × workers`` pages are rendered and not yet gathered at any time.
* Workers get only ``PageImage`` objects. An engine with ``session_per_thread`` gets one session
  per worker thread and per configuration (its language string, parameters and model files),
  reused page after page; any other engine gets one session per configuration, shared by every
  thread (it must be thread-safe). OSD gets one session per thread too. A configuration whose
  open fails is tried once per pass, not once per thread. Every session is closed when the pass
  ends, whether it succeeded or not.
* Pass 2 reads the same PNGs back from disk; no page is held in memory between the passes.

**Models are verified at the moment they're opened** (DESIGN §11.2, decided 2026-11-05):
``store.model_paths`` hashes every model file of a use immediately before ``Engine.open``, and the
OSD model's ``asset_path`` immediately before ``open_osd``, so an engine never loads a file that
wasn't just checked. An ``IntegrityError`` there propagates: a tampered model stops the file.

**Failures.** ``EngineFailure`` never leaves this module. A use that can't open, a read that raises
it, and a use whose engine isn't usable here become a read with status ``failed`` (or
``unavailable`` for an engine that isn't on this platform or doesn't support the language, which is
never counted as a failure). An ``EngineFailure`` from the mixed-script check is an engine failure
on that page: the page counts as unchecked (decided 2026-11-05), so ``langplan`` adds the note
``mixed_script_unchecked`` instead of losing the text silently.

**Warnings on the page** (codes from ``record.WARNING_CODES``; ``record`` holds their wording):
the plan's own warnings, ``render_capped`` from the render, then ``reader_failed`` and
``reader_timeout`` for a degraded use when the page still has a layer, ``vision_language_unsupported``,
the pick rule's ``Pick.warnings`` (``agreement_low``), ``fallback_used`` when the rule fell back,
and ``no_layer_readers_failed`` when no reader worked and the page has no layer. ``notes_for``
gives the page's note codes (``vision_not_on_platform``), which ``PageOCR`` has no field for.

Nothing here writes into a user's folder: only PNGs into the run's work folder in the cache.

**The detection cache** (DESIGN §9.1). Given a ``DetectionCache``, ``gather`` keeps pass 1's result
under a key and, when a later call has the same key (a re-run, ``apply --lang``), hands the stored
result back instead of rendering and detecting again. The key (``_detection_key``) covers everything
pass 1's result depends on: the work PDF's SHA-256, every page class, the OSD results orientation
gave, the platform key and OS release, the Python and Unicode versions, the core version, the base
pack and every pack whose probe runs (id, version, manifest hash, and the probe's configuration), each
engine use pass 1 makes (its configuration with model hashes, its availability here and the engine's
version), the OSD model and Tesseract's version, the calibration, and the versions of the libraries
that render and identify. Installed packs without a probe take no part in pass 1, so installing one
(as a run that fetched a Latin pack does) keeps the key; a pack with a probe changes it (decided
2026-11-06). A different key, or a stored entry that is damaged in any way (its file, its checksum,
its structure, a render's size or hash), is a fresh detection, never a reuse. What an entry holds is
exactly pass 1's result: the renders, OSD's numbers, the script calls, the base pack's two reads
(which detection reads and pass 2 reuses), the probes' counts, the language windows (offsets, codes
and probabilities) and the mixed-script counts. Nothing pass 2 reads, picks, repairs or fills is
ever stored. ``cache_entry`` writes an entry only when it decodes back to exactly what pass 1 found.
**A pass that met a failure is never stored** (decided 2026-11-06): a read that raised, failed or ran
out of time, an engine, probe or OSD session that couldn't open, an OSD call that raised and a
mixed-script check that fell back are conditions of the moment (a timeout under load), not of
anything in the key, so storing them would hand a stale failure to every later run. The result is
used for this run as it is, and the next run detects again. Only a use whose availability says it
can't run here (part of the key) and a base pack without an OSD model are failures the key covers.
``cache_entry`` refuses a read that failed or timed out for any other reason as well.
``flow`` keeps the entries on disk, in ``work/<input sha>-<version>/``; this module never writes them.

**The orientation cache** (DESIGN §9.1, decided 2026-11-06). ``flow``'s orientation step renders
every OCR page twice and reads it with the voter and OSD; that costs more than pass 1 itself, so
its result is kept the same way. ``orientation_key`` covers the work PDF, the pages, the voter's and
OSD's models (by their pins) and engine versions, the platform and OS release, the libraries and
``flow``'s render settings; ``orientation_entry`` and ``read_orientation`` are its checked format.
What is kept is each page's vote (turn, margin, boxes, per-turn scores) and OSD result; the decision
is made again from them every run. ``watch_session`` lets ``flow`` see a voter or OSD failure that
``orient.vote`` swallows, so such a result is never stored either.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .. import __version__, integrity, platforms
from ..errors import EngineFailure, IntegrityError, PackError, SetupError
from ..langpacks import store
from ..params import ValidationContext
from ..pdf.render import PDFIUM_LOCK, Renderer
from ..record import NOTE_CODES, WARNING_CODES
from ..types import (
    READ_STATUSES,
    ROLES,
    EngineRead,
    Fill,
    LayerLine,
    LayerPage,
    Line,
    OsdResult,
    PageImage,
    PageOCR,
    Pick,
    ScriptCall,
    Word,
    to_jsonable,
)
from . import detect, engines, fills, langplan, rules, scripts, textfix
from .detect import LangWindow, MixedScript, PageSignals, ProbeResult
from .engines import Availability

if TYPE_CHECKING:
    from ..langpacks.manifest import EngineUse, MustSurvive, ReaderSet, StepUse
    from ..langpacks.store import InstalledPack
    from ..pdf.classify import PageClass
    from ..protocols import OsdSession, Repair
    from .langplan import PagePlan

DEFAULT_READ_TIMEOUT_S = 300.0      # a use without its own timeout_s (Tesseract's default)
RENDER_AHEAD = 2                    # renders in flight per worker (DESIGN §2.2)
LAYER_KEY = "layer"                 # the must-survive count of the final layer
NOT_READ_STATES = frozenset({"not_on_platform", "unsupported_language"})   # "unavailable", never a failure
WORKING = frozenset({"ok", "empty"})
NO_LAYER_RULE = "none"              # the Pick of a page planned with no pack (--lang none)
NO_LAYER_COMPOSITION = "none"       # the only composition a plan with no pack may have here
_OSD_KEY = "osd"                    # the OSD session's key (a use key is a JSON list, so never this)
DETECT_READERS = "default"          # the base pack's reader set pass 1 reads with, on every platform (``detect_readers``)

CACHE_VERSION = 2                   # the detection cache's format; part of every key (DESIGN §9.1)
CACHE_FILE = "detection.json"       # a cache entry's record of its renders and pass 1's signals
MAX_CACHE_BYTES = 256 * 1024 * 1024  # a larger cache file is never read: a fresh detection instead
# Libraries whose version can change a render, a read or a language window, beyond the engines' own.
CACHE_LIBRARIES = ("pypdfium2", "pillow", "numpy", "py3langid", "onnxruntime", "opencv-python",
                   "opencv-python-headless")
_RENDER_NAME = "ocr-{:05d}.png"     # a pass-1 render's file name in the work folder (and a cache entry)
_SHA_RE = re.compile(r"[0-9a-f]{64}")
# A read status that may go otherwise next time (an engine error, a timeout under load): never stored.
TRANSIENT_STATUSES = frozenset({"failed", "timeout"})
ORIENT_CACHE_FILE = "orientation.json"   # an input's stored orientation votes and OSD results
ORIENT_MAX_BYTES = 4 * 1024 * 1024       # a larger orientation file is never read

_MISSING = object()


# --------------------------------------------------------------------------- failures of the moment

class FailureWatch:
    """Whether anything in a pass failed in a way the next pass may not (DESIGN §9.1): a session that
    couldn't open, a call that raised, a read that failed or timed out. Thread-safe; it only ever
    turns on. A pass whose watch is on is never stored in a cache."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def note(self) -> None:
        self._event.set()

    @property
    def failed(self) -> bool:
        return self._event.is_set()


class _Watched:
    """A session that tells ``watch`` about every read or detect that raised, and every read whose
    status is ``failed`` or ``timeout``; it passes everything else through unchanged."""

    def __init__(self, session: Any, watch: FailureWatch) -> None:
        self._session = session
        self._watch = watch

    def read(self, page: Any, *, timeout_s: float) -> Any:
        try:
            read = self._session.read(page, timeout_s=timeout_s)
        except BaseException:
            self._watch.note()
            raise
        if getattr(read, "status", None) in TRANSIENT_STATUSES:
            self._watch.note()
        return read

    def detect(self, image: Any, dpi: float) -> Any:
        try:
            return self._session.detect(image, dpi)
        except BaseException:
            self._watch.note()
            raise

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if close is not None:
            close()


def watch_session(session: Any, watch: FailureWatch) -> Any:
    """``session`` (an engine or OSD session, or None) reporting its failures to ``watch``."""
    return None if session is None else _Watched(session, watch)


# --------------------------------------------------------------------------- public types

@dataclass
class DocumentSignals:
    pages: list[PageImage]                       # the OCR pages' renders, in page order
    signals: list[PageSignals]                   # one per page of ``pages``, same order
    classes: list[PageClass]                     # every page class gather was given
    osd: dict[int, OsdResult | None]             # page index -> OSD's result (given or measured here)
    # Beyond DESIGN §2.2 (additive): per page index, the render's warning codes (``render_capped``)
    # and note codes (``render_scaled_down``), for the page and the record.
    warnings: dict[int, list[str]] = field(default_factory=dict)
    notes: dict[int, list[str]] = field(default_factory=dict)
    # Beyond DESIGN §2.2 (additive, DESIGN §9.1): True when ``gather`` took this from its detection
    # cache instead of rendering and detecting (flow records it on the ``detect`` step).
    cached: bool = False


class DetectionCache(Protocol):
    """Where ``gather`` keeps pass 1's result between calls (DESIGN §9.1); ``flow`` keeps it on disk.

    ``load`` gives the result stored under ``key``, its renders placed in ``work`` (the folder pass 2
    reads them from), or None; ``keep`` stores ``signals`` under ``key``. Neither raises for a cache
    problem: a cache that can't be read or written only costs a fresh detection. ``read_cache`` and
    ``cache_entry`` are the checks and the format every implementation uses.
    """

    def load(self, key: str, work: Path) -> DocumentSignals | None: ...

    def keep(self, key: str, signals: DocumentSignals) -> None: ...


@dataclass
class ResolvedReaderSet:
    pack: str
    reader_set: ReaderSet
    availability: dict[str, Availability]        # use id -> what the engine said about this use here


# --------------------------------------------------------------------------- engine uses

@dataclass(frozen=True)
class _Use:
    pack: InstalledPack            # whose models the use names (``en:`` refs resolve in the base pack)
    use: EngineUse
    engine: Any
    key: str                       # engine, canonical params and model hashes: one session per key
    avail: Availability


def _asset_sha(pack: InstalledPack, base: InstalledPack, ref: str) -> str:
    owner, key = (base, ref[len(store.BASE_PREFIX):]) if ref.startswith(store.BASE_PREFIX) else (pack, ref)
    for a in owner.manifest.models:
        if a.key == key:
            return a.sha256
    raise PackError("unknown_asset", f"the {owner.manifest.id!r} pack has no model {key!r}",
                    pack=owner.manifest.id, key=key)


def _use_key(pack: InstalledPack, base: InstalledPack, use: EngineUse, engine: Any) -> str:
    """The use's configuration: its engine, its parameters (a model reference as the model's SHA-256,
    so ``rapid-rec-latin`` in the base pack equals ``en:rapid-rec-latin`` elsewhere) and its model
    files' SHA-256s. Two uses with one key read a page the same way."""
    kinds: dict[str, str] = {}
    table = getattr(engine, "PARAMS", None)
    if table is not None:
        for p in table:
            kinds[p.name] = p.kind
    canon: dict[str, Any] = {}
    for name in sorted(use.params):
        value = use.params[name]
        if kinds.get(name) == "asset_ref" and isinstance(value, str):
            canon[name] = "sha256:" + _asset_sha(pack, base, value)
        else:
            canon[name] = to_jsonable(value)
    models = sorted(_asset_sha(pack, base, ref) for ref in use.models)
    return json.dumps([use.engine, canon, models], sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _engine(name: str) -> tuple[Any, Availability | None]:
    """The registered engine, or None and a ``broken`` availability when its module can't load.

    A name no core registry knows is a pack problem and raises PackError.
    """
    try:
        return engines.get(name), None
    except PackError:
        raise
    except SetupError as exc:
        return None, Availability(False, "broken", exc.message)


def _availability(pack: InstalledPack, base: InstalledPack, use: EngineUse, engine: Any,
                  platform_key: str) -> Availability:
    models = store.model_paths(pack, base, use)        # every model hash-checked (IntegrityError propagates)
    try:
        return engine.availability(use.params, models, platform_key)
    except PackError as exc:
        return Availability(False, "broken", exc.message)
    except EngineFailure as exc:
        return Availability(False, "broken", exc.message)
    except Exception as exc:  # an engine bug degrades this reader; it never stops the run
        return Availability(False, "broken", f"the availability check failed: {type(exc).__name__}: {exc}")


def _make_use(pack: InstalledPack, base: InstalledPack, use: EngineUse, platform_key: str) -> _Use:
    engine, broken = _engine(use.engine)
    if engine is None:
        assert broken is not None
        key = json.dumps([use.engine, use.id, "unloadable"], ensure_ascii=True)
        return _Use(pack, use, None, key, broken)
    key = _use_key(pack, base, use, engine)
    return _Use(pack, use, engine, key, _availability(pack, base, use, engine, platform_key))


def resolve_readers(pack: InstalledPack, base: InstalledPack, platform_key: str) -> ResolvedReaderSet:
    """The pack's reader set for ``platform_key`` and what each engine says about its use here.

    Every use's model files are hash-checked (``store.model_paths``; an IntegrityError naming the
    file propagates), then ``Engine.availability`` answers. An engine whose module can't load, or
    whose check raises, is ``broken`` for this run: its use reads as ``failed`` and the page keeps
    its other readers. An engine name no core registry knows raises PackError.
    """
    rs = pack.manifest.readers_for(platform_key)
    avail: dict[str, Availability] = {}
    for use in rs.engines:
        avail[use.id] = _make_use(pack, base, use, platform_key).avail
    return ResolvedReaderSet(pack.manifest.id, rs, avail)


# --------------------------------------------------------------------------- sessions

class _Sessions:
    """Engine and OSD sessions for one pass: per thread for ``session_per_thread`` engines (and OSD),
    shared otherwise; each opened lazily, right after its models are checked, and all closed at the end.

    A configuration whose open fails is tried once per pass, whatever the number of threads: the
    first open of each configuration (and of OSD) is serialised, so a second thread waits for it
    and then sees its failure instead of trying again. Once a first open has succeeded, the other
    threads open their own sessions in parallel."""

    def __init__(self, base: InstalledPack) -> None:
        self._base = base
        self._local = threading.local()
        self._shared: dict[str, Any] = {}
        self._failed: dict[str, EngineFailure] = {}
        self._lock = threading.Lock()
        self._open_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._tried: set[str] = set()
        self._all: list[Any] = []

    def _first_open(self, key: str, opener: Callable[[], Any]) -> Any:
        """``opener()`` for one thread's session of ``key``, raising the recorded failure instead
        when an earlier attempt in this pass failed. The first attempt per key runs alone."""
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
            tried = key in self._tried
        if not tried:
            with key_lock:
                failed = self._failure(key)
                if failed is not None:
                    raise failed
                with self._lock:
                    tried = key in self._tried
                if not tried:                            # this thread makes the pass's first attempt
                    try:
                        return opener()
                    finally:
                        with self._lock:
                            self._tried.add(key)
        failed = self._failure(key)                      # a first attempt has been made: reuse its verdict
        if failed is not None:
            raise failed
        return opener()

    def _remember(self, session: Any) -> Any:
        with self._lock:
            self._all.append(session)
        return session

    def _failure(self, key: str) -> EngineFailure | None:
        with self._lock:
            exc = self._failed.get(key)
        if exc is None:
            return None
        return EngineFailure(exc.code, exc.message, **exc.detail)

    def _open(self, u: _Use) -> Any:
        models = store.model_paths(u.pack, self._base, u.use)   # checked now, immediately before the open (§11.2)
        try:
            session = u.engine.open(u.use.id, u.use.role, u.use.params, models)
        except EngineFailure as exc:
            with self._lock:
                self._failed.setdefault(u.key, exc)
            raise
        return self._remember(session)

    def get(self, u: _Use) -> Any:
        failed = self._failure(u.key)
        if failed is not None:
            raise failed
        if getattr(u.engine, "session_per_thread", True):
            cache = self._local.__dict__.setdefault("sessions", {})
            session = cache.get(u.key)
            if session is None:
                session = self._first_open(u.key, lambda: self._open(u))
                cache[u.key] = session
            return session
        with self._open_lock:
            session = self._shared.get(u.key)
            if session is None:
                failed = self._failure(u.key)
                if failed is not None:
                    raise failed
                session = self._open(u)
                self._shared[u.key] = session
            return session

    def osd(self) -> OsdSession:
        """This thread's OSD session. EngineFailure when OSD can't be opened here; IntegrityError when
        the base pack's OSD model doesn't match its pin."""
        local = self._local.__dict__
        if "osd" in local:
            return local["osd"]

        def opener() -> OsdSession:
            try:
                session = _open_osd(self._base)
            except EngineFailure as exc:
                with self._lock:
                    self._failed.setdefault(_OSD_KEY, exc)
                raise
            return self._remember(session)

        local["osd"] = self._first_open(_OSD_KEY, opener)
        return local["osd"]

    def close(self) -> None:
        with self._lock:
            sessions, self._all = self._all, []
        for s in sessions:
            try:
                s.close()
            except Exception:
                pass


def _osd_key(base: InstalledPack) -> str | None:
    for a in base.manifest.models:
        if a.file == "osd.traineddata":
            return a.key
    return None


def _open_osd(base: InstalledPack) -> OsdSession:
    """An OSD session on the base pack's ``osd.traineddata``, checked against its pin right before it
    opens. EngineFailure when the base pack has no OSD model or Tesseract can't open it here."""
    key = _osd_key(base)
    if key is None:
        raise EngineFailure("osd_unavailable", f"the {base.manifest.id!r} pack has no orientation model (osd.traineddata)")
    path = base.asset_path(key)                       # IntegrityError naming the file propagates
    from .engines import tesseract

    try:
        return tesseract.open_osd({key: path})
    except IntegrityError:
        raise
    except SetupError as exc:
        raise EngineFailure("osd_unavailable", f"OSD can't run here: {exc.message}", engine="tesseract") from None


def _timeout(use: EngineUse) -> float:
    value = use.params.get("timeout_s") if isinstance(use.params, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    return DEFAULT_READ_TIMEOUT_S


def _not_read(u: _Use, status: str, error: str, state: str | None) -> EngineRead:
    meta: dict[str, Any] = {}
    if state is not None:
        meta["state"] = state
    return EngineRead(u.use.id, u.use.engine, f"{u.use.engine}:{u.use.id}", u.use.role, status,  # type: ignore[arg-type]
                      [], 0.0, error, meta)


def _read(sessions: _Sessions, u: _Use, page: PageImage, watch: FailureWatch | None = None) -> EngineRead:
    """``u``'s read of ``page``: never raises EngineFailure (it becomes a ``failed`` read). With a
    ``watch``, a failure that isn't the use's availability (which the cache key covers) is noted."""
    if not u.avail.ok:
        status = "unavailable" if u.avail.state in NOT_READ_STATES else "failed"
        return _not_read(u, status, u.avail.detail or u.avail.state, u.avail.state)
    try:
        session = sessions.get(u)
        read = session.read(page, timeout_s=_timeout(u.use))
    except EngineFailure as exc:
        if watch is not None:
            watch.note()
        return _not_read(u, "failed", exc.message, None)
    if watch is not None and read.status in TRANSIENT_STATUSES:
        watch.note()
    return dataclasses.replace(read, use_id=u.use.id, role=u.use.role, meta=dict(read.meta or {}))


def _first(rs: ReaderSet, role: str) -> EngineUse | None:
    for use in rs.engines:
        if use.role == role:
            return use
    return None


def _workers(workers: Any) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise ValueError(f"workers must be a whole number, not {workers!r}")
    return max(1, workers)


# --------------------------------------------------------------------------- pass 1

@dataclass(frozen=True)
class _ProbeUse:
    pack_id: str
    script: str
    u: _Use
    min_chars: int
    min_conf: float
    min_line_chars: int


def _probe_script(pack: InstalledPack) -> str | None:
    """The non-Latin script a pack's probe counts: its first non-Latin OSD script, else its first non-Latin script."""
    m = pack.manifest
    for name in (*m.detect.osd_scripts, *m.scripts):
        if name != "Latin" and name in scripts.SCRIPTS:
            return name
    return None


@dataclass
class _Gather:
    base: InstalledPack
    sessions: _Sessions
    eng: _Use | None
    checker: _Use | None
    probes: list[_ProbeUse]
    calib: Any
    watch: FailureWatch = field(default_factory=FailureWatch)


def _osd_session(ctx: _Gather) -> Any:
    """This thread's OSD session, watched. EngineFailure when it can't open; that is noted on the
    watch unless the base pack has no OSD model at all (a fact the cache key holds)."""
    try:
        return watch_session(ctx.sessions.osd(), ctx.watch)
    except EngineFailure:
        if _osd_key(ctx.base) is not None:
            ctx.watch.note()
        raise


def _page_osd(ctx: _Gather, page: PageImage) -> OsdResult | None:
    """OSD on the page's render scaled to about 300 dpi; None when OSD can't judge or can't run."""
    from ..pdf.render import upscale_for_osd

    try:
        session = _osd_session(ctx)
    except EngineFailure:
        return None
    try:
        with page.open() as img:
            up = upscale_for_osd(img, float(page.dpi), detect.OSD_TARGET_DPI)
            dpi = float(page.dpi) * (up.size[0] / img.size[0]) if img.size[0] else float(page.dpi)
            return session.detect(up, dpi)
    except EngineFailure:
        return None
    except (OSError, ValueError):
        ctx.watch.note()                               # the render couldn't be read this time
        return None


def _mixed(ctx: _Gather, page: PageImage, eng: EngineRead) -> MixedScript:
    """The mixed-script trigger; an EngineFailure (OSD can't run, the render can't be read) is an
    engine failure on this page, so the page counts as unchecked (decided 2026-11-05)."""
    unchecked = detect.mixed_script_check(page, eng, _NoOsd(), None)   # counts only; touches nothing
    if ctx.calib is None:
        return unchecked
    try:
        osd = _osd_session(ctx)
    except EngineFailure:
        return MixedScript(False, None, unchecked.blocks, None)
    try:
        return detect.mixed_script_check(page, eng, osd, ctx.calib)
    except EngineFailure:
        ctx.watch.note()
        return MixedScript(False, None, unchecked.blocks, None)


class _NoOsd:
    def detect(self, image: Any, dpi: float) -> OsdResult | None:   # never called without a calibration
        raise EngineFailure("osd_unavailable", "no OSD session was opened for this check")


def _probe(ctx: _Gather, page: PageImage, p: _ProbeUse) -> ProbeResult | None:
    """The pack's probe on ``page``; None when the probe can't run here (so it didn't run at all)."""
    if not p.u.avail.ok:
        return None
    try:
        session = ctx.sessions.get(p.u)
    except EngineFailure:
        ctx.watch.note()
        return None
    # run_probe turns a failed read into "found nothing"; the watch still sees it.
    return detect.run_probe(page, watch_session(session, ctx.watch), pack=p.pack_id, script=p.script,
                            min_conf=p.min_conf, min_line_chars=p.min_line_chars, min_chars=p.min_chars)


def _fired(probes: Mapping[str, ProbeResult]) -> bool:
    for pr in probes.values():
        need = pr.min_chars if pr.min_chars is not None else langplan.PROBE_MIN_CHARS_DEFAULT
        if pr.chars >= need:
            return True
    return False


def _missing_use_read(role: str) -> EngineRead:
    return EngineRead(f"no_{role}", "none", "none", role, "unavailable", [], 0.0,  # type: ignore[arg-type]
                      f"the base pack's reader set has no {role} use", {"state": "missing_model"})


def _page_signals(ctx: _Gather, page: PageImage, given: Any) -> tuple[PageSignals, OsdResult | None]:
    """Pass 1 for one page (runs in a worker): OSD, the eng read, the checker read, probes, trigger, windows."""
    osd = _page_osd(ctx, page) if given is _MISSING else given
    eng = _read(ctx.sessions, ctx.eng, page, ctx.watch) if ctx.eng is not None else _missing_use_read("reader")
    checker = _read(ctx.sessions, ctx.checker, page, ctx.watch) if ctx.checker is not None else None
    sc = detect.script_call(osd, read=eng)
    probes: dict[str, ProbeResult] = {}
    mixed = MixedScript(False, None, 0, None)
    windows: list[detect.LangWindow] = []
    if sc.name is None or sc.name == "Latin":
        for p in ctx.probes:
            pr = _probe(ctx, page, p)
            if pr is not None:
                probes[p.pack_id] = pr
        if not _fired(probes):
            mixed = _mixed(ctx, page, eng)
        windows = detect.language_windows(eng)
    return PageSignals(page.index, osd, sc, eng, checker, probes, windows, mixed), osd


def detect_readers(base: InstalledPack) -> ReaderSet:
    """The base pack's reader set pass 1 reads with: its ``default`` set on every platform. Detection's
    thresholds (the Latin gate's Tesseract confidence, the mixed-script trigger's low-confidence lines,
    the function-word rule's checker read) are measured on these reads, so a platform entry the kit picks
    for reading English changes how an English page is read in pass 2, never what detection sees."""
    return base.manifest.readers[DETECT_READERS]


def gather(pdf: Path, classes: Sequence[PageClass], base: InstalledPack, installed: Mapping[str, InstalledPack],
           platform_key: str, work: Path, *, workers: int, osd: Mapping[int, OsdResult | None],
           cache: DetectionCache | None = None) -> DocumentSignals:
    """Pass 1: render every page of ``classes`` that needs OCR and gather its detection signals.

    ``pdf`` is the run's work PDF, uprighted; ``work`` the run's work folder (PNGs land there).
    ``osd`` holds the OSD result orientation already took per page (None: too little text); a page
    missing from it gets OSD on its own render. ``installed`` holds the installed packs; every one
    with a probe (except the base pack) probes the pages that aren't a non-Latin script. Renders
    happen here, in the calling thread, at most ``2 × workers`` ahead of the workers.

    With a ``cache`` (DESIGN §9.1), a result stored under this call's key comes back with its
    renders placed in ``work`` and ``cached=True``, and nothing is rendered or read; otherwise pass 1
    runs and its result is stored. The engine uses are still resolved first, so every model is
    hash-checked either way (a tampered model stops the file, cached or not).
    """
    n = _workers(workers)
    ocr = sorted({c.index: c for c in classes if c.needs_ocr}.values(), key=lambda c: c.index)
    if not ocr:
        return DocumentSignals([], [], list(classes), {})
    base_rs = detect_readers(base)
    eng_use, chk_use = _first(base_rs, "reader"), _first(base_rs, "checker")
    probes: list[_ProbeUse] = []
    for pid in sorted(installed):
        pack = installed[pid]
        m = pack.manifest
        if m.base or pid == base.manifest.id or m.detect.probe is None:
            continue
        script = _probe_script(pack)
        if script is None:
            continue
        pr = m.detect.probe
        probes.append(_ProbeUse(m.id, script, _make_use(pack, base, pr.use, platform_key),
                                int(pr.min_chars), float(pr.min_conf), int(pr.min_line_chars)))
    sessions = _Sessions(base)
    ctx = _Gather(base, sessions,
                  _make_use(base, base, eng_use, platform_key) if eng_use is not None else None,
                  _make_use(base, base, chk_use, platform_key) if chk_use is not None else None,
                  probes, langplan.load_calibration())
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    key: str | None = None
    if cache is not None:
        key = _detection_key(Path(pdf), classes, ctx, platform_key, osd)
        hit = _cache_load(cache, key, work, [c.index for c in ocr]) if key is not None else None
        if hit is not None:
            return dataclasses.replace(hit, classes=list(classes), cached=True)
    pages: list[PageImage] = []
    futures: list[Future[tuple[PageSignals, OsdResult | None]]] = []
    warns: dict[int, list[str]] = {}
    notes: dict[int, list[str]] = {}
    executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="pdfintake-ocr")
    try:
        pending: set[Future[Any]] = set()
        with Renderer(Path(pdf)) as renderer:
            for c in ocr:
                while len(pending) >= RENDER_AHEAD * n:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()                    # a worker's error stops the pass here
                with PDFIUM_LOCK:
                    dpi = renderer.ocr_dpi(c.index)
                    page = renderer.render(c.index, dpi, work / _RENDER_NAME.format(c.index))
                    capped = renderer.render_capped(c.index)
                    scaled = renderer.render_scaled_down(c.index)
                if capped:
                    warns.setdefault(c.index, []).append("render_capped")
                if scaled:
                    notes.setdefault(c.index, []).append("render_scaled_down")
                pages.append(page)
                fut = executor.submit(_page_signals, ctx, page, osd[c.index] if c.index in osd else _MISSING)
                futures.append(fut)
                pending.add(fut)
        results = [f.result() for f in futures]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        sessions.close()
    signals = [r[0] for r in results]
    found = {page.index: r[1] for page, r in zip(pages, results)}
    out = DocumentSignals(pages, signals, list(classes), found, warns, notes)
    if cache is not None and key is not None and not ctx.watch.failed:   # a failure of the moment: not kept
        try:
            cache.keep(key, out)
        except Exception:  # noqa: BLE001 - a cache that can't be written only costs the next call a detection
            pass
    return out


# --------------------------------------------------------------------------- the detection cache (DESIGN §9.1)

def _engine_version(engine: Any) -> str | None:
    if engine is None:
        return None
    try:
        v = engine.version()
    except Exception:  # noqa: BLE001 - an engine that can't say is keyed as unknown
        return None
    return None if v is None else str(v)


def _library_versions() -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for name in CACHE_LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 - not installed (PackageNotFoundError) or a broken distribution
            out[name] = None
    return out


def _os_release() -> str:
    """The OS release an engine's output can change with (Vision with macOS)."""
    mac = platforms.macos_version()
    if mac is not None:
        return f"macOS {mac[0]}.{mac[1]}"
    if sys.platform == "win32":
        v = sys.getwindowsversion()  # type: ignore[attr-defined]
        return f"Windows {v.major}.{v.minor}.{v.build}"
    return sys.platform


def _pack_part(pack: InstalledPack) -> dict[str, Any]:
    m = pack.manifest
    return {"id": m.id, "version": m.version, "manifest": m.raw_sha256}


def _use_part(u: _Use | None) -> dict[str, Any] | None:
    """A use as the key sees it: its configuration (engine, parameters, model hashes), the engine's
    version, and whether and how it can run here (an unavailable use reads as unavailable)."""
    if u is None:
        return None
    return {"key": u.key, "engine": _engine_version(u.engine), "ok": u.avail.ok, "state": u.avail.state}


def _osd_part(base: InstalledPack) -> dict[str, Any]:
    key = _osd_key(base)
    sha = next((a.sha256 for a in base.manifest.models if a.key == key), None) if key is not None else None
    try:
        from .engines import tesseract

        version = _engine_version(tesseract.ENGINE)
    except Exception:  # noqa: BLE001 - Tesseract that can't load is keyed as unknown
        version = None
    return {"model": sha, "tesseract": version}


def _detection_key(pdf: Path, classes: Sequence[PageClass], ctx: _Gather, platform_key: str,
                   osd: Mapping[int, OsdResult | None]) -> str | None:
    """The SHA-256 of everything pass 1's result depends on (see the module docstring); None when a
    part can't be taken (an unreadable work PDF, a value JSON can't hold), which means no cache."""
    try:
        payload = {
            "cache": CACHE_VERSION,
            "core": __version__,
            "pdf": integrity.sha256_file(pdf),
            "classes": to_jsonable(list(classes)),
            "osd": [[int(i), to_jsonable(osd[i])] for i in sorted(osd)],
            "platform": platform_key,
            "os": _os_release(),
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "unicode": unicodedata.unidata_version,
            "base": _pack_part(ctx.base),
            "reader": _use_part(ctx.eng),
            "checker": _use_part(ctx.checker),
            "probes": [{"pack": _pack_part(p.u.pack), "script": p.script, "min_chars": p.min_chars,
                        "min_conf": p.min_conf, "min_line_chars": p.min_line_chars,
                        "min_line_share": detect.PROBE_LINE_MIN_SHARE, "use": _use_part(p.u)}
                       for p in ctx.probes],
            "osd_engine": _osd_part(ctx.base),
            "calibration": to_jsonable(ctx.calib),
            "libraries": _library_versions(),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (IntegrityError, OSError, TypeError, ValueError):
        return None
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _cache_load(cache: DetectionCache, key: str, work: Path, indices: Sequence[int]) -> DocumentSignals | None:
    """The cache's result for ``key`` when it covers exactly these pages with their renders in ``work``."""
    try:
        hit = cache.load(key, work)
    except Exception:  # noqa: BLE001 - a cache that can't be read is a fresh detection
        return None
    if not isinstance(hit, DocumentSignals):
        return None
    if [p.index for p in hit.pages] != list(indices) or [s.index for s in hit.signals] != list(indices):
        return None
    if any(Path(p.path).parent != Path(work) for p in hit.pages):
        return None
    return hit


class _Bad(Exception):
    """A cache file that isn't what ``cache_entry`` writes."""


def _need(ok: bool) -> None:
    if not ok:
        raise _Bad()


def _num(v: Any) -> Any:
    _need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))
    return v


def _int(v: Any) -> int:
    _need(isinstance(v, int) and not isinstance(v, bool))
    return v


def _str(v: Any) -> str:
    _need(isinstance(v, str))
    return v


def _opt(fn: Callable[[Any], Any], v: Any) -> Any:
    return None if v is None else fn(v)


def _obj(v: Any, *keys: str) -> dict[str, Any]:
    _need(isinstance(v, dict) and set(v) == set(keys))
    return v


def _arr(v: Any, n: int | None = None) -> list[Any]:
    _need(isinstance(v, list) and (n is None or len(v) == n))
    return v


def _literals(cls: type, name: str) -> frozenset[str]:
    import typing

    return frozenset(typing.get_args(typing.get_type_hints(cls)[name]))


_SCRIPT_BY = _literals(ScriptCall, "by")


def _e_quad(q: Any) -> list[list[Any]]:
    return [[p[0], p[1]] for p in q]


def _d_quad(v: Any) -> Any:
    return tuple((_num(p[0]), _num(p[1])) for p in (_arr(x, 2) for x in _arr(v, 4)))


def _e_line(ln: Line) -> dict[str, Any]:
    return {"text": ln.text, "conf": ln.conf, "box": _e_quad(ln.box),
            "words": [{"text": w.text, "conf": w.conf, "box": _e_quad(w.box)} for w in ln.words]}


def _d_line(v: Any) -> Line:
    v = _obj(v, "text", "conf", "box", "words")
    words = []
    for w in _arr(v["words"]):
        w = _obj(w, "text", "conf", "box")
        words.append(Word(_str(w["text"]), _num(w["conf"]), _d_quad(w["box"])))
    return Line(_str(v["text"]), _num(v["conf"]), _d_quad(v["box"]), tuple(words))


def _e_read(r: EngineRead | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {"use_id": r.use_id, "engine": r.engine, "spec": r.spec, "role": r.role, "status": r.status,
            "lines": [_e_line(ln) for ln in r.lines], "seconds": r.seconds, "error": r.error, "meta": r.meta}


def _d_read(v: Any) -> EngineRead:
    v = _obj(v, "use_id", "engine", "spec", "role", "status", "lines", "seconds", "error", "meta")
    _need(v["role"] in ROLES and v["status"] in READ_STATUSES and isinstance(v["meta"], dict))
    return EngineRead(_str(v["use_id"]), _str(v["engine"]), _str(v["spec"]), v["role"], v["status"],
                      [_d_line(x) for x in _arr(v["lines"])], _num(v["seconds"]), _opt(_str, v["error"]),
                      dict(v["meta"]))


def _e_osd(o: OsdResult | None) -> dict[str, Any] | None:
    if o is None:
        return None
    return {"script": o.script, "script_conf": o.script_conf, "orient_deg": o.orient_deg, "orient_conf": o.orient_conf}


def _d_osd(v: Any) -> OsdResult | None:
    if v is None:
        return None
    v = _obj(v, "script", "script_conf", "orient_deg", "orient_conf")
    return OsdResult(_opt(_str, v["script"]), _num(v["script_conf"]), _int(v["orient_deg"]), _num(v["orient_conf"]))


def _e_signals(s: PageSignals) -> dict[str, Any]:
    sc, m = s.script, s.mixed
    return {
        "index": s.index, "osd": _e_osd(s.osd),
        "script": {"name": sc.name, "conf": sc.conf, "by": sc.by, "eng_mean_conf": sc.eng_mean_conf},
        "eng_read": _e_read(s.eng_read), "checker_read": _e_read(s.checker_read),
        "probes": [[pack, {"pack": p.pack, "chars": p.chars, "lines": p.lines, "min_chars": p.min_chars}]
                   for pack, p in s.probes.items()],
        "windows": [{"start": w.start, "code": w.code, "prob": w.prob} for w in s.windows],
        "mixed": {"checked": m.checked, "script": m.script, "blocks": m.blocks, "conf": m.conf},
    }


def _d_signals(v: Any) -> PageSignals:
    v = _obj(v, "index", "osd", "script", "eng_read", "checker_read", "probes", "windows", "mixed")
    sc = _obj(v["script"], "name", "conf", "by", "eng_mean_conf")
    _need(sc["by"] in _SCRIPT_BY)
    script = ScriptCall(_opt(_str, sc["name"]), _num(sc["conf"]), sc["by"], _opt(_num, sc["eng_mean_conf"]))
    probes: dict[str, ProbeResult] = {}
    for pair in _arr(v["probes"]):
        pack, p = _arr(pair, 2)
        p = _obj(p, "pack", "chars", "lines", "min_chars")
        _need(_str(pack) not in probes)
        probes[pack] = ProbeResult(_str(p["pack"]), _int(p["chars"]), _int(p["lines"]), _opt(_int, p["min_chars"]))
    windows = []
    for w in _arr(v["windows"]):
        w = _obj(w, "start", "code", "prob")
        windows.append(LangWindow(_int(w["start"]), _str(w["code"]), _num(w["prob"])))
    m = _obj(v["mixed"], "checked", "script", "blocks", "conf")
    _need(isinstance(m["checked"], bool))
    mixed = MixedScript(m["checked"], _opt(_str, m["script"]), _int(m["blocks"]), _opt(_num, m["conf"]))
    return PageSignals(_int(v["index"]), _d_osd(v["osd"]), script, _d_read(v["eng_read"]),
                       _opt(_d_read, v["checker_read"]), probes, windows, mixed)


def _d_codes(v: Any, indices: set[int], allowed: frozenset[str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for pair in _arr(v):
        i, codes = _arr(pair, 2)
        _need(_int(i) in indices and i not in out)
        out[i] = [c for c in _arr(codes) if _str(c) in allowed]
        _need(len(out[i]) == len(codes))
    return out


def _encode_body(sig: DocumentSignals) -> dict[str, Any]:
    pages = []
    for p in sig.pages:
        path = Path(p.path)
        pages.append({"index": p.index, "file": path.name, "width": p.width, "height": p.height, "dpi": p.dpi,
                      "cropbox": list(p.cropbox), "bytes": os.stat(path).st_size,
                      "sha256": integrity.sha256_file(path)})
    return {"pages": pages, "signals": [_e_signals(s) for s in sig.signals],
            "osd": [[i, _e_osd(o)] for i, o in sig.osd.items()],
            "warnings": [[i, list(c)] for i, c in sig.warnings.items()],
            "notes": [[i, list(c)] for i, c in sig.notes.items()]}


def _decode_body(body: Any, folder: Path) -> tuple[DocumentSignals, list[tuple[str, int, str]]]:
    """The signals a cache body holds, each page's render in ``folder``, and each render's (file,
    bytes, SHA-256). _Bad for anything ``_encode_body`` doesn't write."""
    b = _obj(body, "pages", "signals", "osd", "warnings", "notes")
    pages: list[PageImage] = []
    files: list[tuple[str, int, str]] = []
    for v in _arr(b["pages"]):
        v = _obj(v, "index", "file", "width", "height", "dpi", "cropbox", "bytes", "sha256")
        index, name = _int(v["index"]), _str(v["file"])
        _need(index >= 0 and name == _RENDER_NAME.format(index))
        width, height, size, sha = _int(v["width"]), _int(v["height"]), _int(v["bytes"]), _str(v["sha256"])
        _need(width > 0 and height > 0 and size > 0 and _SHA_RE.fullmatch(sha) is not None)
        cropbox = tuple(_num(x) for x in _arr(v["cropbox"], 4))
        pages.append(PageImage(index, folder / name, width, height, _num(v["dpi"]), cropbox))  # type: ignore[arg-type]
        files.append((name, size, sha))
    indices = [p.index for p in pages]
    _need(bool(indices) and indices == sorted(set(indices)))
    signals = [_d_signals(v) for v in _arr(b["signals"])]
    _need([s.index for s in signals] == indices)
    found: dict[int, OsdResult | None] = {}
    for pair in _arr(b["osd"]):
        i, o = _arr(pair, 2)
        _need(_int(i) not in found)
        found[i] = _d_osd(o)
    _need(list(found) == indices)
    have = set(indices)
    return (DocumentSignals(pages, signals, [], found, _d_codes(b["warnings"], have, WARNING_CODES),
                            _d_codes(b["notes"], have, NOTE_CODES)), files)


def _canonical(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _same_result(a: DocumentSignals, b: DocumentSignals) -> bool:
    return (a.pages == b.pages and a.signals == b.signals and a.osd == b.osd and a.warnings == b.warnings
            and a.notes == b.notes)


def _transient(read: EngineRead | None) -> bool:
    """A read that failed or timed out for a reason the cache key doesn't hold."""
    if read is None or read.status not in TRANSIENT_STATUSES:
        return False
    state = (read.meta or {}).get("state") if isinstance(read.meta, Mapping) else None
    return read.status == "timeout" or not isinstance(state, str) or not state


def cache_entry(signals: DocumentSignals, key: str) -> dict[str, Any] | None:
    """The JSON object a cache entry stores for ``signals`` under ``key``, or None when it can't be
    stored faithfully (DESIGN §9.1).

    Every render must be a file named ``ocr-<page>.png`` and all of them in one folder, the entry's:
    each one's size and SHA-256 are taken now, from those files. The object holds only pass 1's result
    (``DocumentSignals`` without its classes, which the key covers) and a SHA-256 over it. It is
    returned only when it decodes back to exactly ``signals``: a read whose ``meta`` holds what JSON
    can't carry (a tuple), a NaN, or text that can't be written as UTF-8 means no entry, never a lossy one.
    A read that timed out, or failed for any reason but its use's availability (whose ``meta`` names
    the ``state``), means no entry either: that failure belongs to the moment, not to the key.
    """
    if not isinstance(key, str) or _SHA_RE.fullmatch(key) is None or not signals.pages:
        return None
    if any(_transient(r) for s in signals.signals for r in (s.eng_read, s.checker_read)):
        return None
    folders = {Path(p.path).parent for p in signals.pages}
    if len(folders) != 1:
        return None
    try:
        body = _encode_body(signals)
        canon = _canonical(body)
        back, _files = _decode_body(json.loads(canon.decode("utf-8")), folders.pop())
    except (_Bad, IntegrityError, OSError, TypeError, ValueError, AttributeError, IndexError, KeyError):
        return None
    if not _same_result(back, signals):
        return None
    return {"v": CACHE_VERSION, "key": key, "sha256": hashlib.sha256(canon).hexdigest(), "body": body}


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        _need(k not in out)
        out[k] = v
    return out


def _no_constant(name: str) -> Any:
    raise _Bad()                                       # NaN and Infinity are never written


class CachedDetection:
    """A cache file ``read_cache`` checked: pass 1's result for one key, its renders not checked yet."""

    def __init__(self, body: Mapping[str, Any]) -> None:
        self._body = body
        _sig, files = _decode_body(body, Path("."))
        self.renders: tuple[str, ...] = tuple(name for name, _size, _sha in files)   # file names, page order

    def signals(self, folder: Path) -> DocumentSignals | None:
        """The stored result with every page's render in ``folder``, once each one there is a regular
        file of its recorded size and SHA-256; None when one is missing, changed or a link."""
        folder = Path(folder)
        try:
            sig, files = _decode_body(self._body, folder)
        except _Bad:
            return None
        for name, size, sha in files:
            path = folder / name
            try:
                st = os.lstat(path)
                if not stat.S_ISREG(st.st_mode) or st.st_size != size or integrity.sha256_file(path) != sha:
                    return None
            except (OSError, IntegrityError):
                return None
        return sig


def read_cache(raw: bytes, key: str) -> CachedDetection | None:
    """A cache file's bytes, checked: its format version, its key (exactly ``key``), the SHA-256 over
    its body, and the body's whole structure. None for anything else (a damaged, truncated or foreign
    file, another key), which means a fresh detection."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_CACHE_BYTES:
        return None
    try:
        data = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_no_constant)
        data = _obj(data, "v", "key", "sha256", "body")
        _need(type(data["v"]) is int and data["v"] == CACHE_VERSION)
        _need(isinstance(data["key"], str) and data["key"] == key)
        _need(isinstance(data["sha256"], str) and data["sha256"] == hashlib.sha256(_canonical(data["body"])).hexdigest())
        return CachedDetection(data["body"])
    except (_Bad, ValueError, TypeError, RecursionError, AttributeError, IndexError, KeyError):
        return None


# --------------------------------------------------------------------------- the orientation cache (DESIGN §9.1)

_TURNS = (0, 90, 180, 270)


def orientation_key(pdf: Path, pages: Sequence[int], base: InstalledPack, platform_key: str,
                    settings: Mapping[str, Any]) -> str | None:
    """The SHA-256 of everything ``flow``'s orientation votes and OSD results depend on: the work PDF,
    the pages, the voter's models (their pins) and RapidOCR's version, the OSD model and Tesseract's
    version, the base pack, the platform and OS release, the Python version, the libraries that
    render and read, the core version and ``settings`` (``flow``'s render dpis). None when a part
    can't be taken (an unreadable PDF, the voter's engine that can't load), which means no cache."""
    try:
        from .engines import rapid

        voter_keys = (rapid.VOTER_DET_KEY, rapid.VOTER_REC_KEY)
        voter_version = _engine_version(rapid.ENGINE)
        pins = {a.key: a.sha256 for a in base.manifest.models if a.key in voter_keys}
        payload = {
            "cache": CACHE_VERSION,
            "kind": "orientation",
            "core": __version__,
            "pdf": integrity.sha256_file(pdf),
            "pages": [int(i) for i in pages],
            "platform": platform_key,
            "os": _os_release(),
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "base": _pack_part(base),
            "voter": {"models": [[k, pins.get(k)] for k in voter_keys], "engine": voter_version},
            "osd_engine": _osd_part(base),
            "libraries": _library_versions(),
            "settings": to_jsonable(dict(settings)),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except Exception:  # noqa: BLE001 - any part that can't be taken (an import, a read, a value) is no cache
        return None
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def check_orientation_models(base: InstalledPack) -> None:
    """Hash-check the voter's and OSD's model files now, as opening them would: a stored orientation
    is never used with a model that doesn't match its pin (IntegrityError, as on a fresh run).
    PackError when the base pack lacks one of them."""
    from .engines import rapid

    for key in (rapid.VOTER_DET_KEY, rapid.VOTER_REC_KEY, _osd_key(base)):
        if key is None:
            raise PackError("osd_unavailable", f"the {base.manifest.id!r} pack has no orientation model")
        base.asset_path(key)


def _e_vote(v: Any) -> dict[str, Any]:
    return {"page": v.page, "turn_cw": v.turn_cw, "margin": v.margin, "boxes": v.boxes,
            "scores": [[int(t), s] for t, s in sorted(dict(v.scores).items())]}


def _d_vote(v: Any, page: int) -> Any:
    from ..pdf.orient import OrientVote

    v = _obj(v, "page", "turn_cw", "margin", "boxes", "scores")
    _need(_int(v["page"]) == page and _int(v["turn_cw"]) in _TURNS and _int(v["boxes"]) >= 0)
    scores: dict[int, float] = {}
    for pair in _arr(v["scores"]):
        t, s = _arr(pair, 2)
        _need(_int(t) in _TURNS and t not in scores)
        scores[t] = _num(s)
    return OrientVote(page, v["turn_cw"], _num(v["margin"]), v["boxes"], scores)


def _decode_orientation(body: Any) -> tuple[dict[int, Any], dict[int, OsdResult | None]]:
    b = _obj(body, "pages")
    votes: dict[int, Any] = {}
    osd: dict[int, OsdResult | None] = {}
    for p in _arr(b["pages"]):
        p = _obj(p, "page", "vote", "osd")
        i = _int(p["page"])
        _need(i >= 0 and i not in votes)
        votes[i] = _d_vote(p["vote"], i)
        osd[i] = _d_osd(p["osd"])
    _need(bool(votes) and list(votes) == sorted(votes))
    return votes, osd


def orientation_entry(votes: Mapping[int, Any], osd: Mapping[int, OsdResult | None], key: str) -> dict[str, Any] | None:
    """The JSON object the orientation cache stores for ``votes`` (page -> ``orient.OrientVote``) and
    ``osd`` (page -> OSD's result) under ``key``, or None when it can't be stored faithfully: the
    same pages in both, a valid key, and an object that decodes back to exactly these values."""
    if not isinstance(key, str) or _SHA_RE.fullmatch(key) is None or not votes or set(votes) != set(osd):
        return None
    try:
        body = {"pages": [{"page": int(i), "vote": _e_vote(votes[i]), "osd": _e_osd(osd[i])} for i in sorted(votes)]}
        canon = _canonical(body)
        back_votes, back_osd = _decode_orientation(json.loads(canon.decode("utf-8")))
    except (_Bad, TypeError, ValueError, AttributeError, IndexError, KeyError):
        return None
    for i in votes:
        if (i not in back_votes or back_votes[i] != votes[i] or dict(back_votes[i].scores) != dict(votes[i].scores)
                or back_osd.get(i, _MISSING) != osd[i]):
            return None
    return {"v": CACHE_VERSION, "kind": "orientation", "key": key, "sha256": hashlib.sha256(canon).hexdigest(),
            "body": body}


def read_orientation(raw: bytes, key: str) -> tuple[dict[int, Any], dict[int, OsdResult | None]] | None:
    """An orientation file's bytes, checked as ``read_cache`` checks a detection file (format version,
    kind, exactly ``key``, the SHA-256 over the body, the body's whole structure): the votes and OSD
    results per page, or None for anything else, which means orienting again."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > ORIENT_MAX_BYTES:
        return None
    try:
        data = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_no_constant)
        data = _obj(data, "v", "kind", "key", "sha256", "body")
        _need(type(data["v"]) is int and data["v"] == CACHE_VERSION and data["kind"] == "orientation")
        _need(isinstance(data["key"], str) and data["key"] == key)
        _need(isinstance(data["sha256"], str)
              and data["sha256"] == hashlib.sha256(_canonical(data["body"])).hexdigest())
        return _decode_orientation(data["body"])
    except (_Bad, ValueError, TypeError, RecursionError, AttributeError, IndexError, KeyError):
        return None


# --------------------------------------------------------------------------- pass 2

@dataclass
class _Prepared:
    pack: InstalledPack
    rs: ReaderSet
    uses: list[_Use]
    rule: Any
    pick_params: dict[str, Any]
    checker_id: str | None
    normalize: tuple[str, ...]
    repairs: list[tuple[StepUse, Repair, dict[str, Any]]]
    fills: tuple[StepUse, ...]
    must: list[MustSurvive]
    script: str | None              # the pack's first script, for fills on a page with no script call


def _prepare(pack: InstalledPack, base: InstalledPack, platform_key: str,
             pool: Mapping[str, InstalledPack]) -> _Prepared:
    """Everything about one pack's reader set that doesn't depend on the page, checked up front so a
    bad name or parameter is a PackError before any page is read."""
    m = pack.manifest
    rs = m.readers_for(platform_key)
    uses = [_make_use(pack, base, use, platform_key) for use in rs.engines]
    use_ids = frozenset(u.use.id for u in uses)
    known_scripts = frozenset(scripts.SCRIPTS)
    rule = rules.get(rs.pick.rule, rs.pick.version)
    pick_params = rule.PARAMS.validate(dict(rs.pick.params),
                                       ValidationContext(use_ids, frozenset(), {}, known_scripts, f"{m.id}.pick.params"))
    for name in m.normalize:
        if name not in textfix.NORMALIZERS:
            raise PackError("unknown_normalizer",
                            f"unknown normaliser {name!r}; this version of pdfintake knows {', '.join(textfix.NORMALIZERS)}",
                            normalizer=name)
    repairs: list[tuple[StepUse, Repair, dict[str, Any]]] = []
    for step in m.repairs:
        repair = textfix.REPAIRS.get((step.name, step.version))
        if repair is None:
            raise PackError("unknown_repair", f"unknown repair {step.name!r} v{step.version}",
                            repair=step.name, version=step.version)
        params = repair.PARAMS.validate(dict(step.params or {}),
                                        ValidationContext(use_ids, frozenset(), {}, known_scripts,
                                                          f"{m.id}.repairs.{step.name}"))
        repairs.append((step, repair, params))
    for step in m.fills:
        if (step.name, step.version) not in fills.FILLS or step.name not in fills.ORDER:
            raise PackError("unknown_fill", f"unknown fill {step.name!r} v{step.version}",
                            fill=step.name, version=step.version)
    checker = _first(rs, "checker")
    return _Prepared(pack, rs, uses, rule, pick_params, checker.id if checker is not None else None,
                     tuple(m.normalize), repairs, tuple(m.fills), list(m.must_survive),
                     m.scripts[0] if m.scripts else None)


def _reusable(signals: DocumentSignals | None, base: InstalledPack, platform_key: str) -> dict[int, dict[str, EngineRead]]:
    """Per page index, the pass-1 reads by use key (the base pack's eng and checker reads)."""
    if signals is None:
        return {}
    keys: dict[str, str] = {}
    for use in detect_readers(base).engines:                # the set pass 1 read with, whatever this platform reads with
        engine, _broken = _engine(use.engine)
        if engine is not None:
            keys[use.id] = _use_key(base, base, use, engine)
    out: dict[int, dict[str, EngineRead]] = {}
    for sig in signals.signals:
        found: dict[str, EngineRead] = {}
        for read in (sig.eng_read, sig.checker_read):
            if read is not None and read.use_id in keys:
                found.setdefault(keys[read.use_id], read)
        out[sig.index] = found
    return out


def _same_render(page: PageImage, gathered: PageImage | None) -> bool:
    """``page`` is the render pass 1 read: same file, pixel size and dpi. A pass-1 read's boxes are in
    that render's pixel space, so another render of the page (other dpi, a re-render) is read again."""
    if gathered is None:
        return False
    return (Path(page.path) == Path(gathered.path) and page.width == gathered.width
            and page.height == gathered.height and float(page.dpi) == float(gathered.dpi))


def _second_packs(plan: PagePlan, pool: Mapping[str, InstalledPack]) -> list[InstalledPack]:
    """The packs of the page's other languages (``languages[1:]``) that this run has, for their counts."""
    out: list[InstalledPack] = []
    taken: set[str] = set()
    for call in plan.languages[1:]:
        for pid in sorted(pool):
            m = pool[pid].manifest
            tags = {str(x.get("tag")) for x in m.languages if x.get("role") == "primary"} | set(m.detect.langid_codes)
            if call.tag in tags and pid != plan.pack and pid not in taken:
                out.append(pool[pid])
                taken.add(pid)
                break
    return out


def _chars_script(chars: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in chars:
        s = scripts.script_of(ch)
        if s is not None and s in scripts.SCRIPTS:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda t: (-t[1], t[0]))[0][0]


def must_survive_count(text: str, ms: MustSurvive) -> int:
    """How many of ``ms``'s units ``text`` holds: characters (``char``), whitespace-separated words
    holding one of its characters (``word``), or trigrams of its characters' script holding one
    (``trigram``, with multiplicity, as ``scripts.trigrams`` counts them). ``scripts.trigrams``
    NFKC-normalises the text, so a trigram holds a character when it holds the character's NFKC
    form (Thai sara am is nikhahit + sara aa there)."""
    chars = set(ms.chars)
    text = text or ""
    if ms.unit == "char":
        return sum(1 for ch in text if ch in chars)
    if ms.unit == "word":
        return sum(1 for w in text.split() if any(ch in chars for ch in w))
    script = _chars_script(ms.chars)
    if script is None:
        return sum(1 for ch in text if ch in chars)
    needles = {unicodedata.normalize("NFKC", ch) for ch in chars}
    return sum(n for gram, n in scripts.trigrams(text, script).items() if any(nd in gram for nd in needles))


def _layer_lines(lines: Sequence[Line], page_fills: Sequence[Fill]) -> list[LayerLine]:
    out = [LayerLine(ln.text, ln.box) for ln in lines if ln.text and ln.text.strip()]
    out += [LayerLine(f.text, f.box) for f in page_fills if f.text and f.text.strip()]
    return out


def _dedupe(codes: Sequence[str]) -> list[str]:
    out: list[str] = []
    for c in codes:
        if c in WARNING_CODES and c not in out:
            out.append(c)
    return out


def _reader_warnings(reads: Sequence[EngineRead], has_layer: bool) -> list[str]:
    out: list[str] = []
    for r in reads:
        state = r.meta.get("state") if isinstance(r.meta, Mapping) else None
        if r.status == "unavailable" and r.engine == "vision" and state == "unsupported_language":
            out.append("vision_language_unsupported")
        if has_layer and r.status == "failed":
            out.append("reader_failed")
        elif has_layer and r.status == "timeout":
            out.append("reader_timeout")
    return out


def _page_script(plan: PagePlan, sig: PageSignals | None) -> ScriptCall:
    if plan.script is not None:
        return plan.script
    if sig is not None:
        return sig.script
    return ScriptCall(None, 0.0, "none", None)


def _read_page(prep: _Prepared | None, page: PageImage, plan: PagePlan, sig: PageSignals | None,
               reuse: Mapping[str, EngineRead], extra: Sequence[str], sessions: _Sessions,
               pool: Mapping[str, InstalledPack]) -> PageOCR:
    """Pass 2 for one page (runs in a worker)."""
    script = _page_script(plan, sig)
    warnings = [*plan.warnings, *extra]
    if prep is None:                                     # planned with no pack: no reads, no layer
        pick = Pick(NO_LAYER_RULE, 0, None, [], {}, False, "no text layer was asked for on this page")
        return PageOCR(page.index, script, list(plan.languages), None, plan.composition, [], pick, [], {}, {},
                       _dedupe(warnings))
    reads: list[EngineRead] = []
    for u in prep.uses:
        raw = reuse.get(u.key)
        if raw is not None:
            read = dataclasses.replace(raw, use_id=u.use.id, role=u.use.role, meta={**(raw.meta or {}), "pass1": True})
        else:
            read = _read(sessions, u, page)
        reads.append(dataclasses.replace(read, lines=textfix.apply_normalizers(prep.normalize, read.lines)))
    by_id = {r.use_id: r for r in reads}
    pick = prep.rule.pick(by_id, prep.pick_params, fallback=prep.rs.fallback, checker_id=prep.checker_id)
    checker = by_id.get(prep.checker_id) if prep.checker_id is not None else None
    lines = list(pick.lines)
    repairs: dict[str, int] = {}
    page_fills: list[Fill] = []
    if pick.chosen is not None:
        for step, repair, params in prep.repairs:
            source = params.get("source")
            lines, counts = repair.repair(lines, by_id.get(source) if source else checker, params,
                                          languages=[c.tag for c in plan.languages])
            for k, v in counts.items():
                repairs[f"{step.name}.{k}"] = repairs.get(f"{step.name}.{k}", 0) + int(v)
        page_fills = fills.run_fills(prep.fills, lines, checker, page, script.name or prep.script or "Latin")
    pick = dataclasses.replace(pick, lines=lines)
    has_layer = pick.chosen is not None
    warnings += _reader_warnings(reads, has_layer)
    warnings += [w for w in pick.warnings if w in rules.PICK_WARNING_CODES]
    if pick.fallback:
        warnings.append("fallback_used")
    if not has_layer:
        warnings.append("no_layer_readers_failed")
    layer_text = "\n".join(ll.text for ll in _layer_lines(lines, page_fills))
    must: dict[str, dict[str, int]] = {}
    items = list(prep.must)
    for other in _second_packs(plan, pool):
        items += list(other.manifest.must_survive)
    for ms in items:
        if ms.name in must:
            continue
        row = {r.use_id: must_survive_count(r.text(), ms) for r in reads if r.status in WORKING}
        row[LAYER_KEY] = must_survive_count(layer_text, ms)
        must[ms.name] = row
    return PageOCR(page.index, script, list(plan.languages), plan.pack, plan.composition, reads, pick, page_fills,
                   repairs, must, _dedupe(warnings))


def read_pages(pages: Sequence[PageImage], plans: Sequence[PagePlan], packs: Mapping[str, InstalledPack],
               base: InstalledPack, platform_key: str, *, signals: DocumentSignals | None, workers: int) -> list[PageOCR]:
    """Pass 2: read each planned page with its pack's reader set; one ``PageOCR`` per plan, in plan order.

    ``packs`` holds every pack a plan names (the base pack may be left out: ``base`` reads its
    pages). ``signals`` is pass 1's result, whose reads are reused where a use is configured the
    same and the page passed here is the render pass 1 read (same file, size and dpi); the kit
    passes None. A plan with no pack and composition ``none`` (``--lang none``) gives a page with no
    reads and no layer; any other plan with no pack (an ``undecided`` page ``langplan.resolve_unsure``
    didn't settle) is a ValueError, so no page loses its text unrecorded. Works without the index or
    the store: a draft manifest in an ``InstalledPack`` reads.
    """
    n = _workers(workers)
    by_index: dict[int, PageImage] = {}
    for p in pages:
        by_index.setdefault(p.index, p)
    plan_list = list(plans)
    seen: set[int] = set()
    for pp in plan_list:
        if pp.index in seen:
            raise ValueError(f"page {pp.index} is planned twice")
        if pp.index not in by_index:
            raise ValueError(f"page {pp.index} is planned but has no render")
        if pp.pack is None and pp.composition != NO_LAYER_COMPOSITION:
            # Only --lang none may leave a page without a layer. An undecided page here would lose its
            # text with nothing in the record, so it is a caller bug (decided 2026-11-05, DESIGN §12.3).
            raise ValueError(f"page {pp.index} is {pp.composition} and has no pack: "
                             "call langplan.resolve_unsure before read_pages")
        seen.add(pp.index)
    pool: dict[str, InstalledPack] = {base.manifest.id: base}
    pool.update(packs)
    prepared: dict[str, _Prepared] = {}
    for pp in plan_list:
        if pp.pack is None or pp.pack in prepared:
            continue
        if pp.pack not in pool:
            raise ValueError(f"page {pp.index} is planned with pack {pp.pack!r}, which wasn't supplied")
        prepared[pp.pack] = _prepare(pool[pp.pack], base, platform_key, pool)
    reuse = _reusable(signals, base, platform_key)
    if signals is not None:                              # a pass-1 read is reused only on the render it read
        rendered = {p.index: p for p in signals.pages}
        reuse = {i: r for i, r in reuse.items() if i in by_index and _same_render(by_index[i], rendered.get(i))}
    sigs = {s.index: s for s in signals.signals} if signals is not None else {}
    extra = signals.warnings if signals is not None else {}
    sessions = _Sessions(base)
    executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix="pdfintake-read")
    try:
        futures = [executor.submit(_read_page, prepared.get(pp.pack) if pp.pack is not None else None,
                                   by_index[pp.index], pp, sigs.get(pp.index), reuse.get(pp.index, {}),
                                   list(extra.get(pp.index, ())), sessions, pool)
                   for pp in plan_list]
        return [f.result() for f in futures]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        sessions.close()


# --------------------------------------------------------------------------- the layer writer's input

def layer_pages(ocr: Sequence[PageOCR], pages: Sequence[PageImage], classes: Sequence[PageClass]) -> list[LayerPage]:
    """The layer writer's input: per OCR'd page, the chosen lines, then the fills, as pixel boxes of
    the page's render. A page with nothing to write keeps an empty ``LayerPage`` (the writer gives it
    no form). ValueError for a page classed as not needing OCR, or missing from ``classes``: a page
    with printed text never gets a layer."""
    by_index = {p.index: p for p in pages}
    needs = {c.index: c.needs_ocr for c in classes}
    out: list[LayerPage] = []
    for o in sorted(ocr, key=lambda x: x.index):
        page = by_index.get(o.index)
        if page is None:
            raise ValueError(f"page {o.index} was read but has no render")
        if not needs.get(o.index, False):              # fails closed: a page missing from ``classes`` too
            raise ValueError(f"page {o.index} isn't a page classed as needing OCR, so it never gets a layer")
        out.append(LayerPage(o.index, page.width, page.height, tuple(_layer_lines(o.pick.lines, o.fills))))
    return out


def notes_for(ocr: PageOCR) -> list[str]:
    """The page's note codes (``record.NOTE_CODES``) that ``PageOCR`` has no field for: today
    ``vision_not_on_platform`` when a Vision use was skipped because Vision isn't on this platform."""
    out: list[str] = []
    for r in ocr.reads:
        state = r.meta.get("state") if isinstance(r.meta, Mapping) else None
        if r.engine == "vision" and r.status == "unavailable" and state == "not_on_platform":
            if "vision_not_on_platform" not in out:
                out.append("vision_not_on_platform")
    return out
