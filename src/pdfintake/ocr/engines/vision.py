"""Apple Vision as an OCR engine (VNRecognizeTextRequest through pyobjc-framework-Vision).

Vision is macOS only and has no model files: the operating system supplies the recogniser, so its
reads change with the macOS version. Three things about it shape this module:

* **Unsupported languages fail silently.** Asked for a language the request's revision and level
  don't list, Vision still returns ok, with junk or with nothing. So ``availability`` checks every
  requested tag against ``supported_languages(revision, level)`` before any read, and a session
  asked for one anyway reads ``unavailable`` without ever calling ``performRequests``.
* **The image handler's options are ``None``, never ``{}``.** pyobjc 13 rejects an empty
  dictionary there ("key does not exist") on some macOS releases (27 does, the macOS 26 CI runner
  doesn't); ``None`` works on every one.
* **A request can hang.** Each read runs the request in a daemon thread joined with a deadline:
  the smaller of the caller's ``timeout_s`` and the pack's ``timeout_s`` param (default 120 s), the
  same cap Tesseract applies. Running out gives status ``timeout``, and the request is cancelled so
  the thread stops early.

Vision is imported lazily, on the first call that needs it, so importing this module costs nothing
and works on every platform. Off macOS ``availability`` answers ``not_on_platform`` and a session
reads ``unavailable``; neither is ever a failure.

Boxes: Vision reports normalized boxes with the origin at the bottom left. They are converted here
to quads in pixels of the shared render (origin top left, y down, clockwise from the top left),
clamped to the image.
"""

from __future__ import annotations

import functools
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ...errors import EngineFailure
from ...params import Param, ParamTable
from ...types import EngineRead, Line, PageImage, Quad, Role
from . import Availability, ContractCase, ModelPaths

if TYPE_CHECKING:
    from ...langpacks.manifest import Asset

__all__ = [
    "DEFAULT_REVISION", "DEFAULT_TIMEOUT_S", "ENGINE", "LANGUAGE_PATTERN", "LEVELS",
    "VisionEngine", "VisionSession", "supported_languages",
]

DEFAULT_REVISION = 3
DEFAULT_TIMEOUT_S = 120.0
LEVELS: tuple[str, ...] = ("accurate", "fast")
_LEVEL_CODES = {"accurate": 0, "fast": 1}          # VNRequestTextRecognitionLevelAccurate / ...Fast
REVISIONS: tuple[int, ...] = (1, 2, 3)              # VNRecognizeTextRequestRevision1..3
PLATFORM_KEYS = frozenset({"darwin-arm64"})

# Vision's own language tags: "en-US", "th-TH", "zh-Hans", "yue-Hant", "ars-SA", "vi-VT".
LANGUAGE_PATTERN = r"[a-z]{2,3}(?:-[A-Z][a-z]{3})?(?:-(?:[A-Z]{2}|[0-9]{3}))?"
_ERROR_CHARS = 200


# --------------------------------------------------------------------------- the lazy Vision import


@dataclass(frozen=True)
class _Bridge:
    """The pyobjc pieces a read needs, imported once."""

    Vision: Any
    objc: Any
    NSURL: Any


_load_lock = threading.Lock()
_bridge: _Bridge | None = None
_load_error: str | None = None


def _on_darwin() -> bool:
    return sys.platform == "darwin"


def _machine() -> str:
    # os.uname is a system call; platform.machine can shell out on some systems (test_hygiene).
    return os.uname().machine if hasattr(os, "uname") else ""


def _vision() -> _Bridge:
    """Vision, objc and NSURL, imported on first use. EngineFailure (remembered) when they can't be."""
    global _bridge, _load_error
    with _load_lock:
        if _bridge is not None:
            return _bridge
        if _load_error is None:
            if not _on_darwin():
                _load_error = "Apple Vision exists only on macOS"
            else:
                try:
                    import objc
                    import Vision
                    from Foundation import NSURL
                except Exception as exc:  # ImportError, or a framework that fails to load
                    _load_error = f"pyobjc-framework-Vision could not be loaded: {type(exc).__name__}: {exc}"[:_ERROR_CHARS]
                else:
                    _bridge = _Bridge(Vision=Vision, objc=objc, NSURL=NSURL)
                    return _bridge
        raise EngineFailure("vision_unavailable", _load_error)


def _reset_for_tests() -> None:
    """Tests only: forget the imported bridge, a remembered import failure and the caches."""
    global _bridge, _load_error
    with _load_lock:
        _bridge = None
        _load_error = None
    for cached in _CACHES:
        cached.cache_clear()


@functools.lru_cache(maxsize=None)
def _revisions_cached() -> tuple[int, ...]:
    request = _vision().Vision.VNRecognizeTextRequest
    return tuple(sorted(int(r) for r in request.supportedRevisions()))


def _supported_revisions() -> tuple[int, ...]:
    """The VNRecognizeTextRequest revisions this macOS offers; () off macOS."""
    if not _on_darwin():
        return ()
    return _revisions_cached()


@functools.lru_cache(maxsize=None)
def _supported_cached(revision: int, level: str) -> tuple[str, ...]:
    bridge = _vision()
    if revision not in _supported_revisions():
        return ()
    with bridge.objc.autorelease_pool():
        request = bridge.Vision.VNRecognizeTextRequest.alloc().init()
        request.setRevision_(revision)
        request.setRecognitionLevel_(_LEVEL_CODES[level])
        languages, error = request.supportedRecognitionLanguagesAndReturnError_(None)
        if error is not None or languages is None:
            raise EngineFailure("vision_languages_failed",
                                f"Vision would not list its languages: {_describe_error(error)}")
        return tuple(str(tag) for tag in languages)


def supported_languages(revision: int, level: str) -> tuple[str, ...]:
    """The language tags Vision's text request reads at this revision and level, on this macOS.

    () off macOS and for a revision this macOS doesn't have. EngineFailure when Vision can't be
    loaded on macOS. Cached for the life of the process.
    """
    if level not in _LEVEL_CODES:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}, not {level!r}")
    if not _on_darwin():
        return ()
    return _supported_cached(int(revision), level)


@functools.lru_cache(maxsize=None)
def _macos_version() -> str:
    return platform.mac_ver()[0] or "unknown"


# Captured here so _reset_for_tests clears the real caches even while a test has patched the names.
_CACHES = (_revisions_cached, _supported_cached, _macos_version)


def _describe_error(error: Any) -> str:
    if error is None:
        return "no error was given"
    try:
        text = f"{error.domain()} {error.code()}: {error.localizedDescription()}"
    except Exception:
        text = str(error)
    return text[:_ERROR_CHARS]


# --------------------------------------------------------------------------- one request


@dataclass(frozen=True)
class _Options:
    languages: tuple[str, ...]
    revision: int
    level: str
    language_correction: bool


@dataclass(frozen=True)
class _Raw:
    """One observation as Vision gives it: the top candidate and a normalized bottom-left box."""

    text: str
    conf: float
    x: float
    y: float
    w: float
    h: float


class _VisionRequestError(Exception):
    """performRequests reported an error."""


class _Cancelled(Exception):
    """The request was cancelled after its deadline passed."""


class _Job:
    """Lets the waiting thread cancel the request the worker thread is running."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request: Any = None
        self.cancelled = False

    def attach(self, request: Any) -> bool:
        with self._lock:
            if self.cancelled:
                return False
            self._request = request
            return True

    def detach(self) -> None:
        with self._lock:
            self._request = None

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            request = self._request
        if request is not None:
            try:
                request.cancel()
            except Exception:
                pass


def _perform(path: Path, opts: _Options, job: _Job) -> list[_Raw]:
    """Run one VNRecognizeTextRequest on the PNG at ``path`` (in the calling worker thread)."""
    bridge = _vision()
    vn = bridge.Vision
    with bridge.objc.autorelease_pool():
        url = bridge.NSURL.fileURLWithPath_(str(path))
        handler = vn.VNImageRequestHandler.alloc().initWithURL_options_(url, None)   # None, never {}
        request = vn.VNRecognizeTextRequest.alloc().init()
        request.setRevision_(opts.revision)
        request.setRecognitionLevel_(_LEVEL_CODES[opts.level])
        request.setUsesLanguageCorrection_(bool(opts.language_correction))
        request.setRecognitionLanguages_(list(opts.languages))
        if not job.attach(request):
            raise _Cancelled()
        try:
            ok, error = handler.performRequests_error_([request], None)
        finally:
            job.detach()
        if job.cancelled:
            raise _Cancelled()
        if not ok:
            raise _VisionRequestError(_describe_error(error))
        raws: list[_Raw] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            best = candidates[0]
            bb = observation.boundingBox()
            raws.append(_Raw(str(best.string()), float(best.confidence()),
                             float(bb.origin.x), float(bb.origin.y), float(bb.size.width), float(bb.size.height)))
        return raws


def _run_with_deadline(work: Callable[[], Any], timeout_s: float, on_timeout: Callable[[], None]) -> tuple[str, Any]:
    """Run ``work`` in a daemon thread and wait at most ``timeout_s``.

    ("done", value), ("error", exception) or ("timeout", None); on a timeout ``on_timeout`` runs
    and the thread is left to finish on its own.
    """
    finished = threading.Event()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:  # handed to the waiting thread, never lost
            box["error"] = exc
        finally:
            finished.set()

    worker = threading.Thread(target=target, name="pdfintake-vision", daemon=True)
    worker.start()
    if not finished.wait(max(float(timeout_s), 0.0)):
        on_timeout()
        return "timeout", None
    if "error" in box:
        return "error", box["error"]
    return "done", box.get("value")


def _clamp(v: float, hi: float) -> float:
    return min(max(v, 0.0), float(hi))


def _to_lines(raws: Sequence[_Raw], width: int, height: int) -> list[Line]:
    """Vision's normalized bottom-left boxes as px quads of the render, top to bottom."""
    out: list[tuple[float, float, Line]] = []
    for r in raws:
        if not r.text:
            continue
        left, right = sorted((r.x, r.x + r.w))
        bottom, top = sorted((r.y, r.y + r.h))
        x0 = _clamp(round(left * width, 1), width)
        x1 = _clamp(round(right * width, 1), width)
        y0 = _clamp(round((1.0 - top) * height, 1), height)
        y1 = _clamp(round((1.0 - bottom) * height, 1), height)
        box: Quad = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        conf = round(min(max(r.conf, 0.0), 1.0), 3)
        out.append((y0, x0, Line(r.text, conf, box)))
    out.sort(key=lambda item: (item[0], item[1]))
    return [line for _, _, line in out]


# --------------------------------------------------------------------------- the engine


def _options(params: Mapping[str, Any]) -> _Options:
    return _Options(
        languages=tuple(params.get("languages") or ()),
        revision=int(params.get("revision", DEFAULT_REVISION)),
        level=str(params.get("level", "accurate")),
        language_correction=bool(params.get("language_correction", True)),
    )


def _limit_s(params: Mapping[str, Any]) -> float:
    """The pack's ``timeout_s`` (validated 10..600), DEFAULT_TIMEOUT_S when absent."""
    return float(params.get("timeout_s", DEFAULT_TIMEOUT_S))


def _language_gate(opts: _Options) -> Availability:
    """Whether this macOS's Vision can read ``opts`` (call only on macOS). EngineFailure if Vision won't load."""
    revisions = _supported_revisions()
    if opts.revision not in revisions:
        offered = ", ".join(str(r) for r in revisions) or "none"
        return Availability(False, "not_on_platform",
                            f"Vision text recognition revision {opts.revision} isn't on macOS {_macos_version()} "
                            f"(this macOS offers {offered})")
    if not opts.languages:
        return Availability(False, "unsupported_language", "no Vision language was requested")
    supported = supported_languages(opts.revision, opts.level)
    missing = [tag for tag in opts.languages if tag not in supported]
    if missing:
        return Availability(False, "unsupported_language",
                            f"Vision revision {opts.revision} ({opts.level}) on macOS {_macos_version()} doesn't read "
                            f"{', '.join(missing)}")
    return Availability(True, "available", "")


class VisionSession:
    """Reads pages with one fixed set of Vision options. Thread-safe: every read is its own request."""

    def __init__(self, engine: VisionEngine, use_id: str, role: Role, opts: _Options, unavailable: str | None,
                 limit_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.engine = engine
        self.use_id = use_id
        self.role = role
        self.opts = opts
        self.limit_s = float(limit_s)           # the pack's timeout_s: no read waits longer than this
        self.spec = _spec(opts)
        self.unavailable = unavailable          # why every read is "unavailable", or None
        self.closed = False

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        if self.closed:
            raise EngineFailure("session_closed", "this Vision session was closed")
        start = time.perf_counter()

        def result(status: str, lines: list[Line], error: str | None = None) -> EngineRead:
            meta: dict[str, Any] = {"revision": self.opts.revision, "level": self.opts.level,
                                    "languages": list(self.opts.languages)}
            version = self.engine.version() if status != "unavailable" else None
            if version is not None:
                meta["version"] = version
            if lines:
                meta["mean_conf"] = round(sum(ln.conf for ln in lines) / len(lines), 4)
            return EngineRead(self.use_id, self.engine.name, self.spec, self.role, status,  # type: ignore[arg-type]
                              lines, time.perf_counter() - start, error, meta)

        if self.unavailable is not None:
            return result("unavailable", [], self.unavailable)
        if not Path(page.path).is_file():
            return result("failed", [], "the page image is missing")
        try:
            limit = min(float(timeout_s), self.limit_s)
        except (TypeError, ValueError):
            limit = self.limit_s
        if not limit > 0:                       # 0, negative or NaN: no time left, so no request
            return result("timeout", [], "no time left for the read")
        job = _Job()
        outcome, value = _run_with_deadline(lambda: _perform(page.path, self.opts, job), limit, job.cancel)
        if outcome == "timeout":
            return result("timeout", [], f"Vision gave no result within {limit:g} s")
        if outcome == "error":
            if isinstance(value, _Cancelled):
                return result("timeout", [], "the Vision request was cancelled at its deadline")
            if isinstance(value, _VisionRequestError):
                return result("failed", [], f"Vision: {value}"[:_ERROR_CHARS])
            if isinstance(value, EngineFailure):
                return result("failed", [], value.message[:_ERROR_CHARS])
            return result("failed", [], f"Vision: {type(value).__name__}: {value}"[:_ERROR_CHARS])
        lines = _to_lines(value, page.width, page.height)
        return result("ok" if lines else "empty", lines)

    def close(self) -> None:
        self.closed = True


def _spec(opts: _Options) -> str:
    spec = f"vision:{'+'.join(opts.languages) or '-'}:rev{opts.revision}:{opts.level}"
    return spec if opts.language_correction else spec + ":nolc"


class VisionEngine:
    name: ClassVar[str] = "vision"
    api: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("languages", "str_list", required=True, pattern=LANGUAGE_PATTERN, max_len=16, min=1, max=8),
        Param("revision", "int", default=DEFAULT_REVISION, choices=REVISIONS),
        Param("level", "str", default="accurate", choices=LEVELS),
        Param("language_correction", "bool", default=True),
        Param("timeout_s", "float", default=DEFAULT_TIMEOUT_S, min=10.0, max=600.0),
    )
    MODEL_KINDS: ClassVar[frozenset[str]] = frozenset()
    platforms: ClassVar[frozenset[str]] = PLATFORM_KEYS
    session_per_thread: ClassVar[bool] = False

    def version(self) -> str | None:
        """"rev3 (macOS 15.4)", the text request revision packs use and the OS that supplies it; None off macOS."""
        if not _on_darwin():
            return None
        try:
            revisions = _supported_revisions()
        except EngineFailure:
            return None
        mac = _macos_version()
        return f"rev{DEFAULT_REVISION} (macOS {mac})" if DEFAULT_REVISION in revisions else f"macOS {mac}"

    def availability(self, params: Mapping[str, Any], models: ModelPaths, platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform", f"Apple Vision runs only on {', '.join(sorted(self.platforms))}")
        if not _on_darwin():
            return Availability(False, "not_on_platform", "Apple Vision exists only on macOS")
        try:
            return _language_gate(_options(params))
        except EngineFailure as exc:
            return Availability(False, "broken", exc.message)

    def open(self, use_id: str, role: Role, params: Mapping[str, Any], models: ModelPaths) -> VisionSession:
        """A session. Off macOS, or for a revision or language this macOS lacks, its reads are
        ``unavailable``. EngineFailure when Vision can't be loaded on macOS."""
        opts = _options(params)
        unavailable: str | None = None
        if not _on_darwin():
            unavailable = "Apple Vision exists only on macOS"
        else:
            gate = _language_gate(opts)            # EngineFailure propagates: a broken install
            if not gate.ok:
                unavailable = gate.detail
        return VisionSession(self, use_id, role, opts, unavailable, _limit_s(params))

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence[Asset]) -> list[dict[str, Any]]:
        """Vision settings the kit can bench for ``lang_tag`` (an ISO 639 code), listed only when
        revision 3 accurate on this macOS reports a matching tag; English rides along as the second
        language, as the launch packs read it."""
        if not _on_darwin():
            return []
        try:
            supported = supported_languages(DEFAULT_REVISION, "accurate")
        except EngineFailure:
            return []
        code = (lang_tag or "").split("-")[0].lower()
        tags = [tag for tag in supported if tag.split("-")[0].lower() == code]
        out: list[dict[str, Any]] = []
        for tag in tags:
            languages = [tag] if code == "en" or "en-US" not in supported else [tag, "en-US"]
            for correction in (True, False):
                out.append({"languages": languages, "revision": DEFAULT_REVISION, "level": "accurate",
                            "language_correction": correction})
        return out

    def contract_case(self) -> ContractCase | None:
        """English at revision 3 on a page of rendered made-up text; None off Apple-silicon macOS
        or when Vision can't be loaded or doesn't read en-US."""
        if not _on_darwin() or _machine() != "arm64":
            return None
        try:
            supported = supported_languages(DEFAULT_REVISION, "accurate")
        except EngineFailure:
            return None
        if "en-US" not in supported:
            return None
        unsupported = next((t for t in ("hu-HU", "el-GR", "he-IL", "zz-ZZ") if t not in supported), None)
        return ContractCase(
            params={"languages": ["en-US"]},
            page=_contract_page,
            unsupported_params={"languages": [unsupported]} if unsupported else None,
            platform_key="darwin-arm64",
            timeout_s=DEFAULT_TIMEOUT_S,
        )


_CONTRACT_TEXT = ("The example invoice lists seven", "sample items and 4521 units.")


def _contract_page(folder: Path) -> PageImage:
    """Two lines of made-up text in Pillow's built-in font, as a 300 dpi PNG."""
    from PIL import Image, ImageDraw, ImageFont

    width, height, dpi = 1800, 500, 300.0
    font = ImageFont.load_default(size=56)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(_CONTRACT_TEXT):
        draw.text((80, 80 + 140 * i), text, font=font, fill=0)
    path = Path(folder) / "vision-contract.png"
    img.save(path, format="PNG", dpi=(dpi, dpi))
    return PageImage(index=0, path=path, width=width, height=height, dpi=dpi,
                     cropbox=(0.0, 0.0, width * 72.0 / dpi, height * 72.0 / dpi))


ENGINE = VisionEngine()
