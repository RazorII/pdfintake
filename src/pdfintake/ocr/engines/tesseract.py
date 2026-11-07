"""Tesseract, in process through tesserocr: a reader for every pack, and the OSD session.

Reading. One ``PyTessBaseAPI`` per session, opened for one language string (``eng``, ``deu+eng``)
and one page segmentation mode, and reused page after page; the pipeline keeps one session per
thread and language string (``session_per_thread``). A read is ``SetImage`` of the rendered PNG,
``SetSourceResolution`` with the render's dpi (the PNG's own pHYs is never relied on), then
``Recognize(timeout=ms)``. Running out of time is the ``timeout`` status, never an exception.

Line text. Tesseract's TEXTLINE text puts a space between Thai glyph clusters whatever
``preserve_interword_spaces`` says, while its page-level text keeps Thai unspaced and keeps the
real gaps. So each line's text comes from the page-level ``GetUTF8Text()``, paired in order with
the ``RIL.TEXTLINE`` iterator, which gives the line's box and its word confidences
(``pair_lines``). Text is passed on raw: normalisers (sara am, NFC) run in the pipeline.

Boxes are quads in pixels of the shared render (origin top-left, y down, clockwise from the
top-left corner). A line's confidence is the mean of its word confidences / 100.

Models. A use's models map asset keys to verified ``<lang>.traineddata`` files. Tesseract loads
every language of a use from one folder, so when the files sit in different folders (a pack's
model beside the base pack's ``eng``) the store's shared ``tessdata`` folder is used, and only
when it holds the same files. ``availability`` loads each model once (cached by file identity)
so a truncated, empty or HTML-body model is reported ``broken``, naming the file: Tesseract's
own message ("Failed to init API") names neither the file nor the reason. When a use loads from
the shared folder, one model is also loaded from there: its bytes are the verified ones, so that
load tests only whether Tesseract can open the folder, which a model's own folder doesn't prove
(on Windows, a non-ASCII cache path the volume gives no 8.3 short name). A wrong-language model
under the right name loads silently; the store's hash check and the pack canary catch that.

Only verified files. ``availability``, ``open`` and ``open_osd`` must be given model paths whose
size and SHA-256 the store (or the kit, for a draft manifest) has just checked against the pin.
The engine can't re-check them (``ModelPaths`` carries no pins), and a same-size model with a
valid header and a corrupted body makes Tesseract hit an internal assert that aborts the whole
process: nothing in process can catch that, so the hash check is the only defence.

Images. A page reaches Tesseract as 8-bit grey or RGB. 16-bit and 32-bit samples are scaled to
8 bits, never clipped; a side over 32767 px (Tesseract's limit) is refused by name.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # before tesserocr loads: one thread per API, workers give the parallelism

try:
    import tesserocr as _tesserocr
    from tesserocr import PSM, RIL, PyTessBaseAPI
except (ImportError, OSError) as _exc:          # no tesserocr wheel on this platform, or a broken install
    _tesserocr = None
    PSM = RIL = PyTessBaseAPI = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR: str | None = f"{type(_exc).__name__}: {_exc}"
else:
    _IMPORT_ERROR = None

import PIL.Image as _PILImage

from ...errors import EngineFailure, PackError, SetupError
from ...params import Param, ParamTable, ValidationContext
from ...protocols import OsdSession
from ...types import EngineRead, Line, OsdResult, PageImage, Quad, Role
from . import Availability, ContractCase, ModelPaths

if TYPE_CHECKING:
    import PIL.Image

    from ...langpacks.manifest import Asset

__all__ = ["ENGINE", "IterLine", "TessOsd", "TesseractEngine", "open_osd", "pair_lines"]

ENGINE_NAME = "tesseract"
# Every supported platform has a tesserocr 2.12.0 wheel; win-arm64 has none (platforms.PLATFORMS).
PLATFORM_KEYS = frozenset({"darwin-arm64", "win-amd64", "linux-x86_64", "linux-aarch64"})
PSM_CHOICES = (3, 4, 6)          # auto, single column, single block
DEFAULT_TIMEOUT_S = 300.0

# One language component is a traineddata stem (eng, tha, chi_sim, aze_cyrl, Thai); up to four joined by +.
_LANG_PART = r"[A-Za-z][A-Za-z0-9]{1,15}(?:_[A-Za-z0-9]{1,8}){0,2}"
LANG_PATTERN = rf"{_LANG_PART}(?:\+{_LANG_PART}){{0,3}}"
NOT_READERS = frozenset({"osd"})   # orientation and script only; it can't read text

_THAI_CHAR = re.compile("[\u0e00-\u0e7f]")   # the Thai block: TEXTLINE puts a space between these clusters
_CONTRACT_LINES = ("The example invoice lists seven sample items",
                   "and a total of 4521 units for the test page.")
_NO_CONTEXT = ValidationContext(frozenset(), frozenset(), {}, frozenset(), "tesseract.params")


# --------------------------------------------------------------------------- line pairing

@dataclass(frozen=True)
class IterLine:
    """One ``RIL.TEXTLINE`` of the result iterator: its word texts, its box and its confidence (0..1)."""

    words: tuple[str, ...]
    box: Quad
    conf: float


def pair_lines(page_text: str, iter_lines: Sequence[IterLine]) -> list[Line]:
    """Lines whose text comes from the page-level text and whose box and confidence come from the iterator.

    The k-th iterator line that has any word text pairs with the k-th non-empty line of
    ``page_text``. That page line is taken (whitespace runs collapsed to one space, ends stripped)
    when it equals the iterator's words with all whitespace removed. Otherwise the iterator's
    words are joined: with no space between two Thai clusters, with one space anywhere else.
    """
    page = [ln for ln in page_text.splitlines() if ln.strip()]
    out: list[Line] = []
    k = 0
    for il in iter_lines:
        words = [_squeeze(w) for w in il.words]
        words = [w for w in words if w]
        if not words:
            continue
        text: str | None = None
        if k < len(page) and _squeeze(page[k]) == "".join(words):
            text = " ".join(page[k].split())
        if text is None:
            text = _join_words(words)
        out.append(Line(text, il.conf, il.box))
        k += 1
    return out


def _squeeze(text: str) -> str:
    return "".join(text.split())


def _join_words(words: Sequence[str]) -> str:
    joined = ""
    for w in words:
        if not joined:
            joined = w
        elif _THAI_CHAR.match(joined[-1]) and _THAI_CHAR.match(w[0]):
            joined += w
        else:
            joined += " " + w
    return joined


# --------------------------------------------------------------------------- models and folders

@dataclass(frozen=True)
class _Plan:
    state: str                        # "available" or an Availability failure state
    detail: str
    tessdata: Path | None = None      # the one folder Tesseract loads every language from
    files: tuple[tuple[str, Path], ...] = ()   # (language, verified model file)


def _components(lang: str) -> list[str]:
    return [c for c in lang.split("+") if c]


def _plan(lang: str, models: ModelPaths) -> _Plan:
    comps = _components(lang)
    if not comps:
        return _Plan("unsupported_language", "no language given")
    for c in comps:
        if c in NOT_READERS:
            return _Plan("unsupported_language", f"{c} is Tesseract's orientation and script model; it can't read text")
    by_name: dict[str, tuple[str, Path]] = {}
    for key in sorted(models):
        path = Path(models[key])
        by_name.setdefault(path.name, (key, path))
    files: list[tuple[str, Path]] = []
    for c in comps:
        name = f"{c}.traineddata"
        hit = by_name.get(name)
        if hit is None:
            return _Plan("missing_model", f"no {name} among this use's models (language {c})")
        key, path = hit
        if not path.is_file():
            return _Plan("missing_model", f"model {key} ({name}) is missing: {path}")
        files.append((c, path))
    parents = {os.path.normcase(os.path.abspath(p.parent)) for _, p in files}
    if len(parents) == 1:
        return _Plan("available", "", files[0][1].parent, tuple(files))
    try:
        shared = Path(_store_tessdata_dir())
    except Exception as exc:  # the store isn't there or can't give its folder
        return _Plan("broken", f"the models for {lang} sit in different folders and the shared tessdata folder "
                               f"isn't available ({type(exc).__name__}: {exc})")
    for c, path in files:
        there = shared / path.name
        if not there.is_file() or not _same_content(there, path):
            return _Plan("broken", f"the shared tessdata folder {shared} doesn't hold the verified {path.name} ({path})")
    return _Plan("available", "", shared, tuple(files))


def _store_tessdata_dir() -> Path:
    """The store's shared tessdata folder (hard links or copies of every installed traineddata)."""
    from ...langpacks import store

    return store.tessdata_dir()


_cache_lock = threading.Lock()
_model_checks: dict[tuple[Any, ...], str | None] = {}
_hashes: dict[tuple[Any, ...], str] = {}


def _stat_key(path: Path) -> tuple[Any, ...]:
    # ctime too: on POSIX an in-place rewrite with its mtime put back still changes ctime, so it misses
    # the caches. Not on Windows, where st_ctime is the creation time (see _stat_follows_writes).
    st = path.stat()
    return (os.path.normcase(os.path.realpath(path)), st.st_size, st.st_mtime_ns, st.st_ctime_ns,
            st.st_ino, st.st_dev)


def _stat_follows_writes() -> bool:
    """Whether every write to a file changes its ``_stat_key``, so a key can stand for its bytes.

    POSIX: yes, a write always moves ctime and nothing can set it back. Windows: no, Python's
    ``st_ctime`` there is the creation time, so a same-size rewrite in place with its mtime put back
    keeps the whole key (the Windows CI jobs, on 3.11 and 3.14, missed such a rewrite through the cache).
    """
    return sys.platform != "win32"


def _sha256(path: Path) -> str:
    """The file's SHA-256. Cached by ``_stat_key`` where that follows every write; read in full each
    time on Windows, because this hash is what keeps a changed shared copy away from Tesseract."""
    key = _stat_key(path) if _stat_follows_writes() else None
    if key is not None:
        with _cache_lock:
            if key in _hashes:
                return _hashes[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if key is not None:
        with _cache_lock:
            _hashes[key] = digest
    return digest


def _same_content(a: Path, b: Path) -> bool:
    try:
        if os.path.samefile(a, b):
            return True
        if a.stat().st_size != b.stat().st_size:
            return False
        return _sha256(a) == _sha256(b)
    except OSError:
        return False


def _model_problem(path: Path) -> str | None:
    """Why Tesseract can't load this traineddata file, or None when it loads. Cached per file identity."""
    try:
        key = _stat_key(path)
    except OSError as exc:
        return f"can't be read ({exc.strerror or exc})"
    with _cache_lock:
        if key in _model_checks:
            return _model_checks[key]
    problem = _check_model(path, key[1])
    with _cache_lock:
        _model_checks[key] = problem
    return problem


def _check_model(path: Path, size: int) -> str | None:
    if size == 0:
        return "the file is empty"
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEADER_READ)
    except OSError as exc:
        return f"can't be read ({exc.strerror or exc})"
    text = head[3:] if head.startswith(b"\xef\xbb\xbf") else head
    if text.lstrip()[:1] == b"<":
        return "the file is a web page (HTML), not a model: a failed download"
    if PyTessBaseAPI is None:
        return None
    try:
        api = PyTessBaseAPI(path=_native_dir(path.parent), lang=path.name[: -len(".traineddata")])
    except RuntimeError:
        folder_problem = _folder_problem(path.parent)
        if folder_problem:
            return f"Tesseract can't open its folder {path.parent}: {folder_problem}"
        if not _header_fits(head, size):
            return "Tesseract can't load it: the file is truncated or isn't a traineddata model"
        return (f"Tesseract can't load it from {path.parent}: its header is intact, so either the file "
                f"is damaged inside or Tesseract can't open that folder")
    api.End()
    return None


_HEADER_READ = 4 + 8 * 64
_MAX_ENTRIES = 64        # Tesseract 5 writes 24 component offsets (11 in osd); anything far beyond isn't a model


def _header_fits(head: bytes, size: int) -> bool:
    """Whether the traineddata header (an int32 count, then that many int64 offsets, -1 for an absent
    component) parses and every component it names starts inside the file, in ascending order."""
    if len(head) < 4:
        return False
    n = int.from_bytes(head[:4], "little", signed=True)
    end = 4 + 8 * n
    if not 0 < n <= _MAX_ENTRIES or len(head) < end:
        return False
    offsets = [int.from_bytes(head[4 + 8 * i: 12 + 8 * i], "little", signed=True) for i in range(n)]
    present = [o for o in offsets if o != -1]
    if not present or any(o < end or o >= size for o in present):
        return False
    return all(a < b for a, b in zip(present, present[1:]))


def _first_broken(files: Sequence[tuple[str, Path]]) -> tuple[Path, str] | None:
    for _, path in files:
        problem = _model_problem(path)
        if problem:
            return path, problem
    return None


def _load_problem(plan: _Plan) -> tuple[Path, str] | None:
    """The first file of an available ``plan`` Tesseract can't load where it will open it, and why;
    None when every one loads.

    Each verified file is loaded from its own folder first, so a damaged model is named by its own
    path. A plan on the shared folder then loads its first model from there too: the copy holds the
    verified bytes (``_plan`` compared them), so that load fails only when Tesseract can't open the
    folder itself, and ``_check_model`` then names the folder's problem (on Windows, a non-ASCII path
    the volume gives no 8.3 short name). Without it the models' own folders load, the use is reported
    available, and every read of it fails with Tesseract's bare "Failed to init API".
    """
    broken = _first_broken(plan.files)
    if broken is not None or plan.tessdata is None or not plan.files:
        return broken
    first = plan.files[0][1]
    there = plan.tessdata / first.name
    if there == first:                                   # the models' own folder: loaded just now
        return None
    problem = _model_problem(there)
    return (there, problem) if problem else None


def _native_dir(folder: Path) -> str:
    """The folder as Tesseract wants it: an absolute path ending in a separator.

    On Windows a non-ASCII folder is given as its 8.3 short path when the volume has one, because
    Tesseract opens model files through the ANSI file API (DESIGN §9.3).
    """
    s = os.path.abspath(str(folder))
    if sys.platform == "win32" and not s.isascii():
        s = _short_path(s) or s
    seps = (os.sep,) if os.altsep is None else (os.sep, os.altsep)   # a backslash is a name character on POSIX
    return s if s.endswith(seps) else s + os.sep


def _short_path(s: str) -> str | None:
    """The Windows 8.3 short form of ``s``, or None when the volume gives it none."""
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, len(buf))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    if 0 < n < len(buf) and buf.value.isascii():
        return buf.value
    return None


def _folder_problem(folder: Path) -> str | None:
    """A known reason Tesseract can't open this folder, asked after a failed load and before the model
    is blamed; None when there's none to name."""
    s = os.path.abspath(str(folder))
    if sys.platform == "win32" and not s.isascii() and _short_path(s) is None:
        return ("its path isn't ASCII and the volume gives it no 8.3 short name, and Tesseract opens "
                "model files through the ANSI file API (move the cache to an ASCII path with PDFINTAKE_CACHE)")
    return None


# --------------------------------------------------------------------------- images

def _tess_image(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """An 8-bit grey or RGB image for SetImage; transparency is laid on white."""
    from PIL import Image

    if img.mode in ("L", "RGB"):
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    if img.mode == "1":
        return img.convert("L")
    if img.mode.startswith("I;16"):
        return img.convert("I").point(lambda v: v * (255 / 65535)).convert("L")
    if img.mode in ("I", "F"):
        # Pillow's convert("L") clips; 8-bit values pass through, wider ones are scaled to 0..255
        top = img.getextrema()[1]
        if top > 255:
            scale = 255 / 65535 if top <= 65535 else 255 / float(top)
            img = img.point(lambda v: v * scale)
        return img.convert("L")
    return img.convert("RGB")


MIN_DPI, MAX_DPI = 1, 10000
TESS_MAX_SIDE = 32767            # Tesseract refuses an image wider or taller than this


def _dpi(value: float) -> tuple[int, bool]:
    """The dpi handed to SetSourceResolution, and whether it differs from the render's own (substituted)."""
    try:
        f = float(value)
        d = int(round(f))
    except (TypeError, ValueError, OverflowError):
        return 300, True
    clamped = min(max(d, MIN_DPI), MAX_DPI)
    return clamped, clamped != f


def _clear(api: Any) -> None:
    """Drop the image and results after a failure, so the next page starts clean."""
    try:
        api.Clear()
    except Exception:  # pragma: no cover - the failure being reported matters more
        pass


def _quad(left: float, top: float, right: float, bottom: float, width: int, height: int) -> Quad:
    x0 = float(min(max(left, 0), width))
    x1 = float(min(max(right, 0), width))
    y0 = float(min(max(top, 0), height))
    y1 = float(min(max(bottom, 0), height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _iter_lines(api: Any, width: int, height: int) -> tuple[list[IterLine], list[int], list[float]]:
    """The TEXTLINEs of the last recognition: lines with any word text, each line's block, every word confidence."""
    ri = api.GetIterator()
    lines: list[IterLine] = []
    blocks: list[int] = []
    all_confs: list[float] = []
    if ri is None:
        return lines, blocks, all_confs
    ri.Begin()
    if ri.Empty(RIL.WORD):
        return lines, blocks, all_confs
    block = -1
    words: list[str] = []
    confs: list[float] = []
    boxes: list[tuple[int, int, int, int]] = []
    line_box: tuple[int, int, int, int] | None = None
    line_block = 0

    def flush() -> None:
        if not words:
            return
        box = line_box
        if box is None and boxes:
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        if box is None:
            box = (0, 0, 0, 0)
        conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
        lines.append(IterLine(tuple(words), _quad(*box, width, height), round(min(max(conf, 0.0), 1.0), 4)))
        blocks.append(line_block)

    while True:
        if ri.IsAtBeginningOf(RIL.BLOCK):
            block += 1
        if ri.IsAtBeginningOf(RIL.TEXTLINE):
            flush()
            words, confs, boxes = [], [], []
            line_box = ri.BoundingBox(RIL.TEXTLINE)
            line_block = max(block, 0)
        text = ri.GetUTF8Text(RIL.WORD)
        if text and text.strip():
            words.append(text)
            conf = float(ri.Confidence(RIL.WORD))
            if conf >= 0:
                confs.append(conf)
                all_confs.append(conf)
            wb = ri.BoundingBox(RIL.WORD)
            if wb is not None:
                boxes.append(wb)
        if not ri.Next(RIL.WORD):
            break
    flush()
    return lines, blocks, all_confs


def _tess_version() -> str | None:
    if PyTessBaseAPI is None:
        return None
    try:
        return str(PyTessBaseAPI.Version())
    except Exception:  # pragma: no cover - a broken native library
        return None


# --------------------------------------------------------------------------- the session

class _TessSession:
    """One Tesseract API for one language string; not shared across threads (``session_per_thread``)."""

    def __init__(self, use_id: str, role: Role, params: Mapping[str, Any], models: ModelPaths) -> None:
        self.use_id = use_id
        self.role = role
        self.lang: str = params["lang"]
        self.psm = int(params.get("psm", 3))
        self.limit_s = float(params.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.spec = f"{ENGINE_NAME}:{self.lang}:psm{self.psm}"
        self._lock = threading.Lock()
        self._api: Any = None
        self._version = _tess_version()
        if PyTessBaseAPI is None:
            raise EngineFailure("engine_unavailable", f"Tesseract can't be loaded here: {_IMPORT_ERROR}",
                                engine=ENGINE_NAME, use_id=use_id)
        plan = _plan(self.lang, models)
        if plan.state != "available" or plan.tessdata is None:
            raise EngineFailure("engine_unavailable", f"Tesseract can't read {self.lang}: {plan.detail}",
                                engine=ENGINE_NAME, use_id=use_id, state=plan.state)
        try:
            self._api = PyTessBaseAPI(path=_native_dir(plan.tessdata), lang=self.lang, psm=self.psm)
        except RuntimeError as exc:
            broken = _load_problem(plan)
            if broken is not None:
                path, problem = broken
                raise EngineFailure("model_broken", f"Tesseract model {path}: {problem}",
                                    engine=ENGINE_NAME, use_id=use_id, file=str(path)) from None
            raise EngineFailure("engine_init_failed",
                                f"Tesseract couldn't start for {self.lang} with the models in {plan.tessdata}: {exc}",
                                engine=ENGINE_NAME, use_id=use_id) from None

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        start = time.perf_counter()
        dpi, substituted = _dpi(page.dpi)
        meta: dict[str, Any] = {"version": self._version, "dpi": dpi, "mean_conf": None, "line_blocks": []}
        if substituted:
            meta["dpi_substituted"] = True
        with self._lock:
            api = self._api
            if api is None:
                raise EngineFailure("session_closed", f"the Tesseract session {self.use_id!r} is closed",
                                    engine=ENGINE_NAME, use_id=self.use_id)
            try:
                limit = min(float(timeout_s), self.limit_s)
            except (TypeError, ValueError):
                limit = self.limit_s
            if not limit > 0:
                return self._result("timeout", [], start, meta, "no time left for the read")
            try:
                with page.open() as img:
                    w, h = img.size
                    if w > TESS_MAX_SIDE or h > TESS_MAX_SIDE:
                        raise EngineFailure("page_too_large",
                                            f"page {page.index}'s render is {w}x{h} px; Tesseract takes at most "
                                            f"{TESS_MAX_SIDE} px on each side",
                                            engine=ENGINE_NAME, use_id=self.use_id, path=str(page.path))
                    api.SetImage(_tess_image(img))
            except RuntimeError as exc:
                raise EngineFailure("engine_failed", f"Tesseract couldn't take page {page.index}: {exc}",
                                    engine=ENGINE_NAME, use_id=self.use_id) from None
            except _PILImage.DecompressionBombError as exc:
                raise EngineFailure("page_unreadable",
                                    f"page {page.index}'s render {page.path} is refused as too many pixels: {exc}",
                                    engine=ENGINE_NAME, use_id=self.use_id, path=str(page.path)) from None
            except (OSError, ValueError, OverflowError) as exc:
                raise EngineFailure("page_unreadable", f"page {page.index}'s render can't be read: {exc}",
                                    engine=ENGINE_NAME, use_id=self.use_id, path=str(page.path)) from None
            try:
                api.SetSourceResolution(dpi)
                api.ClearAdaptiveClassifier()        # each page is read as if it were the first
                remaining = limit - (time.perf_counter() - start)
                if remaining <= 0:
                    api.Clear()
                    return self._result("timeout", [], start, meta, f"timed out after {limit:g} s")
                ms = max(1, int(remaining * 1000))
                began = time.perf_counter()
                ok = api.Recognize(timeout=ms)
                if not ok:
                    ran_ms = (time.perf_counter() - began) * 1000.0
                    api.Clear()
                    if ran_ms >= ms * 0.9:
                        return self._result("timeout", [], start, meta, f"timed out after {limit:g} s")
                    raise EngineFailure("engine_failed", f"Tesseract couldn't recognise page {page.index}",
                                        engine=ENGINE_NAME, use_id=self.use_id)
                page_text = api.GetUTF8Text() or ""
                iter_lines, blocks, confs = _iter_lines(api, page.width, page.height)
                api.Clear()
            except (RuntimeError, OverflowError) as exc:
                _clear(api)
                raise EngineFailure("engine_failed", f"Tesseract failed on page {page.index}: {exc}",
                                    engine=ENGINE_NAME, use_id=self.use_id) from None
        lines = pair_lines(page_text, iter_lines)
        kept = [b for il, b in zip(iter_lines, blocks) if any(w.strip() for w in il.words)]
        meta["line_blocks"] = kept
        meta["words"] = len(confs)
        if confs:
            meta["mean_conf"] = round(sum(confs) / len(confs) / 100.0, 4)
        return self._result("ok" if lines else "empty", lines, start, meta, None)

    def _result(self, status: str, lines: list[Line], start: float, meta: dict[str, Any], error: str | None) -> EngineRead:
        return EngineRead(self.use_id, ENGINE_NAME, self.spec, self.role, status, lines,  # type: ignore[arg-type]
                          time.perf_counter() - start, error, meta)

    def close(self) -> None:
        with self._lock:
            api, self._api = self._api, None
        if api is not None:
            api.End()


# --------------------------------------------------------------------------- OSD

class TessOsd(OsdSession):
    """Tesseract's orientation and script detection (``osd.traineddata``, PSM.OSD_ONLY).

    ``detect`` gives tesserocr's own numbers: ``orient_deg`` is the clockwise rotation it reports
    for the input, ``script`` its script name ("Latin", "Thai", ...). None when the image holds too
    little text to judge. Calls are serialised, so one session may serve several threads.
    """

    def __init__(self, tessdata: Path) -> None:
        if PyTessBaseAPI is None:
            raise SetupError("engine_unavailable", f"Tesseract can't be loaded here: {_IMPORT_ERROR}", engine=ENGINE_NAME)
        self._lock = threading.Lock()
        try:
            self._api: Any = PyTessBaseAPI(path=_native_dir(tessdata), lang="osd", psm=PSM.OSD_ONLY)
        except RuntimeError as exc:
            raise SetupError("model_broken", f"Tesseract's OSD model in {tessdata} can't be loaded: {exc}",
                             engine=ENGINE_NAME, file=str(Path(tessdata) / "osd.traineddata")) from None

    def detect(self, image: "PIL.Image.Image", dpi: float) -> OsdResult | None:
        if image.width == 0 or image.height == 0:
            return None                                          # no pixels: too little text to judge
        if image.width > TESS_MAX_SIDE or image.height > TESS_MAX_SIDE:
            raise EngineFailure("page_too_large", f"the image is {image.width}x{image.height} px; Tesseract "
                                f"takes at most {TESS_MAX_SIDE} px on each side", engine=ENGINE_NAME)
        with self._lock:
            api = self._api
            if api is None:
                raise EngineFailure("session_closed", "the OSD session is closed", engine=ENGINE_NAME)
            try:
                api.SetImage(_tess_image(image))
                api.SetSourceResolution(_dpi(dpi)[0])
                found = api.DetectOrientationScript()
                api.Clear()
            except (RuntimeError, ValueError, OverflowError, OSError) as exc:
                _clear(api)
                raise EngineFailure("osd_failed", f"Tesseract OSD failed: {type(exc).__name__}: {exc}",
                                    engine=ENGINE_NAME) from None
        if not found:
            return None
        script = found.get("script_name") or None
        return OsdResult(script=str(script) if script else None, script_conf=float(found.get("script_conf", 0.0)),
                         orient_deg=int(found.get("orient_deg", 0)), orient_conf=float(found.get("orient_conf", 0.0)))

    def close(self) -> None:
        with self._lock:
            api, self._api = self._api, None
        if api is not None:
            api.End()


def open_osd(models: ModelPaths) -> OsdSession:
    """An OSD session on the ``osd.traineddata`` among ``models`` (any key; the base pack's is ``osd``).

    SetupError naming the file when it's missing or Tesseract can't load it.
    """
    if PyTessBaseAPI is None:
        raise SetupError("engine_unavailable", f"Tesseract can't be loaded here: {_IMPORT_ERROR}", engine=ENGINE_NAME)
    osd: Path | None = None
    for key in sorted(models):
        if Path(models[key]).name == "osd.traineddata":
            osd = Path(models[key])
            break
    if osd is None:
        raise SetupError("missing_model", "Tesseract's orientation model (osd.traineddata) isn't among the models given",
                         engine=ENGINE_NAME, model="osd")
    if not osd.is_file():
        raise SetupError("missing_model", f"Tesseract's orientation model is missing: {osd}",
                         engine=ENGINE_NAME, model="osd", file=str(osd))
    problem = _model_problem(osd)
    if problem:
        raise SetupError("model_broken", f"Tesseract model {osd}: {problem}", engine=ENGINE_NAME, file=str(osd))
    return TessOsd(osd.parent)


# --------------------------------------------------------------------------- the engine

class TesseractEngine:
    """The registered ``tesseract`` engine (``ENGINE``)."""

    name: ClassVar[str] = ENGINE_NAME
    api: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("lang", "str", required=True, pattern=LANG_PATTERN, max_len=80),
        Param("psm", "int", default=3, choices=PSM_CHOICES),
        Param("timeout_s", "float", default=DEFAULT_TIMEOUT_S, min=10.0, max=600.0),
    )
    MODEL_KINDS: ClassVar[frozenset[str]] = frozenset({"traineddata"})
    platforms: ClassVar[frozenset[str]] = PLATFORM_KEYS
    session_per_thread: ClassVar[bool] = True

    def version(self) -> str | None:
        v = _tess_version()
        if v is None or _tesserocr is None:
            return None
        return f"{v} (tesserocr {_tesserocr.__version__})"

    def availability(self, params: Mapping[str, Any], models: ModelPaths, platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform",
                                f"Tesseract runs on {', '.join(sorted(self.platforms))}, not {platform_key}")
        if PyTessBaseAPI is None:
            return Availability(False, "broken", f"tesserocr can't be imported: {_IMPORT_ERROR}")
        try:
            checked = self.PARAMS.validate(params, _NO_CONTEXT)
        except PackError as exc:
            return Availability(False, "broken", str(exc))
        plan = _plan(checked["lang"], models)
        if plan.state != "available":
            return Availability(False, plan.state, plan.detail)  # type: ignore[arg-type]
        broken = _load_problem(plan)
        if broken is not None:
            path, problem = broken
            return Availability(False, "broken", f"Tesseract model {path}: {problem}")
        return Availability(True, "available", "")

    def open(self, use_id: str, role: Role, params: Mapping[str, Any], models: ModelPaths) -> _TessSession:
        return _TessSession(use_id, role, self.PARAMS.validate(params, _NO_CONTEXT), models)

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence["Asset"]) -> list[dict[str, Any]]:
        """For the kit: each Tesseract model of the pack alone, and with ``eng`` for a non-Latin script (as ``tha+eng``)."""
        stems: list[str] = []
        for a in assets:
            if getattr(a, "engine", None) != ENGINE_NAME or getattr(a, "kind", None) != "traineddata":
                continue
            file = str(getattr(a, "file", ""))
            if not file.endswith(".traineddata"):
                continue
            stem = file[: -len(".traineddata")]
            if stem in NOT_READERS or stem in stems or not re.fullmatch(_LANG_PART, stem):
                continue
            stems.append(stem)
        out: list[dict[str, Any]] = []
        for stem in sorted(stems):
            out.append({"lang": stem, "psm": 3})
            if script != "Latin" and stem != "eng":
                out.append({"lang": f"{stem}+eng", "psm": 3})
        return out

    def contract_case(self) -> ContractCase | None:
        """The base pack's ``eng`` on a synthetic English page; None when tesserocr or the base pack isn't here."""
        if PyTessBaseAPI is None:
            return None
        models = _contract_models()
        if not models:
            return None
        return ContractCase(
            params={"lang": "eng", "psm": 3},
            models=models,
            model_kinds={key: "traineddata" for key in models},
            page=_contract_page,
            unsupported_params={"lang": "osd", "psm": 3},
            timeout_s=120.0,
        )


def _contract_models() -> dict[str, Path] | None:
    try:
        from ...langpacks import store

        path = Path(store.base().asset_path("eng"))
    except Exception:  # no store yet, or no base pack installed: the contract suite skips
        return None
    return {"eng": path} if path.is_file() else None


def _contract_page(folder: Path) -> PageImage:
    """A made-up English page at 300 dpi, drawn with Pillow's built-in font."""
    from PIL import Image, ImageDraw, ImageFont

    width, height, dpi = 1400, 420, 300.0
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    for i, text in enumerate(_CONTRACT_LINES):
        draw.text((70, 90 + i * 110), text, font=font, fill=0)
    path = Path(folder) / "tesseract-contract.png"
    img.save(path, format="PNG", dpi=(dpi, dpi))
    return PageImage(index=0, path=path, width=width, height=height, dpi=dpi,
                     cropbox=(0.0, 0.0, width * 72.0 / dpi, height * 72.0 / dpi))


ENGINE = TesseractEngine()
