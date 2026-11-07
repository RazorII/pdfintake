"""Pack canaries: does each engine of a pack read the pack's own test image on this computer?

A hash proves a model file is the one the manifest pinned; it can't prove the pinned file reads the
language it claims (a wrong-language model under the right name loads silently), or that an engine
still reads it after an OS update (Vision changes with macOS and degrades silently: an unsupported
language comes back ``ok`` with junk). So every pack ships a canary: a PNG of a short line of its
language and the strings any working reader must find in it (``canary.must_find``).

``check`` reads the canary with every engine use in the pack's reader set for this platform and
reports, per use, which strings it found. A string counts as found when it appears in the read
after the pack's own normalisers, ignoring whitespace (engines space Thai glyph clusters and line
breaks differently); case and accents count. An engine that isn't on this platform, or doesn't
support the language here, isn't read and isn't a failure on its own: the pipeline degrades that
reader by itself. But a pack where no ``reader`` use read the canary and passed is unusable
(``store`` checks that, with ``skipped``). A canary string that is empty after normalisation would
be found in any read, so it fails every use. Any other failure (missing strings, a failed or
timed-out read, a broken model) makes the pack unusable for this run: ``store.ensure`` raises
PackError ``pack_broken``. Nothing is asked (DESIGN §12.3): the run passes the pack to
``langplan.resolve_unsure(..., broken=...)``, and its pages are read with the base pack, with the
warning ``pack_broken`` and an offer to read them again with that pack later. A base pack whose
canary fails is a SetupError for the run, since nothing can stand in for it.

A pass is cached in ``<cache>/canary/<key>.json``. The key covers the engine, its version, every
model's SHA-256, the canary's SHA-256 and strings, the use's parameters, the pack's normalisers,
the core version and the OS version, so a change to any of them reads the canary again. A failure
is never cached.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__, fsutil, platforms
from ..errors import PackError, PdfIntakeError
from ..types import PageImage

if TYPE_CHECKING:
    from .manifest import EngineUse
    from .store import InstalledPack

PACK_BROKEN = "pack_broken"          # the warning a failed canary becomes (DESIGN §3.2, §12.3)
CACHE_VERSION = 1
READ_TIMEOUT_S = 120.0
NOT_READ_STATES = ("not_on_platform", "unsupported_language")
NOT_READ_PREFIX = "not read: "       # the error of a use this platform skipped (ok=True, nothing read)
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


@dataclass
class CanaryResult:
    pack: str
    use_id: str
    engine: str
    ok: bool
    found: list[str]
    missing: list[str]
    seconds: float
    cached: bool
    error: str | None


def cache_key(engine: str, engine_version: str | None, model_sha256s: Sequence[str], os_version: str) -> str:
    """The canary cache key: ``<engine>-<40 hex>``, a file name on every OS.

    It changes with the engine, its version, any of the hashes (order doesn't matter) and the OS
    version. ``check`` passes the model hashes plus the canary's hash and a digest of the use's
    parameters, strings, normalisers and the core version, so a change to any of them re-reads.
    """
    payload = json.dumps({"v": CACHE_VERSION, "engine": engine, "engine_version": engine_version,
                          "hashes": sorted(str(s) for s in model_sha256s), "os": os_version},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(engine))[:40] or "engine"
    return f"{safe}-{hashlib.sha256(payload.encode('ascii')).hexdigest()[:40]}"


def skipped(result: CanaryResult) -> bool:
    """True for a use that wasn't read here (its engine isn't on this platform, or doesn't support the
    language): ``ok`` without a read. A cached pass was read when it was cached, so it isn't skipped."""
    return result.ok and not result.cached and (result.error or "").startswith(NOT_READ_PREFIX)


def check(pack: InstalledPack, base: InstalledPack, platform_key: str) -> list[CanaryResult]:
    """Read the pack's canary PNG with every engine use of its reader set for ``platform_key``.

    Returns one result per use, in reader-set order; [] for a pack without a canary. The canary and
    every model are hash-checked first (IntegrityError naming the file). Passes are cached.
    """
    m = pack.manifest
    if m.canary is None:
        return []
    from . import store

    png = store._verified_file(pack.root / m.canary.file, m.canary.sha256, m.canary.bytes)
    width, height = _png_size(png)
    dpi = float(m.canary.dpi)
    page = PageImage(index=0, path=png, width=width, height=height, dpi=dpi,
                     cropbox=(0.0, 0.0, width * 72.0 / dpi, height * 72.0 / dpi))
    os_version = _os_version(platform_key)
    return [_check_use(pack, base, use, page, platform_key, os_version)
            for use in m.readers_for(platform_key).engines]


# --------------------------------------------------------------------------- one engine use

def _check_use(pack: InstalledPack, base: InstalledPack, use: EngineUse, page: PageImage,
               platform_key: str, os_version: str) -> CanaryResult:
    from ..ocr import engines
    from . import store

    m = pack.manifest
    must = list(m.canary.must_find)  # type: ignore[union-attr]

    def result(ok: bool, found: list[str], missing: list[str], seconds: float, error: str | None,
               cached: bool = False) -> CanaryResult:
        return CanaryResult(m.id, use.id, use.engine, ok, found, missing, round(seconds, 3), cached, error)

    blank = [s for s in must if not _squash(_normalize_text(m.normalize, s))]
    if blank or not must:
        return result(False, [], must, 0.0, "the canary's must_find "
                      + ("is empty" if not must else f"holds {len(blank)} string(s) that are empty after normalising, "
                         "which any read would contain"))
    try:
        engine = engines.get(use.engine)
    except PdfIntakeError as exc:
        return result(False, [], must, 0.0, exc.message)
    models = store.model_paths(pack, base, use)            # every model hash-checked here
    version = _engine_version(engine)
    key = cache_key(use.engine, version, [*_model_shas(pack, base, use), m.canary.sha256,  # type: ignore[union-attr]
                                          _use_digest(pack, use)], os_version)
    hit = _read_cache(key, m.id, use.id, must)
    if hit is not None:
        return result(True, must, [], float(hit.get("seconds", 0.0)), None, cached=True)
    try:
        avail = engine.availability(use.params, models, platform_key)
    except Exception as exc:  # an engine bug is a failed canary, never a crash
        return result(False, [], must, 0.0, f"availability check failed: {type(exc).__name__}: {exc}")
    if not avail.ok:
        if avail.state in NOT_READ_STATES:
            return result(True, [], [], 0.0, f"{NOT_READ_PREFIX}{avail.state}: {avail.detail}".rstrip(": "))
        return result(False, [], must, 0.0, f"{avail.state}: {avail.detail}".rstrip(": "))
    start = time.perf_counter()
    session = None
    try:
        session = engine.open(use.id, use.role, use.params, models)
        read = session.read(page, timeout_s=_timeout(use))
    except Exception as exc:
        return result(False, [], must, time.perf_counter() - start, f"the read failed: {_message(exc)}")
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
    seconds = time.perf_counter() - start
    if read.status != "ok":
        why = f"the read ended with status {read.status}" + (f" ({read.error})" if read.error else "")
        return result(False, [], must, seconds, why)
    text = _squash("".join(line.text for line in _normalized(m.normalize, read.lines)))
    found = [s for s in must if _squash(_normalize_text(m.normalize, s)) in text]
    missing = [s for s in must if s not in found]
    if missing:
        return result(False, found, missing, seconds, f"didn't find {len(missing)} of the {len(must)} canary strings")
    _write_cache(key, {"v": CACHE_VERSION, "pack": m.id, "use_id": use.id, "engine": use.engine,
                       "engine_version": version, "os": os_version, "must_find": must, "ok": True,
                       "seconds": round(seconds, 3)})
    return result(True, found, [], seconds, None)


def _message(exc: BaseException) -> str:
    if isinstance(exc, PdfIntakeError):
        return exc.message
    return f"{type(exc).__name__}: {exc}"


def _engine_version(engine: Any) -> str | None:
    try:
        v = engine.version()
    except Exception:
        return None
    return None if v is None else str(v)


def _model_shas(pack: InstalledPack, base: InstalledPack, use: EngineUse) -> list[str]:
    out: list[str] = []
    for ref in use.models:
        owner, key = (base, ref[3:]) if ref.startswith("en:") else (pack, ref)
        for a in owner.manifest.models:
            if a.key == key:
                out.append(a.sha256)
                break
        else:
            raise PackError("unknown_asset", f"the {owner.manifest.id!r} pack has no model {key!r}",
                            pack=owner.manifest.id, key=key)
    return out


def _use_digest(pack: InstalledPack, use: EngineUse) -> str:
    from ..types import to_jsonable

    m = pack.manifest
    blob = json.dumps({"use": use.id, "role": use.role, "params": to_jsonable(dict(use.params)),
                       "models": list(use.models), "must_find": list(m.canary.must_find),  # type: ignore[union-attr]
                       "normalize": list(m.normalize), "core": __version__},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _timeout(use: EngineUse) -> float:
    value = use.params.get("timeout_s") if isinstance(use.params, Mapping) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return READ_TIMEOUT_S


# --------------------------------------------------------------------------- matching

def _normalizers() -> Mapping[str, Any]:
    from ..ocr import textfix

    return textfix.NORMALIZERS


def _normalized(names: Sequence[str], lines: Sequence[Any]) -> list[Any]:
    from ..ocr import textfix

    return textfix.apply_normalizers(names, lines)


def _normalize_text(names: Sequence[str], text: str) -> str:
    table = _normalizers()
    for name in names:
        text = table[name](text)
    return text


def _squash(text: str) -> str:
    """NFC, with every whitespace character removed."""
    return "".join(unicodedata.normalize("NFC", text).split())


# --------------------------------------------------------------------------- the cache

def _cache_path(key: str) -> Path:
    return fsutil.work_dir("canary", "") / f"{key}.json"


def _read_cache(key: str, pack: str, use_id: str, must: Sequence[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(_cache_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(data, dict) or data.get("v") != CACHE_VERSION or data.get("ok") is not True
            or data.get("pack") != pack or data.get("use_id") != use_id or data.get("must_find") != list(must)):
        return None
    return data


def _write_cache(key: str, data: Mapping[str, Any]) -> None:
    from . import store

    try:
        store._write_json_atomic(_cache_path(key), data)
    except OSError:
        pass                                           # a cache that can't be written only costs a re-read


# --------------------------------------------------------------------------- the canary image and the OS

def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != _PNG_SIG or head[12:16] != b"IHDR":
        raise PackError("canary_not_png", f"the canary {path} isn't a PNG", path=str(path))
    width, height = struct.unpack(">II", head[16:24])
    if not (0 < width <= 20000 and 0 < height <= 20000):
        raise PackError("canary_not_png", f"the canary {path} says it is {width}x{height} px", path=str(path))
    return width, height


def _os_version(platform_key: str) -> str:
    """The platform key plus the OS release where an engine changes with it (macOS for Vision)."""
    mac = platforms.macos_version()
    if mac is not None:
        return f"{platform_key}/macOS {mac[0]}.{mac[1]}"
    if sys.platform == "win32":
        v = sys.getwindowsversion()  # type: ignore[attr-defined]
        return f"{platform_key}/Windows {v.major}.{v.minor}.{v.build}"
    return platform_key

