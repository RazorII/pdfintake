"""The pack store: the English base pack that ships with the install, the packs fetched since, and
the checks every load runs.

**The chain of trust.** The core release pins ``index.json``; each index entry pins its pack wheel's
SHA-256 and size and its manifest's SHA-256; the manifest pins every model, font and canary file.
Every file is checked against its pin each time it is loaded, not only after a download:
``InstalledPack.asset_path`` hashes the file on every call.

**The base pack** is the installed ``pdfintake-data-en`` distribution, found through
``importlib.metadata`` and never imported. ``uv tool install`` doesn't check the pyproject's
``#sha256=`` pin (measured, DESIGN §6.2), so ``base()`` does the guarding: the version must equal the
index entry, the manifest must hash to the index's pin, the distribution's RECORD and its folder
must hold data only (``datawheel.dist_problems``), and every file must match the manifest.

**A fetch** (DESIGN §3.5) downloads the pinned wheel over HTTPS with the standard library: the CA
bundle is certifi's when it is installed, plus ``SSL_CERT_FILE``; the proxy comes from the
environment; a redirect is followed only to https on a host in ``index.ALLOWED_HOSTS``; the stream
is capped at the pinned size. Then the wheel's SHA-256, the manifest's SHA-256, the manifest itself
(``manifest.parse``) and every member (``datawheel.wheel_problems``: no code, no unlisted member, no
unsafe path, every size and hash) are checked before anything is extracted. The files go to
``packs/.tmp-<random>/`` and move into ``packs/<id>-<version>/`` with one ``os.replace``, so two
fetches at once end with one folder. The pack's canary runs before the folder is marked
``.verified``; ``ensure`` runs it again on every load (``canary.check`` caches a pass).

Store layout under ``platforms.cache_root()``::

    packs/<id>-<version>/manifest.json  models/<file>  fonts/<file>  canary-<id>.png  NOTICE-*  .verified
    tessdata/<file>.traineddata         hard links (copies where linking fails) of base and pack models
    canary/<key>.json                   cached canary passes (canary.py)

A pack is never imported or executed. Only this module writes into ``packs/`` and ``tessdata/``,
and the canary cache goes through ``_write_json_atomic`` here too.
"""

from __future__ import annotations

import errno
import hashlib
import http.client
import importlib.metadata
import json
import os
import re
import secrets
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import __version__, datawheel, integrity, platforms
from ..errors import IntegrityError, PackError, PdfIntakeError, SetupError, UsageError

if TYPE_CHECKING:
    from ..pdf.textlayer import FontSpec
    from .index import Index, IndexEntry
    from .manifest import EngineUse, Manifest

BASE_PREFIX = "en:"
TRAINEDDATA = "traineddata"
PACKS = "packs"
TESSDATA = "tessdata"
VERIFIED = ".verified"
MANIFEST_FILE = "manifest.json"
MAX_MANIFEST_BYTES = 1 << 20

ATTEMPTS = 3
BACKOFF_S = (1.0, 3.0)
TIMEOUT_S = 60.0
STALE_TMP_S = 86400.0
_CHUNK = 1 << 20

_PACK_ID_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]{2,8})?")
_DIR_RE = re.compile(r"(?P<id>[a-z]{2,3}(?:-[a-z0-9]{2,8})?)-(?P<version>(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*))")
_TESS_LOCK = threading.RLock()


# --------------------------------------------------------------------------- what callers get

@dataclass(frozen=True)
class InstalledPack:
    manifest: Manifest
    root: Path

    def asset_path(self, key: str) -> Path:
        """The verified file of this pack's model or font ``key`` (its SHA-256 is checked on every call).

        A Tesseract model is also linked into ``tessdata_dir()``, where Tesseract loads every language
        of one read from. IntegrityError naming the file when it doesn't match its pin; PackError for
        a key this pack doesn't have.
        """
        for a in self.manifest.models:
            if a.key == key:
                path = self.root / "models" / a.file
                _verify(path, a.sha256, a.bytes)
                if a.kind == TRAINEDDATA:
                    _link_tessdata(path, a.file, a.sha256, a.bytes, self.manifest.id, self.manifest.base)
                return path
        for f in self.manifest.fonts:
            if f.key == key:
                path = self.root / "fonts" / f.file
                _verify(path, f.sha256, f.bytes)
                return path
        raise PackError("unknown_asset", f"the {self.manifest.id!r} pack has no model or font {key!r}",
                        pack=self.manifest.id, key=key)

    def font_specs(self) -> list[FontSpec]:
        """This pack's layer fonts, verified, in manifest order (``priority`` 0 first)."""
        from ..pdf.textlayer import FontSpec

        return [FontSpec(key=f.key, path=self.asset_path(f.key), sha256=f.sha256, covers=tuple(f.covers), priority=i)
                for i, f in enumerate(self.manifest.fonts)]


@dataclass
class VerifyResult:
    pack: str
    ok: bool
    problems: list[str]


# --------------------------------------------------------------------------- seams (tests replace these)

def _index() -> Index:
    from . import index

    return index.load()


def _pack_hosts_text() -> str:
    """The hosts the language packs are downloaded from, for a fetch-failure message, read from
    ``index.ALLOWED_HOSTS`` (not the ``_allowed_hosts`` seam the tests widen), so a message always
    names where the packs really live. Packs are GitHub release assets (decided 2026-11-05); the
    "GitHub" label is derived from the hosts, so a host move stays confined to the index, the
    allowlist and the pyproject pin (Decision 9)."""
    from .index import ALLOWED_HOSTS

    hosts = " or ".join(ALLOWED_HOSTS)
    github = all(h.lower() == "github.com" or h.lower().endswith(".githubusercontent.com") for h in ALLOWED_HOSTS)
    return f"GitHub ({hosts})" if github and ALLOWED_HOSTS else hosts


def _attempts_text(n: int) -> str:
    """``3`` -> ``"three attempts"`` (decided 2026-11-05: messages say attempts, in words)."""
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    word = words[n] if 0 <= n < len(words) else str(n)
    return f"{word} attempt{'' if n == 1 else 's'}"


def _allowed_hosts() -> tuple[str, ...]:
    from .index import ALLOWED_HOSTS

    return tuple(h.lower() for h in ALLOWED_HOSTS)


def _parse_manifest(data: bytes, *, base: Manifest | None) -> Manifest:
    from .manifest import parse

    return parse(data, base=base)


def _ssl_context(extra_cafile: str | None = None) -> ssl.SSLContext:
    """Certifi's CA bundle when it is installed (else the system's), plus ``SSL_CERT_FILE`` when set.

    ``extra_cafile`` exists for tests, which trust a throwaway certificate this way rather than
    through an environment variable.
    """
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    env = os.environ.get("SSL_CERT_FILE")
    if env and os.path.isfile(env):
        ctx.load_verify_locations(cafile=env)
    if extra_cafile:
        ctx.load_verify_locations(cafile=extra_cafile)
    return ctx


def _port_allowed(host: str, port: int) -> bool:
    """Whether a download address may name ``port``: never (``index.redirect_problems``). Tests
    replace this to reach their loopback stub, which listens on a random port."""
    return False


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _canary_check(pack: InstalledPack, base_pack: InstalledPack, platform_key: str) -> list[Any]:
    from . import canary

    return canary.check(pack, base_pack, platform_key)


# --------------------------------------------------------------------------- public functions

def base() -> InstalledPack:
    """The installed English base pack, checked from top to bottom (SetupError naming the problem).

    The ``pdfintake-data-en`` distribution is located through ``importlib.metadata`` and never
    imported. Its version must equal the index entry; its ``manifest.json`` must hash to the index's
    pin; ``datawheel.dist_problems`` must find nothing in its RECORD (no ``.py``, ``.pth`` or native
    file, nothing outside the manifest's members), and its folder must hold nothing else; and every
    file must match the manifest's size and SHA-256 (IntegrityError naming the file).
    """
    return _load_base(_index().base(), deep=True)


def installed() -> dict[str, InstalledPack]:
    """Every usable pack by id: the base pack, and each fetched pack of this core's index that has
    passed its canary. A fetched pack that fails a quick check is left out (``ensure`` or ``verify``
    names the problem); a broken base pack raises SetupError."""
    idx = _index()
    b = _load_base(idx.base(), deep=True)
    out = {b.manifest.id: b}
    for entry in idx.entries:
        if entry.base:
            continue
        try:
            found = _open_dir(entry, b, deep=False)
        except PdfIntakeError:
            continue
        if found is not None and found[1]:
            out[entry.id] = found[0]
    return out


def locate(pack_id: str) -> InstalledPack | None:
    """The installed, verified pack ``pack_id`` at this core's index version, or None.

    Every file is hashed: a file tampered with after install raises IntegrityError naming it.
    """
    idx = _index()
    entry = idx.get(pack_id)
    if entry is None:
        return None
    if entry.base:
        return _load_base(entry, deep=True)
    b = _load_base(idx.base(), deep=True)
    found = _open_dir(entry, b, deep=True)
    if found is None or not found[1]:
        return None
    return found[0]


def ensure(entry: IndexEntry, *, allow_network: bool) -> InstalledPack:
    """The pack ``entry`` names, installed and checked, fetching it when ``allow_network`` allows.

    An installed pack is loaded with every hash checked, and its canary runs (a cached pass costs no
    read): a failure raises PackError ``pack_broken``, and the pack can't be used this run. A pack
    whose files no longer match their pins is fetched again when the network is allowed; otherwise
    the IntegrityError naming the file is raised. A pack that isn't installed and may not be fetched
    raises PackError ``pack_not_installed``.

    Nothing here asks, and neither does the run (DESIGN §12.3): when this raises for a pack other
    than the base pack (its canary failed, its download failed, a file no longer matches its pin),
    the run passes the pack to ``langplan.resolve_unsure(..., broken=...)``, and its pages are read
    with the base pack, with the warning ``pack_broken``. The base pack's canary runs here too
    (``base()`` itself runs none), but English has no alternative, so its ``pack_broken`` is a
    SetupError for the run.
    """
    if entry.base or entry.dist.kind == "installed":
        if not entry.base:
            raise PackError("not_fetchable", f"the index entry for {entry.id!r} ships with the install but isn't the base pack",
                            pack=entry.id)
        b = _load_base(entry, deep=True)
        _require_canary(b, b, entry)
        return b
    b = _load_base(_index().base(), deep=True)
    try:
        found = _open_dir(entry, b, deep=True)
    except IntegrityError:
        if not allow_network:
            raise
        return fetch(entry)
    if found is None:
        if not allow_network:
            raise PackError("pack_not_installed",
                            f"The {entry.title} language pack isn't installed, and downloads are off for this run.",
                            pack=entry.id, download_bytes=entry.dist.bytes)
        return fetch(entry)
    pack, verified = found
    _require_canary(pack, b, entry)
    if not verified:
        _mark_verified(pack.root, entry)
    return pack


def fetch(entry: IndexEntry) -> InstalledPack:
    """Download, check and install the pack ``entry`` names (DESIGN §3.5), then run its canary.

    Refused, with nothing installed: a URL or redirect that isn't https on an allowed host
    (PackError), a stream larger than the pinned size (PackError), a wheel or member whose SHA-256 or
    size differs from its pin (IntegrityError), a wheel holding code, an unlisted member or an unsafe
    path (PackError), and a Tesseract model whose file name another pack already uses with other
    contents (PackError ``tessdata_conflict``). A network failure gets three attempts with backoff,
    then raises SetupError naming the host it tried and the GitHub hosts the packs come from. A pack
    already installed and intact is not downloaded again; one whose files no longer match is
    replaced. A canary failure raises PackError ``pack_broken`` and leaves the files in place
    without ``.verified``.

    None of these failures makes the run ask anything (DESIGN §12.3): the run passes the pack to
    ``langplan.resolve_unsure(..., broken=...)``, and its pages are read with the base pack, with
    the warning ``pack_broken``.
    """
    if entry.base or entry.dist.kind != "fetch" or not entry.dist.url:
        raise PackError("not_fetchable", f"the {entry.id!r} pack isn't downloaded; it ships with the install",
                        pack=entry.id)
    b = _load_base(_index().base(), deep=True)
    root = _packs_root()
    dest = root / f"{entry.id}-{entry.version}"
    try:
        found = _open_dir(entry, b, deep=True)
    except IntegrityError:
        _discard(dest)
        found = None
    if found is None:
        problem = _url_problem(entry.dist.url, initial=True)
        if problem:
            raise PackError("url_refused", f"the {entry.id!r} pack's download address {problem}", pack=entry.id)
        _prune_tmp(root)
        tmp = root / f".tmp-{secrets.token_hex(8)}"
        _mkdir_private(tmp)
        try:
            wheel = tmp / "download.whl"
            _download(entry, wheel)
            staged = tmp / "pack"
            manifest = _extract(wheel, entry, b, staged)
            _tessdata_conflicts(manifest, b)
            try:
                os.replace(staged, dest)
            except OSError:
                if not dest.is_dir():
                    raise
                # another fetch placed the pack first; its copy is checked below like ours would be
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        found = _open_dir(entry, b, deep=True)
        if found is None:
            raise PackError("install_failed", f"the {entry.id!r} pack was downloaded but isn't in place at {dest}",
                            pack=entry.id, path=str(dest))
    pack, verified = found
    _require_canary(pack, b, entry)
    if not verified:
        _mark_verified(pack.root, entry)
    return pack


def verify(pack_id: str | None = None) -> list[VerifyResult]:
    """Check the base pack and every fetched pack (or only ``pack_id``) without changing anything."""
    idx = _index()
    if pack_id is not None:
        ids = [pack_id]
    else:
        ids = [e.id for e in idx.entries if e.base or (_packs_root() / f"{e.id}-{e.version}").is_dir()]
    results: list[VerifyResult] = []
    for pid in ids:
        entry = idx.get(pid)
        problems: list[str] = []
        if entry is None:
            problems.append(f"this version of pdfintake has no pack {pid!r}")
        elif entry.base:
            try:
                _load_base(entry, deep=True)
            except PdfIntakeError as exc:
                problems.extend(_problems_of(exc))
        else:
            try:
                b = _load_base(idx.base(), deep=True)
            except PdfIntakeError as exc:
                problems.append("the base pack can't be checked: " + exc.message)
            else:
                try:
                    found = _open_dir(entry, b, deep=True)
                except PdfIntakeError as exc:
                    problems.extend(_problems_of(exc))
                else:
                    if found is None:
                        problems.append(f"the {pid!r} pack isn't installed")
                    elif not found[1]:
                        problems.append(f"the {pid!r} pack is installed, but its canary hasn't passed on this computer")
        results.append(VerifyResult(pid, not problems, problems))
    return results


def remove(pack_id: str) -> bool:
    """Remove every installed version of a fetched pack, with its tessdata links. True if one was there."""
    if not isinstance(pack_id, str) or not _PACK_ID_RE.fullmatch(pack_id):
        raise UsageError("bad_pack_id", f"{pack_id!r} isn't a pack id", pack=str(pack_id))
    idx = _index()
    entry = idx.get(pack_id)
    if (entry is not None and entry.base) or pack_id == idx.base().id:
        raise UsageError("base_pack_kept", "The English base pack ships with pdfintake and can't be removed.",
                         pack=pack_id)
    root = _packs_root()
    removed = False
    for d in sorted(root.iterdir()):
        m = _DIR_RE.fullmatch(d.name)
        if m is None or m["id"] != pack_id or d.is_symlink() or not d.is_dir():
            continue
        _unlink_tessdata_of(d)
        _discard(d)
        removed = True
    return removed


def tessdata_dir() -> Path:
    """The shared folder Tesseract reads every language of one read from: ``<cache>/tessdata``.

    Each traineddata file is linked in when its pack's ``asset_path`` verifies it. On Windows a
    non-ASCII cache path is handed to Tesseract as its 8.3 short form by the Tesseract engine.
    """
    path = _cache_root() / TESSDATA
    _mkdir_private(path)
    return path


def model_paths(pack: InstalledPack, base: InstalledPack, use: EngineUse) -> dict[str, Path]:
    """The verified files of one engine use's models, keyed by the reference as the manifest writes it
    (``"tha"``, ``"en:eng"``); an ``en:`` reference resolves in the base pack.

    **Checked at the moment of the call** (decided 2026-11-05): every file's size and SHA-256 are
    compared with its pin on each call, through ``InstalledPack.asset_path``, and nothing is cached
    between calls, so a model swapped or corrupted after install (or after an earlier call) raises
    IntegrityError naming the file. The pipeline calls this immediately before ``Engine.open``, so an
    engine never loads a file that hasn't just been checked. A Tesseract model is also (re)linked into
    ``tessdata_dir()`` from the file just checked.
    """
    out: dict[str, Path] = {}
    for ref in use.models:
        if ref.startswith(BASE_PREFIX):
            out[ref] = base.asset_path(ref[len(BASE_PREFIX):])
        else:
            out[ref] = pack.asset_path(ref)
    return out


# --------------------------------------------------------------------------- the base pack

def _problems_of(exc: PdfIntakeError) -> list[str]:
    listed = exc.detail.get("problems")
    if isinstance(listed, list) and listed and all(isinstance(x, str) for x in listed):
        return list(listed)
    return [exc.message]


def _load_base(entry: IndexEntry, *, deep: bool) -> InstalledPack:
    name = entry.dist.distribution
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        raise SetupError("base_pack_missing",
                         f"The English data package {name} isn't installed; reinstall pdfintake.",
                         distribution=name) from None
    if dist.version != entry.version:
        raise SetupError("base_pack_version",
                         f"The installed {name} is version {dist.version}; this pdfintake needs {entry.version}. "
                         "Reinstall pdfintake.", distribution=name, installed=dist.version, expected=entry.version)
    files = dist.files
    if files is None:
        raise SetupError("base_pack_no_record", f"The installed {name} has no RECORD file list; reinstall pdfintake.",
                         distribution=name)
    # Only the RECORD entries whose file is still there, on every Python: 3.12 and later leave out the
    # ones that are gone, 3.11 lists them all, and there a deleted model got past the member check
    # below and was reported without the "reinstall" remedy.
    listed = [str(f) for f in files if Path(f.locate()).exists()]
    pkg = name.replace("-", "_")
    rel = f"{pkg}/{MANIFEST_FILE}"
    if rel not in listed:
        raise SetupError("base_pack_no_manifest", f"The installed {name} has no {rel}; reinstall pdfintake.",
                         distribution=name)
    mpath = Path(dist.locate_file(rel))
    raw = _read_capped(mpath, MAX_MANIFEST_BYTES)
    got = integrity.sha256_bytes(raw)
    if got != entry.manifest_sha256:
        raise IntegrityError("hash_mismatch",
                             f"{mpath} has SHA-256 {got}; this pdfintake's pack index pins {entry.manifest_sha256}.",
                             path=str(mpath), expected=entry.manifest_sha256, actual=got)
    data = _json(raw, mpath)
    present = set(listed)
    for member in sorted(datawheel._members(data)):           # listed leaves out files that are gone
        if f"{pkg}/{member}" not in present:
            path = Path(dist.locate_file(f"{pkg}/{member}"))
            raise IntegrityError("file_missing", f"{path} is missing; reinstall pdfintake.", path=str(path))
    problems = datawheel.dist_problems(listed, data)
    if problems:
        raise SetupError("base_pack_not_data_only",
                         f"The installed {name} isn't data only ({len(problems)} problem{'s' if len(problems) != 1 else ''}): "
                         + "; ".join(problems), distribution=name, problems=problems)
    manifest = _parse_manifest(raw, base=None)
    if manifest.id != entry.id or manifest.version != entry.version or not manifest.base:
        raise PackError("base_pack_mismatch", f"The installed {name} carries pack {manifest.id!r} {manifest.version}, "
                                              f"not the base pack {entry.id!r} {entry.version}.", distribution=name)
    root = mpath.parent
    extra = _unexpected_files(root, manifest, allow=())
    if extra:
        raise SetupError("base_pack_not_data_only",
                         f"The installed {name} holds files its RECORD and manifest don't list: " + ", ".join(extra)
                         + ". Reinstall pdfintake.", distribution=name, problems=extra)
    _check_files(root, manifest, deep=deep)
    return InstalledPack(manifest, root)


# Bookkeeping files an OS or a file manager drops into any folder it shows (never code, never read).
OS_METADATA_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def _unexpected_files(root: Path, manifest: Manifest, *, allow: tuple[str, ...]) -> list[str]:
    expected = {MANIFEST_FILE, *allow}
    expected |= {f"models/{a.file}" for a in manifest.models}
    expected |= {f"fonts/{f.file}" for f in manifest.fonts}
    if manifest.canary is not None:
        expected.add(manifest.canary.file)
    expected |= {a.notice for a in manifest.models if a.notice}
    if manifest.fonts:
        expected.add(datawheel.OFL_FILE)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        for name in dirnames:
            if (here / name).is_symlink():
                out.append(str(here / name))
        for name in filenames:
            rel = (here / name).relative_to(root).as_posix()
            if name in OS_METADATA_FILES and not (here / name).is_symlink():
                continue
            if rel not in expected or (here / name).is_symlink():
                out.append(str(here / name))
    return sorted(out)


def _check_files(root: Path, manifest: Manifest, *, deep: bool) -> None:
    """IntegrityError naming the first model, font or canary file that doesn't match its pin.

    ``deep=False`` compares sizes only; the hash is then checked when the file is used.
    """
    pins = [(root / "models" / a.file, a.sha256, a.bytes) for a in manifest.models]
    pins += [(root / "fonts" / f.file, f.sha256, f.bytes) for f in manifest.fonts]
    if manifest.canary is not None:
        c = manifest.canary
        pins.append((root / c.file, c.sha256, c.bytes))
    for path, sha, size in pins:
        if deep:
            _verify(path, sha, size)
            continue
        try:
            st = os.stat(path)
        except OSError:
            raise IntegrityError("file_missing", f"{path} is missing.", path=str(path)) from None
        if st.st_size != size:
            raise IntegrityError("size_mismatch", f"{path} is {st.st_size} bytes; it should be {size}.",
                                 path=str(path), expected=size, actual=st.st_size)


# --------------------------------------------------------------------------- fetched packs on disk

def _open_dir(entry: IndexEntry, b: InstalledPack, *, deep: bool) -> tuple[InstalledPack, bool] | None:
    """(pack, passed its canary) for the pack folder of ``entry``, or None when there is none.

    IntegrityError when the folder's manifest or files don't match their pins; PackError when the
    manifest no longer loads or a Tesseract model clashes with another pack's.
    """
    d = _packs_root() / f"{entry.id}-{entry.version}"
    if not d.is_dir():
        return None
    mpath = d / MANIFEST_FILE
    raw = _read_capped(mpath, MAX_MANIFEST_BYTES)
    got = integrity.sha256_bytes(raw)
    if got != entry.manifest_sha256:
        raise IntegrityError("hash_mismatch",
                             f"{mpath} has SHA-256 {got}; the pack index pins {entry.manifest_sha256}.",
                             path=str(mpath), expected=entry.manifest_sha256, actual=got)
    manifest = _parse_manifest(raw, base=b.manifest)
    if manifest.id != entry.id or manifest.version != entry.version:
        raise IntegrityError("manifest_mismatch", f"{mpath} describes {manifest.id!r} {manifest.version}, "
                                                  f"not {entry.id!r} {entry.version}.", path=str(mpath))
    _check_files(d, manifest, deep=deep)
    _tessdata_conflicts(manifest, b)
    return InstalledPack(manifest, d), _is_verified(d, entry)


def _is_verified(d: Path, entry: IndexEntry) -> bool:
    try:
        data = json.loads((d / VERIFIED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("manifest_sha256") == entry.manifest_sha256


def _mark_verified(d: Path, entry: IndexEntry) -> None:
    _write_json_atomic(d / VERIFIED, {"manifest_sha256": entry.manifest_sha256, "core": __version__,
                                      "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})


def _require_canary(pack: InstalledPack, b: InstalledPack, entry: IndexEntry) -> None:
    """PackError ``pack_broken`` unless the pack's canary passes on this computer.

    Every use that was read must pass, and at least one use with the role ``reader`` (any use, when
    the reader set has no ``reader``) must have read the canary and passed: a use skipped as
    ``not_on_platform`` or ``unsupported_language`` isn't a failure on its own, but a reader set in
    which nothing read would hand the run a pack no engine reads. A fetched pack must have a canary
    (DESIGN §3.4); the base pack is checked when its manifest carries one.
    """
    from . import canary

    platform_key = platforms.current().key
    if pack.manifest.canary is None:
        if pack.manifest.base:
            return
        raise PackError("pack_broken",
                        f"The {entry.title} language pack has no canary, so it can't be checked on this computer "
                        "and isn't used this run.", pack=entry.id, results=[])
    results = _canary_check(pack, b, platform_key)
    details = [{"use_id": r.use_id, "engine": r.engine, "ok": r.ok, "missing": len(r.missing), "error": r.error}
               for r in results]
    failed = [r for r in results if not r.ok]
    if failed:
        what = "; ".join(f"{r.use_id} ({r.engine}): {r.error or 'missing ' + ', '.join(r.missing)}" for r in failed)
        raise PackError("pack_broken",
                        f"The {entry.title} language pack failed its self-check on this computer, so it isn't used "
                        f"this run ({what}).", pack=entry.id, results=details)
    roles = {u.id: u.role for u in pack.manifest.readers_for(platform_key).engines}
    counted = {i for i, role in roles.items() if role == "reader"} or set(roles)
    if not any(r.ok and not canary.skipped(r) for r in results if r.use_id in counted):
        what = "; ".join(f"{r.use_id} ({r.engine}): {r.error}" for r in results) or "its reader set is empty"
        raise PackError("pack_broken",
                        f"No OCR engine on this computer could read the {entry.title} language pack's self-check, "
                        f"so it isn't used this run ({what}).", pack=entry.id, results=details)


def _discard(d: Path) -> None:
    """Move ``d`` out of the way in one rename, then delete it (readers never see half a folder)."""
    if not d.exists() and not d.is_symlink():
        return
    if d.is_symlink() or not d.is_dir():
        d.unlink(missing_ok=True)
        return
    trash = d.parent / f".trash-{secrets.token_hex(8)}"
    try:
        os.replace(d, trash)
    except FileNotFoundError:
        return
    except OSError:                                    # Windows: a file inside is open elsewhere
        shutil.rmtree(d, ignore_errors=True)
        return
    shutil.rmtree(trash, ignore_errors=True)


def _prune_tmp(root: Path) -> None:
    """Remove ``.tmp-*`` and ``.trash-*`` folders a killed process left behind more than a day ago."""
    cutoff = time.time() - STALE_TMP_S
    for d in root.iterdir():
        if not d.name.startswith((".tmp-", ".trash-")):
            continue
        try:
            if os.lstat(d).st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


# --------------------------------------------------------------------------- downloading

class _Retry(Exception):
    pass


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        problem = _url_problem(newurl, initial=False)
        if problem:
            raise PackError("redirect_refused", f"the download was redirected to an address that {problem}; "
                                                "nothing was fetched from it", host=urllib.parse.urlsplit(newurl).hostname)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _url_problem(url: Any, *, initial: bool) -> str | None:
    """Why ``url`` may not be fetched from, or None.

    The rules are ``index.redirect_problems`` (https, an allowed host, no user name, no port, no
    whitespace, control or non-ASCII character, no backslash; tests/test_store.py checks the two
    agree); the first address also carries no query, fragment, ``..`` or ``%``, as
    ``index.url_problems`` requires of every entry (a redirect may carry a query: GitHub signs its
    asset links). The host and port checks go through the ``_allowed_hosts`` and ``_port_allowed``
    seams, so the tests' loopback stub can be reached.
    """
    if not isinstance(url, str) or not url:
        return "is missing"
    if any(ord(c) <= 0x20 or ord(c) >= 0x7F for c in url):
        return "contains whitespace, a control character or a non-ASCII character"
    if "\\" in url:
        return "contains a backslash"
    try:
        u = urllib.parse.urlsplit(url)
        host = (u.hostname or "").lower()
        port = u.port
    except ValueError:
        return "isn't a valid URL"
    if u.scheme != "https":
        return "doesn't use https"
    if host not in _allowed_hosts():
        return f"is on {host or 'no host'}, which pdfintake doesn't download from"
    if u.username is not None or u.password is not None or "@" in u.netloc:
        return "carries a user name or password"
    if port is not None and not _port_allowed(host, port):
        return "names a port"
    if initial and (u.query or u.fragment or "?" in url or "#" in url):
        return "carries a query or a fragment"
    if initial and (".." in url or "%" in url):
        return "contains '..' or '%'"
    return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler(),
                                       urllib.request.HTTPSHandler(context=_ssl_context()),
                                       _RedirectGuard())


def _download(entry: IndexEntry, dest: Path) -> None:
    url = entry.dist.url or ""
    host = urllib.parse.urlsplit(url).hostname or "the pack host"
    last = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            _sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        try:
            _download_once(entry, url, dest)
            return
        except _Retry as exc:
            last = str(exc)
    raise SetupError("download_failed",
                     f"The {entry.title} language pack couldn't be downloaded from {host} after "
                     f"{_attempts_text(ATTEMPTS)} ({last}); the network setting may block downloads from "
                     f"{_pack_hosts_text()}.",
                     pack=entry.id, host=host, reason=last, attempts=ATTEMPTS)


def _download_once(entry: IndexEntry, url: str, dest: Path) -> None:
    size, sha = entry.dist.bytes, entry.dist.sha256
    host = urllib.parse.urlsplit(url).hostname or "the pack host"
    dest.unlink(missing_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": f"pdfintake/{__version__}", "Accept-Encoding": "identity"})
    try:
        resp = _opener().open(req, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        exc.close()                                   # the error body is never read
        if exc.code in (408, 425, 429) or exc.code >= 500:
            raise _Retry(f"HTTP {exc.code}") from None
        raise SetupError("download_refused",
                         f"The {entry.title} language pack couldn't be downloaded: {host} answered HTTP {exc.code}.",
                         pack=entry.id, host=host, status=exc.code) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise SetupError("download_tls",
                             f"The {entry.title} language pack couldn't be downloaded: {host}'s certificate isn't "
                             f"trusted ({reason.verify_message}); the network setting may intercept downloads from "
                             f"{_pack_hosts_text()}.",
                             pack=entry.id, host=host) from None
        raise _Retry(f"{type(reason).__name__ if isinstance(reason, BaseException) else 'URLError'}: {reason}") from None
    except (OSError, ValueError, http.client.HTTPException) as exc:
        # http.client.HTTPException: a garbage status line (BadStatusLine), an oversized header
        # (LineTooLong) from a proxy, a captive portal or a broken mirror
        raise _Retry(f"{type(exc).__name__}: {exc}") from None
    with resp:
        length = resp.headers.get("Content-Length")
        if length is not None and length.strip().isdigit() and int(length) > size:
            raise PackError("download_oversize", f"{host} offered {int(length)} bytes for the {entry.title} language pack; "
                                                 f"the pack index pins {size}. Nothing was installed.",
                            pack=entry.id, host=host, offered=int(length), expected=size)
        h = hashlib.sha256()
        n = 0
        try:
            with open(dest, "xb") as out:
                while True:
                    chunk = resp.read(min(_CHUNK, size + 1 - n))
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > size:
                        raise PackError("download_oversize",
                                        f"{host} sent more than the {size} bytes the pack index pins for the "
                                        f"{entry.title} language pack. Nothing was installed.",
                                        pack=entry.id, host=host, expected=size)
                    h.update(chunk)
                    out.write(chunk)
        except PackError:
            dest.unlink(missing_ok=True)
            raise
        except (OSError, http.client.HTTPException) as exc:
            # IncompleteRead: a chunked stream that breaks off, or bad chunk framing
            dest.unlink(missing_ok=True)
            raise _Retry(f"the download broke off ({type(exc).__name__}: {exc})") from None
    if n != size:
        dest.unlink(missing_ok=True)
        raise _Retry(f"got {n} of {size} bytes")
    digest = h.hexdigest()
    if digest != sha:
        dest.unlink(missing_ok=True)
        raise IntegrityError("wheel_hash_mismatch",
                             f"The {entry.title} language pack from {host} has SHA-256 {digest}; the pack index pins {sha}. "
                             "Nothing was installed.", pack=entry.id, host=host, expected=sha, actual=digest)


# --------------------------------------------------------------------------- checking and extracting a wheel

def _extract(wheel: Path, entry: IndexEntry, b: InstalledPack, staged: Path) -> Manifest:
    pkg = entry.dist.distribution.replace("-", "_")
    mname = f"{pkg}/{MANIFEST_FILE}"
    try:
        with zipfile.ZipFile(wheel) as zf:
            try:
                zi = zf.getinfo(mname)
            except KeyError:
                raise PackError("wheel_no_manifest", f"the {entry.id!r} pack's wheel has no {mname}",
                                pack=entry.id) from None
            raw = _read_member(zf, zi, MAX_MANIFEST_BYTES)
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError) as exc:
        # RuntimeError: an encrypted member; NotImplementedError: a compression zipfile can't read
        raise PackError("wheel_unreadable", f"the {entry.id!r} pack's wheel can't be read: {exc}", pack=entry.id) from None
    got = integrity.sha256_bytes(raw)
    if got != entry.manifest_sha256:
        raise IntegrityError("manifest_hash_mismatch",
                             f"the {entry.id!r} pack's manifest has SHA-256 {got}; the pack index pins "
                             f"{entry.manifest_sha256}", pack=entry.id, expected=entry.manifest_sha256, actual=got)
    manifest = _parse_manifest(raw, base=b.manifest)
    if manifest.id != entry.id or manifest.version != entry.version or manifest.base:
        raise PackError("manifest_mismatch", f"the wheel for {entry.id!r} {entry.version} carries pack "
                                             f"{manifest.id!r} {manifest.version}", pack=entry.id)
    data = _json(raw, Path(mname))
    if datawheel._dist_name(data) != entry.dist.distribution:
        raise PackError("manifest_mismatch", f"the {entry.id!r} pack is published as {entry.dist.distribution}, "
                                             f"but its manifest makes it {datawheel._dist_name(data)}", pack=entry.id)
    problems = datawheel._check_wheel(wheel, data)
    if problems:
        text = "; ".join(msg for _, msg in problems)
        cls = IntegrityError if any(code in ("size", "hash") for code, _ in problems) else PackError
        raise cls("wheel_refused", f"the {entry.id!r} pack's wheel is refused ({len(problems)} "
                                   f"problem{'s' if len(problems) != 1 else ''}): {text}",
                  pack=entry.id, problems=[msg for _, msg in problems])
    members = datawheel._members(data)
    folded: dict[str, str] = {}
    for rel in sorted(members):              # one file where names ignore case (macOS, Windows): refused
        prior = folded.setdefault(rel.casefold(), rel)
        if prior != rel:
            raise PackError("wheel_refused", f"the {entry.id!r} pack's wheel holds {prior!r} and {rel!r}, which are "
                                             "one file where names ignore case", pack=entry.id, member=rel)
    _mkdir_private(staged)
    with zipfile.ZipFile(wheel) as zf:
        for rel, (size, sha, cap) in sorted(members.items()):
            target = staged / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if rel == MANIFEST_FILE:
                    target.write_bytes(raw)
                    continue
                body = zf.getinfo(f"{pkg}/{rel}")
                with zf.open(body) as r, open(target, "xb") as out:
                    n = 0
                    while True:
                        chunk = r.read(min(_CHUNK, cap + 1 - n))
                        if not chunk:
                            break
                        n += len(chunk)
                        if n > cap:
                            raise PackError("wheel_refused", f"member {rel!r} of the {entry.id!r} pack is over {cap} bytes",
                                            pack=entry.id, member=rel)
                        out.write(chunk)
            except (OSError, KeyError, zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError) as exc:
                raise PackError("wheel_refused", f"member {rel!r} of the {entry.id!r} pack can't be extracted "
                                                 f"({type(exc).__name__}: {exc})", pack=entry.id, member=rel) from None
            if sha is not None:
                _verify(target, sha, size)
    return manifest


def _read_member(zf: zipfile.ZipFile, zi: zipfile.ZipInfo, cap: int) -> bytes:
    with zf.open(zi) as r:
        data = r.read(cap + 1)
    if len(data) > cap:
        raise PackError("wheel_refused", f"member {zi.filename!r} is over {cap} bytes")
    return data


# --------------------------------------------------------------------------- the shared tessdata folder

def _traineddata(manifest: Manifest) -> list[tuple[str, str]]:
    return [(a.file, a.sha256) for a in manifest.models if a.kind == TRAINEDDATA]


def _claims(exclude: str, b_manifest: Manifest | None) -> list[tuple[str, str, str]]:
    """(pack id, file, sha256) of every Tesseract model the base pack and the other installed packs ship."""
    out: list[tuple[str, str, str]] = []
    if b_manifest is not None and b_manifest.id != exclude:
        out += [(b_manifest.id, f, s) for f, s in _traineddata(b_manifest)]
    try:
        entries = _index().entries
    except PdfIntakeError:
        return out
    root = _cache_root() / PACKS
    for e in entries:
        if e.base or e.id == exclude:
            continue
        try:
            raw = (root / f"{e.id}-{e.version}" / MANIFEST_FILE).read_bytes()
        except OSError:
            continue
        if integrity.sha256_bytes(raw) != e.manifest_sha256:
            continue
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        for m in data.get("models") or ():
            if isinstance(m, dict) and m.get("kind") == TRAINEDDATA:
                out.append((e.id, str(m.get("file")), str(m.get("sha256"))))
    return out


def _tessdata_conflicts(manifest: Manifest, b: InstalledPack | None) -> None:
    """PackError when another pack ships a Tesseract model under one of this pack's file names with
    other contents: they share one tessdata folder, so one would silently replace the other.

    Names are compared case-insensitively (``casefold``): the tessdata folder sits on a
    case-insensitive file system on macOS and Windows by default, where ``ENG.traineddata`` and
    ``eng.traineddata`` are one file. Two of this pack's own models whose names differ only by case
    are refused the same way.
    """
    mine = _traineddata(manifest)
    if not mine:
        return
    seen: dict[str, str] = {}
    for file, _ in mine:
        prior = seen.setdefault(file.casefold(), file)
        if prior != file:
            raise PackError("tessdata_conflict",
                            f"the {manifest.id!r} pack ships {prior} and {file}, which are one file where names "
                            "ignore case", pack=manifest.id, other=manifest.id, file=file)
    others = _claims(manifest.id, b.manifest if b is not None else None)
    for file, sha in mine:
        for pid, ofile, osha in others:
            if ofile.casefold() == file.casefold() and osha != sha:
                raise PackError("tessdata_conflict",
                                f"the {manifest.id!r} and {pid!r} packs both ship {file} with different contents; "
                                f"Tesseract models share one folder, so the {manifest.id!r} pack can't be used",
                                pack=manifest.id, other=pid, file=file)


def _base_manifest_light() -> Manifest | None:
    try:
        entry = _index().base()
        dist = importlib.metadata.distribution(entry.dist.distribution)
        raw = Path(dist.locate_file(f"{entry.dist.distribution.replace('-', '_')}/{MANIFEST_FILE}")).read_bytes()
        if integrity.sha256_bytes(raw) != entry.manifest_sha256:
            return None
        return _parse_manifest(raw, base=None)
    except (PdfIntakeError, OSError, importlib.metadata.PackageNotFoundError):
        return None


def _link_tessdata(src: Path, file: str, sha: str, size: int, owner: str, owner_is_base: bool) -> None:
    """Make ``tessdata/<file>`` the verified ``src``: a hard link, or a checked copy where linking fails.

    A different file already there is replaced when it is a leftover (an older version of the same
    pack); when another installed pack ships that name, PackError ``tessdata_conflict``.
    """
    tdir = tessdata_dir()
    dest = tdir / file
    with _TESS_LOCK:
        if os.path.lexists(dest):
            try:
                if os.path.samefile(dest, src):
                    return
            except OSError:
                pass
            try:
                if dest.is_file() and dest.stat().st_size == size and integrity.sha256_file(dest) == sha:
                    return
            except (OSError, IntegrityError):              # sha256_file raises IntegrityError on an unreadable
                pass                                       # leftover: replaced below, like any stale one
            b_manifest = None if owner_is_base else _base_manifest_light()
            for pid, ofile, osha in _claims(owner, b_manifest):
                if ofile.casefold() == file.casefold() and osha != sha:
                    raise PackError("tessdata_conflict",
                                    f"the {owner!r} and {pid!r} packs both ship {file} with different contents",
                                    pack=owner, other=pid, file=file)
        tmp = tdir / f".{file}.{secrets.token_hex(6)}.tmp"
        try:
            try:
                os.link(src, tmp)
            except OSError:
                try:
                    shutil.copyfile(src, tmp)
                except OSError as exc:
                    raise IntegrityError("unreadable", f"{src} can't be copied into {tdir} ({exc.strerror or exc}).",
                                         path=str(src)) from None
                _verify(tmp, sha, size)
            _replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)


def _unlink_tessdata_of(d: Path) -> None:
    try:
        data = json.loads((d / MANIFEST_FILE).read_bytes())
    except (OSError, ValueError):
        return
    tdir = _cache_root() / TESSDATA
    for m in data.get("models") or ():
        if not isinstance(m, dict) or m.get("kind") != TRAINEDDATA or not isinstance(m.get("file"), str):
            continue
        link, own = tdir / m["file"], d / "models" / m["file"]
        try:
            if os.path.samefile(link, own) or integrity.sha256_file(link) == m.get("sha256"):
                link.unlink()
        except (OSError, IntegrityError):                  # an unreadable file there isn't known to be ours: kept
            continue


# --------------------------------------------------------------------------- files in the cache

def _cache_root() -> Path:
    root = platforms.cache_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    _mkdir_private(root)
    return root


def _packs_root() -> Path:
    path = _cache_root() / PACKS
    _mkdir_private(path)
    return path


def _mkdir_private(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise


def _replace(src: Path, dst: Path) -> None:
    """os.replace, retried briefly on Windows, where a reader holding ``dst`` open blocks it."""
    for attempt in range(10):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 9:
                raise
            time.sleep(0.05)


def _verified_file(path: Path, sha256: str, size: int) -> Path:
    """``path`` after checking it against its pin (IntegrityError naming it otherwise)."""
    _verify(path, sha256, size)
    return path


def _verify(path: Path, sha256: str, size: int) -> None:
    """``integrity.verify_file``, with an OSError between its stat and its read (a concurrent run
    discarding a corrupt pack folder) turned into the IntegrityError naming the file."""
    try:
        integrity.verify_file(path, sha256, size)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise IntegrityError("file_missing", f"{path} is missing.", path=str(path)) from None
        raise IntegrityError("unreadable", f"{path} can't be read ({exc.strerror or exc}).", path=str(path)) from None


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Write a small JSON file in the cache so readers see the old file or the new one, never half."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        _replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_capped(path: Path, cap: int) -> bytes:
    try:
        with open(path, "rb") as f:
            data = f.read(cap + 1)
    except FileNotFoundError:
        raise IntegrityError("file_missing", f"{path} is missing.", path=str(path)) from None
    except OSError as exc:
        if exc.errno in (errno.EISDIR, errno.ENOTDIR):
            raise IntegrityError("not_a_file", f"{path} isn't a regular file.", path=str(path)) from None
        raise IntegrityError("unreadable", f"{path} can't be read ({exc.strerror or exc}).", path=str(path)) from None
    if len(data) > cap:
        raise IntegrityError("too_large", f"{path} is larger than {cap} bytes.", path=str(path))
    return data


def _json(raw: bytes, where: Path) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise PackError("invalid_manifest", f"{where} isn't valid UTF-8 JSON", path=str(where)) from None
    if not isinstance(data, dict):
        raise PackError("invalid_manifest", f"{where} isn't a JSON object", path=str(where))
    return data
