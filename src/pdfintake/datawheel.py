"""Data wheels: build and check the wheels that carry a language pack's data, never code.

A data wheel holds one pack: its ``manifest.json``, the model, font and canary files the manifest
pins, and the notices those files need. Its layout is fixed::

    pdfintake_lang_th/manifest.json
    pdfintake_lang_th/models/<file>          one per manifest model
    pdfintake_lang_th/fonts/<file>           one per manifest font
    pdfintake_lang_th/canary-th.png          the manifest's canary, when it has one
    pdfintake_lang_th/NOTICE-<name>          every notice a model names, copied from the pack folder
    pdfintake_lang_th/OFL.txt                when the pack has fonts
    pdfintake_lang_th-1.0.0.dist-info/METADATA, WHEEL, RECORD

The base pack is ``pdfintake-data-<id>`` (package folder ``pdfintake_data_<id>``); every other pack is
``pdfintake-lang-<id>``. Nothing in a data wheel is ever imported: there is no ``__init__.py``, and a
member that is code (``.py``, ``.pth``, a native library, a script) is refused when the wheel is
built, when the store fetches it, and in the installed English distribution.

**Reproducible.** The same pack folder and files give the same wheel bytes on every machine and
Python: members in a fixed order, a fixed date and mode, no compression (``COMPRESSION``). Deflate
output depends on the zlib build (CPython 3.14 on Windows ships zlib-ng), so a compressed wheel
could hash differently on the CI runner that rebuilds it for publishing than on the machine that
pinned it in the index.

**Standard library only** (``tests/test_hygiene.py`` checks it), so ``scripts/dev-install.sh`` can
run it in a fresh clone, before pdfintake itself is installed. ``build`` downloads the pinned model
and font sources it doesn't have yet (every file is checked against its manifest size and SHA-256;
an error page is never kept), then writes the wheel and prints its name, size and SHA-256 as JSON.
The canary image has no upstream source, so ``build`` alone fails for any pack with a canary: it
must be drawn into the files folder first. To rebuild a pack's wheel at its pinned canary, run
dev-install, which draws the canary and then calls ``build``::

    PDFINTAKE_DIST=dist scripts/dev-install.sh --wheels-only --packs en

For a new or changed pack, ``pdfintake kit wheel <id>`` (the kit, installed) draws the canary, pins
it in the pack's manifest, then builds.
"""

from __future__ import annotations

import sys


def _drop_script_dir() -> None:
    """Run by path (``python3 src/pdfintake/datawheel.py``), Python puts this file's folder first on
    ``sys.path``, and ``src/pdfintake/types.py`` there would shadow the standard library's ``types``
    (``import argparse`` then fails inside ``enum``). So, before any other import, that entry is
    dropped: this module needs nothing beside it. Only ``sys`` (built in) and string operations are
    used here, since nothing else can be imported safely yet."""
    if __name__ != "__main__" or globals().get("__spec__") is not None or not sys.path:
        return
    if sys.flags.isolated or getattr(sys.flags, "safe_path", False):
        return                                           # -I, -P or PYTHONSAFEPATH: nothing was put there
    here = __file__.replace("\\", "/").rpartition("/")[0]
    first = (sys.path[0] or "").replace("\\", "/").rstrip("/")
    if first and (first == here.rstrip("/") or first.rpartition("/")[2] == here.rstrip("/").rpartition("/")[2]):
        del sys.path[0]


_drop_script_dir()

import argparse  # noqa: E402  (after _drop_script_dir, on purpose)
import base64  # noqa: E402
import hashlib  # noqa: E402
import http.client  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import stat  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable, Mapping  # noqa: E402

WHEEL_TAG = "py3-none-any"
GENERATOR = "pdfintake-datawheel 1"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
COMPRESSION = zipfile.ZIP_STORED
FILE_MODE = 0o100644
OFL_FILE = "OFL.txt"
MANIFEST_FILE = "manifest.json"
DIST_INFO_FILES = ("METADATA", "WHEEL", "RECORD")

MAX_MANIFEST_BYTES = 1 << 20
MAX_NOTICE_BYTES = 1 << 20
MAX_DIST_INFO_BYTES = 1 << 20
MAX_MODEL_BYTES = 104857600
MAX_FONT_BYTES = 10485760
MAX_CANARY_BYTES = 2097152

# Anything that can run: Python code and path hooks, native libraries, executables and scripts.
CODE_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".pyw", ".pyz", ".pth", ".so", ".dylib", ".dll", ".exe",
                 ".com", ".bat", ".cmd", ".ps1", ".sh", ".msi", ".jar")

# Path-safety patterns: the same as manifest.schema.json (tests/test_datawheel.py checks they agree).
ID_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]{2,8})?")
VERSION_RE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
TITLE_RE = re.compile(r"[A-Z][A-Za-z ()-]{0,39}")
SHA_RE = re.compile(r"[0-9a-f]{64}")
MODEL_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,80}")
FONT_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,80}\.(ttf|otf)")
CANARY_FILE_RE = re.compile(r"canary-[a-z0-9-]{1,20}\.png")
NOTICE_RE = re.compile(r"NOTICE-[a-z0-9-]{1,30}")
DIST_NAME_RE = re.compile(r"pdfintake-(data|lang)-[a-z]{2,3}(-[a-z0-9]{2,8})?")

ATTEMPTS = 3
BACKOFF_S = (1.0, 3.0)
TIMEOUT_S = 60.0
_CHUNK = 1 << 20
SCHEMA_PATH = Path(__file__).resolve().parent / "langpacks" / "manifest.schema.json"


class DataWheelError(ValueError):
    """A data wheel can't be built or its sources can't be fetched; the message says why."""


# --------------------------------------------------------------------------- names and members

def _dist_name(manifest: Mapping[str, Any]) -> str:
    kind = "data" if manifest.get("base") is True else "lang"
    return f"pdfintake-{kind}-{manifest['id']}"


def _package_dir(dist_name: str) -> str:
    return dist_name.replace("-", "_")


def _dist_info(dist_name: str, version: str) -> str:
    return f"{_package_dir(dist_name)}-{version}.dist-info"


def _wheel_filename(dist_name: str, version: str) -> str:
    return f"{_package_dir(dist_name)}-{version}-{WHEEL_TAG}.whl"


def _notices(manifest: Mapping[str, Any]) -> list[str]:
    names = {m.get("notice") for m in manifest.get("models") or () if isinstance(m, Mapping) and m.get("notice")}
    out = sorted(n for n in names if isinstance(n, str))
    if manifest.get("fonts"):
        out.append(OFL_FILE)
    return out


def _members(manifest: Mapping[str, Any]) -> dict[str, tuple[int | None, str | None, int]]:
    """The data members, relative to the package folder: path -> (bytes, sha256, cap).

    ``bytes`` and ``sha256`` are None where the manifest pins nothing (the manifest itself, which the
    index pins, and the notices, which the wheel's own hash pins); ``cap`` bounds every read.
    """
    out: dict[str, tuple[int | None, str | None, int]] = {MANIFEST_FILE: (None, None, MAX_MANIFEST_BYTES)}
    for m in manifest.get("models") or ():
        out[f"models/{m['file']}"] = (int(m["bytes"]), m["sha256"], int(m["bytes"]))
    for f in manifest.get("fonts") or ():
        out[f"fonts/{f['file']}"] = (int(f["bytes"]), f["sha256"], int(f["bytes"]))
    canary = manifest.get("canary")
    if canary:
        out[canary["file"]] = (int(canary["bytes"]), canary["sha256"], int(canary["bytes"]))
    for name in _notices(manifest):
        out[name] = (None, None, MAX_NOTICE_BYTES)
    return out


def _full(pattern: re.Pattern[str], value: Any) -> bool:
    return isinstance(value, str) and "\n" not in value and pattern.fullmatch(value) is not None


def _int_in(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _manifest_problems(manifest: Any) -> list[str]:
    """What stops this manifest from describing a data wheel (the full check is the core's loader)."""
    if not isinstance(manifest, Mapping):
        return ["the manifest isn't a JSON object"]
    out: list[str] = []
    if not _full(ID_RE, manifest.get("id")):
        out.append("$.id: missing or not a pack id")
    if not _full(VERSION_RE, manifest.get("version")):
        out.append("$.version: missing or not a version like 1.0.0")
    if "base" in manifest and not isinstance(manifest["base"], bool):
        out.append("$.base: must be true or false")
    if not _full(TITLE_RE, manifest.get("title")):
        out.append("$.title: missing or not a title")
    files: set[str] = set()
    for group, file_re, cap in (("models", MODEL_FILE_RE, MAX_MODEL_BYTES), ("fonts", FONT_FILE_RE, MAX_FONT_BYTES)):
        items = manifest.get(group)
        if not isinstance(items, list):
            out.append(f"$.{group}: must be a list")
            continue
        for i, item in enumerate(items):
            where = f"$.{group}[{i}]"
            if not isinstance(item, Mapping):
                out.append(f"{where}: must be an object")
                continue
            if not _full(file_re, item.get("file")):
                out.append(f"{where}.file: missing or not a safe file name")
            elif (group, item["file"]) in files:
                out.append(f"{where}.file: {item['file']!r} is listed twice")
            else:
                files.add((group, item["file"]))
            if not _int_in(item.get("bytes"), 1, cap):
                out.append(f"{where}.bytes: must be a whole number from 1 to {cap}")
            if not _full(SHA_RE, item.get("sha256")):
                out.append(f"{where}.sha256: missing or not a SHA-256")
            if not isinstance(item.get("licence"), str):
                out.append(f"{where}.licence: missing")
            if group == "models" and item.get("notice") is not None and not _full(NOTICE_RE, item.get("notice")):
                out.append(f"{where}.notice: not a NOTICE-<name> file name")
    canary = manifest.get("canary")
    if canary is not None:
        if not isinstance(canary, Mapping):
            out.append("$.canary: must be an object")
        else:
            if not _full(CANARY_FILE_RE, canary.get("file")):
                out.append("$.canary.file: missing or not canary-<name>.png")
            if not _int_in(canary.get("bytes"), 1, MAX_CANARY_BYTES):
                out.append(f"$.canary.bytes: must be a whole number from 1 to {MAX_CANARY_BYTES}")
            if not _full(SHA_RE, canary.get("sha256")):
                out.append("$.canary.sha256: missing or not a SHA-256")
    return out


def _require_manifest(manifest: Any) -> None:
    problems = _manifest_problems(manifest)
    if problems:
        raise DataWheelError("the manifest can't make a data wheel: " + "; ".join(problems))


_VERSIONED_SO_RE = re.compile(r"\.so(\.[0-9]+)+$")


def _is_code(name: str) -> bool:
    low = name.lower()
    return low.endswith(CODE_SUFFIXES) or _VERSIONED_SO_RE.search(low) is not None


def _unsafe(name: str) -> str | None:
    """Why a member path or RECORD path could escape its folder, or None."""
    if not name:
        return "an empty path"
    if "\x00" in name or "\\" in name:
        return "a NUL or a backslash in the path"
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return "an absolute path"
    parts = name.rstrip("/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        return "a '..', '.' or empty part in the path"
    return None


# --------------------------------------------------------------------------- small helpers

def _require_regular(path: Path, what: str) -> None:
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise DataWheelError(f"{what} {path} can't be read: {exc.strerror or exc}") from None
    if not stat.S_ISREG(st.st_mode):
        raise DataWheelError(f"{what} {path} must be a regular file, not a link or a folder")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _record_hash(digest_hex: str) -> str:
    return "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(digest_hex)).rstrip(b"=").decode("ascii")


def _load_json_bytes(data: bytes, what: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"the field {k!r} appears twice")
            out[k] = v
        return out

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataWheelError(f"{what} isn't valid UTF-8 JSON: {exc}") from None


def _read_capped(path: Path, cap: int, what: str) -> bytes:
    try:
        with open(path, "rb") as f:
            data = f.read(cap + 1)
    except OSError as exc:
        raise DataWheelError(f"{what} {path} can't be read: {exc.strerror or exc}") from None
    if len(data) > cap:
        raise DataWheelError(f"{what} {path} is larger than {cap} bytes")
    return data


def _check_file(path: Path, size: int, sha: str) -> str | None:
    """Why ``path`` isn't the pinned file, or None."""
    try:
        st = os.lstat(path)
    except OSError:
        return f"{path} is missing"
    if not stat.S_ISREG(st.st_mode):
        return f"{path} isn't a regular file (a link, a folder or a device is never copied into a wheel)"
    if st.st_size != size:
        return f"{path} is {st.st_size} bytes; the manifest pins {size}"
    got = _sha256_file(path)
    if got != sha:
        return f"{path} has SHA-256 {got}; the manifest pins {sha}"
    return None


def _metadata(manifest: Mapping[str, Any], dist_name: str) -> bytes:
    licences = sorted({str(x.get("licence")) for g in ("models", "fonts") for x in manifest.get(g) or ()
                       if isinstance(x, Mapping) and x.get("licence")})
    lines = ["Metadata-Version: 2.1", f"Name: {dist_name}", f"Version: {manifest['version']}",
             f"Summary: pdfintake language pack data: {manifest['title']} (models, fonts and a canary; data only)"]
    if licences:
        lines.append("License: " + " AND ".join(licences))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _wheel_file() -> bytes:
    return (f"Wheel-Version: 1.0\nGenerator: {GENERATOR}\nRoot-Is-Purelib: true\nTag: {WHEEL_TAG}\n").encode("ascii")


def _zipinfo(name: str, size: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    zi.compress_type = COMPRESSION
    zi.external_attr = FILE_MODE << 16
    zi.create_system = 3
    zi.create_version = 20
    zi.extract_version = 20
    zi.file_size = size
    return zi


# --------------------------------------------------------------------------- build

def build(pack_dir: Path, files_dir: Path, outdir: Path, *, dist_name: str) -> Path:
    """Write the data wheel for the pack in ``pack_dir`` into ``outdir`` and return its path.

    ``pack_dir`` holds ``manifest.json`` and the notices (``NOTICE-*``, ``OFL.txt``); ``files_dir``
    holds the model, font and canary files under their manifest file names. Every file must match
    its manifest size and SHA-256, and ``dist_name`` must be the pack's distribution name
    (``pdfintake-data-<id>`` for the base pack, ``pdfintake-lang-<id>`` for the others).
    The manifest's bytes go into the wheel unchanged, so the index's manifest SHA-256 holds for both.
    The same inputs give the same bytes (DataWheelError when anything doesn't match).
    """
    pack_dir, files_dir, outdir = Path(pack_dir), Path(files_dir), Path(outdir)
    _require_regular(pack_dir / MANIFEST_FILE, "the manifest")
    raw = _read_capped(pack_dir / MANIFEST_FILE, MAX_MANIFEST_BYTES, "the manifest")
    manifest = _load_json_bytes(raw, f"{pack_dir / MANIFEST_FILE}")
    _require_manifest(manifest)
    want = _dist_name(manifest)
    if not _full(DIST_NAME_RE, dist_name) or dist_name != want:
        raise DataWheelError(f"the distribution for pack {manifest['id']!r} is {want!r}, not {dist_name!r}")
    version = manifest["version"]
    pkg = _package_dir(dist_name)

    sources: list[tuple[str, Path | bytes]] = []          # (member path, file or bytes), data members
    problems: list[str] = []
    for rel, (size, sha, cap) in sorted(_members(manifest).items()):
        if _is_code(rel):
            problems.append(f"{rel} is code (by its name); a data wheel carries no code")
            continue
        if rel == MANIFEST_FILE:
            sources.append((f"{pkg}/{rel}", raw))
            continue
        if size is None:                                 # a notice, from the pack folder
            path = pack_dir / rel
            try:
                st = os.lstat(path)                      # a link is refused: it could pull in any file
            except OSError:
                problems.append(f"{path} is missing (a model names it, or the pack has fonts)")
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
                problems.append(f"{path} must be a regular file (not a link) of at most {cap} bytes")
                continue
            sources.append((f"{pkg}/{rel}", path))
            continue
        path = files_dir / Path(rel).name
        why = _check_file(path, size, sha)  # type: ignore[arg-type]
        if why is not None:
            if rel.startswith("canary-") and not path.exists():
                why += (" (a canary has no upstream source: `scripts/dev-install.sh --wheels-only --packs <id>` "
                        "draws it at its pin before the build; `pdfintake kit wheel <id>` draws and pins a new one)")
            problems.append(why)
            continue
        sources.append((f"{pkg}/{rel}", path))
    if problems:
        raise DataWheelError("the data wheel can't be built: " + "; ".join(problems))

    dist_info = _dist_info(dist_name, version)
    meta = [(f"{dist_info}/METADATA", _metadata(manifest, dist_name)), (f"{dist_info}/WHEEL", _wheel_file())]
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / _wheel_filename(dist_name, version)
    tmp = outdir / f".{dest.name}.{secrets.token_hex(6)}.tmp"
    record: list[str] = []
    try:
        with open(tmp, "wb") as fh, zipfile.ZipFile(fh, "w", allowZip64=False) as zf:
            for name, src in sources + meta:   # type: ignore[operator]
                h = hashlib.sha256()
                if isinstance(src, bytes):
                    data = src
                    h.update(data)
                    zf.writestr(_zipinfo(name, len(data)), data)
                    size = len(data)
                else:
                    size = os.stat(src).st_size
                    with open(src, "rb") as f, zf.open(_zipinfo(name, size), "w") as w:
                        copied = 0
                        while True:
                            chunk = f.read(_CHUNK)
                            if not chunk:
                                break
                            copied += len(chunk)
                            h.update(chunk)
                            w.write(chunk)
                    if copied != size:
                        raise DataWheelError(f"{src} changed while the wheel was being built")
                record.append(f"{name},{_record_hash(h.hexdigest())},{size}")
            record.append(f"{dist_info}/RECORD,,")
            body = ("\n".join(record) + "\n").encode("utf-8")
            zf.writestr(_zipinfo(f"{dist_info}/RECORD", len(body)), body)
        wrong = wheel_problems(tmp, manifest)            # the same check the store runs on a fetch
        if wrong:
            raise DataWheelError("the built wheel fails its own check: " + "; ".join(wrong))
        if os.path.lexists(dest) and not stat.S_ISREG(os.lstat(dest).st_mode):
            raise DataWheelError(f"{dest} exists and isn't a regular file; nothing was written")
        os.replace(tmp, dest)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    return dest


# --------------------------------------------------------------------------- checks

def _check_wheel(path: Path, manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every problem with a data wheel as (code, message); codes: manifest, zip, path, code, unknown,
    missing, size, hash, metadata, record."""
    if _manifest_problems(manifest):
        return [("manifest", p) for p in _manifest_problems(manifest)]
    dist_name = _dist_name(manifest)
    pkg = _package_dir(dist_name)
    dist_info = _dist_info(dist_name, manifest["version"])
    data = {f"{pkg}/{rel}": spec for rel, spec in _members(manifest).items()}
    meta = {f"{dist_info}/{n}" for n in DIST_INFO_FILES}
    out: list[tuple[str, str]] = []
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        return [("zip", f"{path} isn't a readable wheel: {exc}")]
    hashes: dict[str, tuple[str, int]] = {}
    contents: dict[str, bytes] = {}
    with zf:
        seen: set[str] = set()
        for zi in zf.infolist():
            name = zi.filename
            # zipfile's name is cleaned up: cut at a NUL everywhere, and each "\" turned into "/" on
            # Windows. The name the archive stores (orig_filename) is checked first, so a member
            # stored with a backslash or a NUL is refused on every OS, not only where zipfile keeps it.
            why, shown = _unsafe(zi.orig_filename), zi.orig_filename
            if why is None:
                why, shown = _unsafe(name), name
            if why is not None:
                out.append(("path", f"member {shown!r} is refused: {why}"))
                continue
            if name in seen:
                out.append(("path", f"member {name!r} appears twice"))
                continue
            seen.add(name)
            if _is_code(name):
                out.append(("code", f"member {name!r} is code; a data wheel carries no code"))
                continue
            mode = (zi.external_attr >> 16) & 0o170000
            if mode and mode != stat.S_IFREG:
                out.append(("path", f"member {name!r} isn't a regular file (a link or a folder)"))
                continue
            if zi.flag_bits & 0x1:
                out.append(("zip", f"member {name!r} is encrypted"))
                continue
            if zi.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                out.append(("zip", f"member {name!r} uses an unsupported compression"))
                continue
            if name not in data and name not in meta:
                out.append(("unknown", f"member {name!r} isn't part of this pack"))
                continue
            size, sha, cap = data.get(name, (None, None, MAX_DIST_INFO_BYTES))
            h = hashlib.sha256()
            n = 0
            keep = name in meta or name == f"{pkg}/{MANIFEST_FILE}"
            buf = bytearray()
            try:
                with zf.open(zi) as r:
                    while True:
                        chunk = r.read(min(_CHUNK, cap + 1 - n))
                        if not chunk:
                            break
                        n += len(chunk)
                        if n > cap:
                            break
                        h.update(chunk)
                        if keep:
                            buf += chunk
            except (OSError, zipfile.BadZipFile, EOFError, ValueError) as exc:
                out.append(("zip", f"member {name!r} can't be read: {exc}"))
                continue
            if n > cap:
                out.append(("size", f"member {name!r} is larger than {cap} bytes"))
                continue
            digest = h.hexdigest()
            hashes[name] = (digest, n)
            if keep:
                contents[name] = bytes(buf)
            if size is not None and n != size:
                out.append(("size", f"member {name!r} is {n} bytes; the manifest pins {size}"))
            elif sha is not None and digest != sha:
                out.append(("hash", f"member {name!r} has SHA-256 {digest}; the manifest pins {sha}"))
        for name in sorted((set(data) | meta) - seen):
            out.append(("missing", f"member {name!r} is missing"))
    mname = f"{pkg}/{MANIFEST_FILE}"
    if mname in contents:
        try:
            carried = _load_json_bytes(contents[mname], "the wheel's manifest")
        except DataWheelError as exc:
            out.append(("metadata", str(exc)))
        else:
            if carried != dict(manifest):
                out.append(("metadata", f"member {mname!r} isn't the expected manifest"))
    meta_text = contents.get(f"{dist_info}/METADATA", b"").decode("utf-8", "replace")
    fields = dict(ln.split(": ", 1) for ln in meta_text.splitlines() if ": " in ln)
    if f"{dist_info}/METADATA" in contents:
        if fields.get("Name") != dist_name or fields.get("Version") != manifest["version"]:
            out.append(("metadata", f"METADATA names {fields.get('Name')!r} {fields.get('Version')!r}, "
                                    f"not {dist_name!r} {manifest['version']!r}"))
    wheel_text = contents.get(f"{dist_info}/WHEEL", b"").decode("utf-8", "replace")
    if f"{dist_info}/WHEEL" in contents:
        wf = dict(ln.split(": ", 1) for ln in wheel_text.splitlines() if ": " in ln)
        if wf.get("Root-Is-Purelib") != "true" or wf.get("Tag") != WHEEL_TAG:
            out.append(("metadata", f"WHEEL must say Root-Is-Purelib: true and Tag: {WHEEL_TAG}"))
    record_name = f"{dist_info}/RECORD"
    if record_name in contents:
        listed: dict[str, tuple[str, str]] = {}
        for ln in contents[record_name].decode("utf-8", "replace").splitlines():
            if not ln:
                continue
            parts = ln.rsplit(",", 2)
            if len(parts) != 3:
                out.append(("record", f"RECORD line {ln!r} isn't path,hash,size"))
                continue
            listed[parts[0]] = (parts[1], parts[2])
        for name, (digest, n) in sorted(hashes.items()):
            if name == record_name:
                continue
            got = listed.pop(name, None)
            if got is None:
                out.append(("record", f"RECORD doesn't list {name!r}"))
            elif got != (_record_hash(digest), str(n)):
                out.append(("record", f"RECORD's hash or size for {name!r} is wrong"))
        listed.pop(record_name, None)
        for name in sorted(listed):
            out.append(("record", f"RECORD lists {name!r}, which the wheel doesn't hold"))
    return out


def wheel_problems(path: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Why this wheel isn't the data wheel ``manifest`` describes ([] when it is).

    Refused: a member that is code (``.py``, ``.pth``, ``.so``, ``.dylib``, ``.dll`` …), a link, an
    unsafe path (``..``, absolute, a backslash), an encrypted member, a member the manifest doesn't
    list, a missing member, a size or SHA-256 that differs from the manifest, a carried manifest that
    differs from ``manifest``, METADATA or WHEEL that don't fit, and a RECORD that doesn't match.
    Every read is capped, so a member whose header lies about its size can't exhaust memory.
    """
    return [msg for _, msg in _check_wheel(Path(path), manifest)]


def dist_problems(files: Iterable[str], manifest: Mapping[str, Any]) -> list[str]:
    """The same test on an installed distribution's RECORD paths ([] when it is data only).

    Refused: a code file anywhere, an unsafe path (``..`` reaches outside site-packages: scripts,
    headers, a path hook), a file outside the package folder and its own ``.dist-info`` (another
    package, a ``.pth`` at the top level), and a package file the manifest doesn't list. Every listed
    member must be present. An installer's own ``.dist-info`` files (INSTALLER, REQUESTED,
    direct_url.json …) are allowed when they aren't code.
    """
    problems = _manifest_problems(manifest)
    if problems:
        return problems
    dist_name = _dist_name(manifest)
    pkg = _package_dir(dist_name)
    dist_info = _dist_info(dist_name, manifest["version"])
    data = {f"{pkg}/{rel}" for rel in _members(manifest)}
    out: list[str] = []
    seen: set[str] = set()
    for f in files:
        name = str(f)
        why = _unsafe(name)
        if why is not None:
            out.append(f"installed file {name!r} is refused: {why}")
            continue
        if _is_code(name):
            out.append(f"installed file {name!r} is code; the data package carries no code")
            continue
        if name.startswith(pkg + "/"):
            if name not in data:
                out.append(f"installed file {name!r} isn't part of this pack")
            seen.add(name)
        elif not name.startswith(dist_info + "/"):
            out.append(f"installed file {name!r} lies outside {pkg}/ and {dist_info}/")
    for name in sorted(data - seen):
        out.append(f"installed file {name!r} is missing")
    return out


# --------------------------------------------------------------------------- downloading the pinned sources

def _source_patterns() -> dict[str, re.Pattern[str]]:
    """The model and font source allowlist, read from manifest.schema.json beside this file."""
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        defs = schema["$defs"]
        return {"model": re.compile(defs["model"]["properties"]["source"]["pattern"]),
                "font": re.compile(defs["font"]["properties"]["source"]["pattern"])}
    except (OSError, ValueError, KeyError, TypeError, re.error) as exc:
        raise DataWheelError(f"the source allowlist can't be read from {SCHEMA_PATH}: {exc}") from None


def _source_problem(url: Any, pattern: re.Pattern[str]) -> str | None:
    if not isinstance(url, str) or "\n" in url or not pattern.fullmatch(url):
        return "isn't from an allowed source"
    if ".." in url or "?" in url or "#" in url or "%" in url or "\\" in url:
        return "contains '..', a query, a fragment, '%' or a backslash"
    return None


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise urllib.error.URLError(f"refused a redirect away from https ({newurl.split('?')[0]})")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str, timeout: float) -> Any:
    """Open ``url`` (https, redirects kept on https). Tests replace this to reach a local server."""
    opener = urllib.request.build_opener(_HttpsOnly())
    req = urllib.request.Request(url, headers={"User-Agent": GENERATOR, "Accept-Encoding": "identity"})
    return opener.open(req, timeout=timeout)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class _Retry(Exception):
    pass


def _download_once(url: str, dest: Path, size: int, sha: str) -> None:
    part = dest.with_name(f".{dest.name}.{secrets.token_hex(6)}.part")
    try:
        try:
            resp = _open_url(url, TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            exc.close()                                  # the error body is never read, let alone kept
            raise _Retry(f"HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            # HTTPException: a garbage status line (BadStatusLine) or an oversized header (LineTooLong)
            raise _Retry(f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}") from None
        h = hashlib.sha256()
        n = 0
        try:
            with resp, open(part, "xb") as out:
                while True:
                    chunk = resp.read(min(_CHUNK, size + 1 - n))
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > size:
                        raise _Retry(f"the server sent more than the pinned {size} bytes")
                    h.update(chunk)
                    out.write(chunk)
        except (OSError, http.client.HTTPException) as exc:     # IncompleteRead: a broken chunked stream
            raise _Retry(f"the download broke off: {type(exc).__name__}: {exc}") from None
        if n != size:
            raise _Retry(f"got {n} bytes; the manifest pins {size}")
        if h.hexdigest() != sha:
            raise _Retry(f"got SHA-256 {h.hexdigest()}; the manifest pins {sha}")
        os.replace(part, dest)
    finally:
        try:
            os.unlink(part)
        except FileNotFoundError:
            pass


def _attempts_text(n: int) -> str:
    """``3`` -> ``"three attempts"`` (decided 2026-11-05: messages say attempts, in words)."""
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    word = words[n] if 0 <= n < len(words) else str(n)
    return f"{word} attempt{'' if n == 1 else 's'}"


def _hosts_text(url: str) -> str:
    """The hosts a download of ``url`` goes through, for a failure message. github.com answers a raw
    file or a release asset with a redirect to a githubusercontent.com host (raw. and release-assets.,
    checked against live public files on 2026-11-05), so the message for a GitHub source names them
    too; any other source names its own host."""
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        host = ""
    if host == "github.com":
        return "GitHub (github.com, or the githubusercontent.com hosts it redirects to)"
    return host or "the source's host"


def _download(url: str, dest: Path, size: int, sha: str) -> None:
    last = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            _sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        try:
            _download_once(url, dest, size, sha)
            return
        except _Retry as exc:
            last = str(exc)
    raise DataWheelError(f"{dest.name} couldn't be downloaded from {url} after {_attempts_text(ATTEMPTS)} ({last}); "
                         f"nothing was kept. The network setting may block downloads from {_hosts_text(url)}.")


def download_sources(manifest: Mapping[str, Any], dest: Path) -> list[Path]:
    """Every model and font the manifest pins, in ``dest`` under its file name; their paths, in manifest order.

    A file already in ``dest`` with the pinned size and SHA-256 is kept. Others are downloaded from
    their pinned ``source`` (which must be on the manifest schema's allowlist) with three attempts and
    backoff, streamed to a temporary name, capped at the pinned size and checked before they take their
    name. An error page (a 504, or a 200 whose body isn't the file) is never kept as a model.
    """
    _require_manifest(manifest)
    patterns = _source_patterns()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for kind, group in (("model", "models"), ("font", "fonts")):
        for item in manifest.get(group) or ():
            target = dest / item["file"]
            size, sha = int(item["bytes"]), item["sha256"]
            try:
                st = os.lstat(target)
            except FileNotFoundError:
                st = None
            except OSError as exc:
                raise DataWheelError(f"{target} can't be checked: {exc.strerror or exc}") from None
            if st is not None and stat.S_ISDIR(st.st_mode):
                raise DataWheelError(f"{target} is a folder; a model or font file belongs there, so nothing was fetched")
            if _check_file(target, size, sha) is None:
                out.append(target)
                continue
            why = _source_problem(item.get("source"), patterns[kind])
            if why is not None:
                raise DataWheelError(f"the source of {item['file']} {why}: {item.get('source')!r}")
            try:
                os.unlink(target)                        # a wrong or partial file never stays under the name
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise DataWheelError(f"{target} is wrong and can't be removed: {exc.strerror or exc}") from None
            _download(item["source"], target, size, sha)
            out.append(target)
    return out


# --------------------------------------------------------------------------- command line

def _emit(obj: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _load_pack_manifest(pack_dir: Path) -> dict[str, Any]:
    raw = _read_capped(pack_dir / MANIFEST_FILE, MAX_MANIFEST_BYTES, "the manifest")
    manifest = _load_json_bytes(raw, str(pack_dir / MANIFEST_FILE))
    _require_manifest(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """``build PACK_DIR --out DIR``, ``download PACK_DIR --dest DIR``, ``check WHEEL --pack PACK_DIR``.

    Prints one JSON object. Exit 0 on success, 1 when a build, download or check fails, 2 for bad arguments.
    """
    parser = argparse.ArgumentParser(prog="datawheel.py", description="Build and check pdfintake data wheels.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="download the pinned sources if needed, then build the wheel")
    b.add_argument("pack_dir", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--files", type=Path, help="the folder of model, font and canary files (default OUT/sources/ID)")
    b.add_argument("--dist-name", help="the distribution name (default from the manifest)")
    b.add_argument("--offline", action="store_true", help="never download; every file must be in --files")
    d = sub.add_parser("download", help="download the pinned model and font sources")
    d.add_argument("pack_dir", type=Path)
    d.add_argument("--dest", type=Path, required=True)
    c = sub.add_parser("check", help="check a built wheel against its pack's manifest")
    c.add_argument("wheel", type=Path)
    c.add_argument("--pack", type=Path, required=True, help="the pack folder holding manifest.json")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        manifest = _load_pack_manifest(args.pack_dir if args.command != "check" else args.pack)
        if args.command == "download":
            paths = download_sources(manifest, args.dest)
            _emit({"ok": True, "files": [str(p) for p in paths]})
            return 0
        if args.command == "check":
            problems = wheel_problems(args.wheel, manifest)
            _emit({"ok": not problems, "wheel": str(args.wheel), "problems": problems})
            return 0 if not problems else 1
        files = args.files or (args.out / "sources" / manifest["id"])
        if not args.offline:
            download_sources(manifest, files)
        wheel = build(args.pack_dir, files, args.out, dist_name=args.dist_name or _dist_name(manifest))
        _emit({"ok": True, "wheel": str(wheel), "distribution": _dist_name(manifest),
               "version": manifest["version"], "bytes": wheel.stat().st_size, "sha256": _sha256_file(wheel)})
        return 0
    except DataWheelError as exc:
        _emit({"ok": False, "error": str(exc)})
        return 1
    except OSError as exc:                               # a folder where a file belongs, a full disk, no access
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
