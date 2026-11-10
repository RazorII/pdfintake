"""Thorough mode: dots.mocr on Apple silicon, opt-in.

In thorough mode every OCR page also gets a read from dots.mocr, a vision-language OCR model that runs
on the Mac's GPU through MLX. It never reads a page alone: on Latin-script pages it adds the numbers the
layer lacks and the words in a script the pack doesn't cover; on a pack whose pick rule is
``corroborated_trigrams`` it competes as one more reader. It costs about 17 s a page on an M-series Mac
and a one-time 4.4 GB download, so it is off unless the user turns it on.

This module holds what the mode needs outside a run:

* the model, pinned: ``REPO`` at ``REVISION``, every file with its size and SHA-256 (``FILES``);
* the setting (``thorough.json`` in pdfintake's config folder): ``on``, ``off`` or ``declined``; no file
  means the user hasn't decided, which is when a run's record offers the mode;
* ``status()``, ``enable()``, ``disable()`` and ``decline()``, behind ``pdfintake thorough``;
* ``resolve(requested)``, which a run asks whether the mode reads this run and with which model folder.

Nothing here asks anything: the libraries come as the extra ``pdfintake[thorough]`` (this module can't
install them, since ``src/`` never starts a process), and ``enable`` downloads only when it is called.
The download goes to Hugging Face over https; a redirect is followed only to https on ``huggingface.co``
or a ``*.hf.co`` host. Each file is size-capped at its pin and hashed before it is moved into place.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import __version__, platforms
from .errors import IntegrityError, SetupError

ENGINE_NAME = "dots"
PLATFORM_KEY = "darwin-arm64"
MIN_MACOS = (14, 0)                     # mlx's oldest macOS wheel (macosx_14_0_arm64)
LIBRARIES = ("mlx", "tokenizers")
EXTRA = "thorough"

REPO = "mlx-community/dots.mocr-8bit"
REVISION = "6d9e750ba44cf99e3c608cfdf47e59df209dce13"
MODEL_DIR = "dots.mocr-8bit-6d9e750"
HOST = "huggingface.co"
CDN_SUFFIX = ".hf.co"
VERIFIED = ".verified"
SETTING_FILE = "thorough.json"
State = Literal["on", "off", "declined"]
STATES: tuple[str, ...] = ("on", "off", "declined")

ATTEMPTS = 3
BACKOFF_S = (2.0, 6.0)
TIMEOUT_S = 60.0
_CHUNK = 1 << 20
FREE_MARGIN = 512 << 20                 # free space left over after the download, at the least
SECONDS_PER_PAGE = 17                   # dots seconds a page on an M5 Pro, 165 pages: mean 17.2, median 13.9


@dataclass(frozen=True)
class ModelFile:
    name: str
    bytes: int
    sha256: str


# The model card comes too: it carries the model's licence line.
FILES: tuple[ModelFile, ...] = (
    ModelFile("README.md", 771, "23212d86b8c4e1417ef333fc492f7b6b266cad5b605c3e426dffe10536168a3f"),
    ModelFile("chat_template.jinja", 1119, "7bcde5a3e537309d3316239825b950a337dd34562cd5171cdb811134fd9a23ce"),
    ModelFile("chat_template.json", 1147, "c8d629ba2b3133376f1e64b980f82438945741fba5334db768afb281af93c4a9"),
    ModelFile("config.json", 1720, "d4d799c5daac4c963d8ac4d4438236fc087ae1d0a9c6cd1a59eee4bfbeec276a"),
    ModelFile("generation_config.json", 86, "d9ca5bd51bbf3b3b2b418f2b2d296cb4286ef2f02dc76dd7361d82ffd9d4e7ca"),
    ModelFile("model.safetensors", 4412602873, "9d8c0ed06334ed486db9aeb7ea8e1b22e807693c5aad629e93679fc5381da757"),
    ModelFile("model.safetensors.index.json", 83910, "ed21175980235c8bef5e51e456381b9776d9165e17a94a7a903e670b6d0800b2"),
    ModelFile("preprocessor_config.json", 432, "fec8b3187ab0ae340f6d0b1f6f892a53c63b73b328836c83dd7bff1d5bc4c79a"),
    ModelFile("processor_config.json", 735, "69edbef36d2dd554e4d1f8fb6ad742843e8c16728922dc9aa22f9b68fdb8a72b"),
    ModelFile("special_tokens_map.json", 494, "92b060f39417514eed2f7652f454a3d5afb0e4a57b9f4b51dd4d1dfd8fdb9736"),
    ModelFile("tokenizer.json", 11426251, "904d81ff0cfa066dbc0b6a21e10ded6ebb7c2d8df14100d851f90bb7878bd5de"),
    ModelFile("tokenizer_config.json", 702, "f18a7062fdb3b2213bce50801d6297f51c8e8d37f909c8153f80b9be17c0bf39"),
    ModelFile("vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
)
TOTAL_BYTES = sum(f.bytes for f in FILES)


# --------------------------------------------------------------------------- where things live

def model_dir() -> Path:
    """``<cache>/thorough/<MODEL_DIR>``: the model's folder (never created here)."""
    return platforms.cache_root() / "thorough" / MODEL_DIR


def setting_path() -> Path:
    from .passwords import _config_root

    return _config_root() / SETTING_FILE


def install_command() -> str:
    """The command that adds the mode's libraries to an install made with ``uv tool install``."""
    return (f"uv tool install --reinstall 'pdfintake[{EXTRA}] @ "
            f"git+https://github.com/RazorII/pdfintake@v{__version__}'")


# --------------------------------------------------------------------------- the setting

def setting() -> State | None:
    """The user's choice, or None when there is none yet (no file, or one that can't be read)."""
    try:
        raw = setting_path().read_bytes()
    except OSError:
        return None
    if len(raw) > 4096:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    state = data.get("state") if isinstance(data, dict) else None
    return state if state in STATES else None  # type: ignore[return-value]


def _write_setting(state: State) -> None:
    path = setting_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{SETTING_FILE}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps({"state": state, "model": REVISION}) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- what this machine can do

def platform_problem() -> str | None:
    """Why this machine can't run the mode, or None."""
    if platforms.current().key != PLATFORM_KEY:
        return "thorough mode runs only on a Mac with Apple silicon"
    version = platforms.macos_version()
    if version is not None and version < MIN_MACOS:
        return f"thorough mode needs macOS {MIN_MACOS[0]} or later"
    return None


def missing_libraries() -> list[str]:
    """The mode's libraries that aren't importable here."""
    return [name for name in LIBRARIES if importlib.util.find_spec(name) is None]


def _file_ok(path: Path, f: ModelFile, *, deep: bool) -> bool:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size != f.bytes:
            return False
    except OSError:
        return False
    return not deep or _sha256(path) == f.sha256


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def model_problems(*, deep: bool = False) -> list[str]:
    """What's wrong with the installed model: missing, wrong-sized or (``deep``) wrong-hashed files."""
    d = model_dir()
    out = [f.name for f in FILES if not _file_ok(d / f.name, f, deep=deep)]
    if not out and not deep and not (d / VERIFIED).is_file():
        out.append(VERIFIED)
    return out


def model_paths() -> dict[str, Path]:
    """The model's files for the engine, each checked against its pin now (IntegrityError if not)."""
    d = model_dir()
    out: dict[str, Path] = {}
    for f in FILES:
        path = d / f.name
        if not _file_ok(path, f, deep=False):
            raise IntegrityError("thorough_model_changed",
                                 f"the thorough-mode model file {path} isn't the pinned file; run "
                                 "'pdfintake thorough enable' to fetch it again", file=str(path))
        out[f.name] = path
    return out


# --------------------------------------------------------------------------- status

def status() -> dict[str, Any]:
    """Everything ``pdfintake thorough status`` reports; never raises for the machine's state."""
    plat = platform_problem()
    missing = missing_libraries() if plat is None else list(LIBRARIES)
    problems = model_problems() if plat is None else [f.name for f in FILES]
    have = 0 if plat is not None else sum(f.bytes for f in FILES if f.name not in problems)
    state = setting()
    ready = plat is None and not missing and not problems
    return {
        "available": plat is None,
        "why_not": plat,
        "libraries": {"missing": missing, "install": install_command() if missing and plat is None else None},
        "model": {"repo": REPO, "revision": REVISION, "installed": not problems, "folder": str(model_dir()),
                  "download_bytes": max(0, TOTAL_BYTES - have)},
        "setting": state,
        "on": ready and state == "on",
        "ready": ready,
        "seconds_per_page": SECONDS_PER_PAGE,
    }


def resolve(requested: bool | None) -> tuple[bool, str | None]:
    """(read in thorough mode this run, why not) for a run's ``--thorough`` (True), ``--no-thorough``
    (False) or neither (None, the setting decides). ``why not`` is None when the mode reads, or when it
    was neither asked for nor on; otherwise it says what stopped it (the run carries on without it)."""
    want = requested if requested is not None else setting() == "on"
    if not want:
        return False, None
    plat = platform_problem()
    if plat is not None:
        return False, plat
    missing = missing_libraries()
    if missing:
        return False, f"the thorough-mode libraries aren't installed ({', '.join(missing)})"
    problems = model_problems()
    if problems:
        return False, "the thorough-mode model isn't installed; run 'pdfintake thorough enable'"
    return True, None


def should_offer() -> bool:
    """Whether a run's record offers the mode: this machine can run it and the user hasn't decided."""
    return platform_problem() is None and setting() is None


# --------------------------------------------------------------------------- enable, disable, decline

def disable(*, remove: bool = False) -> dict[str, Any]:
    _write_setting("off")
    if remove:
        shutil.rmtree(model_dir(), ignore_errors=True)
    return status()


def decline() -> dict[str, Any]:
    _write_setting("declined")
    return status()


Progress = Callable[[str, int, int], None]      # (file name, bytes done, bytes of the file)


def enable(*, progress: Progress | None = None, canary: bool = True) -> dict[str, Any]:
    """Install the model when it isn't (download, size cap, SHA-256 per file), read the canary, turn the
    mode on. SetupError (exit 3) when the machine can't run it, the libraries are missing, there isn't
    room, or the download fails; IntegrityError when a file arrives with another hash. Nothing is
    turned on unless every file and the canary passed."""
    plat = platform_problem()
    if plat is not None:
        raise SetupError("thorough_not_supported", f"{plat[0].upper()}{plat[1:]}.")
    missing = missing_libraries()
    if missing:
        raise SetupError("thorough_libraries_missing",
                         f"Thorough mode needs {', '.join(missing)}; install them with: {install_command()}",
                         missing=missing, command=install_command())
    d = model_dir()
    todo = [f for f in FILES if not _file_ok(d / f.name, f, deep=True)]
    need = sum(f.bytes for f in todo)
    if todo:
        d.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(d.parent).free
        if free < need + FREE_MARGIN:
            raise SetupError("thorough_no_room",
                             f"Thorough mode needs {_size(need)} for its model and this disk has {_size(free)} free.",
                             need=need, free=free)
        d.mkdir(parents=True, exist_ok=True)
        (d / VERIFIED).unlink(missing_ok=True)
        staging = platforms.cache_root() / "staging" / f"thorough-{secrets.token_hex(4)}"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for f in todo:
                part = staging / f.name
                _download(f, part, progress)
                os.replace(part, d / f.name)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    for f in FILES:                    # every file was hashed above or while it downloaded; sizes again here
        if not _file_ok(d / f.name, f, deep=False):
            raise IntegrityError("thorough_model_changed", f"{d / f.name} isn't the pinned file",
                                 file=str(d / f.name))
    if canary:
        from .ocr.engines import dots

        problem = dots.canary(model_paths())
        if problem is not None:
            raise SetupError("thorough_canary_failed", f"Thorough mode's test page didn't read right: {problem}.")
    (d / VERIFIED).write_text(REVISION + "\n", encoding="utf-8")
    _write_setting("on")
    return status()


# --------------------------------------------------------------------------- downloading

BASE_URL = f"https://{HOST}"           # a seam: the tests point it at a loopback stub


def _url(f: ModelFile) -> str:
    return f"{BASE_URL}/{REPO}/resolve/{REVISION}/{urllib.parse.quote(f.name)}"


def _host_ok(host: str) -> bool:
    """huggingface.co, or a host under hf.co (its download hosts)."""
    host = host.lower()
    return host == HOST or (host.endswith(CDN_SUFFIX) and host.count(".") >= 2)


def _port_allowed(host: str, port: int | None) -> bool:
    """Only the default port (a seam: the tests allow their stub's)."""
    return port is None


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _url_problem(url: str) -> str | None:
    if not isinstance(url, str) or not url or any(ord(c) <= 0x20 or ord(c) >= 0x7F for c in url) or "\\" in url:
        return "isn't a plain address"
    try:
        u = urllib.parse.urlsplit(url)
        port = u.port
    except ValueError:
        return "isn't a valid URL"
    if u.scheme != "https":
        return "doesn't use https"
    if not _host_ok(u.hostname or ""):
        return f"is on {u.hostname or 'no host'}, which thorough mode doesn't download from"
    if u.username is not None or u.password is not None:
        return "carries a user name or a password"
    if not _port_allowed(u.hostname or "", port):
        return "names a port"
    return None


class _Guard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        problem = _url_problem(newurl)
        if problem:
            raise SetupError("thorough_redirect_refused",
                             f"The model download was redirected to an address that {problem}; nothing was fetched.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _ssl_context() -> ssl.SSLContext:
    from .langpacks.store import _ssl_context as ctx

    return ctx()


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler(),
                                       urllib.request.HTTPSHandler(context=_ssl_context()), _Guard())


class _Retry(Exception):
    pass


def _download(f: ModelFile, dest: Path, progress: Progress | None) -> None:
    last = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            _sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        try:
            _download_once(f, dest, progress)
            return
        except _Retry as exc:
            last = str(exc)
    raise SetupError("thorough_download_failed",
                     f"The thorough-mode model file {f.name} couldn't be downloaded from {HOST} after {ATTEMPTS} "
                     f"tries ({last}); the network may block downloads from Hugging Face.",
                     file=f.name, reason=last)


def _download_once(f: ModelFile, dest: Path, progress: Progress | None) -> None:
    dest.unlink(missing_ok=True)
    req = urllib.request.Request(_url(f), headers={"User-Agent": f"pdfintake/{__version__}",
                                                   "Accept-Encoding": "identity"})
    try:
        resp = _opener().open(req, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code in (408, 425, 429) or exc.code >= 500:
            raise _Retry(f"HTTP {exc.code}") from None
        raise SetupError("thorough_download_refused",
                         f"The thorough-mode model file {f.name} couldn't be downloaded: {HOST} answered HTTP {exc.code}.",
                         file=f.name, status=exc.code) from None
    except urllib.error.URLError as exc:
        raise _Retry(f"{type(exc.reason).__name__ if isinstance(exc.reason, BaseException) else 'URLError'}: "
                     f"{exc.reason}") from None
    except (OSError, ValueError) as exc:
        raise _Retry(f"{type(exc).__name__}: {exc}") from None
    with resp:
        length = resp.headers.get("Content-Length")
        if length is not None and length.strip().isdigit() and int(length) != f.bytes:
            raise SetupError("thorough_download_size",
                             f"{HOST} offered {int(length)} bytes for {f.name}; the pin is {f.bytes}. Nothing was kept.",
                             file=f.name, offered=int(length), expected=f.bytes)
        h = hashlib.sha256()
        n = 0
        try:
            with open(dest, "xb") as out:
                while True:
                    chunk = resp.read(min(_CHUNK, f.bytes + 1 - n))
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > f.bytes:
                        raise SetupError("thorough_download_size",
                                         f"{HOST} sent more than the {f.bytes} bytes pinned for {f.name}. Nothing was kept.",
                                         file=f.name, expected=f.bytes)
                    h.update(chunk)
                    out.write(chunk)
                    if progress is not None:
                        progress(f.name, n, f.bytes)
        except (OSError, ConnectionError) as exc:
            dest.unlink(missing_ok=True)
            raise _Retry(f"{type(exc).__name__}: {exc}") from None
        except SetupError:
            dest.unlink(missing_ok=True)
            raise
    if n != f.bytes:
        dest.unlink(missing_ok=True)
        raise _Retry(f"got {n} of {f.bytes} bytes")
    if h.hexdigest() != f.sha256:
        dest.unlink(missing_ok=True)
        raise IntegrityError("thorough_model_changed",
                             f"{f.name} from {HOST} has SHA-256 {h.hexdigest()}, not the pinned {f.sha256}; nothing was kept.",
                             file=f.name)


def _size(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} bytes"


def offer_text() -> tuple[str, str, str]:
    """(summary, label, description) of the record's ``thorough`` offer, in core's words."""
    summary = ("This Mac can read pages more thoroughly with dots.mocr, a vision-language OCR model that runs on "
               "Apple silicon. It finds numbers and lines in other scripts that the regular readers miss, and "
               "on Thai pages its read is chosen where the other readers back it up. It is off unless you turn it on: it needs the "
               f"pdfintake[{EXTRA}] libraries and a one-time {_size(TOTAL_BYTES)} download from Hugging Face, "
               f"and adds about {SECONDS_PER_PAGE} s a page. Now and then it adds a wrong extra word or number "
               "to the invisible text.")
    label = "Read this document again thoroughly"
    description = ("Reads every OCR page again with dots.mocr beside the regular readers. Needs thorough mode "
                   "set up first ('pdfintake thorough enable').")
    return summary, label, description


__all__ = ["ENGINE_NAME", "FILES", "REPO", "REVISION", "TOTAL_BYTES", "decline", "disable", "enable",
           "model_dir", "model_paths", "model_problems", "missing_libraries", "offer_text", "platform_problem",
           "resolve", "setting", "should_offer", "status"]
