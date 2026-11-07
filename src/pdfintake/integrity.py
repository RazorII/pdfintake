"""Hashes, the input seal and the no-subprocess guard.

* ``verify_file`` is how every model, font and canary is checked each time it is loaded. It and
  ``sha256_file`` raise ``IntegrityError`` naming the file (``file_missing`` or ``unreadable``)
  when it can't be opened or read, including when it vanishes between the check and the read.
* ``InputSeal`` records each input's SHA-256, size and mtime before any work, so a file that
  changes during a run, or between a run and ``apply``, is caught (``InputChanged``, exit 5).
* ``no_subprocess()`` makes any attempt to start a process fail while it is active, through an
  audit hook installed once per process. ``flow.run`` and ``flow.apply`` hold it for the whole
  run, so a dependency can't call Ghostscript, a Homebrew ``tesseract`` or ``jpegtran`` either.
  The few process starts that raise no audit event (multiprocessing's spawn and forkserver) are
  wrapped so that they refuse too.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import hashlib
import os
import re
import stat
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InputChanged, IntegrityError, StaleRecord, UsageError

# The one process-start module src/ may import, and only to wrap it (see no_subprocess below):
# a literal import, which test_hygiene allows in this file alone, rather than a name lookup that
# slips past the scan. Importing it starts nothing. Windows has no such module. — decided 2026-11-05
try:
    import _posixsubprocess
except ImportError:                                       # Windows
    _posixsubprocess = None  # type: ignore[assignment]

_CHUNK = 1 << 20
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- hashes

def _unreadable(path: Path, exc: OSError) -> IntegrityError:
    """The IntegrityError naming ``path`` for an OSError raised opening or reading it."""
    if isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT:
        return IntegrityError("file_missing", f"{path} is missing.", path=str(path))
    return IntegrityError("unreadable", f"{path} can't be read ({exc.strerror or type(exc).__name__}).",
                          path=str(path))


def sha256_file(path: Path) -> str:
    """The SHA-256 of ``path``. IntegrityError naming it (file_missing or unreadable) on an OSError."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
    except OSError as exc:
        raise _unreadable(Path(path), exc) from None
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_file(path: Path, sha256: str, size: int | None = None) -> None:
    """IntegrityError naming ``path`` unless it exists with this size (when given) and SHA-256.

    A file that disappears or becomes unreadable after the stat (a concurrent run discarding a
    corrupt pack folder) is ``file_missing`` or ``unreadable`` too, never a bare OSError.
    """
    path = Path(path)
    want = (sha256 or "").strip().lower()
    if not _SHA_RE.match(want):
        raise IntegrityError("bad_expected_hash", f"{path}: the expected SHA-256 {sha256!r} isn't a SHA-256.",
                             path=str(path))
    try:
        st = os.stat(path)
    except OSError as exc:
        raise _unreadable(path, exc) from None
    if not stat.S_ISREG(st.st_mode):
        raise IntegrityError("not_a_file", f"{path} isn't a regular file.", path=str(path))
    if size is not None and st.st_size != size:
        raise IntegrityError("size_mismatch", f"{path} is {st.st_size} bytes; it should be {size}.",
                             path=str(path), expected=size, actual=st.st_size)
    got = sha256_file(path)
    if got != want:
        raise IntegrityError("hash_mismatch", f"{path} has SHA-256 {got}; it should be {want}.",
                             path=str(path), expected=want, actual=got)


# --------------------------------------------------------------------------- the input seal

def _stat_input(path: Path) -> os.stat_result:
    try:
        st = os.stat(path)
    except OSError:
        raise UsageError("input_missing", f"The input {path} doesn't exist or can't be read.", path=str(path)) from None
    if not stat.S_ISREG(st.st_mode):
        raise UsageError("input_not_file", f"The input {path} isn't a regular file.", path=str(path))
    return st


@dataclass
class InputSeal:
    paths: list[Path]
    sha256: list[str]
    sizes: list[int]
    mtimes_ns: list[int]

    @classmethod
    def take(cls, paths: Sequence[Path]) -> InputSeal:
        """Seal the inputs. InputChanged if one changes while it is being hashed."""
        seal = cls([], [], [], [])
        for p in paths:
            path = Path(os.path.abspath(Path(p).expanduser()))
            before = _stat_input(path)
            try:
                digest = sha256_file(path)
            except IntegrityError:
                raise UsageError("input_missing", f"The input {path} doesn't exist or can't be read.",
                                 path=str(path)) from None
            after = _stat_input(path)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise InputChanged("input_changed", f"The input {path} changed while it was being read; run again.",
                                   path=str(path))
            seal.paths.append(path)
            seal.sha256.append(digest)
            seal.sizes.append(after.st_size)
            seal.mtimes_ns.append(after.st_mtime_ns)
        return seal

    def check(self) -> None:
        """InputChanged naming the file unless every input is as sealed.

        Size and mtime are compared first, for every file, because they are cheap; the SHA-256 is
        then always compared too, so an edit that keeps both is still caught.
        """
        for path, size, mtime in zip(self.paths, self.sizes, self.mtimes_ns):
            try:
                st = os.stat(path)
            except OSError:
                raise InputChanged("input_missing", f"The input {path} is gone since it was sealed.",
                                   path=str(path)) from None
            if st.st_size != size:
                raise InputChanged("input_changed", f"The input {path} changed size since it was sealed.",
                                   path=str(path), field="size")
            if st.st_mtime_ns != mtime:
                raise InputChanged("input_changed", f"The input {path} was modified since it was sealed.",
                                   path=str(path), field="mtime")
        for path, digest in zip(self.paths, self.sha256):
            try:
                got = sha256_file(path)
            except (OSError, IntegrityError):
                raise InputChanged("input_missing", f"The input {path} can't be read since it was sealed.",
                                   path=str(path)) from None
            if got != digest:
                raise InputChanged("input_changed", f"The input {path} has different contents since it was sealed.",
                                   path=str(path), field="sha256")

    def to_json(self) -> dict[str, Any]:
        return {"paths": [str(p) for p in self.paths], "sha256": list(self.sha256),
                "sizes": list(self.sizes), "mtimes_ns": list(self.mtimes_ns)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> InputSeal:
        """The seal ``to_json`` wrote. StaleRecord if it is malformed."""
        try:
            paths = [Path(p) for p in data["paths"]]
            shas = [str(s) for s in data["sha256"]]
            sizes = [int(s) for s in data["sizes"]]
            mtimes = [int(m) for m in data["mtimes_ns"]]
        except (KeyError, TypeError, ValueError):
            raise StaleRecord("seal_unreadable", "The record's input seal can't be read; run again.") from None
        ok = (len(paths) == len(shas) == len(sizes) == len(mtimes)
              and all(_SHA_RE.match(s) for s in shas)
              and all(p.is_absolute() for p in paths)
              and all(s >= 0 for s in sizes))
        if not ok:
            raise StaleRecord("seal_unreadable", "The record's input seal can't be read; run again.")
        return cls(paths, shas, sizes, mtimes)


# --------------------------------------------------------------------------- no subprocess

_GUARD_LOCK = threading.Lock()
_GUARD = {"installed": False, "depth": 0}
# The audit events that start a process. os.exec covers every exec*, os.spawn every spawn*,
# os.posix_spawn both posix_spawn and posix_spawnp. Forks are refused too: nothing in a run forks.
_PROCESS_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
    "os.fork", "os.forkpty", "_winapi.CreateProcess",
})
# Process starts that raise no audit event, so the hook can't see them. multiprocessing's spawn and
# forkserver start methods (and ProcessPoolExecutor, which uses them) start a new interpreter through
# _posixsubprocess.fork_exec, which they look up at call time; a fork server started before the guard
# forks a new child on request through forkserver.connect_to_new_process. Each is replaced by a wrapper
# that refuses while a guard is active and otherwise calls the original. (On Windows, multiprocessing
# goes through _winapi.CreateProcess, which the hook already refuses.) subprocess keeps its own
# reference to fork_exec from import time, and subprocess.Popen is audited, so it is unaffected.
# _posixsubprocess is imported above, so its wrapper is in place from the first guard on, before
# anything can look fork_exec up; the fork server module is wrapped only once something else has
# imported it, because a fork server can't be started inside the guard (that needs fork_exec).
_FORKSERVER = "multiprocessing.forkserver"
_WRAPPED_MARK = "__pdfintake_no_subprocess__"


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if _GUARD["depth"] and event in _PROCESS_EVENTS:
        raise PermissionError(f"pdfintake starts no processes during a run ({event} refused)")


def _refusing(what: str, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _GUARD["depth"]:
            raise PermissionError(f"pdfintake starts no processes during a run ({what} refused)")
        return original(*args, **kwargs)

    setattr(wrapper, _WRAPPED_MARK, True)
    return wrapper


def _wrap(module: Any, module_name: str, attr: str) -> None:
    original = getattr(module, attr, None) if module is not None else None
    if original is None or getattr(original, _WRAPPED_MARK, False):
        return
    setattr(module, attr, _refusing(f"{module_name}.{attr}", original))


def _wrap_unaudited_starts() -> None:
    """Put the refusing wrapper around each unaudited process start that is loaded."""
    _wrap(_posixsubprocess, "_posixsubprocess", "fork_exec")
    _wrap(sys.modules.get(_FORKSERVER), _FORKSERVER, "connect_to_new_process")


@contextlib.contextmanager
def no_subprocess() -> Iterator[None]:
    """Refuse every process start in this process while any ``no_subprocess()`` is active."""
    with _GUARD_LOCK:
        if not _GUARD["installed"]:
            sys.addaudithook(_audit)
            _GUARD["installed"] = True
        _wrap_unaudited_starts()
        _GUARD["depth"] += 1
    try:
        yield
    finally:
        with _GUARD_LOCK:
            _GUARD["depth"] -= 1
