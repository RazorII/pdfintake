"""Saved passwords: how a run opens a password-protected PDF without asking anyone (DESIGN §12.1).

A run never asks for a password (decided 2026-11-05). The user saves the passwords they use once,
one per line, in a private file: ``--passwords-file PATH``, or by default ``passwords`` in
pdfintake's config folder (``default_path()``) when that file exists. ``open_with`` tries them in
file order; when none opens a file, the caller skips it (note ``skipped_no_password``) and the batch
carries on.

The file.
  * One password per line, UTF-8 (a leading byte-order mark is ignored). Only the line break is
    removed: spaces at either end are part of the password. Empty lines are skipped, and a password
    listed twice is tried once.
  * At most ``MAX_ENTRIES`` (100) different passwords (a repeated line counts once, in ``candidates``
    and ``add`` alike) and ``MAX_BYTES`` bytes; a larger file is refused.
  * POSIX: it must be a regular file owned by the user running pdfintake, with no access for group
    or others (0600, or the stricter 0400). Anything else is refused before a byte is read as a
    password, with the usage error ``passwords_file_unsafe`` (exit 2). The checks run on the open
    file descriptor, so the file can't be swapped between the check and the read.
  * Windows: the standard library can't read an ACL. What it can check is checked: the file is a
    regular file, and its real path lies inside the user's profile folder, whose inherited ACL gives
    only the user, SYSTEM and Administrators access. The limit: an ACL someone has widened by hand,
    on the file or on a folder above it, isn't seen.

Secrecy. No password reaches an exception message, a record, a log line or ``repr()``: this module
logs nothing, ``candidates`` returns a list whose ``repr`` and ``str`` give only a count, ``check``
returns only a reason code, and every error names the file and the reason, never a line of it.
``open_with`` never says which password worked. One limit is pikepdf's: a ``Pdf`` opened with a
user password keeps it, and ``repr(pdf.encryption)`` prints it, so the caller saves the unencrypted
work copy at once (``encryption=False``, DESIGN §11.1), closes the original and never logs
``pdf.encryption``.

pdfintake never touches a keychain or password manager: a wrapper that keeps the list in one writes
it to a 0600 temporary file for the batch, passes ``--passwords-file`` and deletes the file after.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import UsageError

if TYPE_CHECKING:
    import pikepdf

FILE_NAME = "passwords"
MAX_ENTRIES = 100
MAX_BYTES = 128 * 1024            # 100 passwords of any length a PDF uses, with room to spare
UNSAFE = "passwords_file_unsafe"

_REASONS = {
    "missing": "it doesn't exist",
    "not_a_file": "it isn't a regular file",
    "unreadable": "it can't be read",
    "mode": "other users can read or change it; make it private with: chmod 600 <file>",
    "owner": "it belongs to another user",
    "outside_profile": "it isn't inside your user folder, so pdfintake can't tell who else can read it",
    "too_large": f"it is larger than {MAX_BYTES // 1024} KB",
    "too_many": f"it holds more than {MAX_ENTRIES} passwords, the most pdfintake tries",
    "full": f"it already holds {MAX_ENTRIES} passwords, the most pdfintake tries",
    "not_utf8": "it isn't UTF-8 text",
}


class Candidates(list):
    """The saved passwords, in file order. Its ``repr`` and ``str`` show only how many there are.

    A slice, ``copy()``, ``+`` and ``*`` return a ``Candidates`` too, so trimming or joining the
    list keeps the passwords hidden. ``list(c)`` and ``sorted(c)`` can't be caught and give a plain
    list: pass the object on unchanged rather than copying it.
    """

    def __repr__(self) -> str:
        n = len(self)
        return f"<{n} saved password{'' if n == 1 else 's'}>"

    __str__ = __repr__

    def __getitem__(self, index):
        got = super().__getitem__(index)
        return Candidates(got) if isinstance(index, slice) else got

    def copy(self) -> Candidates:
        return Candidates(self)

    def __add__(self, other):
        return Candidates(list.__add__(self, other))

    def __radd__(self, other):
        return Candidates(list(other) + list(self))

    def __mul__(self, n):
        return Candidates(list.__mul__(self, n))

    __rmul__ = __mul__


# --------------------------------------------------------------------------------- where it lives

def _config_root() -> Path:
    """pdfintake's config folder, derived the way ``platforms.cache_root`` derives the cache.

    ``~/Library/Application Support/pdfintake`` on macOS, ``%LOCALAPPDATA%\\pdfintake`` on Windows
    (local, not roaming, so the list never travels with a roaming profile), and
    ``$XDG_CONFIG_HOME/pdfintake`` or ``~/.config/pdfintake`` elsewhere (a relative
    ``XDG_CONFIG_HOME`` is ignored, as the XDG spec says). Never created here.
    """
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "pdfintake"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else home / "AppData" / "Local"
        return base / "pdfintake"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / "pdfintake"
    return home / ".config" / "pdfintake"


def default_path() -> Path:
    """``<config dir>/pdfintake/passwords``: the file ``candidates(None)`` reads and ``add`` writes."""
    return _config_root() / FILE_NAME


# --------------------------------------------------------------------------------- reading

def _unsafe(path: Path, reason: str) -> UsageError:
    return UsageError(UNSAFE, f"pdfintake won't use the passwords file {path}: {_REASONS[reason]}.",
                      path=str(path), reason=reason)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _inside_profile(path: Path) -> bool:
    real = os.path.normcase(os.path.realpath(path))
    home = os.path.normcase(os.path.realpath(Path.home()))
    try:
        return os.path.commonpath([real, home]) == home
    except ValueError:                       # another drive
        return False


def _check_open_file(path: Path, st: os.stat_result, *, private: bool) -> None:
    """The checks on an open file's ``fstat``. ``private`` adds the owner and mode checks."""
    if not stat.S_ISREG(st.st_mode):
        raise _unsafe(path, "not_a_file")
    if st.st_size > MAX_BYTES:
        raise _unsafe(path, "too_large")
    if not private:
        return
    if _is_windows():
        if not _inside_profile(path):
            raise _unsafe(path, "outside_profile")
        return
    if st.st_uid != os.geteuid():
        raise _unsafe(path, "owner")
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise _unsafe(path, "mode")


def _read(path: Path, *, private: bool) -> tuple[list[str], bool]:
    """The file's passwords (empty lines skipped, duplicates kept) and whether it ends in a line break."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise _unsafe(path, "missing") from None
    except OSError:
        raise _unsafe(path, "unreadable") from None
    try:
        _check_open_file(path, os.fstat(fd), private=private)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:                  # grew after the fstat
                raise _unsafe(path, "too_large")
            chunks.append(chunk)
    except OSError:
        raise _unsafe(path, "unreadable") from None
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    chunks.clear()
    try:
        text: str | None = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = None
    if text is None:
        # Raised out here, after the except block, so no UnicodeDecodeError (whose .object is the
        # file's raw bytes) is chained to it as __context__ (decided 2026-11-06).
        del raw
        raise _unsafe(path, "not_utf8")
    entries = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    return [e for e in entries if e], (not raw or raw.endswith(b"\n"))


def _unique(entries: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for e in entries:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _within_limit(path: Path, entries: list[str]) -> list[str]:
    """``entries`` without repeats, or ``too_many`` when that leaves more than ``MAX_ENTRIES``.

    The one place the limit is counted, so ``add`` can never write a list ``candidates`` refuses
    (decided 2026-11-05): a password listed twice is one password, both when reading and when adding.
    """
    unique = _unique(entries)
    if len(unique) > MAX_ENTRIES:
        raise _unsafe(path, "too_many")
    return unique


def _saved(path: Path) -> list[str]:
    """Every check a run makes on a passwords file (open, ``fstat``, size, UTF-8, the entry count),
    then its passwords without repeats. ``candidates`` and ``check`` both go through here, so they
    can't disagree about a file."""
    entries, _ = _read(path, private=True)
    return _within_limit(path, entries)


def candidates(path: Path | None) -> list[str]:
    """The saved passwords to try, in file order, without repeats (DESIGN §12.1).

    ``path`` is ``--passwords-file``; it must exist. ``None`` reads ``default_path()`` when that
    file exists and returns an empty list when it doesn't. Either way the file must pass the checks
    in the module docstring, or this raises ``UsageError`` ``passwords_file_unsafe`` (exit 2) naming
    the file and the reason, never its contents. More than ``MAX_ENTRIES`` different passwords is
    refused; a repeated line doesn't count twice.
    """
    if path is None:
        path = default_path()
        if not os.path.lexists(path):
            return Candidates()
    failure: UsageError | None = None
    try:
        return Candidates(_saved(Path(path)))
    except UsageError as exc:
        failure = UsageError(exc.code, exc.message, **exc.detail)
    # A fresh error, raised after the except block: the one caught above carries a traceback whose
    # frames (_read, _within_limit) hold the file's lines, and a caller may keep the error for a whole
    # batch. This one's traceback starts here, where no password is held (decided 2026-11-06).
    raise failure


def check(path: Path | None) -> str | None:
    """Why a run would refuse the passwords file, or None when a run would use it.

    The same open, ``fstat`` (regular file, owner and mode, or the profile folder on Windows), size,
    UTF-8 and entry-count checks as ``candidates``, through the same code. The answer is one of the
    reason codes ``passwords_file_unsafe`` carries (``missing``, ``not_a_file``, ``unreadable``,
    ``mode``, ``owner``, ``outside_profile``, ``too_large``, ``too_many``, ``not_utf8``). No password
    is returned or kept: what was read is dropped before this returns, so ``doctor`` can judge the
    file without ever holding a password (decided 2026-11-06). ``None`` as ``path`` checks
    ``default_path()``, and gives None when that file doesn't exist, as ``candidates(None)`` then
    has nothing to refuse.
    """
    if path is None:
        path = default_path()
        if not os.path.lexists(path):
            return None
    try:
        found = _saved(Path(path))
    except UsageError as exc:
        reason = exc.detail.get("reason")
        return reason if isinstance(reason, str) else "unreadable"
    found.clear()
    return None


# --------------------------------------------------------------------------------- opening

def open_with(pdf: Path, candidates: Iterable[str]) -> pikepdf.Pdf | None:
    """``pdf`` opened with the first password that works, or None when none does.

    A file that needs no password to open (not encrypted, or an edit lock only) opens on the first
    try, before any saved password is used: an edit-locked file refuses a wrong password, so trying
    the saved ones first would skip it. Then each candidate is tried in order. Which one worked is
    never recorded or returned. Only a wrong password moves on to the next; any other error (no
    such file, a damaged file) is raised, and its message never holds a password. The caller closes
    the returned ``Pdf`` (see the module docstring about ``pdf.encryption``).
    """
    import pikepdf

    tried: list[str] = []
    password: str | None = None
    opened, error = _try_open(pikepdf, pdf, "")
    if opened is None and error is None:
        for password in candidates:
            if not isinstance(password, str):
                error = TypeError("each saved password must be a str")
                break
            if not password:
                continue
            tried.append(password)
            opened, error = _try_open(pikepdf, pdf, password)
            if opened is not None or error is not None:
                break
    # Raised out here, after every except block has ended, so the error carries no __context__,
    # and its traceback starts in this frame, whose password locals are cleared first
    # (decided 2026-11-05).
    password = None
    del candidates
    if error is not None:
        if _quotes_a_password(error, tried):
            error = pikepdf.PdfError(f"{Path(pdf).name}: the file couldn't be opened")
        tried.clear()
        raise error
    tried.clear()
    return opened


def _try_open(pikepdf, pdf: Path, password: str):
    """``(Pdf, None)`` when ``password`` opens ``pdf``, ``(None, None)`` when it's wrong, and
    ``(None, error)`` for any other failure. The error comes back with its traceback, and every
    traceback in its chain, removed: pikepdf's own frames hold the password as a local."""
    try:
        return pikepdf.open(pdf, password=password), None
    except pikepdf.PasswordError:
        return None, None
    except Exception as exc:                              # noqa: BLE001 (re-raised by open_with)
        return None, _without_tracebacks(exc)


def _chain(exc: BaseException):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _without_tracebacks(exc: BaseException) -> BaseException:
    for link in _chain(exc):
        link.__traceback__ = None
    return exc


def _quotes_a_password(exc: BaseException, tried: list[str]) -> bool:
    """Whether ``exc`` or anything chained to it quotes a password that was tried."""
    for link in _chain(exc):
        try:
            text = f"{link}\n{link!r}"
        except Exception:                                 # noqa: BLE001 (an unprintable error: assume the worst)
            return True
        if any(p and p in text for p in tried):
            return True
    return False


# --------------------------------------------------------------------------------- saving

def _fsync(fd: int) -> None:
    if sys.platform == "darwin":
        import fcntl

        full = getattr(fcntl, "F_FULLFSYNC", None)
        if full is not None:
            try:
                fcntl.fcntl(fd, full)
                return
            except OSError:
                pass
    os.fsync(fd)


def add(from_file: Path, *, to: Path | None = None) -> None:
    """Append the password(s) in ``from_file`` to the saved list (``pdfintake passwords add``).

    ``from_file`` is the short-lived file the skill writes and deletes; each non-empty line in it is
    one password. ``to`` defaults to ``default_path()``. The list is created at 0600 (its folder at
    0700) when it doesn't exist; an existing list must pass the same checks ``candidates`` applies,
    or nothing is written. A password already saved is skipped silently, so adding one twice is
    harmless. Adding past ``MAX_ENTRIES`` different passwords is refused (``passwords_file_unsafe``,
    reason ``full``), counted as ``candidates`` counts them. A list this call creates and then can't
    finish (a failed check or write) is removed again. Nothing is printed, logged or returned.
    """
    target = Path(to) if to is not None else default_path()
    new, _ = _read(Path(from_file), private=False)
    if not new:
        return
    create = not os.path.lexists(target)
    if create:
        existing, ends_clean = [], True
    else:
        existing, ends_clean = _read(target, private=True)
    saved = _within_limit(target, existing)
    known = set(saved)
    fresh = [p for p in _unique(new) if p not in known]
    if not fresh:
        return
    if len(saved) + len(fresh) > MAX_ENTRIES:            # the count candidates() applies
        raise _unsafe(target, "full")
    if create and _is_windows() and not _inside_profile(target):
        raise _unsafe(target, "outside_profile")      # before anything exists on disk

    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # A new list is created exclusively and an existing one is never re-created, so a file that
    # appears or disappears between the read above and this open is refused, not overwritten.
    flags = (os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0))
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(target, flags, 0o600)
    except OSError:
        raise _unsafe(target, "unreadable") from None
    failure: UsageError | None = None
    try:
        st = os.fstat(fd)
        if create and not _is_windows():
            os.fchmod(fd, 0o600)                  # a new file: 0600 whatever the umask left
            st = os.fstat(fd)
        _check_open_file(target, st, private=True)
        data = ("" if ends_clean else "\n") + "".join(p + "\n" for p in fresh)
        view = memoryview(data.encode("utf-8"))
        while view:
            written = os.write(fd, view)
            view = view[written:]
        _fsync(fd)
    except UsageError as exc:
        failure = exc
    except OSError:
        failure = _unsafe(target, "unreadable")
    finally:
        os.close(fd)
    if failure is not None:
        # A list this call created and couldn't finish is removed, so a failed add never leaves a
        # file behind that every later run would refuse (decided 2026-11-05).
        if create:
            try:
                os.unlink(target)
            except OSError:
                pass
        failure.__traceback__ = None
        raise failure
