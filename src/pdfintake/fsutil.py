"""Paths and writes: where an output goes, the one writer into user folders, and the cache folders.

``atomic_write`` is the only function that writes into a folder the user owns, and only
``flow.commit`` and ``flow.apply`` call it (tests/test_write_paths.py checks that statically).
``guard_output`` decides whether a destination is allowed at all; it is called when the output
name is chosen and again right before the write, and ``atomic_write`` itself never replaces a file
unless told to (``overwrite``), so a file that appears after the guard is never lost.
``sweep_partial`` is the one other function that touches a user's folder: it removes the partial
file a killed writer left beside a destination (and puts back an output a killed compare-and-swap
had set aside), never anything else.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import stat
import sys
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from . import platforms
from .errors import PdfIntakeError, StaleRecord, UsageError

PARTIAL_SUFFIX = ".pdfintake-partial"
ASIDE_SUFFIX = ".pdfintake-replaced"   # the output a compare-and-swap is checking
NAME_MAX = 255              # the longest file name APFS, NTFS and ext4 take (in _name_length units)
WORK_KINDS = ("work", "staging", "records", "canary")
PRUNABLE_KINDS = ("work", "staging")     # DESIGN §3.5, §4.4: only these are pruned after 14 days
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}$")
_CHUNK = 1 << 20
_BESIDE_LIMIT = 999
_RENAME_EXCL = 0x00000004   # macOS <stdio.h>: renamex_np fails with EEXIST instead of replacing the target
# renamex_np's answers for "this filesystem can't do an exclusive rename"; the next way is tried.
_NO_RENAME_EXCL = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS})
# link()'s answers for "this filesystem has no hard links" (exFAT, FAT, some SMB shares).
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV,
                            errno.EMLINK})
_RENAMEX: list[object] = []  # renamex_np, looked up once (or None where there is none)


# --------------------------------------------------------------------------- the one writer

def _fsync_file(fd: int) -> None:
    """Flush a file to stable storage. On macOS plain fsync stops at the drive's cache."""
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


def _fsync_dir(folder: Path) -> None:
    """Make a rename durable. Windows can't open a directory for this, and doesn't need it."""
    if sys.platform == "win32":
        return
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _name_length(name: str) -> int:
    """A file name's length as the filesystem counts it: UTF-16 units on APFS and NTFS, bytes elsewhere."""
    if sys.platform in ("darwin", "win32"):
        return len(name.encode("utf-16-le", "surrogatepass")) // 2
    return len(os.fsencode(name))


def _partial_path(dest: Path) -> Path:
    """``<dest name>.pdfintake-partial`` beside ``dest``, or a short hidden name when that one is too long."""
    name = dest.name + PARTIAL_SUFFIX
    if _name_length(name) > NAME_MAX:
        digest = hashlib.sha256(dest.name.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        name = "." + digest + PARTIAL_SUFFIX
    return dest.with_name(name)


def _write_conflict(dest: Path) -> UsageError:
    return UsageError("write_conflict",
                      f"Another pdfintake run is writing {dest}; nothing was written. Wait for it, or choose another name.",
                      path=str(dest))


def _ids(st: os.stat_result) -> tuple[int, int]:
    return st.st_dev, st.st_ino


def _try_lock(fd: int) -> bool:
    """An exclusive, non-blocking lock on an open file; False while another open file holds one.

    The lock lives as long as the open file, so a writer that is killed releases it. Windows needs
    none: a file another process has open can't be removed there, which is the same signal.
    """
    if sys.platform == "win32":
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return False
        raise
    return True


def _appeared(dest: Path) -> UsageError:
    """The usage error for a file that took ``dest`` after ``guard_output`` found the name free."""
    return UsageError("output_exists",
                      f"The output {dest} appeared while pdfintake was writing it; that file was left as it is "
                      "and nothing was written.", path=str(dest))


def _act_if_unlocked(path: Path, act: Any, *, own_only: bool) -> bool | None:
    """Run ``act()`` while holding the lock on ``path``, when it is a regular file no live writer holds (POSIX).

    ``act``'s answer when it ran; None when a live writer holds the lock (the file is left alone);
    False when nothing is there, it isn't a regular file (a link or a FIFO is never followed, opened
    for long or touched), ``own_only`` and another user owns it, or the name changed while it was
    checked. The lock is taken before ``act`` runs, and the name must still be the file that was
    locked.
    """
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return False
    except OSError as e:
        if e.errno == errno.ELOOP:                      # a link, with O_NOFOLLOW
            return False
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return False
        if own_only and hasattr(os, "geteuid") and st.st_uid != os.geteuid():
            return False
        if not _try_lock(fd):
            return None
        try:
            same = _ids(st) == _ids(os.lstat(path))
        except FileNotFoundError:
            return False
        if not same:
            return False
        return bool(act())
    finally:
        os.close(fd)


def _unlink_if_unlocked(partial: Path, *, own_only: bool) -> bool | None:
    """Unlink ``partial`` when it is a regular file no live writer holds (POSIX); see ``_act_if_unlocked``.

    True when it was removed, None when a live writer holds it, False otherwise.
    """
    def unlink() -> bool:
        partial.unlink(missing_ok=True)
        return True

    return _act_if_unlocked(partial, unlink, own_only=own_only)


def _name_taken(partial: Path, dest: Path) -> UsageError:
    return UsageError("write_conflict",
                      f"pdfintake writes {dest} through the name {partial.name}, and what is there isn't a file "
                      "pdfintake left; it was left as it is and nothing was written.", path=str(dest))


def _remove_stale_partial(partial: Path, dest: Path) -> None:
    """Remove a partial file a killed run left behind, by the sweep's rule (decided 2026-11-06).

    Only a regular file owned by this user that no live writer holds is removed: what
    ``sweep_partial`` removes, no more. pdfintake never makes a link, a FIFO or a socket there, and
    another user's file in a shared folder isn't this run's leftover, so anything else at that name
    is left as it is and the write stops with UsageError ``write_conflict``; so does a partial file
    a live writer holds.
    """
    try:
        st = os.lstat(partial)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(st.st_mode):
        raise _name_taken(partial, dest)                 # a link, a FIFO: never followed or removed
    if sys.platform == "win32":
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            raise _write_conflict(dest) from None
        return
    removed = _unlink_if_unlocked(partial, own_only=True)
    if removed is None:
        raise _write_conflict(dest)
    if removed is False and os.path.lexists(partial):  # another user's file, or the name changed
        raise _name_taken(partial, dest)


def _is_one_of(path: Path, paths: Sequence[Path]) -> bool:
    """``path`` is one of ``paths``: the same file, or a path that differs only in case or Unicode form."""
    key = _fold(path)
    return any(_same_file(Path(p), path) or _fold(p) == key for p in paths)


def sweep_partial(dest: Path, *, keep: Sequence[Path] = ()) -> bool:
    """Remove the partial file a killed writer left beside ``dest``; True when one was removed.

    For the next run's orphan sweep (``flow._sweep_orphans``, decided 2026-11-06): a run killed with
    SIGKILL between writing its partial file and renaming it can't remove that file itself, and it
    sits in the user's folder until something writes the same name again. ``dest`` is an absolute
    output path the killed run noted before it began writing.

    Only ``dest``'s own partial name (``<name>.pdfintake-partial``, or the short hidden
    ``.<hash>.pdfintake-partial`` for a long name) is looked at; ``dest`` and every other file are
    never touched. It is removed only when it is a regular file (a link or anything else is left
    alone, never followed), owned by this user, and no live writer holds its lock: the lock is taken
    before the unlink, and the name must still be the file that was locked. A partial file a live
    writer holds gives False, and so does every error: a sweep never raises. On Windows, where no
    lock is taken, a file another process has open can't be removed, which is the same answer.
    The one gap: a writer that has just created its partial file and not yet locked it (a few
    instructions apart) could lose it to a sweep; that writer then stops with ``write_conflict``
    at its pre-rename check, so nothing wrong is ever written.

    ``keep`` is the paths the sweeping run was itself given: a file at the partial name that is one
    of them is that run's input, not a leftover, and is never touched (decided 2026-11-06).

    A killed compare-and-swap (``atomic_write`` with ``expect_sha256``) can also have left the
    output it set aside (``_aside_path``) with ``dest`` free; the sweep puts it back, under the
    same rules (a regular file of this user's that no live writer holds). With ``dest`` taken it is
    left where it is, never removed: it may be the only copy of a version someone saved.
    """
    try:
        dest = Path(dest)
        if not dest.is_absolute() or not dest.name or "\x00" in str(dest):
            return False
        _restore_aside(dest, keep)
        partial = _partial_path(dest)
        if keep and _is_one_of(partial, keep):
            return False
        if sys.platform == "win32":
            if not stat.S_ISREG(os.lstat(partial).st_mode):
                return False
            partial.unlink()
            return True
        return _unlink_if_unlocked(partial, own_only=True) is True
    except (OSError, ValueError, PdfIntakeError):
        return False


def _still_ours(partial: Path, mine: tuple[int, int]) -> bool:
    try:
        return _ids(os.lstat(partial)) == mine
    except OSError:
        return False


def _renamex_np() -> Any:
    """libc's ``renamex_np`` through ctypes on macOS, or None (another system, or not found)."""
    if sys.platform != "darwin":
        return None
    if not _RENAMEX:
        fn: Any = None
        try:
            import ctypes

            fn = ctypes.CDLL(None, use_errno=True).renamex_np
            fn.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            fn.restype = ctypes.c_int
        except (OSError, AttributeError):
            fn = None
        _RENAMEX.append(fn)
    return _RENAMEX[0]


def _rename_excl(src: Path, dst: Path) -> int | None:
    """``renamex_np(src, dst, RENAME_EXCL)``: 0 when renamed, the errno when it failed (EEXIST:
    ``dst`` exists), None where there is no ``renamex_np``."""
    fn = _renamex_np()
    if fn is None:
        return None
    import ctypes

    ctypes.set_errno(0)
    if fn(os.fsencode(src), os.fsencode(dst), _RENAME_EXCL) == 0:
        return 0
    return ctypes.get_errno() or errno.EIO


def _publish(partial: Path, dest: Path, *, overwrite: bool) -> None:
    """Rename the finished partial file to ``dest`` (see ``atomic_write`` for the order of ways)."""
    if overwrite:
        os.replace(partial, dest)
        return
    if sys.platform == "win32":
        try:
            os.rename(partial, dest)                     # Windows' rename never replaces a file
        except FileExistsError:
            raise _appeared(dest) from None
        return
    err = _rename_excl(partial, dest)
    if err == 0:
        return
    if err == errno.EEXIST:
        raise _appeared(dest)
    if err is not None and err not in _NO_RENAME_EXCL:
        raise OSError(err, os.strerror(err), str(partial), None, str(dest))
    try:
        os.link(partial, dest)                           # fails with EEXIST, never replaces
    except FileExistsError:
        raise _appeared(dest) from None
    except OSError as e:
        if e.errno not in _NO_HARD_LINKS:
            raise
    else:
        try:
            os.unlink(partial)                           # dest keeps the bytes; this was a second name
        except OSError:
            pass
        return
    # No exclusive rename and no hard links on this filesystem: check once more, right before the
    # rename. What can still slip in is a file made in the microseconds between the two calls.
    if os.path.lexists(dest):
        raise _appeared(dest)
    os.replace(partial, dest)


def _aside_path(dest: Path) -> Path:
    """The short hidden name a compare-and-swap moves the output it replaces to while it checks it."""
    digest = hashlib.sha256(dest.name.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return dest.with_name("." + digest + ASIDE_SUFFIX)


def _put_back(aside: Path, dest: Path) -> bool:
    """Rename ``aside`` to ``dest`` without ever replacing a file there; True when ``dest`` holds it."""
    try:
        _publish(aside, dest, overwrite=False)
        return True
    except FileNotFoundError:
        return False
    except (UsageError, OSError):
        return False


def _restore_aside(dest: Path, keep: Sequence[Path] = ()) -> None:
    """Put back an output a killed compare-and-swap set aside, when ``dest`` is free; never raises."""
    try:
        aside = _aside_path(dest)
        if not os.path.lexists(aside) or os.path.lexists(dest) or (keep and _is_one_of(aside, keep)):
            return
        if sys.platform == "win32":
            if stat.S_ISREG(os.lstat(aside).st_mode):
                _put_back(aside, dest)
            return
        _act_if_unlocked(aside, lambda: not os.path.lexists(dest) and _put_back(aside, dest), own_only=True)
    except (OSError, ValueError, PdfIntakeError):
        pass


def _set_aside(dest: Path, aside: Path) -> None:
    """Move ``dest`` to ``aside`` in one rename that never replaces a file at ``aside``.

    One rename, never a link and an unlink: a file saved at ``dest`` between those two would be lost.
    """
    err = _rename_excl(dest, aside)
    if err == 0:
        return
    if err == errno.EEXIST:
        raise _write_conflict(dest)
    if err == errno.ENOENT:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(dest))
    if err is not None and err not in _NO_RENAME_EXCL:
        raise OSError(err, os.strerror(err), str(dest), None, str(aside))
    # No exclusive rename here. The aside name is pdfintake's own, and this writer holds dest's
    # partial file locked, so no other pdfintake writer is publishing to dest at this moment.
    if os.path.lexists(aside):
        raise _write_conflict(dest)
    try:
        os.rename(dest, aside)                           # Windows refuses an existing target
    except FileExistsError:
        raise _write_conflict(dest) from None


def _hash_locked(path: Path) -> tuple[int | None, str | None]:
    """``(fd, sha256)`` of the regular file at ``path``, opened without following a link and locked
    (the descriptor stays open, and the lock held, until the caller closes it); ``(None, None)``
    when it isn't a regular file."""
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))
    try:
        fd = os.open(path, flags)
    except OSError as e:
        if e.errno == errno.ELOOP:
            return None, None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or not _try_lock(fd):
            os.close(fd)
            return None, None
        h = hashlib.sha256()
        while chunk := os.read(fd, _CHUNK):
            h.update(chunk)
    except BaseException:
        os.close(fd)
        raise
    return fd, h.hexdigest()


def _output_changed(dest: Path, message: str) -> StaleRecord:
    return StaleRecord("output_changed", message, path=str(dest))


def _replace_if_unchanged(partial: Path, dest: Path, sha256: str) -> None:
    """Publish ``partial`` as ``dest`` only if ``dest`` still holds the bytes with ``sha256`` (decided 2026-11-06).

    ``dest`` is moved aside to ``_aside_path(dest)`` in one rename, hashed there, and only when it
    matches is ``partial`` published, by the same exclusive rename as a new file: a file saved at
    ``dest`` while the old one was aside is never replaced (UsageError ``output_exists``; the old
    output, which that save would have replaced anyway, is then removed). When it doesn't match,
    it is put back and this raises StaleRecord ``output_changed`` (exit 5). So a save that lands on
    ``dest`` at any moment before the publish is kept, whatever it took to build and copy the new
    file.
    """
    aside = _aside_path(dest)
    if os.path.lexists(aside):
        _restore_aside(dest)                             # a killed run's; put back when dest is free
        if os.path.lexists(aside):
            raise UsageError("write_conflict",
                             f"A file pdfintake set aside while replacing {dest} earlier ({aside.name}) is still "
                             "beside it; it was left as it is and nothing was written.", path=str(dest))
    try:
        _set_aside(dest, aside)
    except FileNotFoundError:
        raise _output_changed(dest, f"The output {dest} is gone since the record was written; "
                                    "nothing was written.") from None
    fd: int | None = None
    try:
        fd, digest = _hash_locked(aside)
        if digest != sha256:
            if fd is not None:
                os.close(fd)
                fd = None
            if _put_back(aside, dest):
                raise _output_changed(dest, f"The output {dest} changed since the record was written; it was "
                                            "left as it is and nothing was written.")
            raise _output_changed(dest, f"The output {dest} changed since the record was written and was saved "
                                        f"again while pdfintake checked it; the earlier version is kept as "
                                        f"{aside.name} beside it, and nothing was written.")
        try:
            _publish(partial, dest, overwrite=False)
        except UsageError:
            # Saved at dest while the old output was aside: that save is the newer file and stays.
            # The old output is what it would have replaced, so it goes.
            os.close(fd)                                 # type: ignore[arg-type]
            fd = None
            aside.unlink(missing_ok=True)
            raise
        os.close(fd)                                     # type: ignore[arg-type]
        fd = None
        try:
            aside.unlink()
        except OSError:
            pass
    except StaleRecord:
        raise
    except BaseException:
        if fd is not None:
            os.close(fd)
            fd = None
        if os.path.lexists(aside):
            _put_back(aside, dest)
        raise
    finally:
        if fd is not None:
            os.close(fd)


def _finish(partial: Path, dest: Path, *, overwrite: bool, expect_sha256: str | None) -> None:
    if expect_sha256 is not None:
        _replace_if_unchanged(partial, dest, expect_sha256)
    else:
        _publish(partial, dest, overwrite=overwrite)


def _identity_now(fd: int) -> tuple[int, int] | None:
    try:
        return _ids(os.fstat(fd))
    except OSError:
        return None


def _discard(partial: Path, mine: tuple[int, int] | None) -> bool:
    """Remove ``partial`` when it is still this writer's file (``mine``); never raises.

    False only when it is this writer's file and the unlink was refused (a filesystem that won't
    remove a file that is open, such as some network shares), so the caller can try once it is closed.
    """
    if mine is None or not _still_ours(partial, mine):
        return True
    try:
        partial.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def atomic_write(built: Path, dest: Path, *, overwrite: bool = False, expect_sha256: str | None = None) -> None:
    """Copy ``built`` to ``dest`` so that ``dest`` is either untouched or complete, never partial.

    The bytes go to ``<dest name>.pdfintake-partial`` in dest's folder (a short hidden
    ``.<hash>.pdfintake-partial`` when that name would be too long), are flushed to disk, and the
    partial file is then renamed to ``dest``. On any error this writer's partial file is removed and
    ``dest`` is as it was.

    ``overwrite`` True (``--overwrite``) renames with ``os.replace``. ``overwrite`` False never
    replaces anything: a file that took the name after ``guard_output`` checked it (another program
    saving, a second batch) is left as it is, the partial file is removed, and this raises
    UsageError ``output_exists`` (decided 2026-11-06). The rename is exclusive, in this order:

    * macOS: ``renamex_np`` with ``RENAME_EXCL`` (through ctypes), atomic and exclusive;
    * where that is missing or the filesystem refuses it (other POSIX systems; some SMB and exFAT
      volumes): ``os.link(partial, dest)``, which fails when ``dest`` exists, then the partial
      name is unlinked;
    * Windows: ``os.rename``, which never replaces an existing file;
    * a filesystem with neither (exFAT, FAT and some SMB shares have no hard links): ``dest`` is
      checked once more immediately before ``os.replace``. This last way is not exclusive: a file
      made between that check and the rename, a window of microseconds, would be replaced.

    ``expect_sha256`` (``apply --lang`` replacing pdfintake's own output) is a compare-and-swap:
    ``dest`` is replaced only while it still holds those bytes, checked at the moment of the
    rename, not before the copy (``_replace_if_unchanged``; StaleRecord ``output_changed`` when it
    doesn't). It can't be combined with ``overwrite``.

    The writer holds a lock on its partial file until the rename. A partial file left by a killed
    process has no lock, so the next write to the same name removes it, and so does
    ``sweep_partial``; only a regular file of this user's is ever removed that way. One that a live
    writer holds, and anything that isn't such a file, is never touched: the writer stops with
    UsageError ``write_conflict``. Right before the rename the partial path must still be this
    writer's own file, or it stops the same way, so two runs writing one name can never rename the
    other's half-written bytes over ``dest``. That check reads the open file's identity at that
    moment, not at its creation: FAT and exFAT give an empty file a placeholder inode number and
    change it once its first cluster is written (measured 2026-11-06).

    On an error the partial file is removed the same way, compared and unlinked while it is still
    open and locked (POSIX): on macOS 15's FAT32 the removal made after closing it left the partial
    file behind (CI, 2026-11-07). Only a filesystem that refuses to remove an open file gets a
    second try once it is closed. Windows can't remove a file that is open, and can't rename one
    either, so there the partial file is closed first: its identity is read while it is open and
    compared with the name once more right before the rename, or the removal.
    """
    if overwrite and expect_sha256 is not None:
        raise ValueError("atomic_write: overwrite and expect_sha256 can't be combined")
    built, dest = Path(built), Path(dest)
    partial = _partial_path(dest)
    expected = os.stat(built).st_size
    _remove_stale_partial(partial, dest)
    drop: tuple[int, int] | None = None              # the partial file to remove once it is closed
    closed: tuple[int, int] | None = None            # Windows: published after it is closed
    try:
        with open(built, "rb") as src:
            try:
                out = open(partial, "xb")
            except FileExistsError:
                raise _write_conflict(dest) from None
            with out:
                fd = out.fileno()
                try:
                    if not _try_lock(fd):
                        raise _write_conflict(dest)
                    while chunk := src.read(_CHUNK):
                        out.write(chunk)
                    out.flush()
                    _fsync_file(fd)
                    st = os.fstat(fd)
                    if st.st_size != expected:
                        raise PdfIntakeError("write_incomplete",
                                             f"Writing {dest} stopped after {st.st_size} of {expected} bytes; "
                                             "nothing was written.",
                                             path=str(dest), written=st.st_size, expected=expected)
                    mine = _ids(st)                  # now, with the data written (FAT, exFAT)
                    if sys.platform == "win32":      # Windows can't rename an open file
                        closed = mine
                    else:                            # renamed while still open and locked
                        if not _still_ours(partial, mine):
                            raise _write_conflict(dest)
                        _finish(partial, dest, overwrite=overwrite, expect_sha256=expect_sha256)
                except BaseException:
                    now = _identity_now(fd)
                    if sys.platform == "win32":      # an open file can't be removed there: once closed
                        drop = now
                    elif not _discard(partial, now):  # now, while open and locked; refused: once closed
                        drop = now
                    raise
        if closed is not None:
            drop = closed
            if not _still_ours(partial, closed):
                raise _write_conflict(dest)
            _finish(partial, dest, overwrite=overwrite, expect_sha256=expect_sha256)
            drop = None
    except BaseException:
        _discard(partial, drop)
        raise
    _fsync_dir(dest.parent)


# --------------------------------------------------------------------------- choosing and guarding outputs

def _fold(path: Path | str) -> str:
    """A comparison key for a path on a filesystem that may ignore case and Unicode normal form."""
    resolved = os.path.normcase(str(Path(path).expanduser().resolve()))
    return unicodedata.normalize("NFC", resolved).casefold()


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _inside(child: str, parent: str) -> bool:
    """Both are ``_fold`` keys."""
    if child == parent:
        return True
    prefix = parent if parent.endswith(os.sep) else parent + os.sep
    return child.startswith(prefix)


def guard_output(inputs: Sequence[Path], dest: Path, *, overwrite: bool) -> None:
    """UsageError unless ``dest`` may be written.

    Refused, whatever ``overwrite`` says: ``dest`` is one of the inputs (the same file through a
    symlink or hard link, or a path that differs only in case or Unicode normal form, so Scan.pdf
    and scan.pdf collide on APFS and NTFS); an input sits at a name the write goes through (dest's
    partial name, or the aside name of a compare-and-swap), which the writer would clear first;
    ``dest`` lies inside the cache; ``dest`` is a folder;
    its folder is missing or not writable; its name is longer than a file name can be (255 UTF-16
    units on APFS and NTFS, 255 bytes elsewhere). Refused without ``overwrite``: ``dest`` exists.
    """
    dest = Path(dest)
    if _name_length(dest.name) > NAME_MAX:
        raise UsageError("output_name_too_long",
                         f"The output name {dest.name!r} is longer than a file name can be; choose a shorter one.",
                         path=str(dest))
    dest_key = _fold(dest)
    for src in inputs:
        src = Path(src)
        if _same_file(src, dest) or _fold(src) == dest_key:
            raise UsageError("output_is_input",
                             f"The output {dest} would replace the input {src}; pdfintake never changes its input.",
                             path=str(dest), input=str(src))
    # The writer clears the partial name before it writes, and a compare-and-swap uses the aside
    # name, so an input under either name would be removed (decided 2026-11-06).
    for side in (_partial_path(dest), _aside_path(dest)):
        hit = next((Path(s) for s in inputs if _is_one_of(side, [Path(s)])), None)
        if hit is not None:
            raise UsageError("output_is_input",
                             f"Writing {dest} goes through the name {side.name}, which is the input {hit}; "
                             "pdfintake never changes its input.", path=str(dest), input=str(hit))
    if _inside(dest_key, _fold(platforms.cache_root())):
        raise UsageError("output_in_cache",
                         f"The output {dest} is inside pdfintake's cache, which is pruned; choose another folder.",
                         path=str(dest))
    if dest.is_dir():
        raise UsageError("output_is_folder", f"The output {dest} is a folder; give a file name.", path=str(dest))
    folder = dest.parent
    if not folder.is_dir():
        raise UsageError("output_folder_missing", f"The output folder {folder} doesn't exist.", path=str(dest))
    if not os.access(folder, os.W_OK | os.X_OK):
        raise UsageError("output_folder_not_writable", f"The output folder {folder} isn't writable.", path=str(dest))
    if os.path.lexists(dest) and not overwrite:
        raise UsageError("output_exists",
                         f"The output {dest} already exists; pass --overwrite to replace it.", path=str(dest))


def output_path(inputs: Sequence[Path], *, out: Path | None, outdir: Path | None, suffix: str,
                overwrite: bool) -> Path:
    """The guarded output path for one output document.

    ``out`` is used as given. Otherwise the name is ``<first input's stem><suffix>.pdf`` in ``outdir``
    or, without it, beside the first input; ``suffix`` is the text after the stem, such as
    ``" (searchable)"`` or ``""`` (a suffix already ending in ``.pdf`` gets no second one).
    """
    if not inputs:
        raise UsageError("no_inputs", "No input files were given.")
    if out is not None and outdir is not None:
        raise UsageError("out_and_outdir", "Give either -o or --outdir, not both.")
    first = Path(inputs[0])
    if out is not None:
        dest = Path(out).expanduser()
    else:
        name = first.stem + suffix
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        folder = Path(outdir).expanduser() if outdir is not None else first.parent
        dest = folder / name
    dest = Path(os.path.abspath(dest))
    guard_output(inputs, dest, overwrite=overwrite)
    return dest


def beside_path(lossless: Path) -> Path:
    """``<stem> (smaller).pdf`` beside ``lossless``; ``(smaller 2)``, ``(smaller 3)`` … if taken."""
    lossless = Path(lossless)
    folder, stem = lossless.parent, lossless.stem
    for n in range(1, _BESIDE_LIMIT + 1):
        label = "smaller" if n == 1 else f"smaller {n}"
        candidate = folder / f"{stem} ({label}).pdf"
        if not os.path.lexists(candidate):
            return candidate
    raise UsageError("no_free_name", f"Every name for a smaller copy beside {lossless} is taken.", path=str(lossless))


# --------------------------------------------------------------------------- the cache

def _mkdir_private(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise


def work_dir(kind: Literal["work", "staging", "records", "canary"], key: str) -> Path:
    """``<cache>/<kind>/<key>``, created (owner-only where the OS has modes).

    ``key`` is one path component (a record id, ``<sha256>-<version>``, a canary key); an empty
    key gives the ``<cache>/<kind>`` folder itself, where records and canary results are files.
    """
    if kind not in WORK_KINDS:
        raise UsageError("bad_work_kind", f"Unknown cache folder kind {kind!r}.", kind=str(kind))
    if key and (not _KEY_RE.match(key) or key.endswith(".")):
        raise UsageError("bad_work_key", f"{key!r} can't name a cache folder.", key=str(key))
    root = platforms.cache_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    _mkdir_private(root)
    path = root / kind
    _mkdir_private(path)
    if key:
        path = path / key
        _mkdir_private(path)
    return path


def _newest_mtime(path: Path) -> float:
    """The newest mtime of an entry and, for a real folder, everything inside it. Links aren't followed."""
    newest = os.lstat(path).st_mtime
    if stat.S_ISDIR(os.lstat(path).st_mode):
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            for name in dirnames + filenames:
                try:
                    newest = max(newest, os.lstat(os.path.join(dirpath, name)).st_mtime)
                except OSError:
                    pass
    return newest


def prune(root: Path, older_than_days: int = 14) -> int:
    """Remove each entry of ``root`` that nothing inside has touched for ``older_than_days``.

    ``root`` must be the cache's ``work/`` or ``staging/`` folder or a folder below one (UsageError
    otherwise), so a wrong argument can never delete user files, installed packs or decision records. Links are removed, never followed. Entries that can't be removed (a file in use on
    Windows) are skipped. Returns how many entries were removed; a missing root gives 0.
    """
    root = Path(root)
    key = _fold(root)
    cache = platforms.cache_root()
    if not any(_inside(key, _fold(cache / kind)) for kind in PRUNABLE_KINDS):
        raise UsageError("prune_outside_cache",
                         f"{root} isn't a prunable cache folder (work/ or staging/); nothing was pruned.",
                         path=str(root))
    if not root.is_dir():
        return 0
    cutoff = time.time() - older_than_days * 86400
    removed = 0
    for entry in sorted(root.iterdir()):
        try:
            if _newest_mtime(entry) >= cutoff:
                continue
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink()
            else:
                shutil.rmtree(entry)
            removed += 1
        except OSError:
            continue
    return removed


# --------------------------------------------------------------------------- console

def utf8_stdio() -> None:
    """Write UTF-8 even where the system default is older (Windows sends piped output as cp1252),
    so a file name in a non-Latin script or with an emoji can't crash a run whose output goes to Claude or a log."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
