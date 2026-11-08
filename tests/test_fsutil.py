from __future__ import annotations

import errno
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pdfintake import fsutil, platforms
from pdfintake.errors import EXIT_STALE, EXIT_USAGE, StaleRecord, UsageError

SRC = Path(__file__).resolve().parents[1] / "src"


def _partials(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.name.endswith(fsutil.PARTIAL_SUFFIX))


def _case_insensitive(folder: Path) -> bool:
    probe = folder / "CaseProbe.tmp"
    probe.write_bytes(b"")
    try:
        return (folder / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


@pytest.fixture(autouse=True)
def _own_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test gets an empty cache of its own inside the session's private one."""
    cache = Path(os.environ["PDFINTAKE_CACHE"]) / f"fsutil-{tmp_path.name}"
    monkeypatch.setenv("PDFINTAKE_CACHE", str(cache))
    return cache


@pytest.fixture
def built(tmp_path: Path) -> Path:
    path = tmp_path / "staging" / "built.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 9000 + b"\n%%EOF\n")      # > 2 MiB: several chunks
    return path


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "Desktop"
    folder.mkdir()
    return folder


# --------------------------------------------------------------------------- Windows' file rules, anywhere

_REAL_FSTAT, _REAL_LSTAT = os.fstat, os.lstat
_REAL_UNLINK, _REAL_RENAME, _REAL_REPLACE = os.unlink, os.rename, os.replace
_SHARING_VIOLATION = "The process cannot access the file because it is being used by another process"


class _Platform:
    """``sys`` as fsutil sees it, with another ``platform``."""

    def __init__(self, platform: str) -> None:
        self.platform = platform

    def __getattr__(self, name: str):
        return getattr(sys, name)


def _win_error(cls: type[OSError], err: int, winerror: int, message: str, path) -> OSError:
    exc = cls(err, message, str(path))
    exc.winerror = winerror                 # what Windows' OSError carries; set by hand off Windows
    return exc


class _Held:
    """A file fsutil opened with ``open``, counted as open (by identity) until it is closed."""

    def __init__(self, f, held: dict) -> None:
        self._f, self._held = f, held
        st = _REAL_FSTAT(f.fileno())
        self._id = (st.st_dev, st.st_ino)
        held[self._id] = held.get(self._id, 0) + 1

    def __getattr__(self, name: str):
        return getattr(self._f, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._held.get(self._id):
            self._held[self._id] -= 1
        self._f.close()


def _windows_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows' rules for files, here, so atomic_write takes its Windows branch on every system.

    Python opens a file without FILE_SHARE_DELETE, so while it is open Windows refuses to remove,
    rename or replace it (ERROR_SHARING_VIOLATION, WinError 32; a replace whose target is open gets
    ERROR_ACCESS_DENIED, WinError 5), and ``os.rename`` never replaces a file (MoveFileEx without
    MOVEFILE_REPLACE_EXISTING: FileExistsError). A file counts as open while fsutil holds it through
    ``open``, which is how atomic_write opens its partial file.

    A file's identity is its NTFS file ID, which carries a sequence number: a file created after
    another was removed never gets that one's ID, even when it takes its MFT record. Linux hands a
    freed inode number to the next file at once, so without this a swap after close (the Windows
    order) went unseen on the Linux runners while real Windows would see it.
    """
    held: dict[tuple[int, int], int] = {}
    gens: dict[tuple[int, int], int] = {}               # removals per (dev, ino): the ID's sequence number

    def file_id(st: os.stat_result) -> tuple[int, int]:
        return st.st_dev, (st.st_ino << 20) | gens.get((st.st_dev, st.st_ino), 0)

    def gone(st: os.stat_result | None) -> None:
        if st is not None:
            gens[(st.st_dev, st.st_ino)] = gens.get((st.st_dev, st.st_ino), 0) + 1

    def there(path) -> os.stat_result | None:            # the real lstat, never a test's stand-in
        try:
            return _REAL_LSTAT(path)
        except OSError:
            return None

    def in_use(path) -> bool:
        st = there(path)
        return st is not None and held.get((st.st_dev, st.st_ino), 0) > 0

    def opened(file, mode="r", *a, **k):
        return _Held(open(file, mode, *a, **k), held)

    def unlink(path, *a, **k):
        if in_use(path):
            raise _win_error(PermissionError, errno.EACCES, 32, _SHARING_VIOLATION, path)
        st = there(path)
        out = _REAL_UNLINK(path, *a, **k)
        gone(st)
        return out

    def rename(src, dst, *a, **k):
        if in_use(src):
            raise _win_error(PermissionError, errno.EACCES, 32, _SHARING_VIOLATION, src)
        if there(dst) is not None:
            raise _win_error(FileExistsError, errno.EEXIST, 183, "Cannot create a file when that file already "
                                                                 "exists", src)
        return _REAL_RENAME(src, dst, *a, **k)

    def replace(src, dst, *a, **k):
        if in_use(src):
            raise _win_error(PermissionError, errno.EACCES, 32, _SHARING_VIOLATION, src)
        if in_use(dst):
            raise _win_error(PermissionError, errno.EACCES, 5, "Access is denied", dst)
        st = there(dst)
        out = _REAL_REPLACE(src, dst, *a, **k)
        gone(st)                                            # the replaced file is removed
        return out

    monkeypatch.setattr(fsutil, "sys", _Platform("win32"))
    monkeypatch.setattr(fsutil, "_ids", file_id)
    monkeypatch.setattr(fsutil, "open", opened, raising=False)
    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "remove", unlink)
    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "replace", replace)


@pytest.fixture(params=["native"] if sys.platform == "win32" else ["native", "windows"])
def rules(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """The file rules a test runs under: this system's, and elsewhere Windows' too (emulated).

    The answer is ``"windows"`` or ``"posix"``, for a test whose promise differs between them.
    """
    if request.param == "native":
        return "windows" if sys.platform == "win32" else "posix"
    _windows_rules(monkeypatch)
    return "windows"


# --------------------------------------------------------------------------- atomic_write

def test_atomic_write_copies_and_leaves_nothing_behind(built: Path, out_dir: Path, rules: str):
    dest = out_dir / "Example (searchable).pdf"
    fsutil.atomic_write(built, dest)
    assert dest.read_bytes() == built.read_bytes()
    assert _partials(out_dir) == []
    assert built.exists(), "the built file is copied, not moved"


def test_atomic_write_replaces_an_existing_file_only_with_overwrite(built: Path, out_dir: Path, rules: str):
    dest = out_dir / "old.pdf"
    dest.write_bytes(b"old")
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)                 # the default never replaces
    assert info.value.code == "output_exists" and info.value.exit_code == EXIT_USAGE
    assert dest.read_bytes() == b"old" and _partials(out_dir) == []
    fsutil.atomic_write(built, dest, overwrite=True)
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


@pytest.mark.parametrize("where", ["during_copy", "at_fsync", "at_replace"])
def test_a_simulated_crash_leaves_no_partial_and_dest_untouched(built: Path, out_dir: Path,
                                                               monkeypatch: pytest.MonkeyPatch, where: str):
    dest = out_dir / "kept.pdf"
    dest.write_bytes(b"the previous file")

    class Crash(Exception):
        pass

    def boom(*a, **k):
        raise Crash(where)

    if where == "during_copy":
        real_open = open
        calls = {"n": 0}

        class Writer(io.FileIO):
            def write(self, b):                           # the second chunk dies
                calls["n"] += 1
                if calls["n"] == 2:
                    raise Crash(where)
                return super().write(b)

        def fake_open(file, mode="r", *a, **k):
            if mode == "xb":
                return Writer(file, "x")
            return real_open(file, mode, *a, **k)

        monkeypatch.setattr(fsutil, "open", fake_open, raising=False)
    elif where == "at_fsync":
        monkeypatch.setattr(fsutil, "_fsync_file", boom)
    else:
        monkeypatch.setattr(fsutil.os, "replace", boom)
    with pytest.raises(Crash):
        fsutil.atomic_write(built, dest, overwrite=True)
    assert _partials(out_dir) == []
    assert dest.read_bytes() == b"the previous file"


def test_a_short_write_is_refused(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    real_stat = os.stat

    def bigger(path, *a, **k):
        st = real_stat(path, *a, **k)
        if Path(path) == built:
            return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid,
                                   st.st_size + 1, st.st_atime, st.st_mtime, st.st_ctime))
        return st

    monkeypatch.setattr(fsutil.os, "stat", bigger)
    dest = out_dir / "short.pdf"
    with pytest.raises(Exception, match="stopped after"):
        fsutil.atomic_write(built, dest)
    assert not dest.exists() and _partials(out_dir) == []


_KILLED_WRITER = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from pdfintake import fsutil
real = fsutil._fsync_file
def die(fd):
    os._exit(9)                    # killed after the bytes, before the rename
fsutil._fsync_file = die
fsutil.atomic_write(sys.argv[2], sys.argv[3])
"""


def test_a_killed_writer_never_touches_dest_and_the_next_write_cleans_up(built: Path, out_dir: Path):
    dest = out_dir / "report.pdf"
    dest.write_bytes(b"the previous file")
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", _KILLED_WRITER, str(SRC), str(built), str(dest)],
                          capture_output=True, timeout=120)
    assert proc.returncode == 9, proc.stderr
    assert dest.read_bytes() == b"the previous file"
    assert [p.name for p in _partials(out_dir)] == ["report.pdf" + fsutil.PARTIAL_SUFFIX]
    fsutil.atomic_write(built, dest, overwrite=True)
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


def _pause_first_writer_before_rename(monkeypatch: pytest.MonkeyPatch, during_pause):
    """Writer "A" stops right after its fsync, with its partial file written and locked, and runs
    ``during_pause`` there; the rename follows once that returns."""
    import threading

    real = fsutil._fsync_file

    def fsync(fd):
        real(fd)
        if threading.current_thread().name == "A":
            during_pause()

    monkeypatch.setattr(fsutil, "_fsync_file", fsync)


def _run_as(name: str, fn) -> dict:
    import threading

    result: dict = {}

    def body():
        try:
            fn()
            result["ok"] = True
        except BaseException as exc:          # noqa: BLE001 - collected for the assertion
            result["error"] = exc

    t = threading.Thread(target=body, name=name)
    t.start()
    t.join(30)
    return result


def test_a_second_writer_never_touches_a_live_writers_partial(tmp_path: Path, built: Path, out_dir: Path,
                                                              monkeypatch: pytest.MonkeyPatch):
    other = tmp_path / "other.pdf"
    other.write_bytes(b"B" * (3 << 20))
    dest = out_dir / "a.pdf"
    second: dict = {}

    def second_writer():
        second.update(_run_as("B", lambda: fsutil.atomic_write(other, dest)))
        assert len(_partials(out_dir)) == 1, "the live writer's partial file is left alone"

    _pause_first_writer_before_rename(monkeypatch, second_writer)
    first = _run_as("A", lambda: fsutil.atomic_write(built, dest))
    assert first == {"ok": True}
    err = second.get("error")
    assert isinstance(err, UsageError) and err.code == "write_conflict" and err.exit_code == EXIT_USAGE, second
    assert dest.read_bytes() == built.read_bytes()
    assert _partials(out_dir) == []


def _just_before_the_rename_check(monkeypatch: pytest.MonkeyPatch, during) -> None:
    """Run ``during`` once, right before atomic_write's last check that the partial name is still its
    own file. On Windows that is after the writer has closed the file: the first moment anything
    else can remove or replace it there."""
    real = fsutil._still_ours
    ran: list[bool] = []

    def check(partial: Path, mine):
        if not ran:
            ran.append(True)
            during()
        return real(partial, mine)

    monkeypatch.setattr(fsutil, "_still_ours", check)


def _swapper(partial: Path, refused: list[OSError]):
    """Someone else's half-written file takes the partial name; a refusal (Windows) is noted instead."""
    def swap() -> None:
        try:
            partial.unlink()
        except PermissionError as exc:                # Windows: a file the writer has open stays
            refused.append(exc)
            return
        partial.write_bytes(b"B" * 1000)

    return swap


@pytest.mark.parametrize("when", ["while_open", "before_the_rename"])
def test_a_writer_whose_partial_was_replaced_renames_nothing(tmp_path: Path, built: Path, out_dir: Path,
                                                             monkeypatch: pytest.MonkeyPatch, rules: str, when: str):
    dest = out_dir / "kept.pdf"
    dest.write_bytes(b"the previous file")
    partial = dest.with_name(dest.name + fsutil.PARTIAL_SUFFIX)
    refused: list[OSError] = []
    swap = _swapper(partial, refused)
    if when == "while_open":
        _pause_first_writer_before_rename(monkeypatch, swap)
    else:
        _just_before_the_rename_check(monkeypatch, swap)
    first = _run_as("A", lambda: fsutil.atomic_write(built, dest, overwrite=True))
    if rules == "windows" and when == "while_open":
        # Windows writes the partial file while it is open and refuses to remove an open file
        # (WinError 32), so nobody can swap it then, and the write completes.
        assert len(refused) == 1 and first == {"ok": True}, (refused, first)
        assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []
        return
    assert refused == []
    err = first.get("error")
    assert isinstance(err, UsageError) and err.code == "write_conflict", first
    assert dest.read_bytes() == b"the previous file"
    assert partial.read_bytes() == b"B" * 1000, "the other writer's partial file is left alone"


def test_a_name_too_long_for_a_partial_suffix_still_writes(built: Path, out_dir: Path):
    dest = out_dir / ("x" * 245 + ".pdf")                         # 249 units: fits, but not with the suffix
    fsutil.guard_output([], dest, overwrite=False)
    partial = fsutil._partial_path(dest)
    assert partial.parent == out_dir and partial.name.startswith(".") and partial.name.endswith(fsutil.PARTIAL_SUFFIX)
    assert len(partial.name) <= fsutil.NAME_MAX
    assert fsutil._partial_path(out_dir / "short.pdf").name == "short.pdf" + fsutil.PARTIAL_SUFFIX
    fsutil.atomic_write(built, dest)
    assert dest.read_bytes() == built.read_bytes()
    assert _partials(out_dir) == []


# --------------------------------------------------------------------------- the exclusive publish (overwrite=False)

def _no_renamex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fsutil, "_rename_excl", lambda src, dst: None)


def _no_hard_links(monkeypatch: pytest.MonkeyPatch, err: int = errno.ENOTSUP) -> None:
    def refuse(src, dst, *a, **k):
        raise OSError(err, os.strerror(err), str(src), None, str(dst))

    monkeypatch.setattr(fsutil.os, "link", refuse)


WAYS = ["renamex_np", "link", "recheck"]


def _use(way: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force one way of the exclusive rename (atomic_write's docstring lists the order)."""
    if way == "renamex_np":
        if fsutil._renamex_np() is None:
            pytest.skip("renamex_np is macOS only")
        return
    _no_renamex(monkeypatch)
    if way == "recheck":
        _no_hard_links(monkeypatch)


@pytest.mark.parametrize("way", WAYS)
def test_an_exclusive_publish_writes_a_new_file_and_leaves_one_name(built: Path, out_dir: Path,
                                                                    monkeypatch: pytest.MonkeyPatch, way: str):
    _use(way, monkeypatch)
    dest = out_dir / "Example (searchable).pdf"
    fsutil.atomic_write(built, dest, overwrite=False)
    assert dest.read_bytes() == built.read_bytes()
    assert _partials(out_dir) == []
    if sys.platform != "win32":
        assert os.stat(dest).st_nlink == 1                 # the link way unlinks the partial name


@pytest.mark.parametrize("way", WAYS)
def test_a_file_that_appears_after_the_guard_is_never_replaced(built: Path, out_dir: Path,
                                                               monkeypatch: pytest.MonkeyPatch, way: str):
    """Someone saves under the output's name while pdfintake flushes its partial file."""
    _use(way, monkeypatch)
    dest = out_dir / "Scan (searchable).pdf"
    fsutil.guard_output([], dest, overwrite=False)          # the name is free when it is checked
    late = b"%PDF-1.4\n% a file saved by another program, made up\n"
    real = fsutil._fsync_file

    def someone_saves(fd: int) -> None:
        real(fd)
        dest.write_bytes(late)

    monkeypatch.setattr(fsutil, "_fsync_file", someone_saves)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest, overwrite=False)
    assert info.value.code == "output_exists" and info.value.exit_code == EXIT_USAGE
    assert dest.read_bytes() == late                        # the late file's bytes survive
    assert _partials(out_dir) == []                         # and the partial file is gone


def test_with_overwrite_a_late_file_is_replaced(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    dest = out_dir / "Scan (searchable).pdf"
    real = fsutil._fsync_file

    def someone_saves(fd: int) -> None:
        real(fd)
        dest.write_bytes(b"late")

    monkeypatch.setattr(fsutil, "_fsync_file", someone_saves)
    fsutil.atomic_write(built, dest, overwrite=True)
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


@pytest.mark.skipif(sys.platform != "darwin", reason="renamex_np is macOS only")
def test_renamex_np_is_found_and_refuses_an_existing_target(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    assert fsutil._rename_excl(a, b) == errno.EEXIST
    assert a.read_bytes() == b"A" and b.read_bytes() == b"B"
    b.unlink()
    assert fsutil._rename_excl(a, b) == 0
    assert not a.exists() and b.read_bytes() == b"A"


# renamex_np and the hard link are the POSIX ways; Windows' own way is tested on every system below.
_POSIX_WAYS = pytest.mark.skipif(sys.platform == "win32",
                                 reason="Windows publishes with os.rename, which refuses an existing file itself "
                                        "(MoveFileEx without MOVEFILE_REPLACE_EXISTING): renamex_np and the "
                                        "hard-link fallback are the POSIX ways and never run there")


@_POSIX_WAYS
@pytest.mark.parametrize("err", [errno.ENOTSUP, errno.EINVAL])
def test_a_filesystem_without_renamex_falls_back_to_a_link(built: Path, out_dir: Path,
                                                           monkeypatch: pytest.MonkeyPatch, err: int):
    monkeypatch.setattr(fsutil, "_rename_excl", lambda src, dst: err)
    linked: list[Path] = []
    real_link = os.link
    monkeypatch.setattr(fsutil.os, "link", lambda s, d, *a, **k: (linked.append(Path(d)), real_link(s, d))[1])
    dest = out_dir / "a.pdf"
    fsutil.atomic_write(built, dest, overwrite=False)
    assert linked == [dest] and dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


@_POSIX_WAYS
def test_another_renamex_error_writes_nothing(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fsutil, "_rename_excl", lambda src, dst: errno.EACCES)
    dest = out_dir / "a.pdf"
    with pytest.raises(PermissionError):
        fsutil.atomic_write(built, dest, overwrite=False)
    assert not dest.exists() and _partials(out_dir) == []


@_POSIX_WAYS
@pytest.mark.parametrize("err", [errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP])
def test_a_filesystem_without_hard_links_rechecks_before_the_rename(built: Path, out_dir: Path,
                                                                    monkeypatch: pytest.MonkeyPatch, err: int):
    _no_renamex(monkeypatch)
    _no_hard_links(monkeypatch, err)
    checked: list[Path] = []
    real_lexists = os.path.lexists
    monkeypatch.setattr(fsutil.os.path, "lexists", lambda p: (checked.append(Path(p)), real_lexists(p))[1])
    dest = out_dir / "a.pdf"
    fsutil.atomic_write(built, dest, overwrite=False)
    assert dest in checked and dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


@_POSIX_WAYS
def test_another_link_error_writes_nothing(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _no_renamex(monkeypatch)
    _no_hard_links(monkeypatch, errno.EACCES)
    dest = out_dir / "a.pdf"
    with pytest.raises(PermissionError):
        fsutil.atomic_write(built, dest, overwrite=False)
    assert not dest.exists() and _partials(out_dir) == []


def test_a_crash_at_the_exclusive_publish_leaves_no_partial(built: Path, out_dir: Path,
                                                            monkeypatch: pytest.MonkeyPatch, rules: str):
    class Crash(Exception):
        pass

    def boom(*a, **k):
        raise Crash("publish")

    monkeypatch.setattr(fsutil, "_publish", boom)
    dest = out_dir / "a.pdf"
    with pytest.raises(Crash):
        fsutil.atomic_write(built, dest, overwrite=False)
    assert not dest.exists() and _partials(out_dir) == []


def test_windows_publishes_with_an_exclusive_rename_alone(built: Path, out_dir: Path,
                                                          monkeypatch: pytest.MonkeyPatch):
    """Windows' way, on every system: os.rename after the file is closed, never renamex_np or a link."""
    if sys.platform != "win32":
        _windows_rules(monkeypatch)
    used: list[str] = []
    monkeypatch.setattr(fsutil, "_rename_excl", lambda src, dst: used.append("renamex_np"))
    monkeypatch.setattr(fsutil.os, "link", lambda *a, **k: used.append("link"))
    dest = out_dir / "Scan (searchable).pdf"
    fsutil.atomic_write(built, dest)
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []
    late = b"%PDF-1.4\n% a file another program saved, made up\n"
    dest.write_bytes(late)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)                     # os.rename refuses the file that is there
    assert info.value.code == "output_exists" and dest.read_bytes() == late
    assert _partials(out_dir) == [] and used == []


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows can't remove a file that is open (a sharing violation), so there the partial "
                           "file is removed just after it is closed, by the identity read while it was open")
def test_a_failed_publish_removes_its_partial_before_closing_it(built: Path, out_dir: Path,
                                                               monkeypatch: pytest.MonkeyPatch):
    """On macOS 15's FAT32 a refused publish left its partial file behind (CI, 2026-11-07) when it
    was compared and removed after being closed. One way that happens, simulated here: lstat gives
    the partial file another number once the writer has closed it. It is compared and removed while
    still open instead, as the pre-rename check is."""
    dest = out_dir / "Scan (searchable).pdf"
    late = b"%PDF-1.4\n% a file another program saved, made up\n"
    dest.write_bytes(late)
    partial = fsutil._partial_path(dest)
    closed: list[bool] = []

    class Partial(_Held):
        def close(self) -> None:
            super().close()
            closed.append(True)

    def opened(file, mode="r", *a, **k):
        f = open(file, mode, *a, **k)
        return Partial(f, {}) if Path(file) == partial else f

    def lstat(path, *a, **k):
        st = _REAL_LSTAT(path, *a, **k)
        if closed and Path(path) == partial:
            fields = list(st)
            fields[1] = st.st_ino + 1
            return os.stat_result(fields)
        return st

    monkeypatch.setattr(fsutil, "open", opened, raising=False)
    monkeypatch.setattr(fsutil.os, "lstat", lstat)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)                     # the name is taken: the publish refuses
    assert info.value.code == "output_exists" and dest.read_bytes() == late
    assert closed and _partials(out_dir) == []


def test_a_partial_that_cant_be_removed_while_open_is_removed_once_closed(built: Path, out_dir: Path,
                                                                         monkeypatch: pytest.MonkeyPatch):
    """A filesystem that refuses to remove a file that is open (simulated with EBUSY) still loses the
    partial file of a refused publish: it is tried again once the file is closed."""
    dest = out_dir / "Scan (searchable).pdf"
    late = b"%PDF-1.4\n% a file another program saved, made up\n"
    dest.write_bytes(late)
    held: dict[tuple[int, int], int] = {}
    refused: list[str] = []

    def opened(file, mode="r", *a, **k):
        return _Held(open(file, mode, *a, **k), held)

    def unlink(path, *a, **k):
        try:
            st = _REAL_LSTAT(path)
        except OSError:
            st = None
        if st is not None and held.get((st.st_dev, st.st_ino)):
            refused.append(Path(path).name)
            raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), str(path))
        return _REAL_UNLINK(path, *a, **k)

    monkeypatch.setattr(fsutil, "open", opened, raising=False)
    monkeypatch.setattr(os, "unlink", unlink)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)                     # the name is taken: the publish refuses
    assert info.value.code == "output_exists" and dest.read_bytes() == late
    assert _partials(out_dir) == []
    if sys.platform != "win32":                              # Windows never tries while it is open
        assert refused == [fsutil._partial_path(dest).name]


# --------------------------------------------------------------------------- sweep_partial

def test_sweep_partial_removes_a_killed_writers_partial(built: Path, out_dir: Path):
    dest = out_dir / "report.pdf"
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", _KILLED_WRITER, str(SRC), str(built), str(dest)],
                          capture_output=True, timeout=120)
    assert proc.returncode == 9, proc.stderr
    assert [p.name for p in _partials(out_dir)] == ["report.pdf" + fsutil.PARTIAL_SUFFIX]
    assert fsutil.sweep_partial(dest) is True
    assert _partials(out_dir) == [] and not dest.exists()
    assert fsutil.sweep_partial(dest) is False             # nothing left: nothing to do, no error


def test_sweep_partial_removes_the_hidden_partial_of_a_long_name(out_dir: Path):
    dest = out_dir / ("x" * 245 + ".pdf")
    partial = fsutil._partial_path(dest)
    partial.write_bytes(b"half")
    assert partial.name.startswith(".")
    assert fsutil.sweep_partial(dest) is True and not partial.exists()


def test_sweep_partial_touches_only_dests_own_partial_name(out_dir: Path):
    dest = out_dir / "Scan (searchable).pdf"
    dest.write_bytes(b"the user's own output")
    keep = {
        dest: b"the user's own output",
        out_dir / "Scan (searchable).pdf.bak": b"x",
        out_dir / ("Other.pdf" + fsutil.PARTIAL_SUFFIX): b"another output's partial file",
        out_dir / ("Scan.pdf" + fsutil.PARTIAL_SUFFIX): b"a different name's partial file",
        out_dir / "Scan (searchable).pdf.pdfintake-partial.txt": b"x",
    }
    for path, data in keep.items():
        path.write_bytes(data)
    assert fsutil.sweep_partial(dest) is False
    for path, data in keep.items():
        assert path.read_bytes() == data


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need a privilege on Windows")
def test_sweep_partial_never_follows_or_removes_a_link(tmp_path: Path, out_dir: Path):
    target = tmp_path / "elsewhere.pdf"
    target.write_bytes(b"a file the link points at")
    dest = out_dir / "a.pdf"
    partial = fsutil._partial_path(dest)
    partial.symlink_to(target)
    assert fsutil.sweep_partial(dest) is False
    assert partial.is_symlink() and target.read_bytes() == b"a file the link points at"


@pytest.mark.skipif(sys.platform == "win32", reason="the partial file lock is POSIX only")
def test_sweep_partial_leaves_a_partial_a_live_writer_holds(out_dir: Path):
    import fcntl

    dest = out_dir / "a.pdf"
    partial = fsutil._partial_path(dest)
    partial.write_bytes(b"half written")
    fd = os.open(partial, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)        # what a live writer holds
        assert fsutil.sweep_partial(dest) is False
        assert partial.read_bytes() == b"half written"
    finally:
        os.close(fd)
    assert fsutil.sweep_partial(dest) is True and not partial.exists()


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "geteuid"), reason="POSIX owners")
def test_sweep_partial_leaves_another_users_file(out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    dest = out_dir / "a.pdf"
    partial = fsutil._partial_path(dest)
    partial.write_bytes(b"someone else's")
    monkeypatch.setattr(fsutil.os, "geteuid", lambda: os.stat(partial).st_uid + 1)
    assert fsutil.sweep_partial(dest) is False and partial.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_sweep_partial_never_opens_a_fifo_for_long(out_dir: Path):
    dest = out_dir / "a.pdf"
    partial = fsutil._partial_path(dest)
    os.mkfifo(partial)
    assert fsutil.sweep_partial(dest) is False and partial.exists()


def test_sweep_partial_during_a_live_write_leaves_it_alone(built: Path, out_dir: Path,
                                                         monkeypatch: pytest.MonkeyPatch):
    dest = out_dir / "a.pdf"
    swept: list[bool] = []
    _pause_first_writer_before_rename(monkeypatch, lambda: swept.append(fsutil.sweep_partial(dest)))
    first = _run_as("A", lambda: fsutil.atomic_write(built, dest))
    assert first == {"ok": True}
    assert swept == [False]                                 # locked (POSIX) or open (Windows): left alone
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


@pytest.mark.parametrize("dest", ["relative/a.pdf", "", "/"])
def test_sweep_partial_refuses_a_path_that_names_no_output(dest: str):
    assert fsutil.sweep_partial(Path(dest)) is False


def test_sweep_partial_never_raises(out_dir: Path):
    assert fsutil.sweep_partial(Path(str(out_dir / "a") + "\x00.pdf")) is False
    assert fsutil.sweep_partial(out_dir / "no such folder" / "a.pdf") is False


# --------------------------------------------------------------------------- what sits at the partial name

@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need a privilege on Windows")
def test_a_link_at_the_partial_name_is_left_and_nothing_is_written(tmp_path: Path, built: Path, out_dir: Path):
    """pdfintake never makes a link there, so it isn't a leftover: kept, never followed (decided 2026-11-06)."""
    dest = out_dir / "Out.pdf"
    target = tmp_path / "users-file.txt"
    target.write_bytes(b"a file the link points at")
    link = fsutil._partial_path(dest)
    link.symlink_to(target)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)
    assert info.value.code == "write_conflict"
    assert link.is_symlink() and target.read_bytes() == b"a file the link points at" and not dest.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_fifo_at_the_partial_name_is_left_and_nothing_is_written(built: Path, out_dir: Path):
    dest = out_dir / "Out.pdf"
    os.mkfifo(fsutil._partial_path(dest))
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)
    assert info.value.code == "write_conflict"
    assert os.path.lexists(fsutil._partial_path(dest)) and not dest.exists()


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "geteuid"), reason="POSIX owners")
def test_another_users_file_at_the_partial_name_is_left(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    dest = out_dir / "Out.pdf"
    partial = fsutil._partial_path(dest)
    partial.write_bytes(b"someone else's")
    monkeypatch.setattr(fsutil.os, "geteuid", lambda: os.stat(partial).st_uid + 1)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)
    assert info.value.code == "write_conflict"
    assert partial.read_bytes() == b"someone else's" and not dest.exists()


def test_a_killed_runs_own_partial_is_still_replaced(built: Path, out_dir: Path):
    dest = out_dir / "Out.pdf"
    fsutil._partial_path(dest).write_bytes(b"half written")
    fsutil.atomic_write(built, dest)
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


# --------------------------------------------------------------------------- FAT and exFAT

PLACEHOLDER_INO = 2**64 - 60      # what FAT gives an empty file before its first cluster (measured)


def _inode_moves_once_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every empty regular file report a placeholder inode, as FAT and exFAT do: the number
    changes once data is written, while fstat and lstat of one file always agree."""
    import stat as st_mod

    def fake(real):
        def call(*a, **k):
            st = real(*a, **k)
            if st_mod.S_ISREG(st.st_mode) and st.st_size == 0:
                fields = list(st)
                fields[1] = PLACEHOLDER_INO
                return os.stat_result(fields)
            return st
        return call

    monkeypatch.setattr(fsutil.os, "fstat", fake(os.fstat))
    monkeypatch.setattr(fsutil.os, "lstat", fake(os.lstat))


def test_a_file_whose_inode_changes_once_written_is_published(built: Path, out_dir: Path,
                                                              monkeypatch: pytest.MonkeyPatch):
    """The partial file is identified when it is renamed, not when it was created (decided 2026-11-06)."""
    _inode_moves_once_written(monkeypatch)
    dest = out_dir / "Scan (searchable).pdf"
    fsutil.atomic_write(built, dest)
    assert dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []


def test_a_failed_write_whose_inode_changed_removes_its_partial(built: Path, out_dir: Path,
                                                               monkeypatch: pytest.MonkeyPatch):
    _inode_moves_once_written(monkeypatch)

    def fails(fd: int) -> None:
        raise OSError(errno.EIO, "made-up I/O error")

    monkeypatch.setattr(fsutil, "_fsync_file", fails)
    dest = out_dir / "Scan (searchable).pdf"
    with pytest.raises(OSError):
        fsutil.atomic_write(built, dest)
    assert not dest.exists() and _partials(out_dir) == []


@pytest.mark.parametrize("when", ["while_open", "before_the_rename"])
def test_a_swapped_partial_is_still_caught_when_identity_is_read_late(built: Path, out_dir: Path,
                                                                     monkeypatch: pytest.MonkeyPatch, rules: str,
                                                                     when: str):
    """Reading the identity at the rename keeps the swap check: another file at the name is never renamed.

    ``while_open`` swaps before the identity is read from the open file, ``before_the_rename`` right
    before the name is compared with it (on Windows, after the file is closed)."""
    _inode_moves_once_written(monkeypatch)
    dest = out_dir / "kept.pdf"
    partial = fsutil._partial_path(dest)
    refused: list[OSError] = []
    swap = _swapper(partial, refused)
    if when == "while_open":
        real = fsutil._fsync_file

        def at_fsync(fd: int) -> None:
            real(fd)
            swap()

        monkeypatch.setattr(fsutil, "_fsync_file", at_fsync)
    else:
        _just_before_the_rename_check(monkeypatch, swap)
    if rules == "windows" and when == "while_open":
        fsutil.atomic_write(built, dest)                     # an open file can't be swapped there (WinError 32)
        assert len(refused) == 1 and dest.read_bytes() == built.read_bytes() and _partials(out_dir) == []
        return
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)
    assert refused == []
    assert info.value.code == "write_conflict" and not dest.exists()
    assert partial.read_bytes() == b"B" * 1000


@pytest.fixture(scope="module", params=["MS-DOS", "ExFAT"], ids=["fat32", "exfat"])
def real_volume(request, tmp_path_factory) -> Path:
    """A small FAT32 or exFAT disk image, attached without showing in Finder (macOS only)."""
    import shutil

    if sys.platform != "darwin" or shutil.which("hdiutil") is None:
        pytest.skip("needs hdiutil (macOS)")
    root = tmp_path_factory.mktemp(f"vol-{request.param.lower()}")
    image, mnt = root / "v.dmg", root / "mnt"
    mnt.mkdir()
    made = subprocess.run(["hdiutil", "create", "-quiet", "-size", "8m", "-fs", request.param, "-volname", "PDFI",
                           "-layout", "MBRSPUD", str(image)], capture_output=True, timeout=120)
    if made.returncode != 0:
        pytest.skip(f"hdiutil create failed: {made.stderr.decode(errors='replace')[:200]}")
    attached = subprocess.run(["hdiutil", "attach", "-quiet", "-nobrowse", "-noverify", "-noautoopen",
                               "-mountpoint", str(mnt), str(image)], capture_output=True, timeout=120)
    if attached.returncode != 0 or not os.path.ismount(mnt):
        pytest.skip(f"hdiutil attach failed: {attached.stderr.decode(errors='replace')[:200]}")
    try:
        yield mnt
    finally:
        subprocess.run(["hdiutil", "detach", "-force", "-quiet", str(mnt)], capture_output=True, timeout=120)


def _recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """What atomic_write's identity checks and removals saw, for a failure's message: each check's
    held identity, the name's identity at that moment and the answer, and any refused unlink."""
    seen: list[tuple] = []
    real_ours = fsutil._still_ours

    def still_ours(partial: Path, mine) -> bool:
        try:
            now: object = fsutil._ids(_REAL_LSTAT(partial))
        except OSError as exc:
            now = repr(exc)
        got = real_ours(partial, mine)
        seen.append(("still_ours", mine, now, got))
        return got

    def unlink(path, *a, **k):
        try:
            return _REAL_UNLINK(path, *a, **k)
        except OSError as exc:
            seen.append(("unlink", Path(path).name, repr(exc)))
            raise

    monkeypatch.setattr(fsutil, "_still_ours", still_ours)
    monkeypatch.setattr(os, "unlink", unlink)
    return seen


def test_atomic_write_on_a_real_fat_or_exfat_volume(real_volume: Path, built: Path, monkeypatch: pytest.MonkeyPatch):
    """FAT and exFAT have no hard links, no exclusive rename, and an inode number that changes once
    a file's first cluster is written (measured 2026-11-06)."""
    seen = _recorded(monkeypatch)
    dest = real_volume / "Scan (searchable).pdf"
    fsutil.atomic_write(built, dest)
    assert dest.read_bytes() == built.read_bytes() and _partials(real_volume) == [], seen
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest)                     # an existing name is still never replaced
    assert info.value.code == "output_exists" and _partials(real_volume) == [], seen
    fsutil.atomic_write(built, dest, overwrite=True)
    digest = __import__("hashlib").sha256(dest.read_bytes()).hexdigest()
    fsutil.atomic_write(built, dest, expect_sha256=digest)   # the compare-and-swap works there too
    assert dest.read_bytes() == built.read_bytes() and _partials(real_volume) == [], seen
    assert not os.path.lexists(fsutil._aside_path(dest))
    dest.unlink()


# --------------------------------------------------------------------------- replacing only an unchanged file

def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()
                  if p.name.endswith((fsutil.PARTIAL_SUFFIX, fsutil.ASIDE_SUFFIX)))


@pytest.mark.parametrize("way", WAYS)
def test_an_unchanged_output_is_replaced(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch, way: str):
    _use(way, monkeypatch)
    dest = out_dir / "Scan (searchable).pdf"
    dest.write_bytes(b"%PDF-1.4\n% pdfintake's earlier output, made up\n")
    fsutil.atomic_write(built, dest, expect_sha256=_sha(dest))
    assert dest.read_bytes() == built.read_bytes() and _leftovers(out_dir) == []


@pytest.mark.parametrize("way", WAYS)
def test_an_output_saved_over_during_the_copy_is_kept(built: Path, out_dir: Path,
                                                      monkeypatch: pytest.MonkeyPatch, way: str):
    """A save that lands after the caller's hash check, while the new file is copied (decided 2026-11-06)."""
    _use(way, monkeypatch)
    dest = out_dir / "Scan (searchable).pdf"
    dest.write_bytes(b"%PDF-1.4\n% pdfintake's earlier output, made up\n")
    expected = _sha(dest)
    edited = b"%PDF-1.4\n% the same output with the user's note, made up\n"
    real = fsutil._fsync_file

    def user_saves(fd: int) -> None:
        real(fd)
        tmp = out_dir / ".save-tmp"
        tmp.write_bytes(edited)
        os.replace(tmp, dest)                                # how document apps save

    monkeypatch.setattr(fsutil, "_fsync_file", user_saves)
    with pytest.raises(StaleRecord) as info:
        fsutil.atomic_write(built, dest, expect_sha256=expected)
    assert info.value.code == "output_changed" and info.value.exit_code == EXIT_STALE
    assert dest.read_bytes() == edited and _leftovers(out_dir) == []


def test_a_file_saved_while_the_old_one_is_aside_is_kept(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    dest = out_dir / "Scan (searchable).pdf"
    dest.write_bytes(b"%PDF-1.4\n% pdfintake's earlier output, made up\n")
    expected = _sha(dest)
    real = fsutil._hash_locked

    def during_the_check(path: Path):
        got = real(path)
        dest.write_bytes(LATE)                               # the name is free for a moment
        return got

    monkeypatch.setattr(fsutil, "_hash_locked", during_the_check)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest, expect_sha256=expected)
    assert info.value.code == "output_exists"
    assert dest.read_bytes() == LATE and _leftovers(out_dir) == []


def test_a_gone_output_is_not_written_again(built: Path, out_dir: Path):
    dest = out_dir / "Scan (searchable).pdf"
    with pytest.raises(StaleRecord) as info:
        fsutil.atomic_write(built, dest, expect_sha256="0" * 64)
    assert info.value.code == "output_changed" and not dest.exists() and _leftovers(out_dir) == []


def test_a_crash_during_the_check_puts_the_output_back(built: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    class Crash(Exception):
        pass

    dest = out_dir / "Scan (searchable).pdf"
    dest.write_bytes(b"%PDF-1.4\n% pdfintake's earlier output, made up\n")
    before = dest.read_bytes()

    def boom(path: Path):
        raise Crash("check")

    monkeypatch.setattr(fsutil, "_hash_locked", boom)
    with pytest.raises(Crash):
        fsutil.atomic_write(built, dest, expect_sha256=_sha(dest))
    assert dest.read_bytes() == before and _leftovers(out_dir) == []


def test_overwrite_and_expect_sha256_exclude_each_other(built: Path, out_dir: Path):
    with pytest.raises(ValueError):
        fsutil.atomic_write(built, out_dir / "a.pdf", overwrite=True, expect_sha256="0" * 64)


def test_the_sweep_puts_back_an_output_a_killed_replace_set_aside(out_dir: Path):
    dest = out_dir / "Scan (searchable).pdf"
    aside = fsutil._aside_path(dest)
    aside.write_bytes(b"the output, set aside")
    assert aside.name.startswith(".") and aside.name.endswith(fsutil.ASIDE_SUFFIX)
    fsutil.sweep_partial(dest)
    assert dest.read_bytes() == b"the output, set aside" and not aside.exists()
    aside.write_bytes(b"an older version")                   # dest is taken: left where it is, never removed
    fsutil.sweep_partial(dest)
    assert aside.read_bytes() == b"an older version" and dest.read_bytes() == b"the output, set aside"


def test_a_leftover_aside_with_the_output_present_stops_the_replace(built: Path, out_dir: Path):
    dest = out_dir / "Scan (searchable).pdf"
    dest.write_bytes(b"%PDF-1.4\n% pdfintake's earlier output, made up\n")
    aside = fsutil._aside_path(dest)
    aside.write_bytes(b"an older version")
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest, expect_sha256=_sha(dest))
    assert info.value.code == "write_conflict"
    assert aside.read_bytes() == b"an older version" and _partials(out_dir) == []


# --------------------------------------------------------------------------- an input under a name the write uses

@pytest.mark.parametrize("side", ["partial", "aside"])
def test_guard_output_refuses_an_input_at_a_name_the_write_goes_through(out_dir: Path, side: str):
    dest = out_dir / "Scan (searchable).pdf"
    src = fsutil._partial_path(dest) if side == "partial" else fsutil._aside_path(dest)
    src.write_bytes(b"%PDF-1.4\n% an input, made up\n")
    for overwrite in (False, True):
        err = _refused("output_is_input", [src], dest, overwrite=overwrite)
        assert err.detail["input"] == str(src)
    assert src.exists()


def test_guard_output_refuses_a_case_variant_of_the_partial_name(out_dir: Path):
    dest = out_dir / "Scan (searchable).pdf"
    src = out_dir / ("SCAN (searchable).pdf" + fsutil.PARTIAL_SUFFIX)
    src.write_bytes(b"in")
    _refused("output_is_input", [src], dest, overwrite=False)


def test_sweep_partial_never_touches_a_file_it_was_told_to_keep(out_dir: Path):
    dest = out_dir / "Scan (searchable).pdf"
    partial = fsutil._partial_path(dest)
    partial.write_bytes(b"%PDF-1.4\n% a complete leftover the user is running, made up\n")
    assert fsutil.sweep_partial(dest, keep=[partial]) is False and partial.exists()
    assert fsutil.sweep_partial(dest, keep=[out_dir / "Other.pdf"]) is True and not partial.exists()


LATE = b"%PDF-1.4\n% a file another program saved, made up\n"


# --------------------------------------------------------------------------- guard_output

def test_guard_output_allows_a_new_file(tmp_path: Path, out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    fsutil.guard_output([src], out_dir / "scan (searchable).pdf", overwrite=False)


def _refused(code: str, inputs, dest, *, overwrite: bool) -> UsageError:
    with pytest.raises(UsageError) as info:
        fsutil.guard_output(inputs, dest, overwrite=overwrite)
    assert info.value.code == code, info.value
    assert info.value.exit_code == EXIT_USAGE
    return info.value


@pytest.mark.parametrize("overwrite", [False, True])
def test_guard_output_refuses_the_input_itself(out_dir: Path, overwrite: bool):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    _refused("output_is_input", [src], src, overwrite=overwrite)
    _refused("output_is_input", [src], out_dir / "." / "scan.pdf", overwrite=overwrite)


def test_guard_output_refuses_an_input_through_a_symlinked_file(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    link = out_dir / "link.pdf"
    try:
        link.symlink_to(src)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need a privilege here")
    _refused("output_is_input", [src], link, overwrite=True)
    _refused("output_is_input", [link], src, overwrite=True)


def test_guard_output_refuses_an_input_through_a_symlinked_folder(tmp_path: Path, out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    alias = tmp_path / "alias-folder"
    try:
        alias.symlink_to(out_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need a privilege here")
    _refused("output_is_input", [src], alias / "scan.pdf", overwrite=True)


def test_guard_output_refuses_an_input_through_a_hardlink(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    hard = out_dir / "another name.pdf"
    try:
        os.link(src, hard)
    except (OSError, NotImplementedError):
        pytest.skip("hard links aren't available here")
    _refused("output_is_input", [src], hard, overwrite=True)


def test_guard_output_refuses_a_case_only_difference(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    # Refused on every filesystem: on APFS and NTFS the two names are one file.
    _refused("output_is_input", [src], out_dir / "Scan.pdf", overwrite=True)
    _refused("output_is_input", [src], out_dir / "SCAN.PDF", overwrite=False)


def test_case_only_collision_on_this_filesystem(out_dir: Path):
    if not _case_insensitive(out_dir):
        pytest.skip("this filesystem is case-sensitive")
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    assert os.path.samefile(src, out_dir / "Scan.pdf")
    _refused("output_is_input", [src], out_dir / "Scan.pdf", overwrite=True)


def test_guard_output_refuses_a_unicode_normal_form_difference(out_dir: Path):
    src = out_dir / "Résumé.pdf"              # decomposed
    src.write_bytes(b"in")
    _refused("output_is_input", [src], out_dir / "Résumé.pdf", overwrite=True)


def test_guard_output_refuses_a_path_inside_the_cache(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    staging = fsutil.work_dir("staging", "r-20260101-0a1b2c3d")
    _refused("output_in_cache", [src], staging / "out.pdf", overwrite=True)
    _refused("output_in_cache", [src], platforms.cache_root() / "out.pdf", overwrite=True)


def test_guard_output_refuses_an_existing_file_without_overwrite(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    dest = out_dir / "scan (searchable).pdf"
    dest.write_bytes(b"there")
    _refused("output_exists", [src], dest, overwrite=False)
    fsutil.guard_output([src], dest, overwrite=True)


def test_guard_output_counts_a_dangling_link_as_existing(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    dest = out_dir / "dangling.pdf"
    try:
        dest.symlink_to(out_dir / "nowhere.pdf")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need a privilege here")
    _refused("output_exists", [src], dest, overwrite=False)


def test_guard_output_refuses_a_missing_or_unwritable_folder_and_a_folder_dest(tmp_path: Path, out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    _refused("output_folder_missing", [src], tmp_path / "no such folder" / "x.pdf", overwrite=False)
    _refused("output_is_folder", [src], out_dir, overwrite=True)
    if sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        return                                            # no POSIX modes, or root writes anywhere
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        _refused("output_folder_not_writable", [src], locked / "x.pdf", overwrite=False)
    finally:
        locked.chmod(0o700)


# --------------------------------------------------------------------------- output_path, beside_path

def test_guard_output_refuses_a_name_longer_than_a_file_name_can_be(out_dir: Path):
    _refused("output_name_too_long", [], out_dir / ("x" * 252 + ".pdf"), overwrite=False)
    fsutil.guard_output([], out_dir / ("x" * 251 + ".pdf"), overwrite=False)
    src = out_dir / ("y" * 245 + ".pdf")
    src.write_bytes(b"in")
    with pytest.raises(UsageError) as info:
        fsutil.output_path([src], out=None, outdir=None, suffix=" (searchable)", overwrite=False)
    assert info.value.code == "output_name_too_long"


@pytest.mark.parametrize("plat,units", [("darwin", 255), ("win32", 255), ("linux", 765)])
def test_name_length_counts_what_each_filesystem_counts(monkeypatch: pytest.MonkeyPatch, plat: str, units: int):
    monkeypatch.setattr(fsutil.sys, "platform", plat)
    assert fsutil._name_length("ก" * 255) == units            # APFS and NTFS count UTF-16 units, ext4 bytes
    assert fsutil._name_length("\U0001f600") == (4 if plat == "linux" else 2)


def test_output_path_names(out_dir: Path, tmp_path: Path):
    src = out_dir / "Example Scan.pdf"
    src.write_bytes(b"in")
    assert fsutil.output_path([src], out=None, outdir=None, suffix=" (searchable)", overwrite=False) \
        == out_dir / "Example Scan (searchable).pdf"
    assert fsutil.output_path([src], out=None, outdir=None, suffix=" (compact).pdf", overwrite=False) \
        == out_dir / "Example Scan (compact).pdf"
    other = tmp_path / "Elsewhere"
    other.mkdir()
    assert fsutil.output_path([src], out=None, outdir=other, suffix=" (searchable)", overwrite=False) \
        == other / "Example Scan (searchable).pdf"
    assert fsutil.output_path([src], out=other / "chosen.pdf", outdir=None, suffix=" (searchable)",
                              overwrite=False) == other / "chosen.pdf"


def test_output_path_for_images_is_named_after_the_first(out_dir: Path):
    imgs = [out_dir / "IMG_0002.jpg", out_dir / "IMG_0001.jpg"]
    for p in imgs:
        p.write_bytes(b"\xff\xd8")
    assert fsutil.output_path(imgs, out=None, outdir=None, suffix="", overwrite=False) == out_dir / "IMG_0002.pdf"


def test_output_path_is_guarded(out_dir: Path):
    src = out_dir / "scan.pdf"
    src.write_bytes(b"in")
    with pytest.raises(UsageError, match="never changes its input"):
        fsutil.output_path([src], out=out_dir / "Scan.pdf", outdir=None, suffix="", overwrite=True)
    with pytest.raises(UsageError) as info:
        fsutil.output_path([src], out=None, outdir=None, suffix="", overwrite=True)
    assert info.value.code == "output_is_input"
    with pytest.raises(UsageError) as info:
        fsutil.output_path([src], out=out_dir / "a.pdf", outdir=out_dir, suffix="", overwrite=False)
    assert info.value.code == "out_and_outdir"
    with pytest.raises(UsageError) as info:
        fsutil.output_path([], out=None, outdir=None, suffix="", overwrite=False)
    assert info.value.code == "no_inputs"


def test_output_path_is_absolute(out_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(out_dir)
    Path("scan.pdf").write_bytes(b"in")
    dest = fsutil.output_path([Path("scan.pdf")], out=None, outdir=None, suffix=" (searchable)", overwrite=False)
    assert dest.is_absolute() and dest.name == "scan (searchable).pdf"


def test_beside_path_never_names_an_existing_file(out_dir: Path):
    lossless = out_dir / "Receipt (searchable).pdf"
    lossless.write_bytes(b"x")
    first = fsutil.beside_path(lossless)
    assert first == out_dir / "Receipt (searchable) (smaller).pdf"
    first.write_bytes(b"y")
    second = fsutil.beside_path(lossless)
    assert second == out_dir / "Receipt (searchable) (smaller 2).pdf"
    second.write_bytes(b"z")
    assert fsutil.beside_path(lossless).name == "Receipt (searchable) (smaller 3).pdf"


# --------------------------------------------------------------------------- work_dir, prune

def test_work_dir_kinds_and_keys():
    root = platforms.cache_root()
    assert fsutil.work_dir("records", "") == root / "records"
    assert fsutil.work_dir("canary", "") == root / "canary"
    d = fsutil.work_dir("work", "9f" * 32 + "-0.1.0")
    assert d.parent == root / "work" and d.is_dir()
    if sys.platform != "win32":
        assert (d.stat().st_mode & 0o777) == 0o700
    assert fsutil.work_dir("work", "9f" * 32 + "-0.1.0") == d          # idempotent
    for bad in ["..", ".", "../x", "a/b", "a\\b", "/abs", "C:x", "trailing.", "-lead", "has space"]:
        with pytest.raises(UsageError):
            fsutil.work_dir("staging", bad)
    with pytest.raises(UsageError):
        fsutil.work_dir("packs", "x")                                   # type: ignore[arg-type]


def _age(path: Path, days: float) -> None:
    """Set ``path``'s own times ``days`` back: a link's own, never those of what it points at."""
    t = time.time() - days * 86400
    if os.utime in os.supports_follow_symlinks:
        os.utime(path, (t, t), follow_symlinks=False)
    elif path.is_symlink():                       # Windows: os.utime would follow it
        raise NotImplementedError(f"os.utime can't set a link's own times here: {path}")
    else:
        os.utime(path, (t, t))


def _clock_past(root: Path) -> None:
    """Wait until the clock reads later than every entry's newest time, so a 0-day prune counts them
    all as older than now. Windows' time.time() before Python 3.13 moves in ~15.6 ms steps, so a file
    written a moment ago can carry the very time the clock still reads."""
    newest = max((fsutil._newest_mtime(p) for p in root.iterdir()), default=0.0)
    deadline = time.monotonic() + 5
    while time.time() <= newest and time.monotonic() < deadline:
        time.sleep(0.005)


@pytest.mark.parametrize("link_age", ["aged", "fresh"])
def test_prune_removes_only_entries_untouched_for_the_period(tmp_path: Path, link_age: str):
    if link_age == "aged" and os.utime not in os.supports_follow_symlinks:
        # os.utime(follow_symlinks=False) doesn't exist on Windows: it would age the folder the link
        # points at, never the link itself. The fresh link below still shows a link is removed, not followed.
        pytest.skip("os.utime can't set a link's own times on this platform (Windows has no follow_symlinks=False)")
    root = fsutil.work_dir("work", "")
    old_dir = root / "old"
    old_dir.mkdir()
    (old_dir / "page-1.png").write_bytes(b"x")
    _age(old_dir / "page-1.png", 20)
    _age(old_dir, 20)
    busy_dir = root / "busy"                       # an old folder with a fresh file inside is kept
    busy_dir.mkdir()
    (busy_dir / "fresh.png").write_bytes(b"x")
    _age(busy_dir, 20)
    old_file = root / "old.json"
    old_file.write_text("{}", encoding="utf-8")
    _age(old_file, 15)
    new_file = root / "new.json"
    new_file.write_text("{}", encoding="utf-8")
    outside = tmp_path / "user folder"
    outside.mkdir()
    (outside / "keep.pdf").write_bytes(b"keep")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
        linked = True
    except (OSError, NotImplementedError):
        linked = False                             # symlinks need a privilege on some Windows setups
    aged = linked and link_age == "aged"
    if aged:
        _age(link, 30)

    removed = fsutil.prune(root, older_than_days=14)

    left = sorted(p.name for p in root.iterdir())
    assert removed == 2 + aged, left
    assert left == sorted(["busy", "new.json"] + (["link"] if linked and not aged else [])), left
    assert (outside / "keep.pdf").read_bytes() == b"keep", "a link is removed, never followed"
    assert fsutil.prune(root) == 0
    _clock_past(root)
    assert fsutil.prune(root, older_than_days=0) == 2 + (linked and not aged), sorted(p.name for p in root.iterdir())
    assert list(root.iterdir()) == []
    assert (outside / "keep.pdf").read_bytes() == b"keep", "a link is removed, never followed"


def test_prune_refuses_a_root_outside_the_cache(tmp_path: Path):
    (tmp_path / "doc.pdf").write_bytes(b"x")
    _age(tmp_path / "doc.pdf", 100)
    with pytest.raises(UsageError) as info:
        fsutil.prune(tmp_path)
    assert info.value.code == "prune_outside_cache"
    assert (tmp_path / "doc.pdf").exists()


@pytest.mark.parametrize("where", ["", "records", "packs", "canary", "tessdata"])
def test_prune_refuses_the_cache_root_and_its_durable_folders(where: str):
    root = platforms.cache_root()
    target = root / where if where else root
    target.mkdir(parents=True, exist_ok=True)
    keep = target / "r-20261001-0a1b2c3d.json"
    keep.write_text("{}", encoding="utf-8")
    _age(keep, 100)
    with pytest.raises(UsageError) as info:
        fsutil.prune(target)
    assert info.value.code == "prune_outside_cache"
    assert keep.exists()


def test_prune_accepts_staging_and_a_folder_below_work():
    staging = fsutil.work_dir("staging", "")
    (staging / "r-20261001-0a1b2c3d").mkdir()
    _age(staging / "r-20261001-0a1b2c3d", 30)
    assert fsutil.prune(staging) == 1
    below = fsutil.work_dir("work", "ab" * 32 + "-0.1.0")
    (below / "page-1.png").write_bytes(b"x")
    _age(below / "page-1.png", 30)
    assert fsutil.prune(below) == 1


def test_prune_of_a_missing_root_is_zero():
    assert fsutil.prune(platforms.cache_root() / "staging" / "never-made") == 0


# --------------------------------------------------------------------------- utf8_stdio

def _windows_text_stream(monkeypatch: pytest.MonkeyPatch, raw: io.BytesIO, encoding: str):
    """A text stream that ends lines as one on Windows does. The C io module picks "\\r\\n" when it's
    compiled for Windows, so os.linesep can't change it here; the pure-Python _pyio one reads os.linesep
    when it's made and again on reconfigure, so it stands in for a Windows console or pipe."""
    import _pyio

    monkeypatch.setattr(os, "linesep", "\r\n")
    return _pyio.TextIOWrapper(raw, encoding=encoding)


@pytest.mark.parametrize("line_ends", ["this_os", "windows"])
def test_utf8_stdio_lets_a_cp1252_stream_print_thai_and_emoji(monkeypatch: pytest.MonkeyPatch, line_ends: str):
    raw = io.BytesIO()
    if line_ends == "windows":
        stream = _windows_text_stream(monkeypatch, raw, "cp1252")
        end = "\r\n"
    else:
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        end = os.linesep
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", err)
    with pytest.raises(UnicodeEncodeError):
        print("ตัวอย่าง 📎", file=stream, flush=True)
    raw.seek(0)
    raw.truncate()
    fsutil.utf8_stdio()
    print("ตัวอย่าง 📎", flush=True)
    # the stream ends lines as the platform's text streams do (\r\n on Windows); utf8_stdio changes only the encoding
    assert raw.getvalue().decode("utf-8") == "ตัวอย่าง 📎" + end
    assert sys.stderr.encoding.lower().replace("-", "") == "utf8"


# --------------------------------------------------------------------------- busy files on Windows

class _Clock:
    """``time`` as fsutil sees it: ``sleep`` moves a made-up clock instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _Win:
    """``sys`` as fsutil sees it, with ``platform`` win32."""

    platform = "win32"

    def __getattr__(self, name: str):
        return getattr(sys, name)


class _Posix:
    """``sys`` as fsutil sees it, with ``platform`` linux: the off-Windows half of a test must not
    depend on the runner, since CI runs every test on Windows too."""

    platform = "linux"

    def __getattr__(self, name: str):
        return getattr(sys, name)


def _busy_error(winerror: int = 32) -> PermissionError:
    exc = PermissionError(13, "The process cannot access the file because it is being used by another process")
    exc.winerror = winerror                 # what Windows' OSError carries; set by hand off Windows
    return exc


def test_windows_busy_is_only_errors_5_and_32_on_windows(monkeypatch):
    monkeypatch.setattr(fsutil, "sys", _Posix())
    assert not fsutil.windows_busy(_busy_error(32))              # macOS / Linux: never busy
    monkeypatch.setattr(fsutil, "sys", _Win())
    assert fsutil.windows_busy(_busy_error(5)) and fsutil.windows_busy(_busy_error(32))
    assert not fsutil.windows_busy(_busy_error(3)) and not fsutil.windows_busy(PermissionError(13, "denied"))
    assert not fsutil.windows_busy(FileExistsError(17, "exists"))


def test_retry_busy_waits_out_a_short_busy_spell_on_windows(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(fsutil, "sys", _Win())
    monkeypatch.setattr(fsutil, "time", clock)
    calls = []

    def op(x):
        calls.append(x)
        if len(calls) < 4:
            raise _busy_error(5)
        return "done"

    assert fsutil.retry_busy(op, 1) == "done"
    assert len(calls) == 4 and len(clock.sleeps) == 3 and clock.now < fsutil.REPLACE_PATIENCE_S


def test_retry_busy_raises_at_once_off_windows_and_for_other_errors(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(fsutil, "time", clock)

    def op():
        raise _busy_error(32)

    monkeypatch.setattr(fsutil, "sys", _Posix())
    with pytest.raises(PermissionError):
        fsutil.retry_busy(op)                                     # macOS / Linux: no retry
    monkeypatch.setattr(fsutil, "sys", _Win())

    def other():
        raise _busy_error(3)

    with pytest.raises(PermissionError):
        fsutil.retry_busy(other)                                  # Windows, but not a busy file
    assert clock.sleeps == []


def test_on_windows_overwrite_onto_an_output_held_open_is_output_in_use(tmp_path: Path, monkeypatch):
    """A PDF viewer keeps the file open: the replace is tried for the patience, then the run says why
    (exit 2) instead of an unexplained PermissionError; dest and the folder are as they were."""
    built = tmp_path / "built.pdf"
    built.write_bytes(b"%PDF-1.7 new")
    dest = tmp_path / "out" / "scan (searchable).pdf"
    dest.parent.mkdir()
    dest.write_bytes(b"%PDF-1.7 old, open in a viewer")
    clock = _Clock()
    monkeypatch.setattr(fsutil, "time", clock)
    monkeypatch.setattr(fsutil, "windows_busy", lambda exc: isinstance(exc, PermissionError)
                        and getattr(exc, "winerror", None) in (5, 32))
    real = os.replace

    def replace(src, dst, *a, **k):
        if Path(dst) == dest:
            raise _busy_error(32)
        return real(src, dst, *a, **k)

    monkeypatch.setattr(fsutil.os, "replace", replace)
    with pytest.raises(UsageError) as info:
        fsutil.atomic_write(built, dest, overwrite=True)
    assert info.value.code == "output_in_use" and info.value.exit_code == EXIT_USAGE
    assert str(dest) in info.value.message
    assert fsutil.REPLACE_PATIENCE_S <= clock.now < fsutil.REPLACE_PATIENCE_S + 1
    assert dest.read_bytes() == b"%PDF-1.7 old, open in a viewer"
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


def test_on_windows_overwrite_waits_for_a_reader_that_lets_go(tmp_path: Path, monkeypatch):
    built = tmp_path / "built.pdf"
    built.write_bytes(b"%PDF-1.7 new")
    dest = tmp_path / "out.pdf"
    dest.write_bytes(b"%PDF-1.7 old")
    clock = _Clock()
    monkeypatch.setattr(fsutil, "time", clock)
    monkeypatch.setattr(fsutil, "windows_busy", lambda exc: getattr(exc, "winerror", None) in (5, 32))
    real, refused = os.replace, []

    def replace(src, dst, *a, **k):
        if Path(dst) == dest and len(refused) < 3:               # an indexer reads it for a moment
            refused.append(1)
            raise _busy_error(5)
        return real(src, dst, *a, **k)

    monkeypatch.setattr(fsutil.os, "replace", replace)
    fsutil.atomic_write(built, dest, overwrite=True)
    assert dest.read_bytes() == b"%PDF-1.7 new" and len(refused) == 3


def test_on_windows_a_compare_and_swap_onto_an_open_output_is_output_in_use(tmp_path: Path, monkeypatch):
    """apply replacing its own output: dest can't be set aside while a viewer has it open."""
    dest = tmp_path / "out.pdf"
    dest.write_bytes(b"%PDF-1.7 old")
    clock = _Clock()
    monkeypatch.setattr(fsutil, "time", clock)
    monkeypatch.setattr(fsutil, "windows_busy", lambda exc: getattr(exc, "winerror", None) in (5, 32))
    monkeypatch.setattr(fsutil, "_rename_excl", lambda src, dst: errno.ENOTSUP)   # Windows has no renamex_np
    real = os.rename

    def rename(src, dst, *a, **k):
        if Path(src) == dest:
            raise _busy_error(32)
        return real(src, dst, *a, **k)

    monkeypatch.setattr(fsutil.os, "rename", rename)
    with pytest.raises(UsageError) as info:
        fsutil._set_aside(dest, fsutil._aside_path(dest))
    assert info.value.code == "output_in_use" and dest.read_bytes() == b"%PDF-1.7 old"
