from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from pdfintake import integrity
from pdfintake.errors import EXIT_SETUP, EXIT_STALE, EXIT_USAGE, InputChanged, IntegrityError, StaleRecord, UsageError

EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ABC_SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# --------------------------------------------------------------------------- hashes

def test_known_hashes(tmp_path: Path):
    assert integrity.sha256_bytes(b"") == EMPTY_SHA
    assert integrity.sha256_bytes(b"abc") == ABC_SHA
    p = tmp_path / "abc.bin"
    p.write_bytes(b"abc")
    assert integrity.sha256_file(p) == ABC_SHA
    big = tmp_path / "big.bin"
    data = os.urandom(3 * (1 << 20) + 17)
    big.write_bytes(data)
    assert integrity.sha256_file(big) == integrity.sha256_bytes(data)


def test_verify_file_passes_and_accepts_upper_case(tmp_path: Path):
    p = tmp_path / "model.traineddata"
    p.write_bytes(b"abc")
    integrity.verify_file(p, ABC_SHA)
    integrity.verify_file(p, ABC_SHA.upper(), size=3)


@pytest.mark.parametrize("case,code", [("hash", "hash_mismatch"), ("size", "size_mismatch"),
                                       ("missing", "file_missing"), ("folder", "not_a_file"),
                                       ("bad_hash", "bad_expected_hash")])
def test_verify_file_names_the_file(tmp_path: Path, case: str, code: str):
    p = tmp_path / "fonts" / "Example Sans ไทย.ttf"
    p.parent.mkdir()
    if case == "folder":
        p.mkdir()
    elif case != "missing":
        p.write_bytes(b"abd")
    sha = "not-a-hash" if case == "bad_hash" else ABC_SHA
    size = 4 if case == "size" else 3
    with pytest.raises(IntegrityError) as info:
        integrity.verify_file(p, sha, size)
    err = info.value
    assert err.code == code and err.exit_code == EXIT_SETUP
    assert str(p) in str(err) and err.detail["path"] == str(p)


def test_verify_file_without_size_still_checks_the_hash(tmp_path: Path):
    p = tmp_path / "x.onnx"
    p.write_bytes(b"abcd")
    with pytest.raises(IntegrityError, match="x.onnx has SHA-256"):
        integrity.verify_file(p, ABC_SHA)


def test_a_file_that_vanishes_between_the_stat_and_the_read_is_an_integrity_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The race: verify_file stats the file, then a concurrent run discards it before it is opened.
    p = tmp_path / "models" / "xa.traineddata"
    p.parent.mkdir()
    p.write_bytes(b"abc")
    real_stat = os.stat
    target = os.fspath(p)
    vanished: list[str] = []

    def stat_then_vanish(path, *args, **kwargs):        # os.stat is shared with pathlib: no Path calls here
        st = real_stat(path, *args, **kwargs)
        if not vanished and os.fspath(path) == target:
            vanished.append(target)
            os.unlink(target)
        return st

    monkeypatch.setattr(integrity.os, "stat", stat_then_vanish)
    with pytest.raises(IntegrityError) as info:
        integrity.verify_file(p, ABC_SHA, 3)
    err = info.value
    assert err.code == "file_missing" and err.exit_code == EXIT_SETUP
    assert str(p) in str(err) and err.detail["path"] == str(p)
    assert vanished == [target] and not isinstance(err, OSError)


def test_sha256_file_turns_an_oserror_into_an_integrity_error(tmp_path: Path):
    with pytest.raises(IntegrityError) as info:
        integrity.sha256_file(tmp_path / "gone.onnx")
    assert info.value.code == "file_missing" and info.value.detail["path"] == str(tmp_path / "gone.onnx")
    with pytest.raises(IntegrityError) as info:
        integrity.sha256_file(tmp_path)               # a folder: IsADirectoryError, or PermissionError on Windows
    assert info.value.code == "unreadable" and str(tmp_path) in str(info.value)


def test_a_file_that_fails_mid_read_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = tmp_path / "font.ttf"
    p.write_bytes(b"abc")
    import builtins

    real_open = builtins.open

    class Failing:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

        def read(self, n):
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: Failing(real_open(path, *a, **k))
                        if Path(path) == p else real_open(path, *a, **k))
    with pytest.raises(IntegrityError) as info:
        integrity.verify_file(p, ABC_SHA, 3)
    assert info.value.code == "unreadable" and "Input/output error" in info.value.message


def test_a_stat_refused_for_permission_is_unreadable_not_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Only ENOENT means missing: any other OSError on the stat (EACCES on a parent folder) is unreadable.
    import errno

    p = tmp_path / "models" / "xa.traineddata"
    p.parent.mkdir()
    p.write_bytes(b"abc")
    real_stat = os.stat
    target = os.fspath(p)

    def refuse(path, *args, **kwargs):                  # os.stat is shared with pathlib: no Path calls here
        if os.fspath(path) == target:
            raise PermissionError(errno.EACCES, "Permission denied", target)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(integrity.os, "stat", refuse)
    with pytest.raises(IntegrityError) as info:
        integrity.verify_file(p, ABC_SHA, 3)
    err = info.value
    assert err.code == "unreadable" and err.exit_code == EXIT_SETUP
    assert "Permission denied" in err.message and err.detail["path"] == str(p)


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions and a non-root user")
def test_a_file_in_an_unsearchable_folder_is_unreadable(tmp_path: Path):
    folder = tmp_path / "models"
    folder.mkdir()
    p = folder / "xa.traineddata"
    p.write_bytes(b"abc")
    folder.chmod(0)
    try:
        with pytest.raises(IntegrityError) as info:
            integrity.verify_file(p, ABC_SHA, 3)
        assert info.value.code == "unreadable" and str(p) in str(info.value)
    finally:
        folder.chmod(0o700)


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions and a non-root user")
def test_an_unreadable_file_is_unreadable(tmp_path: Path):
    p = tmp_path / "locked.onnx"
    p.write_bytes(b"abc")
    p.chmod(0)
    try:
        with pytest.raises(IntegrityError) as info:
            integrity.verify_file(p, ABC_SHA, 3)
        assert info.value.code == "unreadable" and str(p) in str(info.value)
    finally:
        p.chmod(0o600)


# --------------------------------------------------------------------------- InputSeal

def _input(tmp_path: Path, name: str = "scan.pdf", data: bytes = b"%PDF-1.7 example body %%EOF") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_seal_takes_hash_size_and_mtime(tmp_path: Path):
    a = _input(tmp_path, "a.pdf", b"abc")
    b = _input(tmp_path, "b.jpg", b"")
    seal = integrity.InputSeal.take([a, b])
    assert seal.paths == [a, b] and all(p.is_absolute() for p in seal.paths)
    assert seal.sha256 == [ABC_SHA, EMPTY_SHA] and seal.sizes == [3, 0]
    assert seal.mtimes_ns == [a.stat().st_mtime_ns, b.stat().st_mtime_ns]
    seal.check()


def test_seal_detects_a_one_byte_change_with_the_same_size_and_mtime(tmp_path: Path):
    p = _input(tmp_path)
    seal = integrity.InputSeal.take([p])
    st = p.stat()
    data = bytearray(p.read_bytes())
    data[10] ^= 0x01
    p.write_bytes(bytes(data))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert p.stat().st_size == st.st_size and p.stat().st_mtime_ns == st.st_mtime_ns
    with pytest.raises(InputChanged) as info:
        seal.check()
    assert info.value.exit_code == EXIT_STALE and info.value.detail["field"] == "sha256"
    assert str(p) in str(info.value)


def test_seal_detects_size_and_mtime_changes_before_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    a = _input(tmp_path, "a.pdf")
    b = _input(tmp_path, "b.pdf")
    seal = integrity.InputSeal.take([a, b])
    hashed: list[Path] = []
    real = integrity.sha256_file
    monkeypatch.setattr(integrity, "sha256_file", lambda p: hashed.append(Path(p)) or real(p))
    b.write_bytes(b"longer than before, so the size changed")
    with pytest.raises(InputChanged) as info:
        seal.check()
    assert info.value.detail["field"] == "size" and hashed == [], "size and mtime are checked before any hash"
    b.write_bytes(a.read_bytes())
    os.utime(b, ns=(0, seal.mtimes_ns[1] + 1_000_000_000))
    with pytest.raises(InputChanged) as info:
        seal.check()
    assert info.value.detail["field"] == "mtime"


def test_seal_detects_a_removed_input(tmp_path: Path):
    p = _input(tmp_path)
    seal = integrity.InputSeal.take([p])
    p.unlink()
    with pytest.raises(InputChanged) as info:
        seal.check()
    assert info.value.code == "input_missing"


def test_take_refuses_missing_and_non_file_inputs(tmp_path: Path):
    with pytest.raises(UsageError) as info:
        integrity.InputSeal.take([tmp_path / "nope.pdf"])
    assert info.value.code == "input_missing" and info.value.exit_code == EXIT_USAGE
    with pytest.raises(UsageError) as info:
        integrity.InputSeal.take([tmp_path])
    assert info.value.code == "input_not_file"


def test_take_and_check_turn_a_vanished_input_into_their_own_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = _input(tmp_path)
    seal = integrity.InputSeal.take([p])
    real = integrity.sha256_file

    def vanish_then_hash(path):
        Path(path).unlink()
        return real(path)                            # IntegrityError file_missing

    monkeypatch.setattr(integrity, "sha256_file", vanish_then_hash)
    with pytest.raises(InputChanged) as info:
        seal.check()
    assert info.value.code == "input_missing" and str(p) in str(info.value)
    q = _input(tmp_path, "other.pdf")
    with pytest.raises(UsageError) as uinfo:
        integrity.InputSeal.take([q])
    assert uinfo.value.code == "input_missing" and str(q) in str(uinfo.value)


def test_take_notices_a_file_changing_while_it_is_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = _input(tmp_path)
    real = integrity.sha256_file

    def hash_then_touch(path):
        digest = real(path)
        with open(path, "ab") as f:
            f.write(b"more")
        return digest

    monkeypatch.setattr(integrity, "sha256_file", hash_then_touch)
    with pytest.raises(InputChanged):
        integrity.InputSeal.take([p])


def test_seal_json_round_trip(tmp_path: Path):
    p = _input(tmp_path, "Ünïcødé ไทย 📎.pdf")
    seal = integrity.InputSeal.take([p])
    data = json.loads(json.dumps(seal.to_json(), ensure_ascii=False))
    assert set(data) == {"paths", "sha256", "sizes", "mtimes_ns"}
    again = integrity.InputSeal.from_json(data)
    assert again == seal
    again.check()


@pytest.mark.parametrize("broken", [
    {},
    {"paths": ["/x"], "sha256": [ABC_SHA], "sizes": [1]},
    {"paths": ["/x", "/y"], "sha256": [ABC_SHA], "sizes": [1], "mtimes_ns": [1]},
    {"paths": ["/x"], "sha256": ["zz"], "sizes": [1], "mtimes_ns": [1]},
    {"paths": ["relative.pdf"], "sha256": [ABC_SHA], "sizes": [1], "mtimes_ns": [1]},
    {"paths": ["/x"], "sha256": [ABC_SHA], "sizes": ["many"], "mtimes_ns": [1]},
    {"paths": ["/x"], "sha256": [ABC_SHA], "sizes": [-1], "mtimes_ns": [1]},
])
def test_from_json_refuses_a_malformed_seal(broken: dict):
    if broken.get("paths") == ["/x"] and sys.platform == "win32":
        broken = {**broken, "paths": ["C:\\x"]}
    with pytest.raises(StaleRecord) as info:
        integrity.InputSeal.from_json(broken)
    assert info.value.code == "seal_unreadable"


# --------------------------------------------------------------------------- no_subprocess

def test_subprocess_raises_inside_no_subprocess_and_works_outside():
    with integrity.no_subprocess():
        with pytest.raises(PermissionError, match="subprocess.Popen"):
            subprocess.run(["true"])
        with pytest.raises(PermissionError):
            subprocess.run([sys.executable, "-c", "pass"])
        with pytest.raises(PermissionError, match="os.system"):
            os.system("true")
        if hasattr(os, "posix_spawn"):
            with pytest.raises(PermissionError, match="os.posix_spawn"):
                os.posix_spawn(sys.executable, [sys.executable, "-c", "pass"], dict(os.environ))
        if hasattr(os, "spawnv"):
            with pytest.raises(PermissionError):
                os.spawnv(os.P_WAIT, sys.executable, [sys.executable, "-c", "pass"])
        with pytest.raises(PermissionError):
            os.popen("true")
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0
    if sys.platform != "win32":
        assert subprocess.run(["true"]).returncode == 0


def test_no_subprocess_nests_and_covers_other_threads():
    errors: list[BaseException] = []

    def worker():
        try:
            subprocess.run([sys.executable, "-c", "pass"])
        except BaseException as exc:          # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    with integrity.no_subprocess():
        with integrity.no_subprocess():
            pass
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        with pytest.raises(PermissionError):
            subprocess.run([sys.executable, "-c", "pass"])
    assert len(errors) == 1 and isinstance(errors[0], PermissionError)
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0


def test_no_subprocess_ends_even_when_the_body_raises():
    with pytest.raises(ValueError):
        with integrity.no_subprocess():
            raise ValueError("boom")
    assert integrity._GUARD["depth"] == 0
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0


def test_the_hook_is_installed_once(monkeypatch: pytest.MonkeyPatch):
    added: list = []
    monkeypatch.setattr(integrity.sys, "addaudithook", added.append)
    monkeypatch.setitem(integrity._GUARD, "installed", False)
    for _ in range(3):
        with integrity.no_subprocess():
            pass
    assert added == [integrity._audit]


def _mp_contexts() -> list[str]:
    import multiprocessing

    return [m for m in ("spawn", "forkserver") if m in multiprocessing.get_all_start_methods()]


@pytest.mark.parametrize("method", _mp_contexts())
def test_multiprocessing_spawn_and_forkserver_raise_inside_no_subprocess(method: str):
    import multiprocessing
    import time

    ctx = multiprocessing.get_context(method)
    with integrity.no_subprocess():
        proc = ctx.Process(target=time.sleep, args=(0,))
        with pytest.raises(PermissionError, match="refused"):
            proc.start()
    proc = ctx.Process(target=time.sleep, args=(0,))
    proc.start()
    proc.join(60)
    assert proc.exitcode == 0, "outside the guard a process starts as before"


@pytest.mark.skipif("forkserver" not in _mp_contexts(), reason="no fork server on this platform")
def test_a_fork_server_started_before_the_guard_forks_nothing_inside_it():
    import multiprocessing
    import time

    ctx = multiprocessing.get_context("forkserver")
    warm = ctx.Process(target=time.sleep, args=(0,))
    warm.start()                                   # the fork server is now running
    warm.join(60)
    assert warm.exitcode == 0
    with integrity.no_subprocess():
        with pytest.raises(PermissionError, match="refused"):
            ctx.Process(target=time.sleep, args=(0,)).start()


def test_a_process_pool_raises_inside_no_subprocess():
    import concurrent.futures
    import multiprocessing

    with integrity.no_subprocess():
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        try:
            with pytest.raises((PermissionError, concurrent.futures.process.BrokenProcessPool)):
                pool.submit(abs, -1).result(timeout=60)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


_LOOKUP_NAMES = ("__import__", "import_module")


def _name_lookups(tree) -> list:
    import ast

    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Attribute) and n.func.attr in _LOOKUP_NAMES)
                 or (isinstance(n.func, ast.Name) and n.func.id in _LOOKUP_NAMES))]


def test_the_lookup_scan_catches_every_spelling():
    import ast

    for src in ("importlib.import_module('m')", "import_module('m')", "__import__('m')",
                "importlib.__import__('m')", "builtins.__import__('m')"):
        assert _name_lookups(ast.parse(src)), src
    assert _name_lookups(ast.parse("import m\nfrom x import y\nm.f()")) == []


def test_posixsubprocess_is_a_literal_import_and_no_name_lookup():
    # The source checks run on every platform, Windows included; only the module identity needs POSIX.
    import ast

    tree = ast.parse(Path(integrity.__file__).read_text(encoding="utf-8"))
    literal = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == "_posixsubprocess" for a in n.names)]
    assert literal, "integrity imports _posixsubprocess literally (decided 2026-11-05)"
    assert _name_lookups(tree) == [], "no importlib or __import__ name lookups left"
    if sys.platform == "win32":
        assert integrity._posixsubprocess is None
    else:
        assert integrity._posixsubprocess is sys.modules["_posixsubprocess"]


def test_the_unaudited_starts_are_wrapped_once():
    with integrity.no_subprocess():
        pass
    posix = sys.modules.get("_posixsubprocess")
    if posix is not None:
        wrapped = posix.fork_exec
        assert getattr(wrapped, integrity._WRAPPED_MARK, False)
        with integrity.no_subprocess():
            pass
        assert posix.fork_exec is wrapped, "a second guard doesn't wrap the wrapper"
