"""Shared test setup.

* The network is blocked for every test that isn't marked ``network``. An audit hook refuses
  socket connects, binds and sends to any address outside loopback, and every DNS lookup except
  ``localhost``, with ``NetworkBlocked`` (an OSError). Loopback stays open because Python builds
  ``socket.socketpair`` from loopback sockets on Windows, which asyncio (and so pyHanko) needs.
  Each refused attempt is also recorded, so a test can prove nothing tried, even when the code
  under test swallows the error: use the ``network_attempts`` fixture. Because loopback is open,
  the proxy variables (``HTTP_PROXY``, ``HTTPS_PROXY``, ``ALL_PROXY`` in both cases) are removed
  and ``NO_PROXY`` is set to ``*`` for those tests too: otherwise a download sent to a local proxy
  would reach the network from outside this process and leave no attempt on record.
* ``PDFINTAKE_CACHE`` points at a temporary folder for the whole session unless the environment
  already sets it (CI does, to test a non-ASCII cache path), so tests never touch a real cache.
* ``pinned_asset(pack, key)`` is the one way a test gets a real model or font (decided 2026-11-05):
  the file ``packs/<pack>/manifest.json`` pins under ``key``, at ``<root>/<pack>/<file>``, checked
  against its pinned size and SHA-256. ``<root>`` is ``PDFINTAKE_TEST_MODELS``, else
  ``~/.pdfintake-dev/models``; ``tools/fetch_test_models.py`` fills it. A missing or different file
  skips the test, naming the file and the command that fetches it. ``pinned_assets("pack:key", ...)``
  does several at once and names every missing one in a single skip. The file itself must be a
  regular file, not a link, the same rule as the tool and ``datawheel.download_sources``. In a test
  marked ``network`` with ``PDFINTAKE_NETWORK_TESTS=1``, ``fetch=True`` downloads what is missing,
  and a download that fails fails the test with the download's own error, never a skip.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import ipaddress
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


class NetworkBlocked(OSError):
    """Raised for a socket operation outside loopback in a test not marked ``network``."""


_GUARD: dict[str, Any] = {"on": False, "attempts": []}

_ADDRESS_EVENTS = {"socket.connect": 1, "socket.bind": 1, "socket.sendto": 1, "socket.sendmsg": 1}
_LOOKUP_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                  "socket.gethostbyaddr", "socket.getnameinfo"}


def _loopback_host(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    h = host.strip().lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _allowed_address(address: Any) -> bool:
    if address is None or isinstance(address, (str, bytes, bytearray)):
        return True                                   # AF_UNIX path, or a send on a connected socket
    if isinstance(address, tuple) and address:
        return _loopback_host(address[0])
    return False


def _network_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _GUARD["on"] or not event.startswith("socket."):
        return
    if event in _ADDRESS_EVENTS:
        address = args[_ADDRESS_EVENTS[event]] if len(args) > 1 else None
        if _allowed_address(address):
            return
    elif event in _LOOKUP_EVENTS:
        if event == "socket.getnameinfo":
            target = args[0][0] if args and isinstance(args[0], tuple) and args[0] else None
        else:
            target = args[0] if args else None
        if _loopback_host(target):
            return
    else:
        return
    _GUARD["attempts"].append((event, repr(args)[:200]))
    raise NetworkBlocked(errno.ENETUNREACH, f"network blocked in tests ({event}); mark the test 'network' if it needs it")


sys.addaudithook(_network_hook)


PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("network") is not None:
        yield
        return
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    _GUARD["attempts"] = []
    _GUARD["on"] = True
    try:
        yield
    finally:
        _GUARD["on"] = False


@pytest.fixture
def network_attempts() -> list[tuple[str, str]]:
    """The socket operations refused so far in this test: (audit event, its arguments)."""
    return _GUARD["attempts"]


_OWN_CACHE: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    # Set before collection, so module-level code in test files sees it too.
    if not os.environ.get("PDFINTAKE_CACHE"):
        folder = tempfile.mkdtemp(prefix="pdfintake-test-cache-")
        _OWN_CACHE.append(folder)
        os.environ["PDFINTAKE_CACHE"] = folder


def pytest_unconfigure(config: pytest.Config) -> None:
    for folder in _OWN_CACHE:
        if os.environ.get("PDFINTAKE_CACHE") == folder:
            del os.environ["PDFINTAKE_CACHE"]
        shutil.rmtree(folder, ignore_errors=True)
    _OWN_CACHE.clear()


# --------------------------------------------------------------------------- the pinned test models

REPO = Path(__file__).resolve().parents[1]
MODELS_ENV = "PDFINTAKE_TEST_MODELS"
DEFAULT_MODELS_ROOT = Path.home() / ".pdfintake-dev" / "models"
FETCH_TOOL = REPO / "tools" / "fetch_test_models.py"

_MANIFESTS: dict[str, dict[str, Any]] = {}
_VERIFIED: dict[tuple[str, int, int, int], str] = {}      # (path, size, mtime_ns, inode) -> sha256


def models_root() -> Path:
    """``PDFINTAKE_TEST_MODELS`` when set, else ``~/.pdfintake-dev/models``."""
    env = os.environ.get(MODELS_ENV)
    return Path(env).expanduser() if env else DEFAULT_MODELS_ROOT


def manifest_entry(pack: str, key: str) -> dict[str, Any]:
    """The model or font entry ``packs/<pack>/manifest.json`` pins under ``key``.

    An unknown pack or key is a mistake in the test, so it raises ``LookupError`` rather than skipping.
    """
    if pack not in _MANIFESTS:
        path = REPO / "packs" / pack / "manifest.json"
        try:
            _MANIFESTS[pack] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LookupError(f"there is no pack {pack!r}: {path} is missing") from None
    data = _MANIFESTS[pack]
    for group in ("models", "fonts"):
        for item in data.get(group) or ():
            if isinstance(item, dict) and item.get("key") == key:
                return item
    raise LookupError(f"packs/{pack}/manifest.json pins no model or font with key {key!r}")


def _sha256(path: Path, st: os.stat_result) -> str:
    memo = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if memo not in _VERIFIED:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _VERIFIED[memo] = h.hexdigest()
    return _VERIFIED[memo]


def _locate(pack: str, key: str) -> tuple[Path | None, str]:
    """The verified path of ``pack:key``, or None and why not."""
    item = manifest_entry(pack, key)
    path = models_root() / pack / str(item["file"])
    label = f"{pack}:{key} ({path})"
    try:
        st = os.lstat(path)                       # a link is refused, as the tool and download_sources do
    except FileNotFoundError:
        return None, f"{label} is missing"
    except OSError as exc:
        return None, f"{label} can't be checked: {exc.strerror or exc}"
    if not stat.S_ISREG(st.st_mode):
        return None, f"{label} isn't a regular file (a link or a folder doesn't count)"
    if st.st_size != item["bytes"]:
        return None, f"{label} is {st.st_size} bytes; the manifest pins {item['bytes']}"
    got = _sha256(path, st)
    if got != item["sha256"]:
        return None, f"{label} has SHA-256 {got[:12]}...; the manifest pins {item['sha256'][:12]}..."
    return path, ""


def _fetch_tool() -> ModuleType:
    name = "_pdfintake_fetch_test_models"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FETCH_TOOL)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def _skip_reason(reasons: list[str]) -> str:
    return (f"pinned test model(s) not available: {'; '.join(reasons)}. "
            f"Fetch them with: python3 tools/fetch_test_models.py (root: {MODELS_ENV} or {DEFAULT_MODELS_ROOT})")


def _network_test() -> bool:
    """True inside a test marked ``network`` (the socket guard is off), False in any other test."""
    return not _GUARD["on"]


def _fetch_missing(missing: list[tuple[str, str, str, str]], found: dict[str, Path]) -> None:
    """Fetch the packs of ``missing``, then fail (never skip) naming each download error and each file still wrong."""
    tool = _fetch_tool()
    errors: list[str] = []
    for pack in sorted({m[1] for m in missing}):
        try:
            rows = tool.fetch_pack(pack, models_root())
        except tool.ManifestProblem as exc:
            errors.append(f"{pack}: {exc}")
            continue
        errors += [f"{pack}/{name}: {detail}" for name, state, detail in rows if state == "failed"]
    still: list[str] = []
    for ref, pack, key, _why in missing:
        path, why = _locate(pack, key)
        if path is None:
            still.append(why)
        else:
            found[ref] = path
    if still:
        pytest.fail(f"fetching the pinned test models into {models_root()} failed: "
                    f"{'; '.join(errors) or 'the download reported no error'}. Still not available: {'; '.join(still)}",
                    pytrace=False)


def pinned_assets(*refs: str, fetch: bool = False) -> dict[str, Path]:
    """``{"pack:key": path}`` for every ref, each the verified pinned file; one skip names all missing.

    ``fetch=True`` is for tests marked ``network`` only, and calling it from any other test fails.
    With ``PDFINTAKE_NETWORK_TESTS=1`` a missing or wrong file's pack is fetched first, with
    ``tools/fetch_test_models.py``, into the same root; if a file is still missing or wrong afterwards
    the test fails with each download error (decided 2026-11-05), because the network job exists to
    catch a broken upstream source. Without ``PDFINTAKE_NETWORK_TESTS=1`` a missing file skips.
    """
    if fetch and not _network_test():
        pytest.fail("pinned_assets(fetch=True) needs a test marked network", pytrace=False)
    pairs: list[tuple[str, str, str]] = []
    for ref in refs:
        pack, sep, key = ref.partition(":")
        if not sep or not pack or not key:
            raise LookupError(f"{ref!r} isn't 'pack:key'")
        pairs.append((ref, pack, key))
    found: dict[str, Path] = {}
    missing: list[tuple[str, str, str, str]] = []
    for ref, pack, key in pairs:
        path, why = _locate(pack, key)
        if path is None:
            missing.append((ref, pack, key, why))
        else:
            found[ref] = path
    if missing and fetch and os.environ.get("PDFINTAKE_NETWORK_TESTS") == "1":
        _fetch_missing(missing, found)
        missing = []
    if missing:
        pytest.skip(_skip_reason([m[3] for m in missing]))
    return found


def pinned_asset(pack: str, key: str, *, fetch: bool = False) -> Path:
    """The verified file ``packs/<pack>/manifest.json`` pins under ``key``, or a skip naming it."""
    return pinned_assets(f"{pack}:{key}", fetch=fetch)[f"{pack}:{key}"]
