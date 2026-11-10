"""Repo hygiene, checked on the source tree beside this file.

* ``src/`` never starts a process: no ``subprocess`` (or ``pty``, ``_posixsubprocess``), no
  ``multiprocessing`` (or ``_multiprocessing``, ``concurrent.futures.process``,
  ``concurrent.futures.ProcessPoolExecutor``), no ``os.system``, ``os.popen``, ``os.exec*``,
  ``os.spawn*``, ``os.posix_spawn*``, ``os.fork*``, ``os.startfile`` and no
  ``asyncio.create_subprocess_*``. The ``platform`` module may be imported, but only the helpers
  in ``PLATFORM_ALLOWED`` may be used: the others can shell out (``architecture``, ``processor``,
  ``uname`` and ``win32_ver`` on some systems). The runtime audit hook covers the dependencies;
  this covers our own code, statically. One exception (``IMPORT_ALLOWED``): ``integrity.py`` imports
  ``_posixsubprocess`` with a literal ``import _posixsubprocess``, because its run guard wraps
  ``fork_exec``. No other file may import it, and ``integrity.py`` may not import it any other way.
* ``src/`` never imports PyMuPDF (``fitz``, ``pymupdf``).
* Every module in ``src/`` imports on all three operating systems, without the optional extras:
  each is imported in a fresh interpreter where the macOS-only modules (``DARWIN_ONLY``) and the
  extras (``OPTIONAL``) can't be imported by pdfintake code, so a top-level import of one fails
  here on the Mac too. Import them inside the function that needs them. ``pdfintake.kit.*`` may
  import the ``[kit]`` and ``[kit-thai]`` extras (``KIT_EXTRAS``). A missing ``pdfintake``
  sibling is allowed only while it is a planned module (``PLANNED``, DESIGN §2.1) not yet on disk.
* Importing any module in ``src/`` touches no network: sockets raise in that interpreter, and any
  attempt (even one the module swallows) fails the test.
* ``datawheel.py`` imports only the standard library, so it runs from a fresh clone.
* ``tests/conftest.py`` works in an interpreter that can't import pdfintake (the CI sandbox job's), and
  still fails when pdfintake is there but its import breaks.
* The manifest schema refuses a model file name with a code suffix and a ``canary.must_find``
  string with no non-whitespace character, and both validators (the loader's walker and
  ``jsonschema``) apply a ``$ref``'s sibling keywords as well as its target.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "pdfintake"

BANNED_MODULES = {"subprocess", "pty", "_posixsubprocess", "multiprocessing", "_multiprocessing",
                  "concurrent.futures.process", "fitz", "pymupdf"}
BANNED_ATTRS = {
    "os": {"system", "popen", "fork", "forkpty", "startfile"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
    "_winapi": {"CreateProcess"},
    "concurrent.futures": {"ProcessPoolExecutor"},
}
BANNED_OS_PREFIXES = ("exec", "spawn", "posix_spawn")
# The banned modules one file may import, by its path inside the package, and only as a plain
# ``import <module>`` (an alias is fine; ``from``, ``__import__`` and ``import_module`` are not):
# integrity.py's run guard wraps _posixsubprocess.fork_exec, which no audit event covers (decided 2026-11-05).
IMPORT_ALLOWED = {"integrity.py": frozenset({"_posixsubprocess"})}
PLATFORM_ALLOWED = {"mac_ver", "python_version", "python_version_tuple", "python_implementation",
                    "freedesktop_os_release"}               # read sys or a file; never start a process
IMPORT_SKIP = {"pdfintake.__main__"}       # runs the CLI when imported

DARWIN_ONLY = {"Vision", "objc", "Quartz", "AppKit", "Foundation", "CoreML", "Cocoa"}
OPTIONAL = {"uharfbuzz", "pythainlp", "pyhanko", "jsonschema"}      # [kit], [kit-thai], [test]
KIT_EXTRAS = {"uharfbuzz", "pythainlp"}                             # pdfintake.kit.* may import these

PLANNED = {f"pdfintake.{m}" for m in (
    "cli flow doctor record datawheel types errors params protocols platforms fsutil integrity refusals quality "
    "intake intake.sniff intake.jpeg intake.images intake.assemble "
    "pdf pdf.render pdf.classify pdf.orient pdf.strip pdf.textlayer pdf.verify pdf.shrink pdf.meta pdf.smaller "
    "pdf.export ocr ocr.scripts ocr.textfix ocr.fills ocr.detect ocr.langplan ocr.pipeline "
    "ocr.engines ocr.engines.tesseract ocr.engines.vision ocr.engines.rapid "
    "ocr.rules ocr.rules.corroborated ocr.rules.primary "
    "langpacks langpacks.manifest langpacks.index langpacks.store langpacks.canary "
    "kit kit.cli kit.corpus kit.render kit.score kit.bench kit.choose kit.verify kit.private kit.report").split()}


def _source_files(pkg: Path = PKG) -> list[Path]:
    return sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path, src: Path = SRC) -> str:
    rel = path.relative_to(src).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _banned_module(name: str) -> bool:
    lowered = name.lower()
    parts = lowered.split(".")
    return any(".".join(parts[:i]) in BANNED_MODULES for i in range(1, len(parts) + 1))


def _banned_attr(module: str, attr: str) -> bool:
    if attr in BANNED_ATTRS.get(module, ()):
        return True
    if module == "platform":
        return attr not in PLATFORM_ALLOWED
    return module == "os" and attr.startswith(BANNED_OS_PREFIXES)


def _dotted(node: ast.AST) -> list[str] | None:
    """``a.b.c`` as ["a", "b", "c"], or None when the chain doesn't start at a plain name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _allowed_imports(path: Path, pkg: Path) -> frozenset[str]:
    """The banned modules ``path`` may import by name: only for the exact file ``IMPORT_ALLOWED`` names."""
    try:
        inside = path.resolve().relative_to(pkg.resolve()).as_posix()
    except ValueError:
        return frozenset()
    return IMPORT_ALLOWED.get(inside, frozenset())


def violations(path: Path, pkg: Path = PKG) -> list[str]:
    """Every process-starting or PyMuPDF use in one file, as "file:line: what".

    ``pkg`` is the package root the file sits in, which decides whether ``IMPORT_ALLOWED`` applies.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    try:
        rel = path.relative_to(SRC.parent).as_posix()
    except ValueError:
        rel = path.name
    allowed = _allowed_imports(path, pkg)
    found: list[str] = []
    aliases: dict[str, str] = {}             # local name -> module (or module attribute) it stands for

    def report(node: ast.AST, what: str) -> None:
        found.append(f"{rel}:{getattr(node, 'lineno', 0)}: {what}")

    for node in ast.walk(tree):                      # imports, and the names modules are bound to
        if isinstance(node, ast.Import):
            for a in node.names:
                if _banned_module(a.name) and a.name not in allowed:
                    report(node, f"import {a.name}")
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    top = a.name.split(".")[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _banned_module(node.module):
                report(node, f"from {node.module} import ...")
            for a in node.names:
                if _banned_attr(node.module, a.name) or _banned_module(f"{node.module}.{a.name}"):
                    report(node, f"from {node.module} import {a.name}")
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    for node in ast.walk(tree):                      # uses: os.system(...), getattr(os, "popen"), import_module("fitz")
        if isinstance(node, ast.Attribute):
            chain = _dotted(node)
            if chain and chain[0] in aliases:
                full = aliases[chain[0]].split(".") + chain[1:]
                if _banned_attr(".".join(full[:-1]), full[-1]):
                    report(node, ".".join(chain))
        elif isinstance(node, ast.Call):
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            args = node.args
            if (fname in ("__import__", "import_module") and args and isinstance(args[0], ast.Constant)
                    and isinstance(args[0].value, str) and _banned_module(args[0].value)):
                report(node, f"{fname}({args[0].value!r})")
            if fname == "getattr" and len(args) >= 2 and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str):
                chain = _dotted(args[0])
                if chain and chain[0] in aliases:
                    module = ".".join(aliases[chain[0]].split(".") + chain[1:])
                    if _banned_attr(module, args[1].value):
                        report(node, f"getattr({'.'.join(chain)}, {args[1].value!r})")
    return found


def test_source_tree_found():
    assert (PKG / "__init__.py").is_file(), f"no package at {PKG}"


def test_src_starts_no_process_and_never_imports_pymupdf():
    found = [v for path in _source_files() for v in violations(path)]
    assert not found, "src/ must not start processes or import PyMuPDF:\n" + "\n".join(found)


PLANTED_BAD = {
    "import subprocess": "import subprocess",
    "import multiprocessing\nmultiprocessing.Process(target=print).start()": "import multiprocessing",
    "from multiprocessing import Pool": "from multiprocessing import",
    "import _multiprocessing": "import _multiprocessing",
    "from concurrent.futures import ProcessPoolExecutor": "import ProcessPoolExecutor",
    "import concurrent.futures\nconcurrent.futures.ProcessPoolExecutor()": "concurrent.futures.ProcessPoolExecutor",
    "import concurrent.futures as cf\ncf.ProcessPoolExecutor()": "cf.ProcessPoolExecutor",
    "from concurrent import futures\nfutures.ProcessPoolExecutor()": "futures.ProcessPoolExecutor",
    "from concurrent.futures.process import ProcessPoolExecutor": "concurrent.futures.process",
    "import platform\nplatform.platform()": "platform.platform",
    "import platform as pf\npf.architecture()": "pf.architecture",
    "from platform import processor": "from platform import processor",
    "import platform\ngetattr(platform, 'uname')()": "getattr(platform, 'uname')",
    "import os\nos.system('true')": "os.system",
    "from os import posix_spawn": "from os import posix_spawn",
    "import os as o\no.execv('x', [])": "o.execv",
    "import importlib\nimportlib.import_module('fitz')": "import_module('fitz')",
    "import _posixsubprocess": "import _posixsubprocess",
}
PLANTED_OK = [
    "import platform\nplatform.mac_ver()\nplatform.python_version()",
    "from concurrent.futures import ThreadPoolExecutor",
    "import concurrent.futures\nconcurrent.futures.ThreadPoolExecutor()",
    "import threading\nthreading.Thread(target=print)",
]


@pytest.mark.parametrize("code", list(PLANTED_BAD), ids=range(len(PLANTED_BAD)))
def test_the_static_ban_catches_planted_process_starts(tmp_path: Path, code: str):
    path = tmp_path / "planted.py"
    path.write_text(code + "\n", encoding="utf-8")
    found = violations(path)
    assert found and PLANTED_BAD[code] in "\n".join(found), found


@pytest.mark.parametrize("code", PLANTED_OK, ids=range(len(PLANTED_OK)))
def test_the_static_ban_allows_threads_and_safe_platform_helpers(tmp_path: Path, code: str):
    path = tmp_path / "planted.py"
    path.write_text(code + "\n", encoding="utf-8")
    assert violations(path) == []


# A literal ``import _posixsubprocess`` passes in the package's integrity.py and nowhere else (decided 2026-11-05).
ALLOWED_ONLY_THERE = "try:\n    import _posixsubprocess\nexcept ImportError:\n    _posixsubprocess = None\n"
STILL_BAD_IN_INTEGRITY = {
    "from _posixsubprocess import fork_exec": "from _posixsubprocess import ...",
    "import importlib\nimportlib.import_module('_posixsubprocess')": "import_module('_posixsubprocess')",
    "__import__('_posixsubprocess')": "__import__('_posixsubprocess')",
    "import _posixsubprocess, subprocess": "import subprocess",
    "import _posixsubprocess\nimport pty": "import pty",
}


def _planted_package(tmp_path: Path) -> Path:
    pkg = tmp_path / "src" / "pdfintake"
    (pkg / "pdf").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return pkg


def test_the_real_integrity_module_is_the_one_allowed():
    assert (PKG / "integrity.py").is_file()
    assert set(IMPORT_ALLOWED) == {"integrity.py"} and IMPORT_ALLOWED["integrity.py"] == {"_posixsubprocess"}


def test_integrity_may_import_posixsubprocess_by_name(tmp_path: Path):
    pkg = _planted_package(tmp_path)
    path = pkg / "integrity.py"
    for code in (ALLOWED_ONLY_THERE, "import _posixsubprocess as _psp\n"):
        path.write_text(code, encoding="utf-8")
        assert violations(path, pkg) == [], code


@pytest.mark.parametrize("where", ["flow.py", "pdf/integrity.py", "integrity/__init__.py", "integrity_extra.py"])
def test_a_planted_posixsubprocess_import_elsewhere_still_fails(tmp_path: Path, where: str):
    pkg = _planted_package(tmp_path)
    path = pkg / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ALLOWED_ONLY_THERE, encoding="utf-8")
    found = violations(path, pkg)
    assert found and "import _posixsubprocess" in "\n".join(found), (where, found)


def test_an_integrity_py_outside_the_package_still_fails(tmp_path: Path):
    path = tmp_path / "integrity.py"                     # same name, but not the package's file
    path.write_text(ALLOWED_ONLY_THERE, encoding="utf-8")
    assert violations(path, _planted_package(tmp_path)) != []
    assert violations(path) != []                       # nor against the real package root


@pytest.mark.parametrize("code", list(STILL_BAD_IN_INTEGRITY), ids=range(len(STILL_BAD_IN_INTEGRITY)))
def test_integrity_may_not_import_it_any_other_way(tmp_path: Path, code: str):
    pkg = _planted_package(tmp_path)
    path = pkg / "integrity.py"
    path.write_text(code + "\n", encoding="utf-8")
    found = violations(path, pkg)
    assert found and STILL_BAD_IN_INTEGRITY[code] in "\n".join(found), found


def test_the_allowance_names_only_that_module(tmp_path: Path):
    pkg = _planted_package(tmp_path)
    path = pkg / "integrity.py"
    path.write_text("import _posixsubprocess, subprocess\n", encoding="utf-8")
    assert [v.split(": ", 1)[1] for v in violations(path, pkg)] == ["import subprocess"]


_SWEEP = r"""
import importlib, importlib.abc, json, re, sys
src, names, blocked, kit_ok = sys.argv[1], json.loads(sys.argv[2]), set(json.loads(sys.argv[3])), set(json.loads(sys.argv[4]))
sys.path.insert(0, src)
attempts, current = [], [None]
NET = {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
       "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.getnameinfo",
       "urllib.Request", "http.client.connect"}
def hook(event, args):
    if event in NET:
        attempts.append({"module": current[0], "event": event, "args": repr(args)[:160]})
        raise OSError("network blocked while importing")
sys.addaudithook(hook)
def importer():
    f = sys._getframe(2)
    while f is not None:
        name = f.f_globals.get("__name__") or ""
        if not (name.startswith("importlib") or name.startswith("_frozen_importlib")):
            return name
        f = f.f_back
    return ""
class Block(importlib.abc.MetaPathFinder):
    # pdfintake code may not import a macOS-only module or an extra at import time (kit: its own extras).
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in blocked:
            return None
        who = importer()
        if not (who == "pdfintake" or who.startswith("pdfintake.")):
            return None
        if top in kit_ok and (who == "pdfintake.kit" or who.startswith("pdfintake.kit.")):
            return None
        raise ModuleNotFoundError("No module named " + repr(fullname) + " (blocked: imported at import time by " + who + ")", name=fullname)
sys.meta_path.insert(0, Block())
results = {}
for name in names:
    current[0] = name
    try:
        importlib.import_module(name)
        results[name] = "ok"
    except ModuleNotFoundError as exc:
        results[name] = "missing module: " + str(exc.name)
    except ImportError as exc:                   # "from . import sibling" when the sibling is absent
        m = re.match(r"cannot import name '([A-Za-z_][A-Za-z0-9_]*)'", str(exc))
        if m and exc.name:
            results[name] = "missing module: " + exc.name + "." + m.group(1)
        else:
            results[name] = "error: ImportError: " + str(exc)[:300]
    except BaseException as exc:
        results[name] = "error: " + type(exc).__name__ + ": " + str(exc)[:300]
print("@@SWEEP@@" + json.dumps({"results": results, "attempts": attempts}))
"""


def _sweep(names: list[str], src: Path = SRC) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _SWEEP, str(src), json.dumps(names),
         json.dumps(sorted(DARWIN_ONLY | OPTIONAL)), json.dumps(sorted(KIT_EXTRAS))],
        capture_output=True, text=True, encoding="utf-8", timeout=600, env=env,
    )
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@SWEEP@@")]
    assert marker, f"the import sweep didn't finish (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
    return json.loads(marker[-1][len("@@SWEEP@@"):])


def _import_problems(results: dict[str, str], src: Path = SRC) -> dict[str, str]:
    """The modules whose import failed for a reason this test doesn't excuse."""
    bad: dict[str, str] = {}
    for name, result in results.items():
        if result == "ok":
            continue
        if result.startswith("missing module: "):
            missing = result[len("missing module: "):]
            if missing in PLANNED and _module_name_exists(missing, src) is False:
                continue                                  # a sibling another unit hasn't built yet
        bad[name] = result
    return bad


def _module_name_exists(name: str, src: Path) -> bool:
    base = src.joinpath(*name.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def test_importing_src_works_everywhere_and_touches_no_network():
    names = [n for n in (_module_name(p) for p in _source_files()) if n not in IMPORT_SKIP]
    out = _sweep(names)
    attempts = out["attempts"]
    assert not attempts, "network at import time:\n" + "\n".join(
        f"{a['module']}: {a['event']} {a['args']}" for a in attempts)
    bad = _import_problems(out["results"])
    assert not bad, ("modules that fail to import (a macOS-only module or an extra imported at module level, "
                     "a missing dependency, or a sibling that isn't planned):\n"
                     + "\n".join(f"{n}: {r}" for n, r in bad.items()))


# onnxruntime's official builds send telemetry to Microsoft by default and keep a device id and
# an event queue under the user's home from the moment onnxruntime is imported (1.31.1 measured on
# macOS, 2026-11-07). Native code sends it, so the socket block above can't see it; what it leaves
# on disk can be seen. Both children get a throwaway home and ORT_DISABLE_TELEMETRY=0.
_ORT_HOMES = ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME", "XDG_CACHE_HOME",
              "XDG_CONFIG_HOME", "XDG_STATE_HOME")


def _ort_child(tmp_path: Path, code: str) -> tuple[str, list[str]]:
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, ORT_DISABLE_TELEMETRY="0", PYTHONDONTWRITEBYTECODE="1")
    for i, name in enumerate(_ORT_HOMES):
        env[name] = str(home if name in ("HOME", "USERPROFILE") else home / f"dir{i}")
    run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    found = sorted(str(f.relative_to(home)) for f in home.rglob("*")
                   if f.is_file() and "pycache" not in str(f) and not f.name.endswith(".pyc"))
    return run.stdout.strip(), found


def test_onnxruntime_telemetry_is_off_whenever_pdfintake_loads_it(tmp_path: Path):
    pytest.importorskip("onnxruntime")
    out, found = _ort_child(tmp_path, "import os\n"
                                      "import pdfintake.ocr.engines.rapid\n"
                                      "import onnxruntime\n"
                                      "onnxruntime.get_device()\n"
                                      "print(os.environ['ORT_DISABLE_TELEMETRY'])\n")
    assert out == "1"
    assert found == [], f"onnxruntime kept telemetry files under the home: {found}"


def test_the_telemetry_check_sees_a_store_when_onnxruntime_keeps_one(tmp_path: Path):
    pytest.importorskip("onnxruntime")
    _out, found = _ort_child(tmp_path, "import onnxruntime\nonnxruntime.get_device()\n")
    if not found:
        pytest.skip("this onnxruntime build keeps no telemetry files, so there is nothing to see")
    assert any("onnxruntime" in f for f in found), found


PLANTED_IMPORTS = {
    "pdfintake.pdf.planted_vision": ("import Vision\n", False),
    "pdfintake.pdf.planted_objc": ("def f():\n    pass\nfrom objc import super\n", False),
    "pdfintake.pdf.planted_thai": ("import pythainlp\n", False),
    "pdfintake.pdf.planted_harfbuzz": ("import uharfbuzz\n", False),
    "pdfintake.pdf.planted_sibling": ("from . import no_such_sibling\n", False),
    "pdfintake.pdf.planted_lazy": ("def f():\n    import Vision\n    import pythainlp\n", True),
    "pdfintake.pdf.planted_planned": ("from ..platforms import current\n", True),   # U1 not on disk in this copy
    "pdfintake.pdf.planted_planned_from": ("from .. import platforms\n", True),
    "pdfintake.pdf.planted_no_name": ("from ..types import NoSuchThing\n", False),
    "pdfintake.kit.planted_kit": ("import uharfbuzz\n", True),
    "pdfintake.kit.planted_kit_test_only": ("import jsonschema\n", False),
}


def test_the_import_sweep_catches_planted_imports(tmp_path: Path):
    src = tmp_path / "src"
    shutil.copytree(PKG, src / "pdfintake", ignore=shutil.ignore_patterns("__pycache__"))
    (src / "pdfintake" / "platforms.py").unlink(missing_ok=True)
    for name, (code, _) in PLANTED_IMPORTS.items():
        src.joinpath(*name.split(".")).with_suffix(".py").write_text(code, encoding="utf-8")
    out = _sweep(list(PLANTED_IMPORTS), src)
    bad = _import_problems(out["results"], src)
    for name, (_, ok) in PLANTED_IMPORTS.items():
        if ok:
            assert name not in bad, f"{name} should import cleanly: {out['results'][name]}"
        else:
            assert name in bad, f"{name} should fail the sweep: {out['results'][name]}"


def test_datawheel_imports_only_the_standard_library():
    path = PKG / "datawheel.py"
    if not path.exists():
        pytest.skip("datawheel.py is not written yet")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"line {node.lineno}: import {a.name}" for a in node.names if a.name.split(".")[0] not in stdlib]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad.append(f"line {node.lineno}: relative import from {'.' * node.level}{node.module or ''}")
            elif (node.module or "").split(".")[0] not in stdlib:
                bad.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call):
            fname = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            first = node.args[0] if node.args else None
            if (fname in ("__import__", "import_module") and isinstance(first, ast.Constant)
                    and isinstance(first.value, str) and first.value.split(".")[0] not in stdlib):
                bad.append(f"line {node.lineno}: {fname}({first.value!r})")
    assert not bad, "datawheel.py must import only the standard library:\n" + "\n".join(bad)


# --------------------------------------------------------------------------- the test-suite guards (tests/conftest.py)


def test_network_is_blocked_by_default(network_attempts):
    import socket

    with socket.socket() as s, pytest.raises(OSError, match="network blocked"):
        s.connect(("192.0.2.10", 80))
    with pytest.raises(OSError, match="network blocked"):
        socket.getaddrinfo("example.com", 443)
    with socket.socket() as s, pytest.raises(OSError, match="network blocked"):
        s.bind(("0.0.0.0", 0))
    with pytest.raises(OSError, match="network blocked"):
        socket.create_connection(("203.0.113.42", 80), timeout=1)
    assert [event for event, _ in network_attempts] == ["socket.connect", "socket.getaddrinfo", "socket.bind",
                                                        "socket.getaddrinfo"]


def test_a_swallowed_attempt_is_still_recorded(network_attempts):
    import urllib.request

    try:
        urllib.request.urlopen("https://example.com/", timeout=1)
    except Exception:
        pass
    assert network_attempts, "the attempt was swallowed but must still be on record"


def test_loopback_stays_open():
    import socket

    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"ping")
        assert b.recv(4) == b"ping"
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5):
            conn, _ = server.accept()
            conn.close()
    assert socket.getaddrinfo("localhost", 80)


def test_proxies_are_cleared():
    import urllib.request

    proxy_vars = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    assert not [v for v in proxy_vars if os.environ.get(v)], "a proxy would carry a download past the socket guard"
    assert os.environ.get("NO_PROXY") == "*"
    assert urllib.request.getproxies() in ({}, {"no": "*"})


@pytest.mark.network
def test_the_network_mark_lifts_the_block(network_attempts):
    import socket

    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))                       # refused without the mark; sends nothing
    assert network_attempts == []


def _default_cache_roots(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The cache roots pdfintake would use with PDFINTAKE_CACHE unset (DESIGN §2.2 platforms.cache_root)."""
    home = Path.home()
    roots = [home / "Library" / "Caches" / "pdfintake", home / ".cache" / "pdfintake"]
    if os.environ.get("XDG_CACHE_HOME"):
        roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "pdfintake")
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "pdfintake" / "Cache")
    try:
        from pdfintake import platforms
    except ImportError:
        return roots                                    # platforms.py (U1) is not built yet
    with monkeypatch.context() as m:
        m.delenv("PDFINTAKE_CACHE", raising=False)
        roots.append(Path(platforms.cache_root()))
    return roots


def test_tests_use_a_private_cache(monkeypatch: pytest.MonkeyPatch):
    cache = os.environ.get("PDFINTAKE_CACHE")
    assert cache, "conftest sets PDFINTAKE_CACHE"
    mine = Path(cache).resolve()
    for root in _default_cache_roots(monkeypatch):
        assert mine != root.resolve() and root.resolve() not in mine.parents, f"tests would use the real cache {root}"
    assert SRC.parent not in mine.parents


# --------------------------------------------------------------------------- conftest without pdfintake

CONFTEST_DIR = Path(__file__).resolve().parent

# Loads this folder's conftest.py as a plugin, in an interpreter where importing pdfintake fails the way
# ``fails_as`` says, and runs one empty test with it (argv: conftest folder, test file, fails_as).
_CONFTEST_PROBE = r"""
import importlib.abc, sys
folder, test_file, fails_as = sys.argv[1:4]

class _Gone(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pdfintake" or name.startswith("pdfintake."):
            raise ModuleNotFoundError(f"No module named {fails_as!r}", name=fails_as)
        return None

for name in [m for m in sys.modules if m == "pdfintake" or m.startswith("pdfintake.")]:
    del sys.modules[name]
sys.meta_path.insert(0, _Gone())
sys.path.insert(0, folder)
import pytest
sys.exit(pytest.main(["-p", "conftest", "-p", "no:cacheprovider", "--noconftest", "-q", test_file]))
"""


def _run_conftest_without_pdfintake(tmp_path: Path, fails_as: str) -> subprocess.CompletedProcess:
    test_file = tmp_path / "test_probe.py"
    test_file.write_text("def test_probe():\n    pass\n", encoding="utf-8")
    return subprocess.run([sys.executable, "-c", _CONFTEST_PROBE, str(CONFTEST_DIR), str(test_file), fails_as],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120)


def test_conftest_works_where_pdfintake_cant_be_imported(tmp_path: Path):
    """The CI sandbox job runs tests/e2e/test_sandbox_smoke.py from an environment that holds only pytest
    (it drives the installed command), and loads this conftest there: no fixture in it may need pdfintake
    importable. 2026-11-10: the thorough-mode fixture imported it, and every smoke test failed at setup."""
    proc = _run_conftest_without_pdfintake(tmp_path, "pdfintake")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout


def test_conftest_still_fails_when_pdfintake_is_there_but_breaks(tmp_path: Path):
    """Only pdfintake's absence is passed over: a pdfintake whose own import fails on a missing dependency
    still fails the test."""
    proc = _run_conftest_without_pdfintake(tmp_path, "some_dependency")
    assert proc.returncode != 0
    assert "some_dependency" in proc.stdout + proc.stderr


# --------------------------------------------------------------------------- the manifest schema's file and canary guards

SCHEMA_PATH = PKG / "langpacks" / "manifest.schema.json"
CODE_NAMES = ["eng.py", "ENG.PY", "hook.pth", "x.Pth", "lib.so", "lib.dylib", "lib.DyLib", "ext.dll", "ext.pyd",
              "a.b.py"]
DATA_NAMES = ["eng.traineddata", "th_PP-OCRv5_rec_mobile_infer.onnx", "keys.txt", "lib.so.1", "pyfile", "a.pyc.txt",
              "x.py.onnx", "a" * 76 + ".onnx"]                 # the longest name the base pattern allows (81)
# Names noCodeSuffix allows but the file node's own pattern refuses: they fail only while a validator applies the
# keywords beside a $ref too (2020-12), not just the $ref's target (draft-07 behaviour) (decided 2026-11-05).
BASE_PATTERN_ONLY = ["x/y.onnx", ".hidden", "a b.onnx", "a" * 77 + ".onnx", "", "-x.onnx"]
BLANK_FINDS = [" ", "\t", " ", "　 "]
GOOD_FINDS = ["4521", " x ", "ตัวอย่าง"]


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validators():
    """The loader's own walker, and jsonschema's where it's installed: both must agree."""
    from pdfintake.langpacks import manifest

    walker = manifest._manifest_schema()
    out = [("loader", lambda value, node: walker.problems(value, node) == [])]
    try:
        import jsonschema
    except ImportError:
        return out
    root = _schema()
    jsonschema.Draft202012Validator.check_schema(root)

    def js(value, node):
        return jsonschema.Draft202012Validator({**node, "$defs": root["$defs"]}).is_valid(value)

    return out + [("jsonschema", js)]


def test_model_file_names_never_end_in_a_code_suffix():
    root = _schema()
    node = root["$defs"]["model"]["properties"]["file"]
    assert {"$ref", "pattern", "type"} <= set(node)                 # the $ref has siblings: that is what's pinned
    suffix_only = root["$defs"]["noCodeSuffix"]
    for who, valid in _validators():
        for name in CODE_NAMES:
            assert not valid(name, node), (who, name)
        for name in DATA_NAMES:
            assert valid(name, node), (who, name)
        for name in BASE_PATTERN_ONLY:
            assert valid(name, suffix_only), (who, "noCodeSuffix alone allows", name)
            assert not valid(name, node), (who, "the file pattern beside the $ref must still apply", name)


# ``\S`` is evaluated with Python ``re`` semantics by both validators, and its whitespace is exactly
# ``str.isspace()``, the set ``canary._squash`` strips with ``str.split()``; an ECMA-262 validator would differ
# (it counts U+FEFF as whitespace, and not U+001C-U+001F or U+0085), and none is used (decided 2026-11-05).
def test_canary_must_find_needs_a_non_whitespace_character():
    node = _schema()["$defs"]["canary"]["properties"]["must_find"]["items"]
    for who, valid in _validators():
        for s in BLANK_FINDS:
            assert not valid(s, node), (who, repr(s))
        for s in GOOD_FINDS:
            assert valid(s, node), (who, s)


def test_canary_blank_means_what_the_canary_normaliser_strips():
    import sys

    from pdfintake.langpacks import canary

    node = _schema()["$defs"]["canary"]["properties"]["must_find"]["items"]
    spaces = [chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace()]
    assert len(spaces) >= 25 and "\u3000" in spaces and "\x1c" in spaces
    for who, valid in _validators():
        for ch in spaces:
            assert canary._squash(ch) == "" and not valid(ch, node), (who, hex(ord(ch)))
        for ch in ("\ufeff", "\u200b"):                  # invisible, but not whitespace to either side
            assert canary._squash(ch) != "" and valid(ch, node), (who, hex(ord(ch)))
