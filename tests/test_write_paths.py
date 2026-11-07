"""Every write into a user's folder converges on one function (DESIGN §0.7, §2.2 U1/U20, §7.1).

A static scan of ``src/pdfintake`` with ``ast``:

* **``fsutil.atomic_write``** is referred to only inside ``flow.commit`` and ``flow.apply`` (and
  defined in ``fsutil.py``). Any other reference fails: a call, a bare reference passed around, an
  import of the name (aliased or not), the string ``"atomic_write"`` for a ``getattr``, and any
  ``getattr`` on ``fsutil`` (or an alias of it) or its ``__dict__`` whose name isn't a constant.
* **Write and delete primitives** (``open`` for writing, ``.save(``, ``.write_bytes``/``.write_text``,
  ``os.replace``/``rename``/``remove``/``unlink``/``link``, ``os.write`` to a descriptor, ``shutil``
  copies, moves and ``rmtree``, ``tempfile.mkstemp`` and the like) appear only in the modules listed
  in ``ALLOWED_MODULES``,
  each of which writes only into pdfintake's cache (records, the pack store, work and staging
  files), into memory, into pdfintake's own config file, or, for the kit and the wheel builder,
  into the contributor's repository. In ``flow.py`` they may appear only inside the helpers in
  ``FLOW_HELPERS``, which refuse any path outside the cache at run time. A module imported under
  another name (``import os as _os``, ``import shutil as sh``) is followed to the module it is.
* ``os.open`` counts as a write unless its flags are spelled out in the call from read-only names
  only (``os.O_RDONLY``, ``os.O_NONBLOCK`` … joined with ``|``, or ``getattr(os, "O_NAME", 0)`` for a
  flag only some platforms have), so a read-only open can be told apart by the scan. Flags in a
  variable, or any other flag, count as a write.
* ``record.save(...)`` is allowed anywhere: it writes only ``<cache>/records/<id>.json``.

So a new module that writes anything has to be added here on purpose, with the reason. The
planted-code tests at the end prove the scan catches each form.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "pdfintake"

ATOMIC = "atomic_write"
ATOMIC_OWNERS = {"flow.py": frozenset({"commit", "apply"})}       # the only callers
ATOMIC_HOME = "fsutil.py"                                        # where it is defined

# Modules that write, and where to. Paths are relative to src/pdfintake; a trailing "/" names a package.
ALLOWED_MODULES: dict[str, str] = {
    "fsutil.py": "atomic_write itself, the partial file beside its dest (sweep_partial removes a killed "
                 "writer's), the aside name a compare-and-swap moves the replaced output to (and back), "
                 "cache folders, pruning <cache>/work and <cache>/staging",
    "record.py": "<cache>/records/<id>.json, atomically",
    "passwords.py": "the user's own saved passwords file (passwords add), created private",
    "datawheel.py": "the data wheel builder: wheels into the --out folder of a fresh clone, downloads into a temp folder",
    "langpacks/store.py": "the pack store under <cache>/packs and <cache>/tessdata, the canary cache",
    "intake/assemble.py": "the assembled work PDF (a dest in the run's work folder)",
    "intake/images.py": "Pillow encodes into in-memory buffers",
    "pdf/render.py": "page renders (PNGs) in the run's work folder",
    "pdf/orient.py": "the uprighted work PDF and the vote's PNGs in the run's work folder",
    "pdf/shrink.py": "the shrunk work PDF in the run's work folder",
    "pdf/smaller.py": "the candidate in <cache>/staging, JPEG encodes into memory",
    "pdf/export.py": "the export, built in staging; flow.commit writes it to the user's path",
    "ocr/engines/tesseract.py": "the contract suite's synthetic page, in a folder the suite hands it",
    "ocr/engines/vision.py": "the contract suite's synthetic page, in a folder the suite hands it",
    "ocr/engines/rapid.py": "the contract suite's synthetic page, in a folder the suite hands it",
    "kit/": "the add-a-language kit: drafts, benches and wheels in the contributor's repository; never part of a run",
}
FLOW = "flow.py"
# _journal_dest writes only to the descriptor _lock_work opened (a work folder's lock file).
FLOW_HELPERS = frozenset({"_save_work", "_copy_into_work", "_remove_work", "_lock_work", "_journal_dest"})

WRITE_ATTRS = frozenset({"save", "write_bytes", "write_text", "touch", "unlink", "rmdir", "rename",
                         "symlink_to", "hardlink_to"})
MODULE_CALLS = {
    "os": frozenset({"replace", "rename", "renames", "remove", "unlink", "rmdir", "removedirs", "link",
                     "symlink", "open", "truncate", "ftruncate", "mkfifo", "write", "pwrite", "writev"}),
    "shutil": frozenset({"move", "copy", "copy2", "copyfile", "copytree", "rmtree", "copyfileobj"}),
    "tempfile": frozenset({"mkstemp", "NamedTemporaryFile", "SpooledTemporaryFile"}),
}
OPEN_FUNCS = {("", "open"), ("io", "open"), ("os", "fdopen"), ("codecs", "open")}
# os.open flags that never let the descriptor write, create, truncate or append.
READ_ONLY_FLAGS = frozenset({"O_RDONLY", "O_NONBLOCK", "O_NOCTTY", "O_BINARY", "O_CLOEXEC", "O_NOFOLLOW",
                             "O_DIRECTORY", "O_NOINHERIT"})
FSUTIL = "fsutil"


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    owner: str
    what: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} in {self.owner or '<module>'}: {self.what}"


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _mode_writes(call: ast.Call, pos: int) -> bool:
    """True when an open()-style call's mode may write (unknown counts as writing)."""
    mode: ast.AST | None = call.args[pos] if len(call.args) > pos else None
    for kw in call.keywords:
        if kw.arg == "mode":
            mode = kw.value
    if mode is None:
        return False                                   # the default mode reads
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(c in mode.value for c in "wax+")
    return True


class _Scan(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.owner: list[str] = []
        self.atomic: list[Finding] = []
        self.writes: list[Finding] = []
        self.aliases: dict[str, tuple[str, str]] = {}     # local name -> (module, function)
        self.modules: dict[str, str] = {}                  # local name -> the module it is (import X as Y)
        self.fsutil_names: set[str] = {FSUTIL}             # local names for fsutil

    # -- where we are
    def _here(self) -> str:
        return self.owner[0] if self.owner else ""

    def _visit_def(self, node: ast.AST, name: str) -> None:
        self.owner.append(name)
        self.generic_visit(node)
        self.owner.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_def(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_def(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.owner:
            self.generic_visit(node)
        else:
            self._visit_def(node, node.name)

    # -- atomic_write, in any form
    def _atomic(self, node: ast.AST, what: str) -> None:
        self.atomic.append(Finding(self.rel, getattr(node, "lineno", 0), self._here(), what))

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == ATOMIC:
            self._atomic(node, "refers to atomic_write")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == ATOMIC:
            self._atomic(node, "refers to .atomic_write")
        if node.attr == "__dict__" and isinstance(node.value, ast.Name) and node.value.id in self.fsutil_names:
            self._atomic(node, "reads fsutil.__dict__")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == ATOMIC:
            self._atomic(node, "names 'atomic_write' in a string")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            target = alias.name if alias.asname else alias.name.split(".")[0]
            if target.split(".")[-1] == FSUTIL:
                self.fsutil_names.add(local)
            self.modules[local] = target
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == ATOMIC:
                self._atomic(node, "imports atomic_write by name")
            if alias.name == FSUTIL:
                self.fsutil_names.add(alias.asname or alias.name)
            mod = (node.module or "").split(".")[-1]
            if mod in MODULE_CALLS and alias.name in MODULE_CALLS[mod]:
                self.aliases[alias.asname or alias.name] = (mod, alias.name)
            if mod in ("io", "codecs", "os") and alias.name in ("open", "fdopen"):
                self.aliases[alias.asname or alias.name] = (mod, alias.name)
        self.generic_visit(node)

    # -- writes
    def _write(self, node: ast.AST, what: str) -> None:
        self.writes.append(Finding(self.rel, getattr(node, "lineno", 0), self._here(), what))

    def _is_os(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and self.modules.get(node.id) == "os"

    def _read_only_flags(self, node: ast.AST | None) -> bool:
        """Whether an ``os.open`` flags expression is built only from read-only ``os`` flags."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._read_only_flags(node.left) and self._read_only_flags(node.right)
        if isinstance(node, ast.Attribute):
            return self._is_os(node.value) and node.attr in READ_ONLY_FLAGS
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            args = node.args
            return (len(args) == 3 and not node.keywords and self._is_os(args[0])
                    and isinstance(args[1], ast.Constant) and args[1].value in READ_ONLY_FLAGS
                    and isinstance(args[2], ast.Constant) and args[2].value == 0)
        return False

    def _os_open_writes(self, call: ast.Call) -> bool:
        flags = call.args[1] if len(call.args) > 1 else None
        for kw in call.keywords:
            if kw.arg == "flags":
                flags = kw.value
        return not self._read_only_flags(flags)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        parts = _dotted(func)
        if parts is not None and len(parts) == 1 and parts[0] in self.aliases:
            parts = list(self.aliases[parts[0]])
        elif parts is not None and len(parts) >= 2 and parts[0] in self.modules:
            parts = [*self.modules[parts[0]].split("."), *parts[1:]]
        if isinstance(func, ast.Name) and func.id == "getattr" and node.args:
            target = _dotted(node.args[0])
            name = node.args[1] if len(node.args) > 1 else None
            if target is not None and (target[0] in self.fsutil_names or target[-1] == FSUTIL) and not (
                    isinstance(name, ast.Constant) and isinstance(name.value, str)):
                self._atomic(node, "getattr on fsutil with a computed name")
        if parts is not None:
            head, name = (parts[0], parts[-1]) if len(parts) >= 2 else ("", parts[0])
            if (head, name) in OPEN_FUNCS or (len(parts) == 1 and name == "open"):
                pos = 1
                if _mode_writes(node, pos):
                    self._write(node, f"{'.'.join(parts)}(..., write mode)")
            elif len(parts) == 2 and head in MODULE_CALLS and name in MODULE_CALLS[head]:
                if head == "os" and name == "open":
                    if self._os_open_writes(node):
                        self._write(node, "os.open(...)")
                else:
                    self._write(node, f"{head}.{name}(...)")
            elif len(parts) >= 2 and name in WRITE_ATTRS:
                if not (name == "save" and parts[-2] == "record"):
                    self._write(node, f".{name}(...)")
        elif isinstance(func, ast.Attribute) and func.attr in WRITE_ATTRS:
            self._write(node, f".{func.attr}(...)")
        if isinstance(func, ast.Attribute):
            if func.attr == "replace" and len(node.args) == 1 and not node.keywords:
                if not (isinstance(func.value, ast.Name) and func.value.id in ("os", "dataclasses")):
                    self._write(node, ".replace(target) (a Path rename)")
            if func.attr == "open" and (parts is None or parts[0] not in ("io", "codecs", "os")):
                # Path.open("w"): a literal mode as the first argument, or mode=; pikepdf.open(path) reads
                first = node.args[0] if node.args else None
                literal = isinstance(first, ast.Constant) and isinstance(first.value, str) \
                    and any(c in first.value for c in "wax+")
                keyword = any(kw.arg == "mode" for kw in node.keywords) and _mode_writes(node, 99)
                if literal or keyword:
                    self._write(node, ".open(write mode)")
        self.generic_visit(node)


def scan(pkg: Path = PKG) -> tuple[list[Finding], list[Finding]]:
    """(atomic_write references, write primitives) over every module of the package."""
    atomic: list[Finding] = []
    writes: list[Finding] = []
    for path in sorted(pkg.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(pkg).as_posix()
        visitor = _Scan(rel)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        atomic += visitor.atomic
        writes += visitor.writes
    return atomic, writes


def _module_allowed(rel: str) -> bool:
    return any(rel == key or (key.endswith("/") and rel.startswith(key)) for key in ALLOWED_MODULES)


def atomic_violations(found: list[Finding]) -> list[Finding]:
    bad = []
    for f in found:
        owners = ATOMIC_OWNERS.get(f.file)
        if owners is not None and f.owner in owners:
            continue
        if f.file == ATOMIC_HOME and f.owner == ATOMIC:
            continue                                   # fsutil.atomic_write naming itself
        bad.append(f)
    return bad


def write_violations(found: list[Finding]) -> list[Finding]:
    bad = []
    for f in found:
        if f.file == FLOW:
            if f.owner not in FLOW_HELPERS:
                bad.append(f)
        elif not _module_allowed(f.file):
            bad.append(f)
    return bad


# --------------------------------------------------------------------------- the source tree

def test_the_package_is_where_the_scan_looks():
    assert (PKG / "flow.py").is_file() and (PKG / "fsutil.py").is_file()


def test_only_commit_and_apply_call_atomic_write():
    atomic, _ = scan()
    assert atomic_violations(atomic) == [], "\n".join(map(str, atomic_violations(atomic)))
    callers = {(f.file, f.owner) for f in atomic if f.file == FLOW}
    assert callers == {(FLOW, "commit"), (FLOW, "apply")}, "commit and apply both write through atomic_write"


def test_nothing_else_writes_a_user_path():
    _, writes = scan()
    bad = write_violations(writes)
    assert bad == [], "write primitives outside the allowlist:\n" + "\n".join(map(str, bad))


def test_flow_writes_only_through_its_cache_helpers():
    _, writes = scan()
    flow_writes = {f.owner for f in writes if f.file == FLOW}
    assert flow_writes and flow_writes <= FLOW_HELPERS


def test_the_allowlist_has_no_stale_entries():
    _, writes = scan()
    writers = {f.file for f in writes}
    for key in ALLOWED_MODULES:
        if key.endswith("/"):
            continue                                   # a package that may not exist yet (the kit)
        assert (PKG / key).is_file(), f"{key} is allowlisted but doesn't exist"
        assert key in writers, f"{key} is allowlisted but writes nothing; drop it from ALLOWED_MODULES"


def test_the_flow_helpers_refuse_paths_outside_the_cache():
    import inspect

    from pdfintake import flow

    for name in ("_save_work", "_copy_into_work", "_lock_work"):
        assert "_work_path(" in inspect.getsource(getattr(flow, name))
    assert "PRUNABLE_KINDS" in inspect.getsource(flow._remove_work)
    journal = inspect.getsource(flow._journal_dest)    # writes only to the lock descriptor it is handed
    assert list(inspect.signature(flow._journal_dest).parameters) == ["lock", "dest"]
    assert "os.write(lock," in journal and "open(" not in journal


# --------------------------------------------------------------------------- planted code

def _plant(tmp_path: Path, rel: str, code: str) -> Path:
    pkg = tmp_path / "pdfintake"
    path = pkg / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return pkg


@pytest.mark.parametrize("rel, code", [
    ("cli.py", "from . import fsutil\ndef main():\n    fsutil.atomic_write(a, b)\n"),
    ("cli.py", "from .fsutil import atomic_write\n"),
    ("cli.py", "from .fsutil import atomic_write as w\ndef main():\n    w(a, b)\n"),
    ("cli.py", "from . import fsutil\nwriter = getattr(fsutil, 'atomic_write')\n"),
    ("cli.py", "from . import fsutil\ndef main():\n    hand = fsutil.atomic_write\n"),
    ("flow.py", "from . import fsutil\ndef run():\n    fsutil.atomic_write(a, b)\n"),
    ("flow.py", "from . import fsutil\ndef _process(d):\n    fsutil.atomic_write(a, b)\n"),
    ("flow.py", "from . import fsutil\ndef apply():\n    pass\ndef later():\n    fsutil.atomic_write(a, b)\n"),
    ("cli.py", "from . import fsutil as F\ndef main():\n    w = getattr(F, 'atomic' + '_write')\n    w(a, b)\n"),
    ("cli.py", "from . import fsutil\ndef main(n):\n    getattr(fsutil, n)(a, b)\n"),
    ("cli.py", "import pdfintake.fsutil as F\ndef main(n):\n    getattr(F, n)(a, b)\n"),
    ("cli.py", "from pdfintake import fsutil as F\ndef main(n):\n    F.__dict__[n](a, b)\n"),
    ("flow.py", "from . import fsutil as F\ndef _process(n):\n    getattr(F, n)(a, b)\n"),
])
def test_the_scan_catches_atomic_write_outside_commit_and_apply(tmp_path: Path, rel: str, code: str):
    atomic, _ = scan(_plant(tmp_path, rel, code))
    assert atomic_violations(atomic)


@pytest.mark.parametrize("code", [
    "from . import fsutil\ndef commit(built, dest):\n    fsutil.atomic_write(built, dest)\n",
    "from . import fsutil\ndef apply(rid):\n    def inner():\n        fsutil.atomic_write(a, b)\n    inner()\n",
])
def test_the_scan_allows_commit_and_apply(tmp_path: Path, code: str):
    atomic, _ = scan(_plant(tmp_path, "flow.py", code))
    assert atomic_violations(atomic) == []


@pytest.mark.parametrize("code", [
    "def f(p):\n    open(p, 'w').write('x')\n",
    "def f(p):\n    open(p, mode='ab')\n",
    "def f(p, m):\n    open(p, m)\n",
    "import io\ndef f(p):\n    io.open(p, 'x')\n",
    "def f(p):\n    p.write_bytes(b'x')\n",
    "def f(p):\n    p.write_text('x')\n",
    "def f(pdf, p):\n    pdf.save(p)\n",
    "import os\ndef f(a, b):\n    os.replace(a, b)\n",
    "from os import replace\ndef f(a, b):\n    replace(a, b)\n",
    "from os import rename as mv\ndef f(a, b):\n    mv(a, b)\n",
    "import shutil\ndef f(a, b):\n    shutil.copyfile(a, b)\n",
    "import shutil\ndef f(a):\n    shutil.rmtree(a)\n",
    "import os\ndef f(a):\n    os.unlink(a)\n",
    "def f(p):\n    p.unlink()\n",
    "def f(p, q):\n    p.replace(q)\n",
    "def f(p):\n    p.open('w')\n",
    "import tempfile\ndef f(d):\n    tempfile.mkstemp(dir=d)\n",
    "import os\ndef f(p):\n    os.open(p, os.O_WRONLY | os.O_CREAT)\n",
    "import os\ndef f(fd):\n    os.fdopen(fd, 'w')\n",
    "import os as _os\ndef f(a, b):\n    _os.replace(a, b)\n",
    "import shutil as sh\ndef f(a, b):\n    sh.copyfile(a, b)\n",
    "import tempfile as tf\ndef f(d):\n    tf.mkstemp(dir=d)\n",
    "import io as _io\ndef f(p):\n    _io.open(p, 'wb')\n",
    "import codecs as cd\ndef f(p):\n    cd.open(p, 'w')\n",
    "import os as _os\ndef f(p):\n    _os.open(p, _os.O_WRONLY)\n",
    "import os\ndef f(fd):\n    os.write(fd, b'x')\n",
    "from os import write\ndef f(fd):\n    write(fd, b'x')\n",
    "import os\ndef f(a, b):\n    os.link(a, b)\n",
    "import os\ndef f(fd):\n    os.ftruncate(fd, 0)\n",
    "import os\ndef f(p, flags):\n    os.open(p, flags)\n",
    "import os\ndef f(p):\n    os.open(p, os.O_RDONLY | os.O_CREAT)\n",
    "import os\ndef f(p):\n    os.open(p, os.O_RDONLY | getattr(os, 'O_TRUNC', 0))\n",
    "import os\ndef f(p):\n    os.open(p, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 1))\n",
    "import os\ndef f(p):\n    os.open(p, flags=os.O_RDWR)\n",
    "import json as os\nimport os as real\ndef f(p):\n    real.open(p, os.O_RDONLY)\n",
    "from os import O_RDONLY\nimport os\ndef f(p):\n    os.open(p, O_RDONLY)\n",
])
def test_the_scan_catches_planted_writes(tmp_path: Path, code: str):
    _, writes = scan(_plant(tmp_path, "cli.py", code))
    assert write_violations(writes), code


@pytest.mark.parametrize("code", [
    "def f(p):\n    return open(p).read()\n",
    "def f(p):\n    return open(p, 'rb').read()\n",
    "def f(s):\n    return s.replace('a', 'b')\n",
    "import dataclasses\ndef f(x):\n    return dataclasses.replace(x, a=1)\n",
    "def f(t):\n    return t.replace(tzinfo=None)\n",
    "from . import record\ndef f(rec):\n    record.save(rec)\n",
    "def f(p):\n    return p.read_bytes()\n",
    "import json as os\ndef f(a, b):\n    return os.replace(a, b)\n",
    "import os.path as osp\ndef f(a):\n    return osp.join(a, 'b')\n",
    "from . import fsutil as F\ndef f():\n    return getattr(F, 'NAME_MAX')\n",
    "import os\ndef f(p):\n    return os.open(p, os.O_RDONLY)\n",
    "import os\ndef f(p):\n    return os.open(p, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0))\n",
    "import os as _os\ndef f(p):\n    return _os.open(p, flags=_os.O_RDONLY | _os.O_CLOEXEC | _os.O_NOFOLLOW)\n",
])
def test_the_scan_allows_reads(tmp_path: Path, code: str):
    _, writes = scan(_plant(tmp_path, "cli.py", code))
    assert write_violations(writes) == []


def test_flow_may_write_only_in_its_helpers(tmp_path: Path):
    ok = "import shutil\ndef _copy_into_work(a, b):\n    shutil.copyfile(a, b)\n"
    _, writes = scan(_plant(tmp_path, "flow.py", ok))
    assert write_violations(writes) == []
    journal = "import os\ndef _journal_dest(lock, dest):\n    os.write(lock, b'x')\n"
    _, writes = scan(_plant(tmp_path / "j", "flow.py", journal))
    assert write_violations(writes) == []
    stray = "import os\ndef commit(lock, dest):\n    os.write(lock, b'x')\n"
    _, writes = scan(_plant(tmp_path / "s", "flow.py", stray))
    assert write_violations(writes)
    bad = "import shutil\ndef _process(a, b):\n    shutil.copyfile(a, b)\n"
    _, writes = scan(_plant(tmp_path / "x", "flow.py", bad))
    assert write_violations(writes)


def test_an_allowlisted_module_name_doesnt_cover_a_lookalike(tmp_path: Path):
    _, writes = scan(_plant(tmp_path, "pdf/shrink_extra.py", "def f(pdf, p):\n    pdf.save(p)\n"))
    assert write_violations(writes)
    _, writes = scan(_plant(tmp_path / "y", "kitchen/x.py", "def f(pdf, p):\n    pdf.save(p)\n"))
    assert write_violations(writes)


def _all_findings() -> Iterator[Finding]:
    atomic, writes = scan()
    yield from atomic
    yield from writes
