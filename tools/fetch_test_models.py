"""Fetch the real models and fonts the tests use, from the upstream sources the packs pin.

Every model and font a ``packs/<id>/manifest.json`` pins is downloaded into::

    <root>/<pack>/<file>

where ``<root>`` is ``--root``, else ``$PDFINTAKE_TEST_MODELS``, else ``~/.pdfintake-dev/models``
(decided 2026-11-05). ``tests/conftest.py``'s ``pinned_asset(pack, key)`` reads the same layout,
so after one run of this tool every real-model test runs instead of skipping.

The download itself is ``src/pdfintake/datawheel.py``'s ``download_sources``, the code that builds the
data wheels: the source must be on the manifest schema's allowlist, redirects stay on https, a file
is streamed to a temporary name, capped at its pinned size, checked against its pinned SHA-256 and
only then given its name, with three attempts. A file already present with the pinned size and
SHA-256 is kept and not fetched again.

Standard library only, so it runs from a fresh clone before pdfintake is installed::

    python3 tools/fetch_test_models.py              # fetch whatever is missing or wrong
    python3 tools/fetch_test_models.py --check      # only check; never touches the network
    python3 tools/fetch_test_models.py --pack th    # one pack (repeatable)

Exit 0 when every requested file is present and correct afterwards, 1 when one is missing or wrong
(``--check``) or couldn't be fetched, 2 for bad arguments or an unreadable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PACKS = REPO / "packs"
DATAWHEEL = REPO / "src" / "pdfintake" / "datawheel.py"
ENV = "PDFINTAKE_TEST_MODELS"
DEFAULT_ROOT = Path.home() / ".pdfintake-dev" / "models"

PACK_ID_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]{2,8})?")
FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,80}")
SHA_RE = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


class ManifestProblem(ValueError):
    """A pack manifest can't be read, or pins a file this tool can't place safely."""


def default_root() -> Path:
    """``$PDFINTAKE_TEST_MODELS`` when set, else ``~/.pdfintake-dev/models``."""
    env = os.environ.get(ENV)
    return Path(env).expanduser() if env else DEFAULT_ROOT


def pack_ids(packs_dir: Path | None = None) -> list[str]:
    """Every pack folder with a manifest, sorted."""
    packs_dir = PACKS if packs_dir is None else packs_dir
    return sorted(p.parent.name for p in packs_dir.glob("*/manifest.json") if PACK_ID_RE.fullmatch(p.parent.name))


def load_manifest(pack: str, packs_dir: Path | None = None) -> dict[str, Any]:
    if not PACK_ID_RE.fullmatch(pack):
        raise ManifestProblem(f"{pack!r} isn't a pack id")
    path = (PACKS if packs_dir is None else packs_dir) / pack / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestProblem(f"there is no pack {pack!r} ({path} is missing)") from None
    except (OSError, ValueError) as exc:
        raise ManifestProblem(f"{path} can't be read: {exc}") from None
    if not isinstance(data, dict):
        raise ManifestProblem(f"{path} isn't a JSON object")
    return data


def pinned_files(manifest: dict[str, Any], pack: str) -> list[dict[str, Any]]:
    """The manifest's model and font entries, models first, each checked to be placeable."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in ("models", "fonts"):
        for item in manifest.get(group) or ():
            if not isinstance(item, dict):
                raise ManifestProblem(f"{pack}: a {group} entry isn't an object")
            key, name, size, sha = item.get("key"), item.get("file"), item.get("bytes"), item.get("sha256")
            if not isinstance(name, str) or not FILE_RE.fullmatch(name) or ".." in name:
                raise ManifestProblem(f"{pack}: {group} entry {key!r} has an unsafe file name {name!r}")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ManifestProblem(f"{pack}: {name} has no positive byte size")
            if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
                raise ManifestProblem(f"{pack}: {name} has no SHA-256 pin")
            if name.casefold() in seen:
                raise ManifestProblem(f"{pack}: {name} is pinned twice")
            seen.add(name.casefold())
            out.append(item)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def problem(path: Path, size: int, sha: str) -> str | None:
    """Why ``path`` isn't the pinned file, or None when it is."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        return f"can't be checked: {exc.strerror or exc}"
    if not stat.S_ISREG(st.st_mode):
        return "isn't a regular file"
    if st.st_size != size:
        return f"is {st.st_size} bytes; the manifest pins {size}"
    try:
        got = sha256_file(path)
    except OSError as exc:
        return f"can't be read: {exc.strerror or exc}"
    if got != sha:
        return f"has SHA-256 {got}; the manifest pins {sha}"
    return None


def _datawheel() -> ModuleType:
    """``src/pdfintake/datawheel.py``, loaded by path (it is standard library only)."""
    name = "_pdfintake_datawheel_for_tests"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DATAWHEEL)
    if spec is None or spec.loader is None:
        raise ManifestProblem(f"{DATAWHEEL} can't be loaded")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def fetch_pack(pack: str, root: Path, *, packs_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Download every file ``pack`` pins that is missing or wrong under ``root/pack``.

    Returns ``(file, state, detail)`` for each pinned file, in manifest order: state ``present`` (it
    was already right), ``fetched`` or ``failed``. Raises ``ManifestProblem`` for an unreadable manifest.
    """
    manifest = load_manifest(pack, packs_dir)
    items = pinned_files(manifest, pack)
    dest = Path(root) / pack
    before = {it["file"]: problem(dest / it["file"], it["bytes"], it["sha256"]) for it in items}
    if all(why is None for why in before.values()):
        return [(it["file"], "present", "") for it in items]
    dw = _datawheel()
    error = ""
    try:
        dw.download_sources(manifest, dest)
    except dw.DataWheelError as exc:
        error = str(exc)
    out: list[tuple[str, str, str]] = []
    for it in items:
        name = it["file"]
        after = problem(dest / name, it["bytes"], it["sha256"])
        if after is None:
            out.append((name, "present" if before[name] is None else "fetched", ""))
        else:
            out.append((name, "failed", error or after))
            error = ""                                   # download_sources stops at its first failure
    return out


def check_pack(pack: str, root: Path, *, packs_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """``(file, state, detail)`` per pinned file, state ``present``, ``missing`` or ``wrong``; no network."""
    manifest = load_manifest(pack, packs_dir)
    out: list[tuple[str, str, str]] = []
    for it in pinned_files(manifest, pack):
        why = problem(Path(root) / pack / it["file"], it["bytes"], it["sha256"])
        state = "present" if why is None else ("missing" if why == "missing" else "wrong")
        out.append((it["file"], state, "" if why in (None, "missing") else why))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_test_models.py",
        description="Download the models and fonts packs/*/manifest.json pin, for the real-model tests.")
    parser.add_argument("--root", type=Path, default=None,
                        help=f"where <pack>/<file> go (default ${ENV}, else {DEFAULT_ROOT})")
    parser.add_argument("--pack", action="append", default=[], metavar="ID",
                        help="only this pack (repeatable; default every pack in packs/)")
    parser.add_argument("--check", action="store_true", help="only check size and SHA-256; download nothing")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of lines")
    args = parser.parse_args(argv)

    root = (args.root.expanduser() if args.root is not None else default_root()).absolute()
    packs = args.pack or pack_ids()
    if not packs:
        print(f"fetch_test_models.py: no pack manifests in {PACKS}", file=sys.stderr)
        return 2
    report: dict[str, list[dict[str, str]]] = {}
    bad = 0
    for pack in packs:
        try:
            rows = check_pack(pack, root) if args.check else fetch_pack(pack, root)
        except ManifestProblem as exc:
            print(f"fetch_test_models.py: {exc}", file=sys.stderr)
            return 2
        report[pack] = [{"file": f, "state": s, "detail": d} for f, s, d in rows]
        bad += sum(1 for _, s, _ in rows if s not in ("present", "fetched"))
    if args.json:
        print(json.dumps({"root": str(root), "ok": bad == 0, "packs": report}, ensure_ascii=False, indent=2))
    else:
        for pack, rows in report.items():
            for row in rows:
                line = f"{row['state']:8} {pack}/{row['file']}"
                print(line + (f"  ({row['detail']})" if row["detail"] else ""))
        total = sum(len(r) for r in report.values())
        verdict = "all present and correct" if bad == 0 else f"{bad} of {total} missing, wrong or not fetched"
        print(f"{root}: {verdict}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
