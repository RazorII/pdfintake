"""Generate ``src/pdfintake/langpacks/index.json`` from ``packs/*/manifest.json`` and the built wheels.

    python3 tools/build_index.py                         # write the index
    python3 tools/build_index.py --check                 # CI: the committed index equals a regeneration
    python3 tools/build_index.py --check --strict        # the release: and nothing in packs/ is ahead of it
    python3 tools/build_index.py --record-release X.Y.Z  # scripts/release.sh: record what vX.Y.Z publishes

Options: ``--packs DIR`` (default ``packs``), ``--index FILE`` (default the packaged index),
``--wheels DIR`` (default ``dist``), ``--published FILE`` (the published record; default
``tools/published-packs.json`` for the packaged index, none for any other index). Every path is
relative to the repo root unless absolute. A ``--published`` file that isn't there is an error,
except for ``--record-release``, which creates it; the default record may be missing (no release
yet), and ``--check`` says which record it read.

**Published or not.** A pack version counts as published only once a release has shipped it, and
``tools/published-packs.json`` is the record of that: ``scripts/release.sh`` writes it with
``--record-release`` and commits it in the commit it tags, so a checkout of any commit (a shallow
CI checkout included, with no tags fetched) knows offline which pack versions are out. The index
never says what's published: an edit to ``index.json`` alone can't mark a version published or
take that back. Before the first release there is no record, and the index is simply regenerated
from the manifests and wheels.

* **A published version** (the record lists its id and version) is fixed: its manifest must hash to
  the recorded ``manifest_sha256`` (the published wheel holds that manifest, and the store checks
  one against the other), a wheel of it in ``--wheels`` must be the recorded one byte for byte, and
  its ``dist`` and ``installed_bytes`` come from the record. To change the pack, give it a new
  version, higher than every published one, and build its wheel.
* **An unpublished version** takes ``dist.sha256``, ``dist.bytes`` and ``installed_bytes`` (the sum of
  the data members) from its wheel in ``--wheels`` (``<distribution>-<version>-py3-none-any.whl``,
  checked with ``datawheel.wheel_problems``). A fetched pack's URL is the release asset of this
  core's tag (``.../releases/download/v<core>/<wheel>``) unless the index already holds that
  version, whose URL it keeps. With no wheel, an entry the index holds at the same version and
  manifest keeps its ``dist`` and ``installed_bytes``; anything else needs its wheel.

**--check** compares the committed index with a regeneration. A pack that ``packs/`` holds ahead of
the index (one the index doesn't have yet, a version it doesn't have yet, or an unpublished
manifest that changed since the index was generated) is a note, not a failure: a pack PR can't
regenerate the index, the maintainer does it with the wheels. The index's entry for that pack is
kept as it is for the comparison, and an entry kept for a published version must be, field for
field, the entry the last release shipped (the record holds its hash). ``--strict`` (the release)
makes those notes failures.

**From each manifest** (checked by ``manifest.parse``, the base pack first): ``id``, ``version``,
``base``, ``title``, ``scripts``, ``langid_codes``, the ``detect`` copy (``index.detect_copy``) and
``manifest_sha256`` (of the file's bytes, which go into the wheel unchanged). Every pack needs a
canary.

**Order.** The base pack first, then the packs the current index already lists, in its order, then
new packs by id. ``core_min`` is the highest ``requires_core`` among the packs (and, while --check
keeps an entry that packs/ is ahead of, no lower than the committed index's). The file is written as
UTF-8 JSON with two-space indent, ``\\n`` line ends on every OS and a final newline; ``--check``
compares the parsed data.

**The record** (``--record-release``, which needs nothing but the committed index and manifests)::

    {"record_version": 1,
     "releases": ["v0.1.0"],
     "packs": {"<id>": {"<version>": {"release": "v0.1.0", "manifest_sha256": "…",
                                      "dist": {…the index entry's dist…}, "installed_bytes": 123,
                                      "index_entry_sha256": "…"}}}}

Every index entry is added under the release's tag, unless the record already lists that id and
version, which must then carry the same pins (``manifest_sha256``, ``dist``, ``installed_bytes``);
``release`` stays the first release that shipped it. ``index_entry_sha256`` is the sha256 of the
whole index entry as the latest recording release shipped it (``entry_sha256``: canonical JSON,
sorted keys, no spaces, UTF-8), so a later core may re-render an entry (a new ``detect`` copy) and
the next release records that. A version the record doesn't list yet must be higher than every
recorded version of its pack. A release tag is recorded once, and never below a recorded one; only
the latest recorded release may be recorded again (a rerun). The
file is UTF-8 JSON with sorted keys, two-space indent, ``\\n`` line ends on every OS and a final
newline, so the same record is the same bytes wherever it's written.

Exit codes: 0 done (or the check passed), 1 a problem (each printed on its own line), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = Path("src/pdfintake/langpacks/index.json")
DEFAULT_PUBLISHED = Path("tools/published-packs.json")
RELEASE_URL = "https://github.com/RazorII/pdfintake/releases/download/v{core}/{wheel}"
RECORD_VERSION = 1
_SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_SHA = re.compile(r"[0-9a-f]{64}")


def _pdfintake() -> None:
    try:
        import pdfintake  # noqa: F401
    except ImportError:                                  # a fresh clone before dev-install
        sys.path.insert(0, str(ROOT / "src"))


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def _write_generated(path: Path, text: str) -> None:
    """Write a generated, committed file as UTF-8 with ``\\n`` line ends. A text-mode write left at
    its default newline turns each ``\\n`` into ``\\r\\n`` on Windows, and these files are compared
    byte for byte and by SHA-256."""
    path.write_text(text, encoding="utf-8", newline="\n")


def _semver(text: str) -> tuple[int, int, int]:
    a, b, c = text.lstrip(">=").split(".")
    return int(a), int(b), int(c)


def entry_sha256(entry: dict[str, Any]) -> str:
    """The sha256 of an index entry's canonical JSON (sorted keys, no spaces, UTF-8)."""
    text = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tag_version(tag: Any) -> tuple[int, int, int] | None:
    m = _SEMVER.fullmatch(tag[1:]) if isinstance(tag, str) and tag.startswith("v") else None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


class Problem(Exception):
    pass


# --------------------------------------------------------------------------- the published record

def load_record(path: Path | None) -> dict[str, Any]:
    """The published record at ``path`` (an empty one when ``path`` is None or missing); Problem if it's malformed."""
    if path is None or not path.exists():
        return {"record_version": RECORD_VERSION, "releases": [], "packs": {}}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Problem(f"{path.name} isn't valid JSON ({exc})") from None
    _validate_record(record, path.name)
    return record


def _validate_record(record: Any, name: str) -> None:
    problems: list[str] = []
    if not isinstance(record, dict) or record.get("record_version") != RECORD_VERSION:
        raise Problem(f"{name}: record_version must be {RECORD_VERSION}")
    releases = record.get("releases")
    if not isinstance(releases, list) or any(_tag_version(t) is None for t in releases):
        raise Problem(f"{name}: releases must be a list of vX.Y.Z tags")
    if [_tag_version(t) for t in releases] != sorted({_tag_version(t) for t in releases}):
        problems.append(f"{name}: releases must be listed once each, oldest first")
    packs = record.get("packs")
    if not isinstance(packs, dict):
        raise Problem(f"{name}: packs must be an object of id -> version -> pins")
    for pid, versions in packs.items():
        if not isinstance(versions, dict) or not versions:
            problems.append(f"{name}: {pid} must map each published version to its pins")
            continue
        for version, rec in versions.items():
            where = f"{name}: {pid} {version}"
            if not _SEMVER.fullmatch(str(version)):
                problems.append(f"{where}: not a version")
                continue
            if not isinstance(rec, dict):
                problems.append(f"{where}: the pins must be an object")
                continue
            dist = rec.get("dist")
            if rec.get("release") not in releases:
                problems.append(f"{where}: its release {rec.get('release')!r} isn't in releases")
            if not _SHA.fullmatch(str(rec.get("manifest_sha256"))):
                problems.append(f"{where}: no manifest_sha256")
            if not (isinstance(dist, dict) and _SHA.fullmatch(str(dist.get("sha256")))
                    and type(dist.get("bytes")) is int and isinstance(dist.get("distribution"), str)):
                problems.append(f"{where}: dist needs distribution, sha256 and bytes")
            if type(rec.get("installed_bytes")) is not int:   # not a bool either
                problems.append(f"{where}: no installed_bytes")
            if not _SHA.fullmatch(str(rec.get("index_entry_sha256"))):
                problems.append(f"{where}: no index_entry_sha256")
    if problems:
        raise Problem("; ".join(problems))


def render_record(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def record_release(record: dict[str, Any], index: dict[str, Any], packs: Path, version: str) -> dict[str, Any]:
    """``record`` with every entry of ``index`` recorded as published by ``v<version>`` (Problem if it can't be)."""
    tag = f"v{version}"
    want = _tag_version(tag)
    if want is None:
        raise Problem(f"not a version: {version}")
    out = copy.deepcopy(record)
    problems: list[str] = []
    later = [t for t in out["releases"] if _tag_version(t) > want]
    if later:
        raise Problem(f"{tag} isn't above the recorded releases (latest {out['releases'][-1]})")
    if tag not in out["releases"]:                       # a rerun of the latest release records it again
        out["releases"].append(tag)
    entries = index.get("packs") if isinstance(index, dict) else None
    if not isinstance(entries, list) or not entries:
        raise Problem("the index lists no packs")
    for entry in entries:
        pid, ver = entry.get("id"), entry.get("version")
        manifest = packs / str(pid) / "manifest.json"
        if not manifest.is_file():
            problems.append(f"{pid}: packs/{pid}/manifest.json is missing")
            continue
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if entry.get("manifest_sha256") != sha:
            problems.append(f"{pid}: the index's manifest_sha256 isn't packs/{pid}/manifest.json's; "
                            "regenerate the index (tools/build_index.py) before releasing")
            continue
        pins = {"manifest_sha256": sha, "dist": copy.deepcopy(entry.get("dist")),
                "installed_bytes": entry.get("installed_bytes")}
        recorded = out["packs"].get(pid, {})
        old = recorded.get(ver)
        if old is not None:
            if {k: old.get(k) for k in pins} != pins:
                problems.append(f"{pid}: {ver} is recorded as published by {old.get('release')} with other pins; "
                                "a published version never changes, so give the pack a new version")
            else:
                old["index_entry_sha256"] = entry_sha256(entry)  # what this release's index ships for it
            continue
        if not _SEMVER.fullmatch(str(ver)):
            problems.append(f"{pid}: the index's version {ver!r} isn't a version")
            continue
        if recorded:
            top = max(recorded, key=_semver)
            if _semver(ver) <= _semver(top):
                problems.append(f"{pid}: version {ver} isn't above {top}, released in {recorded[top]['release']}; "
                                "a new version must be higher than every published one")
                continue
        out["packs"].setdefault(pid, {})[ver] = {"release": tag, **pins, "index_entry_sha256": entry_sha256(entry)}
    if problems:
        raise Problem("; ".join(problems))
    try:                                                 # the result must read back as a valid record
        _validate_record(json.loads(render_record(out)), DEFAULT_PUBLISHED.name)
    except Problem as exc:
        raise Problem(f"the record wouldn't be valid: {exc}") from None
    return out


# --------------------------------------------------------------------------- the index

def _wheel_facts(path: Path, raw: dict[str, Any]) -> tuple[str, int, int]:
    from pdfintake import datawheel

    problems = datawheel.wheel_problems(path, raw)
    if problems:
        raise Problem(f"{path.name}: " + "; ".join(problems))
    data = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        installed = sum(zi.file_size for zi in zf.infolist() if ".dist-info/" not in zi.filename)
    return hashlib.sha256(data).hexdigest(), len(data), installed


def build(packs: Path, current: dict[str, Any] | None, wheels: Path, *, published: dict[str, Any] | None = None,
          check: bool = False, strict: bool = False, notes: list[str] | None = None) -> dict[str, Any]:
    """The index for ``packs`` (Problem listing what's wrong).

    ``published`` is the record (``load_record``); None means nothing is published yet. With
    ``check``, a pack that packs/ holds ahead of ``current`` keeps its ``current`` entry and gets a
    line in ``notes``, or, with ``strict``, is a problem.
    """
    _pdfintake()
    from pdfintake import __version__
    from pdfintake.langpacks import index as I
    from pdfintake.langpacks import manifest as M

    record = (published or {}).get("packs") or {}
    problems: list[str] = []
    notes = notes if notes is not None else []
    dirs = sorted(p for p in packs.iterdir() if (p / "manifest.json").is_file()) if packs.is_dir() else []
    if not dirs:
        raise Problem(f"no pack manifests under {packs}")
    raws: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for d in dirs:
        data = (d / "manifest.json").read_bytes()
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            problems.append(f"{d.name}: manifest.json isn't valid JSON ({exc})")
            continue
        if raw.get("id") != d.name:
            problems.append(f"{d.name}: the manifest's id is {raw.get('id')!r}, not the folder's name")
            continue
        raws[d.name] = (data, raw)
    bases = [pid for pid, (_d, r) in raws.items() if r.get("base") is True]
    if len(bases) != 1:
        raise Problem(f"exactly one pack must be the base pack (found {len(bases)})" + "".join(f"; {p}" for p in problems))
    base_id = bases[0]
    try:
        base_m = M.parse(raws[base_id][0])
    except Exception as exc:
        raise Problem(f"{base_id}: {getattr(exc, 'message', exc)}") from None
    cur_entries = {e["id"]: e for e in (current or {}).get("packs") or [] if isinstance(e, dict) and "id" in e}
    order = [base_id] + [pid for pid in cur_entries if pid in raws and pid != base_id]
    order += sorted(pid for pid in raws if pid not in order)
    entries = []
    core_min = (0, 0, 0)
    kept_behind = False
    for pid in order:
        data, raw = raws[pid]
        try:
            M.parse(data, base=None if pid == base_id else base_m)
        except Exception as exc:
            problems.append(f"{pid}: {getattr(exc, 'message', exc)}")
            continue
        if raw.get("canary") is None:
            problems.append(f"{pid}: a published pack needs a canary (pdfintake kit wheel)")
            continue
        is_base = pid == base_id
        version = raw["version"]
        dist_name = f"pdfintake-{'data' if is_base else 'lang'}-{pid}"
        wheel = wheels / I.wheel_name(dist_name, version)
        old = cur_entries.get(pid)
        manifest_sha = hashlib.sha256(data).hexdigest()
        released = record.get(pid) or {}
        rec = released.get(version)
        if rec is not None:
            # Published: the released wheel carries this manifest, and the store checks the wheel's
            # manifest against manifest_sha256, so neither can change under this version.
            if rec["manifest_sha256"] != manifest_sha:
                problems.append(f"{pid}: the manifest changed under published version {version} "
                                f"(released in {rec['release']}); give the pack a new version and build its wheel")
                continue
            if wheel.is_file():
                try:
                    sha, _size, _installed = _wheel_facts(wheel, raw)
                except Problem as exc:
                    problems.append(f"{pid}: {exc}")
                    continue
                if sha != rec["dist"]["sha256"]:
                    problems.append(f"{pid}: {wheel.name} differs from the {version} wheel released in "
                                    f"{rec['release']}; a published wheel never changes, so give the pack a new version")
                    continue
            dist = copy.deepcopy(rec["dist"])
            installed = int(rec["installed_bytes"])
        else:
            if released:
                top = max(released, key=_semver)
                if _semver(version) <= _semver(top):
                    problems.append(f"{pid}: version {version} isn't above {top}, released in "
                                    f"{released[top]['release']}; a new version must be higher than every published one")
                    continue
            in_index = (old is not None and old.get("version") == version
                        and old.get("manifest_sha256") == manifest_sha)
            if check and not in_index:
                if old is None:
                    why = f"{pid} {version} isn't in the index yet"
                elif old.get("version") != version:
                    why = f"{pid} {version} isn't in the index yet (it has {old.get('version')})"
                else:
                    why = f"{pid}: the {version} manifest changed since the index was generated"
                if strict:
                    problems.append(f"{why}; regenerate the index with its wheel (tools/build_index.py) before releasing")
                    continue
                notes.append(f"note: {why}; the maintainer regenerates the index with its wheel "
                             "when the pack is released")
                if old is not None:
                    # A kept entry for a published version is the one the last release shipped, every
                    # field of it (title, detect copy, installed_bytes, ...), not only its pins.
                    held = released.get(str(old.get("version")))
                    if held is not None and entry_sha256(old) != held["index_entry_sha256"]:
                        problems.append(f"{pid}: the index's entry for {old.get('version')} isn't the one the "
                                        f"last release shipped (first released in {held['release']}); "
                                        "restore it from that release's index.json")
                        continue
                    entries.append(copy.deepcopy(old))
                    kept_behind = True
                continue
            if wheel.is_file():
                try:
                    sha, size, installed = _wheel_facts(wheel, raw)
                except Problem as exc:
                    problems.append(f"{pid}: {exc}")
                    continue
                same_version = old is not None and old.get("version") == version
                url = None if is_base else (old["dist"]["url"] if same_version
                                            else RELEASE_URL.format(core=__version__, wheel=wheel.name))
                dist = {"kind": "installed" if is_base else "fetch", "distribution": dist_name, "url": url,
                        "sha256": sha, "bytes": size}
            elif in_index:
                dist, installed = dict(old["dist"]), int(old["installed_bytes"])
            else:
                problems.append(f"{pid}: version {version} has no wheel in {wheels} (rebuild it at its pinned "
                                f"canary: PDFINTAKE_DIST={wheels} scripts/dev-install.sh --wheels-only --packs {pid}; "
                                f"a new or changed canary: pdfintake kit wheel {pid} --out {wheels}, which draws and "
                                f"pins it in the manifest, then builds)")
                continue
        core_min = max(core_min, _semver(raw["requires_core"]))
        detect = raw["detect"]
        langid = detect.get("langid") or {}
        entries.append({"id": pid, "version": version, "base": is_base, "title": raw["title"],
                        "scripts": list(raw["scripts"]), "langid_codes": list(langid.get("codes") or []),
                        "detect": I.detect_copy(detect), "manifest_sha256": manifest_sha,
                        "dist": dist, "installed_bytes": installed})
    if problems:
        raise Problem("; ".join(problems))
    if kept_behind and isinstance((current or {}).get("core_min"), str):
        core_min = max(core_min, _semver(current["core_min"]))
    index = {"index_version": I.INDEX_VERSION, "core_min": ".".join(map(str, core_min)), "packs": entries}
    try:
        I._parse(json.dumps(index, ensure_ascii=False).encode("utf-8"))
    except Exception as exc:
        raise Problem(f"the generated index doesn't load: {getattr(exc, 'message', exc)}") from None
    return index


def render(index: dict[str, Any]) -> str:
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


def _differs(index: dict[str, Any], current: Any) -> list[str]:
    """Which parts of ``current`` a regeneration changes, for the --check message."""
    if not isinstance(current, dict):
        return ["the whole file"]
    out = [k for k in ("index_version", "core_min") if index.get(k) != current.get(k)]
    new = {e["id"]: e for e in index["packs"]}
    old = {e.get("id"): e for e in current.get("packs") or [] if isinstance(e, dict)}
    out += [pid for pid in new if new[pid] != old.get(pid)]
    out += [f"{pid} (not in a regeneration)" for pid in old if pid not in new]
    if not out and [e["id"] for e in index["packs"]] != [e.get("id") for e in current.get("packs") or []
                                                        if isinstance(e, dict)]:
        out.append("the order of the packs")
    return out or ["its layout"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_index.py", description="Generate the pack index.")
    parser.add_argument("--packs", type=Path, default=Path("packs"))
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--wheels", type=Path, default=Path("dist"))
    parser.add_argument("--published", type=Path, default=None,
                        help="the published record (default tools/published-packs.json for the packaged index)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 unless the index equals a regeneration")
    mode.add_argument("--record-release", metavar="X.Y.Z",
                      help="record every index entry as published by vX.Y.Z (scripts/release.sh)")
    parser.add_argument("--strict", action="store_true", help="with --check: a pack ahead of the index fails")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    if args.strict and not args.check:
        print("--strict goes with --check")
        return 2
    packs, index_path, wheels = _resolve(args.packs), _resolve(args.index), _resolve(args.wheels)
    if args.published is not None:
        record_path: Path | None = _resolve(args.published)
        if not args.record_release and not record_path.is_file():
            # A named record that isn't there would silently mean "nothing is published".
            print(f"--published {args.published}: no such file")
            return 2
    elif index_path.resolve() == (ROOT / DEFAULT_INDEX).resolve():
        record_path = ROOT / DEFAULT_PUBLISHED
    else:
        record_path = None                              # an index elsewhere (tests, scratch) has no record
    try:
        record = load_record(record_path)
    except Problem as exc:
        for line in str(exc).split("; "):
            print(line)
        return 1
    current = None
    if index_path.is_file():
        try:
            current = json.loads(index_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            if args.check or args.record_release:
                print(f"{index_path} isn't valid JSON: {exc}")
                return 1
    elif args.check or args.record_release:
        print(f"{index_path} is missing")
        return 1
    if args.record_release:
        if record_path is None:
            print("--record-release needs --published for an index other than the packaged one")
            return 2
        try:
            new = record_release(record, current, packs, args.record_release)
        except Problem as exc:
            for line in str(exc).split("; "):
                print(line)
            return 1
        if record_path.is_file() and new == record:
            print(f"{record_path.name} already records v{args.record_release}")
            return 0
        record_path.parent.mkdir(parents=True, exist_ok=True)
        _write_generated(record_path, render_record(new))
        print(f"recorded v{args.record_release} in {record_path.name} ({len(current['packs'])} packs)")
        return 0
    if args.check:                                     # which record governs this check, in CI's log
        if record["releases"]:
            print(f"published: {record_path.name} records {', '.join(record['releases'])}")
        elif record_path is not None and record_path.is_file():
            print(f"published: {record_path.name} records no release yet, so no pack version is fixed")
        else:
            where = f"{record_path.name} isn't there" if record_path is not None else "this index has no record"
            print(f"published: nothing yet ({where}), so no pack version is fixed")
    notes: list[str] = []
    try:
        index = build(packs, current, wheels, published=record, check=args.check, strict=args.strict, notes=notes)
    except Problem as exc:
        for line in notes:
            print(line)
        for line in str(exc).split("; "):
            print(line)
        return 1
    for line in notes:
        print(line)
    if args.check:
        if index != current:
            print(f"{index_path} differs from a regeneration ({', '.join(_differs(index, current))}); "
                  "run tools/build_index.py and commit the result")
            return 1
        print(f"{index_path} is up to date ({len(index['packs'])} packs)")
        return 0
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_generated(index_path, render(index))
    print(f"wrote {index_path} ({len(index['packs'])} packs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
