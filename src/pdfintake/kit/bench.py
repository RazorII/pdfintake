"""The bench: every candidate reader set, run through the code ``pdfintake run`` uses, scored per page.

**The shipped code.** ``bench`` reads the bench pages with ``pipeline.read_pages`` (DESIGN §0.5), the
function ``run`` calls for pass 2, with the draft manifest's reader set swapped for each candidate
(``dataclasses.replace(manifest, readers=...)``). So a number in ``bench.json`` describes what a
user gets: the same normalisers, pick rule, repairs and fills, the same model checks.

**Candidates.** ``candidates`` asks every registered engine that runs on the platform for its
``candidate_params`` for the pack's primary language, then combines them:

* each reader alone (pick ``single``);
* each reader with a checker of another engine (``single``, and ``primary_with_floor`` where the
  rule serves the pack's script);
* each pair of readers (``corroborated_trigrams`` on the pack's script, both tie orders' first),
  with and without the checker as a non-competing corroborator.

An engine that proposes several parameter sets appears once per set, its use id numbered from the
second on (``tesseract``, ``tesseract2``: the same engine with its next parameter set, such as
another model; the set's ``params`` say which). Only a parameter that
names a model asset takes the base pack's ``en:`` prefix; a Tesseract ``lang`` string never does.
The draft's own set (set 0) keeps the draft's ids, so the bare id ``tesseract`` can be the pack's
model in set 0 and the base pack's ``eng`` in the others: wherever the kit shows a use to a person
(progress lines, messages, the report) it shows ``use_label``, the id with the model or setting
that tells it apart (``tesseract[vie]``, ``tesseract[eng]``, ``vision2[vi-VT, no correction]``).

Every set is validated as the loader validates a manifest (``manifest.parse``), so a set the loader
would refuse is never benched; ``candidate_sets`` counts the refused ones (``refused_by_loader``)
before the ceiling, so they take no place under it. The draft's own reader set for the platform
comes first. Two sets that are the same once the loader fills in the defaults, use ids aside, are
benched once, under the lower index (``merged_sets`` names the others). Each pick rule keeps at most
``MAX_SETS_PER_RULE`` sets (the draft's own set always stays), and ``candidate_sets`` reports the
total and what was merged and cut, per rule, so neither is ever silent.

This module also holds the kit's pack helpers: where a pack's files live (``packs/<id>/`` in the
repo; models, fonts and pages in the kit work folder in the cache), and how a draft becomes an
``InstalledPack`` the pipeline can read with.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import EXIT_FAILED, PackError, PdfIntakeError, SetupError, UsageError
from ..types import to_jsonable

if TYPE_CHECKING:
    from ..langpacks.manifest import Manifest, ReaderSet
    from ..langpacks.store import InstalledPack
    from .render import BenchPage

# 1.1: must-survive counts only what an engine read, duplicate candidates are merged, and choose's
# policy is maximin-v2. A bench.json from another kit version is benched again (decided 2026-11-06).
KIT_VERSION = "1.1"
DRAFT_FILE = "manifest.draft.json"
MANIFEST_FILE = "manifest.json"
BENCH_FILE = "bench.json"
PRIVATE_FILE = "bench.private.json"
MAX_SETS_PER_RULE = 40       # per pick rule, so no rule's sets crowd out another's (decided 2026-11-05)
TIE = 0.005
_ID_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]{2,8})?")
BASE_ID = "en"


@dataclass
class BenchResult:
    platform: str
    engine_versions: dict[str, str]
    sets: list[dict[str, Any]]
    # A failed set's message, by set index: for the command's own output only. bench.json is
    # committed to a public PR and a message can name a local path, so only the code goes into
    # the set (decided 2026-11-05).
    messages: dict[int, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- where things live

def pack_dir(packs_root: Path, pack_id: str) -> Path:
    if not isinstance(pack_id, str) or not _ID_RE.fullmatch(pack_id):
        raise UsageError("kit_usage", f"{pack_id!r} isn't a pack id (an ISO 639 code such as es, or sr-latn)")
    return Path(packs_root) / pack_id


def work_root(work: Path | None = None) -> Path:
    """The kit's work folder: ``--work`` when given, else ``<cache>/kit``."""
    if work is not None:
        return Path(work)
    from .. import platforms

    return platforms.cache_root() / "kit"


def files_dir(work: Path, pack_id: str) -> Path:
    return Path(work) / pack_id / "files"


def pages_dir(work: Path, pack_id: str) -> Path:
    return Path(work) / pack_id / "pages"


def manifest_file(pdir: Path, *, draft: bool = True) -> Path:
    """The draft when ``draft`` and it exists, else ``manifest.json``."""
    d = Path(pdir) / DRAFT_FILE
    if draft and d.is_file():
        return d
    return Path(pdir) / MANIFEST_FILE


def read_raw(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        raise PackError("kit_no_manifest", f"{path} is missing; start the pack with `pdfintake kit init`") from None
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackError("invalid_manifest", f"{path} isn't valid JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise PackError("invalid_manifest", f"{path} must hold a JSON object")
    return data, raw


class KitFileError(PdfIntakeError):
    """A kit file (bench.json, a candidates list, corpus.json in a report) can't be read: kit exit 1."""

    exit_code = EXIT_FAILED


def read_json(path: Path, *, code: str, what: str) -> Any:
    """A kit file's JSON, or KitFileError ``code`` naming the file when it can't be read or
    parsed, so a malformed file is a kit error with JSON output (exit 1), never a traceback."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KitFileError(code, f"{path} is missing") from None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise KitFileError(code, f"{what} ({Path(path).name}) isn't readable JSON: {exc}") from None


def write_json(path: Path, data: Any) -> None:
    """Write a committed pack file (manifest.json, bench.json) with LF line ends on every OS, since its
    SHA-256 is pinned and .gitattributes keeps every file byte for byte."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def base_manifest(packs_root: Path, pack_id: str) -> "Manifest | None":
    """The base pack's manifest for checking ``pack_id``: None for the base pack itself; else the
    sibling ``packs/en/manifest.json``, or the installed English data wheel's."""
    from ..langpacks import manifest as M

    if pack_id == BASE_ID:
        return None
    sibling = Path(packs_root) / BASE_ID / MANIFEST_FILE
    if sibling.is_file():
        data, _ = read_raw(sibling)
        return M.parse(data)
    from ..langpacks import store

    return store.base().manifest


def load_manifest(packs_root: Path, pack_id: str, *, draft: bool = True) -> tuple["Manifest", dict[str, Any], Path]:
    """The pack's draft (or ``manifest.json``) parsed and checked, its raw JSON, and its path."""
    from ..langpacks import manifest as M

    path = manifest_file(pack_dir(packs_root, pack_id), draft=draft)
    data, raw = read_raw(path)
    if raw.get("id") != pack_id:
        raise PackError("invalid_manifest", f"{path} is for pack {raw.get('id')!r}, not {pack_id!r}")
    return M.parse(data, base=base_manifest(packs_root, pack_id)), raw, path


def ensure_files(raw: Mapping[str, Any], files: Path, *, download: bool) -> list[str]:
    """Download the manifest's pinned models and fonts into ``files`` (hash-checked); with
    ``download`` False, only report which are missing. Returns the missing file names."""
    from .. import datawheel

    files = Path(files)
    if download:
        try:
            datawheel.download_sources(raw, files)
        except datawheel.DataWheelError as exc:
            raise PackError("kit_download", str(exc)) from None
        return []
    missing = []
    for group in ("models", "fonts"):
        for item in raw.get(group) or ():
            p = files / str(item["file"])
            if not p.is_file() or p.stat().st_size != int(item["bytes"]):
                missing.append(str(item["file"]))
    return missing


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size and dst.stat().st_ino == src.stat().st_ino:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def installed_pack(manifest: "Manifest", files: Path, root: Path) -> "InstalledPack":
    """The draft as an ``InstalledPack``: ``root/models`` and ``root/fonts`` linked from ``files``.
    PackError naming every missing file (``kit`` commands download them unless told not to)."""
    from ..langpacks.store import InstalledPack

    root = Path(root)
    missing = []
    for sub, items in (("models", manifest.models), ("fonts", manifest.fonts)):
        (root / sub).mkdir(parents=True, exist_ok=True)
        for a in items:
            src = Path(files) / a.file
            if not src.is_file():
                missing.append(a.file)
                continue
            _link(src, root / sub / a.file)
    if manifest.canary is not None and (Path(files) / manifest.canary.file).is_file():
        _link(Path(files) / manifest.canary.file, root / manifest.canary.file)
    if missing:
        raise PackError("kit_files_missing", f"the {manifest.id!r} pack's files aren't downloaded yet: "
                                             f"{', '.join(missing)} (run the command without --offline)",
                        missing=missing)
    return InstalledPack(manifest, root)


def base_pack(packs_root: Path, work: Path, draft: "InstalledPack", *, download: bool) -> "InstalledPack":
    """The base pack the draft reads with: the draft itself when it is the base pack; else the
    installed English data wheel; else ``packs/en`` with its files in the work folder."""
    if draft.manifest.base:
        return draft
    from ..langpacks import store

    try:
        return store.base()
    except (SetupError, PackError):
        pass
    sibling = Path(packs_root) / BASE_ID / MANIFEST_FILE
    if not sibling.is_file():
        raise SetupError("kit_no_base", "the English base pack isn't installed and packs/en is missing; "
                                        "run scripts/dev-install.sh")
    m, raw, _ = load_manifest(packs_root, BASE_ID, draft=False)
    ensure_files(raw, files_dir(work, BASE_ID), download=download)
    return installed_pack(m, files_dir(work, BASE_ID), Path(work) / BASE_ID / "pack")


# --------------------------------------------------------------------------- manifests as JSON

def _plain(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (frozenset, set)):
        return sorted(_plain(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def reader_set_json(rs: "ReaderSet") -> dict[str, Any]:
    return {"engines": [{"id": u.id, "engine": u.engine, "role": u.role, "params": _plain(u.params),
                         "models": list(u.models)} for u in rs.engines],
            "pick": {"rule": rs.pick.rule, "version": rs.pick.version, "params": _plain(rs.pick.params)},
            "fallback": rs.fallback}


def manifest_json(m: "Manifest") -> dict[str, Any]:
    """A manifest dict for ``m`` that the loader reads back as the same manifest (parameters with
    their defaults filled in, word lists sorted)."""
    det: dict[str, Any] = {"osd_scripts": list(m.detect.osd_scripts)}
    if m.detect.langid_codes:
        det["langid"] = {"codes": list(m.detect.langid_codes)}
        if m.detect.close_group:
            det["langid"]["close_group"] = list(m.detect.close_group)
    if m.detect.function_words:
        det["function_words"] = {"words": sorted(m.detect.function_words), "rivals": sorted(m.detect.rivals),
                                 "min_hits": m.detect.fw_min_hits, "ratio": m.detect.fw_ratio}
    if m.detect.probe is not None:
        p = m.detect.probe
        det["probe"] = {"use": {"id": p.use.id, "engine": p.use.engine, "role": p.use.role,
                                "params": _plain(p.use.params), "models": list(p.use.models)},
                        "min_chars": p.min_chars, "min_conf": p.min_conf, "min_line_chars": p.min_line_chars}
    out: dict[str, Any] = {
        "schema": m.schema, "id": m.id, "version": m.version, "requires_core": m.requires_core, "title": m.title,
        "languages": [dict(x) for x in m.languages], "scripts": list(m.scripts), "detect": det,
        "readers": {k: reader_set_json(rs) for k, rs in m.readers.items()},
        "normalize": list(m.normalize),
        "repairs": [{"name": s.name, "version": s.version, "params": _plain(s.params)} for s in m.repairs],
        "fills": [{"name": s.name, "version": s.version, "params": _plain(s.params)} for s in m.fills],
        "models": [{k: v for k, v in dataclasses.asdict(a).items() if not (k == "notice" and v is None)}
                   for a in m.models],
        "fonts": [{**dataclasses.asdict(f), "covers": list(f.covers)} for f in m.fonts],
        "must_survive": [dataclasses.asdict(x) for x in m.must_survive],
        "bench": _plain(m.bench),
    }
    if m.base:
        out["base"] = True
    if m.canary is not None:
        out["canary"] = {**dataclasses.asdict(m.canary), "must_find": list(m.canary.must_find)}
    return out


def parse_reader_set(raw: Mapping[str, Any], rs_json: Mapping[str, Any], platform_key: str,
                     base: "Manifest | None") -> "ReaderSet":
    """``rs_json`` validated as the loader validates a reader set of this pack on ``platform_key``."""
    from ..langpacks import manifest as M

    variant = dict(raw)
    readers = dict(raw.get("readers") or {})
    readers[platform_key] = dict(rs_json)
    variant["readers"] = readers
    m = M.parse(json.dumps(variant, ensure_ascii=False).encode("utf-8"), base=base)
    return m.readers[platform_key]


# --------------------------------------------------------------------------- candidates

def _refs(draft: "Manifest", base: "Manifest | None") -> list[tuple[str, Any]]:
    out = [(a.key, a) for a in draft.models]
    if base is not None and not draft.base:
        out += [(f"en:{a.key}", a) for a in base.models]
    return out


def _models_for(engine: Any, params: Mapping[str, Any], refs: Sequence[tuple[str, Any]]) -> list[str]:
    """The model references a candidate needs: the ones its parameters name (an asset key, or a
    Tesseract language part matching a traineddata file), then one of each other kind its engine
    reads (a detector, a classifier), this pack's first, then the base pack's."""
    mine = [(ref, a) for ref, a in refs if a.engine == engine.name]
    named: list[str] = []
    strings = [v for v in params.values() if isinstance(v, str)]
    parts = {p for v in strings for p in v.split("+")}
    for ref, a in mine:
        stem = a.file.rsplit(".", 1)[0]
        if ref in strings or (a.kind == "traineddata" and stem in parts and stem != "osd"):
            named.append(ref)
    kinds = {a.kind for ref, a in mine if ref in named}
    for kind in sorted(getattr(engine, "MODEL_KINDS", ()) or ()):
        if kind in kinds:
            continue
        for ref, a in mine:
            if a.kind == kind and kind != "traineddata":
                named.append(ref)
                break
    return named


def _options(draft: "Manifest", platform_key: str, base: "Manifest | None") -> list[tuple[str, dict[str, Any], list[str]]]:
    from ..ocr import engines

    tag = next((x["tag"] for x in draft.languages if x.get("role") == "primary"), draft.id)
    script = draft.scripts[0]
    refs = _refs(draft, base)
    local = {a.key for a in draft.models}
    out: list[tuple[str, dict[str, Any], list[str]]] = []
    for name in engines.names():
        try:
            eng = engines.get(name)
        except PdfIntakeError:
            continue
        if platform_key not in getattr(eng, "platforms", ()):
            continue
        try:
            plist = eng.candidate_params(tag, script, [a for _, a in refs])
        except Exception:
            continue
        # Only a parameter that names a model asset takes the base pack's "en:" prefix. A Tesseract
        # ``lang`` of "eng" is a language string, not an asset reference: prefixed, the loader
        # refused every set holding it, so Tesseract eng was never benched (kit trial es, item 10).
        asset_params = {p.name for p in (getattr(eng, "PARAMS", None) or ()) if getattr(p, "kind", None) == "asset_ref"}
        for params in plist:
            fixed: dict[str, Any] = {}
            for k, v in params.items():
                if (k in asset_params and isinstance(v, str) and v not in local
                        and any(r == f"en:{v}" for r, _ in refs)):
                    v = f"en:{v}"
                fixed[k] = v
            out.append((name, fixed, _models_for(eng, fixed, refs)))
    return out


def use_label(use: Any) -> str:
    """The use's id with what tells it apart, for people: the Tesseract ``lang``, the RapidOCR
    recogniser (without the base pack's ``en:``), Vision's first language and ``no correction`` when
    its language correction is off; the bare id when the use has none of these. ``use`` is a use's
    manifest JSON or an ``EngineUse``."""
    if isinstance(use, Mapping):
        uid, params = use.get("id"), use.get("params")
    else:
        uid, params = getattr(use, "id", None), getattr(use, "params", None)
    params = params if isinstance(params, Mapping) else {}
    parts: list[str] = []
    if isinstance(params.get("lang"), str):
        parts.append(params["lang"])
    if isinstance(params.get("rec"), str):
        parts.append(params["rec"].split(":", 1)[-1])
    langs = params.get("languages")
    if isinstance(langs, (list, tuple)) and langs:
        parts.append(str(langs[0]))
    if params.get("language_correction") is False:
        parts.append("no correction")
    return f"{uid}[{', '.join(parts)}]" if parts else str(uid)


def set_label(rs: Any) -> str:
    """A reader set's uses, labelled (``use_label``) and joined with `` + ``; JSON or a ``ReaderSet``."""
    uses = rs.get("engines") if isinstance(rs, Mapping) else getattr(rs, "engines", None)
    return " + ".join(use_label(u) for u in uses or ())


def _use(uid: str, engine: str, role: str, params: Mapping[str, Any], models: Sequence[str]) -> dict[str, Any]:
    return {"id": uid, "engine": engine, "role": role, "params": dict(params), "models": list(models)}


def _ids(options: Sequence[tuple[str, dict[str, Any], list[str]]]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name, _p, _m in options:
        n = seen.get(name, 0) + 1
        seen[name] = n
        base = re.sub(r"[^a-z0-9_]", "_", name.lower())[:20] or "engine"
        if not base[0].isalpha():
            base = "e" + base[:19]
        out.append(base if n == 1 else f"{base}{n}")
    return out


def _tie_order_in_full(uses: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> list[str]:
    """``corroborated_trigrams``' tie order as the rule applies it: the competitors ``tie_order``
    names, then the ones it leaves out by use id (then any name it holds that isn't a competitor).
    Written out before the ids are renamed, so two sets whose ids sort differently stay apart."""
    lenders = {str(u) for u in params.get("corroborators") or ()}
    competitors = [str(u.get("id")) for u in uses if u.get("role") == "reader" and str(u.get("id")) not in lenders]
    named = [str(u) for u in params.get("tie_order") or ()]
    ranked = [u for u in named if u in competitors] + sorted(u for u in competitors if u not in named)
    return ranked + [u for u in named if u not in ranked]


def _same_key(rs_json: Mapping[str, Any], raw: Mapping[str, Any], platform_key: str, base: "Manifest | None") -> str:
    """What two candidate sets must share to be the same set: the reader set as the loader reads it
    (every parameter's default filled in), each use's model list sorted (its order isn't read), and
    the use ids set aside: they are only names, so each use is renamed by its place (``u0``, ``u1``)
    wherever the pick rule names it. vi, fr, it, pl, pt and hu: the draft's own Tesseract set
    (``tesseract``) came back as a candidate under ``tesseract2``. ``corroborated_trigrams`` ranks
    the readers its ``tie_order`` leaves out by id, so that order is written out first
    (``_tie_order_in_full``)."""
    try:
        rs = reader_set_json(parse_reader_set(raw, rs_json, platform_key, base))
    except (PackError, TypeError, AttributeError):
        rs = dict(rs_json)
    uses = [u for u in rs.get("engines") or () if isinstance(u, Mapping)]
    names = {str(u.get("id")): f"u{i}" for i, u in enumerate(uses)}

    def renamed(v: Any) -> Any:
        if isinstance(v, str):
            return names.get(v, v)
        if isinstance(v, (list, tuple)):
            return [renamed(x) for x in v]
        return v

    engines = [{**u, "id": names[str(u.get("id"))], "models": sorted(u.get("models") or ())} for u in uses]
    pick = dict(rs.get("pick") or {})
    params = dict(pick.get("params") or {})
    if pick.get("rule") == "corroborated_trigrams":
        params["tie_order"] = _tie_order_in_full(uses, params)
    pick["params"] = {k: renamed(v) for k, v in params.items()}
    return json.dumps({**rs, "engines": engines, "pick": pick}, sort_keys=True, default=str)


def candidate_sets(draft: "Manifest", platform_key: str, base: "Manifest | None" = None, *,
                   raw: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The candidate reader sets as manifest JSON (the draft's own set for the platform first), and
    how the list was made: ``{total, refused_by_loader, merged, merged_sets, cut, cut_by_rule, kept,
    max_per_rule}``.

    ``total`` counts the distinct sets proposed. With ``raw`` (the draft's JSON), each is first
    checked the way the loader checks a manifest, and ``refused_by_loader`` counts the ones it
    refuses; they never take a place under the per-rule ceiling. Then a set that is the same as an
    earlier one once the loader fills in the defaults, use ids aside (``_same_key``; es: the draft's
    own RapidOCR set and the generated one; vi: the draft's ``tesseract`` set and the generated
    ``tesseract2`` one; a set that only leaves out a default ``timeout_s``) is merged into the earlier
    one and benched once: ``merged`` counts them, and ``merged_sets`` maps the kept set's index (as a
    string) to the indices the merged sets had before merging. ``cut`` counts the sets the ceiling
    dropped, and ``kept`` is the length of the list returned, so ``kept = total - refused_by_loader
    - merged - cut``."""
    from ..ocr import rules

    script = draft.scripts[0]
    options = _options(draft, platform_key, base)
    ids = _ids(options)
    rule_ok = {}
    for name, version in rules.names():
        try:
            r = rules.get(name, version)
        except PdfIntakeError:
            continue
        scripts_ok = getattr(r, "scripts", None)
        rule_ok[name] = (version, scripts_ok is None or script in scripts_ok)

    def usable(rule: str) -> int | None:
        v = rule_ok.get(rule)
        return v[0] if v and v[1] else None

    sets: list[dict[str, Any]] = [reader_set_json(draft.readers_for(platform_key))]
    n = len(options)
    for i in range(n):
        ri = _use(ids[i], options[i][0], "reader", options[i][1], options[i][2])
        if (v := usable("single")) is not None:
            sets.append({"engines": [ri], "pick": {"rule": "single", "version": v, "params": {"reader": ids[i]}},
                         "fallback": "none"})
        for j in range(n):
            if j == i or options[j][0] == options[i][0]:
                continue
            cj = _use(ids[j], options[j][0], "checker", options[j][1], options[j][2])
            if (v := usable("single")) is not None:
                sets.append({"engines": [ri, cj], "pick": {"rule": "single", "version": v, "params": {"reader": ids[i]}},
                             "fallback": "none"})
            if (v := usable("primary_with_floor")) is not None:
                sets.append({"engines": [ri, cj], "pick": {"rule": "primary_with_floor", "version": v,
                                                           "params": {"primary": ids[i], "checker": ids[j]}},
                             "fallback": "none"})
    if (v := usable("corroborated_trigrams")) is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if options[j][0] == options[i][0]:
                    continue                                  # two settings of one engine don't corroborate
                ri = _use(ids[i], options[i][0], "reader", options[i][1], options[i][2])
                rj = _use(ids[j], options[j][0], "reader", options[j][1], options[j][2])
                pick = {"rule": "corroborated_trigrams", "version": v,
                        "params": {"script": script, "tie_order": [ids[i], ids[j]], "corroborators": []}}
                sets.append({"engines": [ri, rj], "pick": pick, "fallback": "none"})
                for k in range(n):
                    if k in (i, j) or options[k][0] in (options[i][0], options[j][0]):
                        continue
                    ck = _use(ids[k], options[k][0], "checker", options[k][1], options[k][2])
                    pick2 = {**pick, "params": {**pick["params"], "corroborators": [ids[k]]}}
                    sets.append({"engines": [ri, rj, ck], "pick": pick2, "fallback": "none"})
                    break
    unique: list[dict[str, Any]] = []
    keys: set[str] = set()
    for s in sets:
        key = json.dumps(s, sort_keys=True)
        if key not in keys:
            keys.add(key)
            unique.append(s)
    refused = 0
    if raw is not None:
        loadable = []
        for n_set, s in enumerate(unique):
            try:
                parse_reader_set(raw, s, platform_key, base)
            except (PackError, TypeError, AttributeError):
                if n_set > 0:                                 # the draft's own set loaded with the draft
                    refused += 1
                    continue
            loadable.append(s)
    else:
        loadable = unique
    as_loaded = raw if raw is not None else manifest_json(draft)
    distinct: list[tuple[dict[str, Any], list[int]]] = []      # (set, the indices merged into it)
    first: dict[str, int] = {}
    for n_set, s in enumerate(loadable):
        key = _same_key(s, as_loaded, platform_key, base)
        if key in first:
            distinct[first[key]][1].append(n_set)
            continue
        first[key] = len(distinct)
        distinct.append((s, []))
    out: list[dict[str, Any]] = []
    merged_sets: dict[str, list[int]] = {}
    kept: dict[str, int] = {}
    cut: dict[str, int] = {}
    for n_set, (s, merged) in enumerate(distinct):
        rule = str(s["pick"]["rule"])
        if n_set > 0 and kept.get(rule, 0) >= MAX_SETS_PER_RULE:
            cut[rule] = cut.get(rule, 0) + 1
            continue
        kept[rule] = kept.get(rule, 0) + 1
        if merged:
            merged_sets[str(len(out))] = merged
        out.append(s)
    info = {"total": len(unique), "refused_by_loader": refused, "merged": len(loadable) - len(distinct),
            "merged_sets": merged_sets, "cut": sum(cut.values()), "cut_by_rule": dict(sorted(cut.items())),
            "kept": len(out), "max_per_rule": MAX_SETS_PER_RULE}
    return out, info


def candidate_jsons(draft: "Manifest", platform_key: str, base: "Manifest | None" = None) -> list[dict[str, Any]]:
    """The candidate reader sets as manifest JSON, the draft's own set for the platform first."""
    return candidate_sets(draft, platform_key, base)[0]


def candidates(draft: "Manifest", platform_key: str, *, base: "Manifest | None" = None) -> list["ReaderSet"]:
    """Every candidate reader set for ``platform_key`` that the loader accepts, the draft's own first."""
    raw = manifest_json(draft)
    out = []
    for s in candidate_jsons(draft, platform_key, base):
        try:
            out.append(parse_reader_set(raw, s, platform_key, base))
        except PackError:
            continue
    return out


# --------------------------------------------------------------------------- the bench

def _page_images(pages: Sequence["BenchPage"]) -> list[Any]:
    from PIL import Image

    from ..types import PageImage

    out = []
    for i, p in enumerate(pages):
        with Image.open(p.png) as im:
            w, h = im.size
        out.append(PageImage(index=i, path=Path(p.png), width=w, height=h, dpi=float(p.dpi),
                             cropbox=(0.0, 0.0, w * 72.0 / p.dpi, h * 72.0 / p.dpi)))
    return out


def _plans(n: int, draft: "Manifest") -> list[Any]:
    from ..ocr.langplan import PagePlan
    from ..types import LangCall, ScriptCall

    tag = next((x["tag"] for x in draft.languages if x.get("role") == "primary"), draft.id)
    return [PagePlan(index=i, pack=draft.id, composition="forced", languages=[LangCall(tag, 1.0, "forced", 0)],
                     confident=True, reason="kit bench", script=ScriptCall(draft.scripts[0], 1.0, "hint", None))
            for i in range(n)]


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return round(statistics.fmean(vals), 4) if vals else None


def must_coverage(scores: Sequence[Any], name: str, cond: str) -> dict[str, Any]:
    """One must-survive group in one condition, its pages' counts added up: ``truth`` (the group's
    characters in the truth), ``read`` (those inside words or spans an engine read), ``survived``,
    and ``measurable`` (at least ``score.MIN_MEASURABLE`` read characters)."""
    from .score import MIN_MEASURABLE

    totals = [0, 0, 0]
    for s in scores:
        c = (getattr(s, "must_counts", None) or {}).get(name)
        if s.cond == cond and c:
            totals = [x + int(y) for x, y in zip(totals, c)]
    return {"truth": totals[0], "read": totals[1], "survived": totals[2], "measurable": totals[1] >= MIN_MEASURABLE}


def _aggregate(scores: Sequence[Any], pages: Sequence["BenchPage"], primary: str,
               must_names: Sequence[str]) -> dict[str, Any]:
    """A set's numbers. A metric's condition figure is the mean of its pages; a must-survive group's
    is survived / read over the condition's pages together, None where it isn't measurable, with
    the counts under ``must_coverage`` (score module docstring)."""
    from .render import CONDITIONS

    metrics: dict[str, dict[str, float | None]] = {}
    for metric in ("words", "trigrams", "digits"):
        row = {cond: _mean([getattr(s, metric) for s in scores if s.cond == cond]) for cond in CONDITIONS}
        if any(v is not None for v in row.values()):
            metrics[metric] = row
    coverage = {name: {cond: must_coverage(scores, name, cond) for cond in CONDITIONS} for name in must_names}
    must = {name: {cond: (round(c["survived"] / c["read"], 4) if c["measurable"] else None)
                   for cond, c in row.items()} for name, row in coverage.items()}
    prim = [getattr(s, primary) for s in scores if getattr(s, primary) is not None]
    return {"metrics": metrics, "must": must, "must_coverage": coverage,
            "worst_page": round(min(prim), 4) if prim else None,
            "seconds_per_page": round(statistics.fmean([s.seconds for s in scores]), 4) if scores else None,
            "pages": [dataclasses.asdict(s) for s in scores]}


def _as_pack(draft: "Manifest | InstalledPack") -> "InstalledPack":
    from ..langpacks.store import InstalledPack

    if isinstance(draft, InstalledPack):
        return draft
    return InstalledPack(draft, Path("."))


def bench(pages: Sequence["BenchPage"], draft: "Manifest | InstalledPack", base: "InstalledPack", platform_key: str,
          sets: Sequence["ReaderSet"], *, workers: int | None = None,
          progress: Callable[[int, int, "ReaderSet", dict[str, Any], float], None] | None = None,
          merged: Mapping[int, Sequence[int]] | None = None) -> BenchResult:
    """Read every page with every set through ``pipeline.read_pages`` and score it.

    ``draft`` is the pack being built (an ``InstalledPack`` whose files are present, or a manifest
    that names no models of its own). A set that can't run here is kept with its error, never dropped
    silently. ``progress``, when given, is called after each set with its index, the number of sets,
    the set, its entry and the seconds it took (``kit bench`` prints a line on standard error).
    ``merged`` maps a set's index to the candidates merged into it (``candidate_sets``), which its
    entry keeps under ``merged``.
    """
    import time

    from ..ocr import engines, pipeline
    from ..langpacks.store import InstalledPack
    from .score import primary_metric, score_page

    pack = _as_pack(draft)
    m = pack.manifest
    script = m.scripts[0]
    primary = primary_metric(script)
    images = _page_images(pages)
    plans = _plans(len(images), m)
    must_names = [x.name for x in m.must_survive]
    n_workers = workers or min(4, os.cpu_count() or 1)
    used: set[str] = set()
    out_sets: list[dict[str, Any]] = []
    messages: dict[int, str] = {}
    for k, rs in enumerate(sets):
        started = time.monotonic()
        variant = InstalledPack(dataclasses.replace(m, readers={"default": rs, platform_key: rs}), pack.root)
        base_used = variant if m.base else base
        entry: dict[str, Any] = {"index": k, "reader_set": reader_set_json(rs), "engines": len(rs.engines),
                                 "vision": any(u.engine == "vision" for u in rs.engines), "primary_metric": primary}
        if merged and merged.get(k):
            entry["merged"] = [int(i) for i in merged[k]]
        used.update(u.engine for u in rs.engines)
        try:
            ocr = pipeline.read_pages(images, plans, {m.id: variant}, base_used, platform_key,
                                      signals=None, workers=n_workers)
        except PdfIntakeError as exc:
            entry.update({"ok": False, "error": {"code": exc.code}})
            messages[k] = exc.message
        else:
            scores = [score_page(p, o, True, m.must_survive, script) for p, o in zip(pages, ocr)]
            failed = sum(1 for o in ocr for r in o.reads if r.status in ("failed", "timeout"))
            entry.update({"ok": True, "failed_reads": failed, **_aggregate(scores, pages, primary, must_names)})
        out_sets.append(entry)
        if progress is not None:
            progress(k, len(sets), rs, entry, time.monotonic() - started)
    versions: dict[str, str] = {}
    for name in sorted(used):
        try:
            v = engines.get(name).version()
        except Exception:
            v = None
        versions[name] = str(v)[:60] if v else "unavailable"
    return BenchResult(platform_key, versions, out_sets, messages)


def result_json(r: BenchResult) -> dict[str, Any]:
    return {"platform": r.platform, "engine_versions": dict(r.engine_versions), "sets": to_jsonable(r.sets)}


def result_from_json(d: Mapping[str, Any]) -> BenchResult:
    try:
        return BenchResult(str(d["platform"]), dict(d.get("engine_versions") or {}), list(d["sets"]))
    except (KeyError, TypeError):
        raise PackError("kit_bench_invalid", "bench.json holds a result that isn't a kit bench result") from None


# --------------------------------------------------------------------------- the mixed-script trigger

SWEEP_GRID: Mapping[str, tuple[float, ...]] = {"block_max_conf": (0.5, 0.6, 0.7), "osd_min_conf": (2.0, 5.0),
                                               "min_block_px": (24.0,), "max_blocks": (4.0,)}


def _truth_scripts(text: str) -> set[str]:
    from ..ocr import scripts

    out = set()
    for ch in text:
        s = scripts.script_of(ch)
        if s is not None and s in scripts.SCRIPTS:
            out.add(s)
    return out


def mixed_script_sweep(pages: Sequence["BenchPage"], base: "InstalledPack", *, platform_key: str | None = None,
                       osd: Any = None, eng_reads: Sequence[Any] | None = None,
                       corpus_ref: str | None = None) -> dict[str, Any]:
    """Measure the mixed-script trigger (``detect.mixed_script_check``) over a grid of calibrations.

    The trigger exists for non-Latin lines on a Latin page, so a ``mixed_script`` page whose truth
    holds a non-Latin script is a positive (the trigger should name that script), and a page whose
    truth is Latin only is a negative. Any other page (a non-Latin pack's prose, form or microprint)
    is neither and isn't counted (decided 2026-11-05). The first reader of the set pass 1 reads with
    (``pipeline.detect_readers``, the base pack's ``default`` set on every platform) gives the ``eng``
    read and its OSD model the second opinion (``osd`` and ``eng_reads`` replace them in tests).

    The result is numbers only, apart from ``measured`` (``kit <version> <corpus ref>``, the string
    ``ocr/calibration.json`` carries) when ``corpus_ref`` is given: per grid point the recall, the
    false-positive rate and the OSD calls, and the best point (highest recall with no false
    positive, then fewest calls), or None.
    """
    from .. import platforms
    from ..ocr import detect, pipeline
    from ..ocr.langplan import Calibration

    key = platform_key or platforms.current().key
    images = _page_images(pages)
    if eng_reads is None:
        rs = pipeline.detect_readers(base)                # pass 1's eng read, on every platform
        first = next(u for u in rs.engines if u.role == "reader")
        from ..langpacks.manifest import ReaderSet, RuleUse
        from ..langpacks.store import InstalledPack

        single = ReaderSet((first,), RuleUse("single", 1, {"reader": first.id}), "none")
        variant = InstalledPack(dataclasses.replace(base.manifest, readers={"default": single, key: single}), base.root)
        ocr = pipeline.read_pages(images, _plans(len(images), base.manifest), {base.manifest.id: variant}, variant, key,
                                  signals=None, workers=min(4, os.cpu_count() or 1))
        eng_reads = [o.reads[0] for o in ocr]
    if osd is None:
        osd = pipeline._open_osd(base)
    targets: list[str | None] = []          # a script: a positive; "": a negative; None: not counted
    for p in pages:
        other = sorted(_truth_scripts(p.truth) - {"Latin"})
        if not other:
            targets.append("")
        elif p.kind == "mixed_script":
            targets.append(other[0])
        else:
            targets.append(None)
    grid = []
    keys = list(SWEEP_GRID)

    def points(i: int, cur: dict[str, float]) -> list[dict[str, float]]:
        if i == len(keys):
            return [dict(cur)]
        return [pt for v in SWEEP_GRID[keys[i]] for pt in points(i + 1, {**cur, keys[i]: v})]

    for pt in points(0, {}):
        calib = Calibration(pt["block_max_conf"], pt["osd_min_conf"], int(pt["min_block_px"]), int(pt["max_blocks"]),
                            f"kit {KIT_VERSION} sweep")
        hits = fp = pos = neg = calls = 0
        for img, read, target in zip(images, eng_reads, targets):
            if target is None:
                continue
            try:
                res = detect.mixed_script_check(img, read, osd, calib)
            except PdfIntakeError:
                res = None
            calls += res.blocks if res is not None else 0
            found = res.script if res is not None else None
            if target == "":
                neg += 1
                fp += int(found is not None)
            else:
                pos += 1
                hits += int(found == target)
        grid.append({**pt, "recall": round(hits / pos, 4) if pos else None,
                     "false_positive_rate": round(fp / neg, 4) if neg else None, "osd_calls": calls})
    clean = [g for g in grid if g["recall"] is not None and (g["false_positive_rate"] or 0.0) == 0.0]
    best = max(clean, key=lambda g: (g["recall"], -g["osd_calls"])) if clean else None
    out: dict[str, Any] = {"pages": {"mixed": sum(1 for t in targets if t), "latin": sum(1 for t in targets if t == ""),
                                     "not_counted": sum(1 for t in targets if t is None)},
                           "grid": grid, "best": best}
    if corpus_ref:
        out["measured"] = f"kit {KIT_VERSION} {corpus_ref}"
    return out
