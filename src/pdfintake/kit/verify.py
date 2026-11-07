"""``kit verify``: the checks a pack must pass before its PR, and the ones CI re-runs (pack-bench.yml).

Four checks (DESIGN §4.5 step 8), each reported as ``ok``, ``failed`` or ``skipped`` with why:

1. **manifest**: the schema, the core's registries and the source allowlist (``manifest.check``
   against the base pack), every platform's selection made under this kit's policy (``maximin-v2``;
   another fails with a message to run ``kit choose`` again), plus the kit files beside it:
   ``corpus.json`` (allowed licences, pinned revisions), ``bench.json`` and ``bench.private.json``
   (numbers only), a pinned canary, and each
   ``NOTICE-*`` the manifest's models name: it must name exactly those model files, each beside its
   pinned source, because the wheel carries it as it is (kit trial vi shipped ``osd`` and ``eng``).
   A notice no model uses is a warning (``kit report`` lists it under ``leave_out``).
2. **download** (``download=True``): every model and font the pack pins, and the base pack's fonts,
   fetched from the pinned upstream source when the work folder doesn't already hold the file with
   its pinned size and SHA-256, on the allowlisted hosts only (``datawheel.download_sources``).
3. **layer**: every must-survive unit, and the canary's words, are searchable in a PDF whose text
   layer was written with the pack's fonts by the shipped layer writer.
4. **wheel**: the canary re-renders to its pinned bytes, and the data wheel builds and holds data only.
   With ``download`` and no canary in the work folder (a fresh checkout, as in CI), the canary is
   rendered here and kept only when its bytes equal the pin.

Checks 3 and 4 need the pack's files. Without ``download`` they run on files already in the kit
work folder and are skipped (not failed) when those aren't there; with ``download`` they always run,
unless the download itself failed. Every skipped check is listed in ``warnings``.

**Re-bench** (``rebench=True``): the pack's corpus is rendered again and the manifest's reader set
for this platform is benched through ``pipeline.read_pages``, then compared with the numbers the
manifest claims for this platform. Tolerance (DESIGN §7.3): 0.01 absolute for a reader set without
Vision, 0.02 with it; a must-survive floor is absolute. A platform the pack claims nothing for is
measured and checked against the floors only, and the re-bench says so in ``warning`` (and the
result in ``warnings``), so a pass that compared no number is never silent.

``PackVerify.ok`` is False on any failed check. The CLI exits 1 when any of the four checks failed
or the re-bench couldn't run, whatever the re-bench found; only when the re-bench findings are the
only problems does it exit 6 (a floor failed) or 7 (drift beyond tolerance), 6 winning over 7.

The re-bench renders with ``render.BENCH_SEED``, the seed ``kit render`` uses, and refuses a corpus
with fewer than two fonts for any of the pack's scripts (DESIGN §3.6), so CI enforces both.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import PdfIntakeError

if TYPE_CHECKING:
    from ..langpacks.manifest import Manifest

DEFAULT_TOLERANCE: Mapping[str, float] = {"default": 0.01, "vision": 0.02}
CONDS = ("clean", "scan", "small")


@dataclass
class PackVerify:
    ok: bool
    problems: list[str]
    rebench: dict[str, Any] | None
    checks: dict[str, str] = field(default_factory=dict)
    floor_failed: bool = False
    drifted: bool = False
    # What passed without being checked: a skipped check, a platform the pack claims no numbers for.
    # Never a failure on its own; the JSON lists them so a pass that compared nothing isn't silent.
    warnings: list[str] = field(default_factory=list)

    @property
    def other_failed(self) -> bool:
        """A check failed, or the re-bench couldn't run: a plain failure, whatever else was found."""
        return any(v == "failed" for v in self.checks.values()) or bool(self.rebench and "error" in self.rebench)


def _numbers_only(v: Any, path: str, out: list[str]) -> None:
    if isinstance(v, bool) or v is None or isinstance(v, (int, float)):
        return
    if isinstance(v, Mapping):
        for k, x in v.items():
            _numbers_only(x, f"{path}.{k}", out)
        return
    if isinstance(v, list):
        for i, x in enumerate(v):
            _numbers_only(x, f"{path}[{i}]", out)
        return
    out.append(f"{path} holds text; this file holds numbers only")


_BENCH_TEXT_KEYS = {"platform", "engine_versions", "reader_set", "primary_metric", "page", "cond",
                    "kit", "corpus", "measured"}
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,60}")


def _bench_file_problems(data: Any) -> list[str]:
    """bench.json: kit-generated structure (reader sets, ids, metric names) and numbers; no free text."""
    out: list[str] = []

    def walk(v: Any, path: str) -> None:
        if isinstance(v, Mapping):
            for k, x in v.items():
                if k in _BENCH_TEXT_KEYS:
                    continue
                if k == "error":
                    # A failed set keeps only its error code: a message can name a local path.
                    if not (isinstance(x, Mapping) and set(x) == {"code"} and isinstance(x["code"], str)
                            and _ERROR_CODE_RE.fullmatch(x["code"])):
                        out.append(f"bench.json{path}.error holds more than an error code; bench.json holds numbers only")
                    continue
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
        elif isinstance(v, str):
            out.append(f"bench.json{path} holds text; bench.json holds numbers only")

    walk(data, "")
    return out


# A model file name as a notice lists it (one per line, beside its pinned source URL).
_NOTICE_MODEL_RE = re.compile(r"[A-Za-z0-9_.+-]+\.(?:traineddata|onnx)\b")


def notice_findings(pack_dir: Path, raw: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(problems, warnings) for the ``NOTICE-*`` files in ``pack_dir``: a notice the manifest's models
    name must name exactly those model files, each with its pinned source; one no model names is a
    warning (it must not be committed). A notice that is missing is the wheel check's to report."""
    pinned: dict[str, dict[str, str]] = {}
    for m in raw.get("models") or ():
        if isinstance(m, Mapping) and isinstance(m.get("notice"), str):
            pinned.setdefault(m["notice"], {})[str(m.get("file"))] = str(m.get("source"))
    problems: list[str] = []
    warnings: list[str] = []
    for f in sorted(Path(pack_dir).glob("NOTICE-*")):
        want = pinned.get(f.name)
        if not want:
            warnings.append(f"{f.name}: no model in the manifest uses it; don't commit it")
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"{f.name}: not readable ({exc})")
            continue
        named = set(_NOTICE_MODEL_RE.findall(text))
        if named != set(want):
            problems.append(f"{f.name} names {sorted(named)}, but the manifest pins {sorted(want)} with it "
                            "(`pdfintake kit choose` rewrites it)")
        problems += [f"{f.name}: {file} isn't listed with its pinned source {src}"
                     for file, src in want.items() if src not in text]
    return problems, warnings


def _check_manifest(pack_dir: Path, raw: Mapping[str, Any], base: "Manifest | None") -> list[str]:
    from ..langpacks import manifest as M
    from . import corpus as C

    from .choose import POLICY, policy_message

    problems = [f"manifest: {p}" for p in M.check(raw, base=base)]
    if raw.get("canary") is None:
        problems.append("manifest: no canary yet (run `pdfintake kit wheel`)")
    plats = (raw.get("bench") or {}).get("platforms") if isinstance(raw.get("bench"), Mapping) else None
    for plat, claim in sorted((plats or {}).items()) if isinstance(plats, Mapping) else ():
        policy = ((claim or {}).get("selection") or {}).get("policy") if isinstance(claim, Mapping) else None
        if policy != POLICY:
            problems.append(f"manifest: {plat}: {policy_message(policy, str(raw.get('id') or pack_dir.name))}")
    cfile = pack_dir / C.CORPUS_FILE
    if cfile.is_file():
        try:
            C.load(cfile)
        except PdfIntakeError as exc:
            problems.append(f"corpus: {exc.message}")
    for name, check in (("bench.json", _bench_file_problems), ("bench.private.json", None)):
        f = pack_dir / name
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: not readable JSON ({exc})")
            continue
        if check is not None:
            problems += check(data)
        else:
            found: list[str] = []
            _numbers_only(data, "bench.private.json", found)
            problems += found
    return problems


def _fonts(manifest: "Manifest", base: "Manifest | None", files: Path, base_files: Path | None) -> list[Any]:
    """The pack's layer fonts as FontSpecs (base fonts after), when every file is present; else []."""
    from ..pdf.textlayer import FontSpec

    specs = []
    for i, f in enumerate(manifest.fonts):
        p = files / f.file
        if not p.is_file():
            return []
        specs.append(FontSpec(f.key, p, f.sha256, tuple(f.covers), i))
    if base is not None and base_files is not None:
        for j, f in enumerate(base.fonts):
            p = base_files / f.file
            if p.is_file():
                specs.append(FontSpec(f"en-{f.key}", p, f.sha256, tuple(f.covers), len(manifest.fonts) + j))
    return specs


def _sample_words(manifest: "Manifest") -> list[str]:
    """Words the layer must keep: the canary's, and one per must-survive character, each placed
    between two letters of its script so it reads as a word."""
    from .score import chars_script

    letters = {"Latin": "ab", "Thai": "กข", "Greek": "αβ", "Cyrillic": "аб", "Hebrew": "אב", "Arabic": "بت",
               "Devanagari": "कख", "Hangul": "가나"}
    out = list(manifest.canary.must_find) if manifest.canary is not None else []
    for ms in manifest.must_survive:
        sc = chars_script(ms.chars) or manifest.scripts[0]
        a, b = letters.get(sc, "ab")[0], letters.get(sc, "ab")[-1]
        out += [f"{a}{ch}{b}" for ch in ms.chars]
    return out


def layer_round_trip(words: list[str], fonts: list[Any], dest: Path) -> list[str]:
    """Write ``words`` as a text layer on a blank page with ``fonts``; the words pdfium can't find."""
    import pikepdf
    import pypdfium2

    from ..pdf import textlayer
    from ..pdf.render import PDFIUM_LOCK
    from ..types import LayerLine, LayerPage

    fs = textlayer.FontSet.from_specs(fonts)
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    lines = []
    y = 60.0
    for w in words:
        lines.append(LayerLine(w, ((60.0, y), (60.0 + 30.0 * len(w), y), (60.0 + 30.0 * len(w), y + 40.0),
                                   (60.0, y + 40.0))))
        y += 50.0
        if y > 3200:
            break
    textlayer.write_layer(pdf, [LayerPage(0, 2480, 3508, tuple(lines))], fs, marker={})
    pdf.save(dest)
    pdf.close()
    with PDFIUM_LOCK:
        doc = pypdfium2.PdfDocument(str(dest))
        try:
            page = doc[0]
            tp = page.get_textpage()
            text = tp.get_text_range()
            tp.close()
            page.close()
        finally:
            doc.close()
    import unicodedata

    hay = unicodedata.normalize("NFC", "".join(text.split()))
    return [w for w in words[: len(lines)] if unicodedata.normalize("NFC", "".join(w.split())) not in hay]


def compare(claim: Mapping[str, Any] | None, measured: Mapping[str, Any], must: Any, *, tolerance: float) -> dict[str, Any]:
    """The re-bench against the claim: ``drift`` (beyond tolerance) and ``floors`` (absolute) findings."""
    drift: list[str] = []
    floors: list[str] = []
    for ms in must:
        row = (measured.get("must") or {}).get(ms.name) or {}
        for cond in CONDS:
            v = row.get(cond)
            if v is not None and v < ms.floor - 1e-9:
                floors.append(f"{ms.name} {cond}: {v:.4f} is below the floor {ms.floor}")
    if claim:
        rows = dict(measured.get("metrics") or {})
        rows.update({f"must-{k}": v for k, v in (measured.get("must") or {}).items()})
        for name, triple in (claim.get("metrics") or {}).items():
            got = rows.get(name) or {}
            for cond, want in zip(CONDS, triple):
                v = got.get(cond)
                if v is None:
                    drift.append(f"{name} {cond}: claimed {want}, not measured")
                elif abs(float(v) - float(want)) > tolerance + 1e-9:
                    drift.append(f"{name} {cond}: claimed {want}, measured {v:.4f} (tolerance {tolerance})")
    return {"drift": drift, "floors": floors}


def verify_pack(pack_dir: Path, *, download: bool, rebench: bool, tolerance: Mapping[str, float] = DEFAULT_TOLERANCE,
                work: Path | None = None, platform_key: str | None = None) -> PackVerify:
    """Run the four checks (and the re-bench) on ``pack_dir``'s ``manifest.json``."""
    from .. import platforms
    from ..langpacks import manifest as M
    from . import bench as B

    pack_dir = Path(pack_dir)
    packs_root, pack_id = pack_dir.parent, pack_dir.name
    res = PackVerify(True, [], None)
    try:
        data, raw = B.read_raw(pack_dir / B.MANIFEST_FILE)
        base = B.base_manifest(packs_root, pack_id)
    except PdfIntakeError as exc:
        return PackVerify(False, [exc.message], None, {"manifest": "failed"})
    found = _check_manifest(pack_dir, raw, base)
    if not raw.get("base"):
        notice_problems, notice_warnings = notice_findings(pack_dir, raw)
        found += [f"notice: {p}" for p in notice_problems]
        res.warnings += [f"notice: {w}" for w in notice_warnings]
    res.checks["manifest"] = "failed" if found else "ok"
    res.problems += found
    if found:
        res.ok = False
        for name in ("download", "layer", "wheel"):
            res.checks[name] = "skipped: the manifest has problems"
        return res
    manifest = M.parse(data, base=base)
    wroot = B.work_root(work)
    files = B.files_dir(wroot, pack_id)
    base_files = B.files_dir(wroot, B.BASE_ID)

    base_font_error = None
    if download:
        try:
            B.ensure_files(raw, files, download=True)
            res.checks["download"] = "ok"
        except PdfIntakeError as exc:
            res.checks["download"] = "failed"
            res.problems.append(f"download: {exc.message}")
        if base is not None:
            try:
                _, braw = B.read_raw(packs_root / B.BASE_ID / B.MANIFEST_FILE)
                # The base manifest whole, without its models: datawheel refuses a manifest without
                # its id, version and title, and a bare {"fonts": ...} made this a silent no-op, so
                # a fresh work folder never had the base fonts (kit trial es, item 20).
                B.ensure_files({**braw, "models": []}, base_files, download=True)
            except PdfIntakeError as exc:
                base_font_error = exc.message              # the installed base pack's fonts may still serve
    else:
        res.checks["download"] = "skipped: pass --download to fetch from upstream"

    fonts = _fonts(manifest, base, files, base_files if base is not None else None)
    if not fonts and base is not None:
        try:
            from ..langpacks import store

            fonts = store.base().font_specs() if not manifest.fonts else []
        except PdfIntakeError:
            fonts = []
    words = _sample_words(manifest)
    if not fonts:
        if not download:
            res.checks["layer"] = "skipped: the layer fonts aren't in the work folder (pass --download)"
        elif res.checks["download"] == "failed":
            res.checks["layer"] = "skipped: the download failed"
        else:
            res.checks["layer"] = "failed"
            res.problems.append("layer: no layer fonts to write with"
                                + (f" (the base pack's fonts couldn't be downloaded: {base_font_error})"
                                   if base_font_error else ""))
    elif not words:
        res.checks["layer"] = "skipped: no canary or must-survive words to look for"
    else:
        with tempfile.TemporaryDirectory(prefix="pdfintake-kit-verify-") as tmp:
            try:
                lost = layer_round_trip(words, fonts, Path(tmp) / "layer.pdf")
            except PdfIntakeError as exc:
                lost = [f"(the layer couldn't be written: {exc.message})"]
        res.checks["layer"] = "failed" if lost else "ok"
        res.problems += [f"layer: {w!r} isn't searchable in a layer written with the pack's fonts" for w in lost]

    _check_wheel(res, pack_dir, manifest, raw, files, fonts, download=download)
    res.warnings += [f"{name} {state}" for name, state in res.checks.items() if state.startswith("skipped")]

    if rebench:
        key = platform_key or platforms.current().key
        try:
            res.rebench = _rebench(pack_dir, manifest, raw, wroot, key, tolerance)
        except PdfIntakeError as exc:
            res.rebench = {"platform": key, "error": {"code": exc.code, "message": exc.message}}
            res.problems.append(f"rebench: {exc.message}")
        else:
            res.problems += [f"rebench: {d}" for d in res.rebench["drift"]]
            res.problems += [f"rebench: {f}" for f in res.rebench["floors"]]
            res.drifted = bool(res.rebench["drift"])
            res.floor_failed = bool(res.rebench["floors"])
            if res.rebench.get("warning"):
                res.warnings.append(f"rebench: {res.rebench['warning']}")
    res.ok = not res.problems
    return res


def _check_wheel(res: PackVerify, pack_dir: Path, manifest: "Manifest", raw: Mapping[str, Any], files: Path,
                 fonts: list[Any], *, download: bool) -> None:
    """Check 4: the canary re-renders to its pinned bytes and the data wheel builds, data only.

    ``kit wheel`` leaves the canary in the work folder. A fresh checkout has none (CI never runs
    ``kit wheel``), so with ``download`` the canary is rendered here from the manifest's text, fonts
    and dpi, and kept only when its bytes equal the pin; that is the same re-render check, so the
    wheel check runs in CI too (kit trial es, item 20)."""
    from .. import datawheel
    from ..integrity import sha256_file
    from . import bench as B
    from . import render as R

    missing = B.ensure_files(raw, files, download=False)
    canary = manifest.canary
    canary_file = files / canary.file if canary is not None else None
    if missing or canary_file is None:
        if not download:
            res.checks["wheel"] = "skipped: the pack's files aren't in the work folder (pass --download)"
        elif res.checks.get("download") == "failed":
            res.checks["wheel"] = "skipped: the download failed"
        else:
            res.checks["wheel"] = "failed"
            res.problems.append("wheel: the pack's files aren't in the work folder: " + ", ".join(missing or ["canary"]))
        return
    problems: list[str] = []
    if fonts and (download or canary_file.is_file()):
        with tempfile.TemporaryDirectory(prefix="pdfintake-kit-canary-") as tmp:
            again = R.render_canary(canary_text(manifest), [Path(f.path) for f in fonts], Path(tmp) / canary.file,
                                    dpi=canary.dpi)
            got = sha256_file(again)
            if got != canary.sha256:
                problems.append(f"wheel: the canary doesn't re-render to its pinned bytes here ({again.stat().st_size} "
                                f"bytes, SHA-256 {got[:12]}…; the manifest pins {canary.bytes} bytes, "
                                f"{canary.sha256[:12]}…)")
            elif not canary_file.is_file():
                canary_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_dest = canary_file.with_name(canary_file.name + ".tmp")
                shutil.copyfile(again, tmp_dest)
                os.replace(tmp_dest, canary_file)
    if not canary_file.is_file():
        if problems:
            res.checks["wheel"] = "failed"
            res.problems += problems
        elif not download:
            res.checks["wheel"] = "skipped: the canary isn't in the work folder (pass --download, or run kit wheel)"
        else:
            res.checks["wheel"] = "failed"
            res.problems.append("wheel: the canary couldn't be rendered: no layer fonts")
        return
    with tempfile.TemporaryDirectory(prefix="pdfintake-kit-wheel-") as tmp:
        try:
            whl = datawheel.build(pack_dir, files, Path(tmp), dist_name=datawheel._dist_name(raw))
            problems += [f"wheel: {p}" for p in datawheel.wheel_problems(whl, raw)]
        except datawheel.DataWheelError as exc:
            problems.append(f"wheel: {exc}")
    res.checks["wheel"] = "failed" if problems else "ok"
    res.problems += problems


def canary_text(manifest: "Manifest") -> str:
    """The canary line: its ``must_find`` strings, space-separated."""
    return " ".join(manifest.canary.must_find) if manifest.canary is not None else ""


def _rebench(pack_dir: Path, manifest: "Manifest", raw: Mapping[str, Any], wroot: Path, key: str,
             tolerance: Mapping[str, float]) -> dict[str, Any]:
    from . import bench as B
    from . import corpus as C
    from . import render as R

    pack_id = manifest.id
    corpus = C.load(pack_dir / C.CORPUS_FILE)
    files = B.files_dir(wroot, pack_id)
    B.ensure_files(raw, files, download=True)
    draft = B.installed_pack(manifest, files, wroot / pack_id / "pack")
    base = B.base_pack(pack_dir.parent, wroot, draft, download=True)
    fonts = [C.fetch_font(f, wroot / pack_id / "corpus-fonts") for f in corpus.fonts]
    pages = R.render_pages(corpus, fonts, B.pages_dir(wroot, pack_id), seed=R.BENCH_SEED,
                           cache=wroot / pack_id / "texts", scripts=manifest.scripts)
    rs = manifest.readers_for(key)
    result = B.bench(pages, draft, base, key, [rs])
    measured = result.sets[0]
    if not measured.get("ok"):
        err = measured.get("error") or {}
        raise PdfIntakeError("kit_rebench_failed", f"the reader set didn't run here ({err.get('code')}): "
                                                   f"{result.messages.get(0, 'no message')}")
    tol = float(tolerance.get("vision" if measured.get("vision") else "default", 0.01))
    claim = B._plain((manifest.bench.get("platforms") or {}).get(key))
    found = compare(claim, measured, manifest.must_survive, tolerance=tol)
    out = {"platform": key, "tolerance": tol, "claimed": claim or None,
           "measured": {"metrics": measured.get("metrics"), "must": measured.get("must"),
                        "must_coverage": measured.get("must_coverage"),
                        "worst_page": measured.get("worst_page"), "seconds_per_page": measured.get("seconds_per_page")},
           "engine_versions": result.engine_versions, **found}
    if not claim:
        # Not a failure (a pack may not have been benched here yet), but never a silent pass: the
        # re-bench compared no number on this platform (kit trial es, item 18).
        out["warning"] = (f"the pack claims no numbers for {key}, so only the must-survive floors were checked; "
                          f"a `pdfintake kit bench {pack_id}` on {key} and a new `kit choose` give it numbers to compare")
    return out
