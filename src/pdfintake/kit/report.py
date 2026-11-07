"""``kit report``: the pull request body for a pack, built from the files in ``packs/<id>/``.

It holds the manifest summary (languages, scripts, reader sets with each use labelled by its model,
the pack's fills and repairs with their parameters, models and fonts with their sources and licences,
and the base pack's models the reader sets read with, so the table is never empty without a word),
the must-survive groups with their characters, the
corpus (each text's title, source, revision and licence), the bench numbers per platform with the
selection and its margin, and, from ``bench.json``, which set each platform reads with (and the sets
merged into it), its must-survive coverage per condition (read / truth characters, with the
conditions that aren't measurable marked and a warning for a group measurable nowhere) and, when the
tie-break decided, the digit figures that decided it; the numbers of the ``default`` reader set even
when no benched platform reads with it, the supported platforms with no numbers yet, any override
reason (flagged for review),
the optional private aggregate, and the checklist of the language-pack PR template (kit trials it,
pt and vi). Everything comes from the pack's own files; nothing is read from a document.

``pack_files`` says which files of the pack folder the PR commits and which it leaves out.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .bench import BENCH_FILE, MANIFEST_FILE, PRIVATE_FILE, KitFileError, read_json, read_raw, use_label

CONDS = ("clean", "scan", "small")
MARGIN_NOTE = ("The margin is the chosen set's worst-condition score minus the runner-up's (the next set in "
               "the policy's order that reads differently); it is negative when the runner-up scored higher "
               "but lost on the tie-break (worst-condition digit recall, then fewer engines, then speed), or "
               "when an override chose a lower-scoring set.")
COVERAGE_NOTE = ("Must-survive coverage is read / truth: the group's characters inside words (or, for a trigram "
                 "or char group, spans) the set read, of those in the truth; the figure counts only the read "
                 "ones. A condition with fewer than {n} read characters isn't measurable, and its floor doesn't "
                 "apply there.")


def _fmt(v: Any, places: int = 3) -> str:
    return f"{v:.{places}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "–"


def _readers(raw: Mapping[str, Any]) -> list[str]:
    out = []
    for key, rs in (raw.get("readers") or {}).items():
        uses = ", ".join(f"`{use_label(u)}` ({u.get('engine')}, {u.get('role')})" for u in rs.get("engines") or ())
        pick = rs.get("pick") or {}
        out.append(f"- `{key}`: {uses}; pick `{pick.get('rule')}` v{pick.get('version')}, fallback `{rs.get('fallback')}`")
    return out


def _steps(raw: Mapping[str, Any], key: str) -> str:
    """The pack's fills or repairs, each with its version and parameters (the bench read with them)."""
    out = []
    for s in raw.get(key) or ():
        params = s.get("params")
        out.append(f"`{s.get('name')}` v{s.get('version')}"
                   + (f" `` {json.dumps(params, ensure_ascii=False, sort_keys=True)} ``" if params else ""))
    return ", ".join(out) or "none"


def _base_models_used(raw: Mapping[str, Any], pack_dir: Path) -> list[str]:
    """The base pack's models the reader sets read with, as ``en:key`` (with the file where
    ``packs/en`` names it): ``en:`` references, and Tesseract languages the pack pins no model for."""
    own = {str(m.get("file", "")).rsplit(".", 1)[0] for m in raw.get("models") or () if m.get("kind") == "traineddata"}
    refs: list[str] = []
    for rs in (raw.get("readers") or {}).values():
        for u in (rs or {}).get("engines") or ():
            refs += [r for r in u.get("models") or () if isinstance(r, str) and r.startswith("en:")]
            lang = (u.get("params") or {}).get("lang") if u.get("engine") == "tesseract" else None
            if isinstance(lang, str):
                refs += [f"en:{x}" for x in lang.split("+") if x and x not in own]
    files: dict[str, str] = {}
    try:
        _, base = read_raw(Path(pack_dir).parent / "en" / MANIFEST_FILE)
        files = {f"en:{m.get('key')}": str(m.get("file")) for m in base.get("models") or ()}
    except Exception:                                   # no packs/en: the references alone
        pass
    out = []
    for r in dict.fromkeys(refs):
        out.append(f"`{r}` (`{files[r]}`)" if r in files else f"`{r}`")
    return out


def _canon(rs: Any) -> str:
    """A reader set as a comparable string (each use's model list sorted)."""
    if not isinstance(rs, Mapping):
        return ""
    engines = [{**u, "models": sorted(u.get("models") or ())} for u in rs.get("engines") or ()]
    return json.dumps({**rs, "engines": engines}, sort_keys=True, default=str)


def _parsed_readers(pack_dir: Path, raw: Mapping[str, Any]) -> dict[str, str]:
    """Each reader set of the manifest as the loader reads it (defaults filled in), canonical; the raw
    JSON when the manifest can't be parsed here."""
    from . import bench as B

    try:
        m, _raw, _p = B.load_manifest(Path(pack_dir).parent, str(raw.get("id")), draft=False)
        return {k: _canon(B.reader_set_json(rs)) for k, rs in m.readers.items()}
    except Exception:
        return {k: _canon(rs) for k, rs in (raw.get("readers") or {}).items()}


def _triple_rows(s: Mapping[str, Any]) -> list[str]:
    rows = dict(s.get("metrics") or {})
    rows.update({f"must-{k}": v for k, v in (s.get("must") or {}).items()})
    out = ["| Metric | clean | scan | small |", "|---|---|---|---|"]
    for name, row in sorted(rows.items()):
        if isinstance(row, Mapping):
            out.append(f"| {name} | " + " | ".join(_fmt(row.get(c)) for c in CONDS) + " |")
    return out


def _default_lines(pack_dir: Path, raw: Mapping[str, Any], claimed: Mapping[str, Any]) -> list[str]:
    """Which platforms read with ``default``, and its numbers when no benched platform reads with it
    (kit trial vi: the Mac chose a Vision set, and Linux and Windows read with the draft's own set)."""
    from .. import platforms

    readers = raw.get("readers") or {}
    own = {k for k in readers if k != "default"}
    supported = [p.key for p in platforms.PLATFORMS if p.supported]
    readers_default = [k for k in supported if k not in own]
    if not readers_default:
        return []
    shown = [k for k in sorted(claimed) if k not in own]
    head = f"- `default` is read on {', '.join(f'`{k}`' for k in readers_default)}"
    if shown:
        return [f"{head}; its numbers are the `{shown[0]}` table above."]
    canon = _parsed_readers(pack_dir, raw).get("default")
    bench_file = Path(pack_dir) / BENCH_FILE
    found = None
    if canon and bench_file.is_file():
        try:
            data = read_json(bench_file, code="kit_bench_invalid", what="the bench results")
        except KitFileError:
            data = {}
        for plat, res in sorted(((data or {}).get("platforms") or {}).items()):
            for s in (res or {}).get("sets") or ():
                if s.get("ok") and _canon(s.get("reader_set")) == canon:
                    found = (plat, s)
                    break
            if found:
                break
    if found is None:
        return [f"{head}; no platform's bench has measured it yet."]
    plat, s = found
    return [f"{head}. No benched platform reads with it; measured as set {s.get('index')} of the `{plat}` bench "
            f"(worst page {_fmt(s.get('worst_page'))}, {_fmt(s.get('seconds_per_page'))} s/page):", "",
            *_triple_rows(s), ""]


def _bench_data(pack_dir: Path) -> dict[str, Any]:
    """bench.json, or {} when there is none or it can't be read."""
    bench_file = Path(pack_dir) / BENCH_FILE
    if not bench_file.is_file():
        return {}
    try:
        data = read_json(bench_file, code="kit_bench_invalid", what="the bench results")
    except KitFileError:
        return {}
    return data if isinstance(data, dict) else {}


def _must(raw: Mapping[str, Any]) -> list[Any]:
    from ..langpacks.manifest import MustSurvive

    out = []
    for x in raw.get("must_survive") or ():
        try:
            out.append(MustSurvive(str(x["name"]), str(x["chars"]), str(x["unit"]), float(x["floor"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _coverage_lines(s: Mapping[str, Any], must: list[Any]) -> list[str]:
    """Per must-survive group, read / truth characters per condition, with "not measurable" marked."""
    out = []
    for ms in must:
        cov = (s.get("must_coverage") or {}).get(ms.name)
        if not isinstance(cov, Mapping):
            continue
        parts = []
        for c in CONDS:
            x = cov.get(c) or {}
            part = f"{c} {x.get('read', 0)}/{x.get('truth', 0)}"
            parts.append(part if x.get("measurable") else part + " (not measurable)")
        line = f"- Must-survive coverage, `{ms.name}`: " + ", ".join(parts)
        if not any(isinstance(x, Mapping) and x.get("measurable") for x in cov.values()):
            line += (f". **Warning:** `{ms.name}` couldn't be measured in any condition, so its floor wasn't "
                     "checked on this platform")
        out.append(line)
    return out


def _selection_lines(pack_dir: Path, raw: Mapping[str, Any], plat: str, claim: Mapping[str, Any],
                     data: Mapping[str, Any]) -> list[str]:
    """For one claimed platform: which bench set the manifest reads with (and what was merged into it),
    its must-survive coverage, and, when the tie-break decided the choice, what decided it."""
    from . import bench as B
    from . import choose as CH
    from .score import MIN_MEASURABLE

    res = ((data.get("platforms") or {}).get(plat)) if isinstance(data.get("platforms"), Mapping) else None
    if not isinstance(res, Mapping):
        return []
    parsed = _parsed_readers(pack_dir, raw)
    canon = parsed.get(plat) or parsed.get("default")
    sets = [s for s in res.get("sets") or () if isinstance(s, Mapping) and s.get("ok")]
    found = next((s for s in sets if canon and _canon(s.get("reader_set")) == canon), None)
    if found is None:
        return [f"- The `{plat}` reader set isn't one of the sets in bench.json: run `pdfintake kit choose "
                f"{raw.get('id')}` again."]
    merged = found.get("merged") or []
    also = f" (also proposed as {', '.join(f'set {i}' for i in merged)} before merging)" if merged else ""
    must = _must(raw)
    lines = [f"- `{plat}` reads with set {found.get('index')} of its bench{also}."]
    cov = _coverage_lines(found, must)
    if cov:
        lines += cov + [f"- {COVERAGE_NOTE.format(n=MIN_MEASURABLE)}"]
    if (claim.get("selection") or {}).get("override_reason"):
        return lines
    try:
        sel = CH.choose(B.result_from_json(res), must)
    except Exception:                                   # bench.json no longer passes: kit choose says why
        return lines
    if sel.chosen == found.get("index") and sel.tie_break:
        def label(i: int) -> str | None:
            s = next((x for x in sets if x.get("index") == i), None)
            return B.set_label(s.get("reader_set") or {}) if s else None

        lines.append(f"- {CH.tie_break_line(sel.tie_break, label)}")
    return lines


def _benched_platforms(pack_dir: Path) -> set[str]:
    """The platform keys ``bench.json`` holds numbers for (empty without a readable bench.json)."""
    bench_file = Path(pack_dir) / BENCH_FILE
    if not bench_file.is_file():
        return set()
    try:
        data = read_json(bench_file, code="kit_bench_invalid", what="the bench results")
    except KitFileError:
        return set()
    plats = data.get("platforms") if isinstance(data, dict) else None
    return {str(k) for k, v in plats.items() if v} if isinstance(plats, dict) else set()


def pr_body(pack_dir: Path) -> str:
    """The Markdown body for the pack's PR (paste it into the language-pack template)."""
    pack_dir = Path(pack_dir)
    _, raw = read_raw(pack_dir / MANIFEST_FILE)
    pid = raw.get("id")
    lines = [f"### Pack `{pid}` {raw.get('version')}: {raw.get('title')}", ""]
    langs = ", ".join(f"{x.get('tag')} ({x.get('role')})" for x in raw.get("languages") or ())
    lines += [f"- Languages: {langs}", f"- Scripts: {', '.join(raw.get('scripts') or [])}",
              f"- Requires core: `{raw.get('requires_core')}`", "", "**Reader sets**", ""]
    lines += _readers(raw)
    lines += [f"- Fills: {_steps(raw, 'fills')}", f"- Repairs: {_steps(raw, 'repairs')}"]
    lines += ["", "**Models and fonts**", ""]
    if raw.get("models") or raw.get("fonts"):
        lines += ["| File | Kind | Bytes | Licence | Source |", "|---|---|---|---|---|"]
        for m in raw.get("models") or ():
            lines.append(f"| `{m.get('file')}` | {m.get('kind')} | {m.get('bytes')} | {m.get('licence')} | {m.get('source')} |")
        for f in raw.get("fonts") or ():
            lines.append(f"| `{f.get('file')}` | font | {f.get('bytes')} | {f.get('licence')} | {f.get('source')} |")
    else:
        lines.append("- None of its own: the pack pins no model or font, and its wheel carries only the manifest "
                     "and the canary.")
    base_used = _base_models_used(raw, pack_dir)
    if base_used:
        lines += ["", "- The base pack's models the reader sets read with: " + ", ".join(base_used)]
    canary = raw.get("canary")
    if canary:
        words = ", ".join(f"`{w}`" for w in canary.get("must_find") or ())
        lines += ["", f"- Canary: `{canary['file']}`, {canary['bytes']} bytes, must find {words}"]
    else:
        lines += ["", "- Canary: none yet"]
    ms = raw.get("must_survive") or []
    lines.append("- Must survive: " + ("; ".join(f"`{m['name']}` {m['unit']}s holding `{m.get('chars')}`, floor {m['floor']}"
                                               for m in ms) or "none"))

    corpus_file = pack_dir / "corpus.json"
    lines += ["", "**Corpus**", ""]
    if corpus_file.is_file():
        c = read_json(corpus_file, code="invalid_corpus", what="the corpus list")
        if not isinstance(c, dict):
            raise KitFileError("invalid_corpus", f"{corpus_file} must hold a JSON object")
        lines.append(f"- `{c.get('corpus_id')}`: {len(c.get('texts') or [])} texts, {len(c.get('fonts') or [])} fonts, "
                     f"pages {json.dumps(c.get('pages'), sort_keys=True)}")
        for t in c.get("texts") or ():
            title = str(t.get("title") or "").replace("`", "'")
            lines.append(f"  - {title}: {t.get('source')} `{t.get('lang')}` revision {t.get('revid')}, "
                         f"{t.get('licence')}: {t.get('url')}")
    else:
        lines.append("- no corpus.json")

    bench = raw.get("bench") or {}
    lines += ["", "**Bench**", "", f"- Kit {bench.get('kit')}, corpus `{bench.get('corpus')}`"]
    versions = bench.get("engine_versions") or {}
    if versions:
        lines.append("- Engines: " + ", ".join(f"{k} {v}" for k, v in sorted(versions.items())))
    overrides = []
    bench_data = _bench_data(pack_dir)
    for plat, row in sorted((bench.get("platforms") or {}).items()):
        sel = row.get("selection") or {}
        lines += ["", f"`{plat}`: worst page {_fmt(row.get('worst_page'))}, {_fmt(row.get('seconds_per_page'))} s/page, "
                      f"policy `{sel.get('policy')}`, margin {_fmt(sel.get('margin'), 4)}", "",
                  "| Metric | clean | scan | small |", "|---|---|---|---|"]
        for name, triple in sorted((row.get("metrics") or {}).items()):
            lines.append(f"| {name} | " + " | ".join(_fmt(v) for v in triple) + " |")
        extra = _selection_lines(pack_dir, raw, plat, row, bench_data)
        if extra:
            lines += [""] + extra
        if sel.get("override_reason"):
            overrides.append((plat, sel["override_reason"]))
    if not bench.get("platforms"):
        lines.append("- no platform benched yet")
    else:
        lines += ["", f"- {MARGIN_NOTE}"]
    lines += _default_lines(pack_dir, raw, bench.get("platforms") or {})
    from .. import platforms

    claimed = set(bench.get("platforms") or {})
    benched = _benched_platforms(pack_dir)
    # Like `kit choose`'s `unbenched`: a platform bench.json has numbers for is benched, whether or not
    # the manifest claims them yet (kit trial review F3: the two lists contradicted each other).
    unbenched = [p.key for p in platforms.PLATFORMS if p.supported and p.key not in claimed | benched]
    if unbenched:
        lines.append(f"- Not benched yet: {', '.join(f'`{k}`' for k in unbenched)}. CI checks them against the "
                     "must-survive floors only; the maintainer benches them before the pack ships.")
    unclaimed = sorted(benched - claimed)
    if unclaimed:
        lines.append(f"- Benched but no numbers claimed in the manifest yet: {', '.join(f'`{k}`' for k in unclaimed)}. "
                     f"Run `pdfintake kit choose {pid}` again.")
    for plat, reason in overrides:
        lines += ["", f"**Needs review:** the `{plat}` selection overrides the kit's choice: {reason}"]

    private = pack_dir / PRIVATE_FILE
    if private.is_file():
        p = read_json(private, code="kit_private_invalid", what="the private bench")
        if not isinstance(p, dict):
            raise KitFileError("kit_private_invalid", f"{private} must hold a JSON object")
        lines += ["", f"**Private bench** ({p.get('pages')} born-digital pages, numbers only; mean, min, max)", ""]
        for name, triple in sorted((p.get("metrics") or {}).items()):
            lines.append(f"- {name}: " + ", ".join(_fmt(v) for v in triple))

    lines += ["", "**Checklist**", "",
              f"- [ ] The PR changes only `packs/{pid}/**`.",
              "- [ ] No binaries: no model, font, image or PDF.",
              "- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.",
              f"- [ ] `pdfintake kit verify {pid} --download` passes.",
              "- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.",
              "- [ ] The must-survive characters were confirmed by someone who reads the language.",
              "- [ ] Every must-survive group was measurable in at least one condition on every benched platform, "
              "or the warning above is explained.", ""]
    return "\n".join(lines)


# The kit's own files in a pack folder; the notices the wheel needs come from the manifest.
_KIT_FILES = (MANIFEST_FILE, "corpus.json", "bench.json", PRIVATE_FILE)


def pack_files(pack_dir: Path) -> dict[str, list[str]]:
    """Which files in ``packs/<id>/`` the PR commits (``commit``) and which it leaves out
    (``leave_out``): the manifest, corpus, bench files and the notices the pack's wheel carries
    (``datawheel._notices``: one per model notice, ``OFL.txt`` when the pack has fonts) are
    committed; anything else, such as ``manifest.draft.json`` or a notice for a model ``kit
    choose`` dropped, is not (kit trial es, items 5 and 16)."""
    from .. import datawheel

    pack_dir = Path(pack_dir)
    _, raw = read_raw(pack_dir / MANIFEST_FILE)
    wanted = set(_KIT_FILES) | set(datawheel._notices(raw))
    present = sorted(p.name for p in pack_dir.iterdir() if p.is_file()) if pack_dir.is_dir() else []
    return {"commit": [n for n in present if n in wanted], "leave_out": [n for n in present if n not in wanted],
            "missing": sorted(n for n in set(datawheel._notices(raw)) if n not in present)}
