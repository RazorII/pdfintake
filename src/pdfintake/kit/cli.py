"""``pdfintake kit``: the add-a-language commands (docs/add-a-language.md).

Every command prints one JSON object on standard output (``kit bench`` also prints one progress line
per set on standard error, so a bench of many minutes shows it's alive) and exits with a documented
code:

* 0: done (``"ok": true``);
* 1: the command failed (``"ok": false`` and ``error`` or ``problems`` say why);
* 2: bad arguments (also ``kit wheel`` with more ``--must-find`` entries than a canary holds, or with
  canary libraries at other versions than ``tools/canary-env.txt`` pins);
* 3: setup (the English base pack or the ``[kit]`` extra is missing);
* 6: a must-survive or metric floor failed, or ``kit init`` found no Tesseract model for the language;
* 7: ``kit verify --rebench`` measured numbers beyond tolerance of what the pack claims;
* 8: ``kit private`` would have written something other than numbers (nothing was written).

``--json`` is accepted by every command (``pack-bench.yml`` passes it); the output is JSON either way.
Nothing here prompts or reads standard input: every choice is a flag.

The pack's sources live in ``packs/<id>/`` (``--packs``, default ``./packs``). Downloads, rendered
pages and built wheels go to the kit work folder (``--work``, default ``<cache>/kit/<id>/``), never
into the repo.

``register`` adds the ``kit`` group to the main parser; each command sets ``func`` to ``run``, which
returns the exit code. ``main`` runs the group on its own (``python -m pdfintake.kit.cli``).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import __version__
from ..errors import (
    EXIT_FAILED,
    EXIT_KIT_DRIFT,
    EXIT_KIT_FLOOR,
    EXIT_OK,
    EXIT_USAGE,
    PackError,
    PdfIntakeError,
    UsageError,
)

TESSDATA_BEST_COMMIT = "e12c65a915945e4c28e237a9b52bc4a8f39a0cec"   # a tessdata_best commit the launch packs pin
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/{commit}/{name}.traineddata"
MAX_MODEL_BYTES = 104857600
# Language code -> (ISO 639-3 code Tesseract names its model by, English title). Anything else needs
# --iso639-3 and --title.
LANGUAGES: dict[str, tuple[str, str]] = {
    "es": ("spa", "Spanish"), "de": ("deu", "German"), "fr": ("fra", "French"), "it": ("ita", "Italian"),
    "pt": ("por", "Portuguese"), "nl": ("nld", "Dutch"), "pl": ("pol", "Polish"), "cs": ("ces", "Czech"),
    "sk": ("slk", "Slovak"), "sl": ("slv", "Slovenian"), "hr": ("hrv", "Croatian"), "ro": ("ron", "Romanian"),
    "sv": ("swe", "Swedish"), "da": ("dan", "Danish"), "no": ("nor", "Norwegian"), "fi": ("fin", "Finnish"),
    "et": ("est", "Estonian"), "lv": ("lav", "Latvian"), "lt": ("lit", "Lithuanian"), "tr": ("tur", "Turkish"),
    "id": ("ind", "Indonesian"), "ms": ("msa", "Malay"), "vi": ("vie", "Vietnamese"), "hu": ("hun", "Hungarian"),
    "el": ("ell", "Greek"), "ru": ("rus", "Russian"), "uk": ("ukr", "Ukrainian"), "bg": ("bul", "Bulgarian"),
    "th": ("tha", "Thai"), "he": ("heb", "Hebrew"), "ar": ("ara", "Arabic"), "hi": ("hin", "Hindi"),
    "ko": ("kor", "Korean"),
}
_MUST_RE = re.compile(r"^([a-z][a-z0-9-]{0,30})=([^:]{1,64})(?::(word|trigram|char))?(?::([0-9.]+))?$")


class NoModel(PdfIntakeError):
    """Tesseract has no model for the language (kit exit 6)."""

    exit_code = EXIT_KIT_FLOOR


class KitFailed(PdfIntakeError):
    exit_code = EXIT_FAILED


# --------------------------------------------------------------------------- output

def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    sys.stdout.flush()


def _ctx(args: argparse.Namespace) -> tuple[Path, Path, str]:
    from . import bench as B

    pack_id = args.pack
    B.pack_dir(args.packs, pack_id)                      # checks the id
    return Path(args.packs), B.work_root(args.work), pack_id


def _platform(args: argparse.Namespace) -> str:
    from .. import platforms

    key = getattr(args, "platform", None) or platforms.current().key
    if key not in {p.key for p in platforms.PLATFORMS}:
        raise UsageError("kit_usage", f"{key!r} isn't a platform key ({', '.join(p.key for p in platforms.PLATFORMS)})")
    return key


def _draft_pack(packs: Path, work: Path, pack_id: str, *, offline: bool, draft: bool = True):
    from . import bench as B

    m, raw, path = B.load_manifest(packs, pack_id, draft=draft)
    files = B.files_dir(work, pack_id)
    B.ensure_files(raw, files, download=not offline)
    pack = B.installed_pack(m, files, work / pack_id / "pack")
    base = B.base_pack(packs, work, pack, download=not offline)
    return m, raw, path, pack, base


# --------------------------------------------------------------------------- init

def _parse_must(specs: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for s in specs or ():
        m = _MUST_RE.match(s)
        if not m:
            raise UsageError("kit_usage", f"--must-survive {s!r}: use NAME=CHARS[:word|trigram|char[:FLOOR]]")
        floor = float(m.group(4)) if m.group(4) else 0.9
        if not 0.0 <= floor <= 1.0:
            raise UsageError("kit_usage", f"--must-survive {s!r}: the floor is from 0 to 1")
        out.append({"name": m.group(1), "chars": m.group(2), "unit": m.group(3) or "word", "floor": floor})
    return out


def _copy_notice(packs: Path, pdir: Path, name: str) -> bool:
    src = packs / "en" / name
    if src.is_file() and not (pdir / name).exists():
        shutil.copyfile(src, pdir / name)
        return True
    return False


def _sync_notice(packs: Path, pdir: Path, raw: dict[str, Any]) -> str:
    """NOTICE-tessdata follows ``manifest.json`` when the pack has one (it is what the wheel and the PR
    carry), else the draft ``raw``."""
    from . import bench as B
    from .choose import sync_notice

    mfile = pdir / B.MANIFEST_FILE
    governing = B.read_raw(mfile)[1] if mfile.is_file() else raw
    return sync_notice(pdir, governing, packs)


def _tessdata_model(iso3: str, commit: str) -> dict[str, Any]:
    """Tesseract's ``tessdata_best`` model for ``iso3`` at ``commit``, pinned by size and SHA-256
    (NoModel, exit 6, when there is none)."""
    import hashlib

    from . import corpus as C

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise UsageError("kit_usage", "--tessdata-commit is a full 40-character commit id")
    url = TESSDATA_URL.format(commit=commit, name=iso3)
    try:
        data = C._http_get(url, cap=MAX_MODEL_BYTES)
    except PackError as exc:
        if exc.detail.get("status") == 404:
            raise NoModel("kit_no_model", f"Tesseract has no {iso3}.traineddata at tessdata_best {commit[:12]}; "
                                          "this language can't be packed with the engines pdfintake has",
                          language=iso3) from None
        raise
    return {"key": iso3, "file": f"{iso3}.traineddata", "engine": "tesseract", "kind": "traineddata",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "source": url,
            "licence": "Apache-2.0", "notice": "NOTICE-tessdata"}


def _init_next(pdir: Path, pack_id: str, raw: dict[str, Any], *, ofl_copied: bool) -> list[str]:
    """``kit init``'s to-do list: whatever the pack still lacks before it can be benched."""
    from . import corpus as C

    out = []
    if not raw.get("must_survive"):
        out.append("must_survive is empty: confirm with the person which characters must survive, "
                   "then run kit init again with --must-survive NAME=CHARS")
    scripts_ = list(raw.get("scripts") or [])
    if scripts_ and scripts_[0] != "Latin" and not raw.get("fonts"):
        out.append("a non-Latin pack needs a layer font: run kit init again with --font URL (a notofonts file)")
    if ofl_copied:
        out.append("OFL.txt was copied from packs/en, whose first line names the copyright holder of the "
                   "English pack's fonts: replace that line with the one in your font's own OFL.txt")
    try:
        c = C.load(pdir / C.CORPUS_FILE)
        empty = not c.texts or not c.fonts
    except PdfIntakeError:
        empty = True
    if empty:
        out.append(f"the corpus has no texts or fonts yet: pin them with `pdfintake kit corpus {pack_id} "
                   "--wikipedia <ids> --font <url> --font <url>` (step 3)")
    return out


def _init_from_manifest(args: argparse.Namespace, packs: Path, pdir: Path, pack_id: str) -> dict[str, Any]:
    """``kit init <id> --from-manifest``: ``manifest.draft.json`` again from ``manifest.json``, for a
    pack whose draft is gone (kit trial es, re-run R2). The readers, languages, detection, fills,
    fonts and must-survive groups are the manifest's (``--must-survive`` replaces the groups); the
    canary and the bench claims are left out, as in a new draft; and the language's Tesseract model
    is pinned again when ``kit choose`` dropped it, so the next bench proposes it again."""
    from ..langpacks import manifest as M
    from . import bench as B

    mfile = pdir / B.MANIFEST_FILE
    if not mfile.is_file():
        raise UsageError("kit_no_manifest", f"{mfile} is missing; --from-manifest rebuilds the draft from it")
    _data, raw = B.read_raw(mfile)
    if raw.get("id") != pack_id:
        raise UsageError("kit_usage", f"{mfile} is for pack {raw.get('id')!r}, not {pack_id!r}")
    if raw.get("base"):
        raise UsageError("kit_usage", "the base pack has no draft; its manifest is edited directly")
    draft = {k: v for k, v in raw.items() if k != "canary"}
    draft["bench"] = {"kit": B.KIT_VERSION, "corpus": "none", "platforms": {}}
    if args.must_survive:
        draft["must_survive"] = _parse_must(args.must_survive)
    tag = pack_id.split("-")[0]
    primary = next((x for x in raw.get("languages") or () if x.get("role") == "primary"), {})
    iso3 = args.iso639_3 or LANGUAGES.get(tag, (None, None))[0] or primary.get("iso639_3")
    repinned = None
    models = list(draft.get("models") or [])
    if iso3 and not any(m.get("file") == f"{iso3}.traineddata" for m in models):
        if not re.fullmatch(r"[a-z]{3}(_[a-z]+)?", iso3):
            raise UsageError("kit_usage", "--iso639-3 is Tesseract's model name, such as spa or chi_sim")
        model = _tessdata_model(iso3, args.tessdata_commit)
        models.append(model)
        repinned = {k: model[k] for k in ("file", "bytes", "sha256", "source")}
    draft["models"] = models
    problems = M.check(draft, base=B.base_manifest(packs, pack_id))
    if problems:
        raise KitFailed("kit_invalid_draft", f"the draft wouldn't load: {'; '.join(problems)}", problems=problems)
    B.write_json(pdir / B.DRAFT_FILE, draft)
    notice = _sync_notice(packs, pdir, draft)
    return {"draft": str(pdir / B.DRAFT_FILE), "from_manifest": str(mfile), "model": repinned,
            "notice": notice, "next": _init_next(pdir, pack_id, draft, ofl_copied=False)}


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    from ..langpacks import manifest as M
    from ..ocr import scripts
    from . import bench as B
    from . import corpus as C

    packs, _work, pack_id = _ctx(args)
    pdir = B.pack_dir(packs, pack_id)
    if (pdir / B.DRAFT_FILE).exists() and not args.force:
        raise UsageError("kit_exists", f"{pdir / B.DRAFT_FILE} already exists; pass --force to start it again")
    if args.from_manifest:
        if args.script or args.font or args.title:
            raise UsageError("kit_usage", "--from-manifest takes the script, title and fonts from manifest.json; "
                                          "leave out --script, --title and --font")
        return _init_from_manifest(args, packs, pdir, pack_id)
    script = args.script
    if not script:
        raise UsageError("kit_usage", "kit init needs --script (or --from-manifest to rebuild the draft)")
    if script not in scripts.SCRIPTS:
        raise UsageError("kit_usage", f"{script!r} isn't a script this core knows ({', '.join(scripts.SCRIPTS)})")
    tag = pack_id.split("-")[0]
    iso3, title = LANGUAGES.get(tag, (None, None))
    iso3 = args.iso639_3 or iso3
    title = args.title or title
    if not iso3 or not title:
        raise UsageError("kit_usage", f"the kit doesn't know {tag!r}; pass --iso639-3 and --title")
    if not re.fullmatch(r"[a-z]{3}(_[a-z]+)?", iso3):
        raise UsageError("kit_usage", "--iso639-3 is Tesseract's model name, such as spa or chi_sim")
    model = _tessdata_model(iso3, args.tessdata_commit)
    fonts = []
    for i, furl in enumerate(args.font or ()):
        f = C.pin_font(furl)
        name = furl.rsplit("/", 1)[-1]
        key = re.sub(r"[^a-z0-9_.-]", "-", name.rsplit(".", 1)[0].lower())[:40] or f"font-{i}"
        fonts.append({"key": key, "file": name, "covers": [script], "bytes": f["bytes"], "sha256": f["sha256"],
                      "source": furl, "licence": "OFL-1.1"})
    latin = script == "Latin"
    languages = [{"tag": tag, "iso639_3": iso3[:3], "role": "primary"}]
    if not latin:
        languages.append({"tag": "en", "iso639_3": "eng", "role": "secondary"})
    tess = {"id": "tesseract", "engine": "tesseract", "role": "reader",
            "params": {"lang": iso3 if latin else f"{iso3}+eng", "psm": 3},
            "models": [iso3] if latin else [iso3, "en:eng"]}
    if latin:
        checker = {"id": "rapidocr", "engine": "rapidocr", "role": "checker", "params": {"rec": "en:rapid-rec-latin"},
                   "models": ["en:rapid-det", "en:rapid-cls", "en:rapid-rec-latin"]}
        default = {"engines": [tess, checker],
                   "pick": {"rule": "primary_with_floor", "version": 1, "params": {"primary": "tesseract", "checker": "rapidocr"}},
                   "fallback": "none"}
    else:
        default = {"engines": [tess], "pick": {"rule": "single", "version": 1, "params": {"reader": "tesseract"}},
                   "fallback": "none"}
    raw = {"schema": 1, "id": pack_id, "version": "1.0.0", "requires_core": f">={__version__}", "title": title,
           "languages": languages, "scripts": [script] if latin else [script, "Latin"],
           "detect": {"osd_scripts": [script], "langid": {"codes": [tag]}},
           "readers": {"default": default}, "normalize": ["nfc"], "repairs": [],
           "fills": [{"name": "numbers", "version": 1}, {"name": "ids", "version": 1}],
           "models": [model], "fonts": fonts, "must_survive": _parse_must(args.must_survive),
           "bench": {"kit": B.KIT_VERSION, "corpus": "none", "platforms": {}}}
    problems = M.check(raw, base=B.base_manifest(packs, pack_id))
    if problems:
        raise KitFailed("kit_invalid_draft", f"the draft wouldn't load: {'; '.join(problems)}", problems=problems)
    pdir.mkdir(parents=True, exist_ok=True)
    B.write_json(pdir / B.DRAFT_FILE, raw)
    notice = _sync_notice(packs, pdir, raw)
    ofl_copied = _copy_notice(packs, pdir, "OFL.txt") if fonts else False
    if not (pdir / C.CORPUS_FILE).exists():
        C.save(C.empty(), pdir / C.CORPUS_FILE)
    vision = []
    try:
        from ..ocr import engines

        found = [list(c["languages"]) for c in engines.get("vision").candidate_params(tag, script, [])]
        vision = [list(x) for x in dict.fromkeys(tuple(x) for x in found)]   # one per list, not per parameter set
    except Exception:
        vision = []
    return {"draft": str(pdir / B.DRAFT_FILE), "model": {k: model[k] for k in ("file", "bytes", "sha256", "source")},
            "notice": notice, "vision_languages": vision,
            "next": _init_next(pdir, pack_id, raw, ofl_copied=ofl_copied)}


# --------------------------------------------------------------------------- corpus

def _pages(spec: str | None, current: dict[str, int]) -> dict[str, int]:
    from . import corpus as C

    out = dict(current)
    if not spec:
        return out
    for part in spec.split(","):
        k, _, v = part.partition("=")
        if k.strip() not in C.PAGE_KINDS or not v.strip().isdigit():
            raise UsageError("kit_usage", f"--pages takes KIND=N,... with kinds {', '.join(C.PAGE_KINDS)}")
        out[k.strip()] = int(v)
    return out


def cmd_corpus(args: argparse.Namespace) -> dict[str, Any]:
    from . import bench as B
    from . import corpus as C

    packs, _work, pack_id = _ctx(args)
    pdir = B.pack_dir(packs, pack_id)
    m, _raw, _path = B.load_manifest(packs, pack_id)
    tag = args.lang or next((x["tag"] for x in m.languages if x.get("role") == "primary"), pack_id)
    cfile = pdir / C.CORPUS_FILE
    c = C.load(cfile) if cfile.is_file() else C.empty()
    texts = list(c.texts)
    fonts = list(c.fonts)
    added: list[str] = []
    if args.wikipedia:
        items = [x.strip() for x in args.wikipedia.split(",") if x.strip()]
        for t in C.pin_wikipedia(tag, items):
            if all(t.url != x.url for x in texts):
                texts.append(t)
                added.append(t.url)
    if args.url:
        if not args.licence or not args.title:
            raise UsageError("kit_usage", "--url needs --licence and --title")
        t = C.pin_url(args.url, lang=tag, title=args.title, licence=args.licence)
        if all(t.url != x.url for x in texts):
            texts.append(t)
            added.append(t.url)
    for furl in args.font or ():
        if all(f["url"] != furl for f in fonts):
            fonts.append(C.pin_font(furl))
            added.append(furl)
    new = C.Corpus(args.corpus_id or c.corpus_id, tuple(texts), tuple(fonts), _pages(args.pages, dict(c.pages)))
    C.save(new, cfile)
    return {"corpus": str(cfile), "added": added, "texts": len(texts), "fonts": len(fonts),
            "pages": dict(new.pages), "ref": C.corpus_ref(new),
            # Every pinned text with its title, so each Wikidata id can be checked against the
            # article it led to without opening corpus.json (kit trial es, item 9).
            "pinned": [{"title": t.title, "lang": t.lang, "revid": t.revid, "licence": t.licence, "url": t.url}
                       for t in texts]}


# --------------------------------------------------------------------------- render

def cmd_render(args: argparse.Namespace) -> dict[str, Any]:
    from . import bench as B
    from . import corpus as C
    from . import render as R

    packs, work, pack_id = _ctx(args)
    m, _raw, _path = B.load_manifest(packs, pack_id)
    c = C.load(B.pack_dir(packs, pack_id) / C.CORPUS_FILE)
    if not c.texts:
        raise UsageError("kit_corpus_empty", "corpus.json pins no texts yet; run `pdfintake kit corpus` first")
    if not c.fonts:
        raise UsageError("kit_corpus_empty", "corpus.json pins no fonts yet; run `pdfintake kit corpus --font URL`")
    fonts = [C.fetch_font(f, work / pack_id / "corpus-fonts") for f in c.fonts]
    out = B.pages_dir(work, pack_id)
    pages = R.render_pages(c, fonts, out, seed=R.BENCH_SEED, cache=work / pack_id / "texts", scripts=m.scripts)
    kinds: dict[str, int] = {}
    for p in pages:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    return {"folder": str(out), "pages": len(pages), "by_kind": kinds, "conditions": dict(R.CONDITIONS),
            "corpus": C.corpus_ref(c)}


# --------------------------------------------------------------------------- candidates and bench

def _candidates_file(work: Path, pack_id: str, key: str) -> Path:
    return work / pack_id / f"candidates-{key}.json"


def cmd_candidates(args: argparse.Namespace) -> dict[str, Any]:
    from . import bench as B

    packs, work, pack_id = _ctx(args)
    key = _platform(args)
    m, raw, _path = B.load_manifest(packs, pack_id)
    base_m = B.base_manifest(packs, pack_id)
    sets, info = B.candidate_sets(m, key, base_m, raw=raw)
    f = _candidates_file(work, pack_id, key)
    f.parent.mkdir(parents=True, exist_ok=True)
    B.write_json(f, {"platform": key, "sets": sets, "candidates": info})
    # Each set's uses with their parameters: a use id with a number (tesseract2) is the same engine
    # with its next parameter set, and set 0 is the draft's own set with its own ids (kit trial es, item 11).
    # ``uses`` labels each use with its model (tesseract[vie] against tesseract[eng]: the bare id names
    # another model in set 0 than in the others; kit trials fr, it, pt, pl and vi).
    merged = info.get("merged_sets") or {}
    return {"platform": key, "file": str(f), "count": len(sets), "candidates": info,
            "sets": [{"index": i, "engines": [u["id"] for u in s["engines"]],
                      "uses": [B.use_label(u) for u in s["engines"]], "pick": s["pick"]["rule"],
                      "params": {u["id"]: u["params"] for u in s["engines"]},
                      **({"merged": merged[str(i)]} if str(i) in merged else {})}
                     for i, s in enumerate(sets)]}


def cmd_bench(args: argparse.Namespace) -> dict[str, Any]:
    from . import bench as B
    from . import corpus as C
    from . import render as R
    from .choose import _floor_problems, set_score

    packs, work, pack_id = _ctx(args)
    key = _platform(args)
    m, raw, _path, pack, base = _draft_pack(packs, work, pack_id, offline=args.offline)
    base_m = B.base_manifest(packs, pack_id)
    pages = R.load_pages(B.pages_dir(work, pack_id))
    f = _candidates_file(work, pack_id, key)
    info: dict[str, Any] | None = None
    if args.current_only:
        jsons = [B.reader_set_json(m.readers_for(key))]
    elif f.is_file():
        listed = B.read_json(f, code="kit_candidates_invalid", what="the candidates list")
        if not isinstance(listed, dict) or not isinstance(listed.get("sets") or [], list):
            raise B.KitFileError("kit_candidates_invalid", f"{f} isn't a candidates list; run `pdfintake kit candidates` again")
        jsons = list(listed.get("sets") or [])
        info = listed.get("candidates") if isinstance(listed.get("candidates"), dict) else None
        if info is None or not isinstance(info.get("merged_sets"), dict):
            # Made by a kit that didn't merge duplicate sets: it would bench a set twice.
            raise UsageError("kit_candidates_stale", f"{f} was made by an older kit; run `pdfintake kit candidates "
                                                     f"{pack_id}` again, then bench")
    else:
        jsons, info = B.candidate_sets(m, key, base_m, raw=raw)
    listed_merged = (info or {}).get("merged_sets") or {}
    sets = []
    merged: dict[int, list[int]] = {}
    for i, s in enumerate(jsons):
        try:
            sets.append(B.parse_reader_set(raw, s, key, base_m))
        except (PackError, TypeError, AttributeError):
            continue
        if listed_merged.get(str(i)):
            merged[len(sets) - 1] = [int(x) for x in listed_merged[str(i)]]
    if not sets:
        raise KitFailed("kit_no_candidates", f"no candidate reader set loads on {key}")
    result = B.bench(pages, pack, base, key, sets, workers=args.workers, progress=_progress(len(pages)),
                     merged=merged)
    pdir = B.pack_dir(packs, pack_id)
    bfile = pdir / B.BENCH_FILE
    old = B.read_json(bfile, code="kit_bench_invalid", what="the bench results") if bfile.is_file() else {}
    if not isinstance(old, dict):
        raise B.KitFileError("kit_bench_invalid", f"{bfile} must hold a JSON object")
    cfile = pdir / C.CORPUS_FILE
    corpus_ref = C.corpus_ref(C.load(cfile)) if cfile.is_file() else "none"
    if old.get("corpus") not in (None, corpus_ref) or old.get("kit") not in (None, B.KIT_VERSION):
        old = {}                     # a new corpus or another kit: older platforms' numbers no longer compare
    plats = dict(old.get("platforms") or {})
    plats[key] = B.result_json(result)
    data: dict[str, Any] = {"kit": B.KIT_VERSION, "corpus": corpus_ref, "platforms": plats}
    if args.sweep:
        sweeps = dict(old.get("mixed_script_sweep") or {})
        sweeps[key] = B.mixed_script_sweep(pages, base, platform_key=key, corpus_ref=corpus_ref)
        data["mixed_script_sweep"] = sweeps
    B.write_json(bfile, data)

    def floors(s: dict[str, Any]) -> dict[str, Any]:
        if not s.get("ok"):
            return {"floors_ok": None}
        below = _floor_problems(s, m.must_survive)
        return {"floors_ok": not below, **({"below_floor": below} if below else {})}

    return {"platform": key, "bench": str(bfile), "pages": len(pages), "candidates": info,
            "sets": [{"index": s["index"], "ok": s.get("ok"), "engines": s["engines"],
                      "uses": [B.use_label(u) for u in (s.get("reader_set") or {}).get("engines") or ()],
                      "pick": ((s.get("reader_set") or {}).get("pick") or {}).get("rule"), "score": set_score(s),
                      **floors(s), "worst_page": s.get("worst_page"), "seconds_per_page": s.get("seconds_per_page"),
                      **({"merged": s["merged"]} if s.get("merged") else {}),
                      **({"error": {**s["error"], "message": result.messages.get(s["index"], "")}}
                         if s.get("error") else {})}
                     for s in result.sets]}


def progress_line(k: int, total: int, rs: Any, entry: dict[str, Any], seconds: float, n_pages: int,
                  left_s: float) -> str:
    """One ``kit bench`` progress line. ``set k`` is the set's ``index`` in the JSON and ``bench.json``
    (counted from 0), ``(k+1 of total)`` how far the bench is; each use is labelled with its model."""
    from .bench import set_label

    state = "ok" if entry.get("ok") else f"failed ({(entry.get('error') or {}).get('code')})"
    return (f"kit bench: set {k} ({k + 1} of {total}): {set_label(rs)}, {rs.pick.rule}; {state}, "
            f"{seconds:.0f} s for {n_pages} pages; about {left_s / 60:.0f} min left")


def _progress(n_pages: int):
    """``kit bench``'s progress: one line per set on standard error (standard output stays one JSON
    object), so a long bench shows it's alive."""
    import time

    started = time.monotonic()

    def report(k: int, total: int, rs: Any, entry: dict[str, Any], seconds: float) -> None:
        elapsed = time.monotonic() - started
        left = elapsed / (k + 1) * (total - k - 1)
        sys.stderr.write(progress_line(k, total, rs, entry, seconds, n_pages, left) + "\n")
        sys.stderr.flush()

    return report


# --------------------------------------------------------------------------- choose

def cmd_choose(args: argparse.Namespace) -> dict[str, Any]:
    from ..langpacks import manifest as M
    from . import bench as B
    from . import choose as CH
    from .score import MIN_MEASURABLE

    packs, _work, pack_id = _ctx(args)
    pdir = B.pack_dir(packs, pack_id)
    m, raw, _path = B.load_manifest(packs, pack_id)
    bfile = pdir / B.BENCH_FILE
    if not bfile.is_file():
        raise UsageError("kit_no_bench", f"{bfile} is missing; run `pdfintake kit bench` first")
    data = B.read_json(bfile, code="kit_bench_invalid", what="the bench results")
    if not isinstance(data, dict) or not isinstance(data.get("platforms") or {}, dict):
        raise B.KitFileError("kit_bench_invalid", f"{bfile} isn't a kit bench file; run `pdfintake kit bench` again")
    results = {k: B.result_from_json(v) for k, v in (data.get("platforms") or {}).items()}
    if not results:
        raise UsageError("kit_no_bench", "bench.json holds no platform yet")
    if data.get("kit") != B.KIT_VERSION:
        # Kit 1.0 counted must-survive characters on pages no engine read and benched duplicate sets.
        raise UsageError("kit_bench_stale", f"bench.json was made by kit {data.get('kit')}, and this kit "
                                            f"({B.KIT_VERSION}) measures differently; run `pdfintake kit bench "
                                            f"{pack_id}` again (after `kit candidates`), then choose")
    for plat, r in sorted(results.items()):
        # A must-survive group the bench never measured (named after it ran) would pass every floor
        # unmeasured, so the bench is stale, not passed. A group measured as None (the corpus has
        # none of its characters) is a different matter, and stays the policy's.
        ran = [s for s in r.sets if s.get("ok")]
        stale = [ms.name for ms in m.must_survive if ran and not any(ms.name in (s.get("must") or {}) for s in ran)]
        if stale:
            raise UsageError("kit_bench_stale", f"the {plat} bench has no numbers for the must-survive group(s) "
                                                f"{', '.join(stale)}; run `pdfintake kit bench {pack_id}` again")
    overrides: dict[str, int] = {}
    for spec in args.override or ():
        plat, _, idx = spec.partition("=")
        if plat not in results or not idx.isdigit():
            raise UsageError("kit_usage", f"--override {spec!r}: use PLATFORM=SET_INDEX for a benched platform")
        overrides[plat] = int(idx)
    if overrides and not (args.override_reason or "").strip():
        raise UsageError("kit_override_reason", "an override needs --override-reason")
    selections = {p: CH.choose(r, m.must_survive, override=overrides.get(p),
                               override_reason=args.override_reason if p in overrides else None)
                  for p, r in results.items()}
    out = CH.write_manifest(raw, results, selections, corpus=str(data.get("corpus") or "none"))
    for plat, claim in sorted((out["bench"].get("platforms") or {}).items()):
        policy = ((claim or {}).get("selection") or {}).get("policy")
        if policy != CH.POLICY:
            # A platform the manifest claims numbers for that this bench.json doesn't cover keeps its
            # old claim: chosen under another policy, it can't stay.
            raise UsageError("kit_policy", f"the manifest keeps numbers for {plat} chosen under policy {policy!r}, and "
                                           f"this kit chooses with {CH.POLICY} only; bench.json has no {plat} numbers "
                                           f"to choose them again: bench on {plat}, then run `pdfintake kit choose "
                                           f"{pack_id}` again")
    mfile = pdir / B.MANIFEST_FILE
    kept = CH.carry_over(out, B.read_raw(mfile)[1] if mfile.is_file() else None)
    dropped = CH.prune_models(out)
    problems = M.check(out, base=B.base_manifest(packs, pack_id))
    if problems:
        raise KitFailed("kit_invalid_manifest", f"the chosen manifest wouldn't load: {'; '.join(problems)}",
                        problems=problems)
    B.write_json(mfile, out)
    notice = CH.sync_notice(pdir, out, packs)            # the notice follows the models manifest.json keeps
    nxt = list(kept["next"])
    if dropped and kept["canary"] == "kept":
        nxt.append(f"models were dropped, so the wheel changed: run `pdfintake kit wheel {pack_id}` again")
    missing = CH.unbenched(results)

    def label(p: str, i: int | None) -> str | None:
        return B.set_label(CH._set_of(results[p], i).get("reader_set") or {}) if i is not None else None

    def merged(p: str, i: int | None) -> dict[str, Any]:
        got = CH._set_of(results[p], i).get("merged") if i is not None else None
        return {"merged": list(got)} if got else {}

    warnings = []
    for p, s in sorted(selections.items()):
        for name in CH.unmeasured(CH._set_of(results[p], s.chosen), m.must_survive):
            warnings.append(f"{p}: the must-survive group {name} couldn't be measured in any condition (no condition "
                            f"had {MIN_MEASURABLE} of its characters inside words or spans the chosen set read), so "
                            "its floor wasn't checked")

    def selection(p: str, s: Any) -> dict[str, Any]:
        chosen = CH._set_of(results[p], s.chosen)
        out_ = {"chosen": s.chosen, **{f"chosen_{k}": v for k, v in merged(p, s.chosen).items()},
                "runner_up": s.runner_up, "margin": s.margin, "policy": s.policy, "override_reason": s.override_reason,
                "chosen_set": label(p, s.chosen), "runner_up_set": label(p, s.runner_up),
                "chosen_score": s.chosen_score, "runner_up_score": s.runner_up_score,
                "ahead": [{"index": i, "set": label(p, i), "score": CH.set_score(CH._set_of(results[p], i)),
                           **merged(p, i)} for i in s.ahead],
                "must_coverage": chosen.get("must_coverage") or {}}
        if s.tie_break:
            out_["tie_break"] = {**s.tie_break, "summary": CH.tie_break_line(s.tie_break, lambda i: label(p, i))}
        return out_

    return {"manifest": str(mfile), **kept, "next": nxt, "dropped_models": dropped, "notice": notice,
            "unbenched": missing, "warnings": warnings,
            "selections": {p: selection(p, s) for p, s in sorted(selections.items())}}


# --------------------------------------------------------------------------- verify

def cmd_verify(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from . import bench as B
    from . import verify as V

    packs, work, pack_id = _ctx(args)
    res = V.verify_pack(B.pack_dir(packs, pack_id), download=args.download, rebench=args.rebench,
                        work=work, platform_key=_platform(args) if args.platform else None)
    code = EXIT_OK
    if not res.ok:
        # Any failed check is a plain failure; 6 and 7 only when the re-bench findings are all
        # that's wrong, so a drift never hides a broken download (decided 2026-11-05).
        if res.other_failed or not (res.floor_failed or res.drifted):
            code = EXIT_FAILED
        else:
            code = EXIT_KIT_FLOOR if res.floor_failed else EXIT_KIT_DRIFT
    return {"ok": res.ok, "problems": res.problems, "checks": res.checks, "warnings": res.warnings,
            "rebench": res.rebench}, code


# --------------------------------------------------------------------------- wheel

def cmd_wheel(args: argparse.Namespace) -> dict[str, Any]:
    from .. import datawheel
    from ..integrity import sha256_file
    from ..langpacks import canary as K
    from ..langpacks import manifest as M
    from . import bench as B
    from . import render as R

    packs, work, pack_id = _ctx(args)
    pdir = B.pack_dir(packs, pack_id)
    if args.must_find:
        _check_must_find(args.must_find)                 # a bad argument: exit 2 before any download or render
    check_canary_env(repo_root(packs))                   # exit 2 as well, before anything is drawn
    m, raw, _path, pack, base = _draft_pack(packs, work, pack_id, offline=args.offline, draft=False)
    files = B.files_dir(work, pack_id)
    canary = raw.get("canary") or {}
    must_find = list(args.must_find or canary.get("must_find") or [])
    if not must_find:
        raise UsageError("kit_usage", "the pack has no canary yet: pass --must-find WORD (a word of the "
                                      "language with its special characters) and --must-find 4521")
    _check_must_find(must_find)
    dpi = int(canary.get("dpi") or 300)
    fonts = [pack.asset_path(f.key) for f in m.fonts] + [base.asset_path(f.key) for f in base.manifest.fonts
                                                         if base is not pack]
    name = f"canary-{pack_id}.png"
    png = R.render_canary(" ".join(must_find), fonts, files / name, dpi=dpi)
    raw["canary"] = {"file": name, "bytes": png.stat().st_size, "sha256": sha256_file(png), "dpi": dpi,
                     "must_find": must_find}
    problems = M.check(raw, base=B.base_manifest(packs, pack_id))
    if problems:
        raise KitFailed("kit_invalid_manifest", f"the manifest with its canary wouldn't load: {'; '.join(problems)}",
                        problems=problems)
    mfile = pdir / B.MANIFEST_FILE
    before = mfile.read_bytes()
    # The canary is read from the manifest on disk, so the pin is written first. Any failure puts
    # the manifest back byte for byte and leaves no wheel: a pin no reader can read never stays.
    B.write_json(mfile, raw)
    try:
        m2, _raw2, _p = B.load_manifest(packs, pack_id, draft=False)
        pack2 = B.installed_pack(m2, files, work / pack_id / "pack")
        base2 = pack2 if m2.base else base
        reads: list[dict[str, Any]] = []
        failed: list[str] = []
        for key in _canary_keys(m2, _platform(args)):
            uses = {u.id: B.use_label(u) for u in m2.readers_for(key).engines}
            for r in K.check(pack2, base2, key):
                label = uses.get(r.use_id, r.use_id)
                reads.append({"platform": key, "use": r.use_id, "label": label, "ok": r.ok, "found": r.found,
                              "missing": r.missing})
                if not r.ok:
                    failed.append(f"{label} ({key})")
        if failed:
            raise KitFailed("kit_canary_failed", "the canary isn't read here by: " + ", ".join(failed),
                            canary_reads=reads)
        out_dir = Path(args.out) if args.out else work / pack_id / "dist"
        try:
            whl = datawheel.build(pdir, files, out_dir, dist_name=datawheel._dist_name(raw))
        except datawheel.DataWheelError as exc:
            raise KitFailed("kit_wheel", str(exc)) from None
    except BaseException:
        tmp = mfile.with_name(mfile.name + ".tmp")
        tmp.write_bytes(before)
        tmp.replace(mfile)
        raise
    return {"wheel": str(whl), "bytes": whl.stat().st_size, "sha256": sha256_file(whl), "canary": raw["canary"],
            "canary_reads": reads}


def _canary_keys(m: Any, here: str) -> list[str]:
    """The platform keys ``kit wheel`` reads the canary with: ``here`` first, then one key for every
    other reader set the pack names (a platform's own entry, and ``default`` through a platform that
    has none). A run reads the canary with its platform's set (``store.ensure``), so a canary one set
    can't read breaks the pack there; an engine that can't run on this computer is skipped, not
    failed (``canary.check``)."""
    from .. import platforms

    keys, seen = [here], [m.readers_for(here)]
    for p in platforms.PLATFORMS:
        rs = m.readers_for(p.key)
        if not any(rs is s for s in seen):
            keys.append(p.key)
            seen.append(rs)
    return keys


CANARY_ENV = Path("tools") / "canary-env.txt"
# What kit/render.py draws a canary with; scripts/dev-install.sh and pack-publish.yml draw every
# canary again in an environment built from tools/canary-env.txt, at exactly those versions.
CANARY_LIBS = ("fpdf2", "uharfbuzz", "Pillow", "pypdfium2", "pikepdf", "fonttools")


def repo_root(packs: Path) -> Path:
    """The checkout the kit works in: the folder holding ``--packs`` (``packs/`` sits at its top)."""
    return Path(packs).resolve().parent


def _dist_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([0-9][A-Za-z0-9.+!-]*)")


def canary_env_pins(path: Path) -> dict[str, str]:
    """``tools/canary-env.txt``: one ``name==version`` per line, ``#`` comments allowed; names
    normalised (case, ``-``, ``_`` and ``.`` alike). UsageError when it is missing, or when a line
    isn't an exact pin, names a library the canary isn't drawn with, or pins one a second time:
    ``scripts/dev-install.sh`` refuses the same files, so a file this accepts it accepts too."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise UsageError("kit_canary_env", f"{path} is missing. kit wheel pins the canary it draws, and "
                                           "dev-install.sh and pack-publish draw it again with the library versions "
                                           "that file names, so without it the pin can't be checked; it comes with "
                                           "the repository (pull, or restore it with git)") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError("kit_canary_env", f"{path} isn't readable: {exc}") from None
    pins: dict[str, str] = {}
    known = {_dist_key(lib): lib for lib in CANARY_LIBS}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _PIN.fullmatch(line)
        if not m:
            raise UsageError("kit_canary_env", f"{path} line {n} isn't name==version: {line!r}")
        key = _dist_key(m.group(1))
        if key not in known:
            raise UsageError("kit_canary_env", f"{path} line {n}: {m.group(1)} isn't a library the canary is "
                                               f"drawn with ({', '.join(CANARY_LIBS)})")
        if key in pins:
            raise UsageError("kit_canary_env", f"{path} pins {known[key]} twice (line {n} again)")
        pins[key] = m.group(2)
    return pins


def check_canary_env(root: Path, *, versions: Any = None) -> None:
    """Refuse (UsageError, exit 2) unless the canary libraries running here are at exactly the
    versions ``root/tools/canary-env.txt`` pins: a canary drawn with other versions can come out
    with other bytes, and then no fresh clone or release build reproduces the pin. ``versions``
    (a name -> version function) replaces ``importlib.metadata.version`` in tests."""
    from importlib import metadata

    path = Path(root) / CANARY_ENV
    pins = canary_env_pins(path)
    lookup = versions or metadata.version
    diffs = []
    for lib in CANARY_LIBS:
        want = pins.get(_dist_key(lib))
        try:
            got = lookup(lib)
        except metadata.PackageNotFoundError:
            got = None
        if want is None:
            diffs.append(f"{lib} isn't pinned there")
        elif got is None:
            diffs.append(f"{lib} isn't installed ({want} pinned)")
        elif str(got) != want:
            diffs.append(f"{lib} is {got} here, {want} pinned")
    if diffs:
        raise UsageError("kit_canary_env", f"kit wheel draws the canary with the versions {CANARY_ENV.as_posix()} pins, "
                                           f"because dev-install.sh and pack-publish draw it again with them: "
                                           f"{'; '.join(diffs)}. Install those versions into the environment you "
                                           f"run the kit from (`uv pip install -r {CANARY_ENV.as_posix()}`), then "
                                           "run kit wheel again", differs=diffs)


def _must_find_limits() -> tuple[int, int]:
    """The canary schema's limits on ``must_find``: (most entries, most characters per entry)."""
    from ..langpacks import manifest as M

    spec = ((M._manifest_schema().root.get("$defs") or {}).get("canary") or {}).get("properties", {}).get("must_find", {})
    return int(spec.get("maxItems", 4)), int((spec.get("items") or {}).get("maxLength", 40))


def _check_must_find(must_find: Sequence[str]) -> None:
    """``--must-find`` within the canary schema, checked before anything is rendered (exit 2)."""
    most, longest = _must_find_limits()
    if len(must_find) > most:
        raise UsageError("kit_usage", f"--must-find takes at most {most} entries, 4521 included "
                                      f"({len(must_find)} given): keep the words that carry the most special characters")
    for w in must_find:
        if not w.strip() or len(w) > longest:
            raise UsageError("kit_usage", f"--must-find {w!r}: each entry is 1 to {longest} characters, not blank")


# --------------------------------------------------------------------------- private and report

def cmd_private(args: argparse.Namespace) -> dict[str, Any]:
    from . import bench as B
    from . import private as P

    packs, work, pack_id = _ctx(args)
    m, raw, _path, pack, base = _draft_pack(packs, work, pack_id, offline=args.offline)
    agg = P.private_bench(Path(args.folder), pack, base, _platform(args))
    pdir = B.pack_dir(packs, pack_id)
    written = P.write_private(pdir / B.PRIVATE_FILE, agg, m)
    mfile = pdir / B.MANIFEST_FILE
    if mfile.is_file():
        _d, mraw = B.read_raw(mfile)
        mraw.setdefault("bench", {})["private"] = written
        B.write_json(mfile, mraw)
    return {"file": str(pdir / B.PRIVATE_FILE), **written}


def cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    from . import bench as B
    from .report import pack_files, pr_body

    packs, _work, pack_id = _ctx(args)
    pdir = B.pack_dir(packs, pack_id)
    body = pr_body(pdir)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
    files = pack_files(pdir)
    return {"body": body, "commit": [f"packs/{pack_id}/{n}" for n in files["commit"]],
            "leave_out": [f"packs/{pack_id}/{n}" for n in files["leave_out"]],
            **({"missing": [f"packs/{pack_id}/{n}" for n in files["missing"]]} if files["missing"] else {}),
            **({"file": str(args.out)} if args.out else {})}


# --------------------------------------------------------------------------- the parser

COMMANDS = {"init": cmd_init, "corpus": cmd_corpus, "render": cmd_render, "candidates": cmd_candidates,
            "bench": cmd_bench, "choose": cmd_choose, "verify": cmd_verify, "wheel": cmd_wheel,
            "private": cmd_private, "report": cmd_report}


def run(args: argparse.Namespace) -> int:
    """Run the parsed kit command; print its JSON; return its exit code."""
    name = args.kit_command
    try:
        got = COMMANDS[name](args)
        obj, code = got if isinstance(got, tuple) else (got, EXIT_OK)
        obj = {"ok": code == EXIT_OK, "command": f"kit {name}", **obj}
        _emit(obj)
        return code
    except PdfIntakeError as exc:
        _emit({"ok": False, "command": f"kit {name}",
               "error": {"code": exc.code, "message": exc.message,
                         **{k: exc.detail[k] for k in ("problems", "closest", "canary_reads")
                            if isinstance(exc.detail.get(k), list)}}})
        return exc.exit_code
    except Exception as exc:                          # a kit bug or an unreadable file: still one JSON object
        _emit({"ok": False, "command": f"kit {name}",
               "error": {"code": "kit_internal", "message": f"{type(exc).__name__}: {exc}"}})
        return EXIT_FAILED


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("pack", help="the pack id, such as es")
    p.add_argument("--packs", type=Path, default=Path("packs"), help="the folder holding the pack folders (default ./packs)")
    p.add_argument("--work", type=Path, default=None, help="the kit work folder (default <cache>/kit)")
    p.add_argument("--json", action="store_true", help="print JSON (the output is JSON either way)")


def register(subparsers: "argparse._SubParsersAction") -> None:
    """Add ``kit`` and its commands; each sets ``func=run``."""
    kit = subparsers.add_parser("kit", help="build and check a language pack (needs pdfintake[kit])",
                                description="The add-a-language kit; see docs/add-a-language.md.")
    sub = kit.add_subparsers(dest="kit_command", required=True)

    p = sub.add_parser("init", help="start packs/<id>/manifest.draft.json and corpus.json")
    _common(p)
    p.add_argument("--script", help="the language's script, such as Latin or Thai (required unless --from-manifest)")
    p.add_argument("--iso639-3", dest="iso639_3", help="Tesseract's model name when the kit doesn't know the language")
    p.add_argument("--title", help="the language's English name when the kit doesn't know it")
    p.add_argument("--tessdata-commit", default=TESSDATA_BEST_COMMIT, help="the tessdata_best commit to pin")
    p.add_argument("--must-survive", action="append", default=[], metavar="NAME=CHARS[:UNIT[:FLOOR]]")
    p.add_argument("--font", action="append", default=[], metavar="URL", help="a layer font (non-Latin scripts)")
    p.add_argument("--force", action="store_true", help="start the draft again")
    p.add_argument("--from-manifest", action="store_true",
                   help="rebuild manifest.draft.json from manifest.json (and re-pin the language's Tesseract model)")

    p = sub.add_parser("corpus", help="pin corpus texts and fonts in packs/<id>/corpus.json")
    _common(p)
    p.add_argument("--wikipedia", metavar="Q1,Q2", help="Wikidata items whose article in the language to pin")
    p.add_argument("--url", help="a plain-text file under an allowed licence")
    p.add_argument("--licence", help="that file's licence")
    p.add_argument("--title", help="that file's title")
    p.add_argument("--lang", help="the language code of the texts (default the pack's primary language)")
    p.add_argument("--font", action="append", default=[], metavar="URL", help="a notofonts file to render with")
    p.add_argument("--pages", metavar="KIND=N,...", help="page counts per kind")
    p.add_argument("--corpus-id", help="the corpus id (default public-v1)")

    p = sub.add_parser("render", help="render the bench pages into the work folder")
    _common(p)

    for name, helptext in (("candidates", "list the candidate reader sets for this platform"),
                           ("bench", "bench the candidate reader sets on the rendered pages")):
        p = sub.add_parser(name, help=helptext)
        _common(p)
        p.add_argument("--platform", help="the platform key (default this machine's)")
        if name == "bench":
            p.add_argument("--offline", action="store_true", help="never download; the files must be in the work folder")
            p.add_argument("--workers", type=int, default=None)
            p.add_argument("--current-only", action="store_true", help="bench only the draft's own reader set")
            p.add_argument("--sweep", action="store_true", help="also measure the mixed-script trigger")

    p = sub.add_parser("choose", help="pick the reader set (maximin-v2) and write manifest.json")
    _common(p)
    p.add_argument("--override", action="append", default=[], metavar="PLATFORM=INDEX")
    p.add_argument("--override-reason", help="why the kit's choice is overridden (CI flags it for review)")

    p = sub.add_parser("verify", help="check the pack: manifest, download, layer round trip, wheel; optionally re-bench")
    _common(p)
    p.add_argument("--download", action="store_true", help="download every model and font from upstream")
    p.add_argument("--rebench", action="store_true", help="re-bench and compare with the numbers the pack claims")
    p.add_argument("--platform", help="the platform key (default this machine's)")

    p = sub.add_parser("wheel", help="render the canary, pin it, build the data wheel")
    _common(p)
    p.add_argument("--must-find", action="append", default=[], metavar="TEXT",
                   help="a word every reader must find (at most 4, 4521 included)")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--platform", help="the platform key the canary is read on (default this machine's)")
    p.add_argument("--out", help="where to write the wheel (default the work folder)")

    p = sub.add_parser("private", help="bench on your own born-digital PDFs; only numbers are written")
    _common(p)
    p.add_argument("--folder", required=True)
    p.add_argument("--platform", help="the platform key (default this machine's)")
    p.add_argument("--offline", action="store_true")

    p = sub.add_parser("report", help="print the pull request body")
    _common(p)
    p.add_argument("--out", help="also write the body to this file")

    for parser in sub.choices.values():
        parser.set_defaults(func=run)


def main(argv: list[str] | None = None) -> int:
    """``python -m pdfintake.kit.cli kit <command> ...`` (the same as ``pdfintake kit ...``)."""
    from ..fsutil import utf8_stdio

    utf8_stdio()
    parser = argparse.ArgumentParser(prog="pdfintake")
    register(parser.add_subparsers(dest="command", required=True))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
