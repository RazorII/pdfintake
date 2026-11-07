"""``kit private``: a bench on the contributor's own born-digital pages, of which only numbers leave.

Optional (DESIGN §4.5 step 7). The contributor names a folder of PDFs. Only pages classed
``born_digital`` (visible text with a sound font map) are used, because their own text is the truth:
each such page is rendered at 300 dpi, read with the draft's reader set through
``pipeline.read_pages``, and scored against the text pdfium extracts from it. Nothing about a page
is kept: the renders go to a temporary folder deleted afterwards, the texts stay in memory, and
the result is a page count plus one ``[mean, min, max]`` triple per metric.

A must-survive group is counted as the bench counts it (``score``): only its characters inside
words or spans the engine read, and a page counts only with at least ``score.MIN_MEASURABLE`` of
them, so a page with one or two read characters can't make the worst or best 0 or 1
(``must_triple``). When no page has that many, the pages are added up into one figure, given as
the whole triple, or no triple when even all of them together have fewer.

**The privacy guard.** ``write_private`` writes ``packs/<id>/bench.private.json`` only through
``guard``, a schema that allows no string anywhere: the keys are the fixed metric names (``words``,
``trigrams``, ``digits``, ``must-<name>`` for the pack's own must-survive names) and every value
is a number. Anything else trips the guard (kit exit 8) and nothing is written.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import EXIT_KIT_PRIVACY, PdfIntakeError, UsageError

if TYPE_CHECKING:
    from ..langpacks.manifest import Manifest
    from ..langpacks.store import InstalledPack

BASE_METRICS = ("words", "trigrams", "digits")
RENDER_DPI = 300.0
MAX_PAGES = 500


class PrivacyGuard(PdfIntakeError):
    """``kit private`` would have written something other than numbers (kit exit 8)."""

    exit_code = EXIT_KIT_PRIVACY


@dataclass
class PrivateAggregate:
    pages: int
    metrics: dict[str, list[float]]


def allowed_metrics(draft: "Manifest") -> frozenset[str]:
    return frozenset(BASE_METRICS) | {f"must-{ms.name}" for ms in draft.must_survive}


def guard(agg: Any, allowed: frozenset[str]) -> dict[str, Any]:
    """The aggregate as JSON, or PrivacyGuard: a page count and number triples under known names only."""
    problems: list[str] = []
    pages = getattr(agg, "pages", None)
    metrics = getattr(agg, "metrics", None)
    if isinstance(pages, bool) or not isinstance(pages, int) or pages < 1:
        problems.append("pages must be a whole number of at least 1")
    if not isinstance(metrics, Mapping):
        problems.append("metrics must be an object")
        metrics = {}
    out: dict[str, list[float]] = {}
    for name, triple in metrics.items():
        if not isinstance(name, str) or name not in allowed:
            problems.append("a metric name isn't one of the kit's fixed names")
            continue
        if (not isinstance(triple, (list, tuple)) or len(triple) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       or not 0.0 <= v <= 1.0 for v in triple)):
            problems.append(f"{name} must be three numbers from 0 to 1")
            continue
        out[name] = [round(float(v), 4) for v in triple]
    if problems:
        # The message never repeats what tripped the guard: it may be the very text that must not leave.
        raise PrivacyGuard("kit_privacy", f"the private result held something other than numbers "
                                          f"({len(problems)} problem{'s' if len(problems) != 1 else ''}); nothing was written",
                           count=len(problems))
    return {"pages": int(pages), "metrics": out}


def write_private(path: Path, agg: PrivateAggregate, draft: "Manifest") -> dict[str, Any]:
    """Write ``bench.private.json`` through the guard; return what was written."""
    data = guard(agg, allowed_metrics(draft))
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return data


def _page_text(pdf: Path, index: int) -> str:
    import pypdfium2

    from ..pdf.render import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pypdfium2.PdfDocument(str(pdf))
        try:
            page = doc[index]
            try:
                tp = page.get_textpage()
                try:
                    return tp.get_text_range()
                finally:
                    tp.close()
            finally:
                page.close()
        finally:
            doc.close()


def _triple(vals: Sequence[float | None]) -> list[float] | None:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return [round(statistics.fmean(xs), 4), round(min(xs), 4), round(max(xs), 4)]


def must_triple(scores: Sequence[Any], name: str) -> list[float] | None:
    """One must-survive group's ``[mean, min, max]`` over the pages with at least ``MIN_MEASURABLE``
    of its characters inside read words or spans (each page's survived / read); when no page has
    that many, the pages added up as ``[f, f, f]``; None when even those together have fewer."""
    from .score import MIN_MEASURABLE

    counts = [c for c in ((getattr(s, "must_counts", None) or {}).get(name) for s in scores) if c]
    own = [c[2] / c[1] for c in counts if c[1] >= MIN_MEASURABLE]
    if own:
        return _triple(own)
    read, survived = sum(c[1] for c in counts), sum(c[2] for c in counts)
    if read < MIN_MEASURABLE:
        return None
    f = round(survived / read, 4)
    return [f, f, f]


def private_bench(folder: Path, draft: "Manifest | InstalledPack", base: "InstalledPack", platform_key: str,
                  *, max_pages: int = MAX_PAGES) -> PrivateAggregate:
    """Bench the draft's reader set for ``platform_key`` on the born-digital pages of the PDFs in
    ``folder`` (not its subfolders); only the aggregate comes back."""
    import dataclasses

    from ..ocr import pipeline
    from ..pdf.classify import classify
    from ..pdf.render import Renderer
    from .bench import _as_pack, _plans
    from .render import BenchPage
    from .score import score_page

    folder = Path(folder)
    if not folder.is_dir():
        raise UsageError("kit_usage", f"{folder} isn't a folder")
    pack = _as_pack(draft)
    m = pack.manifest
    script = m.scripts[0]
    rs = m.readers_for(platform_key)
    from ..langpacks.store import InstalledPack

    variant = InstalledPack(dataclasses.replace(m, readers={"default": rs, platform_key: rs}), pack.root)
    scores = []
    tmp = Path(tempfile.mkdtemp(prefix="pdfintake-kit-private-"))
    try:
        pages: list[BenchPage] = []
        images = []
        for pdf in sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf" and p.is_file()):
            try:
                classes = classify(pdf, redo_ocr=False, keep_existing_layer=True)
            except Exception:
                continue                                  # unreadable or encrypted: not a bench page
            with Renderer(pdf) as r:
                for c in classes:
                    if c.kind != "born_digital" or len(pages) >= max_pages:
                        continue
                    truth = _page_text(pdf, c.index)
                    if not truth.strip():
                        continue
                    i = len(pages)
                    img = r.render(c.index, RENDER_DPI, tmp / f"{i}.png")
                    images.append(dataclasses.replace(img, index=i))
                    pages.append(BenchPage(str(i), "clean", "prose", img.path, truth, img.dpi))
        if not pages:
            raise UsageError("kit_private_empty", f"{folder} holds no born-digital PDF pages to bench")
        base_used = variant if m.base else base
        ocr = pipeline.read_pages(images, _plans(len(images), m), {m.id: variant}, base_used, platform_key,
                                  signals=None, workers=min(4, os.cpu_count() or 1))
        scores = [score_page(p, o, True, m.must_survive, script) for p, o in zip(pages, ocr)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    metrics: dict[str, list[float]] = {}
    for name in BASE_METRICS:
        t = _triple([getattr(s, name) for s in scores])
        if t is not None:
            metrics[name] = t
    for ms in m.must_survive:
        t = must_triple(scores, ms.name)
        if t is not None:
            metrics[f"must-{ms.name}"] = t
    return PrivateAggregate(len(scores), metrics)
