"""``pdfintake text``: what a PDF's text layer says, page by page (DESIGN §16).

A caller that indexes or searches PDFs reads their text here instead of extracting it itself, so it
gets exactly the layer pdfintake verified. ``read`` returns, for each page asked for:

* ``text``: what pdfium hands back from the page's CropBox, through ``verify.layer_text``, the read
  ``text_retention`` checks after every build. ``clean`` changes only line breaks to ``\\n``, pdfium's
  mark for a hyphen that ends a line back to ``-`` and a line break, a lone surrogate to U+FFFD, and
  the whole to NFC. A page without one non-space character gives ``""``.
* ``layer``: ``pdfintake`` when the text is a layer pdfintake wrote (classify's ``our_layer``),
  ``other`` for any other text (the document's own, or another tool's invisible layer), ``none``
  when the page has no text.
* ``flags``: codes from ``record.TEXT_FLAGS`` saying the text may be missing or wrong:
  ``no_text_layer`` on a page with pictures and no text, classify's ``garbled_text`` and
  ``image_text_not_read``, and, with ``--record``, ``font_map_partial`` and the page's warnings
  from that record.

Which pages have letters their fonts can't map comes from the record's ``fontmap`` step with
``--record``. Without one, ``text`` runs the check ``run`` starts its repair from
(``fontmap.analyse``: pdfium's own map errors, then a probe of the fonts that have them, in a work
folder of the cache removed at once), and classify flags every such page ``garbled_text``: the
file as it is reads wrongly there. A partial repair can't be told from the file alone, so
``font_map_partial`` needs the record. ``font_maps`` in the header says which source decided:
``record``, ``checked``, or ``failed`` when the check couldn't run (the flags then come from
classify's garble ratio alone).

**It never reads a page with OCR, never changes or writes a PDF, and writes no record** (decided
2026-11-17). A PDF without a text layer gives every page ``layer: "none"`` and the warning
``no_text_layer``, which points at ``pdfintake run``. An image is refused (``unsupported_input``,
exit 2): pdfintake reads only documents, and an image becomes a PDF with ``run``.

``--record ID`` names the ``run`` that wrote this PDF. Its ``output.sha256`` must be the PDF's
(StaleRecord ``record_mismatch``, exit 5, otherwise); the record's tool, packs and each OCR'd
page's languages and pack are then copied into the export, and its font-map outcome tells classify
which pages were repaired, so a partly repaired page is flagged ``font_map_partial``.

The text goes only to the caller: standard output, or a file the caller names, which ``stage``
builds in ``<cache>/staging`` and the CLI hands to ``flow.commit``, the one writer into a user's
folder. Nothing here logs or raises a page's text; errors carry codes, counts and page numbers.

A password-protected PDF is refused (``encrypted``, exit 4): ``run`` makes a searchable copy
without a password when a saved password opens it, and ``text`` reads that copy. A file pdfium
can't open is ``damaged`` (exit 4). The input is sealed first and checked again at the end, so the
text is the text of the file whose SHA-256 the export names (InputChanged, exit 5, otherwise).
"""

from __future__ import annotations

import json
import os
import re
import stat
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .. import __version__
from ..errors import InputChanged, PdfIntakeError, Refused, StaleRecord, UsageError

if TYPE_CHECKING:
    from ..integrity import InputSeal
    from ..record import Record

FORMAT = "pdfintake-text"
FORMAT_VERSION = 1
FORMATS: tuple[str, ...] = ("jsonl", "txt")
LayerKind = Literal["pdfintake", "other", "none"]
LAYER_KINDS: tuple[str, ...] = ("pdfintake", "other", "none")

_PAGES_RE = re.compile(r"^\s*([0-9]{1,9})\s*(?:-\s*([0-9]{1,9})\s*)?$")
# pdfium marks a hyphen that ends a line with U+0002 (bounded text) or U+FFFE, and joins the next
# line on without a break (the same marks verify.fonts_mapped folds); whatever break it kept is folded in.
_LINE_HYPHEN_RE = re.compile("[\x02\ufffe](?:\r\n|\r|\n)?")
_SURROGATE_RE = re.compile("[\ud800-\udfff]")
_BASIS = {"forced": "forced", "hint": "hinted"}       # LanguageGuess.by -> the text record's lang_basis


@dataclass
class PageText:
    """One page of the export. ``page`` counts from 1, as everywhere a person reads it."""

    page: int
    text: str
    layer: LayerKind
    flags: list[str] = field(default_factory=list)
    lang: list[str] | None = None
    lang_basis: str | None = None
    pack: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"page": self.page, "text": self.text, "chars": len(self.text), "layer": self.layer,
                "flags": list(self.flags), "lang": None if self.lang is None else list(self.lang),
                "lang_basis": self.lang_basis, "pack": self.pack}


@dataclass
class TextExport:
    """What ``read`` found. Everything but ``lines`` is codes, counts, hashes and page numbers."""

    source: Path
    source_sha256: str
    pages: int                          # pages in the document
    first: int                          # the range exported, from 1, inclusive
    last: int
    lines: list[PageText]
    warnings: list[str]                 # document-level warning codes (record.WARNING_CODES)
    record: dict[str, Any] | None       # what --record contributed, or None
    seal: "InputSeal"
    font_maps: str = "checked"          # record, checked or failed: what said which pages' fonts lack maps

    def layers(self) -> dict[str, int]:
        return {kind: sum(1 for p in self.lines if p.layer == kind) for kind in LAYER_KINDS}

    def flagged(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for p in self.lines:
            for code in p.flags:
                out.setdefault(code, []).append(p.page)
        return {code: out[code] for code in _ordered(out)}

    def warning_messages(self) -> list[dict[str, Any]]:
        from ..record import WARNING_MESSAGES

        return [{"code": code, "message": WARNING_MESSAGES[code], "page": None} for code in self.warnings]

    def header(self) -> dict[str, Any]:
        """The JSONL header line (contract: ``source_sha256``, ``tool_version``, ``pages``, ``producer``)."""
        return {"format": FORMAT, "format_version": FORMAT_VERSION, "source_sha256": self.source_sha256,
                "tool_version": __version__, "pages": self.pages, "range": [self.first, self.last],
                "producer": producer(), "layers": self.layers(), "font_maps": self.font_maps,
                "warnings": self.warning_messages(), "record": self.record}


# --------------------------------------------------------------------------- arguments

def parse_pages(spec: str | None) -> tuple[int, int] | None:
    """``"3"`` -> (3, 3), ``"2-5"`` -> (2, 5), None -> None. UsageError ``bad_pages`` (exit 2) otherwise.

    Pages count from 1. Whether the range fits the document is checked once it is open."""
    if spec is None:
        return None
    m = _PAGES_RE.match(str(spec))
    first = int(m.group(1)) if m else 0
    last = int(m.group(2)) if m and m.group(2) is not None else first
    if not m or first < 1 or last < first:
        raise UsageError("bad_pages", "--pages takes a page number or a range such as 2-5, counting from 1.")
    return first, last


# --------------------------------------------------------------------------- the read

def clean(raw: str) -> str:
    """pdfium's text as a caller can store it (see the module docstring); ``""`` when it holds no
    non-space character."""
    text = _LINE_HYPHEN_RE.sub("-\n", raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SURROGATE_RE.sub("\ufffd", text)
    text = unicodedata.normalize("NFC", text)
    return text if any(not ch.isspace() for ch in text) else ""


def producer() -> dict[str, Any]:
    """What made the text from the PDF's bytes, for the caller's producer key (contract §1.1)."""
    return {"tool": "pdfintake", "tool_version": __version__, "route": "layer", "engines": _pdfium_versions(),
            "packs": []}


def _pdfium_versions() -> dict[str, str]:
    try:
        from pypdfium2 import version
    except ImportError:
        return {}
    return {"pdfium": str(getattr(version, "PDFIUM_INFO", "unknown")),
            "pypdfium2": str(getattr(version, "PYPDFIUM_INFO", "unknown"))}


def check_input(path: Path) -> None:
    """UsageError unless ``path`` is a regular file holding a PDF (exit 2), before anything opens it."""
    from ..intake import sniff

    text = str(path)
    if "\x00" in text:
        shown = text.replace("\x00", "\\x00")
        raise UsageError("input_missing", f"The input {shown} can't exist: its name holds a NUL character.",
                         path=shown)
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        raise UsageError("input_missing", f"The input {path} doesn't exist or can't be read.", path=text) from None
    if not stat.S_ISREG(st.st_mode):
        raise UsageError("input_not_file", f"The input {path} isn't a regular file.", path=text)
    try:
        kind = sniff.sniff(path)                       # UsageError input_not_file for a swapped-in FIFO
    except (OSError, ValueError):
        raise UsageError("input_missing", f"The input {path} doesn't exist or can't be read.", path=text) from None
    if kind == "pdf":
        return
    if kind == "unknown":
        raise UsageError("unsupported_input", f"The input {path} isn't a PDF, so pdfintake text can't read it.",
                         path=text)
    raise UsageError("unsupported_input", f"The input {path} is an image, and pdfintake text reads only PDFs. "
                                          "pdfintake run makes a searchable PDF of it; run text on that.",
                     path=text, kind=kind)


def _page_count(path: Path) -> int:
    """The page count, or Refused: ``encrypted`` for a PDF that needs a password, ``damaged`` otherwise."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    from .render import PDFIUM_LOCK

    try:
        with PDFIUM_LOCK, open(path, "rb") as handle:
            doc = pdfium.PdfDocument(handle)
            try:
                count = len(doc)
            finally:
                doc.close()
    except OSError:
        raise InputChanged("input_missing", f"The input {path} can't be read since it was sealed.",
                           path=str(path)) from None
    except pdfium.PdfiumError as exc:
        if getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
            raise Refused("encrypted", f"{path.name} needs a password to open, so pdfintake text can't read it. "
                                       "pdfintake run makes a searchable copy without a password when a saved "
                                       "password opens it; run text on that copy.") from None
        raise _damaged(path) from None
    if count < 1:
        raise _damaged(path)
    return count


def _damaged(path: Path) -> Refused:
    return Refused("damaged", f"{path.name} can't be read as a PDF, so pdfintake text can't give its text.")


def read(path: Path, *, pages: tuple[int, int] | None = None, record_id: str | None = None) -> TextExport:
    """Read the text layer of ``path`` (an absolute path) for the pages in ``pages`` (from 1, inclusive).

    Raises UsageError (exit 2) for an input that isn't a PDF file, a bad record id or a range the
    document doesn't have; Refused (exit 4) for an encrypted or damaged PDF; StaleRecord (exit 5)
    when ``record_id``'s output isn't this file, or the file changes while it is read.
    """
    from .. import integrity
    from . import classify, verify

    path = Path(path)
    check_input(path)
    seal = integrity.InputSeal.take([path])
    sha = seal.sha256[0]
    rec = _matching_record(record_id, sha) if record_id is not None else None
    count = _page_count(path)
    first, last = (1, count) if pages is None else pages
    if last > count:
        raise UsageError("bad_pages", f"--pages asks for pages {first}-{last}, and this PDF has {count}.",
                         pages=count)
    wanted = range(first - 1, last)
    font_pages, per_page, info = _record_parts(rec) if rec is not None else (None, {}, None)
    font_maps = "record"
    if font_pages is None:
        font_pages, font_maps = _font_check(path)
    try:
        _count, raw = verify.layer_text(path, wanted)
        classes = {c.index: c for c in classify.classify(path, redo_ocr=False, keep_existing_layer=True,
                                                         font_pages=font_pages, pages=wanted)}
    except PdfIntakeError:
        raise
    except Exception as exc:
        if _unreadable_pdf(exc):
            raise _damaged(path) from None
        raise
    seal.check()

    lines: list[PageText] = []
    for i in wanted:
        text = clean(raw.get(i, ""))
        cls = classes.get(i)
        kind = cls.kind if cls is not None else "empty"
        layer: LayerKind = "none" if not text else ("pdfintake" if kind == "our_layer" else "other")
        flags: set[str] = set()
        if layer == "none" and kind == "image_only":
            flags.add("no_text_layer")
        if layer != "none" and cls is not None:
            flags.update(w for w in cls.warnings if w in ("garbled_text", "image_text_not_read"))
            if font_pages.get(i) == "partial":
                flags.add("font_map_partial")
        mine = per_page.get(i, {})
        flags.update(mine.get("flags", ()))
        lines.append(PageText(i + 1, text, layer, _ordered(flags), mine.get("lang"), mine.get("lang_basis"),
                              mine.get("pack")))
    warnings = []
    if not any(p.layer != "none" for p in lines) and any("no_text_layer" in p.flags for p in lines):
        warnings.append("no_text_layer")
    return TextExport(path, sha, count, first, last, lines, warnings, info, seal, font_maps)


def _font_check(path: Path) -> tuple[dict[int, str], str]:
    """Pages (0-based) whose visible letters need a font map they don't have, as ``unrepaired`` for
    classify, and ``checked``; ``({}, "failed")`` when the check can't run on this file."""
    import secrets

    from .. import fsutil
    from . import fontmap

    work: Path | None = None
    try:
        work = fsutil.work_dir("work", f"text-{secrets.token_hex(6)}")
        a = fontmap.analyse(path, work / "fontmap")
        need = a.needs() if a.glyphs else set()
        pages = {g.page for g in a.glyphs if g.visible and (g.font, g.code) in need}
        return {p: "unrepaired" for p in pages}, "checked"
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:                                   # noqa: BLE001 - as in run, a check that can't run is skipped
        return {}, "failed"
    finally:
        if work is not None:
            fontmap.cleanup(work)


def _unreadable_pdf(exc: BaseException) -> bool:
    """An error pikepdf or pdfium raises for a file it can't read (the rest are pdfintake's own bugs)."""
    import pikepdf
    import pypdfium2 as pdfium

    return isinstance(exc, (pikepdf.PdfError, pdfium.PdfiumError))


def _matching_record(record_id: str, sha: str) -> "Record":
    from .. import record

    rec = record.load(record_id)                       # UsageError (exit 2) for a bad or unknown id
    out = rec.output if isinstance(rec.output, Mapping) else None
    if out is None or out.get("sha256") != sha:
        raise StaleRecord("record_mismatch", f"This PDF isn't the output record {record_id} describes (its SHA-256 "
                                             "differs), so the record can't be used. Run text without --record, "
                                             "or pdfintake run again.", record_id=record_id)
    return rec


def _record_parts(rec: "Record") -> tuple[dict[int, str] | None, dict[int, dict[str, Any]], dict[str, Any]]:
    """From a matching record: classify's ``font_pages`` (0-based; None when the record's ``fontmap``
    step is missing or failed, so ``text`` checks for itself), what each OCR'd page contributes
    (``lang``, ``lang_basis``, ``pack`` and ``flags``) and the header's ``record`` object."""
    from ..record import TEXT_FLAGS

    font_pages: dict[int, str] | None = None
    for step in rec.steps or []:
        if not isinstance(step, Mapping) or step.get("step") != "fontmap" or step.get("status") == "failed":
            continue
        font_pages = {}
        detail = step.get("detail") if isinstance(step.get("detail"), Mapping) else {}
        changed = step.get("status") == "changed"
        for key, outcome in (("unrepaired", "unrepaired"), ("partial", "partial"), ("pages", "repaired")):
            for n in _page_numbers(detail.get(key)):
                font_pages[n - 1] = outcome if changed or key == "unrepaired" else "unrepaired"

    per_page: dict[int, dict[str, Any]] = {}
    for entry in rec.pages or []:
        if not isinstance(entry, Mapping) or not _page_numbers([entry.get("page")]):
            continue
        langs = [str(g["tag"]) for g in entry.get("languages") or [] if isinstance(g, Mapping) and g.get("tag")]
        basis = None
        if langs:
            first_by = next((g.get("by") for g in entry["languages"] if isinstance(g, Mapping)), None)
            basis = _BASIS.get(str(first_by), "detected")
        mine = per_page.setdefault(int(entry["page"]) - 1, {"flags": set()})
        mine.update(lang=list(dict.fromkeys(langs)) or None, lang_basis=basis,
                    pack=entry.get("pack") if isinstance(entry.get("pack"), str) else None)
        mine["flags"].update(c for c in entry.get("warnings") or [] if c in TEXT_FLAGS)
    for w in rec.warnings or []:
        if w.code in TEXT_FLAGS and _page_numbers([w.page]):
            per_page.setdefault(int(w.page) - 1, {"flags": set()})["flags"].add(w.code)   # type: ignore[arg-type]
    for mine in per_page.values():
        mine["flags"].discard("no_text_layer")         # only the PDF itself says that

    language = rec.language if isinstance(rec.language, Mapping) else {}
    packs = [{"id": p.get("id"), "version": p.get("version")} for p in language.get("packs") or []
             if isinstance(p, Mapping)]
    info = {"id": rec.id, "tool": dict(rec.tool or {}), "language": {"plan": language.get("plan"), "packs": packs}}
    return font_pages, per_page, info


def _page_numbers(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [int(v) for v in value if type(v) is int and v >= 1]


def _ordered(codes: Any) -> list[str]:
    from ..record import TEXT_FLAGS

    order = {code: n for n, code in enumerate(TEXT_FLAGS)}
    return sorted(set(codes), key=lambda c: (order.get(c, len(order)), c))


# --------------------------------------------------------------------------- the output

def jsonl_lines(export: TextExport) -> Iterator[str]:
    """The JSONL export: the header, then one line per page, each ending in ``\\n``."""
    yield json.dumps(export.header(), ensure_ascii=False) + "\n"
    for p in export.lines:
        yield json.dumps(p.to_json(), ensure_ascii=False) + "\n"


def txt_pages(export: TextExport) -> Iterator[str]:
    """The plain export, as ``pdftotext`` writes it: each page's text, ending in a line break when it
    has any, then a form feed, so page N of the range is the Nth piece of a split on ``\\f``."""
    for p in export.lines:
        yield (p.text if not p.text or p.text.endswith("\n") else p.text + "\n") + "\f"


def chunks(export: TextExport, fmt: str) -> Iterator[str]:
    if fmt == "jsonl":
        return jsonl_lines(export)
    if fmt == "txt":
        return txt_pages(export)
    raise UsageError("bad_format", f"--format is jsonl or txt, not {fmt!r}.")


def stage(export: TextExport, fmt: str, folder: Path) -> Path:
    """The export written into ``folder`` (a ``<cache>/staging`` folder) for ``flow.commit`` to copy
    to the caller's path."""
    built = Path(folder) / f"text.{fmt}"
    with open(built, "w", encoding="utf-8", newline="") as fh:
        for chunk in chunks(export, fmt):
            fh.write(chunk)
    return built


__all__ = ["FORMAT", "FORMAT_VERSION", "FORMATS", "LAYER_KINDS", "PageText", "TextExport", "check_input",
           "chunks", "clean", "parse_pages", "producer", "read", "stage"]
