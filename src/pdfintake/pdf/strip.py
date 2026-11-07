"""Remove an old invisible text layer from pages, operator by operator.

``strip_text`` drops the text-showing operators (``Tj``, ``TJ``, ``'`` and ``"``) and the font
operator ``Tf`` from a page's content and from every Form XObject that content draws with
``Do``, at any depth. Every other operator stays exactly where it was: images, paths, colours,
graphics state, marked content, and the ``BT``/``ET`` pairs themselves (a line width set inside
``BT`` survives). Nothing is rasterised and no image stream is touched.

Only pages classed ``invisible_layer`` or ``our_layer`` (``pdf.classify``) may be stripped: on
those, every piece of text is either render mode 3 or fully transparent paint that clips nothing,
so dropping it changes no pixel. Any other page raises ``ValueError`` before anything is changed,
so printed words can never be erased.

A Form XObject that something else also uses (a page that isn't stripped, an annotation's
appearance, or a tiling pattern, soft-mask group or Type 3 font on any page, none of which the
strip looks inside) is copied first: the stripped page draws the copy, and the original keeps its
text. A copy, and any form or page whose resources had to point at a copy, gets its own Resources
dictionary, so a later ``drop_unused`` can't prune a dictionary the other user still needs. A form
without Resources of its own (the PDF 1.1 style) resolves names through whoever draws it, so it is
stripped once per resolving context, and copied rather than given one caller's resources.

``drop_unused`` then removes the fonts (and anything else) no content stream refers to any more,
with qpdf's ``remove_unreferenced_resources`` on the pages it is given, which copies a shared
dictionary before pruning it.

``flatten_signatures`` is the copy-mode step for signed, certified, usage-rights and timestamped
files (decided 2026-11-05): a copy can't keep a valid signature, so it keeps the stamp as drawn
content and no signature at all. Each signature widget's appearance (the /N stream, or the state
/AS selects) is drawn into its page as a Form XObject, placed as PDF 32000-1 §12.5.5 places it:
the /BBox, transformed by the form's /Matrix, mapped onto the widget's /Rect, and for a NoRotate
widget on a turned page, turned back about the rectangle's upper-left corner. The matrix is written
at full precision (qpdf's own flattening writes 5 decimals, which moved a rotated stamp by a pixel
row, measured 2026-11-05), and the old content is wrapped in q/Q, so the page renders as it did
with form appearances drawn. The wrap counts the page's own save and restore: a stray Q (pdfium
ignores one with nothing saved) gets an extra q in front, and a q the page leaves open gets an
extra Q, so no cm or colour the page sets can reach the stamps. An appearance with no /BBox is
placed as pdfium places it (moved onto the rectangle's corner, unscaled). Every other annotation and form field stays as it was. Then the
signature fields and their widgets, /Perms (with /DocMDP and /UR3), /DSS, the AcroForm's /SigFlags,
and an AcroForm left with no fields are removed. A Hidden widget is removed without being drawn; a
NoView one is drawn, because pdfium draws it with form appearances on, and that render is what
verify compares. ``flow`` calls it after the layer and before shrink, so the stamp can't block OCR.

One case is known not to match pixel for pixel: a NoRotate widget on a page turned 180 degrees,
where pdfium's own drawing differs by one antialiased row (measured 2026-11-05). verify's
``renders_identical`` catches it, so such a copy fails closed rather than being written.

Pages are 0-based indices, as everywhere in pdfintake.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pikepdf

    from .classify import PageClass

STRIPPABLE_KINDS = frozenset({"invisible_layer", "our_layer"})
TEXT_SHOW_OPS = frozenset({"Tj", "TJ", "'", '"'})
DROPPED_OPS = TEXT_SHOW_OPS | {"Tf"}
MAX_FORM_DEPTH = 64               # forms drawn inside forms; pdfium stops drawing well before this
_APPEARANCE_KEYS = ("/N", "/R", "/D")
_STREAM_ONLY_KEYS = ("/Length", "/Filter", "/DecodeParms", "/DL")

# flatten_signatures
ANNOT_HIDDEN = 2
ANNOT_NOROTATE = 16
MAX_FIELD_DEPTH = 64              # /Kids and /Parent chains followed before a field counts as damaged
_SIG_VALUE_KEYS = ("/V", "/FT", "/Lock", "/SV")       # what makes a removed node a signature field
_CATALOG_SIG_KEYS = ("/Perms", "/DSS")                 # /Perms holds /DocMDP and /UR3


@dataclass
class StripReport:
    ops_dropped: int                  # instructions removed, counted once per content stream written
    forms_touched: int                # Form XObjects whose text was removed (in place or in a copy)
    forms_copied: int                 # of those, the ones copied because something else still uses them
    chars_before: dict[int, int]      # page -> character codes the old layer showed (per drawing of a form)


def strip_text(pdf: "pikepdf.Pdf", pages: Iterable[int], classes: Mapping[int, "PageClass"]) -> StripReport:
    """Drop ``Tj TJ ' " Tf`` from ``pages`` of ``pdf`` (in memory) and every form they draw.

    ``classes`` maps page index to its ``PageClass``. Every listed page must be classed
    ``invisible_layer`` or ``our_layer`` (and the class must be that page's); otherwise
    ``ValueError`` names the page, and the document is left unchanged. Every content stream the
    strip will read is parsed first, so a page whose content doesn't parse, or whose forms nest
    deeper than ``MAX_FORM_DEPTH``, also raises ``ValueError`` before anything changes. A form
    shared with a page that keeps its text is copied first.
    """
    import pikepdf

    wanted = sorted({_index(i) for i in pages})
    count = len(pdf.pages)
    for i in wanted:
        if not 0 <= i < count:
            raise ValueError(f"page {i + 1} is out of range: the document has {count} page(s)")
        cls = classes.get(i)
        kind = getattr(cls, "kind", None)
        if kind not in STRIPPABLE_KINDS:
            what = f"classed {kind}" if kind is not None else "not classified"
            raise ValueError(
                f"page {i + 1} is {what}: only a page whose text is all an invisible layer "
                f"(invisible_layer or our_layer) may have its text stripped"
            )
        own = getattr(cls, "index", i)
        if isinstance(own, bool) or not isinstance(own, int) or own != i:
            raise ValueError(f"page {i + 1} was given the class of another page; it can't be stripped safely")

    report = StripReport(ops_dropped=0, forms_touched=0, forms_copied=0, chars_before={})
    if not wanted:
        return report
    keepers = _keeper_forms(pdf, set(wanted))
    stripper = _Stripper(pdf, keepers, report, pikepdf)
    stripper.write = False                    # a dry pass: parse and walk everything, change nothing
    for i in wanted:
        stripper.page(pdf.pages[i], i)
    stripper.reset()
    stripper.write = True
    for i in wanted:
        stripper.page(pdf.pages[i], i)
    return report


def drop_unused(pdf: "pikepdf.Pdf", pages: Iterable[int] | None = None) -> None:
    """Remove every resource that no content stream refers to (qpdf, page by page and through forms).

    ``pages`` limits the pruning to those pages (``flow`` passes the pages it stripped); None
    prunes every page. A page whose resources hold a Type 3 font without Resources of its own is
    always skipped: its glyph procedures draw through the page's resources, and qpdf doesn't see
    the names they use, so it would delete what they draw.
    """
    count = len(pdf.pages)
    wanted = range(count) if pages is None else sorted({_index(i) for i in pages})
    for i in wanted:
        if not 0 <= i < count:
            raise ValueError(f"page {i + 1} is out of range: the document has {count} page(s)")
        page = pdf.pages[i]
        if _has_resourceless_type3(_inherited(page.obj, "/Resources")):
            continue
        page.remove_unreferenced_resources()


def flatten_signatures(pdf: "pikepdf.Pdf") -> list[str]:
    """Draw each signature widget's appearance into its page, then remove every trace of a signature.

    In memory, on the copy only (decided 2026-11-05). A signature field is any field or widget whose
    /FT, inherited through /Parent, is /Sig, signed or empty, in the AcroForm tree or only on a page.
    Each widget with an /N appearance that isn't Hidden is drawn as page content exactly where
    it was; then the fields and widgets go, with /Perms (/DocMDP, /UR3),
    /DSS, /SigFlags and an AcroForm left without fields. The signature value dictionaries
    (/ByteRange, /Contents, /DocTimeStamp) are emptied in place, so nothing can still reach them.

    Returns the fully qualified names of the removed fields, in document order (empty when the
    file has none, and then nothing is changed). Every other annotation and field is kept.
    """
    import pikepdf

    root = pdf.Root
    acro = root.get("/AcroForm")
    acro = acro if _is_dict(acro) else None
    removed: dict[tuple[int, int] | int, Any] = {}          # nodes to remove, by identity
    names: list[str] = []
    values: list[Any] = []

    def note(node: Any) -> None:
        key = _node_key(node)
        if key in removed:
            return
        removed[key] = node
        if "/T" in node:
            name = _full_name(node)
            if name and name not in names and not _has_named_kids(node):
                names.append(name)
        v = node.get("/V")
        if _is_dict(v):
            values.append(v)

    # the AcroForm tree
    tree_nodes: list[Any] = []
    if acro is not None:
        stack: list[tuple[Any, int]] = [(f, 0) for f in _array(acro.get("/Fields"))]
        seen: set[Any] = set()
        while stack:
            node, depth = stack.pop()
            if not _is_dict(node) or depth > MAX_FIELD_DEPTH or _node_key(node) in seen:
                continue
            seen.add(_node_key(node))
            tree_nodes.append(node)
            stack.extend((k, depth + 1) for k in _array(node.get("/Kids")))
        for node in tree_nodes:
            if _is_sig(node):
                note(node)
    # widgets on the pages, in page order
    draw: dict[int, list[Any]] = {}
    for i, page in enumerate(pdf.pages):
        for annot in _array(page.obj.get("/Annots")):
            if not _is_sig_widget(pikepdf, annot):
                continue
            note(annot)
            parent = annot.get("/Parent")
            if _is_dict(parent) and _is_sig(parent):
                note(parent)
            if drawn_appearance(annot) is not None:
                draw.setdefault(i, []).append(annot)
    perms = root.get("/Perms")
    if _is_dict(perms):
        values.extend(v for _k, v in perms.items() if _is_dict(v))
    if not removed and not any(k in root for k in _CATALOG_SIG_KEYS) and not (acro is not None and "/SigFlags" in acro):
        return names

    if draw:
        _draw_widgets(pdf, draw)
    # take the widgets off their pages
    for page in pdf.pages:
        annots = page.obj.get("/Annots")
        if annots is None:
            continue
        items = _array(annots)
        kept = [a for a in items if not (_is_dict(a) and (_node_key(a) in removed or _is_sig_widget(pikepdf, a)))]
        if len(kept) != len(items):
            if kept:
                page.obj.Annots = pikepdf.Array(kept)
            else:
                del page.obj["/Annots"]
    # and the fields out of the tree
    if acro is not None:
        fields = _prune_fields(pikepdf, _array(acro.get("/Fields")))
        if "/SigFlags" in acro:
            del acro["/SigFlags"]
        if fields:
            acro.Fields = pikepdf.Array(fields)
        else:
            del root["/AcroForm"]
    for key in _CATALOG_SIG_KEYS:
        if key in root:
            del root[key]
    _drop_struct_refs(pikepdf, root, removed)
    for node in removed.values():
        for key in _SIG_VALUE_KEYS:
            if key in node:
                del node[key]
    for value in values:
        if value.is_indirect:
            for key in list(value.keys()):
                del value[key]
    return names


def is_signature_widget(annot: Any) -> bool:
    """A widget annotation whose /FT, inherited through /Parent, is /Sig."""
    import pikepdf

    return _is_sig_widget(pikepdf, annot)


def drawn_appearance(annot: Any) -> Any:
    """The appearance stream ``flatten_signatures`` draws for a signature widget, or None.

    None for a Hidden widget (pdfium doesn't draw it) or one without an /N appearance
    (for /N with states, the one /AS selects). ``verify.images_identical`` uses it on copies.
    """
    if not _is_dict(annot) or _flags(annot) & ANNOT_HIDDEN:
        return None
    ap = annot.get("/AP")
    if not _is_dict(ap):
        return None
    n = ap.get("/N")
    if _is_stream(n):
        return n
    if _is_dict(n):
        state = annot.get("/AS")
        chosen = n.get(str(state)) if state is not None else None
        return chosen if _is_stream(chosen) else None
    return None


# --------------------------------------------------------------------------- helpers


def _index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"page indices are integers, not {value!r}")
    return int(value)


def _inherited(obj: "pikepdf.Object", key: str) -> "pikepdf.Object | None":
    """``key`` from a page dictionary or the nearest /Parent that has it."""
    seen: set[tuple[int, int]] = set()
    node: Any = obj
    while node is not None:
        value = node.get(key)
        if value is not None:
            return value
        og = node.objgen
        if og != (0, 0):
            if og in seen:
                return None
            seen.add(og)
        node = node.get("/Parent")
    return None


def _is_dict(obj: Any) -> bool:
    import pikepdf

    return isinstance(obj, pikepdf.Dictionary)


def _is_stream(obj: Any) -> bool:
    import pikepdf

    return isinstance(obj, pikepdf.Stream)


def _is_form(obj: Any) -> bool:
    import pikepdf

    return isinstance(obj, pikepdf.Stream) and obj.stream_dict.get("/Subtype") == pikepdf.Name.Form


def _category(resources: Any, key: str) -> Any:
    if resources is None or not _is_dict(resources):
        return None
    value = resources.get(key)
    return value if _is_dict(value) else None


def _xobjects(resources: Any) -> Any:
    return _category(resources, "/XObject")


def _own_resources(obj: Any) -> Any:
    d = obj.stream_dict if _is_stream(obj) else obj
    res = d.get("/Resources")
    return res if _is_dict(res) else None


def _is_type3(font: Any) -> bool:
    return _is_dict(font) and str(font.get("/Subtype", "")) == "/Type3"


def _side_children(resources: Any) -> list[tuple[Any, Any]]:
    """What ``resources`` can draw other than through ``Do``: (holder, resources it draws with).

    Tiling patterns, ExtGState soft-mask groups and Type 3 fonts. A holder without Resources of its
    own draws through ``resources`` itself. A soft-mask group is a form, so its holder is the form.
    """
    out: list[tuple[Any, Any]] = []
    patterns = _category(resources, "/Pattern")
    if patterns is not None:
        for _name, pat in patterns.items():
            if _is_stream(pat):
                out.append((pat, _own_resources(pat) or resources))
    states = _category(resources, "/ExtGState")
    if states is not None:
        for _name, gs in states.items():
            if not _is_dict(gs):
                continue
            smask = gs.get("/SMask")
            if _is_dict(smask):
                group = smask.get("/G")
                if _is_stream(group):
                    out.append((group, _own_resources(group) or resources))
    fonts = _category(resources, "/Font")
    if fonts is not None:
        for _name, font in fonts.items():
            if _is_type3(font):
                out.append((font, _own_resources(font) or resources))
    return out


def _keeper_forms(pdf: "pikepdf.Pdf", stripped: set[int]) -> set[tuple[int, int]]:
    """Forms something other than a stripped page's ``Do`` uses, which must keep their text.

    That is every form reachable from a page that isn't stripped, from any annotation appearance,
    and, on any page, from a tiling pattern, a soft-mask group or a Type 3 font (the strip never
    looks inside those, so the forms they draw must stay as they are). Walked without recursion.
    """
    keepers: set[tuple[int, int]] = set()
    seen: set[tuple[tuple[int, int], bool]] = set()
    stack: list[tuple[Any, bool]] = []          # (resources, everything below is a keeper)
    for i, page in enumerate(pdf.pages):
        stack.append((_inherited(page.obj, "/Resources"), i not in stripped))
        annots = page.obj.get("/Annots")
        if annots is None:
            continue
        try:
            items = list(annots)
        except TypeError:
            continue
        for annot in items:
            for stream in _appearance_streams(annot):
                if stream.objgen != (0, 0):
                    keepers.add(stream.objgen)
                stack.append((_own_resources(stream), True))

    def visit(holder: Any, resources: Any, keep: bool) -> None:
        og = holder.objgen
        if og != (0, 0):
            if (og, True) in seen or (og, keep) in seen:
                return
            seen.add((og, keep))
        if keep and _is_form(holder) and og != (0, 0):
            keepers.add(og)
        stack.append((resources, keep))

    while stack:
        resources, keep = stack.pop()
        if resources is None or not _is_dict(resources):
            continue
        xobjects = _xobjects(resources)
        if xobjects is not None:
            for _name, xobj in xobjects.items():
                if _is_form(xobj):
                    own = _own_resources(xobj)
                    visit(xobj, own, keep)             # a resourceless form draws through ``resources``
        for holder, res in _side_children(resources):
            visit(holder, res, True)
    return keepers


def _has_resourceless_type3(resources: Any) -> bool:
    """Whether a Type 3 font without its own Resources is in ``resources`` or any form below it."""
    seen: set[tuple[int, int]] = set()
    stack = [resources]
    while stack:
        res = stack.pop()
        if res is None or not _is_dict(res):
            continue
        fonts = _category(res, "/Font")
        if fonts is not None and any(_is_type3(f) and _own_resources(f) is None for _n, f in fonts.items()):
            return True
        xobjects = _xobjects(res)
        if xobjects is not None:
            for _name, xobj in xobjects.items():
                if _is_form(xobj) and xobj.objgen not in seen:
                    seen.add(xobj.objgen)
                    stack.append(_own_resources(xobj))
    return False


def _appearance_streams(annot: Any) -> list[Any]:
    import pikepdf

    out: list[Any] = []
    if not _is_dict(annot):
        return out
    ap = annot.get("/AP")
    if not _is_dict(ap):
        return out
    for key in _APPEARANCE_KEYS:
        value = ap.get(key)
        if isinstance(value, pikepdf.Stream):
            out.append(value)
        elif _is_dict(value):
            out.extend(v for _k, v in value.items() if isinstance(v, pikepdf.Stream))
    return out


def _fresh_resources(pikepdf: Any, resources: Any, replacements: Mapping[str, Any]) -> Any:
    """A new direct Resources dictionary: each category dictionary copied one level, XObjects re-pointed."""
    fresh = pikepdf.Dictionary()
    if resources is not None and _is_dict(resources):
        for key, value in resources.items():
            if _is_dict(value):
                fresh[key] = pikepdf.Dictionary({k: v for k, v in value.items()})
            else:
                fresh[key] = value
    if replacements:
        xobjects = fresh.get("/XObject")
        if xobjects is None:
            xobjects = pikepdf.Dictionary()
            fresh["/XObject"] = xobjects
        for name, obj in replacements.items():
            xobjects[name] = obj
    return fresh


class _Stripper:
    """Walks each stripped page and the forms it draws; ``write=False`` walks without changing anything.

    A form with Resources of its own gives the same result wherever it's drawn, so it's done once.
    A form without (the PDF 1.1 style) resolves names through whoever draws it, so it's done once per
    distinct resolving context, and when its nested forms needed replacing it's copied for that
    context instead of being given the first caller's resources.
    """

    def __init__(self, pdf: "pikepdf.Pdf", keepers: set[tuple[int, int]], report: StripReport, pikepdf: Any) -> None:
        self.pdf = pdf
        self.keepers = keepers
        self.report = report
        self.pikepdf = pikepdf
        self.write = True
        self.index = 0
        self.parsed: dict[Any, list[Any]] = {}            # original instructions, parsed once
        self.written: set[tuple[int, int]] = set()        # forms whose own stream was rewritten
        self.reset()

    def reset(self) -> None:
        self.done: dict[Any, tuple[Any, int]] = {}         # cache key -> (form to draw, chars in its tree)
        self.active: set[tuple[int, int]] = set()

    def _parse(self, key: Any, holder: Any) -> list[Any]:
        if key in self.parsed:
            return self.parsed[key]
        pikepdf = self.pikepdf
        try:
            instructions = list(pikepdf.parse_content_stream(holder))
        except (pikepdf.PdfError, ValueError, TypeError) as exc:
            raise ValueError(f"page {self.index + 1} can't be stripped safely: "
                             f"a content stream it draws doesn't parse") from exc
        self.parsed[key] = instructions
        return instructions

    def filter(self, instructions: list[Any], resources: Any) -> tuple[list[Any], int, int, list[str]]:
        """(kept instructions, dropped count, character codes shown, names drawn with Do)."""
        pikepdf = self.pikepdf
        kept: list[Any] = []
        dropped = 0
        chars = 0
        names: list[str] = []
        two_byte = False
        stack: list[bool] = []
        fonts = resources.get("/Font") if resources is not None and _is_dict(resources) else None
        for ins in instructions:
            if isinstance(ins, pikepdf.ContentStreamInlineImage):
                kept.append(ins)
                continue
            op = str(ins.operator)
            if op == "q":
                stack.append(two_byte)
            elif op == "Q":
                if stack:
                    two_byte = stack.pop()
            elif op == "Tf":
                two_byte = self._two_byte(fonts, ins.operands)
            elif op == "Do" and len(ins.operands) == 1 and isinstance(ins.operands[0], pikepdf.Name):
                names.append(str(ins.operands[0]))
            if op in DROPPED_OPS:
                dropped += 1
                if op in TEXT_SHOW_OPS:
                    chars += _codes_shown(pikepdf, ins.operands, two_byte)
                continue
            kept.append(ins)
        return kept, dropped, chars, names

    @staticmethod
    def _font_is_type0(font: Any) -> bool:
        return bool(font is not None and _is_dict(font) and str(font.get("/Subtype", "")) == "/Type0")

    @classmethod
    def _two_byte(cls, fonts: Any, operands: Any) -> bool:
        if fonts is None or not _is_dict(fonts) or not operands:
            return False
        try:
            font = fonts.get(str(operands[0]))
        except Exception:
            return False
        return cls._font_is_type0(font)

    def _context(self, resources: Any) -> tuple[Any, ...]:
        """What a resourceless form's result depends on in ``resources``: its XObject and font names."""
        xobjects = _xobjects(resources)
        xkey = tuple(sorted((str(k), v.objgen if _is_stream(v) else None) for k, v in xobjects.items())) \
            if xobjects is not None else ()
        fonts = _category(resources, "/Font")
        fkey = tuple(sorted((str(k), self._font_is_type0(v)) for k, v in fonts.items())) if fonts is not None else ()
        return xkey, fkey

    def forms_drawn(self, names: list[str], resources: Any, depth: int) -> tuple[dict[str, Any], int]:
        """Strip each form drawn by name; (replacements for copied forms, characters in their trees)."""
        replacements: dict[str, Any] = {}
        chars = 0
        xobjects = _xobjects(resources)
        if xobjects is None:
            return replacements, chars
        for name in names:
            xobj = xobjects.get(name)
            if not _is_form(xobj):
                continue
            result, form_chars = self.form(xobj, resources, depth + 1)
            chars += form_chars
            if result.objgen != xobj.objgen:
                replacements[name] = result
        return replacements, chars

    def page(self, page: Any, index: int) -> None:
        pikepdf = self.pikepdf
        self.index = index
        resources = _inherited(page.obj, "/Resources")
        has_contents = page.obj.get("/Contents") is not None
        instructions = self._parse(("page", index), page) if has_contents else []
        kept, dropped, chars, names = self.filter(instructions, resources)
        replacements, form_chars = self.forms_drawn(names, resources, 0)
        if not self.write:
            return
        if dropped:
            page.obj.Contents = self.pdf.make_stream(pikepdf.unparse_content_stream(kept))
            self.report.ops_dropped += dropped
        if replacements:
            page.obj.Resources = _fresh_resources(pikepdf, resources, replacements)
        self.report.chars_before[index] = chars + form_chars

    def _copy(self, form: Any, data: bytes, resources: Any, replacements: Mapping[str, Any]) -> Any:
        pikepdf = self.pikepdf
        result = self.pdf.make_stream(data)
        for key, value in form.stream_dict.items():
            if key not in _STREAM_ONLY_KEYS and key != "/Resources":
                result.stream_dict[key] = value
        result.stream_dict["/Resources"] = _fresh_resources(pikepdf, resources, replacements)
        self.report.forms_copied += 1
        self.report.forms_touched += 1
        return result

    def form(self, form: Any, parent_resources: Any, depth: int) -> tuple[Any, int]:
        pikepdf = self.pikepdf
        og = form.objgen
        if depth > MAX_FORM_DEPTH:
            raise ValueError(f"page {self.index + 1} draws forms nested more than {MAX_FORM_DEPTH} deep; "
                             f"it can't be stripped safely")
        own = _own_resources(form)
        resourceless = form.stream_dict.get("/Resources") is None
        resources = own if not resourceless else parent_resources       # PDF 1.1 forms use the caller's
        key = (og, self._context(parent_resources)) if resourceless else og
        if key in self.done:
            return self.done[key]
        if og in self.active:                     # a form that draws itself: leave the loop alone
            return form, 0
        self.active.add(og)
        try:
            kept, dropped, chars, names = self.filter(self._parse(og, form), resources)
            replacements, form_chars = self.forms_drawn(names, resources, depth)
            chars += form_chars
            if not self.write or (not dropped and not replacements):
                result = form
            elif og in self.keepers or (resourceless and replacements):
                # Something else draws the original, or (resourceless) another caller may resolve its
                # names differently: this caller gets its own copy.
                data = pikepdf.unparse_content_stream(kept) if dropped else form.read_bytes()
                result = self._copy(form, data, resources, replacements)
                self.report.ops_dropped += dropped
            else:
                result = form
                if dropped and og not in self.written:
                    form.write(pikepdf.unparse_content_stream(kept))
                    self.written.add(og)
                    self.report.ops_dropped += dropped
                    self.report.forms_touched += 1
                if replacements:                  # only a form with its own Resources gets here
                    form.stream_dict["/Resources"] = _fresh_resources(pikepdf, resources, replacements)
                    if not dropped:
                        self.report.forms_touched += 1
            self.done[key] = (result, chars)
            return result, chars
        finally:
            self.active.discard(og)


def _codes_shown(pikepdf: Any, operands: Any, two_byte: bool) -> int:
    """Character codes a text-showing instruction draws: 2 bytes each in a Type0 font, else 1."""
    total = 0
    for operand in operands:
        if isinstance(operand, pikepdf.String):
            n = len(bytes(operand))
            total += n // 2 if two_byte else n
        elif isinstance(operand, pikepdf.Array):
            for item in operand:
                if isinstance(item, pikepdf.String):
                    n = len(bytes(item))
                    total += n // 2 if two_byte else n
    return total


# --------------------------------------------------------------------------- signature helpers


def _array(obj: Any) -> list[Any]:
    import pikepdf

    return list(obj) if isinstance(obj, pikepdf.Array) else []


def _node_key(node: Any) -> Any:
    og = node.objgen
    return og if og != (0, 0) else id(node)


def _is_sig(node: Any) -> bool:
    """/FT /Sig on the node or the nearest /Parent that carries /FT."""
    import pikepdf

    current = node
    for _ in range(MAX_FIELD_DEPTH):
        if not _is_dict(current):
            return False
        ft = current.get("/FT")
        if ft is not None:
            return ft == pikepdf.Name.Sig
        current = current.get("/Parent")
    return False


def _full_name(node: Any) -> str:
    parts: list[str] = []
    current = node
    for _ in range(MAX_FIELD_DEPTH):
        if not _is_dict(current):
            break
        t = current.get("/T")
        if t is not None:
            parts.append(str(t))
        current = current.get("/Parent")
    return ".".join(reversed(parts))


def _has_named_kids(node: Any) -> bool:
    return any(_is_dict(k) and "/T" in k for k in _array(node.get("/Kids")))


def _flags(annot: Any) -> int:
    try:
        return int(annot.get("/F", 0))
    except (TypeError, ValueError):
        return 0


def _num(value: float) -> bytes:
    """A content-stream number with no exponent and up to 10 decimals (qpdf's 5 can shift a pixel)."""
    if abs(value) < 5e-11:
        return b"0"
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text.encode("ascii")


def _numbers(obj: Any, count: int) -> list[float] | None:
    import pikepdf

    if not isinstance(obj, pikepdf.Array) or len(obj) != count:
        return None
    try:
        values = [float(v) for v in obj]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(v) for v in values) else None


_IDENTITY = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
_TURN_CCW = {90: [0.0, 1.0, -1.0, 0.0, 0.0, 0.0], 180: [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
             270: [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]}
_MATCH_MIN = 0.001                # pdfium's MatchRect keeps scale 1 along an axis narrower than this


def _concat(m1: list[float], m2: list[float]) -> list[float]:
    """``m1`` then ``m2``, in PDF's row-vector convention (a point is p · m1 · m2)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return [a1 * a2 + b1 * c2, a1 * b2 + b1 * d2, c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2]


def _placement(appearance: Any, rect: Any, turn: int = 0) -> list[float] | None:
    """The matrix that draws ``appearance`` on its widget's /Rect (PDF 32000-1 §12.5.5).

    The form's /BBox, transformed by its /Matrix, is mapped onto the rectangle; the form's own
    /Matrix is then applied by ``Do`` itself. ``turn`` is the page's /Rotate for a NoRotate widget:
    the appearance is turned back by it about the rectangle's upper-left corner, so it stays
    upright on screen (§12.5.3, as pdfium and qpdf draw it). None when the rectangle has no area.

    A box pdfium can't scale from follows pdfium's ``CFX_Matrix::MatchRect`` (decided 2026-11-05):
    a missing or malformed /BBox counts as the empty box at the origin, and along an axis where
    the transformed box is under 0.001 wide the scale is 1, so the appearance is only moved onto
    the rectangle's corner. /BBox is required by the spec, so only a malformed producer gets here.
    """
    bbox = _numbers(appearance.stream_dict.get("/BBox"), 4) or [0.0, 0.0, 0.0, 0.0]
    box = _numbers(rect, 4)
    if box is None:
        return None
    form = _numbers(appearance.stream_dict.get("/Matrix"), 6) or list(_IDENTITY)
    x0, x1 = sorted((box[0], box[2]))
    y0, y1 = sorted((box[1], box[3]))
    w, h = x1 - x0, y1 - y0
    turned = _TURN_CCW.get(turn)
    m = _concat(form, turned) if turned is not None else form
    if turn == 90:
        x0, y0, x1, y1 = x0, y1, x0 + h, y1 + w
    elif turn == 180:
        x0, y0, x1, y1 = x0 - w, y1, x0, y1 + h
    elif turn == 270:
        x0, y0, x1, y1 = x0 - h, y1 - w, x0, y1
    corners = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])]
    xs = [m[0] * x + m[2] * y + m[4] for x, y in corners]
    ys = [m[1] * x + m[3] * y + m[5] for x, y in corners]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    sx = (x1 - x0) / (bx1 - bx0) if bx1 - bx0 >= _MATCH_MIN else 1.0
    sy = (y1 - y0) / (by1 - by0) if by1 - by0 >= _MATCH_MIN else 1.0
    fit = [sx, 0.0, 0.0, sy, x0 - bx0 * sx, y0 - by0 * sy]
    return _concat(turned, fit) if turned is not None else fit


def _page_turn(page: Any) -> int:
    value = _inherited(page.obj, "/Rotate")
    try:
        turn = int(value) % 360 if value is not None else 0
    except (TypeError, ValueError):
        return 0
    return turn if turn in _TURN_CCW else 0


def _draw_widgets(pdf: "pikepdf.Pdf", draw: Mapping[int, list[Any]]) -> None:
    """Draw each widget's shown appearance into its page, as a viewer places it.

    The page's old content is wrapped in q/Q so no graphics state leaks into the stamps, and the
    page gets a Resources dictionary of its own (inherited or shared ones are copied one level), so
    no other page sees the new XObject names.
    """
    import pikepdf

    for i, widgets in draw.items():
        page = pdf.pages[i]
        resources = _inherited(page.obj, "/Resources")
        xobjects = _xobjects(resources)
        taken = {str(k) for k in xobjects.keys()} if xobjects is not None else set()
        added: dict[str, Any] = {}
        ops: list[bytes] = []
        for widget in widgets:
            appearance = drawn_appearance(widget)
            turn = _page_turn(page) if _flags(widget) & ANNOT_NOROTATE else 0
            matrix = _placement(appearance, widget.get("/Rect"), turn) if appearance is not None else None
            if matrix is None:
                continue
            if appearance.stream_dict.get("/Subtype") != pikepdf.Name.Form:
                appearance.stream_dict["/Subtype"] = pikepdf.Name.Form
            if not appearance.is_indirect:
                appearance = pdf.make_indirect(appearance)
            n = len(added) + 1
            while f"/FxSig{n}" in taken or f"/FxSig{n}" in added:
                n += 1
            name = f"/FxSig{n}"
            added[name] = appearance
            ops.append(b"q " + b" ".join(_num(v) for v in matrix) + b" cm " + name.encode("ascii") + b" Do Q\n")
        if not ops:
            continue
        stray, left_open = _save_restore_balance(pikepdf, page)
        page.obj.Resources = _fresh_resources(pikepdf, resources, added)
        page.contents_add(pdf.make_stream(b"q\n" * (stray + 1)), prepend=True)
        page.contents_add(pdf.make_stream(b"\n" + b"Q\n" * (left_open + 1) + b"".join(ops)), prepend=False)


def _save_restore_balance(pikepdf: Any, page: Any) -> tuple[int, int]:
    """(unmatched Q, q still open at the end) in the page's own content, as pdfium counts them.

    pdfium ignores a Q with nothing saved, so a page may close more than it opened. Each such Q
    would pop the wrapping q early and leave the page's later cm under the stamps (decided
    2026-11-05), so ``_draw_widgets`` opens one more q per stray Q, and closes every q the page
    left open. Content that doesn't parse counts as balanced; verify still compares the render.
    """
    try:
        instructions = pikepdf.parse_content_stream(page, "q Q")
    except Exception:  # noqa: BLE001 - a page pdfium can't parse either; verify decides
        return 0, 0
    depth = stray = 0
    for instruction in instructions:
        op = str(getattr(instruction, "operator", ""))
        if op == "q":
            depth += 1
        elif op == "Q":
            if depth:
                depth -= 1
            else:
                stray += 1
    return stray, depth


def _is_sig_widget(pikepdf: Any, annot: Any) -> bool:
    return _is_dict(annot) and annot.get("/Subtype") == pikepdf.Name.Widget and _is_sig(annot)


def _prune_fields(pikepdf: Any, nodes: list[Any], depth: int = 0) -> list[Any]:
    """``nodes`` without the signature fields and widgets; a parent whose kids all went goes too.

    A parent keeps whichever kids aren't signatures (a kid's own /FT overrides the parent's).
    """
    kept: list[Any] = []
    for node in nodes:
        if not _is_dict(node):
            kept.append(node)
            continue
        kids = node.get("/Kids")
        if isinstance(kids, pikepdf.Array) and len(kids) and depth < MAX_FIELD_DEPTH:
            left = _prune_fields(pikepdf, list(kids), depth + 1)
            if not left:
                continue
            if len(left) != len(kids):
                node.Kids = pikepdf.Array(left)
        elif _is_sig(node):
            continue
        kept.append(node)
    return kept


def _drop_struct_refs(pikepdf: Any, root: Any, removed: Mapping[Any, Any]) -> None:
    """Remove the structure tree's object references (/OBJR) to the removed widgets."""
    tree = root.get("/StructTreeRoot")
    if not _is_dict(tree):
        return
    seen: set[Any] = set()
    stack: list[tuple[Any, int]] = [(tree, 0)]
    while stack:
        elem, depth = stack.pop()
        if not _is_dict(elem) or depth > MAX_FORM_DEPTH or _node_key(elem) in seen:
            continue
        seen.add(_node_key(elem))
        kids = elem.get("/K")
        items = list(kids) if isinstance(kids, pikepdf.Array) else ([kids] if kids is not None else [])
        left = []
        for kid in items:
            if _is_dict(kid) and kid.get("/Type") == pikepdf.Name("/OBJR"):
                obj = kid.get("/Obj")
                if _is_dict(obj) and _node_key(obj) in removed:
                    continue
            left.append(kid)
            stack.append((kid, depth + 1))
        if len(left) != len(items):
            if not left:
                del elem["/K"]
            elif isinstance(kids, pikepdf.Array):
                elem.K = pikepdf.Array(left)
            else:
                elem.K = left[0]
