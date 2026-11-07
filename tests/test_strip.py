"""pdf/strip.py: an old invisible layer goes, operator by operator, and nothing else moves."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf
import pytest

from fixtures import build
from pdfintake.pdf.strip import StripReport, drop_unused, strip_text

KINDS = ("image_only", "invisible_layer", "our_layer", "hidden_text", "born_digital", "garbled", "clip_text", "empty")


@dataclass
class FakeClass:
    """Stands in for classify.PageClass: strip only reads ``kind``."""

    index: int
    kind: str
    warnings: list[str] = field(default_factory=list)


def classes(kinds: dict[int, str]) -> dict[int, FakeClass]:
    return {i: FakeClass(i, k) for i, k in kinds.items()}


def _lock():
    from pdfintake.pdf.render import PDFIUM_LOCK

    return PDFIUM_LOCK


def renders(path: Path) -> list[bytes]:
    """Every page at 72 dpi, grayscale, as bytes (pdfium)."""
    import numpy as np
    import pypdfium2 as pdfium

    out = []
    with _lock():
        doc = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(doc)):
                page = doc[i]
                bm = page.render(scale=1, grayscale=True, may_draw_forms=False)
                arr = np.frombuffer(bm.buffer, dtype=np.uint8, count=bm.stride * bm.height).reshape(bm.height, bm.stride)
                out.append(arr[:, :bm.width].tobytes() + f"{bm.width}x{bm.height}".encode())
                bm.close()
                page.close()
        finally:
            doc.close()
    return out


def page_texts(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    with _lock():
        doc = pdfium.PdfDocument(str(path))
        try:
            texts = []
            for i in range(len(doc)):
                page = doc[i]
                tp = page.get_textpage()
                texts.append(tp.get_text_range())
                tp.close()
                page.close()
            return texts
        finally:
            doc.close()


def strip_file(src: Path, dest: Path, pages, kinds: dict[int, str], *, drop: bool = True) -> StripReport:
    with pikepdf.open(src) as pdf:
        report = strip_text(pdf, pages, classes(kinds))
        if drop:
            drop_unused(pdf)
        pdf.save(dest)
    return report


def image_streams(path: Path) -> list[bytes]:
    with pikepdf.open(path) as pdf:
        out = []
        for obj in pdf.objects:
            if isinstance(obj, pikepdf.Stream) and obj.stream_dict.get("/Subtype") == pikepdf.Name.Image:
                out.append(bytes(obj.read_raw_bytes()))
        return sorted(out)


def instructions(stream_or_page) -> list[tuple[str, list]]:
    return [(str(i.operator), list(i.operands)) for i in pikepdf.parse_content_stream(stream_or_page)
            if not isinstance(i, pikepdf.ContentStreamInlineImage)]


LAYER = "Example old layer 4521"


def scan_page(*, in_form: bool = False, text: str = LAYER, extra: bytes = b"", jpeg: bool = True) -> build.PageSpec:
    img = build.photo_like(240, 180, seed=3)
    data = build.jpeg_bytes(img) if jpeg else img
    return build.PageSpec(texts=(build.TextSpec(text, mode=3),), images=(build.ImageSpec(data, 0, 0, 612, 792),),
                          in_form=in_form, extra=extra)


# --------------------------------------------------------------------------- the §7.1 cases


def test_a_layer_inside_a_form_xobject_is_removed_to_zero_characters(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page(in_form=True)])
    assert page_texts(src)[0].strip() == LAYER
    report = strip_file(src, tmp_path / "out.pdf", [0], {0: "invisible_layer"})
    assert page_texts(tmp_path / "out.pdf") == [""]
    assert report.forms_touched == 1 and report.forms_copied == 0
    assert report.ops_dropped == 2                       # Tf and Tj
    assert report.chars_before == {0: len(LAYER)}


def test_the_image_stream_and_the_render_stay_identical(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page(), scan_page(in_form=True, jpeg=False)])
    strip_file(src, tmp_path / "out.pdf", [0, 1], {0: "invisible_layer", 1: "invisible_layer"})
    assert image_streams(tmp_path / "out.pdf") == image_streams(src)
    assert renders(tmp_path / "out.pdf") == renders(src)
    assert page_texts(tmp_path / "out.pdf") == ["", ""]


def test_a_transparent_layer_goes_and_the_render_stays_identical(tmp_path: Path):
    # Scanner Pro's layer: fill mode at ExtGState /ca 0, over the scan, on the page and in a form
    def transparent(in_form: bool) -> build.PageSpec:
        img = build.photo_like(240, 180, seed=3)
        return build.PageSpec(texts=(build.TextSpec(LAYER, fill_alpha=0.0),),
                              images=(build.ImageSpec(build.jpeg_bytes(img), 0, 0, 612, 792),), in_form=in_form)

    src = build.write_pdf(tmp_path / "in.pdf", [transparent(False), transparent(True)])
    assert [t.strip() for t in page_texts(src)] == [LAYER, LAYER]
    from pdfintake.pdf.classify import classify

    found = {c.index: c for c in classify(src, redo_ocr=False, keep_existing_layer=False)}
    assert [c.kind for c in found.values()] == ["invisible_layer", "invisible_layer"]
    with pikepdf.open(src) as pdf:
        report = strip_text(pdf, [0, 1], found)
        drop_unused(pdf)
        pdf.save(tmp_path / "out.pdf")
    assert report.chars_before == {0: len(LAYER), 1: len(LAYER)}
    assert image_streams(tmp_path / "out.pdf") == image_streams(src)
    assert renders(tmp_path / "out.pdf") == renders(src)
    assert page_texts(tmp_path / "out.pdf") == ["", ""]


def test_a_line_width_inside_bt_survives(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page(extra=b"BT 0.57 w 3 Tr (x) Tj ET")])
    strip_file(src, tmp_path / "out.pdf", [0], {0: "invisible_layer"})
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        ops = instructions(pdf.pages[0])
    names = [op for op, _ in ops]
    assert not {"Tj", "TJ", "'", '"', "Tf"} & set(names)
    w = [(k, operands) for k, (op, operands) in enumerate(ops) if op == "w"]
    assert len(w) == 1 and float(w[0][1][0]) == pytest.approx(0.57)
    k = w[0][0]
    last_bt = max(j for j in range(k) if ops[j][0] == "BT")
    assert all(ops[j][0] != "ET" for j in range(last_bt, k)), "0.57 w must still sit inside its BT ... ET"
    assert "Tr" in names and "Td" in names and "Do" in names and "cm" in names


def test_fonts_are_dropped_once_tf_is(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page()])

    def fonts(path: Path) -> int:
        with pikepdf.open(path) as pdf:
            return sum(1 for o in pdf.objects if isinstance(o, pikepdf.Dictionary) and o.get("/Type") == pikepdf.Name.Font)

    assert fonts(src) == 1
    with pikepdf.open(src) as pdf:                     # drop_unused alone keeps a font that's still used
        drop_unused(pdf)
        pdf.save(tmp_path / "kept.pdf")
    assert fonts(tmp_path / "kept.pdf") == 1
    strip_file(src, tmp_path / "out.pdf", [0], {0: "invisible_layer"})
    assert fonts(tmp_path / "out.pdf") == 0
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert len(pdf.pages[0].obj.Resources.get("/Font", pikepdf.Dictionary())) == 0
        assert "/Im0" in pdf.pages[0].obj.Resources.XObject


def _shared_form_pdf(path: Path) -> Path:
    """Two pages drawing one form that holds an image and invisible text."""
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica, Encoding=pikepdf.Name.WinAnsiEncoding))
    img = pdf.make_stream(build.jpeg_bytes(build.photo_like(120, 90, seed=5)), Type=pikepdf.Name.XObject,
                          Subtype=pikepdf.Name.Image, Width=120, Height=90, ColorSpace=pikepdf.Name.DeviceRGB,
                          BitsPerComponent=8, Filter=pikepdf.Name.DCTDecode)
    form = pdf.make_stream(b"q 300 0 0 200 50 400 cm /Im0 Do Q BT /F1 12 Tf 3 Tr 72 300 Td (Shared old layer) Tj ET",
                           Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    form.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font), XObject=pikepdf.Dictionary(Im0=img))
    for _ in range(2):
        page = pdf.add_blank_page(page_size=(612, 792))
        page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))
        page.obj.Contents = pdf.make_stream(b"q /Fm0 Do Q")
    pdf.save(path, deterministic_id=True)
    return path


def test_a_shared_form_is_copied(tmp_path: Path):
    src = _shared_form_pdf(tmp_path / "in.pdf")
    report = strip_file(src, tmp_path / "out.pdf", [0], {0: "invisible_layer", 1: "invisible_layer"})
    assert report.forms_copied == 1 and report.forms_touched == 1
    texts = page_texts(tmp_path / "out.pdf")
    assert texts[0] == ""
    assert texts[1].strip() == "Shared old layer"           # the page that keeps its text still has it
    assert renders(tmp_path / "out.pdf") == renders(src)
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        f0 = pdf.pages[0].obj.Resources.XObject.Fm0
        f1 = pdf.pages[1].obj.Resources.XObject.Fm0
        assert f0.objgen != f1.objgen
        assert f0.Resources.XObject.Im0.objgen == f1.Resources.XObject.Im0.objgen   # the image isn't copied
        assert len(f1.Resources.Font) == 1 and len(f0.Resources.get("/Font", {})) == 0


def test_a_form_used_only_by_stripped_pages_is_changed_in_place(tmp_path: Path):
    src = _shared_form_pdf(tmp_path / "in.pdf")
    report = strip_file(src, tmp_path / "out.pdf", [0, 1], {0: "our_layer", 1: "invisible_layer"})
    assert report.forms_copied == 0 and report.forms_touched == 1
    assert report.chars_before == {0: len("Shared old layer"), 1: len("Shared old layer")}
    assert report.ops_dropped == 2                          # counted once for the one form
    assert page_texts(tmp_path / "out.pdf") == ["", ""]
    assert renders(tmp_path / "out.pdf") == renders(src)


@pytest.mark.parametrize("kind", [k for k in KINDS if k not in ("invisible_layer", "our_layer")])
def test_strip_text_refuses_every_page_that_isnt_a_layer(tmp_path: Path, kind: str):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page()])
    with pikepdf.open(src) as pdf:
        before = bytes(pdf.pages[0].obj.Contents.read_bytes())
        with pytest.raises(ValueError, match=kind):
            strip_text(pdf, [0], classes({0: kind}))
        assert bytes(pdf.pages[0].obj.Contents.read_bytes()) == before


def test_strip_text_on_a_born_digital_page_raises_and_changes_nothing(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page(), build.PageSpec(texts=(build.TextSpec("Printed words"),))])
    with pikepdf.open(src) as pdf:
        before = [bytes(p.obj.Contents.read_bytes()) for p in pdf.pages]
        with pytest.raises(ValueError, match="page 2 is classed born_digital"):
            strip_text(pdf, [0, 1], classes({0: "invisible_layer", 1: "born_digital"}))
        assert [bytes(p.obj.Contents.read_bytes()) for p in pdf.pages] == before   # page 1 untouched too


def test_an_unclassified_or_missing_page_raises(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page()])
    with pikepdf.open(src) as pdf:
        with pytest.raises(ValueError, match="not classified"):
            strip_text(pdf, [0], {})
        with pytest.raises(ValueError, match="out of range"):
            strip_text(pdf, [3], classes({3: "invisible_layer"}))
        with pytest.raises(ValueError):
            strip_text(pdf, [True], classes({1: "invisible_layer"}))


# --------------------------------------------------------------------------- more operators and structures


def test_every_text_showing_operator_goes(tmp_path: Path):
    extra = b"BT /F1 9 Tf 3 Tr 72 100 Td [(Ab) -120 (cd)] TJ (ef) ' 1 2 (gh) \" ET"
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page(extra=extra)])
    report = strip_file(src, tmp_path / "out.pdf", [0], {0: "invisible_layer"})
    assert page_texts(tmp_path / "out.pdf") == [""]
    assert report.chars_before[0] == len(LAYER) + 8
    assert report.ops_dropped == 2 + 4                   # Tf Tj, then Tf TJ ' "
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert not {"Tj", "TJ", "'", '"', "Tf"} & {op for op, _ in instructions(pdf.pages[0])}


def test_an_fpdf2_layer_with_two_byte_codes_is_stripped_and_counted(tmp_path: Path):
    text = ["Example sample 4521", build.THAI_SAMPLE]
    font = build.box_font("".join(text))
    src = build.fpdf_text_pdf(tmp_path / "in.pdf", text, font, invisible=True, size=10)
    report = strip_file(src, tmp_path / "out.pdf", [0], {0: "our_layer"})
    assert report.chars_before == {0: sum(len(t) for t in text)}
    assert page_texts(tmp_path / "out.pdf") == [""]
    assert renders(tmp_path / "out.pdf") == renders(src)


def test_nested_forms_are_stripped_at_every_depth(tmp_path: Path):
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica))
    inner = pdf.make_stream(b"BT /F1 10 Tf 3 Tr 80 80 Td (Inner words) Tj ET 0 0 m 10 10 l S",
                            Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    inner.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    outer = pdf.make_stream(b"q /Fx Do Q BT /F1 10 Tf 3 Tr 80 200 Td (Outer words) Tj ET",
                            Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    outer.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font), XObject=pikepdf.Dictionary(Fx=inner))
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fo=outer))
    page.obj.Contents = pdf.make_stream(b"/Fo Do")
    pdf.save(tmp_path / "in.pdf")
    report = strip_file(tmp_path / "in.pdf", tmp_path / "out.pdf", [0], {0: "invisible_layer"})
    assert report.forms_touched == 2
    assert report.chars_before == {0: len("Inner words") + len("Outer words")}
    assert page_texts(tmp_path / "out.pdf") == [""]
    assert renders(tmp_path / "out.pdf") == renders(tmp_path / "in.pdf")


def test_a_form_an_annotation_also_shows_is_copied(tmp_path: Path):
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica))
    form = pdf.make_stream(b"BT /F1 10 Tf 3 Tr 5 5 Td (Stamp) Tj ET", Type=pikepdf.Name.XObject,
                           Subtype=pikepdf.Name.Form, BBox=[0, 0, 100, 40])
    form.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))
    page.obj.Contents = pdf.make_stream(b"q 1 0 0 1 100 100 cm /Fm0 Do Q")
    annot = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Stamp,
                                                 Rect=[300, 300, 400, 340], AP=pikepdf.Dictionary(N=form)))
    page.obj.Annots = pikepdf.Array([annot])
    pdf.save(tmp_path / "in.pdf")
    with pikepdf.open(tmp_path / "in.pdf") as doc:
        report = strip_text(doc, [0], classes({0: "invisible_layer"}))
        assert report.forms_copied == 1
        ap = doc.pages[0].obj.Annots[0].AP.N
        assert ("Tj", [pikepdf.String("Stamp")]) in instructions(ap)
        drawn = doc.pages[0].obj.Resources.XObject.Fm0
        assert drawn.objgen != ap.objgen and "Tj" not in {op for op, _ in instructions(drawn)}


def test_a_content_stream_shared_with_a_kept_page_isnt_rewritten_in_place(tmp_path: Path):
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica))
    contents = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 72 500 Td (Same stream) Tj ET")
    for _ in range(2):
        page = pdf.add_blank_page(page_size=(612, 792))
        page.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
        page.obj.Contents = contents
    pdf.save(tmp_path / "in.pdf")
    strip_file(tmp_path / "in.pdf", tmp_path / "out.pdf", [1], {1: "invisible_layer"})
    texts = page_texts(tmp_path / "out.pdf")
    assert texts[0].strip() == "Same stream" and texts[1] == ""


def test_inherited_resources_are_followed(tmp_path: Path):
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica))
    form = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 72 500 Td (Inherited) Tj ET", Type=pikepdf.Name.XObject,
                           Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    form.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    for _ in range(2):
        page = pdf.add_blank_page(page_size=(612, 792))
        page.obj.Contents = pdf.make_stream(b"/Fm0 Do")
    pdf.Root.Pages.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fm0=form))
    for page in pdf.pages:
        if "/Resources" in page.obj:
            del page.obj["/Resources"]
    pdf.save(tmp_path / "in.pdf")
    assert [t.strip() for t in page_texts(tmp_path / "in.pdf")] == ["Inherited", "Inherited"]
    report = strip_file(tmp_path / "in.pdf", tmp_path / "out.pdf", [0], {0: "invisible_layer"})
    assert report.forms_copied == 1                            # page 2 inherits the same form and keeps it
    texts = page_texts(tmp_path / "out.pdf")
    assert texts[0] == "" and texts[1].strip() == "Inherited"


def test_a_page_with_nothing_to_drop_is_left_byte_identical(tmp_path: Path):
    img = build.photo_like(100, 80)
    src = build.write_pdf(tmp_path / "in.pdf", [build.PageSpec(images=(build.ImageSpec(build.jpeg_bytes(img)),))])
    with pikepdf.open(src) as pdf:
        before = pdf.pages[0].obj.Contents.objgen, bytes(pdf.pages[0].obj.Contents.read_raw_bytes())
        report = strip_text(pdf, [0], classes({0: "invisible_layer"}))
        after = pdf.pages[0].obj.Contents.objgen, bytes(pdf.pages[0].obj.Contents.read_raw_bytes())
    assert before == after
    assert report == StripReport(ops_dropped=0, forms_touched=0, forms_copied=0, chars_before={0: 0})


def test_no_pages_is_a_no_op(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page()])
    with pikepdf.open(src) as pdf:
        assert strip_text(pdf, [], {}) == StripReport(0, 0, 0, {})



# --------------------------------------------------------------------------- review fixes


def _font(pdf: pikepdf.Pdf) -> pikepdf.Object:
    return pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica, Encoding=pikepdf.Name.WinAnsiEncoding))


def _text_form(pdf: pikepdf.Pdf, font: pikepdf.Object, text: bytes, *, mode: int | None = 3, image=None) -> pikepdf.Stream:
    """A form showing ``text``; ``mode=None`` sets no render mode, so the caller's applies."""
    draw = b"q 200 0 0 200 50 50 cm /Im Do Q " if image is not None else b""
    tr = b"" if mode is None else str(mode).encode() + b" Tr "
    form = pdf.make_stream(draw + b"BT /F1 24 Tf " + tr + b"60 400 Td (" + text + b") Tj ET",
                           Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    res = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    if image is not None:
        res.XObject = pikepdf.Dictionary(Im=image)
    form.Resources = res
    return form


def _gray_image(pdf: pikepdf.Pdf, shade: int) -> pikepdf.Stream:
    import zlib

    return pdf.make_stream(zlib.compress(bytes([shade]) * 64), Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image,
                           Width=8, Height=8, ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8,
                           Filter=pikepdf.Name.FlateDecode)


def test_a_resourceless_form_drawn_by_two_pages_is_stripped_for_each(tmp_path: Path):
    pdf = pikepdf.new()
    font = _font(pdf)
    outer = pdf.make_stream(b"/Fx Do", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    for text in (b"Inner one", b"Inner two"):                 # outer has no Resources: /Fx is the page's
        page = pdf.add_blank_page(page_size=(612, 792))
        page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fo=outer, Fx=_text_form(pdf, font, text)))
        page.obj.Contents = pdf.make_stream(b"/Fo Do")
    pdf.save(tmp_path / "in.pdf")
    assert [t.strip() for t in page_texts(tmp_path / "in.pdf")] == ["Inner one", "Inner two"]
    report = strip_file(tmp_path / "in.pdf", tmp_path / "out.pdf", [0, 1], {0: "invisible_layer", 1: "invisible_layer"})
    assert page_texts(tmp_path / "out.pdf") == ["", ""]
    assert report.chars_before == {0: len("Inner one"), 1: len("Inner two")}
    assert renders(tmp_path / "out.pdf") == renders(tmp_path / "in.pdf")


def test_a_resourceless_form_is_never_given_one_callers_resources(tmp_path: Path):
    pdf = pikepdf.new()
    font = _font(pdf)
    ga = _text_form(pdf, font, b"layer A", image=_gray_image(pdf, 0))
    gc = _text_form(pdf, font, b"layer C", image=_gray_image(pdf, 200))
    shared = pdf.make_stream(b"/G Do", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
    for res in (dict(F=shared, G=ga), dict(G=ga), dict(F=shared, G=gc)):
        page = pdf.add_blank_page(page_size=(612, 792))
        page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(**res))
        page.obj.Contents = pdf.make_stream(b"/F Do" if "F" in res else b"/G Do")
    pdf.save(tmp_path / "in.pdf")
    strip_file(tmp_path / "in.pdf", tmp_path / "out.pdf", [0, 2],
               {0: "invisible_layer", 1: "born_digital", 2: "invisible_layer"})
    texts = page_texts(tmp_path / "out.pdf")
    assert texts[0] == "" and texts[1].strip() == "layer A" and texts[2] == ""
    assert renders(tmp_path / "out.pdf") == renders(tmp_path / "in.pdf")      # page 3 still draws its own image


@pytest.mark.parametrize("via", ["pattern", "soft_mask", "type3"])
def test_a_form_another_page_reaches_without_do_keeps_its_text(tmp_path: Path, via: str):
    pdf = pikepdf.new()
    font = _font(pdf)
    form = _text_form(pdf, font, b"VISIBLE WORDS", mode=None)    # visible wherever mode 0 applies
    form.Group = pikepdf.Dictionary(S=pikepdf.Name.Transparency, CS=pikepdf.Name.DeviceGray)
    a = pdf.add_blank_page(page_size=(612, 792))
    a.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=form))
    a.obj.Contents = pdf.make_stream(b"q 3 Tr /Fx Do Q")     # drawn invisibly here (mode 3 inherited)
    b = pdf.add_blank_page(page_size=(612, 792))
    if via == "pattern":
        pat = pdf.make_stream(b"/Fx Do", Type=pikepdf.Name.Pattern, PatternType=1, PaintType=1, TilingType=1,
                              BBox=[0, 0, 612, 792], XStep=612, YStep=792,
                              Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=form)))
        b.obj.Resources = pikepdf.Dictionary(Pattern=pikepdf.Dictionary(P0=pat))
        b.obj.Contents = pdf.make_stream(b"/Pattern cs /P0 scn 0 0 612 792 re f")
    elif via == "soft_mask":
        gs = pikepdf.Dictionary(Type=pikepdf.Name.ExtGState,
                                SMask=pikepdf.Dictionary(Type=pikepdf.Name.Mask, S=pikepdf.Name.Luminosity, G=form,
                                                         BC=[1]))
        b.obj.Resources = pikepdf.Dictionary(ExtGState=pikepdf.Dictionary(GS0=gs))
        b.obj.Contents = pdf.make_stream(b"q /GS0 gs 0 g 0 0 612 792 re f Q")
    else:
        proc = pdf.make_stream(b"1000 0 0 0 1000 1000 d1 /Fx Do")
        t3 = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type3, FontBBox=[0, 0, 1000, 1000],
            FontMatrix=[0.001, 0, 0, 0.001, 0, 0], CharProcs=pikepdf.Dictionary(a=proc),
            Encoding=pikepdf.Dictionary(Differences=[97, pikepdf.Name.a]), FirstChar=97, LastChar=97, Widths=[1000],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=form))))
        b.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(T3=t3))
        b.obj.Contents = pdf.make_stream(b"BT /T3 0.5 Tf 0 0 Td (a) Tj ET")
    pdf.save(tmp_path / "in.pdf")
    report = strip_file(tmp_path / "in.pdf", tmp_path / "out.pdf", [0], {0: "invisible_layer", 1: "born_digital"})
    assert report.forms_copied == 1
    assert page_texts(tmp_path / "out.pdf")[0] == ""
    assert renders(tmp_path / "out.pdf") == renders(tmp_path / "in.pdf")
    with pikepdf.open(tmp_path / "out.pdf") as out:
        kept = [o for o in out.objects if isinstance(o, pikepdf.Stream) and b"VISIBLE WORDS" in o.read_bytes()]
        assert kept, "the original form keeps its text for the page that draws it without Do"


def _type3_page_pdf(path: Path) -> Path:
    """A born-digital page whose Type 3 font has no Resources: its glyph draws the page's /Im0."""
    pdf = pikepdf.new()
    img = _gray_image(pdf, 0)
    proc = pdf.make_stream(b"1000 0 0 0 1000 1000 d1 q 1000 0 0 1000 0 0 cm /Im0 Do Q")
    t3 = pdf.make_indirect(pikepdf.Dictionary(
        Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type3, FontBBox=[0, 0, 1000, 1000],
        FontMatrix=[0.001, 0, 0, 0.001, 0, 0], CharProcs=pikepdf.Dictionary(a=proc),
        Encoding=pikepdf.Dictionary(Differences=[97, pikepdf.Name.a]), FirstChar=97, LastChar=97, Widths=[1000]))
    scan = pdf.add_blank_page(page_size=(612, 792))
    scan.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=_font(pdf)))
    scan.obj.Contents = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 72 700 Td (old layer) Tj ET")
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(T3=t3), XObject=pikepdf.Dictionary(Im0=img))
    page.obj.Contents = pdf.make_stream(b"BT /T3 100 Tf 100 300 Td (a) Tj ET")
    pdf.save(path)
    return path


def test_drop_unused_leaves_a_page_with_a_resourceless_type3_font_alone(tmp_path: Path):
    src = _type3_page_pdf(tmp_path / "in.pdf")
    strip_file(src, tmp_path / "out.pdf", [0], {0: "invisible_layer", 1: "born_digital"})
    assert renders(tmp_path / "out.pdf") == renders(src)
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert "/Im0" in pdf.pages[1].obj.Resources.XObject
        assert len(pdf.pages[0].obj.Resources.get("/Font", pikepdf.Dictionary())) == 0


def test_drop_unused_prunes_only_the_pages_it_is_given(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [scan_page(), scan_page()])
    with pikepdf.open(src) as pdf:
        strip_text(pdf, [0, 1], classes({0: "invisible_layer", 1: "invisible_layer"}))
        drop_unused(pdf, [0])
        fonts = [len(p.obj.Resources.get("/Font", pikepdf.Dictionary())) for p in pdf.pages]
        with pytest.raises(ValueError, match="out of range"):
            drop_unused(pdf, [5])
    assert fonts == [0, 1]


def _deep_chain(pdf: pikepdf.Pdf, depth: int, leaf: pikepdf.Stream) -> pikepdf.Stream:
    inner = leaf
    for _ in range(depth):
        inner = pdf.make_stream(b"/Fx Do", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792],
                                Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=inner)))
    return inner


def test_deep_forms_on_a_page_that_isnt_stripped_need_no_recursion(tmp_path: Path):
    pdf = pikepdf.new()
    font = _font(pdf)
    a = pdf.add_blank_page(page_size=(612, 792))
    a.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    a.obj.Contents = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 72 700 Td (layer) Tj ET")
    leaf = pdf.make_stream(b"0 0 1 1 re f", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 10, 10])
    b = pdf.add_blank_page(page_size=(612, 792))
    b.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=_deep_chain(pdf, 2000, leaf)))
    b.obj.Contents = pdf.make_stream(b"/Fx Do")
    report = strip_text(pdf, [0], classes({0: "invisible_layer", 1: "image_only"}))
    assert report.chars_before == {0: len("layer")}


def test_forms_nested_too_deep_on_a_stripped_page_raise_before_any_change(tmp_path: Path):
    pdf = pikepdf.new()
    font = _font(pdf)
    first = pdf.add_blank_page(page_size=(612, 792))
    first.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    first.obj.Contents = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 72 700 Td (layer) Tj ET")
    deep = pdf.add_blank_page(page_size=(612, 792))
    deep.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=_deep_chain(pdf, 1500, _text_form(pdf, font, b"x"))))
    deep.obj.Contents = pdf.make_stream(b"/Fx Do")
    before = bytes(pdf.pages[0].obj.Contents.read_bytes())
    with pytest.raises(ValueError, match="page 2 .*nested more than"):
        strip_text(pdf, [0, 1], classes({0: "invisible_layer", 1: "invisible_layer"}))
    assert bytes(pdf.pages[0].obj.Contents.read_bytes()) == before


def test_a_content_stream_that_doesnt_parse_raises_before_any_change(tmp_path: Path):
    pdf = pikepdf.new()
    font = _font(pdf)
    first = pdf.add_blank_page(page_size=(612, 792))
    first.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    first.obj.Contents = pdf.make_stream(b"BT /F1 12 Tf 3 Tr 72 700 Td (layer) Tj ET")
    broken = pdf.add_blank_page(page_size=(612, 792))
    broken.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    broken.obj.Contents = pikepdf.Array([pdf.make_stream(b"BT /F1 12 Tf 3 Tr (layer) Tj ET"), None])
    pdf.save(tmp_path / "in.pdf")
    pdf = pikepdf.open(tmp_path / "in.pdf")
    before = bytes(pdf.pages[0].obj.Contents.read_bytes())
    with pytest.raises(ValueError, match="page 2 can't be stripped safely"):
        strip_text(pdf, [0, 1], classes({0: "invisible_layer", 1: "invisible_layer"}))
    assert bytes(pdf.pages[0].obj.Contents.read_bytes()) == before


def test_a_class_for_another_page_is_refused(tmp_path: Path):
    src = build.write_pdf(tmp_path / "in.pdf", [build.PageSpec(texts=(build.TextSpec("Printed words"),)), scan_page()])
    with pikepdf.open(src) as pdf:
        before = bytes(pdf.pages[0].obj.Contents.read_bytes())
        with pytest.raises(ValueError, match="class of another page"):
            strip_text(pdf, [0], {0: FakeClass(1, "invisible_layer")})
        assert bytes(pdf.pages[0].obj.Contents.read_bytes()) == before


# --------------------------------------------------------------------------- flatten_signatures (§10.3, §11.1)


def form_renders(path: Path) -> list[bytes]:
    """Every page at 72 dpi, grayscale, with form-field widgets drawn (what a viewer shows)."""
    import numpy as np
    import pypdfium2 as pdfium

    out = []
    with _lock():
        doc = pdfium.PdfDocument(str(path))
        try:
            doc.init_forms()
            for i in range(len(doc)):
                page = doc[i]
                bm = page.render(scale=1, grayscale=True, may_draw_forms=True)
                arr = np.frombuffer(bm.buffer, dtype=np.uint8, count=bm.stride * bm.height).reshape(bm.height, bm.stride)
                out.append(arr[:, :bm.width].tobytes() + f"{bm.width}x{bm.height}".encode())
                bm.close()
                page.close()
        finally:
            doc.close()
    return out


def flatten_file(src: Path, dest: Path) -> list[str]:
    from pdfintake.pdf.strip import flatten_signatures

    with pikepdf.open(src) as pdf:
        names = flatten_signatures(pdf)
        pdf.save(dest)
    return names


def no_material(path: Path) -> None:
    from pdfintake import refusals
    from pdfintake.pdf import verify

    assert refusals.output_signature_material(path) == []
    check = verify.no_signature_material(path)
    assert check.ok, check.detail
    raw = path.read_bytes()
    for marker in (b"/ByteRange", b"/DocMDP", b"/UR3", b"/DocTimeStamp", b"/SigFlags", b"/Perms"):
        assert marker not in raw, marker


@pytest.fixture
def shells():
    pytest.importorskip("pyhanko")
    from fixtures import refusal_shells

    return refusal_shells


def test_a_pyhanko_stamp_is_drawn_where_it_was_and_no_signature_remains(shells, tmp_path: Path):
    signed = tmp_path / "signed.pdf"
    signed.write_bytes(shells.signed_visible("Signature1"))
    flat = tmp_path / "copy.pdf"
    assert flatten_file(signed, flat) == ["Signature1"]
    assert form_renders(flat) == form_renders(signed)
    assert renders(flat) == form_renders(signed)          # the stamp is page content now, not a widget
    assert renders(signed) != form_renders(signed)        # (it was drawn only as a widget before)
    no_material(flat)
    with pikepdf.open(flat) as pdf:
        assert "/AcroForm" not in pdf.Root and "/Annots" not in pdf.pages[0].obj
        assert pdf.check_pdf_syntax() == []
    from pdfintake.pdf import verify

    assert verify.renders_identical(signed, flat, {}).ok
    assert page_texts(flat)[0].startswith(page_texts(signed)[0].rstrip())


def test_certified_usage_rights_and_timestamped_files_lose_every_signature_dictionary(shells, tmp_path: Path):
    cases = {"certified": shells.certified(2), "timestamped": shells.doc_timestamp()}
    for name, data in cases.items():
        src = tmp_path / f"{name}.pdf"
        src.write_bytes(data)
        with pikepdf.open(src) as pdf:                   # a usage-rights entry beside the certification
            if name == "certified":
                pdf.Root.Perms.UR3 = pdf.make_indirect(pikepdf.Dictionary(
                    Type=pikepdf.Name.Sig, ByteRange=[0, 1, 2, 3], Contents=pikepdf.String(b"\x00" * 4)))
                pdf.save(tmp_path / "certified-ur3.pdf")
                src = tmp_path / "certified-ur3.pdf"
        flat = tmp_path / f"{name}-copy.pdf"
        names = flatten_file(src, flat)
        assert len(names) == 1, name
        no_material(flat)
        assert form_renders(flat) == form_renders(src), name


def test_a_widget_on_a_rotated_page_with_a_matrix_lands_in_the_same_place(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", rotate=90, matrix=[0, 1, -1, 0, 60, 0])
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Example.Signature"]
    assert form_renders(flat) == form_renders(src)
    assert renders(flat) != renders(src)                  # the stamp really became content
    no_material(flat)


@pytest.mark.parametrize("turn", [90, 270])
def test_a_norotate_widget_on_a_turned_page_stays_upright_as_pdfium_draws_it(tmp_path: Path, turn: int):
    src = sig_widget_pdf(tmp_path / "in.pdf", rotate=turn, flags=4 | 16)
    flat = tmp_path / "copy.pdf"
    flatten_file(src, flat)
    assert form_renders(flat) == form_renders(src)
    plain = sig_widget_pdf(tmp_path / "plain.pdf", rotate=turn)
    assert form_renders(plain) != form_renders(src)       # NoRotate really changed where it's drawn
    no_material(flat)


def test_a_noview_widget_is_drawn_because_pdfium_draws_it(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", flags=4 | 32)
    flat = tmp_path / "copy.pdf"
    flatten_file(src, flat)
    assert form_renders(flat) == form_renders(src)


def test_other_fields_and_annotations_stay_and_only_sigflags_goes(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", text_field=True, need_appearances=True, link=True)
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Example.Signature"]
    with pikepdf.open(flat) as pdf:
        acro = pdf.Root.AcroForm
        assert "/SigFlags" not in acro and acro.NeedAppearances is True
        assert [str(f.T) for f in acro.Fields] == ["Example"]
        assert [str(k.T) for k in acro.Fields[0].Kids] == ["Notes"]
        subtypes = sorted(str(a.Subtype) for a in pdf.pages[0].obj.Annots)
        assert subtypes == ["/Link", "/Widget"]
    assert form_renders(flat) == form_renders(src)
    no_material(flat)


def test_a_hidden_signature_widget_is_removed_without_being_drawn(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", flags=2 | 4)
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Example.Signature"]
    assert renders(flat) == renders(src) == form_renders(src) == form_renders(flat)
    no_material(flat)


def test_the_invisible_flag_doesnt_hide_a_widget_so_it_is_drawn(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", flags=1 | 4)
    flat = tmp_path / "copy.pdf"
    flatten_file(src, flat)
    assert form_renders(flat) == form_renders(src)
    assert renders(flat) != renders(src)


def test_the_appearance_state_chosen_by_as_is_the_one_drawn(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", states=True)
    flat = tmp_path / "copy.pdf"
    flatten_file(src, flat)
    assert form_renders(flat) == form_renders(src)
    no_material(flat)


def test_a_structure_reference_to_the_widget_goes_too(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", tagged=True)
    flat = tmp_path / "copy.pdf"
    flatten_file(src, flat)
    no_material(flat)
    with pikepdf.open(flat) as pdf:
        assert "/K" not in pdf.Root.StructTreeRoot.K[0]


def test_a_signed_widget_only_on_the_page_is_removed(shells, tmp_path: Path):
    src = tmp_path / "in.pdf"
    src.write_bytes(shells.signed_widget_not_in_fields())
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Sig1"]
    assert form_renders(flat) == form_renders(src)
    no_material(flat)


def test_old_signature_bytes_from_an_earlier_revision_dont_survive_the_save(shells, tmp_path: Path):
    src = tmp_path / "in.pdf"
    src.write_bytes(shells.signature_removed_later())
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == []
    no_material(flat)


def _with_content(src: Path, *, before: bytes = b"", after: bytes = b"") -> Path:
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        page = pdf.pages[0].obj
        page.Contents = pdf.make_stream(before + page.Contents.read_bytes() + after)
        pdf.save(src)
    return src


@pytest.mark.parametrize("before, after", [
    (b"0.5 0 0 0.5 0 0 cm 1 0 0 rg 0.6 G 4 w ", b""),          # a cm and colours the page never restores
    (b"", b" Q 0.5 0 0 0.5 0 0 cm 1 0 0 rg"),                   # a stray Q, then a cm (pdfium ignores the Q)
    (b"", b" Q Q 2 0 0 2 0 0 cm"),                              # two of them
    (b"q 0.5 0 0 0.5 0 0 cm ", b""),                            # a q the page leaves open
], ids=["unwrapped_cm", "stray_Q", "two_stray_Q", "open_q"])
def test_graphics_state_the_page_leaves_behind_never_reaches_the_stamp(tmp_path: Path, before: bytes, after: bytes):
    from pdfintake.pdf import verify

    src = _with_content(sig_widget_pdf(tmp_path / "in.pdf"), before=before, after=after)
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Example.Signature"]
    assert form_renders(flat) == form_renders(src)
    assert renders(flat) != renders(src)                  # the stamp really became content
    check = verify.renders_identical(src, flat, {})
    assert check.ok, check.detail
    no_material(flat)


@pytest.mark.parametrize("kind", ["popup_parent", "reply_irt"])
def test_a_kept_annotation_pointing_at_the_widget_carries_no_signature(tmp_path: Path, kind: str):
    src = sig_widget_pdf(tmp_path / "in.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        widget = pdf.pages[0].obj.Annots[0]
        if kind == "popup_parent":                       # a popup whose /Parent is the signature widget
            kept = pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Popup, Rect=[0, 0, 10, 10],
                                      Parent=widget, Open=False)
        else:                                            # a markup reply to the widget
            kept = pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Text, Rect=[20, 20, 40, 40],
                                      Contents=pikepdf.String("Example reply"), IRT=widget, F=4)
        pdf.pages[0].obj.Annots.append(pdf.make_indirect(kept))
        pdf.save(src)
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Example.Signature"]
    no_material(flat)
    with pikepdf.open(flat) as pdf:
        annots = pdf.pages[0].obj.Annots
        assert len(annots) == 1 and annots[0].Subtype != pikepdf.Name.Widget
        ref = annots[0].get("/Parent") if kind == "popup_parent" else annots[0].get("/IRT")
        for key in ("/V", "/FT", "/Lock", "/SV"):        # the removed widget it still reaches is no signature
            assert key not in ref, key


def test_a_signature_value_reached_only_through_a_kept_annotation_is_emptied(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        widget = pdf.pages[0].obj.Annots[0]
        widget.Lock = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name("/SigFieldLock"),
                                                           Action=pikepdf.Name.All))
        widget.SV = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name("/SV"), Ff=1))
        keeper = pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Popup, Rect=[0, 0, 10, 10],
                                    Parent=widget, Example=widget.V)  # holds the value dictionary directly too
        pdf.pages[0].obj.Annots.append(pdf.make_indirect(keeper))
        pdf.save(src)
    flat = tmp_path / "copy.pdf"
    flatten_file(src, flat)
    no_material(flat)
    with pikepdf.open(flat) as pdf:
        assert len(pdf.pages[0].obj.Annots[0].Example.keys()) == 0
    assert b"/SigFieldLock" not in flat.read_bytes()


@pytest.mark.parametrize("bbox", [None, [0, 0, 180, 0], [10, 10, 10, 10]], ids=["no_bbox", "flat_bbox", "point_bbox"])
def test_an_appearance_pdfium_cant_scale_is_placed_as_pdfium_places_it(tmp_path: Path, bbox):
    src = sig_widget_pdf(tmp_path / "in.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        n = pdf.pages[0].obj.Annots[0].AP.N
        if bbox is None:
            del n["/BBox"]
        else:
            n.BBox = pikepdf.Array(bbox)
        pdf.save(src)
    flat = tmp_path / "copy.pdf"
    assert flatten_file(src, flat) == ["Example.Signature"]
    # pdfium draws a form with no /BBox unclipped, and clips one with a flat box to nothing
    assert (renders(src) != form_renders(src)) is (bbox is None)
    assert form_renders(flat) == form_renders(src)
    no_material(flat)


def test_a_file_without_signatures_is_left_alone(tmp_path: Path):
    src = sig_widget_pdf(tmp_path / "in.pdf", sig=False, text_field=True)
    from pdfintake.pdf.strip import flatten_signatures

    with pikepdf.open(src) as pdf:
        before = pdf.Root.unparse(resolved=True), pdf.pages[0].obj.unparse(resolved=True)
        assert flatten_signatures(pdf) == []
        assert (pdf.Root.unparse(resolved=True), pdf.pages[0].obj.unparse(resolved=True)) == before


def sig_widget_pdf(path: Path, *, rotate: int = 0, matrix: list[float] | None = None, flags: int = 4,
                   sig: bool = True, text_field: bool = False, need_appearances: bool = False, link: bool = False,
                   states: bool = False, tagged: bool = False) -> Path:
    """A page with printed text and a signature field ``Example.Signature`` drawn only by its widget.

    The signature value is made up (a /ByteRange and zero /Contents): nothing here checks a signature.
    """
    pdf = pikepdf.new()
    font = _font(pdf)
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    page.obj.Contents = pdf.make_stream(b"BT /F1 18 Tf 72 700 Td (Example agreement 4521) Tj ET 0.4 g 72 72 120 40 re f")
    if rotate:
        page.obj.Rotate = rotate
    parent = pdf.make_indirect(pikepdf.Dictionary(T=pikepdf.String("Example"), Kids=pikepdf.Array()))
    annots: list[Any] = []

    def appearance(text: bytes, grey: bytes) -> pikepdf.Stream:
        ap = pdf.make_stream(grey + b" g 0 0 180 50 re f 0 g BT /F1 12 Tf 6 20 Td (" + text + b") Tj ET",
                             Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 180, 50],
                             Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)))
        if matrix is not None:
            ap.Matrix = pikepdf.Array(matrix)
        return ap

    if sig:
        value = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Sig, Filter=pikepdf.Name("/Adobe.PPKLite"),
                                                     ByteRange=[0, 10, 20, 30], Contents=pikepdf.String(b"\x00" * 16)))
        if states:
            ap = pikepdf.Dictionary(N=pikepdf.Dictionary(On=appearance(b"Signed example", b"0.8"),
                                                         Off=appearance(b"Other state", b"0.2")))
        else:
            ap = pikepdf.Dictionary(N=appearance(b"Signed by Jane Doe", b"0.8"))
        widget = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Widget, FT=pikepdf.Name.Sig, T=pikepdf.String("Signature"),
            V=value, F=flags, Rect=[300, 120, 480, 170], AP=ap, P=page.obj, Parent=parent))
        if states:
            widget.AS = pikepdf.Name("/On")
        parent.Kids.append(widget)
        annots.append(widget)
    if text_field:
        tx_ap = pdf.make_stream(b"/Tx BMC BT /F1 12 Tf 2 5 Td (Example note) Tj ET EMC", Type=pikepdf.Name.XObject,
                                Subtype=pikepdf.Name.Form, BBox=[0, 0, 200, 20],
                                Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)))
        tx = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Widget, FT=pikepdf.Name.Tx, T=pikepdf.String("Notes"),
            V=pikepdf.String("Example note"), F=4, Rect=[72, 400, 272, 420], AP=pikepdf.Dictionary(N=tx_ap),
            P=page.obj, Parent=parent))
        parent.Kids.append(tx)
        annots.append(tx)
    if link:
        annots.append(pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Link,
                                                           Rect=[72, 300, 200, 320], Border=[0, 0, 0])))
    if annots:
        page.obj.Annots = pikepdf.Array(annots)
    acro = pikepdf.Dictionary(Fields=pikepdf.Array([parent]), DA=pikepdf.String("/F1 0 Tf 0 g"),
                              DR=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)))
    if sig:
        acro.SigFlags = 3
    if need_appearances:
        acro.NeedAppearances = True
    pdf.Root.AcroForm = acro
    if tagged and sig:
        widget.StructParent = 0
        objr = pikepdf.Dictionary(Type=pikepdf.Name("/OBJR"), Obj=annots[0], Pg=page.obj)
        elem = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.StructElem, S=pikepdf.Name("/Form"), K=objr))
        tree = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.StructTreeRoot, K=pikepdf.Array([elem])))
        elem.P = tree
        pdf.Root.StructTreeRoot = tree
    pdf.save(path)
    return path
