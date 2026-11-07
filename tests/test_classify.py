"""pdf/classify.py: one synthetic page for every PageKind, and the garble ratio."""

from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest

from fixtures import build
from pdfintake.pdf import classify as classify_mod
from pdfintake.pdf.classify import PageClass, classify, garble_ratio
from pdfintake.pdf.render import PDFIUM_LOCK, Renderer

CLASS_WARNINGS = {"image_text_not_read", "garbled_text", "clip_text_kept", "existing_layer_kept",
                  "existing_layer_replaced"}


def _scan(seed: int = 0) -> build.ImageSpec:
    return build.ImageSpec(build.photo_like(306, 396, seed=seed))      # covers the whole Letter page


def _one(tmp_path: Path, spec: build.PageSpec, *, redo_ocr: bool = False, keep: bool = False,
         name: str = "p.pdf") -> PageClass:
    path = build.write_pdf(tmp_path / name, [spec])
    (pc,) = classify(path, redo_ocr=redo_ocr, keep_existing_layer=keep)
    return pc


def _mark(path: Path, *, where: str = "page") -> Path:
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        marker = pikepdf.Dictionary({"/RazorII_pdfintake": pikepdf.Dictionary(
            LastModified=pikepdf.String("D:" + "2000" + "0101" + "000000Z"), Private=pikepdf.Dictionary(Tool=pikepdf.String("example")))})
        holder = pdf.pages[0].obj if where == "page" else pdf.Root
        holder.PieceInfo = marker
        pdf.save(path)
    return path


# --------------------------------------------------------------------------- each class

def test_image_only(tmp_path: Path) -> None:
    pc = _one(tmp_path, build.PageSpec(images=(_scan(),)))
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("image_only", True, False)
    assert pc.image_coverage == pytest.approx(1.0)
    assert pc.visible_chars == pc.invisible_chars == 0 and pc.warnings == []


def test_born_digital(tmp_path: Path) -> None:
    pc = _one(tmp_path, build.PageSpec(texts=(build.TextSpec(build.LATIN_SAMPLE),)))
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("born_digital", False, False)
    assert pc.visible_chars == sum(1 for c in build.LATIN_SAMPLE if not c.isspace())
    assert pc.garble_ratio == 0.0 and pc.warnings == []


def test_brochure_photo_with_visible_caption_is_never_ocrd(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Sample caption under the photo", y=40),))
    pc = _one(tmp_path, spec)
    assert pc.kind == "born_digital" and not pc.needs_ocr and not pc.strip_old_layer
    assert pc.warnings == ["image_text_not_read"]
    # under half the page covered: no warning
    small = build.PageSpec(images=(build.ImageSpec(build.photo_like(60, 60), x=72, y=72, w=200, h=200),),
                           texts=(build.TextSpec("Sample caption", y=40),))
    assert _one(tmp_path, small, name="small.pdf").warnings == []


def test_whitespace_only_visible_text_does_not_count(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("   "), build.TextSpec(" ", y=500)))
    pc = _one(tmp_path, spec)
    assert (pc.kind, pc.needs_ocr, pc.visible_chars) == ("image_only", True, 0)
    blank = _one(tmp_path, build.PageSpec(texts=(build.TextSpec("  "),)), name="blank.pdf")
    assert blank.kind == "empty" and not blank.needs_ocr


def test_invisible_layer_is_stripped_and_read_again(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Old sample layer", mode=3),))
    pc = _one(tmp_path, spec)
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("invisible_layer", True, True)
    assert pc.invisible_chars == len("Oldsamplelayer") and pc.visible_chars == 0
    assert pc.warnings == ["existing_layer_replaced"]
    kept = _one(tmp_path, spec, keep=True, name="kept.pdf")
    assert (kept.kind, kept.needs_ocr, kept.strip_old_layer) == ("invisible_layer", False, False)
    assert kept.warnings == ["existing_layer_kept"]


def test_invisible_layer_inside_a_form(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Old sample layer", mode=3),), in_form=True)
    pc = _one(tmp_path, spec)
    assert pc.kind == "invisible_layer" and pc.needs_ocr and pc.image_coverage == pytest.approx(1.0)


def test_visible_text_inside_a_form_is_born_digital(tmp_path: Path) -> None:
    pc = _one(tmp_path, build.PageSpec(texts=(build.TextSpec("Sample text in a form"),), in_form=True))
    assert pc.kind == "born_digital" and not pc.needs_ocr


def test_our_layer_marker(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Our sample layer", mode=3),))
    path = _mark(build.write_pdf(tmp_path / "ours.pdf", [spec]))
    (pc,) = classify(path, redo_ocr=False, keep_existing_layer=False)
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer, pc.warnings) == ("our_layer", False, False, [])
    (redo,) = classify(path, redo_ocr=True, keep_existing_layer=False)
    assert (redo.kind, redo.needs_ocr, redo.strip_old_layer) == ("our_layer", True, True)
    assert redo.warnings == ["existing_layer_replaced"]
    doc = _mark(build.write_pdf(tmp_path / "doc.pdf", [spec]), where="root")
    assert classify(doc, redo_ocr=False, keep_existing_layer=False)[0].kind == "our_layer"


def test_marker_alone_never_hides_visible_text(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Printed sample words"),))
    path = _mark(build.write_pdf(tmp_path / "p.pdf", [spec]))
    (pc,) = classify(path, redo_ocr=True, keep_existing_layer=False)
    assert pc.kind == "born_digital" and not pc.needs_ocr and not pc.strip_old_layer


def test_transparent_text_is_an_old_invisible_layer(tmp_path: Path) -> None:
    # the way Scanner Pro writes its OCR: fill mode, ExtGState /ca 0 (DESIGN Q8)
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Sample hidden words", fill_alpha=0.0),))
    pc = _one(tmp_path, spec)
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("invisible_layer", True, True)
    assert pc.invisible_chars == len("Samplehiddenwords") and pc.warnings == ["existing_layer_replaced"]
    kept = _one(tmp_path, spec, keep=True, name="kept.pdf")
    assert (kept.kind, kept.needs_ocr, kept.strip_old_layer) == ("invisible_layer", False, False)
    assert kept.warnings == ["existing_layer_kept"]
    # transparent text beside a mode-3 layer, and transparent stroked text, are one old layer too
    mixed = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Layer", mode=3),
                                                      build.TextSpec("Ghost", y=600, fill_alpha=0.0)))
    assert _one(tmp_path, mixed, name="mixed.pdf").kind == "invisible_layer"
    for mode in (1, 2):
        stroked = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Ghost", mode=mode, fill_alpha=0.0),))
        assert _one(tmp_path, stroked, name=f"stroked{mode}.pdf").kind == "invisible_layer"


def test_transparent_text_is_never_taken_for_our_layer(tmp_path: Path) -> None:
    # our marker on a page whose layer isn't all mode 3: not what we write, so it is an old layer
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Layer", mode=3),
                                                     build.TextSpec("Ghost", y=600, fill_alpha=0.0)))
    path = _mark(build.write_pdf(tmp_path / "p.pdf", [spec]))
    (pc,) = classify(path, redo_ocr=False, keep_existing_layer=False)
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("invisible_layer", True, True)


@pytest.mark.parametrize("mode", [4, 5, 6])
def test_transparent_text_that_clips_is_hidden_text(tmp_path: Path, mode: int) -> None:
    # modes 4-6 add the glyphs to the clipping path, so dropping them would change what follows
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Sample hidden words", mode=mode, fill_alpha=0.0),))
    pc = _one(tmp_path, spec)
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("hidden_text", False, False)
    assert pc.warnings == ["existing_layer_kept"]
    with_layer = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Layer", mode=3),
                                                           build.TextSpec("Ghost", y=600, mode=mode, fill_alpha=0.0)))
    assert _one(tmp_path, with_layer, keep=True, name="with_layer.pdf").kind == "hidden_text"


def test_stroked_text_with_transparent_fill_is_visible(tmp_path: Path) -> None:
    # mode 1 strokes the glyphs; a fill alpha of 0 doesn't hide them
    extra = b"q /GSx gs BT /F1 24 Tf 1 Tr 72 400 Td (Outlined sample) Tj ET Q"
    path = build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(images=(_scan(),),
                                                               texts=(build.TextSpec("x", mode=3),), extra=extra)])
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        res = pdf.pages[0].obj.Resources
        res.ExtGState = pikepdf.Dictionary(GSx=pikepdf.Dictionary(Type=pikepdf.Name.ExtGState, ca=0, CA=1))
        pdf.save(path)
    (pc,) = classify(path, redo_ocr=False, keep_existing_layer=False)
    assert pc.kind == "born_digital" and not pc.needs_ocr


def test_clip_text_mode_7(tmp_path: Path) -> None:
    spec = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Sample clip text", mode=7),))
    pc = _one(tmp_path, spec)
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer) == ("clip_text", False, False)
    assert pc.warnings == ["clip_text_kept"]
    both = build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Clip", mode=7), build.TextSpec("Layer", y=600, mode=3)))
    assert _one(tmp_path, both, name="both.pdf").kind == "clip_text"


def test_empty(tmp_path: Path) -> None:
    pc = _one(tmp_path, build.PageSpec())
    assert (pc.kind, pc.needs_ocr, pc.strip_old_layer, pc.image_coverage) == ("empty", False, False, 0.0)


def test_garbled_visible_text(tmp_path: Path) -> None:
    pua = "".join(chr(0xE000 + i) for i in range(12))
    font = build.box_font(pua + "Sample ")
    path = build.fpdf_text_pdf(tmp_path / "g.pdf", [pua, "Sample " + pua[:3]], font)
    (pc,) = classify(path, redo_ocr=False, keep_existing_layer=False)
    assert pc.kind == "garbled" and not pc.needs_ocr
    assert pc.garble_ratio >= 0.05 and pc.warnings == ["garbled_text"]
    clean = build.fpdf_text_pdf(tmp_path / "c.pdf", ["Sample text"], build.box_font("Sample tx"))
    assert classify(clean, redo_ocr=False, keep_existing_layer=False)[0].kind == "born_digital"


def test_many_pages_in_order(tmp_path: Path) -> None:
    specs = [build.PageSpec(images=(_scan(),)), build.PageSpec(texts=(build.TextSpec("Sample"),)),
             build.PageSpec(), build.PageSpec(images=(_scan(1),), texts=(build.TextSpec("Old", mode=3),))]
    got = classify(build.write_pdf(tmp_path / "m.pdf", specs), redo_ocr=False, keep_existing_layer=False)
    assert [p.index for p in got] == [0, 1, 2, 3]
    assert [p.kind for p in got] == ["image_only", "born_digital", "empty", "invisible_layer"]


def test_never_ocr_a_page_with_visible_text(tmp_path: Path) -> None:
    """Across every option, a page with visible text never needs OCR nor a strip."""
    specs = [build.PageSpec(images=(_scan(),), texts=(build.TextSpec("Seen", mode=m),
                                                       build.TextSpec("Layer", y=600, mode=3)))
             for m in (0, 1, 2, 4, 5, 6)]
    path = build.write_pdf(tmp_path / "v.pdf", specs)
    for redo in (False, True):
        for keep in (False, True):
            for pc in classify(path, redo_ocr=redo, keep_existing_layer=keep):
                assert pc.kind == "born_digital" and not pc.needs_ocr and not pc.strip_old_layer


def test_warning_codes_are_record_codes() -> None:
    record = pytest.importorskip("pdfintake.record")
    codes = getattr(record, "WARNING_CODES", None)
    if codes is None:
        pytest.skip("record.WARNING_CODES isn't there yet")
    assert CLASS_WARNINGS <= set(codes)


# --------------------------------------------------------------------------- garble ratio

@pytest.mark.parametrize("text, expected", [
    ("", 0.0),
    ("   \n\t", 0.0),
    ("Sample text", 0.0),
    ("ab�d", 0.25),
    (" ab", 0.5),           # private use
    ("a\x01b\x02", 0.5),                # controls
    ("a͸", 0.5),                   # unassigned
    ("ข้อความ", 0.0),                    # Thai, marks included, is clean
])
def test_garble_ratio(text: str, expected: float) -> None:
    assert garble_ratio(text) == pytest.approx(expected)


# --------------------------------------------------------------------------- the lock

@pytest.mark.skipif(not __debug__, reason="debug assertions are off under python -O")
def test_tally_without_the_lock_is_caught(tmp_path: Path) -> None:
    with Renderer(build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(images=(_scan(),))])) as r:
        with PDFIUM_LOCK:
            page = r._page(0)
        try:
            with pytest.raises(AssertionError, match="PDFIUM_LOCK"):
                classify_mod._tally(page)                  # the page walk reached outside the lock
        finally:
            with PDFIUM_LOCK:
                page.close()
