"""pdf/verify.py: every check passes on a faithful build and fails on its planted defect."""

from __future__ import annotations

import hashlib
import io
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pikepdf
import pytest
from PIL import Image

from fixtures import build
from pdfintake.pdf import verify
from pdfintake.pdf.verify import BuildContext, Check

SAMPLE = "Example sample line 4521"
LONG_LINES = [f"Example invoice line {k:02d} lists sample items 4521" for k in range(20)]   # about 1,000 characters


# --------------------------------------------------------------------------- fixtures


def scan_pdf(path: Path, pages: int = 1, *, jpeg: bool = True, rotate: int = 0, old_layer: str | None = None,
             extra: bytes = b"", cropbox: tuple[float, float, float, float] | None = None) -> Path:
    specs = []
    for k in range(pages):
        img = build.photo_like(160, 120, seed=10 + k)
        data = build.jpeg_bytes(img) if jpeg else img
        texts = (build.TextSpec(old_layer, mode=3),) if old_layer else ()
        specs.append(build.PageSpec(images=(build.ImageSpec(data, 36, 36, 540, 720),), texts=texts, rotate=rotate,
                                    extra=extra, cropbox=cropbox))
    return build.write_pdf(path, specs)


def edit(src: Path, dest: Path, fn: Callable[[pikepdf.Pdf], Any], **save: Any) -> Path:
    """Open, change, save. pikepdf's own XMP version fix is off, as in shrink."""
    save.setdefault("fix_metadata_version", False)
    with pikepdf.open(src) as pdf:
        fn(pdf)
        pdf.save(dest, deterministic_id=True, **save)
    return dest


def _layer_ops(lines: list[tuple[str, float, float]], size: float) -> bytes:
    parts = [b"q BT /FLay %g Tf 3 Tr" % size]
    x0 = y0 = 0.0
    for text, x, y in lines:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        parts.append(b"%g %g Td (%s) Tj" % (x - x0, y - y0, escaped.encode("latin-1")))
        x0, y0 = x, y
    parts.append(b"ET Q")
    return b" ".join(parts)


def add_layer(src: Path, dest: Path, layers: dict[int, list[tuple[str, float, float]]], *, size: float = 8.0,
              **save: Any) -> Path:
    """A stand-in for the text layer writer: invisible Helvetica text appended to each page."""

    def apply(pdf: pikepdf.Pdf) -> None:
        font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                    BaseFont=pikepdf.Name.Helvetica,
                                                    Encoding=pikepdf.Name.WinAnsiEncoding))
        for index, lines in layers.items():
            page = pdf.pages[index]
            res = page.obj.get("/Resources")
            res = pikepdf.Dictionary({k: v for k, v in res.items()}) if res is not None else pikepdf.Dictionary()
            fonts = pikepdf.Dictionary({k: v for k, v in res.get("/Font", pikepdf.Dictionary()).items()})
            fonts["/FLay"] = font
            res["/Font"] = fonts
            page.obj.Resources = res
            page.contents_add(pdf.make_stream(_layer_ops(lines, size)), prepend=False)

    return edit(src, dest, apply, **save)


def lines_at(texts: list[str], *, x: float = 60.0, top: float = 700.0, step: float = 12.0) -> list[tuple[str, float, float]]:
    return [(t, x, top - k * step) for k, t in enumerate(texts)]


def first_image(pdf: pikepdf.Pdf, page: int = 0) -> pikepdf.Stream:
    return pdf.pages[page].obj.Resources.XObject.Im0


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def good(tmp_path: Path) -> tuple[Path, Path]:
    """A one-page scan and its faithful build (an invisible layer added, saved with object streams)."""
    ref = scan_pdf(tmp_path / "ref.pdf")
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at([SAMPLE])},
                      object_stream_mode=pikepdf.ObjectStreamMode.generate)
    return ref, built


# --------------------------------------------------------------------------- text_retention


def test_retention_passes_when_the_layer_comes_back(good):
    _ref, built = good
    checks = verify.text_retention(built, {0: SAMPLE})
    assert [c.ok for c in checks] == [True]
    assert checks[0].name == "text_retention" and checks[0].page is None
    assert checks[0].detail.startswith("1.000 document")


def test_retention_fails_on_zero_ocr_text(good):
    _ref, built = good
    checks = verify.text_retention(built, {0: "", 1: "   "})
    assert len(checks) == 1 and not checks[0].ok and "no OCR text" in checks[0].detail
    assert not verify.text_retention(built, {})[0].ok


def test_retention_fails_when_the_layer_is_truncated(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf")
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at(LONG_LINES[:10])})
    checks = verify.text_retention(built, {0: "\n".join(LONG_LINES)})
    assert not checks[0].ok
    assert [c.page for c in checks[1:]] == [1] and not checks[1].ok


def test_retention_fails_a_page_below_90_percent_even_when_the_document_passes(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", pages=2)
    short = "Sample words 4521 and more items"                      # 28 characters without spaces
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at(LONG_LINES), 1: lines_at(["Sample words 4521"])})
    checks = verify.text_retention(built, {0: "\n".join(LONG_LINES), 1: short})
    doc = checks[0]
    assert float(doc.detail.split()[0]) >= 0.98
    assert not doc.ok and "page 2" in doc.detail
    assert [(c.page, c.ok) for c in checks[1:]] == [(2, False)]


def test_a_sparse_page_has_no_ratio_of_its_own(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", pages=2)
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at(LONG_LINES), 1: lines_at(["Item 45"])})
    checks = verify.text_retention(built, {0: "\n".join(LONG_LINES), 1: "Item 4521"})   # 8 chars, 6 kept
    assert [c.ok for c in checks] == [True]


def test_retention_reads_only_inside_the_cropbox(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", cropbox=(0, 0, 612, 400))
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at([SAMPLE], top=700)})   # above the CropBox
    assert not verify.text_retention(built, {0: SAMPLE})[0].ok
    inside = add_layer(ref, tmp_path / "inside.pdf", {0: lines_at([SAMPLE], top=300)})
    assert verify.text_retention(inside, {0: SAMPLE})[0].ok


def test_retention_handles_thai_marks(tmp_path: Path):
    text = [build.THAI_SAMPLE, "Example 4521"]
    font = build.box_font("".join(text))
    built = build.fpdf_text_pdf(tmp_path / "th.pdf", text, font, invisible=True, size=10)
    checks = verify.text_retention(built, {0: "\n".join(text)})
    assert checks[0].ok and checks[0].detail.startswith("1.000")


def test_retention_names_a_page_missing_from_the_build(good):
    _ref, built = good
    checks = verify.text_retention(built, {0: SAMPLE, 4: "Example sample words on a lost page 4521"})
    assert not checks[0].ok and any(c.page == 5 and "not in the built file" in c.detail for c in checks)


# --------------------------------------------------------------------------- off_page_glyphs


def test_no_off_page_glyphs_on_a_good_build(good):
    _ref, built = good
    check = verify.off_page_glyphs(built)
    assert check.ok and check.name == "off_page_glyphs"


def test_an_off_page_glyph_fails(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf")
    built = add_layer(ref, tmp_path / "built.pdf", {0: [(SAMPLE, 60, 700), ("Lost 4521", -400, 300)]})
    check = verify.off_page_glyphs(built)
    assert not check.ok and check.page == 1 and check.detail.startswith("8 character(s)")


def test_a_glyph_straddling_the_edge_is_on_the_page(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf")
    built = add_layer(ref, tmp_path / "built.pdf", {0: [("W", 605, 300)]}, size=12)    # 605 to 616 pt
    assert verify.off_page_glyphs(built).ok


def test_off_page_glyphs_checks_only_the_pages_asked(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", pages=2)
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at([SAMPLE]), 1: [("Theirs", 60, 1200)]})
    assert verify.off_page_glyphs(built, pages={0}).ok
    whole = verify.off_page_glyphs(built)
    assert not whole.ok and whole.page == 2
    assert not verify.off_page_glyphs(built, pages={7}).ok


# --------------------------------------------------------------------------- renders_identical


def test_renders_identical_on_a_good_build(good):
    ref, built = good
    check = verify.renders_identical(ref, built, {})
    assert check.ok and check.detail == "1/1 pages"


def test_a_shifted_content_stream_fails(good, tmp_path: Path):
    ref, built = good
    shifted = edit(built, tmp_path / "shifted.pdf",
                   lambda pdf: pdf.pages[0].contents_add(pdf.make_stream(b"1 0 0 1 3 0 cm"), prepend=True))
    check = verify.renders_identical(ref, shifted, {})
    assert not check.ok and check.page == 1 and "pixel(s) differ" in check.detail


def test_renders_use_the_turn_for_each_page(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", pages=2)

    def turn_first(pdf: pikepdf.Pdf) -> None:
        pdf.pages[0].obj.Rotate = 90

    turned = edit(ref, tmp_path / "turned.pdf", turn_first)
    assert verify.renders_identical(ref, turned, {0: 90}).ok
    wrong = verify.renders_identical(ref, turned, {})
    assert not wrong.ok and wrong.page == 1
    assert not verify.renders_identical(ref, turned, {0: 45}).ok


def test_a_cleared_flag_compares_against_the_turned_reference(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", rotate=90)

    def clear(pdf: pikepdf.Pdf) -> None:
        del pdf.pages[0].obj["/Rotate"]

    cleared = edit(ref, tmp_path / "cleared.pdf", clear)
    assert verify.renders_identical(ref, cleared, {0: 270}).ok
    assert not verify.renders_identical(ref, cleared, {}).ok


def test_a_different_page_count_fails_renders(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", pages=2)
    one = scan_pdf(tmp_path / "one.pdf", pages=1)
    assert not verify.renders_identical(ref, one, {}).ok


# --------------------------------------------------------------------------- images_identical


def test_images_identical_on_a_good_build(good):
    ref, built = good
    check = verify.images_identical(ref, built)
    assert check.ok and check.detail == "1 image(s) on 1 page(s)"


def _flip_dct_byte(pdf: pikepdf.Pdf) -> None:
    img = first_image(pdf)
    data = bytearray(img.read_raw_bytes())
    data[len(data) // 2] ^= 0x55
    img.write(bytes(data), filter=pikepdf.Name.DCTDecode)


def test_a_changed_image_byte_fails(good, tmp_path: Path):
    ref, built = good
    bad = edit(built, tmp_path / "bad.pdf", _flip_dct_byte)
    check = verify.images_identical(ref, bad)
    assert not check.ok and check.page == 1 and "DCTDecode" in check.detail


def test_a_jpeg_re_encoded_in_place_fails(good, tmp_path: Path):
    ref, built = good

    def reencode(pdf: pikepdf.Pdf) -> None:
        img = first_image(pdf)
        with Image.open(io.BytesIO(bytes(img.read_raw_bytes()))) as decoded:
            out = io.BytesIO()
            decoded.save(out, format="JPEG", quality=80)
        img.write(out.getvalue(), filter=pikepdf.Name.DCTDecode)

    bad = edit(built, tmp_path / "bad.pdf", reencode)
    assert not verify.images_identical(ref, bad).ok


def test_a_flate_image_whose_samples_changed_fails(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", jpeg=False)

    def change(pdf: pikepdf.Pdf) -> None:
        img = first_image(pdf)
        samples = bytearray(img.read_bytes())
        samples[100] ^= 0x01
        img.write(zlib.compress(bytes(samples)), filter=pikepdf.Name.FlateDecode)

    bad = edit(ref, tmp_path / "bad.pdf", change)
    check = verify.images_identical(ref, bad)
    assert not check.ok and "decoded samples" in check.detail


def _up_predicted(samples: bytes, row_bytes: int) -> bytes:
    rows = [samples[i:i + row_bytes] for i in range(0, len(samples), row_bytes)]
    out = bytearray()
    prev = bytes(row_bytes)
    for row in rows:
        out.append(2)
        out += bytes((a - b) & 0xFF for a, b in zip(row, prev))
        prev = row
    return zlib.compress(bytes(out), 9)


def test_lossless_recompression_keeps_images_identical(tmp_path: Path):
    ref = scan_pdf(tmp_path / "ref.pdf", jpeg=False)

    def predict(pdf: pikepdf.Pdf) -> None:
        img = first_image(pdf)
        samples = bytes(img.read_bytes())
        width = int(img.Width)
        img.write(_up_predicted(samples, width * 3), filter=pikepdf.Name.FlateDecode,
                  decode_parms=pikepdf.Dictionary(Predictor=12, Colors=3, BitsPerComponent=8, Columns=width))

    recompressed = edit(ref, tmp_path / "pred.pdf", predict, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    with pikepdf.open(recompressed) as pdf:
        assert first_image(pdf).read_raw_bytes() != b""
    assert verify.images_identical(ref, recompressed).ok
    assert verify.renders_identical(ref, recompressed, {}).ok


def test_only_drawn_images_count(good, tmp_path: Path):
    ref, built = good

    def add_unused(pdf: pikepdf.Pdf) -> None:
        extra = pdf.make_stream(build.jpeg_bytes(build.photo_like(40, 30, seed=99)), Type=pikepdf.Name.XObject,
                                Subtype=pikepdf.Name.Image, Width=40, Height=30, ColorSpace=pikepdf.Name.DeviceRGB,
                                BitsPerComponent=8, Filter=pikepdf.Name.DCTDecode)
        pdf.pages[0].obj.Resources.XObject["/Unused"] = extra

    with_unused = edit(ref, tmp_path / "unused.pdf", add_unused)
    assert verify.images_identical(with_unused, built).ok           # the unused image was dropped: fine


def test_an_image_moved_into_a_form_is_still_the_same_image(good, tmp_path: Path):
    ref, _built = good

    def wrap(pdf: pikepdf.Pdf) -> None:
        page = pdf.pages[0]
        form = pdf.make_stream(bytes(page.obj.Contents.read_bytes()), Type=pikepdf.Name.XObject,
                               Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792])
        form.Resources = page.obj.Resources
        page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Fx=form))
        page.obj.Contents = pdf.make_stream(b"q /Fx Do Q")

    wrapped = edit(ref, tmp_path / "wrapped.pdf", wrap)
    assert verify.images_identical(ref, wrapped).ok
    assert verify.renders_identical(ref, wrapped, {}).ok


def test_a_changed_inline_image_fails(tmp_path: Path):
    inline = b"q 20 0 0 20 50 50 cm BI /W 2 /H 2 /CS /G /BPC 8 ID \x10\x20\x30\x40 EI Q"
    ref = scan_pdf(tmp_path / "ref.pdf", extra=inline)
    same = add_layer(ref, tmp_path / "same.pdf", {0: lines_at([SAMPLE])})
    assert verify.images_identical(ref, same).ok
    other = scan_pdf(tmp_path / "other.pdf", extra=inline.replace(b"\x10\x20\x30\x40", b"\x10\x20\x30\x41"))
    check = verify.images_identical(ref, other)
    assert not check.ok and "inline" in check.detail


def test_a_changed_colour_space_fails(good, tmp_path: Path):
    ref, built = good

    def recolour(pdf: pikepdf.Pdf) -> None:
        first_image(pdf).ColorSpace = pikepdf.Array([pikepdf.Name.CalRGB, pikepdf.Dictionary(WhitePoint=[0.95, 1, 1.09])])

    assert not verify.images_identical(ref, edit(built, tmp_path / "cs.pdf", recolour)).ok


# --------------------------------------------------------------------------- passthrough_identical


@dataclass
class FakeAssembled:
    """Stands in for intake.assemble.AssembledPage."""

    source: str
    route: str
    orientation: int
    bytes: int
    embedded_sha256: str
    source_sha256: str
    warnings: list[str] = field(default_factory=list)


def _jpeg_source(tmp_path: Path) -> tuple[Path, bytes]:
    data = build.jpeg_bytes(build.photo_like(160, 120, seed=21))
    return build.write_pdf(tmp_path / "assembled.pdf", [build.PageSpec(images=(build.ImageSpec(data),))]), data


def test_passthrough_jpeg_bytes_survive(tmp_path: Path):
    assembled, data = _jpeg_source(tmp_path)
    built = add_layer(assembled, tmp_path / "built.pdf", {0: lines_at([SAMPLE])},
                      object_stream_mode=pikepdf.ObjectStreamMode.generate)
    pages = [FakeAssembled("a.jpg", "passthrough", 6, len(data), sha(data), sha(data))]
    check = verify.passthrough_identical(built, pages)
    assert check.ok and check.detail.startswith("1 JPEG(s)")


def test_a_re_encoded_passthrough_jpeg_fails(tmp_path: Path):
    assembled, data = _jpeg_source(tmp_path)

    def reencode(pdf: pikepdf.Pdf) -> None:
        img = first_image(pdf)
        with Image.open(io.BytesIO(data)) as decoded:
            out = io.BytesIO()
            decoded.save(out, format="JPEG", quality=95)
        img.write(out.getvalue(), filter=pikepdf.Name.DCTDecode)

    bad = edit(assembled, tmp_path / "bad.pdf", reencode)
    pages = [FakeAssembled("a.jpg", "passthrough", 1, len(data), sha(data), sha(data))]
    check = verify.passthrough_identical(bad, pages)
    assert not check.ok and check.page == 1


def test_passthrough_needs_the_assembly_to_have_been_faithful(tmp_path: Path):
    assembled, data = _jpeg_source(tmp_path)
    pages = [FakeAssembled("a.jpg", "passthrough", 1, len(data), sha(data), sha(b"other"))]
    assert not verify.passthrough_identical(assembled, pages).ok
    assert not verify.passthrough_identical(assembled, pages * 2).ok        # page count


def _idat_pdf(path: Path, idat: bytes, width: int, height: int) -> Path:
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(400, 300))
    img = pdf.make_stream(idat, Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=width, Height=height,
                          ColorSpace=pikepdf.Name.DeviceRGB, BitsPerComponent=8, Filter=pikepdf.Name.FlateDecode,
                          DecodeParms=pikepdf.Dictionary(Predictor=15, Colors=3, BitsPerComponent=8, Columns=width))
    page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=img))
    page.obj.Contents = pdf.make_stream(b"q 400 0 0 300 0 0 cm /Im0 Do Q")
    pdf.save(path, deterministic_id=True)
    return path


def test_png_idat_bytes_or_a_lossless_recompression_pass(tmp_path: Path):
    width, height = 40, 30
    rows = build.png_samples(width, height, color_type=2, bit_depth=8, seed=4)
    filtered = b"".join(b"\x00" + bytes(r) for r in rows)
    idat = zlib.compress(filtered, 6)
    assembled = _idat_pdf(tmp_path / "assembled.pdf", idat, width, height)
    pages = [FakeAssembled("a.png", "png_idat", 1, len(idat), sha(idat), sha(idat))]
    same = verify.passthrough_identical(assembled, pages)
    assert same.ok and "1 PNG IDAT" in same.detail
    recompressed = _idat_pdf(tmp_path / "re.pdf", zlib.compress(filtered, 1), width, height)
    check = verify.passthrough_identical(recompressed, pages)
    assert check.ok and "recompressed losslessly" in check.detail
    assert verify.images_identical(assembled, recompressed).ok


def test_lossless_and_encoded_routes_are_not_byte_checked(tmp_path: Path):
    assembled, data = _jpeg_source(tmp_path)
    pages = [FakeAssembled("a.heic", "encoded_once", 1, len(data), sha(data), sha(b"heic source"))]
    assert verify.passthrough_identical(assembled, pages).ok


# --------------------------------------------------------------------------- upright


def test_a_touched_page_left_turned_fails_but_an_untouched_landscape_page_passes(tmp_path: Path):
    pdf = build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(rotate=90), build.PageSpec(rotate=90),
                                               build.PageSpec()])
    bad = verify.upright(pdf, frozenset({0, 2}))
    assert not bad.ok and bad.page == 1 and "90" in bad.detail
    assert verify.upright(pdf, frozenset({2})).ok                 # page 2 keeps its /Rotate 90 untouched
    assert verify.upright(pdf, frozenset()).ok
    assert not verify.upright(pdf, frozenset({5})).ok


def test_an_inherited_rotate_counts(tmp_path: Path):
    def inherit(doc: pikepdf.Pdf) -> None:
        doc.Root.Pages.Rotate = 180

    pdf = edit(build.write_pdf(tmp_path / "p.pdf", 1), tmp_path / "inherited.pdf", inherit)
    assert not verify.upright(pdf, frozenset({0})).ok


# --------------------------------------------------------------------------- no_jbig2_written


def _add_jbig2(pdf: pikepdf.Pdf, payload: bytes = b"\x00\x00\x00\x30example jbig2 payload") -> None:
    stream = pdf.make_stream(payload, Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=8, Height=8,
                             ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=1, Filter=pikepdf.Name.JBIG2Decode)
    pdf.pages[0].obj.Resources.XObject["/Jb0"] = stream


def test_a_jbig2_stream_added_fails(good, tmp_path: Path):
    ref, built = good
    assert verify.no_jbig2_written(ref, built).ok
    bad = edit(built, tmp_path / "bad.pdf", _add_jbig2)
    check = verify.no_jbig2_written(ref, bad)
    assert not check.ok and check.detail.startswith("1 JBIG2")


def test_a_jbig2_input_passes_through(tmp_path: Path):
    ref = edit(scan_pdf(tmp_path / "s.pdf"), tmp_path / "ref.pdf", _add_jbig2)
    built = add_layer(ref, tmp_path / "built.pdf", {0: lines_at([SAMPLE])},
                      object_stream_mode=pikepdf.ObjectStreamMode.generate)
    check = verify.no_jbig2_written(ref, built)
    assert check.ok and "1 passed through" in check.detail
    changed = edit(built, tmp_path / "changed.pdf", lambda pdf: _add_jbig2(pdf, b"\x00\x00\x00\x30another payload"))
    assert not verify.no_jbig2_written(ref, changed).ok


# --------------------------------------------------------------------------- no_signature_material


def test_a_byte_range_added_fails(good, tmp_path: Path):
    _ref, built = good
    assert verify.no_signature_material(built).ok

    def plant(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Planted = pdf.make_indirect(pikepdf.Dictionary(ByteRange=[0, 10, 20, 30]))

    check = verify.no_signature_material(edit(built, tmp_path / "bad.pdf", plant))
    assert not check.ok and "/ByteRange" in check.detail


def _sig_field(pdf: pikepdf.Pdf, *, signed: bool) -> None:
    field_dict = pikepdf.Dictionary(FT=pikepdf.Name.Sig, T=pikepdf.String("Signature1"))
    if signed:
        field_dict.V = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Sig, Filter=pikepdf.Name("/Adobe.PPKLite"),
                                                            Contents=pikepdf.String(b"\x00" * 8)))
    field_obj = pdf.make_indirect(field_dict)
    pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([field_obj]))


def test_an_empty_signature_field_passes_and_a_signed_one_fails(good, tmp_path: Path):
    _ref, built = good
    assert verify.no_signature_material(edit(built, tmp_path / "empty.pdf", lambda p: _sig_field(p, signed=False))).ok
    signed = verify.no_signature_material(edit(built, tmp_path / "signed.pdf", lambda p: _sig_field(p, signed=True)))
    assert not signed.ok and "signed /Sig field" in signed.detail and "Sig dictionary" in signed.detail


def test_a_pyhanko_signature_is_signature_material(tmp_path: Path):
    pytest.importorskip("pyhanko")
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import signers

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)       # throwaway, never stored
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Example Test Signer"),
                      x509.NameAttribute(NameOID.EMAIL_ADDRESS, "signer@example.com")])
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(1).not_valid_before(start).not_valid_after(start + datetime.timedelta(days=365))
            .sign(key, hashes.SHA256()))
    (tmp_path / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption()))
    (tmp_path / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    signer = signers.SimpleSigner.load(str(tmp_path / "key.pem"), str(tmp_path / "cert.pem"), key_passphrase=None)
    out = io.BytesIO()
    signers.sign_pdf(IncrementalPdfFileWriter(io.BytesIO(scan_pdf(tmp_path / "s.pdf").read_bytes())),
                     signers.PdfSignatureMetadata(field_name="Signature1"), signer=signer, output=out)
    (tmp_path / "signed.pdf").write_bytes(out.getvalue())
    check = verify.no_signature_material(tmp_path / "signed.pdf")
    assert not check.ok and "/ByteRange" in check.detail and "signed /Sig field" in check.detail


def test_perms_fails(good, tmp_path: Path):
    _ref, built = good

    def perms(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Perms = pikepdf.Dictionary(DocMDP=pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Sig)))

    check = verify.no_signature_material(edit(built, tmp_path / "perms.pdf", perms))
    assert not check.ok and "/Perms" in check.detail


# --------------------------------------------------------------------------- metadata_kept


XMP = (b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
       b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
       b'<rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/" pdf:PDFVersion="1.4" '
       b'pdf:Producer="Example Producer"/>\n'
       b'</rdf:RDF></x:xmpmeta>\n' + b" " * 200 + b'\n<?xpacket end="w"?>')


def _with_meta(pdf: pikepdf.Pdf) -> None:
    pdf.trailer.Info = pdf.make_indirect(pikepdf.Dictionary(Title=pikepdf.String("Example title"),
                                                            Keywords=pikepdf.String("sample, example"),
                                                            Producer=pikepdf.String("Example Producer")))
    pdf.Root.Metadata = pdf.make_stream(XMP, Type=pikepdf.Name.Metadata, Subtype=pikepdf.Name.XML)


@pytest.fixture
def meta_ref(tmp_path: Path) -> Path:
    return edit(scan_pdf(tmp_path / "s.pdf"), tmp_path / "meta.pdf", _with_meta)


def test_metadata_kept_on_a_good_build(meta_ref: Path, tmp_path: Path):
    built = add_layer(meta_ref, tmp_path / "built.pdf", {0: lines_at([SAMPLE])},
                      object_stream_mode=pikepdf.ObjectStreamMode.generate)
    check = verify.metadata_kept(meta_ref, built, frozenset())
    assert check.ok and check.detail == "Info (3 key(s)) and XMP kept"


def test_info_keywords_lost_fails(meta_ref: Path, tmp_path: Path):
    def lose(pdf: pikepdf.Pdf) -> None:
        del pdf.trailer.Info["/Keywords"]

    check = verify.metadata_kept(meta_ref, edit(meta_ref, tmp_path / "bad.pdf", lose), frozenset())
    assert not check.ok and check.detail == "Info Keywords dropped"


def test_xmp_re_serialised_fails(meta_ref: Path, tmp_path: Path):
    def reserialise(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Metadata.write(re.sub(rb">\s+<", b"><", XMP))

    check = verify.metadata_kept(meta_ref, edit(meta_ref, tmp_path / "bad.pdf", reserialise), frozenset())
    assert not check.ok and check.detail == "XMP changed"


def test_a_same_length_pdfversion_patch_matching_the_header_is_accepted(meta_ref: Path, tmp_path: Path):
    def patch(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Metadata.write(XMP.replace(b'pdf:PDFVersion="1.4"', b'pdf:PDFVersion="1.7"'))

    built = edit(meta_ref, tmp_path / "patched.pdf", patch, force_version="1.7")
    check = verify.metadata_kept(meta_ref, built, frozenset())
    assert check.ok and "patched in place to 1.7" in check.detail
    wrong = edit(meta_ref, tmp_path / "wrong.pdf", patch, force_version="1.6")
    assert not verify.metadata_kept(meta_ref, wrong, frozenset()).ok
    longer = edit(meta_ref, tmp_path / "longer.pdf",
                  lambda pdf: pdf.Root.Metadata.write(XMP.replace(b'"1.4"', b'"1.10"')), force_version="1.7")
    assert not verify.metadata_kept(meta_ref, longer, frozenset()).ok


def test_allowed_names_permit_their_changes(meta_ref: Path, tmp_path: Path):
    def producer(pdf: pikepdf.Pdf) -> None:
        pdf.trailer.Info.Producer = pikepdf.String("pdfintake 0.1.0")
        pdf.trailer.Info.CreationDate = pikepdf.String("D:" + build.FIXED_DATE.strftime("%Y%m%d%H%M%S") + "Z")

    built = edit(meta_ref, tmp_path / "p.pdf", producer)
    assert not verify.metadata_kept(meta_ref, built, frozenset()).ok
    assert verify.metadata_kept(meta_ref, built, frozenset({"Info/Producer", "Info/CreationDate"})).ok
    assert verify.metadata_kept(meta_ref, built, frozenset({"Producer", "/CreationDate"})).ok
    assert verify.metadata_kept(meta_ref, built, frozenset({"Info"})).ok
    reser = edit(meta_ref, tmp_path / "x.pdf", lambda pdf: pdf.Root.Metadata.write(XMP.replace(b"\n", b"\r\n")))
    assert verify.metadata_kept(meta_ref, reser, frozenset({"XMP"})).ok


def test_xmp_dropped_or_added_fails(meta_ref: Path, tmp_path: Path):
    def drop(pdf: pikepdf.Pdf) -> None:
        del pdf.Root["/Metadata"]

    dropped = edit(meta_ref, tmp_path / "d.pdf", drop)
    assert verify.metadata_kept(meta_ref, dropped, frozenset()).detail == "XMP dropped"
    assert verify.metadata_kept(dropped, meta_ref, frozenset()).detail == "XMP added"


# --------------------------------------------------------------------------- structure_ok


def test_structure_ok_on_a_good_build(good):
    _ref, built = good
    check = verify.structure_ok(built)
    assert check.ok and "1 page(s)" in check.detail


def test_a_broken_xref_fails_structure(good, tmp_path: Path):
    _ref, built = good
    data = built.read_bytes()
    m = list(re.finditer(rb"startxref\s+(\d+)", data))[-1]
    broken = data[:m.start(1)] + str(int(m.group(1)) + 7).encode() + data[m.end(1):]
    (tmp_path / "broken.pdf").write_bytes(broken)
    check = verify.structure_ok(tmp_path / "broken.pdf")
    assert not check.ok and "qpdf reports" in check.detail


def test_a_file_that_isnt_a_pdf_fails_every_check_without_raising(good, tmp_path: Path):
    ref, _built = good
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a pdf at all")
    ctx = BuildContext(reference=ref, built=junk, turns={}, touched=frozenset({0}), expected={0: SAMPLE},
                       assembled=None, allowed_meta_changes=frozenset())
    checks = verify.verify_build(ctx)
    assert checks and not any(c.ok for c in checks)
    assert all("could not run" in c.detail for c in checks)


# --------------------------------------------------------------------------- text_layer_identical


def test_text_layer_identical_for_a_copy_with_other_images(good, tmp_path: Path):
    _ref, built = good

    def recompress(pdf: pikepdf.Pdf) -> None:
        img = first_image(pdf)
        with Image.open(io.BytesIO(bytes(img.read_raw_bytes()))) as decoded:
            out = io.BytesIO()
            decoded.save(out, format="JPEG", quality=60)
        img.write(out.getvalue(), filter=pikepdf.Name.DCTDecode)

    smaller = edit(built, tmp_path / "smaller.pdf", recompress)
    check = verify.text_layer_identical(built, smaller)
    assert check.ok and check.name == "text_layer_identical"


def test_text_layer_differs_when_text_or_positions_change(good, tmp_path: Path):
    ref, built = good
    other = add_layer(ref, tmp_path / "other.pdf", {0: lines_at(["Example other line 4521"])})
    assert verify.text_layer_identical(built, other).detail == "page 1: the text differs"
    moved = add_layer(ref, tmp_path / "moved.pdf", {0: lines_at([SAMPLE], x=61)})
    assert verify.text_layer_identical(built, moved).detail == "page 1: character positions differ"


# --------------------------------------------------------------------------- strip, then a new layer, then verify_build


def _ctx(ref: Path, built: Path, **kw: Any) -> BuildContext:
    base = dict(reference=ref, built=built, turns={}, touched=frozenset({0}), expected={0: SAMPLE}, assembled=None,
                allowed_meta_changes=frozenset())
    base.update(kw)
    return BuildContext(**base)


def test_verify_build_passes_a_faithful_rebuild_of_an_old_layer(tmp_path: Path):
    from pdfintake.pdf.strip import drop_unused, strip_text

    @dataclass
    class Cls:
        index: int
        kind: str

    ref = edit(scan_pdf(tmp_path / "s.pdf", old_layer="Old example layer 4521"), tmp_path / "ref.pdf", _with_meta)
    with pikepdf.open(ref) as pdf:
        strip_text(pdf, [0], {0: Cls(0, "invisible_layer")})
        drop_unused(pdf)
        pdf.save(tmp_path / "stripped.pdf", fix_metadata_version=False)
    built = add_layer(tmp_path / "stripped.pdf", tmp_path / "built.pdf", {0: lines_at([SAMPLE])},
                      object_stream_mode=pikepdf.ObjectStreamMode.generate)
    checks = verify.verify_build(_ctx(ref, built))
    assert [c.name for c in checks] == ["text_retention", "off_page_glyphs", "renders_identical", "images_identical",
                                        "upright", "no_jbig2_written", "no_signature_material", "metadata_kept",
                                        "structure_ok"]
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_verify_build_runs_passthrough_for_image_inputs_and_skips_retention_without_a_layer(tmp_path: Path):
    assembled, data = _jpeg_source(tmp_path)
    shrunk = edit(assembled, tmp_path / "shrunk.pdf", lambda pdf: None, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    pages = [FakeAssembled("a.jpg", "passthrough", 1, len(data), sha(data), sha(data))]
    checks = verify.verify_build(_ctx(assembled, shrunk, expected={}, touched=frozenset(), assembled=pages))
    names = [c.name for c in checks]
    assert "text_retention" not in names and "passthrough_identical" in names
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_verify_build_reports_each_planted_defect(good, tmp_path: Path):
    ref, built = good
    bad = edit(built, tmp_path / "bad.pdf", _flip_dct_byte)
    failed = {c.name for c in verify.verify_build(_ctx(ref, bad)) if not c.ok}
    assert "images_identical" in failed and "text_retention" not in failed
    turned = edit(built, tmp_path / "turned.pdf", lambda pdf: setattr(pdf.pages[0].obj, "Rotate", 90))
    failed = {c.name for c in verify.verify_build(_ctx(ref, turned, turns={0: 90})) if not c.ok}
    assert failed == {"upright"}


def test_every_check_has_a_name_from_the_closed_list(good):
    ref, built = good
    checks = verify.verify_build(_ctx(ref, built))
    assert {c.name for c in checks} <= set(verify.CHECK_NAMES)
    assert all(isinstance(c, Check) and isinstance(c.detail, str) and c.detail for c in checks)
    assert SAMPLE not in " ".join(c.detail for c in checks)        # details never quote document text


def test_ratios_are_shown_rounded_down():
    assert verify._floor3(0.98) == "0.980"
    assert verify._floor3(0.97999) == "0.979"
    assert verify._floor3(1.0) == "1.000"


def test_pikepdfs_default_save_re_serialises_xmp_and_is_caught(meta_ref: Path, tmp_path: Path):
    # pikepdf's save(fix_metadata_version=True), the default, rewrites the whole packet even when the
    # version doesn't change: every save on the build path must pass fix_metadata_version=False.
    resaved = edit(meta_ref, tmp_path / "resaved.pdf", lambda pdf: None, fix_metadata_version=True)
    check = verify.metadata_kept(meta_ref, resaved, frozenset())
    assert not check.ok and check.detail == "XMP changed"
    kept = edit(meta_ref, tmp_path / "kept.pdf", lambda pdf: None)
    assert verify.metadata_kept(meta_ref, kept, frozenset()).ok


# --------------------------------------------------------------------------- review fixes

PLANTED = "IGNORE_PREVIOUS_INSTRUCTIONS"


def _blank_pdf(path: Path, content: bytes = b"", **save: Any) -> Path:
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(200, 200))
    page.obj.Contents = pdf.make_stream(content)
    pdf.save(path, **save)
    return path


def test_details_never_carry_qpdf_text_names_or_paths_from_the_file(tmp_path: Path):
    marked = tmp_path / f"folder {PLANTED}"
    marked.mkdir()
    dup = _blank_pdf(marked / "dup.pdf", f"/P0 <</{PLANTED} 1 /{PLANTED} 2>> BDC EMC".encode())
    check = verify.structure_ok(dup)
    assert not check.ok and check.detail.startswith("qpdf reports 1 syntax problem(s)")
    raw = _blank_pdf(marked / "obj.pdf", b"", qdf=True, normalize_content=False)
    data = raw.read_bytes()
    old = b"/Type /Page\n"
    assert old in data
    new = (f"/{PLANTED} 1 /{PLANTED} 2".encode() + b"\n")
    raw.write_bytes(data.replace(old, old + new, 1))
    junk = marked / "junk.pdf"
    junk.write_bytes(b"%PDF-1.7\n" + PLANTED.encode())
    checks = [verify.structure_ok(raw), *verify.verify_build(BuildContext(
        reference=dup, built=junk, turns={}, touched=frozenset({0}), expected={0: SAMPLE}, assembled=None,
        allowed_meta_changes=frozenset()))]
    for c in [check, *checks]:
        assert PLANTED not in c.detail and str(tmp_path) not in c.detail and "/Users" not in c.detail, c


def test_metadata_details_name_only_standard_info_keys(meta_ref: Path, tmp_path: Path):
    def hostile(pdf: pikepdf.Pdf) -> None:
        pdf.trailer.Info[f"/{PLANTED}"] = pikepdf.String("example")
        pdf.trailer.Info.Keywords = pikepdf.String("other example")

    check = verify.metadata_kept(meta_ref, edit(meta_ref, tmp_path / "h.pdf", hostile), frozenset())
    assert not check.ok and PLANTED not in check.detail
    assert "1 other key(s) added" in check.detail and "Keywords changed" in check.detail


def _widget_pdf(path: Path, *, value: str | None) -> Path:
    """A page with one filled text field, drawn only by its widget appearance (None: no field at all)."""
    pdf = pikepdf.new()
    font = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
                                                BaseFont=pikepdf.Name.Helvetica))
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.Contents = pdf.make_stream(b"")
    if value is not None:
        ap = pdf.make_stream(b"/Tx BMC BT /F1 18 Tf 2 5 Td (" + value.encode() + b") Tj ET EMC",
                             Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 300, 30],
                             Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)))
        widget = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Widget, FT=pikepdf.Name.Tx, T=pikepdf.String("field"),
            V=pikepdf.String(value), F=4, Rect=[72, 600, 372, 630], AP=pikepdf.Dictionary(N=ap), P=page.obj))
        page.obj.Annots = pikepdf.Array([widget])
        pdf.Root.AcroForm = pikepdf.Dictionary(Fields=[widget], DA=pikepdf.String("/F1 0 Tf 0 g"),
                                               DR=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font)))
    pdf.save(path)
    return path


def test_renders_draw_form_widgets(tmp_path: Path):
    ref = _widget_pdf(tmp_path / "ref.pdf", value="Example Value 4521")
    assert verify.renders_identical(ref, _widget_pdf(tmp_path / "same.pdf", value="Example Value 4521"), {}).ok
    gone = verify.renders_identical(ref, _widget_pdf(tmp_path / "gone.pdf", value=None), {})
    other = verify.renders_identical(ref, _widget_pdf(tmp_path / "other.pdf", value="Example Value 9999"), {})
    assert not gone.ok and gone.page == 1
    assert not other.ok and other.page == 1


def _gradient(pdf: pikepdf.Pdf, bump: bool) -> pikepdf.Stream:
    samples = bytearray((x + y) % 256 for y in range(64) for x in range(64))
    if bump:
        for k in range(0, len(samples), 409):
            samples[k] = (samples[k] + 1) % 256           # a one-level change no 72 dpi render shows
    return pdf.make_stream(zlib.compress(bytes(samples)), Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image,
                           Width=64, Height=64, ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8,
                           Filter=pikepdf.Name.FlateDecode)


def _image_elsewhere(path: Path, via: str, bump: bool) -> Path:
    """A page whose only image is drawn through ``via``, never by its own content's Do."""
    pdf = pikepdf.new()
    img = _gradient(pdf, bump)
    draw = b"q 100 0 0 100 0 0 cm /Im0 Do Q"
    res = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=img))
    page = pdf.add_blank_page(page_size=(612, 792))
    if via == "annotation":
        ap = pdf.make_stream(draw, Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 100, 100],
                             Resources=res)
        page.obj.Contents = pdf.make_stream(b"")
        page.obj.Annots = pikepdf.Array([pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Stamp, Rect=[100, 100, 200, 200], F=4,
            AP=pikepdf.Dictionary(N=ap)))])
    elif via == "pattern":
        pat = pdf.make_stream(draw, Type=pikepdf.Name.Pattern, PatternType=1, PaintType=1, TilingType=1,
                              BBox=[0, 0, 100, 100], XStep=100, YStep=100, Resources=res)
        page.obj.Resources = pikepdf.Dictionary(Pattern=pikepdf.Dictionary(P0=pat))
        page.obj.Contents = pdf.make_stream(b"/Pattern cs /P0 scn 100 100 100 100 re f")
    elif via == "soft_mask":
        group = pdf.make_stream(draw, Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form, BBox=[0, 0, 612, 792],
                                Group=pikepdf.Dictionary(S=pikepdf.Name.Transparency, CS=pikepdf.Name.DeviceGray),
                                Resources=res)
        gs = pikepdf.Dictionary(Type=pikepdf.Name.ExtGState,
                                SMask=pikepdf.Dictionary(Type=pikepdf.Name.Mask, S=pikepdf.Name.Luminosity, G=group))
        page.obj.Resources = pikepdf.Dictionary(ExtGState=pikepdf.Dictionary(GS0=gs))
        page.obj.Contents = pdf.make_stream(b"q /GS0 gs 0 g 0 0 200 200 re f Q")
    else:
        proc = pdf.make_stream(b"1000 0 0 0 1000 1000 d1 q 1000 0 0 1000 0 0 cm /Im0 Do Q")
        t3 = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type3, FontBBox=[0, 0, 1000, 1000],
            FontMatrix=[0.001, 0, 0, 0.001, 0, 0], CharProcs=pikepdf.Dictionary(a=proc),
            Encoding=pikepdf.Dictionary(Differences=[97, pikepdf.Name.a]), FirstChar=97, LastChar=97, Widths=[1000],
            Resources=res))
        page.obj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(T3=t3))
        page.obj.Contents = pdf.make_stream(b"BT /T3 100 Tf 100 100 Td (a) Tj ET")
    pdf.save(path)
    return path


@pytest.mark.parametrize("via", ["annotation", "pattern", "soft_mask", "type3_glyph"])
def test_images_drawn_outside_the_page_content_are_compared(tmp_path: Path, via: str):
    ref = _image_elsewhere(tmp_path / "ref.pdf", via, False)
    same = verify.images_identical(ref, _image_elsewhere(tmp_path / "same.pdf", via, False))
    assert same.ok and same.detail.startswith("1 image(s)")
    changed = verify.images_identical(ref, _image_elsewhere(tmp_path / "changed.pdf", via, True))
    assert not changed.ok and changed.page == 1


def test_renders_and_structure_stay_under_the_pixel_cap(tmp_path: Path, monkeypatch):
    import math

    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.MediaBox = [0, 0, 200000, 200000]
    pdf.save(tmp_path / "huge.pdf")
    small = _blank_pdf(tmp_path / "small.pdf")
    asked: list[float] = []
    real = verify._render_gray

    def spy(page, scale, rotation, **kw):
        w, h = page.get_size()
        asked.append(math.ceil(w * scale) * math.ceil(h * scale))
        return real(page, min(scale, 0.0005), rotation, **kw)

    monkeypatch.setattr(verify, "_render_gray", spy)
    assert verify.structure_ok(tmp_path / "huge.pdf").ok
    verify.renders_identical(tmp_path / "huge.pdf", small, {})        # the reference is the huge one
    verify.renders_identical(small, tmp_path / "huge.pdf", {})
    assert len(asked) == 5 and max(asked) <= verify.RENDER_MAX_PIXELS


def test_a_jbig2_stream_behind_a_recompressed_flate_is_not_new(tmp_path: Path):
    payload = b"\x00\x00\x00\x30example jbig2 payload" * 8

    def jbig2_behind_flate(level: int) -> Callable[[pikepdf.Pdf], None]:
        def apply(pdf: pikepdf.Pdf) -> None:
            stream = pdf.make_stream(zlib.compress(payload, level), Type=pikepdf.Name.XObject,
                                     Subtype=pikepdf.Name.Image, Width=8, Height=8, ColorSpace=pikepdf.Name.DeviceGray,
                                     BitsPerComponent=1,
                                     Filter=pikepdf.Array([pikepdf.Name.FlateDecode, pikepdf.Name.JBIG2Decode]))
            pdf.pages[0].obj.Resources.XObject["/Jb0"] = stream
        return apply

    src = scan_pdf(tmp_path / "s.pdf")
    ref = edit(src, tmp_path / "ref.pdf", jbig2_behind_flate(1))
    built = edit(src, tmp_path / "built.pdf", jbig2_behind_flate(9))
    assert verify.no_jbig2_written(ref, built).ok
    other = edit(src, tmp_path / "other.pdf", lambda pdf: _add_jbig2(pdf, b"\x00\x00\x00\x30another payload"))
    assert not verify.no_jbig2_written(ref, other).ok


def test_info_keys_named_xmp_or_info_waive_only_that_key(meta_ref: Path, tmp_path: Path):
    def change_all(pdf: pikepdf.Pdf) -> None:
        pdf.trailer.Info.Keywords = pikepdf.String("other example")
        pdf.Root.Metadata.write(XMP.replace(b"\n", b"\r\n"))

    built = edit(meta_ref, tmp_path / "b.pdf", change_all)
    for allowed in ({"Info/XMP"}, {"Info/Info"}, {"/XMP", "/Info"}):
        check = verify.metadata_kept(meta_ref, built, frozenset(allowed))
        assert not check.ok and "XMP changed" in check.detail and "Keywords changed" in check.detail, allowed
    assert verify.metadata_kept(meta_ref, built, frozenset({"XMP", "Info/Keywords"})).ok


def test_retention_accepts_page_keys_read_from_json(good):
    _ref, built = good
    checks = verify.text_retention(built, {"0": SAMPLE})
    assert checks[0].ok, checks


# --------------------------------------------------------------------------- copies (§10.2, §11.1)


ORIGINAL_SHA = hashlib.sha256(b"an invented original").hexdigest()
ORIGINAL_NAME = "example-original.pdf"


def _derived_packet(*, ns: str | None = None, name: str = ORIGINAL_NAME, sha: str = ORIGINAL_SHA,
                    note: str | None = None) -> bytes:
    """A packet holding a DerivedFrom block laid out as meta.mark_derived lays it out, one part changed."""
    from pdfintake.pdf import meta

    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            f'<rdf:Description rdf:about="" xmlns:pdfintake="{ns or meta.PDFINTAKE_NS}">\n'
            '<pdfintake:DerivedFrom rdf:parseType="Resource">\n'
            f'<pdfintake:OriginalName>{name}</pdfintake:OriginalName>\n'
            f'<pdfintake:OriginalSHA256>{sha}</pdfintake:OriginalSHA256>\n'
            f'<pdfintake:Note>{meta.DERIVED_NOTE if note is None else note}</pdfintake:Note>\n'
            '</pdfintake:DerivedFrom>\n'
            '</rdf:Description></rdf:RDF></x:xmpmeta>\n<?xpacket end="w"?>').encode("utf-8")


def _set_xmp(packet: bytes) -> Callable[[pikepdf.Pdf], None]:
    def apply(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Metadata = pdf.make_stream(packet, Type=pikepdf.Name.Metadata, Subtype=pikepdf.Name.XML)
    return apply


def _mark(name: str = ORIGINAL_NAME, sha: str = ORIGINAL_SHA) -> Callable[[pikepdf.Pdf], None]:
    def apply(pdf: pikepdf.Pdf) -> None:
        from pdfintake.pdf import meta

        meta.mark_derived(pdf, name, sha)
    return apply


def test_derived_from_present_finds_what_mark_derived_writes(good, tmp_path: Path):
    _ref, built = good
    copy = edit(built, tmp_path / "copy.pdf", _mark())
    check = verify.derived_from_present(copy, ORIGINAL_SHA)
    assert check.ok, check.detail
    assert verify.derived_from_present(copy, ORIGINAL_SHA.upper()).ok
    assert verify.derived_from_present(copy, ORIGINAL_SHA, ORIGINAL_NAME).ok
    with_xmp = edit(built, tmp_path / "x.pdf", _set_xmp(XMP))      # into an existing packet, too
    assert verify.derived_from_present(edit(with_xmp, tmp_path / "copy2.pdf", _mark()), ORIGINAL_SHA).ok
    assert verify.derived_from_present(edit(built, tmp_path / "hand.pdf", _set_xmp(_derived_packet())), ORIGINAL_SHA).ok


def test_derived_from_present_fails_without_the_block_or_with_another_hash(good, tmp_path: Path):
    _ref, built = good
    none = verify.derived_from_present(built, ORIGINAL_SHA)
    assert not none.ok and "no XMP" in none.detail
    other = hashlib.sha256(b"another invented file").hexdigest()
    wrong = verify.derived_from_present(edit(built, tmp_path / "w.pdf", _mark(sha=other)), ORIGINAL_SHA)
    assert not wrong.ok and "SHA-256" in wrong.detail and other not in wrong.detail
    assert not verify.derived_from_present(built, "not a hash").ok
    outside = _derived_packet(sha="") + b"<!-- " + ORIGINAL_SHA.encode() + b" -->"   # the hash, but not in the block
    assert not verify.derived_from_present(edit(built, tmp_path / "s.pdf", _set_xmp(outside)), ORIGINAL_SHA).ok


@pytest.mark.parametrize("ns", ["https://example.com/notpdfintake/", "https://example.com/ns/pdfintake/1.0/",
                                "http://ns.adobe.com/xap/1.0/mm/"])
def test_derived_from_present_needs_pdfintakes_own_namespace_exactly(good, tmp_path: Path, ns: str):
    _ref, built = good
    check = verify.derived_from_present(edit(built, tmp_path / "f.pdf", _set_xmp(_derived_packet(ns=ns))), ORIGINAL_SHA)
    assert not check.ok and "no DerivedFrom block" in check.detail


def test_derived_from_present_needs_the_file_name_and_the_note(good, tmp_path: Path):
    _ref, built = good
    nameless = verify.derived_from_present(edit(built, tmp_path / "n.pdf", _set_xmp(_derived_packet(name=" "))),
                                           ORIGINAL_SHA)
    assert not nameless.ok and "file name" in nameless.detail
    renamed = verify.derived_from_present(edit(built, tmp_path / "r.pdf", _mark(name="another-example.pdf")),
                                          ORIGINAL_SHA, ORIGINAL_NAME)
    assert not renamed.ok and "file name" in renamed.detail and "another-example" not in renamed.detail
    for note in ("", "searchable copy", "Searchable copy, not the original"):
        check = verify.derived_from_present(edit(built, tmp_path / "t.pdf", _set_xmp(_derived_packet(note=note))),
                                            ORIGINAL_SHA)
        assert not check.ok and "searchable copy, not the original" in check.detail, note


def test_not_encrypted_passes_a_plain_copy_and_fails_a_locked_one(good, tmp_path: Path):
    _ref, built = good
    assert verify.not_encrypted(built).ok

    def encrypted(name: str, user: str) -> Path:            # invented passwords, never real ones
        with pikepdf.open(built) as pdf:
            pdf.save(tmp_path / name, encryption=pikepdf.Encryption(
                owner="example-owner-pw", user=user, allow=pikepdf.Permissions(modify_other=False)))
        return tmp_path / name

    check = verify.not_encrypted(encrypted("locked.pdf", ""))
    assert not check.ok and "edit lock" in check.detail
    closed = encrypted("closed.pdf", "example-open-pw")
    check = verify.not_encrypted(closed)
    assert not check.ok and "password" in check.detail and "example" not in check.detail


def test_no_signature_material_also_uses_the_refusals_detector(good, monkeypatch):
    from pdfintake import refusals

    _ref, built = good
    seen: list[Path] = []

    def fake(path: Path) -> list[str]:
        seen.append(Path(path))
        return ["unreadable"]

    monkeypatch.setattr(refusals, "output_signature_material", fake)
    check = verify.no_signature_material(built)
    assert seen == [Path(built)]
    assert not check.ok and "couldn't read" in check.detail
    monkeypatch.setattr(refusals, "output_signature_material", lambda path: ["/Sig field", "/ByteRange"])
    check = verify.no_signature_material(built)
    assert not check.ok and "signed /Sig field (1)" in check.detail and "/ByteRange (1)" in check.detail


def _signed_and_flattened(tmp_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("pyhanko")
    from fixtures import refusal_shells
    from pdfintake.pdf.strip import flatten_signatures

    signed = tmp_path / "signed.pdf"
    signed.write_bytes(refusal_shells.signed_visible("Signature1"))
    flat = tmp_path / "flat.pdf"
    with pikepdf.open(signed) as pdf:
        assert flatten_signatures(pdf) == ["Signature1"]
        pdf.save(flat)
    return signed, flat


def test_a_flattened_pyhanko_stamp_renders_like_the_signed_original(tmp_path: Path):
    signed, flat = _signed_and_flattened(tmp_path)
    assert verify.renders_identical(signed, flat, {}).ok
    assert verify.no_signature_material(flat).ok
    assert not verify.no_signature_material(signed).ok

    def drop_unflattened(pdf: pikepdf.Pdf) -> None:          # the fields removed, the stamp never drawn
        del pdf.Root["/AcroForm"]
        del pdf.pages[0].obj["/Annots"]

    lost = verify.renders_identical(signed, edit(signed, tmp_path / "lost.pdf", drop_unflattened), {})
    assert not lost.ok and lost.page == 1


def _sig_widget_with_states(path: Path, *, flattened: bool) -> Path:
    """A signature widget whose /N shows text and whose /D (down) state draws an image."""
    from pdfintake.pdf.strip import flatten_signatures

    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    page.obj.Contents = pdf.make_stream(b"")
    shown = pdf.make_stream(b"0.5 g 0 0 100 40 re f", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form,
                            BBox=[0, 0, 100, 40])
    down = pdf.make_stream(b"q 100 0 0 40 0 0 cm /Im0 Do Q", Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Form,
                           BBox=[0, 0, 100, 40],
                           Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=_gradient(pdf, False))))
    widget = pdf.make_indirect(pikepdf.Dictionary(
        Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Widget, FT=pikepdf.Name.Sig, T=pikepdf.String("Signature1"),
        F=4, Rect=[100, 100, 200, 140], AP=pikepdf.Dictionary(N=shown, D=down), P=page.obj))
    page.obj.Annots = pikepdf.Array([widget])
    pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([widget]), SigFlags=3)
    if flattened:
        flatten_signatures(pdf)
    pdf.save(path)
    return path


def test_images_identical_counts_only_the_drawn_stamp_on_a_copy(tmp_path: Path):
    ref = _sig_widget_with_states(tmp_path / "ref.pdf", flattened=False)
    copy = _sig_widget_with_states(tmp_path / "copy.pdf", flattened=True)
    assert not verify.images_identical(ref, copy).ok                   # the /D image is gone from the copy
    check = verify.images_identical(ref, copy, flattened_signatures=True)
    assert check.ok, check.detail


def test_verify_build_runs_the_copy_checks_on_a_searchable_copy(tmp_path: Path):
    signed, flat = _signed_and_flattened(tmp_path)
    built = add_layer(flat, tmp_path / "built.pdf", {0: lines_at([SAMPLE])})
    copy = edit(built, tmp_path / "copy.pdf", _mark())
    ctx = _ctx(signed, copy, allowed_meta_changes=frozenset({"XMP"}), original_sha256=ORIGINAL_SHA,
               original_name=ORIGINAL_NAME)
    checks = verify.verify_build(ctx)
    assert [c.name for c in checks] == ["text_retention", "off_page_glyphs", "renders_identical", "images_identical",
                                        "upright", "no_jbig2_written", "no_signature_material", "metadata_kept",
                                        "structure_ok", "not_encrypted", "derived_from_present"]
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert {c.name for c in checks} <= set(verify.CHECK_NAMES)
    bare = verify.verify_build(_ctx(signed, built, allowed_meta_changes=frozenset({"XMP"}), original_sha256=ORIGINAL_SHA))
    assert {c.name for c in bare if not c.ok} == {"derived_from_present"}
    renamed = edit(built, tmp_path / "renamed.pdf", _mark(name="another-example.pdf"))
    misnamed = verify.verify_build(_ctx(signed, renamed, allowed_meta_changes=frozenset({"XMP"}),
                                        original_sha256=ORIGINAL_SHA, original_name=ORIGINAL_NAME))
    assert {c.name for c in misnamed if not c.ok} == {"derived_from_present"}
    unflattened = add_layer(signed, tmp_path / "unflattened.pdf", {0: lines_at([SAMPLE])})
    failed = {c.name for c in verify.verify_build(_ctx(signed, unflattened, original_sha256=ORIGINAL_SHA)) if not c.ok}
    assert {"no_signature_material", "derived_from_present"} <= failed
