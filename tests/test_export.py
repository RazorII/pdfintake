"""pdf/export.py: ``export clean`` and ``export flatten``.

DESIGN §7.1: clean removes the listed keys and keeps the ICC profile and the dates unless
``strip_dates``; flatten keeps the text layer; signed, encrypted, counterparty and evidence inputs
are refused. Every fixture is built here, with made-up text.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import struct
import sys
import types
from pathlib import Path
from typing import Any

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, String
from PIL import Image, ImageCms

from fixtures import build
from pdfintake.errors import PdfIntakeError, Refused, UsageError, VerifyFailed
from pdfintake.pdf import export
from pdfintake.pdf.meta import PDFINTAKE_NS
from pdfintake.pdf.verify import Check

try:
    from fixtures import refusal_shells as shells
except Exception:  # noqa: BLE001 - U3's shells are optional here; the tests that need them skip
    shells = None


# --------------------------------------------------------------------------- fakes for units not built yet

def _fake_strip_jpeg(data: bytes, *, keep_icc: bool = True) -> bytes:
    keep_kinds = {"JFIF", "Adobe"} | ({"ICC"} if keep_icc else set())
    parts = export._jpeg_parts(data)
    assert parts is not None
    out = bytearray(data[:2])
    i = 2
    while True:
        while data[i + 1] == 0xFF:
            i += 1
        marker = data[i + 1]
        if marker == 0xD9:
            out += data[i:i + 2]
            return bytes(out)
        if 0xD0 <= marker <= 0xD7:
            out += data[i:i + 2]
            i += 2
            continue
        end = i + 2 + int.from_bytes(data[i + 2:i + 4], "big")
        if marker == 0xDA:
            j = end
            while True:
                j = data.find(b"\xff", j)
                if data[j + 1] == 0 or 0xD0 <= data[j + 1] <= 0xD7:
                    j += 2
                    continue
                break
            end = j
        is_meta = 0xE0 <= marker <= 0xEF or marker == 0xFE
        if not is_meta or export._app_kind(marker, data[i + 4:end]) in keep_kinds:
            out += data[i:end]
        i = end


def _fake_refusal_read(path: Path) -> Any:
    raw = Path(path).read_bytes()
    encrypted: Any = False
    try:
        with pikepdf.open(path) as pdf:
            if pdf.is_encrypted:
                encrypted = "owner"
    except pikepdf.PasswordError:
        encrypted = "user"
    markers = [m for m in ("/ByteRange [", "/DocMDP", "/UR3", "/DocTimeStamp", "/XFA", "/Encrypt") if m.encode() in raw]
    return types.SimpleNamespace(
        signed_fields=[], empty_sig_fields=0, sig_dicts=0, doc_timestamps=0, certified_p=None,
        usage_rights=False, pdfa=None, pdfua=None, encrypted=encrypted, xfa=False, jbig2_streams=0,
        incremental_updates=0, xmp_readable=True, walk_codes=[], raw_markers=markers, repaired=False,
        qpdf_warnings=0)


@pytest.fixture(autouse=True)
def _dependencies(monkeypatch: pytest.MonkeyPatch):
    """Use refusals (U3) and intake.jpeg (U4) when they exist; otherwise interface-shaped fakes."""
    try:
        from pdfintake.intake import jpeg  # noqa: F401
    except ImportError:
        fake = types.ModuleType("pdfintake.intake.jpeg")
        fake.strip_jpeg = _fake_strip_jpeg  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pdfintake.intake.jpeg", fake)
    try:
        from pdfintake import refusals  # noqa: F401
    except ImportError:
        fake = types.ModuleType("pdfintake.refusals")
        fake.read = _fake_refusal_read  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pdfintake.refusals", fake)
        import pdfintake
        monkeypatch.setattr(pdfintake, "refusals", fake, raising=False)
    yield


_STAGING = itertools.count()


@pytest.fixture
def staging() -> Path:
    """A build folder inside the test cache (exports are built in the cache, never in user folders)."""
    root = Path(os.environ["PDFINTAKE_CACHE"]) / "staging" / f"export-{os.getpid()}-{next(_STAGING)}"
    root.mkdir(parents=True)
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- JPEG fixtures

def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
XMP_PACKET = (b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
              b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" '
              b'xmlns:dc="http://purl.org/dc/elements/1.1/" dc:creator="Jane Doe"/></rdf:RDF></x:xmpmeta>'
              b'<?xpacket end="w"?>')
# The same packet with an XMP pdf:PDFVersion no output of these tests has. pikepdf's default
# fix_metadata_version rewrites a packet only when it carries that field, so a byte-for-byte XMP
# check needs this packet to fail on a save left at the default (decided 2026-11-05).
XMP_PACKET_VERSIONED = XMP_PACKET.replace(
    b'dc:creator="Jane Doe"/>',
    b'xmlns:pdf="http://ns.adobe.com/pdf/1.3/" dc:creator="Jane Doe" pdf:PDFVersion="1.0"/>')
assert XMP_PACKET_VERSIONED != XMP_PACKET


def metadata_jpeg() -> bytes:
    """A camera-style JPEG: ICC, EXIF (a made-up camera serial), XMP, IPTC, a comment and trailing bytes."""
    img = build.photo_like(160, 120, seed=3)
    exif = Image.Exif()
    exif[0x010F] = "Example Camera Co"
    exif[0xA431] = "SN-000000-EXAMPLE"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, icc_profile=SRGB, exif=exif.tobytes())
    data = buf.getvalue()
    extra = (_segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + XMP_PACKET)
             + _segment(0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x0a\x1c\x02\x50\x00\x06Jane D")
             + _segment(0xFE, b"made-up comment"))
    return data[:2] + extra + data[2:] + b"EXTRA-AFTER-EOI"


def jpeg_parts(data: bytes) -> export._JpegParts:
    parts = export._jpeg_parts(data)
    assert parts is not None
    return parts


def _dct_streams(path: Path) -> list[bytes]:
    with pikepdf.open(path) as pdf:
        return [bytes(s.read_raw_bytes()) for s in export._image_streams(pdf)
                if export._filters(s) == ["/DCTDecode"]]


# --------------------------------------------------------------------------- PDF fixtures

def metadata_pdf(path: Path, *, jpeg: bytes | None = None, layer: bool = True) -> Path:
    """A one-page PDF carrying every kind of metadata clean removes, plus an invisible text layer."""
    texts = (build.TextSpec("Example layer words", mode=3),) if layer else ()
    images = (build.ImageSpec(jpeg if jpeg is not None else metadata_jpeg(), 72, 300, 320, 240),)
    data = build.pdf_bytes([build.PageSpec(texts=texts, images=images)])
    pdf = pikepdf.open(io.BytesIO(data))
    pdf.trailer.Info = pdf.make_indirect(Dictionary({
        "/Title": String("Example Title"), "/Author": String("Jane Doe"), "/Creator": String("Example Scanner App"),
        "/Producer": String("Example Producer 1.0"), "/Keywords": String("example, sample"),
        "/CreationDate": String("D:20000101000000Z"), "/ModDate": String("D:20000102000000Z"),
    }))
    xmp = pikepdf.Stream(pdf, XMP_PACKET)
    xmp.Type, xmp.Subtype = Name.Metadata, Name.XML
    pdf.Root.Metadata = pdf.make_indirect(xmp)
    pdf.Root.PieceInfo = Dictionary(RazorII_pdfintake=Dictionary(LastModified=String("D:20000101000000Z"),
                                                                  Private=String("example")))
    page = pdf.pages[0].obj
    page.PieceInfo = Dictionary(Example=Dictionary(Private=String("example")))
    page.LastModified = String("D:20000103000000Z")
    thumb = pikepdf.Stream(pdf, b"\x80" * 12)
    thumb.Width, thumb.Height, thumb.BitsPerComponent, thumb.ColorSpace = 2, 2, 8, Name.DeviceRGB
    page.Thumb = pdf.make_indirect(thumb)
    image = page.Resources.XObject["/Im0"]
    image_xmp = pikepdf.Stream(pdf, XMP_PACKET)
    image_xmp.Type, image_xmp.Subtype = Name.Metadata, Name.XML
    image.Metadata = pdf.make_indirect(image_xmp)
    pdf.save(path, deterministic_id=True)
    pdf.close()
    return path


def _form(pdf: pikepdf.Pdf, content: bytes, bbox: list[float], resources: Dictionary | None = None) -> pikepdf.Stream:
    s = pikepdf.Stream(pdf, content)
    s.Type, s.Subtype, s.BBox = Name.XObject, Name.Form, Array(bbox)
    if resources is not None:
        s.Resources = resources
    return pdf.make_indirect(s)


def form_pdf(path: Path, *, need_appearances: bool = False, field_appearance: bool = True,
             extra_annots: tuple[str, ...] = ()) -> Path:
    """A page whose text layer is our /PdfIntakeOCR0 form, with a filled text field, a square, a hidden
    square, a link and a popup. ``need_appearances`` drops the field's /AP and sets NeedAppearances;
    ``field_appearance=False`` drops the /AP alone."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    font = pdf.make_indirect(Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica,
                                        Encoding=Name.WinAnsiEncoding))
    layer = _form(pdf, b"BT /F1 12 Tf 3 Tr 20 250 Td (Example layer words) Tj ET", [0, 0, 300, 300],
                  Dictionary(Font=Dictionary(F1=font)))
    page.obj.Resources = Dictionary(Font=Dictionary(F1=font), XObject=Dictionary(PdfIntakeOCR0=layer))
    page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, b"0 0 1 rg 10 10 50 50 re f q /PdfIntakeOCR0 Do Q"))
    widget = Dictionary(Type=Name.Annot, Subtype=Name.Widget, FT=Name.Tx, T=String("name"),
                        V=String("Sample value"), Rect=Array([100, 100, 220, 120]), F=4,
                        DA=String("/Helv 10 Tf 0 g"), P=page.obj)
    if field_appearance and not need_appearances:
        widget.AP = Dictionary(N=_form(pdf, b"/Tx BMC BT /F1 10 Tf 0 g 2 5 Td (Sample value) Tj ET EMC",
                                       [0, 0, 120, 20], Dictionary(Font=Dictionary(F1=font))))
    widget = pdf.make_indirect(widget)
    square_ap = _form(pdf, b"1 0 0 RG 2 w 1 1 48 48 re S", [0, 0, 50, 50])
    square = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Square, Rect=Array([200, 200, 250, 250]),
                                          F=4, T=String("Jane Doe"), AP=Dictionary(N=square_ap)))
    hidden = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Square, Rect=Array([20, 200, 70, 250]),
                                          F=2, AP=Dictionary(N=square_ap)))
    popup = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Popup, Rect=Array([150, 150, 290, 190]),
                                         Parent=square))
    square.Popup = popup
    link = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Link, Rect=Array([20, 20, 60, 40]), F=4,
                                        A=Dictionary(S=Name.URI, URI=String("https://example.com/"))))
    annots = [widget, square, hidden, popup, link]
    if "note_without_appearance" in extra_annots:
        annots.append(pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Text, Rect=Array([250, 20, 270, 40]),
                                                   F=4, Contents=String("made-up note"))))
    page.obj.Annots = Array(annots)
    pdf.Root.AcroForm = Dictionary(Fields=Array([widget]), DA=String("/Helv 10 Tf 0 g"),
                                   DR=Dictionary(Font=Dictionary(Helv=font)))
    if need_appearances:
        pdf.Root.AcroForm.NeedAppearances = True
    pdf.save(path, deterministic_id=True)
    return path


def _check_names(err: VerifyFailed) -> set[str]:
    """The failed checks' names; every one is a pdf.verify.Check with ok False (decided 2026-11-05)."""
    checks = err.checks
    assert checks and all(isinstance(c, Check) and c.ok is False for c in checks), checks
    assert all(c.page is None or (isinstance(c.page, int) and c.page >= 1) for c in checks), checks
    from pdfintake.types import to_jsonable

    assert all(set(d) == {"name", "ok", "detail", "page"} for d in to_jsonable(checks)), "records carry them as JSON"
    return {c.name for c in checks}


def page_texts(path: Path) -> list[str]:
    return [text for _, text in export._pdfium_view(path)]


# --------------------------------------------------------------------------- clean

def test_clean_removes_the_listed_metadata_and_keeps_title_dates_and_icc(tmp_path: Path, staging: Path):
    src = metadata_pdf(tmp_path / "in.pdf")
    before_sha = _sha(src)
    jpeg_before = jpeg_parts(_dct_streams(src)[0])
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")

    assert _sha(src) == before_sha, "the input is never changed"
    assert rep.dest == staging / "clean.pdf" and rep.dest.is_file()
    for name in ("Info/Author", "Info/Creator", "Info/Producer", "Info/Keywords", "XMP", "PieceInfo",
                 "Thumb", "XMP/objects"):
        assert name in rep.removed, (name, rep.removed)
    for kind in ("Exif", "XMP", "IPTC", "COM", "trailing_data"):
        assert any(r.startswith(f"jpeg/{kind} ") for r in rep.removed), (kind, rep.removed)
    assert {"Info/Title", "Info/CreationDate", "Info/ModDate", "text_layer"} <= set(rep.kept)
    assert "jpeg/ICC ×1" in rep.kept

    with pikepdf.open(rep.dest) as out:
        assert set(out.trailer.Info.keys()) == {"/Title", "/CreationDate", "/ModDate"}
        assert str(out.trailer.Info.Title) == "Example Title"
        assert "/Metadata" not in out.Root and "/PieceInfo" not in out.Root
        page = out.pages[0].obj
        assert "/Thumb" not in page and "/PieceInfo" not in page
        assert "/LastModified" in page, "a page date stays unless strip_dates"
        image = page.Resources.XObject["/Im0"]
        assert "/Metadata" not in image
    jpeg_after = jpeg_parts(_dct_streams(rep.dest)[0])
    assert jpeg_after.image == jpeg_before.image, "tables, frame and entropy data untouched"
    assert [s for k, s in jpeg_after.meta if k == "ICC"] == [s for k, s in jpeg_before.meta if k == "ICC"]
    assert {k for k, _ in jpeg_after.meta} <= {"JFIF", "ICC", "Adobe"}
    assert jpeg_after.trailing == 0
    raw = rep.dest.read_bytes()
    for leak in (b"Jane Doe", b"SN-000000-EXAMPLE", b"Example Camera Co", b"made-up comment", b"EXTRA-AFTER-EOI"):
        assert leak not in raw, leak
    assert page_texts(rep.dest) == page_texts(src)


def test_strip_dates_removes_the_dates_too(tmp_path: Path, staging: Path):
    src = metadata_pdf(tmp_path / "in.pdf")
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary", strip_dates=True)
    assert {"Info/CreationDate", "Info/ModDate", "LastModified"} <= set(rep.removed)
    assert "Info/CreationDate" not in rep.kept
    with pikepdf.open(rep.dest) as out:
        assert set(out.trailer.Info.keys()) == {"/Title"}
        assert "/LastModified" not in out.pages[0].obj


def test_clean_of_a_pdf_without_metadata_is_a_plain_copy(tmp_path: Path, staging: Path):
    src = build.write_pdf(tmp_path / "plain.pdf", [build.PageSpec(texts=(build.TextSpec("Sample words"),))])
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert rep.removed == [] and rep.warnings == []
    assert rep.kept == ["text_layer"]
    assert page_texts(rep.dest) == page_texts(src)


def test_clean_keeps_a_flate_wrapped_jpeg_and_its_decode_parms(tmp_path: Path, staging: Path):
    import zlib

    jpeg = metadata_jpeg()
    src = build.write_pdf(tmp_path / "in.pdf", [build.PageSpec(images=(build.ImageSpec(jpeg, 72, 300, 320, 240),))])
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        image = pdf.pages[0].obj.Resources.XObject["/Im0"]
        image.write(zlib.compress(jpeg), filter=Array([Name.FlateDecode, Name.DCTDecode]),
                    decode_parms=Array([Dictionary(), Dictionary(ColorTransform=1)]))
        pdf.save(src, deterministic_id=True)
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    with pikepdf.open(rep.dest) as out:
        image = out.pages[0].obj.Resources.XObject["/Im0"]
        assert export._filters(image) == ["/DCTDecode"]
        assert int(image.DecodeParms.ColorTransform) == 1
        parts = jpeg_parts(bytes(image.read_raw_bytes()))
    assert parts.image == jpeg_parts(jpeg).image
    assert any(r.startswith("jpeg/Exif") for r in rep.removed)


def test_clean_is_deterministic(tmp_path: Path, staging: Path):
    src = metadata_pdf(tmp_path / "in.pdf")
    a = export.export_clean(src, staging / "a.pdf", provenance="ordinary")
    b = export.export_clean(src, staging / "b.pdf", provenance="ordinary")
    assert _sha(a.dest) == _sha(b.dest)


def test_clean_warns_about_a_dropped_conformance_claim_and_kept_extras(tmp_path: Path, staging: Path):
    src = metadata_pdf(tmp_path / "in.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        pdf.Root.Metadata.write(XMP_PACKET.replace(
            b'dc:creator="Jane Doe"/>',
            b'dc:creator="Jane Doe"/><rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/" '
            b'pdfaid:part="2" pdfaid:conformance="B"/>'))
        pdf.Root.OCProperties = Dictionary(OCGs=Array([]), D=Dictionary())
        pdf.save(src, deterministic_id=True)
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert "conformance_claim_dropped" in rep.warnings
    assert "optional_content_kept" in rep.warnings
    assert set(rep.warnings) <= export.WARNING_CODES


def test_clean_keeps_forms_and_annotations(tmp_path: Path, staging: Path):
    src = form_pdf(tmp_path / "form.pdf")
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert "annotation_authors_kept" in rep.warnings
    with pikepdf.open(rep.dest) as out:
        assert "/AcroForm" in out.Root
        assert len(out.pages[0].obj.Annots) == 5


def test_a_jpeg_strip_that_touches_image_data_fails_closed(tmp_path: Path, staging: Path,
                                                         monkeypatch: pytest.MonkeyPatch):
    from pdfintake.intake import jpeg as jpeg_module

    real = jpeg_module.strip_jpeg

    def corrupting(data: bytes, *, keep_icc: bool = True) -> bytes:
        good = real(data, keep_icc=keep_icc)
        return good[:-6] + bytes([good[-6] ^ 0x01]) + good[-5:]     # one bit of entropy-coded data

    monkeypatch.setattr(jpeg_module, "strip_jpeg", corrupting)
    src = metadata_pdf(tmp_path / "in.pdf")
    with pytest.raises(VerifyFailed) as err:
        export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert err.value.code == "export_verify_failed"
    assert "jpeg_image_data" in _check_names(err.value)
    assert not (staging / "clean.pdf").exists()


def test_a_jpeg_strip_that_leaves_metadata_fails_closed(tmp_path: Path, staging: Path, monkeypatch: pytest.MonkeyPatch):
    from pdfintake.intake import jpeg as jpeg_module

    monkeypatch.setattr(jpeg_module, "strip_jpeg", lambda d, *, keep_icc=True: d)
    with pytest.raises(VerifyFailed) as err:
        export.export_clean(metadata_pdf(tmp_path / "in.pdf"), staging / "clean.pdf", provenance="ordinary")
    _check_names(err.value)
    assert not (staging / "clean.pdf").exists()


def _raw_pdf(objects: list[bytes]) -> bytes:
    """A classic-xref PDF written by hand, so it can hold objects nothing references."""
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _dct_xobject(data: bytes) -> bytes:
    return (b"<< /Type /XObject /Subtype /Image /Width 160 /Height 120 /BitsPerComponent 8 /ColorSpace /DeviceRGB "
            b"/Filter /DCTDecode /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")


def test_an_unreferenced_jpeg_is_neither_cleaned_nor_compared(tmp_path: Path, staging: Path):
    """An incremental edit can leave an orphan copy of an image; qpdf drops it on save."""
    jpeg = metadata_jpeg()
    content = b"q 160 0 0 120 10 10 cm /Im0 Do Q"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /XObject << /Im0 5 0 R >> >> "
               b"/Contents 4 0 R >>",
               b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
               _dct_xobject(jpeg),
               _dct_xobject(jpeg)]                  # object 6: referenced by nothing
    src = tmp_path / "orphan.pdf"
    src.write_bytes(_raw_pdf(objects))
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert "jpeg/Exif ×1" in rep.removed, rep.removed
    assert "jpeg/ICC ×1" in rep.kept
    assert len(_dct_streams(rep.dest)) == 1


def test_the_copy_gets_a_file_id_of_its_own(tmp_path: Path, staging: Path):
    src = metadata_pdf(tmp_path / "in.pdf")
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    with pikepdf.open(src) as before, pikepdf.open(rep.dest) as out:
        assert bytes(out.trailer.ID[0]) != bytes(before.trailer.ID[0]), "ID[0] would tie the copy to its original"


def _inline_page(path: Path, jpeg: bytes, *, where: str = "page", filters: bytes = b"/DCT") -> Path:
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(200, 200))
    data = jpeg if filters == b"/DCT" else jpeg.hex().encode() + b">"
    drawing = b"q 160 0 0 120 10 10 cm BI /W 160 /H 120 /BPC 8 /CS /RGB /F " + filters + b" ID " + data + b"\nEI Q"
    if where == "page":
        page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, drawing))
    else:
        page.obj.Resources = Dictionary(XObject=Dictionary(Fm0=_form(pdf, drawing, [0, 0, 200, 200])))
        page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, b"q /Fm0 Do Q"))
    pdf.save(path, deterministic_id=True)
    return path


def _inline_jpegs(path: Path) -> list[bytes]:
    with pikepdf.open(path) as pdf:
        return [jpeg for _, jpeg in export._dct_images(pdf) if jpeg is not None]


@pytest.mark.parametrize("where", ["page", "form"])
def test_inline_jpegs_lose_their_metadata_too(tmp_path: Path, staging: Path, where: str):
    jpeg = metadata_jpeg().replace(b"EXTRA-AFTER-EOI", b"")
    src = _inline_page(tmp_path / "inline.pdf", jpeg, where=where)
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    for kind in ("Exif", "XMP", "IPTC", "COM"):
        assert f"jpeg/{kind} ×1" in rep.removed, rep.removed
    assert "jpeg/ICC ×1" in rep.kept and rep.warnings == []
    with pikepdf.open(rep.dest) as out:
        content = b"".join(bytes(s.read_bytes()) for s in export._content_streams(out))
    for leak in (b"SN-000000-EXAMPLE", b"Example Camera Co", b"Jane D", b"made-up comment"):
        assert leak not in content, leak
    [after] = _inline_jpegs(rep.dest)
    assert jpeg_parts(after).image == jpeg_parts(jpeg).image
    assert export._pdfium_view(rep.dest) == export._pdfium_view(src)


def test_an_inline_jpeg_behind_a_second_filter_is_warned_about(tmp_path: Path, staging: Path):
    jpeg = metadata_jpeg().replace(b"EXTRA-AFTER-EOI", b"")
    src = _inline_page(tmp_path / "inline.pdf", jpeg, filters=b"[/AHx /DCT]")
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert rep.warnings == ["inline_jpeg_metadata_kept"]


# Planted defects after the save: each must fail verification and leave nothing behind.

def _planted_strip(monkeypatch: pytest.MonkeyPatch, plant: Any) -> None:
    from pdfintake.pdf import meta

    real = meta.strip_for_export

    def planted(pdf: pikepdf.Pdf, *, keep_dates: bool) -> list[str]:
        removed = real(pdf, keep_dates=keep_dates)
        plant(pdf)
        return removed

    monkeypatch.setattr(meta, "strip_for_export", planted)


def _add_content(pdf: pikepdf.Pdf, data: bytes) -> None:
    pdf.pages[0].contents_add(pdf.make_indirect(pikepdf.Stream(pdf, data)))


def _keep_author(pdf: pikepdf.Pdf) -> None:
    pdf.trailer.Info.Author = String("Jane Doe")


def _keep_pieceinfo(pdf: pikepdf.Pdf) -> None:
    pdf.Root.PieceInfo = Dictionary(Example=Dictionary(Private=String("example")))


def _draw_more(pdf: pikepdf.Pdf) -> None:
    _add_content(pdf, b"0 0 0 rg 0 0 40 40 re f")


def _write_more(pdf: pikepdf.Pdf) -> None:
    pdf.pages[0].obj.Resources.Font = Dictionary(F9=Dictionary(Type=Name.Font, Subtype=Name.Type1,
                                                               BaseFont=Name.Helvetica))
    _add_content(pdf, b"BT /F9 12 Tf 3 Tr 72 100 Td (Planted words) Tj ET")


@pytest.mark.parametrize("plant,check", [
    (_keep_author, "info_clean"),
    (_keep_pieceinfo, "metadata_removed"),
    (_draw_more, "renders_identical"),
    (_write_more, "text_identical"),
], ids=["info", "pieceinfo", "render", "text"])
def test_clean_fails_closed_when_the_saved_copy_is_wrong(tmp_path: Path, staging: Path,
                                                         monkeypatch: pytest.MonkeyPatch, plant: Any, check: str):
    src = metadata_pdf(tmp_path / "in.pdf")
    _planted_strip(monkeypatch, plant)
    with pytest.raises(VerifyFailed) as err:
        export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert err.value.code == "export_verify_failed"
    assert check in _check_names(err.value), err.value.checks
    assert not (staging / "clean.pdf").exists(), "a copy that failed its checks is deleted"


def _planted_save(monkeypatch: pytest.MonkeyPatch, plant: Any) -> None:
    real = pikepdf.Pdf.save

    def planted(self: pikepdf.Pdf, *args: Any, **kwargs: Any) -> None:
        plant(self)
        real(self, *args, **kwargs)

    monkeypatch.setattr(pikepdf.Pdf, "save", planted)


@pytest.mark.parametrize("swap,check", [("original", "jpeg_metadata_removed"), ("other", "images_identical")])
def test_clean_fails_closed_when_a_saved_jpeg_is_wrong(tmp_path: Path, staging: Path,
                                                       monkeypatch: pytest.MonkeyPatch, swap: str, check: str):
    original = metadata_jpeg()
    src = metadata_pdf(tmp_path / "in.pdf", jpeg=original)
    if swap == "original":
        replacement = original                      # the metadata back in, image data unchanged
    else:
        buf = io.BytesIO()
        build.photo_like(160, 120, seed=11).save(buf, format="JPEG", quality=85)
        replacement = buf.getvalue()                # other pixels, no metadata

    def put_back(pdf: pikepdf.Pdf) -> None:
        pdf.pages[0].obj.Resources.XObject["/Im0"].write(replacement, filter=Name.DCTDecode)

    _planted_save(monkeypatch, put_back)
    with pytest.raises(VerifyFailed) as err:
        export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    names = _check_names(err.value)
    assert check in names, err.value.checks
    if swap == "original":
        assert "images_identical" not in names and "renders_identical" not in names
    assert not (staging / "clean.pdf").exists()


# --------------------------------------------------------------------------- flatten

def test_flatten_keeps_the_layer_and_bakes_fields_and_annotations(tmp_path: Path, staging: Path):
    src = form_pdf(tmp_path / "form.pdf")
    before_sha = _sha(src)
    rep = export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert _sha(src) == before_sha
    assert "text_layer" in rep.kept and "annotations/Link ×1" in rep.kept
    assert "annotations/Widget ×1" in rep.removed and "annotations/Square ×1" in rep.removed
    assert "annotations/not_shown ×1" in rep.removed and "AcroForm" in rep.removed
    assert "annotations/Popup ×1" in rep.removed
    assert rep.warnings == ["annotations_not_shown_dropped"]

    with pikepdf.open(src) as before, pikepdf.open(rep.dest) as out:
        assert "/AcroForm" not in out.Root
        assert [str(a.Subtype) for a in out.pages[0].obj.Annots] == ["/Link"]
        layer_before = bytes(before.pages[0].obj.Resources.XObject["/PdfIntakeOCR0"].read_bytes())
        layer_after = bytes(out.pages[0].obj.Resources.XObject["/PdfIntakeOCR0"].read_bytes())
        assert layer_after == layer_before, "the text layer's form is byte-identical"
    text = page_texts(rep.dest)[0]
    assert "Example layer words" in text and "Sample value" in text
    assert export._pdfium_view(rep.dest)[0][0] == export._pdfium_view(src)[0][0], "looks exactly as before"


def test_flatten_generates_missing_field_appearances(tmp_path: Path, staging: Path):
    src = form_pdf(tmp_path / "form.pdf", need_appearances=True)
    rep = export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert "appearances_generated" in rep.warnings
    with pikepdf.open(rep.dest) as out:
        assert "/AcroForm" not in out.Root
    assert "Sample value" in page_texts(rep.dest)[0]
    assert "Example layer words" in page_texts(rep.dest)[0]


def test_flatten_stops_when_an_annotation_has_no_appearance(tmp_path: Path, staging: Path):
    src = form_pdf(tmp_path / "form.pdf", extra_annots=("note_without_appearance",))
    with pytest.raises(PdfIntakeError) as err:
        export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert err.value.code == "flatten_incomplete"
    assert not (staging / "flat.pdf").exists()


def test_flatten_keeps_metadata_and_a_plain_layer_pdf(tmp_path: Path, staging: Path):
    src = metadata_pdf(tmp_path / "in.pdf")
    rep = export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert {"Info", "XMP", "text_layer"} <= set(rep.kept)
    assert rep.removed == [] and rep.warnings == []
    assert page_texts(rep.dest) == page_texts(src)


def test_flatten_is_deterministic(tmp_path: Path, staging: Path):
    src = form_pdf(tmp_path / "form.pdf")
    a = export.export_flatten(src, staging / "a.pdf", provenance="ordinary")
    b = export.export_flatten(src, staging / "b.pdf", provenance="ordinary")
    assert _sha(a.dest) == _sha(b.dest)


def test_flatten_draws_a_field_that_has_no_appearance_and_no_need_appearances(tmp_path: Path, staging: Path):
    src = form_pdf(tmp_path / "form.pdf", field_appearance=False)
    with pikepdf.open(src) as pdf:
        assert "/NeedAppearances" not in pdf.Root.AcroForm
    rep = export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert "appearances_generated" in rep.warnings
    assert "annotations/Widget ×1" in rep.removed
    with pikepdf.open(rep.dest) as out:
        assert "/AcroForm" not in out.Root
    assert "Sample value" in page_texts(rep.dest)[0]


def _planted_flatten(monkeypatch: pytest.MonkeyPatch, plant: Any) -> None:
    real = pikepdf.Pdf.flatten_annotations

    def planted(self: pikepdf.Pdf, *args: Any, **kwargs: Any) -> None:
        real(self, *args, **kwargs)
        plant(self)

    monkeypatch.setattr(pikepdf.Pdf, "flatten_annotations", planted)


def _rewrite_page_content(pdf: pikepdf.Pdf) -> None:
    contents = pdf.pages[0].obj.Contents
    for stream in (list(contents) if isinstance(contents, Array) else [contents]):
        if b"10 10 50 50 re f" in bytes(stream.read_bytes()):         # the original content, not qpdf's q/Q
            stream.write(b"0 1 0 rg 10 10 50 50 re f q /PdfIntakeOCR0 Do Q")


def _empty_the_layer(pdf: pikepdf.Pdf) -> None:
    pdf.pages[0].obj.Resources.XObject["/PdfIntakeOCR0"].write(b"")


def _drop_author(pdf: pikepdf.Pdf) -> None:
    del pdf.trailer.Info["/Author"]


@pytest.mark.parametrize("plant,fixture,checks", [
    (_rewrite_page_content, "form", {"layer_kept", "renders_identical"}),
    (_empty_the_layer, "form", {"layer_kept", "text_kept"}),
    (_drop_author, "metadata", {"metadata_kept"}),
], ids=["content", "layer", "metadata"])
def test_flatten_fails_closed_when_the_saved_copy_is_wrong(tmp_path: Path, staging: Path,
                                                           monkeypatch: pytest.MonkeyPatch, plant: Any,
                                                           fixture: str, checks: set[str]):
    src = form_pdf(tmp_path / "in.pdf") if fixture == "form" else metadata_pdf(tmp_path / "in.pdf")
    _planted_flatten(monkeypatch, plant)
    with pytest.raises(VerifyFailed) as err:
        export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert err.value.code == "export_verify_failed"
    assert checks <= _check_names(err.value), err.value.checks
    assert not (staging / "flat.pdf").exists(), "a copy that failed its checks is deleted"


def _keep_acroform(pdf: pikepdf.Pdf) -> None:
    pdf.Root.AcroForm = Dictionary(Fields=Array([]))


def _leave_a_widget(pdf: pikepdf.Pdf) -> None:
    widget = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Widget, FT=Name.Tx, T=String("left"),
                                          Rect=Array([0, 0, 1, 1]), F=4))
    pdf.pages[0].obj.Annots = Array([widget])


@pytest.mark.filterwarnings("ignore::pikepdf.PageCopyWarning")    # the planted widget has no field
@pytest.mark.parametrize("plant,check", [
    (_keep_acroform, "no_form_fields"),
    (_leave_a_widget, "annotations_baked"),
], ids=["acroform", "widget"])
def test_flatten_fails_closed_when_the_saved_copy_keeps_form_material(tmp_path: Path, staging: Path,
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      plant: Any, check: str):
    src = form_pdf(tmp_path / "in.pdf")
    _planted_save(monkeypatch, plant)
    with pytest.raises(VerifyFailed) as err:
        export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert err.value.code == "export_verify_failed"
    assert check in _check_names(err.value), err.value.checks
    assert not (staging / "flat.pdf").exists(), "a copy that failed its checks is deleted"


@pytest.mark.parametrize("fn", [export.export_clean, export.export_flatten], ids=["clean", "flatten"])
@pytest.mark.parametrize("reader", ["pikepdf", "pdfium"])
def test_a_built_copy_that_cant_be_opened_fails_closed(tmp_path: Path, staging: Path,
                                                       monkeypatch: pytest.MonkeyPatch, fn: Any, reader: str):
    import pypdfium2 as pdfium

    src = form_pdf(tmp_path / "in.pdf") if fn is export.export_flatten else metadata_pdf(tmp_path / "in.pdf")
    dest = staging / "out.pdf"
    if reader == "pikepdf":
        real = pikepdf.Pdf.save

        def unreadable(self: pikepdf.Pdf, *args: Any, **kwargs: Any) -> None:
            real(self, *args, **kwargs)
            Path(args[0]).write_bytes(b"not a PDF at all")

        monkeypatch.setattr(pikepdf.Pdf, "save", unreadable)
    else:
        real_view = export._pdfium_view

        def view(path: Path) -> list[tuple[str, str]]:
            if Path(path) == dest:
                raise pdfium.PdfiumError("planted: the copy can't be opened")
            return real_view(path)

        monkeypatch.setattr(export, "_pdfium_view", view)
    with pytest.raises(VerifyFailed) as err:
        fn(src, dest, provenance="ordinary")
    assert err.value.code == "export_verify_failed"
    assert "renders_identical" in _check_names(err.value)
    assert any(c.detail == "the copy can't be opened" for c in err.value.checks), err.value.checks
    assert not dest.exists(), "a copy that failed its checks is deleted"


def test_flatten_keeps_a_raw_xmp_packet_pikepdf_didnt_write(tmp_path: Path, staging: Path):
    # An ordinary input whose XMP another program wrote: the packet must come out byte for byte,
    # not re-serialised by pikepdf's fix_metadata_version (decided 2026-11-05). Its pdf:PDFVersion
    # differs from the file's, which is what makes that default rewrite it.
    src = metadata_pdf(tmp_path / "raw.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        pdf.Root.Metadata.write(XMP_PACKET_VERSIONED)
        pdf.save(src, deterministic_id=True, fix_metadata_version=False)
    assert _xmp_of(src) == XMP_PACKET_VERSIONED
    rep = export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert "XMP" in rep.kept and "exporting_a_copy" not in rep.warnings
    with pikepdf.open(rep.dest) as out:
        assert out.pdf_version != "1.0"
    assert _xmp_of(rep.dest) == XMP_PACKET_VERSIONED


# --------------------------------------------------------------------------- searchable copies

BOTH = [export.export_clean, export.export_flatten]

# A hand-written DerivedFrom block in the shape DESIGN §10.2 gives (name, SHA-256, the words),
# under pdfintake's real namespace; meta.mark_derived's own block is tested below.
_DERIVED_NS = PDFINTAKE_NS.encode()


def _derived_block(prefix: bytes = b"pdfintake", ns: bytes = _DERIVED_NS) -> bytes:
    return (b'<rdf:Description rdf:about="" xmlns:' + prefix + b'="' + ns + b'"><' + prefix + b':DerivedFrom '
            b'rdf:parseType="Resource"><' + prefix + b':OriginalName>example-scan.pdf</' + prefix + b':OriginalName><'
            + prefix + b':OriginalSHA256>' + b"0" * 64 + b'</' + prefix + b':OriginalSHA256><' + prefix + b':Note>'
            b'searchable copy, not the original</' + prefix + b':Note></' + prefix + b':DerivedFrom></rdf:Description>')


def _with_xmp_extra(path: Path, extra: bytes) -> Path:
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.Root.Metadata.write(XMP_PACKET.replace(b"</rdf:RDF>", extra + b"</rdf:RDF>"))
        # Raw, as a searchable copy carries it (copies are saved with fix_metadata_version=False).
        pdf.save(path, deterministic_id=True, fix_metadata_version=False)
    return path


def _xmp_of(path: Path) -> bytes | None:
    with pikepdf.open(path) as pdf:
        meta = pdf.Root.get("/Metadata")
        return bytes(meta.read_bytes()) if meta is not None else None


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_exporting_a_searchable_copy_is_allowed_with_a_warning(tmp_path: Path, staging: Path, fn: Any):
    src = _with_xmp_extra(metadata_pdf(tmp_path / "copy.pdf"), _derived_block())
    before = _sha(src)
    rep = fn(src, staging / "out.pdf", provenance="ordinary")
    assert "exporting_a_copy" in rep.warnings
    assert rep.warnings == sorted(set(rep.warnings)) and set(rep.warnings) <= export.WARNING_CODES
    assert _sha(src) == before, "the input is never changed"
    xmp = _xmp_of(rep.dest)
    if fn is export.export_clean:
        assert xmp is None, "clean removes the XMP, the DerivedFrom block with it"
    else:
        assert xmp is not None and b"pdfintake:DerivedFrom" in xmp, "flatten keeps the metadata"


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_a_derived_from_block_under_any_prefix_bound_to_the_namespace_counts(tmp_path: Path, staging: Path, fn: Any):
    src = _with_xmp_extra(metadata_pdf(tmp_path / "copy.pdf"), _derived_block(prefix=b"pi"))
    assert "exporting_a_copy" in fn(src, staging / "out.pdf", provenance="ordinary").warnings


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_the_standard_xmpmm_derived_from_is_not_a_searchable_copy(tmp_path: Path, staging: Path, fn: Any):
    xmpmm = (b'<rdf:Description rdf:about="" xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/" '
             b'xmlns:stRef="http://ns.adobe.com/xap/1.0/sType/ResourceRef#"><xmpMM:DerivedFrom '
             b'stRef:documentID="uuid:00000000-0000-0000-0000-000000000000"/></rdf:Description>')
    src = _with_xmp_extra(metadata_pdf(tmp_path / "edited.pdf"), xmpmm)
    assert "exporting_a_copy" not in fn(src, staging / "out.pdf", provenance="ordinary").warnings


def test_a_plain_file_and_a_flate_packet_are_read_right(tmp_path: Path, staging: Path):
    plain = metadata_pdf(tmp_path / "plain.pdf")
    assert "exporting_a_copy" not in export.export_clean(plain, staging / "a.pdf", provenance="ordinary").warnings
    src = metadata_pdf(tmp_path / "flate.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        packet = XMP_PACKET.replace(b"</rdf:RDF>", _derived_block() + b"</rdf:RDF>")
        pdf.Root.Metadata.write(__import__("zlib").compress(packet), filter=Name.FlateDecode)
        pdf.save(src, deterministic_id=True, fix_metadata_version=False)
    assert "exporting_a_copy" in export.export_flatten(src, staging / "b.pdf", provenance="ordinary").warnings


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
@pytest.mark.parametrize("provenance", ["counterparty", "evidence"])
def test_a_copy_of_an_evidence_or_counterparty_original_is_still_refused(tmp_path: Path, staging: Path,
                                                                         fn: Any, provenance: str):
    src = _with_xmp_extra(metadata_pdf(tmp_path / "copy.pdf"), _derived_block())
    with pytest.raises(Refused) as err:
        fn(src, staging / "out.pdf", provenance=provenance)
    assert err.value.code == f"provenance_{provenance}"
    assert not (staging / "out.pdf").exists()


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_a_copy_marked_by_meta_mark_derived_warns(tmp_path: Path, staging: Path, fn: Any):
    from pdfintake.pdf import meta

    src = metadata_pdf(tmp_path / "copy.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        meta.mark_derived(pdf, "example-scan.pdf", "0" * 64)
        # The packet mark_derived wrote, byte for byte, as every copy carries it.
        pdf.save(src, deterministic_id=True, fix_metadata_version=False)
    packet = _xmp_of(src)
    rep = fn(src, staging / "out.pdf", provenance="ordinary")
    assert "exporting_a_copy" in rep.warnings
    if fn is export.export_flatten:
        assert _xmp_of(rep.dest) == packet, "flatten keeps the copy's metadata byte for byte"


_UTF16_BOMS = {"utf-16-le": b"\xff\xfe", "utf-16-be": b"\xfe\xff"}


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
@pytest.mark.parametrize("codec", sorted(_UTF16_BOMS))
def test_a_utf16_copy_marked_by_meta_mark_derived_warns(tmp_path: Path, staging: Path, fn: Any, codec: str):
    from pdfintake.pdf import meta

    src = metadata_pdf(tmp_path / "copy.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        pdf.Root.Metadata.write(_UTF16_BOMS[codec] + XMP_PACKET.decode("utf-8").encode(codec))
        meta.mark_derived(pdf, "example-scan.pdf", "0" * 64)
        pdf.save(src, deterministic_id=True, fix_metadata_version=False)
    assert _xmp_of(src).startswith(_UTF16_BOMS[codec]), "mark_derived keeps the packet's codec"
    assert "exporting_a_copy" in fn(src, staging / "out.pdf", provenance="ordinary").warnings


_MENTIONS = {
    "value": XMP_PACKET.replace(
        b'dc:creator="Jane Doe"/>',
        b'><dc:description><rdf:Alt><rdf:li xml:lang="x-default">notes about pdfintake:DerivedFrom here'
        b'</rdf:li></rdf:Alt></dc:description></rdf:Description>'),
    "comment": XMP_PACKET.replace(b"</rdf:RDF>", b"<!-- <pdfintake:DerivedFrom/> --></rdf:RDF>"),
    "other_namespace": XMP_PACKET.replace(b"</rdf:RDF>", _derived_block(ns=b"https://example.com/ns/pdfintake/1.0/")
                                          + b"</rdf:RDF>"),
}


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
@pytest.mark.parametrize("kind", sorted(_MENTIONS))
def test_a_mention_of_derived_from_is_not_a_searchable_copy(tmp_path: Path, staging: Path, fn: Any, kind: str):
    # Only the block under pdfintake's namespace counts: not the words in a value or a comment,
    # and not a look-alike block under another namespace (decided 2026-11-05).
    src = metadata_pdf(tmp_path / "plain.pdf")
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:
        pdf.Root.Metadata.write(_MENTIONS[kind])
        pdf.save(src, deterministic_id=True, fix_metadata_version=False)
    assert "exporting_a_copy" not in fn(src, staging / "out.pdf", provenance="ordinary").warnings


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
@pytest.mark.parametrize("provenance", ["counterparty", "evidence"])
def test_counterparty_and_evidence_are_refused(tmp_path: Path, staging: Path, fn: Any, provenance: str):
    src = metadata_pdf(tmp_path / "in.pdf")
    with pytest.raises(Refused) as err:
        fn(src, staging / "out.pdf", provenance=provenance)
    assert err.value.code == f"provenance_{provenance}"
    assert err.value.exit_code == 4
    assert not (staging / "out.pdf").exists()


def _shell(name: str, *args: Any) -> bytes:
    if shells is None or not hasattr(shells, name):
        pytest.skip(f"refusal_shells.{name} isn't available")
    if name in ("signed_visible", "signed_invisible", "certified", "doc_timestamp"):
        pytest.importorskip("pyhanko")
    return getattr(shells, name)(*args)


REFUSED = [
    ("signed_visible", (), "signed"),
    ("signed_invisible", (), "signed"),
    ("certified", (2,), "certified"),
    ("doc_timestamp", (), "timestamped"),
    ("ur3", (), "usage_rights"),
    ("encrypted_owner", (), "encrypted"),
    ("encrypted_user", (), "encrypted"),
]


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
@pytest.mark.parametrize("name,args,code", REFUSED, ids=[r[0] for r in REFUSED])
def test_signed_and_encrypted_inputs_are_refused(tmp_path: Path, staging: Path, fn: Any, name: str,
                                                 args: tuple[Any, ...], code: str):
    src = tmp_path / f"{name}.pdf"
    src.write_bytes(_shell(name, *args))
    before = _sha(src)
    with pytest.raises(Refused) as err:
        fn(src, staging / "out.pdf", provenance="ordinary")
    assert err.value.code == code, err.value.detail
    assert not (staging / "out.pdf").exists()
    assert _sha(src) == before


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_every_save_names_its_encryption_and_xmp_handling(tmp_path: Path, staging: Path,
                                                          monkeypatch: pytest.MonkeyPatch, fn: Any):
    real = pikepdf.Pdf.save
    calls: list[dict[str, Any]] = []

    def recording(self: pikepdf.Pdf, *args: Any, **kwargs: Any) -> None:
        calls.append(dict(kwargs))
        real(self, *args, **kwargs)

    src = metadata_pdf(tmp_path / "in.pdf")
    monkeypatch.setattr(pikepdf.Pdf, "save", recording)
    fn(src, staging / "out.pdf", provenance="ordinary")
    assert calls, "the export saved nothing"
    for kwargs in calls:                    # explicit, never pikepdf's defaults (decided 2026-11-05)
        assert kwargs.get("encryption") is False, kwargs
        assert kwargs.get("deterministic_id") is True, kwargs
        assert kwargs.get("fix_metadata_version") is False, kwargs


def _permissions_only(path: Path, *, xmp: bytes | None = None) -> Path:
    """``metadata_pdf`` with an edit lock: an invented owner password and an empty open password.

    ``xmp`` replaces the document's XMP packet, written as given (never fixed up by pikepdf).
    """
    plain = metadata_pdf(path.with_name("plain-" + path.name))
    with pikepdf.open(plain) as pdf:
        if xmp is not None:
            pdf.Root.Metadata.write(xmp)
        pdf.save(path, fix_metadata_version=False,          # qpdf can't derive a deterministic ID for an encrypted file
                 encryption=pikepdf.Encryption(owner="example-owner-pw", user="",
                                               allow=pikepdf.Permissions(modify_other=False, extract=False)))
    return path


def _xmp_bytes(path: Path) -> bytes | None:
    with pikepdf.open(path) as pdf:
        stream = pdf.Root.get("/Metadata")
        return None if stream is None else bytes(stream.read_bytes())


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_an_encrypted_inputs_export_comes_out_unencrypted(tmp_path: Path, staging: Path,
                                                          monkeypatch: pytest.MonkeyPatch, fn: Any):
    # Export refuses encrypted inputs; with that refusal lifted, the save itself must still write no
    # encryption and leave the XMP packet as it was (flatten keeps it; clean removes it).
    # This checks behaviour. pikepdf's default save also drops the encryption, so the encryption
    # half can't tell an explicit encryption=False from the default; the guard that every save
    # names it is test_every_save_names_its_encryption_and_xmp_handling. The XMP half does bite on
    # its own: the packet's pdf:PDFVersion differs from the output's, so a save left at the default
    # fix_metadata_version would rewrite it (decided 2026-11-05).
    real_refuse = export._refuse
    monkeypatch.setattr(export, "_refuse", lambda codes, src: real_refuse(set(codes) - {"encrypted"}, src))
    src = _permissions_only(tmp_path / "locked.pdf", xmp=XMP_PACKET_VERSIONED)
    with pikepdf.open(src) as pdf:
        assert pdf.is_encrypted
    before = _sha(src)
    rep = fn(src, staging / "out.pdf", provenance="ordinary")
    with pikepdf.open(rep.dest) as out:
        assert not out.is_encrypted and "/Encrypt" not in out.trailer
    assert b"/Encrypt" not in rep.dest.read_bytes()
    assert _sha(src) == before                                          # the original keeps its lock
    if fn is export.export_flatten:
        assert _xmp_bytes(src) == XMP_PACKET_VERSIONED
        with pikepdf.open(rep.dest) as out:
            assert out.pdf_version != "1.0"
        assert _xmp_bytes(rep.dest) == XMP_PACKET_VERSIONED               # byte for byte
    else:
        assert _xmp_bytes(rep.dest) is None


def test_xfa_is_refused_by_flatten_only(tmp_path: Path, staging: Path):
    src = tmp_path / "xfa.pdf"
    src.write_bytes(_shell("xfa"))
    with pytest.raises(Refused) as err:
        export.export_flatten(src, staging / "flat.pdf", provenance="ordinary")
    assert err.value.code == "xfa"
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert rep.dest.is_file()


def _zeroed_signature(path: Path) -> Path:
    """A signature dictionary with a zeroed byte range in a filled /Sig field: structure, not a valid signature."""
    pdf = pikepdf.open(io.BytesIO(build.pdf_bytes(1)))
    sig = pdf.make_indirect(Dictionary(Type=Name.Sig, Filter=Name("/Adobe.PPKLite"), SubFilter=Name("/adbe.pkcs7.detached"),
                                       ByteRange=Array([0, 0, 0, 0]), Contents=String(b"\x00" * 16)))
    field = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Widget, FT=Name.Sig, T=String("Sig1"), V=sig,
                                         Rect=Array([0, 0, 0, 0]), P=pdf.pages[0].obj))
    pdf.pages[0].obj.Annots = Array([field])
    pdf.Root.AcroForm = Dictionary(Fields=Array([field]), SigFlags=3)
    pdf.save(path, deterministic_id=True)
    return path


def _clear_report(path: Path) -> Any:
    rep = _fake_refusal_read(path)
    rep.raw_markers = []
    rep.encrypted = False
    return rep


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_the_own_detector_refuses_when_the_refusal_reader_misses_a_signature(
        tmp_path: Path, staging: Path, monkeypatch: pytest.MonkeyPatch, fn: Any):
    from pdfintake import refusals

    monkeypatch.setattr(refusals, "read", _clear_report)
    src = _zeroed_signature(tmp_path / "signed.pdf")
    with pytest.raises(Refused) as err:
        fn(src, staging / "out.pdf", provenance="ordinary")
    assert err.value.code == "signed"


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_the_refusal_reader_alone_is_enough_to_refuse(tmp_path: Path, staging: Path,
                                                      monkeypatch: pytest.MonkeyPatch, fn: Any):
    from pdfintake import refusals

    def says_signed(path: Path) -> Any:
        rep = _clear_report(path)
        rep.raw_markers = ["/ByteRange ["]          # e.g. a signature in a revision the walk can't reach
        return rep

    monkeypatch.setattr(refusals, "read", says_signed)
    src = metadata_pdf(tmp_path / "in.pdf")
    with pytest.raises(Refused) as err:
        fn(src, staging / "out.pdf", provenance="ordinary")
    assert err.value.code == "signed"
    assert not (staging / "out.pdf").exists()


def test_an_empty_signature_field_is_not_a_signature(tmp_path: Path, staging: Path):
    pdf = pikepdf.open(io.BytesIO(build.pdf_bytes(1)))
    field = pdf.make_indirect(Dictionary(Type=Name.Annot, Subtype=Name.Widget, FT=Name.Sig, T=String("Sig1"),
                                         Rect=Array([0, 0, 0, 0]), P=pdf.pages[0].obj))
    pdf.pages[0].obj.Annots = Array([field])
    pdf.Root.AcroForm = Dictionary(Fields=Array([field]))
    src = tmp_path / "empty-field.pdf"
    pdf.save(src, deterministic_id=True)
    rep = export.export_clean(src, staging / "clean.pdf", provenance="ordinary")
    assert rep.dest.is_file()


# --------------------------------------------------------------------------- arguments

@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_the_export_is_built_only_inside_the_cache(tmp_path: Path, fn: Any):
    src = metadata_pdf(tmp_path / "in.pdf")
    with pytest.raises(UsageError) as err:
        fn(src, tmp_path / "out.pdf", provenance="ordinary")
    assert err.value.code == "export_dest_outside_cache"
    assert not (tmp_path / "out.pdf").exists()


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
def test_bad_arguments_are_usage_errors(tmp_path: Path, staging: Path, fn: Any):
    src = metadata_pdf(tmp_path / "in.pdf")
    with pytest.raises(UsageError) as err:
        fn(src, staging / "out.pdf", provenance="mine")
    assert err.value.code == "bad_provenance"
    with pytest.raises(UsageError) as err:
        fn(tmp_path / "missing.pdf", staging / "out.pdf", provenance="ordinary")
    assert err.value.code == "input_missing"
    inside = staging / "in-cache.pdf"
    inside.write_bytes(src.read_bytes())
    with pytest.raises(UsageError) as err:
        fn(inside, inside, provenance="ordinary")
    assert err.value.code == "output_is_input"


@pytest.mark.parametrize("fn", BOTH, ids=["clean", "flatten"])
@pytest.mark.parametrize("kind", ["text", "jpeg", "empty", "truncated"])
def test_a_non_pdf_is_a_usage_error_and_a_damaged_pdf_is_refused(tmp_path: Path, staging: Path, fn: Any, kind: str):
    src = tmp_path / "in.pdf"
    if kind == "text":
        src.write_bytes(b"made-up text, not a PDF\n")
    elif kind == "jpeg":
        src.write_bytes(metadata_jpeg())
    elif kind == "empty":
        src.write_bytes(b"")
    else:
        whole = metadata_pdf(tmp_path / "whole.pdf").read_bytes()
        src.write_bytes(whole[: len(whole) // 3])
    with pytest.raises(PdfIntakeError) as err:
        fn(src, staging / "out.pdf", provenance="ordinary")
    if kind == "truncated":
        assert isinstance(err.value, Refused) and err.value.code == "damaged"
        assert err.value.exit_code == 4
    else:
        assert isinstance(err.value, UsageError) and err.value.code == "not_a_pdf"
        assert err.value.exit_code == 2
    assert not (staging / "out.pdf").exists()


def test_every_warning_code_has_core_wording():
    assert export.WARNING_CODES == frozenset(export.WARNING_MESSAGES)
    assert all(msg.endswith(".") for msg in export.WARNING_MESSAGES.values())
    assert all(code in export._REFUSAL_MESSAGES for code in export.REFUSAL_CODES)


def test_the_interface_matches_the_design():
    import inspect

    assert [f.name for f in __import__("dataclasses").fields(export.ExportReport)] == ["dest", "removed", "kept", "warnings"]
    clean = inspect.signature(export.export_clean)
    assert list(clean.parameters) == ["src", "dest", "provenance", "strip_dates"]
    assert clean.parameters["provenance"].kind is inspect.Parameter.KEYWORD_ONLY
    assert clean.parameters["strip_dates"].default is False
    flatten = inspect.signature(export.export_flatten)
    assert list(flatten.parameters) == ["src", "dest", "provenance"]
    assert flatten.parameters["provenance"].kind is inspect.Parameter.KEYWORD_ONLY
