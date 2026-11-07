"""pdf/shrink.py: lossless, deterministic, never grows (DESIGN §2.2 U9, §5.3, §7.1)."""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

import pikepdf
import pytest
from fixtures import build
from PIL import Image

from pdfintake.pdf import shrink

XMP_14 = (
    b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/" xmlns:dc="http://purl.org/dc/elements/1.1/"'
    b' pdf:PDFVersion="1.4" pdf:Producer="Example Producer"><dc:title><rdf:Alt><rdf:li xml:lang="x-default">'
    b'Example title</rdf:li></rdf:Alt></dc:title></rdf:Description></rdf:RDF></x:xmpmeta>\n'
    + b" " * 512 + b'\n<?xpacket end="w"?>'
)
XMP_NO_VERSION = XMP_14.replace(b' pdf:PDFVersion="1.4"', b"")


# --------------------------------------------------------------------------- builders

def gradient(width: int, height: int, mode: str = "RGB") -> Image.Image:
    """A smooth image: deflate alone does poorly on it, PNG prediction very well."""
    img = Image.new("RGB", (width, height))
    img.putdata([((x * 255) // width, (y * 255) // height, ((x + y) * 127) // (width + height))
                 for y in range(height) for x in range(width)])
    return img.convert(mode) if mode != "RGB" else img


def image_stream(pdf: pikepdf.Pdf, img: Image.Image, *, level: int = 1, predicted: bool = False) -> pikepdf.Stream:
    cs = {"L": pikepdf.Name.DeviceGray, "RGB": pikepdf.Name.DeviceRGB, "CMYK": pikepdf.Name.DeviceCMYK}[img.mode]
    bpp = {"L": 1, "RGB": 3, "CMYK": 4}[img.mode]
    data = img.tobytes()
    parms = None
    if predicted:                                    # PNG Sub on every row, a weaker predictor than ours
        w = img.width * bpp
        rows = []
        for y in range(img.height):
            row = data[y * w:(y + 1) * w]
            rows.append(b"\x01" + bytes((row[i] - (row[i - bpp] if i >= bpp else 0)) & 0xFF for i in range(w)))
        data = b"".join(rows)
        parms = pikepdf.Dictionary(Predictor=15, Colors=bpp, BitsPerComponent=8, Columns=img.width)
    stream = pikepdf.Stream(pdf, zlib.compress(data, level))
    stream.Filter = pikepdf.Name.FlateDecode
    if parms is not None:
        stream.DecodeParms = parms
    stream.Type = pikepdf.Name.XObject
    stream.Subtype = pikepdf.Name.Image
    stream.Width, stream.Height = img.width, img.height
    stream.ColorSpace = cs
    stream.BitsPerComponent = 8
    return stream


def make_pdf(path: Path, *, images: list[pikepdf.Stream] | None = None, version: str = "1.4",
             xmp: bytes | None = None, info: dict[str, str] | None = None, file_id: tuple[bytes, bytes] | None = None,
             jpeg: bytes | None = None, extra=None) -> Path:
    """A one-page PDF with the given image streams placed side by side, saved without touching XMP."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    xobjects = pikepdf.Dictionary()
    ops = []
    streams = [s(pdf) if callable(s) else s for s in (images or [])]
    if jpeg is not None:
        js = pikepdf.Stream(pdf, jpeg)
        js.Filter = pikepdf.Name.DCTDecode
        js.Type, js.Subtype = pikepdf.Name.XObject, pikepdf.Name.Image
        with Image.open(io.BytesIO(jpeg), formats=["JPEG"]) as probe:
            js.Width, js.Height = probe.size
        js.ColorSpace, js.BitsPerComponent = pikepdf.Name.DeviceRGB, 8
        streams.append(js)
    for i, stream in enumerate(streams):
        xobjects[f"/Im{i}"] = pdf.make_indirect(stream)
        ops.append(f"q 200 0 0 150 {10 + 200 * i} 500 cm /Im{i} Do Q")
    ops.append("BT /F1 12 Tf 72 720 Td (Example text) Tj ET")
    page.obj.Resources = pikepdf.Dictionary(
        XObject=xobjects,
        Font=pikepdf.Dictionary(F1=pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica))))
    page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, "\n".join(ops).encode("latin-1")))
    if xmp is not None:
        m = pikepdf.Stream(pdf, xmp)
        m.Type, m.Subtype = pikepdf.Name.Metadata, pikepdf.Name.XML
        pdf.Root.Metadata = pdf.make_indirect(m)
    if info:
        pdf.trailer.Info = pdf.make_indirect(pikepdf.Dictionary({f"/{k}": pikepdf.String(v) for k, v in info.items()}))
    if file_id is not None:
        pdf.trailer.ID = pikepdf.Array([pikepdf.String(file_id[0]), pikepdf.String(file_id[1])])
    if extra is not None:
        extra(pdf)
    pdf.save(path, force_version=version, fix_metadata_version=False, compress_streams=False,
             object_stream_mode=pikepdf.ObjectStreamMode.disable, static_id=file_id is None)
    return path


def append_revision(path: Path, objects: dict[int, bytes]) -> None:
    """Append a classic incremental update replacing the given objects (generation 0)."""
    data = path.read_bytes()
    prev = int(re.findall(rb"startxref\s+(\d+)", data)[-1])
    with pikepdf.open(path) as pdf:
        size = max(int(pdf.trailer.Size), max(objects) + 1)
        root = pdf.Root.objgen[0]
    out = bytearray(data)
    if not out.endswith(b"\n"):
        out += b"\n"
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n"
    for num in sorted(objects):
        out += f"{num} 1\n{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root {root} 0 R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def render(path: Path) -> list[bytes]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            out.append(page.render(scale=1).to_pil().tobytes())
            page.close()
        return out
    finally:
        doc.close()


def page_images(path: Path) -> list[tuple[str, bytes]]:
    """(filter, decoded samples or raw bytes for DCT) of every image placed on every page."""
    out = []
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            xobjects = page.obj.Resources.get("/XObject", pikepdf.Dictionary())
            for key in sorted(xobjects.keys()):
                s = xobjects[key]
                filt = str(s.get("/Filter"))
                out.append((filt, bytes(s.read_raw_bytes() if filt == "/DCTDecode" else s.read_bytes())))
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def mixed(tmp_path: Path) -> Path:
    jpeg = build.jpeg_bytes(build.photo_like(160, 120, seed=3))
    return make_pdf(tmp_path / "mixed.pdf", images=[
        lambda pdf: image_stream(pdf, gradient(240, 180)),
        lambda pdf: image_stream(pdf, gradient(200, 150, "L")),
        lambda pdf: image_stream(pdf, gradient(120, 90, "CMYK"), predicted=True),
    ], jpeg=jpeg, xmp=XMP_NO_VERSION, info={"Title": "Example title", "Keywords": "alpha, beta"},
        version="1.6", file_id=(b"\x11" * 16, b"\x22" * 16))


# --------------------------------------------------------------------------- the §7.1 cases

def test_an_orphan_image_is_dropped(tmp_path: Path):
    src = make_pdf(tmp_path / "orphan.pdf", images=[lambda pdf: image_stream(pdf, gradient(64, 48))])
    marker = b"ORPHAN-IMAGE-PAYLOAD-" * 40
    with pikepdf.open(src) as pdf:
        size = int(pdf.trailer.Size)
    # an image object in the file that nothing references
    append_revision(src, {size: b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 "
                                b"/ColorSpace /DeviceGray /Length %d >>\nstream\n" % len(marker) + marker + b"\nendstream"})
    with pikepdf.open(src) as pdf:
        assert any(isinstance(o, pikepdf.Stream) and marker in bytes(o.read_raw_bytes()) for o in pdf.objects)

    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=False)
    assert not rep.byte_copy
    assert rep.objects_dropped >= 1
    assert rep.revisions_dropped == 1
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert not any(isinstance(o, pikepdf.Stream) and marker in bytes(o.read_raw_bytes()) for o in pdf.objects)
    assert render(src) == render(tmp_path / "out.pdf")


def test_xmp_stays_byte_identical(mixed: Path, tmp_path: Path):
    rep = shrink.lossless_shrink(mixed, tmp_path / "out.pdf", content_changed=True)
    assert not rep.byte_copy and rep.after < rep.before
    assert not rep.xmp_pdfversion_patched
    with pikepdf.open(mixed) as a, pikepdf.open(tmp_path / "out.pdf") as b:
        assert bytes(b.Root.Metadata.read_raw_bytes()) == bytes(a.Root.Metadata.read_raw_bytes()) == XMP_NO_VERSION
        assert "/Filter" not in b.Root.Metadata
        assert {str(k): str(v) for k, v in b.trailer.Info.items()} == {str(k): str(v) for k, v in a.trailer.Info.items()}


def test_xmp_without_a_version_is_untouched_when_the_header_moves(tmp_path: Path):
    src = make_pdf(tmp_path / "v14.pdf", images=[lambda pdf: image_stream(pdf, gradient(200, 150))],
                   xmp=XMP_NO_VERSION, version="1.4")
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=False)
    assert (rep.version_from, rep.version_to, rep.xmp_pdfversion_patched) == ("1.4", "1.5", False)
    with pikepdf.open(tmp_path / "out.pdf") as b:
        assert bytes(b.Root.Metadata.read_raw_bytes()) == XMP_NO_VERSION


def test_the_xmp_version_follows_the_header_at_the_same_length(tmp_path: Path):
    src = make_pdf(tmp_path / "v14.pdf", images=[lambda pdf: image_stream(pdf, gradient(200, 150))],
                   xmp=XMP_14, version="1.4")
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=False)
    assert (rep.version_from, rep.version_to, rep.xmp_pdfversion_patched) == ("1.4", "1.5", True)
    with pikepdf.open(tmp_path / "out.pdf") as b:
        assert b.pdf_version == "1.5"
        new = bytes(b.Root.Metadata.read_raw_bytes())
    assert len(new) == len(XMP_14)
    assert new == XMP_14.replace(b'pdf:PDFVersion="1.4"', b'pdf:PDFVersion="1.5"')
    assert src.read_bytes().count(b'pdf:PDFVersion="1.4"') == 1          # the source is never modified


def test_id0_is_kept(mixed: Path, tmp_path: Path):
    shrink.lossless_shrink(mixed, tmp_path / "out.pdf", content_changed=True)
    with pikepdf.open(tmp_path / "out.pdf") as b:
        assert bytes(b.trailer.ID[0]) == b"\x11" * 16
        assert bytes(b.trailer.ID[1]) != b"\x22" * 16                  # the second half is the new content's


def test_predictor_images_decode_to_identical_samples(mixed: Path, tmp_path: Path):
    rep = shrink.lossless_shrink(mixed, tmp_path / "out.pdf", content_changed=False)
    assert rep.predictor_images == 3                                    # RGB, grey, and CMYK re-predicted
    assert page_images(tmp_path / "out.pdf") == page_images(mixed)      # qpdf's decoder, samples and DCT bytes
    assert render(tmp_path / "out.pdf") == render(mixed)                # pdfium's decoder
    with pikepdf.open(tmp_path / "out.pdf") as b:
        xo = b.pages[0].obj.Resources.XObject
        parms = xo.Im0.DecodeParms
        assert (int(parms.Predictor), int(parms.Colors), int(parms.Columns), int(parms.BitsPerComponent)) == (15, 3, 240, 8)
        assert int(xo.Im1.DecodeParms.Colors) == 1
        assert int(xo.Im2.DecodeParms.Colors) == 4
    # the rows use only Up and Paeth
    with pikepdf.open(tmp_path / "out.pdf") as b:
        predicted = zlib.decompress(bytes(b.pages[0].obj.Resources.XObject.Im0.read_raw_bytes()))
    tags = {predicted[y * (240 * 3 + 1)] for y in range(180)}
    assert tags <= {2, 4}


def test_png_predict_round_trips_through_an_independent_decoder():
    """Our predicted rows, wrapped as a PNG, decode in Pillow to the exact samples."""
    import struct

    for mode, colour_type in (("RGB", 2), ("L", 0)):
        img = build.photo_like(97, 61, mode=mode, seed=5)
        bpp = len(mode)
        rows = shrink.png_predict(img.tobytes(), img.width, img.height, bpp)

        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", img.width, img.height, 8, colour_type, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
        with Image.open(io.BytesIO(png), formats=["PNG"]) as back:
            assert back.tobytes() == img.tobytes()


def test_never_grow_gives_a_byte_copy(mixed: Path, tmp_path: Path):
    once = tmp_path / "once.pdf"
    shrink.lossless_shrink(mixed, once, content_changed=False)
    twice = tmp_path / "twice.pdf"
    rep = shrink.lossless_shrink(once, twice, content_changed=False)
    assert rep.byte_copy
    assert twice.read_bytes() == once.read_bytes()
    assert (rep.before, rep.after, rep.streams_recompressed, rep.objects_dropped) == (once.stat().st_size,) * 2 + (0, 0)
    assert rep.version_to == rep.version_from and not rep.xmp_pdfversion_patched
    assert not list(tmp_path.glob("*.shrink-partial"))


def tiny_classic_pdf(path: Path) -> Path:
    """A hand-written one-page PDF with a tight classic xref: any qpdf rewrite of it is larger."""
    body = (b"%PDF-1.7\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 9 9]>>endobj\n")
    offsets = [m.start() for m in re.finditer(rb"\d 0 obj", body)]
    xref = b"xref\n0 4\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    path.write_bytes(body + xref + b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % len(body))
    return path


def test_never_grow_gives_a_byte_copy_even_when_content_changed(tmp_path: Path):
    src = tiny_classic_pdf(tmp_path / "tiny.pdf")
    with pikepdf.open(src) as pdf:                                   # the rewrite really would be larger
        shrink._save(pdf, tmp_path / "rewrite.pdf", "1.7")
    assert (tmp_path / "rewrite.pdf").stat().st_size > src.stat().st_size
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=True)
    assert rep.byte_copy
    assert (tmp_path / "out.pdf").read_bytes() == src.read_bytes()
    assert rep.after == rep.before == src.stat().st_size
    assert not list(tmp_path.glob("*.shrink-partial"))


def test_the_one_percent_boundary_keeps_a_saving_of_exactly_one_percent():
    assert shrink._keep_rewrite(10_000, 9_900, content_changed=False)            # exactly 1%: kept
    assert not shrink._keep_rewrite(10_000, 9_901, content_changed=False)        # under 1%: byte copy
    assert shrink._keep_rewrite(10_000, 9_999, content_changed=True)             # any saving with a change
    assert not shrink._keep_rewrite(10_000, 10_000, content_changed=True)        # never equal, never larger
    assert not shrink._keep_rewrite(10_000, 10_001, content_changed=True)
    assert not shrink._keep_rewrite(0, 0, content_changed=False)


@pytest.mark.parametrize("xmp_version", [b"1.7", b"1.4"])
def test_a_catalog_version_above_the_header_keeps_the_xmp_version(tmp_path: Path, xmp_version: bytes):
    """Header 1.4, catalog /Version 1.7: the file already reads as 1.7, so the save raises nothing
    and XMP is untouched, whether it says 1.7 (consistent) or 1.4 (a stale value we don't fix)."""
    xmp = XMP_14.replace(b'pdf:PDFVersion="1.4"', b'pdf:PDFVersion="' + xmp_version + b'"')

    def catalog_17(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Version = pikepdf.Name("/1.7")

    src = make_pdf(tmp_path / "catver.pdf", images=[lambda pdf: image_stream(pdf, gradient(200, 150))],
                   xmp=xmp, version="1.4", extra=catalog_17)
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=True)
    assert not rep.byte_copy
    assert (rep.version_from, rep.version_to, rep.xmp_pdfversion_patched) == ("1.4", "1.5", False)
    with pikepdf.open(tmp_path / "out.pdf") as b:
        assert str(b.Root.Version) == "/1.7"
        assert bytes(b.Root.Metadata.read_raw_bytes()) == xmp


def test_a_catalog_version_below_the_target_still_lets_the_xmp_follow(tmp_path: Path):
    def catalog_13(pdf: pikepdf.Pdf) -> None:
        pdf.Root.Version = pikepdf.Name("/1.3")

    src = make_pdf(tmp_path / "catver.pdf", images=[lambda pdf: image_stream(pdf, gradient(200, 150))],
                   xmp=XMP_14, version="1.4", extra=catalog_13)
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=True)
    assert (rep.version_to, rep.xmp_pdfversion_patched) == ("1.5", True)
    with pikepdf.open(tmp_path / "out.pdf") as b:
        assert bytes(b.Root.Metadata.read_raw_bytes()) == XMP_14.replace(b'"1.4"', b'"1.5"')


def test_a_tiny_saving_without_content_change_is_a_byte_copy_but_with_one_is_kept(tmp_path: Path):
    # Almost nothing to gain: one already-tight stream; the object-stream save saves a few bytes only.
    src = make_pdf(tmp_path / "tight.pdf", images=[lambda pdf: image_stream(pdf, build.noise(300, 300, seed=2), level=9)])
    probe = shrink.lossless_shrink(src, tmp_path / "probe.pdf", content_changed=True)
    assert not probe.byte_copy and 0 < probe.before - probe.after < 0.01 * probe.before
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=False)
    assert rep.byte_copy and (tmp_path / "out.pdf").read_bytes() == src.read_bytes()


def test_output_bytes_are_deterministic_across_two_runs(mixed: Path, tmp_path: Path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    shrink.lossless_shrink(mixed, a, content_changed=False)
    shrink.lossless_shrink(mixed, b, content_changed=False)
    assert a.read_bytes() == b.read_bytes()


def test_output_bytes_are_deterministic_across_processes_and_hash_seeds(mixed: Path, tmp_path: Path):
    code = ("import sys; from pathlib import Path; from pdfintake.pdf import shrink; "
            "shrink.lossless_shrink(Path(sys.argv[1]), Path(sys.argv[2]), content_changed=False)")
    digests = set()
    for seed in ("1", "2"):
        out = tmp_path / f"seed{seed}.pdf"
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-c", code, str(mixed), str(out)], check=True, env=env, timeout=120)
        digests.add(sha(out))
    local = tmp_path / "local.pdf"
    shrink.lossless_shrink(mixed, local, content_changed=False)
    digests.add(sha(local))
    assert len(digests) == 1


# --------------------------------------------------------------------------- the rest of the contract

def test_the_source_is_never_modified(mixed: Path, tmp_path: Path):
    before = sha(mixed)
    shrink.lossless_shrink(mixed, tmp_path / "out.pdf", content_changed=True)
    assert sha(mixed) == before


def test_dct_bytes_and_text_are_untouched(mixed: Path, tmp_path: Path):
    shrink.lossless_shrink(mixed, tmp_path / "out.pdf", content_changed=False)
    with pikepdf.open(mixed) as a, pikepdf.open(tmp_path / "out.pdf") as b:
        ja, jb = a.pages[0].obj.Resources.XObject.Im3, b.pages[0].obj.Resources.XObject.Im3
        assert bytes(ja.read_raw_bytes()) == bytes(jb.read_raw_bytes())
        assert str(jb.Filter) == "/DCTDecode" and "/DecodeParms" not in jb
        assert bytes(a.pages[0].Contents.read_bytes()) == bytes(b.pages[0].Contents.read_bytes())


def test_recompress_keeps_a_new_encoding_only_when_16_bytes_smaller():
    pdf = pikepdf.new()
    data = b"example content stream " * 4
    tight = pikepdf.Stream(pdf, zlib.compress(data, 9))
    tight.Filter = pikepdf.Name.FlateDecode
    tight = pdf.make_indirect(tight)
    loose_data = bytes(range(256)) * 40
    loose = pikepdf.Stream(pdf, zlib.compress(loose_data, 0))
    loose.Filter = pikepdf.Name.FlateDecode
    loose = pdf.make_indirect(loose)
    raw_tight = bytes(tight.read_raw_bytes())
    streams, predicted = shrink.recompress_streams(pdf)
    assert (streams, predicted) == (1, 0)
    assert bytes(tight.read_raw_bytes()) == raw_tight
    assert bytes(loose.read_bytes()) == loose_data
    assert len(loose.read_raw_bytes()) < len(zlib.compress(loose_data, 0)) - 16


def test_recompress_leaves_other_filters_metadata_and_damaged_streams_alone():
    pdf = pikepdf.new()
    payload = b"0123456789abcdef" * 64
    meta = pikepdf.Stream(pdf, zlib.compress(payload, 0))
    meta.Filter, meta.Type, meta.Subtype = pikepdf.Name.FlateDecode, pikepdf.Name.Metadata, pikepdf.Name.XML
    meta = pdf.make_indirect(meta)
    chained = pikepdf.Stream(pdf, zlib.compress(payload, 0))
    chained.Filter = pikepdf.Array([pikepdf.Name.FlateDecode, pikepdf.Name.ASCIIHexDecode])
    chained = pdf.make_indirect(chained)
    broken = pikepdf.Stream(pdf, zlib.compress(payload, 0)[:-7])
    broken.Filter = pikepdf.Name.FlateDecode
    broken = pdf.make_indirect(broken)
    trailing = pikepdf.Stream(pdf, zlib.compress(payload, 0) + b"GARBAGE")
    trailing.Filter = pikepdf.Name.FlateDecode
    trailing = pdf.make_indirect(trailing)
    raws = [bytes(s.read_raw_bytes()) for s in (meta, chained, broken, trailing)]
    assert shrink.recompress_streams(pdf) == (0, 0)
    assert [bytes(s.read_raw_bytes()) for s in (meta, chained, broken, trailing)] == raws


def test_an_unfiltered_8bit_image_is_predicted(tmp_path: Path):
    pdf = pikepdf.new()
    img = gradient(120, 80)
    s = pikepdf.Stream(pdf, img.tobytes())
    s.Type, s.Subtype = pikepdf.Name.XObject, pikepdf.Name.Image
    s.Width, s.Height, s.BitsPerComponent, s.ColorSpace = 120, 80, 8, pikepdf.Name.DeviceRGB
    s = pdf.make_indirect(s)
    assert shrink.recompress_streams(pdf) == (1, 1)
    assert bytes(s.read_bytes()) == img.tobytes()


def test_images_that_are_not_8bit_are_only_re_deflated():
    pdf = pikepdf.new()
    img = Image.new("1", (64, 64))
    s = pikepdf.Stream(pdf, zlib.compress(img.tobytes(), 0))
    s.Filter, s.Type, s.Subtype = pikepdf.Name.FlateDecode, pikepdf.Name.XObject, pikepdf.Name.Image
    s.Width, s.Height, s.BitsPerComponent, s.ColorSpace = 64, 64, 1, pikepdf.Name.DeviceGray
    s = pdf.make_indirect(s)
    assert shrink.recompress_streams(pdf) == (1, 0)
    assert "/DecodeParms" not in s and bytes(s.read_bytes()) == img.tobytes()


def test_revisions_are_counted(tmp_path: Path):
    src = make_pdf(tmp_path / "rev.pdf")
    assert shrink._revisions(src) == 1
    with pikepdf.open(src) as pdf:
        info_num = int(pdf.trailer.Size)
    append_revision(src, {info_num: b"<< /Producer (Example) >>"})
    assert shrink._revisions(src) == 2
    lin = tmp_path / "lin.pdf"
    with pikepdf.open(src) as pdf:
        pdf.save(lin, linearize=True)
    assert shrink._revisions(lin) == 1                                  # the first-page "startxref 0" isn't one


# --------------------------------------------------------------------------- never encrypted (DESIGN §10.4, §11.1)

def test_every_save_passes_encryption_false_and_a_deterministic_id(mixed: Path, tmp_path: Path, monkeypatch):
    calls: list[dict] = []
    real = pikepdf.Pdf.save

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pikepdf.Pdf, "save", spy)
    shrink.lossless_shrink(mixed, tmp_path / "out.pdf", content_changed=True)
    assert calls and all(c.get("encryption") is False and c.get("deterministic_id") is True
                         and c.get("fix_metadata_version") is False for c in calls)


def _permissions_only(tmp_path: Path) -> Path:
    src = tmp_path / "locked.pdf"
    with pikepdf.open(tiny_classic_pdf(tmp_path / "tiny.pdf")) as pdf:
        pdf.save(src, encryption=pikepdf.Encryption(owner="owner-example", user="",
                                                    allow=pikepdf.Permissions(modify_other=False)))
    return src


def test_an_encrypted_source_is_rewritten_even_when_the_rewrite_is_not_kept(tmp_path: Path, monkeypatch):
    # Saving without encryption makes this source smaller, so force the never-grow rule to refuse
    # the rewrite: the encrypted-source guard alone must then decide against a byte copy.
    src = _permissions_only(tmp_path)
    monkeypatch.setattr(shrink, "_keep_rewrite", lambda before, after, content_changed: False)
    for changed in (False, True):
        out = tmp_path / f"out-{changed}.pdf"
        rep = shrink.lossless_shrink(src, out, content_changed=changed)
        assert rep.byte_copy is False
        with pikepdf.open(out) as back:
            assert not back.is_encrypted and len(back.pages) == 1
        assert b"/Encrypt" not in out.read_bytes()
    plain = tmp_path / "plain-out.pdf"                                 # control: an unencrypted source is copied
    assert shrink.lossless_shrink(tmp_path / "tiny.pdf", plain, content_changed=True).byte_copy is True


def test_a_permissions_only_source_gives_an_unencrypted_copy_never_a_byte_copy(tmp_path: Path):
    plain = tiny_classic_pdf(tmp_path / "tiny.pdf")
    src = tmp_path / "locked.pdf"
    with pikepdf.open(plain) as pdf:
        pdf.save(src, encryption=pikepdf.Encryption(owner="owner-example", user="",
                                                    allow=pikepdf.Permissions(modify_other=False)))
    with pikepdf.open(src) as pdf:
        assert pdf.is_encrypted
    for changed in (False, True):
        out = tmp_path / f"out-{changed}.pdf"
        rep = shrink.lossless_shrink(src, out, content_changed=changed)
        assert not rep.byte_copy
        with pikepdf.open(out) as back:
            assert not back.is_encrypted and len(back.pages) == 1
        assert b"/Encrypt" not in out.read_bytes()
    assert (tmp_path / "out-False.pdf").read_bytes() == (tmp_path / "out-True.pdf").read_bytes()


def test_a_marked_copy_keeps_its_derived_from_block(tmp_path: Path):
    from pdfintake.pdf import meta

    def mark(pdf: pikepdf.Pdf) -> None:
        meta.mark_derived(pdf, "Example.pdf", "ab" * 32)

    src = make_pdf(tmp_path / "copy.pdf", images=[lambda pdf: image_stream(pdf, gradient(240, 180))], xmp=XMP_14, extra=mark)
    out = tmp_path / "out.pdf"
    rep = shrink.lossless_shrink(src, out, content_changed=True)
    assert not rep.byte_copy and rep.xmp_pdfversion_patched
    with pikepdf.open(out) as back:
        assert meta.derived_from(back) == {"name": "Example.pdf", "sha256": "ab" * 32, "note": meta.DERIVED_NOTE}
        assert not back.is_encrypted
