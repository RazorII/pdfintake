"""intake/assemble.py: every EXIF orientation placed through the page matrix, bytes untouched, /Rotate 0.

The 8-orientation test renders each page in pdfium (and in Quartz on macOS, in-process through
pyobjc) and compares it with the upright reference: Pillow's decode of the very same JPEG bytes,
turned by ``ImageOps.exif_transpose``.
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops, ImageOps, ImageStat

from fixtures import build
from pdfintake import fsutil
from pdfintake.errors import LosslessUnavailable, UsageError
from pdfintake.intake import assemble as A
from pdfintake.intake.assemble import AssembledPage, assemble, orientation_matrix, page_size

INVERSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
           5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_90, 7: Image.Transpose.TRANSVERSE,
           8: Image.Transpose.ROTATE_270}
SIZES = [(403, 301), (1008, 1392)]        # ragged (no whole MCUs: jpegtran -perfect would refuse) and MCU-aligned


def marked(width: int, height: int) -> Image.Image:
    """Continuous tone, coloured corner blocks and an F, so any wrong turn or mirror shows."""
    img = build.photo_like(width, height)
    u = max(width, height) // 40
    img.paste((220, 20, 20), (0, 0, width // 5, height // 6))
    img.paste((20, 20, 220), (width - width // 6, height - height // 5, width, height))
    x, y = width // 3, height // 4
    img.paste((0, 0, 0), (x, y, x + 2 * u, y + 12 * u))
    img.paste((0, 0, 0), (x, y, x + 8 * u, y + 2 * u))
    img.paste((0, 0, 0), (x, y + 5 * u, x + 6 * u, y + 7 * u))
    return img


def mean_diff(a: Image.Image, b: Image.Image) -> float:
    a, b = a.convert("RGB"), b.convert("RGB")
    w, h = min(a.width, b.width), min(a.height, b.height)
    assert abs(a.width - b.width) <= 1 and abs(a.height - b.height) <= 1, (a.size, b.size)
    a, b = a.crop((0, 0, w, h)), b.crop((0, 0, w, h))
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3


def page_dims(pdf_path: Path, index: int = 0) -> tuple[float, float]:
    with pikepdf.open(pdf_path) as pdf:
        box = [float(v) for v in pdf.pages[index].obj.MediaBox]
    return box[2] - box[0], box[3] - box[1]


def render_pdfium(pdf_path: Path, size: tuple[int, int], index: int = 0) -> Image.Image:
    """The page rendered so the image lands 1:1 on ``size`` pixels. The scale comes from the long
    side, which is exactly ``long_side_pt``; the short side's length is rounded to 1/10000 pt."""
    pw, ph = page_dims(pdf_path, index)
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return doc[index].render(scale=max(size) / max(pw, ph)).to_pil().convert("RGB")
    finally:
        doc.close()


def render_quartz(pdf_path: Path, size: tuple[int, int]) -> Image.Image:
    import Quartz
    from Foundation import NSURL

    doc = Quartz.CGPDFDocumentCreateWithURL(NSURL.fileURLWithPath_(str(pdf_path)))
    page = Quartz.CGPDFDocumentGetPage(doc, 1)
    box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFMediaBox)
    w, h = size
    cs = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(None, w, h, 8, w * 4, cs, Quartz.kCGImageAlphaPremultipliedLast)
    Quartz.CGContextSetRGBFillColor(ctx, 1, 1, 1, 1)
    Quartz.CGContextFillRect(ctx, ((0, 0), (w, h)))
    Quartz.CGContextScaleCTM(ctx, w / box.size.width, h / box.size.height)
    Quartz.CGContextDrawPDFPage(ctx, page)
    img = Quartz.CGBitmapContextCreateImage(ctx)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    return Image.frombuffer("RGBA", (w, h), bytes(data), "raw", "RGBA", Quartz.CGImageGetBytesPerRow(img), 1).convert("RGB")


def quartz_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Quartz  # noqa: F401
        from Foundation import NSURL  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- the 8-orientation matrix

@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_all_8_exif_orientations_display_upright_with_the_bytes_untouched(tmp_path: Path, size: tuple[int, int]):
    upright = marked(*size)
    results = {}
    for o in range(1, 9):
        kept = upright.transpose(INVERSE[o]) if o in INVERSE else upright.copy()
        data = build.jpeg_bytes(kept, quality=95, orientation=o)
        src = tmp_path / f"orient{o}.jpg"
        src.write_bytes(data)
        out = tmp_path / f"orient{o}.pdf"
        pages = assemble([src], out)
        assert pages[0].route == "passthrough" and pages[0].orientation == o
        assert pages[0].embedded_sha256 == pages[0].source_sha256
        with pikepdf.open(out) as pdf:
            page = pdf.pages[0]
            assert page.obj.get("/Rotate", 0) == 0 and pdf.Root.Pages.get("/Rotate", 0) == 0
            assert page.obj.Resources.XObject["/Im0"].read_raw_bytes() == data          # byte-identical
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as im:
            reference = ImageOps.exif_transpose(im).convert("RGB")
            as_stored = im.convert("RGB")
        assert reference.size == upright.size
        pw, ph = page_dims(out)
        assert max(pw, ph) == pytest.approx(842.0) and pw / ph == pytest.approx(size[0] / size[1], rel=1e-4)
        shown = render_pdfium(out, reference.size)
        results[o] = {"pdfium": round(mean_diff(shown, reference), 3)}
        assert results[o]["pdfium"] < 1.0, (o, results[o])
        if o != 1 and as_stored.size == reference.size:          # the test can tell a wrong turn apart
            assert mean_diff(shown, as_stored) > 10.0, o
        if quartz_available():
            q = render_quartz(out, reference.size)
            results[o]["quartz"] = round(mean_diff(q, reference), 3)
            assert results[o]["quartz"] < 1.0, (o, results[o])
    print(f"\nmean difference against the upright reference, {size[0]}x{size[1]}: {results}")


def test_the_naive_placement_would_show_the_stored_pixels(tmp_path: Path):
    """Discriminator: orientation-6 bytes placed with the identity matrix are not upright."""
    upright = marked(403, 301)
    data = build.jpeg_bytes(upright.transpose(INVERSE[6]), quality=95, orientation=6)
    src = tmp_path / "six.jpg"
    src.write_bytes(data)
    out = tmp_path / "six.pdf"
    assemble([src], out)
    with pikepdf.open(out, allow_overwriting_input=True) as pdf:
        pw, ph = page_dims(out)
        pdf.pages[0].obj.Contents = pdf.make_stream(f"q {pw} 0 0 {ph} 0 0 cm /Im0 Do Q".encode())
        pdf.save(out)
    shown = render_pdfium(out, (301, 403))
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as im:
        reference = ImageOps.exif_transpose(im).convert("RGB").resize(shown.size)
    assert mean_diff(shown, reference) > 10.0


@pytest.mark.skipif(not quartz_available(), reason="Quartz is checked on macOS only")
def test_quartz_ignores_the_exif_tag_inside_dct_data(tmp_path: Path):
    """Why the matrix is needed in every viewer: Quartz shows a DCT image's stored pixels, tag or not."""
    upright = marked(403, 301)
    kept = upright.transpose(INVERSE[6])
    data = build.jpeg_bytes(kept, quality=95, orientation=6)
    pdf = pikepdf.new()
    image = pikepdf.Stream(pdf, data, Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=kept.width,
                           Height=kept.height, ColorSpace=pikepdf.Name.DeviceRGB, BitsPerComponent=8,
                           Filter=pikepdf.Name.DCTDecode)
    page = pdf.add_blank_page(page_size=(kept.width, kept.height))
    page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=image))
    page.obj.Contents = pdf.make_stream(f"q {kept.width} 0 0 {kept.height} 0 0 cm /Im0 Do Q".encode())
    out = tmp_path / "naive.pdf"
    pdf.save(out)
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as im:
        stored = im.convert("RGB")
    assert mean_diff(render_quartz(out, kept.size), stored) < 1.0


# --------------------------------------------------------------------------- the geometry helpers

def test_page_size_keeps_the_displayed_aspect_and_the_long_side():
    assert page_size(403, 301, 1) == (842.0, round(301 * 842 / 403, 4))
    assert page_size(403, 301, 6) == (round(301 * 842 / 403, 4), 842.0)
    for o in (5, 6, 7, 8):
        w, h = page_size(4000, 3000, o)
        assert (w, h) == (631.5, 842.0)
    assert page_size(100, 100, 1, long_side_pt=595.0) == (595.0, 595.0)
    with pytest.raises(ValueError):
        page_size(0, 10, 1)


def test_orientation_matrix_maps_the_unit_square_onto_the_page():
    w, h = 300.0, 500.0
    for o in range(1, 9):
        a, b, c, d, e, f = orientation_matrix(o, w, h)
        corners = {(a * s + c * t + e, b * s + d * t + f) for s in (0, 1) for t in (0, 1)}
        assert corners == {(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)}, o
        det = a * d - b * c
        assert (det < 0) == (o in (2, 4, 5, 7)), o                       # the mirrored orientations
    assert orientation_matrix(1, w, h) == (w, 0.0, 0.0, h, 0.0, 0.0)
    assert orientation_matrix(6, w, h) == (0.0, -h, w, 0.0, 0.0, h)
    for bad in (0, 9, -1):
        with pytest.raises(ValueError):
            orientation_matrix(bad, w, h)


# --------------------------------------------------------------------------- assemble as a whole

def _inputs(tmp_path: Path) -> list[Path]:
    files = {
        "a.jpg": build.jpeg_bytes(marked(64, 48), orientation=8),
        "b.png": build.raw_png(30, 20, color_type=0, bit_depth=16, seed=2),
        "c.png": build.raw_png(30, 20, color_type=3, bit_depth=2, seed=3),
        "d.tif": None,
        "e.webp": None,
    }
    paths = []
    for name, data in files.items():
        path = tmp_path / name
        if name == "d.tif":
            buf = io.BytesIO()
            rgba = build.photo_like(40, 20).convert("RGBA")
            rgba.putalpha(Image.linear_gradient("L").resize((40, 20)))
            rgba.save(buf, format="TIFF")
            data = buf.getvalue()
        elif name == "e.webp":
            buf = io.BytesIO()
            build.photo_like(40, 20).save(buf, format="WEBP", quality=85)
            data = buf.getvalue()
        path.write_bytes(data)
        paths.append(path)
    return paths


def test_assemble_places_pages_in_order_on_every_route(tmp_path: Path):
    paths = _inputs(tmp_path)
    out = tmp_path / "all.pdf"
    pages = assemble(paths, out)
    assert [p.source for p in pages] == [str(p) for p in paths]
    assert [p.route for p in pages] == ["passthrough", "png_idat", "png_idat", "lossless", "encoded_once"]
    assert [p.orientation for p in pages] == [8, 1, 1, 1, 1]
    assert all(isinstance(p, AssembledPage) and p.warnings == [] for p in pages)
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 5 and pdf.pdf_version >= "1.5"               # 16 bpc
        assert "/Info" not in pdf.trailer
        for page, rec in zip(pdf.pages, pages):
            im0 = page.obj.Resources.XObject["/Im0"]
            raw = im0.read_raw_bytes()
            assert A._sha(raw) == rec.embedded_sha256
            smask = im0.get("/SMask")
            assert rec.bytes == len(raw) + (len(smask.read_raw_bytes()) if smask is not None else 0)
            assert page.obj.get("/Rotate", 0) == 0
        tiff_page = pdf.pages[3].obj.Resources.XObject["/Im0"]
        assert tiff_page.SMask.BitsPerComponent == 8 and tiff_page.Filter == pikepdf.Name.FlateDecode
    for rec, path in zip(pages[1:3], paths[1:3]):
        data = path.read_bytes()
        from pdfintake.intake.images import _png_chunks

        idat = b"".join(p for t, p in _png_chunks(data) if t == b"IDAT")
        assert rec.source_sha256 == rec.embedded_sha256 == A._sha(idat)
    assert pages[3].source_sha256 == A._sha(paths[3].read_bytes()) != pages[3].embedded_sha256
    doc = pdfium.PdfDocument(str(out))
    assert len(doc) == 5
    for i in range(5):
        assert doc[i].render(scale=0.25).to_pil().size[0] > 0
    doc.close()


def test_assemble_is_deterministic(tmp_path: Path):
    paths = _inputs(tmp_path)
    assemble(paths, tmp_path / "one.pdf")
    assemble(paths, tmp_path / "two.pdf")
    assert (tmp_path / "one.pdf").read_bytes() == (tmp_path / "two.pdf").read_bytes()


def test_shared_profiles_are_written_once(tmp_path: Path):
    from test_intake_images import fake_icc

    icc = fake_icc(b"RGB ")
    paths = []
    for i in range(3):
        path = tmp_path / f"p{i}.jpg"
        buf = io.BytesIO()
        build.photo_like(20, 10, seed=i).save(buf, format="JPEG", icc_profile=icc)
        path.write_bytes(buf.getvalue())
        paths.append(path)
    out = tmp_path / "icc.pdf"
    assemble(paths, out)
    with pikepdf.open(out) as pdf:
        objs = {pg.obj.Resources.XObject["/Im0"].ColorSpace[1].objgen for pg in pdf.pages}
        assert len(objs) == 1


def test_an_encoded_image_keeps_its_alpha_turned_with_it(tmp_path: Path):
    upright = marked(120, 80).convert("RGBA")
    alpha = Image.linear_gradient("L").rotate(90).resize((120, 80))
    upright.putalpha(alpha)
    ex = Image.Exif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    upright.transpose(INVERSE[6]).save(buf, format="WEBP", quality=95, exif=ex.tobytes())
    src = tmp_path / "alpha.webp"
    src.write_bytes(buf.getvalue())
    out = tmp_path / "alpha.pdf"
    pages = assemble([src], out)
    assert pages[0].route == "encoded_once" and pages[0].orientation == 1
    with pikepdf.open(out) as pdf:
        im0 = pdf.pages[0].obj.Resources.XObject["/Im0"]
        assert im0.Filter == pikepdf.Name.DCTDecode and (int(im0.Width), int(im0.Height)) == (120, 80)
        assert (int(im0.SMask.Width), int(im0.SMask.Height)) == (120, 80)
    white = Image.new("RGB", upright.size, "white")
    white.paste(upright, mask=upright.getchannel("A"))
    assert mean_diff(render_pdfium(out, (120, 80)), white) < 4.0


def test_a_heic_page_is_upright(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    upright = marked(160, 96)
    ex = Image.Exif()
    ex[0x0112] = 6
    ex[0x010F] = "ExampleCam"
    buf = io.BytesIO()
    try:
        upright.transpose(INVERSE[6]).save(buf, format="HEIF", quality=95, exif=ex.tobytes())
    except Exception as exc:                                             # noqa: BLE001
        pytest.skip(f"pillow-heif can't encode HEIC here: {exc}")
    src = tmp_path / "photo.heic"
    src.write_bytes(buf.getvalue())
    out = tmp_path / "heic.pdf"
    pages = assemble([src], out)
    assert pages[0].route == "encoded_once" and pages[0].orientation == 1
    assert page_dims(out)[0] > page_dims(out)[1]                         # landscape, as displayed
    assert mean_diff(render_pdfium(out, (160, 96)), upright) < 3.0


def test_a_profile_that_does_not_fit_warns_metadata_format_limit(tmp_path: Path):
    from test_intake_images import fake_icc

    src = tmp_path / "x.jpg"
    buf = io.BytesIO()
    build.photo_like(20, 10).save(buf, format="JPEG", icc_profile=fake_icc(b"CMYK"))
    src.write_bytes(buf.getvalue())
    pages = assemble([src], tmp_path / "x.pdf")
    assert pages[0].route == "passthrough" and pages[0].warnings == ["metadata_format_limit"]
    with pikepdf.open(tmp_path / "x.pdf") as pdf:
        assert pdf.pages[0].obj.Resources.XObject["/Im0"].ColorSpace == pikepdf.Name.DeviceRGB


def test_refusals_write_nothing(tmp_path: Path):
    out = tmp_path / "never.pdf"
    with pytest.raises(UsageError):
        assemble([], out)
    pdf_in = tmp_path / "in.pdf"
    pdf_in.write_bytes(build.pdf_bytes(1))
    with pytest.raises(UsageError):
        assemble([pdf_in], out)
    good = tmp_path / "ok.jpg"
    good.write_bytes(build.jpeg_bytes(build.photo_like(10, 10)))
    bad = tmp_path / "rgba16.png"
    bad.write_bytes(build.raw_png(6, 4, color_type=6, bit_depth=16))
    with pytest.raises(LosslessUnavailable) as err:
        assemble([good, bad], out)
    assert "rgba16.png" in str(err.value)
    assert not out.exists()


def test_an_eps_renamed_jpg_is_refused_without_starting_a_process(tmp_path: Path, monkeypatch):
    from PIL import EpsImagePlugin

    seen: list[str] = []

    def hook(event: str, args: tuple) -> None:
        if event.startswith(("subprocess.", "os.system", "os.posix_spawn", "os.exec", "os.spawn")):
            seen.append(event)

    sys.addaudithook(hook)

    def refuse(*a, **k):
        raise AssertionError("EpsImagePlugin was reached")

    monkeypatch.setattr(EpsImagePlugin.EpsImageFile, "_open", refuse)
    src = tmp_path / "scan.jpg"
    src.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n%%EOF\n")
    with pytest.raises(UsageError):
        assemble([src], tmp_path / "eps.pdf")
    assert seen == []


def test_too_many_pixels_is_refused_by_name(tmp_path: Path):
    data = bytearray(build.jpeg_bytes(build.photo_like(16, 16)))
    at = data.index(b"\xff\xc0")
    struct.pack_into(">HH", data, at + 5, 20000, 20000)                  # the frame claims 400 MP
    src = tmp_path / "huge.jpg"
    src.write_bytes(bytes(data))
    with pytest.raises(LosslessUnavailable) as err:
        assemble([src], tmp_path / "huge.pdf")
    assert err.value.code == "too_many_pixels"


def test_assert_upright_catches_an_inherited_rotate():
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(10, 10))
    A._assert_upright(pdf)
    pdf.Root.Pages.Rotate = 90
    with pytest.raises(LosslessUnavailable, match="Rotate"):
        A._assert_upright(pdf)


def test_the_assembled_file_passes_verify_where_it_applies(tmp_path: Path):
    """Integration with pdf/verify.py (U7) when it is there: the AssembledPage hashes are the ones it checks."""
    verify = pytest.importorskip("pdfintake.pdf.verify")
    paths = _inputs(tmp_path)
    out = tmp_path / "all.pdf"
    pages = assemble(paths, out)
    for check in (verify.passthrough_identical(out, pages), verify.upright(out, frozenset(range(len(pages)))),
                  verify.structure_ok(out), verify.no_signature_material(out)):
        assert check.ok, (check.name, check.detail)
    tampered = [AssembledPage(**{**vars(pages[0]), "embedded_sha256": "0" * 64})] + pages[1:]
    assert not verify.passthrough_identical(out, tampered).ok


# --------------------------------------------------------------------------- review fixes: orientation on every route

def _exif(orientation: int, **tags: object) -> bytes:
    ex = Image.Exif()
    ex[0x0112] = orientation
    for k, v in tags.items():
        ex[int(k[1:], 16)] = v
    return ex.tobytes()


def _png_chunk(typ: bytes, payload: bytes) -> bytes:
    import zlib

    return struct.pack(">I", len(payload)) + typ + payload + struct.pack(">I", zlib.crc32(typ + payload) & 0xFFFFFFFF)


def _with_exif_chunk(png: bytes, tiff: bytes) -> bytes:
    """``png`` with an eXIf chunk right after IHDR."""
    return png[:33] + _png_chunk(b"eXIf", tiff) + png[33:]


def _encode(kind: str, stored_img: Image.Image, orientation: int) -> tuple[bytes, str]:
    """``stored_img`` written as ``kind`` carrying EXIF ``orientation``; (bytes, expected route)."""
    buf = io.BytesIO()
    if kind == "png_idat":
        stored_img.save(buf, format="PNG", exif=_exif(orientation))
        return buf.getvalue(), "png_idat"
    if kind == "png_interlaced":
        rows = np.asarray(stored_img).reshape(stored_img.height, -1).tolist()
        png = build.raw_png(stored_img.width, stored_img.height, color_type=2, bit_depth=8, rows=rows, interlace=True)
        return _with_exif_chunk(png, _exif(orientation)), "lossless"
    if kind == "png_trns":
        stored_img.save(buf, format="PNG", exif=_exif(orientation), transparency=(1, 2, 3))
        return buf.getvalue(), "lossless"
    if kind == "webp_lossless":
        stored_img.save(buf, format="WEBP", lossless=True, exif=_exif(orientation))
        return buf.getvalue(), "lossless"
    if kind == "webp_lossy":
        stored_img.save(buf, format="WEBP", quality=95, exif=_exif(orientation))
        return buf.getvalue(), "encoded_once"
    if kind == "tiff_grey":
        stored_img.convert("L").save(buf, format="TIFF", tiffinfo={274: orientation})
        return buf.getvalue(), "lossless"
    if kind == "jpeg":
        return build.jpeg_bytes(stored_img, quality=95, orientation=orientation), "passthrough"
    raise AssertionError(kind)


LOSSLESS_KINDS = ["png_idat", "png_interlaced", "webp_lossless", "tiff_grey"]


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
@pytest.mark.parametrize("kind", LOSSLESS_KINDS)
def test_orientation_is_placed_on_the_png_idat_and_lossless_routes(tmp_path: Path, kind: str, orientation: int):
    upright = marked(64, 40)
    data, route = _encode(kind, upright.transpose(INVERSE[orientation]), orientation)
    src = tmp_path / f"{kind}-{orientation}.img"          # the name says nothing: sniff reads bytes
    src.write_bytes(data)
    out = tmp_path / f"{kind}-{orientation}.pdf"
    pages = assemble([src], out)
    assert pages[0].route == route and pages[0].orientation == orientation
    reference = upright.convert("L") if kind == "tiff_grey" else upright
    assert mean_diff(render_pdfium(out, upright.size), reference) < 1.0


@pytest.mark.parametrize("mode", ["L", "I;16", "P", "RGBA", "CMYK"])
def test_an_uncompressed_tiff_opened_from_a_path_is_placed_upright(tmp_path: Path, mode: str):
    upright = marked(64, 40)
    if mode == "I;16":
        ref = Image.fromarray(np.asarray(upright.convert("L"), np.uint16) * 257)
    elif mode == "P":
        ref = upright.convert("P")
    else:
        ref = upright.convert(mode)
    # The reference is the upright picture stored upright, through the same PDF colour pipeline.
    plain = tmp_path / f"{mode.replace(';', '_')}-upright.tif"
    ref.save(plain, format="TIFF")
    assemble([plain], plain.with_suffix(".pdf"))
    want = render_pdfium(plain.with_suffix(".pdf"), upright.size)
    for orientation in (5, 6, 7, 8):
        src = tmp_path / f"{mode.replace(';', '_')}-{orientation}.tif"
        ref.transpose(INVERSE[orientation]).save(src, format="TIFF", tiffinfo={274: orientation})
        out = src.with_suffix(".pdf")
        pages = assemble([src], out)
        assert pages[0].route == "lossless" and pages[0].orientation == orientation
        assert mean_diff(render_pdfium(out, upright.size), want) < 0.5, (mode, orientation)


def test_the_same_picture_turns_the_same_way_on_every_route(tmp_path: Path):
    upright = marked(64, 40)
    kept = upright.transpose(INVERSE[6])
    for kind in ["jpeg", "png_idat", "png_trns", "webp_lossless", "webp_lossy", "tiff_grey"]:
        data, route = _encode(kind, kept, 6)
        src = tmp_path / f"{kind}.img"
        src.write_bytes(data)
        out = tmp_path / f"{kind}.pdf"
        pages = assemble([src], out)
        assert pages[0].route == route, kind
        assert pages[0].orientation == (1 if route == "encoded_once" else 6), kind
        w, h = page_dims(out)
        assert w > h, kind                                               # displayed landscape, like upright
        reference = upright.convert("L") if kind == "tiff_grey" else upright
        assert mean_diff(render_pdfium(out, upright.size), reference) < 4.0, kind


def test_xmp_orientation_turns_nothing_on_any_route(tmp_path: Path):
    from PIL import PngImagePlugin

    from test_intake_jpeg import xmp

    img = marked(64, 40)
    packet = xmp(orientation=6, w=64, h=40)
    files: dict[str, bytes] = {}
    for name, fmt, kw in [("x.jpg", "JPEG", {"quality": 95}), ("x-lossless.webp", "WEBP", {"lossless": True}),
                          ("x-lossy.webp", "WEBP", {"quality": 95})]:
        buf = io.BytesIO()
        img.save(buf, format=fmt, xmp=packet, **kw)
        files[name] = buf.getvalue()
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", packet.decode("utf-8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    files["x.png"] = buf.getvalue()
    for name, data in files.items():
        src = tmp_path / name
        src.write_bytes(data)
        pages = assemble([src], tmp_path / f"{name}.pdf")
        assert pages[0].orientation == 1, name
        w, h = page_dims(tmp_path / f"{name}.pdf")
        assert w > h, name


# --------------------------------------------------------------------------- review fixes: metadata on the one encode

def _exif_by_hand(ifd0: list[tuple[int, int, int]], sub: list[tuple[int, int, int]]) -> bytes:
    """A little-endian EXIF TIFF block: IFD0 entries, then an Exif sub-IFD (tag, type, value)."""
    n0 = len(ifd0) + 1
    sub_at = 8 + 2 + 12 * n0 + 4
    out = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", n0)
    for tag, typ, val in sorted(ifd0 + [(0x8769, 4, sub_at)]):
        out += struct.pack("<HHI", tag, typ, 1) + (struct.pack("<HH", val, 0) if typ == 3 else struct.pack("<I", val))
    out += struct.pack("<I", 0) + struct.pack("<H", len(sub))
    for tag, typ, val in sorted(sub):
        out += struct.pack("<HHI", tag, typ, 1) + (struct.pack("<HH", val, 0) if typ == 3 else struct.pack("<I", val))
    return out + struct.pack("<I", 0)


@pytest.mark.parametrize("ifd0,sub", [
    ([], [(0x0112, 3, 6)]),                              # Orientation in the Exif sub-IFD
    ([(0xA002, 3, 999)], []),                            # PixelXDimension in IFD0
    ([], [(0x0100, 3, 5)]),                              # ImageWidth in the Exif sub-IFD
], ids=["sub-orientation", "ifd0-pixelx", "sub-width"])
def test_exif_tags_in_either_ifd_are_settled_on_the_one_encode(tmp_path: Path, ifd0, sub):
    buf = io.BytesIO()
    build.photo_like(64, 48).save(buf, format="WEBP", quality=90, exif=_exif_by_hand(ifd0, sub))
    src = tmp_path / "odd-exif.webp"
    src.write_bytes(buf.getvalue())
    pages = assemble([src], tmp_path / "odd.pdf")
    assert pages[0].route == "encoded_once" and pages[0].warnings == []
    with pikepdf.open(tmp_path / "odd.pdf") as pdf:
        jpeg = pdf.pages[0].obj.Resources.XObject["/Im0"].read_raw_bytes()
    from pdfintake.intake.jpeg import check_jpeg

    assert check_jpeg(jpeg, passthrough=False) == []


def test_metadata_the_encode_cannot_settle_is_dropped_with_a_warning(tmp_path: Path):
    # An XMP Orientation written in a form the patch doesn't rewrite: the block is left out, and said so.
    packet = (b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
              b'<tiff:Orientation>6</tiff:Orientation ></x:xmpmeta><?xpacket end="w"?>')
    buf = io.BytesIO()
    build.photo_like(64, 48).save(buf, format="WEBP", quality=90, xmp=packet)
    src = tmp_path / "odd-xmp.webp"
    src.write_bytes(buf.getvalue())
    pages = assemble([src], tmp_path / "odd.pdf")
    assert pages[0].route == "encoded_once" and pages[0].warnings == ["metadata_format_limit"]


def test_an_oversized_exif_block_warns_only_where_an_encode_would_carry_it(tmp_path: Path):
    big = _exif(1, x010E="d" * 70_000)                   # ImageDescription: one block over 64 KB
    rgba = build.photo_like(40, 30).convert("RGBA")
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", exif=big)
    png = tmp_path / "big.png"
    png.write_bytes(buf.getvalue())
    buf = io.BytesIO()
    build.photo_like(40, 30).save(buf, format="WEBP", quality=90, exif=big)
    webp = tmp_path / "big.webp"
    webp.write_bytes(buf.getvalue())
    lossless, encoded = assemble([png, webp], tmp_path / "big.pdf")
    assert lossless.route == "lossless" and lossless.warnings == []
    assert encoded.route == "encoded_once" and encoded.warnings == ["metadata_format_limit"]


# --------------------------------------------------------------------------- review fixes: dest

def test_assemble_never_writes_over_an_input(tmp_path: Path):
    src = tmp_path / "Scan.png"
    src.write_bytes(build.png_bytes(build.photo_like(20, 10)))
    before = src.read_bytes()
    for dest in (src, tmp_path / "scan.png", tmp_path / "sub" / ".." / "Scan.png"):
        with pytest.raises(UsageError) as err:
            assemble([src], dest)
        assert err.value.code == "output_is_input"
    assert src.read_bytes() == before


def test_a_failed_check_leaves_neither_dest_nor_a_partial(tmp_path: Path, monkeypatch):
    src = tmp_path / "a.png"
    src.write_bytes(build.png_bytes(build.photo_like(20, 10)))
    out = tmp_path / "out.pdf"

    def broken(dest, pages):
        raise LosslessUnavailable("assemble_check", "forced")

    monkeypatch.setattr(A, "_check_written", broken)
    with pytest.raises(LosslessUnavailable):
        assemble([src], out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png"]


def test_concurrent_builds_into_one_dest_all_succeed_and_leave_a_whole_file(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    paths = _inputs(tmp_path)
    serial = tmp_path / "serial.pdf"
    assemble(paths, serial)
    shared = tmp_path / "shared.pdf"
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: assemble(paths, shared), range(16)))
    assert len(results) == 16
    assert shared.read_bytes() == serial.read_bytes()
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".partial"]


# --------------------------------------------------------------------------- Windows: a busy replace

class _Platform:
    """``sys`` as assemble sees it, with another ``platform``."""

    def __init__(self, platform: str) -> None:
        self.platform = platform

    def __getattr__(self, name: str):
        return getattr(sys, name)


class _Clock:
    """``time`` as assemble sees it: ``sleep`` moves a made-up clock instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _busy(winerror: int) -> PermissionError:
    exc = PermissionError(13, "The process cannot access the file because it is being used by another process")
    exc.winerror = winerror                 # what Windows' OSError carries; set by hand off Windows
    return exc


def _refuse_replace(monkeypatch, dest: Path, winerror: int, times: int) -> list[Path]:
    """``os.replace`` onto ``dest`` refuses ``times`` times, as Windows does while a handle has a file."""
    import os

    real = os.replace
    refused: list[Path] = []

    def replace(src, dst, *a, **k):
        if Path(dst) == dest and len(refused) < times:
            refused.append(Path(src))
            raise _busy(winerror)
        return real(src, dst, *a, **k)

    monkeypatch.setattr(A.os, "replace", replace)
    return refused


def _one_image(tmp_path: Path) -> Path:
    src = tmp_path / "a.png"
    src.write_bytes(build.png_bytes(build.photo_like(20, 10)))
    return src


@pytest.mark.parametrize("winerror", [5, 32])
def test_on_windows_a_busy_replace_is_tried_again(tmp_path: Path, monkeypatch, winerror: int):
    """Windows refuses a replace while another handle has either file open, and the refusal ends when
    it closes; CI saw WinError 5 with eight builds replacing one dest at once (2026-11-07)."""
    src = _one_image(tmp_path)
    dest = tmp_path / "out.pdf"
    clock = _Clock()
    monkeypatch.setattr(fsutil, "sys", _Platform("win32"))
    monkeypatch.setattr(fsutil, "time", clock)
    refused = _refuse_replace(monkeypatch, dest, winerror, times=3)
    assemble([src], dest)
    assert len(refused) == 3 and len(clock.sleeps) == 3 and clock.now < fsutil.REPLACE_PATIENCE_S
    assert dest.read_bytes().startswith(b"%PDF-")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "out.pdf"]


def test_on_windows_a_replace_still_busy_after_the_patience_raises_and_leaves_no_partial(tmp_path: Path,
                                                                                        monkeypatch):
    src = _one_image(tmp_path)
    dest = tmp_path / "out.pdf"
    clock = _Clock()
    monkeypatch.setattr(fsutil, "sys", _Platform("win32"))
    monkeypatch.setattr(fsutil, "time", clock)
    _refuse_replace(monkeypatch, dest, 5, times=10**6)
    with pytest.raises(PermissionError):
        assemble([src], dest)
    assert fsutil.REPLACE_PATIENCE_S <= clock.now < fsutil.REPLACE_PATIENCE_S + 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png"]


@pytest.mark.parametrize("platform,winerror", [("darwin", 5), ("linux", 32), ("win32", 3)])
def test_any_other_refused_replace_is_raised_at_once(tmp_path: Path, monkeypatch, platform: str, winerror: int):
    """Only Windows' busy-file answers are waited out; POSIX raises a refusal as it did before."""
    src = _one_image(tmp_path)
    dest = tmp_path / "out.pdf"
    clock = _Clock()
    monkeypatch.setattr(fsutil, "sys", _Platform(platform))
    monkeypatch.setattr(fsutil, "time", clock)
    refused = _refuse_replace(monkeypatch, dest, winerror, times=1)
    with pytest.raises(PermissionError):
        assemble([src], dest)
    assert len(refused) == 1 and clock.sleeps == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png"]
