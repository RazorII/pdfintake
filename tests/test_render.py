"""pdf/render.py: renders, placed-image geometry, the OCR dpi and the pdfium lock. Synthetic PDFs only."""

from __future__ import annotations

import math
import threading
import zlib
from pathlib import Path

import pikepdf
import pytest
from PIL import Image

from fixtures import build
from pdfintake.pdf import render
from pdfintake.pdf.render import (MAX_RENDER_PIXELS, MAX_RENDER_SIDE, PDFIUM_LOCK, PlacedImage, Renderer,
                                  assert_locked, upscale_for_osd)
from pdfintake.types import PageImage


# --------------------------------------------------------------------------- helpers

def _image_stream(pdf: pikepdf.Pdf, w: int, h: int, *, gray: bool = False) -> pikepdf.Stream:
    data = bytes((x * 7 + y * 3) % 256 for y in range(h) for x in range(w * (1 if gray else 3)))
    s = pikepdf.Stream(pdf, zlib.compress(data))
    s.Type = pikepdf.Name.XObject
    s.Subtype = pikepdf.Name.Image
    s.Width, s.Height = w, h
    s.ColorSpace = pikepdf.Name.DeviceGray if gray else pikepdf.Name.DeviceRGB
    s.BitsPerComponent = 8
    s.Filter = pikepdf.Name.FlateDecode
    return s


def _custom(path: Path, size: tuple[float, float], content: bytes, *, images: dict[str, tuple[int, int]] | None = None,
            form: tuple[bytes, list[float], dict[str, tuple[int, int]]] | None = None,
            rotate: int = 0, pages_rotate: int = 0) -> Path:
    """A one-page PDF with hand-written content, image XObjects by name and an optional form /Fm0
    (its content, /Matrix and images)."""
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=size)
    xobjects = pikepdf.Dictionary()
    for name, (w, h) in (images or {}).items():
        xobjects[f"/{name}"] = pdf.make_indirect(_image_stream(pdf, w, h))
    if form is not None:
        fcontent, matrix, fimages = form
        fx = pikepdf.Dictionary()
        for name, (w, h) in fimages.items():
            fx[f"/{name}"] = pdf.make_indirect(_image_stream(pdf, w, h))
        fm = pikepdf.Stream(pdf, fcontent)
        fm.Type = pikepdf.Name.XObject
        fm.Subtype = pikepdf.Name.Form
        fm.BBox = pikepdf.Array([0, 0, size[0] * 4, size[1] * 4])
        fm.Matrix = pikepdf.Array(matrix)
        fm.Resources = pikepdf.Dictionary(XObject=fx)
        xobjects["/Fm0"] = pdf.make_indirect(fm)
    page.obj.Resources = pikepdf.Dictionary(XObject=xobjects)
    page.obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, content))
    if rotate:
        page.obj.Rotate = rotate
    if pages_rotate:
        pdf.Root.Pages.Rotate = pages_rotate
    pdf.save(path, deterministic_id=True)
    return path


def _marked_page(path: Path, width: float = 300.0, height: float = 500.0) -> Path:
    """A page with a black square in its top-left corner (unrotated), the rest white."""
    content = f"0 g 0 {height - 60:g} 60 60 re f".encode()
    return _custom(path, (width, height), content)


# --------------------------------------------------------------------------- rendering

def test_inherited_rotate_is_honoured(tmp_path: Path) -> None:
    src = _marked_page(tmp_path / "plain.pdf")
    with pikepdf.open(src) as pdf:
        pdf.Root.Pages.Rotate = 90                    # inherited, not on the page itself
        assert "/Rotate" not in pdf.pages[0].obj
        pdf.save(tmp_path / "rot.pdf")
    with Renderer(tmp_path / "rot.pdf") as r:
        img = r.render_scaled(0, 72.0)
        pi = r.render(0, 72.0, tmp_path / "rot.png")
    assert img.size == (500, 300)                     # displayed landscape
    assert (pi.width, pi.height) == (500, 300)
    # turned 90 degrees clockwise, the top-left square sits top-right
    assert img.convert("L").getpixel((470, 30)) < 50
    assert img.convert("L").getpixel((30, 30)) > 200
    img.close()


def test_render_rotation_argument_is_on_top_of_rotate(tmp_path: Path) -> None:
    src = _marked_page(tmp_path / "p.pdf")
    with Renderer(src) as r:
        upright = r.render_scaled(0, 72.0).convert("L")
        turned = r.render_scaled(0, 72.0, rotation=180, gray=True)
    assert turned.mode == "L"
    assert upright.getpixel((30, 30)) < 50
    assert turned.getpixel((300 - 30, 500 - 30)) < 50   # 180: the square moves to bottom-right
    with Renderer(src) as r, pytest.raises(ValueError):
        r.render_scaled(0, 72.0, rotation=45)
    with Renderer(src) as r, pytest.raises(ValueError):
        r.render_scaled(0, 0.0)


def test_png_carries_phys_and_page_image_fields(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(width=612, height=792, cropbox=(36, 36, 576, 756),
                                                                texts=(build.TextSpec("Sample line"),))])
    with Renderer(src) as r:
        pi = r.render(0, 200.0, tmp_path / "out" / "p0.png")
    assert isinstance(pi, PageImage)
    assert pi.dpi == 200.0 and pi.index == 0
    assert pi.cropbox == (36.0, 36.0, 576.0, 756.0)
    assert (pi.width, pi.height) == (math.ceil(540 * 200 / 72), math.ceil(720 * 200 / 72))
    with Image.open(pi.path, formats=["PNG"]) as png:
        assert png.format == "PNG"
        dpi = png.info["dpi"]
        assert abs(dpi[0] - 200.0) < 0.1 and abs(dpi[1] - 200.0) < 0.1
    with pi.open() as again:                          # the PageImage contract reads it back
        assert again.size == (pi.width, pi.height)


def test_page_index_out_of_range(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "p.pdf", 2)
    with Renderer(src) as r:
        assert r.page_count() == 2
        with pytest.raises(IndexError):
            r.render_scaled(2, 72.0)


def test_renderer_must_be_open(tmp_path: Path) -> None:
    r = Renderer(build.write_pdf(tmp_path / "p.pdf", 1))
    with pytest.raises(RuntimeError):
        r.page_count()


# --------------------------------------------------------------------------- image geometry

def test_ppi_from_the_matrix_at_90_degrees(tmp_path: Path) -> None:
    # 400x200 px; the image's x axis runs up the page (100 pt), its y axis to the left (200 pt)
    src = _custom(tmp_path / "p.pdf", (400, 400), b"q 0 100 -200 0 300 50 cm /Im0 Do Q", images={"Im0": (400, 200)})
    with Renderer(src) as r:
        (im,) = r.placed_images(0)
    assert im.ppi == pytest.approx(400 * 72 / 100)    # 288, from the x axis
    assert (im.width_px, im.height_px) == (400, 200)
    assert im.coverage == pytest.approx(200 * 100 / (400 * 400))
    # the naive width-over-box figure is wrong here: 400 px over a 200 pt wide box is 144
    assert im.ppi != pytest.approx(400 * 72 / 200)


def test_geometry_inside_a_form_xobject(tmp_path: Path) -> None:
    # the form scales by 0.5; inside it the image is drawn 400x400 units, so 200x200 pt on the page
    content = b"q 1 0 0 1 10 20 cm /Fm0 Do Q"
    src = _custom(tmp_path / "p.pdf", (300, 300), content,
                  form=(b"q 400 0 0 400 0 0 cm /Im1 Do Q", [0.5, 0, 0, 0.5, 0, 0], {"Im1": (600, 600)}))
    with Renderer(src) as r:
        (im,) = r.placed_images(0)
        native = r.native_dpi(0)
        with pikepdf.open(src) as pdf:
            img_obj = pdf.pages[0].obj.Resources.XObject.Fm0.Resources.XObject.Im1
            assert im.xobject == img_obj.objgen
    assert im.ppi == pytest.approx(600 * 72 / 200)    # 216
    assert im.coverage == pytest.approx(200 * 200 / (300 * 300))
    assert native is None                              # 44% of the page: not a scan of it


def test_fixture_form_page_and_image_fields(tmp_path: Path) -> None:
    jpeg = build.jpeg_bytes(build.photo_like(300, 200))
    src = build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(width=300, height=200, in_form=True,
                                                                images=(build.ImageSpec(jpeg),))])
    with Renderer(src) as r:
        (im,) = r.placed_images(0)
    assert isinstance(im, PlacedImage)
    assert im.filter == "DCTDecode" and im.bpc == 8 and im.colorspace == "DeviceRGB"
    assert im.bytes == len(jpeg)
    assert im.ppi == pytest.approx(72.0) and im.coverage == pytest.approx(1.0)
    assert im.xobject != (0, 0)


def test_inline_image_is_reported(tmp_path: Path) -> None:
    raw = zlib.compress(bytes(16 * 16 * 3))
    content = b"q 72 0 0 72 0 0 cm BI /W 16 /H 16 /BPC 8 /CS /RGB /F /Fl ID " + raw + b" EI Q"
    src = _custom(tmp_path / "p.pdf", (144, 144), content)
    with Renderer(src) as r:
        (im,) = r.placed_images(0)
    assert im.xobject == (0, 0)
    assert im.filter == "FlateDecode" and im.colorspace == "DeviceRGB" and im.bpc == 8
    assert im.ppi == pytest.approx(16.0) and im.coverage == pytest.approx(0.25)


def test_coverage_is_clipped_and_unioned(tmp_path: Path) -> None:
    # two overlapping images and one hanging off the page
    content = (b"q 100 0 0 100 0 0 cm /Im0 Do Q q 100 0 0 100 50 0 cm /Im0 Do Q "
               b"q 100 0 0 100 150 150 cm /Im0 Do Q")
    src = _custom(tmp_path / "p.pdf", (200, 200), content, images={"Im0": (10, 10)})
    with Renderer(src) as r:
        ims = r.placed_images(0)
        cov = r.image_coverage(0)
    assert [round(i.coverage, 4) for i in ims] == [0.25, 0.25, 0.0625]
    assert cov == pytest.approx((150 * 100 + 50 * 50) / (200 * 200))


def test_cropbox_coverage_and_form_cycle_are_safe(tmp_path: Path) -> None:
    src = _custom(tmp_path / "p.pdf", (200, 200), b"q /Fm0 Do Q",
                  form=(b"q 200 0 0 200 0 0 cm /Im1 Do Q /Fm0 Do", [1, 0, 0, 1, 0, 0], {"Im1": (100, 100)}))
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:       # the form draws itself
        fm = pdf.pages[0].obj.Resources.XObject.Fm0
        fm.Resources.XObject.Fm0 = fm
        pdf.pages[0].obj.CropBox = pikepdf.Array([0, 0, 100, 200])
        pdf.save(src)
    with Renderer(src) as r:
        ims = r.placed_images(0)
    assert len(ims) == 1                                # drawn once, the self-reference stops
    assert ims[0].coverage == pytest.approx(1.0)        # clipped to the 100x200 CropBox


# --------------------------------------------------------------------------- the OCR dpi

def _scan_page(tmp_path: Path, name: str, page: tuple[float, float], px: tuple[int, int]) -> Path:
    w, h = page
    return _custom(tmp_path / name, page, f"q {w:g} 0 0 {h:g} 0 0 cm /Im0 Do Q".encode(), images={"Im0": px})


def test_ocr_dpi_floor(tmp_path: Path) -> None:
    born = build.write_pdf(tmp_path / "b.pdf", [build.PageSpec(texts=(build.TextSpec("Sample"),))])
    low = _scan_page(tmp_path, "low.pdf", (612, 792), (850, 1100))      # 100 ppi
    with Renderer(born) as r:
        assert r.ocr_dpi(0) == 200.0
    with Renderer(low) as r:
        assert r.native_dpi(0) == pytest.approx(100.0)
        assert r.ocr_dpi(0) == 200.0
        assert not r.render_capped(0)


def test_ocr_dpi_native_and_rapid_cap(tmp_path: Path) -> None:
    mid = _scan_page(tmp_path, "mid.pdf", (612, 792), (2550, 3300))     # 300 ppi on Letter
    small = _scan_page(tmp_path, "small.pdf", (227, 297), (2296, 3000))  # about 728 ppi on a small page
    with Renderer(mid) as r:
        assert r.ocr_dpi(0) == 200.0                    # Letter's 2000 px cap (181) gives way to the floor
    with Renderer(small) as r:
        cap = math.floor(72 * 2000 / 297)
        assert r.ocr_dpi(0) == float(cap)               # 484: the long side reaches 2000 px
        assert math.ceil(297 * cap / 72) <= 2000


def test_ocr_dpi_never_above_1200(tmp_path: Path) -> None:
    tiny = _scan_page(tmp_path, "tiny.pdf", (40, 30), (2000, 1500))     # 3600 ppi on a stamp-sized page
    with Renderer(tiny) as r:
        assert r.ocr_dpi(0) == 1200.0


def test_ocr_dpi_rotated_scan_uses_placed_axes(tmp_path: Path) -> None:
    # a 300 ppi scan placed turned 90 degrees on a 227x297 pt page
    w, h = 227, 297
    cm = f"q 0 {h} {-w} 0 {w} 0 cm /Im0 Do Q".encode()     # built at run time: the literal run reads as a phone number
    src = _custom(tmp_path / "p.pdf", (w, h), cm, images={"Im0": (1238, 946)})
    with Renderer(src) as r:
        assert r.native_dpi(0) == pytest.approx(300.0, abs=0.2)
        assert r.ocr_dpi(0) == 300.0


def test_ocr_dpi_60_megapixel_cap(tmp_path: Path) -> None:
    big = build.write_pdf(tmp_path / "big.pdf", [build.PageSpec(width=5000, height=5000)])
    with Renderer(big) as r:
        dpi = r.ocr_dpi(0)
        assert dpi < 200.0                               # the pixel cap wins over the floor
        w, h = render._pixels(5000, 5000, dpi)
        assert w * h <= MAX_RENDER_PIXELS
        w1, h1 = render._pixels(5000, 5000, dpi + 1)
        assert w1 * h1 > MAX_RENDER_PIXELS               # and it is the largest whole dpi that fits
        assert r.render_capped(0)


def test_render_never_exceeds_60_megapixels() -> None:
    # the last two reach _side_cap's rounding guard: floor(72 * 32767 / long side) still rounds up to a
    # 32768 px side there (991 and 1199 dpi are the largest that fit)
    for size in [(14400.0, 14400.0), (14400.0, 200.0), (3000.0, 9000.0), (226.0, 14400.0), (100.0, 14400.0),
                 (10.0, 2378.25), (10.0, 1966.02)]:
        for dpi in (72.0, 200.0, 600.0, 1200.0):
            used = render._cap_dpi(*size, dpi)
            w, h = render._pixels(*size, used)
            assert w * h <= MAX_RENDER_PIXELS
            assert max(w, h) <= MAX_RENDER_SIDE
            assert used <= dpi


# --------------------------------------------------------------------------- the side limit

RECEIPT = (226.0, 14400.0)          # 80 mm wide and 5 m long: a long till receipt's shape


def _receipt(tmp_path: Path) -> Path:
    """A long-receipt-shaped page with a black bar across its top and its bottom."""
    w, h = RECEIPT
    content = f"0 g 0 0 {w:g} 40 re f 0 {h - 40:g} {w:g} 40 re f".encode()
    return _custom(tmp_path / "receipt.pdf", RECEIPT, content)


def test_long_receipt_fits_the_side_limit(tmp_path: Path) -> None:
    assert MAX_RENDER_SIDE == 32767                      # Tesseract's limit
    src = _receipt(tmp_path)
    w, h = RECEIPT
    with Renderer(src) as r:
        assert render._pixels(w, h, 200.0)[1] > MAX_RENDER_SIDE   # the floor alone would be too long
        dpi = r.ocr_dpi(0)
        assert dpi < 200.0                                # the side limit wins over the floor
        assert max(render._pixels(w, h, dpi)) <= MAX_RENDER_SIDE
        assert max(render._pixels(w, h, dpi + 1)) > MAX_RENDER_SIDE   # the largest whole dpi that fits
        assert r.render_scaled_down(0)
        assert not r.render_capped(0)                     # well inside 60 MP: not the pixel cap
        pi = r.render(0, 200.0, tmp_path / "receipt.png")
        assert pi.dpi == dpi
        assert pi.height <= MAX_RENDER_SIDE and pi.width * pi.height <= MAX_RENDER_PIXELS
        assert (pi.width, pi.height) == render._pixels(w, h, dpi)
        turned = r.render_scaled(0, 300.0, rotation=90, gray=True)
        thumb = r.render_scaled(0, 10.0, gray=True)
    with Image.open(pi.path, formats=["PNG"]) as png:
        assert abs(png.info["dpi"][1] - dpi) < 0.1
        assert png.size == (pi.width, pi.height)
    assert turned.size[0] <= MAX_RENDER_SIDE and turned.size[0] > turned.size[1]
    turned.close()
    # the whole page is drawn, scaled rather than cut off: both bars are there
    tw, th = thumb.size
    assert thumb.getpixel((tw // 2, 1)) < 50 and thumb.getpixel((tw // 2, th - 2)) < 50
    assert thumb.getpixel((tw // 2, th // 2)) > 200
    thumb.close()


def test_scaled_down_is_reported_for_any_render_of_the_page(tmp_path: Path) -> None:
    # 226 x 6000 pt: the OCR render fits, a 600 dpi render would not (even after the 60 MP cap)
    src = _custom(tmp_path / "p.pdf", (226, 6000), b"0 g 0 0 226 40 re f")
    with Renderer(src) as r:
        r.ocr_dpi(0)
        assert not r.render_scaled_down(0)
        img = r.render_scaled(0, 600.0, gray=True)
        assert max(img.size) <= MAX_RENDER_SIDE
        img.close()
        assert r.render_scaled_down(0)                    # sticky for the run
        r.render_scaled(0, 20.0, gray=True).close()
        assert r.render_scaled_down(0)


def test_render_to_png_reports_scaled_down(tmp_path: Path) -> None:
    # A fresh Renderer: only render() itself can set the flag here, since the OCR render fits.
    src = _custom(tmp_path / "p.pdf", (226, 6000), b"0 g 0 0 226 40 re f")
    with Renderer(src) as r:
        assert not r.render_scaled_down(0)
        pi = r.render(0, 600.0, tmp_path / "p.png")
        assert pi.dpi == 393.0                            # the 60 MP cap, then the side limit
        assert max(pi.width, pi.height) <= MAX_RENDER_SIDE
        assert r.render_scaled_down(0)


def test_ordinary_page_is_not_scaled_down(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(texts=(build.TextSpec("Sample"),))])
    with Renderer(src) as r:
        r.render(0, r.ocr_dpi(0), tmp_path / "p.png")
        assert not r.render_scaled_down(0)


def test_fit_dpi_flags() -> None:
    assert render._fit_dpi(612.0, 792.0, 300.0) == (300.0, False, False)
    used, capped, scaled = render._fit_dpi(5000.0, 5000.0, 200.0)
    assert capped and not scaled and used < 200.0
    used, capped, scaled = render._fit_dpi(*RECEIPT, 200.0)
    assert not capped and scaled and used == float(math.floor(72 * MAX_RENDER_SIDE / RECEIPT[1]))


# --------------------------------------------------------------------------- OSD upscale

def test_upscale_for_osd() -> None:
    img = Image.new("L", (200, 100), 255)
    up = upscale_for_osd(img, 150.0)
    assert up.size == (400, 200)
    assert upscale_for_osd(img, 300.0) is img
    assert upscale_for_osd(img, 600.0) is img
    huge = Image.new("L", (6000, 6000), 255)
    capped = upscale_for_osd(huge, 100.0)
    assert capped.size[0] * capped.size[1] <= MAX_RENDER_PIXELS
    assert capped.size[0] > 6000
    strip = Image.new("L", (600, 20000), 255)          # a receipt render at 150 dpi: 2x would be 40000 px
    up_strip = upscale_for_osd(strip, 150.0)
    assert max(up_strip.size) <= MAX_RENDER_SIDE
    assert up_strip.size[1] > 30000 and up_strip.size[0] > 600
    with pytest.raises(ValueError):
        upscale_for_osd(img, 0.0)


# --------------------------------------------------------------------------- the lock

def test_two_threads_through_the_lock(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "p.pdf", [build.PageSpec(texts=(build.TextSpec(f"Line {i}"),)) for i in range(3)])
    errors: list[BaseException] = []
    results: dict[int, list[bytes]] = {0: [], 1: []}
    with Renderer(src) as r:
        def work(slot: int) -> None:
            try:
                for i in range(12):
                    with PDFIUM_LOCK:                      # the caller may hold it too: re-entrant
                        img = r.render_scaled(i % 3, 50.0, gray=True)
                    results[slot].append(img.tobytes())
                    r.ocr_dpi(i % 3)
                    r.placed_images(i % 3)
            except BaseException as exc:                   # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors
    assert results[0] == results[1]


@pytest.mark.skipif(not __debug__, reason="debug assertions are off under python -O")
def test_a_call_without_the_lock_is_caught(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        assert_locked()
    with PDFIUM_LOCK:
        assert_locked()
    with Renderer(build.write_pdf(tmp_path / "p.pdf", 1)) as r:
        with pytest.raises(AssertionError, match="PDFIUM_LOCK"):
            r._page(0)                                     # pdfium reached without the lock
        with PDFIUM_LOCK:
            r._page(0).close()


def test_lock_is_held_in_another_thread_not_ours() -> None:
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with PDFIUM_LOCK:
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(5)
    try:
        if __debug__:
            with pytest.raises(AssertionError):
                assert_locked()                            # another thread's hold doesn't count
    finally:
        release.set()
        t.join()
