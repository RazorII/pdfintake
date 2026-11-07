"""intake/sniff.py: kinds by bytes, the one-format open, declared depth, lossless WebP."""

from __future__ import annotations

import io
import os
import stat
import struct
import sys
import threading
import zlib
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from fixtures import build
from pdfintake.errors import LosslessUnavailable, UsageError
from pdfintake.intake import sniff
from pdfintake.intake.sniff import PIL_FORMAT, declared_depth, open_image, webp_is_lossless

# --------------------------------------------------------------------------- an audit hook that records process starts

_PROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile", "os.fork")
_RECORDER: dict[str, object] = {"on": False, "events": []}


def _hook(event: str, args: tuple) -> None:
    if _RECORDER["on"] and event.startswith(_PROCESS_EVENTS):
        _RECORDER["events"].append(event)  # type: ignore[union-attr]


sys.addaudithook(_hook)


@pytest.fixture
def process_starts():
    _RECORDER["events"] = []
    _RECORDER["on"] = True
    try:
        yield _RECORDER["events"]
    finally:
        _RECORDER["on"] = False


# --------------------------------------------------------------------------- fixtures built in code

EPS = (b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n%%EndComments\n"
       b"newpath 0 0 moveto 10 10 lineto stroke\nshowpage\n%%EOF\n")


def _tiff(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    return buf.getvalue()


def _save(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _heif_bytes() -> bytes | None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        return _save(build.photo_like(64, 48), "HEIF", quality=80)
    except Exception:                                    # noqa: BLE001 - no HEVC encoder on this build
        return None


def _avif_bytes() -> bytes | None:
    from PIL import features

    if not features.check("avif"):
        return None
    try:
        return _save(build.photo_like(64, 48), "AVIF", quality=80)
    except Exception:                                    # noqa: BLE001 - no AV1 encoder in this build
        return None


def _bmff(major: bytes, *compat: bytes) -> bytes:
    body = major + b"\x00\x00\x00\x00" + b"".join(compat)
    return struct.pack(">I", 8 + len(body)) + b"ftyp" + body + b"\x00" * 32


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- sniff

def test_each_kind_is_sniffed_by_bytes_not_by_name(tmp_path: Path):
    img = build.photo_like(40, 30)
    cases = {
        "pdf": build.pdf_bytes(1),
        "jpeg": build.jpeg_bytes(img),
        "png": build.png_bytes(img),
        "tiff": _tiff(img),
        "webp": _save(img, "WEBP", lossless=True),
        "gif": _save(img.convert("P"), "GIF"),
        "bmp": _save(img, "BMP"),
        "unknown": b"plain text, not an image\n" * 4,
    }
    heif, avif = _heif_bytes(), _avif_bytes()
    if heif is not None:
        cases["heif"] = heif
    if avif is not None:
        cases["avif"] = avif
    for kind, data in cases.items():
        # every file is named as if it were something else, so only the bytes can decide
        wrong = "x.png" if kind != "png" else "x.jpg"
        assert sniff.sniff(_write(tmp_path, f"{kind}-{wrong}", data)) == kind, kind
    assert set(cases) | {"heif", "avif"} == set(sniff.INPUT_KINDS)


def test_tiff_byte_orders_and_bigtiff_are_tiff():
    for head in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        assert sniff._sniff_bytes(head + b"\x00" * 12) == "tiff"


def test_a_pdf_header_after_leading_junk_is_still_a_pdf():
    assert sniff._sniff_bytes(b"\x00" * 300 + b"%PDF-1.7\n") == "pdf"
    assert sniff._sniff_bytes(b"\x00" * 1100 + b"%PDF-1.7\n") == "unknown"


def test_bmff_brands_decide_heif_or_avif():
    assert sniff._sniff_bytes(_bmff(b"heic", b"mif1", b"heic")) == "heif"
    assert sniff._sniff_bytes(_bmff(b"mif1", b"heic")) == "heif"
    assert sniff._sniff_bytes(_bmff(b"avif", b"mif1", b"miaf")) == "avif"
    assert sniff._sniff_bytes(_bmff(b"mif1", b"avif")) == "avif"
    assert sniff._sniff_bytes(_bmff(b"avis", b"msf1")) == "avif"
    assert sniff._sniff_bytes(_bmff(b"isom", b"mp41")) == "unknown"       # an MP4 is not an image


def test_bmp_needs_a_real_dib_header():
    assert sniff._sniff_bytes(b"BM" + b"\x00" * 12 + struct.pack("<I", 40) + b"\x00" * 20) == "bmp"
    assert sniff._sniff_bytes(b"BMagic words that happen to start with BM") == "unknown"


def test_eps_is_unknown_whatever_its_name(tmp_path: Path):
    assert sniff.sniff(_write(tmp_path, "scan.jpg", EPS)) == "unknown"
    dos_eps = b"\xc5\xd0\xd3\xc6" + struct.pack("<IIIIII", 30, len(EPS), 0, 0, 0, 0) + b"\xff\xff" + EPS
    assert sniff.sniff(_write(tmp_path, "photo.jpeg", dos_eps)) == "unknown"


# --------------------------------------------------------------------------- only a regular file is read (C20-A2)

def _sniff_in_thread(path: Path, *, unstick: Path | None = None, wait: float = 20.0) -> object:
    """sniff(path) on a worker thread: its result, or the exception it raised. A sniff still
    running after ``wait`` seconds fails the test (after opening a writer on ``unstick``, a FIFO,
    so the thread can end)."""
    out: dict[str, object] = {}

    def work() -> None:
        try:
            out["result"] = sniff.sniff(path)
        except BaseException as exc:                    # noqa: BLE001 - handed back to the test
            out["result"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(wait)
    if worker.is_alive():
        if unstick is not None:
            try:
                os.close(os.open(unstick, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass
            worker.join(10)
        pytest.fail(f"sniff waited on {path.name}")
    return out["result"]


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs")
def test_a_fifo_is_refused_without_waiting_for_a_writer(tmp_path: Path):
    fifo = tmp_path / "Pipe.pdf"
    os.mkfifo(fifo)
    before = _open_fds()
    got = _sniff_in_thread(fifo, unstick=fifo)
    assert isinstance(got, UsageError), got
    assert got.code == "input_not_file" and got.exit_code == 2
    assert got.detail["path"] == str(fifo) and "isn't a regular file" in got.message
    assert _open_fds() == before                         # the descriptor was closed on the way out


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs")
def test_a_file_swapped_for_a_fifo_after_the_callers_stat_is_still_refused(tmp_path: Path):
    path = _write(tmp_path, "Scan.pdf", build.pdf_bytes(1))
    assert stat.S_ISREG(os.stat(path).st_mode)          # what the caller checked first
    path.unlink()
    os.mkfifo(path)                                     # swapped before sniff opens it
    got = _sniff_in_thread(path, unstick=path)
    assert isinstance(got, UsageError) and got.code == "input_not_file", got


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX devices and folders")
def test_a_device_or_a_folder_is_refused(tmp_path: Path):
    for path in (Path(os.devnull), tmp_path):
        got = _sniff_in_thread(path)
        assert isinstance(got, UsageError) and got.code == "input_not_file", (path, got)


@pytest.mark.skipif(sys.platform == "win32", reason="/dev/stdin")
def test_dev_stdin_is_never_read(tmp_path: Path):
    # Standard input is made a pipe with a live writer for the length of the test, so the check
    # runs the same however pytest was started, and a sniff that read it would block (and fail
    # on the thread timeout) rather than pass. What could leak to the caller's standard input is
    # O_NONBLOCK on the shared open file description, so its flags are compared, not st_mode.
    # decided 2026-11-06
    import fcntl

    read_end, write_end = os.pipe()
    saved = os.dup(0)
    try:
        os.dup2(read_end, 0)
        flags = fcntl.fcntl(0, fcntl.F_GETFL)
        assert not flags & os.O_NONBLOCK
        got = _sniff_in_thread(Path("/dev/stdin"))
        assert isinstance(got, UsageError) and got.code == "input_not_file", got
        assert fcntl.fcntl(0, fcntl.F_GETFL) == flags   # standard input itself is left alone
        assert fcntl.fcntl(read_end, fcntl.F_GETFL) == flags
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)


def test_a_missing_input_is_an_oserror(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        sniff.sniff(tmp_path / "missing.pdf")


def test_a_header_past_the_first_read_is_still_found_and_the_rest_is_not_read(tmp_path: Path):
    assert sniff.sniff(_write(tmp_path, "late.pdf", b"\x00" * 1000 + b"%PDF-1.7\n" + b"\x00" * 100_000)) == "pdf"
    assert sniff.sniff(_write(tmp_path, "empty.pdf", b"")) == "unknown"
    assert sniff.sniff(_write(tmp_path, "sig.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)) == "png"   # bytes untranslated


# --------------------------------------------------------------------------- open_image

def test_open_image_uses_only_the_sniffed_format(tmp_path: Path):
    path = _write(tmp_path, "x.png", build.jpeg_bytes(build.photo_like(20, 10)))
    with open_image(path, "jpeg") as img:
        assert img.format == "JPEG" and img.size == (20, 10)
    with pytest.raises(UnidentifiedImageError):
        open_image(path, "png")


def test_an_eps_renamed_jpg_raises_and_never_reaches_the_eps_plugin(tmp_path: Path, monkeypatch, process_starts):
    from PIL import EpsImagePlugin

    path = _write(tmp_path, "scan.jpg", EPS)
    accept = getattr(EpsImagePlugin, "_accept", None)
    if accept is not None:
        assert accept(EPS[:64]), "the fixture must be an EPS Pillow's EPS plugin would take"

    def refuse(*args, **kwargs):
        raise AssertionError("EpsImagePlugin was reached")

    monkeypatch.setattr(EpsImagePlugin.EpsImageFile, "_open", refuse)
    monkeypatch.setattr(EpsImagePlugin, "Ghostscript", refuse, raising=False)
    kind = sniff.sniff(path)
    assert kind == "unknown"
    with pytest.raises(UsageError) as err:
        open_image(path, kind)
    assert err.value.code == "not_an_image" and "scan.jpg" in str(err.value)
    with pytest.raises(UnidentifiedImageError):            # even when told it's a JPEG
        open_image(path, "jpeg")
    for kind in PIL_FORMAT:
        with pytest.raises(UnidentifiedImageError):
            open_image(path, kind)
    assert process_starts == [], f"a process was started: {process_starts}"


def test_open_image_refuses_more_than_200_megapixels(tmp_path: Path):
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    def huge(w: int, h: int) -> bytes:
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))

    for w, h in ((15000, 15000), (30000, 30000)):          # above the limit, and above Pillow's own 2x error
        path = _write(tmp_path, f"huge-{w}.png", huge(w, h))
        with pytest.raises(LosslessUnavailable) as err:
            open_image(path, "png")
        assert err.value.code == "too_many_pixels" and path.name in str(err.value)
    assert Image.MAX_IMAGE_PIXELS == sniff.MAX_IMAGE_PIXELS == 200_000_000


# --------------------------------------------------------------------------- declared_depth

@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_declared_depth_of_grey_png(tmp_path: Path, depth: int):
    path = _write(tmp_path, "g.png", build.raw_png(9, 5, color_type=0, bit_depth=depth))
    assert declared_depth(path, "png") == depth


def test_declared_depth_of_colour_and_palette_png(tmp_path: Path):
    assert declared_depth(_write(tmp_path, "c16.png", build.raw_png(4, 3, color_type=2, bit_depth=16)), "png") == 16
    assert declared_depth(_write(tmp_path, "c8.png", build.raw_png(4, 3, color_type=6, bit_depth=8)), "png") == 8
    assert declared_depth(_write(tmp_path, "p4.png", build.raw_png(4, 3, color_type=3, bit_depth=4)), "png") == 4


def _tiff_by_hand(w: int, h: int, bps: tuple[int, ...], photometric: int, *, order: str = ">") -> bytes:
    """An uncompressed single-strip TIFF with any BitsPerSample, written by hand."""
    spp = len(bps)
    row = (w * sum(bps) + 7) // 8
    data = bytes(range(256)) * ((row * h) // 256 + 1)
    data = data[:row * h]
    n = 10
    ifd_at = 8
    extra_at = ifd_at + 2 + 12 * n + 4
    bps_inline = spp * 2 <= 4
    data_at = extra_at + (0 if bps_inline else 2 * spp)
    entries = [
        (256, 3, 1, w), (257, 3, 1, h), (258, 3, spp, None), (259, 3, 1, 1), (262, 3, 1, photometric),
        (273, 4, 1, data_at), (277, 3, 1, spp), (278, 3, 1, h), (279, 4, 1, len(data)), (284, 3, 1, 1),
    ]
    out = (b"MM\x00*" if order == ">" else b"II*\x00") + struct.pack(order + "I", ifd_at)
    out += struct.pack(order + "H", n)
    for tag, typ, cnt, val in entries:
        if tag == 258:
            if bps_inline:
                out += struct.pack(order + "HHI", tag, typ, cnt) + struct.pack(order + "HH", *(list(bps) + [0])[:2])
            else:
                out += struct.pack(order + "HHII", tag, typ, cnt, extra_at)
        elif typ == 3:
            out += struct.pack(order + "HHIHH", tag, typ, cnt, val, 0)
        else:
            out += struct.pack(order + "HHII", tag, typ, cnt, val)
    out += b"\x00\x00\x00\x00"
    if not bps_inline:
        out += struct.pack(order + "H" * spp, *bps)
    return out + data


def test_declared_depth_of_tiff(tmp_path: Path):
    assert declared_depth(_write(tmp_path, "1.tif", _tiff(build.blank(16, 8, mode="1"))), "tiff") == 1
    assert declared_depth(_write(tmp_path, "8.tif", _tiff(build.photo_like(16, 8, mode="L"))), "tiff") == 8
    assert declared_depth(_write(tmp_path, "16.tif", _tiff(build.blank(16, 8, mode="I;16"))), "tiff") == 16
    assert declared_depth(_write(tmp_path, "rgb16.tif", _tiff_by_hand(6, 4, (16, 16, 16), 2)), "tiff") == 16
    assert declared_depth(_write(tmp_path, "rgb16le.tif", _tiff_by_hand(6, 4, (16, 16, 16), 2, order="<")), "tiff") == 16
    assert declared_depth(_write(tmp_path, "g4.tif", _tiff_by_hand(6, 4, (4,), 1)), "tiff") == 4
    assert declared_depth(_write(tmp_path, "rgb8.tif", _tiff(build.photo_like(16, 8))), "tiff") == 8


def test_declared_depth_of_jpeg_bmp_gif_webp(tmp_path: Path):
    img = build.photo_like(16, 8)
    jpg = build.jpeg_bytes(img)
    sof = jpg.index(b"\xff\xc0")
    j12 = jpg[:sof] + b"\xff\xc1" + jpg[sof + 2:sof + 4] + b"\x0c" + jpg[sof + 5:]
    assert declared_depth(_write(tmp_path, "8.jpg", jpg), "jpeg") == 8
    assert declared_depth(_write(tmp_path, "12.jpg", j12), "jpeg") == 12
    assert declared_depth(_write(tmp_path, "24.bmp", _save(img, "BMP")), "bmp") == 8
    assert declared_depth(_write(tmp_path, "1.bmp", _save(build.blank(16, 8, mode="1"), "BMP")), "bmp") == 1
    assert declared_depth(_write(tmp_path, "p.bmp", _save(img.convert("P"), "BMP")), "bmp") == 8
    assert declared_depth(_write(tmp_path, "g.gif", _save(img.convert("P"), "GIF")), "gif") == 8
    assert declared_depth(_write(tmp_path, "w.webp", _save(img, "WEBP")), "webp") == 8


def test_declared_depth_of_bmp_bitfields(tmp_path: Path):
    """A 32-bit BMP with 10-10-10-2 bit fields declares 10 bits per sample."""
    w, h = 2, 2
    masks = (0x3FF00000, 0x000FFC00, 0x000003FF, 0xC0000000)
    pixels = b"\x00" * (w * h * 4)
    header = struct.pack("<IiiHHIIiiII", 56, w, h, 1, 32, 3, len(pixels), 2835, 2835, 0, 0) + struct.pack("<IIII", *masks)
    data = b"BM" + struct.pack("<IHHI", 14 + len(header) + len(pixels), 0, 0, 14 + len(header)) + header + pixels
    path = _write(tmp_path, "ten.bmp", data)
    assert sniff.sniff(path) == "bmp"
    assert declared_depth(path, "bmp") == 10


def test_declared_depth_of_heif_and_avif(tmp_path: Path):
    heif, avif = _heif_bytes(), _avif_bytes()
    if heif is None and avif is None:
        pytest.skip("neither a HEIF nor an AVIF encoder is available here")
    if heif is not None:
        assert declared_depth(_write(tmp_path, "x.heic", heif), "heif") == 8
    if avif is not None:
        assert declared_depth(_write(tmp_path, "x.avif", avif), "avif") == 8


def test_declared_depth_reads_pixi_then_codec_config():
    def box(typ: bytes, body: bytes) -> bytes:
        return struct.pack(">I", 8 + len(body)) + typ + body

    pixi = box(b"pixi", b"\x00\x00\x00\x00" + bytes([3, 10, 10, 10]))
    hvcc = box(b"hvcC", bytes(17) + bytes([0xF8 | 4]) + bytes(5))          # bitDepthLumaMinus8 = 4: 12-bit
    av1c = box(b"av1C", bytes([0x81, 0x00, 0x40, 0x00]))                    # high_bitdepth: 10-bit
    for props, want in ((pixi + hvcc, 10), (hvcc, 12), (av1c, 10), (b"", 8)):
        meta = box(b"meta", b"\x00\x00\x00\x00" + box(b"iprp", box(b"ipco", props)))
        assert sniff._bmff_depth(_bmff(b"heic", b"mif1")[:20] + meta) == want


def test_declared_depth_refuses_non_images(tmp_path: Path):
    path = _write(tmp_path, "x.pdf", build.pdf_bytes(1))
    for kind in ("pdf", "unknown"):
        with pytest.raises(ValueError):
            declared_depth(path, kind)


# --------------------------------------------------------------------------- webp_is_lossless

def test_webp_is_lossless():
    img = build.photo_like(32, 24)
    assert webp_is_lossless(_save(img, "WEBP", lossless=True))
    assert webp_is_lossless(_save(img.convert("RGBA"), "WEBP", lossless=True))
    assert not webp_is_lossless(_save(img, "WEBP", quality=80))
    assert not webp_is_lossless(_save(img.convert("RGBA"), "WEBP", quality=80))      # VP8X + ALPH + VP8
    ex = Image.Exif()
    ex[0x0112] = 6
    assert webp_is_lossless(_save(img, "WEBP", lossless=True, exif=ex.tobytes()))     # VP8X + VP8L + EXIF
    assert not webp_is_lossless(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    assert not webp_is_lossless(build.png_bytes(img))


def test_an_animated_webp_is_lossless_only_when_every_frame_is():
    def chunk(fourcc: bytes, body: bytes) -> bytes:
        return fourcc + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) % 2 else b"")

    def riff(*chunks: bytes) -> bytes:
        body = b"WEBP" + b"".join(chunks)
        return b"RIFF" + struct.pack("<I", len(body)) + body

    frame = bytes(16)
    lossless = chunk(b"ANMF", frame + chunk(b"VP8L", b"\x2f" + bytes(8)))
    lossy = chunk(b"ANMF", frame + chunk(b"VP8 ", bytes(10)))
    head = chunk(b"VP8X", bytes(10)) + chunk(b"ANIM", bytes(6))
    assert webp_is_lossless(riff(head, lossless, lossless))
    assert not webp_is_lossless(riff(head, lossless, lossy))


def test_a_pdf_header_mentioned_inside_text_is_not_a_pdf():
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n% %PDF-1.4 mention\n%PDF-1.4\nshowpage\n"
    assert sniff._sniff_bytes(eps) == "unknown"
    dos_eps = b"\xc5\xd0\xd3\xc6" + b"\x00" * 28 + b"%PDF-1.4\n"
    assert sniff._sniff_bytes(dos_eps) == "unknown"
    assert sniff._sniff_bytes(b"notes: see %PDF-1.7 and 1 0 obj\n") == "unknown"
    assert sniff._sniff_bytes(b"junk line\n%PDF-1.7\n") == "pdf"
    assert sniff._sniff_bytes(b"%PDF-1.7\n") == "pdf"
