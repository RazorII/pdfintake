"""intake/jpeg.py: segment parsing, the frame checks, trailing MPF data, metadata settling and stripping."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fixtures import build
from pdfintake.intake import jpeg as J
from pdfintake.intake.jpeg import (JpegInfo, carried_segments, check_jpeg, jpeg_end, jpeg_info, join_jpeg,
                                   passthrough_ok, settle_metadata, split_jpeg, strip_jpeg)


def seg(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def fake_icc(space: bytes = b"RGB ", size: int = 400) -> bytes:
    """A profile-shaped blob: ICC header fields (size, colour space, 'acsp') and filler."""
    head = bytearray(128)
    struct.pack_into(">I", head, 0, size)
    head[12:16] = b"mntr"
    head[16:20] = space
    head[20:24] = b"XYZ "
    head[36:40] = b"acsp"
    return bytes(head) + bytes((i * 7) % 256 for i in range(size - 128))


def xmp(orientation: int = 6, w: int = 99, h: int = 77) -> bytes:
    return (b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            b'<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/"'
            b' tiff:Orientation="%d" exif:PixelXDimension="%d"><exif:PixelYDimension>%d</exif:PixelYDimension>'
            b'</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>' % (orientation, w, h))


def exif_tiff(*, orientation: int = 6, width: int = 99, height: int = 77, thumbnail: bool = True) -> bytes:
    """An EXIF TIFF block by hand: IFD0 (Orientation SHORT, ImageWidth/Length LONG, Make, Exif IFD),
    the Exif IFD (PixelX LONG, PixelY SHORT) and an IFD1 thumbnail pointer."""
    make = b"ExampleCam\x00"
    ifd0_n, exif_n, ifd1_n = 5, 2, 2
    ifd0 = 8
    exif_at = ifd0 + 2 + 12 * ifd0_n + 4
    ifd1 = exif_at + 2 + 12 * exif_n + 4
    make_at = ifd1 + 2 + 12 * ifd1_n + 4
    thumb_at = make_at + len(make)
    thumb = build.jpeg_bytes(build.blank(8, 8, color=(40, 40, 40)))
    out = b"MM\x00*" + struct.pack(">I", ifd0)
    out += struct.pack(">H", ifd0_n)
    out += struct.pack(">HHIHH", 0x0100, 3, 1, width, 0)            # ImageWidth as SHORT
    out += struct.pack(">HHII", 0x0101, 4, 1, height)               # ImageLength as LONG
    out += struct.pack(">HHII", 0x010F, 2, len(make), make_at)
    out += struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
    out += struct.pack(">HHII", 0x8769, 4, 1, exif_at)
    out += struct.pack(">I", ifd1 if thumbnail else 0)
    out += struct.pack(">H", exif_n)
    out += struct.pack(">HHII", 0xA002, 4, 1, width)
    out += struct.pack(">HHIHH", 0xA003, 3, 1, height, 0)
    out += struct.pack(">I", 0)
    out += struct.pack(">H", ifd1_n)
    out += struct.pack(">HHII", 0x0201, 4, 1, thumb_at)
    out += struct.pack(">HHII", 0x0202, 4, 1, len(thumb))
    out += struct.pack(">I", 0)
    return out + make + thumb


def rich_jpeg(img: Image.Image | None = None, *, icc: bytes | None = None, trailing: bytes = b"",
              mpf: bool = False, extra: tuple[bytes, ...] = ()) -> bytes:
    """A Pillow JPEG with EXIF, XMP, IPTC, a comment, a JFXX thumbnail and an APP12 spliced in.

    With ``mpf``, an MPF index describing ``trailing`` as a second image (as a camera's gain map is).
    """
    img = img or build.photo_like(48, 32)
    buf = io.BytesIO()
    kw = {"quality": 90}
    if icc is not None:
        kw["icc_profile"] = icc
    img.save(buf, format="JPEG", **kw)
    data = buf.getvalue()
    segs, scan = split_jpeg(data)
    meta = [seg(0xE1, J.EXIF_ID + exif_tiff()), seg(0xE1, J.XMP_ID + xmp()), seg(0xED, b"Photoshop 3.0\x008BIM" + bytes(12)),
            seg(0xFE, b"an example comment"), seg(0xE0, b"JFXX\x00\x10" + bytes(4)), seg(0xEC, b"Ducky\x00\x01")]
    meta += list(extra)
    at = 1 if J._kind(segs[0]) == "jfif" else 0

    def build_it(index: bytes | None) -> bytes:
        head = segs[:at] + meta + ([seg(0xE2, index)] if index is not None else []) + segs[at:]
        return join_jpeg(head, scan) + trailing

    if not mpf:
        return build_it(None)
    first = build_it(mpf_index(0, 0, 0))
    tiff_at = first.index(b"MPF\x00") + 4                  # MPF offsets count from its TIFF header
    primary = len(first) - len(trailing)
    return build_it(mpf_index(primary, len(trailing), primary - tiff_at))


def mpf_index(primary_size: int, second_size: int, second_offset: int) -> bytes:
    """An APP2 MPF payload (CIPA DC-007) listing a primary image and one secondary image."""
    entries_at = 8 + 2 + 12 * 3 + 4
    ifd = struct.pack(">H", 3)
    ifd += struct.pack(">HHI", 0xB000, 7, 4) + b"0100"
    ifd += struct.pack(">HHII", 0xB001, 4, 1, 2)
    ifd += struct.pack(">HHII", 0xB002, 7, 32, entries_at)
    ifd += struct.pack(">I", 0)
    entries = struct.pack(">IIIHH", 0x20030000, primary_size, 0, 0, 0)
    entries += struct.pack(">IIIHH", 0x00000000, second_size, second_offset, 0, 0)
    return b"MPF\x00" + b"MM\x00*" + struct.pack(">I", 8) + ifd + entries


# --------------------------------------------------------------------------- segment parsing

def test_split_and_join_round_trip_with_kinds():
    data = rich_jpeg(icc=fake_icc())
    segs, scan = split_jpeg(data)
    assert join_jpeg(segs, scan) == data
    assert all(s[0] == 0xFF and struct.unpack(">H", s[2:4])[0] == len(s) - 2 for s in segs)
    kinds = [J._kind(s) for s in segs]
    for k in ("jfif", "exif", "xmp", "iptc", "com", "jfxx", "app", "icc"):
        assert k in kinds, k
    assert segs[-1][1] == 0xDA                              # SOS ends the header
    assert scan.endswith(b"\xff\xd9") and jpeg_end(data) == len(data)


def test_jpeg_info_fields():
    icc = fake_icc(size=70000)                             # Pillow splits this over two APP2 chunks
    img = build.photo_like(48, 32)
    data = rich_jpeg(img, icc=icc)
    assert sum(1 for s in split_jpeg(data)[0] if J._kind(s) == "icc") == 2
    info = jpeg_info(data)
    with Image.open(io.BytesIO(data), formats=["JPEG"]) as im:
        assert info.icc == im.info["icc_profile"] == icc
    assert info == JpegInfo(width=48, height=32, sof=0xC0, precision=8, components=3, adobe_transform=None,
                            icc=icc, orientation=6, has_mpf=False, trailing_bytes=0)
    grey = jpeg_info(build.jpeg_bytes(build.photo_like(20, 10, mode="L")))
    assert (grey.components, grey.icc, grey.orientation) == (1, None, 1)


def test_cmyk_adobe_transform_and_progressive_frames():
    buf = io.BytesIO()
    build.photo_like(24, 16).convert("CMYK").save(buf, format="JPEG", quality=90)
    cmyk = jpeg_info(buf.getvalue())
    assert cmyk.components == 4 and cmyk.adobe_transform is not None
    prog = build.jpeg_bytes(build.photo_like(64, 48), progressive=True)
    info = jpeg_info(prog)
    assert info.sof == 0xC2 and info.trailing_bytes == 0 and passthrough_ok(info)
    assert jpeg_end(prog) == len(prog)                    # tables between the progressive scans are skipped


def test_fill_bytes_and_standalone_markers_in_the_header_are_skipped():
    data = build.jpeg_bytes(build.photo_like(16, 16))
    padded = data[:2] + b"\xff\xff\xff" + data[2:]          # fill bytes before the first marker
    assert jpeg_info(padded).width == 16
    segs, scan = split_jpeg(padded)
    assert join_jpeg(segs, scan) == data


@pytest.mark.parametrize("bad", [
    b"not a jpeg",
    b"\xff\xd8\xff\xe0\x00",                                # cut off inside a length
    b"\xff\xd8\xff\xe0\x00\x01",                            # length below 2
    b"\xff\xd8\xff\xe0\x00\x10JFIF",                        # runs past the end
])
def test_malformed_headers_raise(bad: bytes):
    with pytest.raises(ValueError):
        jpeg_info(bad)


def test_a_jpeg_without_eoi_raises():
    data = build.jpeg_bytes(build.photo_like(16, 16))
    with pytest.raises(ValueError, match="EOI"):
        jpeg_end(data[:-2])
    assert J._frame(data[:-2])[2:4] == (16, 16)             # the frame header alone still reads


# --------------------------------------------------------------------------- trailing MPF data

def test_trailing_mpf_bytes_after_eoi_stay_in_a_passthrough():
    from pdfintake.intake.images import jpeg_payload

    gain_map = build.jpeg_bytes(build.photo_like(24, 16, seed=5), quality=70)
    data = rich_jpeg(mpf=True, trailing=gain_map)
    info = jpeg_info(data)
    assert info.has_mpf and info.trailing_bytes == len(gain_map)
    assert jpeg_end(data) == len(data) - len(gain_map)
    assert check_jpeg(data, passthrough=True) == []
    payload = jpeg_payload(data, info)
    assert payload.data == data                            # every byte, the gain map included
    segs, scan = split_jpeg(data)
    assert join_jpeg(segs, scan) == data[:jpeg_end(data)]   # split never carries what follows EOI


# --------------------------------------------------------------------------- passthrough_ok

def _with_frame(data: bytes, sof: int, precision: int | None = None, comps: int | None = None) -> bytes:
    at = data.index(b"\xff\xc0")
    head = bytearray(data[at:at + 10])
    head[1] = sof
    if precision is not None:
        head[4] = precision
    if comps is not None:
        head[9] = comps
    return data[:at] + bytes(head) + data[at + 10:]


def test_sof_c1_12_bit_and_sof_c3_fail_passthrough_ok():
    base = build.jpeg_bytes(build.photo_like(16, 16))
    assert passthrough_ok(jpeg_info(base))
    assert passthrough_ok(jpeg_info(_with_frame(base, 0xC1)))                  # extended, 8-bit: fine
    assert not passthrough_ok(jpeg_info(_with_frame(base, 0xC1, precision=12)))
    assert not passthrough_ok(jpeg_info(_with_frame(base, 0xC3)))              # lossless
    assert not passthrough_ok(jpeg_info(_with_frame(base, 0xC9)))              # arithmetic
    assert not passthrough_ok(jpeg_info(_with_frame(base, 0xC0, comps=2)))
    problems = check_jpeg(_with_frame(base, 0xC1, precision=12), passthrough=True)
    assert any("12-bit" in p for p in problems)


# --------------------------------------------------------------------------- settle_metadata

def test_settle_metadata_gives_orientation_1_and_the_new_size():
    img = build.photo_like(48, 32)
    data = rich_jpeg(img, mpf=True)
    before = check_jpeg(data, passthrough=False)
    assert any("Orientation" in p for p in before) and any("MPF" in p for p in before)
    settled = settle_metadata(data, size=(48, 32))
    assert check_jpeg(settled, passthrough=False) == []
    segs, scan = split_jpeg(settled)
    kinds = [J._kind(s) for s in segs]
    assert "mpf" not in kinds and "jfxx" not in kinds
    assert scan == split_jpeg(data)[1]                     # entropy-coded data untouched
    tiff = next(J._payload(s)[6:] for s in segs if J._kind(s) == "exif")
    assert len(tiff) == len(exif_tiff())                    # patched in place, same length
    vals = J._tiff_values(tiff)
    assert vals[0x0112] == 1 and vals[0x0100] == 48 and vals[0x0101] == 32
    assert vals[0xA002] == 48 and vals[0xA003] == 32
    assert struct.unpack_from(">I", tiff, 8 + 2 + 12 * 5)[0] == 0      # IFD1 (the thumbnail) unlinked
    with Image.open(io.BytesIO(settled), formats=["JPEG"]) as im:
        ex = im.getexif()
        assert ex[0x0112] == 1 and ex[0x010F] == "ExampleCam"
        assert ex.get_ifd(0x8769)[0xA002] == 48
        assert np.array_equal(np.asarray(im), np.asarray(Image.open(io.BytesIO(data), formats=["JPEG"])))
    x = next(J._payload(s) for s in segs if J._kind(s) == "xmp")
    assert b'tiff:Orientation="1"' in x and b'exif:PixelXDimension="48"' in x
    assert b"<exif:PixelYDimension>32</exif:PixelYDimension>" in x
    for kind in ("iptc", "com", "app"):                    # everything else carried as it was
        assert kind in kinds


def test_settle_metadata_drops_an_exif_block_it_cannot_patch():
    broken = seg(0xE1, J.EXIF_ID + b"II*\x00\xff\xff\xff\x7f")       # IFD offset past the end
    data = rich_jpeg(extra=(broken,))
    settled = settle_metadata(data, size=(48, 32))
    exifs = [s for s in split_jpeg(settled)[0] if J._kind(s) == "exif"]
    assert len(exifs) == 1 and J._tiff_values(J._payload(exifs[0])[6:])[0x0112] == 1


def test_settle_metadata_refuses_a_size_that_is_not_the_frame():
    with pytest.raises(ValueError):
        settle_metadata(rich_jpeg(), size=(32, 48))


# --------------------------------------------------------------------------- carried segments, strip

def test_carried_segments_are_the_metadata_only():
    data = rich_jpeg(icc=fake_icc(), mpf=True)
    kinds = [J._kind(s) for s in carried_segments(data)]
    assert set(kinds) == {"exif", "xmp", "iptc", "com", "app"}
    assert carried_segments(build.jpeg_bytes(build.photo_like(8, 8))) == []


def test_strip_jpeg_keeps_the_picture_and_its_profile():
    icc = fake_icc()
    gain_map = build.jpeg_bytes(build.photo_like(8, 8))
    data = rich_jpeg(icc=icc, mpf=True, trailing=gain_map)
    stripped = strip_jpeg(data)
    kinds = {J._kind(s) for s in split_jpeg(stripped)[0]}
    assert kinds == {"core", "jfif", "icc"}
    assert split_jpeg(stripped)[1] == split_jpeg(data)[1]
    assert not stripped.endswith(gain_map) and jpeg_end(stripped) == len(stripped)
    assert jpeg_info(stripped).icc == icc
    assert {J._kind(s) for s in split_jpeg(strip_jpeg(data, keep_icc=False))[0]} == {"core", "jfif"}
    with Image.open(io.BytesIO(stripped)) as a, Image.open(io.BytesIO(data)) as b:
        assert np.array_equal(np.asarray(a), np.asarray(b))
    buf = io.BytesIO()
    build.photo_like(16, 16).convert("CMYK").save(buf, format="JPEG")
    assert "adobe" in {J._kind(s) for s in split_jpeg(strip_jpeg(buf.getvalue()))[0]}


# --------------------------------------------------------------------------- check_jpeg

def test_check_jpeg_keeps_a_passthrough_orientation_and_flags_an_encode():
    data = build.jpeg_bytes(build.photo_like(32, 24), orientation=6)
    assert check_jpeg(data, passthrough=True) == []
    assert any("Orientation [6]" in p for p in check_jpeg(data, passthrough=False))
    assert check_jpeg(b"\xff\xd8\xff\xe0garbage", passthrough=True)[0].startswith("unreadable JPEG structure")


def test_check_jpeg_flags_exif_dimensions_that_disagree():
    data = rich_jpeg()                                     # its EXIF says 99 x 77
    settled_wrong = settle_metadata(data, size=(48, 32))
    assert check_jpeg(settled_wrong, passthrough=False) == []
    problems = check_jpeg(J._splice(build.jpeg_bytes(build.photo_like(48, 32)),
                                    [seg(0xE1, J.EXIF_ID + exif_tiff(orientation=1))]), passthrough=False)
    assert any("0xA002" in p for p in problems)


def test_jpeg_files_on_disk(tmp_path: Path):
    path = tmp_path / "x.jpg"
    path.write_bytes(rich_jpeg())
    assert jpeg_info(path.read_bytes()).orientation == 6


# --------------------------------------------------------------------------- review fixes

def test_extraneous_bytes_between_segments_are_skipped_as_libjpeg_does(tmp_path: Path):
    from pdfintake.intake.assemble import assemble
    from pdfintake.intake.images import route_for

    data = build.jpeg_bytes(build.photo_like(48, 32))
    app0_end = 2 + 2 + struct.unpack_from(">H", data, 4)[0]
    padded = data[:app0_end] + b"\x00\x13" + data[app0_end:]          # two stray bytes before DQT
    info = jpeg_info(padded)
    assert (info.width, info.height) == (48, 32) and passthrough_ok(info)
    assert check_jpeg(padded, passthrough=True) == []
    src = tmp_path / "padded.jpg"
    src.write_bytes(padded)
    assert route_for("jpeg", src) == "passthrough"
    pages = assemble([src], tmp_path / "padded.pdf")
    assert pages[0].route == "passthrough" and pages[0].embedded_sha256 == pages[0].source_sha256
    with pytest.raises(ValueError):
        jpeg_info(data[:app0_end] + b"\x00" * 40)                      # junk to the end is still cut off


def _icc_segments(profile: bytes, parts: int, *, seqs: list[int] | None = None, count: int | None = None) -> list[bytes]:
    size = -(-len(profile) // parts)
    chunks = [profile[i:i + size] for i in range(0, len(profile), size)]
    seqs = seqs or list(range(1, len(chunks) + 1))
    return [seg(0xE2, J.ICC_ID + bytes((s, count or len(chunks))) + c) for s, c in zip(seqs, chunks)]


def _with_segments(data: bytes, extra: list[bytes]) -> bytes:
    return data[:2] + b"".join(extra) + data[2:]


def test_an_icc_profile_is_used_only_when_its_chunks_make_one_whole_profile():
    base = build.jpeg_bytes(build.photo_like(16, 16))
    profile = fake_icc(b"RGB ", 600)
    assert jpeg_info(_with_segments(base, _icc_segments(profile, 2))).icc == profile
    missing = _icc_segments(profile, 2)[:1]
    duplicate = _icc_segments(profile, 2, seqs=[1, 1])
    wrong_count = _icc_segments(profile, 2, count=3)
    short = _icc_segments(profile[:300], 1)                          # header says 600 bytes
    for segs in (missing, duplicate, wrong_count, short):
        assert jpeg_info(_with_segments(base, segs)).icc == b""
    assert jpeg_info(base).icc is None


def test_a_broken_icc_profile_falls_back_to_the_device_space_with_a_warning(tmp_path: Path):
    from pdfintake.intake.assemble import assemble

    base = build.jpeg_bytes(build.photo_like(16, 16))
    src = tmp_path / "half.jpg"
    src.write_bytes(_with_segments(base, _icc_segments(fake_icc(b"RGB ", 600), 2)[:1]))
    pages = assemble([src], tmp_path / "half.pdf")
    assert pages[0].route == "passthrough" and pages[0].warnings == ["metadata_format_limit"]
    import pikepdf

    with pikepdf.open(tmp_path / "half.pdf") as pdf:
        assert pdf.pages[0].obj.Resources.XObject["/Im0"].ColorSpace == pikepdf.Name.DeviceRGB


def test_settle_metadata_patches_tags_in_either_ifd():
    def entries(items):
        return b"".join(struct.pack("<HHIHH", t, 3, 1, v, 0) for t, v in sorted(items))

    ifd0 = [(0x0112, 6), (0xA002, 999)]
    sub = [(0x0112, 8), (0x0100, 5), (0xA003, 7)]
    n0 = len(ifd0) + 1
    sub_at = 8 + 2 + 12 * n0 + 4
    tiff = (b"II*\x00" + struct.pack("<IH", 8, n0) + entries(ifd0)
            + struct.pack("<HHII", 0x8769, 4, 1, sub_at) + struct.pack("<I", 0)
            + struct.pack("<H", len(sub)) + entries(sub) + struct.pack("<I", 0))
    jpeg = build.jpeg_bytes(build.photo_like(40, 24))
    with_exif = _with_segments(jpeg, [seg(0xE1, J.EXIF_ID + tiff)])
    assert check_jpeg(with_exif, passthrough=False) != []
    settled = settle_metadata(with_exif, size=(40, 24))
    assert check_jpeg(settled, passthrough=False) == []
    exif = next(s for s in split_jpeg(settled)[0] if J._kind(s) == "exif")
    vals = J._tiff_values(exif[10:])
    assert vals[0x0112] == 1 and vals[0xA002] == 40 and vals[0xA003] == 24 and vals[0x0100] == 40


@pytest.mark.parametrize("prefix", [b"", J.EXIF_ID, J.EXIF_ID + J.EXIF_ID])
def test_exif_orientation_reads_a_block_with_or_without_its_prefix(prefix: bytes):
    ex = Image.Exif()
    ex[0x0112] = 6
    tiff = ex.tobytes()
    tiff = tiff[6:] if tiff.startswith(J.EXIF_ID) else tiff
    assert J.exif_orientation(prefix + tiff) == 6
    assert J.exif_orientation(None) == J.exif_orientation(b"") == 1


def test_exif_orientation_ignores_types_other_than_short_and_long():
    slong = b"II*\x00" + struct.pack("<IH", 8, 1) + struct.pack("<HHIi", 0x0112, 9, 1, 6) + struct.pack("<I", 0)
    assert J.exif_orientation(slong) == 1
