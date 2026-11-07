"""JPEG structure in pure Python: segments, the frame header, metadata, and the checks.

A JPEG is SOI, header segments up to and including SOS, entropy-coded scan data through EOI,
and sometimes bytes after EOI (an MPF gain map or depth image, or junk). Here a *segment* is
always the whole marker segment as bytes (``FF xx``, the two-byte length, the payload), and the
*scan* is everything from the end of the SOS header through EOI.

A passthrough JPEG is never touched: its bytes, trailing data and camera Orientation tag
included, go into the PDF as they are, and the page matrix does the turn (assemble.py).
``settle_metadata`` is only for a JPEG pdfintake encoded itself; ``strip_jpeg`` only for
``export clean``.
"""

from __future__ import annotations

import io
import re
import struct
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}   # DHT, JPG and DAC aren't frames
DCT_SOF = frozenset({0xC0, 0xC1, 0xC2})       # baseline, extended and progressive Huffman: what DCTDecode reads
LOSSLESS_SOF = frozenset({0xC3, 0xC7, 0xCB, 0xCF})
HIERARCHICAL_SOF = frozenset({0xC5, 0xC6, 0xC7, 0xCD, 0xCE, 0xCF})
EXIF_ID = b"Exif\x00\x00"
XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"
XMP_EXT_ID = b"http://ns.adobe.com/xmp/extension/\x00"
ICC_ID = b"ICC_PROFILE\x00"
MAX_PAYLOAD = 0xFFFF - 2

PICTURE_KINDS = frozenset({"core", "jfif", "icc", "adobe"})           # what strip_jpeg keeps
METADATA_KINDS = frozenset({"exif", "xmp", "xmpext", "iptc", "com", "app"})   # what an encode carries
_STANDALONE = frozenset({0x01}) | frozenset(range(0xD0, 0xD8))       # TEM, RSTn: no length field


@dataclass(frozen=True)
class JpegInfo:
    width: int
    height: int
    sof: int                     # the frame marker, 0xC0..0xCF
    precision: int               # bits per sample
    components: int
    adobe_transform: int | None  # APP14 Adobe transform flag, None without that segment
    icc: bytes | None            # ICC_PROFILE chunks joined in sequence order; b"" when chunks are
                                 # present but don't form one whole profile (missing, repeated, cut)
    orientation: int             # EXIF Orientation, 1 when absent or out of range
    has_mpf: bool                # an APP2 MPF index (gain map, depth or other secondary images)
    trailing_bytes: int          # bytes after the first image's EOI


# --------------------------------------------------------------------------- structure

def _parse_header(data: bytes) -> tuple[list[tuple[int, int, int]], int]:
    """[(marker, start, end)] for every header segment through SOS, and where the scan begins.

    Extraneous bytes between two segments are skipped up to the next 0xFF, as libjpeg, Pillow
    and pdfium do (libjpeg warns "extraneous bytes before marker" and reads on), so such a file
    stays a JPEG DCTDecode carries rather than one refused as undecodable.
    """
    if data[:2] != SOI:
        raise ValueError("not a JPEG (no SOI)")
    i, n, segs = 2, len(data), []
    while True:
        if i < n and data[i] != 0xFF:
            j = data.find(b"\xff", i)
            if j < 0:
                raise ValueError(f"JPEG header cut off at byte {i}")
            i = j
        if i + 2 > n:
            raise ValueError(f"malformed JPEG header at byte {i}")
        m = data[i + 1]
        if m == 0xFF:                                    # fill byte before a marker
            i += 1
            continue
        if m in _STANDALONE:
            i += 2
            continue
        if m in (0xD8, 0xD9):
            raise ValueError(f"marker FF{m:02X} inside the JPEG header")
        if i + 4 > n:
            raise ValueError(f"JPEG header cut off at byte {i}")
        length = struct.unpack_from(">H", data, i + 2)[0]
        end = i + 2 + length
        if length < 2 or end > n:
            raise ValueError(f"JPEG segment FF{m:02X} at byte {i} has a bad length")
        segs.append((m, i, end))
        i = end
        if m == 0xDA:
            return segs, i


def _scan_end(data: bytes, i: int) -> int:
    """Offset just past the EOI of the image whose scan data starts at ``i``."""
    n = len(data)
    while i < n - 1:
        j = data.find(b"\xff", i)
        if j < 0 or j + 1 >= n:
            break
        m = data[j + 1]
        if m == 0x00 or 0xD0 <= m <= 0xD7:               # stuffed byte, restart marker
            i = j + 2
        elif m == 0xFF:                                  # fill byte before a marker
            i = j + 1
        elif m == 0xD9:
            return j + 2
        else:                                            # DHT, SOS, DQT, DRI between progressive scans
            if j + 4 > n:
                break
            i = j + 2 + struct.unpack_from(">H", data, j + 2)[0]
    raise ValueError("JPEG has no EOI")


def jpeg_end(data: bytes) -> int:
    """Offset just past the first image's EOI. What follows (an MPF gain map, junk) isn't that image."""
    _, scan = _parse_header(data)
    return _scan_end(data, scan)


def split_jpeg(data: bytes) -> tuple[list[bytes], bytes]:
    """(every header segment through SOS, each as whole segment bytes; the scan data through EOI).

    Bytes after EOI are not returned, so ``join_jpeg(*split_jpeg(d)) == d[:jpeg_end(d)]`` for a
    JPEG without fill bytes, standalone markers or extraneous bytes in its header.
    """
    segs, scan = _parse_header(data)
    return [data[s:e] for _, s, e in segs], data[scan:_scan_end(data, scan)]


def join_jpeg(segments: Sequence[bytes], scan: bytes) -> bytes:
    return SOI + b"".join(segments) + scan


def _segment(marker: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"a JPEG segment holds at most {MAX_PAYLOAD} bytes, not {len(payload)}")
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def _payload(segment: bytes) -> bytes:
    return segment[4:]


def _kind(segment: bytes) -> str:
    """What a header segment is: ``core`` (tables, frame, scan) or a named APPn/COM."""
    m, p = segment[1], segment[4:]
    if m == 0xE0:
        return "jfif" if p[:5] == b"JFIF\x00" else "jfxx" if p[:5] == b"JFXX\x00" else "app"
    if m == 0xE1:
        return ("exif" if p[:6] == EXIF_ID else "xmp" if p.startswith(XMP_ID)
                else "xmpext" if p.startswith(XMP_EXT_ID) else "app")
    if m == 0xE2:
        return "icc" if p[:12] == ICC_ID else "mpf" if p[:4] == b"MPF\x00" else "app"
    if m == 0xED:
        return "iptc"
    if m == 0xEE:
        return "adobe" if p[:5] == b"Adobe" else "app"
    if m == 0xFE:
        return "com"
    return "app" if 0xE0 <= m <= 0xEF else "core"


def _frame(data: bytes) -> tuple[int, int, int, int, int]:
    """(sof, precision, width, height, components) from the header alone (no scan walk)."""
    segs, _ = _parse_header(data)
    for m, s, e in segs:
        if m in SOF_MARKERS:
            p = data[s + 4:e]
            if len(p) < 6:
                raise ValueError("JPEG frame header too short")
            precision, height, width, comps = struct.unpack_from(">BHHB", p, 0)
            return m, precision, width, height, comps
    raise ValueError("JPEG has no frame header")


def jpeg_info(data: bytes) -> JpegInfo:
    """The frame, colour and metadata facts of a JPEG. ValueError for a malformed one (no EOI included)."""
    segs, scan = _parse_header(data)
    end = _scan_end(data, scan)
    sof, precision, width, height, comps = _frame(data)
    adobe: int | None = None
    icc_chunks: list[tuple[int, int, bytes]] = []
    orientation: int | None = None
    has_mpf = False
    for m, s, e in segs:
        seg = data[s:e]
        kind, p = _kind(seg), seg[4:]
        if kind == "adobe" and adobe is None and len(p) >= 12:
            adobe = p[11]
        elif kind == "icc" and len(p) >= 14:
            icc_chunks.append((p[12], p[13], p[14:]))
        elif kind == "exif" and orientation is None:
            orientation = exif_orientation(p)
        elif kind == "mpf":
            has_mpf = True
    icc = _join_icc(icc_chunks) if icc_chunks else None
    return JpegInfo(width=width, height=height, sof=sof, precision=precision, components=comps,
                    adobe_transform=adobe, icc=icc, orientation=orientation or 1, has_mpf=has_mpf,
                    trailing_bytes=len(data) - end)


def _join_icc(chunks: list[tuple[int, int, bytes]]) -> bytes:
    """The profile the ICC_PROFILE chunks carry, joined by sequence number; b"" unless the
    sequence numbers are exactly 1..N, every chunk declares N, and the joined bytes are as long
    as the profile header says (a profile that is cut, padded or repeated isn't written)."""
    total = len(chunks)
    if sorted(seq for seq, _, _ in chunks) != list(range(1, total + 1)) or any(n != total for _, n, _ in chunks):
        return b""
    icc = b"".join(c for _, _, c in sorted(chunks, key=lambda c: c[0]))
    if len(icc) < 128 or struct.unpack_from(">I", icc, 0)[0] != len(icc):
        return b""
    return icc


def passthrough_ok(info: JpegInfo) -> bool:
    """True when a PDF's DCTDecode carries the JPEG as it is: SOF C0/C1/C2, 8-bit, 1/3/4 components."""
    return (info.sof in DCT_SOF and info.precision == 8 and info.components in (1, 3, 4)
            and info.width > 0 and info.height > 0)


# --------------------------------------------------------------------------- metadata

def strip_jpeg(data: bytes, *, keep_icc: bool = True) -> bytes:
    """The same entropy-coded bytes with only the picture's own segments (tables, frame, JFIF,
    Adobe, and the ICC profile unless ``keep_icc`` is false) and nothing after EOI.
    Used only by ``export clean``."""
    keep = PICTURE_KINDS if keep_icc else PICTURE_KINDS - {"icc"}
    segs, scan = split_jpeg(data)
    return join_jpeg([s for s in segs if _kind(s) in keep], scan)


def carried_segments(data: bytes) -> list[bytes]:
    """The metadata segments an encode carries over from a JPEG source: EXIF, XMP (and its
    extension), IPTC, comments and other APPn. Never JFIF, JFXX thumbnails, ICC (an encode writes
    the profile itself), Adobe or an MPF index."""
    segs, _ = split_jpeg(data)
    return [s for s in segs if _kind(s) in METADATA_KINDS]


def _splice(data: bytes, extra: Sequence[bytes]) -> bytes:
    """``extra`` segments inserted right after JFIF (or after SOI when there is none)."""
    if not extra:
        return data
    segs, _ = _parse_header(data)
    at = 2
    for m, s, e in segs:
        if _kind(data[s:e]) == "jfif":
            at = e
            break
    return data[:at] + b"".join(extra) + data[at:]


def settle_metadata(data: bytes, *, size: tuple[int, int]) -> bytes:
    """Make an encoded JPEG's metadata true to its pixels, after a real encode only.

    EXIF and XMP say Orientation 1 and ``size`` (width, height); the EXIF thumbnail (IFD1) is
    unlinked; a JFXX thumbnail and an MPF index (whose secondary images aren't there) are dropped.
    An EXIF block that can't be patched safely, or a segment grown past 64 KB, is dropped rather
    than left saying the picture is sideways. Bytes after EOI are not kept. ValueError when
    ``size`` isn't the frame's size.
    """
    segs, scan = split_jpeg(data)
    _, _, fw, fh, _ = _frame(data)
    w, h = size
    if (fw, fh) != (w, h):
        raise ValueError(f"the frame is {fw}x{fh}, not {w}x{h}")
    out: list[bytes] = []
    for seg in segs:
        kind = _kind(seg)
        if kind in ("mpf", "jfxx"):
            continue
        if kind == "exif":
            tiff = bytearray(_payload(seg)[6:])
            try:
                _patch_tiff(tiff, w, h)
            except (struct.error, ValueError, IndexError):
                continue
            seg = _segment(0xE1, EXIF_ID + bytes(tiff))
        elif kind == "xmp":
            try:
                seg = _segment(0xE1, _patch_xmp(_payload(seg), w, h))
            except ValueError:
                continue
        out.append(seg)
    return join_jpeg(out, scan)


def exif_orientation(exif: bytes | None) -> int:
    """The one orientation rule every route uses: IFD0 Orientation (0x0112) of an EXIF block,
    typed SHORT or LONG, 1..8; 1 when absent, unreadable, out of range or of another type.

    ``exif`` is a TIFF block with or without the ``Exif\\0\\0`` prefix (a JPEG APP1 payload, a PNG
    eXIf chunk, Pillow's ``info["exif"]``). XMP ``tiff:Orientation`` is never read: browsers and
    image viewers turn by EXIF alone, so a page turned by XMP would differ from what the user sees.
    """
    if not exif:
        return 1
    exif = bytes(exif)
    for _ in range(2):                                   # Pillow prefixes a PNG eXIf that may carry one already
        if exif.startswith(EXIF_ID):
            exif = exif[6:]
    return _tiff_orientation(exif)


def _tiff_orientation(tiff: bytes) -> int:
    """IFD0 Orientation from an EXIF TIFF block; 1 when absent, unreadable or out of range."""
    try:
        bo = {b"II": "<", b"MM": ">"}[bytes(tiff[:2])]
        ifd = struct.unpack_from(bo + "I", tiff, 4)[0]
        n = struct.unpack_from(bo + "H", tiff, ifd)[0]
        for k in range(n):
            e = ifd + 2 + 12 * k
            tag, typ, cnt = struct.unpack_from(bo + "HHI", tiff, e)
            if tag == 0x0112 and cnt >= 1:
                if typ == 3:
                    v = struct.unpack_from(bo + "H", tiff, e + 8)[0]
                elif typ == 4:
                    v = struct.unpack_from(bo + "I", tiff, e + 8)[0]
                else:
                    return 1
                return v if 1 <= v <= 8 else 1
    except (KeyError, struct.error):
        return 1
    return 1


def _tiff_values(tiff: bytes) -> dict[int, int]:
    """Single SHORT/LONG values of IFD0 and the Exif IFD, by tag (for the checks)."""
    out: dict[int, int] = {}
    try:
        bo = {b"II": "<", b"MM": ">"}[bytes(tiff[:2])]

        def read(off: int) -> int | None:
            n = struct.unpack_from(bo + "H", tiff, off)[0]
            sub = None
            for k in range(n):
                e = off + 2 + 12 * k
                tag, typ, cnt = struct.unpack_from(bo + "HHI", tiff, e)
                if tag == 0x8769 and cnt == 1:
                    sub = struct.unpack_from(bo + "I", tiff, e + 8)[0]
                elif cnt == 1 and typ in (3, 4):
                    out[tag] = struct.unpack_from(bo + ("H" if typ == 3 else "I"), tiff, e + 8)[0]
            return sub

        sub = read(struct.unpack_from(bo + "I", tiff, 4)[0])
        if sub:
            read(sub)
    except (KeyError, struct.error):
        pass
    return out


def _patch_tiff(t: bytearray, w: int, h: int) -> None:
    """Patch an EXIF TIFF block in place, same length so every offset stays valid: Orientation 1,
    pixel dimensions w x h, the IFD1 thumbnail unlinked. Raises on a malformed block."""
    if bytes(t[:2]) not in (b"II", b"MM"):
        raise ValueError("not a TIFF header")
    bo = "<" if t[:2] == b"II" else ">"

    def u16(o: int) -> int:
        return struct.unpack_from(bo + "H", t, o)[0]

    def u32(o: int) -> int:
        return struct.unpack_from(bo + "I", t, o)[0]

    def patch(off: int, want: dict[int, int], cut_next: bool) -> int | None:
        n, sub = u16(off), None
        for k in range(n):
            e = off + 2 + 12 * k
            tag, typ, cnt = u16(e), u16(e + 2), u32(e + 4)
            if tag in want and cnt == 1 and typ in (3, 4):
                v = want[tag]
                if typ == 3 and v <= 0xFFFF:
                    struct.pack_into(bo + "H", t, e + 8, v)
                elif typ == 4:
                    struct.pack_into(bo + "I", t, e + 8, v)
            elif tag == 0x8769:
                sub = u32(e + 8)
        if cut_next:
            struct.pack_into(bo + "I", t, off + 2 + 12 * n, 0)    # IFD0 -> IFD1 (the thumbnail)
        return sub

    # Both IFDs get every tag ``check_jpeg`` reads (``_tiff_values`` merges them), so a tag a
    # writer put in the "wrong" IFD is patched too and the encode's own check agrees by construction.
    want = {0x0112: 1, 0x0100: w, 0x0101: h, 0xA002: w, 0xA003: h}
    exif_ifd = patch(u32(4), want, cut_next=True)
    if exif_ifd:
        patch(exif_ifd, want, cut_next=False)


_XMP_FIELDS = (b"tiff:Orientation", b"exif:PixelXDimension", b"exif:PixelYDimension",
               b"tiff:ImageWidth", b"tiff:ImageLength")


def _patch_xmp(p: bytes, w: int, h: int) -> bytes:
    """The same edits in an XMP packet: tiff:Orientation 1, pixel dimensions w x h."""
    for tag, v in zip(_XMP_FIELDS, (1, w, h, w, h)):
        val = str(v).encode()
        p = re.sub(rb"(" + tag + rb"\s*=\s*[\"'])\s*\d+\s*([\"'])", rb"\g<1>" + val + rb"\g<2>", p)
        p = re.sub(rb"(<" + tag + rb">)\s*\d+\s*(</" + tag + rb">)", rb"\g<1>" + val + rb"\g<2>", p)
    return p


def _xmp_orientations(p: bytes) -> set[int]:
    return {int(v) for v in re.findall(rb"tiff:Orientation\s*(?:=\s*[\"']|>)\s*(\d+)", p)}


# --------------------------------------------------------------------------- checks

def check_jpeg(data: bytes, *, passthrough: bool) -> list[str]:
    """Why a JPEG can't go into a PDF as it is (an empty list when it can).

    Always: the structure parses through EOI, the frame is one DCTDecode reads, and Pillow reads
    the header at the frame's size. For a JPEG pdfintake encoded (``passthrough=False``) also:
    EXIF and XMP say Orientation 1, EXIF pixel dimensions match the frame, and there is no MPF
    index. A passthrough keeps its camera Orientation tag; the page matrix places it.
    """
    try:
        info = jpeg_info(data)
        segs, _ = split_jpeg(data)
    except ValueError as exc:
        return [f"unreadable JPEG structure: {exc}"]
    problems: list[str] = []
    if not passthrough_ok(info):
        problems.append(f"frame SOF 0x{info.sof:02X}, {info.precision}-bit, {info.components} components "
                        "is not something DCTDecode carries")
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=["JPEG"]) as im:
                if im.size != (info.width, info.height):
                    problems.append(f"Pillow reads {im.size[0]}x{im.size[1]}, the frame says {info.width}x{info.height}")
    except Image.DecompressionBombError:
        pass                                             # the pixel limit is the caller's check (sniff.MAX_IMAGE_PIXELS)
    except Exception as exc:                             # noqa: BLE001 - any reader failure is the finding
        problems.append(f"Pillow can't read the header: {type(exc).__name__}")
    if not passthrough:
        orients: set[int] = set()
        for seg in segs:
            kind = _kind(seg)
            if kind == "exif":
                tiff = _payload(seg)[6:]
                vals = _tiff_values(tiff)
                if 0x0112 in vals:
                    orients.add(vals[0x0112])
                for tag in (0xA002, 0xA003, 0x0100, 0x0101):
                    want = info.width if tag in (0xA002, 0x0100) else info.height
                    if tag in vals and vals[tag] != want:
                        problems.append(f"EXIF tag 0x{tag:04X} says {vals[tag]}, the frame says {want}")
            elif kind == "xmp":
                orients |= _xmp_orientations(_payload(seg))
            elif kind == "mpf":
                problems.append("an MPF index points at images that aren't there")
        if orients - {1}:
            problems.append(f"Orientation {sorted(orients - {1})} left on upright pixels")
    return problems
