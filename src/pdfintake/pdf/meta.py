"""Document metadata: snapshots, diffs and the few edits pdfintake is allowed to make.

A PDF input's metadata is never rewritten by a build. The Info dictionary is copied through
untouched, and the XMP packet keeps every byte except one: when the lossless save raises the
file's version (object streams need PDF 1.5), ``patch_xmp_pdfversion`` raises the
``pdf:PDFVersion`` value in place, with the packet keeping its exact length. It never lowers it.

pikepdf's ``open_metadata()`` is never used on a build: on exit it re-serialises the packet and
syncs Info from XMP, which drops Info keys the packet doesn't carry (Keywords, for one). Every edit
here works on the packet's text, so whatever it doesn't name stays byte for byte as it was.

Edits that do change metadata are narrow and named:

* ``set_built_info``: a PDF built from images gets ``/Producer`` and ``/CreationDate`` only.
* ``mark_derived`` and ``drop_conformance_claims``: searchable copies only (DESIGN §10.2, §11.1).
  The copy's XMP gains a ``pdfintake:DerivedFrom`` block naming the original's file name and
  SHA-256, and loses its PDF/A and PDF/UA identification, which the added layer would break.
  ``mark_derived`` is the one edit that may create a packet, when the original had none.
* ``strip_for_export``: ``export clean`` only, on a copy that leaves the house.

None of these saves a file. Every save of a copy, the decrypt-once work copy included, passes
``encryption=False``, ``deterministic_id=True`` and ``fix_metadata_version=False`` (decided
2026-11-05), as ``shrink`` does. pikepdf's default ``fix_metadata_version=True`` re-serialises the
packet on save (it rewrites ``pdf:PDFVersion``, reorders attributes and drops the padding), which
would break the rule above before any edit here runs.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pikepdf

__all__ = [
    "MetaSnapshot", "MetaDiff", "snapshot", "diff", "set_built_info", "patch_xmp_pdfversion",
    "mark_derived", "derived_from", "drop_conformance_claims", "strip_for_export",
    "PDFINTAKE_NS", "DERIVED_NOTE",
]

PDF_NS = b"http://ns.adobe.com/pdf/1.3/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
PDFINTAKE_NS = "https://github.com/RazorII/pdfintake/ns/1.0/"   # the namespace of the DerivedFrom block
PDFINTAKE_PREFIX = "pdfintake"
DERIVED_NOTE = "searchable copy, not the original"             # DESIGN §10.2, word for word
# PDF/A and PDF/UA identification schemas (ISO 19005, ISO 14289); written with or without the slash.
CONFORMANCE_NS = {
    "http://www.aiim.org/pdfa/ns/id/": "pdfaid", "http://www.aiim.org/pdfa/ns/id": "pdfaid",
    "http://www.aiim.org/pdfua/ns/id/": "pdfuaid", "http://www.aiim.org/pdfua/ns/id": "pdfuaid",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME = r"[A-Za-z_][A-Za-z0-9_.\-]*"
_VERSION_RE = re.compile(r"^[0-9]\.[0-9]{1,2}$")
_DATE_KEYS = ("/CreationDate", "/ModDate")


@dataclass(frozen=True)
class MetaSnapshot:
    info: dict[str, str]          # Info key without its slash -> value as text (strings decoded, others as PDF syntax)
    xmp_sha256: str | None        # of the decoded document XMP stream, None when there is none
    xmp_bytes: int                # its decoded length, 0 when there is none
    pdf_version: str              # the header version, e.g. "1.7"


@dataclass
class MetaDiff:
    kept: list[str]
    changed: list[str]
    dropped: list[str]


# --------------------------------------------------------------------------- reading

def _value_text(value: Any) -> str:
    import pikepdf

    if isinstance(value, pikepdf.String):
        return str(value)
    if isinstance(value, pikepdf.Object):
        try:
            return bytes(value.unparse()).decode("latin-1")
        except Exception:  # noqa: BLE001 - a malformed value still gets a stable text form
            return repr(value)
    return str(value)


def _info_dict(pdf: "pikepdf.Pdf") -> "pikepdf.Dictionary | None":
    import pikepdf

    # pdf.docinfo would create and attach an empty Info when there is none; read the trailer instead.
    info = pdf.trailer.get("/Info")
    return info if isinstance(info, pikepdf.Dictionary) else None


def _xmp_stream(pdf: "pikepdf.Pdf") -> "pikepdf.Stream | None":
    import pikepdf

    meta = pdf.Root.get("/Metadata")
    return meta if isinstance(meta, pikepdf.Stream) else None


def _xmp_data(stream: "pikepdf.Stream") -> bytes:
    try:
        return bytes(stream.read_bytes())
    except Exception:  # noqa: BLE001 - an undecodable packet is still fingerprinted, by its raw bytes
        return bytes(stream.read_raw_bytes())


def _snapshot_open(pdf: "pikepdf.Pdf") -> MetaSnapshot:
    info: dict[str, str] = {}
    raw_info = _info_dict(pdf)
    if raw_info is not None:
        for key in sorted(str(k) for k in raw_info.keys()):
            info[key[1:] if key.startswith("/") else key] = _value_text(raw_info[key])
    stream = _xmp_stream(pdf)
    if stream is None:
        sha, size = None, 0
    else:
        data = _xmp_data(stream)
        sha, size = hashlib.sha256(data).hexdigest(), len(data)
    return MetaSnapshot(info=info, xmp_sha256=sha, xmp_bytes=size, pdf_version=str(pdf.pdf_version))


def snapshot(pdf: "Path | pikepdf.Pdf") -> MetaSnapshot:
    """Info, an XMP fingerprint and the header version. Never changes the PDF it reads."""
    import pikepdf

    if isinstance(pdf, pikepdf.Pdf):
        return _snapshot_open(pdf)
    with pikepdf.open(Path(pdf)) as opened:
        return _snapshot_open(opened)


def diff(before: MetaSnapshot, after: MetaSnapshot) -> MetaDiff:
    """What happened to the metadata between two snapshots.

    ``"Info"`` is reported whole when the Info dictionaries are equal; when they differ, each key
    is reported on its own as ``"Info/<Key>"`` (a key that appears counts as changed). ``"XMP"``
    is always reported whole, as kept, changed or dropped, since only its hash is compared (a
    packet that appears counts as changed). A part absent on both sides isn't listed. The header
    version isn't metadata here: the ShrinkReport records it.
    """
    kept: list[str] = []
    changed: list[str] = []
    dropped: list[str] = []

    if before.info or after.info:
        if before.info == after.info:
            kept.append("Info")
        else:
            for key in sorted(set(before.info) | set(after.info)):
                name = f"Info/{key}"
                if key not in after.info:
                    dropped.append(name)
                elif key not in before.info or before.info[key] != after.info[key]:
                    changed.append(name)
                else:
                    kept.append(name)

    if before.xmp_sha256 is not None or after.xmp_sha256 is not None:
        if after.xmp_sha256 is None:
            dropped.append("XMP")
        elif before.xmp_sha256 == after.xmp_sha256:
            kept.append("XMP")
        else:
            changed.append("XMP")

    return MetaDiff(kept=kept, changed=changed, dropped=dropped)


# --------------------------------------------------------------------------- edits

def _pdf_date(moment: datetime) -> str:
    from pikepdf.models.metadata import encode_pdf_date

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return encode_pdf_date(moment)


def set_built_info(pdf: "pikepdf.Pdf", *, producer: str, created: datetime) -> None:
    """For a PDF built from images only: Info holds exactly ``/Producer`` and ``/CreationDate``.

    A naive ``created`` is taken as UTC. No XMP packet is created.
    """
    import pikepdf

    pdf.trailer.Info = pdf.make_indirect(pikepdf.Dictionary({
        "/CreationDate": pikepdf.String(_pdf_date(created)),
        "/Producer": pikepdf.String(producer),
    }))


def _pdf_prefixes(xmp: bytes) -> list[bytes]:
    """Every namespace prefix bound to the Adobe PDF schema in the packet ("pdf" in practice)."""
    found = re.findall(rb"xmlns:([A-Za-z_][A-Za-z0-9_.\-]*)\s*=\s*([\"'])" + re.escape(PDF_NS) + rb"\2", xmp)
    return sorted({prefix for prefix, _ in found})


def _version_tuple(version: str) -> tuple[int, ...] | None:
    """``"1.7"`` -> ``(1, 7)``; None for anything that isn't a plain PDF version."""
    if not _VERSION_RE.match(version):
        return None
    return tuple(int(p) for p in version.split("."))


def _patched_packet(xmp: bytes, version: str) -> bytes | None:
    if xmp[:2] in (b"\xfe\xff", b"\xff\xfe") or xmp[:4] in (b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00"):
        return None                                   # UTF-16/32 packets aren't patched
    new_value = version.encode("ascii")
    new_tuple = _version_tuple(version)
    out = xmp
    hits = 0
    for prefix in _pdf_prefixes(xmp):
        name = re.escape(prefix + b":PDFVersion")
        attr = re.compile(rb"(\s" + name + rb"\s*=\s*([\"']))([^\"'<>]*)(\2)")
        elem = re.compile(rb"(<" + name + rb"(?:\s[^>]*)?>)\s*([^<\s]*)\s*(</" + name + rb"\s*>)")
        spans: list[tuple[int, int]] = []
        for m in attr.finditer(out):
            spans.append(m.span(3))
        for m in elem.finditer(out):
            spans.append(m.span(2))
        for start, end in sorted(spans, reverse=True):
            old_value = out[start:end]
            old_tuple = _version_tuple(old_value.decode("ascii", "replace"))
            if old_tuple is None or new_tuple is None or old_tuple >= new_tuple:
                continue                              # only ever raised: an equal, higher or odd value stays
            if len(old_value) != len(new_value):
                return None                           # never shift a byte of the packet
            out = out[:start] + new_value + out[end:]
            hits += 1
    return out if hits else None


def patch_xmp_pdfversion(pdf: "pikepdf.Pdf", version: str) -> bool:
    """Raise XMP ``pdf:PDFVersion`` to ``version`` in place, keeping the packet's length.

    Only a value lower than ``version`` is replaced; an equal or higher one, or one that isn't a
    plain version, is left as it is. Returns True when the packet was changed. False, and nothing
    touched, when there is no packet, no lower ``pdf:PDFVersion``, the new value has a different
    length, the packet isn't UTF-8, or the stream uses a filter other than one plain FlateDecode.
    """
    import pikepdf

    if not _VERSION_RE.match(version):
        raise ValueError(f"not a PDF version: {version!r}")
    stream = _xmp_stream(pdf)
    if stream is None:
        return False
    filters = stream.get("/Filter")
    if filters is None:
        flate = False
    else:
        names = [str(f) for f in filters] if isinstance(filters, pikepdf.Array) else [str(filters)]
        if names != ["/FlateDecode"] or stream.get("/DecodeParms") is not None:
            return False
        flate = True
    try:
        data = bytes(stream.read_bytes())
    except Exception:  # noqa: BLE001 - an undecodable packet is left exactly as it is
        return False
    patched = _patched_packet(data, version)
    if patched is None or len(patched) != len(data):
        return False
    if flate:
        stream.write(zlib.compress(patched, 9), filter=pikepdf.Name.FlateDecode)
    else:
        stream.write(patched)
    return True


# --------------------------------------------------------------------------- packet text helpers

def _decode_packet(data: bytes) -> tuple[str, str] | None:
    """(text, codec) for a UTF-8 or BOM-marked UTF-16 packet, None for anything else.

    A UTF-8 byte-order mark stays in the text (as U+FEFF), so encoding the text back with the
    same codec gives the same bytes for every character an edit didn't touch.
    """
    if data[:4] in (b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00"):
        return None                                   # UTF-32 packets aren't edited
    for bom, codec in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")):
        if data.startswith(bom):
            try:
                return "﻿" + data[2:].decode(codec), codec
            except UnicodeDecodeError:
                return None
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return None


def _encode_packet(text: str, codec: str) -> bytes:
    if codec == "utf-8":
        return text.encode("utf-8")
    body = text[1:] if text.startswith("﻿") else text
    return (b"\xff\xfe" if codec == "utf-16-le" else b"\xfe\xff") + body.encode(codec)


def _parse(text: str) -> "ET.Element | None":
    """The packet's XML tree, or None when it won't parse. A DOCTYPE or ENTITY never parses here:
    XMP never needs one, and refusing it means no entity is ever expanded (as ``refusals`` does)."""
    probe = text.upper()
    if "<!DOCTYPE" in probe or "<!ENTITY" in probe:
        return None
    body = re.sub(r"^﻿?\s*<\?xml[^>]*\?>", "", text.lstrip("﻿ \t\r\n\x00"))
    try:
        return ET.fromstring(body)
    except (ET.ParseError, ValueError):
        return None


def _write_packet(stream: "pikepdf.Stream", data: bytes) -> None:
    """Replace a packet's bytes: a plain FlateDecode packet stays Flate, anything else is written unfiltered."""
    import pikepdf

    names = _filters_of(stream)
    if names == ["/FlateDecode"] and stream.get("/DecodeParms") is None:
        stream.write(zlib.compress(data, 9), filter=pikepdf.Name.FlateDecode)
    else:
        stream.write(data)                            # also removes /Filter and /DecodeParms (measured)


def _filters_of(stream: "pikepdf.Stream") -> list[str]:
    import pikepdf

    filters = stream.get("/Filter")
    if filters is None:
        return []
    return [str(f) for f in filters] if isinstance(filters, pikepdf.Array) else [str(filters)]


def _xml_text(value: str) -> str:
    """``value`` as XML character data: escaped, line ends kept as references, and characters
    XML 1.0 can't carry (controls, lone surrogates, U+FFFE/U+FFFF) replaced with U+FFFD."""
    out: list[str] = []
    for ch in value:
        o = ord(ch)
        if ch == "&":
            out.append("&amp;")
        elif ch == "<":
            out.append("&lt;")
        elif ch == ">":
            out.append("&gt;")
        elif ch == '"':
            out.append("&quot;")
        elif ch in "\t\n\r":
            out.append(f"&#{o};")
        elif o < 0x20 or 0xD800 <= o <= 0xDFFF or o in (0xFFFE, 0xFFFF):
            out.append("�")
        else:
            out.append(ch)
    return "".join(out)


def _prefixes_bound(text: str, uris: Callable[[str], bool]) -> set[str]:
    return {m.group(1) for m in re.finditer(r"xmlns:(" + _NAME + r")\s*=\s*([\"'])(.*?)\2", text, re.S)
            if uris(m.group(3))}


# --------------------------------------------------------------------------- the DerivedFrom block

_OWN_BLOCK = re.compile(
    # The about value is matched by its own delimiter: a reused value may hold the other quote.
    r"<rdf:Description xmlns:rdf=\"" + re.escape(RDF_NS) + r"\" rdf:about=(?:\"[^\"]*\"|'[^']*') xmlns:"
    + PDFINTAKE_PREFIX + r"=\"" + re.escape(PDFINTAKE_NS) + r"\">.*?</rdf:Description>", re.S)


def _derived_block(about: str, quote: str, name: str, sha256: str) -> str:
    # rdf is declared on the block itself, so it reads the same whatever prefix the packet binds.
    p = PDFINTAKE_PREFIX
    return (f'<rdf:Description xmlns:rdf="{RDF_NS}" rdf:about={quote}{about}{quote} xmlns:{p}="{PDFINTAKE_NS}">'
            f'<{p}:DerivedFrom rdf:parseType="Resource">'
            f"<{p}:OriginalName>{_xml_text(name)}</{p}:OriginalName>"
            f"<{p}:OriginalSHA256>{sha256}</{p}:OriginalSHA256>"
            f"<{p}:Note>{DERIVED_NOTE}</{p}:Note>"
            f"</{p}:DerivedFrom></rdf:Description>")


def _new_packet(block: str) -> str:
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="{RDF_NS}">{block}</rdf:RDF></x:xmpmeta>\n'
            '<?xpacket end="w"?>')


def _derived_in(tree: "ET.Element") -> dict[str, str] | None:
    el = tree.find(f".//{{{PDFINTAKE_NS}}}DerivedFrom")
    if el is None:
        return None
    fields = {"name": "OriginalName", "sha256": "OriginalSHA256", "note": "Note"}
    return {k: (el.findtext(f"{{{PDFINTAKE_NS}}}{local}") or "") for k, local in fields.items()}


def derived_from(pdf: "pikepdf.Pdf") -> dict[str, str] | None:
    """The document XMP's ``pdfintake:DerivedFrom`` block as ``{"name", "sha256", "note"}``, or None.

    Read by namespace, so XMP's own ``xmpMM:DerivedFrom`` (which editing apps write) never counts.
    A packet that won't decode or parse carries no block we could read.
    """
    stream = _xmp_stream(pdf)
    if stream is None:
        return None
    decoded = _decode_packet(_xmp_data(stream))
    tree = _parse(decoded[0]) if decoded else None
    return _derived_in(tree) if tree is not None else None


def mark_derived(pdf: "pikepdf.Pdf", original_name: str, original_sha256: str) -> None:
    """Write the ``pdfintake:DerivedFrom`` block into the document XMP (searchable copies only).

    The block names the original's file name, its SHA-256 and the words "searchable copy, not the
    original" (DESIGN §10.2). It goes in as its own ``rdf:Description`` before the packet's closing
    ``</rdf:RDF>``, so every other byte of the packet stays; a block this function wrote before is
    replaced. With no packet, or an empty one, a minimal packet holding only the block is created.
    Info is never touched. ``original_name`` is a file name, not a path; ``original_sha256`` is 64
    lowercase hex digits. ValueError, with nothing written, for bad arguments, a packet that isn't
    UTF-8 or UTF-16, has no ``rdf:RDF`` element, or doesn't parse with the block in it.
    """
    import pikepdf

    if not isinstance(original_name, str) or not original_name or "/" in original_name or "\x00" in original_name:
        raise ValueError("original_name must be a file name, not a path")
    if not isinstance(original_sha256, str) or not _SHA256_RE.match(original_sha256):
        raise ValueError("original_sha256 must be 64 lowercase hex digits")

    stream = _xmp_stream(pdf)
    data = _xmp_data(stream) if stream is not None else b""
    if not data.strip(b" \t\r\n\x00"):
        text, codec = _new_packet(_derived_block("", '"', original_name, original_sha256)), "utf-8"
    else:
        decoded = _decode_packet(data)
        if decoded is None:
            raise ValueError("the XMP packet isn't UTF-8 or UTF-16, so the DerivedFrom block can't be added")
        text, codec = decoded
        about = re.search(r"<(?:" + _NAME + r":)?Description\b[^>]*?\s(?:" + _NAME + r":)?about\s*=\s*([\"'])(.*?)\1",
                          text, re.S)
        quote, about_value = (about.group(1), about.group(2)) if about else ('"', "")
        block = _derived_block(about_value, quote, original_name, original_sha256)
        if _OWN_BLOCK.search(text):
            text = _OWN_BLOCK.sub(lambda _m: block, text, count=1)
        else:
            rdf_prefixes = _prefixes_bound(text, lambda uri: uri == RDF_NS)
            default_rdf = re.search(r"\sxmlns\s*=\s*([\"'])" + re.escape(RDF_NS) + r"\1", text) is not None
            closing = [m for m in re.finditer(r"</(?:(" + _NAME + r"):)?RDF\s*>", text)
                       if (m.group(1) in rdf_prefixes if m.group(1) else default_rdf)]
            if not closing:
                raise ValueError("the XMP packet has no rdf:RDF element to add the DerivedFrom block to")
            at = closing[-1].start()
            text = text[:at] + block + text[at:]

    tree = _parse(text)
    found = _derived_in(tree) if tree is not None else None
    if found is None or found["sha256"] != original_sha256 or found["note"] != DERIVED_NOTE \
            or len(tree.findall(f".//{{{PDFINTAKE_NS}}}DerivedFrom")) != 1:
        raise ValueError("the XMP packet doesn't parse with the DerivedFrom block in it")

    new = _encode_packet(text, codec)
    if stream is None:
        packet = pikepdf.Stream(pdf, new)
        packet.Type, packet.Subtype = pikepdf.Name.Metadata, pikepdf.Name.XML
        pdf.Root.Metadata = pdf.make_indirect(packet)
    else:
        _write_packet(stream, new)
        stream.Type, stream.Subtype = pikepdf.Name.Metadata, pikepdf.Name.XML


# --------------------------------------------------------------------------- conformance claims

def _conformance_left(tree: "ET.Element") -> bool:
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        names = [el.tag, *el.attrib]
        if any(n.startswith("{") and n[1:].partition("}")[0] in CONFORMANCE_NS for n in names):
            return True
    return False


def _drop_claims_text(text: str) -> tuple[str, list[str]]:
    """``text`` without any ``pdfaid:*`` or ``pdfuaid:*`` property (element or attribute form) or
    the namespace declarations that bound them. Returns the new text and the names removed, under
    their canonical prefixes."""
    removed: list[str] = []
    for prefix in sorted(_prefixes_bound(text, lambda uri: uri in CONFORMANCE_NS)):
        canon = _canonical(text, prefix)
        p = re.escape(prefix)
        element = re.compile(r"<(" + p + r"):(" + _NAME + r")(?:\s[^>]*)?(?:/>|>.*?</\1:\2\s*>)", re.S)
        attribute = re.compile(r"\s+" + p + r":(" + _NAME + r")\s*=\s*([\"']).*?\2", re.S)
        removed += [f"{canon}:{m.group(2)}" for m in element.finditer(text)]
        text = element.sub("", text)
        removed += [f"{canon}:{m.group(1)}" for m in attribute.finditer(text)]
        text = attribute.sub("", text)
        if not re.search(r"(?:<|</|\s)" + p + r":", text):
            text = re.sub(r"\s+xmlns:" + p + r"\s*=\s*([\"'])[^\"']*\1", "", text)
    return text, removed


def _canonical(text: str, prefix: str) -> str:
    """The usual prefix (``pdfaid`` or ``pdfuaid``) for the schema ``prefix`` is bound to."""
    m = re.search(r"xmlns:" + re.escape(prefix) + r"\s*=\s*([\"'])(.*?)\1", text, re.S)
    return CONFORMANCE_NS.get(m.group(2), prefix) if m else prefix


def _packets(pdf: "pikepdf.Pdf") -> Iterator["pikepdf.Stream"]:
    """Every XMP stream reachable from the trailer, each once: the catalog's first."""
    import pikepdf

    seen: set[tuple[int, int]] = set()
    found: list[pikepdf.Stream] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, pikepdf.Stream) and (obj.get("/Type") == pikepdf.Name.Metadata
                                                or obj.get("/Subtype") == pikepdf.Name.XML):
            if obj.objgen not in seen:
                seen.add(obj.objgen)
                found.append(obj)

    catalog = _xmp_stream(pdf)
    if catalog is not None:
        visit(catalog)
    _walk(pdf.trailer, visit)
    yield from found


def drop_conformance_claims(pdf: "pikepdf.Pdf") -> list[str]:
    """Remove PDF/A and PDF/UA identification (``pdfaid:*``, ``pdfuaid:*``) from every XMP packet
    (searchable copies only: the added layer breaks the claim, DESIGN §10.1).

    The document's packet and any page's or image's own are edited as text; everything else in
    them stays byte for byte. Returns the properties removed, such as ``["pdfaid:conformance",
    "pdfaid:part"]``, sorted and without repeats; empty when nothing claimed conformance (then no
    packet is rewritten). A packet that won't decode or parse is left alone, except the document's
    own, which raises ValueError when it still makes a claim, so a copy never keeps a claim it breaks.
    """
    removed: set[str] = set()
    catalog = _xmp_stream(pdf)
    for stream in _packets(pdf):
        is_catalog = catalog is not None and stream.objgen == catalog.objgen
        decoded = _decode_packet(_xmp_data(stream))
        tree = _parse(decoded[0]) if decoded else None
        if decoded is None or tree is None:
            if is_catalog and re.search(rb"pdf(?:ua|a)id", _xmp_data(stream)):
                raise ValueError("the document's XMP can't be read, so its conformance claim can't be removed")
            continue
        if not _conformance_left(tree):
            continue
        text, names = _drop_claims_text(decoded[0])
        after = _parse(text)
        if after is None or _conformance_left(after):
            if is_catalog:
                raise ValueError("the document's conformance claim couldn't be removed from its XMP")
            continue
        _write_packet(stream, _encode_packet(text, decoded[1]))
        removed.update(names)
    return sorted(removed)


# --------------------------------------------------------------------------- export clean

def _walk(root: Any, visit: Callable[[Any], None]) -> None:
    """Visit every dictionary and stream reachable from ``root``, direct ones included, each
    indirect object once. ``visit`` runs before an object's values are followed, so a key it
    deletes isn't walked into."""
    import pikepdf

    seen: set[tuple[int, int]] = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if not isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream, pikepdf.Array)):
            continue
        if obj.is_indirect:
            if obj.objgen in seen:
                continue
            seen.add(obj.objgen)
        if isinstance(obj, pikepdf.Array):
            stack.extend(list(obj))
            continue
        visit(obj)
        for key in list(obj.keys()):
            try:
                stack.append(obj[key])
            except Exception:  # noqa: BLE001 - an unreadable value holds nothing we could remove
                continue


def _drop_key(obj: Any, key: str) -> bool:
    if key in obj:
        del obj[key]
        return True
    return False


def strip_for_export(pdf: "pikepdf.Pdf", *, keep_dates: bool) -> list[str]:
    """``export clean`` only: remove Info except Title (and the dates when ``keep_dates``), every
    XMP packet, ``/PieceInfo`` and page ``/Thumb`` images. Returns what was removed.

    The walk starts at the trailer, so ``/Metadata`` and ``/PieceInfo`` on direct dictionaries (an
    annotation written inline in ``/Annots``, say) go too, and so does any other key that points
    straight at an XMP stream, or any array element that is one, nested arrays included (decided
    2026-11-05). pikepdf's ``open_metadata()`` isn't used, so no
    Info key is lost through an XMP sync (the reason the old scrub had to call
    ``load_from_docinfo`` first): each key is kept or removed here by name. ``/LastModified`` (the
    date that accompanies PieceInfo) goes from pages, and from any dictionary that lost its
    PieceInfo, unless ``keep_dates``. Unreferenced objects (old thumbnails, packets) disappear when
    the copy is saved.
    """
    import pikepdf

    removed: list[str] = []
    keep = {"/Title"} | (set(_DATE_KEYS) if keep_dates else set())

    info = _info_dict(pdf)
    if info is not None:
        for key in sorted(str(k) for k in info.keys()):
            if key not in keep:
                del info[key]
                removed.append(f"Info/{key[1:]}")
        if len(info.keys()) == 0:
            del pdf.trailer["/Info"]
    elif "/Info" in pdf.trailer:
        del pdf.trailer["/Info"]                      # a malformed Info that isn't a dictionary
        removed.append("Info")

    if _drop_key(pdf.Root, "/Metadata"):
        removed.append("XMP")
    if _drop_key(pdf.Root, "/PieceInfo"):
        removed.append("PieceInfo")

    counts = {"xmp": 0, "piece": 0, "dates": 0}
    pruned: set[tuple[int, int]] = set()

    def is_packet(value: Any) -> bool:
        return isinstance(value, pikepdf.Stream) and value.get("/Type") == pikepdf.Name.Metadata

    def prune(array: Any) -> None:
        """Drop the packets an array (or an array nested in it) holds, so the walk never reaches them."""
        if array.is_indirect:
            if array.objgen in pruned:
                return
            pruned.add(array.objgen)
        for i in reversed(range(len(array))):
            try:
                value = array[i]
            except Exception:  # noqa: BLE001 - an unreadable element isn't a packet we could see
                continue
            if is_packet(value):
                del array[i]
                counts["xmp"] += 1
            elif isinstance(value, pikepdf.Array):
                prune(value)

    def visit(obj: Any) -> None:
        counts["xmp"] += _drop_key(obj, "/Metadata")
        if _drop_key(obj, "/PieceInfo"):
            counts["piece"] += 1
            if not keep_dates:
                counts["dates"] += _drop_key(obj, "/LastModified")
        for key in list(obj.keys()):                  # any other key that points straight at a packet
            try:
                value = obj[key]
            except Exception:  # noqa: BLE001 - an unreadable value isn't a packet we could see
                continue
            if is_packet(value):
                del obj[key]
                counts["xmp"] += 1
            elif isinstance(value, pikepdf.Array):    # or holds one in an array (decided 2026-11-05)
                prune(value)

    _walk(pdf.trailer, visit)
    thumbs = 0
    for page in pdf.pages:
        thumbs += _drop_key(page.obj, "/Thumb")
        if not keep_dates:
            counts["dates"] += _drop_key(page.obj, "/LastModified")
    if counts["xmp"]:
        removed.append("XMP/objects")
    if counts["piece"] and "PieceInfo" not in removed:
        removed.append("PieceInfo")
    if thumbs:
        removed.append("Thumb")
    if counts["dates"]:
        removed.append("LastModified")
    return removed
