"""pdf/meta.py: snapshots, diffs, the same-length PDFVersion patch and the two named edits (DESIGN §2.2 U9, §5.3)."""

from __future__ import annotations

import io
import re
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pikepdf
import pytest
from fixtures import build
from pikepdf.models.metadata import decode_pdf_date

from pdfintake.pdf import meta, shrink

XMP_ATTR = (
    b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:pdf="http://ns.adobe.com/pdf/1.3/" pdf:PDFVersion="1.4"'
    b' pdf:Producer="Example Producer"/></rdf:RDF></x:xmpmeta>\n' + b" " * 256 + b'\n<?xpacket end="w"?>'
)
XMP_ELEM = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b"<rdf:Description rdf:about='' xmlns:ex='http://ns.adobe.com/pdf/1.3/'>"
    b"<ex:PDFVersion>1.3</ex:PDFVersion><ex:Keywords>alpha</ex:Keywords></rdf:Description></rdf:RDF></x:xmpmeta>\n"
    b'<?xpacket end="w"?>'
)


def new_pdf(*, xmp: bytes | None = None, info: dict[str, str] | None = None, flate_xmp: bool = False) -> pikepdf.Pdf:
    pdf = pikepdf.open(io.BytesIO(build.pdf_bytes([build.PageSpec(texts=(build.TextSpec(build.LATIN_SAMPLE),))])))
    if xmp is not None:
        s = pikepdf.Stream(pdf, zlib.compress(xmp, 9) if flate_xmp else xmp)
        if flate_xmp:
            s.Filter = pikepdf.Name.FlateDecode
        s.Type, s.Subtype = pikepdf.Name.Metadata, pikepdf.Name.XML
        pdf.Root.Metadata = pdf.make_indirect(s)
    if info is not None:
        pdf.trailer.Info = pdf.make_indirect(pikepdf.Dictionary({f"/{k}": pikepdf.String(v) for k, v in info.items()}))
    return pdf


def save(pdf: pikepdf.Pdf, path: Path) -> Path:
    pdf.save(path, fix_metadata_version=False, deterministic_id=True)
    return path


def xmp_of(pdf: pikepdf.Pdf) -> bytes:
    return bytes(pdf.Root.Metadata.read_bytes())


# --------------------------------------------------------------------------- Info without XMP keeps Keywords

def test_info_without_xmp_keeps_keywords_through_snapshot_patch_and_shrink(tmp_path: Path):
    info = {"Title": "Example title", "Keywords": "alpha, beta", "Author": "Example Author"}
    pdf = new_pdf(info=info)
    snap = meta.snapshot(pdf)
    assert snap.info == info and snap.xmp_sha256 is None and snap.xmp_bytes == 0
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is False           # no packet: nothing to patch, none created
    assert "/Metadata" not in pdf.Root
    src = save(pdf, tmp_path / "info-only.pdf")
    rep = shrink.lossless_shrink(src, tmp_path / "out.pdf", content_changed=True)
    assert rep.version_to == "1.5"
    after = meta.snapshot(tmp_path / "out.pdf")
    assert after.info == info and after.info["Keywords"] == "alpha, beta"
    assert after.xmp_sha256 is None
    d = meta.diff(meta.snapshot(src), after)
    assert (d.kept, d.changed, d.dropped) == (["Info"], [], [])


def test_snapshot_leaves_out_info_keys_whose_value_is_null(tmp_path: Path):
    # /Author null is the same as no /Author, and qpdf writes neither: never "dropped" by a build
    path = build.with_null_info(build.write_pdf(tmp_path / "x.pdf"), "Example title", ["Author", "Trapped"])
    assert b"/Author null" in path.read_bytes()
    with pikepdf.open(path) as pdf:
        assert sorted(str(k) for k, _ in pdf.trailer.Info.items()) == ["/Author", "/Title", "/Trapped"]
        before = meta.snapshot(pdf)
        assert before.info == {"Title": "Example title"}
        buf = io.BytesIO(); pdf.save(buf)
    after = meta.snapshot(pikepdf.open(io.BytesIO(buf.getvalue())))
    assert meta.diff(before, after).dropped == []


def test_verify_compares_dictionaries_without_their_null_keys(tmp_path: Path):
    from pdfintake.pdf import verify

    path = build.with_null_info(build.write_pdf(tmp_path / "x.pdf"), "Example title", ["Author"])
    with pikepdf.open(path) as pdf:
        assert verify._canon(pdf.trailer.Info) == verify._canon(pikepdf.Dictionary(Title=pikepdf.String("Example title")))


def test_snapshot_never_adds_an_info_dictionary():
    pdf = new_pdf()
    assert "/Info" not in pdf.trailer
    snap = meta.snapshot(pdf)
    assert snap.info == {} and "/Info" not in pdf.trailer


def test_snapshot_of_a_path_equals_snapshot_of_the_open_pdf(tmp_path: Path):
    pdf = new_pdf(xmp=XMP_ATTR, info={"Title": "Example title"})
    path = save(pdf, tmp_path / "x.pdf")
    with pikepdf.open(path) as opened:
        assert meta.snapshot(path) == meta.snapshot(opened)
    snap = meta.snapshot(path)
    assert snap.xmp_bytes == len(XMP_ATTR) and snap.pdf_version == "1.3"


def test_snapshot_fingerprints_decoded_xmp():
    plain = meta.snapshot(new_pdf(xmp=XMP_ATTR))
    packed = meta.snapshot(new_pdf(xmp=XMP_ATTR, flate_xmp=True))
    assert plain.xmp_sha256 == packed.xmp_sha256 and plain.xmp_bytes == packed.xmp_bytes


# --------------------------------------------------------------------------- the same-length PDFVersion patch

def test_patch_attribute_form_same_length():
    pdf = new_pdf(xmp=XMP_ATTR)
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is True
    new = xmp_of(pdf)
    assert len(new) == len(XMP_ATTR)
    assert new == XMP_ATTR.replace(b'pdf:PDFVersion="1.4"', b'pdf:PDFVersion="1.5"')
    assert "/Filter" not in pdf.Root.Metadata


def test_patch_element_form_with_another_prefix():
    pdf = new_pdf(xmp=XMP_ELEM)
    assert meta.patch_xmp_pdfversion(pdf, "1.7") is True
    assert xmp_of(pdf) == XMP_ELEM.replace(b"<ex:PDFVersion>1.3<", b"<ex:PDFVersion>1.7<")


def test_patch_a_flate_packet_stays_flate():
    pdf = new_pdf(xmp=XMP_ATTR, flate_xmp=True)
    assert meta.patch_xmp_pdfversion(pdf, "2.0") is True
    assert str(pdf.Root.Metadata.Filter) == "/FlateDecode"
    assert xmp_of(pdf) == XMP_ATTR.replace(b'"1.4"', b'"2.0"')


def test_patch_refuses_a_different_length_or_an_equal_value():
    pdf = new_pdf(xmp=XMP_ATTR)
    assert meta.patch_xmp_pdfversion(pdf, "1.10") is False
    assert meta.patch_xmp_pdfversion(pdf, "1.4") is False
    assert xmp_of(pdf) == XMP_ATTR


def test_patch_never_lowers_a_higher_value_or_touches_an_odd_one():
    higher = XMP_ATTR.replace(b'pdf:PDFVersion="1.4"', b'pdf:PDFVersion="1.7"')
    pdf = new_pdf(xmp=higher)
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is False and xmp_of(pdf) == higher
    odd = XMP_ATTR.replace(b'pdf:PDFVersion="1.4"', b'pdf:PDFVersion="x.y"')
    pdf = new_pdf(xmp=odd)
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is False and xmp_of(pdf) == odd


def test_patch_leaves_a_packet_without_pdfversion_or_with_an_odd_filter():
    no_version = XMP_ATTR.replace(b' pdf:PDFVersion="1.4"', b"")
    pdf = new_pdf(xmp=no_version)
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is False and xmp_of(pdf) == no_version
    odd = new_pdf(xmp=XMP_ATTR)
    odd.Root.Metadata.write(XMP_ATTR.hex().encode() + b">", filter=pikepdf.Name.ASCIIHexDecode)
    assert meta.patch_xmp_pdfversion(odd, "1.5") is False


def test_patch_ignores_pdfversion_in_other_namespaces():
    foreign = XMP_ATTR.replace(b'xmlns:pdf="http://ns.adobe.com/pdf/1.3/"', b'xmlns:pdf="http://example.com/other/"')
    pdf = new_pdf(xmp=foreign)
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is False


def test_patch_rejects_a_value_that_isnt_a_version():
    with pytest.raises(ValueError):
        meta.patch_xmp_pdfversion(new_pdf(xmp=XMP_ATTR), '1.5" evil="1')


# --------------------------------------------------------------------------- diff

def snap(info: dict[str, str], xmp: str | None, version: str = "1.7") -> meta.MetaSnapshot:
    return meta.MetaSnapshot(info=info, xmp_sha256=xmp, xmp_bytes=0 if xmp is None else 10, pdf_version=version)


def test_diff_whole_parts_kept():
    a = snap({"Title": "Example", "Keywords": "alpha"}, "aa")
    d = meta.diff(a, snap(dict(a.info), "aa", "1.5"))                 # the header version isn't metadata here
    assert (d.kept, d.changed, d.dropped) == (["Info", "XMP"], [], [])


def test_diff_per_key_when_info_differs():
    a = snap({"Title": "Example", "Keywords": "alpha", "Author": "Example Author"}, "aa")
    b = snap({"Title": "Example", "Keywords": "beta", "Producer": "Example Producer"}, "bb")
    d = meta.diff(a, b)
    assert d.kept == ["Info/Title"]
    assert d.changed == ["Info/Keywords", "Info/Producer", "XMP"]
    assert d.dropped == ["Info/Author"]


def test_diff_xmp_added_dropped_and_absent():
    assert meta.diff(snap({}, None), snap({}, None)) == meta.MetaDiff([], [], [])
    assert meta.diff(snap({}, "aa"), snap({}, None)).dropped == ["XMP"]
    assert meta.diff(snap({}, None), snap({}, "aa")).changed == ["XMP"]


def test_diff_after_a_real_version_patch_names_xmp_changed(tmp_path: Path):
    src = save(new_pdf(xmp=XMP_ATTR, info={"Title": "Example title"}), tmp_path / "v.pdf")
    out = tmp_path / "out.pdf"
    rep = shrink.lossless_shrink(src, out, content_changed=True)
    assert rep.xmp_pdfversion_patched
    d = meta.diff(meta.snapshot(src), meta.snapshot(out))
    assert (d.kept, d.changed, d.dropped) == (["Info"], ["XMP"], [])


# --------------------------------------------------------------------------- set_built_info

def test_set_built_info_writes_only_producer_and_creation_date():
    pdf = new_pdf(info={"Title": "Example title", "Keywords": "alpha"})
    created = datetime(2001, 2, 3, 4, 5, 6, tzinfo=timezone(timedelta(hours=-3)))
    meta.set_built_info(pdf, producer="pdfintake 0.1.0", created=created)
    info = meta.snapshot(pdf).info
    assert sorted(info) == ["CreationDate", "Producer"] and info["Producer"] == "pdfintake 0.1.0"
    assert re.fullmatch(r"D:\d{14}-03'00", info["CreationDate"])
    assert decode_pdf_date(info["CreationDate"]) == created
    assert "/Metadata" not in pdf.Root


def test_set_built_info_takes_a_naive_time_as_utc():
    pdf = new_pdf()
    meta.set_built_info(pdf, producer="pdfintake 0.1.0", created=datetime(2001, 1, 1))
    stamp = meta.snapshot(pdf).info["CreationDate"]
    assert stamp.endswith("+00'00") and decode_pdf_date(stamp) == datetime(2001, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- strip_for_export

def _with_extras(pdf: pikepdf.Pdf) -> pikepdf.Pdf:
    page = pdf.pages[0].obj
    page.PieceInfo = pikepdf.Dictionary(Example_Marker=pikepdf.Dictionary(Private=pikepdf.Name.Example))
    page.LastModified = pikepdf.String("D:2001")
    thumb = pikepdf.Stream(pdf, b"\x00" * 4)
    thumb.Width, thumb.Height, thumb.BitsPerComponent, thumb.ColorSpace = 2, 2, 8, pikepdf.Name.DeviceGray
    page.Thumb = pdf.make_indirect(thumb)
    font = page.Resources.Font.F1
    font.Metadata = pdf.make_indirect(pikepdf.Stream(pdf, XMP_ATTR))
    return pdf


def test_strip_for_export_keeps_title_and_dates_when_asked():
    info = {"Title": "Example title", "Author": "Example Author", "Keywords": "alpha",
            "CreationDate": "D:2001", "ModDate": "D:2002", "Producer": "Example Producer"}
    pdf = _with_extras(new_pdf(xmp=XMP_ATTR, info=info))
    removed = meta.strip_for_export(pdf, keep_dates=True)
    assert removed == ["Info/Author", "Info/Keywords", "Info/Producer", "XMP", "XMP/objects", "PieceInfo", "Thumb"]
    assert meta.snapshot(pdf).info == {"CreationDate": "D:2001", "ModDate": "D:2002",
                                       "Title": "Example title"}
    page = pdf.pages[0].obj
    assert "/Metadata" not in pdf.Root and "/PieceInfo" not in page and "/Thumb" not in page
    assert "/LastModified" in page


def test_strip_for_export_drops_dates_otherwise(tmp_path: Path):
    pdf = _with_extras(new_pdf(info={"Title": "Example title", "CreationDate": "D:2001"}))
    removed = meta.strip_for_export(pdf, keep_dates=False)
    assert "Info/CreationDate" in removed and "LastModified" in removed
    assert meta.snapshot(pdf).info == {"Title": "Example title"}
    out = save(pdf, tmp_path / "clean.pdf")
    assert b"pdf:Producer" not in out.read_bytes()                     # the font's packet is gone once unreferenced


def test_strip_for_export_removes_an_info_left_empty():
    pdf = new_pdf(info={"Keywords": "alpha"})
    assert meta.strip_for_export(pdf, keep_dates=True) == ["Info/Keywords"]
    assert "/Info" not in pdf.trailer


# --------------------------------------------------------------------------- strip_for_export walks from the trailer

def _reachable_leftovers(path: Path) -> set[str]:
    """What export's leftover check looks for, walked from the trailer as it does."""
    found: set[str] = set()
    with pikepdf.open(path) as pdf:
        def visit(obj):
            for key in ("/Metadata", "/PieceInfo"):
                if key in obj:
                    found.add(key)
            if obj.get("/Type") == pikepdf.Name.Metadata:
                found.add("XMP stream")
        meta._walk(pdf.trailer, visit)
    return found


def test_strip_for_export_reaches_direct_dictionaries(tmp_path: Path):
    pdf = new_pdf(xmp=XMP_ATTR, info={"Title": "Example title"})
    page = pdf.pages[0].obj
    packet = pikepdf.Stream(pdf, XMP_ATTR)
    packet.Type, packet.Subtype = pikepdf.Name.Metadata, pikepdf.Name.XML
    inline = pikepdf.Dictionary(                                      # an annotation written inline in /Annots
        Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Text, Rect=[0, 0, 10, 10],
        Metadata=pdf.make_indirect(packet), LastModified=pikepdf.String("D:2001"),
        PieceInfo=pikepdf.Dictionary(Example_App=pikepdf.Dictionary(Private=pikepdf.Name.Example)))
    page.Annots = pikepdf.Array([inline])
    page.Example_Ref = pdf.make_indirect(pikepdf.Stream(pdf, XMP_ATTR, Type=pikepdf.Name.Metadata))

    def packet_ref():
        return pdf.make_indirect(pikepdf.Stream(pdf, XMP_ATTR, Type=pikepdf.Name.Metadata))

    page.Example_Arr = pikepdf.Array([1, packet_ref(), pikepdf.Name.Example])      # an array holding packets,
    page.Example_Nest = pdf.make_indirect(pikepdf.Array([pikepdf.Array([packet_ref()]), 2]))  # one nested
    removed = meta.strip_for_export(pdf, keep_dates=False)
    assert "XMP/objects" in removed and "PieceInfo" in removed and "LastModified" in removed
    annot = pdf.pages[0].obj.Annots[0]
    assert not annot.is_indirect
    assert "/Metadata" not in annot and "/PieceInfo" not in annot and "/LastModified" not in annot
    assert "/Example_Ref" not in pdf.pages[0].obj
    assert list(pdf.pages[0].obj.Example_Arr) == [1, pikepdf.Name.Example]          # the rest stays
    assert len(pdf.pages[0].obj.Example_Nest[0]) == 0 and pdf.pages[0].obj.Example_Nest[1] == 2
    out = save(pdf, tmp_path / "clean.pdf")
    assert _reachable_leftovers(out) == set()
    assert b"pdf:Producer" not in out.read_bytes()


def test_strip_for_export_keeps_an_annotations_date_when_asked():
    pdf = new_pdf()
    pdf.pages[0].obj.Annots = pikepdf.Array([pikepdf.Dictionary(
        Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Text, Rect=[0, 0, 1, 1], LastModified=pikepdf.String("D:2001"),
        PieceInfo=pikepdf.Dictionary(Example_App=pikepdf.Dictionary()))])
    assert meta.strip_for_export(pdf, keep_dates=True) == ["PieceInfo"]
    assert "/LastModified" in pdf.pages[0].obj.Annots[0]


# --------------------------------------------------------------------------- mark_derived and derived_from

SHA = "0123456789abcdef" * 4
RDF = b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'


def _tree(data: bytes):
    import xml.etree.ElementTree as ET

    text = data.decode("utf-8") if not data.startswith((b"\xff\xfe", b"\xfe\xff")) else data.decode("utf-16")
    return ET.fromstring(text.lstrip("﻿"))


def test_mark_derived_creates_a_packet_when_there_is_none(tmp_path: Path):
    info = {"Title": "Example title", "Keywords": "alpha"}
    pdf = new_pdf(info=info)
    meta.mark_derived(pdf, "Example scan.pdf", SHA)
    assert meta.derived_from(pdf) == {"name": "Example scan.pdf", "sha256": SHA, "note": meta.DERIVED_NOTE}
    assert meta.DERIVED_NOTE == "searchable copy, not the original"
    stream = pdf.Root.Metadata
    assert stream.is_indirect and stream.Type == pikepdf.Name.Metadata and stream.Subtype == pikepdf.Name.XML
    _tree(xmp_of(pdf))                                                 # well-formed XML
    out = save(pdf, tmp_path / "copy.pdf")
    with pikepdf.open(out) as back:
        assert meta.derived_from(back)["sha256"] == SHA
        assert meta.snapshot(back).info == info                       # Info untouched
    assert b"pdfintake:DerivedFrom" in xmp_of(pikepdf.open(out))      # the form export's detector reads


def test_mark_derived_inserts_its_block_and_keeps_every_other_byte():
    pdf = new_pdf(xmp=XMP_ATTR)
    meta.mark_derived(pdf, "Example scan.pdf", SHA)
    new = xmp_of(pdf)
    at = XMP_ATTR.rindex(b"</rdf:RDF>")
    assert new.startswith(XMP_ATTR[:at]) and new.endswith(XMP_ATTR[at:])
    block = new[at:len(new) - len(XMP_ATTR) + at]
    assert block.startswith(b"<rdf:Description " + RDF + b' rdf:about=""')
    assert meta.derived_from(pdf)["name"] == "Example scan.pdf"
    assert meta.patch_xmp_pdfversion(pdf, "1.5") is True               # the version patch still finds its value


def test_mark_derived_keeps_a_flate_packet_flate_and_reuses_rdf_about():
    xmp = XMP_ELEM.replace(b"rdf:about=''", b"rdf:about='uuid:00000000-0000-0000-0000-000000000000'")
    pdf = new_pdf(xmp=xmp, flate_xmp=True)
    meta.mark_derived(pdf, "Example.pdf", SHA)
    assert str(pdf.Root.Metadata.Filter) == "/FlateDecode"
    assert b"rdf:about='uuid:00000000-0000-0000-0000-000000000000' xmlns:pdfintake=" in xmp_of(pdf)
    assert meta.derived_from(pdf)["sha256"] == SHA


def test_mark_derived_with_another_rdf_prefix_and_a_utf16_packet():
    other = XMP_ATTR.replace(b"rdf:", b"r:").replace(b"xmlns:rdf=", b"xmlns:r=")
    pdf = new_pdf(xmp=other)
    meta.mark_derived(pdf, "Example.pdf", SHA)
    assert xmp_of(pdf).index(b"<rdf:Description") < xmp_of(pdf).rindex(b"</r:RDF>")
    assert meta.derived_from(pdf)["name"] == "Example.pdf"

    utf16 = b"\xff\xfe" + XMP_ATTR.decode("utf-8").replace("﻿", "").encode("utf-16-le")
    pdf = new_pdf(xmp=utf16)
    meta.mark_derived(pdf, "Example.pdf", SHA)
    assert xmp_of(pdf).startswith(b"\xff\xfe")
    assert meta.derived_from(pdf)["sha256"] == SHA


def test_mark_derived_escapes_the_name_and_drops_what_xml_cant_carry():
    name = 'Jane & <Doe> "scan"\tone\x01.pdf'
    pdf = new_pdf(xmp=XMP_ATTR)
    meta.mark_derived(pdf, name, SHA)
    assert meta.derived_from(pdf)["name"] == 'Jane & <Doe> "scan"\tone�.pdf'
    for evil in ("Example<\\pdfintake:OriginalName>.pdf", "]]><x&amp;y>.pdf"):
        pdf = new_pdf()
        meta.mark_derived(pdf, evil, SHA)
        assert meta.derived_from(pdf)["name"] == evil


def test_marking_again_replaces_the_block():
    pdf = new_pdf(xmp=XMP_ATTR)
    meta.mark_derived(pdf, "First.pdf", SHA)
    once = len(xmp_of(pdf))
    other = "f" * 64
    meta.mark_derived(pdf, "Again.pdf", other)
    assert meta.derived_from(pdf) == {"name": "Again.pdf", "sha256": other, "note": meta.DERIVED_NOTE}
    assert xmp_of(pdf).count(b"DerivedFrom rdf:parseType") == 1 and len(xmp_of(pdf)) == once


@pytest.mark.parametrize("about", [b"rdf:about='say \"hi\"'", b"rdf:about=\"it's\""])
def test_marking_again_replaces_the_block_when_rdf_about_holds_the_other_quote(about: bytes):
    pdf = new_pdf(xmp=XMP_ATTR.replace(b'rdf:about=""', about))
    meta.mark_derived(pdf, "First.pdf", SHA)
    once = xmp_of(pdf)
    assert once.count(about) == 2                                     # the block reuses it as written
    other = "f" * 64
    meta.mark_derived(pdf, "Again.pdf", other)
    assert meta.derived_from(pdf) == {"name": "Again.pdf", "sha256": other, "note": meta.DERIVED_NOTE}
    assert xmp_of(pdf).count(b"DerivedFrom rdf:parseType") == 1 and len(xmp_of(pdf)) == len(once)


def test_a_copy_save_with_the_documented_flags_keeps_the_xmp_bytes(tmp_path: Path):
    """The flags the module docstring gives every save of a copy, the decrypt-once work copy
    included, keep an encrypted original's packet byte for byte; pikepdf's default doesn't."""
    src = tmp_path / "locked.pdf"
    new_pdf(xmp=XMP_ATTR).save(src, fix_metadata_version=False,
                               encryption=pikepdf.Encryption(owner="owner-example", user=""))
    with pikepdf.open(src) as pdf:
        assert pdf.is_encrypted and xmp_of(pdf) == XMP_ATTR
        pdf.save(tmp_path / "work.pdf", encryption=False, deterministic_id=True, fix_metadata_version=False)
        pdf.save(tmp_path / "default.pdf", encryption=False, deterministic_id=True)
    with pikepdf.open(tmp_path / "work.pdf") as work:
        assert not work.is_encrypted and xmp_of(work) == XMP_ATTR
    with pikepdf.open(tmp_path / "default.pdf") as default:
        assert xmp_of(default) != XMP_ATTR                            # why the flag is required


def test_mark_derived_replaces_a_whitespace_only_packet():
    pdf = new_pdf(xmp=b"  \n ")
    meta.mark_derived(pdf, "Example.pdf", SHA)
    assert meta.derived_from(pdf)["name"] == "Example.pdf"


@pytest.mark.parametrize("name, sha", [
    ("folder/Example.pdf", SHA), ("", SHA), ("Example.pdf", SHA.upper()), ("Example.pdf", SHA[:-1]),
    ("Example.pdf", "g" * 64)])
def test_mark_derived_refuses_bad_arguments(name: str, sha: str):
    pdf = new_pdf(xmp=XMP_ATTR)
    with pytest.raises(ValueError):
        meta.mark_derived(pdf, name, sha)
    assert xmp_of(pdf) == XMP_ATTR


@pytest.mark.parametrize("xmp", [
    b"<x:xmpmeta xmlns:x='adobe:ns:meta/'/>",                                              # no rdf:RDF
    XMP_ATTR.replace(b"</rdf:RDF>", b"</rdf:RDF><broken"),                                 # won't parse
    b'<!DOCTYPE x [<!ENTITY e "e">]><rdf:RDF ' + RDF + b"></rdf:RDF>",                      # an entity
    b"\xff\xff\xfe junk",                                                                   # not UTF-8
])
def test_mark_derived_leaves_a_packet_it_cant_edit(xmp: bytes):
    pdf = new_pdf(xmp=xmp)
    with pytest.raises(ValueError):
        meta.mark_derived(pdf, "Example.pdf", SHA)
    assert xmp_of(pdf) == xmp


def test_derived_from_ignores_xmpmm_derived_from_and_absent_packets():
    assert meta.derived_from(new_pdf()) is None
    xmpmm = XMP_ATTR.replace(b"/></rdf:RDF>", b' xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">'
                             b'<xmpMM:DerivedFrom rdf:parseType="Resource"/></rdf:Description></rdf:RDF>')
    pdf = new_pdf(xmp=xmpmm)
    _tree(xmpmm)
    assert meta.derived_from(pdf) is None
    assert meta.derived_from(new_pdf(xmp=b"not xml")) is None


# --------------------------------------------------------------------------- drop_conformance_claims

PDFA_ATTR = XMP_ATTR.replace(
    b' pdf:Producer="Example Producer"/>',
    b' pdf:Producer="Example Producer"/><rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/"'
    b' pdfaid:part="2" pdfaid:conformance="B"/>')
PDFA_ELEM = XMP_ELEM.replace(
    b"<ex:Keywords>alpha</ex:Keywords></rdf:Description>",
    b"<ex:Keywords>alpha</ex:Keywords></rdf:Description><rdf:Description rdf:about='' "
    b"xmlns:a='http://www.aiim.org/pdfa/ns/id' xmlns:ua='http://www.aiim.org/pdfua/ns/id/'>"
    b"<a:part>3</a:part>\n <a:conformance>U</a:conformance><a:amd/><ua:part>1</ua:part></rdf:Description>")


def _claims(path: Path):
    from pdfintake import refusals

    rep = refusals.read(path)
    return rep.pdfa, rep.pdfua, sorted(m for m in rep.raw_markers if "id:part" in m), \
        sorted(c for c in rep.walk_codes if c.startswith("object_"))


def test_drop_conformance_claims_attribute_form(tmp_path: Path):
    src = save(new_pdf(xmp=PDFA_ATTR), tmp_path / "a.pdf")
    assert _claims(src)[0] == "2B"
    with pikepdf.open(src) as pdf:
        assert meta.drop_conformance_claims(pdf) == ["pdfaid:conformance", "pdfaid:part"]
        new = xmp_of(pdf)
        assert new.replace(b'<rdf:Description rdf:about=""/>', b"") == XMP_ATTR
        out = save(pdf, tmp_path / "out.pdf")
    assert _claims(out) == (None, None, [], [])


def test_drop_conformance_claims_element_form_other_prefixes_and_flate(tmp_path: Path):
    src = save(new_pdf(xmp=PDFA_ELEM), tmp_path / "e.pdf")
    assert _claims(src)[:2] == ("3U", "1")
    pdf = new_pdf(xmp=PDFA_ELEM, flate_xmp=True)                     # in memory: qpdf saves XMP unfiltered
    removed = meta.drop_conformance_claims(pdf)
    assert removed == ["pdfaid:amd", "pdfaid:conformance", "pdfaid:part", "pdfuaid:part"]
    assert str(pdf.Root.Metadata.Filter) == "/FlateDecode"
    new = xmp_of(pdf)
    assert b"aiim.org" not in new and b"<ex:Keywords>alpha</ex:Keywords>" in new
    _tree(new)
    out = save(pdf, tmp_path / "out.pdf")
    assert _claims(out) == (None, None, [], [])


def test_drop_conformance_claims_reaches_a_pages_own_packet(tmp_path: Path):
    pdf = new_pdf(xmp=XMP_ATTR)
    own = pikepdf.Stream(pdf, PDFA_ATTR)
    own.Type, own.Subtype = pikepdf.Name.Metadata, pikepdf.Name.XML
    pdf.pages[0].obj.Metadata = pdf.make_indirect(own)
    src = save(pdf, tmp_path / "p.pdf")
    assert _claims(src)[3] == ["object_pdfa"]
    with pikepdf.open(src) as opened:
        assert meta.drop_conformance_claims(opened) == ["pdfaid:conformance", "pdfaid:part"]
        assert xmp_of(opened) == XMP_ATTR                              # the document's packet had no claim
        out = save(opened, tmp_path / "out.pdf")
    assert _claims(out) == (None, None, [], [])


def test_drop_conformance_claims_without_a_claim_changes_nothing():
    pdf = new_pdf(xmp=XMP_ATTR, flate_xmp=True)
    raw = bytes(pdf.Root.Metadata.read_raw_bytes())
    assert meta.drop_conformance_claims(pdf) == []
    assert bytes(pdf.Root.Metadata.read_raw_bytes()) == raw
    assert meta.drop_conformance_claims(new_pdf()) == []


def test_drop_conformance_claims_refuses_an_unreadable_document_packet_that_claims():
    bad = PDFA_ATTR.replace(b"</rdf:RDF>", b"</rdf:RDF><broken")
    pdf = new_pdf(xmp=bad)
    with pytest.raises(ValueError):
        meta.drop_conformance_claims(pdf)
    assert xmp_of(pdf) == bad


def test_a_copy_gets_both_edits_and_keeps_info(tmp_path: Path):
    info = {"Title": "Example title", "Author": "Jane Doe"}
    pdf = new_pdf(xmp=PDFA_ATTR, info=info)
    assert meta.drop_conformance_claims(pdf)
    meta.mark_derived(pdf, "Example.pdf", SHA)
    out = save(pdf, tmp_path / "copy.pdf")
    with pikepdf.open(out) as back:
        assert meta.derived_from(back)["name"] == "Example.pdf" and meta.snapshot(back).info == info
    assert _claims(out) == (None, None, [], [])
