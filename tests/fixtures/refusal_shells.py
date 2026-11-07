"""PDFs pdfintake refuses or only copies (and a few it leaves alone), built in code for tests/test_refusals.py.

Since DESIGN §10 and §11.1 (decided 2026-11-05) most of these are copy reasons: the original stays
untouched and a searchable copy goes beside it. Only XFA, unreadable XMP and a disagreement between
the two detectors still refuse.

* pyHanko-built: visible and invisible signatures, certified P=1 and P=2, an empty signature
  field, a signed field without a widget, a document timestamp. pyHanko is a ``[test]`` extra and
  is imported only inside these builders. Keys and certificates are generated at test time
  (CN "Example Test Signer" and "Example Test TSA", O "example.com") and never written to disk.
* pikepdf shells: UR3 usage rights, PDF/A-2B, PDF/UA, owner- and user-password encryption, XFA,
  unreadable XMP. Signature dictionaries in the shells carry zeroed byte ranges: the refusal
  reader looks at structure, never at whether a signature verifies.
* Damage: every fixture also comes with its ``startxref`` offset broken and with its trailer cut
  off (``DAMAGES``).
* ``append_update`` writes a classic incremental update, used for an unreferenced signature
  dictionary and for a signature an update removed from the form.
* Hostile and edge-case files for the raw scan: unreferenced signature dictionaries hidden in
  object streams with predictors or filter chains, behind name escapes, comments, indirect
  arrays, fake stream keywords and broken ``endstream`` keywords; marker text in strings; XMP
  that belongs to a page or an image rather than the document.

Every text is made up. Nothing here reads a real document.
"""

from __future__ import annotations

import datetime
import functools
import io
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fixtures import build

PASSWORD_OWNER = "example-owner-password"
PASSWORD_USER = "example-user-password"

XMP_PDFA_2B = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
   <pdfaid:part>2</pdfaid:part>
   <pdfaid:conformance>B</pdfaid:conformance>
  </rdf:Description>
  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">Example archival sample</rdf:li></rdf:Alt></dc:title>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

XMP_PDFA_ATTR = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/" pdfaid:part="3" pdfaid:conformance="u"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

# The same claim under a prefix other than "pdfaid": both detectors follow the namespace binding.
XMP_PDFA_OTHER_PREFIX = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:archive="http://www.aiim.org/pdfa/ns/id/" archive:part="1" archive:conformance="B"/>
 </rdf:RDF>
</x:xmpmeta>"""

# Element form, the namespace URI without its trailing slash, single quotes.
XMP_PDFA_OTHER_PREFIX_ELEMENT = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:a.id = 'http://www.aiim.org/pdfa/ns/id'>
   <a.id:part>1</a.id:part>
   <a.id:conformance>B</a.id:conformance>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""

XMP_PDFUA_OTHER_PREFIX = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:ua="http://www.aiim.org/pdfua/ns/id/" ua:part="1"/>
 </rdf:RDF>
</x:xmpmeta>"""

# A "part" property in some other namespace: not a PDF/A claim to either detector.
XMP_UNRELATED_PART = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:archive="http://example.com/ns/archive/" archive:part="1"/>
 </rdf:RDF>
</x:xmpmeta>"""

XMP_PDFUA_1 = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:pdfuaid="http://www.aiim.org/pdfua/ns/id/" pdfuaid:part="1"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

XMP_PLAIN = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:CreatorTool="Example Writer 1.0"/>
 </rdf:RDF>
</x:xmpmeta>"""

XMP_BROKEN = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" <broken attribute="example"
</x:xmpmeta>"""

XMP_DOCTYPE = """<?xml version="1.0"?>
<!DOCTYPE x:xmpmeta [<!ENTITY sample "example">]>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">&sample;</rdf:RDF></x:xmpmeta>"""


# --------------------------------------------------------------------------- base documents

def base_pdf(pages: int = 1, *, object_streams: bool = False) -> bytes:
    """A plain PDF with a line of made-up text on each page, deterministic.

    Classic xref table by default; ``object_streams`` packs the dictionaries into object streams
    behind an xref stream, as most current writers do.
    """
    specs = [build.PageSpec(texts=(build.TextSpec(f"Example page {i + 1}: {build.LATIN_SAMPLE}"),))
             for i in range(pages)]
    data = build.pdf_bytes(specs)
    if object_streams:
        import pikepdf

        data = _with(data, lambda pdf: None, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    return data


def _save(pdf: Any, **kwargs: Any) -> bytes:
    buf = io.BytesIO()
    kwargs.setdefault("deterministic_id", "encryption" not in kwargs)
    kwargs.setdefault("fix_metadata_version", False)       # pikepdf would otherwise rewrite the XMP
    pdf.save(buf, **kwargs)
    return buf.getvalue()


def _with(data: bytes, edit: Callable[[Any], None], **save: Any) -> bytes:
    import pikepdf

    with pikepdf.open(io.BytesIO(data)) as pdf:
        edit(pdf)
        return _save(pdf, **save)


def _metadata(pdf: Any, xmp: str | bytes) -> Any:
    import pikepdf

    data = xmp.encode("utf-8") if isinstance(xmp, str) else xmp
    stream = pikepdf.Stream(pdf, data)
    stream.Type = pikepdf.Name.Metadata
    stream.Subtype = pikepdf.Name.XML
    return pdf.make_indirect(stream)


def _add_flate_metadata(data: bytes, xmp: str | bytes, *, filter_text: str = "/FlateDecode") -> bytes:
    """``data`` plus an incremental update giving the catalog a Flate-compressed XMP stream.

    qpdf always writes /Metadata streams uncompressed, so a compressed one has to be written by hand.
    ``filter_text`` is the /Filter value as written (``[/FlateDecode]``, ``/Fl``, ...).
    """
    import zlib

    import pikepdf

    raw = xmp.encode("utf-8") if isinstance(xmp, str) else xmp
    packed = zlib.compress(raw, 9)
    with pikepdf.open(io.BytesIO(data)) as pdf:
        placeholder = pdf.make_indirect(pikepdf.Stream(pdf, b""))
        pdf.Root.Metadata = placeholder
        num, root = placeholder.objgen[0], pdf.Root.objgen[0]
        root_text = pdf.Root.unparse(resolved=True)
    stream = (f"<< /Type /Metadata /Subtype /XML /Filter {filter_text} /Length {len(packed)} >>\nstream\n"
              .encode("ascii") + packed + b"\nendstream")
    return append_update(data, {num: stream, root: root_text})


def _zero_sig(pdf: Any, *, kind: str = "/Sig", reference: Any = None) -> Any:
    import pikepdf

    sig = pikepdf.Dictionary(
        Type=pikepdf.Name(kind), Filter=pikepdf.Name("/Adobe.PPKLite"),
        SubFilter=pikepdf.Name("/adbe.pkcs7.detached"),
        ByteRange=pikepdf.Array([0, 0, 0, 0]), Contents=pikepdf.String(b"\x00" * 32))
    if reference is not None:
        sig.Reference = reference
    return pdf.make_indirect(sig)


# --------------------------------------------------------------------------- pikepdf shells

def ur3(*, object_streams: bool = False) -> bytes:
    """Usage rights: /Perms /UR3 pointing at a signature dictionary with a UR3 transform.

    With ``object_streams`` the catalog is compressed; qpdf never puts a signature dictionary
    in an object stream, so the dictionary itself stays readable in the file.
    """
    import pikepdf

    def edit(pdf: Any) -> None:
        ref = pikepdf.Array([pikepdf.Dictionary(
            Type=pikepdf.Name.SigRef, TransformMethod=pikepdf.Name.UR3,
            TransformParams=pikepdf.Dictionary(Type=pikepdf.Name.TransformParams, V=pikepdf.Name("/2.2"),
                                               Document=pikepdf.Array([pikepdf.Name.FullSave])))])
        pdf.Root.Perms = pikepdf.Dictionary(UR3=_zero_sig(pdf, reference=ref))

    mode = pikepdf.ObjectStreamMode.generate if object_streams else pikepdf.ObjectStreamMode.disable
    return _with(base_pdf(), edit, object_stream_mode=mode)


def _keep_streams() -> dict[str, Any]:
    """Save options that write every stream exactly as built (no decoding, no recompression)."""
    import pikepdf

    return {"compress_streams": False, "stream_decode_level": pikepdf.StreamDecodeLevel.none}


def with_xmp(xmp: str | bytes, *, compress: bool = False, filter_text: str = "/FlateDecode") -> bytes:
    """The base PDF with this XMP as the catalog's /Metadata, byte for byte (Flate when ``compress``)."""
    if compress:
        return _add_flate_metadata(base_pdf(), xmp, filter_text=filter_text)

    def edit(pdf: Any) -> None:
        pdf.Root.Metadata = _metadata(pdf, xmp)

    return _with(base_pdf(), edit, **_keep_streams())


def pdfa(xmp: str = XMP_PDFA_2B, *, compress: bool = False) -> bytes:
    """PDF/A-2B by default: the claim lives only in the catalog's XMP."""
    return with_xmp(xmp, compress=compress)


def pdfua(*, compress: bool = False) -> bytes:
    """PDF/UA-1: the XMP claim plus a marked, minimally tagged catalog."""
    import pikepdf

    def edit(pdf: Any) -> None:
        if not compress:
            pdf.Root.Metadata = _metadata(pdf, XMP_PDFUA_1)
        pdf.Root.MarkInfo = pikepdf.Dictionary(Marked=True)
        pdf.Root.StructTreeRoot = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.StructTreeRoot,
                                                                       K=pikepdf.Array([])))
        pdf.Root.Lang = pikepdf.String("en")

    data = _with(base_pdf(), edit, **_keep_streams())
    return _add_flate_metadata(data, XMP_PDFUA_1) if compress else data


def xmp_unreadable(xmp: str = XMP_BROKEN) -> bytes:
    return with_xmp(xmp)


def encrypted_owner() -> bytes:
    """An owner password with restricted permissions and an empty user password: opens without asking."""
    import pikepdf

    with pikepdf.open(io.BytesIO(base_pdf())) as pdf:
        return _save(pdf, encryption=pikepdf.Encryption(
            owner=PASSWORD_OWNER, user="", allow=pikepdf.Permissions(modify_other=False, extract=False)))


def encrypted_user() -> bytes:
    """A user password: nothing opens without it."""
    import pikepdf

    with pikepdf.open(io.BytesIO(base_pdf())) as pdf:
        return _save(pdf, encryption=pikepdf.Encryption(owner=PASSWORD_OWNER, user=PASSWORD_USER))


def encrypt(data: bytes, *, user: str = PASSWORD_USER, object_streams: bool = False) -> bytes:
    """``data`` saved with encryption: an open password ``user`` ("" for permissions only).

    With ``object_streams`` the file's object streams are kept, so whatever they hold is encrypted
    stream data the raw scan can't read.
    """
    import pikepdf

    mode = pikepdf.ObjectStreamMode.preserve if object_streams else pikepdf.ObjectStreamMode.disable
    with pikepdf.open(io.BytesIO(data)) as pdf:
        return _save(pdf, object_stream_mode=mode, encryption=pikepdf.Encryption(
            owner=PASSWORD_OWNER, user=user, allow=pikepdf.Permissions(modify_other=False, extract=False)))


def decrypt(data: bytes, password: str = PASSWORD_USER) -> bytes:
    """The unencrypted work copy the flow makes of an encrypted input (§11.1: ``encryption=False``)."""
    import pikepdf

    with pikepdf.open(io.BytesIO(data), password=password) as pdf:
        return _save(pdf, encryption=False)


def xfa(*, object_streams: bool = False) -> bytes:
    """An AcroForm with an /XFA packet (inside a compressed object stream when ``object_streams``)."""
    import pikepdf

    def edit(pdf: Any) -> None:
        packet = pdf.make_stream(b'<xdp:xdp xmlns:xdp="http://ns.adobe.com/xdp/"><template/></xdp:xdp>')
        pdf.Root.AcroForm = pdf.make_indirect(pikepdf.Dictionary(Fields=pikepdf.Array([]), XFA=packet))

    mode = pikepdf.ObjectStreamMode.generate if object_streams else pikepdf.ObjectStreamMode.disable
    return _with(base_pdf(), edit, object_stream_mode=mode)


def jbig2_image() -> bytes:
    """A page whose image stream claims /JBIG2Decode (placeholder bytes; only the filter matters here)."""
    import pikepdf

    def edit(pdf: Any) -> None:
        img = pikepdf.Stream(pdf, b"\x00" * 64)
        img.Type, img.Subtype = pikepdf.Name.XObject, pikepdf.Name.Image
        img.Width, img.Height, img.BitsPerComponent = 8, 8, 1
        img.ColorSpace, img.Filter = pikepdf.Name.DeviceGray, pikepdf.Name.JBIG2Decode
        page = pdf.pages[0]
        page.obj.Resources.XObject = pikepdf.Dictionary(Im9=pdf.make_indirect(img))

    return _with(base_pdf(), edit)


def inherited_sig_field() -> bytes:
    """A field tree: the parent carries /FT /Sig, its kid (a merged field and widget) the signature value."""
    import pikepdf

    def edit(pdf: Any) -> None:
        page = pdf.pages[0].obj
        parent = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name.Sig, T=pikepdf.String("Parent")))
        kid = pdf.make_indirect(pikepdf.Dictionary(
            T=pikepdf.String("Kid"), Parent=parent, V=_zero_sig(pdf), Type=pikepdf.Name.Annot,
            Subtype=pikepdf.Name.Widget, Rect=pikepdf.Array([0, 0, 0, 0]), P=page))
        parent.Kids = pikepdf.Array([kid])
        page.Annots = pikepdf.Array([kid])
        pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([parent]), SigFlags=3)

    return _with(base_pdf(), edit)


def direct_sig_value() -> bytes:
    """A signed field whose signature dictionary is a direct object inside the field."""
    import pikepdf

    def edit(pdf: Any) -> None:
        sig = pikepdf.Dictionary(Type=pikepdf.Name.Sig, ByteRange=pikepdf.Array([0, 0, 0, 0]),
                                 Contents=pikepdf.String(b"\x00" * 16))
        field = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name.Sig, T=pikepdf.String("Direct"), V=sig))
        pdf.Root.AcroForm = pikepdf.Dictionary(Fields=pikepdf.Array([field]))

    return _with(base_pdf(), edit)


# --------------------------------------------------------------------------- pyHanko

@dataclass(frozen=True)
class _Identity:
    cert: Any
    key: Any


def _identity(common_name: str, *, timestamping: bool = False) -> _Identity:
    from asn1crypto import keys as akeys
    from asn1crypto import x509 as ax509
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "example.com")])
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    builder = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
               .serial_number(1).not_valid_before(start).not_valid_after(start + datetime.timedelta(days=3650))
               .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True))
    if timestamping:
        builder = builder.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
    cert = builder.sign(key, hashes.SHA256())
    return _Identity(
        cert=ax509.Certificate.load(cert.public_bytes(serialization.Encoding.DER)),
        key=akeys.PrivateKeyInfo.load(key.private_bytes(serialization.Encoding.DER,
                                                        serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption())))


@functools.lru_cache(maxsize=None)
def _signer() -> Any:
    from pyhanko.sign import signers
    from pyhanko_certvalidator.registry import SimpleCertificateStore

    ident = _identity("Example Test Signer")
    return signers.SimpleSigner(signing_cert=ident.cert, signing_key=ident.key,
                                cert_registry=SimpleCertificateStore.from_certs([ident.cert]))


@functools.lru_cache(maxsize=None)
def _timestamper() -> Any:
    from pyhanko.sign.timestamps.dummy_client import DummyTimeStamper

    ident = _identity("Example Test TSA", timestamping=True)
    return DummyTimeStamper(ident.cert, ident.key,
                            fixed_dt=datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc))




def _sign(data: bytes, *, field: str = "Sig1", box: tuple[int, int, int, int] | None = None,
          certify: bool = False, p: int | None = None) -> bytes:
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import fields, signers
    from pyhanko.sign.fields import MDPPerm

    meta_kwargs: dict[str, Any] = {"field_name": field}
    if certify:
        meta_kwargs.update(certify=True, docmdp_permissions={1: MDPPerm.NO_CHANGES, 2: MDPPerm.FILL_FORMS,
                                                             3: MDPPerm.ANNOTATE}[p or 2])
    writer = IncrementalPdfFileWriter(io.BytesIO(data))
    out = io.BytesIO()
    signers.PdfSigner(signers.PdfSignatureMetadata(**meta_kwargs), signer=_signer(),
                      new_field_spec=fields.SigFieldSpec(field, box=box)).sign_pdf(writer, output=out)
    return out.getvalue()


def signed_visible(field: str = "Sig1", *, object_streams: bool = False) -> bytes:
    return _sign(base_pdf(object_streams=object_streams), field=field, box=(72, 72, 272, 132))


def signed_invisible(field: str = "Sig1", *, object_streams: bool = False) -> bytes:
    return _sign(base_pdf(object_streams=object_streams), field=field)


def certified(p: int, *, object_streams: bool = False) -> bytes:
    return _sign(base_pdf(object_streams=object_streams), certify=True, p=p)


def empty_field() -> bytes:
    """An unsigned signature field with a visible widget: nothing to refuse."""
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import fields

    writer = IncrementalPdfFileWriter(io.BytesIO(base_pdf()))
    fields.append_signature_field(writer, fields.SigFieldSpec("Sig1", box=(72, 72, 272, 132)))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def doc_timestamp() -> bytes:
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import signers

    writer = IncrementalPdfFileWriter(io.BytesIO(base_pdf()))
    out = io.BytesIO()
    signers.PdfTimeStamper(_timestamper()).timestamp_pdf(writer, "sha256", output=out)
    return out.getvalue()


_WIDGET_KEYS = ("/Type", "/Subtype", "/Rect", "/F", "/P", "/AP", "/MK", "/Border")


def signed_no_widget() -> bytes:
    """A signed field listed in /Fields whose widget is gone: no /Subtype /Widget, not in any /Annots."""
    import pikepdf

    data = signed_invisible()
    with pikepdf.open(io.BytesIO(data)) as pdf:
        field = pdf.Root.AcroForm.Fields[0]
        page = pdf.pages[0].obj
        for k in _WIDGET_KEYS:
            if k in field:
                del field[k]
        kept = pikepdf.Array([a for a in page.get("/Annots", pikepdf.Array()) if a.objgen != field.objgen])
        page.Annots = kept
        objects = {field.objgen[0]: field.unparse(resolved=True), page.objgen[0]: page.unparse(resolved=True)}
    return append_update(data, objects)


def signed_widget_not_in_fields() -> bytes:
    """A signed widget still on the page, while an update emptied the form's /Fields."""
    import pikepdf

    data = signed_visible()
    with pikepdf.open(io.BytesIO(data)) as pdf:
        acroform = pdf.Root.AcroForm
        if not acroform.is_indirect:
            raise ValueError("expected an indirect AcroForm")
        acroform.Fields = pikepdf.Array([])
        objects = {acroform.objgen[0]: acroform.unparse(resolved=True)}
    return append_update(data, objects)


def signature_removed_later() -> bytes:
    """Signed, then an incremental update drops /AcroForm and the page's /Annots: only old bytes remain."""
    data = signed_visible()
    import pikepdf

    with pikepdf.open(io.BytesIO(data)) as pdf:
        root, page = pdf.Root, pdf.pages[0].obj
        for k in ("/AcroForm", "/Perms"):
            if k in root:
                del root[k]
        if "/Annots" in page:
            del page["/Annots"]
        objects = {root.objgen[0]: root.unparse(resolved=True), page.objgen[0]: page.unparse(resolved=True)}
    return append_update(data, objects)


# --------------------------------------------------------------------------- incremental updates

_STARTXREF = re.compile(rb"startxref\s*(\d+)")


def _last_startxref(data: bytes) -> tuple[int, int]:
    """(position of the last startxref keyword, the offset it names)."""
    i = data.rfind(b"startxref")
    m = _STARTXREF.match(data, i)
    if i < 0 or m is None:
        raise ValueError("no startxref")
    return i, int(m.group(1))


def append_update(data: bytes, objects: Mapping[int, bytes], *, trailer_extra: bytes = b"") -> bytes:
    """``data`` plus a classic incremental update that (re)defines ``objects`` (number -> object body).

    ``trailer_extra`` is added to the new trailer as written (an encrypted file's ``/Encrypt`` entry).
    """
    import pikepdf

    with pikepdf.open(io.BytesIO(data)) as pdf:
        root = pdf.trailer.Root.objgen[0]
        size = int(pdf.trailer.get("/Size", 0))
        ids = pdf.trailer.get("/ID")
        id_text = ids.unparse() if ids is not None else b""
    _, prev = _last_startxref(data)
    out = bytearray(data)
    if not out.endswith(b"\n"):
        out += b"\n"
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii") + bytes(objects[num]) + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n"
    for num in sorted(objects):
        out += f"{num} 1\n{offsets[num]:010d} 00000 n \n".encode("ascii")
    size = max([size] + [n + 1 for n in objects])
    out += f"trailer\n<< /Size {size} /Root {root} 0 R /Prev {prev}".encode("ascii")
    if id_text:
        out += b" /ID " + id_text
    out += trailer_extra
    out += f" >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    return bytes(out)


def orphan_byterange() -> bytes:
    """A plain PDF whose incremental update adds a signature-like dictionary nothing references."""
    import pikepdf

    data = base_pdf()
    with pikepdf.open(io.BytesIO(data)) as pdf:
        num = int(pdf.trailer.Size)
    return append_update(data, {num: b"<< /ByteRange [0 0 0 0] /Contents <00> >>"})


# --------------------------------------------------------------------------- hostile and edge-case files

ORPHAN_SIG = (b"<< /Type /Sig /Filter /Adobe.PPKLite /SubFilter /adbe.pkcs7.detached "
              b"/ByteRange [0 10 20 30] /Contents <00> >>")


def _free_number(data: bytes) -> int:
    import pikepdf

    with pikepdf.open(io.BytesIO(data)) as pdf:
        return int(pdf.trailer.Size)


def png_predict(data: bytes, *, columns: int, kind: int, bpp: int = 1) -> bytes:
    """``data`` encoded with one PNG predictor type (0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth) on every row."""
    rows = [data[i:i + columns].ljust(columns, b" ") for i in range(0, len(data), columns)]
    out = bytearray()
    prev = bytes(columns)
    for row in rows:
        enc = bytearray()
        for i, x in enumerate(row):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            ul = prev[i - bpp] if i >= bpp else 0
            if kind == 0:
                pred = 0
            elif kind == 1:
                pred = left
            elif kind == 2:
                pred = up
            elif kind == 3:
                pred = (left + up) >> 1
            else:
                p = left + up - ul
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                pred = left if pa <= pb and pa <= pc else up if pb <= pc else ul
            enc.append((x - pred) & 0xFF)
        out.append(kind)
        out += enc
        prev = row
    return bytes(out)


def objstm(objs: Mapping[int, bytes], *, encoding: str = "flate", keyword: bytes = b"stream\n") -> bytes:
    """An object stream body (dictionary, keyword, data, endstream) holding ``objs``.

    ``encoding``: ``flate``, ``predictor`` (Flate with PNG Up rows), ``ahx_flate``
    (``[/ASCIIHexDecode /FlateDecode]``), ``a85`` (``/ASCII85Decode``), ``lzw`` (a filter the raw
    scan doesn't decode) or ``none``.
    """
    import base64
    import zlib

    offsets, pos = [], 0
    for num, body in objs.items():
        offsets.append(f"{num} {pos}")
        pos += len(body) + 1
    header = (" ".join(offsets) + " ").encode("ascii")
    payload = header + b"\n".join(objs.values()) + b"\n"
    parms = ""
    if encoding == "flate":
        data, filt = zlib.compress(payload), "/FlateDecode"
    elif encoding == "predictor":
        data, filt = zlib.compress(png_predict(payload, columns=8, kind=2)), "/FlateDecode"
        parms = " /DecodeParms << /Predictor 12 /Columns 8 >>"
    elif encoding == "ahx_flate":
        data, filt = zlib.compress(payload).hex().encode("ascii") + b">", "[/ASCIIHexDecode /FlateDecode]"
    elif encoding == "a85":
        data, filt = base64.a85encode(payload) + b"~>", "/ASCII85Decode"
    elif encoding == "lzw":
        data, filt = payload, "/LZWDecode"
    else:
        data, filt = payload, ""
    filter_entry = f" /Filter {filt}" if filt else ""
    return (f"<< /Type /ObjStm /N {len(objs)} /First {len(header)}{filter_entry}{parms} /Length {len(data)} >>\n"
            .encode("ascii") + keyword + data + b"\nendstream")


def orphan_in_objstm(encoding: str = "flate", *, keyword: bytes = b"stream\n", pad: int = 0) -> bytes:
    """An unreferenced signature dictionary inside an unreferenced object stream (after ``pad`` bytes of padding)."""
    data = base_pdf()
    num = _free_number(data)
    objs = {num + 2: b"<< /Example (" + b" " * pad + b") >>"} if pad else {}
    objs[num] = ORPHAN_SIG
    return append_update(data, {num + 1: objstm(objs, encoding=encoding, keyword=keyword)})


def orphan_variant(kind: str) -> bytes:
    """An unreferenced signature dictionary the raw scan has to look harder for."""
    data = base_pdf()
    num = _free_number(data)
    if kind == "name_escaped":
        return append_update(data, {num: ORPHAN_SIG.replace(b"/ByteRange", b"/Byte#52ange")})
    if kind == "comment":
        return append_update(data, {num: ORPHAN_SIG.replace(b"/ByteRange [", b"/ByteRange % example\n[")})
    if kind == "indirect":
        return append_update(data, {num: ORPHAN_SIG.replace(b"[0 10 20 30]", f"{num + 1} 0 R".encode()),
                                    num + 1: b"[0 10 20 30]"})
    if kind == "fake_stream_in_string":
        return append_update(data, {num + 1: b"<< /Title (>>stream\n) >>", num: ORPHAN_SIG,
                                    num + 2: b"<< /Example (endstream) >>"})
    if kind == "paren_in_comment":
        return append_update(data, {num + 1: b"<< /Example 1 >> % an example ( in a comment", num: ORPHAN_SIG,
                                    num + 2: b"<< /Example (a closing ) paren) >>"})
    if kind == "unclosed_string":
        return append_update(data, {num + 1: b"<< /Example (never closed >>", num: ORPHAN_SIG,
                                    num + 2: b"<< /Example 2 >> )"})
    if kind == "broken_endstream":
        out = append_update(data, {num: ORPHAN_SIG})
        k = out.rfind(b"endstream", 0, out.rfind(b"/ByteRange"))
        return out[:k] + b"endstreaX" + out[k + 9:]
    raise ValueError(kind)


def break_endstream_before(data: bytes, needle: bytes) -> bytes:
    """The last ``endstream`` keyword before the last ``needle`` damaged into ``endstreaX``."""
    k = data.rfind(b"endstream", 0, data.rfind(needle))
    if k < 0:
        raise ValueError("no endstream before the needle")
    return data[:k] + b"endstreaX" + data[k + 9:]


def owner_encrypted_plus_orphan() -> bytes:
    """An owner-password file whose incremental update adds an unreferenced signature dictionary."""
    data = encrypted_owner()
    m = re.search(rb"/Encrypt\s+(\d+)\s+(\d+)\s+R", data)
    if m is None:
        raise ValueError("expected an indirect /Encrypt")
    num = _free_number(data)
    return append_update(data, {num: ORPHAN_SIG}, trailer_extra=b" /Encrypt " + m.group(0)[len(b"/Encrypt"):].strip())


def unreferenced_encrypt_dict() -> bytes:
    """A plain file with an unreferenced object that looks like a standard security handler."""
    data = base_pdf()
    return append_update(data, {_free_number(data): b"<< /Filter /Standard /V 4 /R 4 /Length 128 >>"})


def with_title(title: str) -> bytes:
    """The base PDF whose Info title is ``title`` (written as a literal string)."""
    import pikepdf

    def edit(pdf: Any) -> None:
        pdf.docinfo["/Title"] = pikepdf.String(title)

    return _with(base_pdf(), edit)


def object_xmp(where: str, xmp: str = XMP_PDFA_2B, *, compress_streams: bool = False) -> bytes:
    """XMP with a PDF/A claim on a page (``page``), an image XObject (``image``), or nothing (``unreferenced``)."""
    import pikepdf

    data = base_pdf()
    if where == "unreferenced":
        raw = xmp.encode("utf-8")
        body = (f"<< /Type /Metadata /Subtype /XML /Length {len(raw)} >>\nstream\n".encode("ascii")
                + raw + b"\nendstream")
        return append_update(data, {_free_number(data): body})

    def edit(pdf: Any) -> None:
        meta = pdf.make_stream(xmp.encode("utf-8"), Type=pikepdf.Name.Metadata, Subtype=pikepdf.Name.XML)
        if where == "page":
            pdf.pages[0].obj.Metadata = meta
            return
        img = pdf.make_stream(b"\x00" * 4, Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image, Width=2,
                              Height=2, ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8)
        img.Metadata = meta
        pdf.pages[0].obj.Resources.XObject = pikepdf.Dictionary(Im9=img)

    return _with(data, edit, compress_streams=compress_streams)


def big_pdf(megabytes: int) -> bytes:
    """A valid PDF carrying ``megabytes`` of zero bytes in an unreferenced-free page image (for the truncation test)."""
    import pikepdf

    def edit(pdf: Any) -> None:
        side = 1024
        rows = max(1, megabytes * 1024 * 1024 // side)
        img = pdf.make_stream(b"\x00" * (side * rows), Type=pikepdf.Name.XObject, Subtype=pikepdf.Name.Image,
                              Width=side, Height=rows, ColorSpace=pikepdf.Name.DeviceGray, BitsPerComponent=8)
        pdf.pages[0].obj.Resources.XObject = pikepdf.Dictionary(Im9=img)

    return _with(base_pdf(), edit, compress_streams=False)


# --------------------------------------------------------------------------- damage

def break_startxref(data: bytes) -> bytes:
    """The last startxref names a wrong offset (3 bytes past the real one)."""
    i, off = _last_startxref(data)
    return data[:i] + b"startxref\n" + str(off + 3).encode("ascii") + b"\n%%EOF\n"


def cut_trailer(data: bytes) -> bytes:
    """Everything from the last trailer on is gone (for an xref stream: from its object on)."""
    i = data.rfind(b"trailer")
    _, off = _last_startxref(data)
    if i < 0 or i < off:                         # the last section is an xref stream
        i = off
    return data[:i]


DAMAGES: dict[str, Callable[[bytes], bytes]] = {
    "intact": lambda d: d,
    "startxref": break_startxref,
    "no_trailer": cut_trailer,
}


# --------------------------------------------------------------------------- the catalogue

@dataclass(frozen=True)
class Shell:
    build: Callable[[], bytes]
    code: str | None              # the code the file carries (first of _codes(..., adds_layer=True))
    code_no_layer: str | None     # the same with adds_layer=False
    pyhanko: bool = False
    # Whether ``code`` refuses (a HARD code) or only puts the file in copy mode (COPYABLE) is
    # refusals' own split; tests check each shell against both refusal() and copy_reasons().


SHELLS: dict[str, Shell] = {
    "signed_visible": Shell(signed_visible, "signed", "signed", pyhanko=True),
    "signed_invisible": Shell(signed_invisible, "signed", "signed", pyhanko=True),
    "certified_p1": Shell(lambda: certified(1), "certified", "certified", pyhanko=True),
    "certified_p2": Shell(lambda: certified(2), "certified", "certified", pyhanko=True),
    "empty_field": Shell(empty_field, None, None, pyhanko=True),
    "signed_no_widget": Shell(signed_no_widget, "signed", "signed", pyhanko=True),
    "doc_timestamp": Shell(doc_timestamp, "timestamped", "timestamped", pyhanko=True),
    "ur3": Shell(ur3, "usage_rights", "usage_rights"),
    "pdfa_2b": Shell(pdfa, "pdfa", "pdfa"),
    "pdfua": Shell(pdfua, "pdfua_needs_layer", None),
    "encrypted_owner": Shell(encrypted_owner, "encrypted", "encrypted"),
    "encrypted_user": Shell(encrypted_user, "encrypted", "encrypted"),
    "xfa": Shell(xfa, "xfa", "xfa"),
    "xmp_unreadable": Shell(xmp_unreadable, "xmp_unreadable", "xmp_unreadable"),
}
