"""Refusals: the PDFs pdfintake leaves exactly as they are, decided before any work.

Two answers come out of one report (DESIGN §10.1 and §11.1, decided 2026-11-05):

* ``refusal`` names a **hard** code (``HARD``: ``xfa``, ``xmp_unreadable``,
  ``detectors_disagree``, ``damaged``). Such a file is refused: nothing is written (exit 4).
* ``copy_reasons`` names the **copyable** codes (``COPYABLE``: provenance, signatures,
  certification, usage rights, timestamps, PDF/A, PDF/UA that needs a layer, encryption). Such a
  file is never changed either, but pdfintake writes a separate searchable copy beside it.

A file can carry both: a hard code wins, and the caller refuses.

Two independent detectors read every input PDF:

* **The object walk** opens the file with pikepdf (qpdf repairs a damaged file while opening
  it) and follows every object reachable from the trailer. It reads the AcroForm field tree
  (``/Fields`` and ``/Kids`` with inherited ``/FT`` and ``/V``), every page widget, every
  dictionary with a ``/ByteRange``, the catalog's ``/Perms`` (``/DocMDP`` certification with its
  P value, ``/UR3`` usage rights), document timestamps (``/DocTimeStamp``), encryption, the
  AcroForm ``/XFA`` entry and the catalog's XMP (``pdfaid:part``, ``pdfuaid:part``).
  It also reads every other reachable XMP stream (a page's, an image's), whose PDF/A or PDF/UA
  claim is not the document's: it only lets the two detectors agree about it.
* **The raw byte scan** never parses the object graph. It lexes the file's bytes (string literals
  and comments honoured), looks for marker tokens outside stream data, and inside the bodies of
  XMP metadata and object streams (decoded through their filters under size caps), so it also
  sees objects the walk can't reach: an unreferenced object, a revision an incremental update
  replaced, a section qpdf skipped. What it can't read, it reports (``unscanned``), and a file
  with no trailer at all is reported too (``no_trailer``): either one is a disagreement.

Either detector's finding counts, and when the two don't name the same things the file is
refused with ``detectors_disagree``. Every message is written here; nothing taken from the
document (field names, XMP values beyond a checked pattern) reaches a message or a label in the
report.

Damaged files (decided 2026-11-06). A file that fails structure checks before any work is refused
with its own hard code, ``damaged``, so its record says what happened rather than that the
detectors disagree: pikepdf can't open it at all (``open_failed``, not a password), the raw scan
finds neither a trailer nor an xref stream (``no_trailer``: the file was cut short or its end is
gone, and qpdf could only rebuild it blind), or the walk stopped part way through a file qpdf had
to repair (``walk_failed`` with ``repaired``). A walk that stops on an intact file (the object
cap, say) is still ``detectors_disagree``, and so is anything the two detectors name differently
on a file that is also damaged: both codes are listed, ``damaged`` first, so ``refusal`` names it.
A file qpdf repairs from a broken cross-reference table (a bad ``startxref``, a missing xref table)
and then reads in full is not damaged: it is processed with the warning ``input_repaired``. A file
with no trailer and no xref stream is damaged even when qpdf rebuilds it and the walk reads it all.

Report vocabulary.
``walk_codes`` uses the codes the walk can find (``signed``, ``certified``,
``usage_rights``, ``timestamped``, ``pdfa``, ``encrypted``, ``xfa``, ``xmp_unreadable``) plus
``pdfua`` (a PDF/UA claim, a copy reason only when a layer would be added), two states,
``open_failed`` (pikepdf could not open the file at all) and ``walk_failed`` (the walk stopped
part way), and ``object_pdfa`` / ``object_pdfua`` (a claim in a page's or an image's own XMP,
which never counts). ``raw_markers`` uses the strings in ``RAW_MARKERS``.

When the walk can't open the file, or needs a password it doesn't have, what the raw scan saw
stands on its own; a file pikepdf can't open at all is refused as ``damaged``.

Passwords. This module never takes one (decided 2026-11-05). A file that needs an open password
reads as ``encrypted="user"``, a copy reason, and its report is provisional: the walk saw nothing
inside it, and the raw scan can't read its encrypted streams (an XFA form or an XMP claim inside an
object stream stays hidden). The flow decrypts the original once into an unencrypted work copy and
reads that copy again; a hard code in either report refuses the file.

Encrypted inputs and ``unscanned`` (decided 2026-11-05). The raw scan can't decode an encrypted
object stream, so every encrypted file with object streams reports ``unscanned``. On an encrypted
input that marker is not a disagreement: ``encrypted`` always means copy mode, the original is never
written, the copy is saved from the current revision only and carries no signature material
(``output_signature_material`` checks it), so a signature hidden in an earlier revision's encrypted
object stream changes nothing. What does change the outcome is still caught: the walk reads the
current revision of a permissions-only file, and the decrypted work copy of a password-protected one.
On an unencrypted input ``unscanned`` still counts as ``detectors_disagree``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pikepdf

from .errors import UsageError

CODES = ("signed", "certified", "usage_rights", "timestamped", "pdfa", "pdfua_needs_layer", "encrypted",
         "xfa", "xmp_unreadable", "detectors_disagree", "damaged", "provenance_counterparty", "provenance_evidence")

# §11.1: what gets a searchable copy, and what is still refused. Together they are CODES.
COPYABLE = frozenset({"provenance_counterparty", "provenance_evidence", "signed", "certified", "usage_rights",
                      "timestamped", "pdfa", "pdfua_needs_layer", "encrypted"})
# "damaged" (decided 2026-11-06): a file that fails the structure checks gets its own code, so the
# record no longer calls a broken file a disagreement between the detectors.
HARD = frozenset({"xfa", "xmp_unreadable", "detectors_disagree", "damaged"})

PROVENANCE_VALUES = ("ordinary", "counterparty", "evidence")

# What the raw scan can report, in report order.
RAW_MARKERS = ("/ByteRange [", "/DocMDP", "/UR3", "/DocTimeStamp", "/XFA", "/Encrypt", "pdfaid:part", "pdfuaid:part",
               "unscanned", "no_trailer")

# Walk-only states (not refusal codes).
WALK_OPEN_FAILED = "open_failed"
WALK_FAILED = "walk_failed"
WALK_PDFUA = "pdfua"
WALK_OBJECT_PDFA = "object_pdfa"       # a PDF/A claim in a page's or an image's XMP (never refuses)
WALK_OBJECT_PDFUA = "object_pdfua"

# Caps that keep a hostile file from costing unbounded time or memory.
MAX_WALK_OBJECTS = 5_000_000          # reachable objects visited before the walk gives up (walk_failed)
MAX_INFLATE_STREAM = 64 * 1024 * 1024  # bytes inflated from one metadata or object stream
MAX_INFLATE_TOTAL = 256 * 1024 * 1024  # bytes inflated in one raw scan
MAX_XMP_BYTES = 32 * 1024 * 1024       # decoded XMP larger than this counts as unreadable
_DICT_LOOKBACK = 8192                  # bytes searched back from a stream keyword for its dictionary


@dataclass
class RefusalReport:
    signed_fields: list[str]; empty_sig_fields: int; sig_dicts: int; doc_timestamps: int
    certified_p: int | None; usage_rights: bool; pdfa: str | None; pdfua: str | None
    encrypted: Literal[False, "owner", "user"]; xfa: bool; jbig2_streams: int; incremental_updates: int
    xmp_readable: bool
    walk_codes: list[str]                 # what the pikepdf object walk found
    raw_markers: list[str]                # what the raw byte scan found (RAW_MARKERS)
    repaired: bool; qpdf_warnings: int    # qpdf repaired a damaged file while opening it


# --------------------------------------------------------------------------- messages (core-worded)

# Hard codes: nothing is written. Copyable codes: the original is untouched and a copy goes beside it.
_KEPT = "pdfintake leaves it exactly as it is and makes no copy."
_COPY = "pdfintake leaves the original exactly as it is and writes a separate searchable copy beside it."

_MESSAGES: dict[str, str] = {
    "signed": "This PDF is digitally signed. Any change to the file would break the signature, so " + _COPY,
    "certified": ("This PDF is certified: its author signed it with a certification signature that "
                  "limits what may change. Any change would break it, so " + _COPY),
    "usage_rights": ("This PDF carries usage rights (a UR3 signature that unlocks features in Adobe "
                     "Reader). Any change would invalidate them, so " + _COPY),
    "timestamped": ("This PDF carries a document timestamp, a signature that proves when the file "
                    "existed. Any change would break it, so " + _COPY),
    "pdfa": ("This PDF is an archival PDF/A file{detail}. Any change to the file would break its "
             "conformance, so " + _COPY),
    "pdfua_needs_layer": ("This PDF is an accessible PDF/UA file{detail}, and its pages would need a text "
                          "layer. An added layer would not be tagged and would break that conformance, so "
                          + _COPY),
    "encrypted": ("This PDF is encrypted{detail}, so " + _COPY[:-1]
                  + ", saved without a password so that it can be searched."),
    "xfa": ("This PDF contains an XFA form, a dynamic form whose pages the form itself can "
            "generate. Its content can't be checked safely, so " + _KEPT),
    "xmp_unreadable": ("This PDF's XMP metadata can't be read, so pdfintake can't tell whether it is an "
                       "archival or otherwise protected file. To be safe, " + _KEPT),
    "detectors_disagree": ("pdfintake's two checks for signatures, certification, encryption and archival "
                           "markers don't agree about this PDF, or it is too damaged for them to be sure "
                           "(for example, signature data left in an earlier version of the file). To be "
                           "safe, " + _KEPT),
    "damaged": ("This PDF is damaged: it can't be opened, it was cut short or lost its end, or it couldn't "
                "be read through after a repair, so pdfintake can't read it reliably or check it safely. "
                "To be safe, " + _KEPT),
    "provenance_counterparty": ("You said this file came from the other side of a contract, deal or "
                                "dispute. pdfintake leaves the original exactly as received and writes a "
                                "separate searchable copy beside it."),
    "provenance_evidence": ("You said this file is an original kept as evidence. pdfintake leaves the "
                            "original exactly as it is and writes a separate searchable copy beside it."),
}

_PDFA_RE = re.compile(r"^[0-9]{1,2}[A-Z]{0,2}$")
_PDFUA_RE = re.compile(r"^[0-9]{1,2}$")


def _message(code: str, report: RefusalReport | None = None) -> str:
    text = _MESSAGES[code]
    if "{detail}" not in text:
        return text
    detail = ""
    if report is not None:
        if code == "pdfa" and report.pdfa and _PDFA_RE.match(report.pdfa):
            detail = f" (PDF/A-{report.pdfa})"
        elif code == "pdfua_needs_layer" and report.pdfua and _PDFUA_RE.match(report.pdfua):
            detail = f" (PDF/UA-{report.pdfua})"
        elif code == "encrypted":
            detail = {"user": " and needs a password to open",
                      "owner": " with an owner password that restricts what may be done with it"}.get(
                report.encrypted or "", "")
    return text.format(detail=detail)


# --------------------------------------------------------------------------- the decision

_WALK_ORDER = ("certified", "signed", "timestamped", "usage_rights", "xfa", "pdfa")
_MARKER_FAMILY = {"/ByteRange [": "signature", "/DocMDP": "certified", "/UR3": "usage_rights",
                  "/DocTimeStamp": "timestamped", "/XFA": "xfa", "pdfaid:part": "pdfa",
                  "pdfuaid:part": "pdfua", "/Encrypt": "encrypted"}
_FAMILY_CODE = {"signature": "signed", "certified": "certified", "usage_rights": "usage_rights",
                "timestamped": "timestamped", "xfa": "xfa", "pdfa": "pdfa"}


def _walk_families(report: RefusalReport) -> set[str]:
    fam: set[str] = set()
    walk = set(report.walk_codes)
    if report.sig_dicts > 0:
        fam.add("signature")
    if report.certified_p is not None:
        fam.add("certified")
    if report.usage_rights:
        fam.add("usage_rights")
    if report.doc_timestamps > 0:
        fam.add("timestamped")
    if report.xfa:
        fam.add("xfa")
    if report.pdfa is not None or WALK_OBJECT_PDFA in walk:
        fam.add("pdfa")
    if report.pdfua is not None or WALK_OBJECT_PDFUA in walk:
        fam.add("pdfua")
    return fam


def _raw_families(report: RefusalReport) -> set[str]:
    return {_MARKER_FAMILY[m] for m in report.raw_markers if m in _MARKER_FAMILY} - {"encrypted"}


def _codes(report: RefusalReport, *, adds_layer: bool) -> list[str]:
    """Every code that applies, most important first (empty: nothing to say).

    ``refusal`` and ``copy_reasons`` split this list into hard and copyable codes, so every code
    that applies is listed: ``damaged`` and then ``detectors_disagree`` are appended whenever they
    apply, even after another code.
    """
    walk = set(report.walk_codes)
    raw_m = set(report.raw_markers)
    open_failed = WALK_OPEN_FAILED in walk
    raw_encrypt = "/Encrypt" in raw_m
    # A file that fails the structure checks is ``damaged`` (module notes, decided 2026-11-06).
    damaged = open_failed or NO_TRAILER in raw_m or (WALK_FAILED in walk and report.repaired)
    out: list[str] = []
    # Encryption comes from the walk, or from the raw scan when the walk couldn't open the file or
    # rebuilt it (qpdf loses /Encrypt with a cut trailer).
    if report.encrypted or "encrypted" in walk or (raw_encrypt and (open_failed or report.repaired)):
        out.append("encrypted")
    raw = _raw_families(report)
    # On an encrypted input ``unscanned`` is never a disagreement (module notes, decided 2026-11-05):
    # the user branch returns before it is looked at, and the owner branch below doesn't test it.
    if open_failed or report.encrypted == "user":
        # The walk saw nothing; what the raw scan saw stands on its own.
        # A byte range alone is a signature; next to /DocMDP, /DocTimeStamp or /UR3 it may be theirs.
        out += [_FAMILY_CODE[f] for f in ("certified", "timestamped", "usage_rights", "signature", "xfa", "pdfa")
                if f in raw and _FAMILY_CODE[f] not in out]
        if adds_layer and "pdfua" in raw:
            out.append("pdfua_needs_layer")
        if damaged:
            out.append("damaged")
        return out
    out += [c for c in _WALK_ORDER if c in walk and c not in out]
    if adds_layer and WALK_PDFUA in walk:
        out.append("pdfua_needs_layer")
    if "xmp_unreadable" in walk or not report.xmp_readable:
        out.append("xmp_unreadable")
    if damaged:
        out.append("damaged")
    # The walk stopping on an intact file still means the two detectors can't agree; on a damaged
    # one the damage says why, and only a real difference between them adds detectors_disagree. A
    # damaged file whose raw /Encrypt the walk lost has already been read as encrypted (above).
    lost_encryption = raw_encrypt and not report.encrypted and not (damaged and "encrypted" in out)
    disagree = (WALK_FAILED in walk and not damaged) or lost_encryption
    if report.encrypted == "owner":
        # The raw scan can't read inside encrypted streams: whatever it saw, the walk must have seen.
        disagree = disagree or not raw <= _walk_families(report)
    else:
        disagree = disagree or UNSCANNED in raw_m or raw != _walk_families(report)
    if disagree:
        out.append("detectors_disagree")
    return out


def refusal(report: RefusalReport, *, adds_layer: bool = False) -> tuple[str, str] | None:
    """(hard code, core-worded message) when the file must be refused, else None.

    Only ``HARD`` codes refuse (decided 2026-11-05): ``xfa``; ``xmp_unreadable`` when XMP
    exists but won't parse; ``damaged`` when the file fails the structure checks (it can't be
    opened, has no trailer left, or a repaired file couldn't be walked; decided 2026-11-06);
    ``detectors_disagree`` when the walk and the raw scan don't name the same things. A damaged
    file is named ``damaged`` even when the detectors also differ. Everything else a file carries is a copy
    reason (``copy_reasons``), not a refusal. ``adds_layer`` is accepted for the earlier call
    shape and changes nothing: no hard code depends on it.

    For a report with ``encrypted="user"`` the answer is provisional (see the module notes):
    read the decrypted work copy before going on.
    """
    hard = [c for c in _codes(report, adds_layer=adds_layer) if c in HARD]
    if not hard:
        return None
    return hard[0], _message(hard[0], report)


def copy_reasons(report: RefusalReport, provenance: str, *, adds_layer: bool) -> list[str]:
    """The ``COPYABLE`` codes that apply, in ``CODES`` order (empty: an ordinary output).

    Any reason means copy mode: the original is never written, and a separate searchable copy
    goes beside it. ``provenance`` is checked as ``check_provenance`` checks it (UsageError for a
    value outside ``PROVENANCE_VALUES``). ``adds_layer`` is required (decided 2026-11-05): it is
    the page classification's answer, and ``pdfua_needs_layer`` applies only when it is True (a
    PDF/UA file that needs no layer is shrunk only, §2.2), so a caller can't leave it out and get
    the wrong mode. Hard codes are never listed here: ask ``refusal`` first, since a hard code
    refuses whatever the copy reasons say. ``copy_message`` words each reason.
    """
    found = set(_codes(report, adds_layer=adds_layer)) & COPYABLE
    given = check_provenance(provenance)
    if given is not None:
        found.add(given[0])
    return [c for c in CODES if c in found]


def copy_message(code: str, report: RefusalReport | None = None) -> str:
    """The core-worded sentence for one copy reason (a code from ``COPYABLE``), for reporting copy mode.

    ``report`` fills in the PDF/A part, the PDF/UA part and the kind of encryption when given.
    ValueError for a code outside ``COPYABLE``: hard codes are worded by ``refusal``.
    """
    if code not in COPYABLE:
        raise ValueError(f"not a copy reason: {code!r}")
    return _message(code, report)


def check_provenance(value: str) -> tuple[str, str] | None:
    """None for ``ordinary``; (copy reason, message) for ``counterparty`` and ``evidence``; UsageError otherwise.

    Since §10 (decided 2026-11-05) neither provenance refuses: both put the file in copy mode.
    """
    if value == "ordinary":
        return None
    if value == "counterparty":
        return "provenance_counterparty", _MESSAGES["provenance_counterparty"]
    if value == "evidence":
        return "provenance_evidence", _MESSAGES["provenance_evidence"]
    raise UsageError("bad_provenance", "--provenance must be one of: ordinary, counterparty, evidence.",
                     allowed=list(PROVENANCE_VALUES))


# --------------------------------------------------------------------------- the raw byte scan
#
# The scan reads the file with ordinary buffered reads (never a memory map: a file another process
# truncates while it is mapped kills the interpreter with SIGBUS), then lexes it once from the start:
#
# * string literals ``( ... )`` outside streams are dropped, so a title or field value that mentions
#   a marker can't trip it (a string counts only when it closes before the next ``endobj``;
#   otherwise its bytes are scanned as they are);
# * comments are honoured, so a ``(`` in a comment never opens a string;
# * a stream body ends where ``/Length`` says when ``endstream`` sits there, and otherwise at the
#   earliest of the next ``endstream``, the next line-start ``N G obj``, ``xref`` or ``trailer``;
#   with no terminator at all, the rest of the file is scanned as plain bytes;
# * object streams and XMP streams are decoded through their filter chain (Flate with PNG or TIFF
#   predictors, ASCIIHex, ASCII85) under byte caps; one that can't be decoded, or that a cap cut
#   short, adds the marker ``unscanned``, which counts as disagreement (the scan fails closed);
# * ``#xx`` escapes in names are decoded before matching, and a comment or an indirect array may
#   sit after ``/ByteRange``.

_W = rb"[\x00\t\n\x0c\r ]"
_END = rb"(?=[\x00\t\n\x0c\r /<>\[\]()%{}]|\Z)"
_GAP = rb"(?>(?:" + _W + rb"|%[^\r\n]*+[\r\n])*+)"           # whitespace and comments, no backtracking
_REF = rb"[0-9]{1,10}" + _W + rb"+[0-9]{1,5}" + _W + rb"+R" + _END
_RAW_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("/ByteRange [", re.compile(rb"/ByteRange" + _GAP + rb"(?:\[|" + _REF + rb")")),
    ("/DocMDP", re.compile(rb"/DocMDP" + _END)),
    ("/UR3", re.compile(rb"/UR3" + _END)),
    ("/DocTimeStamp", re.compile(rb"/DocTimeStamp" + _END)),
    ("/XFA", re.compile(rb"/XFA" + _END)),
    ("/Encrypt", re.compile(rb"/Encrypt" + _GAP + rb"(?:" + _REF + rb"|<<)|/Filter" + _GAP
                            + rb"/(?:Standard|Adobe\.PubSec)" + _END + rb"|/CFM" + _GAP + rb"/(?:AESV2|AESV3|V2)" + _END)),
    ("pdfaid:part", re.compile(rb"pdfaid:part")),
    ("pdfaid:part", re.compile(re.escape("pdfaid:part".encode("utf-16-le")))),
    ("pdfaid:part", re.compile(re.escape("pdfaid:part".encode("utf-16-be")))),
    ("pdfuaid:part", re.compile(rb"pdfuaid:part")),
    ("pdfuaid:part", re.compile(re.escape("pdfuaid:part".encode("utf-16-le")))),
    ("pdfuaid:part", re.compile(re.escape("pdfuaid:part".encode("utf-16-be")))),
)
# The PDF/A and PDF/UA identification namespaces may be bound to any prefix (decided 2026-11-05): the
# walk resolves the namespace, so the raw scan finds each prefix bound to it and looks for
# ``<prefix>:part`` too, reported as the usual marker. A declaration sits right before its URI, so only
# a short stretch before each URI is read, for at most _NS_HITS URIs per buffer (past that, a missed
# prefix just means the detectors disagree, which refuses).
_NS_URIS = (("pdfaid:part", "http://www.aiim.org/pdfa/ns/id"), ("pdfuaid:part", "http://www.aiim.org/pdfua/ns/id"))
_NCNAME = r"[A-Za-z_][A-Za-z0-9._-]{0,63}"
_NS_DECL = {marker: re.compile(r"xmlns:(" + _NCNAME + r")[\t\n\r ]*=[\t\n\r ]*[\"']" + re.escape(uri) + r"/?[\"']\Z")
            for marker, uri in _NS_URIS}
_NS_HITS = 64
_NS_LOOKBACK = 96                      # characters before a URI: "xmlns:", a 64-character prefix, "=", quote


def _ns_prefix_parts(buf: bytes) -> set[str]:
    """Markers for ``<prefix>:part`` where the buffer binds ``prefix`` to a PDF/A or PDF/UA id namespace.

    UTF-8 and both UTF-16 byte orders, as the literal patterns are matched.
    """
    found: set[str] = set()
    for marker, uri in _NS_URIS:
        for enc, width in (("utf-8", 1), ("utf-16-le", 2), ("utf-16-be", 2)):
            needle = uri.encode(enc)
            prefixes: set[str] = set()
            start = hits = 0
            while hits < _NS_HITS:
                at = buf.find(needle, start)
                if at < 0:
                    break
                hits += 1
                start = at + len(needle)
                lo = at - width * _NS_LOOKBACK
                lo = lo if lo >= 0 else at % width        # keep the code-unit alignment of the URI
                hi = start + width * 2                    # room for an optional "/" and the closing quote
                text = buf[lo:hi].decode(enc, errors="replace")
                for cut in (len(text), len(text) - 1):    # the URI ends with or without its "/"
                    m = _NS_DECL[marker].search(text[:cut])
                    if m:
                        prefixes.add(m.group(1))
                        break
            if any((p + ":part").encode(enc) in buf for p in prefixes):
                found.add(marker)
                break
    return found


_WS = b"\x00\t\n\x0c\r "
_LEX =re.compile(rb"[(%]|(?<![A-Za-z0-9#])stream(?![A-Za-z0-9#])")
_STRCH = re.compile(rb"[()\\]")
_EOL = re.compile(rb"[\r\n]")
_NAME_ESC = re.compile(rb"#([0-9A-Fa-f]{2})")
_TYPE_OBJSTM = re.compile(rb"/Type" + _GAP + rb"/ObjStm" + _END)
_TYPE_META = re.compile(rb"/Type" + _GAP + rb"/Metadata" + _END + rb"|/Subtype" + _GAP + rb"/XML" + _END)
_TYPE_XREF = re.compile(rb"/Type" + _GAP + rb"/XRef" + _END)
_TRAILER = re.compile(rb"(?<![A-Za-z0-9/#])trailer" + _END)
# An object head can't backtrack: a run of digits is tried once, from its first digit.
_OBJ_HEAD = re.compile(rb"(?<![0-9])[0-9]{1,10}" + _W + rb"+[0-9]{1,5}" + _W + rb"+obj" + _END)
_OBJ_LINE = re.compile(rb"[\r\n][\x00\t\x0c ]*+[0-9]{1,10}" + _W + rb"+[0-9]{1,5}" + _W + rb"+obj" + _END)
_XREF_LINE = re.compile(rb"[\r\n][\x00\t\x0c ]*+xref" + _END)
_INT_OBJ = re.compile(rb"(?<![0-9])([0-9]{1,10})" + _W + rb"+([0-9]{1,5})" + _W + rb"+obj" + _W
                      + rb"*+([0-9]{1,15})" + _W + rb"*+endobj")
_STARTXREF = re.compile(rb"startxref" + _W + rb"*([0-9]{1,15})")
_PREV = re.compile(rb"/Prev" + _W + rb"+([0-9]{1,15})")

MAX_RAW_BYTES = 1024 * 1024 * 1024    # a larger file isn't scanned at all: "unscanned"
MAX_PREDICTED = 8 * 1024 * 1024        # bytes run through a PNG or TIFF predictor in one stream
_LENGTH_SLACK = 16                     # whitespace allowed between a /Length end and endstream

UNSCANNED = "unscanned"                # part of the file the raw scan couldn't read
NO_TRAILER = "no_trailer"              # neither a trailer nor an xref stream: qpdf rebuilt the file blind


@dataclass
class _Raw:
    markers: list[str]
    incremental_updates: int
    unfinished_update: bool      # object definitions after the last startxref (a revision qpdf ignores)


def _read_file(path: Path) -> bytes | None:
    """The file's bytes by buffered reads, or None when it is larger than MAX_RAW_BYTES.

    A file that shrinks while it is read just gives fewer bytes; the input seal reports the change.
    """
    with open(path, "rb") as fh:
        data = fh.read(MAX_RAW_BYTES + 1)
    return None if len(data) > MAX_RAW_BYTES else data


class _Next:
    """First occurrence of a needle at or after a position, cached for scans that only move forward."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self._cache: dict[Any, tuple[int, int]] = {}

    def __call__(self, needle: bytes | re.Pattern[bytes], start: int) -> int:
        hit = self._cache.get(needle)
        if hit is not None and hit[0] <= start and (hit[1] < 0 or hit[1] >= start):
            return hit[1]
        if isinstance(needle, bytes):
            pos = self.data.find(needle, start)
        else:
            m = needle.search(self.data, start)
            pos = m.start() if m else -1
        self._cache[needle] = (start, pos)
        return pos


def _string_close(data: bytes, start: int, limit: int) -> int | None:
    """End of the balanced string literal opening at ``start``, or None when it doesn't close before ``limit``."""
    depth, p = 1, start + 1
    while True:
        m = _STRCH.search(data, p, limit)
        if m is None:
            return None
        c = data[m.start()]
        if c == 0x5C:                     # backslash: the next byte is escaped
            p = m.start() + 2
            continue
        depth += 1 if c == 0x28 else -1
        if depth == 0:
            return m.end()
        p = m.end()


def _lex(data: bytes, on_stream: Any = None) -> bytes:
    """``data`` without its string literals (each becomes ``()``) and, with ``on_stream``, its stream bodies.

    ``on_stream(pos)`` is called at each ``stream`` keyword and returns ``(body_start, body_end)``
    for a stream it took, or None for a keyword that doesn't start one.
    """
    nxt = _Next(data)
    n = len(data)
    parts: list[bytes] = []
    pos = span = 0
    strings_off_until = -1                # after a string that doesn't close, scan its bytes as they are
    while pos < n:
        m = _LEX.search(data, pos)
        if m is None:
            break
        t = m.start()
        c = data[t]
        if c == 0x25:                     # a comment, to the end of its line
            e = _EOL.search(data, t)
            pos = e.end() if e else n
            continue
        if c == 0x28:
            if t < strings_off_until:
                pos = t + 1
                continue
            limit = nxt(b"endobj", t)
            limit = n if limit < 0 else limit
            close = _string_close(data, t, limit)
            if close is None:
                strings_off_until = limit
                pos = t + 1
                continue
            parts += (data[span:t], b"()")
            span = pos = close
            continue
        got = on_stream(t) if on_stream is not None else None
        if got is None:
            pos = m.end()
            continue
        body, end = got
        parts += (data[span:body], b"\n")
        span = pos = max(end, body)
    parts.append(data[span:n])
    return b"".join(parts)


def _unescape(buf: bytes) -> bytes:
    return _NAME_ESC.sub(lambda m: bytes((int(m.group(1), 16),)), buf) if b"#" in buf else buf


def _depth1(text: bytes, pattern: re.Pattern[bytes]) -> re.Match[bytes] | None:
    """The first match of ``pattern`` at the dictionary's top level (one ``<<`` deep)."""
    for m in pattern.finditer(text):
        before = text[:m.start()]
        if before.count(b"<<") - before.count(b">>") == 1:
            return m
    return None


_KEY_LENGTH = re.compile(rb"/Length" + _W + rb"*+([0-9]{1,15})(?:" + _W + rb"+([0-9]{1,5})" + _W + rb"+R" + _END + rb")?")
_KEY_FILTER = re.compile(rb"/Filter" + _W + rb"*+(\[[^\]]*\]|/[^\x00\t\n\x0c\r /<>\[\]()%{}]*|[0-9]{1,10}" + _W + rb"+[0-9])")
_KEY_PARMS = re.compile(rb"/DecodeParms" + _W + rb"*+(<<.*?>>|\[.*?\]|null|[0-9]{1,10}" + _W + rb"+[0-9])", re.S)
_KEY_INT = {k: re.compile(rb"/" + k.encode() + _W + rb"*+([0-9]{1,10})" + _END)
            for k in ("Predictor", "Colors", "BitsPerComponent", "Columns", "N", "First")}
_FILTER_KIND = {b"/FlateDecode": "flate", b"/Fl": "flate", b"/ASCIIHexDecode": "ahx", b"/AHx": "ahx",
                b"/ASCII85Decode": "a85", b"/A85": "a85"}


def _int_key(text: bytes, key: str, default: int | None = None) -> int | None:
    m = _depth1(text, _KEY_INT[key])
    return int(m.group(1)) if m else default


def _filters(dtext: bytes) -> tuple[list[str], list[bytes | None]] | None:
    """(filter kinds, decode parms text per filter) from a stream dictionary; None when they can't be read."""
    fm = _depth1(dtext, _KEY_FILTER)
    if fm is None:
        return [], []
    value = fm.group(1)
    if value[:1].isdigit():
        return None                       # an indirect /Filter
    names = re.findall(rb"/[^\x00\t\n\x0c\r /<>\[\]()%{}]*", value)
    kinds = [_FILTER_KIND.get(nm) for nm in names]
    if any(k is None for k in kinds):
        return None
    parms: list[bytes | None] = [None] * len(kinds)
    pm = _depth1(dtext, _KEY_PARMS)
    if pm is not None:
        pv = pm.group(1)
        if pv[:1].isdigit():
            return None
        if pv.startswith(b"<<"):
            if len(kinds) != 1:
                return None
            parms[0] = pv
        elif pv.startswith(b"["):
            items = re.findall(rb"null|<<.*?>>|" + _REF, pv[1:-1], re.S)
            if len(items) != len(kinds) or any(it[:1].isdigit() for it in items):
                return None
            parms = [None if it == b"null" else it for it in items]
    return [k for k in kinds if k is not None], parms


def _inflate(body: bytes, cap: int) -> bytes:
    """Inflate a zlib stream, keeping what decodes before any error, at most ``cap`` bytes."""
    out, _complete = _inflate_checked(body, cap)
    return out


def _inflate_checked(body: bytes, cap: int) -> tuple[bytes, bool]:
    """(what decodes, at most ``cap`` bytes; False when the data is corrupt or the cap cut it short)."""
    d = zlib.decompressobj()
    out = bytearray()
    step = 1 << 16
    try:
        for i in range(0, len(body), step):
            out += d.decompress(body[i:i + step], max(1, cap + 1 - len(out)))
            if len(out) > cap or d.unconsumed_tail:
                return bytes(out[:cap]), False
            if d.eof:
                break
    except zlib.error:
        return bytes(out[:cap]), False
    return bytes(out), True


def _ahx(body: bytes) -> bytes | None:
    end = body.find(b">")
    text = re.sub(rb"[\x00\t\n\x0c\r ]", b"", body if end < 0 else body[:end])
    if len(text) % 2:
        text += b"0"
    try:
        return bytes.fromhex(text.decode("ascii"))
    except ValueError:
        return None


def _a85(body: bytes) -> bytes | None:
    import base64

    end = body.find(b"~>")
    text = (body if end < 0 else body[:end]).strip(_WS)
    if text.startswith(b"<~"):
        text = text[2:]
    try:
        return base64.a85decode(text, adobe=False, ignorechars=_WS)
    except ValueError:
        return None


def _predict(data: bytes, parms: bytes | None) -> bytes | None:
    """Undo a PNG (10-15) or TIFF (2) predictor; None for one this scan doesn't handle."""
    if parms is None:
        return data
    pred = _int_key(parms, "Predictor", 1)
    if pred == 1:
        return data
    colors = _int_key(parms, "Colors", 1) or 1
    bpc = _int_key(parms, "BitsPerComponent", 8) or 8
    cols = _int_key(parms, "Columns", 1) or 1
    if len(data) > MAX_PREDICTED or colors > 64 or bpc not in (1, 2, 4, 8, 16) or cols > 1 << 24:
        return None
    import numpy as np

    rowlen = (colors * bpc * cols + 7) // 8
    bpp = max(1, colors * bpc // 8)
    if pred == 2:
        if bpc != 8 or not rowlen:
            return None
        rows = len(data) // rowlen
        a = np.frombuffer(data[:rows * rowlen], dtype=np.uint8).reshape(rows, cols, colors)
        return np.cumsum(a, axis=1, dtype=np.uint8).tobytes()
    if not 10 <= (pred or 0) <= 15:
        return None
    out = bytearray()
    prev = np.zeros(rowlen, dtype=np.uint8)
    stride = rowlen + 1
    for r in range(len(data) // stride):
        kind = data[r * stride]
        row = np.frombuffer(data, dtype=np.uint8, count=rowlen, offset=r * stride + 1).copy()
        if kind == 1:                       # Sub, one byte lane at a time
            for lane in range(bpp):
                row[lane::bpp] = np.cumsum(row[lane::bpp], dtype=np.uint8)
        elif kind == 2:                     # Up
            row = row + prev
        elif kind in (3, 4):                # Average, Paeth: sequential
            cur = bytearray(row.tobytes())
            up = prev.tobytes()
            for i in range(rowlen):
                left = cur[i - bpp] if i >= bpp else 0
                if kind == 3:
                    cur[i] = (cur[i] + ((left + up[i]) >> 1)) & 0xFF
                else:
                    ul = up[i - bpp] if i >= bpp else 0
                    p = left + up[i] - ul
                    pa, pb, pc = abs(p - left), abs(p - up[i]), abs(p - ul)
                    cur[i] = (cur[i] + (left if pa <= pb and pa <= pc else up[i] if pb <= pc else ul)) & 0xFF
            row = np.frombuffer(bytes(cur), dtype=np.uint8)
        elif kind != 0:
            return None
        out += row.tobytes()
        prev = row
    return bytes(out)


def _decode(body: bytes, dtext: bytes, cap: int) -> bytes | None:
    """The stream's decoded bytes under ``cap``, or None when the chain can't be decoded or the cap cut it."""
    chain = _filters(dtext)
    if chain is None:
        return None
    data: bytes | None = body
    for kind, parms in zip(*chain):
        assert data is not None
        if kind == "flate":
            data, complete = _inflate_checked(data, cap)
            if not complete:
                return None
            data = _predict(data, parms)
        elif parms is not None:
            return None
        elif kind == "ahx":
            data = _ahx(data)
        else:
            data = _a85(data)
        if data is None or len(data) > cap:
            return None
    return data


_OBJSTM_PAIR = re.compile(rb"[0-9]{1,10}" + _W + rb"++[0-9]{1,15}")


def _objstm_ok(decoded: bytes, dtext: bytes) -> bool:
    """True when the decoded object stream starts with its /N number-offset pairs before /First."""
    n, first = _int_key(dtext, "N"), _int_key(dtext, "First")
    if n is None or first is None or first > len(decoded):
        return False
    head = decoded[:first]
    pairs = _OBJSTM_PAIR.findall(head)
    rest = _OBJSTM_PAIR.sub(b"", head).strip(_WS)
    return len(pairs) == n and not rest


def _int_objects(data: bytes) -> dict[tuple[int, int], int]:
    """Every ``N G obj <integer> endobj`` in the file (the last definition wins), for indirect lengths."""
    return {(int(m.group(1)), int(m.group(2))): int(m.group(3)) for m in _INT_OBJ.finditer(data)}


def _count(data: bytes, needle: bytes) -> int:
    return data.count(needle)


def _incremental_updates(data: bytes) -> tuple[int, bool]:
    """(updates after the first revision, objects defined after the last startxref)."""
    n = len(data)
    if not n:
        return 0, False
    linearized = data.find(b"/Linearized", 0, min(n, 2048)) >= 0
    last = data.rfind(b"startxref")
    unfinished = last >= 0 and _OBJ_HEAD.search(data, last) is not None
    sections = 0
    if last >= 0:
        m = _STARTXREF.match(data, last)
        seen: set[int] = set()
        off = int(m.group(1)) if m else -1
        while 0 <= off < n and off not in seen and len(seen) < 4096:
            seen.add(off)
            k = off
            while k < n and data[k] in _WS:
                k += 1
            if data[k:k + 4] == b"xref":
                t = data.find(b"trailer", k)
                if t < 0:
                    break
                stop = data.find(b"startxref", t)
                text = data[t:stop if stop > t else min(n, t + 8192)]
            elif _OBJ_HEAD.match(data, k):
                s = data.find(b"stream", k, min(n, k + 65536))
                text = data[k:s if s > k else min(n, k + 8192)]
            else:
                break
            sections += 1
            p = _PREV.search(text)
            off = int(p.group(1)) if p else -1
    if sections == 0:                      # damaged: count the startxref keywords instead
        sections = _count(data, b"startxref")
    if linearized and sections > 1:
        sections -= 1
    return max(0, sections - 1), unfinished


class _Scan:
    """One raw scan over a file's bytes: the markers found and the state the lexer needs."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.found: set[str] = set()
        self.inflated = 0
        self.trailer = False
        self.last_end = 0
        self.nxt = _Next(data)
        self._ints: dict[tuple[int, int], int] | None = None

    def search(self, buf: bytes) -> None:
        for candidate in (buf, _unescape(buf)) if b"#" in buf else (buf,):
            for name, pat in _RAW_PATTERNS:
                if name not in self.found and pat.search(candidate):
                    self.found.add(name)
            if not {"pdfaid:part", "pdfuaid:part"} <= self.found:
                self.found |= _ns_prefix_parts(candidate)

    def _length(self, dtext: bytes) -> int | None:
        m = _depth1(dtext, _KEY_LENGTH)
        if m is None:
            return None
        if m.group(2) is None:
            return int(m.group(1))
        if self._ints is None:
            self._ints = _int_objects(self.data)
        return self._ints.get((int(m.group(1)), int(m.group(2))))

    def _body_end(self, body: int, dtext: bytes) -> int:
        data, nxt = self.data, self.nxt
        e = nxt(b"endstream", body)
        length = self._length(dtext)
        if length is not None and e >= 0 and body + length <= e <= body + length + _LENGTH_SLACK \
                and not data[body + length:e].strip(_WS):
            return e
        ends = [c for c in (e, nxt(_OBJ_LINE, body), nxt(_XREF_LINE, body), nxt(_TRAILER, body)) if c >= 0]
        return min(ends) if ends else body       # no terminator: the rest is scanned as plain bytes

    def on_stream(self, t: int) -> tuple[int, int] | None:
        data = self.data
        j = t
        while j > self.last_end and data[j - 1] in _WS:
            j -= 1
        if j - 2 < self.last_end or data[j - 2:j] != b">>":
            return None
        k = t + 6
        while k < len(data) and data[k] in b"\x00\t\x0c ":
            k += 1
        if data[k:k + 2] == b"\r\n":
            k += 2
        elif data[k:k + 1] in (b"\n", b"\r"):
            k += 1
        lo = max(self.last_end, t - _DICT_LOOKBACK)
        head = data.rfind(b"obj", lo, t)
        dtext = _unescape(_lex(data[head + 3 if head >= 0 else lo:j]))
        end = self._body_end(k, dtext)
        self.last_end = end
        self._take(data[k:end], dtext)
        return k, end

    def _take(self, body: bytes, dtext: bytes) -> None:
        if _TYPE_XREF.search(dtext):
            self.trailer = True
        objstm = _TYPE_OBJSTM.search(dtext) is not None
        if not objstm and _TYPE_META.search(dtext) is None:
            return                                    # page content, images, fonts: never scanned
        cap = min(MAX_INFLATE_STREAM, MAX_INFLATE_TOTAL - self.inflated)
        decoded = _decode(body, dtext, cap) if cap > 0 else None
        if decoded is None or (objstm and not _objstm_ok(decoded, dtext)):
            self.found.add(UNSCANNED)
            return
        self.inflated += len(decoded)
        self.search(_lex(decoded) if objstm else decoded)

    def run(self) -> None:
        plain = _lex(self.data, self.on_stream)
        self.search(plain)
        if _TRAILER.search(plain):
            self.trailer = True


def _raw_scan(path: Path) -> _Raw:
    data = _read_file(path)
    if data is None:
        return _Raw(markers=[UNSCANNED], incremental_updates=0, unfinished_update=False)
    scan = _Scan(data)
    scan.run()
    if not scan.trailer:
        scan.found.add(NO_TRAILER)
    updates, unfinished = _incremental_updates(data)
    return _Raw(markers=[m for m in RAW_MARKERS if m in scan.found], incremental_updates=updates,
                unfinished_update=unfinished)


# --------------------------------------------------------------------------- XMP

_PDFA_NS = "http://www.aiim.org/pdfa/ns/id"
_PDFUA_NS = "http://www.aiim.org/pdfua/ns/id"


def _split(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, _, local = tag[1:].partition("}")
        return ns.rstrip("/"), local
    return "", tag


def _parse_xmp(data: bytes) -> tuple[bool, str | None, str | None]:
    """(readable, pdfa, pdfua) for an XMP packet. Empty or whitespace-only XMP is readable and claims nothing.

    A DOCTYPE or ENTITY declaration counts as unreadable: XMP never needs one, and refusing it
    means no entity is ever expanded.
    """
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            text: str | bytes = data.decode("utf-16")
        except UnicodeDecodeError:
            return False, None, None
        text = text.strip(" \t\r\n\x00\ufeff")
        probe = text.upper()
        if not text:
            return True, None, None
        if "<!DOCTYPE" in probe or "<!ENTITY" in probe:
            return False, None, None
        text = re.sub(r"^<\?xml[^>]*\?>", "", text)
    else:
        text = data.strip(b" \t\r\n\x00")
        if not text:
            return True, None, None
        probe_b = text.upper()
        if b"<!DOCTYPE" in probe_b or b"<!ENTITY" in probe_b:
            return False, None, None
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, ValueError, UnicodeError):
        return False, None, None
    found: dict[tuple[str, str], str] = {}
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for key, value in el.attrib.items():
            ns, local = _split(key)
            if ns in (_PDFA_NS, _PDFUA_NS):
                found.setdefault((ns, local), value)
        ns, local = _split(el.tag)
        if ns in (_PDFA_NS, _PDFUA_NS):
            found.setdefault((ns, local), (el.text or "").strip())
    pdfa = pdfua = None
    if (_PDFA_NS, "part") in found:
        part = found[(_PDFA_NS, "part")].strip()
        conf = found.get((_PDFA_NS, "conformance"), "").strip().upper()
        pdfa = f"{part}{conf}" if re.fullmatch(r"[0-9]{1,2}", part) and re.fullmatch(r"[A-Z]{0,2}", conf) else "unknown"
    if (_PDFUA_NS, "part") in found:
        part = found[(_PDFUA_NS, "part")].strip()
        pdfua = part if re.fullmatch(r"[0-9]{1,2}", part) else "unknown"
    return True, pdfa, pdfua


# --------------------------------------------------------------------------- the object walk

_DICT, _STREAM, _ARRAY, _NAME = (pikepdf.ObjectType.dictionary, pikepdf.ObjectType.stream,
                                 pikepdf.ObjectType.array, pikepdf.ObjectType.name_)


@dataclass
class _Walk:
    codes: list[str]
    signed_fields: list[str]
    empty_sig_fields: int
    sig_dicts: int
    doc_timestamps: int
    certified_p: int | None
    usage_rights: bool
    pdfa: str | None
    pdfua: str | None
    encrypted: Literal[False, "owner", "user"]
    xfa: bool
    jbig2_streams: int
    xmp_readable: bool
    warnings: int
    object_claims: set[str]           # WALK_OBJECT_PDFA / WALK_OBJECT_PDFUA from other reachable XMP


def _empty_walk(codes: list[str], encrypted: Literal[False, "owner", "user"] = False) -> _Walk:
    return _Walk(codes=codes, signed_fields=[], empty_sig_fields=0, sig_dicts=0, doc_timestamps=0,
                 certified_p=None, usage_rights=False, pdfa=None, pdfua=None, encrypted=encrypted,
                 xfa=False, jbig2_streams=0, xmp_readable=True, warnings=0, object_claims=set())


def _tc(obj: Any) -> Any:
    """The pikepdf object type, or None for the Python scalars pikepdf converts (int, bool, Decimal, None)."""
    return getattr(obj, "_type_code", None)


def _key(obj: Any) -> tuple[Any, ...]:
    """A dedupe key: the object and generation number, or the serialised object when it is direct.

    pikepdf hands out a new wrapper on every access, so a wrapper's identity can't stand for a
    direct object (and ids are reused once a wrapper is freed).
    """
    if obj.is_indirect:
        return obj.objgen
    return ("direct", bytes(obj.unparse()))


def _get(d: Any, key: str) -> Any:
    """d[key], or None when absent or null (pikepdf drops null-valued keys)."""
    try:
        return d.get(key)
    except Exception:
        return None


def _is_dictlike(obj: Any) -> bool:
    return _tc(obj) in (_DICT, _STREAM)


def _is_array(obj: Any) -> bool:
    return _tc(obj) == _ARRAY


def _name(obj: Any) -> str | None:
    return str(obj) if _tc(obj) == _NAME else None


def _has_filter(stream: Any, filter_name: str) -> bool:
    f = _get(stream, "/Filter")
    if _is_array(f):
        return any(_name(x) == filter_name for x in f)
    return _name(f) == filter_name


def _reachable(pdf: Any) -> Iterator[Any]:
    """Every dictionary and stream reachable from the trailer, each indirect object once."""
    seen: set[tuple[int, int]] = set()
    stack: list[Any] = [pdf.trailer]
    visited = 0
    while stack:
        obj = stack.pop()
        code = _tc(obj)
        if code not in (_DICT, _STREAM, _ARRAY):
            continue
        if obj.is_indirect:
            og = obj.objgen
            if og in seen:
                continue
            seen.add(og)
        visited += 1
        if visited > MAX_WALK_OBJECTS:
            raise RuntimeError("too many objects")
        if code == _ARRAY:
            stack.extend(v for v in obj if _tc(v) in (_DICT, _STREAM, _ARRAY))
            continue
        yield obj
        for _k, v in obj.items():
            if _tc(v) in (_DICT, _STREAM, _ARRAY):
                stack.append(v)


def _inherited(node: Any, key: str, limit: int = 64) -> Any:
    cur: Any = node
    seen: set[Any] = set()
    for _ in range(limit):
        if not _is_dictlike(cur):
            return None
        v = _get(cur, key)
        if v is not None:
            return v
        k = _key(cur)
        if k in seen:
            return None
        seen.add(k)
        cur = _get(cur, "/Parent")
    return None


def _terminal_fields(acroform: Any) -> Iterator[Any]:
    """Terminal fields of the AcroForm tree (a field whose kids are only widgets is terminal)."""
    fields = _get(acroform, "/Fields")
    if not _is_array(fields):
        return
    stack = list(fields)
    seen: set[Any] = set()
    while stack:
        node = stack.pop()
        if not _is_dictlike(node):
            continue
        k = _key(node)
        if k in seen:
            continue
        seen.add(k)
        kids = _get(node, "/Kids")
        child_fields = []
        if _is_array(kids):
            child_fields = [c for c in kids if _is_dictlike(c) and ("/T" in c or "/Kids" in c or "/FT" in c)]
        if child_fields:
            stack.extend(child_fields)
        else:
            yield node


def _field_of_widget(widget: Any) -> Any:
    if "/T" in widget or "/FT" in widget:
        return widget
    parent = _get(widget, "/Parent")
    return parent if _is_dictlike(parent) else widget


def _docmdp_p(sig: Any) -> int | None:
    """The P value of a certification signature (2 when absent or invalid), None when the dict isn't one."""
    refs = _get(sig, "/Reference")
    if not _is_array(refs):
        return None
    for ref in refs:
        if _is_dictlike(ref) and _name(_get(ref, "/TransformMethod")) == "/DocMDP":
            params = _get(ref, "/TransformParams")
            p = _get(params, "/P") if _is_dictlike(params) else None
            if isinstance(p, int) and not isinstance(p, bool) and 1 <= p <= 3:
                return p
            return 2
    return None


def _xmp_bytes(meta: Any) -> bytes | None:
    """The decoded bytes of an XMP stream, or None when they can't be read within MAX_XMP_BYTES.

    Only no filter or a single Flate filter without /DecodeParms is decoded, by the capped
    inflate, so a bomb never expands: any other filter, a /DecodeParms entry, or more than
    MAX_XMP_BYTES decoded counts as unreadable. ``read_raw_bytes`` is never used on a
    stream whose /Length already exceeds the cap.
    """
    if _tc(meta) != _STREAM:
        return None
    filt = _get(meta, "/Filter")
    if filt is None:
        names: list[str | None] = []
    elif _is_array(filt):
        names = [_name(x) for x in filt]
    else:
        names = [_name(filt)]
    parms = _get(meta, "/DecodeParms")
    if parms is not None and not (_is_array(parms) and all(x is None for x in parms)):
        return None
    length = _get(meta, "/Length")
    if isinstance(length, int) and length > MAX_XMP_BYTES and not names:
        return None
    try:
        raw = meta.read_raw_bytes()
    except Exception:
        return None
    if not names:
        data = raw
    elif names in (["/FlateDecode"], ["/Fl"]):
        data, complete = _inflate_checked(raw, MAX_XMP_BYTES)
        if not complete:
            return None
    else:
        return None
    return bytes(data) if len(data) <= MAX_XMP_BYTES else None


def _read_xmp(root: Any) -> tuple[bool, str | None, str | None]:
    meta = _get(root, "/Metadata")
    if meta is None:
        return True, None, None
    data = _xmp_bytes(meta)
    if data is None:
        return False, None, None
    return _parse_xmp(data)


def _is_xmp_stream(obj: Any) -> bool:
    """A stream the raw scan treats as XMP: /Type /Metadata or /Subtype /XML."""
    return _tc(obj) == _STREAM and (_name(_get(obj, "/Type")) == "/Metadata" or _name(_get(obj, "/Subtype")) == "/XML")


def _all_fields(pdf: Any, root: Any) -> tuple[list[Any], dict[Any, int]]:
    """Every terminal form field, then every page widget's field the form doesn't list; and each field's page."""
    page_of: dict[Any, int] = {}
    widget_fields: list[Any] = []
    for i, page in enumerate(pdf.pages):
        annots = _get(page.obj, "/Annots")
        if not _is_array(annots):
            continue
        for a in annots:
            if _is_dictlike(a) and _name(_get(a, "/Subtype")) == "/Widget":
                f = _field_of_widget(a)
                page_of.setdefault(_key(f), i + 1)
                widget_fields.append(f)
    terminal: list[Any] = []
    keys: set[Any] = set()
    acroform = _get(root, "/AcroForm")
    candidates = list(_terminal_fields(acroform)) if _is_dictlike(acroform) else []
    for f in candidates + widget_fields:
        if _key(f) not in keys:
            keys.add(_key(f))
            terminal.append(f)
    return terminal, page_of


def _walk_open(pdf: Any, encrypted: Literal[False, "owner", "user"]) -> _Walk:
    w = _empty_walk([], encrypted)
    root = _get(pdf.trailer, "/Root")
    if not _is_dictlike(root):
        w.codes = [WALK_FAILED]
        return w

    terminal, page_of = _all_fields(pdf, root)
    acroform = _get(root, "/AcroForm")
    if _is_dictlike(acroform):
        w.xfa = _get(acroform, "/XFA") is not None

    # /Perms: certification and usage rights.
    perms = _get(root, "/Perms")
    ur_sigs: set[Any] = set()
    docmdp_sig = None
    if _is_dictlike(perms):
        for k in ("/UR3", "/UR"):
            ur = _get(perms, k)
            if ur is not None:
                w.usage_rights = True
                if _is_dictlike(ur):
                    ur_sigs.add(_key(ur))
        docmdp = _get(perms, "/DocMDP")
        if docmdp is not None:
            docmdp_sig = docmdp if _is_dictlike(docmdp) else None
            w.certified_p = (_docmdp_p(docmdp_sig) if docmdp_sig is not None else None) or 2

    # Signature fields: signed, timestamp, or empty.
    field_sigs: set[Any] = set()
    timestamps: set[Any] = set()
    n_field = 0
    for f in terminal:
        if _name(_inherited(f, "/FT")) != "/Sig":
            continue
        n_field += 1
        v = _inherited(f, "/V")
        if not _is_dictlike(v) or not ("/ByteRange" in v or "/Contents" in v):
            w.empty_sig_fields += 1
            continue
        field_sigs.add(_key(v))
        if _name(_get(v, "/Type")) == "/DocTimeStamp":
            timestamps.add(_key(v))
            continue
        page = page_of.get(_key(f))
        w.signed_fields.append(f"field {n_field}, page {page}" if page else f"field {n_field}, no widget")

    # Every reachable dictionary: signature dictionaries anywhere, JBIG2 images, other XMP.
    sig_keys: set[Any] = set()
    jbig2: set[Any] = set()
    orphan = 0
    catalog_meta = _get(root, "/Metadata")
    catalog_meta_key = _key(catalog_meta) if _is_dictlike(catalog_meta) else None
    xmp_budget = MAX_INFLATE_TOTAL
    for obj in _reachable(pdf):
        if "/ByteRange" in obj:
            k = _key(obj)
            if k in sig_keys:
                continue
            sig_keys.add(k)
            if _name(_get(obj, "/Type")) == "/DocTimeStamp":
                timestamps.add(k)
            elif k not in field_sigs and k not in ur_sigs:
                orphan += 1
                w.signed_fields.append(f"signature {orphan}, no field")
            p = _docmdp_p(obj)
            if p is not None and w.certified_p is None:
                w.certified_p = p
        elif _tc(obj) == _STREAM and _has_filter(obj, "/JBIG2Decode"):
            jbig2.add(_key(obj))
        elif _is_xmp_stream(obj) and xmp_budget > 0 and _key(obj) != catalog_meta_key:
            # A page's or an image's own XMP: its claim isn't the document's, but the raw scan sees it.
            data = _xmp_bytes(obj)
            if data is not None:
                xmp_budget -= len(data)
                _readable, a, ua = _parse_xmp(data)
                if a is not None:
                    w.object_claims.add(WALK_OBJECT_PDFA)
                if ua is not None:
                    w.object_claims.add(WALK_OBJECT_PDFUA)
    w.sig_dicts = len(sig_keys)
    w.doc_timestamps = len(timestamps)
    w.jbig2_streams = len(jbig2)

    w.xmp_readable, w.pdfa, w.pdfua = _read_xmp(root)

    codes = []
    if encrypted:
        codes.append("encrypted")
    if w.certified_p is not None:
        codes.append("certified")
    if w.signed_fields:
        codes.append("signed")
    if w.doc_timestamps:
        codes.append("timestamped")
    if w.usage_rights:
        codes.append("usage_rights")
    if w.xfa:
        codes.append("xfa")
    if w.pdfa is not None:
        codes.append("pdfa")
    if w.pdfua is not None:
        codes.append(WALK_PDFUA)
    if not w.xmp_readable:
        codes.append("xmp_unreadable")
    codes += [c for c in (WALK_OBJECT_PDFA, WALK_OBJECT_PDFUA) if c in w.object_claims]
    w.codes = codes
    return w


def _walk(path: Path) -> _Walk:
    try:
        pdf = pikepdf.open(path, suppress_warnings=True, access_mode=pikepdf.AccessMode.stream)
    except pikepdf.PasswordError:
        return _empty_walk(["encrypted"], "user")
    except Exception:
        return _empty_walk([WALK_OPEN_FAILED])
    warnings = 0
    try:
        warnings += len(pdf.get_warnings())
        encrypted: Literal[False, "owner", "user"] = "owner" if pdf.is_encrypted else False
        try:
            w = _walk_open(pdf, encrypted)
        except Exception:
            w = _empty_walk(["encrypted", WALK_FAILED] if encrypted else [WALK_FAILED], encrypted)
        warnings += len(pdf.get_warnings())
        w.warnings = warnings
        return w
    finally:
        pdf.close()


# --------------------------------------------------------------------------- public readers

def read(pdf: Path) -> RefusalReport:
    """Both detectors over one PDF. Never raises for content; a password prompt becomes ``encrypted="user"``.

    The file is only read. OSError (a missing or unreadable path) still propagates.
    """
    pdf = Path(pdf)
    raw = _raw_scan(pdf)
    w = _walk(pdf)
    return RefusalReport(
        signed_fields=w.signed_fields, empty_sig_fields=w.empty_sig_fields, sig_dicts=w.sig_dicts,
        doc_timestamps=w.doc_timestamps, certified_p=w.certified_p, usage_rights=w.usage_rights,
        pdfa=w.pdfa, pdfua=w.pdfua, encrypted=w.encrypted, xfa=w.xfa, jbig2_streams=w.jbig2_streams,
        incremental_updates=raw.incremental_updates, xmp_readable=w.xmp_readable,
        walk_codes=w.codes, raw_markers=raw.markers,
        repaired=w.warnings > 0 or raw.unfinished_update, qpdf_warnings=w.warnings,
    )


def output_signature_material(pdf: Path) -> list[str]:
    """Signature material in a built output: "/ByteRange", "/Perms", "/Sig field" (one that holds a value).

    A /Sig field counts when its value holds /ByteRange or /Contents, the same test ``read``
    uses, so an empty signature field (which ``read`` lets through) isn't material. "/Perms"
    comes from the catalog only; the raw scan adds unreferenced byte ranges. A file pikepdf can't
    open, or one the raw scan couldn't read in full (outside encrypted streams), gives
    "unreadable", so a check built on this fails closed.
    """
    pdf = Path(pdf)
    found: set[str] = set()
    raw = _raw_scan(pdf).markers
    if "/ByteRange [" in raw:
        found.add("/ByteRange")
    encrypted = False
    try:
        with pikepdf.open(pdf, suppress_warnings=True, access_mode=pikepdf.AccessMode.stream) as doc:
            encrypted = bool(doc.is_encrypted)
            root = _get(doc.trailer, "/Root")
            if _is_dictlike(root):
                if _get(root, "/Perms") is not None:
                    found.add("/Perms")
                for f in _all_fields(doc, root)[0]:
                    v = _inherited(f, "/V")
                    if _name(_inherited(f, "/FT")) == "/Sig" and _is_dictlike(v) and ("/ByteRange" in v or "/Contents" in v):
                        found.add("/Sig field")
                        break
            for obj in _reachable(doc):
                if "/ByteRange" in obj:
                    found.add("/ByteRange")
                    break
    except Exception:
        found.add("unreadable")
    if NO_TRAILER in raw or (UNSCANNED in raw and not encrypted):
        found.add("unreadable")
    return [label for label in ("/ByteRange", "/Perms", "/Sig field", "unreadable") if label in found]
