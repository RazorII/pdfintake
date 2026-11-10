"""Font programs, outlines and reference fonts for the font-map repair (DESIGN §15).

Everything here reads fonts; nothing writes a PDF. ``pdf/fontmap.py`` uses it to find out which
letter each glyph of an embedded font draws:

* ``Program`` wraps an embedded font program (TrueType from ``/FontFile2``, CFF from
  ``/FontFile3``, OpenType) read with fontTools. A glyph's identity is the hash of its outline,
  decomposed and rounded to whole font units (``Program.glyph_hash``), so two programs that hold
  the same glyph give the same hash whatever they number it.
* ``font_dirs`` lists the platform's font folders; ``ReferenceIndex`` finds the installed faces of a
  family there (read-only), and ``Reference`` holds one face's glyphs: outline hash and the letters
  its cmap gives, plus what its GSUB makes of them read backwards (ligatures and positional
  variants). No font is downloaded, copied or named in a record.
* ``parse_cmap`` and ``cmap_stream`` read and write ``/ToUnicode`` CMaps (``bfchar`` and ``bfrange``).

Every limit below bounds what an untrusted file can make pdfintake do. A program or CMap that
fontTools can't read, or that passes a limit, makes its font out of scope; it never stops the run.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import threading
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_PROGRAM_BYTES = 64 * 1024 * 1024        # an embedded font program larger than this isn't read
MAX_GLYPHS = 70_000                         # nor one with more glyphs (the format's own ceiling is 65,535)
MAX_CMAP_BYTES = 8 * 1024 * 1024            # a /ToUnicode stream larger than this is unusable
MAX_CMAP_ENTRIES = 200_000                  # bfchar entries plus codes covered by bfranges
MAX_RANGE = 65_536                          # one bfrange never spans more codes than this
MAX_DST_CHARS = 16                          # a destination string longer than this is unusable
MAX_REFERENCE_FILES = 20_000                # font files looked at in the platform's folders
MAX_REFERENCE_BYTES = 256 * 1024 * 1024     # a reference file larger than this is skipped

FONT_SUFFIXES = (".ttf", ".otf", ".ttc", ".otc")

# fontTools logs what it repairs in a font it reads ("1 extra bytes in post.stringData array"); a run
# reports through its record, never stderr chatter about the user's files.
import logging as _logging

_logging.getLogger("fontTools").setLevel(_logging.ERROR)
_TAG_RE = re.compile(r"^[A-Z]{6}\+")
_STYLE_TOKENS = ("psmt", "ps", "mt")


# --------------------------------------------------------------------------- names

def strip_subset_tag(base: str) -> str:
    """``ABCDEF+Times-Bold`` -> ``Times-Bold``."""
    name = base[1:] if base.startswith("/") else base
    return _TAG_RE.sub("", name)


def family_key(name: str) -> str:
    """A font's family, comparable across a PDF's ``/BaseFont`` and an installed face's names.

    The subset tag and everything from the first ``,`` or ``-`` go; then a trailing ``PSMT``,
    ``PS`` or ``MT``; then anything that isn't a letter or digit; lower case. ``TimesNewRomanPS-BoldMT``,
    ``TimesNewRoman,Bold`` and ``Times New Roman`` all give ``timesnewroman``.
    """
    base = strip_subset_tag(name)
    base = re.split(r"[,\-]", base, maxsplit=1)[0]
    low = base.lower()
    for token in _STYLE_TOKENS:
        if low.endswith(token) and len(low) > len(token) + 2:
            low = low[: -len(token)]
            break
    return "".join(ch for ch in low if ch.isalnum())


def style_key(name: str) -> str:
    """A font's family and style: ``family_key`` plus the part after the first ``,`` or ``-``, folded
    the same way (``Regular`` when there is none). ``TimesNewRomanPS-ItalicMT`` and
    ``TimesNewRoman,Italic`` both give ``timesnewroman-italic``."""
    base = strip_subset_tag(name)
    parts = re.split(r"[,\-]", base, maxsplit=1)
    style = parts[1].lower() if len(parts) > 1 else ""
    for token in _STYLE_TOKENS:
        if style.endswith(token) and len(style) > len(token):
            style = style[: -len(token)]
            break
    style = "".join(ch for ch in style if ch.isalnum()) or "regular"
    return f"{family_key(name)}-{style}"


# --------------------------------------------------------------------------- text

_BAD_CATEGORIES = frozenset({"Cc", "Cs", "Cn"})


def usable_text(text: str) -> bool:
    """A mapping a reader can use: not empty, not too long, no control, surrogate or unassigned
    character, no U+FFFD or U+FFFE. Private use counts as usable: symbol fonts map there on purpose."""
    if not text or len(text) > MAX_DST_CHARS:
        return False
    for ch in text:
        if ch in ("\ufffd", "\ufffe", "\uffff") or unicodedata.category(ch) in _BAD_CATEGORIES:
            return False
    return True


def agl_text(name: str | None) -> str | None:
    """A glyph name through the Adobe Glyph List (``fontTools.agl``), ligatures included; None when it
    names no usable text (``g255``, ``C0``, ``.notdef``)."""
    if not name or name.startswith(".") or len(name) > 256:
        return None
    from fontTools.agl import toUnicode

    try:
        text = toUnicode(name)
    except Exception:  # noqa: BLE001 - a malformed name names nothing
        return None
    return text if usable_text(text) else None


# --------------------------------------------------------------------------- outlines

def _hash_recording(value: Sequence[tuple[str, Sequence[Any]]]) -> str | None:
    if not value:
        return None
    parts = []
    for op, pts in value:
        coords = ",".join(f"{round(float(x))}:{round(float(y))}" for x, y in pts)
        parts.append(f"{op}({coords})")
    return hashlib.sha256(";".join(parts).encode("ascii")).hexdigest()[:32]


class OutlineError(Exception):
    """A glyph that can't be drawn (a broken charstring, a component loop)."""


def draw_hash(glyphset: Any, name: str) -> str | None:
    """The outline hash of glyph ``name`` (None for an empty glyph); OutlineError when it can't be drawn."""
    from fontTools.pens.recordingPen import DecomposingRecordingPen

    pen = DecomposingRecordingPen(glyphset)
    try:
        glyphset[name].draw(pen)
    except Exception as exc:  # noqa: BLE001 - any failure of an untrusted program
        raise OutlineError(type(exc).__name__) from None
    return _hash_recording(pen.value)


class ProgramError(Exception):
    """A font program pdfintake won't read (unreadable, too large, an unsupported kind)."""


@dataclass
class Program:
    """One embedded font program. ``kind`` is ``truetype`` or ``cff``; ``cid_keyed`` for a CID-keyed
    CFF, whose glyphs are found by CID through its charset."""

    kind: str
    glyph_order: list[str]
    glyphset: Any
    cid_keyed: bool = False
    cid_gid: dict[int, int] = field(default_factory=dict)
    _ttfont: Any = None
    _cff: Any = None
    _hashes: dict[int, str | None] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def num_glyphs(self) -> int:
        return len(self.glyph_order)

    def glyph_hash(self, gid: int) -> str | None:
        """The outline hash of glyph ``gid``; None when the glyph is empty or doesn't exist or can't be drawn.

        Use ``has_glyph`` first to tell a missing glyph from an empty one.
        """
        with self._lock:
            if gid in self._hashes:
                return self._hashes[gid]
            value: str | None = None
            if 0 <= gid < len(self.glyph_order):
                try:
                    value = draw_hash(self.glyphset, self.glyph_order[gid])
                except OutlineError:
                    value = None
            self._hashes[gid] = value
            return value

    def has_glyph(self, gid: int) -> bool:
        return 0 <= gid < len(self.glyph_order)

    def drawable(self, gid: int) -> bool:
        """The glyph exists and draws without error (an empty glyph is drawable)."""
        if not self.has_glyph(gid):
            return False
        try:
            draw_hash(self.glyphset, self.glyph_order[gid])
        except OutlineError:
            return False
        return True

    def glyph_name(self, gid: int) -> str | None:
        """The glyph's own name, when the program carries real names (a ``post`` table, a name-keyed CFF)."""
        if not self.has_glyph(gid):
            return None
        name = self.glyph_order[gid]
        if self.cid_keyed or re.fullmatch(r"glyph\d+", name):
            return None
        if self.kind == "truetype":
            post = self._ttfont.get("post") if self._ttfont is not None and "post" in self._ttfont else None
            if post is None or getattr(post, "formatType", 3.0) not in (1.0, 2.0):
                return None
        return name

    def gid_for_name(self, name: str) -> int | None:
        try:
            return self.glyph_order.index(name)
        except ValueError:
            return None

    def cmap_text(self) -> dict[int, set[str]]:
        """gid -> the texts the program's own Unicode cmap gives it ((3,1), (3,10), (0,x))."""
        out: dict[int, set[str]] = {}
        tt = self._ttfont
        if tt is None or "cmap" not in tt:
            return out
        names = {n: i for i, n in enumerate(self.glyph_order)}
        for table in tt["cmap"].tables:
            if not ((table.platformID == 3 and table.platEncID in (1, 10)) or table.platformID == 0):
                continue
            for cp, gname in table.cmap.items():
                if not 0 < cp <= 0x10FFFF:
                    continue
                gid = names.get(gname)
                ch = chr(cp)
                if gid is not None and usable_text(ch):
                    out.setdefault(gid, set()).add(ch)
        return out

    def symbol_cmap(self) -> dict[int, int]:
        """code -> gid for a simple symbolic TrueType font: the (3,0) cmap at 0xF000 + code (or at code),
        else the (1,0) cmap at code."""
        out: dict[int, int] = {}
        tt = self._ttfont
        if tt is None or "cmap" not in tt:
            return out
        names = {n: i for i, n in enumerate(self.glyph_order)}
        tables = {(t.platformID, t.platEncID): t for t in tt["cmap"].tables}
        sym = tables.get((3, 0))
        mac = tables.get((1, 0))
        for code in range(256):
            gname = None
            if sym is not None:
                gname = sym.cmap.get(0xF000 + code) or sym.cmap.get(code)
            if gname is None and mac is not None:
                gname = mac.cmap.get(code)
            if gname is not None and gname in names:
                out[code] = names[gname]
        return out

    def builtin_encoding(self) -> dict[int, str]:
        """code -> glyph name from a CFF program's own encoding (a simple Type1C font with no /Encoding)."""
        cff = self._cff
        if cff is None or self.cid_keyed:
            return {}
        enc = getattr(cff, "Encoding", None)
        if not isinstance(enc, list):
            if enc == "StandardEncoding":
                from fontTools.encodings.StandardEncoding import StandardEncoding

                return {i: n for i, n in enumerate(StandardEncoding) if n != ".notdef"}
            return {}
        return {i: n for i, n in enumerate(enc) if n and n != ".notdef"}


def load_program(data: bytes, subtype: str) -> Program:
    """A ``Program`` from an embedded font stream's bytes; ``subtype`` is ``FontFile2``, ``Type1C``,
    ``CIDFontType0C`` or ``OpenType``. ProgramError when it can't or mustn't be read."""
    if len(data) > MAX_PROGRAM_BYTES:
        raise ProgramError("program_too_large")
    try:
        if subtype in ("FontFile2", "OpenType"):
            from fontTools.ttLib import TTFont

            tt = TTFont(io.BytesIO(data), lazy=True, recalcBBoxes=False, recalcTimestamp=False)
            order = tt.getGlyphOrder()
            if len(order) > MAX_GLYPHS:
                raise ProgramError("too_many_glyphs")
            if "glyf" in tt:
                return Program("truetype", order, tt.getGlyphSet(), _ttfont=tt)
            if "CFF " in tt:
                top = tt["CFF "].cff[tt["CFF "].cff.fontNames[0]]
                return _cff_program(top, ttfont=tt)
            raise ProgramError("no_outlines")
        if subtype in ("Type1C", "CIDFontType0C"):
            from fontTools.cffLib import CFFFontSet

            cff = CFFFontSet()
            cff.decompile(io.BytesIO(data), None)
            if not cff.fontNames:
                raise ProgramError("no_outlines")
            return _cff_program(cff[cff.fontNames[0]])
    except ProgramError:
        raise
    except Exception as exc:  # noqa: BLE001 - an unreadable program makes its font out of scope
        raise ProgramError("program_unreadable") from exc
    raise ProgramError("program_kind")


def _cff_program(top: Any, ttfont: Any = None) -> Program:
    charset = list(top.charset)
    if len(charset) > MAX_GLYPHS:
        raise ProgramError("too_many_glyphs")
    cid_keyed = hasattr(top, "ROS")
    cid_gid: dict[int, int] = {}
    if cid_keyed:
        for gid, name in enumerate(charset):
            if name == ".notdef":
                cid_gid[0] = gid
            elif name.startswith("cid") and name[3:].isdigit():
                cid_gid[int(name[3:])] = gid
    return Program("cff", charset, top.CharStrings, cid_keyed=cid_keyed, cid_gid=cid_gid, _ttfont=ttfont, _cff=top)


# --------------------------------------------------------------------------- reference fonts

def font_dirs() -> list[Path]:
    """The platform's font folders, read-only; the ones that don't exist are left out."""
    home = Path.home()
    if sys.platform == "darwin":
        dirs = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
                home / "Library" / "Fonts"]
    elif sys.platform == "win32":
        windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
        local = os.environ.get("LOCALAPPDATA")
        dirs = [Path(windir) / "Fonts"]
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                (Path(xdg) if xdg and os.path.isabs(xdg) else home / ".local" / "share") / "fonts", home / ".fonts"]
    return [d for d in dirs if d.is_dir()]


def _font_files(dirs: Iterable[Path]) -> Iterator[Path]:
    seen = 0
    for top in dirs:
        for root, subdirs, files in os.walk(top):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
            for name in sorted(files):
                if not name.lower().endswith(FONT_SUFFIXES):
                    continue
                seen += 1
                if seen > MAX_REFERENCE_FILES:
                    return
                yield Path(root) / name


@dataclass(frozen=True)
class Face:
    path: Path
    index: int              # the face's number in a collection (0 for a single font)
    ps_name: str
    family: str             # family_key of the face's names


class ReferenceIndex:
    """The installed faces of the platform's font folders, by family key; built on first use."""

    def __init__(self, dirs: Sequence[Path] | None = None) -> None:
        self._dirs = list(dirs) if dirs is not None else None
        self._faces: dict[str, list[Face]] | None = None
        self._lock = threading.Lock()

    def _build(self) -> dict[str, list[Face]]:
        from fontTools.ttLib import TTCollection, TTFont

        out: dict[str, list[Face]] = {}
        dirs = self._dirs if self._dirs is not None else font_dirs()
        for path in _font_files(dirs):
            try:
                if path.stat().st_size > MAX_REFERENCE_BYTES:
                    continue
                if path.suffix.lower() in (".ttc", ".otc"):
                    fonts = list(TTCollection(str(path), lazy=True).fonts)
                else:
                    fonts = [TTFont(str(path), lazy=True)]
            except Exception:  # noqa: BLE001 - a font we can't open is no reference
                continue
            for i, tt in enumerate(fonts):
                try:
                    name = tt["name"]
                    ps = name.getDebugName(6) or ""
                    fam = name.getDebugName(16) or name.getDebugName(1) or ""
                except Exception:  # noqa: BLE001
                    continue
                keys = {k for k in (family_key(ps), family_key(fam)) if k}
                for k in keys:
                    out.setdefault(k, []).append(Face(path, i, ps, k))
        return out

    def faces(self, family: str) -> list[Face]:
        with self._lock:
            if self._faces is None:
                self._faces = self._build()
            return list(self._faces.get(family, ()))


@dataclass
class Reference:
    """One installed face: per glyph its outline hash and the texts it draws."""

    face: Face
    gid_hash: dict[int, str]                 # glyphs with a text and an outline
    gid_text: dict[int, set[str]]            # cmap and reverse GSUB
    hash_text: dict[str, set[str]]

    def text_at(self, gid: int, outline: str) -> set[str] | None:
        """The texts of glyph ``gid`` when it has outline ``outline`` there, else None."""
        if self.gid_hash.get(gid) == outline:
            return set(self.gid_text.get(gid, ()))
        return None


def _gsub_reverse(tt: Any, gid_text: dict[int, set[str]], names: Mapping[str, int]) -> None:
    """Extend ``gid_text`` with what GSUB produces, read backwards: a single or alternate
    substitution's output draws its input's text, a ligature draws its components' texts joined.

    Only glyphs the cmap doesn't encode get texts this way: an encoded glyph is what its code point
    says (Times New Roman's ``dtls`` turns i into the encoded ı, which read backwards would make ı
    an i; measured 2026-11-10)."""
    if "GSUB" not in tt:
        return
    encoded = frozenset(gid_text)
    try:
        lookups = tt["GSUB"].table.LookupList.Lookup if tt["GSUB"].table.LookupList else []
    except Exception:  # noqa: BLE001
        return
    subtables: list[tuple[int, Any]] = []
    for lookup in lookups:
        for st in getattr(lookup, "SubTable", []) or []:
            kind = lookup.LookupType
            if kind == 7:
                kind = getattr(st, "ExtensionLookupType", 0)
                st = getattr(st, "ExtSubTable", None)
            if kind in (1, 3, 4) and st is not None:
                subtables.append((kind, st))
    for _ in range(4):                       # outputs that feed further lookups (a variant of a ligature)
        changed = False
        for kind, st in subtables:
            pairs: list[tuple[list[str], str]] = []
            if kind == 1:
                pairs = [([a], b) for a, b in (getattr(st, "mapping", None) or {}).items()]
            elif kind == 3:
                for a, outs in (getattr(st, "alternates", None) or {}).items():
                    pairs.extend(([a], b) for b in outs)
            elif kind == 4:
                for first, ligs in (getattr(st, "ligatures", None) or {}).items():
                    for lig in ligs:
                        pairs.append(([first, *lig.Component], lig.LigGlyph))
            for inputs, out in pairs:
                out_gid = names.get(out)
                if out_gid is None or out_gid in encoded:
                    continue
                texts = [gid_text.get(names.get(n, -1)) for n in inputs]
                if any(not t for t in texts):
                    continue
                joined = {"".join(t) for t in _product(texts) if len("".join(t)) <= MAX_DST_CHARS}
                joined = {j for j in joined if usable_text(j)}
                have = gid_text.setdefault(out_gid, set())
                if not joined <= have:
                    have |= joined
                    changed = True
        if not changed:
            break


def _product(sets: Sequence[set[str] | None], cap: int = 8) -> Iterator[tuple[str, ...]]:
    """Every combination of one text per set, at most ``cap`` of them."""
    combos: list[tuple[str, ...]] = [()]
    for s in sets:
        combos = [c + (t,) for c in combos for t in sorted(s or ())][:cap]
    yield from combos


def load_reference(face: Face) -> Reference | None:
    """A face's glyphs with their texts; None when it can't be read."""
    from fontTools.ttLib import TTFont

    try:
        tt = TTFont(str(face.path), fontNumber=face.index, lazy=True)
        order = tt.getGlyphOrder()
        if len(order) > MAX_GLYPHS:
            return None
        names = {n: i for i, n in enumerate(order)}
        gid_text: dict[int, set[str]] = {}
        cmap = tt.getBestCmap() or {}
        for table in tt["cmap"].tables if "cmap" in tt else ():
            if table.isUnicode():
                for cp, gname in table.cmap.items():
                    if 0 < cp <= 0x10FFFF and usable_text(chr(cp)) and gname in names:
                        gid_text.setdefault(names[gname], set()).add(chr(cp))
        for cp, gname in cmap.items():
            if 0 < cp <= 0x10FFFF and usable_text(chr(cp)) and gname in names:
                gid_text.setdefault(names[gname], set()).add(chr(cp))
        _gsub_reverse(tt, gid_text, names)
        glyphset = tt.getGlyphSet()
    except Exception:  # noqa: BLE001 - a face we can't read is no reference
        return None
    gid_hash: dict[int, str] = {}
    hash_text: dict[str, set[str]] = {}
    for gid, texts in gid_text.items():
        try:
            h = draw_hash(glyphset, order[gid])
        except OutlineError:
            continue
        if h is None:
            continue
        gid_hash[gid] = h
        hash_text.setdefault(h, set()).update(texts)
    return Reference(face, gid_hash, gid_text, hash_text)


# --------------------------------------------------------------------------- ToUnicode CMaps

class CMapError(Exception):
    """A /ToUnicode stream that can't be used (too large, malformed)."""


_HEX_RE = re.compile(rb"<([0-9A-Fa-f\s]*)>")
_BLOCK_RE = re.compile(rb"begin(bfchar|bfrange|codespacerange)(.*?)end\1", re.S)


def _hex(token: bytes) -> bytes:
    digits = re.sub(rb"\s+", b"", token)
    if len(digits) % 2:
        digits += b"0"
    return bytes.fromhex(digits.decode("ascii"))


def _utf16(data: bytes) -> str | None:
    if len(data) % 2:
        return None
    try:
        return data.decode("utf-16-be")
    except UnicodeDecodeError:
        return None


def parse_cmap(data: bytes) -> dict[int, str | None]:
    """code -> text from a ToUnicode CMap's ``bfchar`` and ``bfrange`` blocks.

    A destination that isn't valid UTF-16 maps to None. CMapError when the stream passes a limit.
    """
    if len(data) > MAX_CMAP_BYTES:
        raise CMapError("cmap_too_large")
    out: dict[int, str | None] = {}
    count = 0
    for kind, body in _BLOCK_RE.findall(data):
        if kind == b"codespacerange":
            continue
        tokens = re.findall(rb"<[0-9A-Fa-f\s]*>|\[|\]", body)
        i = 0
        if kind == b"bfchar":
            while i + 1 < len(tokens):
                src, dst = tokens[i], tokens[i + 1]
                i += 2
                if not (src.startswith(b"<") and dst.startswith(b"<")):
                    continue
                out[int.from_bytes(_hex(src[1:-1]) or b"\0", "big")] = _utf16(_hex(dst[1:-1]))
                count += 1
        else:
            while i + 2 < len(tokens):
                lo, hi = tokens[i], tokens[i + 1]
                if not (lo.startswith(b"<") and hi.startswith(b"<")):
                    i += 1
                    continue
                a = int.from_bytes(_hex(lo[1:-1]) or b"\0", "big")
                b = int.from_bytes(_hex(hi[1:-1]) or b"\0", "big")
                if b < a or b - a >= MAX_RANGE:
                    raise CMapError("cmap_range")
                if tokens[i + 2] == b"[":
                    j = i + 3
                    k = 0
                    while j < len(tokens) and tokens[j] != b"]":
                        if a + k <= b:
                            out[a + k] = _utf16(_hex(tokens[j][1:-1]))
                        k += 1
                        j += 1
                    i = j + 1
                    count += k
                else:
                    first = _utf16(_hex(tokens[i + 2][1:-1]))
                    for k in range(b - a + 1):
                        if first is None or not first:
                            out[a + k] = None
                        else:
                            last = ord(first[-1]) + k
                            out[a + k] = first[:-1] + chr(last) if last <= 0x10FFFF else None
                    count += b - a + 1
                    i += 3
                if count > MAX_CMAP_ENTRIES:
                    raise CMapError("cmap_too_many")
        if count > MAX_CMAP_ENTRIES:
            raise CMapError("cmap_too_many")
    return out


def cmap_bytes(entries: Mapping[int, str], *, two_byte: bool) -> bytes:
    """A ToUnicode CMap holding ``entries`` (code -> text) as ``bfchar`` blocks of at most 100."""
    width = 4 if two_byte else 2
    lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
             "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def", "1 begincodespacerange",
             f"<{0:0{width}X}> <{(0xFFFF if two_byte else 0xFF):0{width}X}>", "endcodespacerange"]
    items = sorted(entries.items())
    for start in range(0, len(items), 100):
        chunk = items[start:start + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        lines.extend(f"<{code:0{width}X}> <{text.encode('utf-16-be').hex().upper()}>" for code, text in chunk)
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return ("\n".join(lines) + "\n").encode("ascii")


# --------------------------------------------------------------------------- scripts and lookalikes

def script_of(ch: str) -> str | None:
    """``LATIN``, ``GREEK``, ``CYRILLIC`` or ``THAI`` for a letter of one of those scripts, else None."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    if not unicodedata.category(ch).startswith("L"):
        return "THAI" if name.startswith("THAI ") else None
    for script in ("LATIN", "GREEK", "CYRILLIC", "THAI"):
        if name.startswith(script + " "):
            return script
    if "GREEK" in name:
        return "GREEK"
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "LATIN" in name:
        return "LATIN"
    return None


# Single-letter homoglyphs between Latin, Greek and Cyrillic (DESIGN §15.3). Each group is one shape;
# a letter's twins are the other members of its own groups, one step, never chained.
_TWIN_GROUPS = (
    "aа", "cс", "eе", "oоο", "pрρ", "xхχ", "yу", "sѕ", "iі", "jј", "hһ", "ıι", "vν",
    "AАΑ", "BВΒ", "CС", "EЕΕ", "HНΗ", "IІΙӀ", "JЈ", "KКΚ", "MМΜ", "NΝ", "OОΟ", "PРΡ", "SЅ", "TТΤ", "XХΧ",
    "YΥҮ", "ZΖ", "ƂБ", "ᴦг",
)
TWINS: dict[str, frozenset[str]] = {}
for _group in _TWIN_GROUPS:
    for _ch in _group:
        TWINS[_ch] = frozenset(TWINS.get(_ch, frozenset()) | (set(_group) - {_ch}))


def skeleton(text: str) -> str:
    """Case-folded, marks removed, dotless i as i: what two reads of one shape have in common."""
    folded = unicodedata.normalize("NFD", text.replace("ı", "i").replace("İ", "I")).casefold()
    return "".join(ch for ch in folded if not unicodedata.category(ch).startswith("M"))
