"""Writing systems: which script a character belongs to, script runs, and per-script trigrams.

``SCRIPTS`` is the table of scripts this core knows. A pack names scripts by these keys (its
``scripts``, ``detect.osd_scripts``, a rule's ``script`` parameter), the loader checks them here,
and the layer writer asks each font to cover a script's ``required`` code points.

Each ``Script`` carries:

* ``ranges``: the code points that belong to the script (closed intervals, sorted). The script's
  own combining marks are inside them (Thai tone marks are Thai). Generic combining marks that
  belong to no script (U+0300 to U+036F, for example) are "Inherited": they take the script of
  the character they sit on.
* ``trigram_ranges``: the code points that count for trigram corroboration. For Thai that is
  U+0E01 to U+0E4F, so Thai digits never count, and the baht sign (U+0E3F) does: it is Common
  (``script_of`` gives None) but sits inside Thai words as readers write them. A trigram range
  may reach outside ``ranges`` only to such a symbol (Unicode category S*) of no script.
* ``spaced``: whether the script separates words with spaces.
* ``osd_names``: the names Tesseract's OSD reports for it ("Fraktur" is Latin, "Korean" is Hangul,
  "Japanese" is reported as Hiragana here because Han alone can't tell Japanese from Chinese).
* ``required``: the code points a layer font must cover to claim the script, so a font that only
  calls itself a Thai font is refused.

A character outside every table (digits, punctuation, spaces and symbols: Unicode's "Common") has
no script: ``script_of`` returns None.
"""

from __future__ import annotations

import bisect
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

INHERITED = "Inherited"
COMMON = "Common"

Ranges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Script:
    name: str
    ranges: Ranges
    trigram_ranges: Ranges
    spaced: bool
    osd_names: tuple[str, ...]
    required: frozenset[int]          # code points a layer font must cover

    def __post_init__(self) -> None:
        for label, rs in (("ranges", self.ranges), ("trigram_ranges", self.trigram_ranges)):
            prev = -1
            for lo, hi in rs:
                if not (0 <= lo <= hi <= 0x10FFFF) or lo <= prev:
                    raise ValueError(f"script {self.name}: {label} must be sorted, disjoint closed intervals")
                prev = hi
        for cp in _excess(self.trigram_ranges, self.ranges):
            if not unicodedata.category(chr(cp)).startswith("S"):
                raise ValueError(f"script {self.name}: trigram code point U+{cp:04X} is outside its ranges "
                                 "and not a symbol")
        outside = [cp for cp in self.required if not _in(cp, self.ranges)]
        if outside:
            raise ValueError(f"script {self.name}: required code points outside its ranges: "
                             + ", ".join(f"U+{cp:04X}" for cp in sorted(outside)[:8]))

    def contains(self, ch: str) -> bool:
        return len(ch) == 1 and _in(ord(ch), self.ranges)

    def counts_for_trigrams(self, ch: str) -> bool:
        return len(ch) == 1 and _in(ord(ch), self.trigram_ranges)


def _in(cp: int, ranges: Ranges) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def _excess(rs: Ranges, inside: Ranges) -> list[int]:
    """The code points of ``rs`` not in ``inside``; refuses a large excess rather than walking it."""
    out: list[int] = []
    for lo, hi in rs:
        cp = lo
        while cp <= hi:
            cover = next(((a, b) for a, b in inside if a <= cp <= b), None)
            if cover is not None:
                cp = cover[1] + 1
                continue
            out.append(cp)
            if len(out) > 64:
                raise ValueError(f"trigram range {lo:04X}-{hi:04X} reaches far outside its script's ranges")
            cp += 1
    return out


def _cps(*spans: tuple[int, int] | str) -> frozenset[int]:
    out: set[int] = set()
    for span in spans:
        if isinstance(span, str):
            out.update(ord(c) for c in span)
        else:
            out.update(range(span[0], span[1] + 1))
    return frozenset(out)


def _without(cps: Iterable[int], drop: Iterable[int]) -> frozenset[int]:
    return frozenset(set(cps) - set(drop))


_LATIN_LETTERS: Ranges = (
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00AA, 0x00AA), (0x00BA, 0x00BA),
    (0x00C0, 0x00D6), (0x00D8, 0x00F6), (0x00F8, 0x02AF), (0x1D00, 0x1D25), (0x1D2C, 0x1D5C),
    (0x1D62, 0x1D65), (0x1D6B, 0x1D77), (0x1D79, 0x1DBE), (0x1E00, 0x1EFF), (0x2071, 0x2071),
    (0x207F, 0x207F), (0x2090, 0x209C), (0x212A, 0x212B), (0x2132, 0x2132), (0x214E, 0x214E),
    (0x2160, 0x2188), (0x2C60, 0x2C7F), (0xA722, 0xA787), (0xA78B, 0xA7FF), (0xAB30, 0xAB5A),
    (0xAB5C, 0xAB64), (0xFB00, 0xFB06), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A),
)

_SCRIPT_LIST: tuple[Script, ...] = (
    Script(
        name="Latin",
        ranges=_LATIN_LETTERS,
        trigram_ranges=_LATIN_LETTERS,
        spaced=True,
        osd_names=("Latin", "Fraktur"),
        # Basic Latin and Latin-1 letters plus Latin Extended-A (Central European letters: ő, ű, č, ł ...)
        required=_without(_cps((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0xFF), (0x0100, 0x017F)), (0xD7, 0xF7)),
    ),
    Script(
        name="Thai",
        ranges=((0x0E01, 0x0E3A), (0x0E40, 0x0E5B)),       # U+0E3F (the baht sign) is Common
        # U+0E01-U+0E4F less the unassigned U+0E3B-U+0E3E: the baht sign counts; no Thai digits, no ๚ ๛
        trigram_ranges=((0x0E01, 0x0E3A), (0x0E3F, 0x0E4F)),
        spaced=False,
        osd_names=("Thai",),
        required=_cps((0x0E01, 0x0E3A), (0x0E40, 0x0E5B)),
    ),
    Script(
        name="Greek",
        ranges=((0x0370, 0x0373), (0x0375, 0x0377), (0x037A, 0x037D), (0x037F, 0x037F), (0x0384, 0x0384),
                (0x0386, 0x0386), (0x0388, 0x038A), (0x038C, 0x038C), (0x038E, 0x03A1), (0x03A3, 0x03E1),
                (0x03F0, 0x03FF), (0x1F00, 0x1FFF)),
        trigram_ranges=((0x0386, 0x0386), (0x0388, 0x038A), (0x038C, 0x038C), (0x038E, 0x03A1),
                        (0x03A3, 0x03E1), (0x03F0, 0x03FF), (0x1F00, 0x1FFF)),
        spaced=True,
        osd_names=("Greek",),
        required=_cps((0x0391, 0x03A1), (0x03A3, 0x03A9), (0x03B1, 0x03C9), (0x0386, 0x0386),
                      (0x0388, 0x038A), (0x038C, 0x038C), (0x038E, 0x0390), (0x03AA, 0x03B0), (0x03CA, 0x03CE)),
    ),
    Script(
        name="Cyrillic",
        ranges=((0x0400, 0x0484), (0x0487, 0x052F), (0x1C80, 0x1C88), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
        trigram_ranges=((0x0400, 0x0481), (0x048A, 0x052F)),
        spaced=True,
        osd_names=("Cyrillic",),
        required=_cps((0x0400, 0x045F)),
    ),
    Script(
        name="Hebrew",
        ranges=((0x0591, 0x05C7), (0x05D0, 0x05EA), (0x05EF, 0x05F4), (0xFB1D, 0xFB4F)),
        trigram_ranges=((0x05D0, 0x05EA),),
        spaced=True,
        osd_names=("Hebrew",),
        required=_cps((0x05D0, 0x05EA)),
    ),
    Script(
        name="Arabic",
        ranges=((0x0600, 0x0604), (0x0606, 0x060B), (0x060D, 0x061A), (0x061C, 0x061E), (0x0620, 0x063F),
                (0x0641, 0x064A), (0x0656, 0x066F), (0x0671, 0x06DC), (0x06DE, 0x06FF), (0x0750, 0x077F),
                (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFC)),
        trigram_ranges=((0x0620, 0x063F), (0x0641, 0x064A), (0x0671, 0x06D3), (0x0750, 0x077F)),
        spaced=True,
        osd_names=("Arabic",),
        required=_cps((0x0621, 0x063A), (0x0641, 0x064A)),
    ),
    Script(
        name="Devanagari",
        ranges=((0x0900, 0x0950), (0x0955, 0x0963), (0x0966, 0x097F), (0xA8E0, 0xA8FF)),
        trigram_ranges=((0x0900, 0x0950), (0x0955, 0x0963), (0x0971, 0x097F)),
        spaced=True,
        osd_names=("Devanagari",),
        required=_cps((0x0905, 0x0939), (0x093E, 0x094D)),
    ),
    Script(
        name="Han",
        ranges=((0x2E80, 0x2E99), (0x2E9B, 0x2EF3), (0x2F00, 0x2FD5), (0x3005, 0x3005), (0x3007, 0x3007),
                (0x3021, 0x3029), (0x3038, 0x303B), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
                (0x20000, 0x2FA1F)),
        trigram_ranges=((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)),
        spaced=False,
        osd_names=("Han",),
        # A small core of very common ideographs: a full CJK font covers far more, any CJK font covers these.
        required=_cps("一二三四五六七八九十人大中日月年上下山水"),
    ),
    Script(
        name="Hiragana",
        ranges=((0x3041, 0x3096), (0x309D, 0x309F)),
        trigram_ranges=((0x3041, 0x3096),),
        spaced=False,
        osd_names=("Hiragana", "Japanese"),
        required=_cps((0x3041, 0x3093)),
    ),
    Script(
        name="Katakana",
        ranges=((0x30A1, 0x30FA), (0x30FD, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF6F), (0xFF71, 0xFF9D)),
        trigram_ranges=((0x30A1, 0x30FA),),
        spaced=False,
        osd_names=("Katakana",),
        required=_cps((0x30A1, 0x30F6)),
    ),
    Script(
        name="Hangul",
        ranges=((0x1100, 0x11FF), (0x3131, 0x318E), (0xA960, 0xA97C), (0xAC00, 0xD7A3), (0xD7B0, 0xD7C6),
                (0xD7CB, 0xD7FB), (0xFFA0, 0xFFDC)),
        trigram_ranges=((0xAC00, 0xD7A3),),
        spaced=True,
        osd_names=("Hangul", "Korean"),
        required=_cps((0xAC00, 0xD7A3)),
    ),
)

SCRIPTS: Mapping[str, Script] = MappingProxyType({s.name: s for s in _SCRIPT_LIST})

# One sorted table of (start, end, script name) for bisect lookups.
_TABLE: list[tuple[int, int, str]] = sorted((lo, hi, s.name) for s in _SCRIPT_LIST for lo, hi in s.ranges)
_STARTS: list[int] = [lo for lo, _, _ in _TABLE]


def _check_table() -> None:
    prev_hi, prev_name = -1, ""
    for lo, hi, name in _TABLE:
        if lo <= prev_hi:
            raise ValueError(f"scripts {prev_name} and {name} overlap at U+{lo:04X}")
        prev_hi, prev_name = hi, name
    for sc in _SCRIPT_LIST:                 # a trigram code point outside the ranges belongs to no script
        for cp in _excess(sc.trigram_ranges, sc.ranges):
            if any(lo <= cp <= hi for lo, hi, _ in _TABLE):
                raise ValueError(f"script {sc.name}: trigram code point U+{cp:04X} belongs to another script")


_check_table()


def script_of(ch: str) -> str | None:
    """The script ``ch`` belongs to, ``"Inherited"`` for a combining mark of no script, else None.

    None means Unicode's Common (digits, punctuation, spaces, symbols) or a script this core
    doesn't know.
    """
    if len(ch) != 1:
        raise ValueError(f"script_of takes one character, got {len(ch)}")
    cp = ord(ch)
    i = bisect.bisect_right(_STARTS, cp) - 1
    if i >= 0:
        lo, hi, name = _TABLE[i]
        if lo <= cp <= hi:
            return name
    if unicodedata.category(ch) in ("Mn", "Me", "Mc") or ch in ("‌", "‍"):
        return INHERITED
    return None


def script_runs(text: str) -> list[tuple[str, str]]:
    """``text`` cut into (script, run) pieces whose concatenation is ``text``.

    Characters of no script (spaces, digits, punctuation) and Inherited marks never start a new
    run: they stay with the run before them, or with the first script run when the text starts
    with them. A text with no script character at all is one ``"Common"`` run.
    """
    runs: list[list[str]] = []          # [script or None, accumulated text]
    for ch in text:
        s = script_of(ch)
        if s is None or s == INHERITED:
            if runs:
                runs[-1][1] += ch
            else:
                runs.append([None, ch])  # type: ignore[list-item]
            continue
        if runs and runs[-1][0] is None:
            runs[-1][0] = s              # leading neutral characters join the first script run
            runs[-1][1] += ch
        elif runs and runs[-1][0] == s:
            runs[-1][1] += ch
        else:
            runs.append([s, ch])
    return [(s if s is not None else COMMON, t) for s, t in runs]


def script_text(text: str, script: str) -> str:
    """The characters of ``text`` in ``script`` (with the Inherited marks on them), runs joined by one space."""
    _get(script)
    pieces: list[str] = []
    cur: list[str] = []
    in_run = False
    for ch in text:
        s = script_of(ch)
        if s == script or (s == INHERITED and in_run):
            cur.append(ch)
            in_run = True
        else:
            if cur:
                pieces.append("".join(cur))
                cur = []
            in_run = False
    if cur:
        pieces.append("".join(cur))
    return " ".join(pieces)


def trigrams(text: str, script: str) -> Counter[str]:
    """Character trigrams of ``text``'s trigram characters in ``script``, its runs joined.

    The text is NFKC-normalised first, so a character that readers write in different
    compositions (Thai sara am as U+0E33 or as nikhahit + sara aa) compares the same; spacing
    and line breaks differ between readers, so runs are joined with nothing between them.
    """
    sc = _get(script)
    t = "".join(ch for ch in unicodedata.normalize("NFKC", text or "") if sc.counts_for_trigrams(ch))
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def corroborated(mine: Counter[str], others: set[str]) -> int:
    """How many of ``mine``'s trigrams, with multiplicity, appear at least once in ``others``."""
    return sum(n for g, n in mine.items() if g in others)


def _get(script: str) -> Script:
    try:
        return SCRIPTS[script]
    except KeyError:
        raise KeyError(f"unknown script {script!r}; this core knows {', '.join(SCRIPTS)}") from None
