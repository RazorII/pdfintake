"""Pass-1 detection: the signals ``langplan.plan`` decides each page's language pack from.

For every page that will be OCR'd, the pipeline gathers (``PageSignals``):

* ``script``: ``script_call`` over OSD's result on the about-300-dpi render and the mean confidence
  of Tesseract's first ``eng`` read. A non-Latin OSD script at a confidence of at least
  ``langplan.OSD_NONLATIN_MIN_CONF`` names the page's script; a page counts as Latin only when the
  ``eng`` read's mean confidence is at least ``langplan.LATIN_ENG_MIN_MEANCONF``; anything else is
  left undecided (``by="none"``) for the document vote in ``langplan``.
* ``windows``: ``language_windows`` cuts the ``eng`` read's Latin-script words (word confidence of
  at least ``WORD_MIN_CONF``) into windows of about 120 characters and classifies each with a
  registered language identifier (``IDENTIFIERS``; py3langid by default).
* ``function_word_score``: function-word counts of a read against a pack's word and rival lists
  (the function-word page score; the lists are pack data).
* ``probes``: ``run_probe`` reads the page with an installed pack's cheap probe and counts the
  characters of the pack's script on confident lines written mostly in that script (at least
  ``PROBE_LINE_MIN_SHARE`` of a line's letters). Lines are never swapped into a read.
* ``mixed``: ``mixed_script_check``, the pack-independent mixed-script trigger. Low-confidence
  blocks of the ``eng`` read are cropped, scaled to about 300 dpi and given to OSD; a non-Latin
  script at the calibrated confidence is reported. Without a calibration the trigger is off and
  only counts the blocks below ``UNCALIBRATED_BLOCK_MAX_CONF`` (``langplan`` turns that count
  into the ``mixed_script_unchecked`` note).

Scales. ``Line.conf`` and ``Word.conf`` are in [0, 1]. ``OsdResult.script_conf`` is OSD's own
scale (the 5.0 threshold). ``eng_mean_conf`` is on Tesseract's 0-100 scale (100 times the [0, 1]
mean): ``read_mean_conf`` computes it from a read.

Nothing here keeps or reports document text: the results are counts, codes and probabilities.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import EngineFailure, PackError, SetupError
from ..protocols import LanguageIdentifier, OsdSession
from ..types import EngineRead, Line, OsdResult, PageImage, ScriptCall

if TYPE_CHECKING:
    from .engines import EngineSession
    from .langplan import Calibration

# --------------------------------------------------------------------------- constants

WORD_MIN_CONF = 0.4                 # a word counts for the language windows at this confidence or more
OSD_TARGET_DPI = 300.0              # OSD is unreliable on coarser renders (measured)
PROBE_TIMEOUT_S = 120.0             # one probe read of one page
# A probe line counts only when its script's characters are at least this share of the line's
# letters (its own script's characters plus the letters of every other script; digits, punctuation
# and unattached marks count for neither). A probe's recogniser reads accented Latin letters as
# stray characters of its own script: on the Latin kit pages (Vietnamese above all) the Thai probe
# found 2 to 10 Thai characters on confident lines of 9 to 171 letters, never more than a third of
# a line, and on a dense Vietnamese page they added up to 21 to 77, past min_chars. Every confident
# line on the Thai kit pages, plain and shaped, was at least 0.875 Thai (measured 2026-11-06). Half
# sits in that gap, keeps a short Thai line on a Latin page (all of it is Thai), and is the same
# rule for every probe pack.
PROBE_LINE_MIN_SHARE = 0.5
UNCALIBRATED_BLOCK_MAX_CONF = 0.5   # "an eng block below 50 mean confidence" (DESIGN §3.5)
BLOCK_PAD_FRACTION = 0.25           # a block crop is padded by this share of its height (at least 2 px)

# The function-word page score: English function words never count against a page in the
# document rule (``other + en <= 2``), so they are core logic, not pack data.
EN_WORDS: frozenset[str] = frozenset(
    "the and of to is for that with this are you your from by be on".split())

_WORD_RE = re.compile(r"[^\W\d_]+")   # a run of letters (the page score's tokeniser)


# --------------------------------------------------------------------------- language identifiers

class Py3LangId:
    """py3langid 0.4.0 with normalised probabilities (its model ships inside the package and is
    checked against ``PY3LANGID_MODEL_SHA256`` before it is parsed).

    The model loads on the first ``classify`` call, once per process, under a lock. Classification
    only reads the model, so one instance serves every thread.
    """

    name: ClassVar[str] = "py3langid"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ident: Any = None

    def _identifier(self) -> Any:
        if self._ident is None:
            with self._lock:
                if self._ident is None:
                    self._ident = _load_py3langid()
        return self._ident

    def classify(self, text: str) -> list[tuple[str, float]]:
        """Every language the model knows, with its probability, best first. Empty text gives []."""
        if not text or not text.strip():
            return []
        ranked = self._identifier().rank(text)
        return [(str(code), _unit(prob)) for code, prob in ranked]


# py3langid 0.4.0's bundled model (``py3langid/data/model.npz.xz``), pinned like every other model
# and checked before it is parsed (DESIGN §11.2, decided 2026-11-05). Measured from the installed
# 0.4.0 wheel; it agrees with the wheel's own RECORD entry.
PY3LANGID_VERSION = "0.4.0"
PY3LANGID_MODEL_SHA256 = "f4f4a2c3465ca1f081541037f9cac23021d68151c2f709c55ae5eedfca522963"
PY3LANGID_MODEL_SIZE = 4_586_720


def _installed_py3langid_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("py3langid")
    except metadata.PackageNotFoundError:
        return "unknown"


def _load_py3langid() -> Any:
    """py3langid's bundled model, checked against its pinned size and SHA-256, then parsed.

    The model file is read once; the hash is taken over those bytes and the identifier is built
    from the same bytes, so what is parsed is what was checked. A missing, truncated, unreadable or
    different model is a SetupError naming the file.
    """
    import hashlib
    import io
    import lzma
    from pathlib import Path

    try:
        import numpy as np
        from py3langid.langid import MODEL_DIR, MODEL_FILE, LanguageIdentifier as _LangId
        from py3langid.modelio import load_model
    except ImportError as exc:
        raise SetupError("identifier_broken", f"the language identifier py3langid isn't installed ({type(exc).__name__})",
                         identifier="py3langid") from None
    path = Path(MODEL_FILE)
    if not path.is_absolute():                      # as py3langid's own from_model_file resolves it
        path = Path(MODEL_DIR) / path
    reinstall = f"reinstall pdfintake, which needs py3langid {PY3LANGID_VERSION}"
    try:
        with open(path, "rb") as fh:
            raw = fh.read(PY3LANGID_MODEL_SIZE + 1)
    except OSError as exc:
        raise SetupError("identifier_broken",
                         f"the language identifier py3langid can't read its model ({type(exc).__name__}); {reinstall}",
                         identifier="py3langid", file=str(path)) from None
    got = hashlib.sha256(raw).hexdigest()
    if len(raw) != PY3LANGID_MODEL_SIZE or got != PY3LANGID_MODEL_SHA256:
        raise SetupError("identifier_broken",
                         f"the language identifier py3langid's model at {path} isn't the one pdfintake pins for "
                         f"py3langid {PY3LANGID_VERSION} (installed: {_installed_py3langid_version()}; "
                         f"{len(raw)} bytes, SHA-256 {got}; expected {PY3LANGID_MODEL_SIZE} bytes, SHA-256 "
                         f"{PY3LANGID_MODEL_SHA256}); {reinstall}",
                         identifier="py3langid", file=str(path), sha256=got, expected_sha256=PY3LANGID_MODEL_SHA256)
    try:
        ptc, pc, classes, nextmove, row, output = load_model(io.BytesIO(raw))
        # The body of py3langid 0.4.0's LanguageIdentifier.from_model_file, fed the checked bytes.
        return _LangId(np.asarray(ptc), np.asarray(pc), classes, nextmove, output, norm_probs=True, tk_row=row)
    except (OSError, EOFError, ValueError, KeyError, TypeError, lzma.LZMAError) as exc:
        raise SetupError("identifier_broken",
                         f"the language identifier py3langid can't load its model ({type(exc).__name__}); {reinstall}",
                         identifier="py3langid", file=str(path)) from None


_IDENTIFIERS: dict[str, LanguageIdentifier] = {"py3langid": Py3LangId()}
IDENTIFIERS: Mapping[str, LanguageIdentifier] = MappingProxyType(_IDENTIFIERS)
DEFAULT_IDENTIFIER = "py3langid"


def _identifier(name: str) -> LanguageIdentifier:
    try:
        return IDENTIFIERS[name]
    except KeyError:
        raise PackError("unknown_identifier",
                        f"unknown language identifier {name!r}; this core knows {', '.join(sorted(IDENTIFIERS))}",
                        name=name) from None


# --------------------------------------------------------------------------- results

@dataclass(frozen=True)
class LangWindow:
    start: int          # character offset of the window in the read's joined Latin words
    code: str           # the identifier's best language code for the window
    prob: float


@dataclass(frozen=True)
class ProbeResult:
    pack: str
    chars: int          # characters of the pack's script on qualifying lines
    lines: int          # qualifying lines: enough script characters at enough confidence
    # Beyond DESIGN §2.2 (additive): the installed manifest's ``probe.min_chars``, which the index's
    # detect copy reduces to ``"probe": true``. ``langplan`` compares ``chars`` with it when set.
    min_chars: int | None = None


@dataclass(frozen=True)
class FWScore:
    hits: int           # tokens in the pack's function words
    rival_hits: int     # tokens in the pack's rival words
    en_hits: int        # tokens in EN_WORDS
    words: int          # all tokens


@dataclass(frozen=True)
class MixedScript:
    checked: bool                 # False when there is no calibration (the trigger is off)
    script: str | None            # the non-Latin script found, or None
    blocks: int                   # blocks given to OSD; unchecked: eng blocks below UNCALIBRATED_BLOCK_MAX_CONF
    conf: float | None            # OSD's script confidence for ``script``


@dataclass
class PageSignals:
    index: int
    osd: OsdResult | None
    script: ScriptCall
    eng_read: EngineRead
    checker_read: EngineRead | None
    probes: dict[str, ProbeResult] = field(default_factory=dict)
    windows: list[LangWindow] = field(default_factory=list)
    mixed: MixedScript = field(default_factory=lambda: MixedScript(False, None, 0, None))


# --------------------------------------------------------------------------- scripts

def _script_name(osd_name: str | None) -> str | None:
    """The core script name for a name OSD reports ("Fraktur" -> "Latin"); unknown names pass through."""
    if not osd_name:
        return None
    from . import scripts

    if osd_name in scripts.SCRIPTS:
        return osd_name
    for sc in scripts.SCRIPTS.values():
        if osd_name in sc.osd_names:
            return sc.name
    return osd_name


def read_mean_conf(read: EngineRead | None) -> float | None:
    """A read's mean confidence on Tesseract's 0-100 scale, for ``script_call``; None without text.

    ``meta["mean_conf"]`` (the engine's own mean, in [0, 1]) when the engine gave one, otherwise the
    mean of the word confidences, or of the line confidences for lines without words.
    """
    if read is None or read.status != "ok" or not read.lines:
        return None
    meta = read.meta.get("mean_conf") if isinstance(read.meta, Mapping) else None
    if isinstance(meta, (int, float)) and not isinstance(meta, bool) and math.isfinite(meta):
        return 100.0 * _unit(float(meta))
    confs: list[float] = []
    for ln in read.lines:
        if ln.words:
            confs.extend(w.conf for w in ln.words)
        else:
            confs.append(ln.conf)
    if not confs:
        return None
    return 100.0 * _unit(sum(confs) / len(confs))


def script_call(osd: OsdResult | None, eng_mean_conf: float | None = None, *,
                read: EngineRead | None = None) -> ScriptCall:
    """The page's script from OSD and the ``eng`` read (DESIGN §4.3 step 1).

    * A non-Latin OSD script with ``script_conf >= langplan.OSD_NONLATIN_MIN_CONF`` is the script.
    * Otherwise the page is Latin when ``eng_mean_conf >= langplan.LATIN_ENG_MIN_MEANCONF``
      (0-100 scale).
    * Otherwise it is undecided (``name=None``, ``by="none"``) and takes the document's vote.

    ``eng_mean_conf`` is on Tesseract's 0-100 scale, NOT the [0, 1] ``EngineRead.meta["mean_conf"]``:
    pass the ``eng`` read itself as ``read=`` (beyond DESIGN §2.2, additive) and ``read_mean_conf``
    converts it, or pass ``read_mean_conf(read)``. ``conf`` is always OSD's script confidence (0.0
    without OSD). ``by="osd"`` names step 1's OSD-and-eng gate as a whole: a Latin call made by the
    eng gate alone, with no OSD script, is also ``"osd"``, with ``conf`` 0.0.
    """
    from .langplan import LATIN_ENG_MIN_MEANCONF, OSD_NONLATIN_MIN_CONF

    if read is not None:
        if eng_mean_conf is not None:
            raise ValueError("pass eng_mean_conf or read=, not both")
        eng_mean_conf = read_mean_conf(read)
    if eng_mean_conf is not None:
        if not (isinstance(eng_mean_conf, (int, float)) and math.isfinite(eng_mean_conf)
                and 0.0 <= eng_mean_conf <= 100.0):
            raise ValueError(f"eng_mean_conf is on the 0-100 scale, got {eng_mean_conf!r}")
        eng_mean_conf = float(eng_mean_conf)
    osd_conf = 0.0
    osd_script = None
    if osd is not None:
        osd_conf = float(osd.script_conf) if math.isfinite(osd.script_conf) else 0.0
        osd_script = _script_name(osd.script)
    if osd_script is not None and osd_script != "Latin" and osd_conf >= OSD_NONLATIN_MIN_CONF:
        return ScriptCall(osd_script, osd_conf, "osd", eng_mean_conf)
    if eng_mean_conf is not None and eng_mean_conf >= LATIN_ENG_MIN_MEANCONF:
        return ScriptCall("Latin", osd_conf, "osd", eng_mean_conf)
    return ScriptCall(None, osd_conf, "none", eng_mean_conf)


# --------------------------------------------------------------------------- language windows

def _is_latin_token(token: str) -> bool:
    """True when ``token`` has a letter and every letter in it is Latin script (marks may sit on them)."""
    from . import scripts

    has_letter = False
    for ch in token:
        if not unicodedata.category(ch).startswith("L"):
            continue
        if scripts.script_of(ch) != "Latin":
            return False
        has_letter = True
    return has_letter


def _latin_words(read: EngineRead | None, *, min_conf: float = WORD_MIN_CONF) -> list[str]:
    """The read's Latin-script words with confidence >= ``min_conf``, in reading order.

    A line with word boxes uses each word's confidence; a line without them is split on
    whitespace and every token takes the line's confidence.
    """
    if read is None or read.status != "ok":
        return []
    out: list[str] = []
    for ln in read.lines:
        if ln.words:
            pairs = [(w.text, w.conf) for w in ln.words]
        else:
            pairs = [(tok, ln.conf) for tok in ln.text.split()]
        for text, conf in pairs:
            for tok in text.split():
                if conf >= min_conf and _is_latin_token(tok):
                    out.append(tok)
    return out


def _latin_text(read: EngineRead | None) -> str:
    return " ".join(_latin_words(read))


def _cut_windows(words: Sequence[str], size: int) -> list[tuple[int, str]]:
    """Consecutive windows of whole words, each at least ``size`` characters, as (start, text).

    A tail shorter than ``size`` joins the window before it; a text shorter than ``size`` gives no
    window at all (too little text to call a language).
    """
    out: list[tuple[int, str]] = []
    cur: list[str] = []
    cur_len = 0
    start = 0
    pos = 0                       # offset of the next word in " ".join(words)
    for w in words:
        if not cur:
            start = pos
        cur.append(w)
        cur_len += len(w) + (1 if len(cur) > 1 else 0)
        pos += len(w) + 1
        if cur_len >= size:
            out.append((start, " ".join(cur)))
            cur, cur_len = [], 0
    if cur and out:
        s, t = out[-1]
        out[-1] = (s, t + " " + " ".join(cur))
    return out


def language_windows(read: EngineRead, *, size: int = 120, identifier: str = DEFAULT_IDENTIFIER) -> list[LangWindow]:
    """The read's Latin words cut into windows of about ``size`` characters, each classified.

    Only Latin-script words with confidence >= ``WORD_MIN_CONF`` count. Every window holds at
    least ``size`` characters, so a page with less text has no windows (``langplan`` treats it as
    sparse). Each window keeps its identifier's best code and probability.
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    ident = _identifier(identifier)
    out: list[LangWindow] = []
    for start, text in _cut_windows(_latin_words(read), size):
        ranked = ident.classify(text)
        if not ranked:
            continue
        code, prob = ranked[0]
        out.append(LangWindow(start, str(code), _unit(prob)))
    return out


def page_language(read: EngineRead, *, identifier: str = DEFAULT_IDENTIFIER) -> list[tuple[str, float]]:
    """The identifier's ranking for the whole page's Latin words (the "probability on the page")."""
    return list(_identifier(identifier).classify(_latin_text(read)))


# --------------------------------------------------------------------------- function words

def function_word_score(read: EngineRead, words: frozenset[str], rivals: frozenset[str]) -> FWScore:
    """Function-word counts of a read against a pack's word and rival lists.

    Tokens are runs of letters, lower-cased. ``hits`` counts tokens in ``words``, ``rival_hits``
    tokens in ``rivals``, ``en_hits`` tokens in ``EN_WORDS``, and ``words`` all tokens. Text and
    lists are NFC-normalised first (a decomposed accent would split a word), and a dotted capital
    I lower-cases to a plain i, so an all-caps Turkish page keeps its rival words.
    """
    if read is None or read.status != "ok":
        return FWScore(0, 0, 0, 0)
    wl = frozenset(_fold(w) for w in words)
    rl = frozenset(_fold(w) for w in rivals)
    toks = [t for ln in read.lines for t in _WORD_RE.findall(_fold(ln.text or ""))]
    return FWScore(hits=sum(t in wl for t in toks), rival_hits=sum(t in rl for t in toks),
                   en_hits=sum(t in EN_WORDS for t in toks), words=len(toks))


def _fold(text: str) -> str:
    """NFC, then lower case with U+0130 (dotted capital I) as a plain i."""
    return unicodedata.normalize("NFC", str(text)).replace("\u0130", "i").lower()


# --------------------------------------------------------------------------- probes

def _line_script_counts(text: str, script: str) -> tuple[int, int]:
    """(characters of ``script``, letters of every other script) in ``text``.

    Characters of no script (digits, punctuation, spaces) and Inherited marks count for neither.
    """
    from . import scripts

    own = other = 0
    for ch in text:
        s = scripts.script_of(ch)
        if s == script:
            own += 1
        elif s is not None and s != scripts.INHERITED:
            other += 1
    return own, other


def run_probe(page: PageImage, session: "EngineSession", *, pack: str, script: str,
              min_conf: float, min_line_chars: int, min_chars: int | None = None,
              min_line_share: float = PROBE_LINE_MIN_SHARE) -> ProbeResult:
    """Read ``page`` with a pack's probe and count its script on confident lines written in it.

    A line qualifies with at least ``min_line_chars`` characters of ``script``, a confidence of at
    least ``min_conf``, and ``script`` making up at least ``min_line_share`` of its letters (its
    characters of ``script`` plus its letters of every other script; ``PROBE_LINE_MIN_SHARE`` by
    default, 0 turns the share off). ``chars`` sums the qualifying lines' script characters
    (``langplan`` compares it with the pack's ``min_chars``). The share keeps the stray script
    characters a probe's recogniser reads into accented Latin text (one or a few per line) from
    adding up on a dense Latin page, while a line of the pack's own script, however short, counts.
    Pass the installed manifest's ``probe.min_chars`` as ``min_chars`` (beyond DESIGN §2.2,
    additive): the index only says that a pack has a probe, so without it ``langplan`` falls back
    to ``PROBE_MIN_CHARS_DEFAULT``. A probe that fails, times out or raises ``EngineFailure`` finds
    nothing: the mixed-script trigger still covers the page. The probe's read is never merged into
    another read.
    """
    from . import scripts

    if script not in scripts.SCRIPTS:
        raise ValueError(f"unknown script {script!r}")
    if not (0.0 <= float(min_conf) <= 1.0):
        raise ValueError(f"min_conf must be in [0, 1], got {min_conf!r}")
    if int(min_line_chars) < 1:
        raise ValueError(f"min_line_chars must be at least 1, got {min_line_chars!r}")
    if min_chars is not None and (isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1):
        raise ValueError(f"min_chars must be a positive integer, got {min_chars!r}")
    if (isinstance(min_line_share, bool) or not isinstance(min_line_share, (int, float))
            or not math.isfinite(min_line_share) or not 0.0 <= min_line_share <= 1.0):
        raise ValueError(f"min_line_share must be a number in [0, 1], got {min_line_share!r}")
    try:
        read = session.read(page, timeout_s=PROBE_TIMEOUT_S)
    except EngineFailure:
        return ProbeResult(pack, 0, 0, min_chars)
    if read.status != "ok":
        return ProbeResult(pack, 0, 0, min_chars)
    chars = lines = 0
    for ln in read.lines:
        n, other = _line_script_counts(ln.text or "", script)
        if n >= min_line_chars and ln.conf >= min_conf and n >= min_line_share * (n + other) - 1e-9:
            chars += n
            lines += 1
    return ProbeResult(pack, chars, lines, min_chars)


# --------------------------------------------------------------------------- mixed-script trigger

@dataclass
class _Block:
    x0: float
    y0: float
    x1: float
    y1: float
    confs: list[float]
    order: int

    @property
    def conf(self) -> float:
        return sum(self.confs) / len(self.confs)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


def _bbox(line: Line) -> tuple[float, float, float, float] | None:
    xs = [p[0] for p in line.box]
    ys = [p[1] for p in line.box]
    if not all(math.isfinite(v) for v in xs + ys):
        return None
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return x0, y0, x1, y1


def _line_block_ids(read: EngineRead) -> list[int] | None:
    """Tesseract's block id per line (``meta["line_blocks"]``) when it lines up with the read's lines."""
    ids = read.meta.get("line_blocks") if isinstance(read.meta, Mapping) else None
    if not isinstance(ids, (list, tuple)) or len(ids) != len(read.lines):
        return None
    if not all(isinstance(b, int) and not isinstance(b, bool) for b in ids):
        return None
    return list(ids)


def _low_blocks(read: EngineRead | None, max_conf: float) -> list[_Block]:
    """Runs of consecutive ``read`` lines below ``max_conf``, merged into blocks.

    When the read carries Tesseract's block ids (``meta["line_blocks"]``), consecutive low lines of
    the same block form one block. Otherwise a line joins the block before it when it overlaps it
    across and the vertical gap between them is under the smaller line height. A confidence that
    isn't a finite number counts as low (as 0.0). Blocks keep their reading order (``order``).
    """
    if read is None or read.status != "ok":
        return []
    ids = _line_block_ids(read)
    blocks: list[_Block] = []
    prev_low = False
    prev_id: int | None = None
    for k, ln in enumerate(read.lines):
        bb = _bbox(ln)
        conf = float(ln.conf) if isinstance(ln.conf, (int, float)) and math.isfinite(ln.conf) else 0.0
        if bb is None or conf >= max_conf:
            prev_low = False
            continue
        x0, y0, x1, y1 = bb
        bid = ids[k] if ids is not None else None
        if prev_low and blocks:
            b = blocks[-1]
            if ids is not None:
                joins = bid == prev_id
            else:
                h = min(y1 - y0, b.y1 - b.y0)
                gap = max(y0 - b.y1, b.y0 - y1, 0.0)
                joins = min(x1, b.x1) > max(x0, b.x0) and gap < h
            if joins:
                b.x0, b.y0, b.x1, b.y1 = min(b.x0, x0), min(b.y0, y0), max(b.x1, x1), max(b.y1, y1)
                b.confs.append(conf)
                continue
        blocks.append(_Block(x0, y0, x1, y1, [conf], len(blocks)))
        prev_low = True
        prev_id = bid
    return blocks


def _unreadable(page: PageImage, why: str) -> EngineFailure:
    return EngineFailure("page_unreadable", f"page {page.index}'s render can't be read for the mixed-script check: {why}",
                         path=str(page.path))


def mixed_script_check(page: PageImage, eng: EngineRead, osd: OsdSession,
                       calib: "Calibration | None") -> MixedScript:
    """The pack-independent mixed-script trigger.

    With a calibration: the ``eng`` read's blocks whose mean line confidence is below
    ``calib.block_max_conf`` (a [0, 1] confidence), that overlap the page, and whose shorter side
    measures at least ``calib.min_block_px`` pixels at about 300 dpi are taken, lowest confidence
    first (then larger first), at most ``calib.max_blocks``. Each is cropped from the page render
    with a small margin (clipped to the page), scaled up to about 300 dpi and given to OSD. The
    most confident non-Latin script at ``script_conf >= calib.osd_min_conf`` is reported;
    ``blocks`` counts the crops actually given to OSD.

    Without one (``calib is None``) nothing is given to OSD: ``checked`` is False and ``blocks``
    counts the eng blocks below ``UNCALIBRATED_BLOCK_MAX_CONF``.

    A page render that can't be opened, or a page dpi that isn't a positive number, raises
    ``EngineFailure("page_unreadable")``, as the engines do for the same render.
    """
    if calib is None:
        return MixedScript(False, None, len(_low_blocks(eng, UNCALIBRATED_BLOCK_MAX_CONF)), None)
    dpi = page.dpi
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
        raise _unreadable(page, f"its dpi is {dpi!r}, not a positive number")
    scale = OSD_TARGET_DPI / float(dpi)
    pw, ph = page.width, page.height
    cands = [b for b in _low_blocks(eng, calib.block_max_conf)
             if b.x1 > 0 and b.y1 > 0 and b.x0 < pw and b.y0 < ph
             and (min(b.x1, pw) - max(b.x0, 0)) > 0 and (min(b.y1, ph) - max(b.y0, 0)) > 0
             and min(min(b.x1, pw) - max(b.x0, 0), min(b.y1, ph) - max(b.y0, 0)) * scale >= calib.min_block_px]
    cands.sort(key=lambda b: (b.conf, -b.area, b.order))
    cands = cands[:calib.max_blocks]
    if not cands:
        return MixedScript(True, None, 0, None)
    from PIL import UnidentifiedImageError

    from ..pdf.render import upscale_for_osd

    best: tuple[str, float] | None = None
    given = 0
    try:
        img_cm = page.open()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise _unreadable(page, type(exc).__name__) from None
    with img_cm as img:
        w, h = img.size
        for b in cands:
            pad = max(2.0, BLOCK_PAD_FRACTION * (b.y1 - b.y0))
            box = (max(0, int(math.floor(b.x0 - pad))), max(0, int(math.floor(b.y0 - pad))),
                   min(w, int(math.ceil(b.x1 + pad))), min(h, int(math.ceil(b.y1 + pad))))
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            try:
                crop = img.crop(box)
                up = upscale_for_osd(crop, float(dpi), OSD_TARGET_DPI)
            except (OSError, ValueError) as exc:
                raise _unreadable(page, type(exc).__name__) from None
            up_dpi = float(dpi) * (up.size[0] / crop.size[0]) if crop.size[0] else float(dpi)
            given += 1
            try:
                res = osd.detect(up, up_dpi)
            except EngineFailure:
                continue
            if res is None:
                continue
            name = _script_name(res.script)
            conf = float(res.script_conf) if math.isfinite(res.script_conf) else 0.0
            if name is None or name == "Latin" or conf < calib.osd_min_conf:
                continue
            if best is None or conf > best[1]:
                best = (name, conf)
    if best is None:
        return MixedScript(True, None, given, None)
    return MixedScript(True, best[0], given, best[1])


# --------------------------------------------------------------------------- helpers

def _unit(x: float) -> float:
    """``x`` as a float clamped to [0, 1] (NaN -> 0)."""
    x = float(x)
    if not math.isfinite(x):
        return 0.0 if math.isnan(x) else (1.0 if x > 0 else 0.0)
    return min(1.0, max(0.0, x))
