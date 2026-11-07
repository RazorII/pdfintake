"""Orientation: make every page that will be OCR'd upright before any OCR render.

Three steps, all on pages that will be OCR'd only:

1. ``vote``: the page's *displayed* appearance (a render that honours ``/Rotate``) is read by a
   RapidOCR voter (det + latin rec, cls off) at each of the four right angles. Only horizontal
   line boxes count, so the upright turn wins by a wide margin: lines of a sideways page are
   vertical and drop out, and an upside-down page reads as junk with low confidence.
2. ``decide``: the vote decides alone at a margin of ``VOTE_CONFIDENT_MARGIN`` or more. Below
   that, Tesseract OSD (``orient_conf >= OSD_ORIENT_MIN_CONF``) is the second opinion: the page
   turns only when OSD agrees. When the vote finds no text at all, a confident OSD call is the
   turn. Anything else keeps the page as displayed.
3. ``normalize``: per page, ``net = (/Rotate + turn) % 360``. ``net == 0`` clears the flag
   (lossless; the stored content is already upright). Otherwise the content is baked: the page
   becomes a Form XObject drawn through a rotation matrix, clipped to the old CropBox, with a new
   MediaBox and ``/Rotate 0``. No image is decoded or re-encoded. An unconfident decision has a
   turn of 0, so the flag is baked into the content (the display is kept), never cleared unverified.
   A page with any annotation other than a Link (widgets included) is never baked.

``/Rotate`` is read the way pdfium (the renderer the vote and the verify suite use) reads it:
``int(value / 90)`` right angles, truncated toward zero, modulo 4. So 135 displays as 90, -135 as
270 and 89 as 0, and a malformed flag gives the same net turn here as on screen. An encrypted
input keeps its encryption and permissions in the work copy (``encryption=True`` on save).

Turns are always clockwise, as seen on screen, applied to the displayed appearance.
"""

from __future__ import annotations

import math
import os
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..errors import EngineFailure
from ..types import EngineRead, OsdResult, PageImage

if TYPE_CHECKING:
    import PIL.Image
    import pikepdf

    from ..ocr.engines import EngineSession

VOTE_CONFIDENT_MARGIN = 10.0        # RapidOCR vote margin (measured 60-90x on real scans)
OSD_ORIENT_MIN_CONF = 3.0           # 529/552 correct calls kept, none wrong (public renders, about 300 dpi)

# Private knobs of the RapidOCR rotation vote.
VOTE_LONG_SIDE = 1100               # the vote image is scaled down to this long side (px)
VOTE_SCORE_FLOOR = 1.0              # the runner-up score is floored here, so a page with almost no
                                    # text can't reach the margin (best must score >= 10 chars x conf)
VOTE_TIMEOUT_S = 60.0               # per voter read
VOTE_DEFAULT_DPI = 110.0            # assumed when the image carries no dpi
FORM_NAME = "/PdfIntakeOrient0"     # fixed, so a bake gives the same bytes every time

TURNS: tuple[int, ...] = (0, 90, 180, 270)

# Why a decision came out the way it did. Closed set; the record keeps only turn_cw, by and action.
REASONS: frozenset[str] = frozenset({
    "vote_confident",             # margin >= 10x, OSD silent or below 3
    "vote_confident_osd_differs", # margin >= 10x, a confident OSD call disagreed; the vote wins
    "vote_and_osd_agree",         # margin >= 10x and a confident OSD call agreed
    "weak_vote_osd_agrees",       # margin < 10x, a confident OSD call agreed
    "osd_disagrees",              # margin < 10x, a confident OSD call disagreed: kept as displayed
    "weak_vote_no_osd",           # margin < 10x, OSD silent or below 3: kept as displayed
    "osd_only",                   # the vote found no text; a confident OSD call is the turn
    "no_signal",                  # no text for the vote and no confident OSD call: kept as displayed
})
# Decision reasons that the flow reports as a page warning (record.WARNING_CODES).
REASON_WARNINGS: Mapping[str, str] = {"osd_disagrees": "orientation_disagree"}

Turn = Literal[0, 90, 180, 270]
Matrix = tuple[float, float, float, float, float, float]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class OrientVote:
    page: int
    turn_cw: Literal[0, 90, 180, 270]
    margin: float
    boxes: int
    # Per-turn scores (non-space characters x confidence on horizontal lines), for the record and the kit.
    scores: Mapping[int, float] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class OrientDecision:
    page: int
    turn_cw: Literal[0, 90, 180, 270]
    by: Literal["vote", "vote+osd", "osd", "none"]
    confident: bool
    reason: str


@dataclass
class OrientReport:
    changed: bool
    baked: list[int]
    cleared: list[int]
    kept_flag: list[int]          # pages with a non-zero /Rotate kept as it is and no turn called for: no
                                  # decision (not OCR'd), or a non-Link-annotated page whose decision has no turn
    kept_annotated: list[int]     # pages a turn was called for but that carry non-Link annotations: not turned


# --------------------------------------------------------------------------- vote

def vote(image: "PIL.Image.Image", page: int, voter: "EngineSession", *, dpi: float | None = None,
         workdir: Path | None = None, timeout_s: float = VOTE_TIMEOUT_S) -> OrientVote:
    """Read the displayed page at the four right angles and return the clockwise turn that reads best.

    ``image`` is the page as displayed (rendered honouring ``/Rotate``). Each turned copy is written
    as a PNG (to ``workdir``, else a temporary folder that is removed afterwards) and read with
    ``voter``; an engine failure or a non-ok read scores 0 for that turn. ``margin`` is the best score
    over the runner-up (floored at ``VOTE_SCORE_FLOOR``); ``boxes`` counts the horizontal text boxes
    of the winning turn, 0 when nothing was read.
    """
    from PIL import Image

    img = image.convert("RGB") if image.mode != "RGB" else image.copy()
    src_dpi = _image_dpi(image) if dpi is None else float(dpi)
    long_side = max(img.size)
    if long_side > VOTE_LONG_SIDE:
        s = VOTE_LONG_SIDE / long_side
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                         Image.Resampling.BICUBIC)
        src_dpi *= s
    scores: dict[int, float] = {}
    boxes: dict[int, int] = {}
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
        _vote_all(img, page, voter, Path(workdir), src_dpi, timeout_s, scores, boxes)
    else:
        with tempfile.TemporaryDirectory(prefix="pdfintake-orient-") as tmp:
            _vote_all(img, page, voter, Path(tmp), src_dpi, timeout_s, scores, boxes)
    best = max(TURNS, key=lambda r: (scores[r], -r))          # ties go to the smaller turn, 0 first
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] <= 0.0:
        return OrientVote(page, 0, 0.0, 0, scores)
    margin = ranked[0] / max(ranked[1], VOTE_SCORE_FLOOR)
    return OrientVote(page, best, round(margin, 4), boxes[best], scores)  # type: ignore[arg-type]


def _vote_all(img: "PIL.Image.Image", page: int, voter: "EngineSession", folder: Path, dpi: float,
              timeout_s: float, scores: dict[int, float], boxes: dict[int, int]) -> None:
    for r in TURNS:
        turned = turn_image(img, r)
        path = folder / f"orient-{page:04d}-{r:03d}.png"
        turned.save(path, format="PNG", dpi=(dpi, dpi))
        w, h = turned.size
        pi = PageImage(index=page, path=path, width=w, height=h, dpi=dpi,
                       cropbox=(0.0, 0.0, w * 72.0 / dpi, h * 72.0 / dpi))
        try:
            read = voter.read(pi, timeout_s=timeout_s)
        except EngineFailure:
            read = None
        scores[r], boxes[r] = _score(read)
        try:
            os.unlink(path)
        except OSError:
            pass


def _score(read: EngineRead | None) -> tuple[float, int]:
    """Confident characters on horizontal lines (box width >= height), and how many such lines."""
    if read is None or read.status != "ok":
        return 0.0, 0
    total = 0.0
    n = 0
    for line in read.lines:
        chars = sum(1 for ch in line.text if not ch.isspace())
        if not chars:
            continue
        xs = [p[0] for p in line.box]
        ys = [p[1] for p in line.box]
        if (max(xs) - min(xs)) < (max(ys) - min(ys)):
            continue
        conf = float(line.conf)
        if not math.isfinite(conf) or conf <= 0.0:
            continue
        total += chars * min(conf, 1.0)
        n += 1
    return total, n


def turn_image(image: "PIL.Image.Image", turn_cw: int) -> "PIL.Image.Image":
    """``image`` turned ``turn_cw`` degrees clockwise (0, 90, 180, 270), pixel-exact."""
    from PIL import Image

    t = _check_turn(turn_cw)
    if t == 0:
        return image
    op = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[t]
    return image.transpose(op)


def _image_dpi(image: "PIL.Image.Image") -> float:
    dpi = image.info.get("dpi")
    try:
        value = float(dpi[0]) if isinstance(dpi, tuple) else float(dpi)
    except (TypeError, ValueError, IndexError):
        return VOTE_DEFAULT_DPI
    return value if math.isfinite(value) and value > 1.0 else VOTE_DEFAULT_DPI


# --------------------------------------------------------------------------- decide

def osd_turn_cw(osd: OsdResult | None) -> int | None:
    """The clockwise turn OSD calls for, or None when OSD has no confident call.

    Content turned r degrees counter-clockwise reports ``orient_deg = (360 - r) % 360``, so the
    fix is ``(360 - orient_deg) % 360`` clockwise. Below ``OSD_ORIENT_MIN_CONF`` there is no call.
    """
    if osd is None:
        return None
    try:
        conf = float(osd.orient_conf)
        deg = int(osd.orient_deg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(conf) or conf < OSD_ORIENT_MIN_CONF or deg % 90 != 0:
        return None
    return (360 - deg % 360) % 360


def decide(vote: OrientVote, osd: OsdResult | None, rotate_flag: int) -> OrientDecision:
    """The turn for one page, from its vote and its OSD result.

    * margin >= ``VOTE_CONFIDENT_MARGIN``: the vote's turn (``by`` is ``vote+osd`` when a confident
      OSD call agrees, else ``vote``).
    * margin below that: the vote's turn only if a confident OSD call agrees; otherwise no turn and
      not confident (``osd_disagrees`` when OSD confidently said something else).
    * a vote with no boxes: OSD's turn when it is confident, else no turn.

    ``rotate_flag`` is the page's current ``/Rotate``; it doesn't change the turn (a turn is always
    relative to the displayed page). It must be an integer; a value that isn't a right angle is read
    the way pdfium reads it (see ``normalize_rotate``), so a malformed file never raises here.
    """
    _check_flag(rotate_flag)
    page = vote.page
    o = osd_turn_cw(osd)
    if vote.boxes <= 0 or not (vote.margin > 0.0):
        if o is not None:
            return OrientDecision(page, o, "osd", True, "osd_only")  # type: ignore[arg-type]
        return OrientDecision(page, 0, "none", False, "no_signal")
    v = _check_turn(vote.turn_cw)
    if vote.margin >= VOTE_CONFIDENT_MARGIN:
        if o is None:
            return OrientDecision(page, v, "vote", True, "vote_confident")  # type: ignore[arg-type]
        if o == v:
            return OrientDecision(page, v, "vote+osd", True, "vote_and_osd_agree")  # type: ignore[arg-type]
        return OrientDecision(page, v, "vote", True, "vote_confident_osd_differs")  # type: ignore[arg-type]
    if o is None:
        return OrientDecision(page, 0, "none", False, "weak_vote_no_osd")
    if o == v:
        return OrientDecision(page, v, "vote+osd", True, "weak_vote_osd_agrees")  # type: ignore[arg-type]
    return OrientDecision(page, 0, "none", False, "osd_disagrees")


def decision_warnings(decision: OrientDecision) -> list[str]:
    """Page warning codes this decision raises (today only ``orientation_disagree``)."""
    code = REASON_WARNINGS.get(decision.reason)
    return [code] if code else []


# --------------------------------------------------------------------------- normalize and bake

def normalize(src: Path, dest: Path, decisions: Sequence[OrientDecision], *,
              password: str | None = None) -> OrientReport:
    """Apply ``decisions`` to ``src`` and save the result to ``dest``; ``dest`` is untouched when nothing changes.

    Only pages with a decision are touched. Per page ``net = (/Rotate + turn) % 360``: ``net == 0``
    with a flag clears the flag; ``net != 0`` bakes the content, unless the page has annotations
    other than Links: then the page is left exactly as it is, listed in ``kept_annotated`` when a turn
    was called for (the flow's ``not_turned_annotations`` warning) and in ``kept_flag`` when none was
    (only the flag bake was skipped; the display is the same either way). An unconfident decision
    counts as no turn. Pages without a decision that carry a non-zero ``/Rotate`` are listed in
    ``kept_flag``. Every flag cleared or baked becomes an explicit ``/Rotate 0``.

    ``password`` opens an input encrypted with an open password (never logged or stored). An
    encrypted input, permissions-only included, is saved with ``encryption=True``, so the work copy
    keeps the original's encryption, password and permissions (pikepdf's default save drops them).
    """
    import pikepdf

    src = Path(src)
    dest = Path(dest)
    if _same_path(src, dest):
        raise ValueError("normalize needs a destination other than its source")
    with pikepdf.open(src, password=password or "") as pdf:
        count = len(pdf.pages)
        by_page: dict[int, OrientDecision] = {}
        for d in decisions:
            if not isinstance(d.page, int) or not 0 <= d.page < count:
                raise ValueError(f"orientation decision for page index {d.page!r}, but the PDF has {count} pages")
            if d.page in by_page:
                raise ValueError(f"two orientation decisions for page index {d.page}")
            _check_turn(d.turn_cw)
            by_page[d.page] = d
        report = OrientReport(False, [], [], [], [])
        for index in range(count):
            page = pdf.pages[index]
            flag = _page_rotate(page.obj)
            d = by_page.get(index)
            if d is None:
                if flag:
                    report.kept_flag.append(index)
                continue
            turn = d.turn_cw if d.confident else 0
            net = (flag + turn) % 360
            if net == 0:
                if flag or _has_raw_rotate(page.obj):
                    page.obj.Rotate = 0           # explicit, also over an inherited or malformed flag
                    report.cleared.append(index)
                continue
            if has_blocking_annotations(page.obj):
                (report.kept_annotated if turn else report.kept_flag).append(index)
                continue
            bake(pdf, index, net)  # type: ignore[arg-type]
            report.baked.append(index)
        report.changed = bool(report.baked or report.cleared)
        if report.changed:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".orient-partial")
            try:
                if pdf.is_encrypted:
                    # qpdf refuses deterministic_id on an encrypted write (measured, pikepdf 10.17),
                    # so an encrypted work copy keeps the original's encryption but not byte identity
                    pdf.save(tmp, encryption=True)
                else:
                    pdf.save(tmp, deterministic_id=True)
                os.replace(tmp, dest)
            finally:
                if tmp.exists():
                    tmp.unlink()
    return report


def page_action(report: OrientReport, index: int) -> str:
    """The record's ``orientation.action`` for a page: baked, cleared, kept_annotated, kept_flag or none."""
    for name in ("baked", "cleared", "kept_annotated", "kept_flag"):
        if index in getattr(report, name):
            return name
    return "none"


def bake(pdf: "pikepdf.Pdf", index: int, degrees: Literal[90, 180, 270]) -> None:
    """Turn page ``index``'s content ``degrees`` clockwise into its stored content, losslessly.

    The page's content becomes a Form XObject (``as_form_xobject(handle_transformations=False)``)
    with its BBox set to the old visible area (CropBox within MediaBox). The new content draws it
    through a rotation matrix, clipped to the new MediaBox, which is the old visible area turned.
    ``/Rotate`` becomes 0 and the CropBox goes; TrimBox, BleedBox, ArtBox and Link annotation
    rectangles are mapped through the same matrix. Image streams are never touched. The result
    renders like the old page with ``degrees`` more of display rotation, ignoring its old ``/Rotate``.
    """
    import pikepdf

    d = _check_turn(degrees)
    if d == 0:
        raise ValueError("bake needs 90, 180 or 270 degrees")
    page = pdf.pages[index]
    obj = page.obj
    if has_blocking_annotations(obj):
        raise ValueError(f"page index {index} has annotations other than Links; it can't be baked")
    visible = visible_box(obj)
    m = turn_matrix(visible, d)
    new_w, new_h = _turned_size(visible, d)

    resources = _inherited(obj, "/Resources")
    if resources is not None and not resources.is_indirect:
        obj.Resources = pdf.make_indirect(resources)      # the form shares it; it must outlive the page's key
    form = page.as_form_xobject(handle_transformations=False)
    # qpdf fills the form's data lazily from the page's /Contents at save time; the page's contents
    # are replaced below, so the data is materialised now (decoded content, stored Flate).
    form.write(zlib.compress(form.read_bytes(), 9), filter=pikepdf.Name.FlateDecode)
    form.BBox = pikepdf.Array([_num_obj(v) for v in visible])
    if "/Matrix" in form:
        del form["/Matrix"]
    form = pdf.make_indirect(form)

    content = (
        "q\n"
        f"0 0 {_num(new_w)} {_num(new_h)} re W n\n"
        f"{' '.join(_num(v) for v in m)} cm\n"
        f"{FORM_NAME} Do\n"
        "Q\n"
    ).encode("ascii")
    obj.Contents = pdf.make_indirect(pikepdf.Stream(pdf, content))
    obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary({FORM_NAME: form}))

    for key in ("/TrimBox", "/BleedBox", "/ArtBox"):           # not inheritable (ISO 32000 §14.11.2)
        raw = obj.get(key)
        if raw is None:
            continue
        mapped = _clip_rect(_map_rect(_rect(raw), m), (0.0, 0.0, new_w, new_h))
        if mapped is None:
            del obj[key]
        else:
            obj[key] = pikepdf.Array([_num_obj(v) for v in mapped])
    obj.MediaBox = pikepdf.Array([0, 0, _num_obj(new_w), _num_obj(new_h)])
    if "/CropBox" in obj:
        del obj["/CropBox"]
    obj.Rotate = 0
    if "/Thumb" in obj:
        del obj["/Thumb"]                # a thumbnail of the old orientation would be wrong

    annots = obj.get("/Annots")
    if annots is not None:
        for annot in annots:
            if not isinstance(annot, pikepdf.Dictionary):
                continue
            rect = annot.get("/Rect")
            if rect is not None:
                annot.Rect = pikepdf.Array([_num_obj(v) for v in _map_rect(_rect(rect), m)])
            quads = annot.get("/QuadPoints")
            if quads is not None:
                vals = [float(v) for v in quads]
                out: list[Any] = []
                for i in range(0, len(vals) - 1, 2):
                    x, y = _apply(m, vals[i], vals[i + 1])
                    out.extend((_num_obj(x), _num_obj(y)))
                annot.QuadPoints = pikepdf.Array(out)


# --------------------------------------------------------------------------- geometry

def turn_matrix(visible: Rect, degrees: int) -> Matrix:
    """The ``cm`` matrix that turns the area ``visible`` clockwise by ``degrees`` onto ``[0 0 W H]``."""
    x0, y0, x1, y1 = visible
    w, h = x1 - x0, y1 - y0
    t = _check_turn(degrees)
    if t == 0:
        return (1.0, 0.0, 0.0, 1.0, -x0, -y0)
    if t == 90:
        return (0.0, -1.0, 1.0, 0.0, -y0, w + x0)
    if t == 180:
        return (-1.0, 0.0, 0.0, -1.0, x0 + w, y0 + h)
    return (0.0, 1.0, -1.0, 0.0, y0 + h, -x0)


def visible_box(page_obj: Any) -> Rect:
    """The page's visible area: the CropBox (inherited, default MediaBox) within the MediaBox."""
    media_raw = _inherited(page_obj, "/MediaBox")
    if media_raw is None:
        raise ValueError("page has no MediaBox")
    media = _rect(media_raw)
    crop_raw = _inherited(page_obj, "/CropBox")
    crop = _rect(crop_raw) if crop_raw is not None else media
    vis = _clip_rect(crop, media)
    if vis is None:
        raise ValueError("page's CropBox doesn't overlap its MediaBox")
    return vis


def has_blocking_annotations(page_obj: Any) -> bool:
    """True when the page has any annotation that isn't a Link (widgets included)."""
    import pikepdf

    annots = page_obj.get("/Annots")
    if annots is None:
        return False
    if not isinstance(annots, pikepdf.Array):
        return True
    for annot in annots:
        if not isinstance(annot, pikepdf.Dictionary):
            return True
        if annot.get("/Subtype") != pikepdf.Name.Link:
            return True
    return False


def _turned_size(visible: Rect, degrees: int) -> tuple[float, float]:
    w, h = visible[2] - visible[0], visible[3] - visible[1]
    return (h, w) if degrees in (90, 270) else (w, h)


def _apply(m: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def _map_rect(r: Rect, m: Matrix) -> Rect:
    pts = [_apply(m, r[0], r[1]), _apply(m, r[2], r[1]), _apply(m, r[2], r[3]), _apply(m, r[0], r[3])]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _clip_rect(r: Rect, bound: Rect) -> Rect | None:
    x0, y0 = max(r[0], bound[0]), max(r[1], bound[1])
    x1, y1 = min(r[2], bound[2]), min(r[3], bound[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _rect(raw: Any) -> Rect:
    vals = [float(v) for v in raw]
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        raise ValueError("a page box must be four finite numbers")
    return (min(vals[0], vals[2]), min(vals[1], vals[3]), max(vals[0], vals[2]), max(vals[1], vals[3]))


def _inherited(page_obj: Any, key: str) -> Any:
    """``key`` on the page or the nearest /Pages ancestor that has it (MediaBox, CropBox, Rotate, Resources)."""
    node = page_obj
    seen = 0
    while node is not None and seen < 64:
        value = node.get(key)
        if value is not None:
            return value
        node = node.get("/Parent")
        seen += 1
    return None


def normalize_rotate(value: int) -> int:
    """A ``/Rotate`` value as pdfium displays it: ``int(value / 90)`` right angles (truncated toward
    zero), modulo 4. 135 -> 90, -135 -> 270, 89 -> 0, -90 -> 270, 450 -> 90."""
    return (int(value / 90) % 4) * 90


def _raw_rotate(page_obj: Any) -> int | None:
    raw = _inherited(page_obj, "/Rotate")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value)                     # pdfium reads a real /Rotate as an integer, truncated


def _has_raw_rotate(page_obj: Any) -> bool:
    """True when the page shows a non-zero raw ``/Rotate`` (say 45) that pdfium displays as 0."""
    raw = _raw_rotate(page_obj)
    return raw is not None and raw != 0


def _page_rotate(page_obj: Any) -> int:
    raw = _raw_rotate(page_obj)
    return 0 if raw is None else normalize_rotate(raw)


def _check_turn(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in TURNS:
        raise ValueError(f"a turn must be 0, 90, 180 or 270, not {value!r}")
    return value


def _check_flag(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"/Rotate must be an integer, not {value!r}")
    return normalize_rotate(value)


def _num(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _num_obj(v: float) -> Any:
    from decimal import Decimal

    if v == int(v):
        return int(v)
    return Decimal(_num(v))


def _same_path(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
