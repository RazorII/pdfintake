"""pdf/orient.py: the four-way vote, the OSD mapping, the decide table, lossless bakes and normalize.

The vote's logic is tested with a fake voter. The real-model tests read a synthetic page with RapidOCR
(det + latin rec, cls off). The models come from ``conftest.pinned_asset(pack, key)``: the base
pack's ``rapid-det``, ``rapid-cls`` and ``rapid-rec-latin`` under ``PDFINTAKE_TEST_MODELS`` (default
``~/.pdfintake-dev/models``) as ``en/<file>``, each checked against its pinned size and SHA-256
(``tools/fetch_test_models.py`` fetches them); a missing or different file skips the test.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from conftest import pinned_assets
from fixtures import build
from pdfintake.errors import EngineFailure
from pdfintake.pdf import orient
from pdfintake.pdf.orient import OrientDecision, OrientVote, decide, osd_turn_cw
from pdfintake.types import EngineRead, Line, OsdResult, PageImage

pikepdf = pytest.importorskip("pikepdf")
pdfium = pytest.importorskip("pypdfium2")

TURNS = (0, 90, 180, 270)


# --------------------------------------------------------------------------- helpers

def render(data: bytes | Path, index: int = 0, *, rotation: int = 0, scale: float = 1.0) -> np.ndarray:
    """pdfium at 72 dpi (scale 1), grey, with pdfium's own extra display rotation."""
    src = data if isinstance(data, bytes) else Path(data).read_bytes()
    doc = pdfium.PdfDocument(src)
    try:
        page = doc[index]
        bitmap = page.render(scale=scale, rotation=rotation, grayscale=True)
        return bitmap.to_numpy().copy()
    finally:
        doc.close()


def same_render(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def page_raw_images(pdf: Any, index: int) -> list[bytes]:
    """Raw stream bytes of every image reachable from a page, through forms."""
    out: list[bytes] = []

    def walk(res: Any, depth: int) -> None:
        if res is None or depth > 8:
            return
        xobjs = res.get("/XObject")
        if xobjs is None:
            return
        for _, x in xobjs.items():
            if x.get("/Subtype") == pikepdf.Name.Image:
                out.append(bytes(x.read_raw_bytes()))
            elif x.get("/Subtype") == pikepdf.Name.Form:
                walk(x.get("/Resources"), depth + 1)

    walk(pdf.pages[index].obj.get("/Resources"), 0)
    return sorted(out)


def rich_page(**kw: Any) -> build.PageSpec:
    """A page with a JPEG, a Flate image, text and a diagonal stroke: anything a bake could disturb."""
    jpg = build.jpeg_bytes(build.text_like(400, 300, mode="RGB"))
    flate = build.photo_like(120, 90)
    return build.PageSpec(
        texts=(build.TextSpec("Example heading", x=80, y=600, size=20),
               build.TextSpec("Sample line 4521", x=300, y=100, size=16)),
        images=(build.ImageSpec(jpg, x=50, y=200, w=400, h=300), build.ImageSpec(flate, x=100, y=550, w=200, h=150)),
        extra=b"q 0.2 g 3 w 20 20 m 590 770 l S Q",
        **kw,
    )


# --------------------------------------------------------------------------- a fake voter

def marker_page(width: int = 600, height: int = 400) -> Image.Image:
    """An upright page: a dark marker block in the top-left corner and some bars."""
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, 60, 60), fill=0)
    for k in range(3):
        d.rectangle((100, 100 + 70 * k, width - 60, 120 + 70 * k), fill=90)
    return img


class MarkerVoter:
    """Reads a marker page: upright gives three long horizontal lines; sideways gives vertical boxes;
    upside down gives short horizontal junk at low confidence (like a real reader with cls off)."""

    def __init__(self, *, fail: str | None = None) -> None:
        self.fail = fail
        self.sizes: list[tuple[int, int]] = []
        self.pages: list[PageImage] = []

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        self.pages.append(page)
        if self.fail == "raise":
            raise EngineFailure("engine_crashed", "the fake voter failed")
        if self.fail == "failed":
            return EngineRead("orient", "fake", "fake:orient", "checker", "failed", [], 0.0, error="boom")
        with page.open() as img:
            a = np.asarray(img.convert("L")) < 40
            self.sizes.append(img.size)
        h, w = a.shape
        ys, xs = np.nonzero(a)
        if len(xs) == 0:
            return EngineRead("orient", "fake", "fake:orient", "checker", "empty", [], 0.0)
        cx, cy = xs.mean() / w, ys.mean() / h
        if cx < 0.5 and cy < 0.5:                    # upright
            lines = [Line("Sample words read upright", 0.95, build.quad(100, 100 + 70 * k, 540, 120 + 70 * k))
                     for k in range(3)]
        elif cx >= 0.5 and cy >= 0.5:                # upside down
            lines = [Line("xz", 0.3, build.quad(100, 100, 540, 120))]
        else:                                        # sideways: boxes taller than wide
            lines = [Line("Sample words read upright", 0.9, build.quad(100 + 70 * k, 100, 120 + 70 * k, 540))
                     for k in range(3)]
        return EngineRead("orient", "fake", "fake:orient", "checker", "ok", lines, 0.0)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("shown", TURNS)
def test_vote_fake_voter_finds_the_turn(shown: int) -> None:
    upright = marker_page()
    displayed = orient.turn_image(upright, shown)            # the page as displayed: upright turned `shown` cw
    v = orient.vote(displayed, 3, MarkerVoter())
    assert v.page == 3
    assert v.turn_cw == (360 - shown) % 360
    assert v.margin >= orient.VOTE_CONFIDENT_MARGIN
    assert v.boxes == 3


def test_vote_writes_pngs_with_dpi_and_cleans_up(tmp_path: Path) -> None:
    voter = MarkerVoter()
    orient.vote(marker_page(), 0, voter, dpi=150.0, workdir=tmp_path)
    assert len(voter.pages) == 4
    assert all(p.dpi == 150.0 for p in voter.pages)
    assert {(p.width, p.height) for p in voter.pages} == {(600, 400), (400, 600)}
    assert list(tmp_path.iterdir()) == []


def test_vote_scales_large_pages_down() -> None:
    voter = MarkerVoter()
    big = marker_page().resize((3000, 2000))
    v = orient.vote(big, 0, voter)
    assert max(max(s) for s in voter.sizes) <= orient.VOTE_LONG_SIDE
    assert v.turn_cw == 0


@pytest.mark.parametrize("fail", ["raise", "failed"])
def test_vote_engine_failure_counts_as_no_text(fail: str) -> None:
    v = orient.vote(marker_page(), 0, MarkerVoter(fail=fail))
    assert (v.turn_cw, v.margin, v.boxes) == (0, 0.0, 0)


def test_vote_blank_page_has_no_boxes() -> None:
    v = orient.vote(Image.new("RGB", (500, 700), "white"), 1, MarkerVoter())
    assert v.boxes == 0 and v.margin == 0.0 and v.turn_cw == 0


def test_vote_little_text_never_reaches_the_margin() -> None:
    class Tiny(MarkerVoter):
        def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
            with page.open() as img:
                upright = img.size == (600, 400) and np.asarray(img.convert("L"))[30, 30] < 40
            lines = [Line("ab", 0.9, build.quad(10, 10, 100, 20))] if upright else []
            return EngineRead("orient", "fake", "fake:orient", "checker", "ok" if lines else "empty", lines, 0.0)

    v = orient.vote(marker_page(), 0, Tiny())
    assert v.turn_cw == 0 and v.boxes == 1
    assert v.margin < orient.VOTE_CONFIDENT_MARGIN       # 1.8 / floor 1.0


# --------------------------------------------------------------------------- the real voter

def _pinned_models() -> dict[str, Path]:
    """The base pack's RapidOCR det, cls and latin rec, or a skip naming what's missing."""
    found = pinned_assets("en:rapid-det", "en:rapid-cls", "en:rapid-rec-latin")
    return {"det": found["en:rapid-det"], "cls": found["en:rapid-cls"], "latin": found["en:rapid-rec-latin"]}


class RapidVoter:
    """Test-side voter: RapidOCR det + latin rec with cls off, every model path explicit."""

    def __init__(self, models: dict[str, Path]) -> None:
        rapidocr = pytest.importorskip("rapidocr")
        from rapidocr.utils.typings import LangDet, LangRec, OCRVersion

        self.engine = rapidocr.RapidOCR(params={
            "Global.use_cls": False, "Global.log_level": "error",
            "Det.model_path": str(models["det"]),
            "Det.ocr_version": OCRVersion.PPOCRV4, "Det.lang_type": LangDet.CH,
            "Rec.model_path": str(models["latin"]),
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.lang_type": LangRec.LATIN,
            "Cls.model_path": str(models["cls"]),       # cls is off, but RapidOCR wants a path
        })

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        with page.open() as img:
            arr = np.asarray(img.convert("RGB"))
        res = self.engine(arr)
        lines = []
        boxes = res.boxes if res.boxes is not None else []
        for text, score, box in zip(res.txts or (), res.scores or (), boxes):
            q = tuple((float(p[0]), float(p[1])) for p in box)
            lines.append(Line(str(text), float(score), q))  # type: ignore[arg-type]
        return EngineRead("orient", "rapidocr", "rapidocr:orient", "checker", "ok" if lines else "empty", lines, 0.0)

    def close(self) -> None:
        pass


def text_page() -> Image.Image:
    """A synthetic letter-size page at 100 dpi with made-up English lines in Pillow's bundled font."""
    font = ImageFont.load_default(size=28)
    img = Image.new("L", (850, 1100), 255)
    d = ImageDraw.Draw(img)
    texts = [
        "The example invoice lists seven sample items",
        "and a total of 4521 units for the test page.",
        "Please read this sample page the right way up.",
        "Orientation voting needs some real words here.",
        "Another made-up line of text for the reader.",
    ]
    for i, t in enumerate(texts):
        d.text((60, 80 + 60 * i), t, fill=0, font=font)
    return img


MEASURED_MARGINS: dict[int, float] = {}


@pytest.mark.parametrize("shown", TURNS)
def test_vote_real_rapidocr_turns(shown: int) -> None:
    models = _pinned_models()
    voter = RapidVoter(models)
    displayed = orient.turn_image(text_page(), shown)
    v = orient.vote(displayed, 0, voter, dpi=100.0)
    MEASURED_MARGINS[shown] = v.margin
    assert v.turn_cw == (360 - shown) % 360, v
    assert v.margin >= orient.VOTE_CONFIDENT_MARGIN, v
    assert v.boxes >= 3


def test_vote_with_the_shipped_voter() -> None:
    """The same vote through ``ocr.engines.rapid.open_voter`` once that engine is built."""
    models = _pinned_models()
    try:
        from pdfintake.ocr.engines import rapid
        voter = rapid.open_voter({"rapid-det": models["det"], "rapid-rec-latin": models["latin"]})
    except (ImportError, AttributeError, EngineFailure) as exc:
        pytest.skip(f"rapid.open_voter isn't usable here: {exc}")
    try:
        for shown in TURNS:
            v = orient.vote(orient.turn_image(text_page(), shown), 0, voter, dpi=100.0)
            assert v.turn_cw == (360 - shown) % 360 and v.margin >= orient.VOTE_CONFIDENT_MARGIN, (shown, v)
    finally:
        voter.close()


def test_vote_real_rapidocr_through_a_pdf_rotate_flag(tmp_path: Path) -> None:
    """The displayed render of a page with /Rotate 90 votes for the turn that undoes it."""
    models = _pinned_models()
    pdf = build.pdf_bytes([build.PageSpec(width=612, height=792, rotate=90,
                                          images=(build.ImageSpec(text_page()),))])
    doc = pdfium.PdfDocument(pdf)
    try:
        shown = doc[0].render(scale=100 / 72, grayscale=True).to_pil()
    finally:
        doc.close()
    v = orient.vote(shown, 0, RapidVoter(models), dpi=100.0)
    d = decide(v, None, 90)
    assert (d.turn_cw, d.by, d.confident) == (270, "vote", True)
    src = tmp_path / "in.pdf"
    src.write_bytes(pdf)
    rep = orient.normalize(src, tmp_path / "out.pdf", [d])
    assert rep.cleared == [0] and rep.baked == []           # net (90 + 270) % 360 = 0: the flag goes


# --------------------------------------------------------------------------- OSD mapping

@pytest.mark.parametrize("content_ccw", TURNS)
def test_osd_mapping(content_ccw: int) -> None:
    # Content turned r degrees counter-clockwise reports orient_deg = (360 - r) % 360; the fix is r clockwise.
    osd = OsdResult(script="Latin", script_conf=4.0, orient_deg=(360 - content_ccw) % 360, orient_conf=5.0)
    assert osd_turn_cw(osd) == content_ccw


def test_osd_mapping_needs_confidence() -> None:
    assert osd_turn_cw(None) is None
    assert osd_turn_cw(OsdResult("Latin", 9.0, 90, 2.99)) is None
    assert osd_turn_cw(OsdResult("Latin", 9.0, 90, orient.OSD_ORIENT_MIN_CONF)) == 270
    assert osd_turn_cw(OsdResult("Latin", 9.0, 45, 9.0)) is None
    assert osd_turn_cw(OsdResult("Latin", 9.0, 90, float("nan"))) is None


# --------------------------------------------------------------------------- decide

def osd_for(turn_cw: int | None, conf: float = 6.0) -> OsdResult | None:
    if turn_cw is None:
        return None
    return OsdResult("Latin", 5.0, (360 - turn_cw) % 360, conf)


DECIDE_TABLE = [
    # vote (turn, margin, boxes), OSD turn (None = no result), OSD conf -> turn, by, confident, reason
    ((90, 40.0, 5), None, 6.0, (90, "vote", True, "vote_confident")),                  # vote alone
    ((90, 10.0, 5), None, 6.0, (90, "vote", True, "vote_confident")),                  # exactly 10x
    ((90, 40.0, 5), 90, 6.0, (90, "vote+osd", True, "vote_and_osd_agree")),
    ((90, 40.0, 5), 180, 6.0, (90, "vote", True, "vote_confident_osd_differs")),      # >= 10x: the vote wins
    ((90, 40.0, 5), 180, 2.0, (90, "vote", True, "vote_confident")),
    ((180, 4.0, 5), 180, 6.0, (180, "vote+osd", True, "weak_vote_osd_agrees")),       # under 10x, OSD agrees
    ((180, 4.0, 5), 90, 6.0, (0, "none", False, "osd_disagrees")),                    # under 10x, OSD disagrees
    ((180, 4.0, 5), 180, 2.9, (0, "none", False, "weak_vote_no_osd")),                # under 10x, OSD below 3
    ((180, 4.0, 5), None, 6.0, (0, "none", False, "weak_vote_no_osd")),
    ((0, 0.0, 0), 270, 6.0, (270, "osd", True, "osd_only")),                          # no boxes, OSD confident
    ((0, 0.0, 0), 270, 1.0, (0, "none", False, "no_signal")),                         # no boxes, OSD weak
    ((0, 0.0, 0), None, 6.0, (0, "none", False, "no_signal")),
]


@pytest.mark.parametrize("vote_args,osd_turn,osd_conf,expected", DECIDE_TABLE)
def test_decide_table(vote_args: tuple[int, float, int], osd_turn: int | None, osd_conf: float,
                      expected: tuple[int, str, bool, str]) -> None:
    v = OrientVote(2, vote_args[0], vote_args[1], vote_args[2])  # type: ignore[arg-type]
    d = decide(v, osd_for(osd_turn, osd_conf), 0)
    assert (d.page, d.turn_cw, d.by, d.confident, d.reason) == (2, *expected)
    assert d.reason in orient.REASONS


def test_decide_warnings_and_flag_check() -> None:
    d = decide(OrientVote(0, 180, 4.0, 5), osd_for(90), 270)
    assert orient.decision_warnings(d) == ["orientation_disagree"]
    assert orient.decision_warnings(decide(OrientVote(0, 90, 40.0, 5), None, 0)) == []
    # a malformed flag is read the way pdfium reads it, never raised on (decide ignores the flag)
    for flag in (45, 100, 135, -135, 179, 450):
        assert decide(OrientVote(0, 90, 40.0, 5), None, flag).turn_cw == 90
    with pytest.raises(ValueError):
        decide(OrientVote(0, 90, 40.0, 5), None, "90")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 45, 89, 90, 100, 135, 179, 180, 270, 360, 450, -45, -90, -135, -180, -270])
def test_rotate_flag_matches_pdfium(value: int) -> None:
    raw = build.pdf_bytes([rich_page()])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.pages[0].obj.Rotate = value
        buf = io.BytesIO()
        pdf.save(buf)
    doc = pdfium.PdfDocument(buf.getvalue())
    try:
        shown = doc[0].get_rotation()
    finally:
        doc.close()
    with pikepdf.open(io.BytesIO(buf.getvalue())) as pdf:
        assert orient._page_rotate(pdf.pages[0].obj) == shown
    assert orient.normalize_rotate(value) == shown


# --------------------------------------------------------------------------- bake

CROPS = [None, (30, 40, 500, 700), (30.5, 40, 500, 700.25)]


@pytest.mark.parametrize("crop", CROPS, ids=["plain", "offset-crop", "fractional-crop"])
@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_bake_renders_identically_and_keeps_images(crop: Any, degrees: int) -> None:
    raw = build.pdf_bytes([rich_page(cropbox=crop)])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        before = page_raw_images(pdf, 0)
        orient.bake(pdf, 0, degrees)  # type: ignore[arg-type]
        buf = io.BytesIO()
        pdf.save(buf, deterministic_id=True)
    baked = buf.getvalue()
    assert same_render(render(raw, rotation=degrees), render(baked))
    with pikepdf.open(io.BytesIO(baked)) as out:
        page = out.pages[0].obj
        assert int(page.get("/Rotate", 0)) == 0
        assert "/CropBox" not in page
        w = (crop[2] - crop[0]) if crop else 612
        h = (crop[3] - crop[1]) if crop else 792
        media = [float(v) for v in page.MediaBox]
        assert media == ([0, 0, h, w] if degrees in (90, 270) else [0, 0, w, h])
        assert page_raw_images(out, 0) == before                 # stored image bytes untouched
        assert orient.FORM_NAME in page.Resources.XObject


def test_bake_is_deterministic() -> None:
    raw = build.pdf_bytes([rich_page(cropbox=(30, 40, 500, 700))])
    outs = []
    for _ in range(2):
        with pikepdf.open(io.BytesIO(raw)) as pdf:
            orient.bake(pdf, 0, 90)
            buf = io.BytesIO()
            pdf.save(buf, deterministic_id=True)
            outs.append(buf.getvalue())
    assert outs[0] == outs[1]


def test_bake_honours_inherited_boxes_and_rotate() -> None:
    raw = build.pdf_bytes([rich_page()])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        page = pdf.pages[0].obj
        parent = page.Parent
        parent.MediaBox = page.MediaBox
        parent.Rotate = 90
        del page["/MediaBox"]
        buf = io.BytesIO()
        pdf.save(buf)
    inherited = buf.getvalue()
    shown = render(inherited)
    assert shown.shape == (612, 792)                           # pdfium honours the inherited flag
    src = io.BytesIO(inherited)
    with pikepdf.open(src) as pdf:
        assert orient._page_rotate(pdf.pages[0].obj) == 90
        orient.bake(pdf, 0, 90)                               # bake the flag itself
        out = io.BytesIO()
        pdf.save(out)
    assert same_render(shown, render(out.getvalue()))


def test_bake_empty_page_and_bad_degrees() -> None:
    raw = build.pdf_bytes(1)
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        orient.bake(pdf, 0, 270)
        buf = io.BytesIO()
        pdf.save(buf)
        with pytest.raises(ValueError):
            orient.bake(pdf, 0, 0)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            orient.bake(pdf, 0, 45)  # type: ignore[arg-type]
    assert same_render(render(raw, rotation=270), render(buf.getvalue()))


def test_bake_maps_link_rectangles() -> None:
    raw = build.pdf_bytes([rich_page()])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        link = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Link, Rect=[100, 700, 200, 720],
            QuadPoints=[100, 720, 200, 720, 100, 700, 200, 700], Border=[0, 0, 0],
            A=pikepdf.Dictionary(S=pikepdf.Name.URI, URI=pikepdf.String("https://example.com/"))))
        pdf.pages[0].obj.Annots = pikepdf.Array([link])
        orient.bake(pdf, 0, 90)
        rect = [float(v) for v in pdf.pages[0].obj.Annots[0].Rect]
        quads = [float(v) for v in pdf.pages[0].obj.Annots[0].QuadPoints]
    # cw 90 on a 612x792 page: (x, y) -> (y, 612 - x)
    assert rect == [700, 412, 720, 512]
    assert quads[:4] == [720, 512, 720, 412]


# --------------------------------------------------------------------------- normalize

def dec(page: int, turn: int, confident: bool = True, by: str = "vote") -> OrientDecision:
    return OrientDecision(page, turn, by, confident, "vote_confident" if confident else "no_signal")  # type: ignore[arg-type]


def test_normalize_mixed_document(tmp_path: Path) -> None:
    pages = [
        rich_page(),                       # 0: confident turn 90 -> baked
        rich_page(rotate=90),              # 1: confident turn 270 -> net 0 -> flag cleared
        rich_page(rotate=180),             # 2: unconfident -> flag baked, display kept
        rich_page(),                       # 3: confident turn 0, no flag -> untouched
        rich_page(rotate=90),              # 4: no decision (not OCR'd) -> flag kept
    ]
    src = build.write_pdf(tmp_path / "in.pdf", pages)
    raw = src.read_bytes()
    dest = tmp_path / "out.pdf"
    rep = orient.normalize(src, dest, [dec(0, 90), dec(1, 270), dec(2, 0, confident=False, by="none"), dec(3, 0)])
    assert rep.changed
    assert rep.baked == [0, 2] and rep.cleared == [1] and rep.kept_flag == [4] and rep.kept_annotated == []
    assert src.read_bytes() == raw                                     # the input is never modified
    out = dest.read_bytes()
    assert same_render(render(raw, 0, rotation=90), render(out, 0))
    assert same_render(render(raw, 1, rotation=270), render(out, 1))   # upright now
    assert same_render(render(raw, 2), render(out, 2))                 # displayed look kept
    assert same_render(render(raw, 3), render(out, 3))
    assert same_render(render(raw, 4), render(out, 4))
    with pikepdf.open(dest) as pdf:
        rot = [int(p.obj.get("/Rotate", 0)) for p in pdf.pages]
        assert rot == [0, 0, 0, 0, 90]
        for i in range(5):
            with pikepdf.open(src) as before:
                assert page_raw_images(pdf, i) == page_raw_images(before, i)
    assert [orient.page_action(rep, i) for i in range(5)] == ["baked", "cleared", "baked", "none", "kept_flag"]


def test_normalize_unconfident_turn_is_ignored(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "in.pdf", [rich_page(rotate=90)])
    # a decision that claims a turn but isn't confident must not clear the flag
    rep = orient.normalize(src, tmp_path / "out.pdf", [OrientDecision(0, 270, "none", False, "no_signal")])
    assert rep.baked == [0] and rep.cleared == []
    assert same_render(render(src), render(tmp_path / "out.pdf"))


def test_normalize_leaves_dest_untouched_when_nothing_changes(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "in.pdf", [rich_page(), rich_page(rotate=90)])
    dest = tmp_path / "out.pdf"
    rep = orient.normalize(src, dest, [dec(0, 0)])
    assert not rep.changed and rep.kept_flag == [1]
    assert not dest.exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_normalize_keeps_annotated_pages(tmp_path: Path) -> None:
    raw = build.pdf_bytes([rich_page(), rich_page(), rich_page(rotate=90)])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        note = pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Text, Rect=[50, 50, 70, 70],
                                  Contents=pikepdf.String("example note"))
        widget = pikepdf.Dictionary(Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Widget, Rect=[50, 90, 150, 110],
                                    FT=pikepdf.Name.Tx, T=pikepdf.String("field1"))
        pdf.pages[0].obj.Annots = pikepdf.Array([pdf.make_indirect(note)])
        pdf.pages[1].obj.Annots = pikepdf.Array([pdf.make_indirect(widget)])
        pdf.pages[2].obj.Annots = pikepdf.Array([pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Square, Rect=[10, 10, 40, 40]))])
        src = tmp_path / "in.pdf"
        pdf.save(src)
    dest = tmp_path / "out.pdf"
    # page 2: turn 270 on /Rotate 90 is net 0, so its flag is cleared (no bake needed) even though annotated
    rep = orient.normalize(src, dest, [dec(0, 90), dec(1, 180), dec(2, 270)])
    assert rep.kept_annotated == [0, 1] and rep.baked == [] and rep.cleared == [2]
    with pikepdf.open(dest) as pdf:
        assert [int(p.obj.get("/Rotate", 0)) for p in pdf.pages] == [0, 0, 0]
        assert [float(v) for v in pdf.pages[0].obj.MediaBox] == [0, 0, 612, 792]
    assert same_render(render(src, 0), render(dest, 0))                # display kept
    assert same_render(render(src, 1), render(dest, 1))
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.pages[0].obj.Annots = pikepdf.Array([pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Square, Rect=[0, 0, 1, 1]))])
        with pytest.raises(ValueError):
            orient.bake(pdf, 0, 90)                  # bake itself refuses an annotated page


@pytest.mark.parametrize("value,shown", [(135, 90), (-135, 270), (179, 90), (100, 90)])
def test_normalize_malformed_rotate_follows_the_display(tmp_path: Path, value: int, shown: int) -> None:
    # the stored content is upright; pdfium displays the page turned `shown` clockwise, so the vote
    # calls for (360 - shown) % 360 and the output must be upright
    raw = build.pdf_bytes([rich_page()])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.pages[0].obj.Rotate = value
        src = tmp_path / "in.pdf"
        pdf.save(src)
    displayed = render(src)
    assert same_render(displayed, render(raw, rotation=shown))
    turn = (360 - shown) % 360
    dest = tmp_path / "out.pdf"
    rep = orient.normalize(src, dest, [dec(0, turn)])
    assert rep.cleared == [0] and rep.baked == []
    out = render(dest)
    assert same_render(out, render(src, rotation=turn))     # what verify.renders_identical expects
    assert same_render(out, render(raw))                    # upright
    with pikepdf.open(dest) as pdf:
        assert int(pdf.pages[0].obj.Rotate) == 0
    # an unconfident decision bakes the displayed look in, whatever the malformed flag
    dest2 = tmp_path / "out2.pdf"
    rep2 = orient.normalize(src, dest2, [dec(0, 0, confident=False, by="none")])
    assert rep2.baked == [0]
    assert same_render(render(dest2), displayed)


def test_normalize_clears_a_rotate_that_displays_as_zero(tmp_path: Path) -> None:
    raw = build.pdf_bytes([rich_page()])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.pages[0].obj.Rotate = 45
        src = tmp_path / "in.pdf"
        pdf.save(src)
    dest = tmp_path / "out.pdf"
    rep = orient.normalize(src, dest, [dec(0, 0)])
    assert rep.cleared == [0]
    with pikepdf.open(dest) as pdf:
        assert int(pdf.pages[0].obj.Rotate) == 0
    assert same_render(render(dest), render(src))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_normalize_annotated_page_without_a_turn_is_kept_flag(tmp_path: Path) -> None:
    raw = build.pdf_bytes([rich_page(rotate=90), rich_page(rotate=180)])
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        for i in range(2):
            pdf.pages[i].obj.Annots = pikepdf.Array([pdf.make_indirect(pikepdf.Dictionary(
                Type=pikepdf.Name.Annot, Subtype=pikepdf.Name.Square, Rect=[10, 10, 40, 40]))])
        src = tmp_path / "in.pdf"
        pdf.save(src)
    rep = orient.normalize(src, tmp_path / "out.pdf",
                           [dec(0, 0, confident=False, by="none"), dec(1, 0)])
    assert rep.kept_flag == [0, 1] and rep.kept_annotated == [] and not rep.changed
    assert [orient.page_action(rep, i) for i in range(2)] == ["kept_flag", "kept_flag"]
    rep = orient.normalize(src, tmp_path / "out.pdf", [dec(0, 90), dec(1, 0)])
    assert rep.kept_annotated == [0] and rep.kept_flag == [1]


ENC_PERMS = dict(extract=False, print_lowres=False, print_highres=False, modify_other=False)


def _encrypted(tmp_path: Path, user: str) -> Path:
    src = tmp_path / "enc.pdf"
    with pikepdf.open(io.BytesIO(build.pdf_bytes([rich_page(), rich_page(rotate=90)]))) as pdf:
        pdf.save(src, encryption=pikepdf.Encryption(owner="owner-example", user=user,
                                                    allow=pikepdf.Permissions(**ENC_PERMS)))
    return src


def test_normalize_keeps_permissions_only_encryption(tmp_path: Path) -> None:
    src = _encrypted(tmp_path, "")
    with pikepdf.open(src) as pdf:
        before = (pdf.encryption.R, pdf.encryption.P, pdf.allow)
    dest = tmp_path / "out.pdf"
    rep = orient.normalize(src, dest, [dec(0, 90), dec(1, 270)])
    assert rep.baked == [0] and rep.cleared == [1]
    with pikepdf.open(dest) as pdf:
        assert pdf.is_encrypted
        assert (pdf.encryption.R, pdf.encryption.P, pdf.allow) == before
        assert not pdf.allow.extract and not pdf.allow.print_highres
        assert [int(p.obj.get("/Rotate", 0)) for p in pdf.pages] == [0, 0]
    assert same_render(render(dest, 0), render(src, 0, rotation=90))


def test_normalize_open_password_input(tmp_path: Path) -> None:
    src = _encrypted(tmp_path, "user-example")
    dest = tmp_path / "out.pdf"
    with pytest.raises(pikepdf.PasswordError):
        orient.normalize(src, dest, [dec(0, 90)])
    assert not dest.exists()
    rep = orient.normalize(src, dest, [dec(0, 90)], password="user-example")
    assert rep.baked == [0]
    with pytest.raises(pikepdf.PasswordError):
        pikepdf.open(dest)                                   # the work copy needs the same password
    with pikepdf.open(dest, password="user-example") as pdf:
        assert pdf.is_encrypted and not pdf.allow.extract


def test_normalize_rejects_bad_decisions(tmp_path: Path) -> None:
    src = build.write_pdf(tmp_path / "in.pdf", 2)
    with pytest.raises(ValueError):
        orient.normalize(src, tmp_path / "o.pdf", [dec(5, 90)])
    with pytest.raises(ValueError):
        orient.normalize(src, tmp_path / "o.pdf", [dec(0, 90), dec(0, 180)])
    with pytest.raises(ValueError):
        orient.normalize(src, tmp_path / "o.pdf", [OrientDecision(0, 45, "vote", True, "vote_confident")])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        orient.normalize(src, src, [dec(0, 90)])
    assert not (tmp_path / "o.pdf").exists()


def test_turn_image_is_clockwise() -> None:
    img = Image.new("L", (4, 2), 255)
    img.putpixel((0, 0), 0)                          # top-left
    turned = orient.turn_image(img, 90)
    assert turned.size == (2, 4)
    assert turned.getpixel((1, 0)) == 0              # top-left goes to top-right on a clockwise turn
    assert orient.turn_image(img, 180).getpixel((3, 1)) == 0
    assert orient.turn_image(img, 270).getpixel((0, 3)) == 0
