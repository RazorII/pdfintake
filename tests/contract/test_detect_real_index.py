"""The probe rule against the real index and the probe settings its packs ship.

Every pack the real index lists with a probe is checked with the settings its own manifest pins
(``detect.probe``: ``min_chars``, ``min_conf``, ``min_line_chars``), as ``pipeline.gather`` passes
them to ``detect.run_probe``, and ``langplan.plan`` decides with the real index:

* a Vietnamese page whose probe read carries stray Thai characters (the horned letters read as
  Thai lookalikes, several per line) stays Vietnamese, although the same characters counted line
  by line without ``detect.PROBE_LINE_MIN_SHARE`` pass the pack's ``min_chars``;
* a Latin page with a Thai block (English or Vietnamese around it) goes to the Thai pack through
  its probe, and so does a page of Thai lines with English words in them.

The synthetic reads are built from the shipped settings (their confidence clears ``min_conf``,
every line holds ``min_line_chars``, the stray lines stay under the line share), so they hold
whatever probe settings a pack ships; only the rule's outcome is asserted.

With the pinned test models (``conftest.pinned_assets``; skipped, naming the files, without them)
the same holds for the Thai probe's real recogniser: on a dense page of made-up Vietnamese its
stray Thai characters don't send the page to th (skipped, naming the settings, when the shipped
settings already keep them below ``min_chars`` without the line share); on a page with a shaped
Thai block among English lines, the block does. Every text here is made up.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import pinned_assets
from fixtures import build
from pdfintake.langpacks import manifest
from pdfintake.langpacks.index import Index, load
from pdfintake.ocr import detect, engines, langplan, pipeline
from pdfintake.ocr.detect import MixedScript, PageSignals, ProbeResult
from pdfintake.types import PageImage, ScriptCall

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
REAL_INDEX = REPO / "src" / "pdfintake" / "langpacks" / "index.json"
PACKS = REPO / "packs"

VIETNAMESE = [
    "Mùa mưa năm nay đến sớm hơn mọi năm, nước sông dâng lên từng ngày.",
    "Người dân trong làng dựng lại hàng rào tre trước khi cơn bão đi qua.",
    "Trên cánh đồng, những thửa ruộng mới cấy đã xanh mướt sau hai tuần.",
    "Ở chợ phiên cuối tuần, người ta bán rau, cá và những giỏ trứng tươi.",
    "Thư viện của trường mở cửa đến tối để học sinh có chỗ ôn bài yên tĩnh.",
    "Con đường dẫn lên đồi được lát đá, hai bên trồng hàng cây phượng đỏ.",
    "Mỗi sáng, ông bà dậy sớm tưới nước cho vườn hoa trước khi trời nắng.",
    "Chiếc thuyền nhỏ neo bên bờ sông, chờ nước lên để ra khơi đánh cá.",
    "Buổi chiều, trẻ em chơi đùa trên sân, tiếng cười vang khắp xóm nhỏ.",
    "Người thợ mộc làm xong chiếc bàn gỗ mới cho lớp học ở đầu thôn.",
]
_AS_THAI = str.maketrans({"ư": "น", "ừ": "น", "ứ": "น", "ữ": "น", "ự": "น", "ử": "น",
                          "ơ": "ซ", "ờ": "ซ", "ớ": "ซ", "ở": "ซ", "ợ": "ซ"})
ENGLISH = ["Rivers carry water from the hills down to the sea, and the sea gives it back to the sky.",
           "Clouds form when that vapour cools high above the land, and rain brings the water back.",
           "The garden behind the school has flowers of many colours and a small pond with fish.",
           "In the evening the children play on the field until the lights come on along the road."]
THAI = ["ฝนตกหนักตลอดทั้งคืนที่ผ่านมา", "สวนหลังบ้านมีดอกไม้หลายสี", "ดวงจันทร์โคจรรอบโลกเดือนละหนึ่งรอบ"]
THAI_WITH_ENGLISH = "ห้องสมุดเปิด Monday ถึง Friday"                     # 15 Thai of 27 letters


# --------------------------------------------------------------------------- the real index and settings

@pytest.fixture(scope="module")
def real() -> Index:
    assert REAL_INDEX.is_file(), f"the real index is missing ({REAL_INDEX}; tools/build_index.py writes it)"
    return load(REAL_INDEX)


@pytest.fixture(scope="module")
def base_manifest() -> manifest.Manifest:
    return manifest.parse((PACKS / "en" / "manifest.json").read_bytes())


def _probe_packs(idx: Index) -> list[str]:
    return [e.id for e in idx.entries if langplan._probe_min_chars(e) is not None]


def _settings(pack_id: str, base: manifest.Manifest) -> tuple[Any, str]:
    """The pack's probe block and the script its probe counts, read as ``pipeline.gather`` reads them."""
    m = manifest.parse((PACKS / pack_id / "manifest.json").read_bytes(), base=base)
    assert m.detect.probe is not None, f"the index says {pack_id} has a probe, its manifest has none"
    script = pipeline._probe_script(SimpleNamespace(manifest=m))
    assert script is not None, f"{pack_id}'s probe counts no non-Latin script"
    return m.detect.probe, script


class _Session:
    def __init__(self, lines: list[tuple[str, float]]) -> None:
        self.read_ = build.engine_read("probe", lines=[build.line(t, conf=c) for t, c in lines], role="probe")

    def read(self, page: Any, *, timeout_s: float) -> Any:
        return self.read_


def _run_probe(page: PageImage, session: Any, pack_id: str, probe: Any, script: str, **kw: Any) -> ProbeResult:
    return detect.run_probe(page, session, pack=pack_id, script=script, min_conf=float(probe.min_conf),
                            min_line_chars=int(probe.min_line_chars), min_chars=int(probe.min_chars), **kw)


def _signals(i: int, eng_lines: list[str], probe: ProbeResult) -> PageSignals:
    eng = build.engine_read("tesseract", eng_lines)
    return PageSignals(index=i, osd=None, script=ScriptCall("Latin", 0.0, "osd", 88.0), eng_read=eng,
                       checker_read=build.engine_read("rapidocr", eng_lines, role="checker"), probes={probe.pack: probe},
                       windows=detect.language_windows(eng), mixed=MixedScript(False, None, 0, None))


def _plan_page(idx: Index, sig: PageSignals) -> langplan.PagePlan:
    p = langplan.plan([sig], idx, lang=[], force_lang=[], installed={e.id for e in idx.entries}, fetch="auto")
    return langplan.resolve_unsure(p, idx).pages[0]


def test_the_real_index_lists_the_thai_probe(real: Index, base_manifest: manifest.Manifest):
    assert "th" in _probe_packs(real)
    for pid in _probe_packs(real):
        probe, script = _settings(pid, base_manifest)
        assert probe.min_chars >= 1 and 0.0 <= probe.min_conf <= 1.0 and probe.min_line_chars >= 1
        assert script in real.get(pid).scripts


# --------------------------------------------------------------------------- the rule on synthetic probe reads

def _conf(probe: Any) -> float:
    """A line confidence at or above the pack's ``min_conf``, whatever it is."""
    return min(1.0, float(probe.min_conf) + 0.5 * (1.0 - float(probe.min_conf)))


def _lengthened(text: str, min_thai: int) -> str:
    """``text`` repeated until it holds at least ``min_thai`` Thai characters. Repeating keeps the
    line's Thai share, so a line built here passes the pack's ``min_line_chars`` whatever it is
    and the line-share rule alone decides whether it counts."""
    n, _ = detect._line_script_counts(text, "Thai")
    assert n >= 1, text
    return " ".join([text] * max(1, -(-int(min_thai) // n)))


def _thai_count(text: str) -> int:
    return detect._line_script_counts(text, "Thai")[0]


def _share(text: str) -> float:
    n, other = detect._line_script_counts(text, "Thai")
    return n / (n + other)


def _stray_lines(probe: Any) -> list[tuple[str, float]]:
    """Vietnamese lines read with stray Thai, at a confidence above the bar, each holding at least
    the pack's ``min_line_chars`` Thai characters (lengthened in proportion with its Vietnamese), until
    the stray characters counted without the line share reach twice the pack's ``min_chars``."""
    out: list[tuple[str, float]] = []
    total = 0
    while total < 2 * probe.min_chars:
        t = _lengthened(VIETNAMESE[len(out) % len(VIETNAMESE)].translate(_AS_THAI), probe.min_line_chars)
        assert _share(t) < detect.PROBE_LINE_MIN_SHARE, t     # stray by construction, not by the settings
        out.append((t, _conf(probe)))
        total += _thai_count(t)
    return out


def _thai_lines(probe: Any) -> list[tuple[str, float]]:
    """A Thai line with English words in it, then Thai lines, until they hold the pack's ``min_chars``."""
    first = _lengthened(THAI_WITH_ENGLISH, probe.min_line_chars)
    assert _share(first) >= detect.PROBE_LINE_MIN_SHARE, first
    out: list[tuple[str, float]] = [(first, _conf(probe))]
    total = _thai_count(first)
    while total < probe.min_chars:
        t = _lengthened(THAI[len(out) % len(THAI)], probe.min_line_chars)
        out.append((t, _conf(probe)))
        total += _thai_count(t)
    return out


def test_stray_thai_on_a_vietnamese_page_does_not_take_it_from_vietnamese(real: Index, base_manifest, tmp_path):
    probe, script = _settings("th", base_manifest)
    assert script == "Thai"
    page = build.page_image(tmp_path)
    stray = _stray_lines(probe)
    pr = _run_probe(page, _Session(stray), "th", probe, script)
    assert pr.chars < probe.min_chars and pr.chars == 0
    old = _run_probe(page, _Session(stray), "th", probe, script, min_line_share=0.0)
    assert old.chars >= 2 * probe.min_chars                         # what the probe used to count
    pp = _plan_page(real, _signals(0, VIETNAMESE[:6], pr))
    assert pp.pack != "th"
    if real.get("vi") is not None:
        assert (pp.pack, pp.composition) == ("vi", "language")
    taken = _plan_page(real, _signals(0, VIETNAMESE[:6], old))
    assert (taken.pack, taken.composition) == ("th", "probe")


def test_a_thai_block_still_goes_to_the_thai_pack(real: Index, base_manifest, tmp_path):
    probe, script = _settings("th", base_manifest)
    page = build.page_image(tmp_path)
    english = [(t, _conf(probe)) for t in ENGLISH]
    thai = _thai_lines(probe)
    lines = english[:2] + thai + english[2:]
    pr = _run_probe(page, _Session(lines), "th", probe, script)
    assert pr.chars >= probe.min_chars and pr.lines == len(thai)
    pp = _plan_page(real, _signals(0, ENGLISH, pr))
    assert (pp.pack, pp.composition) == ("th", "probe") and "mixed_script_page" in pp.warnings
    # A Vietnamese page with the same Thai block, its stray characters around it: still th.
    mixed = _stray_lines(probe)[:4] + thai + _stray_lines(probe)[4:6]
    pr2 = _run_probe(page, _Session(mixed), "th", probe, script)
    assert pr2.chars == pr.chars and pr2.lines == len(thai)
    assert _plan_page(real, _signals(0, VIETNAMESE[:6], pr2)).pack == "th"


# --------------------------------------------------------------------------- the real Thai probe recogniser

def _probe_session(probe: Any) -> Any:
    refs = [r if ":" in r else f"th:{r}" for r in probe.use.models]
    found = pinned_assets(*refs)
    models = {orig: found[ref] for orig, ref in zip(probe.use.models, refs)}
    return engines.get(probe.use.engine).open(probe.use.id, probe.use.role, probe.use.params, models)


def _vietnamese_page(folder: Path) -> PageImage:
    """A dense A4 page of made-up Vietnamese at 300 dpi (9 pt Noto Sans, 60 lines)."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = pinned_assets("en:noto-sans")["en:noto-sans"]
    dpi = 300
    w, h = int(8.27 * dpi), int(11.69 * dpi)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), size=int(9 * dpi / 72))
    y = int(0.7 * dpi)
    for k in range(60):
        draw.text((int(0.6 * dpi), y), VIETNAMESE[k % len(VIETNAMESE)], font=font, fill=0)
        y += int(9 * 1.45 * dpi / 72)
    return build.page_image(folder, image=img, dpi=dpi, name="vietnamese.png")


def _english_with_thai_page(folder: Path) -> PageImage:
    """English lines with a shaped Thai block (HarfBuzz through fpdf2, rendered by pdfium at 300 dpi)."""
    pytest.importorskip("uharfbuzz")
    import pypdfium2 as pdfium
    from fpdf import FPDF

    fonts = pinned_assets("en:noto-sans", "th:noto-sans-thai")
    rows = [(ENGLISH[0], "L"), (ENGLISH[1], "L"), (THAI[0], "T"), (THAI[2], "T"), (ENGLISH[2], "L"), (ENGLISH[3], "L")]
    pdf = FPDF(unit="pt", format=(595.28, 420.0))
    pdf.set_auto_page_break(False)
    pdf.set_margin(0)
    pdf.add_font("L", fname=str(fonts["en:noto-sans"]))
    pdf.add_font("T", fname=str(fonts["th:noto-sans-thai"]))
    pdf.set_text_shaping(True)
    pdf.add_page()
    pdf.c_margin = 0
    y = 60.0
    for text, fam in rows:
        pdf.set_font(fam, size=12)
        pdf.set_xy(40.0, y - 0.8 * 12)
        pdf.cell(w=pdf.get_string_width(text), h=12, text=text)
        y += 30.0
    doc = pdfium.PdfDocument(bytes(pdf.output()))
    try:
        pg = doc[0]
        img = pg.render(scale=300 / 72.0, grayscale=True).to_pil().convert("L")
        pg.close()
    finally:
        doc.close()
    return build.page_image(folder, image=img, dpi=300, name="english-thai.png")


def test_the_real_thai_probe_on_a_vietnamese_page(real: Index, base_manifest, tmp_path):
    probe, script = _settings("th", base_manifest)
    vi_page = _vietnamese_page(tmp_path)
    session = _probe_session(probe)
    try:
        read = session.read(vi_page, timeout_s=detect.PROBE_TIMEOUT_S)
    finally:
        session.close()
    assert read.status == "ok" and len(read.lines) >= 50, (read.status, read.error, len(read.lines))
    replay = SimpleNamespace(read=lambda page, *, timeout_s: read)
    old = _run_probe(vi_page, replay, "th", probe, script, min_line_share=0.0)
    if old.chars < probe.min_chars:
        pytest.skip(f"th's shipped probe (min_chars {probe.min_chars}, min_conf {probe.min_conf}, "
                    f"min_line_chars {probe.min_line_chars}) keeps this page's stray Thai below min_chars "
                    f"without the line share ({old.chars} characters), so the page doesn't exercise it")
    pr = _run_probe(vi_page, replay, "th", probe, script)
    assert pr.chars < probe.min_chars, pr
    pp = _plan_page(real, _signals(0, VIETNAMESE[:6], pr))
    assert pp.pack != "th", pp
    if real.get("vi") is not None:
        assert pp.pack == "vi", pp


def test_the_real_thai_probe_on_a_thai_block(real: Index, base_manifest, tmp_path):
    probe, script = _settings("th", base_manifest)
    th_page = _english_with_thai_page(tmp_path)
    session = _probe_session(probe)
    try:
        pr = _run_probe(th_page, session, "th", probe, script)
    finally:
        session.close()
    assert pr.chars >= probe.min_chars and pr.lines >= 2, pr
    assert _plan_page(real, _signals(0, ENGLISH, pr)).pack == "th"
