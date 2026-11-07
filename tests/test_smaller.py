"""pdf/smaller.py: which images are candidates (R1, R2), the candidate build, and when to offer it.

Every PDF here is built in code from synthetic images. ``build`` calls ``verify.text_layer_identical``
and ``meta.diff``/``meta.snapshot``; when those modules aren't importable yet (they belong to other
build units), the ``deps`` fixture puts small stand-ins in their place.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import sys
import types
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pikepdf
import pytest
from PIL import Image

from fixtures import build as fb
from pdfintake import quality
from pdfintake.pdf import smaller


# --------------------------------------------------------------------------- dependencies


def _fake_modules() -> dict[str, types.ModuleType]:
    verify = types.ModuleType("pdfintake.pdf.verify")
    meta = types.ModuleType("pdfintake.pdf.meta")

    @dataclass
    class Check:
        name: str
        ok: bool
        detail: str
        page: int | None = None

    @dataclass
    class MetaDiff:
        kept: list[str]
        changed: list[str]
        dropped: list[str]

    def text_layer_identical(a: Path, b: Path) -> Check:
        import pypdfium2 as pdfium

        da, db = pdfium.PdfDocument(str(a)), pdfium.PdfDocument(str(b))
        try:
            same = len(da) == len(db) and all(
                da[i].get_textpage().get_text_range() == db[i].get_textpage().get_text_range() for i in range(len(da)))
        finally:
            da.close()
            db.close()
        return Check("text_layer_identical", same, "")

    def snapshot(path: Path) -> tuple[dict[str, str], bytes | None]:
        with pikepdf.open(path) as pdf:
            info = pdf.trailer.get("/Info")
            xmp = pdf.Root.get("/Metadata")
            return ({str(k): str(v) for k, v in info.items()} if info is not None else {},
                    bytes(xmp.read_bytes()) if xmp is not None else None)

    def diff(before: tuple, after: tuple) -> MetaDiff:
        kept, changed = [], []
        (kept if before[0] == after[0] else changed).append("Info")
        (kept if before[1] == after[1] else changed).append("XMP")
        return MetaDiff(kept, changed, [])

    verify.Check, verify.text_layer_identical = Check, text_layer_identical
    meta.MetaDiff, meta.snapshot, meta.diff = MetaDiff, snapshot, diff
    return {"verify": verify, "meta": meta}


@pytest.fixture(autouse=True)
def deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import pdfintake.pdf as pkg

    fakes = None
    for name in ("verify", "meta"):
        try:
            importlib.import_module(f"pdfintake.pdf.{name}")
        except ImportError:
            fakes = fakes or _fake_modules()
            monkeypatch.setitem(sys.modules, f"pdfintake.pdf.{name}", fakes[name])
            monkeypatch.setattr(pkg, name, fakes[name], raising=False)


# --------------------------------------------------------------------------- fixtures


def scene(width: int, height: int, *, seed: int = 1, grain: float = 1.0) -> Image.Image:
    """Smooth colour fields with a little luminance grain: a photo that lossless can't shrink."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.stack([150 + 70 * np.sin(xx / 53.0) * np.cos(yy / 71.0),
                    140 + 60 * np.cos(xx / 91.0 - yy / 47.0),
                    130 + 50 * np.sin((xx + yy) / 67.0)], axis=-1)
    img += rng.normal(0.0, grain, (height, width, 1))
    return Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8))


def one_image_pdf(path: Path, image, *, width_pt: float, height_pt: float, **page) -> Path:
    return fb.write_pdf(path, [fb.PageSpec(width=width_pt, height=height_pt, images=(fb.ImageSpec(image),), **page)])


def page_images(path: Path) -> dict[tuple[int, str], dict]:
    """Every image a page's resources name, keyed by (0-based page, resource name)."""
    out = {}
    with pikepdf.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            for name, obj in page.obj.Resources.get("/XObject", {}).items():
                if obj.get("/Subtype") != "/Image":
                    continue
                mask = obj.get("/SMask")
                out[(i, str(name))] = {"objgen": obj.objgen, "filter": str(obj.get("/Filter")), "w": int(obj.Width),
                                       "h": int(obj.Height), "cs": str(obj.get("/ColorSpace")),
                                       "raw": bytes(obj.read_raw_bytes()),
                                       "smask": bytes(mask.read_raw_bytes()) if mask is not None else None}
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- R1


def test_photo_like_flate_page_triggers_r1_and_saves_60pct(tmp_path: Path) -> None:
    src = one_image_pdf(tmp_path / "photo.pdf", scene(1200, 900), width_pt=432, height_pt=324)   # 200 ppi
    cands = smaller.find(src)
    assert len(cands) == 1
    c = cands[0]
    assert (c.rule, c.page, c.target_ppi) == ("R1", 0, None)
    assert c.source["filter"] == "FlateDecode" and c.source["ppi"] == pytest.approx(200.0)

    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    assert len(rep.pages) == 1 and rep.kept_lossless == []
    row = rep.pages[0]
    assert row["page"] == 1 and row["rule"] == "R1" and row["to"]["filter"] == "DCTDecode"
    assert row["to"]["q"] in (88, 90, 92, 95)
    assert row["score"] >= 90.0 and rep.min_score == row["score"]
    saving = 100 * (rep.bytes_before - rep.bytes_after) / rep.bytes_before
    assert saving >= 60.0
    assert rep.text_layer_identical and smaller.worth_offering(rep)
    after = page_images(tmp_path / "candidate.pdf")[(0, "/Im0")]
    assert after["filter"] == "/DCTDecode"
    assert (after["w"], after["h"], after["cs"]) == (1200, 900, "/DeviceRGB")


def test_the_lowest_passing_quality_is_chosen(tmp_path: Path) -> None:
    img = scene(1200, 900)
    src = one_image_pdf(tmp_path / "photo.pdf", img, width_pt=432, height_pt=324)
    rep = smaller.build(src, tmp_path / "candidate.pdf", smaller.find(src))
    q = rep.pages[0]["to"]["q"]
    ref = np.asarray(img)
    for lower in (88, 90, 92, 95):
        if lower >= q:
            break
        data = smaller._encode(img, lower)
        with Image.open(__import__("io").BytesIO(data)) as dec:
            assert quality.worst_tile(ref, np.asarray(dec.convert("RGB"))).score < 90.0


# --------------------------------------------------------------------------- R2


def test_r2_size_rule() -> None:
    assert smaller.r2_size(6000, 4000, 600.0) == (3000, 2000, 300.0)      # 300 ppi binds
    w, h, ppi = smaller.r2_size(3000, 2400, 600.0)                        # the 2,000 px floor binds
    assert (w, h) == (2000, 1600) and ppi == pytest.approx(400.0)
    assert smaller.r2_size(2000, 1500, 600.0) is None                     # already at the floor
    assert smaller.r2_size(5000, 4000, 400.0) is None                     # not above 400 ppi


def test_600ppi_page_triggers_r2_with_the_2000px_floor(tmp_path: Path) -> None:
    jpeg = fb.jpeg_bytes(scene(3000, 2400, seed=2), quality=92)
    src = one_image_pdf(tmp_path / "hires.pdf", jpeg, width_pt=360, height_pt=288)              # 600 ppi
    cands = smaller.find(src)
    assert [c.rule for c in cands] == ["R2"]
    assert cands[0].source["ppi"] == pytest.approx(600.0) and cands[0].target_ppi == pytest.approx(400.0)
    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    assert len(rep.pages) == 1 and rep.pages[0]["score"] >= 90.0
    after = page_images(tmp_path / "candidate.pdf")[(0, "/Im0")]
    assert (after["w"], after["h"]) == (2000, 1600) and after["filter"] == "/DCTDecode"
    assert smaller.worth_offering(rep)


# --------------------------------------------------------------------------- never candidates


def test_typical_dct_scan_gives_no_candidates(tmp_path: Path) -> None:
    jpeg = fb.jpeg_bytes(fb.text_like(1500, 1200, lines=30).convert("RGB"), quality=85)
    src = one_image_pdf(tmp_path / "scan.pdf", jpeg, width_pt=360, height_pt=288)               # 300 ppi
    assert smaller.find(src) == []


def test_bilevel_and_compressible_grey_give_no_candidates(tmp_path: Path) -> None:
    text = fb.text_like(1500, 1200, lines=30)
    pages = [fb.PageSpec(width=360, height=288, images=(fb.ImageSpec(text.convert("1")),)),
             fb.PageSpec(width=360, height=288, images=(fb.ImageSpec(text),))]                   # grey, < 4 bpp stored
    src = fb.write_pdf(tmp_path / "mixed.pdf", pages)
    assert smaller.find(src) == []


def _add_smask(src: Path, dest: Path, *, matte: bool = False) -> None:
    with pikepdf.open(src) as pdf:
        img = pdf.pages[0].obj.Resources.XObject.Im0
        w, h = int(img.Width), int(img.Height)
        alpha = scene(w, h, seed=9).convert("L")
        mask = pikepdf.Stream(pdf, zlib.compress(alpha.tobytes(), 9))
        mask.Type, mask.Subtype = pikepdf.Name.XObject, pikepdf.Name.Image
        mask.Width, mask.Height, mask.BitsPerComponent = w, h, 8
        mask.ColorSpace, mask.Filter = pikepdf.Name.DeviceGray, pikepdf.Name.FlateDecode
        if matte:
            mask.Matte = pikepdf.Array([1, 1, 1])
        img.SMask = pdf.make_indirect(mask)
        pdf.save(dest, deterministic_id=True)


def test_soft_masks_are_never_candidates_and_are_kept(tmp_path: Path) -> None:
    base = one_image_pdf(tmp_path / "base.pdf", scene(1200, 900), width_pt=432, height_pt=324)
    src = tmp_path / "smask.pdf"
    _add_smask(base, src)
    with pikepdf.open(src) as pdf:
        image_key = pdf.pages[0].obj.Resources.XObject.Im0.objgen
    cands = smaller.find(src)
    assert [c.xobject for c in cands] == [image_key]                              # the mask itself is not one
    before = page_images(src)[(0, "/Im0")]
    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    assert len(rep.pages) == 1
    after = page_images(tmp_path / "candidate.pdf")[(0, "/Im0")]
    assert after["filter"] == "/DCTDecode"
    assert after["smask"] == before["smask"] is not None                           # still attached, untouched


def test_an_image_whose_soft_mask_has_matte_is_never_resized(tmp_path: Path) -> None:
    # ISO 32000, 11.6.5 (soft-mask images): a soft mask with /Matte must have its parent's Width and Height.
    base = one_image_pdf(tmp_path / "base.pdf", scene(2400, 2000, seed=3), width_pt=288, height_pt=240)  # 600 ppi
    plain = smaller.find(base)
    assert [c.rule for c in plain] == ["R2"]                                       # without /Matte: R2
    src = tmp_path / "matte.pdf"
    _add_smask(base, src, matte=True)
    cands = smaller.find(src)
    assert [(c.rule, c.target_ppi) for c in cands] == [("R1", None)]               # falls back to R1, own size
    forged = [dataclasses.replace(cands[0], rule="R2", target_ppi=plain[0].target_ppi)]
    rep = smaller.build(src, tmp_path / "forged.pdf", forged)                       # build guards it too
    assert rep.pages == [] and "/Matte" in rep.kept_lossless[0]["reason"]
    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    with pikepdf.open(tmp_path / "candidate.pdf") as pdf:
        im = pdf.pages[0].obj.Resources.XObject.Im0
        assert (int(im.Width), int(im.Height)) == (int(im.SMask.Width), int(im.SMask.Height)) == (2400, 2000)
        assert "/Matte" in im.SMask


def _strip_pdf(path: Path, image: Image.Image, w_pt: float, h_pt: float) -> Path:
    return fb.write_pdf(path, [fb.PageSpec(images=(fb.ImageSpec(image, x=36, y=36, w=w_pt, h=h_pt),))])


def test_a_thin_r2_strip_is_scored_or_kept_never_a_crash(tmp_path: Path) -> None:
    # 20 x 3000 px at 600 ppi: R2 makes it 13 x 2000, whose short side alone gives 390 ppi.
    src = _strip_pdf(tmp_path / "strip.pdf", scene(20, 3000, seed=5), 2.4, 360)
    cands = smaller.find(src)
    assert [(c.rule, c.target_ppi) for c in cands] == [("R2", pytest.approx(400.0))]
    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    assert len(rep.pages) + len(rep.kept_lossless) == 1
    assert all(k["reason"].startswith("scored ") for k in rep.kept_lossless)      # it was scored
    if rep.pages:
        after = page_images(tmp_path / "candidate.pdf")[(0, "/Im0")]
        assert (after["w"], after["h"]) == (13, 2000)


def test_a_strip_too_thin_to_score_stays_lossless(tmp_path: Path) -> None:
    # 7 x 24000 px shown 0.7 pt wide: 3 px across at 300 ppi, below SSIMULACRA 2's 8 px minimum.
    noise = Image.fromarray(np.random.default_rng(1).integers(0, 256, (24000, 7, 3), dtype=np.uint8))
    src = _strip_pdf(tmp_path / "thin.pdf", noise, 0.7, 720)
    cands = smaller.find(src)
    assert [c.rule for c in cands] == ["R2"]
    before = page_images(src)[(0, "/Im0")]
    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    assert rep.pages == [] and len(rep.kept_lossless) == 1
    assert rep.kept_lossless[0]["reason"].startswith("the image could not be scored")
    assert page_images(tmp_path / "candidate.pdf")[(0, "/Im0")]["raw"] == before["raw"]


# --------------------------------------------------------------------------- what build keeps


def _layered_pdf(tmp_path: Path) -> Path:
    """An image inside a form plus an invisible text layer, with Info and XMP."""
    page = fb.PageSpec(width=432, height=324, in_form=True, images=(fb.ImageSpec(scene(1200, 900)),),
                       texts=(fb.TextSpec("Example invisible layer 4521", x=40, y=200, mode=3),))
    raw = tmp_path / "raw.pdf"
    fb.write_pdf(raw, [page])
    out = tmp_path / "layered.pdf"
    with pikepdf.open(raw) as pdf:
        pdf.docinfo["/Title"] = "Example title"
        pdf.docinfo["/Keywords"] = "sample, example"
        with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as xmp:
            xmp["dc:title"] = "Example title"
        pdf.save(out, deterministic_id=True)
    return out


def _streams(path: Path) -> list[bytes]:
    out = []
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            out.append(bytes(page.obj.Contents.read_raw_bytes()))
            for name, xo in page.obj.Resources.XObject.items():
                if xo.get("/Subtype") == "/Form":
                    out.append(bytes(xo.read_raw_bytes()))
        out.append(bytes(pdf.Root.Metadata.read_raw_bytes()))
    return out


def test_content_and_layer_streams_stay_byte_identical(tmp_path: Path) -> None:
    src = _layered_pdf(tmp_path)
    cands = smaller.find(src)
    assert [c.rule for c in cands] == ["R1"]                                       # found through the form
    assert cands[0].source["ppi"] == pytest.approx(200.0)
    dest = tmp_path / "candidate.pdf"
    rep = smaller.build(src, dest, cands)
    assert len(rep.pages) == 1
    assert _streams(dest) == _streams(src)
    assert rep.text_layer_identical
    assert rep.metadata.changed == [] and rep.metadata.dropped == []
    with pikepdf.open(dest) as pdf:
        assert str(pdf.docinfo["/Title"]) == "Example title" and str(pdf.docinfo["/Keywords"]) == "sample, example"


def test_a_page_below_the_bar_stays_lossless(tmp_path: Path) -> None:
    pages = [fb.PageSpec(width=432, height=324, images=(fb.ImageSpec(scene(1200, 900)),)),
             fb.PageSpec(width=396, height=360, images=(fb.ImageSpec(fb.noise(1100, 1000, seed=4)),))]
    src = fb.write_pdf(tmp_path / "two.pdf", pages)
    cands = smaller.find(src)
    assert [c.rule for c in cands] == ["R1", "R1"]
    before = page_images(src)
    rep = smaller.build(src, tmp_path / "candidate.pdf", cands)
    assert [p["page"] for p in rep.pages] == [1]
    assert len(rep.kept_lossless) == 1 and rep.kept_lossless[0]["page"] == 2
    assert rep.kept_lossless[0]["reason"].startswith("scored ") and "below the bar of 90" in rep.kept_lossless[0]["reason"]
    after = page_images(tmp_path / "candidate.pdf")
    assert after[(1, "/Im0")]["raw"] == before[(1, "/Im0")]["raw"] and after[(1, "/Im0")]["filter"] == "/FlateDecode"
    assert after[(0, "/Im0")]["filter"] == "/DCTDecode"


def test_two_builds_give_the_same_sha256(tmp_path: Path) -> None:
    src = _layered_pdf(tmp_path)
    cands = smaller.find(src)
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    ra = smaller.build(src, a, cands)
    rb = smaller.build(src, b, cands)
    assert sha(a) == sha(b) and ra.pages == rb.pages


def test_build_refuses_to_overwrite_its_source(tmp_path: Path) -> None:
    src = one_image_pdf(tmp_path / "photo.pdf", scene(1200, 900), width_pt=432, height_pt=324)
    with pytest.raises(ValueError):
        smaller.build(src, src, smaller.find(src))


# --------------------------------------------------------------------------- the offer


def _metadiff(kept: list[str], changed: list[str], dropped: list[str]):
    return importlib.import_module("pdfintake.pdf.meta").MetaDiff(kept, changed, dropped)


def _report(before: int, after: int, **kw) -> smaller.SmallerReport:
    row = {"page": 1, "rule": "R1", "from": {"filter": "FlateDecode", "bytes": before},
           "to": {"filter": "DCTDecode", "q": 92, "bytes": after}, "score": 92.0}
    fields = {"pages": [row], "kept_lossless": [], "text_layer_identical": True,
              "metadata": _metadiff(["Info/Title", "XMP"], [], []), "min_score": 92.0}
    fields.update(kw)
    return smaller.SmallerReport(bytes_before=before, bytes_after=after, **fields)


def test_threshold_is_30pct_with_no_byte_floor() -> None:
    assert not smaller.worth_offering(_report(1000, 701))           # 29.9%
    assert smaller.worth_offering(_report(1000, 700))               # 30.0%
    assert smaller.worth_offering(_report(10_000, 7_000))           # a small file saving 30% is offered
    assert smaller.worth_offering(_report(400, 280))
    assert not smaller.worth_offering(_report(10_000, 7_000), min_bytes=1_000_000)
    assert not smaller.worth_offering(_report(1000, 500), min_pct=60.0)


def test_never_offered_without_a_change_or_with_a_broken_layer() -> None:
    assert not smaller.worth_offering(_report(1000, 100, pages=[]))
    assert not smaller.worth_offering(_report(1000, 100, text_layer_identical=False))
    assert not smaller.worth_offering(_report(1000, 1000))
    assert not smaller.worth_offering(_report(0, 0))


def test_never_offered_when_metadata_changed_or_was_dropped() -> None:
    assert smaller.worth_offering(_report(1000, 100))
    assert not smaller.worth_offering(_report(1000, 100, metadata=_metadiff([], ["Info/Title"], [])))
    assert not smaller.worth_offering(_report(1000, 100, metadata=_metadiff(["Info/Title"], [], ["XMP"])))


# --------------------------------------------------------------------------- geometry and limits


def test_an_image_shown_twice_is_judged_at_its_largest_placement(tmp_path: Path) -> None:
    jpeg = fb.jpeg_bytes(scene(3000, 2400, seed=2), quality=92)
    # full page: 600 ppi; a second, smaller placement of the same image: 1200 ppi
    src = one_image_pdf(tmp_path / "twice.pdf", jpeg, width_pt=360, height_pt=288,
                        extra=b"q 180 0 0 144 36 36 cm /Im0 Do Q")
    cands = smaller.find(src)
    assert len(cands) == 1 and cands[0].source["ppi"] == pytest.approx(600.0)


def test_r1_needs_a_megapixel(tmp_path: Path) -> None:
    small = one_image_pdf(tmp_path / "small.pdf", scene(999, 1000), width_pt=360, height_pt=360)
    big = one_image_pdf(tmp_path / "big.pdf", scene(1000, 1000), width_pt=360, height_pt=360)
    assert smaller.find(small) == []
    assert [c.rule for c in smaller.find(big)] == ["R1"]


def test_the_bar_stays_90_on_libjxls_scale() -> None:
    # decided 2026-11-05: the score became libjxl's own (quality.METRIC_ID ssimulacra2-libjxl/1);
    # the bar it is held to stays 90, the value measured with that tool.
    import inspect

    assert inspect.signature(smaller.build).parameters["bar"].default == 90.0
    assert quality.METRIC_ID.startswith("ssimulacra2-libjxl/1;")
