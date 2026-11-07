"""flow.py and doctor.py: the run and apply orchestration, its records and its one writer.

Most tests use fake OCR engines registered through a test-only override of ``engines._REGISTRY``
and packs built in code (a box-glyph font for the layer), with the store and index seams of
``flow`` pointed at them, so a run is fast and deterministic. One test at the end uses the real
engines and the pinned English models (``conftest.pinned_assets``; it skips, naming the files,
when they aren't fetched). Every text, file name and password here is made up.

Every test that runs a file also proves the input was never written, renamed or touched: its
SHA-256, size, mtime and inode are the same afterwards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from fixtures import build
from pdfintake import doctor, flow, fsutil, integrity, record, refusals
from pdfintake.errors import EngineFailure, InputChanged, PackError, StaleRecord, UsageError
from pdfintake.langpacks import index as I
from pdfintake.langpacks import store
from pdfintake.langpacks.manifest import Detect, EngineUse, FontAsset, Manifest, ReaderSet, RuleUse
from pdfintake.ocr import engines
from pdfintake.ocr.engines import Availability
from pdfintake.params import Param, ParamTable
from pdfintake.types import EngineRead, PageImage

KEY = "test-os"                          # a made-up platform key: the fake engines run only here
REPO = Path(__file__).resolve().parents[1]
FIXTURE_INDEX = REPO / "tests" / "fixtures" / "index.test.json"

EN_LINES = ["The example invoice lists seven sample items",
            "and a total of 4521 units for the test account.",
            "Reference AB123456 was noted on the order form."]
HU_LINES = ["A kertészek látják tavasszal, hogy a régi fák és",
            "a virágok minden reggel egyre erősebben árad,",
            "és az erdő csak akkor igazán hűvös, ha már egy",
            "másik vagy még több felhő érkezett; ezt kell",
            "tudni, és az egy felhő már csak vagy még kell."]
PASSWORD = "made up open phrase"         # a made-up password
WRONG = "made up wrong phrase"


# --------------------------------------------------------------------------- fake engines and packs

class FakeSession:
    def __init__(self, engine: FakeOcr, use_id: str, role: str, params: Mapping[str, Any]) -> None:
        self.engine, self.use_id, self.role, self.params = engine, use_id, role, dict(params)

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        lang = str(self.params.get("lang", "en"))
        self.engine.reads.append((self.use_id, lang, page.index))
        texts = self.engine.text(lang, page.index)
        lines = build.lines_for(texts, char_w=14.0, with_words=True)
        return EngineRead(self.use_id, self.engine.name, f"{self.engine.name}:{lang}", self.role,  # type: ignore[arg-type]
                          "ok" if lines else "empty", lines, 0.001, None, {"mean_conf": 0.9})

    def close(self) -> None:
        pass


class FakeOcr:
    api = 1
    MODEL_KINDS = frozenset({"onnx_rec"})
    session_per_thread = True

    def __init__(self, name: str, text: Callable[[str, int], list[str]]) -> None:
        self.name = name
        self.PARAMS = ParamTable(Param("lang", "str", pattern=r"[a-z]{2,3}", default="en"),
                                 Param("timeout_s", "float", min=1.0, max=600.0))
        self.platforms = frozenset({KEY})
        self.text = text
        self.reads: list[tuple[str, str, int]] = []

    def version(self) -> str | None:
        return "1.0"

    def availability(self, params: Mapping[str, Any], models: Mapping[str, Path], platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform", "fake")
        return Availability(True, "available", "")

    def open(self, use_id: str, role: str, params: Mapping[str, Any], models: Mapping[str, Path]) -> FakeSession:
        return FakeSession(self, use_id, role, params)

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence[Any]) -> list[dict[str, Any]]:
        return []

    def contract_case(self) -> None:
        return None


def latin_chars() -> list[int]:
    from pdfintake.ocr import scripts

    cps = set(scripts.SCRIPTS["Latin"].required)
    cps |= set(range(0x20, 0x7F))
    cps |= {cp for cp in range(0xA0, 0x180) if __import__("unicodedata").category(chr(cp))[0] in "LNPSZ"}
    return sorted(cps)


def make_pack(root: Path, pid: str, lang: str, *, base: bool, font: Path) -> store.InstalledPack:
    folder = root / "packs" / f"{pid}-1.0.0"
    (folder / "fonts").mkdir(parents=True, exist_ok=True)
    data = font.read_bytes()
    (folder / "fonts" / font.name).write_bytes(data)
    fonts = (FontAsset("lat", font.name, ("Latin",), len(data), hashlib.sha256(data).hexdigest(),
                       "https://example.com/font", "OFL-1.1"),)
    reader = EngineUse("reader", "fakeocr", "reader", {"lang": lang}, ())
    checker = EngineUse("checker", "fakechk", "checker", {"lang": lang}, ())
    rs = ReaderSet((reader, checker), RuleUse("primary_with_floor", 1, {"primary": "reader", "checker": "checker"}),
                   "none")
    m = Manifest(schema=1, id=pid, version="1.0.0", requires_core=">=0.1.0", base=base, title=pid.upper(),
                 languages=({"tag": lang, "iso639_3": "xxx", "role": "primary"},), scripts=("Latin",),
                 detect=Detect(("Latin",), (lang,), (), frozenset(), frozenset(), 3, 1 / 3, None),
                 readers={KEY: rs, "default": rs}, normalize=("nfc",), repairs=(), fills=(), models=(),
                 fonts=fonts, canary=None, must_survive=(), bench={}, raw_sha256="0" * 64)
    return store.InstalledPack(m, folder)


@dataclass
class Env:
    root: Path
    cache: Path
    base: store.InstalledPack
    available: dict[str, store.InstalledPack]
    installed_ids: set[str]
    reader: FakeOcr
    checker: FakeOcr
    lines: dict[str, list[str]] = field(default_factory=lambda: {"*": list(EN_LINES)})
    fetches: list[str] = field(default_factory=list)
    broken: set[str] = field(default_factory=set)

    def text(self, lang: str, page: int) -> list[str]:
        return list(self.lines.get(lang, self.lines["*"]))

    def records(self) -> list[Path]:
        folder = self.cache / "records"
        return sorted(folder.glob("*.json")) if folder.is_dir() else []

    def work_left(self) -> list[Path]:
        """The run folders left in ``<cache>/work``: everything but the inputs' detection caches."""
        folder = self.cache / "work"
        return sorted(p for p in folder.iterdir() if not flow.CACHE_DIR_RE.match(p.name)) if folder.is_dir() else []

    def detection_caches(self) -> list[Path]:
        """The inputs' detection caches in ``<cache>/work`` (DESIGN §9.1)."""
        folder = self.cache / "work"
        return sorted(p for p in folder.iterdir() if flow.CACHE_DIR_RE.match(p.name)) if folder.is_dir() else []


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return make_env(tmp_path, monkeypatch)


def make_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    """The fakes a run test uses, set up through ``monkeypatch`` (also by the killed child run)."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("PDFINTAKE_CACHE", str(cache))
    font = build.write_box_font(tmp_path / "ExampleSans-Regular.ttf", latin_chars(), family="Example Sans")
    holder: dict[str, Env] = {}
    reader = FakeOcr("fakeocr", lambda lang, page: holder["env"].text(lang, page))
    checker = FakeOcr("fakechk", lambda lang, page: holder["env"].text(lang, page))
    monkeypatch.setitem(engines._REGISTRY, "fakeocr", reader)
    monkeypatch.setitem(engines._REGISTRY, "fakechk", checker)
    base = make_pack(tmp_path, "en", "en", base=True, font=font)
    hu = make_pack(tmp_path, "hu", "hu", base=False, font=font)
    e = Env(tmp_path, cache, base, {"hu": hu}, {"en"}, reader, checker)
    holder["env"] = e
    idx = I.load(FIXTURE_INDEX)

    def fake_ensure(entry: Any, *, allow_network: bool) -> store.InstalledPack:
        if entry.id == "en":
            return e.base
        if entry.id in e.broken or entry.id not in e.available:
            raise PackError("pack_broken", f"The {entry.title} language pack failed its self-check (made up).",
                            pack=entry.id)
        if entry.id not in e.installed_ids:
            if not allow_network:
                raise PackError("pack_not_installed", "not installed (made up)", pack=entry.id)
            e.fetches.append(entry.id)
            e.installed_ids.add(entry.id)
        return e.available[entry.id]

    monkeypatch.setattr(flow, "_platform_key", lambda: KEY)
    monkeypatch.setattr(flow, "_load_index", lambda: idx)
    monkeypatch.setattr(flow, "_ensure", fake_ensure)
    monkeypatch.setattr(flow, "_installed",
                        lambda: {pid: (e.base if pid == "en" else e.available[pid]) for pid in e.installed_ids})
    monkeypatch.setattr(flow, "_open_voter", lambda base: None)
    monkeypatch.setattr(flow, "_open_osd", lambda base: None)
    return e


# --------------------------------------------------------------------------- inputs and proofs

def page_image(width: int = 850, height: int = 1100) -> Any:
    return build.text_like(width, height, lines=8)


def image_pdf(path: Path, pages: int = 1, *, image: Any = None) -> Path:
    img = image if image is not None else page_image()
    return build.write_pdf(path, [build.PageSpec(images=(build.ImageSpec(img),)) for _ in range(pages)])


def text_pdf(path: Path) -> Path:
    return build.write_pdf(path, [build.PageSpec(texts=(build.TextSpec("Example text that is printed."),))])


def smooth_photo(width: int = 1300, height: int = 1000, *, seed: int = 0) -> Any:
    """Smooth random colour: lossless Flate can't shrink it, a JPEG at q95 or lower keeps it at 90+."""
    import random

    from PIL import Image

    rng = random.Random(seed)
    w, h = width // 32, height // 32
    return Image.frombytes("RGB", (w, h), rng.randbytes(w * h * 3)).resize((width, height), Image.Resampling.BICUBIC)


def tight_text_pdf(path: Path, tmp: Path) -> Path:
    """A born-digital PDF already as small as the lossless shrink makes it."""
    from pdfintake.pdf import shrink

    loose = text_pdf(tmp / "loose-input.pdf")
    shrink.lossless_shrink(loose, path, content_changed=True)
    loose.unlink()
    return path


def fingerprint(paths: Sequence[Path]) -> dict[str, tuple[str, int, int, int]]:
    out = {}
    for p in paths:
        st = os.stat(p)
        out[str(p)] = (hashlib.sha256(Path(p).read_bytes()).hexdigest(), st.st_size, st.st_mtime_ns, st.st_ino)
    return out


@pytest.fixture
def untouched():
    """Call with the inputs before a run; the check runs at teardown, whatever the test did."""
    watched: list[tuple[list[Path], dict[str, Any]]] = []

    def watch(*paths: Path) -> None:
        ps = [Path(p) for p in paths]
        watched.append((ps, fingerprint(ps)))

    yield watch
    for ps, before in watched:
        assert fingerprint(ps) == before, "an input was written, renamed or touched"


def pdf_text(path: Path) -> str:
    import pypdfium2

    from pdfintake.pdf.render import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pypdfium2.PdfDocument(str(path))
        try:
            out = []
            for i in range(len(doc)):
                page = doc[i]
                tp = page.get_textpage()
                out.append(tp.get_text_range())
                tp.close()
                page.close()
            return "\n".join(out)
        finally:
            doc.close()


def assert_record_ok(rec: record.Record, *, password: str | None = None) -> dict[str, Any]:
    data = record.to_json(rec)
    assert record.validate(data, forbid=[password] if password else ()) == []
    assert rec.exit in (0, 1, 2, 3, 4, 5) and rec.exit != 10
    loaded = record.load(rec.id)                       # it was saved
    assert record.to_json(loaded) == json.loads(json.dumps(data))
    return data


# --------------------------------------------------------------------------- a run, end to end (fakes)

def test_an_image_only_pdf_gets_a_searchable_output(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Scan.pdf", 2)
    untouched(src)
    recs = flow.run([src], flow.RunOptions(workers=2))
    assert len(recs) == 1
    rec = recs[0]
    data = assert_record_ok(rec)
    assert (rec.status, rec.exit) == ("done", 0)
    out = tmp_path / "Scan (searchable).pdf"
    assert out.is_file()
    assert rec.output == {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                          "bytes": out.stat().st_size, "kind": "searchable"}
    text = pdf_text(out)
    assert "4521" in text and "AB123456" in text
    assert [c["name"] for c in data["verify"]] and all(c["ok"] for c in data["verify"])
    assert {n.code for n in rec.notes} >= {"input_unchanged"}
    assert rec.provenance == "ordinary" and rec.input["kind"] == "pdf" and rec.input["pages"] == 2
    assert [p["page"] for p in data["pages"]] == [1, 2]
    assert data["language"]["plan"] in record.LANGUAGE_PLANS
    assert env.work_left() == []                       # the run's work folder is gone
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".pdf") == ["Scan (searchable).pdf", "Scan.pdf"]
    assert not list(tmp_path.glob("*.pdfintake-partial"))
    assert "invoice" not in json.dumps(data)           # no document text in a record


def test_run_never_asks_and_never_returns_10(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    def no_input(*a: Any, **k: Any) -> Any:
        raise AssertionError("a run read standard input")

    monkeypatch.setattr("builtins.input", no_input)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    a, b = image_pdf(tmp_path / "a.pdf"), text_pdf(tmp_path / "b.pdf")
    untouched(a, b)
    recs = flow.run([a, b])
    assert all(r.exit != 10 for r in recs)
    assert record.overall_exit(recs) == 0


def test_a_born_digital_pdf_writes_nothing(env: Env, tmp_path: Path, untouched):
    src = tight_text_pdf(tmp_path / "Letter.pdf", tmp_path)
    untouched(src)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("done", 0, None)
    assert {n.code for n in rec.notes} >= {"nothing_written", "input_unchanged"}
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Letter.pdf"]
    assert env.reader.reads == []                      # a page with printed text is never read


def test_a_scan_with_another_apps_clip_or_clipping_transparent_layer_is_kept_and_recorded(env: Env, tmp_path: Path,
                                                                                            untouched):
    # Transparent text in a clipping mode (hidden_text) and a mode-7 page (clip_text): neither is
    # read, and the record still saves. Their class names end in "_text", which the record's text
    # guard once refused, failing the whole file.
    photo = build.ImageSpec(smooth_photo(400, 300), x=40, y=40, w=500, h=700)
    src = build.write_pdf(tmp_path / "Scan.pdf", [
        build.PageSpec(images=(photo,), texts=(build.TextSpec("Sample hidden words", mode=4, fill_alpha=0.0),)),
        build.PageSpec(images=(photo,), texts=(build.TextSpec("Clipped sample", mode=7),)),
    ])
    untouched(src)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert (rec.status, rec.exit) == ("done", 0)
    classify_step = next(s for s in rec.steps if s["step"] == "classify")
    assert classify_step["detail"]["kinds"] == {"clip_text": 1, "hidden_text": 1}
    assert {w.code for w in rec.warnings} >= {"existing_layer_kept", "clip_text_kept"}
    assert env.reader.reads == []


def test_a_scanner_apps_transparent_layer_is_replaced_by_a_read(env: Env, tmp_path: Path, untouched):
    # Scanner Pro draws its OCR at fill alpha 0 over the scan (DESIGN Q8): an old layer, so the
    # page is read and the old text replaced; with --keep-existing-layer it stays and nothing is read.
    spec = build.PageSpec(images=(build.ImageSpec(page_image()),),
                          texts=(build.TextSpec("Zebra hidden words", fill_alpha=0.0),))
    src = build.write_pdf(tmp_path / "Scanner app.pdf", [spec])
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert (rec.status, rec.exit) == ("done", 0), rec.steps
    classify_step = next(s for s in rec.steps if s["step"] == "classify")
    assert classify_step["detail"]["kinds"] == {"invisible_layer": 1}
    assert "existing_layer_replaced" in {w.code for w in rec.warnings}
    assert {c["name"]: c["ok"] for c in data["verify"]}["renders_identical"] is True
    text = pdf_text(Path(rec.output["path"]))
    assert "Zebra" not in text and "4521" in text

    kept_src = build.write_pdf(tmp_path / "Kept.pdf", [spec])
    untouched(kept_src)
    reads = len(env.reader.reads)
    kept = flow.run([kept_src], flow.RunOptions(keep_existing_layer=True))[0]
    assert_record_ok(kept)
    assert kept.status == "done" and "existing_layer_kept" in {w.code for w in kept.warnings}
    assert len(env.reader.reads) == reads


def test_a_scan_whose_info_holds_null_values_is_written(env: Env, tmp_path: Path, untouched):
    # Info entries set to null (some scanner apps and PDF writers leave /Author null) aren't
    # carried by the save, and the metadata check once counted them as dropped and wrote nothing.
    src = build.with_null_info(image_pdf(tmp_path / "Scan.pdf"), "A scan", ["Author", "Subject", "Trapped"])
    untouched(src)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert (rec.status, rec.exit) == ("done", 0)
    assert [v.ok for v in rec.verify if v.name == "metadata_kept"] == [True]
    assert rec.output is not None and Path(rec.output["path"]).is_file()


def test_a_compressible_pdf_gets_a_compact_copy(env: Env, tmp_path: Path, untouched):
    import pikepdf

    import zlib

    # Printed text (so nothing is read) and a small picture stored with no compression at all.
    raw = build.pdf_bytes([build.PageSpec(texts=tuple(build.TextSpec(f"Example line {i}", y=700 - 12 * i)
                                                      for i in range(40)),
                                          images=(build.ImageSpec(smooth_photo(400, 300), x=300, y=40,
                                                                  w=200, h=150),))])
    src = tmp_path / "Loose.pdf"
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        for obj in pdf.objects:
            if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == pikepdf.Name.Image:
                obj.write(zlib.compress(obj.read_bytes(), 0), filter=pikepdf.Name.FlateDecode)
        pdf.save(src, compress_streams=False, object_stream_mode=pikepdf.ObjectStreamMode.disable)
    untouched(src)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert rec.exit == 0 and rec.output is not None
    out = Path(rec.output["path"])
    assert out.name == "Loose (compact).pdf" and out.stat().st_size < src.stat().st_size * 0.99


def test_images_become_one_pdf_named_after_the_first(env: Env, tmp_path: Path, untouched):
    a = build.write_png(page_image(), tmp_path / "Receipt.png", dpi=100)
    b = build.write_png(page_image(), tmp_path / "Back.png", dpi=100)
    untouched(a, b)
    recs = flow.run([a, b])
    assert len(recs) == 1
    rec = recs[0]
    data = assert_record_ok(rec)
    out = tmp_path / "Receipt (searchable).pdf"
    assert rec.exit == 0 and rec.output["path"] == str(out)
    assert rec.input["pages"] == 2 and rec.input["kind"] == "images"
    assert "passthrough_identical" in {c["name"] for c in data["verify"]}
    import pikepdf

    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 2 and str(pdf.docinfo.get("/Producer")) == flow.PRODUCER


def test_separate_images_get_distinct_names(env: Env, tmp_path: Path, untouched):
    a = build.write_png(page_image(), tmp_path / "Card.png", dpi=100)
    b = tmp_path / "Card.jpg"
    b.write_bytes(build.jpeg_bytes(page_image().convert("RGB")))
    untouched(a, b)
    recs = flow.run([a, b], flow.RunOptions(separate=True))
    assert [r.exit for r in recs] == [0, 0]
    names = [Path(r.output["path"]).name for r in recs]
    assert names == ["Card (searchable).pdf", "Card (searchable 2).pdf"]


def test_plan_writes_nothing(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(plan_only=True))[0]
    data = assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("plan", 0, None)
    assert [p["id"] for p in data["language"]["packs"]] == ["en"]
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Scan.pdf"]
    assert env.fetches == []


def test_an_old_invisible_layer_is_replaced(env: Env, tmp_path: Path, untouched):
    src = build.write_pdf(tmp_path / "Old layer.pdf", [build.PageSpec(images=(build.ImageSpec(page_image()),),
                                                                    texts=(build.TextSpec("Zebra hidden words", mode=3),))])
    untouched(src)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert rec.exit == 0
    assert "existing_layer_replaced" in {w.code for w in rec.warnings}
    text = pdf_text(Path(rec.output["path"]))
    assert "Zebra" not in text and "4521" in text


def test_a_sideways_flag_is_baked_and_the_xmp_kept(env: Env, tmp_path: Path, untouched):
    import pikepdf

    from fixtures import refusal_shells as shells

    raw = build.pdf_bytes([build.PageSpec(rotate=90, images=(build.ImageSpec(page_image()),))])
    src = tmp_path / "Turned.pdf"
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.Root.Metadata = shells._metadata(pdf, shells.XMP_PLAIN)
        pdf.save(src)
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    checks = {c["name"]: c for c in data["verify"]}
    assert checks["upright"]["ok"] and checks["metadata_kept"]["ok"] and checks["renders_identical"]["ok"]
    assert data["pages"][0]["orientation"]["action"] == "baked"
    with pikepdf.open(rec.output["path"]) as pdf:
        assert int(pdf.pages[0].obj.get("/Rotate", 0)) == 0


# --------------------------------------------------------------------------- copy mode (§10, §11)

def test_a_counterparty_file_gets_a_searchable_copy(env: Env, tmp_path: Path, untouched):
    import pikepdf

    from pdfintake.pdf import meta

    src = image_pdf(tmp_path / "Lease.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src], flow.RunOptions(provenance="counterparty"))[0]
    data = assert_record_ok(rec)
    out = tmp_path / "Lease (searchable copy).pdf"
    assert rec.exit == 0 and rec.provenance == "counterparty"
    assert data["output"]["kind"] == "searchable_copy"
    assert data["output"]["original"] == {"name": "Lease.pdf", "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}
    assert data["output"]["copy_reasons"] == ["provenance_counterparty"]
    assert rec.offers == [] and rec.smaller is None    # no smaller copy for a copy
    assert {"not_encrypted", "derived_from_present"} <= {c["name"] for c in data["verify"]}
    with pikepdf.open(out) as pdf:
        found = meta.derived_from(pdf)
    assert found is not None and found["name"] == "Lease.pdf" and found["note"] == meta.DERIVED_NOTE
    assert "4521" in pdf_text(out)


def test_a_pdfa_scan_gets_a_copy_without_the_claim(env: Env, tmp_path: Path, untouched):
    import pikepdf

    from fixtures import refusal_shells as shells

    raw = build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))])
    src = tmp_path / "Archive.pdf"
    with pikepdf.open(io.BytesIO(raw)) as pdf:
        pdf.Root.Metadata = shells._metadata(pdf, shells.XMP_PDFA_2B)
        pdf.save(src)
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    assert data["output"]["copy_reasons"] == ["pdfa"]
    assert "conformance_claim_dropped" in {n.code for n in rec.notes}
    with pikepdf.open(rec.output["path"]) as pdf:
        xmp = pdf.Root.Metadata.read_bytes()
    assert b"pdfaid" not in xmp and b"DerivedFrom" in xmp


def test_a_copy_never_overwrites(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Exhibit.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(provenance="evidence", overwrite=True))[0]
    assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("failed", 2, None)
    assert rec.steps[-1]["detail"]["code"] == "copy_never_overwrites"
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Exhibit.pdf"]


def test_a_signed_born_digital_pdf_writes_nothing(env: Env, tmp_path: Path, untouched):
    pytest.importorskip("pyhanko")
    from fixtures import refusal_shells as shells

    src = tmp_path / "Signed.pdf"
    src.write_bytes(shells.signed_visible())
    untouched(src)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert (rec.exit, rec.output) == (0, None)
    assert {n.code for n in rec.notes} >= {"nothing_written"}


def test_a_signed_scan_gets_a_copy_without_the_signature(env: Env, tmp_path: Path, untouched):
    pytest.importorskip("pyhanko")
    from fixtures import refusal_shells as shells

    src = tmp_path / "Signed scan.pdf"
    src.write_bytes(shells._sign(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                 box=(72, 72, 272, 132)))
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    assert data["output"]["kind"] == "searchable_copy" and "signed" in data["output"]["copy_reasons"]
    assert "signature_not_carried" in {n.code for n in rec.notes}
    out = Path(rec.output["path"])
    assert out.name == "Signed scan (searchable copy).pdf"
    assert refusals.output_signature_material(out) == []


def test_an_edit_locked_pdf_gets_an_unlocked_copy(env: Env, tmp_path: Path, untouched):
    import pikepdf

    from fixtures import refusal_shells as shells

    src = tmp_path / "Locked.pdf"
    src.write_bytes(shells.encrypt(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                   user=""))
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    assert data["output"]["copy_reasons"] == ["encrypted"]
    assert "copy_unlocked" in {n.code for n in rec.notes}
    with pikepdf.open(rec.output["path"]) as pdf:
        assert not pdf.is_encrypted
    assert env.work_left() == []                       # the decrypted work copy is gone


def private_file(path: Path, lines: Sequence[str]) -> Path:
    path.write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def test_a_saved_password_opens_the_file_and_never_reaches_a_record(env: Env, tmp_path: Path, untouched):
    from fixtures import refusal_shells as shells

    src = tmp_path / "Statement.pdf"
    src.write_bytes(shells.encrypt(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                   user=PASSWORD))
    pw = private_file(tmp_path / "saved-passwords", [WRONG, PASSWORD])
    untouched(src, pw)
    rec = flow.run([src], flow.RunOptions(passwords_file=pw))[0]
    data = assert_record_ok(rec, password=PASSWORD)
    assert rec.exit == 0, rec.steps
    assert data["output"]["kind"] == "searchable_copy" and data["output"]["copy_reasons"] == ["encrypted"]
    assert {"copy_unlocked", "input_unchanged"} <= {n.code for n in rec.notes}
    for path in [*env.records(), *env.cache.rglob("*")]:
        if path.is_file():
            assert PASSWORD.encode() not in path.read_bytes() and WRONG.encode() not in path.read_bytes()
    assert env.work_left() == []
    assert "password" not in json.dumps(data["options"])


def test_no_saved_password_skips_the_file_and_the_batch_carries_on(env: Env, tmp_path: Path, untouched):
    from fixtures import refusal_shells as shells

    locked = tmp_path / "Locked.pdf"
    locked.write_bytes(shells.encrypt(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                      user=PASSWORD))
    plain = image_pdf(tmp_path / "Plain.pdf")
    pw = private_file(tmp_path / "saved-passwords", [WRONG])
    untouched(locked, plain, pw)
    recs = flow.run([locked, plain], flow.RunOptions(passwords_file=pw))
    assert [r.exit for r in recs] == [0, 0]
    first, second = recs
    assert_record_ok(first, password=WRONG)
    assert first.output is None and first.status == "done"
    assert {n.code for n in first.notes} >= {"skipped_no_password", "input_unchanged"}
    assert second.output is not None and Path(second.output["path"]).name == "Plain (searchable).pdf"
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Locked.pdf", "Plain (searchable).pdf", "Plain.pdf"]


def test_no_passwords_file_at_all_skips_too(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    from fixtures import refusal_shells as shells
    from pdfintake import passwords

    monkeypatch.setattr(passwords, "_config_root", lambda: tmp_path / "config")
    locked = tmp_path / "Locked.pdf"
    locked.write_bytes(shells.encrypt(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                      user=PASSWORD))
    untouched(locked)
    rec = flow.run([locked])[0]
    assert (rec.exit, rec.output) == (0, None)
    assert "skipped_no_password" in {n.code for n in rec.notes}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_an_unsafe_passwords_file_fails_only_the_files_that_need_it(env: Env, tmp_path: Path, untouched):
    from fixtures import refusal_shells as shells

    locked = tmp_path / "Locked.pdf"
    locked.write_bytes(shells.encrypt(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                      user=PASSWORD))
    plain = image_pdf(tmp_path / "Plain.pdf")
    pw = tmp_path / "saved-passwords"
    pw.write_text(PASSWORD + "\n", encoding="utf-8")
    os.chmod(pw, 0o644)
    untouched(locked, plain, pw)
    recs = flow.run([locked, plain], flow.RunOptions(passwords_file=pw))
    assert [r.exit for r in recs] == [2, 0]
    assert recs[0].steps[-1]["detail"]["code"] == "passwords_file_unsafe"
    assert record.overall_exit(recs) == 2
    assert PASSWORD not in json.dumps(record.to_json(recs[0]))


def test_an_xfa_form_is_refused(env: Env, tmp_path: Path, untouched):
    from fixtures import refusal_shells as shells

    src = tmp_path / "Form.pdf"
    src.write_bytes(shells.xfa())
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("refused", 4, None)
    assert data["refusal"]["code"] == "xfa" and data["refusal"]["message"]
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Form.pdf"]


# --------------------------------------------------------------------------- usage, outputs, failures

def test_invocations_that_cant_run(env: Env, tmp_path: Path):
    pdf, png = image_pdf(tmp_path / "a.pdf"), build.write_png(page_image(), tmp_path / "b.png")
    with pytest.raises(UsageError) as info:
        flow.run([pdf, png])
    assert info.value.code == "mixed_inputs"
    with pytest.raises(UsageError):
        flow.run([])
    with pytest.raises(UsageError) as info:
        flow.run([pdf, image_pdf(tmp_path / "c.pdf")], flow.RunOptions(out=tmp_path / "x.pdf"))
    assert info.value.code == "out_needs_one_output"
    for bad in (flow.RunOptions(provenance="mine"), flow.RunOptions(fetch="ask"),  # type: ignore[arg-type]
                flow.RunOptions(workers=0), flow.RunOptions(lang=("../x",)),
                flow.RunOptions(out=tmp_path / "x.pdf", outdir=tmp_path)):
        with pytest.raises(UsageError):
            flow.run([pdf], bad)
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["a.pdf", "c.pdf"]


def test_a_missing_or_unknown_input_fails_on_its_own(env: Env, tmp_path: Path, untouched):
    junk = tmp_path / "notes.txt"
    junk.write_text("not a document", encoding="utf-8")
    good = image_pdf(tmp_path / "Good.pdf")
    untouched(junk, good)
    recs = flow.run([tmp_path / "missing.pdf", junk, good])
    assert [r.exit for r in recs] == [2, 2, 0]
    assert [r.steps[-1]["detail"]["code"] for r in recs[:2]] == ["input_missing", "unsupported_input"]


def test_an_existing_output_is_exit_2_before_any_ocr_and_overwrite_replaces_it(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    first = flow.run([src])[0]
    assert first.exit == 0
    reads = len(env.reader.reads)
    again = flow.run([src])[0]
    assert (again.exit, again.steps[-1]["detail"]["code"]) == (2, "output_exists")
    assert len(env.reader.reads) == reads              # refused before any page was read
    third = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert third.exit == 0 and third.output["path"] == first.output["path"]


def test_an_output_that_would_be_the_input_is_exit_2(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "scan.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(out=tmp_path / "Scan.pdf", overwrite=True))[0]
    assert rec.exit == 2 and rec.steps[-1]["detail"]["code"] == "output_is_input"
    rec = flow.run([src], flow.RunOptions(out=src, overwrite=True))[0]
    assert rec.exit == 2 and rec.steps[-1]["detail"]["code"] == "output_is_input"


def test_the_input_changing_mid_run_is_exit_5(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pdfintake.ocr import pipeline

    src = image_pdf(tmp_path / "Scan.pdf")
    real = pipeline.read_pages

    def touch_input(*a: Any, **k: Any) -> Any:
        with open(src, "ab") as fh:                    # someone else edits the file during the run
            fh.write(b"\n% edited\n")
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "read_pages", touch_input)
    rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("failed", 5, None)
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Scan.pdf"]
    assert env.work_left() == []


def test_a_failed_check_writes_nothing_and_its_check_is_recorded_unchanged(env: Env, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake.pdf import verify

    real = verify.verify_build
    planted = verify.Check("renders_identical", False, "page 1: 3 pixel(s) differ, by up to 9", 1)
    monkeypatch.setattr(verify, "verify_build", lambda ctx: [*real(ctx), planted])
    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("failed", 1, None)
    assert {"name": "renders_identical", "ok": False, "detail": "page 1: 3 pixel(s) differ, by up to 9",
            "page": 1} in data["verify"]
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Scan.pdf"]


def test_one_file_failing_never_stops_the_batch(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake.pdf import classify

    first, second = image_pdf(tmp_path / "One.pdf"), image_pdf(tmp_path / "Two.pdf")
    untouched(first, second)
    real = classify.classify
    calls: list[Path] = []

    def boom(pdf: Path, **k: Any) -> Any:
        calls.append(pdf)
        if len(calls) == 1:
            raise RuntimeError("made-up crash with a made-up message")
        return real(pdf, **k)

    monkeypatch.setattr(classify, "classify", boom)
    recs = flow.run([first, second])
    assert [r.exit for r in recs] == [1, 0]
    assert recs[0].steps[-1]["detail"]["code"] == "internal_error"
    assert "made-up message" not in json.dumps(record.to_json(recs[0]))
    assert env.work_left() == []


def test_no_process_can_start_during_a_run(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake.pdf import classify

    marker = tmp_path / "started"
    depth_at_warm_up: list[int] = []
    real_warm = engines.warm_up
    monkeypatch.setattr(engines, "warm_up", lambda: (depth_at_warm_up.append(integrity._GUARD["depth"]), real_warm())[1])

    def start(pdf: Path, **k: Any) -> Any:
        subprocess.run([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"], check=False)
        raise AssertionError("unreachable")

    monkeypatch.setattr(classify, "classify", start)
    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    rec = flow.run([src])[0]
    assert (rec.exit, rec.steps[-1]["detail"]["code"]) == (1, "process_refused")
    assert not marker.exists()
    assert depth_at_warm_up == [0]                     # engines import before the guard goes up
    assert integrity._GUARD["depth"] == 0              # and it comes down after the run


def test_atomic_write_is_reached_only_through_commit_and_apply(env: Env, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch, untouched):
    callers: list[str] = []
    real = fsutil.atomic_write

    def spy(built: Path, dest: Path, **kw: Any) -> None:
        callers.append(sys._getframe(1).f_code.co_name)
        assert integrity._GUARD["depth"] >= 1           # no process can start while it writes
        assert kw == {"overwrite": False}               # neither may replace a file without --overwrite
        real(built, dest, **kw)

    monkeypatch.setattr(fsutil, "atomic_write", spy)
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src])[0]
    assert rec.exit == 0
    assert any(o.kind == "smaller" for o in rec.offers)
    flow.apply(rec.id, choice="smaller")
    assert callers == ["commit", "apply"]


def test_work_files_stay_inside_the_cache(env: Env, tmp_path: Path):
    import pikepdf

    with pytest.raises(Exception) as info:
        with pikepdf.new() as pdf:
            flow._save_work(pdf, tmp_path / "outside.pdf")
    assert getattr(info.value, "code", None) == "not_a_work_path"
    with pytest.raises(Exception):
        flow._copy_into_work(image_pdf(tmp_path / "a.pdf"), tmp_path / "b.pdf")
    assert not (tmp_path / "outside.pdf").exists() and not (tmp_path / "b.pdf").exists()
    keep = tmp_path / "keep.txt"
    keep.write_text("kept", encoding="utf-8")
    flow._remove_work(keep)                            # outside the cache: left alone
    assert keep.exists()


# --------------------------------------------------------------------------- the smaller copy (§12.4)

def test_a_smaller_copy_is_offered_then_saved_by_apply(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0
    offers = [o for o in rec.offers if o.kind == "smaller"]
    assert len(offers) == 1 and offers[0].apply_flag == "--choice" and [o.id for o in offers[0].options] == ["smaller"]
    assert data["smaller"]["offered"] is True and data["smaller"]["min_score"] >= flow.SMALLER_BAR
    out = Path(rec.output["path"])
    beside = out.with_name(f"{out.stem} (smaller).pdf")
    assert not beside.exists()                         # nothing lossy is written during a run
    done = flow.apply(rec.id, choice="smaller")
    assert len(done) == 1 and done[0].id == rec.id
    assert (done[0].status, done[0].exit) == ("resolved", 0)
    assert beside.is_file() and beside.stat().st_size < out.stat().st_size
    assert hashlib.sha256(beside.read_bytes()).hexdigest() == data["smaller"]["candidate"]["sha256"]
    saved = record.load(rec.id)
    assert saved.history[-1]["flag"] == "--choice" and saved.history[-1]["answer"] == "smaller"
    assert saved.smaller["applied"] is True and not saved.offers
    with pytest.raises(UsageError) as info:
        flow.apply(rec.id, choice="smaller")
    assert info.value.code == "already_applied"


def test_apply_rebuilds_a_pruned_candidate_with_the_same_hash(env: Env, tmp_path: Path, untouched):
    import shutil

    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src])[0]
    assert any(o.kind == "smaller" for o in rec.offers)
    shutil.rmtree(env.cache / "staging" / rec.id)
    done = flow.apply(rec.id, choice="smaller")[0]
    assert done.status == "resolved"
    assert Path(done.smaller["written"]).is_file()


def test_apply_refuses_a_changed_output_or_input(env: Env, tmp_path: Path):
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    rec = flow.run([src])[0]
    assert any(o.kind == "smaller" for o in rec.offers)
    out = Path(rec.output["path"])
    before = out.read_bytes()
    out.write_bytes(before + b"\n% edited by someone\n")
    with pytest.raises(StaleRecord):
        flow.apply(rec.id, choice="smaller")
    out.write_bytes(before)
    with open(src, "ab") as fh:
        fh.write(b"\n% edited\n")
    with pytest.raises(InputChanged):
        flow.apply(rec.id, choice="smaller")
    assert not list(tmp_path.glob("*(smaller).pdf"))


def test_run_with_smaller_writes_it_beside_without_an_offer(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src], flow.RunOptions(smaller=True))[0]
    assert_record_ok(rec)
    assert rec.exit == 0 and not any(o.kind == "smaller" for o in rec.offers)
    assert rec.smaller["applied"] is True
    written = Path(rec.smaller["written"])
    assert written.name == "Photo (searchable) (smaller).pdf" and written.is_file()
    assert not (env.cache / "staging" / rec.id).exists()


def test_apply_takes_exactly_one_known_answer(env: Env, tmp_path: Path):
    src = image_pdf(tmp_path / "Scan.pdf")
    rec = flow.run([src])[0]
    with pytest.raises(UsageError):
        flow.apply(rec.id)
    with pytest.raises(UsageError):
        flow.apply(rec.id, choice="smaller", lang="en")
    with pytest.raises(UsageError):
        flow.apply("../" + rec.id, choice="smaller")
    with pytest.raises(UsageError):
        flow.apply("r-20990101-0000beef", choice="smaller")
    if not any(o.kind == "smaller" for o in rec.offers):
        with pytest.raises(UsageError) as info:
            flow.apply(rec.id, choice="smaller")
        assert info.value.code == "no_offer"
    with pytest.raises(UsageError) as info:
        flow.apply(rec.id, lang="th")
    assert info.value.code == "no_offer"


# --------------------------------------------------------------------------- language (§12.3)

def test_an_offline_run_reads_with_english_and_apply_lang_reads_again(env: Env, tmp_path: Path, untouched):
    env.lines = {"*": list(HU_LINES)}
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(fetch="never"))[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    assert env.fetches == []
    assert data["pages"][0]["pack"] == "en"
    warn = [w for w in rec.warnings if w.code == "pack_not_installed"]
    assert warn and "Hungarian" in warn[0].message
    offers = [o for o in rec.offers if o.kind == "language"]
    assert offers and offers[0].apply_flag == "--lang" and "hu" in {o.id for o in offers[0].options}
    assert data["language"]["plan"] == "uncertain"
    out = Path(rec.output["path"])
    before = out.read_bytes()
    for not_offered, code in (("none", "bad_language_code"), ("th", "not_an_option")):
        with pytest.raises(UsageError) as info:       # only a language the offer lists (§12.5)
            flow.apply(rec.id, lang=not_offered)
        assert info.value.code == code
    assert out.read_bytes() == before and record.load(rec.id).status == "done"

    children = flow.apply(rec.id, lang="hu")
    assert len(children) == 1
    child = children[0]
    child_data = assert_record_ok(child)
    assert child.parent == rec.id and child.exit == 0, child.steps
    assert env.fetches == ["hu"]                       # the request is the person's yes to that download
    assert child_data["pages"][0]["pack"] == "hu"
    assert child.output["path"] == str(out)            # the earlier output is replaced
    assert child.output["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert env.work_left() == []
    parent = record.load(rec.id)
    assert parent.status == "resolved" and parent.offers == []
    assert parent.history[-1] == {**parent.history[-1], "flag": "--lang", "answer": "hu", "child": child.id}
    with pytest.raises(UsageError):
        flow.apply(rec.id, lang="hu")                  # nothing left to act on


def test_a_pack_that_fails_its_check_is_replaced_by_english(env: Env, tmp_path: Path, untouched):
    env.lines = {"*": list(HU_LINES)}
    env.broken.add("hu")
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    assert data["pages"][0]["pack"] == "en"
    assert "pack_broken" in {w.code for w in rec.warnings}
    assert data["language"]["detection"]["broken"] == ["hu"]
    assert any(o.kind == "language" for o in rec.offers)


def test_a_published_pack_is_fetched_without_asking(env: Env, tmp_path: Path, untouched):
    env.lines = {"*": list(HU_LINES)}
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src])[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0
    assert env.fetches == ["hu"] and data["pages"][0]["pack"] == "hu"
    assert "pack_fetched" in {n.code for n in rec.notes}
    assert [p["fetched"] for p in data["language"]["packs"] if p["id"] == "hu"] == [True]


def test_lang_none_gives_no_layer(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(force_lang=("none",)))[0]
    data = assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("done", 0, None)   # an image-only PDF gains nothing
    assert {"nothing_written", "input_unchanged"} <= {n.code for n in rec.notes}
    assert data["pages"] and all(p["pack"] is None and p["composition"] == "none" and p["readers"] == []
                                 for p in data["pages"])
    assert data["language"]["plan"] == "forced" and data["language"]["packs"] == []
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["Scan.pdf"]


def test_apply_lang_refuses_a_changed_output(env: Env, tmp_path: Path, untouched):
    env.lines = {"*": list(HU_LINES)}
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(fetch="never"))[0]
    out = Path(rec.output["path"])
    out.write_bytes(out.read_bytes() + b"\n% edited by someone\n")
    edited = out.read_bytes()
    with pytest.raises(StaleRecord):
        flow.apply(rec.id, lang="hu")
    assert out.read_bytes() == edited and env.fetches == []
    assert record.load(rec.id).status == "done"


def test_apply_lang_replaces_a_searchable_copy(env: Env, tmp_path: Path, untouched):
    env.lines = {"*": list(HU_LINES)}
    src = image_pdf(tmp_path / "Lease.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(provenance="counterparty", fetch="never"))[0]
    assert rec.exit == 0 and rec.output["kind"] == "searchable_copy"
    assert any(o.kind == "language" for o in rec.offers)
    child = flow.apply(rec.id, lang="hu")[0]
    data = assert_record_ok(child)
    assert (child.exit, child.status) == (0, "done"), child.steps
    assert data["output"]["kind"] == "searchable_copy" and child.output["path"] == rec.output["path"]
    assert Path(child.output["path"]).name == "Lease (searchable copy).pdf"
    assert data["pages"][0]["pack"] == "hu"
    assert record.load(rec.id).status == "resolved"


def _locked_pdf(path: Path, password: str = PASSWORD) -> Path:
    from fixtures import refusal_shells as shells

    path.write_bytes(shells.encrypt(build.pdf_bytes([build.PageSpec(images=(build.ImageSpec(page_image()),))]),
                                    user=password))
    return path


def test_apply_lang_on_an_encrypted_input_uses_the_saved_passwords(env: Env, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake import passwords

    monkeypatch.setattr(passwords, "_config_root", lambda: tmp_path / "config")   # no default list here
    env.lines = {"*": list(HU_LINES)}
    src = _locked_pdf(tmp_path / "Statement.pdf")
    pw = private_file(tmp_path / "saved-passwords", [PASSWORD])
    untouched(src, pw)
    rec = flow.run([src], flow.RunOptions(fetch="never", passwords_file=pw))[0]
    assert rec.exit == 0 and rec.output["copy_reasons"] == ["encrypted"]
    assert any(o.kind == "language" for o in rec.offers)
    out = Path(rec.output["path"])
    before = out.read_bytes()

    skipped = flow.apply(rec.id, lang="hu")[0]          # no passwords file given to apply this time
    assert_record_ok(skipped, password=PASSWORD)
    assert (skipped.exit, skipped.output) == (0, None)
    assert "skipped_no_password" in {n.code for n in skipped.notes}
    assert out.read_bytes() == before and record.load(rec.id).status == "done"

    child = flow.apply(rec.id, lang="hu", passwords_file=pw)[0]
    assert_record_ok(child, password=PASSWORD)
    assert child.exit == 0 and child.output["path"] == str(out), child.steps
    assert record.load(rec.id).status == "resolved"
    assert env.work_left() == [] and env.detection_caches() == []


def test_a_decrypted_work_copy_is_private_and_gone_afterwards(env: Env, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch, untouched):
    src = _locked_pdf(tmp_path / "Statement.pdf")
    pw = private_file(tmp_path / "saved-passwords", [PASSWORD])
    untouched(src, pw)
    seen: list[tuple[str, int, int]] = []
    real = flow._save_work

    def spy(pdf: Any, dest: Path) -> None:
        real(pdf, dest)
        seen.append((Path(dest).name, stat.S_IMODE(os.stat(Path(dest).parent).st_mode),
                     stat.S_IMODE(os.stat(dest).st_mode)))

    monkeypatch.setattr(flow, "_save_work", spy)
    rec = flow.run([src], flow.RunOptions(passwords_file=pw))[0]
    assert rec.exit == 0, rec.steps
    assert ("decrypted.pdf", 0o700, 0o600) in seen if sys.platform != "win32" else seen
    assert env.work_left() == [] and env.detection_caches() == []       # no decrypted page outlives the run


def test_drop_unused_gets_only_the_stripped_pages(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 untouched):
    from pdfintake.pdf import strip

    src = build.write_pdf(tmp_path / "Mixed.pdf", [
        build.PageSpec(images=(build.ImageSpec(page_image()),)),
        build.PageSpec(images=(build.ImageSpec(page_image()),), texts=(build.TextSpec("Zebra hidden words", mode=3),)),
    ])
    untouched(src)
    calls: list[list[int]] = []
    real = strip.drop_unused
    monkeypatch.setattr(strip, "drop_unused", lambda pdf, pages: (calls.append(list(pages)), real(pdf, pages))[1])
    rec = flow.run([src])[0]
    assert rec.exit == 0, rec.steps
    assert calls == [[1]]


def test_same_stem_documents_keep_their_names_when_one_fails(env: Env, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake.pdf import classify

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "out").mkdir()
    a = image_pdf(tmp_path / "a" / "Scan.pdf")
    b = image_pdf(tmp_path / "b" / "scan.pdf", image=page_image(900, 1100))
    untouched(a, b)
    opts = flow.RunOptions(outdir=tmp_path / "out")
    first = flow.run([a, b], opts)
    assert [Path(r.output["path"]).name for r in first] == ["Scan (searchable).pdf", "scan (searchable 2).pdf"]
    kept = {p.name: p.read_bytes() for p in (tmp_path / "out").iterdir()}

    real = classify.classify
    calls: list[Path] = []

    def first_fails(pdf: Path, **k: Any) -> Any:
        calls.append(pdf)
        if len(calls) == 1:
            raise RuntimeError("made-up failure")
        return real(pdf, **k)

    monkeypatch.setattr(classify, "classify", first_fails)
    again = flow.run([a, b], dataclasses.replace(opts, overwrite=True))
    assert [(r.exit, r.output and Path(r.output["path"]).name) for r in again] == [(1, None),
                                                                                 (0, "scan (searchable 2).pdf")]
    assert (tmp_path / "out" / "Scan (searchable).pdf").read_bytes() == kept["Scan (searchable).pdf"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(kept)


@pytest.mark.parametrize("order", ["output_name_second", "output_name_first", "outdir"])
def test_no_output_ever_lands_on_another_input(env: Env, tmp_path: Path, order: str, untouched):
    if order == "outdir":
        (tmp_path / "x").mkdir()
        (tmp_path / "y").mkdir()
        scan = image_pdf(tmp_path / "x" / "Scan.pdf")
        other = image_pdf(tmp_path / "y" / "Scan (searchable).pdf", image=page_image(900, 1100))
        inputs, opts = [scan, other], flow.RunOptions(overwrite=True, outdir=tmp_path / "y")
    else:
        scan = image_pdf(tmp_path / "Scan.pdf")
        other = image_pdf(tmp_path / "Scan (searchable).pdf", image=page_image(900, 1100))
        inputs = [scan, other] if order == "output_name_second" else [other, scan]
        opts = flow.RunOptions(overwrite=True)
    untouched(scan, other)                             # the teardown check is the point of this test
    recs = flow.run(inputs, opts)
    by_input = {r.input["paths"][0]: r for r in recs}
    hit = by_input[str(scan)]
    assert (hit.exit, hit.output, hit.steps[-1]["detail"]["code"]) == (2, None, "output_is_input")
    assert by_input[str(other)].exit == 0


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs")
def test_a_fifo_or_folder_input_is_never_opened(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    import threading

    from pdfintake.intake import sniff

    fifo = tmp_path / "Pipe.pdf"
    os.mkfifo(fifo)
    folder = tmp_path / "Folder.pdf"
    folder.mkdir()
    plain = image_pdf(tmp_path / "Plain.pdf")
    untouched(plain)
    sniffed: list[Path] = []
    real = sniff.sniff
    monkeypatch.setattr(sniff, "sniff", lambda p: (sniffed.append(Path(p)), real(p))[1])
    out: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: out.setdefault("recs", flow.run([fifo, folder, plain])), daemon=True)
    worker.start()
    worker.join(60)
    if worker.is_alive():                              # unstick it before failing
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(30)
        pytest.fail("the run waited on a FIFO input")
    recs = out["recs"]
    assert [(r.exit, r.steps[-1]["detail"]["code"] if r.exit else None) for r in recs] == [
        (2, "input_not_file"), (2, "input_not_file"), (0, None)]
    assert sniffed == [plain]


@pytest.mark.skipif(sys.platform == "win32", reason="FIFOs")
def test_an_input_swapped_for_a_fifo_after_the_stat_fails_only_itself(env: Env, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch, untouched):
    """sniff's own refusal (input_not_file) becomes that input's record; the batch carries on."""
    fifo = tmp_path / "Pipe.pdf"
    os.mkfifo(fifo)
    plain = image_pdf(tmp_path / "Plain.pdf")
    untouched(plain)
    monkeypatch.setattr(flow, "_input_problem", lambda p: None)   # the stat saw a regular file
    recs = flow.run([fifo, plain])
    assert [(r.exit, r.steps[-1]["detail"]["code"] if r.exit else None) for r in recs] == [
        (2, "input_not_file"), (0, None)]


def test_a_path_no_file_can_have_fails_only_itself(env: Env, tmp_path: Path, untouched):
    first = image_pdf(tmp_path / "First.pdf")
    last = image_pdf(tmp_path / "Last.pdf")
    untouched(first, last)
    nul = Path(str(tmp_path / "Bad") + "\x00.pdf")
    odd = Path(str(tmp_path / "Odd") + "\udcff.pdf")    # how Python hands over a name that isn't UTF-8
    recs = flow.run([first, nul, odd, last])
    assert [r.exit for r in recs] == [0, 2, 2, 0]
    for r in recs:
        assert record.load(r.id).id == r.id            # every record was saved
    assert "\\udcff" in recs[2].input["paths"][0] and "\\x00" in json.dumps(record.to_json(recs[1]))
    assert sorted(p.name for p in tmp_path.glob("*(searchable).pdf")) == ["First (searchable).pdf",
                                                                          "Last (searchable).pdf"]


def test_a_long_input_name_gets_a_shortened_output_name(env: Env, tmp_path: Path, untouched):
    # The longest stem this filesystem takes with ".pdf", which " (searchable).pdf" makes too long.
    # Ä is one unit on APFS and NTFS (UTF-16) and two on Linux (bytes), so the count is platform's.
    count = (fsutil.NAME_MAX - fsutil._name_length(".pdf")) // fsutil._name_length("\u00c4")
    stem = "\u00c4" * count
    src = image_pdf(tmp_path / f"{stem}.pdf")
    other = image_pdf(tmp_path / "Other.pdf")
    untouched(src, other)
    recs = flow.run([src, other])
    assert [r.exit for r in recs] == [0, 0], recs[0].steps
    name = Path(recs[0].output["path"]).name
    assert name.endswith(flow.ELLIPSIS + " (searchable).pdf") and name.startswith("\u00c4" * (count * 4 // 5))
    assert fsutil._name_length(name) <= fsutil.NAME_MAX


@pytest.mark.parametrize("unit", ["e\u0301", "\U0001F1ED\U0001F1FA", "\U0001F469\u200d\U0001F4BB",
                                  "\U0001F44D\U0001F3FD", "a"])
def test_a_shortened_stem_never_splits_a_character(unit: str):
    import unicodedata

    stem = unit * 300
    cut = flow._fit_stem(stem)
    assert cut.endswith(flow.ELLIPSIS)
    kept = cut[:-1]
    assert stem.startswith(kept) and len(kept) % len(unit) == 0, (len(kept), len(unit))
    assert fsutil._name_length(cut + flow.SLOT_LABEL_ROOM) <= fsutil.NAME_MAX
    assert flow._fit_stem("Short name") == "Short name"
    assert not flow._extends(stem[len(kept)]) and not unicodedata.combining(stem[len(kept)])


def test_an_image_changed_and_put_back_during_the_run_cant_reach_the_output(env: Env, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch):
    import pikepdf

    from pdfintake.intake import assemble

    img = build.write_png(page_image(850, 1100), tmp_path / "Photo.png", dpi=100)
    original = img.read_bytes()
    st = os.stat(img)
    other = build.write_png(page_image(400, 300), tmp_path / "other.png", dpi=100).read_bytes()
    real = assemble.assemble

    def swap(paths: Any, dest: Any, *a: Any, **k: Any) -> Any:
        img.write_bytes(other)                         # someone saves another picture under the name
        try:
            return real(paths, dest, *a, **k)
        finally:
            img.write_bytes(original)                  # and puts the old one back
            os.utime(img, ns=(st.st_atime_ns, st.st_mtime_ns))

    monkeypatch.setattr(assemble, "assemble", swap)
    rec = flow.run([img])[0]
    assert rec.exit == 0, rec.steps
    with pikepdf.open(rec.output["path"]) as pdf:
        box = [float(x) for x in pdf.pages[0].mediabox]
    assert abs((box[2] - box[0]) / (box[3] - box[1]) - 850 / 1100) < 0.01   # the sealed picture's shape


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM and SIGHUP end a POSIX process")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_terminated_run_removes_its_decrypted_copy(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     name: str, untouched):
    import signal
    import time

    from pdfintake.pdf import classify

    sig = getattr(signal, name)
    if signal.getsignal(sig) is not signal.SIG_DFL:
        pytest.skip(f"{name} already has a handler in this process")
    src = _locked_pdf(tmp_path / "Statement.pdf")
    pw = private_file(tmp_path / "saved-passwords", [PASSWORD])
    untouched(src, pw)
    held: list[list[str]] = []

    def stopped(pdf: Path, **k: Any) -> Any:
        held.append(sorted(p.name for p in Path(pdf).parent.iterdir()))
        assert signal.getsignal(sig) is flow._end_the_call      # never send it without the handler
        os.kill(os.getpid(), sig)                              # launchd, a closed terminal, a dropped session
        time.sleep(5)
        raise AssertionError("the signal didn't end the run")

    monkeypatch.setattr(classify, "classify", stopped)
    with pytest.raises(SystemExit) as info:
        flow.run([src], flow.RunOptions(passwords_file=pw))
    assert info.value.code == 128 + int(sig)
    assert "decrypted.pdf" in held[0]
    assert env.work_left() == []                       # the decrypted copy went on the way out
    assert signal.getsignal(sig) is signal.SIG_DFL     # and the handler was put back


@pytest.mark.skipif(sys.platform == "win32", reason="the work-folder lock is POSIX only")
def test_a_killed_runs_work_folder_goes_at_the_next_run(env: Env, tmp_path: Path, untouched):
    import fcntl

    work = env.cache / "work"
    work.mkdir(parents=True)
    orphan, live, starting, old = (work / n for n in ("orphan-1", "live-1", "starting-1", "old-1"))
    for folder in (orphan, live, starting, old):
        folder.mkdir()
        (folder / "decrypted.pdf").write_bytes(b"%PDF-1.7 made up")
    for folder in (orphan, live):
        (folder / flow.LOCK_NAME).write_bytes(b"")
    fd = os.open(live / flow.LOCK_NAME, os.O_RDONLY)  # a live run holds this one
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        long_ago = 0.0
        os.utime(old, (long_ago + 1e9, long_ago + 1e9))
        src = image_pdf(tmp_path / "Scan.pdf")
        untouched(src)
        rec = flow.run([src])[0]
        assert rec.exit == 0
        assert [p.name for p in env.work_left()] == ["live-1", "starting-1"]
    finally:
        os.close(fd)


# A child process that runs one file and stops for good between staging and publish: the partial
# file is written and flushed beside the output's name, and the rename is next. The parent kills it.
_KILLED_RUN = r"""
import os, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import pytest
import test_flow
from pdfintake import flow, fsutil
root, cache, src, mark = (Path(a) for a in sys.argv[3:7])
root.mkdir(parents=True, exist_ok=True)
mp = pytest.MonkeyPatch()
test_flow.make_env(root, mp)
mp.setenv("PDFINTAKE_CACHE", str(cache))
real = fsutil._fsync_file
def staged(fd):
    real(fd)
    mark.write_text("staged")
    time.sleep(600)
fsutil._fsync_file = staged
flow.run([src])
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL and the work-folder lock are POSIX only")
def test_a_run_killed_between_staging_and_publish_is_swept_by_the_next_run(env: Env, tmp_path: Path, untouched):
    """SIGKILL runs no cleanup at all, so the next run removes the partial file it left in the user's folder."""
    import signal
    import time

    folder = tmp_path / "Desktop"
    folder.mkdir()
    src = image_pdf(folder / "Scan.pdf")
    dest = folder / "Scan (searchable).pdf"
    partial = fsutil._partial_path(dest)
    mark = tmp_path / "staged"
    untouched(src)
    log = tmp_path / "child.log"
    with open(log, "wb") as out:
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", _KILLED_RUN, str(REPO / "src"), str(REPO / "tests"),
                                  str(tmp_path / "child"), str(env.cache), str(src), str(mark)],
                                 stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while not mark.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert mark.exists(), log.read_text("utf-8", "replace")[-4000:]
        # Staged, not published: the partial file sits beside the output's still-free name, and the
        # run noted that name in its work folder's lock file before it began writing.
        assert partial.is_file() and not dest.exists()
        locks = list((env.cache / "work").glob(f"*/{flow.LOCK_NAME}"))
        assert len(locks) == 1 and locks[0].read_bytes() == os.fsencode(os.path.abspath(dest)) + b"\0"
        child.send_signal(signal.SIGKILL)
        child.wait(30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(30)
    assert child.returncode == -signal.SIGKILL
    assert partial.is_file() and not dest.exists()     # nothing was cleaned up on the way out
    assert len(env.work_left()) == 1

    elsewhere = tmp_path / "Elsewhere"
    elsewhere.mkdir()
    second = image_pdf(elsewhere / "Letter.pdf")
    untouched(second)
    rec = flow.run([second])[0]                        # the next run, on another file in another folder
    assert rec.exit == 0
    assert not partial.exists() and not dest.exists()  # the killed run's partial file is gone
    assert sorted(p.name for p in folder.iterdir()) == ["Scan.pdf"]
    assert env.work_left() == []                       # and so is its work folder


@pytest.mark.skipif(sys.platform == "win32", reason="the work-folder and partial-file locks are POSIX only")
def test_the_sweep_removes_only_a_dead_runs_own_partial_files(env: Env, tmp_path: Path, untouched):
    import fcntl

    folder = tmp_path / "Desktop"
    folder.mkdir()

    def partial_of(dest: Path) -> Path:
        return fsutil._partial_path(dest)

    stale = folder / "Stale (searchable).pdf"          # the dead run's partial file: removed
    held = folder / "Held (searchable).pdf"            # a live writer holds this partial file: kept
    linked = folder / "Linked (searchable).pdf"        # this partial name is a link: kept, never followed
    users = folder / "Users (searchable).pdf"          # the user's own file under a noted name: untouched
    cut = folder / "Cut (searchable).pdf"              # an entry a kill cut short (no NUL): ignored
    unnoted = folder / "Unnoted (searchable).pdf"      # never noted by any run: its partial file is kept
    live = folder / "Live (searchable).pdf"            # noted by a run that is still alive: kept
    for d in (stale, held, cut, unnoted):
        partial_of(d).write_bytes(b"half written")
    partial_of(live).write_bytes(b"a live run's half")
    target = tmp_path / "elsewhere.pdf"
    target.write_bytes(b"a file of the user's")
    partial_of(linked).symlink_to(target)
    users.write_bytes(b"the user's own file")

    work = env.cache / "work"
    dead, alive = work / "dead-1", work / "alive-1"
    for f in (dead, alive):
        f.mkdir(parents=True)
    noted = b"".join(os.fsencode(str(p)) + b"\0" for p in (stale, held, linked, users))
    (dead / flow.LOCK_NAME).write_bytes(noted + b"relative/x.pdf\0" + os.fsencode(str(cut)))
    (alive / flow.LOCK_NAME).write_bytes(os.fsencode(str(live)) + b"\0")
    fds = [os.open(alive / flow.LOCK_NAME, os.O_RDONLY), os.open(partial_of(held), os.O_RDONLY)]
    try:
        for fd in fds:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        src = image_pdf(tmp_path / "Scan.pdf")
        untouched(src)
        rec = flow.run([src])[0]
        assert rec.exit == 0
    finally:
        for fd in fds:
            os.close(fd)
    assert not partial_of(stale).exists()
    assert partial_of(held).read_bytes() == b"half written"
    assert partial_of(linked).is_symlink() and target.read_bytes() == b"a file of the user's"
    assert users.read_bytes() == b"the user's own file"
    assert partial_of(cut).exists() and partial_of(unnoted).exists()
    assert partial_of(live).read_bytes() == b"a live run's half"
    assert [p.name for p in env.work_left()] == ["alive-1"]


def test_a_finished_run_notes_each_output_before_writing_it(env: Env, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch, untouched):
    noted: list[tuple[bytes, bool]] = []
    real = fsutil.atomic_write

    def spy(built: Path, dest: Path, **kw: Any) -> None:
        locks = list((env.cache / "work").glob(f"*/{flow.LOCK_NAME}"))
        noted.append((b"".join(p.read_bytes() for p in locks), os.path.lexists(fsutil._partial_path(dest))))
        real(built, dest, **kw)

    monkeypatch.setattr(fsutil, "atomic_write", spy)
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src], flow.RunOptions(smaller=True))[0]
    assert rec.exit == 0 and rec.smaller["applied"] is True
    lossless, beside = Path(rec.output["path"]), Path(rec.smaller["written"])
    if sys.platform != "win32":
        entry = lambda p: os.fsencode(str(p)) + b"\0"     # noqa: E731
        assert noted == [(entry(lossless), False), (entry(lossless) + entry(beside), False)]
    rec2 = flow.run([image_pdf(tmp_path / "Second.pdf", image=smooth_photo(seed=1))])[0]
    assert rec2.exit == 0 and any(o.kind == "smaller" for o in rec2.offers)
    noted.clear()
    flow.apply(rec2.id, choice="smaller")                # apply notes its one write as well
    if sys.platform != "win32":
        assert noted == [(os.fsencode(str(Path(rec2.output["path"]).with_name("Second (searchable) (smaller).pdf")))
                          + b"\0", False)]
    assert env.work_left() == []


def _someone_saves(path: Path, data: bytes, monkeypatch: pytest.MonkeyPatch, *, on_write: int = 1) -> None:
    """Another program saves ``data`` as ``path`` during pdfintake's ``on_write``-th write, after the
    partial file is flushed and before the rename: after every guard said the name was free."""
    real = fsutil._fsync_file
    calls: list[int] = []

    def flushed(fd: int) -> None:
        real(fd)
        calls.append(fd)
        if len(calls) == on_write and not os.path.lexists(path):
            path.write_bytes(data)

    monkeypatch.setattr(fsutil, "_fsync_file", flushed)


LATE = b"%PDF-1.4\n% a file another program saved, made up\n"


def test_a_file_saved_under_the_output_name_during_the_write_is_kept(env: Env, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch, untouched):
    """The guard found the name free; the publish never replaces a file that appeared there since."""
    src = image_pdf(tmp_path / "Scan.pdf")
    dest = tmp_path / "Scan (searchable).pdf"
    untouched(src)
    with monkeypatch.context() as m:
        _someone_saves(dest, LATE, m)
        rec = flow.run([src])[0]
    assert_record_ok(rec)
    assert (rec.status, rec.exit, rec.output) == ("failed", 2, None)
    assert rec.steps[-1]["step"] == "commit" and rec.steps[-1]["detail"]["code"] == "output_exists"
    assert dest.read_bytes() == LATE                   # the late file's bytes survive
    assert not list(tmp_path.glob("*" + fsutil.PARTIAL_SUFFIX)) and env.work_left() == []
    with monkeypatch.context() as m:                   # with --overwrite the replace is what was asked for
        _someone_saves(dest, LATE, m)
        again = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert again.exit == 0 and dest.read_bytes() != LATE
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == again.output["sha256"]


def test_run_smaller_keeps_a_file_saved_under_the_beside_name(env: Env, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch, untouched):
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    beside = tmp_path / "Photo (searchable) (smaller).pdf"
    untouched(src)
    _someone_saves(beside, LATE, monkeypatch, on_write=2)        # the first write is the lossless file
    rec = flow.run([src], flow.RunOptions(smaller=True))[0]
    assert_record_ok(rec)
    assert rec.exit == 0 and Path(rec.output["path"]).is_file()  # the lossless file is written
    assert beside.read_bytes() == LATE
    step = [s for s in rec.steps if s["step"] == "smaller"][-1]
    assert step["status"] == "failed" and step["detail"]["code"] == "output_exists"
    # Its staged candidate is removed, so the record keeps none and offers nothing (decided 2026-11-06).
    assert rec.smaller is None and not any(o.kind == "smaller" for o in rec.offers)
    assert not (env.cache / "staging" / rec.id).exists()
    assert not list(tmp_path.glob("*" + fsutil.PARTIAL_SUFFIX)) and env.work_left() == []


def test_apply_smaller_never_replaces_a_file_saved_under_its_name(env: Env, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch, untouched):
    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src])[0]
    assert any(o.kind == "smaller" for o in rec.offers)
    beside = tmp_path / "Photo (searchable) (smaller).pdf"
    with monkeypatch.context() as m:
        _someone_saves(beside, LATE, m)
        with pytest.raises(UsageError) as info:
            flow.apply(rec.id, choice="smaller")
    assert info.value.code == "output_exists"
    assert beside.read_bytes() == LATE
    assert not list(tmp_path.glob("*" + fsutil.PARTIAL_SUFFIX)) and env.work_left() == []
    again = flow.apply(rec.id, choice="smaller")[0]    # the offer still stands: the next free name
    assert again.status == "resolved" and beside.read_bytes() == LATE
    assert Path(again.smaller["written"]).name == "Photo (searchable) (smaller 2).pdf"


# --------------------------------------------------------------------------- commit

def test_apply_smaller_rechecks_the_refusals(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake.errors import Refused

    src = image_pdf(tmp_path / "Photo.pdf", image=smooth_photo())
    untouched(src)
    rec = flow.run([src])[0]
    assert any(o.kind == "smaller" for o in rec.offers)
    beside = Path(rec.output["path"]).with_name("Photo (searchable) (smaller).pdf")
    reads: list[Path] = []
    real_read = refusals.read
    with monkeypatch.context() as m:
        m.setattr(refusals, "read", lambda p: (reads.append(Path(p)), real_read(p))[1])
        m.setattr(refusals, "refusal", lambda report, **k: ("xfa", "made-up hard refusal"))
        with pytest.raises(Refused):
            flow.apply(rec.id, choice="smaller")
        assert reads and all(env.cache in p.parents for p in reads)     # read from a sealed work copy
        m.setattr(refusals, "refusal", lambda report, **k: None)
        m.setattr(refusals, "copy_reasons", lambda *a, **k: ["signed"])
        with pytest.raises(Refused) as info:
            flow.apply(rec.id, choice="smaller")
        assert info.value.code == "no_smaller_for_a_copy"
    assert not beside.exists() and env.work_left() == []
    assert flow.apply(rec.id, choice="smaller")[0].status == "resolved" and beside.is_file()


def test_commit_rechecks_refusals_and_the_seal(env: Env, tmp_path: Path):
    from fixtures import refusal_shells as shells

    src = image_pdf(tmp_path / "In.pdf")
    built = env.cache / "built.pdf"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_bytes(src.read_bytes())
    seal = integrity.InputSeal.take([src])
    ok = flow.commit(built, tmp_path / "Out.pdf", seal, refusals.read(src), "ordinary", adds_layer=True,
                     inputs=[src], overwrite=False)
    assert ok.sha256 == hashlib.sha256(built.read_bytes()).hexdigest() and ok.path.is_file()
    xfa = tmp_path / "Form.pdf"
    xfa.write_bytes(shells.xfa())
    from pdfintake.errors import Refused

    with pytest.raises(Refused):
        flow.commit(built, tmp_path / "Out2.pdf", integrity.InputSeal.take([xfa]), refusals.read(xfa), "ordinary",
                    adds_layer=True, inputs=[xfa], overwrite=False)
    with pytest.raises(UsageError) as info:
        flow.commit(built, tmp_path / "Out3.pdf", seal, refusals.read(src), "evidence", adds_layer=True,
                    inputs=[src], overwrite=True)
    assert info.value.code == "copy_never_overwrites"
    with open(src, "ab") as fh:
        fh.write(b"%")
    with pytest.raises(InputChanged):
        flow.commit(built, tmp_path / "Out4.pdf", seal, None, "ordinary", adds_layer=True, inputs=[src],
                    overwrite=False)
    assert not (tmp_path / "Out2.pdf").exists() and not (tmp_path / "Out3.pdf").exists()
    assert not (tmp_path / "Out4.pdf").exists()


# --------------------------------------------------------------------------- every write is noted; inputs at written names

def _lock_notes(env: Env) -> bytes:
    return b"".join(p.read_bytes() for p in sorted((env.cache / "work").glob(f"*/{flow.LOCK_NAME}")))


@pytest.mark.skipif(sys.platform == "win32", reason="the work-folder lock is POSIX only")
def test_commit_without_a_journal_notes_its_dest_itself(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        untouched):
    """export calls commit with no journal; commit notes the write in a work folder of its own (decided 2026-11-06)."""
    src = image_pdf(tmp_path / "In.pdf")
    untouched(src)
    built = fsutil.work_dir("staging", "export-made-up") / "built.pdf"
    built.write_bytes(src.read_bytes())
    dest = tmp_path / "In (clean).pdf"
    seen: list[tuple[bytes, bool]] = []
    real = fsutil.atomic_write

    def spy(b: Path, d: Path, **kw: Any) -> None:
        seen.append((_lock_notes(env), os.path.lexists(fsutil._partial_path(d))))
        real(b, d, **kw)

    monkeypatch.setattr(fsutil, "atomic_write", spy)
    ref = flow.commit(built, dest, integrity.InputSeal.take([src]), None, "ordinary", adds_layer=False,
                      inputs=[src], overwrite=False)
    assert ref.path == dest and dest.read_bytes() == built.read_bytes()
    assert seen == [(os.fsencode(str(dest)) + b"\0", False)]
    assert env.work_left() == []                          # the folder goes with the write


_KILLED_COMMIT = r"""
import os, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1]]
cache, src, built, dest, mark = (Path(a) for a in sys.argv[2:7])
os.environ["PDFINTAKE_CACHE"] = str(cache)
from pdfintake import flow, fsutil, integrity
real = fsutil._fsync_file
def staged(fd):
    real(fd)
    mark.write_text("staged")
    time.sleep(600)
fsutil._fsync_file = staged
flow.commit(built, dest, integrity.InputSeal.take([src]), None, "ordinary", adds_layer=False, inputs=[src],
            overwrite=False)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL and the work-folder lock are POSIX only")
def test_an_export_killed_between_staging_and_publish_is_swept_by_the_next_run(env: Env, tmp_path: Path, untouched):
    """export builds in staging, which no sweep reads; commit's own note is what lets the next run clean up."""
    import signal
    import time

    folder = tmp_path / "Desktop"
    folder.mkdir()
    src = image_pdf(folder / "In.pdf")
    untouched(src)
    built = fsutil.work_dir("staging", "export-made-up") / "built.pdf"
    built.write_bytes(src.read_bytes())
    dest = folder / "In (clean).pdf"
    partial = fsutil._partial_path(dest)
    mark = tmp_path / "staged"
    log = tmp_path / "child.log"
    with open(log, "wb") as out:
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", _KILLED_COMMIT, str(REPO / "src"), str(env.cache),
                                  str(src), str(built), str(dest), str(mark)], stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while not mark.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert mark.exists(), log.read_text("utf-8", "replace")[-4000:]
        assert partial.is_file() and not dest.exists()
        child.send_signal(signal.SIGKILL)
        child.wait(30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(30)
    assert partial.is_file() and len(env.work_left()) == 1
    rec = flow.run([image_pdf(tmp_path / "Other.pdf")])[0]
    assert rec.exit == 0
    assert not partial.exists() and not dest.exists() and env.work_left() == []
    assert sorted(p.name for p in folder.iterdir()) == ["In.pdf"]


def test_an_input_at_another_outputs_partial_name_is_never_removed(env: Env, tmp_path: Path, untouched):
    """A killed run's leftover is a complete PDF; run with its siblings, it is an input (decided 2026-11-06)."""
    leftover = image_pdf(tmp_path / ("Scan (searchable).pdf" + fsutil.PARTIAL_SUFFIX), pages=2)
    scan = image_pdf(tmp_path / "Scan.pdf")
    untouched(leftover, scan)
    recs = flow.run([leftover, scan])
    assert recs[0].exit == 0 and leftover.is_file()
    assert (recs[1].exit, recs[1].output) == (2, None)
    assert recs[1].steps[-1]["detail"]["code"] == "output_is_input"
    assert not (tmp_path / "Scan (searchable).pdf").exists()


def test_o_whose_partial_name_is_the_input_is_refused(env: Env, tmp_path: Path, untouched):
    src = image_pdf(tmp_path / ("Letter.pdf" + fsutil.PARTIAL_SUFFIX))
    untouched(src)
    rec = flow.run([src], flow.RunOptions(out=tmp_path / "Letter.pdf"))[0]
    assert (rec.exit, rec.output) == (2, None) and rec.steps[-1]["detail"]["code"] == "output_is_input"
    assert src.is_file() and not (tmp_path / "Letter.pdf").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the work-folder lock is POSIX only")
def test_the_sweep_at_run_start_never_removes_an_input_of_that_run(env: Env, tmp_path: Path, untouched):
    folder = tmp_path / "Desktop"
    folder.mkdir()
    dest = folder / "Scan (searchable).pdf"
    leftover = image_pdf(fsutil._partial_path(dest))       # a killed run's complete partial file
    untouched(leftover)
    dead = env.cache / "work" / "dead-1"
    dead.mkdir(parents=True)
    (dead / flow.LOCK_NAME).write_bytes(os.fsencode(str(dest)) + b"\0")
    rec = flow.run([leftover], flow.RunOptions(outdir=tmp_path))[0]   # the user runs pdfintake on it
    assert rec.exit == 0 and leftover.is_file()
    assert env.work_left() == []                           # the dead folder itself still goes


def test_apply_lang_keeps_an_output_saved_over_during_its_write(env: Env, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch, untouched):
    """The replace is a compare-and-swap at the rename, not only a hash check before the copy (decided 2026-11-06)."""
    env.lines = {"*": list(HU_LINES)}
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(fetch="never"))[0]
    out = Path(rec.output["path"])
    edited = out.read_bytes() + b"\n% the user's note, made up\n"
    real = fsutil._fsync_file

    def user_saves(fd: int) -> None:                       # after commit's hash check and guard
        real(fd)
        tmp = out.with_name(".save-tmp")
        tmp.write_bytes(edited)
        os.replace(tmp, out)                               # how document apps save

    with monkeypatch.context() as m:
        m.setattr(fsutil, "_fsync_file", user_saves)
        child = flow.apply(rec.id, lang="hu")[0]
    assert child.exit == 5 and child.output is None
    assert child.steps[-1]["detail"]["code"] == "output_changed"
    assert out.read_bytes() == edited
    assert not [p for p in tmp_path.iterdir() if p.name.endswith((fsutil.PARTIAL_SUFFIX, fsutil.ASIDE_SUFFIX))]
    assert record.load(rec.id).status == "done" and env.work_left() == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
@pytest.mark.parametrize("case", ["not_utf8", "too_many"])
def test_a_refused_passwords_file_leaves_no_password_in_the_batch(tmp_path: Path, case: str):
    """The error a batch keeps for every file holds none of the file's lines (DESIGN §12.1, decided 2026-11-06)."""
    import gc

    secret = "Invented-Batch-Secret-" + os.urandom(4).hex()
    digest = hashlib.sha256(secret.encode()).hexdigest()
    lines = [secret] + ([f"Invented-{i:03d}" for i in range(120)] if case == "too_many" else [])
    data = "\n".join(lines).encode() + b"\n" + (b"\xff\xfe made up\n" if case == "not_utf8" else b"")
    del secret, lines
    pw = tmp_path / "pw"
    pw.write_bytes(data)
    del data
    os.chmod(pw, 0o600)
    batch = flow._Batch(flow.RunOptions(passwords_file=pw))
    for _ in range(2):                                     # the first failure and the kept one
        with pytest.raises(UsageError) as info:
            batch.passwords()
        assert info.value.code == "passwords_file_unsafe" and info.value.detail["reason"] == case
    del info

    def holds(obj: Any) -> bool:
        if isinstance(obj, str):
            return len(obj) < 200 and hashlib.sha256(obj.encode("utf-8", "surrogatepass")).hexdigest() == digest
        if isinstance(obj, (bytes, bytearray)):
            return any(hashlib.sha256(line).hexdigest() == digest for line in bytes(obj).split(b"\n"))
        return False

    gc.collect()
    found = []
    for obj in gc.get_objects():
        if isinstance(obj, (list, tuple, set, frozenset)):
            if any(holds(x) for x in obj):
                found.append(type(obj).__name__)
        elif isinstance(obj, dict):
            if any(holds(x) for x in obj.values()):
                found.append(type(obj).__name__)
        elif isinstance(obj, BaseException) and holds(getattr(obj, "object", None)):
            found.append(type(obj).__name__)
    assert found == []
    assert batch._errors["passwords"].__traceback__ is None and batch._errors["passwords"].__context__ is None


# --------------------------------------------------------------------------- the detection cache (DESIGN §9.1)

@pytest.fixture
def ocr_renders(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every pass-1 render (a PNG named ocr-<page>.png), by page index."""
    from pdfintake.pdf import render

    out: list[int] = []
    real = render.Renderer.render

    def counting(self: Any, index: int, dpi: float, dest: Path) -> PageImage:
        if Path(dest).name.startswith("ocr-"):
            out.append(index)
        return real(self, index, dpi, dest)

    monkeypatch.setattr(render.Renderer, "render", counting)
    return out


def _detect_step(rec: record.Record) -> dict[str, Any]:
    steps = [s for s in rec.steps if s["step"] == "detect"]
    assert len(steps) == 1, rec.steps
    return steps[0]["detail"]


def _entries(cache: Path) -> list[Path]:
    return sorted(p for p in cache.iterdir() if p.name.startswith(flow.CACHE_ENTRY_PREFIX))


def test_a_rerun_takes_detection_from_the_cache(env: Env, tmp_path: Path, ocr_renders: list[int], untouched):
    src = image_pdf(tmp_path / "Scan.pdf", 2)
    untouched(src)
    first = flow.run([src], flow.RunOptions(workers=2))[0]
    assert first.exit == 0, first.steps
    assert _detect_step(first) == {"pages": 2, "cached": False} and ocr_renders == [0, 1]
    (cache,) = env.detection_caches()
    assert cache.name == f"{hashlib.sha256(src.read_bytes()).hexdigest()}-{flow.__version__}"
    reads = (len(env.reader.reads), len(env.checker.reads))

    again = flow.run([src], flow.RunOptions(workers=2, overwrite=True))[0]
    assert again.exit == 0, again.steps
    assert _detect_step(again) == {"pages": 2, "cached": True}
    assert ocr_renders == [0, 1]                                         # no page rendered again
    assert (len(env.reader.reads), len(env.checker.reads)) == reads      # nothing read: pass 2 reused pass 1's reads
    assert again.output["sha256"] == first.output["sha256"]              # the same output
    assert record.to_json(again)["pages"] == record.to_json(first)["pages"]
    assert env.work_left() == [] and env.detection_caches() == [cache] and len(_entries(cache)) == 1


def test_apply_lang_takes_detection_from_the_cache(env: Env, tmp_path: Path, ocr_renders: list[int], untouched):
    env.lines = {"*": list(HU_LINES)}
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src], flow.RunOptions(fetch="never"))[0]
    assert rec.exit == 0 and any(o.kind == "language" for o in rec.offers), rec.steps
    assert _detect_step(rec)["cached"] is False and ocr_renders == [0]
    reads = len(env.reader.reads)
    english = [r for r in env.reader.reads + env.checker.reads if r[1] == "en"]

    child = flow.apply(rec.id, lang="hu")[0]
    assert child.exit == 0, child.steps
    assert _detect_step(child) == {"pages": 1, "cached": True}
    assert ocr_renders == [0]                                            # no page rendered again
    assert [r for r in env.reader.reads + env.checker.reads if r[1] == "en"] == english   # no pass-1 read again
    assert {r[1] for r in env.reader.reads[reads:]} == {"hu"}            # only the Hungarian pack read the page
    assert record.to_json(child)["pages"][0]["pack"] == "hu"
    assert env.work_left() == []


def test_an_encrypted_input_is_never_cached(env: Env, tmp_path: Path, ocr_renders: list[int], untouched):
    src = _locked_pdf(tmp_path / "Statement.pdf")
    pw = private_file(tmp_path / "saved-passwords", [PASSWORD])
    untouched(src, pw)
    first = flow.run([src], flow.RunOptions(passwords_file=pw))[0]
    (tmp_path / "again").mkdir()
    again = flow.run([src], flow.RunOptions(passwords_file=pw, outdir=tmp_path / "again"))[0]
    assert (first.exit, again.exit) == (0, 0), (first.steps, again.steps)
    assert [_detect_step(r)["cached"] for r in (first, again)] == [False, False]
    assert ocr_renders == [0, 0]                                         # rendered and detected each time
    assert env.detection_caches() == [] and env.work_left() == []        # its decrypted pages went with the run


def test_a_damaged_detection_cache_is_a_fresh_detection(env: Env, tmp_path: Path, ocr_renders: list[int], untouched):
    from pdfintake.ocr import pipeline

    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    first = flow.run([src])[0]
    (cache,) = env.detection_caches()
    (entry,) = _entries(cache)
    f = entry / pipeline.CACHE_FILE
    f.write_bytes(f.read_bytes()[: f.stat().st_size // 2])
    again = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert again.exit == 0 and _detect_step(again)["cached"] is False and ocr_renders == [0, 0]
    assert again.output["sha256"] == first.output["sha256"]
    assert _entries(cache) != [entry] and len(_entries(cache)) == 1      # the damaged entry was replaced
    third = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert _detect_step(third)["cached"] is True and ocr_renders == [0, 0]


def _orient_step(rec: record.Record) -> dict[str, Any]:
    steps = [s for s in rec.steps if s["step"] == "orientation"]
    assert len(steps) == 1, rec.steps
    return steps[0]["detail"]


class FakeVoter:
    """The orientation voter: the same made-up line at every turn, so the vote keeps the page as it is."""

    def __init__(self) -> None:
        self.reads = 0
        self.fail = False

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        self.reads += 1
        if self.fail:
            raise EngineFailure("engine_failed", "made-up failure under load")
        lines = build.lines_for(["Bread rises as the dough warms"], char_w=14.0, with_words=True)
        return EngineRead("voter", "fakevote", "fakevote:en", "reader", "ok", lines, 0.001, None, {})

    def close(self) -> None:
        pass


class FakeOrientOsd:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def detect(self, image: Any, dpi: float) -> Any:
        from pdfintake.types import OsdResult

        self.calls += 1
        if self.fail:
            raise EngineFailure("osd_failed", "made-up OSD failure under load")
        return OsdResult("Latin", 5.0, 0, 5.0)

    def close(self) -> None:
        pass


@dataclass
class Orienting:
    voter: FakeVoter
    osd: FakeOrientOsd
    checks: list[str]


@pytest.fixture
def orienting(env: Env, monkeypatch: pytest.MonkeyPatch) -> Orienting:
    """A voter and OSD that open, and the hash check of their models recorded (the fake base pack has no
    model files to check)."""
    from pdfintake.ocr import pipeline

    o = Orienting(FakeVoter(), FakeOrientOsd(), [])
    monkeypatch.setattr(flow, "_open_voter", lambda base: o.voter)
    monkeypatch.setattr(flow, "_open_osd", lambda base: o.osd)
    monkeypatch.setattr(pipeline, "check_orientation_models", lambda base: o.checks.append(base.manifest.id))
    return o


def test_a_rerun_takes_orientation_from_the_cache(env: Env, tmp_path: Path, orienting: Orienting,
                                                  ocr_renders: list[int], untouched):
    from pdfintake.ocr import pipeline

    src = image_pdf(tmp_path / "Scan.pdf", 2)
    untouched(src)
    first = flow.run([src], flow.RunOptions(workers=2))[0]
    assert first.exit == 0, first.steps
    assert _orient_step(first)["cached"] is False and _orient_step(first)["voter"] is True
    assert (orienting.voter.reads, orienting.osd.calls) == (8, 2)        # four turns and one OSD call a page
    (cache,) = env.detection_caches()
    assert (cache / pipeline.ORIENT_CACHE_FILE).is_file()

    again = flow.run([src], flow.RunOptions(workers=2, overwrite=True))[0]
    assert again.exit == 0, again.steps
    assert _orient_step(again) == {**_orient_step(first), "cached": True}
    assert (orienting.voter.reads, orienting.osd.calls) == (8, 2)        # nothing rendered or read for orientation
    assert orienting.checks == ["en"]                                    # the models were still hash-checked
    assert _detect_step(again)["cached"] is True and ocr_renders == [0, 1]
    assert again.output["sha256"] == first.output["sha256"]
    assert record.to_json(again)["pages"] == record.to_json(first)["pages"]


@pytest.mark.parametrize("what", ["voter", "osd"])
def test_an_orientation_that_met_a_failure_is_never_kept(env: Env, tmp_path: Path, orienting: Orienting, what: str,
                                                         untouched):
    from pdfintake.ocr import pipeline

    getattr(orienting, what).fail = True
    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    first = flow.run([src])[0]
    assert first.exit == 0 and _orient_step(first)["cached"] is False, first.steps
    (cache,) = env.detection_caches()
    assert not (cache / pipeline.ORIENT_CACHE_FILE).exists()
    getattr(orienting, what).fail = False                                # it works again
    again = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert again.exit == 0 and _orient_step(again)["cached"] is False    # oriented again, not the failure reused
    third = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert third.exit == 0 and _orient_step(third)["cached"] is True
    assert (orienting.voter.reads, orienting.osd.calls) == (8, 2)


def test_a_tampered_model_stops_the_file_on_an_orientation_hit(env: Env, tmp_path: Path, orienting: Orienting,
                                                               monkeypatch: pytest.MonkeyPatch, untouched):
    from pdfintake.errors import IntegrityError
    from pdfintake.ocr import pipeline

    src = image_pdf(tmp_path / "Scan.pdf")
    untouched(src)
    assert flow.run([src])[0].exit == 0

    def tampered(base: Any) -> None:
        raise IntegrityError("hash_mismatch", "made up: a model file doesn't match its pin", path="det.onnx")

    monkeypatch.setattr(pipeline, "check_orientation_models", tampered)
    again = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert again.exit != 0 and not again.output, again.steps
    assert again.steps[-1]["step"] == "orientation" and again.steps[-1]["detail"]["code"] == "hash_mismatch"
    assert orienting.voter.reads == 4                                    # and nothing was read either


class Clock(datetime):
    """``flow``'s clock, a minute later at every call: a re-run never lands in the same second."""

    at = datetime(2031, 3, 1, 9, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz: Any = None) -> Any:
        Clock.at = Clock.at + timedelta(minutes=1)
        return Clock.at if tz is None else Clock.at.astimezone(tz)


def _created(path: Path) -> str:
    from pdfintake.pdf import meta

    return meta.snapshot(path).info["CreationDate"]


def test_a_rerun_of_an_image_takes_detection_from_the_cache(env: Env, tmp_path: Path, ocr_renders: list[int],
                                                            monkeypatch: pytest.MonkeyPatch, untouched):
    monkeypatch.setattr(flow, "datetime", Clock)
    src = build.write_png(page_image(), tmp_path / "Receipt.png", dpi=100)
    untouched(src)
    first = flow.run([src])[0]
    assert first.exit == 0 and _detect_step(first)["cached"] is False, first.steps
    (cache,) = env.detection_caches()
    assert (cache / flow.ASSEMBLED_FILE).is_file()
    again = flow.run([src], flow.RunOptions(overwrite=True))[0]
    assert again.exit == 0 and _detect_step(again)["cached"] is True, again.steps
    assert ocr_renders == [0]                                            # no page rendered again
    assert again.output["sha256"] == first.output["sha256"]
    assert _created(Path(again.output["path"])).startswith("D:20310301090")   # the first build's date, kept


def test_apply_lang_on_images_takes_detection_from_the_cache(env: Env, tmp_path: Path, ocr_renders: list[int],
                                                             monkeypatch: pytest.MonkeyPatch, untouched):
    monkeypatch.setattr(flow, "datetime", Clock)
    env.lines = {"*": list(HU_LINES)}
    a = build.write_png(page_image(), tmp_path / "Levél 1.png", dpi=100)
    b = build.write_png(page_image(900, 1100), tmp_path / "Levél 2.png", dpi=100)
    untouched(a, b)
    rec = flow.run([a, b], flow.RunOptions(fetch="never"))[0]
    assert rec.exit == 0 and _detect_step(rec)["cached"] is False, rec.steps
    child = flow.apply(rec.id, lang="hu")[0]
    assert child.exit == 0, child.steps
    assert _detect_step(child) == {"pages": 2, "cached": True} and ocr_renders == [0, 1]
    assert record.to_json(child)["pages"][0]["pack"] == "hu"


def test_a_damaged_assembly_date_is_replaced(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, untouched):
    monkeypatch.setattr(flow, "datetime", Clock)
    src = build.write_png(page_image(), tmp_path / "Receipt.png", dpi=100)
    untouched(src)
    assert flow.run([src])[0].exit == 0
    (cache,) = env.detection_caches()
    for bad in (b"not json", b'{"v": 1, "created": "2031-13-01T00:00:00Z"}', b'{"v": 2, "created": "2031-03-01T09:01:00Z"}',
                b'{"v": 1, "created": "2099-01-01T00:00:00Z"}'):           # the last is in the future
        (cache / flow.ASSEMBLED_FILE).write_bytes(bad)
        rec = flow.run([src], flow.RunOptions(overwrite=True))[0]
        assert rec.exit == 0 and _detect_step(rec)["cached"] is False, (bad, rec.steps)
        stored = json.loads((cache / flow.ASSEMBLED_FILE).read_text("utf-8"))
        assert set(stored) == {"v", "created"} and stored["created"].startswith("2031-03-01T")


def test_the_detection_cache_holds_no_page_text_beyond_detection(env: Env, tmp_path: Path, untouched):
    from pdfintake.ocr import pipeline

    pass_two = ["Volcanoes form where molten rock rises through the crust,",
                "and the Moon has old lava plains anyone can see at night."]
    env.lines = {"*": list(HU_LINES), "hu": pass_two}                    # the Hungarian pack reads other text
    src = image_pdf(tmp_path / "Levél.pdf")
    untouched(src)
    rec = flow.run([src])[0]
    assert rec.exit == 0 and record.to_json(rec)["pages"][0]["pack"] == "hu", rec.steps
    assert "Volcanoes" in pdf_text(Path(rec.output["path"]))             # pass 2's text is in the output
    (cache,) = env.detection_caches()
    files = [p for p in cache.rglob("*") if p.is_file()]
    assert sorted(p.name for p in files) == [pipeline.CACHE_FILE, "ocr-00000.png"]   # no PDF, no work copy
    blob = b"".join(p.read_bytes() for p in files)
    for text in (*pass_two, "Volcanoes", "molten", "lava"):
        assert text.encode("utf-8") not in blob
    assert HU_LINES[0].encode("utf-8") in blob                           # the eng read detection read is there


def test_the_detection_cache_never_writes_outside_pdfintakes_cache(env: Env, tmp_path: Path):
    from pdfintake.ocr import pipeline

    outside = tmp_path / "Documents"
    outside.mkdir()
    cache = flow._DetectionCache(outside)
    cache.keep("a" * 64, pipeline.DocumentSignals([], [], [], {}))
    assert cache.load("a" * 64, outside) is None
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="the orphan sweep is POSIX only")
def test_the_sweep_keeps_a_detection_cache_and_the_prune_removes_it_after_14_days(env: Env, tmp_path: Path,
                                                                                  untouched):
    import time

    def age(folder: Path, seconds: float) -> None:
        when = time.time() - seconds
        for p in [*folder.rglob("*"), folder]:
            os.utime(p, (when, when), follow_symlinks=False)

    src = image_pdf(tmp_path / "Scan.pdf")
    other = image_pdf(tmp_path / "Other.pdf", image=page_image(900, 1100))
    untouched(src, other)
    assert flow.run([src])[0].exit == 0
    (cache,) = env.detection_caches()
    age(cache, flow.ORPHAN_GRACE_S + 3600)                               # no lock file, and past the sweep's grace
    assert flow.run([other])[0].exit == 0
    assert cache in env.detection_caches() and len(_entries(cache)) == 1   # the sweep left it alone
    age(cache, 15 * 86400)
    assert flow.run([other], flow.RunOptions(overwrite=True))[0].exit == 0
    assert cache not in env.detection_caches()                          # the 14-day prune removed it
    assert len(env.detection_caches()) == 1                             # the other input's cache stays


# --------------------------------------------------------------------------- doctor

def test_doctor_reports_without_a_pack_index_and_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pdfintake import passwords

    monkeypatch.setenv("PDFINTAKE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(passwords, "_config_root", lambda: tmp_path / "config")
    monkeypatch.setattr(passwords, "candidates", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read")))

    def no_index(path: Any = None) -> Any:
        raise PackError("index_missing", "this pdfintake install has no pack index (made up)")

    monkeypatch.setattr(I, "load", no_index)
    rep = doctor.report()
    assert doctor.exit_code(rep) == 3 and rep["blocking"] is True
    assert rep["index"]["ok"] is False and rep["base_pack"]["ok"] is False
    assert rep["cache"] == str(tmp_path / "cache")
    assert set(rep["engines"]) >= {"tesseract", "vision", "rapidocr"}
    assert rep["passwords_file"] == {"path": str(tmp_path / "config" / "passwords"), "exists": False, "private": None,
                                     "reason": None}
    json.dumps(rep)


def test_doctor_is_happy_with_a_good_base_pack(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pdfintake import passwords

    monkeypatch.setattr(passwords, "_config_root", lambda: tmp_path / "config")
    (tmp_path / "config").mkdir()
    pw = tmp_path / "config" / "passwords"
    pw.write_text(PASSWORD + "\n", encoding="utf-8")
    os.chmod(pw, 0o600)
    monkeypatch.setattr(I, "load", lambda path=None: _fixture_index())
    monkeypatch.setattr(store, "base", lambda: env.base)
    depth: list[int] = []
    monkeypatch.setattr(store, "verify", lambda pack_id=None: (depth.append(integrity._GUARD["depth"]),
                                                               [store.VerifyResult("en", True, [])])[1])
    monkeypatch.setattr(store, "installed", lambda: {"en": env.base})
    monkeypatch.setattr(passwords, "candidates", lambda *a, **k: pytest.fail("doctor read the saved passwords"))
    rep = doctor.report(canary=True, check_network=True)
    assert doctor.exit_code(rep) == 0, rep["problems"]
    assert rep["base_pack"] == {"ok": True, "id": "en", "version": "1.0.0"}
    assert rep["canaries"] == [{"pack": "en", "use_id": None, "engine": None, "ok": True, "skipped": True,
                                "reason": "this pack has no canary"}]
    assert rep["network"]["ok"] is False               # the network is blocked in tests: reported, never raised
    assert depth == [1]                                # doctor holds the no-subprocess guard too
    if sys.platform != "win32":
        assert rep["passwords_file"]["private"] is True
    assert PASSWORD not in json.dumps(rep)


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX modes, and root reads anything")
@pytest.mark.parametrize("mode, count, reason", [(0o000, 1, "unreadable"), (0o600, 101, "too_many"),
                                                 (0o644, 1, "mode")])
def test_doctor_judges_the_passwords_file_as_a_run_does(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        mode: int, count: int, reason: str):
    from pdfintake import passwords

    monkeypatch.setattr(passwords, "_config_root", lambda: tmp_path / "config")
    (tmp_path / "config").mkdir()
    pw = tmp_path / "config" / "passwords"
    pw.write_text("".join(f"Invented-{i:04d}\n" for i in range(count)), encoding="utf-8")
    os.chmod(pw, mode)
    try:
        with monkeypatch.context() as m:              # doctor judges it without reading a password
            m.setattr(passwords, "candidates", lambda *a, **k: pytest.fail("doctor read the saved passwords"))
            got = doctor._passwords_file()
        with pytest.raises(UsageError) as info:
            passwords.candidates(None)                 # what every run does with it
    finally:
        os.chmod(pw, 0o600)
    assert got == {"path": str(pw), "exists": True, "private": False, "reason": reason}
    assert info.value.detail["reason"] == reason
    assert "Invented" not in json.dumps(got)


_REAL_INDEX_LOAD = I.load


def _fixture_index() -> Any:
    return _REAL_INDEX_LOAD(FIXTURE_INDEX)


# --------------------------------------------------------------------------- the real engines

def _link(src: Path, dest: Path) -> None:
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def test_a_photographed_page_becomes_searchable_with_the_real_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                      untouched):
    """The real English pack (pinned models), the real orientation voter and OSD, a sideways photo."""
    pytest.importorskip("tesserocr")
    from PIL import Image, ImageDraw, ImageFont

    from conftest import pinned_asset
    from pdfintake import platforms
    from pdfintake.langpacks import manifest as M

    plat = platforms.current()
    if not plat.supported:
        pytest.skip(f"{plat.key} isn't a supported platform")
    m = M.parse((REPO / "packs" / "en" / "manifest.json").read_bytes())
    root = tmp_path / "real" / "en-1.0.0"
    for a in m.models:
        _link(pinned_asset("en", a.key), root / "models" / a.file)
    for f in m.fonts:
        _link(pinned_asset("en", f.key), root / "fonts" / f.file)
    base = store.InstalledPack(m, root)
    monkeypatch.setenv("PDFINTAKE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(flow, "_load_index", _fixture_index)
    monkeypatch.setattr(flow, "_ensure", lambda entry, *, allow_network: base)
    monkeypatch.setattr(flow, "_installed", lambda: {"en": base})

    font = ImageFont.truetype(str(pinned_asset("en", "noto-sans")), 46)
    img = Image.new("L", (1700, 1100), 255)
    draw = ImageDraw.Draw(img)
    lines = ["The example invoice lists seven sample items", "and a total of 4521 units for the test",
             "customer. Reference AB123456 is made up."]
    for k, text in enumerate(lines):
        draw.text((90, 120 + 110 * k), text, font=font, fill=0)
    sideways = img.rotate(90, expand=True)             # shot sideways; EXIF says nothing
    src = tmp_path / "Photo of a letter.jpg"
    src.write_bytes(build.jpeg_bytes(sideways.convert("RGB"), quality=92))
    untouched(src)
    rec = flow.run([src], flow.RunOptions(workers=2))[0]
    data = assert_record_ok(rec)
    assert rec.exit == 0, rec.steps
    out = Path(rec.output["path"])
    assert out.name == "Photo of a letter (searchable).pdf"
    text = pdf_text(out)
    assert "4521" in text and "invoice" in text
    assert data["pages"][0]["orientation"]["turn_cw"] in (90, 270)
    assert all(c["ok"] for c in data["verify"])
