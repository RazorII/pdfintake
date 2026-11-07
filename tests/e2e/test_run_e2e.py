"""End to end (DESIGN §7.2, as changed by §12): the real CLI, the real index and the real pack store.

Nothing here replaces a seam. The CLI runs in this process (``cli.main``), on the packaged
``index.json`` and the installed English data wheel, with the real engines, so every CI job runs
the whole pipeline. The network is blocked for every call (the root conftest's guard), and each
call proves that nothing tried to reach it: the offline CI job runs this file again under
``unshare -rn`` on the warm cache, which is how it shows nothing downloads outside the pack store.

* **The batch:** three images make one PDF: a JPEG stored sideways with EXIF Orientation 6 (it
  displays upright), a 16-bit grey PNG, and a lossless photo-like PNG (the smaller copy's R1).
  ``run --json`` exits 0 and writes the lossless file, and its record offers a smaller copy
  (nothing is asked, §12.4). The output keeps ``/Rotate 0``, the JPEG's bytes, every PNG sample,
  and its placeholder words are searchable. ``apply <id> --choice smaller`` then writes
  ``… (smaller).pdf`` beside it, and a second run on the output finds its own layer and writes
  nothing.
* **Copies and refusals (§10.1):** one file per row of the table, each a picture of text: the
  provenance answers, a pyHanko signature, a certification, a document timestamp, usage rights,
  PDF/A, PDF/UA, an edit lock, an open password a saved list opens, and one it doesn't (skipped,
  exit 0). Each copy lands beside its original as ``<stem> (searchable copy).pdf`` and the
  original is unchanged. XFA, unreadable XMP, a damaged file and two detectors that disagree exit
  4 with nothing written.

Without the English data wheel (``scripts/dev-install.sh`` installs it) the file skips, except on
CI, where a missing base pack fails it. Every text is made up and on a neutral topic.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import struct
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pikepdf")
pytest.importorskip("pypdfium2")

from fixtures import build  # noqa: E402
from fixtures import refusal_shells as rs  # noqa: E402

TIMEOUT_NOTE = "a run reads every page with the real engines; allow minutes, not seconds"

WATER = ["Water covers most of the surface of the planet,",
         "and rivers carry it from the hills to the sea.",
         "The sample reference 4521 is a made-up number."]
SUN = ["The Sun rises in the east and sets in the west,",
       "and its light reaches us in about eight minutes.",
       "Plants use that light to turn water into sugar."]
MOON = ["The Moon turns once on its axis each time it",
        "goes around the Earth, so one face looks at us."]
SAVED_PASSWORD = "made up garden phrase"       # invented; opens one fixture below
OTHER_PASSWORD = "made up river phrase"        # invented; no saved list holds it
PAGE_DPI = 200


# --------------------------------------------------------------------------- the environment

def _require_base_pack() -> Any:
    """The installed English base pack, checked by the store itself; a skip (a failure on CI) without it."""
    from pdfintake import platforms
    from pdfintake.errors import PdfIntakeError
    from pdfintake.langpacks import store

    plat = platforms.current()
    if not plat.supported:
        pytest.skip(f"{plat.key} isn't a supported platform")
    try:
        return store.base()
    except PdfIntakeError as exc:
        why = (f"the English base pack (pdfintake-data-en) can't be loaded: {exc.code}: {exc.message.rstrip('.')}. "
               "scripts/dev-install.sh builds and installs it")
        if os.environ.get("CI"):
            pytest.fail(why, pytrace=False)
        pytest.skip(why)


@contextlib.contextmanager
def _network_blocked() -> Iterator[list[tuple[str, str]]]:
    """The root conftest's socket guard for a whole module fixture; yields the refused attempts.

    The autouse guard is per test, and a module fixture is set up before it, so this turns the same
    guard on (with the proxy variables gone, as the autouse fixture does) for the calls made here.
    """
    import conftest

    guard = conftest._GUARD
    with pytest.MonkeyPatch.context() as mp:
        for name in conftest.PROXY_VARS:
            mp.delenv(name, raising=False)
        mp.setenv("NO_PROXY", "*")
        mp.setenv("no_proxy", "*")
        was = guard["on"]
        guard["attempts"] = []
        guard["on"] = True
        try:
            yield guard["attempts"]
        finally:
            guard["on"] = was


def cli(*argv: str) -> tuple[int, dict[str, Any]]:
    """``pdfintake <argv> --json`` in this process: (exit code, the parsed JSON it printed)."""
    from pdfintake import cli as C

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = C.main([*argv, "--json"])
    text = out.getvalue()
    try:
        data = json.loads(text)
    except ValueError:
        raise AssertionError(f"pdfintake {' '.join(argv)} printed no JSON object: {text[:2000]!r}") from None
    return code, data


# --------------------------------------------------------------------------- fixture content, built in code

def text_image(lines: Sequence[str], font_path: Path, *, width: int = 1600, height: int = 1100,
               size: int = 44) -> Any:
    """A grey picture of ``lines`` drawn with the base pack's own font (what a scan of a page looks like)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), size)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for k, text in enumerate(lines):
        draw.text((90, 110 + int(size * 2.3) * k), text, font=font, fill=0)
    return img


def png16(samples: Any, *, dpi: float = PAGE_DPI) -> bytes:
    """A 16-bit grey PNG of ``samples`` (a 2-D uint16 array), written by hand: Pillow can't write
    16-bit grey with every sample kept."""
    import numpy as np

    data = np.asarray(samples).astype(">u2")
    h, w = data.shape
    raw = b"".join(b"\x00" + data[r].tobytes() for r in range(h))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    ppm = round(dpi / 0.0254)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
            + chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1)) + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def sixteen_bit(img: Any) -> Any:
    """The 8-bit text picture spread over 16 bits, with low bits an 8-bit image can't hold."""
    import numpy as np

    g = np.asarray(img, dtype=np.uint16) * 257
    yy, xx = np.indices(g.shape)
    return (g - ((xx + yy) % 5).astype(np.uint16) * (g > 0)).astype(np.uint16)


def photo(width: int = 1600, height: int = 1100, *, seed: int = 3) -> Any:
    """A photo-like RGB picture: soft colour fields and a little grain. Lossless it is large; as a
    JPEG at the smaller copy's qualities it clears the score bar (measured about 91 to 92)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 256, (height // 40, width // 40, 3)).astype(np.uint8)
    soft = Image.fromarray(cells, "RGB").resize((width, height), Image.Resampling.BICUBIC)
    arr = np.asarray(soft, dtype=np.float32) + rng.normal(0.0, 0.7, (height, width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def scanned_pdf(lines: Sequence[str], font_path: Path) -> bytes:
    """A one-page PDF whose only content is a picture of ``lines``: a scan pdfintake reads."""
    img = text_image(lines, font_path, width=1300, height=420, size=42)
    w_pt, h_pt = img.width * 72.0 / PAGE_DPI, img.height * 72.0 / PAGE_DPI
    return build.pdf_bytes([build.PageSpec(width=w_pt, height=h_pt, images=(build.ImageSpec(img),))])


# --------------------------------------------------------------------------- reading results

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Seal:
    sha256: str
    size: int
    mtime_ns: int


def seal(path: Path) -> Seal:
    st = path.stat()
    return Seal(sha256(path), st.st_size, st.st_mtime_ns)


def listing(folder: Path) -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}


def pdf_text(path: Path) -> list[str]:
    """Each page's text as a reader's search sees it (pdfium, inside the CropBox)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            try:
                out.append(tp.get_text_bounded())
            finally:
                tp.close()
                page.close()
        return out
    finally:
        doc.close()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def word_recall(truth: Sequence[str], text: str) -> float:
    want = [w for w in words(" ".join(truth)) if len(w) >= 3 or w.isdigit()]
    got = set(words(text))
    return sum(w in got for w in want) / len(want)


def codes(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [i["code"] for i in items]


def only_record(data: Mapping[str, Any]) -> dict[str, Any]:
    recs = data["records"]
    assert len(recs) == 1, [r.get("input") for r in recs]
    return recs[0]


def record_for(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    found = [r for r in data["records"] if any(Path(p).name == name for p in r["input"]["paths"])]
    assert len(found) == 1, f"no single record for {name}"
    return found[0]


def retention_of(rec: Mapping[str, Any]) -> float:
    check = next(c for c in rec["verify"] if c["name"] == "text_retention" and c.get("page") is None)
    m = re.match(r"(\d+\.\d+) document", check["detail"])
    assert m, check
    return float(m.group(1))


# --------------------------------------------------------------------------- the image batch

@dataclass
class Batch:
    folder: Path
    inputs: dict[str, Path]
    sources: dict[str, Any]
    seals_before: dict[str, Seal]
    first: tuple[int, dict[str, Any]]
    after_first: dict[str, str]
    apply: tuple[int, dict[str, Any]]
    after_apply: dict[str, str]
    second: tuple[int, dict[str, Any]]
    after_second: dict[str, str]
    seals_after: dict[str, Seal]
    attempts: list[tuple[str, str]] = field(default_factory=list)


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> Batch:
    import numpy as np

    base = _require_base_pack()
    font = base.asset_path("noto-sans")
    folder = tmp_path_factory.mktemp("e2e-batch")

    upright = text_image(WATER, font).convert("RGB")
    jpeg = build.jpeg_bytes(upright.rotate(90, expand=True), quality=90, orientation=6)   # displays upright
    grey16 = sixteen_bit(text_image(SUN, font))
    picture = photo()
    inputs = {"jpeg": folder / "a water page.jpg", "png16": folder / "b sun page.png",
              "photo": folder / "c photo.png"}
    inputs["jpeg"].write_bytes(jpeg)
    inputs["png16"].write_bytes(png16(grey16))
    inputs["photo"].write_bytes(build.png_bytes(picture, dpi=PAGE_DPI))
    sources = {"jpeg": jpeg, "png16": np.asarray(grey16, dtype=">u2").tobytes(), "photo": picture.tobytes()}
    before = {k: seal(p) for k, p in inputs.items()}

    with _network_blocked() as attempts:
        first = cli("run", *(str(p) for p in inputs.values()))
        after_first = listing(folder)
        rec = only_record(first[1])
        applied = cli("apply", rec["id"], "--choice", "smaller")
        after_apply = listing(folder)
        out = Path(rec["output"]["path"]) if rec.get("output") else folder / "missing.pdf"
        second = cli("run", str(out))
        after_second = listing(folder)
        found = list(attempts)
    return Batch(folder, inputs, sources, before, first, after_first, applied, after_apply, second, after_second,
                 {k: seal(p) for k, p in inputs.items()}, found)


def test_the_first_run_exits_0_and_offers_a_smaller_copy(batch: Batch):
    code, data = batch.first
    rec = only_record(data)
    assert code == 0 and data["exit"] == 0, rec["steps"]
    assert rec["status"] == "done" and rec["exit"] == 0
    assert rec["provenance"] == "ordinary"
    assert "decision" not in rec                              # nothing is asked (DESIGN §12)
    out = Path(rec["output"]["path"])
    assert out.parent == batch.folder and out.name == "a water page (searchable).pdf"
    assert rec["output"]["kind"] == "searchable" and out.is_file()
    assert sha256(out) == rec["output"]["sha256"]
    # The lossless file only: the smaller copy is offered, never written by a plain run (§12.4).
    assert set(batch.after_first) == {p.name for p in batch.inputs.values()} | {out.name}
    smaller = [o for o in rec["offers"] if o["kind"] == "smaller"]
    assert len(smaller) == 1, rec["offers"]
    assert smaller[0]["apply_flag"] == "--choice"
    assert [o["id"] for o in smaller[0]["options"]] == ["smaller"]
    assert rec["smaller"]["offered"] is True and rec["smaller"]["saving"]["pct"] >= 30
    assert rec["smaller"]["min_score"] >= 90
    assert not [o for o in rec["offers"] if o["kind"] == "language"], rec["offers"]


def test_the_output_is_upright_and_keeps_every_image_exactly(batch: Batch):
    import pikepdf

    rec = only_record(batch.first[1])
    out = Path(rec["output"]["path"])
    raw_dct: list[bytes] = []
    samples: dict[tuple[int, int], list[bytes]] = {}
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 3
        for page in pdf.pages:
            assert int(page.obj.get("/Rotate", 0)) == 0
            xobjects = page.obj.Resources.get("/XObject", pikepdf.Dictionary())
            for _name, obj in xobjects.items():
                if obj.get("/Subtype") != pikepdf.Name.Image:
                    continue
                if obj.get("/Filter") == pikepdf.Name.DCTDecode:
                    raw_dct.append(obj.read_raw_bytes())
                else:
                    samples.setdefault((int(obj.BitsPerComponent), int(obj.Width)), []).append(obj.read_bytes())
        # Orientation 6 turns the stored portrait picture into a landscape page.
        box = [float(v) for v in pdf.pages[0].mediabox]
        assert box[2] - box[0] > box[3] - box[1]
    assert raw_dct == [batch.sources["jpeg"]], "the JPEG's bytes aren't in the PDF unchanged"
    assert samples[(16, 1600)] == [batch.sources["png16"]], "the 16-bit PNG's samples changed"
    assert samples[(8, 1600)] == [batch.sources["photo"]], "the photo PNG's samples changed"
    for page in rec["pages"]:
        assert page["orientation"]["turn_cw"] == 0, page


def test_the_placeholder_words_are_searchable_and_retained(batch: Batch):
    rec = only_record(batch.first[1])
    pages = pdf_text(Path(rec["output"]["path"]))
    assert word_recall(WATER, pages[0]) >= 0.95, pages[0]
    assert word_recall(SUN, pages[1]) >= 0.95, pages[1]
    assert "4521" in pages[0]
    assert retention_of(rec) >= 0.98
    assert rec["verify"] and all(c["ok"] for c in rec["verify"]), [c for c in rec["verify"] if not c["ok"]]
    names = {c["name"] for c in rec["verify"]}
    assert {"text_retention", "renders_identical", "images_identical", "passthrough_identical", "upright"} <= names
    assert rec["language"]["packs"] and {p["id"] for p in rec["language"]["packs"]} == {"en"}
    assert not any(p["fetched"] for p in rec["language"]["packs"])


def test_the_inputs_are_never_touched(batch: Batch):
    assert batch.seals_after == batch.seals_before
    for rec in [only_record(batch.first[1]), only_record(batch.second[1])]:
        assert "input_unchanged" in codes(rec["notes"])


def test_apply_choice_smaller_writes_the_smaller_copy_beside_the_output(batch: Batch):
    code, data = batch.apply
    rec = only_record(data)
    first = only_record(batch.first[1])
    assert code == 0 and data["exit"] == 0, rec["steps"]
    assert rec["id"] == first["id"] and rec["status"] == "resolved"
    out = Path(first["output"]["path"])
    beside = out.with_name("a water page (searchable) (smaller).pdf")
    assert rec["smaller"]["applied"] is True and Path(rec["smaller"]["written"]) == beside
    assert set(batch.after_apply) == set(batch.after_first) | {beside.name}
    assert batch.after_apply[out.name] == batch.after_first[out.name]          # the lossless file is kept
    assert beside.stat().st_size < out.stat().st_size * 0.7
    assert sha256(beside) == first["smaller"]["candidate"]["sha256"]
    assert not [o for o in rec["offers"] if o["kind"] == "smaller"]
    assert pdf_text(beside) == pdf_text(out)                                   # the same text layer


def test_a_second_run_on_the_output_finds_its_own_layer_and_writes_nothing(batch: Batch):
    code, data = batch.second
    rec = only_record(data)
    assert code == 0 and data["exit"] == 0, rec["steps"]
    assert rec["status"] == "done" and rec.get("output") is None
    assert "nothing_written" in codes(rec["notes"])
    kinds = next(s for s in rec["steps"] if s["step"] == "classify")["detail"]["kinds"]
    assert kinds.get("our_layer", 0) >= 2, kinds
    assert batch.after_second == batch.after_apply


def test_nothing_reached_for_the_network(batch: Batch):
    """With every pack the batch needs already installed, no call opened a socket: the offline CI
    job relies on this, running the same file with no network at all."""
    assert batch.attempts == []


# --------------------------------------------------------------------------- §10.1: copies and refusals

@dataclass
class Copies:
    folder: Path
    seals_before: dict[str, Seal]
    ordinary: tuple[int, dict[str, Any]]
    counterparty: tuple[int, dict[str, Any]]
    evidence: tuple[int, dict[str, Any]]
    hard: tuple[int, dict[str, Any]]
    listing_before_hard: dict[str, str]
    listing_after_hard: dict[str, str]
    seals_after: dict[str, Seal]
    attempts: list[tuple[str, str]]


def _signature_variants(data: bytes) -> dict[str, bytes]:
    """The signed, certified and timestamped versions of ``data`` (pyHanko, throwaway keys made at test time)."""
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import signers

    stamped = io.BytesIO()
    signers.PdfTimeStamper(rs._timestamper()).timestamp_pdf(IncrementalPdfFileWriter(io.BytesIO(data)), "sha256",
                                                             output=stamped)
    return {"signed": rs._sign(data, box=(20, 20, 160, 60)), "certified": rs._sign(data, certify=True, p=2),
            "timestamped": stamped.getvalue()}


def _marker_variants(data: bytes) -> dict[str, bytes]:
    import pikepdf

    def ur3(pdf: Any) -> None:
        ref = pikepdf.Array([pikepdf.Dictionary(
            Type=pikepdf.Name.SigRef, TransformMethod=pikepdf.Name.UR3,
            TransformParams=pikepdf.Dictionary(Type=pikepdf.Name.TransformParams, V=pikepdf.Name("/2.2"),
                                               Document=pikepdf.Array([pikepdf.Name.FullSave])))])
        pdf.Root.Perms = pikepdf.Dictionary(UR3=rs._zero_sig(pdf, reference=ref))

    def pdfa(pdf: Any) -> None:
        pdf.Root.Metadata = rs._metadata(pdf, rs.XMP_PDFA_2B)

    def pdfua(pdf: Any) -> None:
        pdf.Root.Metadata = rs._metadata(pdf, rs.XMP_PDFUA_1)
        pdf.Root.MarkInfo = pikepdf.Dictionary(Marked=True)
        pdf.Root.StructTreeRoot = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.StructTreeRoot,
                                                                       K=pikepdf.Array([])))
        pdf.Root.Lang = pikepdf.String("en")

    keep = rs._keep_streams()
    return {"usage_rights": rs._with(data, ur3), "pdfa": rs._with(data, pdfa, **keep),
            "pdfua": rs._with(data, pdfua, **keep),
            "edit_lock": rs.encrypt(data, user=""),
            "open_password": rs.encrypt(data, user=SAVED_PASSWORD),
            "unknown_password": rs.encrypt(data, user=OTHER_PASSWORD)}


@pytest.fixture(scope="module")
def copies(tmp_path_factory: pytest.TempPathFactory) -> Copies:
    pytest.importorskip("pyhanko")
    base = _require_base_pack()
    font = base.asset_path("noto-sans")
    folder = tmp_path_factory.mktemp("e2e-copies")
    plain = scanned_pdf(MOON, font)

    copyable = {**_signature_variants(plain), **_marker_variants(plain)}
    files = {k: folder / f"{k.replace('_', ' ')} page.pdf" for k in copyable}
    for k, data in copyable.items():
        files[k].write_bytes(data)
    for k in ("counterparty", "evidence"):
        files[k] = folder / f"{k} page.pdf"
        files[k].write_bytes(plain)
    hard = {"xfa": rs.xfa(), "xmp_unreadable": rs.xmp_unreadable(), "damaged": rs.cut_trailer(plain),
            "detectors_disagree": rs.orphan_byterange()}
    for k, data in hard.items():
        files[k] = folder / f"{k.replace('_', ' ')}.pdf"
        files[k].write_bytes(data)
    saved = folder / "saved-passwords.txt"
    saved.write_text(f"{OTHER_PASSWORD[::-1]}\n{SAVED_PASSWORD}\n", encoding="utf-8")
    os.chmod(saved, 0o600)
    before = {k: seal(p) for k, p in files.items()}

    with _network_blocked() as attempts:
        ordinary = cli("run", *(str(files[k]) for k in copyable), "--passwords-file", str(saved))
        counterparty = cli("run", str(files["counterparty"]), "--provenance", "counterparty")
        evidence = cli("run", str(files["evidence"]), "--provenance", "evidence")
        listing_before = listing(folder)
        refused = cli("run", *(str(files[k]) for k in hard))
        listing_after = listing(folder)
        found = list(attempts)
    return Copies(folder, before, ordinary, counterparty, evidence, refused, listing_before, listing_after,
                  {k: seal(p) for k, p in files.items()}, found)


COPY_ROWS = {
    # fixture key: (file name, the copy reason, the notes the copy carries)
    "signed": ("signed page.pdf", "signed", {"signature_not_carried"}),
    "certified": ("certified page.pdf", "certified", {"signature_not_carried"}),
    "timestamped": ("timestamped page.pdf", "timestamped", {"signature_not_carried"}),
    "usage_rights": ("usage rights page.pdf", "usage_rights", {"signature_not_carried"}),
    "pdfa": ("pdfa page.pdf", "pdfa", {"conformance_claim_dropped"}),
    "pdfua": ("pdfua page.pdf", "pdfua_needs_layer", {"conformance_claim_dropped"}),
    "edit_lock": ("edit lock page.pdf", "encrypted", {"copy_unlocked"}),
    "open_password": ("open password page.pdf", "encrypted", {"copy_unlocked"}),
}


def _assert_copy(rec: Mapping[str, Any], name: str, reason: str, notes: set[str], before: Seal) -> Path:
    import pikepdf

    assert rec["status"] == "done" and rec["exit"] == 0, (name, rec["steps"])
    out = Path(rec["output"]["path"])
    assert out.name == f"{Path(name).stem} (searchable copy).pdf" and out.is_file(), out
    assert rec["output"]["kind"] == "searchable_copy"
    assert reason in rec["output"]["copy_reasons"], rec["output"]
    assert rec["output"]["original"]["name"] == name and rec["output"]["original"]["sha256"] == before.sha256
    assert notes <= set(codes(rec["notes"])), (name, codes(rec["notes"]))
    assert all(c["ok"] for c in rec["verify"]), [c for c in rec["verify"] if not c["ok"]]
    assert not [o for o in rec["offers"] if o["kind"] == "smaller"]           # a copy gets no smaller version
    assert word_recall(MOON, " ".join(pdf_text(out))) >= 0.9
    with pikepdf.open(out) as pdf:
        assert not pdf.is_encrypted
        assert "/Perms" not in pdf.Root
        acro = pdf.Root.get("/AcroForm")
        fields = list(acro.get("/Fields", [])) if acro is not None else []
        assert not [f for f in fields if f.get("/FT") == pikepdf.Name.Sig]
        xmp = bytes(pdf.Root.Metadata.read_bytes()) if "/Metadata" in pdf.Root else b""
        assert b"DerivedFrom" in xmp and before.sha256.encode() in xmp
        assert b"pdfaid:" not in xmp and b"pdfuaid:" not in xmp
    return out


@pytest.mark.parametrize("key", sorted(COPY_ROWS))
def test_each_copyable_file_gets_a_searchable_copy_beside_it(copies: Copies, key: str):
    name, reason, notes = COPY_ROWS[key]
    code, data = copies.ordinary
    _assert_copy(record_for(data, name), name, reason, notes, copies.seals_before[key])
    assert copies.seals_after[key] == copies.seals_before[key]


def test_a_file_no_saved_password_opens_is_skipped_and_the_batch_exits_0(copies: Copies):
    code, data = copies.ordinary
    assert code == 0 and data["exit"] == 0, [(r["input"]["paths"], r["exit"]) for r in data["records"]]
    rec = record_for(data, "unknown password page.pdf")
    assert rec["exit"] == 0 and rec.get("output") is None
    assert "skipped_no_password" in codes(rec["notes"])
    assert not (copies.folder / "unknown password page (searchable copy).pdf").exists()
    text = json.dumps(data)
    assert SAVED_PASSWORD not in text and OTHER_PASSWORD not in text and OTHER_PASSWORD[::-1] not in text


@pytest.mark.parametrize("key", ["counterparty", "evidence"])
def test_the_provenance_answers_make_a_copy(copies: Copies, key: str):
    code, data = getattr(copies, key)
    assert code == 0 and data["exit"] == 0
    rec = only_record(data)
    assert rec["provenance"] == key
    _assert_copy(rec, f"{key} page.pdf", f"provenance_{key}", set(), copies.seals_before[key])
    assert copies.seals_after[key] == copies.seals_before[key]


def test_hard_codes_exit_4_and_write_nothing(copies: Copies):
    code, data = copies.hard
    assert code == 4 and data["exit"] == 4
    got = {}
    for rec in data["records"]:
        assert rec["status"] == "refused" and rec["exit"] == 4 and rec.get("output") is None, rec["steps"]
        failed = [s for s in rec["steps"] if s["status"] == "failed"]
        got[Path(rec["input"]["paths"][0]).stem] = failed[-1]["detail"]["code"]
    assert got == {"xfa": "xfa", "xmp unreadable": "xmp_unreadable", "damaged": "damaged",
                   "detectors disagree": "detectors_disagree"}
    assert copies.listing_after_hard == copies.listing_before_hard
    for key in ("xfa", "xmp_unreadable", "damaged", "detectors_disagree"):
        assert copies.seals_after[key] == copies.seals_before[key]


def test_no_copy_or_refusal_reached_for_the_network(copies: Copies):
    assert copies.attempts == []
