"""Thorough mode inside a run (DESIGN §14.1, §14.2): pass 2 with a dots read, the record, the offer, apply.

dots is a stand-in here (the fake engines of ``test_pipeline`` and ``test_flow``), handed to the
pipeline as thorough mode's use, so every test runs on every OS without the model. What is checked:

* a Latin pack keeps its pick and takes from dots only the numbers the layer lacks and the words
  in a script the pack doesn't cover, as fills with source ``dots``;
* a ``corroborated_trigrams`` pack lets dots compete, last in the tie order, and only a
  corroborated dots read wins;
* a dots read that fails changes nothing but the page's ``thorough_failed`` warning;
* a run offers the mode while it is undecided, records ``options.thorough`` and the dots reader,
  warns ``thorough_unavailable`` when asked for but unable, and ``apply --choice thorough`` reads
  the document again in the mode.

Every text is made up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import test_flow as tf
import test_pipeline as tp
from pdfintake import flow, record, thorough
from pdfintake.errors import SetupError, UsageError
from pdfintake.langpacks.manifest import EngineUse, ReaderSet, RuleUse
from pdfintake.ocr import pipeline
from pdfintake.ocr.engines import Availability

GREEK = "Αθήνα Θεσσαλονίκη"
fakes = tp.fakes                         # test_pipeline's fake engines, registered, as a fixture here too


def pages(folder: Path) -> list[Any]:
    folder.mkdir(parents=True, exist_ok=True)
    return tp.pages_for(folder, 1)


def dots_use(base: Any, fake: Any, avail: Availability | None = None) -> pipeline._Use:
    """Thorough mode's use with a stand-in engine, as ``pipeline.thorough_use`` builds it."""
    use = EngineUse(pipeline.THOROUGH_USE_ID, "dots", "reader", {}, ())
    return pipeline._Use(base, use, fake, json.dumps(["dots", "test"]), avail or Availability(True, "available", ""),
                         lambda: {})


# --------------------------------------------------------------------------- pass 2

def test_a_latin_pack_takes_only_numbers_and_other_script_words(tmp_path: Path, fakes: dict[str, tp.FakeOcr]):
    root = tmp_path / "packs"
    en, hu = tp.base_pack(root), tp.hu_pack(root)
    dots = tp.FakeOcr("fk_dots", lambda lang, i: [*tp.HU_TEXT, "Számla 98765 kertészeknek", GREEK])
    [plain] = tp.pipeline.read_pages(pages(tmp_path / "a"), [tp.plan_for(0, "hu")], {"hu": hu}, en, tp.KEY,
                                     signals=None, workers=2)
    [page] = tp.pipeline.read_pages(pages(tmp_path / "b"), [tp.plan_for(0, "hu")], {"hu": hu}, en, tp.KEY,
                                    signals=None, workers=2, thorough=dots_use(en, dots))
    assert [ln.text for ln in page.pick.lines] == [ln.text for ln in plain.pick.lines]
    assert page.pick.chosen == plain.pick.chosen
    mine = [(f.kind, f.text) for f in page.fills if f.source == "dots"]
    assert ("number", "98765") in mine
    assert ("word", GREEK) in mine
    assert not any("kertészeknek" in t for _, t in mine)        # a Latin word the pack reads: never
    assert [f for f in page.fills if f.source != "dots"] == list(plain.fills)
    [d] = [r for r in page.reads if r.use_id == "dots"]
    assert d.role == "checker" and d.status == "ok"
    assert "thorough_failed" not in page.warnings


def thai_corroborated(root: Path) -> Any:
    rs = ReaderSet((tp.use("vision", "fk_va"), tp.use("tesseract", "fk_tb"),
                    tp.use("rapidocr", "fk_rc", "checker")),
                   RuleUse("corroborated_trigrams", 1, {"script": "Thai", "tie_order": ["vision", "tesseract"],
                                                        "corroborators": ["rapidocr"]}), "none")
    return tp.make_pack(root, "th", {"default": rs}, scripts=("Thai", "Latin"), osd_scripts=("Thai",),
                        normalize=("nfc", "thai_sara_am"))


@pytest.fixture
def thai_fakes(monkeypatch: pytest.MonkeyPatch, fakes: dict[str, tp.FakeOcr]) -> dict[str, tp.FakeOcr]:
    right = list(tp.THAI_TEXT)
    out = {"fk_va": tp.FakeOcr("fk_va", lambda lang, i: [t[:6] + "กกกก" for t in right]),
           "fk_tb": tp.FakeOcr("fk_tb", lambda lang, i: [t[:4] for t in right]),
           "fk_rc": tp.FakeOcr("fk_rc", lambda lang, i: right)}
    tp.register(monkeypatch, *out.values())
    return out


def test_the_word_fill_adds_every_occurrence_the_layer_lacks():
    """Measured with the words dots read less the layer's, counted: a sentence printed twice goes in twice,
    each at its own line; occurrences the chosen text already has are taken off first."""
    from pdfintake.ocr import fills
    from pdfintake.types import EngineRead, Line

    box = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
    box2 = ((0.0, 9.0), (10.0, 9.0), (10.0, 14.0), (0.0, 14.0))
    dots = EngineRead("dots", "dots", "dots", "checker", "ok",
                      [Line("Ένα παράδειγμα κειμένου", 1.0, box), Line("Ένα παράδειγμα κειμένου", 1.0, box2),
                       Line("Ein Beispiel", 1.0, box)], 1.0)
    chosen = [Line("Ein Beispel mit κειμένου", 0.9, box)]
    out = fills.script_words(dots, chosen, frozenset({"Latin"}))
    assert [(f.text, f.box) for f in out] == [("Ένα παράδειγμα", box), ("Ένα παράδειγμα κειμένου", box2)]
    assert all(f.kind == "word" and f.source == "dots" for f in out)


def test_the_number_fill_adds_numbers_with_a_run_the_layer_lacks():
    """Counted, every length, each number as dots wrote it: "4,5" read as "45" comes back; a number the
    layer has as often as dots does doesn't."""
    from pdfintake.ocr import fills
    from pdfintake.types import EngineRead, Line

    box = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
    box2 = ((0.0, 9.0), (10.0, 9.0), (10.0, 14.0), (0.0, 14.0))
    dots = EngineRead("dots", "dots", "dots", "checker", "ok",
                      [Line("Rund 4,5 Mio. Tonnen im Jahr 2024", 1.0, box), Line("Seite 2024 von 12", 1.0, box2)], 1.0)
    chosen = [Line("Rund 45 Mio. Tonnen im Jahr 2024, Seite 12", 0.9, box)]
    out = fills.missing_numbers(dots, chosen)
    assert [(f.kind, f.text, f.box) for f in out] == [("number", "4,5", box), ("number", "2024", box2)]
    assert fills.missing_numbers(dots, [*chosen, Line("4 5 2024", 0.9, box)]) == []


def test_a_corroborated_pack_lets_dots_compete(tmp_path: Path, thai_fakes: dict[str, tp.FakeOcr]):
    root = tmp_path / "packs"
    en, th = tp.base_pack(root), thai_corroborated(root)
    good = tp.FakeOcr("fk_dots", lambda lang, i: list(tp.THAI_TEXT))
    [page] = tp.pipeline.read_pages(tp.pages_for(tmp_path, 1), [tp.plan_for(0, "th", script="Thai")], {"th": th}, en,
                                    tp.KEY, signals=None, workers=2, thorough=dots_use(en, good))
    assert page.pick.chosen == "dots" and page.pick.rule == "corroborated_trigrams"
    [d] = [r for r in page.reads if r.use_id == "dots"]
    assert d.role == "reader"
    assert not [f for f in page.fills if f.source == "dots"]     # it competes; it doesn't fill


def test_an_uncorroborated_dots_read_doesnt_win(tmp_path: Path, thai_fakes: dict[str, tp.FakeOcr]):
    root = tmp_path / "packs"
    en, th = tp.base_pack(root), thai_corroborated(root)
    junk = tp.FakeOcr("fk_dots", lambda lang, i: ["ฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟฟ"])
    [page] = tp.pipeline.read_pages(tp.pages_for(tmp_path, 1), [tp.plan_for(0, "th", script="Thai")], {"th": th}, en,
                                    tp.KEY, signals=None, workers=2, thorough=dots_use(en, junk))
    assert page.pick.chosen == "vision"


def test_a_tie_goes_to_the_packs_readers(tmp_path: Path, thai_fakes: dict[str, tp.FakeOcr]):
    root = tmp_path / "packs"
    en, th = tp.base_pack(root), thai_corroborated(root)
    same = tp.FakeOcr("fk_dots", thai_fakes["fk_va"].script)
    [page] = tp.pipeline.read_pages(tp.pages_for(tmp_path, 1), [tp.plan_for(0, "th", script="Thai")], {"th": th}, en,
                                    tp.KEY, signals=None, workers=2, thorough=dots_use(en, same))
    assert page.pick.chosen == "vision"


def test_a_latin_pack_with_the_corroborated_rule_still_takes_fills(tmp_path: Path, thai_fakes: dict[str, tp.FakeOcr]):
    """de and vi pick with corroborated_trigrams over Latin: dots fills there, it never competes."""
    root = tmp_path / "packs"
    en = tp.base_pack(root)
    rs = ReaderSet((tp.use("tesseract", "fk_va"), tp.use("rapidocr", "fk_tb")),
                   RuleUse("corroborated_trigrams", 1, {"script": "Latin", "tie_order": ["tesseract", "rapidocr"]}),
                   "none")
    xl = tp.make_pack(root, "xl", {"default": rs})
    thai_fakes["fk_va"].script = lambda lang, i: list(tp.HU_TEXT)
    thai_fakes["fk_tb"].script = lambda lang, i: list(tp.HU_TEXT)
    dots = tp.FakeOcr("fk_dots", lambda lang, i: [*tp.HU_TEXT, "Ref 98765", GREEK])
    [page] = tp.pipeline.read_pages(pages(tmp_path / "p"), [tp.plan_for(0, "xl")], {"xl": xl}, en, tp.KEY,
                                    signals=None, workers=2, thorough=dots_use(en, dots))
    assert page.pick.chosen == "tesseract"
    [d] = [r for r in page.reads if r.use_id == "dots"]
    assert d.role == "checker"
    mine = {(f.kind, f.text) for f in page.fills if f.source == "dots"}
    assert ("number", "98765") in mine and ("word", GREEK) in mine


@pytest.mark.parametrize("outcome", ["failed", "timeout"])
def test_a_failed_dots_read_changes_nothing_but_its_warning(tmp_path: Path, fakes: dict[str, tp.FakeOcr],
                                                             outcome: str):
    root = tmp_path / "packs"
    en, hu = tp.base_pack(root), tp.hu_pack(root)
    broken = tp.FakeOcr("fk_dots", lambda lang, i: outcome)
    [plain] = tp.pipeline.read_pages(pages(tmp_path / "a"), [tp.plan_for(0, "hu")], {"hu": hu}, en, tp.KEY,
                                     signals=None, workers=1)
    [page] = tp.pipeline.read_pages(pages(tmp_path / "b"), [tp.plan_for(0, "hu")], {"hu": hu}, en, tp.KEY,
                                    signals=None, workers=1, thorough=dots_use(en, broken))
    assert [ln.text for ln in page.pick.lines] == [ln.text for ln in plain.pick.lines]
    assert list(page.fills) == list(plain.fills)
    assert "thorough_failed" in page.warnings and "reader_failed" not in page.warnings
    assert "reader_timeout" not in page.warnings


# --------------------------------------------------------------------------- a run and its record

@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tf.Env:
    return tf.make_env(tmp_path, monkeypatch)


@pytest.fixture
def stand_in(env: tf.Env, monkeypatch: pytest.MonkeyPatch) -> tf.FakeOcr:
    """Thorough mode set up, with a stand-in dots that also reads an order number."""
    dots = tf.FakeOcr("fk_dots", lambda lang, page: [*tf.EN_LINES, "Order 98765 shipped"])
    monkeypatch.setattr(thorough, "resolve", lambda requested: (requested is not False, None))
    monkeypatch.setattr(pipeline, "thorough_use", lambda key: dots_use(env.base, dots))
    return dots


def test_an_undecided_mac_gets_the_offer(env: tf.Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flow, "_thorough_offered", lambda: True)
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")])[0]
    tf.assert_record_ok(rec)
    [offer] = [o for o in rec.offers if o.kind == "thorough"]
    assert offer.apply_flag == "--choice" and [o.id for o in offer.options] == ["thorough"]
    assert "4.4 GB" in offer.summary and "Hugging Face" in offer.summary
    text = flow.run([tf.text_pdf(tmp_path / "Printed.pdf")])[0]
    assert not any(o.kind == "thorough" for o in text.offers)        # no page was read: nothing to offer


def test_no_offer_unless_the_mac_is_undecided(env: tf.Env, tmp_path: Path):
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")])[0]           # conftest: _thorough_offered is False
    assert not any(o.kind == "thorough" for o in rec.offers)


def test_a_thorough_run_records_the_dots_reader(env: tf.Env, tmp_path: Path, stand_in: tf.FakeOcr,
                                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flow, "_thorough_offered", lambda: True)
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")], flow.RunOptions(thorough=True))[0]
    data = tf.assert_record_ok(rec)
    assert rec.exit == 0 and data["options"]["thorough"] is True
    [step] = [s for s in data["steps"] if s["step"] == "ocr"]
    assert step["detail"]["thorough"] is True
    readers = data["pages"][0]["readers"]
    assert [r["use_id"] for r in readers][-1] == "dots" and readers[-1]["status"] == "ok"
    assert data["pages"][0]["fills"].get("number", 0) >= 1
    assert "98765" in tf.pdf_text(Path(rec.output["path"]))
    assert not any(o.kind == "thorough" for o in rec.offers)          # it already read this document
    assert stand_in.reads


def test_no_thorough_overrides_the_setting(env: tf.Env, tmp_path: Path, stand_in: tf.FakeOcr):
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")], flow.RunOptions(thorough=False))[0]
    data = tf.assert_record_ok(rec)
    assert data["options"]["thorough"] is False and not stand_in.reads
    assert all(r["use_id"] != "dots" for r in data["pages"][0]["readers"])


def test_asked_for_but_unable_warns_and_reads_without_it(env: tf.Env, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(thorough, "resolve", lambda requested: (False, "thorough mode runs only on a Mac"))
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")], flow.RunOptions(thorough=True))[0]
    tf.assert_record_ok(rec)
    assert rec.exit == 0 and rec.output is not None
    assert [w.code for w in rec.warnings].count("thorough_unavailable") == 1


def test_apply_choice_thorough(env: tf.Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(flow, "_thorough_offered", lambda: True)
    src = tf.image_pdf(tmp_path / "Scan.pdf")
    rec = flow.run([src])[0]
    assert any(o.kind == "thorough" for o in rec.offers)
    monkeypatch.setattr(thorough, "resolve", lambda requested: (False, "the model isn't installed"))
    with pytest.raises(SetupError) as info:
        flow.apply(rec.id, choice="thorough")
    assert info.value.code == "thorough_not_ready" and "pdfintake thorough enable" in info.value.message
    dots = tf.FakeOcr("fk_dots", lambda lang, page: [*tf.EN_LINES, "Order 98765 shipped"])
    monkeypatch.setattr(thorough, "resolve", lambda requested: (requested is not False, None))
    monkeypatch.setattr(pipeline, "thorough_use", lambda key: dots_use(env.base, dots))
    [child] = flow.apply(rec.id, choice="thorough")
    data = tf.assert_record_ok(child)
    assert child.parent == rec.id and child.exit == 0 and data["options"]["thorough"] is True
    assert child.output["path"] == rec.output["path"] and "98765" in tf.pdf_text(Path(child.output["path"]))
    parent = record.load(rec.id)
    assert parent.status == "resolved" and not parent.offers
    assert parent.history[-1]["flag"] == "--choice" and parent.history[-1]["answer"] == "thorough"
    with pytest.raises(UsageError) as again:
        flow.apply(rec.id, choice="thorough")
    assert again.value.code == "already_applied"


def test_the_real_thorough_use_reaches_the_dots_engine(env: tf.Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No stand-in for ``pipeline.thorough_use`` here: it builds the use from the registered dots engine and the
    pinned files. On the fake platform dots isn't available, so a run asked for the mode warns and reads
    without it. (A NameError in this path once failed every page of a real thorough run: 2026-11-10.)"""
    folder = tmp_path / "model"
    folder.mkdir()
    files = {}
    for name in thorough.FILES:
        (folder / name.name).write_bytes(b"made-up")
        files[name.name] = folder / name.name
    monkeypatch.setattr(thorough, "model_paths", lambda: files)
    use = pipeline.thorough_use(tf.KEY)
    assert use is not None and use.use.id == "dots" and use.use.engine == "dots" and use.engine.name == "dots"
    assert use.avail.state == "not_on_platform" and use.models is not None
    monkeypatch.setattr(thorough, "resolve", lambda requested: (True, None))
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")], flow.RunOptions(thorough=True))[0]
    tf.assert_record_ok(rec)
    assert rec.exit == 0 and "thorough_unavailable" in [w.code for w in rec.warnings]


def test_apply_choice_thorough_needs_the_offer(env: tf.Env, tmp_path: Path):
    rec = flow.run([tf.image_pdf(tmp_path / "Scan.pdf")])[0]
    with pytest.raises(UsageError) as info:
        flow.apply(rec.id, choice="thorough")
    assert info.value.code == "no_offer"
