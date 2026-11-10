"""The font-map repair (DESIGN §15): fonts built in code, text made up.

Each PDF draws its text with a Type0 font whose program has no cmap and no glyph names and whose
dictionary has no ``/ToUnicode``, so pdfium reads garbage. The tests check which evidence decides
each glyph, that nothing unproven is guessed, and that the written map reads back while every page
renders exactly as before.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from fixtures import garbled as G
from pdfintake.pdf import fontmap, fontprog, verify
from pdfintake.types import EngineRead, Line

LATIN = "abcdefghijklmnopqrstuvwxyz "
WORDS = ["a quick brown fox jumps", "over the lazy dog", "while sixty zebras quietly wander", "back home"]


def _text(path: Path, page: int = 0) -> str:
    import pypdfium2

    from pdfintake.pdf.render import PDFIUM_LOCK

    with PDFIUM_LOCK:
        doc = pypdfium2.PdfDocument(str(path))
        try:
            tp = doc[page].get_textpage()
            try:
                return tp.get_text_range()
            finally:
                tp.close()
        finally:
            doc.close()


def _apply(src: Path, res: fontmap.FontMapResult, dest: Path) -> Path:
    import pikepdf

    with pikepdf.open(src) as pdf:
        fontmap.write(pdf, res)
        pdf.save(dest)
    return dest


def _refs(folder: Path, *fonts: G.ShapeFont, names: Sequence[str] = ("ExampleSerif-Regular.ttf",)) -> fontprog.ReferenceIndex:
    folder.mkdir(parents=True, exist_ok=True)
    for font, name in zip(fonts, names):
        (folder / name).write_bytes(font.data)
    return fontprog.ReferenceIndex([folder])


def _garbled(tmp_path: Path, texts: Sequence[str], *, chars: str = LATIN, order: str | None = None,
             extra: Sequence[G.FontSpec] = (), pages: Sequence[G.Page] | None = None, **font: Any) -> Path:
    prog = G.shape_font(chars, order=order if order is not None else chars[::-1], cmap=False, names=False, **font)
    spec = G.FontSpec(prog)
    return G.garbled_pdf(tmp_path / "in.pdf", pages or [G.Page(G.lines_of(texts))], [spec, *extra])


# --------------------------------------------------------------------------- fontprog

@pytest.mark.parametrize(("name", "family", "style"), [
    ("ABCDEF+TimesNewRomanPS-BoldMT", "timesnewroman", "timesnewroman-bold"),
    ("TimesNewRoman,Bold", "timesnewroman", "timesnewroman-bold"),
    ("TimesNewRomanPSMT", "timesnewroman", "timesnewroman-regular"),
    ("Times New Roman", "timesnewroman", "timesnewroman-regular"),
    ("MinionPro-Regular", "minionpro", "minionpro-regular"),
    ("/QWERTY+Example-Italic", "example", "example-italic"),
])
def test_family_and_style_keys(name: str, family: str, style: str):
    assert fontprog.family_key(name) == family
    assert fontprog.style_key(name) == style


def test_usable_text_refuses_controls_and_replacement_but_keeps_private_use():
    assert fontprog.usable_text("a") and fontprog.usable_text("fi") and fontprog.usable_text("")
    for bad in ("", "\x02", "�", "￾", "\ud800", "x" * 17):
        assert not fontprog.usable_text(bad)


def test_agl_names():
    assert fontprog.agl_text("a") == "a"
    assert fontprog.agl_text("f_i") == "fi"
    assert fontprog.agl_text("uni0131") == "ı"
    for bad in ("g255", "C0", ".notdef", None, "glyph12"):
        assert fontprog.agl_text(bad) is None


def test_cmap_round_trip_and_ranges():
    data = fontprog.cmap_bytes({1: "a", 0x2F: "fi", 0x30A: "ı"}, two_byte=True)
    assert fontprog.parse_cmap(data) == {1: "a", 0x2F: "fi", 0x30A: "ı"}
    one = fontprog.cmap_bytes({0x41: "A"}, two_byte=False)
    assert b"<00> <FF>" in one and fontprog.parse_cmap(one) == {0x41: "A"}
    ranges = (b"begincmap 2 beginbfrange <0010> <0012> <0061> <0020> <0021> [<0041> <0066 0069>] endbfrange "
              b"1 beginbfchar <0030> <D8> endbfchar endcmap")
    assert fontprog.parse_cmap(ranges) == {0x10: "a", 0x11: "b", 0x12: "c", 0x20: "A", 0x21: "fi", 0x30: None}


@pytest.mark.parametrize("data", [
    b"4 beginbfrange" + b" <0000> <FFFF> <0041>" * 4 + b" endbfrange",   # more entries than the limit
    b"1 beginbfrange <0000> <1FFFF> <0041> endbfrange",                  # one range past the limit
    b"x" * (fontprog.MAX_CMAP_BYTES + 1),
])
def test_cmap_limits(data: bytes):
    with pytest.raises(fontprog.CMapError):
        fontprog.parse_cmap(data)


def test_twins_are_one_step_and_skeleton_folds():
    assert fontprog.TWINS["a"] == {"а"}
    assert "i" not in fontprog.TWINS["ı"] and fontprog.TWINS["ı"] == {"ι"}
    assert fontprog.skeleton("İ") == fontprog.skeleton("ı") == "i"
    assert fontprog.script_of("а") == "CYRILLIC" and fontprog.script_of("ı") == "LATIN"
    assert fontprog.script_of("ก") == "THAI" and fontprog.script_of("่") == "THAI" and fontprog.script_of("1") is None


def test_outline_hash_ignores_numbering():
    one = G.shape_font("abc", order="abc")
    two = G.shape_font("abc", order="cab", cmap=False, names=False)
    p1, p2 = fontprog.load_program(one.data, "FontFile2"), fontprog.load_program(two.data, "FontFile2")
    for c in "abc":
        assert p1.glyph_hash(one.gid[c]) == p2.glyph_hash(two.gid[c]) is not None
    assert len({p1.glyph_hash(one.gid[c]) for c in "abc"}) == 3


def test_a_broken_program_is_refused():
    with pytest.raises(fontprog.ProgramError):
        fontprog.load_program(b"not a font", "FontFile2")
    with pytest.raises(fontprog.ProgramError):
        fontprog.load_program(b"x" * 10, "Type1")


def test_reference_reads_gsub_backwards_but_not_onto_encoded_glyphs(tmp_path: Path):
    font = G.shape_font("fiı", ligatures={"fi": "f_i"}, encoded_variants={"i": "ı"})
    idx = _refs(tmp_path / "fonts", font)
    faces = idx.faces("exampleserif")
    assert faces and faces[0].ps_name == "ExampleSerif-Regular"
    ref = fontprog.load_reference(faces[0])
    assert ref is not None
    assert ref.gid_text[font.gid["fi"]] == {"fi"}                       # the ligature, from GSUB
    assert ref.gid_text[font.gid["ı"]] == {"ı"}                         # dtls makes ı from i: still only ı
    assert fontprog.ReferenceIndex([tmp_path / "nothing"]).faces("exampleserif") == []


# --------------------------------------------------------------------------- describe

def _font_dicts(path: Path) -> list[fontmap.Font]:
    import pikepdf

    with pikepdf.open(path) as pdf:
        return [fontmap.describe(f) for f in fontmap._font_dicts(pdf)]


def test_scope_of_a_type0_font(tmp_path: Path):
    src = _garbled(tmp_path, ["abc"])
    (f,) = _font_dicts(src)
    assert f.scope is None and f.two_byte and not f.existing and f.family == "exampleserif"
    no_map = G.garbled_pdf(tmp_path / "nomap.pdf", [G.Page(G.lines_of(["abc"]))],
                           [G.FontSpec(G.shape_font("abc", cmap=False, names=False), cidtogid=None)])
    assert _font_dicts(no_map)[0].scope == "cidtogid_missing"


def test_scope_of_other_fonts(tmp_path: Path):
    import pikepdf

    pdf = pikepdf.new()
    t3 = pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type3)
    plain = pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica)
    other = pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type0, BaseFont=pikepdf.Name.X,
                               Encoding=pikepdf.Name("/90ms-RKSJ-H"), DescendantFonts=[pikepdf.Dictionary()])
    scopes = [fontmap.describe(pdf.make_indirect(d)).scope for d in (t3, plain, other)]
    assert scopes == ["type3", "not_embedded", "encoding_not_identity"]


# --------------------------------------------------------------------------- the gate and the probe

def test_a_pdf_pdfium_reads_needs_nothing(tmp_path: Path):
    font = G.shape_font(LATIN)
    src = G.garbled_pdf(tmp_path / "ok.pdf", [G.Page(G.lines_of(WORDS))],
                        [G.FontSpec(font, tounicode={c: c for c in LATIN})])
    res, a = fontmap.repair(src, tmp_path / "w", references=None)
    assert not a.fonts and not a.glyphs and not res.changed      # no map error: no font is even read


def test_the_probe_finds_every_glyph(tmp_path: Path):
    src = _garbled(tmp_path, WORDS)
    a = fontmap.analyse(src, tmp_path / "w")
    drawn = sum(len(w.replace(" ", "")) + w.count(" ") for w in WORDS)
    assert len(a.glyphs) == drawn and all(g.visible for g in a.glyphs)
    assert all(g.map_error for g in a.glyphs if g.code != 0)
    assert {(g.font, g.code) for g in a.glyphs} == a.needs()


# --------------------------------------------------------------------------- evidence 1-3

def test_an_installed_font_decides_every_glyph(tmp_path: Path):
    src = _garbled(tmp_path, WORDS)
    refs = _refs(tmp_path / "fonts", G.shape_font(LATIN))
    res, a = fontmap.repair(src, tmp_path / "w", references=refs)
    assert not a.open_codes()
    assert res.codes.get("installed", 0) == len(set(LATIN) - {" "}) and res.codes.get("program") == 1   # the space
    out = _apply(src, res, tmp_path / "out.pdf")
    assert all(w in _text(out) for w in WORDS)
    assert res.pages_repaired == [0] and not res.pages_partial


def test_kept_numbering_is_decided_by_glyph_id(tmp_path: Path):
    """A subset that keeps the reference's glyph numbers is decided by GID, which settles a pair of
    letters sharing one outline even with no word context."""
    chars = "bcdeaа "
    ref = G.shape_font(chars, same_shape={"а": "a"})
    sub = G.shape_font(chars, cmap=False, names=False, same_shape={"а": "a"})   # same order: numbering kept
    src = G.garbled_pdf(tmp_path / "in.pdf", [G.Page(G.lines_of(["bcde", "a", "а"]))], [G.FontSpec(sub)])
    res, a = fontmap.repair(src, tmp_path / "w", references=_refs(tmp_path / "fonts", ref))
    texts = {a.fonts[k[0]].code_gid[k[1]]: d.text for k, d in a.decided.items()}
    assert texts[sub.gid["a"]] == "a" and texts[sub.gid["а"]] == "а"
    assert all(d.evidence == "installed" for d in a.decided.values() if d.text.strip())


def test_the_document_s_own_mapped_font_decides(tmp_path: Path):
    prog = G.shape_font(LATIN, cmap=False, names=False)
    mapped = G.FontSpec(prog, base="QWERTY+ExampleSerif-Regular", tounicode={c: c for c in "abcdefghij "})
    bare = G.FontSpec(prog)
    page = G.Page([[G.Run(1, "abcdefghij")], [G.Run(0, "a bad cafe")]])
    src = G.garbled_pdf(tmp_path / "in.pdf", [page], [bare, mapped])
    res, a = fontmap.repair(src, tmp_path / "w", references=None)
    assert res.codes.get("document", 0) >= 5 and not a.open_codes()
    assert "a bad cafe" in _text(_apply(src, res, tmp_path / "out.pdf"))


def test_the_program_s_own_names_decide(tmp_path: Path):
    prog = G.shape_font(LATIN, order=LATIN[::-1], cmap=False, names=True)
    src = G.garbled_pdf(tmp_path / "in.pdf", [G.Page(G.lines_of(WORDS))], [G.FontSpec(prog)])
    res, a = fontmap.repair(src, tmp_path / "w", references=None)
    if not a.glyphs:
        pytest.skip("pdfium read this program's glyph names itself")
    assert res.codes.get("program", 0) >= 20 and not a.open_codes()


# --------------------------------------------------------------------------- lookalikes

LOOK = "abdбдгаrst "


def _lookalike_pdf(tmp_path: Path, lines: Sequence[str]) -> tuple[Path, G.ShapeFont]:
    sub = G.shape_font(LOOK, order=LOOK[::-1], cmap=False, names=False, same_shape={"а": "a"})
    src = G.garbled_pdf(tmp_path / "in.pdf", [G.Page(G.lines_of(lines))], [G.FontSpec(sub)])
    return src, sub


def test_lookalikes_follow_their_words(tmp_path: Path):
    src, sub = _lookalike_pdf(tmp_path, ["bad dab", "бад габ", "rats star"])
    ref = G.shape_font(LOOK, same_shape={"а": "a"})
    res, a = fontmap.repair(src, tmp_path / "w", references=_refs(tmp_path / "fonts", ref))
    texts = {a.fonts[k[0]].code_gid[k[1]]: d.text for k, d in a.decided.items()}
    assert texts[sub.gid["a"]] == "a" and texts[sub.gid["а"]] == "а"
    out = _text(_apply(src, res, tmp_path / "out.pdf"))
    assert "bad dab" in out and "бад габ" in out


def test_a_lookalike_with_no_context_is_unproven(tmp_path: Path):
    src, sub = _lookalike_pdf(tmp_path, ["bad", "а"])                 # the Cyrillic а stands alone
    ref = G.shape_font(LOOK, same_shape={"а": "a"})
    res, a = fontmap.repair(src, tmp_path / "w", references=_refs(tmp_path / "fonts", ref))
    key = next(k for k in a.needs() if a.fonts[k[0]].code_gid[k[1]] == sub.gid["а"])
    assert key not in a.decided
    out = _text(_apply(src, res, tmp_path / "out.pdf"))
    assert "�" in out and "bad" in out
    assert res.pages_partial == [0] and res.codes.get("unproven") == 1


def test_settle():
    assert fontmap.settle({"a"}) == (None, {"a", "а"})
    assert fontmap.settle({"b"})[0] == "b"
    assert fontmap.settle({"-", "­"})[0] == "-"
    assert fontmap.settle({";", ";"})[0] == ";"
    assert fontmap.settle({"ǧ", "ğ"})[0] == "ğ"                       # one script: the lowest code point
    assert fontmap.settle({"ı", "ɩ"})[0] is None                        # ı's twin ι makes it a choice of script
    assert fontmap.settle({"fi", "ﬁ"})[0] == "fi"                 # the compatibility ligature goes


# --------------------------------------------------------------------------- OCR evidence (made-up reads)

def _reads_for(path: Path, lines: Sequence[str], page_h: float, *, size: float = 24.0, engines: Sequence[str] = ("one", "two"),
               swap: Mapping[str, Mapping[str, str]] | None = None) -> Any:
    """A reader that returns each line's true text (with ``swap`` applied per engine) as a read whose
    line boxes cover the line across the page, in a 300 dpi render's pixels."""
    from pdfintake.types import PageImage

    scale = 300 / 72

    def read(pages: Sequence[int]) -> dict[int, tuple[Any, list[EngineRead]]]:
        out = {}
        for p in pages:
            image = PageImage(p, path, int(612 * scale), int(page_h * scale), 300.0, (0, 0, 612, page_h))
            reads = []
            for eng in engines:
                rl = []
                for k, text in enumerate(lines):
                    y = (page_h - (760 - k * size * 1.6)) * scale
                    x1 = (60 + sum((G.SPACE_ADVANCE if c == " " else G.ADVANCE) for c in text) * size / G.UPM) * scale
                    for src_ch, dst in (swap or {}).get(eng, {}).items():
                        text = text.replace(src_ch, dst)
                    x0 = 60 * scale
                    rl.append(Line(text, 0.9, ((x0, y - size * scale), (x1, y - size * scale),
                                              (x1, y + 0.3 * size * scale), (x0, y + 0.3 * size * scale))))
                reads.append(EngineRead(eng, eng, "x", "reader", "ok", rl, 0.1))
            out[p] = (image, reads)
        return out

    return read


def test_ocr_decides_what_nothing_else_can(tmp_path: Path):
    src = _garbled(tmp_path, WORDS)
    res, a = fontmap.repair(src, tmp_path / "w", references=None, read=_reads_for(src, WORDS, 792))
    sub = G.shape_font(LATIN, order=LATIN[::-1], cmap=False, names=False)
    by_gid = {g: c for c, g in sub.gid.items()}
    wrong = [(by_gid[a.fonts[k[0]].code_gid[k[1]]], d.text) for k, d in a.decided.items()
             if by_gid[a.fonts[k[0]].code_gid[k[1]]] != d.text and not d.text.isspace()]
    assert not wrong
    assert res.codes.get("ocr", 0) >= 20 and res.ocr_pages == [0]


def test_one_engine_s_habit_is_not_proof(tmp_path: Path):
    """Both engines read ``o`` as ``0`` in a document where ``0`` has no other glyph: one engine alone
    would need three occurrences; two engines agreeing is proof. Here only one engine says 0 and the
    other says o: the code stays open."""
    lines = ["foo", "boo"]
    src = _garbled(tmp_path, lines, chars="fob ")
    read = _reads_for(src, lines, 792, swap={"one": {"o": "0"}})
    res, a = fontmap.repair(src, tmp_path / "w", references=None, read=read)
    sub = G.shape_font("fob ", order=" bof", cmap=False, names=False)
    by_gid = {g: c for c, g in sub.gid.items()}
    o_key = next(k for k in a.needs() if by_gid[a.fonts[k[0]].code_gid[k[1]]] == "o")
    assert o_key not in a.decided


def test_a_letter_another_shape_already_has_is_set_aside(tmp_path: Path):
    """A reader that writes o for ö can't map ö to o when the plain o is already known: its votes are set
    aside, and the other engine's ö decides."""
    chars = "boösel "
    prog = G.shape_font(chars, order=chars[::-1], cmap=False, names=False)
    mapped = G.FontSpec(prog, base="QWERTY+ExampleSerif-Regular", tounicode={"o": "o", "s": "s", "e": "e", " ": " "})
    lines = ["böse", "löse", "lös"]                     # three sightings: one engine alone needs three
    page = G.Page([*G.lines_of(lines), [G.Run(1, "so es")]])
    src = G.garbled_pdf(tmp_path / "in.pdf", [page], [G.FontSpec(prog), mapped])
    read = _reads_for(src, [*lines, "so es"], 792, swap={"one": {"ö": "o"}})
    res, a = fontmap.repair(src, tmp_path / "w", references=None, read=read)
    by_gid = {g: c for c, g in prog.gid.items()}
    got = {by_gid[a.fonts[k[0]].code_gid[k[1]]]: d.text for k, d in a.decided.items() if k[0] == 0}
    assert got.get("ö") == "ö"
    assert a.tallies[next(k for k in a.tallies if k[0] == 0 and by_gid[a.fonts[0].code_gid[k[1]]] == "ö")] == {"ö": 3}


# --------------------------------------------------------------------------- writing and verify

def test_the_map_changes_no_pixel_and_reads_back(tmp_path: Path):
    src = _garbled(tmp_path, WORDS)
    res, _a = fontmap.repair(src, tmp_path / "w", references=_refs(tmp_path / "fonts", G.shape_font(LATIN)))
    out = _apply(src, res, tmp_path / "out.pdf")
    assert verify.renders_identical(src, out, {}).ok
    check = verify.fonts_mapped(out, res.expected)
    assert check.ok, check.detail
    quartz = verify.renders_identical_quartz(src, out, set(res.expected))
    assert quartz is None or quartz.ok
    bad = {p: [(box, "Z") for box, _t in items] for p, items in res.expected.items()}
    assert not verify.fonts_mapped(out, bad).ok


def test_only_the_tounicode_key_changes(tmp_path: Path):
    import pikepdf

    src = _garbled(tmp_path, WORDS)
    res, _a = fontmap.repair(src, tmp_path / "w", references=_refs(tmp_path / "fonts", G.shape_font(LATIN)))
    out = _apply(src, res, tmp_path / "out.pdf")
    with pikepdf.open(src) as a, pikepdf.open(out) as b:
        fa = a.pages[0].Resources.Font.F0
        fb = b.pages[0].Resources.Font.F0
        assert set(fb.keys()) - set(fa.keys()) == {"/ToUnicode"}
        assert a.pages[0].Contents.read_bytes() == b.pages[0].Contents.read_bytes()
        da, db = fa.DescendantFonts[0].FontDescriptor.FontFile2, fb.DescendantFonts[0].FontDescriptor.FontFile2
        assert da.read_bytes() == db.read_bytes()


def test_nothing_proven_writes_nothing(tmp_path: Path):
    """Only the space (an empty glyph) is known: no map, and the page stays garbled."""
    src = _garbled(tmp_path, WORDS)
    res, a = fontmap.repair(src, tmp_path / "w", references=None)
    assert not res.changed and res.pages_unrepaired == [0] and res.fonts_left == 1


def test_plan_counts_the_ocr_pages_and_reads_nothing(tmp_path: Path):
    src = _garbled(tmp_path, WORDS)
    called = []
    res, a = fontmap.repair(src, tmp_path / "w", references=None, plan_only=True,
                            read=lambda pages: called.append(pages) or {})
    assert not called and res.ocr_pages == [0]


def test_a_failed_outline_is_never_a_space(tmp_path: Path):
    src = _garbled(tmp_path, ["ab"])
    a = fontmap.analyse(src, tmp_path / "w")
    f = a.fonts[0]
    assert fontmap._hash_of(f, 0) == (None, False)                    # .notdef is never decided
    assert fontmap._hash_of(f, 10_000) == (None, False)


def test_expected_lists_every_repaired_glyph(tmp_path: Path):
    src = _garbled(tmp_path, WORDS)
    res, a = fontmap.repair(src, tmp_path / "w", references=_refs(tmp_path / "fonts", G.shape_font(LATIN)))
    assert sum(len(v) for v in res.expected.values()) == len(a.glyphs)
    assert all(unicodedata.is_normalized("NFC", t) for v in res.expected.values() for _b, t in v)



# --------------------------------------------------------------------------- a run, end to end

@pytest.fixture
def run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A run with a private cache, the installed fonts a test gives, and no OCR evidence unless the test
    supplies a reader (``flow._fontmap_read``)."""
    from pdfintake import flow

    monkeypatch.setenv("PDFINTAKE_CACHE", str(tmp_path / "cache"))
    state: dict[str, Any] = {"refs": fontprog.ReferenceIndex([tmp_path / "no-fonts"]), "read": flow._fontmap_read}
    monkeypatch.setattr(flow, "_font_index", lambda: state["refs"])
    monkeypatch.setattr(flow, "_fontmap_read", lambda d, pages: {})
    monkeypatch.setattr(flow, "_thorough_offered", lambda: False)
    return state


def _run(src: Path, **opts: Any) -> Any:
    from pdfintake import flow, record

    rec = flow.run([src], flow.RunOptions(workers=1, fetch="never", **opts))[0]
    data = record.to_json(rec)
    assert record.validate(data) == []
    return rec, data


def _codes(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    return [(m["code"], m["page"]) for m in items]


def test_a_run_repairs_the_map_and_writes_a_searchable_file(tmp_path: Path, run_env: dict[str, Any]):
    import hashlib

    src = _garbled(tmp_path, WORDS)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    run_env["refs"] = _refs(tmp_path / "fonts", G.shape_font(LATIN))
    rec, data = _run(src)
    assert rec.exit == 0 and rec.status == "done", data["steps"]
    out = Path(rec.output["path"])
    assert out.name == "in (searchable).pdf" and data["output"]["kind"] == "searchable"
    assert all(w in _text(out) for w in WORDS)
    step = next(s for s in data["steps"] if s["step"] == "fontmap")
    assert step["status"] == "changed" and step["detail"]["fonts"] == 1 and step["detail"]["pages"] == [1]
    assert ("font_map_repaired", 1) in _codes(data["notes"])
    assert not any(c == "garbled_text" for c, _p in _codes(data["warnings"]))
    checks = {c["name"]: c["ok"] for c in data["verify"]}
    assert checks["fonts_mapped"] and checks["renders_identical"] and all(checks.values())
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before


def test_a_counterparty_file_gets_a_searchable_copy(tmp_path: Path, run_env: dict[str, Any]):
    src = _garbled(tmp_path, WORDS)
    run_env["refs"] = _refs(tmp_path / "fonts", G.shape_font(LATIN))
    rec, data = _run(src, provenance="counterparty")
    assert rec.exit == 0 and data["output"]["kind"] == "searchable_copy"
    assert "provenance_counterparty" in data["output"]["copy_reasons"]
    assert all(w in _text(Path(rec.output["path"])) for w in WORDS)


def test_plan_says_what_it_would_do_and_writes_nothing(tmp_path: Path, run_env: dict[str, Any]):
    src = _garbled(tmp_path, WORDS)
    run_env["refs"] = _refs(tmp_path / "fonts", G.shape_font(LATIN))
    rec, data = _run(src, plan_only=True)
    assert rec.output is None and not list(tmp_path.glob("*searchable*"))
    step = next(s for s in data["steps"] if s["step"] == "fontmap")
    assert step["detail"]["planned"] is True and step["detail"]["fonts"] == 1


def test_nothing_proven_leaves_the_page_garbled(tmp_path: Path, run_env: dict[str, Any]):
    src = _garbled(tmp_path, WORDS)
    rec, data = _run(src)
    assert rec.exit == 0 and rec.status == "done"
    assert rec.output is None or "(searchable)" not in rec.output["path"]    # at most the lossless shrink
    assert ("garbled_text", 1) in _codes(data["warnings"])
    assert not any(c.startswith("font_map") for c, _p in _codes(data["notes"]) + _codes(data["warnings"]))
    step = next(s for s in data["steps"] if s["step"] == "fontmap")
    assert step["status"] == "ok" and step["detail"]["unrepaired"] == [1]


def test_a_partial_map_warns(tmp_path: Path, run_env: dict[str, Any]):
    src, _sub = _lookalike_pdf(tmp_path, ["bad", "а"])
    run_env["refs"] = _refs(tmp_path / "fonts", G.shape_font(LOOK, same_shape={"а": "a"}))
    rec, data = _run(src)
    assert rec.exit == 0
    assert ("font_map_partial", 1) in _codes(data["warnings"])
    assert "�" in _text(Path(rec.output["path"]))


def test_an_ordinary_pdf_reads_no_font(tmp_path: Path, run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    font = G.shape_font(LATIN)
    src = G.garbled_pdf(tmp_path / "ok.pdf", [G.Page(G.lines_of(WORDS))],
                        [G.FontSpec(font, tounicode={c: c for c in LATIN})])
    monkeypatch.setattr(fontmap, "describe", lambda f: pytest.fail("a font was read"))
    rec, data = _run(src)
    step = next(s for s in data["steps"] if s["step"] == "fontmap")
    assert step == {"step": "fontmap", "status": "ok", "detail": {"fonts": 0}}
    assert rec.exit == 0


def test_a_repair_that_breaks_leaves_the_file_as_before(tmp_path: Path, run_env: dict[str, Any],
                                                        monkeypatch: pytest.MonkeyPatch):
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("made up")

    monkeypatch.setattr(fontmap, "repair", boom)
    src = _garbled(tmp_path, WORDS)
    rec, data = _run(src)
    step = next(s for s in data["steps"] if s["step"] == "fontmap")
    assert step["status"] == "failed" and step["detail"] == {"error": "RuntimeError"}
    assert rec.exit == 0 and (rec.output is None or "(searchable)" not in rec.output["path"])
    assert ("garbled_text", 1) in _codes(data["warnings"])


def test_ocr_evidence_through_the_real_reader(tmp_path: Path, run_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    """The real English pack (pinned models) reads a page drawn with Noto Sans glyphs whose font has no
    map and no names; the letters OCR proves are right, and nothing it can't prove is guessed."""
    pytest.importorskip("tesserocr")
    from conftest import pinned_asset
    from pdfintake import flow, platforms
    from pdfintake.langpacks import manifest as M
    from pdfintake.langpacks import store

    plat = platforms.current()
    if not plat.supported:
        pytest.skip(f"{plat.key} isn't a supported platform")
    repo = Path(__file__).resolve().parents[1]
    m = M.parse((repo / "packs" / "en" / "manifest.json").read_bytes())
    root = tmp_path / "real" / "en-1.0.0"
    for asset in [*m.models, *m.fonts]:
        dest = root / ("models" if asset in m.models else "fonts") / asset.file
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(pinned_asset("en", asset.key).read_bytes())
    base = store.InstalledPack(m, root)
    from test_flow import _fixture_index

    monkeypatch.setattr(flow, "_load_index", _fixture_index)
    monkeypatch.setattr(flow, "_ensure", lambda entry, *, allow_network: base)
    monkeypatch.setattr(flow, "_installed", lambda: {"en": base})
    monkeypatch.setattr(flow, "_fontmap_read", run_env["read"])          # the real reader, this once
    texts = ["The example invoice lists seven sample items", "and a total of 4521 units for the test",
             "customer. Reference AB123456 is made up.", "Seven items were sent to the test customer."]
    chars = "".join(sorted(set("".join(texts))))
    noto = G.noto_subset(pinned_asset("en", "noto-sans"), chars)
    src = G.garbled_pdf(tmp_path / "in.pdf", [G.Page(G.lines_of(texts), size=20)],
                        [G.FontSpec(noto, base="ABCDEF+NotoSans-Regular")])
    rec, data = _run(src)
    step = next(s for s in data["steps"] if s["step"] == "fontmap")
    assert step["detail"]["codes"]["ocr"] >= 15, step
    out = Path(rec.output["path"])
    text = _text(out)
    # every proven letter is right: each read word is the true word or holds U+FFFD where unproven
    truth = " ".join(texts).split()
    got = text.split()
    assert len(got) == len(truth)
    for t, g in zip(truth, got):
        assert len(t) == len(g) and all(gc in (tc, "�") for tc, gc in zip(t, g)), (t, g)

