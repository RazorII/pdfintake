"""kit/corpus.py and ``kit init`` / ``kit corpus``: texts pinned by revision and hash, drift caught.

The network is never touched: ``corpus._http_get`` is replaced by ``FakeWeb``, which answers with
made-up bytes. ``kit init`` runs on a copy of the fixture packs (tests/fixtures/packs).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from pdfintake.errors import IntegrityError, PackError, UsageError
from pdfintake.kit import cli
from pdfintake.kit import corpus as C
from pdfintake.langpacks import manifest as M
from test_kit_render import FONT_URL, WIKITEXT, FakeWeb, latin_font

FIXTURE_PACKS = Path(__file__).resolve().parent / "fixtures" / "packs"


def test_wikitext_to_plain_keeps_only_words():
    plain = C.wikitext_to_plain(WIKITEXT + "<!-- hidden -->\n{| class=x\n| cell\n|}\n[[File:A.png|thumb|a [[b|caption]]]]\n"
                                "[https://example.com/a external label]\n")
    assert plain.startswith("Example is a made-up article.")
    for gone in ("Infobox", "A note", "Category", "hidden", "cell", "caption", "thumb", "<ref", "'''", "=="):
        assert gone not in plain
    assert "History" in plain and "external label" in plain
    assert C.wikitext_to_plain(WIKITEXT) == C.wikitext_to_plain(WIKITEXT)


def test_wikitext_keeps_the_space_between_a_number_and_its_unit():
    # Spanish Wikipedia writes the space as {{esd}}; dropping it drew "99,97°C" and "1atm" (kit trial es).
    raw = "El punto de ebullición de la muestra es de 99,97{{esd}}°C (373,12{{esd}}K) a 1{{esd}}atm.\n"
    assert C.wikitext_to_plain(raw) == "El punto de ebullición de la muestra es de 99,97 °C (373,12 K) a 1 atm."


def test_wikitext_keeps_numbers_dates_and_units_held_in_templates():
    raw = ("The example mill opened in {{date|March 1907}}, {{circa|1850}} by one count, and ran {{convert|3.5|km}} "
           "of belts at {{convert|40|C|F}}; it made {{formatnum:12500}} loaves a day, 5{{endash}}10 kg each, "
           "{{nowrap|[[Example unit|12 units]] apiece}}.{{citation needed|date=May 2026}}<ref>{{cite web|title=A note}}</ref>\n"
           "{{Infobox mill|length={{convert|3|km}}|year=1907}}\n"
           "Its rating is {{val|1.5|u=kW}}, never {{val|5.1|e=-10|u=Pa}}, in the {{lang|la|exemplum}} style.\n")
    assert C.wikitext_to_plain(raw) == (
        "The example mill opened in March 1907, c. 1850 by one count, and ran 3.5 km of belts at 40 °C; "
        "it made 12500 loaves a day, 5–10 kg each, 12 units apiece.\n"
        "Its rating is 1.5 kW, never , in the style.")
    assert C.wikitext_to_plain(raw) == C.wikitext_to_plain(raw)


@pytest.mark.parametrize("raw, want", [
    ("del {{siglo|XXI||s}} y del {{siglo|V|a|s}}, en el {{siglo|XX}}. {{Siglo|XIX||S}}",
     "del siglo XXI y del siglo V a. C., en el XX. Siglo XIX"),
    ("au {{s-|XIV}}, au {{-s-|V}}, en 1223 {{av JC}}, en {{date-|août 2018}}",
     "au XIVe siècle, au Ve siècle av. J.-C., en 1223 av. J.-C., en août 2018"),
    ("no {{séc|XVII}} e no {{-séc|III}}, em 204-{{AC|203|X}}, {{fmtn|332900}} vezes",
     "no século XVII e no século III a.C., em 204-203 a.C., 332900 vezes"),
    ("{{szám|0|C}} és {{unité|5000|tonnes}}, {{nombre|5.5|milliards}}, {{tmp|100|°C}}, {{chem|H|2|O}}",
     "0 °C és 5000 tonnes, 5.5 milliards, 100 °C, H2O"),
    ("L{{'}}esempio misura {{M|5777|u=K}} e {{Tutto attaccato|1 360 km}}, {{Citation|une phrase}} {{incise|en passant}}",
     "L'esempio misura 5777 K e 1 360 km, « une phrase » — en passant —"),
], ids=["es", "fr", "pt", "units", "it-fr"])
def test_wikitext_reads_dates_eras_and_values(raw: str, want: str):
    assert C.wikitext_to_plain(raw + "\n") == want


@pytest.mark.parametrize("raw, want", [
    # A unit's power is written against its unit, as the wiki's text reads without its <sup> tags.
    ("La masse molaire vaut {{Unité|18,5|g||mol|-1}}, la cuve tient {{unité|60|m|3}} et la plaque "
     "{{unité|7|m|2|s|-1}}; la masse volumique est de {{unité|1500 kg m -3}} ou {{Unité|8,5 J mol-1 K-1}}.",
     "La masse molaire vaut 18,5 g mol−1, la cuve tient 60 m3 et la plaque 7 m2 s−1; "
     "la masse volumique est de 1500 kg m−3 ou 8,5 J mol−1 K−1."),
    # A power of ten has no plain-text form ("2 e30" would read as the number 2, "×1030" as 1030): dropped.
    ("Une masse de {{unité|2 e30|kg}}, {{unité|4|e=5|m}}, {{unité|1 e14 m −3}} et {{unité|e38 réactions}}.",
     "Une masse de , , et ."),
    ("A lake of {{convert|380|e9m3}} and {{convert|1.5|e9km3|e6mi3}}, {{convert|50|au|e9km}} away.",
     "A lake of and , 50 au away."),
    ("Il Sole emette {{Converti|3,8e26|J|YJ}} e {{converti|40|K|C}}.", "Il Sole emette e 40 K."),
    ("Évi {{szám|2e6|tonna}} és {{szám|170|millió|tonnát}}.", "Évi és 170 millió tonnát."),
    # Range ends and the words the wiki shows between them.
    ("De {{unité|0|–=11|km}}, de {{unité|40|et=80|millions}}, de {{unité|5|à=7|km|2}}.",
     "De 0–11 km, de 40 et 80 millions, de 5 à 7 km2."),
    # {{=}} is an equals sign inside another template's argument.
    ("Its albedo is {{nowrap|0.3 {{=}} 0.3}} and {{nowrap|''E'' {{=}} ''mc''<sup>2</sup>}}; {{=}}.",
     "Its albedo is 0.3 = 0.3 and E = mc2; =."),
], ids=["unit-powers", "powers-of-ten-fr", "powers-of-ten-en", "powers-of-ten-it", "hu", "unite-ranges", "equals"])
def test_wikitext_writes_powers_ranges_and_equals_like_the_wiki(raw: str, want: str):
    assert C.wikitext_to_plain(raw + "\n") == want


@pytest.mark.parametrize("lang, raw, want", [
    ("en", "A ratio of {{val|0.52|–|0.54|u=°}}, {{val|5|–|30|u=%}}, {{val|10|-|20|u=K}}, {{val|1|to|2|u=m}}, "
           "{{val|3|x|4|u=m}}, {{val|1.5|u=%}} and {{val|100|u=°C}}.",
     "A ratio of 0.52°–0.54°, 5%–30%, 10–20 K, 1 to 2 m, 3 m × 4 m, 1.5% and 100 °C."),
    ("it", "Tra {{M|22|and|26|ul=°C}}, {{M|180|to|275|u=°C}}, {{M|5|-|7|u=%}} e {{val|1|to|2|u=g}}.",
     "Tra 22 e 26 °C, 180 fino a 275 °C, 5%-7% e 1 fino a 2 g."),
    ("pt", "De {{val|1|-|2|u=km}} e {{val|3|to|4|u=g}}.", "De 1–2 km e 3 a 4 g."),
    ("", "A range of {{val|10|-|20|u=K}} and {{val|1|to|2|u=m}}.", "A range of 10-20 K and 1 to 2 m."),
], ids=["en", "it", "pt", "other"])
def test_wikitext_reads_val_ranges_in_the_wikis_words(lang: str, raw: str, want: str):
    assert C.wikitext_to_plain(raw + "\n", lang) == want


def test_page_truth_keeps_whole_numbers_and_closes_brackets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The truth used to drop every word without a letter: "ponad 300 piekarń" lost its number (kit trial
    # pl) and "(sống vào năm 23 đến 79)" its years and closing bracket (kit trial vi).
    from pdfintake.kit import render as R
    from test_kit_render import font_files, latin_font2

    raw = ("The example town had over 300 [[bakery|bakeries]] and 12 mills.\n"
           "The example writer (who lived from 23 to 79) and the example astronomer (about 150) wrote on it.\n"
           "The example dough holds some salt (0,5 %) and a gap ({{lang|la|exemplum}}) in it.\n")
    monkeypatch.setattr(C, "_http_get", FakeWeb(wikitext=raw))
    c = C.Corpus("test-v1", tuple(C.pin_wikipedia("en", ["Q1"])), (), {"prose": 1, "form": 1})
    fonts = font_files(tmp_path)
    (tmp_path / "ExampleSerif-Regular.ttf").write_bytes(latin_font2())
    pages = R.render_pages(c, [*fonts, tmp_path / "ExampleSerif-Regular.ttf"], tmp_path / "pages",
                           cache=tmp_path / "texts", scripts=["Latin"])
    prose = next(p for p in pages if p.id == "prose-0-clean").truth
    for number in ("300", "12", "23", "79)", "150)", "(0,5", "%)"):
        assert number in prose.split()
    assert prose.count("(") == prose.count(")") == 3          # the empty "()" a dropped template leaves is left out
    form = next(p for p in pages if p.id == "form-0-clean").truth
    labels = [part for line in form.splitlines() for part in line.split(":")[0].split()]
    assert labels and all(any(ch.isalpha() for ch in w) for w in labels)    # a form label is a word, never a number


def test_pin_wikipedia_pins_revision_and_hash(monkeypatch: pytest.MonkeyPatch):
    web = FakeWeb(revid=777)
    monkeypatch.setattr(C, "_http_get", web)
    (t,) = C.pin_wikipedia("es", ["Q1"])
    assert t.source == "wikipedia" and t.lang == "es" and t.revid == 777 and t.title == "Example Q1"
    assert t.url == "https://es.wikipedia.org/w/index.php?oldid=777&action=raw"
    assert t.sha256 == hashlib.sha256(WIKITEXT.encode()).hexdigest() and t.licence == "CC-BY-SA-4.0"
    with pytest.raises(UsageError):
        C.pin_wikipedia("es", ["283"])
    with pytest.raises(UsageError):
        C.pin_wikipedia("Spanish", ["Q1"])


def test_fetch_text_detects_drift_and_uses_its_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(C, "_http_get", FakeWeb())
    (t,) = C.pin_wikipedia("en", ["Q1"])
    assert "made-up article" in C.fetch_text(t, cache=tmp_path)
    assert (tmp_path / f"{t.sha256}.txt").is_file()
    drifted = FakeWeb(wikitext=WIKITEXT + "an edit")
    monkeypatch.setattr(C, "_http_get", drifted)
    assert "made-up article" in C.fetch_text(t, cache=tmp_path)        # the cached revision, no download
    assert drifted.calls == []
    with pytest.raises(IntegrityError, match="corpus.json pins"):
        C.fetch_text(t)


def test_corpus_json_is_checked_strictly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(C, "_http_get", FakeWeb({FONT_URL: latin_font()}))
    c = C.Corpus("public-v1", tuple(C.pin_wikipedia("en", ["Q1"])), (C.pin_font(FONT_URL),), dict(C.DEFAULT_PAGES))
    C.save(c, tmp_path / "corpus.json")
    again = C.load(tmp_path / "corpus.json")
    assert again == c and C.corpus_digest(again) == C.corpus_digest(c)
    assert C.corpus_ref(c) == f"public-v1@{C.corpus_digest(c)}"
    other = C.Corpus("public-v1", c.texts, c.fonts, {**c.pages, "prose": 9})
    assert C.corpus_digest(other) != C.corpus_digest(c)
    data = C.to_json(c)
    data["texts"][0]["licence"] = "CC-BY-ND-4.0"
    data["fonts"][0]["url"] = "https://example.com/font.ttf"
    data["pages"]["poster"] = 1
    with pytest.raises(PackError) as exc:
        C.from_json(data)
    text = " ".join(exc.value.detail["problems"])
    assert "licence" in text and "fonts[0].url" in text and "$.pages" in text


def test_only_allowed_licences_and_font_hosts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(C, "_http_get", FakeWeb())
    with pytest.raises(UsageError, match="isn't one the kit allows"):
        C.pin_url("https://example.com/a.txt", lang="en", title="A", licence="all-rights-reserved")
    with pytest.raises(UsageError, match="allowed font source"):
        C.pin_font("https://example.com/Font-Regular.ttf")


def _packs(tmp: Path) -> Path:
    dest = tmp / "packs"
    shutil.copytree(FIXTURE_PACKS, dest)
    return dest


def test_kit_init_writes_a_draft_the_loader_accepts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    (packs / "en" / "NOTICE-tessdata").write_text("Example notice.\n", encoding="utf-8")
    model = b"made-up tessdata model bytes"
    url = cli.TESSDATA_URL.format(commit=cli.TESSDATA_BEST_COMMIT, name="spa")
    monkeypatch.setattr(C, "_http_get", FakeWeb({url: model}))
    code = cli.main(["kit", "init", "es", "--script", "Latin", "--packs", str(packs), "--json",
                     "--must-survive", "acute=áéíóúñ:word:0.9"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["ok"], out
    draft = json.loads((packs / "es" / "manifest.draft.json").read_text(encoding="utf-8"))
    assert draft["models"][0]["sha256"] == hashlib.sha256(model).hexdigest()
    assert draft["models"][0]["bytes"] == len(model) and draft["models"][0]["source"] == url
    assert draft["must_survive"] == [{"name": "acute", "chars": "áéíóúñ", "unit": "word", "floor": 0.9}]
    base = M.parse((packs / "en" / "manifest.json").read_bytes())
    assert M.check(draft, base=base) == []
    assert C.load(packs / "es" / "corpus.json").texts == ()
    assert (packs / "es" / "NOTICE-tessdata").is_file()
    assert cli.main(["kit", "init", "es", "--script", "Latin", "--packs", str(packs)]) == 2


def test_kit_init_stops_with_6_when_tesseract_has_no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    monkeypatch.setattr(C, "_http_get", FakeWeb())
    code = cli.main(["kit", "init", "xx", "--script", "Latin", "--iso639-3", "xxx", "--title", "Example",
                     "--packs", str(packs)])
    out = json.loads(capsys.readouterr().out)
    assert code == 6 and out["error"]["code"] == "kit_no_model"
    assert not (packs / "xx").exists()
    assert cli.main(["kit", "init", "zz", "--script", "Latin", "--packs", str(packs)]) == 2


def test_kit_corpus_pins_into_corpus_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]):
    packs = _packs(tmp_path)
    monkeypatch.setattr(C, "_http_get", FakeWeb({FONT_URL: latin_font()}))
    code = cli.main(["kit", "corpus", "hu", "--packs", str(packs), "--wikipedia", "Q1,Q2", "--font", FONT_URL,
                     "--pages", "prose=2", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["texts"] == 2 and out["fonts"] == 1 and out["pages"]["prose"] == 2
    c = C.load(packs / "hu" / "corpus.json")
    assert [t.lang for t in c.texts] == ["hu", "hu"] and out["ref"] == C.corpus_ref(c)
    assert cli.main(["kit", "corpus", "hu", "--packs", str(packs), "--wikipedia", "Q1"]) == 0   # no duplicate
    assert len(C.load(packs / "hu" / "corpus.json").texts) == 2
    capsys.readouterr()
    assert cli.main(["kit", "corpus", "hu", "--packs", str(packs), "--url", "https://example.com/a.txt",
                     "--title", "A", "--licence", "proprietary"]) == 2


def test_url_texts_come_only_from_allowlisted_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # `kit verify --rebench` fetches corpus texts in CI, so a text host is allowlisted like a model
    # host, and corpus.json with any other host fails the check before anything is fetched.
    allowed = "https://www.gutenberg.org/cache/epub/0/example.txt"
    web = FakeWeb({allowed: b"An example text in the public domain."})
    monkeypatch.setattr(C, "_http_get", web)
    t = C.pin_url(allowed, lang="en", title="Example", licence="public-domain")
    assert t.source == "url" and t.revid == 0
    for url in ("https://example.com/a.txt", "https://www.gutenberg.org.example.com/a.txt",
                "https://user@www.gutenberg.org/a.txt", "https://www.gutenberg.org:8443/a.txt"):
        with pytest.raises(UsageError, match="isn't on a host"):
            C.pin_url(url, lang="en", title="A", licence="public-domain")
        assert not C.text_host_allowed(url)
    assert web.calls == [allowed]
    for ok in ("https://en.wikisource.org/wiki/Example?action=raw", "https://eur-lex.europa.eu/example.txt"):
        assert C.text_host_allowed(ok)
    data = C.to_json(C.Corpus("public-v1", (t,), (), dict(C.DEFAULT_PAGES)))
    C.from_json(data)
    data["texts"][0]["url"] = "https://example.com/a.txt"
    with pytest.raises(PackError, match=r"texts\[0\]\.url: not a host"):
        C.from_json(data)
    (tmp_path / "corpus.json").write_text(json.dumps(data), encoding="utf-8")
    from pdfintake.kit import verify as V

    problems = V._check_manifest(tmp_path, {}, None)
    assert any("corpus:" in p and "not a host" in p for p in problems)


def test_kit_init_lists_each_vision_language_list_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]):
    from pdfintake.ocr import engines

    class Vision:
        def candidate_params(self, tag, script, assets):
            return [{"languages": ["es-ES", "en-US"], "language_correction": flag} for flag in (True, False)]

    real_get = engines.get
    monkeypatch.setattr(engines, "get", lambda name, *a, **k: Vision() if name == "vision" else real_get(name, *a, **k))
    packs = _packs(tmp_path)
    url = cli.TESSDATA_URL.format(commit=cli.TESSDATA_BEST_COMMIT, name="spa")
    monkeypatch.setattr(C, "_http_get", FakeWeb({url: b"made-up tessdata model bytes"}))
    assert cli.main(["kit", "init", "es", "--script", "Latin", "--packs", str(packs)]) == 0
    assert json.loads(capsys.readouterr().out)["vision_languages"] == [["es-ES", "en-US"]]


def test_a_corpus_text_never_redirects_off_the_allowlist(monkeypatch: pytest.MonkeyPatch):
    import urllib.request

    captured: dict = {}

    def capture(*handlers):
        captured["h"] = next(h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
        raise PackError("test_stop", "no network in tests")

    monkeypatch.setattr(urllib.request, "build_opener", capture)

    def redirect(url: str, new: str) -> None:
        with pytest.raises(PackError):
            C._http_get(url, cap=10)                          # the capture stops it before any request
        captured["h"].redirect_request(urllib.request.Request(url), None, 302, "Found", {}, new)

    text = "https://www.gutenberg.org/cache/epub/0/example.txt"
    with pytest.raises(PackError, match="redirects off the corpus text hosts"):
        redirect(text, "https://mirror.example.com/example.txt")
    redirect(text, "https://www.gutenberg.org/files/0/example.txt")
    font = "https://github.com/notofonts/latin-greek-cyrillic/releases/download/example-v1/ExampleSans-Regular.ttf"
    redirect(font, "https://objects.example.com/ExampleSans-Regular.ttf")   # fonts are checked by their pin
