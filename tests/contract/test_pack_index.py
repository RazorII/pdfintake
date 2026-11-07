"""The pack index contract, run against tests/fixtures/index.test.json and, once it exists, the real index.

For every index:
* every entry's manifest is there, has the pinned SHA-256, and parses against the running core;
* every rule and version, engine, fill, repair and normaliser it names is registered;
* Tesseract model file names are unique across the whole index (they share one tessdata directory);
* each entry's ``detect`` copy, scripts, language codes, title, version and base flag match its manifest;
* every published pack has a canary;
* every fetch URL's host is in ``ALLOWED_HOSTS``.

The real pack sources (``packs/*/manifest.json``) are checked on their own as well, without
index.json, so a broken source manifest fails CI before an index is ever generated. Those tests
fail rather than skip (decided 2026-11-05).

The real index holds every launch pack (``LAUNCH_PACKS``), and the docs that name the languages
(README, the plugin README, the skill and plugin.json) each name exactly the index's packs, in one
shared sentence (decided 2026-11-06).

The rest of this file tests the index loader itself (langpacks/index.py) on made-up indexes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from pdfintake.errors import SetupError, PackError
from pdfintake.langpacks import index as pack_index
from pdfintake.langpacks import manifest
from pdfintake.langpacks.index import ALLOWED_HOSTS, detect_copy, load
from pdfintake.ocr import engines, fills, rules, textfix

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
FIXTURE_INDEX = TESTS / "fixtures" / "index.test.json"
REAL_INDEX = REPO / "src" / "pdfintake" / "langpacks" / "index.json"

SOURCES = {
    "fixture": (FIXTURE_INDEX, TESTS / "fixtures" / "packs"),
    "real": (REAL_INDEX, REPO / "packs"),
}


@pytest.fixture(params=sorted(SOURCES), scope="module")
def published(request: pytest.FixtureRequest) -> dict[str, Any]:
    index_path, packs_dir = SOURCES[request.param]
    if not index_path.is_file():
        pytest.skip(f"{request.param} index not generated yet ({index_path.name}; tools/build_index.py writes it)")
    idx = load(index_path)
    raws: dict[str, bytes] = {}
    for e in idx.entries:
        path = packs_dir / e.id / "manifest.json"
        assert path.is_file(), f"index entry {e.id!r} has no manifest at {path}"
        raws[e.id] = path.read_bytes()
    names = set()
    for data in raws.values():
        d = json.loads(data)
        for rs in d["readers"].values():
            names.update(u["engine"] for u in rs["engines"])
        if d["detect"].get("probe"):
            names.add(d["detect"]["probe"]["use"]["engine"])
    for name in sorted(names):
        try:
            engines.get(name)
        except SetupError as exc:
            if exc.code == "engine_import_failed":
                pytest.skip(f"engine {name!r} isn't built yet: {exc.message}")
            raise
    return {"index": idx, "raw": raws, "data": {k: json.loads(v) for k, v in raws.items()}}


def test_every_manifest_parses(published: dict[str, Any]) -> None:
    idx = published["index"]
    raws = published["raw"]
    base_entry = idx.base()
    assert hashlib.sha256(raws[base_entry.id]).hexdigest() == base_entry.manifest_sha256
    base = manifest.parse(raws[base_entry.id])
    assert base.base and base.id == base_entry.id
    for e in idx.entries:
        assert hashlib.sha256(raws[e.id]).hexdigest() == e.manifest_sha256, f"{e.id}: manifest_sha256 is stale"
        m = base if e.base else manifest.parse(raws[e.id], base=base)
        assert m.id == e.id and m.version == e.version and m.raw_sha256 == e.manifest_sha256


def test_every_referenced_member_exists(published: dict[str, Any]) -> None:
    rule_names = set(rules.names())
    for pack_id, d in published["data"].items():
        for key, rs in d["readers"].items():
            assert (rs["pick"]["rule"], int(rs["pick"]["version"])) in rule_names, f"{pack_id} {key}"
            rules.get(rs["pick"]["rule"], int(rs["pick"]["version"]))
            for u in rs["engines"]:
                assert u["engine"] in engines.names(), f"{pack_id} {key} {u['id']}"
        for st in d["fills"]:
            assert (st["name"], int(st["version"])) in fills.FILLS, f"{pack_id} fill {st['name']}"
        for st in d["repairs"]:
            assert (st["name"], int(st["version"])) in textfix.REPAIRS, f"{pack_id} repair {st['name']}"
        for n in d["normalize"]:
            assert n in textfix.NORMALIZERS, f"{pack_id} normaliser {n}"


def _tessdata_clashes(packs: dict[str, dict[str, Any]]) -> list[str]:
    """Tesseract model file names that repeat across packs, compared ignoring case (one shared tessdata directory)."""
    seen: dict[str, str] = {}
    out: list[str] = []
    for pack_id, d in packs.items():
        for m in d.get("models", []):
            if m.get("kind") == manifest.TRAINEDDATA:
                key = m["file"].casefold()
                if key in seen:
                    out.append(f"{m['file']} is in both {seen[key]} and {pack_id}")
                seen.setdefault(key, pack_id)
    return out


def test_tessdata_names_unique(published: dict[str, Any]) -> None:
    assert _tessdata_clashes(published["data"]) == []


def test_detect_copies_match(published: dict[str, Any]) -> None:
    for e in published["index"].entries:
        d = published["data"][e.id]
        assert json.loads(json.dumps(_thaw(e.detect))) == detect_copy(d["detect"]), f"{e.id}: detect copy differs"
        assert list(e.scripts) == d["scripts"]
        assert list(e.langid_codes) == list((d["detect"].get("langid") or {}).get("codes", []))
        assert e.title == d["title"] and e.version == d["version"] and e.base == (d.get("base") is True)


def test_every_published_pack_has_a_canary(published: dict[str, Any]) -> None:
    for pack_id, d in published["data"].items():
        assert d.get("canary"), f"{pack_id} has no canary"


def test_fetch_hosts_allowed(published: dict[str, Any]) -> None:
    for e in published["index"].entries:
        if e.dist.kind == "fetch":
            host = urllib.parse.urlsplit(e.dist.url).hostname
            assert host in ALLOWED_HOSTS, f"{e.id}: {host}"
            assert host == pack_index.ENTRY_HOST, f"{e.id}: an entry names the github.com release download"
        else:
            assert e.base and e.dist.url is None


# --------------------------------------------------------------------------- the real pack sources

PACK_SOURCES = sorted((REPO / "packs").glob("*/manifest.json"))
# The packs that shipped at launch: each was built with the kit, verified, and is in the index.
LAUNCH_PACKS = {"en", "th", "hu", "de", "es", "fr", "it", "pt", "pl", "vi"}

# The docs that name the languages, each in the same sentence (language_sentence).
LANGUAGE_DOCS = (
    REPO / "README.md",
    REPO / "claude-plugin" / "README.md",
    REPO / "claude-plugin" / "skills" / "pdfintake" / "SKILL.md",
    REPO / "claude-plugin" / ".claude-plugin" / "plugin.json",
)


def _source_base() -> manifest.Manifest:
    return manifest.parse((REPO / "packs" / "en" / "manifest.json").read_bytes())


def test_launch_pack_sources_present() -> None:
    assert {p.parent.name for p in PACK_SOURCES} >= LAUNCH_PACKS


def _real_index() -> pack_index.Index:
    """The generated index. It ships in the core wheel, so a missing one fails rather than skips."""
    assert REAL_INDEX.is_file(), f"{REAL_INDEX.relative_to(REPO)} is missing; tools/build_index.py writes it"
    return load(REAL_INDEX)


def test_real_index_holds_the_launch_packs() -> None:
    assert {e.id for e in _real_index().entries} >= LAUNCH_PACKS


def language_sentence(titles: list[str], base: str) -> str:
    """The one sentence every doc uses to name the languages: the base pack's language, then the
    other packs' titles in alphabetical order."""
    rest = sorted(titles)
    listed = rest[0] if len(rest) == 1 else ", ".join(rest[:-1]) + " and " + rest[-1]
    return (f"{base} is built in, and {listed} are language packs, "
            "downloaded the first time a document needs one.")


def test_language_sentence() -> None:
    assert language_sentence(["Thai", "German", "Hungarian"], "English") == (
        "English is built in, and German, Hungarian and Thai are language packs, "
        "downloaded the first time a document needs one.")


@pytest.mark.parametrize("doc", LANGUAGE_DOCS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_docs_name_exactly_the_published_packs(doc: Path) -> None:
    """Each doc names the languages in the same sentence, built from the index's titles, so a doc
    never promises a pack that isn't published and never leaves one out."""
    idx = _real_index()
    base = idx.base()
    sentence = language_sentence([e.title for e in idx.entries if not e.base], base.title)
    text = doc.read_text(encoding="utf-8")
    if doc.suffix == ".json":
        text = json.loads(text)["description"]
    assert sentence in text, f"{doc.relative_to(REPO)} should say: {sentence}"
    # No second, stale list elsewhere in the doc: outside the shared sentence, no sentence names
    # three or more pack languages (one or two, as in a worked example, is fine).
    titles = _pack_titles() - {base.title}
    rest = text.replace(sentence, " ")
    for part in re.split(r"(?<=[.!?])\s+|\n", rest):
        named = sorted(t for t in titles if re.search(rf"\b{re.escape(t)}\b", part))
        assert len(named) < 3, f"{doc.relative_to(REPO)} lists languages outside the shared sentence: {part.strip()!r}"


def _pack_titles() -> set[str]:
    """Every pack title the repo knows: the index's, and those of every pack folder's manifest or draft."""
    titles = {e.title for e in _real_index().entries}
    for p in sorted((REPO / "packs").glob("*/manifest*.json")):
        t = json.loads(p.read_text(encoding="utf-8")).get("title")
        if isinstance(t, str) and t:
            titles.add(t)
    return titles


def test_stale_list_check_catches_a_second_list() -> None:
    """The check above, on a made-up doc: the shared sentence passes, a second list doesn't."""
    titles = {"German", "Hungarian", "Thai", "Polish"}
    ok = "One example in Thai."
    stale = "Packs: Thai, Hungarian and German."
    count = lambda s: sum(bool(re.search(rf"\b{t}\b", s)) for t in titles)  # noqa: E731
    assert count(ok) < 3 and count(stale) >= 3


@pytest.mark.parametrize("path", PACK_SOURCES, ids=lambda p: p.parent.name)
def test_pack_source_manifest_valid(path: Path) -> None:
    """Every packs/*/manifest.json passes the loader and jsonschema (Draft 2020-12), with packs/en as the base."""
    import jsonschema   # a test dependency: a missing one is a failure here, not a skip

    raw = path.read_bytes()
    data = json.loads(raw)
    base = None if data.get("base") is True else _source_base()
    assert manifest.check(data, base=base) == []
    m = manifest.parse(raw, base=base)
    assert m.id == path.parent.name
    schema = json.loads((REPO / "src" / "pdfintake" / "langpacks" / "manifest.schema.json").read_text(encoding="utf-8"))
    errors = [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(data)]
    assert errors == []


def test_pack_source_tessdata_names_unique() -> None:
    packs = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in PACK_SOURCES}
    assert _tessdata_clashes(packs) == []


def test_hu_function_word_ratio_is_the_exact_third() -> None:
    """The function-word rule is the exact third, 3 x hits >= rival hits (§11.2); langplan compares
    hits >= ratio x rivals - 1e-9, so the manifest's ratio must give the same answer on every count."""
    fw = json.loads((REPO / "packs" / "hu" / "manifest.json").read_text(encoding="utf-8"))["detect"]["function_words"]
    ratio = fw.get("ratio", manifest.FW_RATIO_DEFAULT)
    for hu in range(0, 400):
        for other in range(0, 1300):
            assert (hu >= ratio * other - 1e-9) == (3 * hu >= other), (hu, other, ratio)


# Model file names as a NOTICE lists them (one per line, beside its pinned source URL).
_NOTICE_MODEL_RE = re.compile(r"[A-Za-z0-9_.+-]+\.(?:traineddata|onnx)\b")


@pytest.mark.parametrize("path", PACK_SOURCES, ids=lambda p: p.parent.name)
def test_pack_notices_name_their_own_models(path: Path) -> None:
    """Each packs/<id>/NOTICE-* names exactly the model files its manifest pins with that notice,
    each beside its pinned source, and no other model. The wheel carries the pack folder's NOTICE
    as it is, so a NOTICE copied from another pack ships the wrong attribution. A NOTICE that no
    pinned model uses must not be in the folder at all."""
    d = json.loads(path.read_text(encoding="utf-8"))
    folder = path.parent
    pinned: dict[str, dict[str, str]] = {}
    for m in d.get("models", []):
        if m.get("notice"):
            pinned.setdefault(m["notice"], {})[m["file"]] = m["source"]
    present = sorted(p.name for p in folder.glob("NOTICE-*"))
    missing = sorted(set(pinned) - set(present))
    assert not missing, f"{folder.name}: the manifest names notice files that aren't there: {missing}"
    for name in present:
        text = (folder / name).read_text(encoding="utf-8")
        want = pinned.get(name, {})
        named = set(_NOTICE_MODEL_RE.findall(text))
        assert want, f"{folder.name}/{name}: no model in the manifest uses this notice (it names {sorted(named)}); delete it"
        assert named == set(want), (f"{folder.name}/{name} names {sorted(named)}, "
                                    f"but the manifest pins {sorted(want)} with it")
        for file, source in want.items():
            assert source in text, f"{folder.name}/{name}: {file} isn't listed with its pinned source {source}"


def test_notice_check_reads_model_names() -> None:
    text = ("  vie.traineddata  https://example.com/raw/abc/vie.traineddata\n"
            "  ch_ppocr_mobile_v2.0_cls_mobile.onnx  https://example.com/x.onnx\n")
    assert set(_NOTICE_MODEL_RE.findall(text)) == {"vie.traineddata", "ch_ppocr_mobile_v2.0_cls_mobile.onnx", "x.onnx"}


# Invented, neutral sentences per launch language (water, the Moon, the Sun, bread, chess, volcanoes).
# The real index's Hungarian function-word lists must fire on the Hungarian ones and on no other.
FW_SAMPLES = {
    "hu": ["A víz a Föld felszínének nagy részét borítja, és az óceánok tárolják a legtöbbet.",
           "Ez azt jelenti, hogy a folyók vize végül a tengerbe jut, ahol pedig elpárolog.",
           "A Hold egy kőzetből álló égitest, amely nem bocsát ki saját fényt, csak visszaveri a Nap sugarait."],
    "en": ["Water covers most of the surface of the Earth, and the oceans hold almost all of it.",
           "The Moon is a rocky body that gives off no light of its own; it only reflects the light of the Sun.",
           "A game of chess begins with sixteen pieces on each side of the board."],
    "de": ["Das Wasser der Flüsse fließt am Ende in das Meer, wo es verdunstet und als Regen zurückkehrt.",
           "Die Sonne ist ein Stern, und der Mond ist ein Begleiter der Erde.",
           "Für ein gutes Brot braucht man nur Mehl, Wasser, Salz und etwas Zeit."],
    "es": ["El agua cubre la mayor parte de la superficie de la Tierra y los océanos guardan casi toda.",
           "La Luna es un cuerpo rocoso que no tiene luz propia, solo refleja la luz del Sol.",
           "Para hacer pan solo hacen falta harina, agua, sal y un poco de tiempo."],
    "fr": ["L'eau couvre la plus grande partie de la surface de la Terre, et les océans en gardent presque tout.",
           "La Lune est un corps rocheux qui ne produit pas de lumière ; elle renvoie seulement celle du Soleil.",
           "Une partie d'échecs commence avec seize pièces de chaque côté de l'échiquier."],
    "it": ["L'acqua copre la maggior parte della superficie della Terra e gli oceani ne contengono quasi tutta.",
           "La Luna è un corpo roccioso che non produce luce propria, ma riflette quella del Sole.",
           "Per fare il pane bastano farina, acqua, sale e un po' di tempo."],
    "pt": ["A água cobre a maior parte da superfície da Terra, e os oceanos guardam quase toda ela.",
           "A Lua é um corpo rochoso que não tem luz própria; ela apenas reflete a luz do Sol.",
           "Um vulcão nem sempre está ativo, e muitos ficam calmos durante séculos."],
    "pl": ["Woda pokrywa większą część powierzchni Ziemi, a oceany przechowują prawie całą jej ilość.",
           "Księżyc jest skalistym ciałem, które nie świeci własnym światłem, tylko odbija światło Słońca.",
           "Do upieczenia chleba potrzeba tylko mąki, wody, soli i odrobiny czasu."],
    "vi": ["Nước bao phủ phần lớn bề mặt Trái Đất, và các đại dương chứa gần như toàn bộ lượng nước đó.",
           "Mặt Trăng là một thiên thể bằng đá, không tự phát sáng mà chỉ phản chiếu ánh sáng của Mặt Trời.",
           "Một ván cờ vua bắt đầu với mười sáu quân cờ ở mỗi bên bàn cờ."],
    "th": ["น้ำปกคลุมพื้นผิวโลกส่วนใหญ่ และมหาสมุทรเก็บน้ำไว้เกือบทั้งหมด",
           "ดวงจันทร์ไม่มีแสงในตัวเอง แต่สะท้อนแสงจากดวงอาทิตย์",
           "การทำขนมปังใช้เพียงแป้ง น้ำ เกลือ และเวลาเล็กน้อย"],
}


def test_hu_function_words_hold_against_every_launch_language() -> None:
    """§13.1 / U31: the real index's Hungarian word and rival lists fire on Hungarian and on no
    other launch language. An edit to packs/hu's lists, or a new
    Latin pack whose text hits Hungarian words, fails here rather than in a scratch run."""
    from fixtures import build
    from pdfintake.ocr import detect, langplan

    assert set(FW_SAMPLES) >= LAUNCH_PACKS, f"add samples for {sorted(LAUNCH_PACKS - set(FW_SAMPLES))}"
    hu = _real_index().get("hu")
    assert hu is not None
    fw = langplan._function_words(hu)
    assert fw is not None, "the index's hu entry has no function_words"
    for lang, lines in FW_SAMPLES.items():
        score = detect.function_word_score(build.engine_read("checker", lines), fw.words, fw.rivals)
        assert langplan._fires(score, fw) == (lang == "hu"), (lang, score)


def _thaw(v: Any) -> Any:
    if isinstance(v, (dict,)) or hasattr(v, "items"):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_thaw(x) for x in v]
    return v


# --------------------------------------------------------------------------- the loader itself

def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_INDEX.read_text(encoding="utf-8"))


def write(tmp_path: Path, data: Any) -> Path:
    p = tmp_path / "index.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def problems_of(tmp_path: Path, data: Any) -> list[str]:
    with pytest.raises(PackError) as exc:
        load(write(tmp_path, data))
    assert exc.value.code == "invalid_index"
    return exc.value.detail["problems"]


def has(found: list[str], path: str, *words: str) -> bool:
    return any(p.startswith(path + ":") and all(w in p for w in words) for p in found)


def test_lookups() -> None:
    idx = load(FIXTURE_INDEX)
    assert idx.version == 1 and [e.id for e in idx.entries] == ["en", "th", "hu"]
    assert idx.base().id == "en" and idx.get("th").dist.kind == "fetch" and idx.get("xx") is None
    assert [e.id for e in idx.for_script("Thai")] == ["th"]
    assert [e.id for e in idx.for_script("Latin")] == ["en", "hu"]
    assert [e.id for e in idx.for_language("hu")] == ["hu"] and idx.for_language("sw") == []
    hu = idx.get("hu")
    assert "és" in hu.detect["function_words"]["words"] and hu.detect["probe"] is False
    assert idx.get("th").detect["probe"] is True
    with pytest.raises(TypeError):
        hu.detect["probe"] = True  # type: ignore[index]


def test_url_rules(tmp_path: Path) -> None:
    d = fixture()
    th = d["packs"][1]
    good = th["dist"]["url"]
    cases = {
        "http://" + good[len("https://"):]: "https",
        good.replace("github.com", "github.com.example.com"): "isn't allowed",
        good.replace("github.com", "files.pythonhosted.org"): "isn't allowed",
        good.replace("github.com", "release-assets.githubusercontent.com"): "only where GitHub redirects",
        good.replace("github.com", "objects.githubusercontent.com"): "isn't allowed",
        good + "?x=1": "query",
        good + "#x": "fragment",
        good.replace("github.com", "user@github.com"): "user name",
        good.replace("github.com", "github.com:8443"): "port",
        good.replace("/releases/", "/releases/../"): "'..'",
        good.replace("pdfintake_lang_th-1.0.0", "pdfintake_lang_hu-1.0.0"): "must end in",
        good.replace("RazorII", "Razor\nII"): "unreserved",
        good.replace("RazorII", "Razor\tII"): "unreserved",
        good.replace("RazorII", "Razor II"): "unreserved",
        good.replace("/download/", "/download/a@b/"): "unreserved",
        good.replace("RazorII/pdfintake", "someone-else/evil"): "release asset",
    }
    for url, word in cases.items():
        th["dist"]["url"] = url
        assert has(problems_of(tmp_path, d), "$.packs[1].dist.url", word), url
    th["dist"]["url"] = None
    assert has(problems_of(tmp_path, d), "$.packs[1].dist.url", "needs a URL")


def test_base_rules(tmp_path: Path) -> None:
    d = fixture()
    d["packs"][1]["base"] = True
    found = problems_of(tmp_path, d)
    assert has(found, "$.packs[1].base", "only the base pack")
    assert has(found, "$.packs", "exactly one")
    d = fixture()
    d["packs"][0]["dist"]["kind"] = "fetch"
    d["packs"][0]["dist"]["url"] = "https://github.com/RazorII/pdfintake/releases/download/x/pdfintake_data_en-1.0.0-py3-none-any.whl"
    found = problems_of(tmp_path, d)
    assert has(found, "$.packs[0].dist.kind", "installed")
    assert has(found, "$.packs[0].dist.url", "null")
    d = fixture()
    d["packs"][1]["dist"]["kind"] = "installed"
    assert has(problems_of(tmp_path, d), "$.packs[1].dist.kind", "fetched")
    d = fixture()
    d["packs"][2]["dist"]["distribution"] = "pdfintake-lang-de"
    assert has(problems_of(tmp_path, d), "$.packs[2].dist.distribution", "pdfintake-lang-hu")


def test_entry_shape_and_consistency(tmp_path: Path) -> None:
    d = fixture()
    d["packs"].append(copy.deepcopy(d["packs"][2]))
    d["packs"][1]["langid_codes"] = ["lo"]
    d["packs"][1]["scripts"] = ["Thai", "Klingon"]
    d["packs"][1]["detect"]["probe"] = {"use": {}}
    d["packs"][0]["extra"] = 1
    d["packs"][2]["dist"]["sha256"] = "nope"
    d["core_min"] = "9.0.0"
    found = problems_of(tmp_path, d)
    assert has(found, "$.packs[3].id", "twice")
    assert has(found, "$.packs[1].langid_codes", "detect.langid.codes")
    assert has(found, "$.packs[1].scripts[1]", "Klingon")
    assert has(found, "$.packs[1].detect.probe", "true or false")
    assert has(found, "$.packs[0].extra", "unknown field")
    assert has(found, "$.packs[2].dist.sha256")
    assert has(found, "$.core_min", "9.0.0")


@pytest.mark.parametrize("data, word", [
    (b'{"index_version": 1, "index_version": 1}', "twice"),
    (b"[]", "JSON object"),
    (b'{"index_version": Infinity}', "Infinity"),
])
def test_strict_json(tmp_path: Path, data: bytes, word: str) -> None:
    p = tmp_path / "index.json"
    p.write_bytes(data)
    with pytest.raises(PackError) as exc:
        load(p)
    assert exc.value.code == "invalid_index" and word in exc.value.message


def test_missing_index(tmp_path: Path) -> None:
    with pytest.raises(PackError) as exc:
        load(tmp_path / "nope.json")
    assert exc.value.code == "index_missing"
    if not REAL_INDEX.is_file():
        with pytest.raises(PackError) as exc:
            load()
        assert exc.value.code == "index_missing"


def test_detect_copy_reduces_probe() -> None:
    d = {"osd_scripts": ["Thai"], "probe": {"use": {"id": "probe"}, "min_chars": 1, "min_conf": 0.5}}
    assert detect_copy(d) == {"osd_scripts": ["Thai"], "probe": True}
    assert detect_copy({"osd_scripts": []}) == {"osd_scripts": [], "probe": False}
    assert pack_index.wheel_name("pdfintake-lang-th", "1.2.3") == "pdfintake_lang_th-1.2.3-py3-none-any.whl"


def test_url_problems_is_for_entry_urls_and_redirect_problems_for_redirects() -> None:
    """GitHub answers a release download with a redirect to a signed address on
    release-assets.githubusercontent.com. The entry rule refuses that query string; the redirect
    rule (§3.5: https on an allowed host) takes it."""
    signed = ("https://release-assets.githubusercontent.com/github-production-release-asset/1/abc"
              "?sp=r&sv=2018&se=2026&sig=made-up")
    assert any("query" in p for p in pack_index.url_problems(signed))
    assert pack_index.redirect_problems(signed) == []
    refused = {
        "http://release-assets.githubusercontent.com/x": "https",
        "https://downloads.example.invalid/x.whl": "isn't allowed",
        "https://github.com.example.com/x": "isn't allowed",
        "https://user:secret@github.com/x": "user name",
        "https://github.com:8443/x": "port",
        "https://github.com/x\ny": "control",
        "https://github.com/x y": "whitespace",
        "": "missing",
    }
    for url, word in refused.items():
        found = pack_index.redirect_problems(url)
        assert any(word in p for p in found), (url, found)
