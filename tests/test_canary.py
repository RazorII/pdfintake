"""langpacks/canary.py: every engine of a pack must read the pack's canary on this computer.

The engine here is test_store.TextEngine, which reads back the text of its model files, so a pack's
"language" is whatever its model says: a wrong-language model under the right name is a model file
with other text, pinned by the manifest (its hash matches, so only the canary can catch it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdfintake import platforms
from pdfintake.errors import EngineFailure, IntegrityError, PackError
from pdfintake.langpacks import canary, store
from pdfintake.langpacks import index as I
from pdfintake.langpacks import manifest as M
from pdfintake.ocr import engines
from test_store import Env, TextEngine, index_entry, make_env, make_pack, write_pack_dir

KEY = platforms.current().key


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    return make_env(tmp_path, monkeypatch, None)


def installed(env: Env, pack, *, into_store: bool = False) -> store.InstalledPack:
    """The pack's files on disk and its InstalledPack (in the store's folder when ``into_store``)."""
    root = env.pack_dir(pack.id, pack.version) if into_store else env.tmp / "packs" / f"{pack.id}-{pack.version}"
    write_pack_dir(pack, root)
    b = store.base()
    return store.InstalledPack(M.parse(pack.raw, base=b.manifest), root)


def cache_files() -> list[str]:
    folder = platforms.cache_root() / "canary"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


# --------------------------------------------------------------------------- the cache key

def test_the_cache_key_changes_with_engine_version_model_sha_or_os():
    shas = ["a" * 64, "b" * 64]
    key = canary.cache_key("tesseract", "5.5.3", shas, "darwin-arm64/macOS 15.4")
    assert key.startswith("tesseract-") and len(key) == len("tesseract-") + 40
    assert key == canary.cache_key("tesseract", "5.5.3", list(reversed(shas)), "darwin-arm64/macOS 15.4")
    variants = [
        canary.cache_key("tesseract", "5.5.4", shas, "darwin-arm64/macOS 15.4"),
        canary.cache_key("tesseract", None, shas, "darwin-arm64/macOS 15.4"),
        canary.cache_key("tesseract", "5.5.3", ["a" * 64, "c" * 64], "darwin-arm64/macOS 15.4"),
        canary.cache_key("tesseract", "5.5.3", shas, "darwin-arm64/macOS 15.5"),
        canary.cache_key("vision", "5.5.3", shas, "darwin-arm64/macOS 15.4"),
    ]
    assert len({key, *variants}) == 6
    assert canary.cache_key("../weird name", None, [], "x").startswith("___weird_name-")   # always a plain file name


# --------------------------------------------------------------------------- a pass, and its cache

def test_a_good_pack_passes_and_the_pass_is_cached(env: Env):
    pack = installed(env, make_pack("xa"))
    results = canary.check(pack, store.base(), KEY)
    assert [(r.use_id, r.engine, r.ok, r.found, r.missing, r.cached, r.error) for r in results] == [
        ("reader", "fake", True, ["ตัวอย่าง", "4521"], [], False, None)]
    assert env.engine.reads == 1 and len(cache_files()) == 1
    again = canary.check(pack, store.base(), KEY)
    assert again[0].ok and again[0].cached and env.engine.reads == 1


def test_a_changed_engine_version_or_os_reads_again(env: Env, monkeypatch: pytest.MonkeyPatch):
    pack = installed(env, make_pack("xa"))
    canary.check(pack, store.base(), KEY)
    env.engine.ver = "1.1"
    assert not canary.check(pack, store.base(), KEY)[0].cached and env.engine.reads == 2
    monkeypatch.setattr(canary, "_os_version", lambda key: f"{key}/macOS 99.0")
    assert not canary.check(pack, store.base(), KEY)[0].cached and env.engine.reads == 3
    assert canary.check(pack, store.base(), KEY)[0].cached and env.engine.reads == 3


def test_a_corrupt_cache_file_is_ignored(env: Env):
    pack = installed(env, make_pack("xa"))
    canary.check(pack, store.base(), KEY)
    (platforms.cache_root() / "canary" / cache_files()[0]).write_text("{not json", encoding="utf-8")
    assert canary.check(pack, store.base(), KEY)[0].ok and env.engine.reads == 2


def test_the_os_version_names_the_macos_release(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(platforms, "macos_version", lambda: (15, 4))
    assert canary._os_version("darwin-arm64") == "darwin-arm64/macOS 15.4"
    monkeypatch.setattr(platforms, "macos_version", lambda: None)
    if canary.sys.platform != "win32":
        assert canary._os_version("linux-x86_64") == "linux-x86_64"


# --------------------------------------------------------------------------- failures

def test_a_wrong_language_model_under_the_right_name_fails_the_canary(env: Env):
    wrong = make_pack("xa", models={"xa.traineddata": "Beispieltext"})   # pinned, so its hash matches
    pack = installed(env, wrong)
    [r] = canary.check(pack, store.base(), KEY)
    assert not r.ok and r.found == ["4521"] and r.missing == ["ตัวอย่าง"]
    assert "didn't find 1 of the 2" in (r.error or "")
    assert cache_files() == []                                    # a failure is never cached
    canary.check(pack, store.base(), KEY)
    assert env.engine.reads == 2


def test_a_failed_canary_makes_the_pack_unusable_this_run(env: Env):
    wrong = make_pack("xa", models={"xa.traineddata": "Beispieltext"})
    installed(env, wrong, into_store=True)
    entry = index_entry(wrong, url="https://127.0.0.1/pdfintake_lang_xa-1.0.0-py3-none-any.whl", wheel=b"made-up")
    env.entries["xa"] = entry
    with pytest.raises(PackError) as info:
        store.ensure(entry, allow_network=False)
    assert info.value.code == canary.PACK_BROKEN == "pack_broken"
    assert info.value.detail["results"][0]["missing"] == 1
    assert store.locate("xa") is None


def test_a_vision_stub_returning_ok_junk_fails_it(env: Env, monkeypatch: pytest.MonkeyPatch):
    vision = TextEngine("vision")
    vision.junk = "Lorem ipsvm dolor"                             # availability says ok; the read is ok and junk
    monkeypatch.setitem(engines._REGISTRY, "vision", vision)
    pack = installed(env, make_pack("xa", extra_uses=({"id": "vision", "engine": "vision", "role": "reader",
                                                        "params": {"lang": "xa"}, "models": []},)))
    results = {r.use_id: r for r in canary.check(pack, store.base(), KEY)}
    assert results["reader"].ok
    assert not results["vision"].ok and results["vision"].missing == ["ตัวอย่าง", "4521"]
    assert results["vision"].engine == "vision"


@pytest.mark.parametrize("state,ok", [("not_on_platform", True), ("unsupported_language", True),
                                      ("missing_model", False), ("broken", False)])
def test_an_unavailable_engine(env: Env, state: str, ok: bool):
    pack = installed(env, make_pack("xa"))
    env.engine.state = state
    [r] = canary.check(pack, store.base(), KEY)
    assert r.ok is ok and env.engine.reads == 0
    assert state in (r.error or "")
    if ok:
        assert r.found == [] and r.missing == [] and (r.error or "").startswith("not read")


def test_an_engine_that_raises_fails_the_canary(env: Env, monkeypatch: pytest.MonkeyPatch):
    pack = installed(env, make_pack("xa"))

    def boom(self, page, *, timeout_s):  # noqa: ANN001
        raise EngineFailure("model_broken", "Tesseract model made-up.traineddata: the file is truncated")

    monkeypatch.setattr("test_store.TextSession.read", boom)
    [r] = canary.check(pack, store.base(), KEY)
    assert not r.ok and "truncated" in (r.error or "") and cache_files() == []


def test_an_empty_read_fails_the_canary(env: Env):
    pack = installed(env, make_pack("xa"))
    env.engine.junk = ""
    [r] = canary.check(pack, store.base(), KEY)
    assert not r.ok and "status empty" in (r.error or "")


def test_an_engine_this_core_no_longer_registers_fails_the_canary(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(engines._REGISTRY, "gone", TextEngine("gone"))
    pack = installed(env, make_pack("xa", engine="gone", base_refs=()))
    b = store.base()
    monkeypatch.delitem(engines._REGISTRY, "gone")
    [r] = canary.check(pack, b, KEY)
    assert not r.ok and "unknown engine" in (r.error or "")


# --------------------------------------------------------------------------- matching and the image

def test_matching_ignores_whitespace_but_not_accents(env: Env):
    spaced = installed(env, make_pack("xa", models={"xa.traineddata": "ตัว อย่าง"}))     # engines space Thai clusters
    assert canary.check(spaced, store.base(), KEY)[0].ok
    accents = installed(env, make_pack("xb", models={"xb.traineddata": "orlo fuzfa"}, must_find=("őrlő",)))
    [r] = canary.check(accents, store.base(), KEY)
    assert not r.ok and r.missing == ["őrlő"]


def test_a_pack_without_a_canary_has_nothing_to_check(env: Env):
    pack = installed(env, make_pack("xa", canary=False))
    assert canary.check(pack, store.base(), KEY) == []


def test_a_tampered_canary_or_model_is_an_integrity_error(env: Env):
    pack = installed(env, make_pack("xa"))
    png = pack.root / pack.manifest.canary.file  # type: ignore[union-attr]
    good = png.read_bytes()
    png.write_bytes(good[:-1] + b"\x00")
    with pytest.raises(IntegrityError, match="canary-xa.png"):
        canary.check(pack, store.base(), KEY)
    png.write_bytes(good)
    model = pack.root / "models" / "xa.traineddata"
    model.write_bytes(model.read_bytes()[:-2] + b"X\n")
    with pytest.raises(IntegrityError, match="xa.traineddata"):
        canary.check(pack, store.base(), KEY)
    assert env.engine.reads == 0


def test_the_cache_holds_no_read_text(env: Env):
    pack = installed(env, make_pack("xa", models={"xa.traineddata": "ตัวอย่าง plus a made-up secret line"}))
    canary.check(pack, store.base(), KEY)
    [name] = cache_files()
    text = (platforms.cache_root() / "canary" / name).read_text(encoding="utf-8")
    assert "secret" not in text and json.loads(text)["ok"] is True


# --------------------------------------------------------------------------- review fixes

@pytest.mark.parametrize("blank", [(" ",), (" ", " "), ("4521", "　")])
def test_a_blank_canary_string_is_refused_when_the_pack_loads(env: Env, blank: tuple[str, ...]):
    # Decided 2026-11-05 (DESIGN §11.2): the schema refuses a must_find item with no non-whitespace
    # character, so a blank string never reaches the canary ('' would be "found" in every read).
    with pytest.raises(PackError) as e:
        installed(env, make_pack("xa", must_find=blank))
    assert e.value.code == "invalid_manifest" and "must_find" in str(e.value)
    assert env.engine.reads == 0 and cache_files() == []


def test_skipped_tells_a_skip_from_a_read(env: Env):
    pack = installed(env, make_pack("xa"))
    env.engine.state = "not_on_platform"
    [skip] = canary.check(pack, store.base(), KEY)
    assert skip.ok and canary.skipped(skip)
    env.engine.state = "available"
    [read] = canary.check(pack, store.base(), KEY)
    [cached] = canary.check(pack, store.base(), KEY)
    assert read.ok and not canary.skipped(read)
    assert cached.cached and not canary.skipped(cached)       # a cached pass was a real read when it was cached
    env.engine.state = "broken"
    env.engine.ver = "9.9"
    [failed] = canary.check(pack, store.base(), KEY)
    assert not failed.ok and not canary.skipped(failed)


def test_the_canary_and_store_docs_say_a_broken_pack_is_replaced_not_asked_about():
    # DESIGN §12.3 (decided 2026-11-05): a pack that fails its canary, its download or its hash
    # check goes to langplan.resolve_unsure(..., broken=...) and its pages are read with the base
    # pack, warning pack_broken. No docstring may still say the run asks a language decision.
    docs = {"canary": canary.__doc__, "store.ensure": store.ensure.__doc__, "store.fetch": store.fetch.__doc__}
    for name, doc in docs.items():
        assert doc, name
        flat = " ".join(doc.split())
        assert "decision" not in flat.lower() and "the run asks" not in flat, name
        assert "resolve_unsure(..., broken=...)" in flat and "pack_broken" in flat and "base pack" in flat, name
    assert canary.PACK_BROKEN == "pack_broken"
