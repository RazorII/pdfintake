"""``tools/fetch_test_models.py`` and conftest's ``pinned_asset``/``pinned_assets`` (decided 2026-11-05).

Every test here uses a made-up pack ``xx`` in a temporary folder and a stand-in for
``datawheel.download_sources``, so nothing reaches the network and no real model is needed. What is
covered: ``--check`` on a present, missing and wrong file; a second fetch making no download call; a
failed download reported with its own error; unsafe file names; exit codes 0, 1 and 2; links refused
the same way by the tool and by conftest; and ``pinned_assets`` skipping on a size or hash mismatch,
raising ``LookupError`` for a mistake in the test, and failing (never skipping) when a fetch in a
network test doesn't produce the file.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

import conftest

TOOL = conftest._fetch_tool()

MODEL = b"made-up model bytes for a test\n" * 7
FONT = b"made-up font bytes\n" * 3


def _entry(key: str, name: str, body: bytes) -> dict[str, Any]:
    return {"key": key, "file": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "source": f"https://github.com/example/models/raw/0123456789abcdef0123456789abcdef01234567/{name}"}


def _manifest(**extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"models": [_entry("xx-model", "xx.model", MODEL)],
                            "fonts": [_entry("xx-font", "XxSans-Regular.ttf", FONT)]}
    data.update(extra)
    return data


BODIES = {"xx.model": MODEL, "XxSans-Regular.ttf": FONT}


@pytest.fixture
def packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A packs folder holding pack ``xx``, which the tool reads instead of the repo's."""
    folder = tmp_path / "packs"
    (folder / "xx").mkdir(parents=True)
    (folder / "xx" / "manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    monkeypatch.setattr(TOOL, "PACKS", folder)
    return folder


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "models"
    folder.mkdir()
    monkeypatch.setenv(TOOL.ENV, str(folder))
    return folder


class FakeDataWheel:
    """Stands in for ``datawheel``: writes the pinned bodies, or raises, and counts its calls."""

    class DataWheelError(Exception):
        pass

    def __init__(self, fail: str | None = None) -> None:
        self.calls = 0
        self.fail = fail

    def download_sources(self, manifest: dict[str, Any], dest: Path) -> list[Path]:
        self.calls += 1
        if self.fail:
            raise self.DataWheelError(self.fail)
        dest.mkdir(parents=True, exist_ok=True)
        out = []
        for group in ("models", "fonts"):
            for item in manifest[group]:
                target = dest / item["file"]
                if target.is_symlink():
                    target.unlink()
                target.write_bytes(BODIES[item["file"]])
                out.append(target)
        return out


@pytest.fixture
def wheel(monkeypatch: pytest.MonkeyPatch) -> FakeDataWheel:
    fake = FakeDataWheel()
    monkeypatch.setattr(TOOL, "_datawheel", lambda: fake)
    return fake


def _fill(root: Path, **override: bytes) -> None:
    (root / "xx").mkdir(exist_ok=True)
    for name, body in BODIES.items():
        (root / "xx" / name).write_bytes(override.get(name, body))


# --------------------------------------------------------------------------- the tool

def test_check_present_is_exit_0_and_touches_no_network(packs: Path, root: Path, wheel: FakeDataWheel,
                                                        capsys: pytest.CaptureFixture[str]):
    _fill(root)
    assert TOOL.main(["--check", "--pack", "xx"]) == 0
    assert wheel.calls == 0
    out = capsys.readouterr().out
    assert "present  xx/xx.model" in out and "all present and correct" in out


def test_check_missing_is_exit_1(packs: Path, root: Path, wheel: FakeDataWheel,
                                 capsys: pytest.CaptureFixture[str]):
    assert TOOL.main(["--check", "--pack", "xx", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert [r["state"] for r in report["packs"]["xx"]] == ["missing", "missing"]
    assert wheel.calls == 0


def test_check_wrong_size_and_wrong_hash(packs: Path, root: Path):
    _fill(root, **{"xx.model": MODEL[:-1], "XxSans-Regular.ttf": bytes(len(FONT))})
    rows = {f: (s, d) for f, s, d in TOOL.check_pack("xx", root)}
    assert rows["xx.model"][0] == "wrong" and f"pins {len(MODEL)}" in rows["xx.model"][1]
    zero_sha = hashlib.sha256(bytes(len(FONT))).hexdigest()
    assert rows["XxSans-Regular.ttf"] == ("wrong", f"has SHA-256 {zero_sha}; the manifest pins "
                                                   f"{hashlib.sha256(FONT).hexdigest()}")
    assert TOOL.main(["--check", "--pack", "xx"]) == 1


def test_check_refuses_a_link(packs: Path, root: Path, tmp_path: Path):
    _fill(root)
    real = tmp_path / "elsewhere.model"
    real.write_bytes(MODEL)
    (root / "xx" / "xx.model").unlink()
    (root / "xx" / "xx.model").symlink_to(real)
    rows = {f: (s, d) for f, s, d in TOOL.check_pack("xx", root)}
    assert rows["xx.model"] == ("wrong", "isn't a regular file")


def test_fetch_then_second_run_makes_no_download_call(packs: Path, root: Path, wheel: FakeDataWheel):
    assert TOOL.main(["--pack", "xx"]) == 0
    assert wheel.calls == 1
    assert [s for _, s, _ in TOOL.fetch_pack("xx", root)] == ["present", "present"]
    assert TOOL.main(["--pack", "xx"]) == 0
    assert wheel.calls == 1
    assert (root / "xx" / "xx.model").read_bytes() == MODEL


def test_fetch_failure_carries_the_download_error(packs: Path, root: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]):
    fake = FakeDataWheel(fail="the source of xx.model answered 404")
    monkeypatch.setattr(TOOL, "_datawheel", lambda: fake)
    rows = TOOL.fetch_pack("xx", root)
    assert rows[0] == ("xx.model", "failed", "the source of xx.model answered 404")
    assert rows[1] == ("XxSans-Regular.ttf", "failed", "missing")
    assert TOOL.main(["--pack", "xx"]) == 1
    assert "answered 404" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["../xx.model", "sub/xx.model", ".hidden", "", None])
def test_unsafe_file_name_is_exit_2(packs: Path, root: Path, wheel: FakeDataWheel, name: Any,
                                    capsys: pytest.CaptureFixture[str]):
    bad = _manifest()
    bad["models"][0]["file"] = name
    (packs / "xx" / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(TOOL.ManifestProblem, match="unsafe file name"):
        TOOL.check_pack("xx", root)
    assert TOOL.main(["--pack", "xx"]) == 2
    assert wheel.calls == 0
    assert "unsafe file name" in capsys.readouterr().err


@pytest.mark.parametrize("pack", ["zz", "../xx", "XX", "x"])
def test_unknown_or_unsafe_pack_is_exit_2(packs: Path, root: Path, wheel: FakeDataWheel, pack: str):
    assert TOOL.main(["--check", "--pack", pack]) == 2
    assert TOOL.main(["--pack", pack]) == 2
    assert wheel.calls == 0


def test_unreadable_manifest_is_exit_2(packs: Path, root: Path):
    (packs / "xx" / "manifest.json").write_text("{not json", encoding="utf-8")
    assert TOOL.main(["--check", "--pack", "xx"]) == 2


def test_every_pack_by_default(packs: Path, root: Path, wheel: FakeDataWheel):
    (packs / "not-a-pack-id!").mkdir()
    assert TOOL.pack_ids() == ["xx"]
    assert TOOL.main([]) == 0 and wheel.calls == 1


# --------------------------------------------------------------------------- conftest's pinned_asset

@pytest.fixture
def fake_pack(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Pack ``xx`` as conftest sees it, with the models root at ``root``."""
    monkeypatch.setitem(conftest._MANIFESTS, "xx", _manifest())
    monkeypatch.setenv(conftest.MODELS_ENV, str(root))
    return root


def test_pinned_asset_returns_the_verified_path(fake_pack: Path):
    _fill(fake_pack)
    assert conftest.pinned_asset("xx", "xx-model") == fake_pack / "xx" / "xx.model"
    assert conftest.pinned_assets("xx:xx-model", "xx:xx-font") == {
        "xx:xx-model": fake_pack / "xx" / "xx.model", "xx:xx-font": fake_pack / "xx" / "XxSans-Regular.ttf"}


def test_pinned_asset_skips_on_a_missing_file(fake_pack: Path):
    with pytest.raises(pytest.skip.Exception, match=r"xx:xx-model .* is missing.*fetch_test_models\.py"):
        conftest.pinned_asset("xx", "xx-model")


def test_pinned_asset_skips_on_a_size_mismatch(fake_pack: Path):
    _fill(fake_pack, **{"xx.model": MODEL + b"!"})
    with pytest.raises(pytest.skip.Exception, match=f"{len(MODEL) + 1} bytes; the manifest pins {len(MODEL)}"):
        conftest.pinned_asset("xx", "xx-model")


def test_pinned_asset_skips_on_a_same_size_hash_mismatch(fake_pack: Path):
    _fill(fake_pack, **{"xx.model": bytes(len(MODEL))})
    got = hashlib.sha256(bytes(len(MODEL))).hexdigest()[:12]
    with pytest.raises(pytest.skip.Exception, match=f"has SHA-256 {got}"):
        conftest.pinned_asset("xx", "xx-model")


def test_pinned_asset_refuses_a_link_like_the_tool(packs: Path, fake_pack: Path, tmp_path: Path):
    _fill(fake_pack)
    real = tmp_path / "elsewhere.model"
    real.write_bytes(MODEL)
    (fake_pack / "xx" / "xx.model").unlink()
    (fake_pack / "xx" / "xx.model").symlink_to(real)
    with pytest.raises(pytest.skip.Exception, match="isn't a regular file"):
        conftest.pinned_asset("xx", "xx-model")
    states = {f: s for f, s, _ in TOOL.check_pack("xx", fake_pack)}
    assert states["xx.model"] == "wrong"                          # one rule in both places


def test_one_skip_names_every_missing_file(fake_pack: Path):
    with pytest.raises(pytest.skip.Exception) as info:
        conftest.pinned_assets("xx:xx-model", "xx:xx-font")
    assert "xx:xx-model" in str(info.value) and "xx:xx-font" in str(info.value)


@pytest.mark.parametrize("pack, key", [("xx", "nope"), ("qq", "anything")])
def test_unknown_pack_or_key_is_a_lookup_error(fake_pack: Path, pack: str, key: str):
    with pytest.raises(LookupError):
        conftest.pinned_asset(pack, key)


@pytest.mark.parametrize("ref", ["xx", ":xx-model", "xx:"])
def test_malformed_ref_is_a_lookup_error(fake_pack: Path, ref: str):
    with pytest.raises(LookupError):
        conftest.pinned_assets(ref)


def test_fetch_outside_a_network_test_fails(fake_pack: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PDFINTAKE_NETWORK_TESTS", "1")
    with pytest.raises(pytest.fail.Exception, match="needs a test marked network"):
        conftest.pinned_asset("xx", "xx-model", fetch=True)


def _network_fetch(monkeypatch: pytest.MonkeyPatch, rows: Any) -> list[str]:
    """Pretend to be in a network test, with the fetch tool replaced; returns the packs fetched."""
    fetched: list[str] = []

    def fetch_pack(pack: str, root: Path) -> list[tuple[str, str, str]]:
        fetched.append(pack)
        if isinstance(rows, Exception):
            raise rows
        return rows(root) if callable(rows) else rows

    monkeypatch.setattr(conftest, "_network_test", lambda: True)
    monkeypatch.setattr(conftest, "_fetch_tool",
                        lambda: types.SimpleNamespace(fetch_pack=fetch_pack, ManifestProblem=TOOL.ManifestProblem))
    return fetched


def test_network_fetch_failure_fails_with_the_download_error(fake_pack: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PDFINTAKE_NETWORK_TESTS", "1")
    fetched = _network_fetch(monkeypatch, [("xx.model", "failed", "the source of xx.model answered 404"),
                                           ("XxSans-Regular.ttf", "failed", "missing")])
    with pytest.raises(pytest.fail.Exception) as info:
        conftest.pinned_asset("xx", "xx-model", fetch=True)
    assert fetched == ["xx"]
    assert "answered 404" in str(info.value) and "xx:xx-model" in str(info.value)


def test_network_fetch_manifest_problem_fails(fake_pack: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PDFINTAKE_NETWORK_TESTS", "1")
    _network_fetch(monkeypatch, TOOL.ManifestProblem("xx: models entry 'xx-model' has an unsafe file name"))
    with pytest.raises(pytest.fail.Exception, match="unsafe file name"):
        conftest.pinned_asset("xx", "xx-model", fetch=True)


def test_network_fetch_that_works_returns_the_path(fake_pack: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PDFINTAKE_NETWORK_TESTS", "1")

    def rows(root: Path) -> list[tuple[str, str, str]]:
        _fill(root)
        return [("xx.model", "fetched", ""), ("XxSans-Regular.ttf", "fetched", "")]

    fetched = _network_fetch(monkeypatch, rows)
    assert conftest.pinned_asset("xx", "xx-model", fetch=True) == fake_pack / "xx" / "xx.model"
    assert fetched == ["xx"]


def test_network_test_without_the_switch_skips_and_fetches_nothing(fake_pack: Path,
                                                                   monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PDFINTAKE_NETWORK_TESTS", raising=False)
    fetched = _network_fetch(monkeypatch, [])
    with pytest.raises(pytest.skip.Exception):
        conftest.pinned_asset("xx", "xx-model", fetch=True)
    assert fetched == []


def test_network_test_follows_the_guard():
    assert conftest._network_test() is False                     # this test isn't marked network


def test_the_tool_imports_only_the_standard_library():
    import ast
    import sys

    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    tree = ast.parse(conftest.FETCH_TOOL.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in stdlib]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] not in stdlib:
            bad.append(node.module or "")
    assert not bad, f"tools/fetch_test_models.py must run from a fresh clone; it imports {bad}"
