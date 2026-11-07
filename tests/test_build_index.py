"""tools/build_index.py: when a pack version counts as published, and what --check does about packs/.

A pack version is published only once a release has shipped it, and ``tools/published-packs.json``
(written by ``build_index.py --record-release``, which ``scripts/release.sh`` runs) is the record of
that. Before any release the index is simply regenerated; after one, a published manifest can't
change under its version, a bump can. Everything here runs on the fixture packs (en, th, hu) in a
temporary folder, offline; a wheel is stood in for by monkeypatching ``_wheel_facts``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PACKS = ROOT / "tests" / "fixtures" / "packs"
FIXTURE_INDEX = ROOT / "tests" / "fixtures" / "index.test.json"
NEW_SHA = "5e" * 32


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("pdfintake_build_index_unit", ROOT / "tools" / "build_index.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bi() -> Any:
    return _load()


class Repo:
    """Fixture packs, a copy of the fixture index, an empty dist and a record path, in tmp."""

    def __init__(self, tmp: Path, bi: Any) -> None:
        self.bi = bi
        self.packs = tmp / "packs"
        shutil.copytree(FIXTURE_PACKS, self.packs)
        self.index = tmp / "index.json"
        shutil.copyfile(FIXTURE_INDEX, self.index)
        self.dist = tmp / "dist"
        self.dist.mkdir()
        self.record = tmp / "published-packs.json"

    def args(self, *extra: str, record: bool | None = None) -> list[str]:
        """The CLI arguments. ``--published`` names the record when it exists or ``--record-release``
        will create it; ``record`` True or False forces it on or off."""
        out = ["--packs", str(self.packs), "--index", str(self.index), "--wheels", str(self.dist)]
        if record if record is not None else (self.record.exists() or "--record-release" in extra):
            out += ["--published", str(self.record)]
        return out + list(extra)

    def manifest(self, pid: str) -> dict[str, Any]:
        return json.loads((self.packs / pid / "manifest.json").read_text(encoding="utf-8"))

    def write_manifest(self, pid: str, raw: dict[str, Any]) -> None:
        (self.packs / pid / "manifest.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    def edit(self, pid: str, **changes: Any) -> dict[str, Any]:
        """A content change to a pack's manifest (one more function word), plus any field changes."""
        raw = self.manifest(pid)
        raw["detect"]["function_words"]["words"].append("példa")
        raw.update(changes)
        self.write_manifest(pid, raw)
        return raw

    def index_data(self) -> dict[str, Any]:
        return json.loads(self.index.read_text(encoding="utf-8"))

    def write_index(self, data: dict[str, Any]) -> None:
        self.index.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def entry(self, pid: str) -> dict[str, Any] | None:
        return next((e for e in self.index_data()["packs"] if e["id"] == pid), None)

    def wheel(self, monkeypatch: pytest.MonkeyPatch, pid: str, version: str, sha: str = NEW_SHA) -> None:
        """A wheel for ``pid`` at ``version`` in dist, whose facts are ``sha``, 1234 bytes, 5678 installed."""
        from pdfintake.langpacks import index as I

        kind = "data" if pid == "en" else "lang"
        (self.dist / I.wheel_name(f"pdfintake-{kind}-{pid}", version)).write_bytes(b"a made-up wheel")
        monkeypatch.setattr(self.bi, "_wheel_facts", lambda path, raw: (sha, 1234, 5678))

    def release(self, version: str = "0.1.0") -> None:
        assert self.bi.main(self.args("--record-release", version)) == 0

    def set_entry(self, pid: str, change: Any) -> None:
        """Apply ``change(entry)`` to the index's entry for ``pid``, in place."""
        data = self.index_data()
        change(next(e for e in data["packs"] if e["id"] == pid))
        self.write_index(data)


@pytest.fixture
def repo(tmp_path: Path, bi: Any) -> Repo:
    return Repo(tmp_path, bi)


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


@pytest.fixture
def windows_text_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Text files written as Windows writes them: a text-mode write whose ``newline`` is left at its
    default turns each "\\n" into "\\r\\n" there. CPython fixes that default when it's built, so this
    opens such writes with ``newline="\\r\\n"``, which translates the same way. ``Path.write_text``,
    ``open`` and ``os.fdopen`` all go through ``io.open``."""
    import builtins
    import io

    real = io.open

    def windows_open(file: Any, mode: str = "r", buffering: int = -1, encoding: str | None = None,
                     errors: str | None = None, newline: str | None = None, closefd: bool = True,
                     opener: Any = None) -> Any:
        if newline is None and "b" not in mode and any(c in mode for c in "wax"):
            newline = "\r\n"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(io, "open", windows_open)
    monkeypatch.setattr(builtins, "open", windows_open)
    probe = tmp_path / "probe.txt"
    probe.write_text("a\n", encoding="utf-8")
    assert probe.read_bytes() == b"a\r\n"                  # the stand-in really writes as Windows does
    probe.unlink()


# --------------------------------------------------------------------------- before any release

def test_the_untouched_fixture_checks_with_or_without_a_record(repo: Repo, capsys: pytest.CaptureFixture[str]):
    assert repo.bi.main(repo.args("--check")) == 0
    assert "up to date (3 packs)" in _out(capsys) and not repo.record.exists()
    repo.release()
    assert repo.bi.main(repo.args("--check")) == 0
    assert repo.bi.main(repo.args("--check", "--strict")) == 0


def test_an_unreleased_manifest_edit_is_allowed(repo: Repo, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]):
    raw = repo.edit("hu")
    before = repo.index.read_bytes()
    # A pack PR's CI has no wheel: --check notes that the index is behind, and passes.
    assert repo.bi.main(repo.args("--check")) == 0, _out(capsys)
    out = _out(capsys)
    assert "note: hu: the 1.0.0 manifest changed since the index was generated" in out
    assert "under published version" not in out
    # The release's strict check refuses an index that's behind.
    assert repo.bi.main(repo.args("--check", "--strict")) == 1
    assert "regenerate the index with its wheel" in _out(capsys)
    # The maintainer regenerates from the wheels: no wheel, no regeneration...
    assert repo.bi.main(repo.args()) == 1
    assert "hu: version 1.0.0 has no wheel" in _out(capsys)
    assert repo.index.read_bytes() == before
    # ...and with it, the entry takes the new manifest and wheel under the same version.
    repo.wheel(monkeypatch, "hu", "1.0.0")
    assert repo.bi.main(repo.args()) == 0, _out(capsys)
    hu = repo.entry("hu")
    assert hu["version"] == "1.0.0" and hu["dist"]["sha256"] == NEW_SHA and hu["installed_bytes"] == 5678
    assert hu["detect"]["function_words"]["words"] == raw["detect"]["function_words"]["words"]
    assert repo.bi.main(repo.args("--check", "--strict")) == 0


def test_a_pack_the_index_doesnt_have_yet_is_a_note(repo: Repo, capsys: pytest.CaptureFixture[str]):
    shutil.copytree(repo.packs / "hu", repo.packs / "xx")
    repo.write_manifest("xx", dict(repo.manifest("xx"), id="xx", title="Example"))
    assert repo.bi.main(repo.args("--check")) == 0
    out = _out(capsys)
    assert "note: xx 1.0.0 isn't in the index yet" in out and "has no wheel" not in out
    assert [e["id"] for e in repo.index_data()["packs"]] == ["en", "th", "hu"]
    assert repo.bi.main(repo.args("--check", "--strict")) == 1
    assert "xx 1.0.0 isn't in the index yet" in _out(capsys)
    assert repo.bi.main(repo.args()) == 1
    assert "xx: version 1.0.0 has no wheel" in _out(capsys)


def test_a_new_pack_is_a_note_after_a_release_too(repo: Repo, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]):
    repo.release()
    shutil.copytree(repo.packs / "hu", repo.packs / "xx")
    repo.write_manifest("xx", dict(repo.manifest("xx"), id="xx", title="Example"))
    assert repo.bi.main(repo.args("--check")) == 0
    assert "note: xx 1.0.0 isn't in the index yet" in _out(capsys)
    # Added with its wheel, it's fetched from this core's release.
    from pdfintake import __version__

    repo.wheel(monkeypatch, "xx", "1.0.0")
    assert repo.bi.main(repo.args()) == 0, _out(capsys)
    xx = repo.entry("xx")
    assert xx["dist"]["url"].endswith(f"/download/v{__version__}/pdfintake_lang_xx-1.0.0-py3-none-any.whl")
    assert [e["id"] for e in repo.index_data()["packs"]] == ["en", "th", "hu", "xx"]


# --------------------------------------------------------------------------- after a release

def test_a_released_manifest_edit_is_refused(repo: Repo, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]):
    repo.release()
    repo.edit("hu")
    before = repo.index.read_bytes()
    want = "hu: the manifest changed under published version 1.0.0 (released in v0.1.0)"
    assert repo.bi.main(repo.args("--check")) == 1
    assert want in _out(capsys)
    assert repo.bi.main(repo.args()) == 1
    assert want in _out(capsys)
    # A wheel built from the edited manifest doesn't change that.
    repo.wheel(monkeypatch, "hu", "1.0.0")
    assert repo.bi.main(repo.args()) == 1
    assert want in _out(capsys)
    assert repo.index.read_bytes() == before


def test_a_released_wheel_never_changes(repo: Repo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    repo.release()
    repo.wheel(monkeypatch, "th", "1.0.0", sha="7a" * 32)
    assert repo.bi.main(repo.args()) == 1
    assert "differs from the 1.0.0 wheel released in v0.1.0" in _out(capsys)
    # The released bytes are fine.
    repo.wheel(monkeypatch, "th", "1.0.0", sha=repo.entry("th")["dist"]["sha256"])
    assert repo.bi.main(repo.args("--check")) == 0, _out(capsys)


def test_a_released_pack_can_be_bumped(repo: Repo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    repo.release()
    repo.edit("hu", version="1.0.1")
    # The PR: a note, not a failure, and the index keeps the published 1.0.0 until the next release.
    assert repo.bi.main(repo.args("--check")) == 0, _out(capsys)
    assert "note: hu 1.0.1 isn't in the index yet (it has 1.0.0)" in _out(capsys)
    assert repo.bi.main(repo.args("--check", "--strict")) == 1
    _out(capsys)
    # The maintainer regenerates with the new wheel; its URL is this core's release.
    from pdfintake import __version__

    repo.wheel(monkeypatch, "hu", "1.0.1")
    assert repo.bi.main(repo.args()) == 0, _out(capsys)
    hu = repo.entry("hu")
    assert hu["version"] == "1.0.1" and hu["dist"]["sha256"] == NEW_SHA
    assert hu["dist"]["url"].endswith(f"/download/v{__version__}/pdfintake_lang_hu-1.0.1-py3-none-any.whl")
    assert repo.bi.main(repo.args("--check", "--strict")) == 0
    # The next release records 1.0.1 beside 1.0.0.
    repo.release("0.2.0")
    record = json.loads(repo.record.read_text(encoding="utf-8"))
    assert record["releases"] == ["v0.1.0", "v0.2.0"]
    assert {v: r["release"] for v, r in record["packs"]["hu"].items()} == {"1.0.0": "v0.1.0", "1.0.1": "v0.2.0"}
    assert record["packs"]["th"]["1.0.0"]["release"] == "v0.1.0"


def test_a_bump_must_go_above_every_published_version(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.release()
    repo.edit("hu", version="0.9.0")
    assert repo.bi.main(repo.args("--check")) == 1
    assert "hu: version 0.9.0 isn't above 1.0.0, released in v0.1.0" in _out(capsys)


# --------------------------------------------------------------------------- the record, not the index

def test_an_index_edit_cant_fake_the_published_manifest(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.release()
    repo.edit("hu")
    # Point the index's entry at the edited manifest, as if 1.0.0 had always held it.
    data = repo.index_data()
    hu = next(e for e in data["packs"] if e["id"] == "hu")
    import hashlib

    hu["manifest_sha256"] = hashlib.sha256((repo.packs / "hu" / "manifest.json").read_bytes()).hexdigest()
    repo.write_index(data)
    assert repo.bi.main(repo.args("--check")) == 1
    assert "hu: the manifest changed under published version 1.0.0" in _out(capsys)


def test_dropping_a_pack_from_the_index_doesnt_unpublish_it(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.release()
    data = repo.index_data()
    data["packs"] = [e for e in data["packs"] if e["id"] != "hu"]
    repo.write_index(data)
    repo.edit("hu")
    assert repo.bi.main(repo.args("--check")) == 1
    assert "hu: the manifest changed under published version 1.0.0" in _out(capsys)
    assert repo.bi.main(repo.args()) == 1
    assert "under published version" in _out(capsys)


def test_a_published_pin_comes_back_from_the_record(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.release()
    pinned = repo.entry("th")["dist"]
    data = repo.index_data()
    next(e for e in data["packs"] if e["id"] == "th")["dist"]["sha256"] = "0d" * 32
    repo.write_index(data)
    assert repo.bi.main(repo.args("--check")) == 1
    assert "differs from a regeneration (th)" in _out(capsys)
    assert repo.bi.main(repo.args()) == 0
    assert repo.entry("th")["dist"] == pinned


def _probe_flip(e: dict[str, Any]) -> None:
    e["detect"]["probe"] = not e["detect"]["probe"]


@pytest.mark.parametrize("field, change", [
    ("dist.sha256", lambda e: e["dist"].update(sha256="0d" * 32)),
    ("dist.url", lambda e: e["dist"].update(url="https://example.com/other.whl")),
    ("manifest_sha256", lambda e: e.update(manifest_sha256="0d" * 32)),
    ("installed_bytes", lambda e: e.update(installed_bytes=e["installed_bytes"] + 999)),
    ("detect.probe", _probe_flip),
    ("detect.function_words", lambda e: e["detect"]["function_words"]["words"].append("példa")),
    ("title", lambda e: e.update(title="Something else")),
    ("scripts", lambda e: e.update(scripts=e["scripts"] + ["Greek"])),
    ("langid_codes", lambda e: e.update(langid_codes=e["langid_codes"] + ["xx"])),
])
def test_an_index_kept_behind_a_bump_must_be_the_published_entry(repo: Repo, capsys: pytest.CaptureFixture[str],
                                                                   field: str, change: Any):
    """While packs/ holds a bump, --check keeps the index's entry for the published version, and
    that entry must be, in every field, the one the release shipped."""
    repo.release()
    repo.edit("hu", version="1.0.1")
    assert repo.bi.main(repo.args("--check")) == 0, _out(capsys)
    _out(capsys)
    repo.set_entry("hu", change)
    assert repo.bi.main(repo.args("--check")) == 1, field
    assert "hu: the index's entry for 1.0.0 isn't the one the last release shipped (first released in v0.1.0)" \
        in _out(capsys)


def test_before_a_release_the_index_marks_nothing_published(repo: Repo, capsys: pytest.CaptureFixture[str]):
    """With no record, nothing in the index turns the guard on, not even release-asset URLs."""
    data = repo.index_data()
    for e in data["packs"]:
        if e["dist"]["url"]:
            e["dist"]["url"] = e["dist"]["url"].replace(f"/packs-{e['id']}-1.0.0/", "/v0.1.0/")
    repo.write_index(data)
    repo.edit("hu")
    assert repo.bi.main(repo.args("--check")) == 0, _out(capsys)
    assert "note: hu: the 1.0.0 manifest changed" in _out(capsys)


# --------------------------------------------------------------------------- --record-release

def test_recording_a_release_is_idempotent_and_deterministic(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.release()
    first = repo.record.read_bytes()
    assert "recorded v0.1.0" in _out(capsys)
    repo.release()
    assert "already records v0.1.0" in _out(capsys)
    assert repo.record.read_bytes() == first
    record = json.loads(first)
    assert record["record_version"] == 1 and record["releases"] == ["v0.1.0"]
    index = repo.index_data()
    for e in index["packs"]:
        rec = record["packs"][e["id"]][e["version"]]
        assert rec == {"release": "v0.1.0", "manifest_sha256": e["manifest_sha256"], "dist": e["dist"],
                       "installed_bytes": e["installed_bytes"], "index_entry_sha256": repo.bi.entry_sha256(e)}
    assert first.decode("utf-8") == repo.bi.render_record(record)


def test_the_record_and_the_index_are_the_same_bytes_when_written_on_windows(
        repo: Repo, windows_text_writes: None, capsys: pytest.CaptureFixture[str]):
    """Both files are committed and compared byte for byte (the record by the release, the index by
    SHA-256 once checked out with no line-end conversion), so a Windows run must write "\\n", not "\\r\\n"."""
    repo.release()
    record = json.loads(repo.record.read_bytes())
    assert repo.record.read_bytes() == repo.bi.render_record(record).encode("utf-8")
    assert repo.bi.main(repo.args()) == 0, _out(capsys)          # regenerates and writes the index
    index = repo.index_data()
    assert repo.index.read_bytes() == repo.bi.render(index).encode("utf-8")
    assert b"\r" not in repo.record.read_bytes() + repo.index.read_bytes()


def test_recording_refuses_an_older_release_and_changed_pins(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.release("0.2.0")
    _out(capsys)
    assert repo.bi.main(repo.args("--record-release", "0.1.0")) == 1
    assert "v0.1.0 isn't above the recorded releases (latest v0.2.0)" in _out(capsys)
    repo.release("0.3.0")
    before = repo.record.read_bytes()
    assert repo.bi.main(repo.args("--record-release", "0.2.0")) == 1   # recorded, but no longer the latest
    assert "v0.2.0 isn't above the recorded releases (latest v0.3.0)" in _out(capsys)
    assert repo.record.read_bytes() == before
    data = repo.index_data()
    next(e for e in data["packs"] if e["id"] == "th")["dist"]["bytes"] += 1
    repo.write_index(data)
    before = repo.record.read_bytes()
    assert repo.bi.main(repo.args("--record-release", "0.3.0")) == 1
    assert "th: 1.0.0 is recorded as published by v0.2.0 with other pins" in _out(capsys)
    assert repo.record.read_bytes() == before


def test_a_later_release_records_the_entry_it_ships_for_a_version_already_published(
        repo: Repo, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """A later core may render a published version's entry differently (a new detect copy, say);
    the pins stay, the entry hash follows the latest release, and a kept entry is checked against it."""
    repo.release()
    first = json.loads(repo.record.read_bytes())["packs"]["hu"]["1.0.0"]["index_entry_sha256"]
    repo.set_entry("hu", lambda e: e.update(title="Hungarian (re-rendered)"))
    repo.release("0.2.0")
    rec = json.loads(repo.record.read_bytes())["packs"]["hu"]["1.0.0"]
    assert rec["release"] == "v0.1.0" and rec["index_entry_sha256"] == repo.bi.entry_sha256(repo.entry("hu")) != first
    repo.edit("hu", version="1.0.1")
    assert repo.bi.main(repo.args("--check")) == 0, _out(capsys)
    repo.set_entry("hu", lambda e: e.update(title=repo.manifest("hu")["title"]))   # back to the v0.1.0 entry
    assert repo.bi.main(repo.args("--check")) == 1
    assert "isn't the one the last release shipped" in _out(capsys)


def test_recording_refuses_a_new_version_below_a_published_one(repo: Repo, capsys: pytest.CaptureFixture[str]):
    import hashlib

    repo.release()
    repo.edit("hu", version="0.9.0")
    sha = hashlib.sha256((repo.packs / "hu" / "manifest.json").read_bytes()).hexdigest()
    repo.set_entry("hu", lambda e: e.update(version="0.9.0", manifest_sha256=sha))
    before = repo.record.read_bytes()
    _out(capsys)
    assert repo.bi.main(repo.args("--record-release", "0.2.0")) == 1
    assert "hu: version 0.9.0 isn't above 1.0.0, released in v0.1.0" in _out(capsys)
    assert repo.record.read_bytes() == before


def test_recording_refuses_an_index_that_isnt_the_manifests(repo: Repo, capsys: pytest.CaptureFixture[str]):
    repo.edit("hu")
    assert repo.bi.main(repo.args("--record-release", "0.1.0")) == 1
    assert "hu: the index's manifest_sha256 isn't packs/hu/manifest.json's" in _out(capsys)
    assert not repo.record.exists()


@pytest.mark.parametrize("body, says", [
    ("not json", "isn't valid JSON"),
    ('{"record_version": 2, "releases": [], "packs": {}}', "record_version must be 1"),
    ('{"record_version": 1, "releases": ["0.1.0"], "packs": {}}', "releases must be a list of vX.Y.Z tags"),
    ('{"record_version": 1, "releases": ["v0.2.0", "v0.1.0"], "packs": {}}', "oldest first"),
    ('{"record_version": 1, "releases": ["v0.1.0"], "packs": {"hu": {"1.0.0": {"release": "v0.9.0"}}}}',
     "isn't in releases"),
    ('{"record_version": 1, "releases": ["v0.1.0"], "packs": {"hu": {"1.0.0": {"release": "v0.1.0"}}}}',
     "no index_entry_sha256"),
])
def test_a_malformed_record_stops_everything(repo: Repo, capsys: pytest.CaptureFixture[str], body: str, says: str):
    repo.record.write_text(body, encoding="utf-8")
    assert repo.bi.main(repo.args("--check")) == 1
    assert says in _out(capsys)


def test_a_named_record_that_isnt_there_is_an_error(repo: Repo, capsys: pytest.CaptureFixture[str]):
    """A mistyped --published, or a deleted record, must not quietly turn the guard off."""
    repo.release()
    repo.edit("hu")
    missing = str(repo.record.with_name("published-packs-typo.json"))
    for extra in (("--check",), ()):
        assert repo.bi.main(repo.args(*extra, "--published", missing, record=False)) == 2
        assert "published-packs-typo.json: no such file" in _out(capsys)
    assert repo.bi.main(repo.args("--check")) == 1
    out = _out(capsys)
    assert "published: published-packs.json records v0.1.0" in out and "under published version 1.0.0" in out


def test_check_says_which_record_it_read(repo: Repo, capsys: pytest.CaptureFixture[str]):
    assert repo.bi.main(repo.args("--check")) == 0
    assert "published: nothing yet (this index has no record), so no pack version is fixed" in _out(capsys)
    repo.record.write_text('{"record_version": 1, "releases": [], "packs": {}}', encoding="utf-8")
    assert repo.bi.main(repo.args("--check")) == 0
    assert "published: published-packs.json records no release yet" in _out(capsys)


def test_bad_flag_combinations(repo: Repo, capsys: pytest.CaptureFixture[str]):
    assert repo.bi.main(repo.args("--strict")) == 2
    assert repo.bi.main(repo.args("--check", "--record-release", "0.1.0")) == 2
    assert repo.bi.main(repo.args("--record-release", "0.1.0", record=False)) == 2
    assert repo.bi.main(repo.args("--record-release", "v0.1.0")) == 1
    assert "not a version: v0.1.0" in _out(capsys)


# --------------------------------------------------------------------------- which record applies

def test_the_packaged_index_reads_tools_published_packs_json_and_no_other_index_does(
        tmp_path: Path, bi: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """The repo's record governs the packaged index only: a test or scratch index elsewhere has no
    record unless --published names one, so fixture data never meets the real record."""
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE_PACKS, root / "packs")
    (root / "src/pdfintake/langpacks").mkdir(parents=True)
    shutil.copyfile(FIXTURE_INDEX, root / "src/pdfintake/langpacks/index.json")
    monkeypatch.setattr(bi, "ROOT", root)
    assert bi.main(["--record-release", "0.1.0"]) == 0
    assert (root / "tools" / "published-packs.json").is_file()
    raw = json.loads((root / "packs/hu/manifest.json").read_text(encoding="utf-8"))
    raw["detect"]["function_words"]["words"].append("példa")
    (root / "packs/hu/manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    _out(capsys)
    assert bi.main(["--check"]) == 1
    assert "under published version 1.0.0" in _out(capsys)
    elsewhere = tmp_path / "elsewhere.json"
    shutil.copyfile(FIXTURE_INDEX, elsewhere)
    assert bi.main(["--index", str(elsewhere), "--check"]) == 0
    assert "under published version" not in _out(capsys)
    (root / "tools" / "published-packs.json").unlink()
    assert bi.main(["--check"]) == 0
    assert "published: nothing yet (published-packs.json isn't there)" in _out(capsys)



def test_the_kit_writes_committed_pack_files_with_lf_on_every_os(tmp_path: Path, windows_text_writes: None):
    """manifest.json and bench.json (kit/bench.py write_json), corpus.json (kit/corpus.py save) and
    bench.private.json are committed, and a manifest's SHA-256 is pinned in the index; with
    `* -text` in .gitattributes, a kit run on Windows would otherwise commit "\\r\\n" bytes."""
    from pdfintake.kit import bench as B
    from pdfintake.kit import corpus as C

    B.write_json(tmp_path / "manifest.json", {"id": "xx", "list": [1, 2]})
    assert b"\r" not in (tmp_path / "manifest.json").read_bytes()
    C.save(C.empty(), tmp_path / "corpus.json")
    assert b"\r" not in (tmp_path / "corpus.json").read_bytes()
