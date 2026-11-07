from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from pdfintake import platforms
from pdfintake.errors import EXIT_SETUP, SetupError

# sysconfig.get_platform() strings as each interpreter build reports them, and the row they map to.
TAGS = [
    ("macosx-11.0-arm64", "", "darwin-arm64"),
    ("macosx-15.0-arm64", "arm64", "darwin-arm64"),
    ("macosx-10.13-universal2", "arm64", "darwin-arm64"),
    ("macosx-10.13-universal2", "x86_64", "darwin-x86_64"),      # a universal2 Python under Rosetta
    ("macosx-10.9-x86_64", "", "darwin-x86_64"),
    ("macosx-10.6-intel", "x86_64", "darwin-x86_64"),
    ("win-amd64", "", "win-amd64"),
    ("win-arm64", "", "win-arm64"),
    ("linux-x86_64", "", "linux-x86_64"),
    ("linux-aarch64", "", "linux-aarch64"),
    ("linux-arm64", "", "linux-aarch64"),
]


@pytest.mark.parametrize("tag,machine,key", TAGS)
def test_key_mapping_for_each_sysconfig_string(tag: str, machine: str, key: str):
    plat = platforms._platform_for(tag, machine)
    assert plat.key == key
    assert plat in platforms.PLATFORMS


@pytest.mark.parametrize("tag,key", [("win32", "win-x86"), ("linux-i686", "linux-i686"),
                                     ("linux-armv7l", "linux-armv7l"), ("freebsd-14.1-RELEASE-amd64", "freebsd-amd64")])
def test_unlisted_platforms_are_unsupported_with_a_reason(tag: str, key: str):
    plat = platforms._platform_for(tag)
    assert plat.key == key and not plat.supported
    assert plat.notes == platforms.UNLISTED_NOTE
    assert plat not in platforms.PLATFORMS


def test_a_32_bit_python_is_unsupported_even_on_a_listed_platform():
    plat = platforms._platform_for("linux-x86_64", bits=32)
    assert not plat.supported and plat.notes == platforms.BITS_NOTE


def test_the_table_has_unique_keys_and_every_unsupported_row_says_why():
    keys = [p.key for p in platforms.PLATFORMS]
    assert len(keys) == len(set(keys))
    for p in platforms.PLATFORMS:
        assert p.supported or p.notes, p.key
        assert p.supported == bool(p.ci_runners), p.key
        assert p.sys_platform in {"darwin", "win32", "linux"}


@pytest.mark.parametrize("key", ["darwin-x86_64", "win-arm64"])
def test_require_supported_names_the_reason(monkeypatch: pytest.MonkeyPatch, key: str):
    row = next(p for p in platforms.PLATFORMS if p.key == key)
    monkeypatch.setattr(platforms, "current", lambda: row)
    with pytest.raises(SetupError) as info:
        platforms.require_supported()
    err = info.value
    assert err.exit_code == EXIT_SETUP and err.code == "unsupported_platform"
    assert row.notes in str(err) and key in str(err)


def test_require_supported_names_an_unlisted_platform(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(platforms, "current", lambda: platforms._platform_for("linux-riscv64"))
    with pytest.raises(SetupError, match="linux-riscv64: pdfintake has no build"):
        platforms.require_supported()


def test_current_runs_here_and_matches_sys_platform():
    plat = platforms.current()
    assert plat.sys_platform == sys.platform or not plat.supported
    if plat.supported:
        assert platforms.require_supported() is plat


def test_current_reads_sysconfig(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(platforms.sysconfig, "get_platform", lambda: "win-arm64")
    assert platforms.current().key == "win-arm64"


def test_macos_version(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(platforms.sys, "platform", "darwin")
    for release, want in [("15.4", (15, 4)), ("27.2", (27, 2)), ("26", (26, 0)), ("", None), ("x.y", None)]:
        monkeypatch.setattr(platforms._platform, "mac_ver", lambda r=release: (r, ("", "", ""), ""))
        assert platforms.macos_version() == want
    monkeypatch.setattr(platforms.sys, "platform", "linux")
    assert platforms.macos_version() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_version_reads_this_mac():
    ver = platforms.macos_version()
    assert ver is not None and ver[0] >= 11


@pytest.mark.parametrize("name", ["cache", "Ünïcødé キャッシュ ไทย 📎"])
def test_cache_root_honours_pdfintake_cache_including_non_ascii(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                 name: str):
    target = tmp_path / name
    monkeypatch.setenv("PDFINTAKE_CACHE", str(target))
    root = platforms.cache_root()
    assert root == target and root.is_absolute()
    assert not root.exists(), "cache_root never creates the folder"
    from pdfintake import fsutil

    made = fsutil.work_dir("work", "abc-0.1.0")
    assert made == target / "work" / "abc-0.1.0" and made.is_dir()
    (made / "page-1.png").write_bytes(b"x")
    assert (target / "work" / "abc-0.1.0" / "page-1.png").read_bytes() == b"x"


def test_a_relative_override_is_made_absolute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDFINTAKE_CACHE", "rel-cache")
    root = platforms.cache_root()
    assert root.is_absolute() and root == Path(os.getcwd()) / "rel-cache"


@pytest.mark.parametrize("plat,env,want", [
    ("darwin", {}, ("Library", "Caches", "pdfintake")),
    ("linux", {}, (".cache", "pdfintake")),
    ("linux", {"XDG_CACHE_HOME": "relative/xdg"}, (".cache", "pdfintake")),
    ("win32", {}, ("AppData", "Local", "pdfintake", "Cache")),
])
def test_default_cache_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plat: str, env: dict, want: tuple):
    monkeypatch.delenv("PDFINTAKE_CACHE", raising=False)
    for var in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(platforms.sys, "platform", plat)
    monkeypatch.setattr(platforms.Path, "home", classmethod(lambda cls: tmp_path))
    assert platforms.cache_root() == tmp_path.joinpath(*want)


def test_xdg_and_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("PDFINTAKE_CACHE", raising=False)
    monkeypatch.setattr(platforms.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert platforms.cache_root() == tmp_path / "xdg" / "pdfintake"
    monkeypatch.setattr(platforms.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Lokale Daten"))
    assert platforms.cache_root() == tmp_path / "Lokale Daten" / "pdfintake" / "Cache"
