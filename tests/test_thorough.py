"""thorough.py and ``pdfintake thorough``: the setting, what a run asks of it, and the model download.

The download runs against a loopback HTTPS stub (the store tests' ``HttpsStub``), with two made-up
files pinned in place of the real 4.4 GB model, through the module's seams (``BASE_URL``,
``_host_ok``, ``_port_allowed``, ``_ssl_context``, ``_sleep``); nothing reaches Hugging Face.
``conftest._thorough_isolated`` keeps the setting in a temp folder; ``PDFINTAKE_CACHE`` keeps the
model there too. The machine checks (Apple silicon, the libraries) are replaced where a test needs
them either way, so every test runs on every OS.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pdfintake import cli, thorough
from pdfintake.errors import IntegrityError, SetupError
from pdfintake.langpacks import store
from test_store import HttpsStub

FILE_A = b"made-up model config\n"
FILE_B = bytes(range(256)) * 40          # 10 KiB of made-up weights


def pin(name: str, data: bytes) -> thorough.ModelFile:
    return thorough.ModelFile(name, len(data), hashlib.sha256(data).hexdigest())


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "cache"
    monkeypatch.setenv("PDFINTAKE_CACHE", str(folder))
    return folder


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine can run the mode and has the libraries."""
    monkeypatch.setattr(thorough, "platform_problem", lambda: None)
    monkeypatch.setattr(thorough, "missing_libraries", lambda: [])


@pytest.fixture(scope="module")
def https(tmp_path_factory: pytest.TempPathFactory):
    stub = HttpsStub(tmp_path_factory.mktemp("tls"))
    yield stub
    stub.close()


@pytest.fixture
def hub(https: HttpsStub, cache: Path, mac: None, monkeypatch: pytest.MonkeyPatch) -> HttpsStub:
    """The stub as the model's host, serving two pinned made-up files; the canary passes."""
    https.routes.clear()
    https.hits.clear()
    files = (pin("config.json", FILE_A), pin("model.safetensors", FILE_B))
    monkeypatch.setattr(thorough, "FILES", files)
    monkeypatch.setattr(thorough, "TOTAL_BYTES", sum(f.bytes for f in files))
    monkeypatch.setattr(thorough, "BASE_URL", https.base)
    monkeypatch.setattr(thorough, "_host_ok", lambda host: host in ("127.0.0.1", "huggingface.co"))
    monkeypatch.setattr(thorough, "_port_allowed", lambda host, port: port in (None, https.port))
    monkeypatch.setattr(thorough, "_ssl_context", lambda: store._ssl_context(extra_cafile=str(https.cert)))
    sleeps: list[float] = []
    monkeypatch.setattr(thorough, "_sleep", sleeps.append)
    https.sleeps = sleeps  # type: ignore[attr-defined]
    from pdfintake.ocr.engines import dots

    monkeypatch.setattr(dots, "canary", lambda models: None)
    for f, data in zip(files, (FILE_A, FILE_B)):
        https.serve(path_of(f), data)
    return https


def path_of(f: thorough.ModelFile) -> str:
    return f"/{thorough.REPO}/resolve/{thorough.REVISION}/{f.name}"


# --------------------------------------------------------------------------- the setting

def test_no_setting_until_one_is_made():
    assert thorough.setting() is None
    thorough.decline()
    assert thorough.setting() == "declined"
    thorough.disable()
    assert thorough.setting() == "off"
    data = json.loads(thorough.setting_path().read_text(encoding="utf-8"))
    assert data == {"state": "off", "model": thorough.REVISION}


@pytest.mark.parametrize("raw", [b"not json", b'{"state": "maybe"}', b"[1, 2]", b'{"state": "on"}' + b" " * 5000])
def test_an_unreadable_setting_counts_as_none(raw: bytes):
    path = thorough.setting_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    assert thorough.setting() is None


def test_should_offer_only_where_it_runs_and_nobody_decided(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(thorough, "platform_problem", lambda: None)
    assert thorough.should_offer() is True
    thorough.decline()
    assert thorough.should_offer() is False
    thorough.setting_path().unlink()
    monkeypatch.setattr(thorough, "platform_problem", lambda: "thorough mode runs only on a Mac with Apple silicon")
    assert thorough.should_offer() is False


def test_resolve(cache: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(thorough, "platform_problem", lambda: None)
    monkeypatch.setattr(thorough, "missing_libraries", lambda: [])
    assert thorough.resolve(None) == (False, None)            # nobody asked, the setting isn't on
    assert thorough.resolve(False) == (False, None)
    on, why = thorough.resolve(True)
    assert not on and "pdfintake thorough enable" in (why or "")
    monkeypatch.setattr(thorough, "model_problems", lambda deep=False: [])
    assert thorough.resolve(True) == (True, None)
    thorough._write_setting("on")
    assert thorough.resolve(None) == (True, None) and thorough.resolve(False) == (False, None)
    monkeypatch.setattr(thorough, "missing_libraries", lambda: ["mlx"])
    assert thorough.resolve(None) == (False, "the thorough-mode libraries aren't installed (mlx)")
    monkeypatch.setattr(thorough, "platform_problem", lambda: "thorough mode runs only on a Mac with Apple silicon")
    assert thorough.resolve(None) == (False, "thorough mode runs only on a Mac with Apple silicon")


def test_status_off_the_mac(cache: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(thorough, "platform_problem", lambda: "thorough mode runs only on a Mac with Apple silicon")
    st = thorough.status()
    assert st["available"] is False and st["ready"] is False and st["on"] is False
    assert st["libraries"]["install"] is None and st["model"]["installed"] is False
    assert st["model"]["download_bytes"] == thorough.TOTAL_BYTES


def test_the_pins_add_up():
    assert thorough.TOTAL_BYTES == sum(f.bytes for f in thorough.FILES) == 4_426_897_073
    assert len({f.name for f in thorough.FILES}) == 13
    assert all(len(f.sha256) == 64 for f in thorough.FILES)
    from pdfintake.ocr.engines import dots

    assert set(dots.REQUIRED_FILES) <= {f.name for f in thorough.FILES}


# --------------------------------------------------------------------------- addresses

@pytest.mark.parametrize("url,ok", [
    ("https://huggingface.co/x/resolve/abc/model.safetensors", True),
    ("https://us.aws.cdn.hf.co/xet-bridge/abc", True),
    ("https://cas-bridge.xethub.hf.co/abc", True),
    ("http://huggingface.co/x", False),
    ("https://example.com/x", False),
    ("https://hf.co.example.com/x", False),
    ("https://examplehf.co/x", False),
    ("https://hf.co/x", False),
    ("https://huggingface.co:8443/x", False),
    ("https://user@huggingface.co/x", False),
    ("https://huggingface.co/a b", False),
])
def test_which_addresses_a_redirect_may_go_to(url: str, ok: bool):
    assert (thorough._url_problem(url) is None) is ok


# --------------------------------------------------------------------------- enable

def test_enable_downloads_checks_and_turns_it_on(hub: HttpsStub):
    seen: list[tuple[str, int, int]] = []
    st = thorough.enable(progress=lambda name, done, total: seen.append((name, done, total)))
    folder = thorough.model_dir()
    assert (folder / "config.json").read_bytes() == FILE_A and (folder / "model.safetensors").read_bytes() == FILE_B
    assert (folder / thorough.VERIFIED).read_text() == thorough.REVISION + "\n"
    assert st["on"] is True and st["ready"] is True and thorough.setting() == "on"
    assert seen[-1] == ("model.safetensors", len(FILE_B), len(FILE_B))
    assert not list((thorough.platforms.cache_root() / "staging").glob("thorough-*"))
    hits = dict(hub.hits)
    thorough.enable()                                  # installed: nothing is fetched again
    assert hub.hits == hits


def test_a_file_with_another_hash_is_never_kept(hub: HttpsStub):
    hub.serve(path_of(thorough.FILES[1]), bytes(reversed(FILE_B)))
    with pytest.raises(IntegrityError) as info:
        thorough.enable()
    assert info.value.code == "thorough_model_changed"
    assert not (thorough.model_dir() / "model.safetensors").exists()
    assert thorough.setting() is None


@pytest.mark.parametrize("body,length", [(FILE_B + b"x", True), (FILE_B + b"x", False)])
def test_more_bytes_than_the_pin_are_refused(hub: HttpsStub, body: bytes, length: bool):
    hub.serve(path_of(thorough.FILES[1]), body, length=length)
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_download_size"
    assert not (thorough.model_dir() / "model.safetensors").exists()


def test_a_redirect_off_hugging_face_is_refused(hub: HttpsStub):
    hub.redirect(path_of(thorough.FILES[1]), "https://example.com/model.safetensors")
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_redirect_refused"


def test_a_redirect_to_an_allowed_host_is_followed(hub: HttpsStub):
    hub.serve("/cdn/weights", FILE_B)
    hub.redirect(path_of(thorough.FILES[1]), f"{hub.base}/cdn/weights")
    assert thorough.enable()["on"] is True


def test_a_refusal_and_a_flaky_host(hub: HttpsStub):
    hub.fail(path_of(thorough.FILES[1]), 404)
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_download_refused" and info.value.detail["status"] == 404
    hub.fail(path_of(thorough.FILES[1]), 503)
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_download_failed"
    assert hub.sleeps == list(thorough.BACKOFF_S)  # type: ignore[attr-defined]


def test_no_room_downloads_nothing(hub: HttpsStub, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(thorough.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12, 1000))
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_no_room" and hub.hits == {}


def test_a_failed_canary_leaves_it_off(hub: HttpsStub, monkeypatch: pytest.MonkeyPatch):
    from pdfintake.ocr.engines import dots

    monkeypatch.setattr(dots, "canary", lambda models: "the test page's text didn't come back")
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_canary_failed"
    assert thorough.setting() is None and not (thorough.model_dir() / thorough.VERIFIED).exists()
    assert thorough.model_problems() == [thorough.VERIFIED]   # the files stay; enable checks again


def test_enable_refuses_where_it_cant_run(cache: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(thorough, "platform_problem", lambda: "thorough mode runs only on a Mac with Apple silicon")
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_not_supported"
    monkeypatch.setattr(thorough, "platform_problem", lambda: None)
    monkeypatch.setattr(thorough, "missing_libraries", lambda: ["mlx", "tokenizers"])
    with pytest.raises(SetupError) as info:
        thorough.enable()
    assert info.value.code == "thorough_libraries_missing" and "pdfintake[thorough]" in info.value.message


def test_a_changed_file_is_named_before_a_run_uses_it(hub: HttpsStub):
    thorough.enable()
    (thorough.model_dir() / "config.json").write_bytes(FILE_A + b"edited")
    with pytest.raises(IntegrityError) as info:
        thorough.model_paths()
    assert info.value.code == "thorough_model_changed"
    assert thorough.resolve(None)[0] is False


def test_disable_keeps_the_model_unless_told(hub: HttpsStub):
    thorough.enable()
    assert thorough.disable()["on"] is False and thorough.model_dir().is_dir()
    thorough.disable(remove=True)
    assert not thorough.model_dir().exists() and thorough.setting() == "off"


# --------------------------------------------------------------------------- the CLI

def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, Any]:
    code = cli.main(list(args))
    out = capsys.readouterr().out
    return code, json.loads(out) if "--json" in args else out


def test_cli_status_enable_disable_decline(hub: HttpsStub, capsys: pytest.CaptureFixture[str]):
    code, out = run_cli(capsys, "thorough", "status", "--json")
    assert code == 0 and out["exit"] == 0 and out["setting"] is None and out["model"]["installed"] is False
    code, out = run_cli(capsys, "thorough", "enable", "--json")
    assert code == 0 and out["on"] is True
    code, out = run_cli(capsys, "thorough", "status")
    assert code == 0 and "thorough mode: on" in out
    code, out = run_cli(capsys, "thorough", "disable", "--remove", "--json")
    assert code == 0 and out["setting"] == "off" and out["model"]["installed"] is False
    code, out = run_cli(capsys, "thorough", "decline", "--json")
    assert code == 0 and out["setting"] == "declined"


def test_cli_enable_reports_a_setup_error(cache: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(thorough, "platform_problem", lambda: None)
    monkeypatch.setattr(thorough, "missing_libraries", lambda: ["mlx"])
    code, out = run_cli(capsys, "thorough", "enable", "--json")
    assert code == 3 and out["error"]["code"] == "thorough_libraries_missing"


def test_cli_run_takes_one_thorough_flag(capsys: pytest.CaptureFixture[str]):
    code, out = run_cli(capsys, "run", "x.pdf", "--thorough", "--no-thorough", "--json")
    assert code == 2 and out["exit"] == 2


def test_cli_passes_the_flag_to_the_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    from pdfintake import flow

    seen: list[Any] = []
    monkeypatch.setattr(flow, "run", lambda inputs, opts: seen.append(opts.thorough) or [])
    for flag, want in (("--thorough", True), ("--no-thorough", False), (None, None)):
        run_cli(capsys, *(a for a in ("run", "x.pdf", flag, "--json") if a))
    assert seen == [True, False, None]


def test_cli_apply_takes_choice_thorough(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    from pdfintake import flow

    seen: list[Any] = []
    monkeypatch.setattr(flow, "apply", lambda rid, **kw: seen.append((rid, kw["choice"], kw["lang"])) or [])
    code, _ = run_cli(capsys, "apply", "r-20261110-1a2b3c4d", "--choice", "thorough", "--json")
    assert code == 0 and seen == [("r-20261110-1a2b3c4d", "thorough", None)]
