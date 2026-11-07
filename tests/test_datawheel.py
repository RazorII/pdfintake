"""datawheel.py: reproducible data-only wheels, their checks, and the pinned-source downloader.

Every pack here is made up: model files are short texts, fonts are fontTools box fonts, the canary
is a generated PNG, and source URLs point at an allowlisted host but are never really fetched
(downloads go to a local HTTP server through the ``_open_url`` seam).
"""

from __future__ import annotations

import hashlib
import http.server
import io
import socket
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from fixtures import build as fx
from pdfintake import datawheel

SRC = Path(__file__).resolve().parents[1] / "src"
SCHEMA = SRC / "pdfintake" / "langpacks" / "manifest.schema.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_pack(root: Path, pack_id: str = "xa", *, base: bool = False, version: str = "1.0.0",
              models: dict[str, bytes] | None = None, font: bool = True, canary: bool = True) -> dict[str, Any]:
    """A made-up pack: ``root/pack`` (manifest and notices) and ``root/files`` (models, fonts, canary)."""
    pack_dir, files = root / "pack", root / "files"
    pack_dir.mkdir(parents=True, exist_ok=True)
    files.mkdir(parents=True, exist_ok=True)
    models = models if models is not None else {f"{pack_id}.traineddata": b"Example model text 4521\n"}
    manifest: dict[str, Any] = {
        "schema": 1, "id": pack_id, "version": version, "requires_core": ">=0.1.0", "base": base, "title": "Example",
        "languages": [{"tag": "xa", "iso639_3": "xaa", "role": "primary"}], "scripts": ["Latin"],
        "detect": {"osd_scripts": ["Latin"]}, "readers": {}, "normalize": ["nfc"], "repairs": [], "fills": [],
        "models": [], "fonts": [], "must_survive": [], "bench": {"kit": "1.0", "corpus": "none", "platforms": {}},
    }
    for name, data in models.items():
        (files / name).write_bytes(data)
        manifest["models"].append({
            "key": name.split(".")[0], "file": name, "engine": "tesseract", "kind": "traineddata",
            "bytes": len(data), "sha256": sha(data),
            "source": f"https://github.com/tesseract-ocr/tessdata_best/raw/{COMMIT}/{name}",
            "licence": "Apache-2.0", "notice": "NOTICE-example"})
    (pack_dir / "NOTICE-example").write_text("Example notice for made-up test models.\n", encoding="utf-8")
    if font:
        data = fx.box_font("ABC")
        (files / "ExampleSans-Regular.ttf").write_bytes(data)
        manifest["fonts"].append({
            "key": "example-sans", "file": "ExampleSans-Regular.ttf", "covers": ["Latin"], "bytes": len(data),
            "sha256": sha(data), "licence": "OFL-1.1",
            "source": "https://github.com/notofonts/latin-greek-cyrillic/releases/download/v1/ExampleSans-Regular.ttf"})
        (pack_dir / "OFL.txt").write_text("Example OFL text for a made-up font.\n", encoding="utf-8")
    if canary:
        data = fx.png_bytes(fx.text_like(240, 60, lines=1), dpi=300)
        (files / f"canary-{pack_id}.png").write_bytes(data)
        manifest["canary"] = {"file": f"canary-{pack_id}.png", "bytes": len(data), "sha256": sha(data), "dpi": 300,
                              "must_find": ["4521"]}
    write_manifest(pack_dir, manifest)
    return manifest


def write_manifest(pack_dir: Path, manifest: dict[str, Any]) -> bytes:
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (pack_dir / "manifest.json").write_bytes(raw)
    return raw


def build_pack(tmp: Path, **kw: Any) -> tuple[Path, dict[str, Any]]:
    manifest = make_pack(tmp, **kw)
    wheel = datawheel.build(tmp / "pack", tmp / "files", tmp / "out", dist_name=datawheel._dist_name(manifest))
    return wheel, manifest


def rewrite(wheel: Path, dest: Path, *, add: dict[str, bytes] | None = None, drop: tuple[str, ...] = (),
            replace: dict[str, bytes] | None = None, fix_record: bool = True,
            attrs: dict[str, int] | None = None, stored: dict[str, str] | None = None) -> Path:
    """A copy of ``wheel`` with members added, dropped or replaced; RECORD rewritten to match unless told not to.

    Every name is stored exactly as given: ``zipfile.ZipInfo`` would turn a backslash into "/" on
    Windows and cut a name at a NUL. ``stored`` maps a member to another name to store it under,
    while RECORD keeps listing it under its own."""
    with zipfile.ZipFile(wheel) as zf:
        members = [(zi.orig_filename, zf.read(zi)) for zi in zf.infolist()]
    members = [(n, (replace or {}).get(n, d)) for n, d in members if n not in drop]
    members += list((add or {}).items())
    record = next(n for n, _ in members if n.endswith(".dist-info/RECORD"))
    if fix_record:
        lines = [f"{n},{datawheel._record_hash(sha(d))},{len(d)}" for n, d in members if n != record]
        lines.append(f"{record},,")
        members = [(n, ("\n".join(lines) + "\n").encode() if n == record else d) for n, d in members]
    with zipfile.ZipFile(dest, "w") as zf:
        for n, d in members:
            zi = zipfile.ZipInfo(n, date_time=datawheel.ZIP_DATE)
            zi.filename = (stored or {}).get(n, n)          # verbatim, on every OS
            zi.external_attr = (attrs or {}).get(n, datawheel.FILE_MODE) << 16
            zf.writestr(zi, d)
    return dest


# --------------------------------------------------------------------------- building

def test_building_twice_gives_identical_bytes(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    first = wheel.read_bytes()
    for path in (tmp_path / "files").iterdir():               # times and modes must not leak into the wheel
        os.utime(path, (1_900_000_000, 1_900_000_000))
        os.chmod(path, 0o600)
    again = datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out2",
                            dist_name=datawheel._dist_name(manifest))
    assert again.read_bytes() == first
    assert again.name == wheel.name == "pdfintake_lang_xa-1.0.0-py3-none-any.whl"
    assert datawheel.wheel_problems(wheel, manifest) == []


def test_the_wheel_holds_exactly_the_pack_layout(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        infos = zf.infolist()
        raw = zf.read("pdfintake_lang_xa/manifest.json")
        meta = zf.read("pdfintake_lang_xa-1.0.0.dist-info/METADATA").decode()
    assert names == [
        "pdfintake_lang_xa/NOTICE-example", "pdfintake_lang_xa/OFL.txt", "pdfintake_lang_xa/canary-xa.png",
        "pdfintake_lang_xa/fonts/ExampleSans-Regular.ttf", "pdfintake_lang_xa/manifest.json",
        "pdfintake_lang_xa/models/xa.traineddata",
        "pdfintake_lang_xa-1.0.0.dist-info/METADATA", "pdfintake_lang_xa-1.0.0.dist-info/WHEEL",
        "pdfintake_lang_xa-1.0.0.dist-info/RECORD"]
    assert raw == (tmp_path / "pack" / "manifest.json").read_bytes()       # the index's manifest pin holds for both
    assert all(zi.compress_type == zipfile.ZIP_STORED and zi.date_time == datawheel.ZIP_DATE for zi in infos)
    assert "Name: pdfintake-lang-xa" in meta and "Version: 1.0.0" in meta
    assert "License: Apache-2.0 AND OFL-1.1" in meta
    assert "http" not in meta                                  # no host appears in a wheel's metadata
    assert not any(n.endswith(".py") for n in names)


def test_the_base_pack_is_the_data_distribution(tmp_path: Path):
    wheel, _ = build_pack(tmp_path, pack_id="en", base=True)
    assert wheel.name == "pdfintake_data_en-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as zf:
        assert "pdfintake_data_en/models/en.traineddata" in zf.namelist()


def test_build_refuses_files_that_dont_match_their_pins(tmp_path: Path):
    manifest = make_pack(tmp_path)
    good = (tmp_path / "files" / "xa.traineddata").read_bytes()
    (tmp_path / "files" / "xa.traineddata").write_bytes(good[:-1] + b"X")
    with pytest.raises(datawheel.DataWheelError, match="xa.traineddata has SHA-256"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-lang-xa")
    (tmp_path / "files" / "xa.traineddata").write_bytes(good)
    (tmp_path / "files" / "canary-xa.png").unlink()
    with pytest.raises(datawheel.DataWheelError, match="kit wheel"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-lang-xa")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())
    with pytest.raises(datawheel.DataWheelError, match="pdfintake-lang-xa"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-data-xa")
    assert manifest["id"] == "xa"


def test_build_refuses_a_missing_notice(tmp_path: Path):
    make_pack(tmp_path)
    (tmp_path / "pack" / "NOTICE-example").unlink()
    with pytest.raises(datawheel.DataWheelError, match="NOTICE-example is missing"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-lang-xa")


def test_path_patterns_agree_with_the_schema():
    defs = json.loads(SCHEMA.read_text(encoding="utf-8"))["$defs"]
    props = json.loads(SCHEMA.read_text(encoding="utf-8"))["properties"]

    def bare(p: str) -> str:
        assert p.startswith("^") and p.endswith("$")
        return p[1:-1]

    assert datawheel.MODEL_FILE_RE.pattern == bare(defs["model"]["properties"]["file"]["pattern"])
    assert datawheel.FONT_FILE_RE.pattern == bare(defs["font"]["properties"]["file"]["pattern"])
    assert datawheel.CANARY_FILE_RE.pattern == bare(defs["canary"]["properties"]["file"]["pattern"])
    assert datawheel.NOTICE_RE.pattern == bare(defs["model"]["properties"]["notice"]["pattern"])
    assert datawheel.SHA_RE.pattern == bare(defs["sha256"]["pattern"])
    assert datawheel.ID_RE.pattern == bare(props["id"]["pattern"])
    assert datawheel.VERSION_RE.pattern == bare(props["version"]["pattern"])
    assert datawheel.TITLE_RE.pattern == bare(props["title"]["pattern"])
    assert defs["model"]["properties"]["bytes"]["maximum"] == datawheel.MAX_MODEL_BYTES
    assert defs["font"]["properties"]["bytes"]["maximum"] == datawheel.MAX_FONT_BYTES
    assert defs["canary"]["properties"]["bytes"]["maximum"] == datawheel.MAX_CANARY_BYTES


# --------------------------------------------------------------------------- wheel_problems

@pytest.mark.parametrize("name", ["pdfintake_lang_xa/models/loader.py", "pdfintake_lang_xa/hook.pth", "evil.pth",
                                  "pdfintake_lang_xa/models/native.so", "pdfintake_lang_xa/lib.dylib",
                                  "pdfintake_lang_xa/lib.DLL", "pdfintake_lang_xa/models/cache.pyc"])
def test_wheel_problems_flag_code(tmp_path: Path, name: str):
    wheel, manifest = build_pack(tmp_path)
    bad = rewrite(wheel, tmp_path / "bad.whl", add={name: b"print('made-up')\n"})
    problems = datawheel.wheel_problems(bad, manifest)
    assert any("is code" in p and name in p for p in problems), problems


@pytest.mark.parametrize("case,expect", [
    ("dotdot", "'..'"), ("absolute", "absolute"), ("backslash", "backslash"), ("extra", "isn't part of this pack"),
    ("missing", "is missing"), ("hash", "the manifest pins"), ("record", "RECORD"), ("symlink", "isn't a regular file"),
])
def test_wheel_problems_flag_unsafe_unknown_and_wrong_members(tmp_path: Path, case: str, expect: str):
    wheel, manifest = build_pack(tmp_path)
    model = "pdfintake_lang_xa/models/xa.traineddata"
    kw: dict[str, Any] = {
        "dotdot": {"add": {"pdfintake_lang_xa/../escape.txt": b"x"}},
        "absolute": {"add": {"/etc/escape.txt": b"x"}},
        "backslash": {"add": {"pdfintake_lang_xa\\models\\x.txt": b"x"}},
        "extra": {"add": {"pdfintake_lang_xa/models/extra.bin": b"x"}},
        "missing": {"drop": (model,)},
        "hash": {"replace": {model: b"Example model text 4529\n"}},
        "record": {"replace": {model: b"Example model text 4529\n"}, "fix_record": False},
        "symlink": {"add": {"pdfintake_lang_xa/models/link.bin": b"xa.traineddata"},
                    "attrs": {"pdfintake_lang_xa/models/link.bin": 0o120777}},
    }[case]
    bad = rewrite(wheel, tmp_path / "bad.whl", **kw)
    problems = datawheel.wheel_problems(bad, manifest)
    assert any(expect in p for p in problems), problems


@pytest.mark.parametrize("sep", ["/", "\\"], ids=["posix", "windows"])
@pytest.mark.parametrize("stored", ["pdfintake_lang_xa\\models\\xa.traineddata",
                                    "pdfintake_lang_xa/models/xa.traineddata\x00.txt"], ids=["backslash", "nul"])
def test_a_member_stored_under_a_name_zipfile_cleans_up_is_refused_on_every_os(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sep: str, stored: str):
    """The model stored under a name zipfile reads back as the model's (it cuts at a NUL, and on
    Windows turns each "\\" into "/"), with RECORD listing the model as usual: a check of zipfile's
    name alone accepts that wheel there and refuses it elsewhere. ``os.sep`` set to "\\" is how
    zipfile names members on Windows."""
    wheel, manifest = build_pack(tmp_path)
    model = "pdfintake_lang_xa/models/xa.traineddata"
    bad = rewrite(wheel, tmp_path / "bad.whl", stored={model: stored})
    with zipfile.ZipFile(bad) as zf:
        assert stored in [zi.orig_filename for zi in zf.infolist()]        # the archive holds that name
    monkeypatch.setattr(os, "sep", sep)
    monkeypatch.setattr(os, "altsep", "/" if sep == "\\" else None)
    with zipfile.ZipFile(bad) as zf:
        assert (model in zf.namelist()) is (sep == "\\" or "\x00" in stored)   # zipfile's view of it
    problems = datawheel.wheel_problems(bad, manifest)
    assert f"member {stored!r} is refused: a NUL or a backslash in the path" in problems, problems


def test_wheel_problems_reject_a_different_manifest_and_a_non_zip(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    other = dict(manifest, title="Another")
    assert any("isn't the expected manifest" in p for p in datawheel.wheel_problems(wheel, other))
    junk = tmp_path / "junk.whl"
    junk.write_bytes(b"<html>Gateway Timeout</html>")
    assert any("isn't a readable wheel" in p for p in datawheel.wheel_problems(junk, manifest))


def test_a_member_whose_header_lies_about_its_size_is_capped(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    model = "pdfintake_lang_xa/models/xa.traineddata"
    big = b"0" * (5 * 1024 * 1024)
    bad = rewrite(wheel, tmp_path / "bad.whl", replace={model: big})
    problems = datawheel.wheel_problems(bad, manifest)
    assert any("larger than" in p and model in p for p in problems), problems


# --------------------------------------------------------------------------- dist_problems

def installed_files(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        names = [n for n in zf.namelist()]
    di = next(n.split("/")[0] for n in names if ".dist-info/" in n)
    return names + [f"{di}/INSTALLER", f"{di}/REQUESTED", f"{di}/direct_url.json"]


def test_dist_problems_accept_an_installed_data_wheel(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    assert datawheel.dist_problems(installed_files(wheel), manifest) == []


@pytest.mark.parametrize("extra,expect", [
    ("pdfintake_lang_xa/models/loader.py", "is code"), ("evil.pth", "is code"), ("zzz_hook.pth", "is code"),
    ("pdfintake_lang_xa/native.so", "is code"), ("pdfintake/cli.py", "is code"),
    ("pdfintake/data.json", "lies outside"), ("../../bin/example-tool", "'..'"),
    ("pdfintake_lang_xa/models/extra.bin", "isn't part of this pack"),
    ("pdfintake_lang_xa-1.0.0.dist-info/entry_points.py", "is code"),
])
def test_dist_problems_flag_code_and_strays(tmp_path: Path, extra: str, expect: str):
    wheel, manifest = build_pack(tmp_path)
    problems = datawheel.dist_problems(installed_files(wheel) + [extra], manifest)
    assert any(expect in p for p in problems), problems


def test_dist_problems_flag_a_missing_member(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    files = [f for f in installed_files(wheel) if not f.endswith("xa.traineddata")]
    assert any("is missing" in p for p in datawheel.dist_problems(files, manifest))


def test_an_installed_wheel_reads_back_through_importlib_metadata(tmp_path: Path):
    import importlib.metadata

    wheel, manifest = build_pack(tmp_path)
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(site)
    dists = list(importlib.metadata.distributions(path=[str(site)]))
    assert [d.metadata["Name"] for d in dists] == ["pdfintake-lang-xa"]
    assert dists[0].version == "1.0.0"
    assert datawheel.dist_problems([str(f) for f in dists[0].files or ()], manifest) == []


# --------------------------------------------------------------------------- download_sources

class _Server:
    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes]] = {}
        self.hits: dict[str, int] = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                outer.hits[self.path] = outer.hits.get(self.path, 0) + 1
                status, body = outer.routes.get(self.path, (404, b"<html>Not Found</html>"))
                self.send_response(status)
                self.send_header("Content-Type", "text/html" if status != 200 else "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch):
    srv = _Server()

    def local(url: str, timeout: float) -> Any:
        assert url.startswith("https://github.com/")
        return urllib.request.urlopen(srv.base + url[len("https://github.com"):], timeout=timeout)

    monkeypatch.setattr(datawheel, "_open_url", local)
    monkeypatch.setattr(datawheel, "_sleep", lambda s: None)
    yield srv
    srv.close()


def _paths(manifest: dict[str, Any]) -> dict[str, str]:
    return {item["file"]: item["source"][len("https://github.com"):] for g in ("models", "fonts") for item in manifest[g]}


def test_download_sources_fetches_checks_and_keeps_good_files(tmp_path: Path, server: _Server):
    manifest = make_pack(tmp_path / "src")
    for file, path in _paths(manifest).items():
        server.routes[path] = (200, (tmp_path / "src" / "files" / file).read_bytes())
    dest = tmp_path / "dest"
    got = datawheel.download_sources(manifest, dest)
    assert [p.name for p in got] == ["xa.traineddata", "ExampleSans-Regular.ttf"]
    assert all(p.read_bytes() == (tmp_path / "src" / "files" / p.name).read_bytes() for p in got)
    hits = dict(server.hits)
    datawheel.download_sources(manifest, dest)                 # already there: nothing is fetched again
    assert server.hits == hits
    assert sorted(os.listdir(dest)) == ["ExampleSans-Regular.ttf", "xa.traineddata"]


@pytest.mark.parametrize("status,body", [(504, b"<html><body>504 Gateway Time-out</body></html>"),
                                         (200, b"<html><body>Please sign in</body></html>")])
def test_an_error_body_is_never_kept_as_a_model(tmp_path: Path, server: _Server, status: int, body: bytes):
    manifest = make_pack(tmp_path / "src")
    paths = _paths(manifest)
    server.routes[paths["xa.traineddata"]] = (status, body)
    dest = tmp_path / "dest"
    with pytest.raises(datawheel.DataWheelError, match="nothing was kept") as info:
        datawheel.download_sources(manifest, dest)
    assert server.hits[paths["xa.traineddata"]] == datawheel.ATTEMPTS
    msg = str(info.value)                                       # §11.2, decided 2026-11-05
    assert "after three attempts" in msg and "retries" not in msg and "3 attempts" not in msg
    assert "GitHub (github.com, or the githubusercontent.com hosts it redirects to)" in msg
    assert "pythonhosted" not in msg.lower() and "pypi" not in msg.lower()
    assert os.listdir(dest) == []                               # no model, no partial file


def test_a_wrong_file_already_in_place_is_replaced(tmp_path: Path, server: _Server):
    manifest = make_pack(tmp_path / "src")
    for file, path in _paths(manifest).items():
        server.routes[path] = (200, (tmp_path / "src" / "files" / file).read_bytes())
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "xa.traineddata").write_bytes(b"<html>an old error page</html>")
    datawheel.download_sources(manifest, dest)
    assert (dest / "xa.traineddata").read_bytes() == (tmp_path / "src" / "files" / "xa.traineddata").read_bytes()


def test_a_source_off_the_allowlist_is_never_fetched(tmp_path: Path, server: _Server):
    manifest = make_pack(tmp_path / "src")
    manifest["models"][0]["source"] = "https://raw.example.com/models/xa.traineddata"
    with pytest.raises(datawheel.DataWheelError, match="allowed source"):
        datawheel.download_sources(manifest, tmp_path / "dest")
    manifest["models"][0]["source"] = f"https://github.com/tesseract-ocr/tessdata_best/raw/{COMMIT}/xa.traineddata?x=1"
    with pytest.raises(datawheel.DataWheelError):
        datawheel.download_sources(manifest, tmp_path / "dest")
    assert server.hits == {}


# --------------------------------------------------------------------------- from a fresh clone

@pytest.mark.parametrize("flags", [("-S",), (), ("-I", "-S")], ids=["no-site", "as-dev-install-runs-it", "isolated"])
def test_datawheel_runs_from_a_fresh_clone_with_the_standard_library_only(tmp_path: Path, flags: tuple[str, ...]):
    # The clone keeps every top-level module beside datawheel.py, as a real clone does: run by path,
    # src/pdfintake/ is first on sys.path, and its types.py must not shadow the standard library's.
    clone = tmp_path / "clone" / "src" / "pdfintake"
    (clone / "langpacks").mkdir(parents=True)
    for module in (SRC / "pdfintake").glob("*.py"):
        shutil.copy2(module, clone / module.name)
    assert (clone / "types.py").is_file()
    shutil.copy2(SCHEMA, clone / "langpacks" / "manifest.schema.json")
    manifest = make_pack(tmp_path / "pk")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run = lambda *args: subprocess.run(  # noqa: E731
        [sys.executable, *flags, str(clone / "datawheel.py"), *args],
        capture_output=True, text=True, encoding="utf-8", timeout=120, env=env, cwd=tmp_path / "clone")
    out = run("build", str(tmp_path / "pk" / "pack"), "--out", str(tmp_path / "dist"),
              "--files", str(tmp_path / "pk" / "files"), "--offline")
    assert out.returncode == 0, out.stderr + out.stdout
    report = json.loads(out.stdout)
    wheel = Path(report["wheel"])
    assert report["distribution"] == "pdfintake-lang-xa" and report["sha256"] == sha(wheel.read_bytes())
    in_process = datawheel.build(tmp_path / "pk" / "pack", tmp_path / "pk" / "files", tmp_path / "dist2",
                                 dist_name="pdfintake-lang-xa")
    assert in_process.read_bytes() == wheel.read_bytes()
    checked = run("check", str(wheel), "--pack", str(tmp_path / "pk" / "pack"))
    assert checked.returncode == 0 and json.loads(checked.stdout)["problems"] == []
    bad = rewrite(wheel, tmp_path / "bad.whl", add={"pdfintake_lang_xa/models/x.py": b"pass\n"})
    checked = run("check", str(bad), "--pack", str(tmp_path / "pk" / "pack"))
    assert checked.returncode == 1 and json.loads(checked.stdout)["problems"]
    assert run("build").returncode == 2
    assert manifest["id"] == "xa"


def test_main_reports_a_failed_build_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    make_pack(tmp_path)
    (tmp_path / "files" / "xa.traineddata").unlink()
    code = datawheel.main(["build", str(tmp_path / "pack"), "--out", str(tmp_path / "out"),
                           "--files", str(tmp_path / "files"), "--offline"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False and "xa.traineddata is missing" in out["error"]


def test_datawheel_source_has_no_hosts_beyond_the_schema():
    text = (SRC / "pdfintake" / "datawheel.py").read_text(encoding="utf-8")
    hosts = set(re.findall(r"https?://([A-Za-z0-9.-]+)", text))
    assert hosts == set(), hosts                                # hosts live in the schema, index.json and pyproject only


# --------------------------------------------------------------------------- review fixes: build writes data only

@pytest.mark.parametrize("name", ["hook.pth", "loader.so", "evil.py", "lib.dylib", "mod.cpython-312-darwin.so",
                                  "libx.so.1", "libx.so.1.2"])
def test_build_refuses_a_model_file_that_is_code(tmp_path: Path, name: str):
    make_pack(tmp_path, models={name: b"Example model text 4521\n"})
    with pytest.raises(datawheel.DataWheelError, match="is code"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-lang-xa")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_wheel_problems_flag_a_versioned_shared_library(tmp_path: Path):
    wheel, manifest = build_pack(tmp_path)
    for name in ("pdfintake_lang_xa/models/libx.so.1", "pdfintake_lang_xa/libx.SO.1.2"):
        bad = rewrite(wheel, tmp_path / "bad.whl", add={name: b"made-up\n"})
        assert any("is code" in p and name in p for p in datawheel.wheel_problems(bad, manifest))
    assert not datawheel._is_code("pdfintake_lang_xa/models/x.solid") and not datawheel._is_code("a.so.x")


@pytest.mark.parametrize("which", ["notice", "model", "manifest"])
def test_build_refuses_a_symlinked_source(tmp_path: Path, which: str):
    make_pack(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(b"MADE-UP-SECRET-0123\n")
    target = {"notice": tmp_path / "pack" / "NOTICE-example", "model": tmp_path / "files" / "xa.traineddata",
              "manifest": tmp_path / "pack" / "manifest.json"}[which]
    if which != "notice":                                      # the link points at the right bytes: still refused
        outside.write_bytes(target.read_bytes())
    target.unlink()
    try:
        os.symlink(outside, target)
    except (OSError, NotImplementedError):
        pytest.skip("this file system can't make a symlink")
    with pytest.raises(datawheel.DataWheelError, match="link"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-lang-xa")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_build_checks_the_wheel_it_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    make_pack(tmp_path)
    monkeypatch.setattr(datawheel, "wheel_problems", lambda path, manifest: ["made-up problem"])
    with pytest.raises(datawheel.DataWheelError, match="made-up problem"):
        datawheel.build(tmp_path / "pack", tmp_path / "files", tmp_path / "out", dist_name="pdfintake-lang-xa")
    assert list((tmp_path / "out").iterdir()) == []            # neither the wheel nor its temporary file stays


def test_main_reports_os_errors_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    manifest = make_pack(tmp_path)
    (tmp_path / "out" / "pdfintake_lang_xa-1.0.0-py3-none-any.whl").mkdir(parents=True)   # a folder where the wheel goes
    code = datawheel.main(["build", str(tmp_path / "pack"), "--out", str(tmp_path / "out"),
                           "--files", str(tmp_path / "files"), "--offline"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False and "isn't a regular file" in out["error"]
    (tmp_path / "dest" / "xa.traineddata").mkdir(parents=True)  # a folder where a model goes
    with pytest.raises(datawheel.DataWheelError, match="is a folder"):
        datawheel.download_sources(manifest, tmp_path / "dest")
    code = datawheel.main(["download", str(tmp_path / "pack"), "--dest", str(tmp_path / "dest")])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and "is a folder" in out["error"]


def test_main_turns_an_unexpected_os_error_into_json(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                   monkeypatch: pytest.MonkeyPatch):
    make_pack(tmp_path)

    def full(*args: Any, **kw: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(datawheel, "build", full)
    code = datawheel.main(["build", str(tmp_path / "pack"), "--out", str(tmp_path / "out"),
                           "--files", str(tmp_path / "files"), "--offline"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and out["ok"] is False and "No space left" in out["error"]


# --------------------------------------------------------------------------- review fixes: broken HTTP replies

class _RawServer:
    """Answers every connection with fixed raw bytes (not always valid HTTP), then closes."""

    def __init__(self, reply: bytes) -> None:
        self.reply = reply
        self.hits = 0
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.base = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                self.hits += 1
                conn.sendall(self.reply)
                conn.close()
            except OSError:
                pass


@pytest.mark.parametrize("kind", ["not-http", "chunked-cut-off", "bad-chunk-size", "long-header"])
def test_a_broken_reply_is_retried_then_a_named_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str], kind: str):
    manifest = make_pack(tmp_path / "src", font=False)
    body = (tmp_path / "src" / "files" / "xa.traineddata").read_bytes()
    head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
    srv = _RawServer({
        "not-http": b"made-up-garbage\r\n\r\n",
        "chunked-cut-off": head + b"%x\r\n" % (len(body) + 100) + body,
        "bad-chunk-size": head + b"zz\r\n" + body,
        "long-header": b"HTTP/1.1 200 OK\r\nX-Made-Up: " + b"a" * 70000 + b"\r\n\r\n",
    }[kind])
    try:
        monkeypatch.setattr(datawheel, "_open_url",
                            lambda url, timeout: urllib.request.urlopen(srv.base + "/x", timeout=timeout))
        monkeypatch.setattr(datawheel, "_sleep", lambda s: None)
        with pytest.raises(datawheel.DataWheelError, match="nothing was kept"):
            datawheel.download_sources(manifest, tmp_path / "dest")
        assert srv.hits == datawheel.ATTEMPTS and os.listdir(tmp_path / "dest") == []
        code = datawheel.main(["download", str(tmp_path / "src" / "pack"), "--dest", str(tmp_path / "dest")])
        assert code == 1 and json.loads(capsys.readouterr().out)["ok"] is False
    finally:
        srv.sock.close()


def test_a_failure_message_names_the_source_host():
    """§11.2 (decided 2026-11-05): GitHub sources name GitHub's hosts; another source names its own."""
    assert datawheel._attempts_text(datawheel.ATTEMPTS) == "three attempts"
    assert datawheel._attempts_text(1) == "one attempt"
    assert datawheel._hosts_text("https://github.com/tesseract-ocr/tessdata_best/raw/" + COMMIT + "/xa.traineddata") \
        == "GitHub (github.com, or the githubusercontent.com hosts it redirects to)"
    assert datawheel._hosts_text("https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v1.0.0/x.onnx") \
        == "www.modelscope.cn"
    assert datawheel._hosts_text("not a url") == "the source's host"
