"""langpacks/store.py: the base distribution check, fetching, the store's refusals and its cache.

A local HTTPS server (``http.server`` with a throwaway self-signed certificate made at test time)
serves wheels built by ``datawheel.build`` from made-up packs. The certificate is trusted through
the ``_ssl_context`` seam, never through an environment variable, and the allowlist is the
loopback address. The base pack is a data wheel "installed" into a temporary site folder.
Every test runs with a non-ASCII cache path.

The helpers above the tests (``TextEngine``, ``make_pack``, ``Env``) are shared with test_canary.py.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import http.server
import io
import importlib
import importlib.metadata
import json
import os
import ssl
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from fixtures import build as fx
from pdfintake import datawheel, platforms
from pdfintake.errors import IntegrityError, PackError, SetupError, UsageError
from pdfintake.langpacks import index as I
from pdfintake.langpacks import manifest as M
from pdfintake.langpacks import store
from pdfintake.ocr import engines
from pdfintake.ocr.engines import Availability
from pdfintake.params import Param, ParamTable
from pdfintake.types import EngineRead

COMMIT = "0123456789abcdef0123456789abcdef01234567"
BASE_TEXT = "Example 4521"
NON_ASCII = "cache-ünïcødé-ตัวอย่าง"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- a fake engine whose read is its model's text

class TextSession:
    def __init__(self, engine: "TextEngine", use_id: str, role: str, models: dict[str, Path]) -> None:
        self.engine, self.use_id, self.role, self.models = engine, use_id, role, dict(models)

    def read(self, page: Any, *, timeout_s: float) -> EngineRead:
        with page.open():                                  # the canary must be a readable PNG of the right size
            pass
        with self.engine.lock:
            self.engine.reads += 1
        if self.engine.junk is not None:
            texts = [self.engine.junk]
        else:
            texts = [Path(p).read_text(encoding="utf-8").strip() for _, p in sorted(self.models.items())]
        lines = fx.lines_for([t for t in texts if t])
        return EngineRead(self.use_id, self.engine.name, f"{self.engine.name}:text", self.role,  # type: ignore[arg-type]
                          "ok" if lines else "empty", lines, 0.001)

    def close(self) -> None:
        pass


class TextEngine:
    """Reads back the text of its model files: a pack's "language" is whatever its model says."""

    name = "fake"
    api = 1
    PARAMS = ParamTable(Param("lang", "str", pattern=r"[a-z]{2,3}", default="en"))
    MODEL_KINDS = frozenset({"traineddata"})
    platforms = frozenset({p.key for p in platforms.PLATFORMS} | {platforms.current().key})
    session_per_thread = False

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.reads = 0
        self.lock = threading.Lock()
        self.ver: str | None = "1.0"
        self.junk: str | None = None
        self.state: str = "available"

    def version(self) -> str | None:
        return self.ver

    def availability(self, params: Any, models: Any, platform_key: str) -> Availability:
        if self.state != "available":
            return Availability(False, self.state, "made-up reason")  # type: ignore[arg-type]
        for key, path in models.items():
            if not Path(path).is_file():
                return Availability(False, "missing_model", f"model {key} is missing")
        return Availability(True, "available", "")

    def open(self, use_id: str, role: str, params: Any, models: Any) -> TextSession:
        return TextSession(self, use_id, role, models)

    def candidate_params(self, lang_tag: str, script: str, assets: Any) -> list[dict[str, Any]]:
        return []

    def contract_case(self) -> None:
        return None


# --------------------------------------------------------------------------- made-up packs

@dataclass
class Pack:
    manifest: dict[str, Any]
    raw: bytes
    files: dict[str, bytes]                # model, font and canary files by file name
    notices: dict[str, bytes]

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def dist(self) -> str:
        return datawheel._dist_name(self.manifest)


def canary_png() -> bytes:
    return fx.png_bytes(fx.text_like(240, 60, lines=1), dpi=300)


def make_pack(pack_id: str = "xa", *, base: bool = False, version: str = "1.0.0",
              models: dict[str, str] | None = None, must_find: tuple[str, ...] = ("ตัวอย่าง", "4521"),
              font: bool = False, engine: str = "fake", canary: bool = True, base_refs: tuple[str, ...] = ("en:eng",),
              extra_uses: tuple[dict[str, Any], ...] = ()) -> Pack:
    """A schema-valid manifest for the TextEngine, with its files. Model texts are made up."""
    if models is None:
        models = {"eng.traineddata": BASE_TEXT} if base else {f"{pack_id.replace('-', '')}.traineddata": "ตัวอย่าง"}
    files: dict[str, bytes] = {}
    entries = []
    for name, text in models.items():
        data = (text + "\n").encode("utf-8")
        files[name] = data
        entries.append({"key": name.split(".")[0], "file": name, "engine": engine, "kind": "traineddata",
                        "bytes": len(data), "sha256": sha(data),
                        "source": f"https://github.com/tesseract-ocr/tessdata_best/raw/{COMMIT}/{name}",
                        "licence": "Apache-2.0", "notice": "NOTICE-example"})
    refs = [e["key"] for e in entries] + ([] if base else list(base_refs))
    fonts = []
    if font:
        data = fx.box_font("ABC")
        files["ExampleSans-Regular.ttf"] = data
        fonts.append({"key": "example-sans", "file": "ExampleSans-Regular.ttf", "covers": ["Latin"],
                      "bytes": len(data), "sha256": sha(data), "licence": "OFL-1.1",
                      "source": "https://github.com/notofonts/latin-greek-cyrillic/releases/download/v1/ExampleSans-Regular.ttf"})
    tag = "en" if base else "xa"
    manifest: dict[str, Any] = {
        "schema": 1, "id": pack_id, "version": version, "requires_core": ">=0.1.0", "base": base,
        "title": "English" if base else "Example",
        "languages": [{"tag": tag, "iso639_3": "eng" if base else "xaa", "role": "primary"}],
        "scripts": ["Latin"],
        "detect": {"osd_scripts": ["Latin"], "langid": {"codes": [tag]}},
        "readers": {"default": {
            "engines": [{"id": "reader", "engine": engine, "role": "reader", "params": {"lang": tag}, "models": refs},
                        *extra_uses],
            "pick": {"rule": "single", "version": 1, "params": {"reader": "reader"}}, "fallback": "none"}},
        "normalize": ["nfc"], "repairs": [], "fills": [], "models": entries, "fonts": fonts,
        "must_survive": [], "bench": {"kit": "1.0", "corpus": "none", "platforms": {}},
    }
    if canary:
        data = canary_png()
        files[f"canary-{pack_id}.png"] = data
        manifest["canary"] = {"file": f"canary-{pack_id}.png", "bytes": len(data), "sha256": sha(data), "dpi": 300,
                              "must_find": list(must_find if not base else ("4521",))}
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    notices = {"NOTICE-example": b"Example notice for made-up test models.\n"}
    if font:
        notices["OFL.txt"] = b"Example OFL text for a made-up font.\n"
    return Pack(manifest, raw, files, notices)


def write_pack_dir(pack: Pack, root: Path) -> Path:
    """The pack laid out as the store keeps it (manifest, models/, fonts/, canary, notices)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_bytes(pack.raw)
    for name, data in pack.files.items():
        sub = "models" if name.endswith(".traineddata") else "fonts" if name.endswith((".ttf", ".otf")) else ""
        target = root / sub / name if sub else root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name, data in pack.notices.items():
        (root / name).write_bytes(data)
    return root


def build_wheel(pack: Pack, tmp: Path) -> Path:
    work = tmp / f"build-{pack.id}-{pack.version}-{os.urandom(3).hex()}"
    pack_dir, files = work / "pack", work / "files"
    pack_dir.mkdir(parents=True)
    files.mkdir()
    (pack_dir / "manifest.json").write_bytes(pack.raw)
    for name, data in pack.notices.items():
        (pack_dir / name).write_bytes(data)
    for name, data in pack.files.items():
        (files / name).write_bytes(data)
    return datawheel.build(pack_dir, files, work / "out", dist_name=pack.dist)


def rewrite_wheel(wheel: Path, dest: Path, *, add: dict[str, bytes] | None = None,
                  replace: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(wheel) as zf:
        members = [(zi.filename, zf.read(zi)) for zi in zf.infolist()]
    members = [(n, (replace or {}).get(n, d)) for n, d in members] + list((add or {}).items())
    record = next(n for n, _ in members if n.endswith(".dist-info/RECORD"))
    lines = [f"{n},{datawheel._record_hash(sha(d))},{len(d)}" for n, d in members if n != record] + [f"{record},,"]
    members = [(n, ("\n".join(lines) + "\n").encode() if n == record else d) for n, d in members]
    with zipfile.ZipFile(dest, "w") as zf:
        for n, d in members:
            zf.writestr(zipfile.ZipInfo(n, date_time=datawheel.ZIP_DATE), d)
    return dest


def install_wheel(wheel: Path, site: Path) -> Path:
    """Unpack a wheel the way an installer does, adding the installer's own dist-info file."""
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(site)
        di = next(n.split("/")[0] for n in zf.namelist() if ".dist-info/" in n)
    (site / di / "INSTALLER").write_text("example-installer\n", encoding="utf-8")
    with open(site / di / "RECORD", "a", encoding="utf-8") as f:
        f.write(f"{di}/INSTALLER,,\n")
    return site / di


def index_entry(pack: Pack, *, url: str | None, wheel: bytes) -> I.IndexEntry:
    d = pack.manifest["detect"]
    return I.IndexEntry(id=pack.id, version=pack.version, base=pack.manifest["base"], title=pack.manifest["title"],
                        scripts=tuple(pack.manifest["scripts"]), langid_codes=tuple(d["langid"]["codes"]),
                        detect=I.detect_copy(d), manifest_sha256=sha(pack.raw),
                        dist=I.Dist("installed" if url is None else "fetch", pack.dist, url, sha(wheel), len(wheel)),
                        installed_bytes=sum(len(b) for b in pack.files.values()) + len(pack.raw))


# --------------------------------------------------------------------------- the HTTPS server

def _make_cert(folder: Path) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    import ipaddress

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Example Test Server")])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(minutes=5)).not_valid_after(now + _dt.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                        x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_path, key_path


Route = Callable[[http.server.BaseHTTPRequestHandler], None]


class HttpsStub:
    def __init__(self, folder: Path) -> None:
        self.cert, key = _make_cert(folder)
        self.routes: dict[str, Route] = {}
        self.hits: dict[str, int] = {}
        self.lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                with outer.lock:
                    outer.hits[self.path] = outer.hits.get(self.path, 0) + 1
                route = outer.routes.get(self.path)
                if route is None:
                    self.send_error(404)
                    return
                route(self)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert, key)
        self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        self.base = f"https://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def serve(self, path: str, body: bytes, *, status: int = 200, length: bool = True, delay: float = 0.0) -> None:
        def route(h: http.server.BaseHTTPRequestHandler) -> None:
            h.send_response(status)
            h.send_header("Content-Type", "application/octet-stream")
            if length:
                h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            if delay:
                half = len(body) // 2
                h.wfile.write(body[:half])
                h.wfile.flush()
                time.sleep(delay)
                h.wfile.write(body[half:])
            else:
                h.wfile.write(body)
        self.routes[path] = route

    def redirect(self, path: str, location: str, status: int = 302) -> None:
        def route(h: http.server.BaseHTTPRequestHandler) -> None:
            h.send_response(status)
            h.send_header("Location", location)
            h.send_header("Content-Length", "0")
            h.end_headers()
        self.routes[path] = route

    def fail(self, path: str, status: int) -> None:
        def route(h: http.server.BaseHTTPRequestHandler) -> None:
            body = b"<html><body>made-up error page</body></html>"
            h.send_response(status)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        self.routes[path] = route

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="module")
def https(tmp_path_factory: pytest.TempPathFactory):
    stub = HttpsStub(tmp_path_factory.mktemp("tls"))
    yield stub
    stub.close()


# --------------------------------------------------------------------------- the environment

@dataclass
class Env:
    tmp: Path
    server: HttpsStub | None
    engine: TextEngine
    base_pack: Pack
    site: Path
    entries: dict[str, I.IndexEntry] = field(default_factory=dict)
    sleeps: list[float] = field(default_factory=list)

    def index(self) -> I.Index:
        return I.Index(1, tuple(self.entries.values()))

    def publish(self, pack: Pack, *, wheel: bytes | None = None, path: str | None = None,
                url: str | None = None, sha256: str | None = None, size: int | None = None) -> I.IndexEntry:
        """Serve the pack's wheel (or ``wheel`` bytes) and add its index entry."""
        assert self.server is not None
        data = wheel if wheel is not None else build_wheel(pack, self.tmp).read_bytes()
        path = path or f"/releases/download/v0.1.0/{I.wheel_name(pack.dist, pack.version)}"
        self.server.serve(path, data)
        entry = index_entry(pack, url=url or self.server.base + path, wheel=data)
        if sha256 is not None or size is not None:
            entry = I.IndexEntry(**{**entry.__dict__, "dist": I.Dist(
                "fetch", entry.dist.distribution, entry.dist.url, sha256 or entry.dist.sha256,
                size if size is not None else entry.dist.bytes)})
        self.entries[pack.id] = entry
        return entry

    def pack_dir(self, pack_id: str, version: str = "1.0.0") -> Path:
        return platforms.cache_root() / "packs" / f"{pack_id}-{version}"

    def leftovers(self) -> list[str]:
        root = platforms.cache_root() / "packs"
        return sorted(p.name for p in root.iterdir() if p.name.startswith(".")) if root.is_dir() else []


def make_env(tmp: Path, monkeypatch: pytest.MonkeyPatch, server: HttpsStub | None) -> Env:
    cache = tmp / NON_ASCII
    monkeypatch.setenv("PDFINTAKE_CACHE", str(cache))
    engine = TextEngine()
    monkeypatch.setitem(engines._REGISTRY, "fake", engine)
    base_pack = make_pack("en", base=True, font=True)
    wheel = build_wheel(base_pack, tmp)
    site = tmp / "site"
    install_wheel(wheel, site)
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()
    env = Env(tmp, server, engine, base_pack, site)
    env.entries["en"] = index_entry(base_pack, url=None, wheel=wheel.read_bytes())
    monkeypatch.setattr(store, "_index", env.index)
    monkeypatch.setattr(store, "_allowed_hosts", lambda: ("127.0.0.1",))
    monkeypatch.setattr(store, "_sleep", env.sleeps.append)
    if server is not None:
        trust(monkeypatch, server.cert, server.port)
    return env


def trust(monkeypatch: pytest.MonkeyPatch, cert: Path, port: int) -> None:
    """Trust a throwaway certificate and let the store reach ``127.0.0.1:<port>`` (production allows no port)."""
    real = store._ssl_context
    monkeypatch.setattr(store, "_ssl_context", lambda: real(extra_cafile=str(cert)))
    monkeypatch.setattr(store, "_port_allowed", lambda host, p: host == "127.0.0.1" and p == port)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, https: HttpsStub) -> Env:
    https.routes.clear()
    https.hits.clear()
    return make_env(tmp_path, monkeypatch, https)


def hits(env: Env, entry: I.IndexEntry) -> int:
    assert env.server is not None and entry.dist.url
    return env.server.hits.get(entry.dist.url[len(env.server.base):], 0)


# --------------------------------------------------------------------------- fetch: the good path

def test_fetch_installs_checks_and_marks_a_pack(env: Env):
    entry = env.publish(make_pack("xa", font=True))
    pack = store.fetch(entry)
    d = env.pack_dir("xa")
    assert pack.root == d and pack.manifest.id == "xa"
    assert sorted(p.name for p in d.iterdir()) == [".verified", "NOTICE-example", "OFL.txt", "canary-xa.png", "fonts",
                                                    "manifest.json", "models"]
    assert json.loads((d / ".verified").read_text(encoding="utf-8"))["manifest_sha256"] == entry.manifest_sha256
    assert env.engine.reads == 1                               # the canary ran before .verified was written
    assert env.leftovers() == []
    assert NON_ASCII in str(d)                                 # the cache path isn't ASCII, and nothing minded
    assert store.locate("xa") == pack
    assert set(store.installed()) == {"en", "xa"}
    again = store.ensure(entry, allow_network=False)           # installed: no download, canary cached
    assert again == pack and hits(env, entry) == 1 and env.engine.reads == 1


def test_model_paths_resolve_base_refs_and_link_tessdata(env: Env):
    entry = env.publish(make_pack("xa", font=True))
    pack = store.fetch(entry)
    base = store.base()
    use = pack.manifest.readers_for(platforms.current().key).engines[0]
    paths = store.model_paths(pack, base, use)
    assert set(paths) == {"xa", "en:eng"}
    assert paths["xa"] == pack.root / "models" / "xa.traineddata"
    assert paths["en:eng"] == base.root / "models" / "eng.traineddata"
    tess = store.tessdata_dir()
    assert tess == platforms.cache_root() / "tessdata"
    for p in paths.values():
        assert os.path.samefile(tess / p.name, p)              # Tesseract finds every language of the read in one folder
    specs = pack.font_specs()
    assert [(s.key, s.path.name, s.priority, s.covers) for s in specs] == [
        ("example-sans", "ExampleSans-Regular.ttf", 0, ("Latin",))]
    with pytest.raises(PackError, match="no model or font"):
        pack.asset_path("en:eng")


def test_a_redirect_within_the_allowlist_is_followed(env: Env):
    pack = make_pack("xa")
    wheel = build_wheel(pack, env.tmp).read_bytes()
    entry = env.publish(pack, wheel=wheel, path="/real.whl", url=env.server.base + "/moved.whl")  # type: ignore[union-attr]
    env.server.redirect("/moved.whl", env.server.base + "/real.whl?signature=made-up")  # type: ignore[union-attr]
    env.server.serve("/real.whl?signature=made-up", wheel)  # type: ignore[union-attr]
    assert store.fetch(entry).manifest.id == "xa"


# --------------------------------------------------------------------------- fetch: every refusal of DESIGN §3.5

def test_a_wrong_wheel_hash_is_refused(env: Env):
    entry = env.publish(make_pack("xa"), sha256="0" * 64)
    with pytest.raises(IntegrityError, match="pack index pins 0000") as info:
        store.fetch(entry)
    assert info.value.code == "wheel_hash_mismatch"
    assert not env.pack_dir("xa").exists() and env.leftovers() == []
    assert hits(env, entry) == 1                                # a hash mismatch isn't retried


def test_a_wrong_member_hash_is_refused(env: Env):
    pack = make_pack("xa")
    good = build_wheel(pack, env.tmp)
    model = "pdfintake_lang_xa/models/xa.traineddata"
    bad = rewrite_wheel(good, env.tmp / "bad.whl", replace={model: "ภาษาอื่น\n".encode("utf-8")})
    entry = env.publish(pack, wheel=bad.read_bytes())          # the index pins this wheel: only the member is wrong
    with pytest.raises(IntegrityError, match="xa.traineddata"):
        store.fetch(entry)
    assert not env.pack_dir("xa").exists() and env.leftovers() == []


@pytest.mark.parametrize("member,expect", [
    ("pdfintake_lang_xa/models/extra.bin", "isn't part of this pack"),
    ("pdfintake_lang_xa/../escape.txt", "'..'"),
    ("/tmp/escape.txt", "absolute"),
    ("pdfintake_lang_xa/models/hook.pth", "is code"),
    ("pdfintake_lang_xa/loader.py", "is code"),
])
def test_an_extra_unsafe_or_code_member_is_refused(env: Env, member: str, expect: str):
    pack = make_pack("xa")
    bad = rewrite_wheel(build_wheel(pack, env.tmp), env.tmp / "bad.whl", add={member: b"made-up\n"})
    entry = env.publish(pack, wheel=bad.read_bytes())
    with pytest.raises(PackError, match=expect) as info:
        store.fetch(entry)
    assert not isinstance(info.value, IntegrityError)
    assert not env.pack_dir("xa").exists() and env.leftovers() == []
    assert not (env.tmp / "escape.txt").exists() and not (platforms.cache_root() / "escape.txt").exists()


@pytest.mark.parametrize("length", [True, False], ids=["content-length", "no-length"])
def test_an_oversize_stream_is_refused(env: Env, length: bool):
    pack = make_pack("xa")
    wheel = build_wheel(pack, env.tmp).read_bytes()
    entry = env.publish(pack, wheel=wheel, size=len(wheel) - 100)
    env.server.serve(entry.dist.url[len(env.server.base):], wheel, length=length)  # type: ignore[union-attr,index]
    with pytest.raises(PackError, match="pins") as info:
        store.fetch(entry)
    assert info.value.code == "download_oversize"
    assert not env.pack_dir("xa").exists() and env.leftovers() == []


@pytest.mark.parametrize("location", ["https://downloads.example.invalid/pdfintake_lang_xa-1.0.0-py3-none-any.whl",
                                      "http://127.0.0.1:1/pdfintake_lang_xa-1.0.0-py3-none-any.whl"],
                         ids=["other-host", "plain-http"])
def test_a_redirect_off_the_allowlist_is_refused(env: Env, network_attempts: list, location: str):
    entry = env.publish(make_pack("xa"))
    env.server.redirect(entry.dist.url[len(env.server.base):], location)  # type: ignore[union-attr,index]
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    assert info.value.code == "redirect_refused"
    assert network_attempts == []                                # not even a DNS lookup of the other host
    assert not env.pack_dir("xa").exists()


@pytest.mark.parametrize("url", ["http://127.0.0.1/x.whl", "https://files.example.com/x.whl",
                                 "https://user:secret@127.0.0.1/x.whl", "https://127.0.0.1/x.whl?y=1"])
def test_a_download_address_off_the_allowlist_is_never_opened(env: Env, url: str):
    pack = make_pack("xa")
    entry = env.publish(pack, url=url)
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    assert info.value.code == "url_refused"
    assert env.server.hits == {}  # type: ignore[union-attr]


def test_network_failures_are_retried_then_name_the_host(env: Env):
    entry = env.publish(make_pack("xa"))
    env.server.fail(entry.dist.url[len(env.server.base):], 503)  # type: ignore[union-attr,index]
    with pytest.raises(SetupError, match="couldn't be downloaded from 127.0.0.1 after three attempts") as info:
        store.fetch(entry)
    assert info.value.code == "download_failed"
    assert_names_github_not_pypi(info.value.message)
    assert info.value.detail["host"] == "127.0.0.1" and info.value.detail["attempts"] == 3
    assert hits(env, entry) == store.ATTEMPTS == 3
    assert env.sleeps == [1.0, 3.0]
    assert env.leftovers() == []


def assert_names_github_not_pypi(message: str) -> None:
    """§11.2 (decided 2026-11-05): a fetch-failure hint names the GitHub hosts the packs live on."""
    assert "GitHub (github.com or release-assets.githubusercontent.com)" in message, message
    assert "pythonhosted" not in message.lower() and "pypi" not in message.lower()
    assert "retries" not in message and "3 attempts" not in message


def test_the_pack_hosts_are_github_only():
    assert I.ALLOWED_HOSTS == ("github.com", "release-assets.githubusercontent.com")
    assert store._attempts_text(store.ATTEMPTS) == "three attempts"
    assert store._attempts_text(1) == "one attempt" and store._attempts_text(12) == "12 attempts"


def test_the_hosts_label_says_github_only_for_github_hosts(monkeypatch: pytest.MonkeyPatch):
    # Decision 9: a host move touches only the index, ALLOWED_HOSTS and the pyproject pin
    assert store._pack_hosts_text() == "GitHub (github.com or release-assets.githubusercontent.com)"
    monkeypatch.setattr(I, "ALLOWED_HOSTS", ("files.example.org", "github.com"))
    assert store._pack_hosts_text() == "files.example.org or github.com"


def test_a_missing_asset_is_not_retried(env: Env):
    entry = env.publish(make_pack("xa"))
    env.server.fail(entry.dist.url[len(env.server.base):], 404)  # type: ignore[union-attr,index]
    with pytest.raises(SetupError, match="HTTP 404") as info:
        store.fetch(entry)
    assert info.value.code == "download_refused" and hits(env, entry) == 1


def test_a_manifest_that_isnt_the_pinned_one_is_refused(env: Env):
    pack = make_pack("xa")
    entry = env.publish(pack)
    env.entries["xa"] = I.IndexEntry(**{**entry.__dict__, "manifest_sha256": "f" * 64})
    with pytest.raises(IntegrityError, match="manifest"):
        store.fetch(env.entries["xa"])
    assert not env.pack_dir("xa").exists()


def test_a_server_that_isnt_trusted_is_refused(env: Env, monkeypatch: pytest.MonkeyPatch):
    entry = env.publish(make_pack("xa"))
    monkeypatch.setattr(store, "_ssl_context", lambda: ssl.create_default_context())   # the stub's cert isn't trusted
    with pytest.raises(SetupError, match="certificate") as info:
        store.fetch(entry)
    assert info.value.code == "download_tls"
    assert_names_github_not_pypi(info.value.message)


# --------------------------------------------------------------------------- concurrency

def test_concurrent_fetches_end_with_one_directory(env: Env):
    pack = make_pack("xa")
    wheel = build_wheel(pack, env.tmp).read_bytes()
    entry = env.publish(pack, wheel=wheel)
    env.server.serve(entry.dist.url[len(env.server.base):], wheel, delay=0.3)  # type: ignore[union-attr,index]
    gate = threading.Barrier(4)
    results: list[Any] = []

    def worker() -> None:
        gate.wait()
        try:
            results.append(store.fetch(entry))
        except BaseException as exc:  # pragma: no cover - reported below
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert all(isinstance(r, store.InstalledPack) for r in results), results
    assert {r.root for r in results} == {env.pack_dir("xa")}
    packs = platforms.cache_root() / "packs"
    assert sorted(p.name for p in packs.iterdir()) == ["xa-1.0.0"]   # one folder, no temporary leftovers
    assert store.verify("xa")[0].ok


# --------------------------------------------------------------------------- loads after install

def test_a_file_tampered_with_after_install_is_caught_at_the_next_load(env: Env):
    entry = env.publish(make_pack("xa"))
    pack = store.fetch(entry)
    model = pack.root / "models" / "xa.traineddata"
    good = model.read_bytes()
    model.write_bytes(good[:-2] + b"X\n")                        # same size, other content
    with pytest.raises(IntegrityError, match=str(model).replace("\\", "\\\\")):
        store.locate("xa")
    with pytest.raises(IntegrityError, match="xa.traineddata"):
        pack.asset_path("xa")
    with pytest.raises(IntegrityError, match="xa.traineddata"):
        store.ensure(entry, allow_network=False)
    problems = store.verify("xa")[0].problems
    assert problems and "xa.traineddata" in problems[0]
    repaired = store.ensure(entry, allow_network=True)           # downloads it again, checked as on first install
    assert (repaired.root / "models" / "xa.traineddata").read_bytes() == good
    assert hits(env, entry) == 2 and env.leftovers() == []


def test_model_paths_checks_every_file_at_the_moment_it_is_called(env: Env):
    """§11.2 (decided 2026-11-05): no check is remembered between calls, so a model swapped after
    install, or after an earlier call that passed, is refused by the very next call."""
    entry = env.publish(make_pack("xa"))
    pack = store.fetch(entry)
    base = store.base()
    use = pack.manifest.readers_for(platforms.current().key).engines[0]
    model = pack.root / "models" / "xa.traineddata"
    base_model = base.root / "models" / "eng.traineddata"
    good, base_good = model.read_bytes(), base_model.read_bytes()
    tess = store.tessdata_dir() / "xa.traineddata"
    assert store.model_paths(pack, base, use)["xa"] == model     # installed and checked: fine
    assert os.path.samefile(tess, model)

    def swap(path: Path, data: bytes) -> None:                  # a new file under the old name, as a sync tool would
        tmp = path.with_name(path.name + ".swap")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    with open(model, "r+b") as f:                                # rewritten in place: the tessdata link shares the inode
        f.write(b"Z")
    assert tess.read_bytes() != good
    with pytest.raises(IntegrityError, match="xa.traineddata") as info:
        store.model_paths(pack, base, use)
    assert info.value.code == "hash_mismatch"
    swap(model, good)
    assert store.model_paths(pack, base, use)["xa"] == model
    assert tess.read_bytes() == good and os.path.samefile(tess, model)   # the corrupted link was replaced

    swap(model, good[:-2] + b"Y\n")                             # same size, other bytes
    with pytest.raises(IntegrityError, match="xa.traineddata") as info:
        store.model_paths(pack, base, use)
    assert info.value.code == "hash_mismatch"
    assert tess.read_bytes() == good                             # the bad file never reached tessdata/

    swap(model, good)                                            # put back: the next call passes again
    assert store.model_paths(pack, base, use)["xa"] == model

    swap(model, good + b"more")                                  # another size
    with pytest.raises(IntegrityError, match="xa.traineddata") as info:
        store.model_paths(pack, base, use)
    assert info.value.code == "size_mismatch"
    swap(model, good)
    assert store.model_paths(pack, base, use)["xa"] == model

    swap(base_model, base_good[:-2] + b"Q\n")                   # an en: reference is checked in the base pack
    with pytest.raises(IntegrityError, match="eng.traineddata"):
        store.model_paths(pack, base, use)
    model.unlink()                                               # and a missing model is named too
    swap(base_model, base_good)
    with pytest.raises(IntegrityError, match="xa.traineddata") as info:
        store.model_paths(pack, base, use)
    assert info.value.code == "file_missing"


def test_a_pack_that_isnt_installed_needs_the_network(env: Env, network_attempts: list):
    entry = env.publish(make_pack("xa"))
    with pytest.raises(PackError) as info:
        store.ensure(entry, allow_network=False)
    assert info.value.code == "pack_not_installed" and isinstance(info.value, SetupError)
    assert hits(env, entry) == 0 and network_attempts == []
    assert store.locate("xa") is None
    assert store.ensure(entry, allow_network=True).manifest.id == "xa"


def test_a_failed_canary_leaves_the_pack_unverified_and_unused(env: Env):
    entry = env.publish(make_pack("xa", models={"xa.traineddata": "Beispieltext"}))   # not what the canary says
    with pytest.raises(PackError, match="self-check") as info:
        store.fetch(entry)
    assert info.value.code == "pack_broken"
    d = env.pack_dir("xa")
    assert d.is_dir() and not (d / ".verified").exists()
    assert store.locate("xa") is None and "xa" not in store.installed()
    assert "canary" in store.verify("xa")[0].problems[0]
    reads = env.engine.reads
    with pytest.raises(PackError, match="self-check"):
        store.ensure(entry, allow_network=False)                 # the canary runs again; nothing is downloaded
    assert env.engine.reads == reads + 1 and hits(env, entry) == 1


def test_a_canary_that_starts_passing_marks_the_pack(env: Env):
    entry = env.publish(make_pack("xa"))
    env.engine.junk = "made-up junk"
    with pytest.raises(PackError, match="self-check"):
        store.fetch(entry)
    env.engine.junk = None
    assert store.ensure(entry, allow_network=False).manifest.id == "xa"
    assert (env.pack_dir("xa") / ".verified").exists() and hits(env, entry) == 1


# --------------------------------------------------------------------------- tessdata

def test_the_same_tessdata_file_name_with_a_different_hash_is_refused(env: Env):
    a = env.publish(make_pack("xa", models={"xx.traineddata": "ตัวอย่าง"}))
    store.fetch(a).asset_path("xx")
    b = env.publish(make_pack("xb", models={"xx.traineddata": "ตัวอย่าง อื่น"}))
    with pytest.raises(PackError) as info:
        store.fetch(b)
    assert info.value.code == "tessdata_conflict" and "xx.traineddata" in info.value.message
    assert not env.pack_dir("xb").exists()
    tess = store.tessdata_dir() / "xx.traineddata"
    assert os.path.samefile(tess, env.pack_dir("xa") / "models" / "xx.traineddata")


def test_a_pack_reusing_a_base_tessdata_name_is_refused(env: Env):
    entry = env.publish(make_pack("xa", models={"eng.traineddata": "ตัวอย่าง"}, base_refs=()))
    with pytest.raises(PackError, match="eng.traineddata"):
        store.fetch(entry)
    assert not env.pack_dir("xa").exists()


def test_a_newer_version_replaces_its_own_stale_tessdata_link(env: Env):
    old = env.publish(make_pack("xa", version="1.0.0"))
    store.fetch(old).asset_path("xa")
    new = env.publish(make_pack("xa", version="1.1.0", models={"xa.traineddata": "ตัวอย่าง ใหม่"}))
    pack = store.ensure(new, allow_network=True)
    pack.asset_path("xa")
    assert os.path.samefile(store.tessdata_dir() / "xa.traineddata", pack.root / "models" / "xa.traineddata")


def test_a_copy_is_used_where_hard_links_fail(env: Env, monkeypatch: pytest.MonkeyPatch):
    entry = env.publish(make_pack("xa"))
    pack = store.fetch(entry)

    def no_link(src: Any, dst: Any) -> None:
        raise OSError(18, "Invalid cross-device link")

    tess = store.tessdata_dir() / "xa.traineddata"
    assert os.path.samefile(tess, pack.root / "models" / "xa.traineddata")   # linked when the canary read it
    tess.unlink()
    monkeypatch.setattr(os, "link", no_link)
    path = pack.asset_path("xa")
    assert not os.path.samefile(tess, path) and tess.read_bytes() == path.read_bytes()
    tess.write_bytes(path.read_bytes()[:-2] + b"X\n")             # a stale copy is replaced, not trusted
    pack.asset_path("xa")
    assert tess.read_bytes() == path.read_bytes()


def _unreadable_in_tessdata(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Make ``integrity.sha256_file`` raise its IntegrityError for every file in tessdata/, as it does
    for a file there that can't be read; the pack's own files still hash. The paths it refused."""
    from pdfintake import integrity

    real = integrity.sha256_file
    tdir = store.tessdata_dir()
    refused: list[Path] = []

    def sha256_file(path: Any) -> str:
        if Path(path).parent == tdir:
            refused.append(Path(path))
            raise IntegrityError("unreadable", f"{path} can't be read (Permission denied).", path=str(path))
        return real(path)

    monkeypatch.setattr(integrity, "sha256_file", sha256_file)
    return refused


def test_an_unreadable_leftover_in_tessdata_is_replaced(env: Env, monkeypatch: pytest.MonkeyPatch):
    pack = store.fetch(env.publish(make_pack("xa")))
    path = pack.asset_path("xa")
    tess = store.tessdata_dir() / "xa.traineddata"
    tess.unlink()
    tess.write_bytes(path.read_bytes()[:-2] + b"X\n")             # a leftover of the same size: only its hash tells
    refused = _unreadable_in_tessdata(monkeypatch)
    assert pack.asset_path("xa") == path                          # not an IntegrityError about the leftover
    assert refused == [tess] and os.path.samefile(tess, path)


def test_remove_survives_an_unreadable_file_in_tessdata(env: Env, monkeypatch: pytest.MonkeyPatch):
    pack = store.fetch(env.publish(make_pack("xa")))
    path = pack.asset_path("xa")
    tess = store.tessdata_dir() / "xa.traineddata"
    tess.unlink()
    tess.write_bytes(path.read_bytes())                           # a copy, not the link: only its hash would tell
    refused = _unreadable_in_tessdata(monkeypatch)
    assert store.remove("xa") is True
    assert refused == [tess] and not env.pack_dir("xa").exists()
    assert tess.exists()                                          # not known to be the pack's, so it stays


# --------------------------------------------------------------------------- base()

def test_base_finds_the_installed_data_distribution(env: Env):
    b = store.base()
    assert b.manifest.id == "en" and b.manifest.base
    assert b.root == env.site / "pdfintake_data_en"
    assert b.asset_path("eng").read_text(encoding="utf-8").strip() == BASE_TEXT
    assert store.locate("en") == b and store.installed()["en"] == b


def test_base_refuses_a_pth_in_its_record(env: Env):
    record = next(env.site.glob("*.dist-info")) / "RECORD"
    (env.site / "zz_pdfintake_data_en.pth").write_text("# made-up\n", encoding="utf-8")   # site isn't a site dir: never run
    with open(record, "a", encoding="utf-8") as f:                 # base() skips RECORD entries with no file
        f.write("zz_pdfintake_data_en.pth,sha256=made-up,10\n")
    with pytest.raises(SetupError, match="zz_pdfintake_data_en.pth") as info:
        store.base()
    assert info.value.code == "base_pack_not_data_only" and not isinstance(info.value, IntegrityError)


def test_base_refuses_code_or_strays_in_its_folder(env: Env):
    (env.site / "pdfintake_data_en" / "models" / "made_up.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SetupError, match="made_up.py"):
        store.base()


def test_base_names_a_tampered_model(env: Env):
    model = env.site / "pdfintake_data_en" / "models" / "eng.traineddata"
    model.write_bytes(model.read_bytes()[:-2] + b"9\n")
    with pytest.raises(IntegrityError) as info:
        store.base()
    assert str(model) in info.value.message
    assert any(str(model) in p for p in store.verify("en")[0].problems)


class _RecordAsWritten:
    """A distribution whose ``files`` lists every RECORD line whether or not its file is still there,
    as ``importlib.metadata`` does on Python 3.11 (3.12 and later leave out the files that are gone)."""

    def __init__(self, dist: importlib.metadata.Distribution) -> None:
        self._dist = dist

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    @property
    def files(self) -> list[importlib.metadata.PackagePath] | None:
        text = self._dist.read_text("RECORD")
        if text is None:
            return None
        out = []
        for row in csv.reader(text.splitlines()):
            p = importlib.metadata.PackagePath(row[0])
            p.hash, p.size, p.dist = None, None, self._dist
            out.append(p)
        return out


@pytest.mark.parametrize("record", ["this_python", "as_python_311"])
def test_base_names_a_missing_model(env: Env, monkeypatch: pytest.MonkeyPatch, record: str):
    model = env.site / "pdfintake_data_en" / "models" / "eng.traineddata"
    model.unlink()                                               # an antivirus quarantine, a half-finished install
    if record == "as_python_311":
        real = importlib.metadata.distribution
        monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _RecordAsWritten(real(name)))
        files = importlib.metadata.distribution(env.entries["en"].dist.distribution).files
        assert files is not None and any(str(f).endswith("models/eng.traineddata") for f in files)
    with pytest.raises(IntegrityError, match="is missing; reinstall pdfintake") as info:
        store.base()
    assert str(model) in info.value.message and info.value.code == "file_missing"


def test_base_refuses_a_manifest_other_than_the_pinned_one(env: Env):
    env.entries["en"] = I.IndexEntry(**{**env.entries["en"].__dict__, "manifest_sha256": "e" * 64})
    with pytest.raises(IntegrityError, match="pack index pins eeee"):
        store.base()


def test_base_refuses_a_version_other_than_the_index_entry(env: Env):
    env.entries["en"] = I.IndexEntry(**{**env.entries["en"].__dict__, "version": "1.0.1"})
    with pytest.raises(SetupError, match="needs 1.0.1") as info:
        store.base()
    assert info.value.code == "base_pack_version"


def test_base_missing_is_a_setup_error(env: Env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(env.site)])
    importlib.invalidate_caches()
    name = env.entries["en"].dist.distribution
    real = importlib.metadata.distribution

    def distribution(dist_name: str):                          # a dev venv also holds the real English data
        if dist_name == name:
            raise importlib.metadata.PackageNotFoundError(dist_name)
        return real(dist_name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    with pytest.raises(SetupError, match="reinstall pdfintake") as info:
        store.base()
    assert info.value.code == "base_pack_missing"


# --------------------------------------------------------------------------- verify and remove

def test_verify_reports_every_installed_pack(env: Env):
    store.fetch(env.publish(make_pack("xa")))
    results = {r.pack: r for r in store.verify()}
    assert set(results) == {"en", "xa"} and all(r.ok and not r.problems for r in results.values())
    assert store.verify("zz")[0].problems == ["this version of pdfintake has no pack 'zz'"]
    env.publish(make_pack("xb", models={"xb.traineddata": "ตัวอย่าง"}))
    assert store.verify("xb")[0].problems == ["the 'xb' pack isn't installed"]


def test_remove_deletes_the_pack_and_its_tessdata_links(env: Env):
    entry = env.publish(make_pack("xa"))
    store.fetch(entry).asset_path("xa")
    tess = store.tessdata_dir()
    assert (tess / "xa.traineddata").exists()
    store.base().asset_path("eng")
    assert store.remove("xa") is True
    assert not env.pack_dir("xa").exists() and not (tess / "xa.traineddata").exists()
    assert (tess / "eng.traineddata").exists()                   # the base pack's links stay
    assert store.remove("xa") is False and env.leftovers() == []
    with pytest.raises(UsageError):
        store.remove("en")
    with pytest.raises(UsageError):
        store.remove("../xa")


def test_store_source_names_no_host():
    import re as _re

    text = Path(store.__file__).read_text(encoding="utf-8")
    assert _re.findall(r"https?://[A-Za-z0-9.-]+", text) == []   # hosts live in index.json, ALLOWED_HOSTS and pyproject


# --------------------------------------------------------------------------- review fixes: broken responses

class RawTls:
    """A TLS server that answers every request with fixed raw bytes (not always valid HTTP), then closes."""

    def __init__(self, folder: Path, reply: bytes) -> None:
        import socket

        self.cert, key = _make_cert(folder)
        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ctx.load_cert_chain(self.cert, key)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.reply = reply
        self.hits = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                s = self.ctx.wrap_socket(conn, server_side=True)
                s.recv(65536)
                self.hits += 1
                s.sendall(self.reply)
                s.close()
            except OSError:
                pass

    def close(self) -> None:
        self.sock.close()


@pytest.mark.parametrize("kind", ["chunked-cut-off", "not-http", "long-header", "bad-chunk-size"])
def test_a_broken_response_is_retried_then_a_setup_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    env = make_env(tmp_path, monkeypatch, None)
    pack = make_pack("xa")
    wheel = build_wheel(pack, tmp_path).read_bytes()
    head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
    reply = {
        "chunked-cut-off": head + b"%x\r\n" % len(wheel) + wheel[: len(wheel) // 2],     # http.client IncompleteRead
        "not-http": b"made-up-garbage\r\n\r\n",                                           # BadStatusLine
        "long-header": b"HTTP/1.1 200 OK\r\nX-Made-Up: " + b"a" * 70000 + b"\r\n\r\n",     # LineTooLong
        "bad-chunk-size": head + b"zz\r\n" + wheel,                                       # IncompleteRead
    }[kind]
    server = RawTls(tmp_path, reply)
    try:
        trust(monkeypatch, server.cert, server.port)
        entry = index_entry(pack, url=f"https://127.0.0.1:{server.port}/{I.wheel_name(pack.dist, pack.version)}",
                            wheel=wheel)
        env.entries["xa"] = entry
        with pytest.raises(SetupError) as info:
            store.fetch(entry)
        assert type(info.value) is SetupError and info.value.code == "download_failed"
        assert server.hits == store.ATTEMPTS and env.sleeps == [1.0, 3.0]
        assert not env.pack_dir("xa").exists() and env.leftovers() == []
    finally:
        server.close()


# --------------------------------------------------------------------------- review fixes: the canary floor

@pytest.mark.parametrize("state", ["not_on_platform", "unsupported_language"])
def test_a_pack_no_engine_reads_here_is_broken(env: Env, state: str):
    entry = env.publish(make_pack("xa"))
    env.engine.state = state                                     # every use skipped: nothing reads the canary
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    assert info.value.code == "pack_broken" and "No OCR engine" in info.value.message
    assert env.engine.reads == 0
    assert not (env.pack_dir("xa") / ".verified").exists() and store.locate("xa") is None


def test_a_skipped_use_beside_a_reader_that_passes_is_fine(env: Env, monkeypatch: pytest.MonkeyPatch):
    gone = TextEngine("vision")
    gone.state = "not_on_platform"
    monkeypatch.setitem(engines._REGISTRY, "vision", gone)
    pack = make_pack("xa", extra_uses=({"id": "vision", "engine": "vision", "role": "reader",
                                        "params": {"lang": "xa"}, "models": []},))
    assert store.fetch(env.publish(pack)).manifest.id == "xa"
    assert (env.pack_dir("xa") / ".verified").exists()


def test_a_fetched_pack_without_a_canary_is_broken(env: Env):
    entry = env.publish(make_pack("xa", canary=False))
    with pytest.raises(PackError, match="no canary") as info:
        store.fetch(entry)
    assert info.value.code == "pack_broken"
    assert env.engine.reads == 0 and not (env.pack_dir("xa") / ".verified").exists()


def test_ensure_runs_the_base_packs_canary(env: Env):
    entry = env.entries["en"]
    assert store.ensure(entry, allow_network=False) == store.base() and env.engine.reads == 1
    store.ensure(entry, allow_network=False)
    assert env.engine.reads == 1                                 # a pass is cached
    env.engine.ver = "2.0"                                       # a new engine version reads again
    env.engine.junk = "made-up junk"
    with pytest.raises(PackError) as info:
        store.ensure(entry, allow_network=False)
    assert info.value.code == "pack_broken" and isinstance(info.value, SetupError)
    assert store.base().manifest.id == "en"                      # base() itself runs no canary


# --------------------------------------------------------------------------- review fixes: names that ignore case

def _reraw(pack: Pack) -> Pack:
    return Pack(pack.manifest, (json.dumps(pack.manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
                pack.files, pack.notices)


def zip_wheel(pack: Pack, dest: Path) -> Path:
    """The pack's wheel assembled in memory (datawheel.build reads files from a folder, where two names
    that differ only by case are one file on macOS and Windows)."""
    pkg, di = pack.dist.replace("-", "_"), f"{pack.dist.replace('-', '_')}-{pack.version}.dist-info"
    members = [(f"{pkg}/manifest.json", pack.raw)]
    for name, data in pack.files.items():
        sub = "models/" if name.endswith(".traineddata") else "fonts/" if name.endswith((".ttf", ".otf")) else ""
        members.append((f"{pkg}/{sub}{name}", data))
    members += [(f"{pkg}/{n}", d) for n, d in pack.notices.items()]
    members += [(f"{di}/METADATA", datawheel._metadata(pack.manifest, pack.dist)), (f"{di}/WHEEL", datawheel._wheel_file())]
    lines = [f"{n},{datawheel._record_hash(sha(d))},{len(d)}" for n, d in members] + [f"{di}/RECORD,,"]
    members.append((f"{di}/RECORD", ("\n".join(lines) + "\n").encode()))
    with zipfile.ZipFile(dest, "w") as zf:
        for n, d in members:
            zf.writestr(zipfile.ZipInfo(n, date_time=datawheel.ZIP_DATE), d)
    return dest


def test_a_tessdata_name_that_differs_only_by_case_is_refused(env: Env):
    pack = make_pack("xb", models={"ENG.traineddata": "ตัวอย่าง"}, base_refs=())   # the base ships eng.traineddata
    pack.manifest["models"][0]["key"] = "eng-upper"
    pack.manifest["readers"]["default"]["engines"][0]["models"] = ["eng-upper"]
    entry = env.publish(_reraw(pack))
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    # manifest.check compares file names with the base pack's ignoring case (§11.2), so the parse
    # refuses it before the store's own tessdata check would
    assert info.value.code == "invalid_manifest" and "ENG.traineddata" in info.value.message
    assert not env.pack_dir("xb").exists()


def test_a_tessdata_name_another_pack_ships_in_other_case_is_refused(env: Env):
    """manifest.check knows only the base pack, so the store's claims check is the one guard
    between two downloaded packs whose model names differ only by case."""
    a = env.publish(make_pack("xa", models={"xx.traineddata": "ตัวอย่าง"}))
    store.fetch(a).asset_path("xx")
    pack = make_pack("xb", models={"XX.traineddata": "ตัวอย่าง อื่น"})
    pack.manifest["models"][0]["key"] = "xx-upper"
    pack.manifest["readers"]["default"]["engines"][0]["models"] = ["xx-upper", "en:eng"]
    b = env.publish(_reraw(pack))
    with pytest.raises(PackError) as info:
        store.fetch(b)
    assert info.value.code == "tessdata_conflict" and "XX.traineddata" in info.value.message
    assert info.value.detail["other"] == "xa"
    assert not env.pack_dir("xb").exists() and env.leftovers() == []
    tess = store.tessdata_dir() / "xx.traineddata"
    assert os.path.samefile(tess, env.pack_dir("xa") / "models" / "xx.traineddata")


def test_two_members_that_differ_only_by_case_are_refused(env: Env):
    pack = make_pack("xc", models={"xc.traineddata": "ตัวอย่าง", "XC.traineddata": "ตัวอย่าง อื่น"})
    pack.manifest["models"][1]["key"] = "xc-upper"
    pack.manifest["readers"]["default"]["engines"][0]["models"] = ["xc", "xc-upper", "en:eng"]
    pack = _reraw(pack)
    wheel = zip_wheel(pack, env.tmp / "case.whl").read_bytes()
    entry = env.publish(pack, wheel=wheel)
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    assert info.value.code == "invalid_manifest"                  # the parse refuses first (§11.2)
    assert "ignor" in info.value.message and "case" in info.value.message
    assert not env.pack_dir("xc").exists() and env.leftovers() == []


def _parse_without_check(data: bytes, *, base: M.Manifest | None) -> M.Manifest:
    """manifest.parse with its problems ignored, to reach the store's own guards behind it."""
    sem = M._Semantic(json.loads(data), base, M.__version__)
    sem.run()
    return sem.build(sha(data))


def _case_pair_pack() -> Pack:
    pack = make_pack("xc", models={"xc.traineddata": "ตัวอย่าง", "XC.traineddata": "ตัวอย่าง อื่น"})
    pack.manifest["models"][1]["key"] = "xc-upper"
    pack.manifest["readers"]["default"]["engines"][0]["models"] = ["xc", "xc-upper", "en:eng"]
    return _reraw(pack)


def test_the_store_refuses_wheel_members_that_differ_only_by_case(env: Env, monkeypatch: pytest.MonkeyPatch):
    # defence in depth behind manifest.check: with the parse lenient, _extract's own member check refuses
    monkeypatch.setattr(store, "_parse_manifest", _parse_without_check)
    pack = _case_pair_pack()
    entry = env.publish(pack, wheel=zip_wheel(pack, env.tmp / "case.whl").read_bytes())
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    assert info.value.code == "wheel_refused" and "ignore case" in info.value.message
    assert info.value.detail["member"].casefold() == "models/xc.traineddata"
    assert not env.pack_dir("xc").exists() and env.leftovers() == []


def test_the_store_refuses_a_packs_own_tessdata_names_that_differ_only_by_case(env: Env):
    # defence in depth: the fetch path stops at the member check first, and an installed folder
    # can't hold both names on a case-insensitive file system, so the guard is called directly
    pack = _case_pair_pack()
    manifest = _parse_without_check(pack.raw, base=store.base().manifest)
    with pytest.raises(PackError) as info:
        store._tessdata_conflicts(manifest, store.base())
    assert info.value.code == "tessdata_conflict" and "ignore case" in info.value.message
    assert "xc.traineddata" in info.value.message and "XC.traineddata" in info.value.message


# --------------------------------------------------------------------------- review fixes: named errors only

def test_an_encrypted_manifest_member_is_a_named_refusal(env: Env):
    pack = make_pack("xa")
    data = bytearray(build_wheel(pack, env.tmp).read_bytes())
    name = b"pdfintake_lang_xa/manifest.json"
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
        local = zf.getinfo(name.decode()).header_offset
    data[local + 6] |= 0x1                                       # the encrypted flag, in the local header
    central = data.index(b"PK\x01\x02")                         # the central directory's first record
    while data[central + 46:central + 46 + len(name)] != name:
        central = data.index(b"PK\x01\x02", central + 4)
    data[central + 8] |= 0x1                                     # and in the central directory: zipfile wants a password
    entry = env.publish(pack, wheel=bytes(data))
    with pytest.raises(PackError) as info:
        store.fetch(entry)
    assert info.value.code == "wheel_unreadable" and "encrypted" in info.value.message
    assert not env.pack_dir("xa").exists() and env.leftovers() == []


def test_a_file_that_vanishes_mid_check_is_an_integrity_error(env: Env, monkeypatch: pytest.MonkeyPatch):
    pack = store.fetch(env.publish(make_pack("xa")))
    from pdfintake import integrity

    def vanished(path: Any, sha256: str, size: Any = None) -> None:
        raise FileNotFoundError(2, "No such file or directory", str(path))   # a concurrent run discarded the folder

    monkeypatch.setattr(integrity, "verify_file", vanished)
    with pytest.raises(IntegrityError) as info:
        pack.asset_path("xa")
    assert info.value.code == "file_missing" and "xa.traineddata" in info.value.message
    with pytest.raises(IntegrityError):
        store.locate("xa")


def test_os_bookkeeping_files_in_the_base_folder_are_ignored(env: Env):
    (env.site / "pdfintake_data_en" / ".DS_Store").write_bytes(b"\x00made-up")
    (env.site / "pdfintake_data_en" / "models" / "Thumbs.db").write_bytes(b"\x00made-up")
    assert store.base().manifest.id == "en"
    (env.site / "pdfintake_data_en" / "stray.bin").write_bytes(b"made-up")
    with pytest.raises(SetupError, match="stray.bin.*Reinstall pdfintake"):
        store.base()


# --------------------------------------------------------------------------- review fixes: one rule set for addresses

@pytest.mark.parametrize("url", [
    "https://release-assets.githubusercontent.com/x?sig=1",
    "https://release-assets.githubusercontent.com:8443/x?sig=1",
    "https://github.com:8443/RazorII/pdfintake/releases/download/v1/x.whl",
    "https://release-assets.githubusercontent.com/a b",
    "https://release-assets.githubusercontent.com/a\tb",
    "https://release-assets.githubusercontent.com/café",
    "https://release-assets.githubusercontent.com/a\\b",
    "https://user@github.com/x",
    "http://github.com/x",
    "https://example.com/x",
    "",
])
def test_redirect_rules_agree_with_the_index(url: str):
    assert (store._url_problem(url, initial=False) is None) == (I.redirect_problems(url) == []), url


def test_a_redirect_to_another_port_is_refused(env: Env, network_attempts: list):
    entry = env.publish(make_pack("xa"))
    env.server.redirect(entry.dist.url[len(env.server.base):], "https://127.0.0.1:1/x.whl")  # type: ignore[union-attr,index]
    with pytest.raises(PackError, match="names a port") as info:
        store.fetch(entry)
    assert info.value.code == "redirect_refused" and network_attempts == []
