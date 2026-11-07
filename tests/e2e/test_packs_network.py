"""The real language packs, fetched from the index, and the language end to end (DESIGN §7.2).

Runs only with ``PDFINTAKE_NETWORK_TESTS=1`` (the CI network job); otherwise every test skips with
that reason, test by test, so a run selected by another marker deselects them instead. Every test is marked ``network``.

**Which packs.** Every fetched pack the packaged ``index.json`` lists is fetched and checked, so a
pack is covered here as soon as it is indexed. Every launch pack (DESIGN §13.1) is required: one
of them missing from the index fails, except a pack in ``PENDING`` (not built yet), which skips
with that reason.

**Where the packs come from.** The index pins each pack's wheel to an asset of a GitHub release
(``.../releases/download/<tag>/<wheel>``). Whether that release is published is asked of the
release page itself (``.../releases/tag/<tag>``), never inferred from one asset:

* **Published release** (the page answers): every asset of it must answer too, so a missing upload,
  a renamed wheel or a wrong index URL fails here. Those packs are fetched exactly as a user's run
  fetches them: the real index entry, the real store.
* **Unpublished release** (the page answers 404): those packs come from a CI-built local index
  instead. Each wheel is built here from its pinned upstream files (``datawheel``, with the canary
  rendered as ``kit wheel`` renders it), must hash to the index's pin, and is served over HTTPS
  from loopback with a throwaway certificate. Only the download address changes (those entries'
  URLs, and the store's host, port and CA seams, widened to the loopback server); the wheel,
  manifest, member and canary checks are the store's own. ``PDFINTAKE_PACK_WHEELS`` may name a
  folder of wheels built earlier; one is used only when it hashes to its pin.

A release page that answers neither (no network, a 5xx) fails the run: the network job needs GitHub.

**The language runs** go through the CLI in this process, on pages drawn here from made-up lines
on neutral topics:

* a shaped Thai page reaches at least 0.9 trigram recall, with its tone marks in the layer;
* a Hungarian page keeps ő and ű in the layer;
* a Latin page with a Thai block goes to the Thai pack through its probe once the pack is
  installed, and on a fresh cache through the mixed-script trigger or the
  ``mixed_script_unchecked`` note.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import http.server
import io
import json
import os
import shutil
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# A skip marker on each test, not a module-level skip: `pytest -m vision` (scripts/vision-local.sh, the
# release's Vision gate) then deselects these tests instead of counting a skip, which the gate refuses.
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("PDFINTAKE_NETWORK_TESTS") != "1",
                       reason="the pack network tests run only with PDFINTAKE_NETWORK_TESTS=1 (the CI network job); "
                              "they download the real language packs"),
]

REPO = Path(__file__).resolve().parents[2]
REQUIRED = ("de", "es", "fr", "hu", "it", "pl", "pt", "th", "vi")   # every launch pack but the built-in en
PENDING: frozenset[str] = frozenset()   # required, but not built yet: skips while the index lacks it
WHEELS_ENV = "PDFINTAKE_PACK_WHEELS"


def _pack_ids() -> list[str]:
    """Every fetched pack in the packaged index, plus the required ones (so a dropped one fails)."""
    from pdfintake.langpacks import index as I

    return sorted({e.id for e in I.load().entries if not e.base} | set(REQUIRED))


PACK_IDS = _pack_ids()

THAI = ["ดวงอาทิตย์ขึ้นทางทิศตะวันออกทุกเช้า",
        "น้ำในแม่น้ำไหลลงสู่ทะเลอย่างช้า ๆ",
        "พืชใช้แสงแดดเพื่อสร้างอาหารให้ตัวเอง",
        "ดวงจันทร์โคจรรอบโลกเดือนละหนึ่งรอบ"]
THAI_TONES = "่้"                    # mai ek and mai tho, both in the lines above
HUNGARIAN = ["A víz körforgása a felhőkből és a folyókból áll, és ez",
             "nem egy új dolog: a hűvös erdő is őrzi a nedvességet.",
             "A tavaszi eső után a föld puha lesz, és a fű gyorsan nő,",
             "hogy a szél és a nap még több vizet vigyen fel az égre.",
             "Ez a kör minden évben újra kezdődik, és soha nem ér véget."]
ENGLISH = ["Rivers carry water from the hills down to the sea,",
           "and the sea gives it back to the sky as warm vapour.",
           "Clouds form when that vapour cools high above the land,",
           "and rain brings the water back to the hills once more."]


# --------------------------------------------------------------------------- the environment

def _base() -> Any:
    from pdfintake import platforms
    from pdfintake.errors import PdfIntakeError
    from pdfintake.langpacks import store

    if not platforms.current().supported:
        pytest.skip(f"{platforms.current().key} isn't a supported platform")
    try:
        return store.base()
    except PdfIntakeError as exc:
        pytest.fail(f"the English base pack can't be loaded ({exc.code}: {exc.message}); "
                    "scripts/dev-install.sh installs it", pytrace=False)


def cli(*argv: str) -> tuple[int, dict[str, Any]]:
    """``pdfintake <argv> --json`` in this process: (exit code, the parsed JSON it printed)."""
    from pdfintake import cli as C

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = C.main([*argv, "--json"])
    try:
        return code, json.loads(out.getvalue())
    except ValueError:
        raise AssertionError(f"pdfintake {' '.join(argv)} printed no JSON: {out.getvalue()[:2000]!r}") from None


def _answers(url: str) -> bool | None:
    """True when ``url`` answers a HEAD, False on 404, None when unsure (no network, another error)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pdfintake-tests"})
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True
    except urllib.error.HTTPError as exc:
        return False if exc.code == 404 else None
    except (urllib.error.URLError, OSError):
        return None


def _release_tag(url: str) -> str:
    """The release tag of an index entry's asset URL (``.../releases/download/<tag>/<wheel>``)."""
    from pdfintake.langpacks import index as I

    path = urllib.parse.urlsplit(url).path
    assert path.startswith(I.RELEASES_PATH), f"{url} isn't a release asset under {I.RELEASES_PATH}"
    tag = path[len(I.RELEASES_PATH):].split("/", 1)[0]
    assert tag, f"{url} names no release tag"
    return tag


def _release_page(tag: str) -> str:
    from pdfintake.langpacks import index as I

    assert I.RELEASES_PATH.endswith("/download/")
    return f"https://{I.ENTRY_HOST}{I.RELEASES_PATH[:-len('download/')]}tag/{tag}"


def _sources(entries: Sequence[Any]) -> dict[str, str]:
    """pack id -> "hosted" (its release is published) or "local" (its release isn't yet).

    A published release must answer for every one of its assets; the run fails naming each that
    doesn't, rather than quietly building that pack here."""
    by_tag: dict[str, list[Any]] = {}
    for e in entries:
        by_tag.setdefault(_release_tag(e.dist.url), []).append(e)
    sources: dict[str, str] = {}
    problems: list[str] = []
    for tag, group in sorted(by_tag.items()):
        page = _release_page(tag)
        published = _answers(page)
        if published is None:
            pytest.fail(f"can't tell whether release {tag} is published ({page}): the network job needs GitHub",
                        pytrace=False)
        for e in group:
            sources[e.id] = "hosted" if published else "local"
            if not published:
                continue
            found = _answers(e.dist.url)
            if found is not True:
                problems.append(f"{e.id}: {e.dist.url} "
                                + ("answers 404" if found is False else "can't be reached"))
    if problems:
        pytest.fail("a published release is missing pack assets the index points at (a missing upload, "
                    "a renamed wheel or a wrong index URL):\n  " + "\n  ".join(problems), pytrace=False)
    return sources


# --------------------------------------------------------------------------- the CI-built local index

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prebuilt(entry: Any) -> Path | None:
    from pdfintake.langpacks.index import wheel_name

    name = wheel_name(entry.dist.distribution, entry.version)
    folders = [Path(p) for p in os.environ.get(WHEELS_ENV, "").split(os.pathsep) if p] + [REPO / "dist"]
    for folder in folders:
        for found in sorted(folder.rglob(name)) if folder.is_dir() else ():
            if found.is_file() and _sha256(found) == entry.dist.sha256:
                return found
    return None


def _build_wheel(entry: Any, base: Any, work: Path) -> Path:
    """The pack's wheel built from its pinned upstream files, as ``kit wheel`` builds it."""
    from conftest import models_root
    from pdfintake import datawheel
    from pdfintake.kit import render

    pack_dir = REPO / "packs" / entry.id
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    sources = models_root() / entry.id
    datawheel.download_sources(manifest, sources)              # checks every file against its pin
    files = work / "files" / entry.id
    files.mkdir(parents=True, exist_ok=True)
    for group in ("models", "fonts"):
        for item in manifest.get(group) or ():
            shutil.copyfile(sources / item["file"], files / item["file"])
    canary = manifest["canary"]
    fonts = [files / f["file"] for f in manifest.get("fonts") or ()]
    fonts += [base.asset_path(f.key) for f in base.manifest.fonts]
    png = render.render_canary(" ".join(canary["must_find"]), fonts, files / canary["file"],
                               dpi=int(canary.get("dpi") or 300))
    assert _sha256(png) == canary["sha256"], (
        f"the {entry.id} canary rendered here doesn't match its pin, so the pack wheel can't be rebuilt here")
    wheel = datawheel.build(pack_dir, files, work / "dist", dist_name=entry.dist.distribution)
    assert _sha256(wheel) == entry.dist.sha256, (
        f"the {entry.id} wheel built from its pinned files doesn't hash to the index pin")
    return wheel


def _make_cert(folder: Path) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    import ipaddress

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "example.com")])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(days=1)).not_valid_after(now + _dt.timedelta(days=2))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                           critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = folder / "cert.pem", folder / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return cert_path, key_path


class _WheelServer:
    """A loopback HTTPS server for the wheels: GET /<wheel name>, nothing else."""

    def __init__(self, folder: Path, wheels: dict[str, Path]) -> None:
        self.cert, key = _make_cert(folder)
        self.hits: list[str] = []
        served, hits = dict(wheels), self.hits

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = served.get(self.path.lstrip("/"))
                hits.append(self.path)
                if path is None:
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert, key)
        self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@dataclass
class Packs:
    source: dict[str, str]                      # pack id -> "hosted" or "local"
    index: Any
    installed: dict[str, Any]
    missing: dict[str, str]                     # required pack -> why it isn't tested

    @property
    def mode(self) -> str:
        kinds = set(self.source.values())
        return kinds.pop() if len(kinds) == 1 else "mixed"


@pytest.fixture(scope="module")
def packs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Packs]:
    from pdfintake import flow
    from pdfintake.langpacks import index as I
    from pdfintake.langpacks import store

    base = _base()
    idx = I.load()
    fetched = [e for e in idx.entries if not e.base]
    assert fetched, "the index lists no fetched packs"
    missing = {pid: f"{pid} isn't in this core's pack index yet" for pid in REQUIRED if idx.get(pid) is None}
    source = _sources(fetched)
    local = [e for e in fetched if source[e.id] == "local"]

    with pytest.MonkeyPatch.context() as mp:
        server = None
        local_idx = idx
        if local:
            work = tmp_path_factory.mktemp("local-index")
            wheels = {}
            for e in local:
                wheels[I.wheel_name(e.dist.distribution, e.version)] = _prebuilt(e) or _build_wheel(e, base, work)
            server = _WheelServer(work, wheels)
            url = f"https://127.0.0.1:{server.port}/"
            local_idx = dataclasses.replace(idx, entries=tuple(
                dataclasses.replace(e, dist=dataclasses.replace(
                    e.dist, url=url + I.wheel_name(e.dist.distribution, e.version)))
                if not e.base and source[e.id] == "local" else e
                for e in idx.entries))
            real_ctx, real_hosts, real_port = store._ssl_context, store._allowed_hosts(), store._port_allowed
            port = server.port
            # Only the loopback server when every pack is local; the real hosts as well when some
            # packs come from a published release.
            hosts = ("127.0.0.1",) if len(local) == len(fetched) else (*real_hosts, "127.0.0.1")
            mp.setattr(store, "_index", lambda: local_idx)
            mp.setattr(flow, "_load_index", lambda: local_idx)
            mp.setattr(store, "_allowed_hosts", lambda: hosts)
            mp.setattr(store, "_port_allowed",
                       lambda host, p: (host == "127.0.0.1" and p == port) or real_port(host, p))
            mp.setattr(store, "_ssl_context", lambda: real_ctx(extra_cafile=str(server.cert)))
        try:
            installed = {}
            for e in fetched:
                entry = local_idx.get(e.id)
                pack = store.ensure(entry, allow_network=True)
                assert pack.manifest.id == e.id and pack.manifest.version == e.version
                installed[e.id] = pack
            if server is not None:
                assert sorted(h.lstrip("/") for h in server.hits) == sorted(
                    I.wheel_name(e.dist.distribution, e.version) for e in local)
            yield Packs(source, local_idx, installed, missing)
        finally:
            if server is not None:
                server.close()


# --------------------------------------------------------------------------- pages, drawn here

def shaped_page(dest: Path, lines: Sequence[tuple[str, Path]], *, dpi: float = 300.0) -> Path:
    """A picture of ``lines`` (each in its own font), shaped by HarfBuzz through fpdf2 and rendered by pdfium."""
    pytest.importorskip("uharfbuzz")
    import pypdfium2 as pdfium
    from fpdf import FPDF

    pdf = FPDF(unit="pt", format=(595.28, 420.0))
    pdf.set_creation_date(_dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc))
    pdf.set_auto_page_break(False)
    pdf.set_margin(0)
    families: dict[Path, str] = {}
    for _text, font in lines:
        if font not in families:
            families[font] = f"Example{len(families)}"
            pdf.add_font(families[font], fname=str(font))
    pdf.set_text_shaping(True)
    pdf.add_page()
    y = 60.0
    for text, font in lines:
        pdf.set_font(families[font], size=16)
        pdf.text(48.0, y, text)
        y += 34.0
    doc = pdfium.PdfDocument(bytes(pdf.output()))
    try:
        page = doc[0]
        img = page.render(scale=dpi / 72.0, grayscale=True).to_pil()
        page.close()
    finally:
        doc.close()
    img.convert("L").save(dest, format="PNG", dpi=(dpi, dpi))
    return dest


def pdf_text(path: Path) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        parts = []
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            parts.append(tp.get_text_bounded())
            tp.close()
            page.close()
        return "\n".join(parts)
    finally:
        doc.close()


def run_one(path: Path) -> dict[str, Any]:
    code, data = cli("run", str(path))
    (rec,) = data["records"]
    assert code == 0 and rec["exit"] == 0, rec["steps"]
    return rec


def detection(rec: dict[str, Any], page: int = 1) -> dict[str, Any]:
    return next(p for p in rec["language"]["detection"]["pages"] if p["page"] == page)


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("pack_id", PACK_IDS)
def test_each_indexed_pack_is_fetched_from_the_index(packs: Packs, pack_id: str):
    from pdfintake.langpacks import store

    if pack_id in packs.missing:
        if pack_id in PENDING:
            pytest.skip(f"{packs.missing[pack_id]} (not built yet)")
        pytest.fail(f"{packs.missing[pack_id]}: it is one of the required packs {', '.join(REQUIRED)}",
                    pytrace=False)
    pack = packs.installed[pack_id]
    entry = packs.index.get(pack_id)
    print(f"\n{pack_id} {entry.version}: {packs.source[pack_id]} ({entry.dist.url})")
    assert (pack.root / ".verified").is_file()
    assert pack.root.parent.name == "packs" and pack.root.name == f"{pack_id}-{entry.version}"
    found = store.locate(pack_id)
    assert found is not None and found.root == pack.root
    (result,) = store.verify(pack_id)               # the store's own deep check: manifest pin, every
    assert result.ok, result.problems               # file's size and SHA-256, the canary passed
    for asset in (*pack.manifest.models, *pack.manifest.fonts):
        assert pack.asset_path(asset.key).is_file()  # asset_path checks size and SHA-256 again


def test_a_shaped_thai_page_is_read_by_the_thai_pack(packs: Packs, tmp_path: Path):
    if "th" not in packs.installed:
        pytest.skip(packs.missing.get("th", "the Thai pack isn't installed"))
    from pdfintake.kit import score

    font = packs.installed["th"].asset_path("noto-sans-thai")
    page = shaped_page(tmp_path / "thai lines.png", [(t, font) for t in THAI])
    rec = run_one(page)
    assert detection(rec)["pack"] == "th", rec["language"]
    text = pdf_text(Path(rec["output"]["path"]))
    recall = score.trigram_recall(" ".join(THAI), text, "Thai")
    print(f"\nThai trigram recall {recall:.3f} ({packs.mode} packs)")
    assert recall >= 0.9, text
    assert all(mark in text for mark in THAI_TONES), text


def test_a_hungarian_page_keeps_its_double_accents(packs: Packs, tmp_path: Path):
    if "hu" not in packs.installed:
        pytest.skip(packs.missing.get("hu", "the Hungarian pack isn't installed"))
    base = _base()
    font = base.asset_path("noto-sans")
    page = shaped_page(tmp_path / "hungarian lines.png", [(t, font) for t in HUNGARIAN])
    rec = run_one(page)
    assert detection(rec)["pack"] == "hu", rec["language"]
    text = pdf_text(Path(rec["output"]["path"]))
    assert "ő" in text and "ű" in text, text
    for word in ("felhőkből", "hűvös"):
        assert word in text, text


def _mixed_page(tmp_path: Path, packs: Packs) -> Path:
    base = _base()
    latin = base.asset_path("noto-sans")
    thai = packs.installed["th"].asset_path("noto-sans-thai")
    lines = [(ENGLISH[0], latin), (ENGLISH[1], latin), (THAI[0], thai), (THAI[2], thai),
             (ENGLISH[2], latin), (ENGLISH[3], latin)]
    return shaped_page(tmp_path / "mixed lines.png", lines)


def test_a_latin_page_with_a_thai_block_goes_to_the_thai_pack_through_its_probe(packs: Packs, tmp_path: Path):
    if "th" not in packs.installed:
        pytest.skip(packs.missing.get("th", "the Thai pack isn't installed"))
    rec = run_one(_mixed_page(tmp_path, packs))
    det = detection(rec)
    assert det["pack"] == "th" and det["composition"] == "probe", rec["language"]
    assert "mixed_script_page" in [w["code"] for w in rec["warnings"]]
    text = pdf_text(Path(rec["output"]["path"]))
    assert "ดวงอาทิตย์" in text and "water" in text.lower(), text


def test_on_a_fresh_cache_the_thai_block_is_caught_by_the_trigger_or_noted(packs: Packs, tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch):
    """No Thai pack installed, so no probe can run: a calibrated trigger routes the page (fetching
    the pack), and with the trigger uncalibrated the page is read with the base pack and noted."""
    if "th" not in packs.installed:
        pytest.skip(packs.missing.get("th", "the Thai pack isn't installed"))
    page = _mixed_page(tmp_path, packs)
    monkeypatch.setenv("PDFINTAKE_CACHE", str(tmp_path / "fresh cache"))
    from pdfintake.langpacks import store

    assert store.locate("th") is None
    rec = run_one(page)
    det = detection(rec)
    notes = [n["code"] for n in rec["notes"]]
    print(f"\nfresh cache: page 1 read with {det['pack']} ({det['composition']}); notes {notes}")
    if det["composition"] == "trigger":
        assert det["pack"] == "th", rec["language"]
    else:
        assert "mixed_script_unchecked" in notes, (rec["language"], notes)
        assert det["pack"] == "en"
