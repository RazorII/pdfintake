"""The real language packs, fetched from the index, and the language end to end (DESIGN §7.2).

Runs only with ``PDFINTAKE_NETWORK_TESTS=1`` (the CI network job); otherwise every test skips with
that reason, test by test, so a run selected by another marker deselects them instead. Every test is marked ``network``.

**Which packs.** Every fetched pack the packaged ``index.json`` lists is fetched and checked, so a
pack is covered here as soon as it is indexed. Every launch pack (DESIGN §13.1) is required: one
of them missing from the index fails, except a pack in ``PENDING`` (not built yet), which skips
with that reason.

**Where the packs come from.** The index pins each pack's wheel to an asset of a GitHub release
(``.../releases/download/<tag>/<wheel>``). Whether that release is published is asked of GitHub's
releases API (``api.github.com/repos/<owner>/<repo>/releases/tags/<tag>``), never inferred from one
asset, and never from the release page (``.../releases/tag/<tag>``): that page answers for a bare tag
too, so on the tag's own commit, pushed before the release exists, it called the release published
and every asset missing. ``PDFINTAKE_TEST_GITHUB_TOKEN`` (the network job's read-only token) lifts the
API's anonymous rate limit; without it the call goes unauthenticated:

* **Published release** (the API answers, not a draft): every asset of it must answer too, so a missing upload,
  a renamed wheel or a wrong index URL fails here. Those packs are fetched exactly as a user's run
  fetches them: the real index entry, the real store.
* **Unpublished release** (the API answers 404: no release, a bare tag or a draft): those packs come from a CI-built local index
  instead. Each wheel is built here from its pinned upstream files (``datawheel``, with the canary
  rendered as ``kit wheel`` renders it), must hash to the index's pin, and is served over HTTPS
  from loopback with a throwaway certificate. Only the download address changes (those entries'
  URLs, and the store's host, port and CA seams, widened to the loopback server); the wheel,
  manifest, member and canary checks are the store's own. ``PDFINTAKE_PACK_WHEELS`` may name a
  folder of wheels built earlier; one is used only when it hashes to its pin.

An API that answers neither (no network, a 5xx, a rate limit) fails the run: the network job needs GitHub.

**The language runs** go through the CLI in this process, on pages drawn here from made-up lines
on neutral topics:

* every pack reads a page of its own language (``PAGES``): pass 1 routes the page to it, at least
  0.9 of the words (Thai: character trigrams) come back, and every character group the manifest
  says must survive does, at that group's own floor. With a published release this proves each
  pack end to end from the public assets, on every OS the network job runs on;
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

# One page of made-up lines per pack, on the same neutral topic, each carrying every character
# group the pack's manifest says must survive. Thai and Hungarian reuse the lines above.
PAGES: dict[str, list[str]] = {
    "de": ["Das Wasser fließt von den Bergen hinunter ins Meer,",
           "und die Sonne wärmt es, bis es als Dampf aufsteigt.",
           "Über dem Land kühlt der Dampf ab und bildet Wolken,",
           "aus denen später Regen auf Wälder und Wiesen fällt.",
           "So schließt sich der Kreislauf jedes Jahr aufs Neue."],
    "es": ["El agua de los ríos baja de las montañas hasta el mar,",
           "donde el sol la calienta y la convierte en vapor.",
           "Más tarde, ese vapor se enfría y forma las nubes,",
           "y la lluvia vuelve a caer sobre los campos y árboles.",
           "En la orilla, una cigüeña observa el río en silencio.",
           "Así comienza otra vez el ciclo cada año, sin pausa."],
    "fr": ["Les rivières descendent des collines à la mer,",
           "où le soleil les réchauffe et les change en vapeur.",
           "Plus haut, cette vapeur se refroidit en nuages gris,",
           "puis la pluie tombe sur les forêts et les châteaux.",
           "Un garçon regarde la pluie tomber, le cœur tranquille.",
           "Ce cycle se répète chaque année, été comme hiver."],
    "hu": HUNGARIAN,
    "it": ["Ogni mattina il sole scalda il mare vicino alla città,",
           "così una parte del mare sale in cielo come vapore.",
           "Più in alto il vapore si raffredda e forma le nuvole,",
           "perché lassù fa freddo anche quando è estate.",
           "Poi la pioggia cade sui boschi e il ciclo ricomincia.",
           "Però nessuno vede quando il ciclo comincia davvero."],
    "pl": ["Woda z rzek spływa z gór aż do morza,",
           "gdzie słońce ogrzewa ją i zamienia w parę.",
           "Wysoko nad ziemią para się ochładza i tworzy chmury,",
           "a deszcz znów pada na lasy, łąki i pola.",
           "Można to zobaczyć, gdy rano mgła unosi się nad rzeką.",
           "Późną jesienią woda zamarza w małych stawach.",
           "Ten obieg powtarza się każdego roku bez końca."],
    "pt": ["A água dos rios desce das montanhas até o mar,",
           "onde o sol a aquece e a transforma em vapor.",
           "Lá no alto, o vapor esfria e forma as nuvens,",
           "e a chuva volta a cair sobre as árvores e os campos.",
           "O trânsito da água entre o céu e a terra nunca para.",
           "Os cientistas reúnem informações sobre essas mudanças,",
           "e o ciclo também se repete em cada estação do ano."],
    "th": THAI,
    "vi": ["Nước từ các con sông chảy xuống biển mỗi ngày,",
           "mặt trời làm nóng nước và biến nó thành hơi.",
           "Trên cao, hơi nước lạnh đi và tạo thành mây,",
           "rồi mưa lại rơi xuống rừng, đồng ruộng và làng quê.",
           "Vòng tuần hoàn này lặp lại quanh năm không ngừng."],
}
WORD_RECALL = 0.9       # Latin pages: share of the page's words read exactly (case-folded)
TRIGRAM_RECALL = 0.9    # Thai pages: share of the page's character trigrams


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


TOKEN_ENV = "PDFINTAKE_TEST_GITHUB_TOKEN"


def _release_api(tag: str) -> str:
    from pdfintake.langpacks import index as I

    assert I.RELEASES_PATH.endswith("/releases/download/")
    slug = I.RELEASES_PATH[:-len("/releases/download/")].strip("/")
    assert I.ENTRY_HOST == "github.com" and slug.count("/") == 1, (I.ENTRY_HOST, I.RELEASES_PATH)
    return f"https://api.github.com/repos/{slug}/releases/tags/{urllib.parse.quote(tag)}"


def _published(tag: str) -> bool | None:
    """True when release ``tag`` is published, False when GitHub has no published release of that
    name (none, a bare tag, a draft), None when unsure (no network, another error)."""
    headers = {"User-Agent": "pdfintake-tests", "Accept": "application/vnd.github+json"}
    token = os.environ.get(TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(_release_api(tag), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.load(resp)
    except urllib.error.HTTPError as exc:
        return False if exc.code == 404 else None
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if release.get("tag_name") != tag:
        return None
    return not release.get("draft", False)


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
        published = _published(tag)
        if published is None:
            pytest.fail(f"can't tell whether release {tag} is published ({_release_api(tag)}): the network job "
                        "needs GitHub", pytrace=False)
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


def test_every_pack_has_a_page():
    assert sorted(PAGES) == sorted(REQUIRED), "every launch pack needs a page in PAGES"


@pytest.mark.parametrize("pack_id", PACK_IDS)
def test_each_pack_reads_a_page_in_its_language(packs: Packs, pack_id: str, tmp_path: Path):
    """The pack as fetched (from the published release when there is one) reads a page of its own
    language through the CLI: pass 1 routes the page to it, and its readers keep the words and every
    character group its manifest says must survive, at that group's own floor."""
    if pack_id not in packs.installed:
        pytest.skip(packs.missing.get(pack_id, f"the {pack_id} pack isn't installed"))
    if pack_id not in PAGES:
        pytest.fail(f"{pack_id} is indexed but has no page in PAGES", pytrace=False)
    from pdfintake.kit import score

    pack = packs.installed[pack_id]
    lines = PAGES[pack_id]
    fonts = [pack.asset_path(f.key) for f in pack.manifest.fonts] or [_base().asset_path("noto-sans")]
    page = shaped_page(tmp_path / f"{pack_id} lines.png", [(t, fonts[0]) for t in lines])
    rec = run_one(page)
    det = detection(rec)
    truth = " ".join(lines)
    text = pdf_text(Path(rec["output"]["path"]))
    script = pack.manifest.scripts[0]               # the primary script, as the kit's bench takes it
    metric = score.primary_metric(script)
    if metric == "words":
        recall, floor = score.word_recall(truth, text, script), WORD_RECALL
    else:
        recall, floor = score.trigram_recall(truth, text, script), TRIGRAM_RECALL
    survived = {ms.name: score.must_survive_recall(truth, text, ms, script) for ms in pack.manifest.must_survive}
    print(f"\n{pack_id} ({packs.source[pack_id]}): routed to {det['pack']} ({det['composition']}), "
          f"{metric} recall {recall:.3f}, must survive {survived}")
    assert det["pack"] == pack_id, rec["language"]
    assert recall >= floor, text
    for ms in pack.manifest.must_survive:
        got = survived[ms.name]
        assert got is not None, f"{ms.name}: none of its characters were read\n{text}"
        assert got >= ms.floor, f"{ms.name}: {got:.3f} < {ms.floor}\n{text}"


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
