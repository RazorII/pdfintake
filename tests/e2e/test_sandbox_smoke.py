"""The sandbox smoke test (DESIGN §7.2): a non-editable install, then ``doctor`` and one run.

The CI ``sandbox`` job installs pdfintake the way a user gets it (``uv tool install .`` in a root
container like the claude.ai sandbox) and runs this file from a separate virtual environment that
holds only pytest. So this file imports nothing from pdfintake: it drives the ``pdfintake`` on
PATH, and asks that install's own interpreter the questions only it can answer.

A package-data file missing from the wheel (``langpacks/index.json``,
``langpacks/manifest.schema.json``, ``ocr/calibration.json``) fails here, where an editable install
would read it from the source tree and hide it.

Everywhere else the file skips with the reason: no ``pdfintake`` on PATH, or an editable one (every
other CI job, and a developer's checkout). In a job that must run it, those skips are failures:
the CI ``sandbox`` job itself (``GITHUB_ACTIONS=true`` with ``GITHUB_JOB=sandbox``), so the guarantee
doesn't hang on the workflow setting a variable, and anywhere ``PDFINTAKE_SANDBOX_SMOKE=1`` is set.
Every text is made up.

The ``install-smoke`` workflow runs the same file against a published release installed from its tag
on Linux, Windows and macOS, and sets three more variables:

* ``PDFINTAKE_SMOKE_PYTHON``: the install's own interpreter. A Windows ``pdfintake.exe`` is a
  launcher with no shebang line to read it from.
* ``PDFINTAKE_SMOKE_VERSION``: the version the install must report (the tag without its ``v``).
* ``PDFINTAKE_SMOKE_FETCH``: a pack id. A page in that language is drawn and run with the default
  ``--fetch auto``, so the pack comes from the release's own assets, as a user's first run gets it.
  Unset (the CI ``sandbox`` job, which installs an unreleased checkout), that test skips.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.cli_subprocess

SANDBOX_JOB = "sandbox"                     # the job id in .github/workflows/test.yml
REQUIRED = (os.environ.get("PDFINTAKE_SANDBOX_SMOKE") == "1"
            or (os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_JOB") == SANDBOX_JOB))
PACKAGE_DATA = (("pdfintake.langpacks", "index.json"), ("pdfintake.langpacks", "manifest.schema.json"),
                ("pdfintake.ocr", "calibration.json"))
LINES = ["Volcanoes form where hot rock rises from deep",
         "below the ground and cools into new mountains."]
# A page per pack the fetch test can draw with the base pack's Latin font (word, must appear in the layer).
FETCH_PAGES: dict[str, tuple[list[str], tuple[str, ...]]] = {
    "de": (["Das Wasser fließt von den Bergen hinunter ins Meer,",
            "und die Sonne wärmt es, bis es als Dampf aufsteigt.",
            "Über dem Land kühlt der Dampf ab und bildet Wolken."], ("fließt", "wärmt", "kühlt")),
    "hu": (["A víz körforgása a felhőkből és a folyókból áll,",
            "és a hűvös erdő is őrzi a nedvességet egész nyáron.",
            "A tavaszi eső után a föld puha lesz, és a fű nő."], ("felhőkből", "hűvös", "őrzi")),
}
TIMEOUT_S = 900
PYTHON_ENV, VERSION_ENV, FETCH_ENV = "PDFINTAKE_SMOKE_PYTHON", "PDFINTAKE_SMOKE_VERSION", "PDFINTAKE_SMOKE_FETCH"


def _not_here(why: str) -> None:
    if REQUIRED:
        pytest.fail(f"the sandbox smoke test must run here (the CI {SANDBOX_JOB} job or "
                    f"PDFINTAKE_SANDBOX_SMOKE=1), but {why}", pytrace=False)
    pytest.skip(f"the sandbox smoke test runs on a non-editable install only (the CI sandbox job): {why}")


def _interpreter(cli: str) -> str:
    """The Python the ``pdfintake`` script runs on: ``PDFINTAKE_SMOKE_PYTHON`` when set, otherwise
    the script's shebang line (a uv tool or venv script)."""
    named = os.environ.get(PYTHON_ENV)
    if named:
        if not Path(named).is_file():
            _not_here(f"{PYTHON_ENV} names {named}, which isn't a file")
        return named
    try:
        with open(cli, "rb") as fh:
            first = fh.readline(4096)
    except OSError as exc:
        _not_here(f"{cli} can't be read: {exc}")
    if not first.startswith(b"#!"):
        _not_here(f"{cli} isn't a script with a shebang line (a Windows launcher?)")
    parts = first[2:].decode("utf-8", "replace").strip().split()
    if not parts:
        _not_here(f"{cli} has an empty shebang line")
    if Path(parts[0]).name == "env" and len(parts) > 1:      # #!/usr/bin/env python3
        found = shutil.which(parts[1])
        if found is None:
            _not_here(f"{cli} names {parts[1]!r}, which isn't on PATH")
        return found
    return parts[0]


def _env(cache: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PDFINTAKE_CACHE"] = str(cache)
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONPATH", None)                              # the install alone, never the checkout's src/
    return env


def _python(py: str, code: str, cache: Path, *args: str) -> Any:
    """Run ``code`` in the install's own interpreter (isolated mode) and parse the JSON it prints."""
    proc = subprocess.run([py, "-I", "-c", code, *args], capture_output=True, text=True, encoding="utf-8",
                          env=_env(cache), timeout=TIMEOUT_S)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout)


def _cli(cli: str, cache: Path, *args: str) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run([cli, *args, "--json"], capture_output=True, text=True, encoding="utf-8",
                          env=_env(cache), timeout=TIMEOUT_S)
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        raise AssertionError(f"pdfintake {' '.join(args)} exited {proc.returncode} with no JSON: "
                             f"{proc.stdout[-2000:]!r} {proc.stderr[-2000:]!r}") from None
    return proc.returncode, data


@pytest.fixture(scope="module")
def install(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    cli = shutil.which("pdfintake")
    if cli is None:
        _not_here("there is no pdfintake on PATH")
    py = _interpreter(cli)
    cache = tmp_path_factory.mktemp("smoke-cache")
    facts = _python(py, (
        "import importlib.metadata as m, json, sys\n"
        "d = m.distribution('pdfintake')\n"
        "direct = json.loads(d.read_text('direct_url.json') or '{}')\n"
        "import pdfintake\n"
        "print(json.dumps({'version': d.version, 'editable': bool(direct.get('dir_info', {}).get('editable')),\n"
        "                  'package': pdfintake.__file__, 'prefix': sys.prefix}))\n"), cache)
    if facts["editable"]:
        _not_here(f"the pdfintake on PATH ({cli}) is an editable install")
    return {"cli": cli, "python": py, "cache": cache, **facts}


def test_the_install_is_the_version_asked_for(install: dict[str, Any]):
    want = os.environ.get(VERSION_ENV)
    if not want:
        pytest.skip(f"{VERSION_ENV} isn't set (only the install-smoke workflow names a version)")
    assert install["version"] == want


def test_the_install_is_not_the_checkout(install: dict[str, Any]):
    package = Path(install["package"]).resolve()
    checkout = Path(__file__).resolve().parents[2]
    assert checkout not in package.parents, f"pdfintake is imported from the checkout: {package}"
    assert Path(install["prefix"]).resolve() in package.parents


def test_the_package_data_ships_in_the_wheel(install: dict[str, Any]):
    found = _python(install["python"], (
        "import json, sys\n"
        "from importlib import resources\n"
        "out = {}\n"
        "for spec in sys.argv[1:]:\n"
        "    pkg, name = spec.split(':')\n"
        "    res = resources.files(pkg).joinpath(name)\n"
        "    out[spec] = {'file': res.is_file(), 'bytes': len(res.read_bytes()) if res.is_file() else 0,\n"
        "                 'path': str(res)}\n"
        "print(json.dumps(out))\n"), install["cache"], *(f"{p}:{n}" for p, n in PACKAGE_DATA))
    for spec, info in found.items():
        assert info["file"] and info["bytes"] > 0, f"{spec} is missing from the installed package: {info}"
        json.loads(Path(info["path"]).read_text(encoding="utf-8"))       # and it is JSON
    loaded = _python(install["python"], (
        "import json\n"
        "from pdfintake.langpacks import index\n"
        "idx = index.load()\n"
        "print(json.dumps({'base': idx.base().id, 'packs': [e.id for e in idx.entries]}))\n"), install["cache"])
    assert loaded["base"] == "en" and "en" in loaded["packs"]


def test_doctor_says_ready(install: dict[str, Any]):
    code, data = _cli(install["cli"], install["cache"], "doctor")
    assert code == 0 and data["exit"] == 0, data.get("problems")
    assert data["base_pack"]["ok"] is True and data["base_pack"]["id"] == "en"
    assert data["index"]["ok"] is True


def _draw(install: dict[str, Any], image: Path, lines: list[str]) -> None:
    """A page of ``lines`` drawn by the install's own Pillow, with the base pack's font: this
    environment has neither, and only pytest."""
    _python(install["python"], (
        "import json, sys\n"
        "from PIL import Image, ImageDraw, ImageFont\n"
        "from pdfintake.langpacks import store\n"
        "font = ImageFont.truetype(str(store.base().asset_path('noto-sans')), 44)\n"
        "img = Image.new('L', (1500, 120 + 110 * (len(sys.argv) - 2)), 255)\n"
        "draw = ImageDraw.Draw(img)\n"
        "for k, line in enumerate(sys.argv[2:]):\n"
        "    draw.text((80, 100 + 110 * k), line, font=font, fill=0)\n"
        "img.save(sys.argv[1], dpi=(200, 200))\n"
        "print(json.dumps({'ok': True}))\n"), install["cache"], str(image), *lines)


def _layer_text(install: dict[str, Any], pdf: Path) -> str:
    return _python(install["python"], (
        "import json, sys\n"
        "import pypdfium2 as pdfium\n"
        "doc = pdfium.PdfDocument(sys.argv[1])\n"
        "page = doc[0]\n"
        "tp = page.get_textpage()\n"
        "text = tp.get_text_bounded()\n"
        "tp.close(); page.close(); doc.close()\n"
        "print(json.dumps({'text': text}))\n"), install["cache"], str(pdf))["text"]


def test_one_run_writes_a_searchable_pdf(install: dict[str, Any], tmp_path: Path):
    folder = tmp_path / "in"
    folder.mkdir()
    image = folder / "volcano note.png"
    _draw(install, image, LINES)
    before = image.read_bytes()
    code, data = _cli(install["cli"], install["cache"], "run", str(image))
    assert code == 0 and data["exit"] == 0, data
    (rec,) = data["records"]
    out = Path(rec["output"]["path"])
    assert out == folder / "volcano note (searchable).pdf" and out.is_file()
    assert image.read_bytes() == before
    text = _layer_text(install, out).lower()
    for word in ("volcanoes", "mountains"):
        assert word in text, text


def test_a_first_run_fetches_the_pack_it_needs_and_reads_with_it(install: dict[str, Any], tmp_path: Path):
    pack_id = os.environ.get(FETCH_ENV)
    if not pack_id:
        pytest.skip(f"{FETCH_ENV} isn't set (only the install-smoke workflow fetches from a published release)")
    if pack_id not in FETCH_PAGES:
        pytest.fail(f"{FETCH_ENV}={pack_id!r}: this file has pages for {', '.join(sorted(FETCH_PAGES))}",
                    pytrace=False)
    lines, words = FETCH_PAGES[pack_id]
    image = tmp_path / f"{pack_id} note.png"
    _draw(install, image, lines)
    code, data = _cli(install["cli"], install["cache"], "run", str(image))
    assert code == 0 and data["exit"] == 0, data
    (rec,) = data["records"]
    pages = rec["language"]["detection"]["pages"]
    assert [p["pack"] for p in pages] == [pack_id], rec["language"]
    code, doctor = _cli(install["cli"], install["cache"], "doctor")
    assert code == 0, doctor.get("problems")
    text = _layer_text(install, Path(rec["output"]["path"]))
    for word in words:
        assert word in text, text
