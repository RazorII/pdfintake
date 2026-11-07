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
TIMEOUT_S = 900


def _not_here(why: str) -> None:
    if REQUIRED:
        pytest.fail(f"the sandbox smoke test must run here (the CI {SANDBOX_JOB} job or "
                    f"PDFINTAKE_SANDBOX_SMOKE=1), but {why}", pytrace=False)
    pytest.skip(f"the sandbox smoke test runs on a non-editable install only (the CI sandbox job): {why}")


def _interpreter(cli: str) -> str:
    """The Python the ``pdfintake`` script runs on, from its shebang line (a uv tool or venv script)."""
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


def test_one_run_writes_a_searchable_pdf(install: dict[str, Any], tmp_path: Path):
    folder = tmp_path / "in"
    folder.mkdir()
    image = folder / "volcano note.png"
    # The page is drawn by the install's own Pillow, with the base pack's font: this environment has
    # neither, and only pytest.
    _python(install["python"], (
        "import json, sys\n"
        "from PIL import Image, ImageDraw, ImageFont\n"
        "from pdfintake.langpacks import store\n"
        "font = ImageFont.truetype(str(store.base().asset_path('noto-sans')), 44)\n"
        "img = Image.new('L', (1500, 500), 255)\n"
        "draw = ImageDraw.Draw(img)\n"
        "for k, line in enumerate(sys.argv[2:]):\n"
        "    draw.text((80, 100 + 110 * k), line, font=font, fill=0)\n"
        "img.save(sys.argv[1], dpi=(200, 200))\n"
        "print(json.dumps({'ok': True}))\n"), install["cache"], str(image), *LINES)
    before = image.read_bytes()
    code, data = _cli(install["cli"], install["cache"], "run", str(image))
    assert code == 0 and data["exit"] == 0, data
    (rec,) = data["records"]
    out = Path(rec["output"]["path"])
    assert out == folder / "volcano note (searchable).pdf" and out.is_file()
    assert image.read_bytes() == before
    text = _python(install["python"], (
        "import json, sys\n"
        "import pypdfium2 as pdfium\n"
        "doc = pdfium.PdfDocument(sys.argv[1])\n"
        "page = doc[0]\n"
        "tp = page.get_textpage()\n"
        "text = tp.get_text_bounded()\n"
        "tp.close(); page.close(); doc.close()\n"
        "print(json.dumps({'text': text}))\n"), install["cache"], str(out))["text"].lower()
    for word in ("volcanoes", "mountains"):
        assert word in text, text
