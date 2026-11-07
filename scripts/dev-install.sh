#!/usr/bin/env bash
# Install pdfintake for development or CI, with English built here from its pinned sources.
#
# pyproject pins pdfintake-data-en (the English models, font and canary) to a release-asset URL.
# This script builds that wheel with src/pdfintake/datawheel.py from the upstream files the
# manifest pins, then points the pin at the local wheel with uv's --overrides, which replaces a
# direct-URL pin. So a fresh clone works before the asset exists, and after any change to packs/en.
#
# usage: scripts/dev-install.sh [--python PY] [--packs en|all|ID[,ID...]] [--extras LIST] [--tool|--wheels-only]
#   (default)  editable install with [kit,test], into $VIRTUAL_ENV or into .venv (made if missing);
#              an active environment takes the install (it says so); afterwards activate .venv
#   --tool     `uv tool install .`: a non-editable install, the way a user gets it (sandbox CI job)
#   --wheels-only  build the --packs wheels into the wheel folder and stop: nothing is installed,
#              and English is built only when --packs names it (pack-publish.yml and release.sh
#              rebuild this way)
#   --packs    the data wheels to build into dist/ (default en; all = every packs/*/manifest.json)
#   --python   the Python uv installs for: a version such as 3.12, or a path (default: uv's choice)
#   --extras   the extras of the editable install (default kit,test; empty for none)
# Environment: PDFINTAKE_DIST is the wheel folder (default dist/); UV is the uv command. A wheel
# already in the folder is reused only when datawheel.wheel_problems finds nothing wrong with it.
# A pack's canary has no upstream source: before a wheel is built, the canary is drawn into the
# pack's files folder with src/pdfintake/kit/render.py, as `pdfintake kit wheel` draws it, in a
# throwaway uv environment holding only the libraries it draws with, at the exact versions
# tools/canary-env.txt pins (never pyproject.toml's ranges, which a new release could satisfy with
# other bytes). datawheel.py then refuses a canary whose bytes differ from the manifest's pin.
# Inside GitHub Actions the install's bin folder is added to GITHUB_PATH, so later steps find
# `python` and `pdfintake`.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

py=""; packs="en"; extras="kit,test"; tool=0; wheels_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    --python) py="${2:?--python needs a value}"; shift 2 ;;
    --packs) packs="${2:?--packs needs a value}"; shift 2 ;;
    --extras) [ $# -ge 2 ] || { echo "dev-install.sh: --extras needs a value" >&2; exit 2; }; extras="$2"; shift 2 ;;
    --tool) tool=1; shift ;;
    --wheels-only) wheels_only=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "dev-install.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ "$tool" = 1 ] && [ "$wheels_only" = 1 ]; then
  echo "dev-install.sh: --wheels-only installs nothing, so it can't take --tool" >&2; exit 2
fi

uv="${UV:-uv}"
command -v "$uv" >/dev/null 2>&1 || { echo "dev-install.sh needs uv (https://docs.astral.sh/uv/)" >&2; exit 3; }
# Any Python 3 runs the build: datawheel.py is standard library only.
boot="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
[ -n "$boot" ] || { echo "dev-install.sh needs a python3 on PATH to build the data wheels" >&2; exit 3; }
[ -f src/pdfintake/datawheel.py ] || { echo "src/pdfintake/datawheel.py is missing" >&2; exit 3; }

dist="${PDFINTAKE_DIST:-dist}"
mkdir -p "$dist"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/helper.py" <<'PY'
"""dev-install.sh's helper: pick the packs, build or reuse their wheels, write the override."""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
WORK = Path(__file__).resolve().parent
# What src/pdfintake/kit/render.py draws a canary with: fpdf2 lays the line out, uharfbuzz shapes it,
# fontTools reads the fonts, pypdfium2 (with pikepdf, which the renderer opens the page with) renders
# it, and Pillow writes the PNG. Installed at the exact versions CANARY_ENV pins them to.
RENDER_LIBS = ("fpdf2", "uharfbuzz", "pillow", "pypdfium2", "pikepdf", "fonttools")
CANARY_ENV = "tools/canary-env.txt"


def die(msg, code=3):
    print(f"dev-install.sh: {msg}", file=sys.stderr)
    sys.exit(code)


def manifest_of(pack):
    return json.loads((ROOT / "packs" / pack / "manifest.json").read_text(encoding="utf-8"))


def pack_ids(spec, wheels_only="0"):
    have = sorted(p.parent.name for p in (ROOT / "packs").glob("*/manifest.json"))
    if "en" not in have and wheels_only != "1":
        die("packs/en/manifest.json is missing: English is built for every install")
    if spec == "all":
        ids = have
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
        missing = [i for i in ids if i not in have]
        if missing:
            die(f"no packs/{missing[0]}/manifest.json", 2)
    if wheels_only == "1":
        print("\n".join(dict.fromkeys(ids)))
    else:
        print("\n".join(["en"] + [i for i in dict.fromkeys(ids) if i != "en"]))


def datawheel():
    spec = importlib.util.spec_from_file_location("pdfintake_datawheel", ROOT / "src" / "pdfintake" / "datawheel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stem(manifest):
    kind = "data" if manifest.get("base") else "lang"
    return f"pdfintake_{kind}_{manifest['id'].replace('-', '_')}"


def prefix(manifest):
    return f"{stem(manifest)}-{manifest['version']}-"


def run(cmd, what):
    # This helper's stdout is a value the shell captures, so every tool's output goes to stderr.
    rc = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=sys.stderr).returncode
    if rc != 0:
        die(f"{what} exited {rc}", 1)


def render_pins():
    """The exact pins CANARY_ENV gives RENDER_LIBS, in RENDER_LIBS order: one name==version line for
    each library and nothing else (blank lines and # comments aside). A range, a second pin or an
    unknown name stops the install, since any of them could change a canary's bytes."""
    try:
        text = (ROOT / CANARY_ENV).read_text(encoding="utf-8")
    except OSError as exc:
        die(f"{CANARY_ENV} can't be read ({exc.strerror or exc}); it pins the canary renderer's libraries", 1)
    found = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([0-9][A-Za-z0-9.+!-]*)", line)
        if not m:
            die(f"{CANARY_ENV} line {n}: {line!r} isn't an exact name==version pin", 1)
        name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
        if name not in RENDER_LIBS:
            die(f"{CANARY_ENV} line {n}: {m.group(1)} isn't a library the canary renderer draws with", 1)
        if name in found:
            die(f"{CANARY_ENV} pins {name} twice", 1)
        found[name] = f"{name}=={m.group(2)}"
    missing = [name for name in RENDER_LIBS if name not in found]
    if missing:
        die(f"{CANARY_ENV} has no pin for {', '.join(missing)}, which the canary renderer needs", 1)
    return [found[name] for name in RENDER_LIBS]


def render_python(uv, py):
    """The throwaway environment's Python, made once per dev-install run (in its work folder)."""
    env = WORK / "render-env"
    exe = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ready = WORK / "render-env.ready"
    if not ready.exists():
        pins = render_pins()
        print(f"dev-install.sh: a throwaway environment draws the canaries ({CANARY_ENV}): " + " ".join(pins),
              file=sys.stderr)
        shutil.rmtree(env, ignore_errors=True)
        run([uv, "venv", *(["--python", py] if py else []), env], "uv venv (the canary environment)")
        run([uv, "pip", "install", "--python", exe, *pins], "uv pip install (the canary environment)")
        ready.write_text("ready\n", encoding="utf-8")
    return exe


def canary_fonts(manifest, files, dw):
    """The fonts the canary is drawn with, as `pdfintake kit wheel` passes them: the pack's own, then
    the base pack's (downloaded once per run, checked against their pins) for any other pack."""
    fonts = [files / f["file"] for f in manifest.get("fonts") or ()]
    if manifest.get("base") is True:
        return fonts
    bases = [p.parent.name for p in sorted((ROOT / "packs").glob("*/manifest.json"))
             if manifest_of(p.parent.name).get("base") is True]
    if len(bases) != 1:
        die(f"the {manifest['id']} canary needs the base pack's fonts, and packs/ holds {len(bases)} base packs", 1)
    base = manifest_of(bases[0])
    folder = WORK / "base-fonts"
    try:
        dw.download_sources({**base, "models": []}, folder)
    except (ValueError, OSError) as exc:
        die(f"the base pack's fonts for the {manifest['id']} canary: {exc}", 1)
    return fonts + [folder / f["file"] for f in base.get("fonts") or ()]


def draw_canary(pack, manifest, files, dw, uv, py):
    canary = manifest["canary"]
    job = WORK / f"canary-{pack}.json"
    job.write_text(json.dumps({
        "src": str(ROOT / "src"),
        "text": " ".join(canary["must_find"]),
        "fonts": [str(f) for f in canary_fonts(manifest, files, dw)],
        "dest": str(files / canary["file"]),
        "dpi": int(canary.get("dpi") or 300),
    }, ensure_ascii=False), encoding="utf-8")
    run([render_python(uv, py), "-I", WORK / "render_canary.py", job], f"drawing the {pack} canary")


def wheel(pack, dist, uv="uv", py=""):
    dist = Path(dist)
    manifest = manifest_of(pack)
    dw = datawheel()
    lead = prefix(manifest)
    found = sorted(p for p in dist.glob("*.whl") if p.name.startswith(lead))
    if len(found) == 1 and not dw.wheel_problems(found[0], manifest):
        print(found[0].resolve())
        return
    out = Path(tempfile.mkdtemp(prefix=f".build-{pack}-", dir=dist))
    try:
        files = out / "sources" / pack
        hint = ""
        if manifest.get("canary"):
            try:
                dw.download_sources(manifest, files)
            except (ValueError, OSError) as exc:
                die(f"the {pack} models and fonts: {exc}", 1)
            draw_canary(pack, manifest, files, dw, uv, py)
            hint = (f". If it names {manifest['canary']['file']}: the canary drawn here differs from the pin in "
                    f"packs/{pack}/manifest.json, so kit/render.py or {CANARY_ENV} has changed since the pin; "
                    f"`pdfintake kit wheel {pack}` draws and pins it again")
        # datawheel.py reports in JSON on stdout. This command's stdout is the wheel path alone (the
        # shell captures it), so the build's report goes to stderr with the rest of the log.
        rc = subprocess.run([sys.executable, str(ROOT / "src" / "pdfintake" / "datawheel.py"), "build",
                             f"packs/{pack}", "--out", str(out), "--files", str(files)],
                            cwd=ROOT, stdout=sys.stderr).returncode
        if rc != 0:
            die(f"datawheel.py build packs/{pack} exited {rc}{hint}", 1)
        built = sorted(out.glob("*.whl"))
        if len(built) != 1:
            die(f"datawheel.py build packs/{pack} wrote {len(built)} wheels, expected 1", 1)
        problems = dw.wheel_problems(built[0], manifest)
        if problems:
            die(f"the {pack} wheel datawheel.py built has problems: " + "; ".join(map(str, problems)), 1)
        # Only this pack's older wheels: the exact stem, so rebuilding `br` never removes `pt-br`'s.
        for old in dist.glob(f"{stem(manifest)}-*.whl"):
            old.unlink()
        dest = dist / built[0].name
        shutil.move(str(built[0]), str(dest))
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(dest.resolve())


def override(wheel_path, dist):
    if not wheel_path.endswith(".whl") or not Path(wheel_path).is_file():
        die(f"no English wheel to point the pin at (got {wheel_path!r})", 1)
    target = Path(dist) / "overrides.txt"
    uri = Path(wheel_path).resolve().as_uri()
    target.write_text(f"pdfintake-data-en @ {uri}\n", encoding="utf-8")
    print(target.resolve())


def venv_python(target):
    target = Path(target)
    print(target / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def venv_bin(target):
    print(Path(target).resolve() / ("Scripts" if os.name == "nt" else "bin"))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"packs": pack_ids, "wheel": wheel, "override": override, "render-pins": lambda: print("\n".join(render_pins())),
     "venv-python": venv_python, "venv-bin": venv_bin}[cmd](*args)
PY

cat > "$work/render_canary.py" <<'PY'
"""dev-install.sh's canary step, run with -I by the throwaway environment's Python: draw one pack's
canary with src/pdfintake/kit/render.py from the job file the helper wrote (text, fonts, dpi, where)."""
import json
import sys
from pathlib import Path

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sys.path.insert(0, job["src"])
from pdfintake.kit.render import render_canary  # noqa: E402

dest = render_canary(job["text"], [Path(f) for f in job["fonts"]], Path(job["dest"]), dpi=int(job["dpi"]))
# One line for the log: when a pin differs on another OS, the pixel digest says whether the drawing
# or only the PNG encoding (Pillow's zlib build) differs.
import hashlib  # noqa: E402
from PIL import Image, features  # noqa: E402

data = dest.read_bytes()
with Image.open(dest) as im:
    pixels, (w, h) = hashlib.sha256(im.tobytes()).hexdigest(), im.size
print(f"dev-install.sh: drew {dest.name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]}; "
      f"pixels {w}x{h} sha256 {pixels[:16]}; zlib {features.version('zlib')}", file=sys.stderr)
PY

# On Windows, Git Bash keeps the \r of Python's \r\n line ends in $(...), which turned the pack id
# "en" into "en\r". Every value the script reads back from the helper goes through here.
helper() { "$boot" "$work/helper.py" "$@" | tr -d '\r'; }

ids="$(helper packs "$packs" "$wheels_only")"
[ -n "$ids" ] || { echo "dev-install.sh: no packs to build" >&2; exit 3; }
en_wheel=""
for id in $ids; do
  w="$(helper wheel "$id" "$dist" "$uv" "$py")"
  echo "data wheel for $id: $w"
  if [ "$id" = en ]; then en_wheel="$w"; fi
done
if [ "$wheels_only" = 1 ]; then
  echo "data wheels built in $dist; nothing installed (--wheels-only)"
  exit 0
fi
overrides="$(helper override "$en_wheel" "$dist")"

pyargs=()
[ -n "$py" ] && pyargs=(--python "$py")

if [ "$tool" = 1 ]; then
  "$uv" tool install --force ${pyargs[@]+"${pyargs[@]}"} --overrides "$overrides" .
  bindir="$("$uv" tool dir --bin | tr -d '\r')"
  echo "pdfintake installed as a uv tool (non-editable), English from $en_wheel"
else
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    target="$VIRTUAL_ENV"
    # An environment that is active when this runs takes the install, even another project's.
    if [ "$(cd "$target" 2>/dev/null && pwd -P)" != "$(cd "$root" && pwd -P)/.venv" ]; then
      echo "dev-install.sh: installing into the active virtual environment $target, not $root/.venv;" \
           "deactivate it first to install into .venv" >&2
    fi
  else
    target="$root/.venv"
    [ -d "$target" ] || "$uv" venv ${pyargs[@]+"${pyargs[@]}"} "$target"
  fi
  spec="."
  [ -n "$extras" ] && spec=".[$extras]"
  "$uv" pip install --python "$(helper venv-python "$target")" --overrides "$overrides" -e "$spec"
  bindir="$(helper venv-bin "$target")"
  echo "pdfintake installed (editable, $spec) in $target, English from $en_wheel"
  # pdfintake is on the PATH only inside the environment: say how to get there.
  if [ "${VIRTUAL_ENV:-}" != "$target" ]; then
    case "$(uname -s 2>/dev/null)" in
      MINGW*|MSYS*|CYGWIN*) activate="$target/Scripts/activate" ;;
      *) activate="$target/bin/activate" ;;
    esac
    echo "next: source \"$activate\" (or run $bindir/pdfintake)"
  fi
fi

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$bindir" >> "$GITHUB_PATH"
fi
