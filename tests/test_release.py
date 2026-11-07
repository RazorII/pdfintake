"""The command, the plugin and the skill carry one version, and a release can only go out whole.

Claude Code only hands out a new copy of a plugin when its version changes, and the skill upgrades
the command when it's older than the version the skill names. So all three move together on every
release. pdfintake also pins its data wheels: the pyproject pin of the English wheel must equal the
index's ``en`` entry, every published pack needs a canary, and ``index.json`` must regenerate
identically (strictly at release: nothing in packs/ ahead of it).

The rest checks the release machinery itself, never against GitHub:

* the workflows keep their safety properties (pack-bench on ``pull_request`` only with a read-only
  token and no secrets; pack-publish on ``workflow_dispatch`` behind an environment, uploading to
  GitHub releases, no PyPI and no stored token);
* ``scripts/release.sh`` runs against a fake ``git``, ``gh`` and release scanner in a temporary
  repo. It refuses to run at all without a usable ``RELEASE_SCAN``, and refuses to tag on every
  gate: a skipped or failed repo test, the Vision gate, immutable releases off, a wheel whose hash
  differs from its pin, an earlier release that doesn't hold its wheel immutably, and a scanner
  finding. It rebuilds the wheels it hosts through the real ``scripts/dev-install.sh --wheels-only``
  (a fake ``uv`` and ``datawheel.py``), which draws each canary before the build. Before the tag it
  records the pack versions the release publishes in ``tools/published-packs.json`` and commits that
  file, once. After the tag, it follows GitHub's
  documented order for immutable releases: create a draft, attach every asset to it, check every
  digest, publish. It publishes only a draft whose
  assets carry exactly the pinned digests. The fake ``gh`` behaves
  as GitHub does once a release is immutable (no asset added, no edit), and no real ``gh`` or
  network is ever reached;
* ``scripts/vision-local.sh`` passes only when Vision tests ran and none skipped;
* ``scripts/dev-install.sh`` builds or reuses the English wheel and installs with ``--overrides``,
  and draws canaries in an environment holding exactly ``tools/canary-env.txt``'s pins.

Files other units write (the plugin, the index, the index tool) skip while missing and fail under
PDFINTAKE_RELEASE_STRICT=1, which scripts/release.sh sets.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest

from pdfintake import __version__

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "claude-plugin"
SKILL = PLUGIN / "skills" / "pdfintake" / "SKILL.md"
INDEX = ROOT / "src" / "pdfintake" / "langpacks" / "index.json"
BUILD_INDEX = ROOT / "tools" / "build_index.py"
CANARY_ENV = ROOT / "tools" / "canary-env.txt"
WORKFLOWS = ROOT / ".github" / "workflows"
SCRIPTS = ROOT / "scripts"
PIN = re.compile(r"^pdfintake-data-en\s*@\s*(\S+?)#sha256=([0-9a-f]{64})\s*$")
ASSET_URL = re.compile(r"^https://github\.com/RazorII/pdfintake/releases/download/v\d+\.\d+\.\d+/([A-Za-z0-9_.+-]+\.whl)$")
PLACEHOLDER_SHA = "0" * 64

posix = pytest.mark.skipif(sys.platform == "win32", reason="the release scripts are bash; releases are cut on the Mac")


def _strict() -> bool:
    return os.environ.get("PDFINTAKE_RELEASE_STRICT") == "1"


def _not_yet(msg: str) -> None:
    if _strict():
        pytest.fail(msg)
    pytest.skip(msg)


def _need(path: Path) -> Path:
    if not path.exists():
        _not_yet(f"{path.relative_to(ROOT)} isn't there yet")
    return path


def _pin() -> tuple[str, str]:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    pins = [m for m in (PIN.match(d) for d in deps) if m]
    assert len(pins) == 1, "pyproject.toml needs exactly one 'pdfintake-data-en @ URL#sha256=HASH' dependency"
    return pins[0].group(1), pins[0].group(2)


def _index() -> dict[str, Any]:
    return json.loads(_need(INDEX).read_text(encoding="utf-8"))


def _wheel_name(distribution: str, version: str) -> str:
    return f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"


# --- one version everywhere (gdocout's, with the name swapped)

def test_plugin_version_matches_package():
    manifest = json.loads(_need(PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["version"] == __version__


def test_skill_names_the_current_version():
    skill = _need(SKILL).read_text(encoding="utf-8")
    named = set(re.findall(r"pdfintake (\d+\.\d+\.\d+) or newer", skill)) | set(re.findall(r"older than (\d+\.\d+\.\d+)", skill))
    assert named == {__version__}, named


def test_skill_installs_its_own_release():
    skill = _need(SKILL).read_text(encoding="utf-8")
    pins = set(re.findall(r"github\.com/RazorII/pdfintake@v(\d+\.\d+\.\d+)", skill))
    assert pins == {__version__}, pins
    assert not re.search(r"github\.com/RazorII/pdfintake(?!@v|/)", skill), "an install line is not pinned"


# --- the data wheel pins

def test_pyproject_pin_matches_the_index_en_entry():
    url, sha = _pin()
    entries = [e for e in _index()["packs"] if e.get("base")]
    assert [e["id"] for e in entries] == ["en"], "the index needs exactly one base pack, en"
    en = entries[0]
    if sha == PLACEHOLDER_SHA:
        _not_yet("the pyproject pin is still the placeholder (scripts/dev-install.sh overrides it until release)")
    assert sha == en["dist"]["sha256"], f"pyproject pins {sha}, the index's en wheel is {en['dist']['sha256']}"
    m = ASSET_URL.match(url)
    assert m, f"the English pin {url} isn't a RazorII/pdfintake release asset"
    assert m.group(1) == _wheel_name(en["dist"]["distribution"], en["version"])


def test_every_fetched_pack_is_a_release_asset():
    for entry in _index()["packs"]:
        if entry.get("base"):
            continue
        dist = entry["dist"]
        assert dist["kind"] == "fetch", entry["id"]
        m = ASSET_URL.match(dist.get("url") or "")
        assert m, f"{entry['id']}: {dist.get('url')} isn't a RazorII/pdfintake release asset"
        assert m.group(1) == _wheel_name(dist["distribution"], entry["version"])
        assert re.fullmatch(r"[0-9a-f]{64}", dist["sha256"]), entry["id"]


def test_every_published_index_entry_has_a_canary():
    for entry in _index()["packs"]:
        manifest = ROOT / "packs" / entry["id"] / "manifest.json"
        assert manifest.is_file(), f"the index publishes {entry['id']}, but packs/{entry['id']}/manifest.json is missing"
        canary = json.loads(manifest.read_text(encoding="utf-8")).get("canary")
        assert canary and re.fullmatch(r"[0-9a-f]{64}", canary.get("sha256", "")) and canary.get("bytes", 0) > 0, \
            f"{entry['id']} is published without a pinned canary"


def test_index_regenerates_identically():
    _index()
    spec = importlib.util.spec_from_file_location("pdfintake_build_index", _need(BUILD_INDEX))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A pack PR may hold packs/ ahead of the index (a note); the release's strict run may not.
    argv = ["--check", "--strict"] if _strict() else ["--check"]
    assert module.main(argv) == 0, "src/pdfintake/langpacks/index.json differs from a regeneration"


# --- the workflows

yaml = pytest.importorskip("yaml")


def _wf(name: str) -> dict[str, Any]:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))           # YAML 1.1 reads the key `on` as true
    return data


def _uses(wf: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in wf["jobs"].values() for step in job.get("steps", [])]


def test_no_workflow_uses_pull_request_target_or_stored_secrets():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
        assert "pull_request_target" not in text, path.name
        assert "secrets." not in text, f"{path.name} reads a stored secret"
        for step in _uses(_wf(path.name)):
            if str(step.get("uses", "")).startswith("actions/checkout"):
                assert step.get("with", {}).get("persist-credentials") is False, f"{path.name}: checkout keeps its token"


def test_pack_bench_is_pull_request_only_and_read_only():
    wf = _wf("pack-bench.yml")
    assert list(wf["on"]) == ["pull_request"]
    assert set(wf["on"]["pull_request"]["paths"]) == {"packs/**", "src/pdfintake/ocr/**"}
    assert wf["permissions"] == {"contents": "read"}
    for job in wf["jobs"].values():
        assert "permissions" not in job and "environment" not in job
    text = (WORKFLOWS / "pack-bench.yml").read_text(encoding="utf-8")
    assert "kit verify" in text and "--rebench" in text and "GITHUB_STEP_SUMMARY" in text and "needs-review" in text
    # A pack id from the PR reaches the shell only through an environment variable.
    assert "${{ matrix.pack }}" not in "".join(s.get("run", "") for s in _uses(wf))


def test_pack_publish_uploads_to_github_releases_behind_an_approval():
    wf = _wf("pack-publish.yml")
    assert list(wf["on"]) == ["workflow_dispatch"]
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    assert job["environment"] == "pack-publish"
    assert job["permissions"] == {"contents": "write"}
    text = (WORKFLOWS / "pack-publish.yml").read_text(encoding="utf-8").lower()
    assert "gh release upload" in text and "--clobber" not in text
    assert "github.token" in text
    for word in ("pypi", "twine", "pypa/", "id-token"):
        assert word not in text, word
    assert "${{ inputs.tag }}" not in "".join(s.get("run", "") for s in _uses(wf))
    # The token is in the env of the gh steps only, never of the job or the rebuild that downloads.
    assert "GH_TOKEN" not in job.get("env", {})
    holders = [st for st in job["steps"] if "GH_TOKEN" in st.get("env", {})]
    assert holders and all("gh " in st["run"] for st in holders)
    # The rebuild is found by the command it runs as well as by name, so rewording a comment can't drop the check.
    rebuild = [st for st in job["steps"]
               if "scripts/dev-install.sh" in st.get("run", "") or "datawheel.py" in st.get("run", "")]
    assert rebuild and all("GH_TOKEN" not in st.get("env", {}) for st in rebuild)


def test_resolve_runs_weekly():
    wf = _wf("resolve.yml")
    assert set(wf["on"]) == {"schedule", "workflow_dispatch"}
    (cron,) = [c["cron"] for c in wf["on"]["schedule"]]
    minute, hour, dom, month, dow = cron.split()
    assert dom == "*" and month == "*" and dow.isdigit()
    text = (WORKFLOWS / "resolve.yml").read_text(encoding="utf-8")
    assert "uv pip compile" in text
    # A source-only dependency would need a native build the sandbox can't do: wheels only.
    assert "--only-binary :all:" in text


def test_the_test_matrix_has_every_job_of_the_design():
    wf = _wf("test.yml")
    assert set(wf["jobs"]) == {"suite", "sandbox", "network", "offline"}
    names = {r["job"] for r in wf["jobs"]["suite"]["strategy"]["matrix"]["include"]}
    assert names == {"mac-floor", "mac-new", "win-311", "win-314", "linux-sandbox-base", "linux-new", "linux-arm"}
    for job_name in ("suite", "sandbox", "network", "offline"):
        runs = [s.get("run", "") for s in wf["jobs"][job_name]["steps"]]
        install = next(i for i, r in enumerate(runs) if "scripts/dev-install.sh" in r)
        doctor = next(i for i, r in enumerate(runs) if "pdfintake doctor --json" in r)
        tests = max(i for i, r in enumerate(runs) if "pytest" in r)
        assert install < doctor <= tests, job_name
        if doctor == tests:
            assert runs[doctor].index("pdfintake doctor") < runs[doctor].rindex("pytest"), job_name
    assert wf["jobs"]["sandbox"]["container"]["image"] == "ubuntu:24.04"
    sandbox = "".join(s.get("run", "") for s in wf["jobs"]["sandbox"]["steps"])
    assert "externally-managed-environment" in sandbox and "--tool" in sandbox and "test_sandbox_smoke.py" in sandbox
    assert wf["jobs"]["network"]["env"]["PDFINTAKE_NETWORK_TESTS"] == "1"
    network = [s.get("run", "") for s in wf["jobs"]["network"]["steps"]]
    assert any("tests/e2e/test_packs_network.py" in r for r in network)
    # §11.2: the Thai and Hungarian Tesseract reads run here, or they skip on every job.
    assert any(re.search(r"pytest\s+-m\s+network\s+tests/test_tesseract\.py", r) for r in network), network
    assert "unshare -rn" in "".join(s.get("run", "") for s in wf["jobs"]["offline"]["steps"])
    for job_name in ("suite", "sandbox", "network", "offline"):
        steps = wf["jobs"][job_name]["steps"]
        cache = [i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith("actions/cache")]
        install = next(i for i, s in enumerate(steps) if "scripts/dev-install.sh" in s.get("run", ""))
        assert cache and cache[0] < install, f"{job_name} doesn't restore the data wheels before installing"
        assert "hashFiles('packs/*/manifest.json')" in steps[cache[0]]["with"]["key"], job_name
        assert "dist" in str(steps[cache[0]]["with"]["path"]).split(), job_name
    suite_runs = " ".join(r for r in (s.get("run", "") for s in wf["jobs"]["suite"]["steps"]))
    assert "--canary" in suite_runs


def test_codeowners_require_razorii_on_packs():
    lines = [l.split() for l in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    owners = {l[0]: l[1:] for l in lines}
    assert owners["*"] == ["@RazorII"] and owners["/packs/"] == ["@RazorII"]


def test_pull_request_templates():
    pack = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / "language-pack.md").read_text(encoding="utf-8")
    code = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE" / "code.md").read_text(encoding="utf-8")
    assert "packs/<id>/**" in pack and "kit report" in pack and "kit verify" in pack
    assert "docs/add-a-language.md" in pack and "code.md" in pack
    assert "tests/test_hygiene.py" in code and "version" in code


# --- release.sh, against fakes

def test_release_sh_is_docpdfs_with_the_name_swapped():
    text = (SCRIPTS / "release.sh").read_text(encoding="utf-8")
    for line in ('grep -qx "__version__ = \\"$v\\"" src/pdfintake/__init__.py',
                 'publisher="RazorII"',
                 'git tag -a "$tag" -m "pdfintake $v"',
                 'git push --atomic origin HEAD "refs/tags/$tag"',
                 'git ls-remote --exit-code --tags origin "refs/tags/$tag"',
                 '[ -z "$(git status --porcelain)" ]'):
        assert line in text, line
    assert "docpdf" not in text and "gdocout" not in text
    assert "pypi" not in text.lower() and "twine" not in text


FAKE_GIT = r'''
import json, os, sys
state_path = os.environ["FAKE_STATE"]
state = json.load(open(state_path, encoding="utf-8"))
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("git " + " ".join(a) + "\n")
def out(s=""):
    print(s)
def save():
    json.dump(state, open(state_path, "w", encoding="utf-8"))
if a == ["rev-parse", "--show-toplevel"]:
    out(state["root"])
elif a == ["status", "--porcelain"]:
    sys.stdout.write(state["porcelain"])
elif a[0] == "log":
    out("\n".join(state["authors"]))
elif a == ["config", "user.name"]:
    out(state["user"])
elif a[:2] == ["cat-file", "-t"]:
    tag = a[2].rsplit("/", 1)[1]
    if tag in state["tags"]:
        out(state["tag_types"].get(tag, "tag"))
    else:
        sys.exit(128)
elif a[0] == "rev-parse" and a[1] == "HEAD":
    out(state["head"])
elif a[0] == "rev-parse":
    tag = a[1].split("/")[-1].split("^")[0]
    if tag not in state["tags"]:
        sys.exit(128)
    out(state["tags"][tag])
elif a[0] == "tag":
    tag = a[2]
    if tag in state["tags"]:
        print(f"fatal: tag '{tag}' already exists", file=sys.stderr)
        sys.exit(128)
    state["tags"][tag] = state["head"]
    save()
elif a[:2] == ["push", "--atomic"]:
    if not state["push_ok"]:
        sys.exit(1)
    state["remote_tags"].append(a[4].rsplit("/", 1)[1])
    save()
elif a[:2] == ["add", "--"]:
    state.setdefault("staged", []).extend(a[2:])
    save()
elif a[0] == "commit":
    # release.sh commits only the published record, by path, with a message.
    paths = a[a.index("--") + 1:]
    assert "-m" in a and paths and set(paths) <= set(state.get("staged", [])), a
    state["head"] = state["head"] + "+"
    state.setdefault("commits", []).append({"message": a[a.index("-m") + 1], "paths": paths, "head": state["head"]})
    state["staged"] = []
    save()
elif a[0] == "ls-remote":
    sys.exit(0 if a[-1].rsplit("/", 1)[1] in state["remote_tags"] else 2)
else:
    print("fake git: unexpected " + " ".join(a), file=sys.stderr)
    sys.exit(99)
'''

FAKE_GH = r'''
import json, os, sys
state_path = os.environ["FAKE_STATE"]
state = json.load(open(state_path, encoding="utf-8"))
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("gh " + " ".join(a) + "\n")
def save():
    json.dump(state, open(state_path, "w", encoding="utf-8"))
def emit(obj):
    print(json.dumps(obj))
def opt(name):
    return a[a.index(name) + 1]
def event(*e):
    state.setdefault("events", []).append(list(e))
slug = "repos/RazorII/pdfintake"
def gh_failure(kind):
    # What gh itself prints when the call never gets a proper answer: nothing on stdout, exit 1.
    print({"network": "error connecting to api.github.com\ncheck your internet connection or https://githubstatus.com",
           "auth": "gh: Bad credentials (HTTP 401)"}[kind], file=sys.stderr)
    sys.exit(1)
if a[0] == "api":
    path = a[1]
    if path in (f"{slug}/environments/pack-publish", f"{slug}/immutable-releases") and \
            path.rsplit("/", 1)[1] in state.get("gh_fails", {}):
        gh_failure(state["gh_fails"][path.rsplit("/", 1)[1]])
    if path == f"{slug}/environments/pack-publish":
        if state["environment"] is None:
            emit({"message": "Not Found", "status": "404"})
            sys.exit(1)
        emit(state["environment"])
    elif path == f"{slug}/immutable-releases":
        # GitHub answers 200 {enabled, enforced_by_owner} when they're on and 404 when they're off;
        # "disabled" is a 200 that says enabled: false, which must refuse just the same.
        if state["immutable"] == "disabled":
            emit({"enabled": False, "enforced_by_owner": False})
        elif state["immutable"]:
            emit({"enabled": True, "enforced_by_owner": False})
        else:
            emit({"message": "Not Found", "status": "404"})
            sys.exit(1)
    elif path.startswith(f"{slug}/releases/tags/"):
        tag = path.rsplit("/", 1)[1]
        rel = [r for r in state["releases"] if r["tag_name"] == tag and not r["draft"]]
        if not rel:
            emit({"message": "Not Found"})
            sys.exit(1)
        emit(rel[0])
    elif path.startswith(f"{slug}/releases?"):
        emit(state["releases"])
    elif path.startswith(f"{slug}/actions/workflows/pack-publish.yml/runs"):
        emit({"workflow_runs": state["runs"]})
    elif path.startswith(f"{slug}/releases/"):
        rid = int(path.rsplit("/", 1)[1])
        rel = [r for r in state["releases"] if r["id"] == rid][0]
        event("read", rel["tag_name"], rel["draft"], sorted(x["name"] for x in rel["assets"]))
        save()
        emit(rel)
    else:
        sys.exit(99)
elif a[:2] == ["release", "create"]:
    # Only these options: a file named on the command line would be attached at creation, before
    # the draft exists for the digest check.
    rest, i = a[3:], 0
    while i < len(rest):
        if rest[i] in ("--repo", "--title", "--notes"):
            i += 2
        elif rest[i] in ("--draft", "--verify-tag"):
            i += 1
        else:
            print("fake gh: unexpected release create argument " + rest[i], file=sys.stderr)
            sys.exit(99)
    assert "--draft" in a and "--verify-tag" in a and opt("--repo") == "RazorII/pdfintake"
    state["releases"].insert(0, {"id": 500 + len(state["releases"]), "tag_name": a[2], "draft": True,
                                 "immutable": False, "assets": []})
    event("create", a[2], True, [])
    save()
elif a[:2] == ["workflow", "run"]:
    assert a[2] == "pack-publish.yml" and opt("--ref") == a[a.index("-f") + 1].split("=", 1)[1]
    state["runs"].insert(0, {"id": 900 + len(state["runs"]), "head_branch": opt("--ref")})
    save()
elif a[:2] == ["run", "watch"]:
    assert "--exit-status" in a
    run = [r for r in state["runs"] if r["id"] == int(a[2])][0]
    attach = state["ci_fail_assets"] if state["ci_fail"] else state["ci_assets"]
    for rel in state["releases"]:
        if rel["tag_name"] == run["head_branch"] and not rel["draft"] and rel["immutable"]:
            print("fake gh: cannot upload assets to an immutable release", file=sys.stderr)
            sys.exit(1)
        if rel["tag_name"] == run["head_branch"] and rel["draft"]:
            event("attach", rel["tag_name"], rel["draft"], sorted(attach))
            have = {x["name"]: x["digest"] for x in rel["assets"]}
            if not state["ci_fail"]:
                # pack-publish uploads only what's missing, and never over an existing asset
                assert not set(have) & set(attach) - {n for n in have if have[n] == attach.get(n)}, have
            have.update(attach)
            rel["assets"] = [{"name": n, "digest": d} for n, d in sorted(have.items())]
    save()
    if state["ci_fail"]:
        sys.exit(1)
elif a[:2] == ["release", "edit"]:
    assert "--draft=false" in a
    for rel in state["releases"]:
        if rel["tag_name"] == a[2]:
            if not rel["draft"] and rel["immutable"]:
                print("fake gh: an immutable release can't be edited", file=sys.stderr)
                sys.exit(1)
            rel["draft"] = False
            rel["immutable"] = state["publish_immutable"]
            event("publish", rel["tag_name"], False, sorted(x["name"] for x in rel["assets"]))
    save()
else:
    print("fake gh: unexpected " + " ".join(a), file=sys.stderr)
    sys.exit(99)
'''

FAKE_DATAWHEEL = r'''
"""A stand-in for datawheel.py: a deterministic 'wheel' per pack, made of its manifest's words."""
import json, os, sys
from pathlib import Path


def name_and_bytes(manifest):
    kind = "data" if manifest.get("base") else "lang"
    name = f"pdfintake_{kind}_{manifest['id'].replace('-', '_')}-{manifest['version']}-py3-none-any.whl"
    return name, f"fake wheel {manifest['id']} {manifest['version']} {manifest.get('salt', '')}".encode()


def wheel_problems(path, manifest):
    name, data = name_and_bytes(manifest)
    return [] if Path(path).name == name and Path(path).read_bytes() == data else ["not the pinned wheel"]


def download_sources(manifest, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)       # nothing upstream: a fake pack's files are its canary
    return []


def main(argv):
    assert argv[0] == "build" and argv[2] == "--out", argv
    manifest = json.loads(Path(argv[1], "manifest.json").read_text(encoding="utf-8"))
    canary = manifest.get("canary")
    files = Path(argv[argv.index("--files") + 1]) if "--files" in argv else None
    if canary and (files is None or not (files / canary["file"]).is_file()):
        # As the real build does: a canary has no upstream source, so it must already be drawn into --files.
        sys.stdout.write(json.dumps({"ok": False, "error": f"{canary['file']} isn't in the files folder"}) + "\n")
        return 1
    name, data = name_and_bytes(manifest)
    Path(argv[3], name).write_bytes(data)
    # The real datawheel.py reports on stdout as indented JSON; callers that capture stdout must cope.
    sys.stdout.write(json.dumps({"ok": True, "wheel": str(Path(argv[3], name)), "bytes": len(data)}, indent=2) + "\n")
    log = os.environ.get("FAKE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"datawheel build {argv[1]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def _tool(bin_dir: Path, name: str, code: str) -> None:
    script = bin_dir / f"{name}_fake.py"
    script.write_text(code, encoding="utf-8")
    wrapper = bin_dir / name
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)


def _shell(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def _bash() -> str:
    # The release Mac's /bin/bash is 3.2: the scripts must run there.
    return "/bin/bash" if sys.platform == "darwin" and Path("/bin/bash").exists() else (shutil.which("bash") or "bash")


def _fake_wheel(manifest: dict[str, Any]) -> tuple[str, bytes]:
    kind = "data" if manifest.get("base") else "lang"
    name = f"pdfintake_{kind}_{manifest['id'].replace('-', '_')}-{manifest['version']}-py3-none-any.whl"
    return name, f"fake wheel {manifest['id']} {manifest['version']} {manifest.get('salt', '')}".encode()


class FakeRelease:
    """A temporary repo with release.sh, fake tools on PATH, and a state file both fakes share."""

    VERSION = "0.2.0"

    def __init__(self, tmp: Path, *, en_host: str = "v0.2.0", th_host: str = "v0.2.0") -> None:
        self.tmp = tmp
        self.root = tmp / "my répo"
        self.bin = tmp / "bin"
        self.log = tmp / "calls.log"
        self.state_path = tmp / "state.json"
        self.bin.mkdir()
        for d in ("src/pdfintake/langpacks", "packs/en", "packs/th", "scripts", "tests", "tools"):
            (self.root / d).mkdir(parents=True)
        shutil.copy(BUILD_INDEX, self.root / "tools/build_index.py")
        (self.root / "src/pdfintake/__init__.py").write_text(f'__version__ = "{self.VERSION}"\n', encoding="utf-8")
        (self.root / "src/pdfintake/datawheel.py").write_text(FAKE_DATAWHEEL, encoding="utf-8")
        shutil.copy(SCRIPTS / "release.sh", self.root / "scripts/release.sh")
        shutil.copy(SCRIPTS / "dev-install.sh", self.root / "scripts/dev-install.sh")   # the real rebuild
        shutil.copy(CANARY_ENV, self.root / "tools/canary-env.txt")
        _shell(self.root / "scripts/vision-local.sh", 'echo "vision-local" >> "$FAKE_LOG"\nexit "${FAKE_VISION_RC:-0}"\n')
        for name in ("test_release", "test_docs", "test_plugin_folder", "test_platform_table"):
            (self.root / "tests" / f"{name}.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
        canary = {"file": "canary-x.png", "bytes": 10, "sha256": "ab" * 32, "dpi": 300, "must_find": ["x"]}
        self.manifests = {"en": {"id": "en", "version": "1.0.0", "base": True, "canary": canary},
                          "th": {"id": "th", "version": "1.0.0", "canary": canary}}
        self.hosts = {"en": en_host, "th": th_host}
        self.index_salt: dict[str, str] = {}
        self.pin_sha: str | None = None
        self.write_packs()
        _tool(self.bin, "git", FAKE_GIT)
        _tool(self.bin, "gh", FAKE_GH)
        _tool(self.bin, "uv", FAKE_UV)
        (self.bin / "draw_fake.py").write_text(FAKE_DRAW, encoding="utf-8")
        _shell(self.bin / "scanner", 'echo "scanner $*" >> "$FAKE_LOG"\nexit "${FAKE_SCAN_RC:-0}"\n')
        self.state: dict[str, Any] = {
            "root": str(self.root), "porcelain": "", "authors": ["RazorII"], "user": "RazorII",
            "tags": {}, "tag_types": {}, "head": "c0ffee", "remote_tags": [], "push_ok": True,
            "immutable": True, "releases": [], "runs": [], "ci_fail": False, "publish_immutable": True,
            "ci_assets": {}, "ci_fail_assets": {},
            "environment": {"name": "pack-publish", "protection_rules": [
                {"type": "required_reviewers", "prevent_self_review": False,
                 "reviewers": [{"type": "User", "reviewer": {"login": "RazorII"}}]}]},
        }
        self.state["ci_assets"] = self.correct_assets()

    def wheel(self, pid: str, *, salt: str = "") -> tuple[str, str, int]:
        name, data = _fake_wheel(dict(self.manifests[pid], salt=salt))
        return name, hashlib.sha256(data).hexdigest(), len(data)

    def url(self, pid: str) -> str:
        return f"https://github.com/RazorII/pdfintake/releases/download/{self.hosts[pid]}/{self.wheel(pid)[0]}"

    def write_packs(self) -> None:
        for pid, m in self.manifests.items():
            (self.root / "packs" / pid / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        entries = []
        for pid in ("en", "th"):
            _, sha, size = self.wheel(pid, salt=self.index_salt.get(pid, ""))
            dist = ({"kind": "installed", "distribution": "pdfintake-data-en", "url": None} if pid == "en" else
                    {"kind": "fetch", "distribution": f"pdfintake-lang-{pid}", "url": self.url(pid)})
            dist.update(sha256=sha, bytes=size)
            manifest_sha = hashlib.sha256((self.root / "packs" / pid / "manifest.json").read_bytes()).hexdigest()
            entries.append({"id": pid, "version": "1.0.0", "base": pid == "en", "manifest_sha256": manifest_sha,
                            "dist": dist, "installed_bytes": size + 100})
        (self.root / "src/pdfintake/langpacks/index.json").write_text(
            json.dumps({"index_version": 1, "packs": entries}), encoding="utf-8")
        pin_sha = self.pin_sha or entries[0]["dist"]["sha256"]
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "pdfintake"\ndependencies = [\n  "certifi",\n'
            f'  "pdfintake-data-en @ {self.url("en")}#sha256={pin_sha}",\n]\n', encoding="utf-8")

    def correct_assets(self, tag: str = "v0.2.0") -> dict[str, str]:
        return {self.wheel(p)[0]: "sha256:" + self.wheel(p)[1] for p in ("en", "th") if self.hosts[p] == tag}

    def add_published(self, tag: str, assets: dict[str, str], *, immutable: bool = True) -> None:
        self.state["releases"].append({"id": 100 + len(self.state["releases"]), "tag_name": tag, "draft": False,
                                       "immutable": immutable,
                                       "assets": [{"name": n, "digest": d} for n, d in assets.items()]})

    # The release scan the fake runs: a command on PATH plus an argument, so the split is exercised.
    SCAN = "scanner check"

    def run(self, version: str | None = None, *, scan: str | None = SCAN, **env: str) -> subprocess.CompletedProcess[str]:
        """Run release.sh with ``RELEASE_SCAN=scan``; ``scan=None`` leaves the variable unset."""
        self.state_path.write_text(json.dumps(self.state), encoding="utf-8")
        e = {k: v for k, v in os.environ.items() if k not in {"PDFINTAKE_RELEASE_STRICT", "RELEASE_SCAN", "VIRTUAL_ENV",
                                                                "PDFINTAKE_DIST", "UV", "GITHUB_PATH"}}
        e.update(PATH=f"{self.bin}{os.pathsep}{e['PATH']}", FAKE_STATE=str(self.state_path), FAKE_LOG=str(self.log),
                 FAKE_DRAW=str(self.bin / "draw_fake.py"), PYTHON=sys.executable, HOME=str(self.tmp),
                 PYTHONDONTWRITEBYTECODE="1", **env)
        if scan is not None:
            e["RELEASE_SCAN"] = scan
        proc = subprocess.run([_bash(), str(self.root / "scripts/release.sh"), version or self.VERSION],
                              cwd=self.tmp, env=e, capture_output=True, text=True, timeout=300)
        self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        return proc

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def index_of(self, prefix: str) -> int:
        return next(i for i, c in enumerate(self.calls()) if c.startswith(prefix))

    def tagged(self) -> bool:
        return any(c.startswith(("git tag", "git push")) for c in self.calls())

    def events(self, tag: str = "v0.2.0") -> list[tuple[str, bool, list[str]]]:
        """What the fake GitHub saw happen to one release: (what, draft at that moment, asset names)."""
        return [(e[0], e[2], e[3]) for e in self.state.get("events", []) if e[1] == tag]


@pytest.fixture
def rel(tmp_path: Path) -> FakeRelease:
    return FakeRelease(tmp_path)


@posix
def test_release_happy_path_publishes_an_immutable_release_with_every_pinned_digest(rel: FakeRelease):
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert "v0.2.0 is published as an immutable release" in proc.stdout
    order = [rel.index_of(p) for p in ("vision-local", "gh api repos/RazorII/pdfintake/immutable-releases",
                                       "gh api repos/RazorII/pdfintake/environments/pack-publish",
                                       "datawheel build packs/en", "scanner check", "git tag -a v0.2.0",
                                       "git push --atomic", "gh release create v0.2.0", "gh workflow run pack-publish.yml",
                                       "gh run watch", "gh release edit v0.2.0")]
    assert order == sorted(order), rel.calls()
    scanned = [c for c in rel.calls() if c.startswith("scanner")][0]
    assert scanned.startswith("scanner check ") and rel.wheel("en")[0] in scanned and rel.wheel("th")[0] in scanned
    (release,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert release["draft"] is False and release["immutable"] is True
    assert {a["name"]: a["digest"] for a in release["assets"]} == rel.correct_assets()


@posix
def test_release_rebuilds_the_wheels_it_hosts_through_dev_install(rel: FakeRelease):
    """datawheel.py alone can't build a pack with a canary (the canary has no upstream source), so Gate 4
    rebuilds through dev-install.sh --wheels-only, as pack-publish.yml does: one canary environment at
    tools/canary-env.txt's pins and pack-publish's Python version, each canary drawn before its build,
    and nothing installed."""
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    calls = rel.calls()
    for pid in ("en", "th"):
        assert calls.index(f"draw canary-{pid}") < calls.index(f"datawheel build packs/{pid}") < rel.index_of("scanner")
    made = [c for c in calls if c.startswith("uv venv") and c.endswith("render-env")]
    assert len(made) == 1, calls
    (job,) = _wf("pack-publish.yml")["jobs"].values()
    publish = re.findall(r"scripts/dev-install\.sh --wheels-only --python (\S+)", "\n".join(s.get("run", "") for s in job["steps"]))
    assert len(publish) == 1 and made[0].split()[2:4] == ["--python", publish[0]], (publish, made)
    installs = [c for c in calls if c.startswith(f"uv pip install --python {made[0].split()[-1]}/bin/python ")]
    assert [c.split()[5:] for c in installs] == [_canary_env_pins()]
    assert not any("--overrides" in c for c in calls), "a release rebuild installs nothing"


@posix
def test_release_refuses_to_tag_when_dev_install_cant_rebuild(rel: FakeRelease):
    (rel.root / "tools/canary-env.txt").unlink()
    proc = rel.run()
    assert proc.returncode != 0 and "tools/canary-env.txt can't be read" in proc.stderr, proc.stderr
    assert not rel.tagged() and not any(c.startswith(("scanner", "datawheel")) for c in rel.calls())


@posix
def test_release_refuses_to_tag_when_dev_install_builds_other_wheels_than_it_hosts(rel: FakeRelease):
    """packs/th says 1.0.1 while the index still pins 1.0.0: dev-install.sh builds a wheel the release
    doesn't host and none of the one it does, so Gate 4 stops by name before anything is scanned or tagged."""
    (rel.root / "packs/th/manifest.json").write_text(json.dumps(dict(rel.manifests["th"], version="1.0.1")),
                                                     encoding="utf-8")
    proc = rel.run()
    assert proc.returncode != 0, proc.stderr
    assert "scripts/dev-install.sh built" in proc.stderr and "pdfintake_lang_th-1.0.1-py3-none-any.whl" in proc.stderr
    assert "datawheel build packs/th" in rel.calls(), "the rebuild ran; its result is what's refused"
    assert not rel.tagged() and not any(c.startswith("scanner") for c in rel.calls())


@posix
def test_release_follows_githubs_order_draft_assets_digests_publish(rel: FakeRelease):
    """GitHub's documented order for immutable releases (decided 2026-11-05): create the release as a
    draft, attach every asset to the draft, check them, and only then publish."""
    assert rel.run().returncode == 0
    names = sorted(rel.correct_assets())
    assert rel.events() == [
        ("create", True, []),              # a draft, with nothing attached at creation
        ("attach", True, names),           # every asset goes onto the draft
        ("read", True, names),             # the digest check reads the draft as it now stands
        ("publish", False, names),         # published only after that check
        ("read", False, names),            # and read back: published, immutable, same digests
    ], rel.events()
    create = [c for c in rel.calls() if c.startswith("gh release create")]
    assert len(create) == 1 and "--draft" in create[0].split()


@posix
@pytest.mark.parametrize("answer", ["404", "disabled"])
def test_release_refuses_to_tag_while_immutable_releases_are_off(rel: FakeRelease, answer: str):
    rel.state["immutable"] = False if answer == "404" else "disabled"
    proc = rel.run()
    assert proc.returncode != 0 and "immutable releases are off" in proc.stderr
    assert not rel.tagged()
    gh = [c for c in rel.calls() if c.startswith("gh ")]
    assert gh and gh[-1] == "gh api repos/RazorII/pdfintake/immutable-releases", "the gate is the last gh call"
    assert "events" not in rel.state


@posix
@pytest.mark.parametrize("kind, said", [("network", "error connecting to api.github.com"),
                                        ("auth", "Bad credentials (HTTP 401)")])
@pytest.mark.parametrize("endpoint, setting", [("immutable-releases", "the immutable-releases setting"),
                                               ("pack-publish", "the pack-publish environment")])
def test_a_failed_settings_read_says_so_instead_of_calling_the_setting_off(
        rel: FakeRelease, kind: str, said: str, endpoint: str, setting: str):
    """A gh failure other than a 404 is reported as itself, never as the setting being off
    (decided 2026-11-05): the user mustn't be sent to switch on what's already on."""
    rel.state["gh_fails"] = {endpoint: kind}
    proc = rel.run()
    assert proc.returncode != 0 and not rel.tagged(), proc.stderr
    assert f"couldn't read {setting} for RazorII/pdfintake: " in proc.stderr and said in proc.stderr, proc.stderr
    assert "immutable releases are off" not in proc.stderr
    assert "no pack-publish environment" not in proc.stderr
    assert "events" not in rel.state


@posix
def test_release_never_publishes_a_draft_missing_an_asset(rel: FakeRelease):
    name = rel.wheel("th")[0]
    del rel.state["ci_assets"][name]
    proc = rel.run()
    assert proc.returncode != 0 and f"{name} isn't attached" in proc.stderr, proc.stderr
    (release,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert release["draft"] is True
    assert [e[0] for e in rel.events()] == ["create", "attach", "read"]


@posix
def test_release_never_publishes_a_draft_with_an_unpinned_asset(rel: FakeRelease):
    """The draft check is exact on the first run too: a published release is immutable, so an extra
    file attached to the draft would stay on it for good (decided 2026-11-05)."""
    rel.state["ci_assets"]["extra-1.0.0-py3-none-any.whl"] = "sha256:" + "a" * 64
    proc = rel.run()
    assert proc.returncode != 0, proc.stdout
    assert "extra-1.0.0-py3-none-any.whl is attached but not pinned" in proc.stderr, proc.stderr
    (release,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert release["draft"] is True
    assert [e[0] for e in rel.events()] == ["create", "attach", "read"]
    assert not any(c.startswith("gh release edit") for c in rel.calls())


@posix
def test_release_refuses_to_tag_when_the_vision_gate_fails(rel: FakeRelease):
    proc = rel.run(FAKE_VISION_RC="1")
    assert proc.returncode != 0 and not rel.tagged()
    assert not any(c.startswith("gh ") for c in rel.calls())


@posix
def test_release_refuses_to_tag_when_a_repo_test_skips(rel: FakeRelease):
    (rel.root / "tests/test_docs.py").write_text("import pytest\n\ndef test_ok():\n    pytest.skip('not yet')\n",
                                                 encoding="utf-8")
    proc = rel.run()
    assert proc.returncode != 0 and "1 skipped" in proc.stderr
    assert not rel.tagged() and "vision-local" not in rel.calls()


@posix
def test_release_runs_the_repo_tests_in_strict_mode(rel: FakeRelease):
    (rel.root / "tests/test_release.py").write_text(
        "import os\n\ndef test_ok():\n    assert os.environ['PDFINTAKE_RELEASE_STRICT'] == '1'\n", encoding="utf-8")
    assert rel.run().returncode == 0


@posix
def test_release_refuses_to_tag_when_a_rebuilt_wheel_differs_from_its_pin(rel: FakeRelease):
    rel.index_salt["th"] = "tampered"
    rel.write_packs()
    proc = rel.run()
    assert proc.returncode != 0 and "rebuilt as sha256" in proc.stderr
    assert not rel.tagged() and not any(c.startswith("scanner") for c in rel.calls())


@posix
def test_release_refuses_when_the_pyproject_pin_and_the_index_disagree(rel: FakeRelease):
    rel.pin_sha = "1" * 64
    rel.write_packs()
    proc = rel.run()
    assert proc.returncode != 0 and "pyproject pin's sha256" in proc.stderr and not rel.tagged()


@posix
def test_release_refuses_a_published_pack_without_a_canary(rel: FakeRelease):
    del rel.manifests["th"]["canary"]
    rel.write_packs()
    proc = rel.run()
    assert proc.returncode != 0 and "needs a canary" in proc.stderr and not rel.tagged()


@posix
def test_release_refuses_to_tag_on_a_scanner_finding(rel: FakeRelease):
    proc = rel.run(FAKE_SCAN_RC="1")
    assert proc.returncode != 0 and "the release scan stopped the release" in proc.stderr and not rel.tagged()
    assert any(c.startswith("scanner check ") for c in rel.calls())


@posix
@pytest.mark.parametrize(("scan", "says"), [(None, "RELEASE_SCAN is unset"), ("", "RELEASE_SCAN is unset"),
                                             ("  \t ", "holds no command"),
                                             ("scanner\ncheck", "more than one line"),
                                             ("no-such-scanner check", "command isn't found")])
def test_release_refuses_to_run_without_a_usable_release_scan(rel: FakeRelease, scan: str | None, says: str):
    """The scan is required: without one, release.sh stops before any gate, so nothing runs at all."""
    proc = rel.run(scan=scan)
    assert proc.returncode != 0 and says in proc.stderr and "refusing to release" in proc.stderr
    assert rel.calls() == [], rel.calls()


@posix
def test_wheels_an_earlier_release_hosts_must_be_there_immutable_with_the_pinned_digest(tmp_path: Path):
    rel = FakeRelease(tmp_path, th_host="v0.1.0")
    th_name, th_sha, _ = rel.wheel("th")
    rel.add_published("v0.1.0", {th_name: f"sha256:{th_sha}", "pdfintake_lang_xx-1.0.0-py3-none-any.whl": "sha256:" + "f" * 64})
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert "datawheel build packs/th" not in rel.calls(), "a wheel an earlier release hosts is never rebuilt"
    assert "draw canary-th" not in rel.calls() and "draw canary-en" in rel.calls()
    scanned = [c for c in rel.calls() if c.startswith("scanner")][0]
    assert th_name not in scanned and rel.wheel("en")[0] in scanned
    (draft,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert [a["name"] for a in draft["assets"]] == [rel.wheel("en")[0]]


@posix
@pytest.mark.parametrize("problem", ["digest", "mutable", "missing"])
def test_release_refuses_to_tag_when_an_earlier_release_doesnt_hold_its_wheel(tmp_path: Path, problem: str):
    rel = FakeRelease(tmp_path, th_host="v0.1.0")
    th_name, th_sha, _ = rel.wheel("th")
    if problem != "missing":
        digest = "sha256:" + ("e" * 64 if problem == "digest" else th_sha)
        rel.add_published("v0.1.0", {th_name: digest}, immutable=(problem != "mutable"))
    proc = rel.run()
    assert proc.returncode != 0 and not rel.tagged(), proc.stderr
    assert {"digest": "the index pins", "mutable": "isn't immutable", "missing": "isn't on GitHub"}[problem] in proc.stderr


@posix
def test_release_never_publishes_a_draft_whose_assets_dont_match(rel: FakeRelease):
    name = rel.wheel("th")[0]
    rel.state["ci_assets"][name] = "sha256:" + "d" * 64
    proc = rel.run()
    assert proc.returncode != 0 and "the index pins" in proc.stderr
    (release,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert release["draft"] is True
    assert not any(c.startswith("gh release edit") for c in rel.calls())


@posix
def test_release_stops_when_the_upload_job_fails(rel: FakeRelease):
    rel.state["ci_fail"] = True
    proc = rel.run()
    assert proc.returncode != 0
    assert not any(c.startswith("gh release edit") for c in rel.calls())


@posix
def test_release_says_so_when_github_didnt_make_it_immutable(rel: FakeRelease):
    rel.state["publish_immutable"] = False
    proc = rel.run()
    assert proc.returncode != 0 and "isn't immutable" in proc.stderr


@posix
def test_a_release_hosting_no_wheels_publishes_without_the_upload_job(tmp_path: Path):
    rel = FakeRelease(tmp_path, en_host="v0.1.0", th_host="v0.1.0")
    rel.add_published("v0.1.0", {rel.wheel(p)[0]: "sha256:" + rel.wheel(p)[1] for p in ("en", "th")})
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith(("gh workflow", "scanner", "datawheel")) for c in rel.calls())
    (release,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert release["draft"] is False and release["assets"] == []


@posix
def test_a_rerun_after_a_full_release_changes_nothing(rel: FakeRelease):
    assert rel.run().returncode == 0
    rel.log.unlink()
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith(("git commit", "git tag", "gh release create", "gh workflow", "gh release edit"))
                   for c in rel.calls())


@posix
def test_a_rerun_after_a_failed_upload_resumes_from_the_draft(rel: FakeRelease):
    rel.state["ci_fail"] = True
    assert rel.run().returncode != 0
    rel.state["ci_fail"] = False
    rel.log.unlink()
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith(("git tag", "gh release create")) for c in rel.calls())
    assert any(c.startswith("gh workflow run") for c in rel.calls())


@posix
def test_a_rerun_after_a_partial_upload_uploads_the_rest(rel: FakeRelease):
    en, th = rel.wheel("en"), rel.wheel("th")
    rel.state["ci_fail"] = True
    rel.state["ci_fail_assets"] = {en[0]: f"sha256:{en[1]}"}
    assert rel.run().returncode != 0
    (draft,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert [a["name"] for a in draft["assets"]] == [en[0]] and draft["draft"] is True
    rel.state["ci_fail"] = False
    rel.log.unlink()
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert any(c.startswith("gh workflow run") for c in rel.calls())
    (release,) = [r for r in rel.state["releases"] if r["tag_name"] == "v0.2.0"]
    assert release["draft"] is False and {a["name"]: a["digest"] for a in release["assets"]} == rel.correct_assets()
    assert th[0] in {a["name"] for a in release["assets"]}


@posix
@pytest.mark.parametrize("bad", ["digest", "unpinned"])
def test_a_rerun_stops_on_a_draft_holding_a_wrong_asset(rel: FakeRelease, bad: str):
    en = rel.wheel("en")
    rel.state["ci_fail"] = True
    rel.state["ci_fail_assets"] = ({en[0]: "sha256:" + "c" * 64} if bad == "digest" else
                                   {en[0]: f"sha256:{en[1]}", "stray.whl": "sha256:" + "c" * 64})
    assert rel.run().returncode != 0
    rel.state["ci_fail"] = False
    rel.log.unlink()
    proc = rel.run()
    assert proc.returncode != 0 and "remove those assets on GitHub" in proc.stderr, proc.stderr
    assert not any(c.startswith(("gh workflow run", "gh release edit")) for c in rel.calls())


@posix
@pytest.mark.parametrize("case", ["missing", "no-rules", "someone-else", "self-review"])
def test_release_refuses_to_tag_unless_pack_publish_needs_razoriis_approval(rel: FakeRelease, case: str):
    if case == "missing":
        rel.state["environment"] = None
    elif case == "no-rules":
        rel.state["environment"]["protection_rules"] = [{"type": "branch_policy"}]
    elif case == "someone-else":
        rel.state["environment"]["protection_rules"][0]["reviewers"] = [
            {"type": "User", "reviewer": {"login": "someone-else"}}]
    else:
        rel.state["environment"]["protection_rules"][0]["prevent_self_review"] = True
    proc = rel.run()
    assert proc.returncode != 0 and not rel.tagged(), proc.stderr
    assert {"missing": "no pack-publish environment", "no-rules": "doesn't require RazorII's approval",
            "someone-else": "doesn't require RazorII's approval", "self-review": "prevents self-review"}[case] in proc.stderr


def _build_index() -> Any:
    spec = importlib.util.spec_from_file_location("pdfintake_build_index_release", _need(BUILD_INDEX))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@posix
def test_release_records_the_published_packs_in_the_tagged_commit(rel: FakeRelease):
    """The tag points at a commit whose tools/published-packs.json lists every pack version the
    release pins, so build_index.py knows offline, in any checkout of it, what's published."""
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    (commit,) = rel.state["commits"]
    assert commit["paths"] == ["tools/published-packs.json"]
    assert commit["message"] == "Record the packs pdfintake 0.2.0 publishes"
    assert rel.state["tags"]["v0.2.0"] == commit["head"] == rel.state["head"]
    order = [rel.index_of(p) for p in ("scanner check", "git add -- tools/published-packs.json",
                                       "git commit", "git tag -a v0.2.0", "git push --atomic")]
    assert order == sorted(order), rel.calls()
    record = _build_index().load_record(rel.root / "tools" / "published-packs.json")
    assert record["releases"] == ["v0.2.0"]
    index = json.loads((rel.root / "src/pdfintake/langpacks/index.json").read_text(encoding="utf-8"))
    for entry in index["packs"]:
        assert record["packs"][entry["id"]]["1.0.0"] == {
            "release": "v0.2.0", "manifest_sha256": entry["manifest_sha256"], "dist": entry["dist"],
            "installed_bytes": entry["installed_bytes"], "index_entry_sha256": _build_index().entry_sha256(entry)}


@posix
def test_a_rerun_commits_no_second_record(rel: FakeRelease):
    rel.state["push_ok"] = False
    assert rel.run().returncode != 0
    assert len(rel.state["commits"]) == 1 and rel.state["tags"]["v0.2.0"] == rel.state["head"]
    rel.state["push_ok"] = True
    rel.log.unlink()
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    assert len(rel.state["commits"]) == 1
    assert not any(c.startswith(("git add", "git commit", "git tag")) for c in rel.calls()), rel.calls()
    assert "git push --atomic origin HEAD refs/tags/v0.2.0" in rel.calls()


@posix
def test_a_later_release_keeps_what_an_earlier_one_recorded(tmp_path: Path):
    rel = FakeRelease(tmp_path, th_host="v0.1.0")
    th_name, th_sha, _ = rel.wheel("th")
    rel.add_published("v0.1.0", {th_name: f"sha256:{th_sha}"})
    index = json.loads((rel.root / "src/pdfintake/langpacks/index.json").read_text(encoding="utf-8"))
    th = next(e for e in index["packs"] if e["id"] == "th")
    earlier = {"record_version": 1, "releases": ["v0.1.0"],
               "packs": {"th": {"1.0.0": {"release": "v0.1.0", "manifest_sha256": th["manifest_sha256"],
                                          "dist": th["dist"], "installed_bytes": th["installed_bytes"],
                                          "index_entry_sha256": _build_index().entry_sha256(th)}}}}
    (rel.root / "tools" / "published-packs.json").write_text(json.dumps(earlier), encoding="utf-8")
    proc = rel.run()
    assert proc.returncode == 0, proc.stderr
    record = _build_index().load_record(rel.root / "tools" / "published-packs.json")
    assert record["releases"] == ["v0.1.0", "v0.2.0"]
    assert record["packs"]["th"]["1.0.0"]["release"] == "v0.1.0"
    assert record["packs"]["en"]["1.0.0"]["release"] == "v0.2.0"


@posix
def test_release_refuses_to_tag_when_the_record_cant_be_written(rel: FakeRelease):
    """An index whose manifest hash isn't the manifest's can't be recorded: nothing is committed or tagged."""
    rel.manifests["th"]["title"] = "changed after the index"
    (rel.root / "packs/th/manifest.json").write_text(json.dumps(rel.manifests["th"]), encoding="utf-8")
    proc = rel.run()
    assert proc.returncode != 0, proc.stdout
    assert "th: the index's manifest_sha256 isn't packs/th/manifest.json's" in proc.stdout
    assert not rel.tagged() and "commits" not in rel.state
    assert not (rel.root / "tools" / "published-packs.json").exists()


@posix
def test_a_tag_elsewhere_stops_the_release_before_the_record_commit(rel: FakeRelease):
    rel.state["tags"]["v0.2.0"] = "0ld"
    proc = rel.run()
    assert proc.returncode != 0 and "already exists" in proc.stderr
    assert "commits" not in rel.state and not any(c.startswith("git push") for c in rel.calls())


@posix
@pytest.mark.parametrize("record", ["none", "earlier only"])
def test_a_tag_at_head_without_its_record_stops_the_release(rel: FakeRelease, record: str):
    """This release's annotated tag already at HEAD, but the record at HEAD doesn't list the
    release: the tagged commit can't gain the record, because a tag never moves."""
    rel.state["tags"]["v0.2.0"] = rel.state["head"]
    if record == "earlier only":
        index = json.loads((rel.root / "src/pdfintake/langpacks/index.json").read_text(encoding="utf-8"))
        th = next(e for e in index["packs"] if e["id"] == "th")
        earlier = {"record_version": 1, "releases": ["v0.1.0"],
                   "packs": {"th": {"1.0.0": {"release": "v0.1.0", "manifest_sha256": th["manifest_sha256"],
                                              "dist": th["dist"], "installed_bytes": th["installed_bytes"],
                                              "index_entry_sha256": _build_index().entry_sha256(th)}}}}
        (rel.root / "tools" / "published-packs.json").write_text(json.dumps(earlier), encoding="utf-8")
    before = (rel.root / "tools" / "published-packs.json").read_bytes() if record != "none" else None
    proc = rel.run()
    assert proc.returncode != 0, proc.stdout
    assert "v0.2.0 is already tagged at HEAD, but tools/published-packs.json doesn't record it" in proc.stderr
    assert "commits" not in rel.state
    assert not any(c.startswith(("git add", "git commit", "git tag", "git push")) for c in rel.calls()), rel.calls()
    record_path = rel.root / "tools" / "published-packs.json"
    assert (record_path.read_bytes() if record_path.exists() else None) == before


@posix
def test_a_lightweight_tag_is_never_replaced(rel: FakeRelease):
    rel.state["tags"]["v0.2.0"] = "c0ffee"
    rel.state["tag_types"]["v0.2.0"] = "commit"
    proc = rel.run()
    assert proc.returncode != 0 and "already exists" in proc.stderr
    assert not any(c.startswith(("git push", "git commit")) for c in rel.calls())


@posix
@pytest.mark.parametrize("case", ["version", "dirty", "publisher", "not-a-version"])
def test_docpdfs_gates_still_hold(rel: FakeRelease, case: str):
    if case == "dirty":
        rel.state["porcelain"] = " M README.md\n"
    elif case == "publisher":
        rel.state["authors"] = ["RazorII", "Someone Else"]
    proc = rel.run({"version": "0.3.0", "not-a-version": "v0.2"}.get(case))
    assert proc.returncode != 0 and not rel.tagged()
    assert {"version": "is not at 0.3.0", "dirty": "commit everything first", "publisher": "must be by RazorII",
            "not-a-version": "not a version"}[case] in proc.stderr


# --- vision-local.sh

def _vision_repo(tmp: Path, test_body: str) -> tuple[Path, dict[str, str]]:
    root = tmp / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    shutil.copy(SCRIPTS / "vision-local.sh", root / "scripts/vision-local.sh")
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\nmarkers = ["vision: needs Vision"]\n', encoding="utf-8")
    (root / "tests/test_v.py").write_text("import os, pytest\n\n" + textwrap.dedent(test_body), encoding="utf-8")
    fake = tmp / "bin"
    fake.mkdir()
    _shell(fake / "uname", 'case "$1" in -s) echo "${FAKE_UNAME_S:-Darwin}" ;; -m) echo "${FAKE_UNAME_M:-arm64}" ;; esac\n')
    env = dict(os.environ, PATH=f"{fake}{os.pathsep}{os.environ['PATH']}", PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PDFINTAKE_REQUIRE_VISION", None)
    return root, env


def _vision(root: Path, env: dict[str, str], **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([_bash(), str(root / "scripts/vision-local.sh")], cwd=root.parent, env=dict(env, **extra),
                          capture_output=True, text=True, timeout=300)


PASSING = '''
@pytest.mark.vision
def test_reads():
    assert os.environ["PDFINTAKE_REQUIRE_VISION"] == "1"

def test_not_vision():
    assert False, "deselected by -m vision"
'''


@posix
def test_vision_gate_passes_when_every_vision_test_ran(tmp_path: Path):
    root, env = _vision_repo(tmp_path, PASSING)
    proc = _vision(root, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 Vision tests ran, none skipped" in proc.stdout


@posix
@pytest.mark.parametrize("body,why", [
    ("""
@pytest.mark.vision
def test_a():
    pass

@pytest.mark.vision
def test_b():
    pytest.skip('no Vision')
""", "1 skipped"),
    ("@pytest.mark.vision\ndef test_a():\n    assert False\n", "1 failed"),
    ("def test_a():\n    pass\n", "no Vision test was collected"),
    ("@pytest.mark.vision\ndef test_a():\n    pass\n\nimport not_a_module_anywhere\n", "the Vision gate failed"),
])
def test_vision_gate_refuses_a_skip_a_failure_or_nothing(tmp_path: Path, body: str, why: str):
    root, env = _vision_repo(tmp_path, body)
    proc = _vision(root, env)
    assert proc.returncode != 0 and why in proc.stderr, proc.stdout + proc.stderr


@posix
@pytest.mark.parametrize("system,machine", [("Linux", "x86_64"), ("Darwin", "x86_64")])
def test_vision_gate_runs_only_on_an_apple_silicon_mac(tmp_path: Path, system: str, machine: str):
    root, env = _vision_repo(tmp_path, PASSING)
    proc = _vision(root, env, FAKE_UNAME_S=system, FAKE_UNAME_M=machine)
    assert proc.returncode != 0 and "Apple-silicon Mac" in proc.stderr


# --- dev-install.sh

FAKE_UV = r'''
import os, sys
from pathlib import Path
a = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write("uv " + " ".join(a) + "\n")
if a[0] == "venv":
    target = Path(a[-1])
    (target / ("Scripts" if os.name == "nt" else "bin")).mkdir(parents=True)
    if target.name == "render-env":            # the canary environment: its python draws a stand-in canary
        py = target / "bin" / "python"
        py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{os.environ["FAKE_DRAW"]}" "$@"\n', encoding="utf-8")
        py.chmod(0o755)
elif a[:3] == ["tool", "dir", "--bin"]:
    print(os.environ["FAKE_TOOL_BIN"])
elif a[0] in ("pip", "tool") and "--overrides" in a:
    overrides = Path(a[a.index("--overrides") + 1])
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
        log.write("overrides: " + overrides.read_text(encoding="utf-8"))
elif a[:2] != ["pip", "install"]:            # the canary environment's install, logged above
    sys.exit(99)
'''

# The canary environment's python: dev-install.sh runs it as `python -I render_canary.py JOB`; this
# writes a stand-in canary where the job says and logs the job's name (canary-<pack>).
FAKE_DRAW = r'''
import json, os, sys
from pathlib import Path
assert sys.argv[1] == "-I", sys.argv
job = Path(sys.argv[3])
Path(json.loads(job.read_text(encoding="utf-8"))["dest"]).write_bytes(b"stand-in canary")
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write(f"draw {job.stem}\n")
'''


class FakeInstall:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.root = tmp / "my répo"
        self.bin = tmp / "bin"
        self.log = tmp / "calls.log"
        self.bin.mkdir()
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "src/pdfintake").mkdir(parents=True)
        shutil.copy(SCRIPTS / "dev-install.sh", self.root / "scripts/dev-install.sh")
        (self.root / "src/pdfintake/datawheel.py").write_text(FAKE_DATAWHEEL, encoding="utf-8")
        self.manifests = {"en": {"id": "en", "version": "1.0.0", "base": True}, "th": {"id": "th", "version": "1.0.0"},
                          "hu": {"id": "hu", "version": "1.2.0"}}
        for pid, m in self.manifests.items():
            (self.root / "packs" / pid).mkdir(parents=True)
            (self.root / "packs" / pid / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        _tool(self.bin, "uv", FAKE_UV)
        (self.bin / "draw_fake.py").write_text(FAKE_DRAW, encoding="utf-8")
        self.github_path = tmp / "github_path"

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        e = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PDFINTAKE_DIST", "UV", "GITHUB_PATH"}}
        e.update(PATH=f"{self.bin}{os.pathsep}{e['PATH']}", FAKE_LOG=str(self.log), FAKE_TOOL_BIN=str(self.tmp / "toolbin"),
                 FAKE_DRAW=str(self.bin / "draw_fake.py"), PYTHONDONTWRITEBYTECODE="1", **env)
        return subprocess.run([_bash(), str(self.root / "scripts/dev-install.sh"), *args], cwd=self.tmp, env=e,
                              capture_output=True, text=True, timeout=120)

    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def wheel(self, pid: str) -> Path:
        return self.root / "dist" / _fake_wheel(self.manifests[pid])[0]

    def with_canary(self, pid: str = "en") -> Path:
        """Give ``pid`` a canary (made up) and the repo the real tools/canary-env.txt; returns that file."""
        self.manifests[pid]["canary"] = {"file": f"canary-{pid}.png", "bytes": 10, "sha256": "ab" * 32, "dpi": 300,
                                         "must_find": ["x"]}
        (self.root / "packs" / pid / "manifest.json").write_text(json.dumps(self.manifests[pid]), encoding="utf-8")
        (self.root / "tools").mkdir(exist_ok=True)
        return Path(shutil.copy(CANARY_ENV, self.root / "tools/canary-env.txt"))


@pytest.fixture
def inst(tmp_path: Path) -> FakeInstall:
    return FakeInstall(tmp_path)


@posix
def test_dev_install_builds_english_and_installs_editable_with_the_override(inst: FakeInstall):
    proc = inst.run(GITHUB_PATH=str(inst.github_path))
    assert proc.returncode == 0, proc.stderr
    calls = inst.calls()
    assert calls[0] == "datawheel build packs/en"
    assert calls[1] == f"uv venv {inst.root / '.venv'}"
    assert calls[2] == f"uv pip install --python {inst.root / '.venv/bin/python'} --overrides {inst.root / 'dist/overrides.txt'} -e .[kit,test]"
    assert calls[3] == f"overrides: pdfintake-data-en @ {inst.wheel('en').as_uri()}"
    assert inst.wheel("en").read_bytes() == _fake_wheel(inst.manifests["en"])[1]
    assert inst.github_path.read_text(encoding="utf-8").strip() == str(inst.root / ".venv/bin")
    assert "%C3%A9" in calls[3], "the file URI is percent-encoded"


@posix
def test_dev_install_reuses_a_sound_wheel_and_rebuilds_a_bad_one(inst: FakeInstall):
    assert inst.run().returncode == 0
    inst.log.unlink()
    assert inst.run().returncode == 0
    assert not any(c.startswith("datawheel") for c in inst.calls())
    inst.wheel("en").write_bytes(b"tampered")
    inst.log.unlink()
    assert inst.run().returncode == 0
    assert inst.calls()[0] == "datawheel build packs/en"
    assert inst.wheel("en").read_bytes() == _fake_wheel(inst.manifests["en"])[1]


@posix
def test_dev_install_tool_mode_is_a_non_editable_uv_tool_install(inst: FakeInstall):
    proc = inst.run("--tool", "--python", "3.12", GITHUB_PATH=str(inst.github_path))
    assert proc.returncode == 0, proc.stderr
    assert f"uv tool install --force --python 3.12 --overrides {inst.root / 'dist/overrides.txt'} ." in inst.calls()
    assert not any(c.startswith(("uv venv", "uv pip")) for c in inst.calls())
    assert inst.github_path.read_text(encoding="utf-8").strip() == str(inst.tmp / "toolbin")


@posix
def test_dev_install_builds_every_pack_on_request(inst: FakeInstall):
    assert inst.run("--packs", "all").returncode == 0
    assert [c for c in inst.calls() if c.startswith("datawheel")] == [
        "datawheel build packs/en", "datawheel build packs/hu", "datawheel build packs/th"]
    inst.log.unlink()
    assert inst.run("--packs", "th").returncode == 0
    assert not any(c.startswith("datawheel") for c in inst.calls())
    proc = inst.run("--packs", "xx")
    assert proc.returncode == 2 and "no packs/xx/manifest.json" in proc.stderr


@posix
def test_dev_install_rebuilding_one_pack_keeps_a_pack_whose_id_ends_the_same(inst: FakeInstall):
    for pid in ("br", "pt-br"):
        inst.manifests[pid] = {"id": pid, "version": "1.0.0"}
        (inst.root / "packs" / pid).mkdir()
        (inst.root / "packs" / pid / "manifest.json").write_text(json.dumps(inst.manifests[pid]), encoding="utf-8")
    assert inst.run("--packs", "all").returncode == 0
    inst.wheel("br").write_bytes(b"tampered")
    inst.log.unlink()
    proc = inst.run("--packs", "br")
    assert proc.returncode == 0, proc.stderr
    assert [c for c in inst.calls() if c.startswith("datawheel")] == ["datawheel build packs/br"]
    assert inst.wheel("pt-br").read_bytes() == _fake_wheel(inst.manifests["pt-br"])[1]
    assert inst.wheel("br").read_bytes() == _fake_wheel(inst.manifests["br"])[1]


@posix
def test_dev_install_uses_the_active_virtualenv(inst: FakeInstall, tmp_path: Path):
    venv = tmp_path / "active env"
    proc = inst.run("--extras", "", VIRTUAL_ENV=str(venv))
    assert proc.returncode == 0, proc.stderr
    assert not any(c.startswith("uv venv") for c in inst.calls())
    assert f"uv pip install --python {venv / 'bin/python'} --overrides {inst.root / 'dist/overrides.txt'} -e ." in inst.calls()


@posix
def test_dev_install_needs_uv(inst: FakeInstall):
    proc = inst.run(UV="no-such-uv-command")
    assert proc.returncode == 3 and "needs uv" in proc.stderr


@posix
def test_dev_install_fails_loudly_when_no_english_wheel_comes_out(inst: FakeInstall):
    # A build that writes nothing must stop the install, never point the pin at a folder.
    (inst.root / "src/pdfintake/datawheel.py").write_text(
        FAKE_DATAWHEEL.replace("Path(argv[3], name).write_bytes(data)", "pass"), encoding="utf-8")
    proc = inst.run()
    assert proc.returncode == 1 and "wrote 0 wheels" in proc.stderr
    assert not any(c.startswith("uv pip") for c in inst.calls())
    (inst.root / "packs/en/manifest.json").unlink()
    proc = inst.run()
    assert proc.returncode == 3 and "packs/en/manifest.json is missing" in proc.stderr


# --- the canary environment: tools/canary-env.txt

def _canary_env_pins() -> list[str]:
    lines = [ln.split("#", 1)[0].strip() for ln in CANARY_ENV.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def _render_libs() -> list[str]:
    """dev-install.sh's RENDER_LIBS, read from the script's helper."""
    m = re.search(r"^RENDER_LIBS = \(([^)]*)\)", (SCRIPTS / "dev-install.sh").read_text(encoding="utf-8"), re.M)
    assert m, "dev-install.sh has no RENDER_LIBS"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_canary_env_pins_every_renderer_library_exactly_and_within_pyproject():
    from packaging.requirements import Requirement

    text = CANARY_ENV.read_text(encoding="utf-8")
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert header and text.startswith("#") and "kit wheel" in " ".join(header)
    pins = _canary_env_pins()
    names = [re.fullmatch(r"([a-z0-9-]+)==(\d[\w.+!-]*)", p) for p in pins]
    assert all(names), pins
    assert [m.group(1) for m in names] == _render_libs()
    # Each exact pin is a version pyproject.toml allows, so the drawn canaries match what users install.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    reqs = [Requirement(r) for r in project["dependencies"] + [r for g in project["optional-dependencies"].values() for r in g]]
    for m in names:
        allowed = [r.specifier for r in reqs if re.sub(r"[-_.]+", "-", r.name).lower() == m.group(1)]
        assert allowed and all(spec.contains(m.group(2)) for spec in allowed), (m.group(0), allowed)


@posix
def test_dev_install_draws_canaries_with_exactly_the_canary_env_pins(inst: FakeInstall):
    inst.with_canary("en")
    assert not (inst.root / "pyproject.toml").exists()          # the pins come from tools/canary-env.txt alone
    proc = inst.run("--wheels-only", "--packs", "en")
    assert proc.returncode == 0, proc.stderr
    calls = inst.calls()
    made = [c for c in calls if c.startswith("uv venv") and c.endswith("render-env")]
    assert len(made) == 1, calls
    installs = [c for c in calls if c.startswith(f"uv pip install --python {made[0].split()[-1]}/bin/python ")]
    assert [c.split()[5:] for c in installs] == [_canary_env_pins()]
    assert calls.index("draw canary-en") < calls.index("datawheel build packs/en")
    assert inst.wheel("en").read_bytes() == _fake_wheel(inst.manifests["en"])[1]


CANARY_ENV_DAMAGE = {
    "missing": (lambda t: None, "tools/canary-env.txt can't be read"),
    "a range": (lambda t: re.sub(r"(?m)^pillow==\S+", "pillow>=13,<14", t), "isn't an exact name==version pin"),
    "a library left out": (lambda t: re.sub(r"(?m)^fonttools==\S+\n", "", t), "has no pin for fonttools"),
    "a library twice": (lambda t: t + "fpdf2==2.8.0\n", "pins fpdf2 twice"),
    "another library": (lambda t: t + "numpy==2.0.0\n", "numpy isn't a library the canary renderer draws with"),
}


@posix
@pytest.mark.parametrize("damage", sorted(CANARY_ENV_DAMAGE))
def test_dev_install_refuses_a_canary_env_that_isnt_exact_pins(inst: FakeInstall, damage: str):
    env_file = inst.with_canary("en")
    change, says = CANARY_ENV_DAMAGE[damage]
    text = change(env_file.read_text(encoding="utf-8"))
    if text is None:
        env_file.unlink()
    else:
        assert text != env_file.read_text(encoding="utf-8"), damage
        env_file.write_text(text, encoding="utf-8")
    proc = inst.run("--wheels-only", "--packs", "en")
    assert proc.returncode == 1 and says in proc.stderr, proc.stderr
    assert not any(c.startswith(("uv pip", "draw", "datawheel")) for c in inst.calls()), inst.calls()
