"""Workflow hygiene the release tests don't cover (tests/test_release.py checks each workflow's own
safety properties: triggers, tokens, secrets, checkout credentials).

* Every action is pinned to a 40-hex commit SHA, with its tag in a comment on the same line, so a
  moved tag can't change what runs. One action is pinned to one commit across the workflows, and a
  commit always carries the same tag.
* No run script holds a ``${{ ... }}`` expression: a value reaches the shell only through an
  environment variable, so nothing a workflow is handed (an input, a pack id, a ref) is ever pasted
  into a script.
* ``bench-platforms.yml`` and ``install-smoke.yml`` start by hand only (workflow_dispatch) with a
  read-only token, and no job asks for more. install-smoke runs the sandbox smoke test with its skips
  as failures, on every OS of the network job.
* The steps that need a particular setting have it: the sandbox smoke test runs with
  ``PDFINTAKE_SANDBOX_SMOKE=1``, pack-bench's summary prints ``kit verify``'s warnings, and
  pack-publish rebuilds the wheels the way ``scripts/dev-install.sh`` does (the canary drawn first).
* pack-publish, whose job holds a write token that every action gets by default, runs first-party
  actions only, with no cache, and installs uv by a run step at one version and wheel digest.
* Every workflow runs that same uv version: each ``astral-sh/setup-uv`` step names it exactly, so the
  wheels CI tests are built by the uv that pack-publish rebuilds them with.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
FILES = sorted(WORKFLOWS.glob("*.yml"))

USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
PINNED = re.compile(r"^(?P<action>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?)@(?P<sha>[0-9a-f]{40})$")
TAG_COMMENT = re.compile(r"^\s+#\s*(?P<tag>v\d+(?:\.\d+){0,2})\s*$")


def _uses_lines() -> list[tuple[str, int, str, str]]:
    """(workflow, line number, ref, rest of the line) for every ``uses:`` line, read as text."""
    out = []
    for path in FILES:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = USES_LINE.match(line)
            if m:
                out.append((path.name, n, m["ref"].strip("'\""), m["rest"]))
    return out


def test_there_are_workflows():
    assert {p.name for p in FILES} >= {"test.yml", "pack-bench.yml", "pack-publish.yml", "resolve.yml",
                                      "bench-platforms.yml", "install-smoke.yml"}


def test_every_action_is_pinned_to_a_commit_with_its_tag_in_a_comment():
    lines = _uses_lines()
    assert lines
    for name, n, ref, rest in lines:
        if ref.startswith("./"):                         # an action in this repo: the checkout pins it
            continue
        assert PINNED.match(ref), f"{name}:{n}: {ref} isn't pinned to a 40-hex commit SHA"
        assert TAG_COMMENT.match(rest), f"{name}:{n}: {ref} needs its tag in a comment (# vX.Y.Z)"


def test_one_commit_per_action_and_one_tag_per_commit():
    actions: dict[str, set[str]] = {}
    tags: dict[str, set[str]] = {}
    for _name, _n, ref, rest in _uses_lines():
        m, c = PINNED.match(ref), TAG_COMMENT.match(rest)
        if m and c:
            actions.setdefault(m["action"], set()).add(m["sha"])
            tags.setdefault(m["sha"], set()).add(c["tag"])
    assert {a: s for a, s in actions.items() if len(s) > 1} == {}, "an action is pinned to two commits"
    assert {s: t for s, t in tags.items() if len(t) > 1} == {}, "a commit carries two different tags"


# --- read as YAML (each of these skips without PyYAML; the text checks above never do)

def _wf(path: Path) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))           # YAML 1.1 reads the key `on` as true
    return data


def _steps(wf: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in wf["jobs"].values() for step in job.get("steps", [])]


def test_the_text_scan_sees_every_uses_the_yaml_holds():
    for path in FILES:
        wf = _wf(path)
        parsed = [str(s["uses"]) for s in _steps(wf) if "uses" in s]
        parsed += [str(j["uses"]) for j in wf["jobs"].values() if "uses" in j]   # reusable workflows
        scanned = [ref for name, _n, ref, _rest in _uses_lines() if name == path.name]
        assert sorted(parsed) == sorted(scanned), path.name


def test_no_run_script_holds_an_expression():
    for path in FILES:
        for job_id, job in _wf(path)["jobs"].items():
            for step in job.get("steps", []):
                assert "${{" not in str(step.get("run", "")), (
                    f"{path.name} job {job_id}, step {step.get('name') or step.get('uses')}: pass the value "
                    "through env: instead of an expression inside the script")


@pytest.mark.parametrize("name", ["bench-platforms.yml", "install-smoke.yml"])
def test_hand_started_workflows_have_a_read_only_token(name: str):
    wf = _wf(WORKFLOWS / name)
    assert list(wf["on"]) == ["workflow_dispatch"]
    assert wf["permissions"] == {"contents": "read"}
    for job_id, job in wf["jobs"].items():
        perms = job.get("permissions")
        if perms is None:
            continue
        assert isinstance(perms, dict), f"{job_id}: permissions: {perms!r} is wider than contents: read"
        wider = {k: v for k, v in perms.items() if v != "none" and not (k == "contents" and v == "read")}
        assert wider == {}, f"{job_id} asks for more than contents: read: {wider}"
    # No run script of it holds ${{ }}: test_no_run_script_holds_an_expression checks every workflow.


def test_the_sandbox_smoke_test_must_run():
    (step,) = [s for s in _wf(WORKFLOWS / "test.yml")["jobs"]["sandbox"]["steps"]
               if "test_sandbox_smoke.py" in str(s.get("run", ""))]
    assert step.get("env", {}).get("PDFINTAKE_SANDBOX_SMOKE") == "1"


def test_install_smoke_installs_the_tag_and_must_run_the_smoke_test_everywhere():
    wf = _wf(WORKFLOWS / "install-smoke.yml")
    (job,) = wf["jobs"].values()
    assert job["env"].get("PDFINTAKE_SANDBOX_SMOKE") == "1"
    network = _wf(WORKFLOWS / "test.yml")["jobs"]["network"]["strategy"]["matrix"]["os"]
    assert sorted(job["strategy"]["matrix"]["os"]) == sorted(network)
    runs = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert 'uv tool install "git+https://github.com/RazorII/pdfintake@$TAG"' in runs
    assert "tests/e2e/test_sandbox_smoke.py" in runs
    assert "dev-install.sh" not in runs, "install-smoke must install the release, not this checkout"


def test_pack_bench_summary_prints_the_warnings():
    (step,) = [s for s in _steps(_wf(WORKFLOWS / "pack-bench.yml")) if "kit verify" in str(s.get("run", ""))]
    run = step["run"]
    assert 'result.get("problems"' in run and 'result.get("warnings"' in run
    assert "GITHUB_STEP_SUMMARY" in run


UV_PIN = re.compile(r"^uv==\d+\.\d+\.\d+ --hash=sha256:[0-9a-f]{64}$")


def test_pack_publish_rebuilds_the_way_dev_install_does():
    """datawheel.py alone can't rebuild a pack with a canary (the canary has no upstream source):
    the rebuild goes through dev-install.sh --wheels-only, which draws it first, after uv is installed
    at one pinned version and wheel digest."""
    steps = _wf(WORKFLOWS / "pack-publish.yml")["jobs"]["publish"]["steps"]
    runs = [str(s.get("run", "")) for s in steps]
    rebuild = [i for i, r in enumerate(runs) if "scripts/dev-install.sh --wheels-only" in r]
    assert len(rebuild) == 1
    uv = [i for i, r in enumerate(runs) if "--require-hashes" in r and "uv==" in r]
    assert len(uv) == 1 and uv[0] < rebuild[0]
    pins = re.findall(r"'(uv==[^']*)'", runs[uv[0]])
    assert len(pins) == 1 and UV_PIN.match(pins[0]), pins
    assert not any(re.search(r"datawheel\.py\"?,?\s*\"?build", r) for r in runs), "a bare datawheel.py build"


def test_pack_publish_runs_no_third_party_action():
    """GitHub hands the job's token (contents: write here) to every action as its default
    github-token input, whatever the step's env says, so the publish job runs first-party actions
    only, and none of them caches."""
    for job_id, job in _wf(WORKFLOWS / "pack-publish.yml")["jobs"].items():
        for step in job.get("steps", []):
            if "uses" not in step:
                continue
            action = str(step["uses"]).split("@", 1)[0]
            assert action.startswith("actions/"), f"{job_id}: {action} would receive the job's write token"
            assert not action.startswith("actions/cache"), f"{job_id}: a release job doesn't cache"
            assert "cache" not in (step.get("with") or {}), f"{job_id}: {action} restores a cache"


UV_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def test_every_workflow_runs_one_uv_version():
    """A setup-uv step without a version takes whatever uv is newest that day; the release rebuild
    installs one exact uv by digest. Every way a workflow gets uv must name the same version."""
    found: dict[str, set[str]] = {}
    setup_steps = pinned_runs = 0
    for path in FILES:
        for job_id, job in _wf(path)["jobs"].items():
            for step in job.get("steps", []):
                where = f"{path.name} job {job_id}, step {step.get('name') or step.get('uses')}"
                if str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    setup_steps += 1
                    version = str((step.get("with") or {}).get("version", ""))
                    assert UV_VERSION.match(version), f"{where}: setup-uv needs an exact version: (got {version!r})"
                    found.setdefault(version, set()).add(where)
                for version in re.findall(r"\buv==([^\s'\"]+)", str(step.get("run", ""))):
                    pinned_runs += 1
                    assert UV_VERSION.match(version), f"{where}: uv=={version} isn't an exact version"
                    found.setdefault(version, set()).add(where)
    assert setup_steps and pinned_runs, "found no setup-uv step or no uv==X.Y.Z install to compare"
    assert len(found) == 1, f"workflows run more than one uv version: {found}"


def test_every_file_checks_out_byte_for_byte():
    """Windows runners check text out with CRLF unless told not to, and a pinned file then hashes
    differently: the first Windows CI runs failed `doctor` with pdfintake_data_en/manifest.json's
    SHA-256 not matching the index. `.gitattributes` turns the conversion off for every file."""
    rules = [ln.split("#", 1)[0].split() for ln in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()]
    assert ["*", "-text"] in rules
