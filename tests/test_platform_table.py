"""The README's support table and the CI matrix match ``platforms.PLATFORMS``.

A platform is one row in ``PLATFORMS``. Adding one means a CI runner and a README row as well;
these tests say which is missing.

* **README.** A Markdown table whose header names "Platform" has exactly one row per
  ``PLATFORMS`` key (the key written in the row, for example ``darwin-arm64``). A supported row
  says yes (``Yes``, ``✓`` or ``Supported``) and names its ``min_os``; an unsupported row says no.
* **test.yml.** The suite's runners are exactly the supported rows' ``ci_runners``; a row whose
  notes call its CI optional runs with ``optional: true`` (continue-on-error); the Mac rows require
  Vision; the Python versions span the floor and the newest allowed by ``requires-python``. The
  network job runs on Linux, macOS and Windows.
* **resolve.yml** resolves for every supported platform (uv's target name for each row) and for
  every Python from the floor up to the cap it watches.
* **pack-bench.yml** re-benches on Linux, Windows and macOS.

The README part skips until the README exists, and fails under PDFINTAKE_RELEASE_STRICT=1.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

from pdfintake.platforms import PLATFORMS

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
WORKFLOWS = ROOT / ".github" / "workflows"

# uv's --python-platform name for each supported row. A new platform needs its name here.
UV_TARGETS = {
    "darwin-arm64": "aarch64-apple-darwin",
    "win-amd64": "x86_64-pc-windows-msvc",
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-aarch64": "aarch64-unknown-linux-gnu",
}
YES = ("yes", "✓", "✅", "supported")
NO = ("no", "✗", "❌", "not supported", "unsupported")
SUPPORTED = [p for p in PLATFORMS if p.supported]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _python_range() -> tuple[tuple[int, int], tuple[int, int]]:
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    lo = re.search(r">=\s*3\.(\d+)", spec)
    hi = re.search(r"<\s*3\.(\d+)", spec)
    assert lo and hi, f"requires-python {spec!r} needs a floor and a cap"
    return (3, int(lo.group(1))), (3, int(hi.group(1)))


def _ver(s: str) -> tuple[int, int]:
    major, minor = str(s).split(".")[:2]
    return int(major), int(minor)


# --- README

def _plain(cell: str) -> str:
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell).replace("`", "").replace("*", "").strip()


def support_rows(text: str) -> list[list[str]]:
    rows, block = [], []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append([_plain(c) for c in line.strip().strip("|").split("|")])
            continue
        if len(block) >= 2 and any("platform" in c.lower() for c in block[0]):
            rows += [r for r in block[2:] if not all(re.fullmatch(r"[\s:-]*", c) for c in r)]
        block = []
    return rows


def _says(row: list[str], words: tuple[str, ...]) -> bool:
    for cell in row:
        c = cell.lower()
        if any(c == w or re.match(rf"^{re.escape(w)}(?![a-z])", c) for w in words):
            return True
    return False


def readme_problems(text: str) -> list[str]:
    rows = support_rows(text)
    if not rows:
        return ["no table with a 'Platform' header"]
    problems = []
    for p in PLATFORMS:
        mine = [r for r in rows if any(re.search(rf"(?<![\w-]){re.escape(p.key)}(?![\w-])", c) for c in r)]
        if len(mine) != 1:
            problems.append(f"{p.key}: {len(mine)} rows, expected 1")
            continue
        row = mine[0]
        if p.supported:
            if not _says(row, YES) or _says(row, NO):
                problems.append(f"{p.key}: supported, but the row doesn't say yes")
            if p.min_os.lower() not in " ".join(row).lower():
                problems.append(f"{p.key}: the row doesn't name {p.min_os}")
        elif not _says(row, NO) or _says(row, YES):
            problems.append(f"{p.key}: not supported, but the row doesn't say no")
    keys = {p.key for p in PLATFORMS}
    for r in rows:
        if not any(re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", c) for k in keys for c in r):
            problems.append(f"a row names no PLATFORMS key: {' | '.join(c for c in r if c)}")
    return problems


def test_readme_support_table_matches_platforms():
    if not README.is_file():
        msg = "README.md isn't written yet"
        if os.environ.get("PDFINTAKE_RELEASE_STRICT") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    assert readme_problems(README.read_text(encoding="utf-8")) == []


def test_the_readme_rules_catch_mismatches():
    good = ["| Platform | Supported | Minimum | Notes |", "|---|---|---|---|"]
    for p in PLATFORMS:
        good.append(f"| `{p.key}` | {'Yes' if p.supported else 'No'} | {p.min_os or '-'} | {p.notes} |")
    assert readme_problems("\n".join(good)) == []
    flipped = "\n".join(good).replace("| `win-arm64` | No |", "| `win-arm64` | Yes |")
    assert readme_problems(flipped) == ["win-arm64: not supported, but the row doesn't say no"]
    dropped = "\n".join(line for line in good if "linux-aarch64" not in line)
    assert readme_problems(dropped) == ["linux-aarch64: 0 rows, expected 1"]
    extra = "\n".join(good + ["| `freebsd-amd64` | No | - | |"])
    assert readme_problems(extra) == ["a row names no PLATFORMS key: freebsd-amd64 | No | -"]
    no_floor = "\n".join(good).replace("| macOS 15 |", "| macOS |")
    assert readme_problems(no_floor) == ["darwin-arm64: the row doesn't name macOS 15"]
    assert readme_problems("no tables here") == ["no table with a 'Platform' header"]


# --- the CI matrix

def _suite_rows() -> list[dict]:
    return _workflow("test.yml")["jobs"]["suite"]["strategy"]["matrix"]["include"]


def test_suite_runners_are_the_supported_platforms_runners():
    runners = {r["os"] for r in _suite_rows()}
    expected = {run for p in SUPPORTED for run in p.ci_runners}
    assert runners == expected
    for p in PLATFORMS:
        if not p.supported:
            assert not set(p.ci_runners) & runners, f"{p.key} isn't supported but has a CI row"


def test_every_supported_platform_has_a_required_row_unless_optional():
    rows = _suite_rows()
    for p in SUPPORTED:
        optional = "optional" in p.notes.lower()
        mine = [r for r in rows if r["os"] in p.ci_runners]
        assert mine, f"{p.key} has no CI row"
        assert all(bool(r["optional"]) is optional for r in mine), f"{p.key}: optional should be {optional}"
    assert _workflow("test.yml")["jobs"]["suite"]["continue-on-error"] == "${{ matrix.optional }}"


def test_macs_require_vision_and_nothing_else_does():
    darwin = {run for p in SUPPORTED if p.sys_platform == "darwin" for run in p.ci_runners}
    for r in _suite_rows():
        assert str(r["vision"]) == ("1" if r["os"] in darwin else "0"), r
    assert _workflow("test.yml")["jobs"]["suite"]["env"]["PDFINTAKE_REQUIRE_VISION"] == "${{ matrix.vision }}"


def test_suite_pythons_span_requires_python():
    (lo, hi) = _python_range()
    pythons = {_ver(r["python"]) for r in _suite_rows()}
    newest = (hi[0], hi[1] - 1)
    assert lo in pythons and newest in pythons, pythons
    assert all(lo <= v <= newest for v in pythons), pythons
    for p in SUPPORTED:
        if "optional" in p.notes.lower():
            continue
        mine = {_ver(r["python"]) for r in _suite_rows() if r["os"] in p.ci_runners}
        assert lo in mine or newest in mine, f"{p.key} tests neither the floor nor the newest Python"


def test_windows_rows_use_a_non_ascii_cache():
    for r in _suite_rows():
        if r["os"].startswith("windows"):
            assert not r["cache_name"].isascii(), r


def test_network_job_runs_on_every_os_family():
    oses = _workflow("test.yml")["jobs"]["network"]["strategy"]["matrix"]["os"]
    assert {o.split("-")[0] for o in oses} == {"ubuntu", "macos", "windows"}
    assert len(oses) == 3


def test_resolve_covers_every_supported_platform_and_python():
    matrix = _workflow("resolve.yml")["jobs"]["compile"]["strategy"]["matrix"]
    missing = [p.key for p in SUPPORTED if p.key not in UV_TARGETS]
    assert missing == [], f"add uv's --python-platform name for {missing} to UV_TARGETS"
    assert set(matrix["target"]) == {UV_TARGETS[p.key] for p in SUPPORTED}
    (lo, hi) = _python_range()
    assert sorted(_ver(v) for v in matrix["python"]) == [(3, m) for m in range(lo[1], hi[1] + 1)]


def test_pack_bench_covers_three_operating_systems():
    oses = _workflow("pack-bench.yml")["jobs"]["bench"]["strategy"]["matrix"]["os"]
    assert {o.split("-")[0] for o in oses} == {"ubuntu", "macos", "windows"}
