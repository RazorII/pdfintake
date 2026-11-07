#!/usr/bin/env bash
# Run the Apple Vision tests on this Mac, and pass only when every one of them ran and passed.
#
# GitHub's macOS VMs may not be able to run Vision, so scripts/release.sh runs this on the release
# Mac before it tags: each new macOS gets a Vision check before a release. With
# PDFINTAKE_REQUIRE_VISION=1 the Vision tests fail instead of skipping, and this script also
# refuses a skip that slips through, a failure, an error, or a run that collected no Vision test.
#
# usage: scripts/vision-local.sh [extra pytest arguments]
# Environment: PYTHON is the interpreter with pdfintake[test] installed (default python3).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "the Vision gate runs on an Apple-silicon Mac (this is $(uname -s) $(uname -m))" >&2
  exit 1
fi

py="${PYTHON:-python3}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

rc=0
PDFINTAKE_REQUIRE_VISION=1 PYTHONDONTWRITEBYTECODE=1 \
  "$py" -m pytest -m vision -p no:cacheprovider -q -rs --junitxml "$work/vision.xml" "$@" || rc=$?

"$py" - "$work/vision.xml" "$rc" <<'PY'
import sys
import xml.etree.ElementTree as ET

path, rc = sys.argv[1], int(sys.argv[2])
try:
    root = ET.parse(path).getroot()
except (OSError, ET.ParseError):
    print(f"the Vision gate failed: pytest wrote no report (exit {rc})", file=sys.stderr)
    sys.exit(1)
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
count = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "skipped", "failures", "errors")}
ran = count["tests"] - count["skipped"]
problems = []
if count["tests"] == 0 or rc == 5:
    problems.append("no Vision test was collected")
if count["skipped"]:
    problems.append(f"{count['skipped']} skipped")
if count["failures"] or count["errors"]:
    problems.append(f"{count['failures']} failed, {count['errors']} errors")
if rc not in (0, 5) and not problems:
    problems.append(f"pytest exited {rc}")
if problems:
    print("the Vision gate failed: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
print(f"the Vision gate passed: {ran} Vision tests ran, none skipped")
PY
