#!/usr/bin/env bash
# Run thorough mode's real-model tests on this Mac, and pass only when every one of them ran and passed.
#
# CI never downloads the 4.4 GB dots.mocr model, so its tests skip there. scripts/release.sh runs
# this on the release Mac before it tags: the dots engine's contract and the model port's reads,
# on the pinned model. With PDFINTAKE_REQUIRE_THOROUGH=1 those tests fail instead of skipping, and
# this script also refuses a skip that slips through, a failure, an error, or a run that collected
# no thorough test.
#
# The model must be installed and verified where PDFINTAKE_CACHE points (pdfintake thorough enable
# does that), or PDFINTAKE_THOROUGH_CACHE names a cache folder that holds it.
#
# usage: scripts/thorough-local.sh [extra pytest arguments]
# Environment: PYTHON is the interpreter with pdfintake[test,thorough] installed (default python3).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "the thorough gate runs on an Apple-silicon Mac (this is $(uname -s) $(uname -m))" >&2
  exit 1
fi

py="${PYTHON:-python3}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
if [ -n "${PDFINTAKE_THOROUGH_CACHE:-}" ]; then
  export PDFINTAKE_CACHE="$PDFINTAKE_THOROUGH_CACHE"
fi
if [ -z "${PDFINTAKE_CACHE:-}" ]; then
  echo "the thorough gate needs PDFINTAKE_CACHE (or PDFINTAKE_THOROUGH_CACHE) naming a cache that holds the model" >&2
  exit 1
fi

rc=0
PDFINTAKE_REQUIRE_THOROUGH=1 PYTHONDONTWRITEBYTECODE=1 \
  "$py" -m pytest -m thorough -p no:cacheprovider -q -rs --junitxml "$work/thorough.xml" "$@" || rc=$?

"$py" - "$work/thorough.xml" "$rc" <<'PY'
import sys
import xml.etree.ElementTree as ET

path, rc = sys.argv[1], int(sys.argv[2])
try:
    root = ET.parse(path).getroot()
except (OSError, ET.ParseError):
    print(f"the thorough gate failed: pytest wrote no report (exit {rc})", file=sys.stderr)
    sys.exit(1)
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
count = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "skipped", "failures", "errors")}
ran = count["tests"] - count["skipped"]
problems = []
if count["tests"] == 0 or rc == 5:
    problems.append("no thorough test was collected")
if count["skipped"]:
    problems.append(f"{count['skipped']} skipped")
if count["failures"] or count["errors"]:
    problems.append(f"{count['failures']} failed, {count['errors']} errors")
if rc not in (0, 5) and not problems:
    problems.append(f"pytest exited {rc}")
if problems:
    print("the thorough gate failed: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
print(f"the thorough gate passed: {ran} thorough tests ran, none skipped")
PY
