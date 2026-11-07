#!/usr/bin/env bash
# Tag a pdfintake release, push it, publish it as an immutable GitHub release, and prove it.
#
# The plugin's skill installs pdfintake pinned to its release tag, so a tag that never reached
# GitHub breaks every new install. `git push --follow-tags` skips lightweight tags, so a tag can
# stay local without anyone noticing.
#
# pdfintake also hosts its data wheels (English and the language packs) as release assets, and the
# core pins them by SHA-256. So before anything is tagged this script refuses unless:
#   - the repo-level tests pass with nothing skipped (missing docs or index fail, not skip);
#   - the Apple Vision tests pass on this Mac with nothing skipped (scripts/vision-local.sh);
#   - GitHub's immutable releases are on for the repo, so a published asset and its tag can
#     never change, even for someone holding the account;
#   - the pack-publish environment exists with RazorII as a required reviewer (GitHub creates a
#     missing environment with no protection the first time a job names it, so an unchecked
#     environment would upload without anyone approving);
#   - every data wheel the index and the pyproject pin name exists with the pinned hash: the
#     wheels this release hosts are rebuilt here with the command and Python version pack-publish.yml
#     rebuilds them with (scripts/dev-install.sh --wheels-only --python 3.12, which draws each
#     pack's canary before the build; the computer and its uv are this one's), and must match byte
#     for byte, and the ones an earlier release hosts must already be published there, immutable,
#     with the same digest;
#   - the release scan (RELEASE_SCAN, below) finds nothing in the wheels this release hosts.
# Then it records every pack version the index pins as published by this release, in
# tools/published-packs.json (tools/build_index.py --record-release), and commits that file, so the
# tag points at a commit that knows offline which pack versions are out: from then on
# build_index.py refuses a manifest change under a recorded version. A rerun finds the release
# already recorded and commits nothing.
# Then, in GitHub's documented order for immutable releases: create the release as a draft, let
# pack-publish.yml (approved in GitHub) attach the wheels, check every asset's digest, publish.
# A published release never changes: a mistake is fixed by releasing the next patch version.
# A rerun resumes: a draft holding some of the pinned wheels, each at its pinned digest, gets the
# upload job again, and that job attaches only the wheels still missing.
#
# usage: RELEASE_SCAN='<command> [args...]' scripts/release.sh X.Y.Z
# Environment: PYTHON (default python3) has pdfintake[test] installed; uv is on PATH (or UV names
# it) for scripts/dev-install.sh's canary environment. RELEASE_SCAN is required:
# the maintainer's release scan, a command (split on whitespace, never globbed; one line) that is
# run with the wheels this release hosts as its arguments (a relative command path is taken from
# where you start the script). A non-zero exit stops the release before anything is tagged.
# Without RELEASE_SCAN, or when its command isn't found, the script refuses to run at all.
set -euo pipefail
v="${1:?usage: scripts/release.sh X.Y.Z}"
[[ "$v" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || { echo "not a version: $v" >&2; exit 1; }
no_scan="RELEASE_SCAN must hold the release scan command, which gets the built wheels as arguments; refusing to release"
[ -n "${RELEASE_SCAN:-}" ] || { echo "RELEASE_SCAN is unset or empty. $no_scan" >&2; exit 1; }
case "$RELEASE_SCAN" in *$'\n'*) echo "RELEASE_SCAN holds more than one line. $no_scan" >&2; exit 1 ;; esac
read -ra scan <<< "$RELEASE_SCAN"
[ ${#scan[@]} -gt 0 ] || { echo "RELEASE_SCAN holds no command. $no_scan" >&2; exit 1; }
case "${scan[0]}" in /*) ;; */*) scan[0]="$PWD/${scan[0]}" ;; esac   # the script cds to the repo top below
command -v -- "${scan[0]}" >/dev/null || { echo "RELEASE_SCAN's command isn't found: ${scan[0]}. $no_scan" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
grep -qx "__version__ = \"$v\"" src/pdfintake/__init__.py || { echo "src/pdfintake/__init__.py is not at $v" >&2; exit 1; }
slug="RazorII/pdfintake"
py="${PYTHON:-python3}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
[ -z "$(git status --porcelain)" ] || { echo "commit everything first" >&2; exit 1; }
# pdfintake is published as RazorII. A fresh clone takes the global git name, so
# check every commit and the name the tag will carry before anything is pushed.
publisher="RazorII"
names="$( { git log --format='%an%n%cn'; git config user.name; } | sort -u)"
[ "$names" = "$publisher" ] || { echo "commits and the tag must be by $publisher (git config user.name $publisher); found: $(echo $names)" >&2; exit 1; }

cat > "$work/helper.py" <<'PY'
"""release.sh's helper: what the index pins, and whether GitHub holds exactly that."""
import hashlib
import json
import re
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

SLUG = "RazorII/pdfintake"
ASSET_URL = re.compile(r"^https://github\.com/" + re.escape(SLUG) +
                       r"/releases/download/(v(\d+)\.(\d+)\.(\d+))/([A-Za-z0-9_.+-]+\.whl)$")
PIN = re.compile(r"^pdfintake-data-en\s*@\s*(\S+?)#sha256=([0-9a-f]{64})\s*$")


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def wheel_name(distribution, version):
    return f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"


def plan(version):
    """One row per pinned data wheel: id, hosting tag, file, sha256, bytes."""
    want = tuple(int(x) for x in version.split("."))
    index_path = Path("src/pdfintake/langpacks/index.json")
    if not index_path.is_file():
        die("src/pdfintake/langpacks/index.json is missing: the release pins the packs through it")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    deps = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    pins = [m for m in (PIN.match(d) for d in deps) if m]
    if len(pins) != 1:
        die("pyproject.toml needs exactly one 'pdfintake-data-en @ URL#sha256=HASH' dependency")
    pin_url, pin_sha = pins[0].groups()
    rows, problems = [], []
    for entry in index.get("packs", []):
        pid, dist = entry.get("id"), entry.get("dist") or {}
        manifest = Path("packs") / str(pid) / "manifest.json"
        if not manifest.is_file():
            problems.append(f"{pid}: packs/{pid}/manifest.json is missing")
        elif not json.loads(manifest.read_text(encoding="utf-8")).get("canary"):
            problems.append(f"{pid}: a published pack needs a canary")
        if entry.get("base"):
            if dist.get("kind") != "installed":
                problems.append(f"{pid}: the base pack ships with the install (dist.kind 'installed')")
            url, sha = pin_url, pin_sha
            if sha != dist.get("sha256"):
                problems.append(f"{pid}: the pyproject pin's sha256 {sha} isn't the index's {dist.get('sha256')}")
        else:
            if dist.get("kind") != "fetch":
                problems.append(f"{pid}: only the base pack is installed; others are fetched")
            url, sha = dist.get("url") or "", dist.get("sha256") or ""
        m = ASSET_URL.match(url)
        if not m:
            problems.append(f"{pid}: {url!r} isn't a {SLUG} release asset")
            continue
        tag, file = m.group(1), m.group(5)
        expected = wheel_name(dist.get("distribution", ""), entry.get("version", ""))
        if file != expected:
            problems.append(f"{pid}: the URL names {file}, the index entry is {expected}")
        if tuple(int(x) for x in m.group(2, 3, 4)) > want:
            problems.append(f"{pid}: its URL points at {tag}, a release after v{version}")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            problems.append(f"{pid}: no sha256 pinned")
        rows.append((pid, tag, file, sha, str(dist.get("bytes", 0))))
    if not any(r for r in index.get("packs", []) if r.get("base")):
        problems.append("the index has no base pack")
    if problems:
        die("the data wheel pins don't hold:\n  " + "\n  ".join(problems))
    for row in rows:
        print("\t".join(row))


def check_built(folder, expect, tag):
    """The folder holds exactly the wheels ``tag`` hosts, each at its pinned hash and size; prints their paths."""
    rows = [line.split("\t") for line in Path(expect).read_text(encoding="utf-8").splitlines()]
    want = {file: (sha, size) for _pid, t, file, sha, size in rows if t == tag}
    built = sorted(p.name for p in Path(folder).glob("*.whl"))
    if built != sorted(want):
        die(f"scripts/dev-install.sh built {built}, expected {sorted(want)}")
    for file, (sha, size) in sorted(want.items()):
        data = (Path(folder) / file).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != sha or len(data) != int(size):
            die(f"{file} rebuilt as sha256 {got} ({len(data)} bytes); the index pins {sha} ({size} bytes)")
    for file in sorted(want):
        print(Path(folder) / file)


def rows_for(expect, tag):
    out = {}
    for line in Path(expect).read_text(encoding="utf-8").splitlines():
        pid, t, file, sha, size = line.split("\t")
        if t == tag:
            out[file] = sha
    return out


def asset_problems(release, want, exact):
    have = {a.get("name"): a.get("digest") for a in release.get("assets", [])}
    problems = []
    for file, sha in sorted(want.items()):
        if file not in have:
            problems.append(f"{file} isn't attached")
        elif have[file] != f"sha256:{sha}":
            problems.append(f"{file} has digest {have[file]}, the index pins sha256:{sha}")
    if exact:
        problems += [f"{name} is attached but not pinned" for name in sorted(set(have) - set(want))]
    return problems


def state(listing, tag, expect):
    """missing | published | draft-empty | draft-partial | draft-complete, and the release id.

    draft-partial: every attached asset is a pinned wheel at its pinned digest, but some are
    still missing (an upload job that stopped part way). The upload job runs again for the rest.
    """
    found = [r for r in json.loads(Path(listing).read_text(encoding="utf-8")) if r.get("tag_name") == tag]
    if not found:
        print("missing 0")
        return
    if len(found) > 1:
        die(f"GitHub has {len(found)} releases for {tag}; delete the extra drafts")
    rel = found[0]
    want = rows_for(expect, tag)
    if not rel.get("draft"):
        print(f"published {rel['id']}")
        return
    if not rel.get("assets"):
        print(f"draft-empty {rel['id']}")
        return
    problems = asset_problems(rel, want, exact=True)
    if not problems:
        print(f"draft-complete {rel['id']}")
        return
    # Only missing wheels, nothing attached that's unpinned or at another digest: resumable.
    have = {a.get("name") for a in rel.get("assets", [])}
    if problems == [f"{file} isn't attached" for file in sorted(set(want) - have)]:
        print(f"draft-partial {rel['id']}")
        return
    die(f"the {tag} draft has assets that don't match the pins: " + "; ".join(problems)
        + ". A draft is still editable: remove those assets on GitHub, then run this again")


def check(release_json, expect, tag, mode):
    """mode draft: exactly the pinned assets; published or older: also published and immutable."""
    rel = json.loads(Path(release_json).read_text(encoding="utf-8"))
    problems = asset_problems(rel, rows_for(expect, tag), exact=(mode != "older"))
    if mode == "draft" and not rel.get("draft"):
        problems.append("it's no longer a draft")
    if mode in ("published", "older"):
        if rel.get("draft"):
            problems.append("it's still a draft")
        if rel.get("immutable") is not True:
            problems.append("it isn't immutable")
    if problems:
        die(f"release {tag}: " + "; ".join(problems))


def gh_answer(path, err, rc, what):
    """What gh api answered: the JSON body, or None for a 404.

    Any other failure (auth, network, rate limit) stops here with gh's own words, so a setting that
    is already on is never reported as off (decided 2026-11-05).
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = None
    try:
        said = [ln.strip() for ln in Path(err).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except OSError:
        said = []
    if int(rc) == 0:
        if body is None:
            die(f"couldn't read {what} for {SLUG}: GitHub's answer isn't JSON")
        return body
    if any("HTTP 404" in ln for ln in said) or (
            isinstance(body, dict) and (str(body.get("status")) == "404" or body.get("message") == "Not Found")):
        return None
    die(f"couldn't read {what} for {SLUG}: " + (said[0] if said else f"gh api exited {rc}"))


def environment(path, err, rc):
    """pack-publish must hold the upload until RazorII approves it."""
    env = gh_answer(path, err, rc, "the pack-publish environment")
    if not isinstance(env, dict) or env.get("name") != "pack-publish":
        die(f"{SLUG} has no pack-publish environment: create it (Settings, Environments) with "
            "RazorII as a required reviewer before tagging")
    rules = [r for r in env.get("protection_rules") or [] if r.get("type") == "required_reviewers"]
    logins = {(rv.get("reviewer") or {}).get("login") for r in rules for rv in r.get("reviewers") or []
              if rv.get("type") == "User"}
    if "RazorII" not in logins:
        die(f"the pack-publish environment of {SLUG} doesn't require RazorII's approval: add RazorII as a "
            "required reviewer (Settings, Environments, pack-publish) before tagging")
    if any(r.get("prevent_self_review") for r in rules):
        die("the pack-publish environment prevents self-review, so RazorII, who starts the upload, could "
            "never approve it: turn 'Prevent self-review' off before tagging")


def immutable(path, err, rc):
    """The endpoint answers 404 while immutable releases are off, and 200 {enabled: false} too."""
    answer = gh_answer(path, err, rc, "the immutable-releases setting")
    if not (isinstance(answer, dict) and answer.get("enabled") is True):
        die(f"immutable releases are off for {SLUG}: turn them on in the repo's settings "
            "(Settings, General, Releases) before tagging")


def new_run(before, after, tag):
    seen = {r["id"] for r in json.loads(Path(before).read_text(encoding="utf-8")).get("workflow_runs", [])}
    runs = [r for r in json.loads(Path(after).read_text(encoding="utf-8")).get("workflow_runs", [])
            if r["id"] not in seen and r.get("head_branch") == tag]
    print(max(r["id"] for r in runs) if runs else "")


def no_skips(report, rc):
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    count = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "skipped", "failures", "errors")}
    if int(rc) != 0 or count["skipped"] or count["failures"] or count["errors"] or not count["tests"]:
        die(f"the repo-level tests must all run and pass: {count['tests']} tests, {count['skipped']} skipped, "
            f"{count['failures']} failed, {count['errors']} errors (pytest exit {rc})")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"plan": plan, "check-built": check_built, "state": state, "check": check,
     "new-run": new_run, "no-skips": no_skips, "immutable": immutable, "environment": environment}[cmd](*args)
PY
helper() { "$py" "$work/helper.py" "$@"; }

# Gate 1: the repo-level tests, strict. Missing docs, plugin files or index fail instead of skipping.
rc=0
PDFINTAKE_RELEASE_STRICT=1 PYTHONDONTWRITEBYTECODE=1 "$py" -m pytest -q -p no:cacheprovider -rs \
  --junitxml "$work/repo.xml" tests/test_release.py tests/test_docs.py tests/test_plugin_folder.py \
  tests/test_platform_table.py || rc=$?
[ -f "$work/repo.xml" ] || { echo "the repo-level tests didn't run (pytest exit $rc)" >&2; exit 1; }
helper no-skips "$work/repo.xml" "$rc"

# Gate 2: Apple Vision on this Mac, nothing skipped.
PYTHON="$py" scripts/vision-local.sh

# Gate 3: immutable releases on. The endpoint answers 404 while they're off; any other gh failure
# is reported as itself.
rc=0
gh api "repos/$slug/immutable-releases" > "$work/immutable.json" 2> "$work/immutable.err" || rc=$?
helper immutable "$work/immutable.json" "$work/immutable.err" "$rc"

# Gate 3b: the upload waits for RazorII's approval. The endpoint answers 404 for a missing environment.
rc=0
gh api "repos/$slug/environments/pack-publish" > "$work/environment.json" 2> "$work/environment.err" || rc=$?
helper environment "$work/environment.json" "$work/environment.err" "$rc"

# Gate 4: every pinned data wheel exists with its pinned hash. The wheels this release hosts are
# rebuilt with pack-publish.yml's command and Python version: scripts/dev-install.sh --wheels-only
# draws each pack's canary (datawheel.py alone can't: a canary has no upstream source), then builds
# the wheel.
helper plan "$v" > "$work/expect.tsv"
tag="v$v"
hosted=""
older=""
while IFS=$'\t' read -r id host _rest <&3; do
  if [ "$host" = "$tag" ]; then
    hosted="${hosted:+$hosted,}$id"
  else
    case " $older " in *" $host "*) ;; *) older="$older $host" ;; esac
  fi
done 3< "$work/expect.tsv"
built=()
if [ -n "$hosted" ]; then
  PDFINTAKE_DIST="$work/wheels" scripts/dev-install.sh --wheels-only --python 3.12 --packs "$hosted" </dev/null
  helper check-built "$work/wheels" "$work/expect.tsv" "$tag" > "$work/built.txt"
  while IFS= read -r w; do built+=("$w"); done < "$work/built.txt"
fi
for host in $older; do
  gh api "repos/$slug/releases/tags/$host" > "$work/release-$host.json" </dev/null \
    || { echo "release $host, which hosts pinned wheels, isn't on GitHub" >&2; exit 1; }
  helper check "$work/release-$host.json" "$work/expect.tsv" "$host" older
done

# Gate 5: the release scan (RELEASE_SCAN) on the wheels this release will host. A release that
# hosts none (every pinned wheel is an earlier release's) has nothing new to scan.
if [ ${#built[@]} -gt 0 ]; then
  "${scan[@]}" "${built[@]}" || { echo "the release scan stopped the release: see its findings above" >&2; exit 1; }
fi

# The published record: every pack version the index pins counts as published from this release
# on. It goes into the commit the tag points at, so it's checked before anything is committed: a tag
# that exists already must be this release's own annotated tag at HEAD (a rerun), and a rerun finds
# the release recorded. The record is written to the work folder first and copied in only if it
# changed.
record="tools/published-packs.json"
tag_type="$(git cat-file -t "refs/tags/$tag" 2>/dev/null || true)"
if [ -n "$tag_type" ] && { [ "$tag_type" != tag ] || [ "$(git rev-parse "refs/tags/$tag^{commit}")" != "$(git rev-parse HEAD)" ]; }; then
  echo "fatal: tag '$tag' already exists and isn't this release's annotated tag at HEAD; it is never replaced" >&2
  exit 1
fi
[ ! -f "$record" ] || cp "$record" "$work/record.json"
"$py" tools/build_index.py --record-release "$v" --published "$work/record.json" </dev/null
if ! cmp -s "$work/record.json" "$record"; then
  [ -z "$tag_type" ] || { echo "$tag is already tagged at HEAD, but $record doesn't record it, and a tag never moves: stopping" >&2; exit 1; }
  cp "$work/record.json" "$record"
  git add -- "$record"
  git commit -q -m "Record the packs pdfintake $v publishes" -- "$record"
fi

# A rerun after a failed push finds its own annotated tag already at HEAD and pushes it again. A tag anywhere
# else, or a lightweight one, still stops here: git tag -a refuses to replace it.
if [ "$(git cat-file -t "refs/tags/$tag" 2>/dev/null)" != tag ] || [ "$(git rev-parse "refs/tags/$tag^{commit}")" != "$(git rev-parse HEAD)" ]; then
  git tag -a "$tag" -m "pdfintake $v"
fi
# --atomic: main and the tag land together or not at all, so a refused tag can't leave main published
# without its release (installs pin to the tag).
git push --atomic origin HEAD "refs/tags/$tag"
git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null || { echo "$tag did not reach GitHub" >&2; exit 1; }
echo "$tag is on GitHub"

# The release: draft, assets, digests, publish. A rerun picks up wherever the last run stopped.
gh api "repos/$slug/releases?per_page=100" > "$work/releases.json"
st="$(helper state "$work/releases.json" "$tag" "$work/expect.tsv")"
state="${st%% *}"; id="${st##* }"
if [ "$state" = missing ]; then
  gh release create "$tag" --repo "$slug" --draft --verify-tag --title "pdfintake $v" \
    --notes "pdfintake $v. Install: uv tool install git+https://github.com/$slug@$tag"
  # GitHub's release listing can lag the create by a few seconds (the v0.1.0 draft was missing from
  # the first look), so look again before giving up.
  for _ in $(seq 1 10); do
    gh api "repos/$slug/releases?per_page=100" > "$work/releases.json"
    st="$(helper state "$work/releases.json" "$tag" "$work/expect.tsv")"
    state="${st%% *}"; id="${st##* }"
    [ "$state" = missing ] || break
    sleep 3
  done
  [ "$state" = draft-empty ] || { echo "the $tag draft didn't appear on GitHub (state: $state)" >&2; exit 1; }
fi
if { [ "$state" = draft-empty ] || [ "$state" = draft-partial ]; } && [ -n "$(cut -f2 "$work/expect.tsv" | grep -x "$tag" || true)" ]; then
  gh api "repos/$slug/actions/workflows/pack-publish.yml/runs?event=workflow_dispatch&per_page=20" > "$work/runs-before.json"
  gh workflow run pack-publish.yml --repo "$slug" --ref "$tag" -f tag="$tag"
  run=""
  for _ in $(seq 1 30); do
    gh api "repos/$slug/actions/workflows/pack-publish.yml/runs?event=workflow_dispatch&per_page=20" > "$work/runs-after.json"
    run="$(helper new-run "$work/runs-before.json" "$work/runs-after.json" "$tag")"
    [ -n "$run" ] && break
    sleep 2
  done
  [ -n "$run" ] || { echo "pack-publish.yml didn't start for $tag" >&2; exit 1; }
  echo "pack-publish.yml is uploading the wheels: approve its pack-publish environment in GitHub"
  gh run watch "$run" --repo "$slug" --exit-status
fi
if [ "$state" != published ]; then
  gh api "repos/$slug/releases/$id" > "$work/draft.json"
  helper check "$work/draft.json" "$work/expect.tsv" "$tag" draft
  gh release edit "$tag" --repo "$slug" --draft=false
fi
gh api "repos/$slug/releases/$id" > "$work/published.json"
helper check "$work/published.json" "$work/expect.tsv" "$tag" published
echo "$tag is published as an immutable release, every asset at its pinned digest"
