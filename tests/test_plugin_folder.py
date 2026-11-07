"""The plugin folder holds only what the Claude plugin directory accepts.

* No ``bin/``, no package-manager config, no hooks and no ``.mcp.json``: the plugin is one skill
  that installs the pinned command, nothing that runs on its own.
* Only text and image files of the allowed types, each non-image under 256 KiB, at most 512 files.
* SKILL.md's frontmatter keys are a subset of the directory's list, and no file names
  ``${CLAUDE_*}`` (a plugin-runtime variable a directory plugin must not depend on).
* The frontmatter grants no question tool: a run never asks (DESIGN §12, decided 2026-11-05), so
  ``allowed-tools`` never lists AskUserQuestion. tests/test_docs.py checks the skill's text.
* The plugin README has at least 40 words outside code blocks and a ``## Examples`` list of at
  least three prompts (directory policy 3E).

Each rule is a function that returns problems, tested on planted folders as well as the real one.
A missing plugin folder skips while it is being written, and fails under PDFINTAKE_RELEASE_STRICT=1
(scripts/release.sh).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "claude-plugin"
SKILL = PLUGIN / "skills" / "pdfintake" / "SKILL.md"

ALLOWED_SUFFIXES = {".md", ".json", ".png", ".txt"}
IMAGE_SUFFIXES = {".png"}
MAX_TEXT_BYTES = 256 * 1024
MAX_FILES = 512
FRONTMATTER_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
PACKAGE_MANAGER_FILES = {
    ".npmrc", ".yarnrc", ".yarnrc.yml", ".pnpmfile.cjs", "package.json", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "bun.lockb", "bun.lock", "uv.toml", "uv.lock", "pyproject.toml", "setup.py", "setup.cfg",
    "requirements.txt", "Pipfile", "Pipfile.lock", "poetry.lock", "pip.conf", "pip.ini", ".python-version",
    "Gemfile", "Gemfile.lock", "Cargo.toml", "Cargo.lock", "go.mod", "go.sum", "Brewfile",
}
CLAUDE_VAR = re.compile(r"\$\{CLAUDE_[A-Za-z0-9_]*\}")
QUESTION_TOOLS = ("AskUserQuestion",)


def _need(path: Path, what: str) -> None:
    if path.exists():
        return
    msg = f"{what} isn't there yet: {path.relative_to(ROOT)}"
    if os.environ.get("PDFINTAKE_RELEASE_STRICT") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _files(plugin: Path) -> list[Path]:
    return sorted(p for p in plugin.rglob("*") if p.is_file() and p.name != ".DS_Store")


def structure_problems(plugin: Path) -> list[str]:
    problems = []
    for p in sorted(plugin.rglob("*")):
        rel = p.relative_to(plugin).as_posix()
        if p.is_dir() and p.name in {"bin", "hooks"}:
            problems.append(f"{rel}/: a plugin folder has no {p.name}/")
        if p.is_file():
            if p.name in PACKAGE_MANAGER_FILES or p.name.startswith("requirements") and p.suffix == ".txt":
                problems.append(f"{rel}: package-manager config")
            if p.name in {".mcp.json", "hooks.json"}:
                problems.append(f"{rel}: no MCP servers or hooks")
    manifest = plugin / ".claude-plugin" / "plugin.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("hooks", "mcpServers", "lspServers"):
            if key in data:
                problems.append(f"plugin.json declares {key}")
    return problems


def file_problems(plugin: Path) -> list[str]:
    files = _files(plugin)
    problems = []
    if len(files) > MAX_FILES:
        problems.append(f"{len(files)} files, at most {MAX_FILES}")
    for p in files:
        rel = p.relative_to(plugin).as_posix()
        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            problems.append(f"{rel}: file type {p.suffix or '(none)'} isn't allowed")
        elif p.suffix.lower() not in IMAGE_SUFFIXES and p.stat().st_size > MAX_TEXT_BYTES:
            problems.append(f"{rel}: {p.stat().st_size} bytes, over 256 KiB")
        if p.suffix.lower() not in IMAGE_SUFFIXES and p.suffix.lower() in ALLOWED_SUFFIXES:
            if CLAUDE_VAR.search(p.read_text(encoding="utf-8", errors="replace")):
                problems.append(f"{rel}: names a ${{CLAUDE_*}} variable")
    return problems


def frontmatter_lines(text: str) -> list[str] | None:
    """The lines between a SKILL.md's opening and closing ``---``; None without a closed block."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[1:i]
    return None


def frontmatter(text: str) -> dict[str, str] | None:
    """The top-level keys of a SKILL.md frontmatter block (no YAML parser needed for the key set)."""
    lines = frontmatter_lines(text)
    if lines is None:
        return None
    keys: dict[str, str] = {}
    for line in lines:
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if m:
            keys[m.group(1)] = m.group(2)
    return keys


def skill_problems(skill: Path) -> list[str]:
    fm = frontmatter(skill.read_text(encoding="utf-8"))
    if fm is None:
        return ["SKILL.md has no frontmatter block"]
    problems = [f"frontmatter key {k!r} isn't allowed" for k in sorted(set(fm) - FRONTMATTER_KEYS)]
    for k in ("name", "description"):
        if not fm.get(k, "").strip():
            problems.append(f"frontmatter has no {k}")
    block = "\n".join(frontmatter_lines(skill.read_text(encoding="utf-8")) or [])
    for tool in QUESTION_TOOLS:
        if re.search(rf"(?<![A-Za-z0-9_]){tool}(?![A-Za-z0-9_])", block):
            problems.append(f"frontmatter grants {tool}; a run never asks")
    return problems


def readme_problems(readme: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    words = len(re.sub(r"```.*?```", " ", text, flags=re.S).split())
    problems = [] if words >= 40 else [f"README has {words} words outside code blocks, at least 40 needed"]
    ex = re.search(r"^## Examples\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    n = len(re.findall(r"^- ", ex.group(1), flags=re.M)) if ex else 0
    if n < 3:
        problems.append(f"README '## Examples' lists {n} prompts, at least 3 needed")
    return problems


# --- the real plugin folder

def test_plugin_folder_structure():
    _need(PLUGIN, "the plugin folder")
    assert structure_problems(PLUGIN) == []


def test_plugin_files_are_allowed_types_and_sizes():
    _need(PLUGIN, "the plugin folder")
    assert file_problems(PLUGIN) == []


def test_skill_frontmatter():
    _need(SKILL, "SKILL.md")
    assert skill_problems(SKILL) == []
    assert frontmatter(SKILL.read_text(encoding="utf-8"))["name"].strip() == "pdfintake"


def test_plugin_readme_words_and_examples():
    _need(PLUGIN / "README.md", "the plugin README")
    assert readme_problems(PLUGIN / "README.md") == []


# --- the rules catch what they're for

def _plugin(tmp_path: Path) -> Path:
    p = tmp_path / "claude-plugin"
    (p / ".claude-plugin").mkdir(parents=True)
    (p / ".claude-plugin" / "plugin.json").write_text('{"name": "example"}', encoding="utf-8")
    (p / "skills" / "example").mkdir(parents=True)
    (p / "skills" / "example" / "SKILL.md").write_text("---\nname: example\ndescription: Does a thing.\n---\nBody.\n",
                                                       encoding="utf-8")
    return p


def test_a_clean_planted_folder_passes(tmp_path: Path):
    p = _plugin(tmp_path)
    assert structure_problems(p) == [] and file_problems(p) == []
    assert skill_problems(p / "skills" / "example" / "SKILL.md") == []


@pytest.mark.parametrize("plant", ["bin/run", "hooks/x.md", ".mcp.json", "uv.toml", ".npmrc",
                                   "skills/example/package.json", "requirements-dev.txt"])
def test_structure_rules_catch_plants(tmp_path: Path, plant: str):
    p = _plugin(tmp_path)
    (p / plant).parent.mkdir(parents=True, exist_ok=True)
    (p / plant).write_text("x", encoding="utf-8")
    assert structure_problems(p)


def test_manifest_hooks_and_mcp_servers_are_caught(tmp_path: Path):
    p = _plugin(tmp_path)
    (p / ".claude-plugin" / "plugin.json").write_text('{"name": "example", "mcpServers": {}}', encoding="utf-8")
    assert structure_problems(p) == ["plugin.json declares mcpServers"]


def test_file_rules_catch_types_sizes_counts_and_variables(tmp_path: Path):
    p = _plugin(tmp_path)
    (p / "tool.py").write_text("print()", encoding="utf-8")
    (p / "big.md").write_text("x" * (MAX_TEXT_BYTES + 1), encoding="utf-8")
    (p / "uses.md").write_text("run ${CLAUDE_PLUGIN_ROOT}/x", encoding="utf-8")
    (p / "icon.png").write_bytes(b"\x89PNG" + b"\0" * (MAX_TEXT_BYTES + 10))     # images may be bigger
    found = " | ".join(file_problems(p))
    assert "tool.py: file type .py" in found and "big.md" in found and "uses.md: names a" in found
    assert "icon.png" not in found
    many = tmp_path / "many"
    many.mkdir()
    for i in range(MAX_FILES + 1):
        (many / f"{i}.txt").write_text("x", encoding="utf-8")
    assert f"{MAX_FILES + 1} files" in file_problems(many)[0]


def test_frontmatter_rules(tmp_path: Path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("---\nname: x\ndescription: y\nmodel: opus\nhooks: []\n---\n", encoding="utf-8")
    assert skill_problems(skill) == ["frontmatter key 'hooks' isn't allowed", "frontmatter key 'model' isn't allowed"]
    skill.write_text("no frontmatter\n", encoding="utf-8")
    assert skill_problems(skill) == ["SKILL.md has no frontmatter block"]
    skill.write_text("---\nname: x\n---\n", encoding="utf-8")
    assert skill_problems(skill) == ["frontmatter has no description"]
    skill.write_text("---\nname: x\ndescription: y\n---\nBody.\n", encoding="utf-8")
    assert skill_problems(skill) == []
    skill.write_text("---\nname: x\ndescription: y\n", encoding="utf-8")
    assert skill_problems(skill) == ["SKILL.md has no frontmatter block"]


@pytest.mark.parametrize("tools", ["allowed-tools: Bash, AskUserQuestion",
                                   "allowed-tools:\n  - Bash\n  - AskUserQuestion"])
def test_the_frontmatter_grants_no_question_tool(tmp_path: Path, tools: str):
    skill = tmp_path / "SKILL.md"
    skill.write_text(f"---\nname: x\ndescription: y\n{tools}\n---\nBody.\n", encoding="utf-8")
    assert skill_problems(skill) == ["frontmatter grants AskUserQuestion; a run never asks"]
    skill.write_text("---\nname: x\ndescription: y\nallowed-tools: Bash, Read\n---\nBody.\n", encoding="utf-8")
    assert skill_problems(skill) == []


def test_readme_rules(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("Short.\n\n```\n" + "word " * 100 + "\n```\n\n## Examples\n\n- one\n- two\n", encoding="utf-8")
    assert readme_problems(readme) == ["README has 7 words outside code blocks, at least 40 needed",
                                       "README '## Examples' lists 2 prompts, at least 3 needed"]
    readme.write_text("word " * 40 + "\n\n## Examples\n\n- a\n- b\n- c\n\n## Install\n", encoding="utf-8")
    assert readme_problems(readme) == []
