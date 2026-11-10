"""The public docs agree with the code they describe, and none of them describes a run that asks.

* **Nothing is asked during a run** (DESIGN §12, decided 2026-11-05). No doc names AskUserQuestion,
  exit 10 (``EXIT_DECISION``), or a question flow: asking the user, a numbered option list, a
  ``provenance`` or ``password`` decision, ``decision.question``, ``--password-stdin``, the
  one-password ``--password-file`` (``--passwords-file`` is the saved list and fine),
  ``password_wrong``, ``--fetch ask``, ``--smaller ask`` or ``--choice keep|supply|decline``. A
  sentence that says it *doesn't* happen ("``run`` never exits 10", "exit 10 is retired") is fine, as
  long as the negation governs the match itself and not a condition before it;
  AskUserQuestion isn't named at all, not even to forbid it. Every Markdown file in the repo and
  the plugin manifest are scanned.
* The exit-code tables equal the ``errors.EXIT_*`` constants. The skill and docs/decision-record.md
  give exactly the codes of ``run`` and ``apply`` (``MAIN_CODES``, which has no 10): the kit's codes
  are for contributors, and a record never carries one (decided 2026-11-05). README.md gives those
  codes and either all three kit codes or none of them.
  A table counts when its first header cell is ``Code``, ``Exit``, ``Exit code``, ``Exit status`` or
  ``Outcome``; the skill may instead give one bold bullet per outcome (``- **0.** …``, ``- **1:** …``).
* Where SKILL.md, README.md or docs/decision-record.md give the order in which several records decide
  the process exit ("2, then 3, 1, 4, 5, 0"), it is ``record.EXIT_PRIORITY`` exactly.
* Every warning code in ``record.WARNING_CODES`` is explained in SKILL.md, and so are
  ``copy_unlocked`` and ``skipped_no_password``, which the skill always reports (DESIGN §10.2, §12.6).
* Every refusal code (``refusals.CODES``, ``damaged`` included, and the codes ``flow`` refuses
  ``apply`` with, such as ``no_smaller_for_a_copy``) and every usage code (``record.USAGE_CODES``,
  ``input_not_file`` included) is explained in SKILL.md and in docs/decision-record.md. Each exit
  table says ``apply`` can exit 4 too, and each of the three docs says a stop by SIGTERM or SIGHUP
  exits 128 plus the signal number (decided 2026-11-06).
* The skill's password steps hold: the password file is written with the file tool, made private
  with ``chmod 600``, then saved with ``passwords add --from-file`` (or passed once with
  ``--passwords-file``); every reason ``passwords.py`` turns a passwords file down for is explained
  in SKILL.md and docs/decision-record.md; the skill merges images by default, chmods only a path
  it chose, gives the three notes with their own summary lines their exemption in both places, and
  lists everything ``export`` refuses.
* Every code a record can carry is documented in docs/decision-record.md, and nothing else is: the
  warning, note and usage codes each in their own section, exactly; the offers table names every
  offer kind with its apply flag and option ids (there is no decisions table any more); and the
  refusal codes, output kinds, statuses, apply flags, provenances and language plans appear in
  the page, each in backticks, and so do the step statuses.
* Every download host (``langpacks.index.ALLOWED_HOSTS`` and the pyproject pin of the English
  data wheel) is named in the plugin README and in PRIVACY.md.
* Every runtime dependency in pyproject has a row in THIRD-PARTY.md (its first column).
* The ``steps[]`` details in docs/decision-record.md name exactly the keywords ``flow.py`` passes to
  ``_step(d, "detect", ...)`` and ``_step(d, "orientation", ...)`` (read with ``ast``), plus record
  codes. PRIVACY.md's list of what stays in ``work/`` has one item per file the cache keeps, and
  every file name ``ocr/pipeline.py`` and ``flow.py`` spell out is accounted for
  (``WORK_FILE_NAMES``): each string or f-string, constant or inline (``entry / "x.json"``),
  read with ``ast``. A file named only through a variable or a call is out of its reach.

A doc or module that isn't written yet skips, and fails under PDFINTAKE_RELEASE_STRICT=1
(scripts/release.sh), as does the placeholder pin.
"""

from __future__ import annotations

import ast
import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pytest

from pdfintake import errors

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PRIVACY = ROOT / "PRIVACY.md"
THIRD_PARTY = ROOT / "THIRD-PARTY.md"
PLUGIN_README = ROOT / "claude-plugin" / "README.md"
SKILL = ROOT / "claude-plugin" / "skills" / "pdfintake" / "SKILL.md"
RECORD_DOC = ROOT / "docs" / "decision-record.md"
PLUGIN_MANIFEST = ROOT / "claude-plugin" / ".claude-plugin" / "plugin.json"

KIT_CODES = {errors.EXIT_KIT_FLOOR, errors.EXIT_KIT_DRIFT, errors.EXIT_KIT_PRIVACY}
ALL_CODES = {getattr(errors, n) for n in dir(errors) if n.startswith("EXIT_")}
# The "decision needed" exit, retired with every question (DESIGN §12.5, decided 2026-11-05).
RETIRED_DECISION_EXIT = 10
MAIN_CODES = ALL_CODES - KIT_CODES - {RETIRED_DECISION_EXIT}
PIN = re.compile(r"^pdfintake-data-en\s*@\s*(\S+?)#sha256=([0-9a-f]{64})\s*$")
PLACEHOLDER_HOSTS = {"placeholder.invalid"}


def _strict() -> bool:
    return os.environ.get("PDFINTAKE_RELEASE_STRICT") == "1"


def _not_yet(msg: str) -> None:
    if _strict():
        pytest.fail(msg)
    pytest.skip(msg)


def _read(path: Path) -> str:
    if not path.is_file():
        _not_yet(f"{path.relative_to(ROOT)} isn't written yet")
    return path.read_text(encoding="utf-8")


# --- parsers

def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(cell: str) -> str:
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    return cell.replace("`", "").replace("*", "").strip()


def tables(text: str) -> list[list[list[str]]]:
    """Every Markdown table: a list of rows (header first, separator dropped), each a list of cells."""
    out, block = [], []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2 and re.fullmatch(r"[\s|:-]+", block[1]):
            out.append([_cells(block[0])] + [_cells(r) for r in block[2:]])
        block = []
    return out


EXIT_TABLE_HEADERS = {"code", "exit", "exit code", "exit codes", "exit status", "outcome", "return code"}


def exit_codes_in(text: str) -> set[int]:
    codes: set[int] = set()
    for table in tables(text):
        if _plain(table[0][0]).lower() in EXIT_TABLE_HEADERS:
            for row in table[1:]:
                codes |= {int(n) for n in re.findall(r"\d+", _plain(row[0]))}
    if not codes:
        for m in re.finditer(r"^\s*(?:[-*]|\d+\.)\s+\*\*([0-9][0-9, or]*?)[.:]?\*\*", text, flags=re.M):
            codes |= {int(n) for n in re.findall(r"\d+", m.group(1))}
    return codes


def section(text: str, heading: str) -> str:
    """The body under ``## heading`` (any level), up to the next heading of the same or a higher level."""
    m = re.search(rf"^(#+)\s+{re.escape(heading)}\s*$", text, flags=re.M)
    if not m:
        return ""
    level = len(m.group(1))
    end = re.search(rf"^#{{1,{level}}}\s", text[m.end():], flags=re.M)
    return text[m.end(): m.end() + end.start()] if end else text[m.end():]


def codes_in(text: str) -> set[str]:
    """Every snake_case code written in backticks (``layer_shrank``; not ``WARNING_MESSAGES``)."""
    return {c for c in re.findall(r"`([^`\n]+)`", text) if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", c)}


def explained(code: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(code)}(?![A-Za-z0-9_])", text) is not None


def _dist_name(requirement: str) -> str:
    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else ""


def third_party_names(text: str) -> set[str]:
    names = set()
    for table in tables(text):
        for row in table:
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", _plain(row[0]))
            if m:
                names.add(re.sub(r"[-_.]+", "-", m.group(1)).lower())
    return names


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def pin_host() -> str | None:
    pins = [m for m in (PIN.match(d) for d in _pyproject()["project"]["dependencies"]) if m]
    return urlparse(pins[0].group(1)).hostname if len(pins) == 1 else None


# --- nothing is asked during a run (DESIGN §12)

# Named at all, even to forbid it, is a finding: the skill has nothing to ask with it.
ALWAYS_FORBIDDEN = (("AskUserQuestion", re.compile(r"AskUserQuestion")),)
# A finding unless the match itself is negated (see `_negated`).
QUESTION_FLOW = (
    ("exit 10", re.compile(r"\bexit(?:s|ed|ing)?\s+(?:code\s+|status\s+|with\s+(?:code\s+)?)?`?10`?(?![0-9])", re.I)),
    ("exit 10", re.compile(r"\bEXIT_DECISION\b")),
    ("exit 10", re.compile(r"^\s*(?:[-*]|\d+\.)\s+\*\*(?:[0-9]+(?:,\s*|\s+or\s+))*10[.:]?\*\*", re.M)),
    ("exit 10", re.compile(r"\breturn(?:s|ed|ing)?\s+(?:exit\s+|code\s+)?`?10`?(?![0-9])", re.I)),
    ("a question flow", re.compile(r"\bask(?:s|ed)?\s+(?:the\s+user|them|the\s+question|it\s+like|again|"
                                   r"the\s+record's|(?:its|the|a|each|one)\s+questions?)\b", re.I)),
    ("a question flow", re.compile(r"\basks?\s+you\b", re.I)),
    ("a question flow", re.compile(r"\bquestions?\s+(?:pdfintake|claude|the\s+skill|it|a\s+run)\s+asks?\b", re.I)),
    ("a question flow", re.compile(r"\bhands?\s+back\s+a\s+question\b", re.I)),
    ("a question flow", re.compile(r"\banswer(?:s|ing)?\s+(?:a|the|its|each)\s+question", re.I)),
    ("a question flow", re.compile(r"\b(?:one\s+question\s+per|(?:report|show)s?\s+(?:each|the|its)\s+question)\b", re.I)),
    ("a question flow", re.compile(r"\bnumbered\s+(?:option\s+)?list\b", re.I)),
    ("a question flow", re.compile(r"\bwait(?:s|ing)?\s+for\s+(?:the|their|an?|your)\s+(?:reply|answer|user)", re.I)),
    ("a question flow", re.compile(r"\bprompt(?:s|ed|ing)?\s+(?:the\s+user|them|you)\b", re.I)),
    ("a question flow", re.compile(r"\bwhen\s+(?:you|the\s+user)\s+agrees?\b", re.I)),
    ("a question flow", re.compile(r"\bbefore\s+I\s+start\b|\bdid\s+any\s+of\s+these\s+come\s+from\b", re.I)),
    ("a question flow", re.compile(r"\bdecision\.(?:question|apply_flag|options|kind|answer)\b")),
    ("a question flow", re.compile(r"\bdecisions?\s+of\s+kind\b", re.I)),
    ("a question flow", re.compile(r"\b(?:a|an|the|its|this|that|same)\s+`?(?:provenance|password|language|smaller)`?"
                                   r"\s+decisions?\b", re.I)),
    ("a question flow", re.compile(r"record's\s+`decision`|`decision`\s+(?:field|key|object|status)\b|"
                                   r"\bstatus\s+`decision`", re.I)),
    ("a question flow", re.compile(r"\bapply\b[^\n|]*?--provenance\b")),
    ("a question flow", re.compile(r"--password-stdin\b|--password-file\b(?!s)|\bpassword_wrong\b")),
    ("a question flow", re.compile(r"--(?:fetch|smaller)[ =](?:[a-z]+\|)*ask\b")),
    ("a question flow", re.compile(r"--choice[ =](?:keep|supply|decline)\b")),
)
# "Ask for / whether / which …" is a question flow only when Claude, pdfintake or the skill does the
# asking (an imperative, or one of `_ASKERS` as the subject): "when you ask for it" and "if the person
# later asks for one" are the user's own request, which is fine. "You" asks only a question: in
# SKILL.md "you" is Claude, while "the file you asked for" is the user's request.
ASK_FOR = re.compile(r"\bask(?:s|ed|ing)?\s+for\b", re.I)
ASK_WHETHER = re.compile(r"\bask(?:s|ed|ing)?\s+(?:whether|which|if|what|about)\b(?!\s+you\s+want)", re.I)
_ASKERS = {"claude", "pdfintake", "the skill", "skill", "it", "run", "the run", "the cli", "we", "i"}
_FILLERS = {"then", "first", "also", "now", "just", "always", "only", "please", "should", "will", "can",
            "must", "may", "would", "could", "later", "so", "and"}
_CLAUSE_END = re.compile(r"[.!?;,](?=\s|$)|:\s|\s(?:and|but|then|so|or|while|when)\s|\n")
_NEGATION_WORD = re.compile(r"^\W*(?:never|no|not|nor|nothing|none|nobody|without)\W*$|n['’]t\W*$", re.I)
# A clause that opens with one of these is a condition: "If nothing opens it ask the user" negates the
# condition, not the asking, so only a negation right before the match counts there.
_CONDITIONAL = {"if", "unless", "when", "whenever", "once"}
_NEGATION_REACH = 4  # "Nothing in a record waits for an answer": the negation is 4 words before
_RETIRED_AFTER = re.compile(r"^\W*(?:is|are|was|were|has\s+been|have\s+been)\s+(?:retired|removed|gone)\b", re.I)
# A run of exit codes in the order that decides the process exit: "(2, then 3, 1, 4, 5, 0)",
# "in the order 2, 3, 1, 4, 5, 0", "2 > 3 > 1 > 4 > 5 > 0".
_CODE_RUN = re.compile(r"(?<!\d)(?<!\d\.)\d+(?:\s*(?:,\s*(?:then\s+|and\s+)?|>|\bthen\b|\band\b)\s*\d+(?!\d|\.\d)){2,}")
_PRIORITY_WORDS = re.compile(r"priorit|most\s+important|\border\b", re.I)


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    """The text of the match's own clause before it and after it."""
    before = [m.end() for m in _CLAUSE_END.finditer(text, 0, start)]
    left = before[-1] if before else 0
    after = _CLAUSE_END.search(text, end)
    return text[left:start], text[end: after.start() if after else len(text)]


def _words(before: str) -> list[str]:
    return [w for w in (re.sub(r"[`*_>\"“”]", "", w).lower() for w in before.split()) if w and w != "-"]


def _negated(text: str, start: int, end: int) -> bool:
    """True when the match itself is negated: a negation among the few words right before it ("never
    exits 10", "doesn't ask the user", "Nothing in a record waits for an answer", "without stopping to
    ask you"), or "is retired / removed / gone" right after it. A negation in another clause doesn't
    count ("writes nothing and returns exit 10"), and in a clause that opens with a condition only the
    word right before the match does: "If no saved password opens it ask the user" asks."""
    before, after = _clause(text, start, end)
    if _RETIRED_AFTER.search(after):
        return True
    words = _words(before)
    conditional = any(w.strip("()") in _CONDITIONAL for w in words) or (words[:1] == ["without"] and len(words) > 1)
    reach = words[-1:] if conditional else words[-_NEGATION_REACH:]
    return any(_NEGATION_WORD.search(w) for w in reach)


def _asked_by_the_skill(text: str, start: int, end: int, *, you_count: bool) -> bool:
    """True when the asking is Claude's, pdfintake's or the skill's: the clause has no subject (an
    imperative, "Ask which language") or one of `_ASKERS`, or "you" when ``you_count``."""
    before, _ = _clause(text, start, end)
    words = [w for w in _words(before) if not re.fullmatch(r"\d+\.|\(?[a-z0-9]\)", w)]
    subject = " ".join(w for w in words if w not in _FILLERS and not _NEGATION_WORD.search(w))
    return subject == "" or subject in _ASKERS or (you_count and subject == "you")


def priority_sequences(text: str) -> list[tuple[int, ...]]:
    """Every run of three or more exit codes whose sentence speaks of their priority or order."""
    out = []
    for m in _CODE_RUN.finditer(text):
        sentence_start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        if _PRIORITY_WORDS.search(text[sentence_start: m.start()]):
            out.append(tuple(int(n) for n in re.findall(r"\d+", m.group())))
    return out


def question_flow_problems(text: str) -> list[str]:
    """Each place ``text`` describes asking during a run, as ``line N: what: excerpt``."""
    hits = []
    for what, pattern in ALWAYS_FORBIDDEN:
        hits += [(m.start(), what) for m in pattern.finditer(text)]
    for what, pattern in QUESTION_FLOW:
        hits += [(m.start(), what) for m in pattern.finditer(text) if not _negated(text, m.start(), m.end())]
    for pattern, you_count in ((ASK_FOR, False), (ASK_WHETHER, True)):
        hits += [(m.start(), "a question flow") for m in pattern.finditer(text)
                 if not _negated(text, m.start(), m.end())
                 and _asked_by_the_skill(text, m.start(), m.end(), you_count=you_count)]
    for m in _CODE_RUN.finditer(text):
        sentence_start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        if (RETIRED_DECISION_EXIT in {int(n) for n in re.findall(r"\d+", m.group())}
                and (_PRIORITY_WORDS.search(text[sentence_start: m.start()])
                     or re.search(r"\bexit", text[sentence_start: m.start()], re.I))):
            hits.append((m.start(), "exit 10 in an exit-priority list"))
    problems = []
    for start, what in sorted(set(hits)):
        line_no = text.count("\n", 0, start) + 1
        line = text.splitlines()[line_no - 1].strip()
        problems.append(f"line {line_no}: {what}: {line[:100]}")
    for code in sorted(exit_codes_in(text) & {RETIRED_DECISION_EXIT}):
        problems.append(f"an exit-code table or bullet lists {code}")
    return problems


def scanned_docs() -> list[Path]:
    """Every Markdown file in the repo (git data, tests and virtualenvs aside), and the plugin manifest."""
    skip = {".git", "tests", ".venv", "venv", "node_modules", "build", "dist"}
    docs = [p for p in ROOT.rglob("*.md") if not skip & set(p.relative_to(ROOT).parts[:-1])]
    if PLUGIN_MANIFEST.is_file():
        docs.append(PLUGIN_MANIFEST)
    return sorted(docs)


# --- the real docs

@pytest.mark.parametrize("doc", scanned_docs(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_doc_describes_a_run_that_asks(doc: Path):
    problems = question_flow_problems(doc.read_text(encoding="utf-8"))
    assert problems == [], f"{doc.relative_to(ROOT)} describes asking during a run (DESIGN §12): " + " | ".join(problems)


def test_the_skill_and_the_record_doc_are_scanned():
    for doc in (SKILL, RECORD_DOC, README, PLUGIN_README):
        _read(doc)
        assert doc in scanned_docs(), doc


def test_errors_has_no_decision_exit():
    assert not hasattr(errors, "EXIT_DECISION"), "exit 10 is retired (DESIGN §12.5)"
    assert RETIRED_DECISION_EXIT not in ALL_CODES, "no EXIT_* constant may be 10 (DESIGN §12.5)"


def test_readme_exit_table_equals_the_constants():
    codes = exit_codes_in(_read(README))
    assert codes in (MAIN_CODES, ALL_CODES), f"README documents {sorted(codes)}; errors.py has {sorted(ALL_CODES)}"


def test_skill_exit_table_is_exactly_the_run_and_apply_codes():
    codes = exit_codes_in(_read(SKILL))
    assert codes == MAIN_CODES, f"SKILL.md documents {sorted(codes)}; run and apply exit with {sorted(MAIN_CODES)}"


def test_record_doc_exit_table_is_exactly_the_run_and_apply_codes():
    codes = exit_codes_in(_read(RECORD_DOC))
    assert codes == MAIN_CODES, f"decision-record.md documents {sorted(codes)}; a record carries {sorted(MAIN_CODES)}"


@pytest.mark.parametrize("doc,required", [(SKILL, True), (RECORD_DOC, True), (README, False)],
                         ids=["SKILL", "decision-record", "README"])
def test_the_exit_priority_order_is_the_records(doc: Path, required: bool):
    record = _record_module()
    found = priority_sequences(_read(doc))
    if required:
        assert found, f"{doc.relative_to(ROOT)} doesn't give the exit priority order"
    wrong = [seq for seq in found if seq != tuple(record.EXIT_PRIORITY)]
    assert wrong == [], (f"{doc.relative_to(ROOT)} gives the exit priority as {wrong}; "
                         f"record.EXIT_PRIORITY is {tuple(record.EXIT_PRIORITY)} (DESIGN §12.5)")


def test_every_warning_code_is_explained_in_the_skill():
    from pdfintake.record import WARNING_CODES

    skill = _read(SKILL)
    missing = sorted(c for c in WARNING_CODES | {"copy_unlocked", "skipped_no_password"} if not explained(c, skill))
    assert missing == [], f"SKILL.md doesn't explain {missing}"


def _record_module():
    try:
        from pdfintake import record
    except ModuleNotFoundError as exc:
        if exc.name != "pdfintake.record":
            raise
        _not_yet("pdfintake.record isn't written yet")
    return record


@pytest.mark.parametrize("heading,name", [("Warning codes", "WARNING_CODES"), ("Note codes", "NOTE_CODES"),
                                          ("Usage codes", "USAGE_CODES")])
def test_every_record_code_is_documented_in_its_section(heading: str, name: str):
    codes = set(getattr(_record_module(), name))
    body = section(_read(RECORD_DOC), heading)
    assert body, f"decision-record.md has no '{heading}' section"
    listed = codes_in(body)
    assert sorted(codes - listed) == [], f"decision-record.md → {heading} doesn't list {sorted(codes - listed)}"
    assert sorted(listed - codes) == [], f"decision-record.md → {heading} lists {sorted(listed - codes)}, which record.{name} doesn't have"


def test_every_offer_kind_is_documented_with_its_flag_and_options():
    record = _record_module()
    doc = _read(RECORD_DOC)
    assert section(doc, "Decisions") == "", "decision-record.md still has a Decisions section; a run never asks (§12.5)"
    rows = {}
    for table in tables(section(doc, "Offers")):
        if [_plain(c).lower() for c in table[0][:3]] == ["kind", "apply flag", "option ids"]:
            rows.update({_plain(r[0]): r for r in table[1:]})
    assert set(rows) == set(record.OFFER_KINDS), f"the offers table has {sorted(rows)}"
    for kind, row in rows.items():
        assert _plain(row[1]) == record.APPLY_FLAG_FOR[kind], kind
        options = record._KIND_OPTIONS[kind]
        if options is not None:
            assert set(re.findall(r"`([^`]+)`", row[2])) == set(options), kind


def named(word: str, text: str) -> bool:
    """``word`` written in backticks on its own (```done```, ```--lang```)."""
    return f"`{word}`" in text


def test_every_record_vocabulary_word_is_in_the_record_doc():
    record = _record_module()
    from pdfintake import refusals

    doc = _read(RECORD_DOC)
    words = (set(refusals.CODES) | set(record.OUTPUT_KINDS) | set(record.STATUSES) | set(record.STEP_STATUSES)
             | set(record.APPLY_FLAGS) | set(record.OFFER_KINDS) | set(record.PROVENANCES)
             | set(record.LANGUAGE_PLANS))
    missing = sorted(w for w in words if not named(w, doc))
    assert missing == [], f"decision-record.md doesn't name {missing} in backticks"


# --- every refusal and usage code, exit 4 for apply, the stop signals (decided 2026-11-06)

FLOW_SRC = ROOT / "src" / "pdfintake" / "flow.py"


def refused_codes_in(source: str) -> set[str]:
    """Every literal code a module raises ``Refused`` with (``Refused("no_smaller_for_a_copy", …)``)."""
    import ast

    codes = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Refused"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            codes.add(node.args[0].value)
    return codes


def refusal_codes() -> set[str]:
    """What a `run` or `apply` can refuse with: refusals.CODES and the codes flow raises itself."""
    from pdfintake import refusals

    return set(refusals.CODES) | (refused_codes_in(FLOW_SRC.read_text(encoding="utf-8")) if FLOW_SRC.is_file() else set())


def titled_section(text: str, title: str) -> str:
    """The body under the heading titled ``title``, with or without a number ("## 3. Passwords")."""
    m = re.search(rf"^(#+)\s+(?:\d+\.\s+)?{re.escape(title)}\s*$", text, flags=re.M)
    if not m:
        return ""
    level = len(m.group(1))
    end = re.search(rf"^#{{1,{level}}}\s", text[m.end():], flags=re.M)
    return text[m.end(): m.end() + end.start()] if end else text[m.end():]


def exit_rows(text: str) -> dict[int, str]:
    """Each exit-table row by its code, as the row's whole text."""
    rows: dict[int, str] = {}
    for table in tables(text):
        if _plain(table[0][0]).lower() in EXIT_TABLE_HEADERS:
            for row in table[1:]:
                for n in re.findall(r"\d+", _plain(row[0])):
                    rows[int(n)] = " | ".join(row)
    return rows


def test_refused_codes_in_reads_literal_codes():
    src = 'raise Refused("no_smaller_for_a_copy", "m")\nraise Refused(code, m)\nraise Other("x")\n'
    assert refused_codes_in(src) == {"no_smaller_for_a_copy"}


def test_the_refusal_codes_include_damaged_and_applys_own():
    codes = refusal_codes()
    assert {"damaged", "detectors_disagree", "xfa", "xmp_unreadable"} <= codes
    if FLOW_SRC.is_file():
        assert "no_smaller_for_a_copy" in codes


@pytest.mark.parametrize("doc", [SKILL, RECORD_DOC], ids=["SKILL", "decision-record"])
def test_every_refusal_code_is_explained(doc: Path):
    text = _read(doc)
    missing = sorted(c for c in refusal_codes() if f"`{c}`" not in text)
    assert missing == [], f"{doc.relative_to(ROOT)} doesn't explain the refusal codes {missing}"


@pytest.mark.parametrize("doc", [SKILL, RECORD_DOC, README], ids=["SKILL", "decision-record", "README"])
def test_damaged_names_a_failed_read_after_a_repair(doc: Path):
    # A repaired file the walk couldn't read through is damaged too (decided 2026-11-06).
    line = next((ln for ln in _read(doc).splitlines() if "`damaged`" in ln and "repair" in ln), "")
    assert "read through" in line, f"{doc.relative_to(ROOT)}: the damaged explanation misses a failed read after a repair"


def test_the_record_doc_names_which_repairs_are_harmless():
    # Only a broken cross-reference table is a harmless repair; a file with no trailer is damaged
    # even when qpdf rebuilds it (decided 2026-11-06).
    text = _read(RECORD_DOC)
    line = next((ln for ln in text.splitlines() if ln.startswith("- `damaged`")), "")
    assert "`startxref`" in line and "no trailer" in line, line


@pytest.mark.parametrize("doc", [SKILL, RECORD_DOC], ids=["SKILL", "decision-record"])
def test_every_usage_code_is_explained(doc: Path):
    record = _record_module()
    text = _read(doc)
    missing = sorted(c for c in record.USAGE_CODES | {"input_not_file"} if f"`{c}`" not in text)
    assert missing == [], f"{doc.relative_to(ROOT)} doesn't explain the usage codes {missing}"


@pytest.mark.parametrize("doc,heading", [(SKILL, "Why a file was refused (exit 4)"), (RECORD_DOC, "Refusal codes")],
                         ids=["SKILL", "decision-record"])
def test_every_code_that_refuses_is_explained_where_refusals_are(doc: Path, heading: str):
    # The copy reasons have their own place; every code that refuses (exit 4) is explained under the
    # refusals heading, damaged and apply's no_smaller_for_a_copy included.
    from pdfintake import refusals

    body = titled_section(_read(doc), heading)
    assert body, f"{doc.relative_to(ROOT)} has no '{heading}' section"
    missing = sorted(c for c in refusal_codes() - refusals.COPYABLE if f"`{c}`" not in body)
    assert missing == [], f"{doc.relative_to(ROOT)} → {heading} doesn't explain {missing}"


@pytest.mark.parametrize("doc", [SKILL, README, RECORD_DOC], ids=["SKILL", "README", "decision-record"])
def test_every_exit_table_says_apply_can_be_refused(doc: Path):
    rows = exit_rows(_read(doc))
    assert 4 in rows, f"{doc.relative_to(ROOT)} has no exit 4 row"
    assert "apply" in rows[4], f"{doc.relative_to(ROOT)}: exit 4 is for apply --choice smaller too (no_smaller_for_a_copy)"
    assert "no_smaller_for_a_copy" in rows[4], doc.relative_to(ROOT)


@pytest.mark.parametrize("doc", [SKILL, README, RECORD_DOC], ids=["SKILL", "README", "decision-record"])
def test_every_exit_doc_gives_the_stop_signal_exit(doc: Path):
    text = _read(doc)
    for word in ("SIGTERM", "SIGHUP", "128 plus the signal number", "143", "129"):
        assert word in text, f"{doc.relative_to(ROOT)} doesn't say {word!r} (a stopped run exits 128 + the signal)"
    assert not {128, 129, 143} & exit_codes_in(text), "a signal's exit is a sentence, not an exit-table row"


@pytest.mark.parametrize("doc", [SKILL, README, RECORD_DOC], ids=["SKILL", "README", "decision-record"])
def test_input_not_file_is_documented(doc: Path):
    text = _read(doc)
    assert "`input_not_file`" in text
    rows = exit_rows(text)
    assert "input_not_file" in rows.get(2, ""), f"{doc.relative_to(ROOT)}: exit 2 covers input_not_file"
    for what in ("folder", "pipe", "device"):
        assert what in text, (doc.relative_to(ROOT), what)


# --- the skill's password steps and the N25 fixes (folded in from the Wave C review, decided 2026-11-06)

# What each reason passwords.py turns a passwords file down for must be explained with, per doc.
PASSWORD_REASON_WORDS = {
    "missing": ("doesn't exist", "missing"),
    "not_a_file": ("regular file",),
    "unreadable": ("can't be read",),
    "mode": ("other users can read or change", "read or changed by other users"),
    "owner": ("another user",),
    "outside_profile": ("user's folder",),
    "too_large": ("128 KB",),
    "too_many": ("more than 100",),
    "full": ("already holds 100",),
    "not_utf8": ("UTF-8",),
}


def _passwords_module():
    try:
        from pdfintake import passwords
    except ModuleNotFoundError as exc:
        if exc.name != "pdfintake.passwords":
            raise
        _not_yet("pdfintake.passwords isn't written yet")
    return passwords


def test_the_reason_words_cover_every_reason_passwords_reports():
    assert set(PASSWORD_REASON_WORDS) == set(_passwords_module()._REASONS)


def _unsafe_entries(text: str) -> list[str]:
    """Each `passwords_file_unsafe` bullet: from the bullet to the next top-level bullet or heading."""
    return [m.group(0) for m in re.finditer(r"^- `passwords_file_unsafe`:.*?(?=^- |^#|\Z)", text, flags=re.M | re.S)]


def _uncovered(entry: str) -> list[str]:
    return sorted(r for r, words in PASSWORD_REASON_WORDS.items() if not any(w in entry for w in words))


@pytest.mark.parametrize("doc", [SKILL, RECORD_DOC], ids=["SKILL", "decision-record"])
def test_every_passwords_file_reason_is_explained(doc: Path):
    # The entry that explains the code (the skill also has a short how-to bullet for it) covers every reason.
    entries = _unsafe_entries(_read(doc))
    assert entries, f"{doc.relative_to(ROOT)} has no `passwords_file_unsafe` entry"
    entry = min(entries, key=lambda e: len(_uncovered(e)))
    assert _uncovered(entry) == [], f"{doc.relative_to(ROOT)}'s passwords_file_unsafe entry doesn't cover {_uncovered(entry)}"
    assert "names the file and the" in entry and "reason" in entry


def test_the_skill_saves_a_password_with_passwords_add_from_file():
    skill = _read(SKILL)
    assert "passwords add --from-file" in skill
    steps = titled_section(skill, "Passwords")
    assert steps, "SKILL.md has no Passwords section"
    tool, chmod, add = steps.find("with your file tool"), steps.find("chmod 600 'FILE'"), steps.find("passwords add --from-file")
    assert -1 < tool < chmod < add, "the file tool writes it, chmod 600 makes it private, then passwords add saves it"
    one_run = steps.find("pass `--passwords-file 'FILE'`")
    assert one_run > chmod, "the don't-keep path makes the file private before passing --passwords-file"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_skills_password_steps_give_a_file_pdfintake_reads(tmp_path: Path):
    # The file tool writes 0644; pdfintake refuses that, and takes it after chmod 600 (N25-1).
    passwords = _passwords_module()
    from pdfintake.errors import UsageError

    folder = tmp_path / "pw"
    folder.mkdir(mode=0o700)
    f = folder / "pw.txt"
    old = os.umask(0o022)
    try:
        f.write_text("made up docs phrase\n", encoding="utf-8")
    finally:
        os.umask(old)
    with pytest.raises(UsageError) as err:
        passwords.candidates(f)
    assert err.value.code == "passwords_file_unsafe"
    os.chmod(f, 0o600)
    assert len(passwords.candidates(f)) == 1


def test_the_skill_merges_images_by_default():
    flags = titled_section(_read(SKILL), "Turn the request into flags")
    assert flags, "SKILL.md has no 'Turn the request into flags' section"
    assert "Otherwise pass `--separate`" not in flags
    line = next((ln for ln in flags.splitlines() if "Several images" in ln), "")
    assert "one PDF" in line and "default" in line and "--separate" in line


def test_the_skill_chmods_only_a_path_it_chose():
    body = titled_section(_read(SKILL), "Read every record")
    assert body, "SKILL.md has no 'Read every record' section"
    assert "chmod 600 'FILE'" not in body
    assert "rather than copying it from the message" in body


def test_the_notes_with_their_own_lines_are_exempt_in_both_places():
    skill = _read(SKILL)
    own = ("`copy_unlocked`", "`signature_not_carried`", "`skipped_no_password`")
    summary = titled_section(skill, "One summary after the whole batch")
    closing = next((p for p in summary.split("\n\n") if "closing line" in p), "")
    assert closing and all(code in closing for code in own), "the summary's closing-line rule names the three notes"
    notes = titled_section(skill, "Notes")
    intro = notes.strip().split("\n\n")[0]
    assert all(code in intro for code in own), "the Notes heading exempts the same three notes"
    assert "Notes other than `copy_unlocked` go" not in skill


EXPORT_REFUSAL_WORDS = {
    "certified": "certified", "usage_rights": "usage-rights", "timestamped": "timestamped", "signed": "signed",
    "encrypted": "password-protected", "xfa": "XFA", "provenance_counterparty": "other side",
    "provenance_evidence": "evidence", "damaged": "`damaged`",
}


def test_the_skill_lists_everything_export_refuses():
    from pdfintake.pdf import export

    assert set(EXPORT_REFUSAL_WORDS) == set(export.REFUSAL_CODES)
    body = titled_section(_read(SKILL), "Export only when asked")
    assert body, "SKILL.md has no 'Export only when asked' section"
    missing = sorted(code for code, words in EXPORT_REFUSAL_WORDS.items() if words not in body)
    assert missing == [], f"SKILL.md's export section doesn't mention {missing}"


def _hosts() -> list[str]:
    hosts = []
    try:
        from pdfintake.langpacks.index import ALLOWED_HOSTS
    except ModuleNotFoundError as exc:
        if exc.name != "pdfintake.langpacks.index":
            raise
        _not_yet("pdfintake.langpacks.index isn't written yet")
    hosts += list(ALLOWED_HOSTS)
    host = pin_host()
    assert host, "pyproject.toml needs exactly one 'pdfintake-data-en @ URL#sha256=HASH' dependency"
    if host in PLACEHOLDER_HOSTS:
        if _strict():
            pytest.fail(f"the pyproject pin still names {host}")
    else:
        hosts.append(host)
    return sorted(set(hosts))


@pytest.mark.parametrize("doc", [PLUGIN_README, PRIVACY], ids=["plugin-README", "PRIVACY"])
def test_every_download_host_is_named(doc: Path):
    text = _read(doc)
    hosts = _hosts()
    missing = [h for h in hosts if h not in text]
    assert missing == [], f"{doc.relative_to(ROOT)} doesn't name {missing}"


def test_every_runtime_dependency_has_a_third_party_row():
    rows = third_party_names(_read(THIRD_PARTY))
    deps = sorted({_dist_name(d) for d in _pyproject()["project"]["dependencies"]})
    missing = [d for d in deps if d not in rows]
    assert missing == [], f"THIRD-PARTY.md has no row for {missing}"


# --- the steps[] details and PRIVACY.md's work/ list follow the code

FLOW = ROOT / "src" / "pdfintake" / "flow.py"
CACHED_STEPS = ("detect", "orientation")
# The modules that make the work/ folders and the cache in them (flow.py's _cache_folder and _DetectionCache,
# ocr/pipeline.py's cache records).
WORK_CODE = ("ocr/pipeline.py", "flow.py", "pdf/fontmap.py")
# Where PRIVACY.md accounts for a file that the cache doesn't keep, by the sentence that says so.
NOT_CACHED = {
    "run": "A run's own working files are deleted when the run ends",   # in work/, for one run
    "staged": "a staged smaller copy in `staging/`",
    "output": "writes the new PDF there",                               # beside the input
}
# Every file name WORK_CODE spells out (a string, or an f-string with each field as {}), by where PRIVACY.md
# accounts for it: a phrase of exactly one item of its work/ list for a file the cache keeps, or a NOT_CACHED key.
WORK_FILE_NAMES: dict[tuple[str, str], str] = {
    ("ocr/pipeline.py", "ocr-{:05d}.png"): "pictures of its pages",
    ("ocr/pipeline.py", "detection.json"): "what detection found",
    ("ocr/pipeline.py", "orientation.json"): "orientation checks",
    ("flow.py", "assembled.json"): "the date it was first built",
    ("flow.py", ".pdfintake-run.lock"): "run",
    ("flow.py", "input.pdf"): "run",
    ("flow.py", "decrypted.pdf"): "run",
    ("flow.py", "assembled-raw.pdf"): "run",
    ("flow.py", "assembled.pdf"): "run",
    ("flow.py", "built.pdf"): "run",
    ("flow.py", "final.pdf"): "run",
    ("flow.py", "stripped.pdf"): "run",
    ("flow.py", "upright.pdf"): "run",
    ("flow.py", "fontmapped.pdf"): "run",
    ("pdf/fontmap.py", "probe-{}.pdf"): "run",
    ("flow.py", "candidate.pdf"): "staged",
    ("flow.py", "{}{}.pdf"): "output",
}
FILE_NAME = re.compile(r"\S*\.[a-z]{2,5}")


def step_details(source: str, step: str) -> list[set[str]]:
    """The keyword names of each ``_step(d, "<step>", status, ...)`` call in ``source``, read with ast."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_step"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == step):
            assert all(k.arg is not None for k in node.keywords), f"_step(d, {step!r}, ...) passes **detail"
            found.append({str(k.arg) for k in node.keywords})
    return found


def step_bullet_names(doc: str, step: str) -> set[str]:
    """The backticked names in the ``steps[]`` details bullet for ``step`` (``- `detect`: ...``)."""
    body = section(doc, "`steps[]` details")
    m = re.search(rf"^- `{re.escape(step)}`:(.*?)(?=^- |^\s*$|\Z)", body, flags=re.M | re.S)
    return set(re.findall(r"`([^`\n]+)`", m.group(1))) if m else set()


def file_names(source: str) -> set[str]:
    """Every string in ``source`` that is a file name or a suffix (no whitespace, ending in a dot and 2-5
    lowercase letters): constants and inline names alike (``detection.json``, ``work / "input.pdf"``,
    ``".tmp"``), and f-strings with each field as ``{}`` (``f"{stem}{label}.pdf"`` is ``{}{}.pdf``).
    Docstrings don't count."""
    tree = ast.parse(source)
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
        elif isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)       # an f-string's parts count only as the whole
    found = set()
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
        else:
            continue
        if FILE_NAME.fullmatch(text):
            found.add(text)
    return found


def work_cache_list(privacy: str) -> list[str]:
    """The items nested under PRIVACY.md's bullet that names `work/`, each joined onto one line."""
    lines = privacy.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^\s*- .*`work/`", ln)), None)
    if start is None:
        return []
    indent = len(lines[start]) - len(lines[start].lstrip())
    items: list[str] = []
    for ln in lines[start + 1:]:
        if not ln.strip() or len(ln) - len(ln.lstrip()) <= indent:
            break
        if ln.lstrip().startswith("- "):
            items.append(ln.strip()[2:])
        elif items:
            items[-1] += " " + ln.strip()
    return items


@pytest.mark.parametrize("step", CACHED_STEPS)
def test_the_step_details_name_exactly_what_flow_records(step: str):
    record = _record_module()
    calls = step_details(FLOW.read_text(encoding="utf-8"), step)
    assert len(calls) == 1, f"flow.py has {len(calls)} _step(d, {step!r}, ...) calls"
    names = step_bullet_names(_read(RECORD_DOC), step)
    assert names, f"decision-record.md → `steps[]` details has no `{step}` bullet"
    codes = set(record.WARNING_CODES) | set(record.NOTE_CODES)
    assert sorted(calls[0] - names) == [], f"decision-record.md's `{step}` bullet doesn't name {sorted(calls[0] - names)}"
    extra = names - calls[0] - codes
    assert sorted(extra) == [], f"decision-record.md's `{step}` bullet names {sorted(extra)}, which flow doesn't record"


def test_privacy_lists_every_file_the_cache_keeps():
    found = {(rel, name) for rel in WORK_CODE
             for name in file_names((ROOT / "src" / "pdfintake" / rel).read_text(encoding="utf-8"))}
    new, gone = sorted(found - set(WORK_FILE_NAMES)), sorted(set(WORK_FILE_NAMES) - found)
    assert new == [] and gone == [], (
        f"file names added {new}, removed {gone}: say in WORK_FILE_NAMES where PRIVACY.md accounts for each "
        "(a cache file needs its own item in PRIVACY.md's work/ list)")
    text = _read(PRIVACY)
    items = work_cache_list(text)
    phrases = sorted({p for p in WORK_FILE_NAMES.values() if p not in NOT_CACHED})
    assert len(items) == len(phrases), f"PRIVACY.md's work/ list has {len(items)} items: {items}"
    for phrase in phrases:
        assert sum(phrase in item for item in items) == 1, f"{phrase!r} isn't in exactly one item: {items}"
    for where in sorted(set(WORK_FILE_NAMES.values()) & set(NOT_CACHED)):
        assert NOT_CACHED[where] in text, f"PRIVACY.md doesn't say {NOT_CACHED[where]!r}"


# --- the parsers do what they claim

def test_exit_codes_from_a_table():
    text = ("| Code | Name | run |\n|---|---|---|\n| 0 | OK | done |\n| `1` | FAILED | x |\n| **10** | DECISION | y |\n"
            "\n| Other | x |\n|---|---|\n| 99 | not an exit table |\n")
    assert exit_codes_in(text) == {0, 1, 10}


def test_exit_codes_from_bold_bullets():
    text = "4. Act on its exit:\n   - **0.** Report.\n   - **10.** Ask.\n   - **1, 2 or 3.** Report the error.\n"
    assert exit_codes_in(text) == {0, 1, 2, 3, 10}


def test_exit_codes_from_colon_bullets_and_outcome_tables():
    assert exit_codes_in("- **0:** done.\n- **10:** a decision is waiting.\n") == {0, 10}
    assert exit_codes_in("| Outcome | Meaning |\n|---|---|\n| 0 | done |\n| 10 | waiting |\n") == {0, 10}


def test_priority_sequences():
    assert priority_sequences("the most important code among them (2, then 3, 1, 4, 5, 0).") == [(2, 3, 1, 4, 5, 0)]
    assert priority_sequences("exits with the highest-priority code, in the order 2, 3, 1, 4, 5, 0.") == [(2, 3, 1, 4, 5, 0)]
    assert priority_sequences("The exit priority is 2 > 3 > 1 > 4 > 5 > 0.") == [(2, 3, 1, 4, 5, 0)]
    assert priority_sequences("Python 3.11, 3.12 and 3.13 in that order.") == []
    assert priority_sequences("Pages 1, 5, 10 were read.") == []


def test_a_table_wins_over_bullets():
    assert exit_codes_in("| Exit | x |\n|:-:|---|\n| 5 | stale |\n\n- **7.** kit\n") == {5}


def test_section_and_codes_in():
    text = ("# Page\n\n## Warning codes\n\n| Group | Codes |\n|---|---|\n| a | `layer_shrank`, `pack_broken` |\n\n"
            "### Detail\n\n`clip_text_kept` and `WARNING_MESSAGES` in `record.py`, flag `--choice`.\n\n"
            "## Note codes\n\n- `input_unchanged`\n")
    body = section(text, "Warning codes")
    assert codes_in(body) == {"layer_shrank", "pack_broken", "clip_text_kept"}
    assert codes_in(section(text, "Note codes")) == {"input_unchanged"}
    assert section(text, "Usage codes") == ""


def test_explained_needs_the_whole_code():
    assert explained("layer_shrank", "the `layer_shrank` warning")
    assert not explained("layer_shrank", "layer_shrank_more")
    assert not explained("pack_broken", "pack broken")


def test_third_party_names_normalise():
    text = ("| Package | Licence |\n|---|---|\n| pikepdf 10.17 | MPL-2.0 |\n| [Pillow](https://example.com) 13 | MIT-CMU |\n"
            "| `pyobjc-framework-Vision` ≥ 13 | MIT |\n| **py3langid** ==0.4.0 | BSD |\n")
    assert {"pikepdf", "pillow", "pyobjc-framework-vision", "py3langid"} <= third_party_names(text)
    assert _dist_name("pyobjc-framework-Vision>=13; sys_platform == 'darwin'") == "pyobjc-framework-vision"
    assert _dist_name("pdfintake-data-en @ https://example.com/x.whl") == "pdfintake-data-en"
    assert _dist_name("fonttools>=4.68,<5") == "fonttools"


def test_the_constants_cover_both_tables():
    assert MAIN_CODES == {0, 1, 2, 3, 4, 5}
    assert KIT_CODES == {6, 7, 8}


# --- the question-flow scanner catches what it's for, and lets "never" through

OLD_FLOWS = [
    "In Claude Code, use AskUserQuestion, one question per decision.",
    "No AskUserQuestion here.",
    "When it needs a choice, `run` stops with exit 10 and a decision record.",
    "The first run without `--provenance` writes nothing and returns exit 10 with the record.",
    "- **10.** Act on the record.",
    "- **1, 2 or 10.** Report it.",
    "Ask the record's question with its options, the recommended one first.",
    "Ask the user which language, then run apply.",
    "Otherwise Claude asks you first, and names the download size.",
    "### Questions pdfintake asks",
    "Its skill tells Claude how to ask its questions.",
    "It hands back a question in a JSON record.",
    "pdfintake apply r-20261105-1a2b3c4d --choice smaller     # answer a question a run asked",
    "Show the options as a short numbered list.",
    "Then wait for the reply.",
    "Report each question and its options, then stop.",
    "Before I start: did any of these come from a landlord?",
    "Run `pdfintake apply <id> <decision.apply_flag> <option id>`.",
    "A decision of kind `password` means the file can't be opened without its password.",
    "Ask it like any other decision.",
    "The same `password` decision comes back.",
    "Then run `pdfintake apply <id> --provenance counterparty --json`.",
    "Read it from standard input (`--password-stdin`).",
    "If it comes back with the note `password_wrong`, try again.",
    "Write the password to a 0600 temp file and pass `--password-file 'FILE'` to apply.",
    "`run` takes `--password-file PATH` for a password the user gave up front.",
    "`--fetch ask` asks first every time.",
    "Use `--fetch auto|ask|never` and `--smaller ask|never`.",
    "Run `--choice decline` to leave it.",
    "`keep` deletes the candidate: `pdfintake apply <id> --choice keep`.",
    "The CLI uses `EXIT_DECISION` for that.",
    "| Code | Meaning |\n|---|---|\n| 0 | done |\n| 10 | a question |\n",
    # exit 10 in the exit-priority lists the committed docs carried (N26-R1)
    "With several files, the process exit is the most important code among the records (2, then 3, 1, 5, 10, 4, 0).",
    "With several records, the process exits with the highest-priority code among them, in the order 2, 3, 1, 5, 10, 4, 0.",
    "The exit priority is 2 > 3 > 1 > 10 > 4 > 0.",
    # a negated condition doesn't cover the asking after it (N26-R2)
    "If the file doesn't open ask the user for its password.",
    "If no saved password opens it ask the user for one.",
    "If nothing opens it ask the user for the password.",
    "Without a password ask the user for it.",
    "If the file can't be opened `run` exits 10 with a password decision.",
    "- If no saved password opens a file ask the user for it before going on.",
    # the ordinary ways of telling Claude to ask (N26-R3)
    "Ask for the password, then run apply.",
    "Ask whether these came from a landlord before running.",
    "Ask which language the pages are in.",
    "Claude asks which language the pages are in.",
    "Prompt the user for the password.",
    "Then wait for the user to reply.",
    "Read the record's `decision` field and show its question.",
    "A record with status `decision` holds the question.",
    "`run` returns 10 when it needs an answer.",
    "- **10:** a decision is waiting.",
    "| Outcome | Meaning |\n|---|---|\n| 0 | done |\n| 10 | waiting |\n",
    "lossless shrinking, and a smaller copy only when you agree.",
]

ALLOWED = [
    "**A run never waits for an answer.** pdfintake doesn't ask anything during a run, and `run` never exits 10.",
    "Exit 10 is retired.",
    "The old exit 10 was removed.",
    "Provenance and passwords are never offers or questions.",
    "pdfintake chose a language for some pages without asking, and another pack could read them again.",
    "The skill never asks about them during or before a batch.",
    "The skill doesn't ask the user anything during a run.",
    "It works through a pile of files without stopping to ask you anything.",
    "Read every record. Nothing in a record waits for an answer.",
    "Ask questions in [Discussions](https://github.com/RazorII/pdfintake/discussions).",
    "Questions, option labels and warnings are worded in core.",
    "A `language` offer reads the page again; run `pdfintake apply <id> --lang bs --json`.",
    "pdfintake apply r-20261105-1a2b3c4d --choice smaller",
    "Pass `--provenance counterparty` to `pdfintake run` when the user says so.",
    "A page of 10 lines exits nothing; exit 1 is a failure and exit 100 isn't a code.",
    "The language warnings that replace what used to be a question:",
    "Stop and ask the human which characters matter.",
    "| Code | Meaning |\n|---|---|\n| 0 | done |\n| 5 | stale |\n",
    "With several files, the process exit is the most important code among the records (2, then 3, 1, 4, 5, 0).",
    "If the file is damaged, `run` never exits 10.",
    "If no saved password opens it, the file is skipped and `run` never exits 10.",
    "pdfintake never asks for a password.",
    "Pass `--passwords-file 'FILE'` to that one run instead.",
    "`--passwords-file PATH` points a run at another saved list.",
    "A smaller copy is made only when you ask for it (`--smaller`).",
    "GitHub sees your IP address and the file you asked for.",
    "If the person later asks for the smaller copy, run apply.",
    "Only when the user asks for smaller files.",
    "The registry imports the module only when a pack asks for the engine.",
    "12 files could be made about 60% smaller; ask if you want smaller copies.",
    "pdfintake never prompts and never waits for an answer.",
    "Python 3.11, 3.12 and 3.13 are tested, in that order.",
]


@pytest.mark.parametrize("text", OLD_FLOWS)
def test_the_scanner_catches_a_run_that_asks(text: str):
    assert question_flow_problems(text)


@pytest.mark.parametrize("text", ALLOWED)
def test_the_scanner_lets_through_what_says_it_does_not_ask(text: str):
    assert question_flow_problems(text) == []


def test_the_scanner_names_the_line():
    problems = question_flow_problems("# Title\n\nFine.\n\nUse AskUserQuestion.\n")
    assert problems == ["line 5: AskUserQuestion: Use AskUserQuestion."]


def test_step_details_and_their_bullet():
    src = 'def f(d):\n    _step(d, "detect", "ok", pages=1, cached=False)\n    _step(d, "other", "ok", x=1)\n'
    assert step_details(src, "detect") == [{"pages", "cached"}]
    assert step_details(src, "orientation") == []
    doc = ("### `steps[]` details\n\nIntro with `cached`.\n\n- `detect`: `pages` and\n  `cached`, see `layer_shrank`.\n"
           "- `other`: `x`.\n\n## Next\n\n- `detect`: `elsewhere`\n")
    assert step_bullet_names(doc, "detect") == {"pages", "cached", "layer_shrank"}
    assert step_bullet_names(doc, "orientation") == set()


def test_work_cache_list_and_file_names():
    text = ("- **Kept.** In it:\n  - working files in `work/`. What stays:\n    - pictures;\n    - what detection\n"
            "      found;\n  - a staged copy in `staging/`.\n")
    assert work_cache_list(text) == ["pictures;", "what detection found;"]
    src = ('"""module.txt"""\nA_FILE = "a.json"\nB = " (copy).pdf"\nC_NAME: str = "ocr-{:05d}.png"\nD = "%Y-%m-%d"\n'
           'E = "osd"\n\n\ndef keep(entry, n, stem):\n    """notes.md"""\n    write(entry / "reads.json")\n'
           '    write(entry / f"votes-{n:03d}.json", entry / f"{stem} (copy).pdf", f"{n}")\n'
           '    return entry.with_suffix(".tmp"), "%s.lock" % stem, "Run it again."\n')
    assert file_names(src) == {"a.json", "ocr-{:05d}.png", "reads.json", "votes-{}.json", ".tmp", "%s.lock"}
