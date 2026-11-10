"""The ``pdfintake`` command: argument parsing, the JSON a skill reads, and the exit code.

Commands (DESIGN §4.1 as changed by §10-§12)::

    pdfintake run INPUT... [--paths-from FILE] [--provenance ordinary|counterparty|evidence]
                  [-o OUT | --outdir DIR] [--lang CODE[,CODE]|none] [--force-lang CODE[,CODE]|none]
                  [--fetch auto|never] [--smaller] [--separate] [--redo-ocr] [--keep-existing-layer]
                  [--overwrite] [--plan] [--workers N] [--thorough | --no-thorough]
                  [--passwords-file FILE] [--json]
    pdfintake apply RECORD-ID (--choice smaller|thorough | --lang CODE) [--passwords-file FILE] [--json]
    pdfintake doctor [--canary] [--check-network] [--json]
    pdfintake thorough status|enable|disable [--remove]|decline [--json]
    pdfintake packs list|show ID|fetch ID|verify [ID]|remove ID|path [--json]
    pdfintake record show RECORD-ID [--json]
    pdfintake export clean INPUT -o OUT --provenance VALUE [--strip-dates] [--overwrite] [--json]
    pdfintake export flatten INPUT -o OUT --provenance VALUE [--overwrite] [--json]
    pdfintake passwords add --from-file FILE [--passwords-file FILE] [--json]
    pdfintake kit ...                       (needs pdfintake[kit])
    pdfintake --version

**Nothing is ever asked** (DESIGN §12, decided 2026-11-05). No command prompts, reads standard
input or waits for an answer, and no command exits 10: every choice is a flag or a default, and
what a person could still do is an offer in the record, acted on later with ``apply``. A list of
input paths comes only from ``--paths-from FILE``, a regular file the caller wrote; standard input
is never read, even as ``-``.

**Output.** ``run`` and ``apply`` print ``{"exit": N, "records": [...]}`` with ``--json`` (the
records ``flow`` saved, unchanged); an invocation that can't run at all adds ``"error": {code,
message}`` with no records. The other commands print one JSON object with ``exit`` and their result,
or ``exit`` and ``error``. Without ``--json`` a short human summary goes to standard output and
errors to standard error. Output is UTF-8 everywhere (``fsutil.utf8_stdio``), so a file name in a
script the console's code page lacks can't crash a run piped through cp1252.

**Exit codes** are the ``EXIT_*`` constants in ``errors.py``: 0 done, 1 failed, 2 usage, 3 setup,
4 refused, 5 stale; with several records the highest priority wins (2 > 3 > 1 > 4 > 5 > 0). The kit
adds 6, 7 and 8. A bad argument is exit 2, ``--help`` and ``--version`` exit 0, an interrupt 130.

**Passwords.** The only way in is ``--passwords-file`` (a saved list, one per line) or the default
list in the config folder. ``passwords add --from-file`` appends to that list. A password is never
an argument value, never read from the environment or standard input, and never printed.

**Writes.** This module writes nothing itself. ``run`` and ``apply`` write through ``flow``;
``export`` builds into the cache and hands the file to ``flow.commit``, the one writer into a
user's folder (tests/test_write_paths.py checks both).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .errors import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_SETUP,
    EXIT_USAGE,
    PdfIntakeError,
    Refused,
    UsageError,
)

PROG = "pdfintake"
PROVENANCE_CHOICES = ("ordinary", "counterparty", "evidence")
FETCH_CHOICES = ("auto", "never")
MAX_PATHS_FROM_BYTES = 16 * 1024 * 1024   # a list of a hundred thousand long paths fits with room to spare
KIT_EXTRA_MODULES = ("uharfbuzz",)         # what `pip install pdfintake[kit]` adds (pyproject [kit])
KIT_EXTRA_MESSAGE = ("The add-a-language kit needs the [kit] extra, which isn't installed: install pdfintake[kit] "
                     "(for example: uv tool install --force 'pdfintake[kit] @ git+https://github.com/RazorII/pdfintake'), "
                     "or run scripts/dev-install.sh in a clone.")
EXIT_INTERRUPTED = 130


# --------------------------------------------------------------------------- the parser

class _ArgError(Exception):
    """argparse's complaint about the arguments, turned into exit 2 instead of a SystemExit."""

    def __init__(self, message: str, prog: str) -> None:
        super().__init__(message)
        self.message = message
        self.prog = prog


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser that raises on a bad argument and never guesses an abbreviation.

    Abbreviations are off so ``--password`` can't quietly mean ``--passwords-file``, and a later
    option can't change what an old command line meant. Subparsers inherit this class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> Any:  # type: ignore[override]
        raise _ArgError(message, self.prog)


def _lang_codes(value: str | None, flag: str) -> tuple[str, ...]:
    """``de,fr`` -> ``("de", "fr")``; ``flow`` checks each code.

    A value with no code in it (``''``, ``' '``, ``','``, as an empty variable in a caller's script
    gives) is exit 2, never read as "no language given" (decided 2026-11-05).
    """
    if value is None:
        return ()
    codes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not codes:
        raise UsageError("bad_language_code", f"{flag} takes language codes such as en or fr, and was given none.",
                         flag=flag)
    return codes


def build_parser() -> argparse.ArgumentParser:
    """The whole command tree (the kit group included when its module imports)."""
    parser = _Parser(prog=PROG, description="Turn scans, phone photos and PDFs into small, searchable PDFs. "
                                            "Nothing is ever asked: every choice is a flag or a default.")
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", parser_class=_Parser)

    # run ------------------------------------------------------------------------------------------
    p = sub.add_parser("run", help="make searchable PDFs; writes beside the inputs or into --outdir",
                       description="Make a searchable (or compact) PDF of each input. PDFs are never merged; "
                                   "images become one PDF in the order given (one each with --separate). "
                                   "Nothing is asked: what is left to decide is listed as offers in each record.")
    p.add_argument("inputs", nargs="*", metavar="INPUT", help="PDF or image files")
    p.add_argument("--paths-from", action="append", default=[], metavar="FILE",
                   help="read more input paths from FILE, one UTF-8 path per line (never standard input)")
    p.add_argument("--provenance", choices=PROVENANCE_CHOICES, default=None,
                   help="counterparty or evidence: the original stays as it is and a searchable copy is written "
                        "beside it (default ordinary)")
    where = p.add_mutually_exclusive_group()
    where.add_argument("-o", "--out", metavar="OUT", help="the output file (one output document only)")
    where.add_argument("--outdir", metavar="DIR", help="write every output into DIR")
    p.add_argument("--lang", metavar="CODE[,CODE]|none", help="the language of pages detection was unsure about")
    p.add_argument("--force-lang", metavar="CODE[,CODE]|none", help="the language of every page that is read")
    p.add_argument("--fetch", choices=FETCH_CHOICES, default="auto",
                   help="auto: download the language packs a file needs (default); never: stay offline")
    p.add_argument("--smaller", action="store_true",
                   help="also write '<name> (smaller).pdf' wherever a smaller copy saves at least 30%% and "
                        "still looks the same")
    p.add_argument("--separate", action="store_true", help="one PDF per image instead of one for all of them")
    p.add_argument("--redo-ocr", action="store_true", help="replace a text layer pdfintake added before")
    p.add_argument("--keep-existing-layer", action="store_true",
                   help="leave pages that carry another tool's invisible text layer as they are")
    p.add_argument("--overwrite", action="store_true", help="replace an output that already exists")
    p.add_argument("--plan", action="store_true", help="say what would be done and downloaded; write nothing")
    p.add_argument("--workers", type=int, default=None, metavar="N", help="engine reads at once (1-16)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--thorough", dest="thorough", action="store_const", const=True, default=None,
                      help="also read every OCR page with dots.mocr (Apple silicon; see 'pdfintake thorough')")
    mode.add_argument("--no-thorough", dest="thorough", action="store_const", const=False,
                      help="don't, even when thorough mode is on")
    p.add_argument("--passwords-file", metavar="FILE",
                   help="the saved passwords to try on a password-protected file (default: the config folder's "
                        "'passwords' file, when it exists)")
    p.add_argument("--json", action="store_true", help="print the records as JSON")
    p.set_defaults(func=_cmd_run)
    parser.set_defaults(_run_parser=p)

    # apply ----------------------------------------------------------------------------------------
    p = sub.add_parser("apply", help="act on an offer of an earlier run",
                       description="Act on an offer in a record, only when a person asks for it: "
                                   "--choice smaller writes the smaller copy, --choice thorough reads the "
                                   "document again in thorough mode, --lang CODE reads the pages again.")
    p.add_argument("record_id", metavar="RECORD-ID")
    answer = p.add_mutually_exclusive_group(required=True)
    answer.add_argument("--choice", choices=("smaller", "thorough"),
                        help="smaller: write the smaller copy the record offers; thorough: read the document "
                             "again in thorough mode (set it up first with 'pdfintake thorough enable')")
    answer.add_argument("--lang", metavar="CODE", help="read the pages again with this language")
    p.add_argument("--passwords-file", metavar="FILE", help="the saved passwords, for a password-protected input")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_apply)

    # doctor ---------------------------------------------------------------------------------------
    p = sub.add_parser("doctor", help="check that pdfintake can run here, and say what is missing")
    p.add_argument("--canary", action="store_true", help="also read every installed pack's test image")
    p.add_argument("--check-network", action="store_true", help="also check that the pack host answers")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_doctor)

    # packs ----------------------------------------------------------------------------------------
    p = sub.add_parser("packs", help="language packs: list, show, fetch, verify, remove, path")
    psub = p.add_subparsers(dest="packs_command", required=True, metavar="ACTION", parser_class=_Parser)
    for name, helptext, arg in (("list", "every pack this version knows, and which are installed", None),
                                ("show", "one pack's details", "required"),
                                ("fetch", "download and check one pack", "required"),
                                ("verify", "check the installed packs (or one)", "optional"),
                                ("remove", "remove a downloaded pack", "required"),
                                ("path", "where packs and the cache live", None)):
        q = psub.add_parser(name, help=helptext)
        if arg == "required":
            q.add_argument("pack_id", metavar="ID")
        elif arg == "optional":
            q.add_argument("pack_id", metavar="ID", nargs="?")
        q.add_argument("--json", action="store_true")
        q.set_defaults(func=_cmd_packs)

    # record ---------------------------------------------------------------------------------------
    p = sub.add_parser("record", help="show a saved record")
    rsub = p.add_subparsers(dest="record_command", required=True, metavar="ACTION", parser_class=_Parser)
    q = rsub.add_parser("show", help="print a saved record")
    q.add_argument("record_id", metavar="RECORD-ID")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=_cmd_record_show)

    # export ---------------------------------------------------------------------------------------
    p = sub.add_parser("export", help="a copy to share (clean) or a non-editable copy (flatten)")
    esub = p.add_subparsers(dest="export_command", required=True, metavar="KIND", parser_class=_Parser)
    for name, helptext in (("clean", "a copy to share: identifying metadata removed, the text layer kept"),
                           ("flatten", "a non-editable copy: form fields and annotations baked into the page")):
        q = esub.add_parser(name, help=helptext)
        q.add_argument("input", metavar="INPUT")
        q.add_argument("-o", "--out", required=True, metavar="OUT")
        q.add_argument("--provenance", choices=PROVENANCE_CHOICES, required=True)
        if name == "clean":
            q.add_argument("--strip-dates", action="store_true", help="also remove the creation and change dates")
        q.add_argument("--overwrite", action="store_true", help="replace OUT when it already exists")
        q.add_argument("--json", action="store_true")
        q.set_defaults(func=_cmd_export)

    # thorough -------------------------------------------------------------------------------------
    p = sub.add_parser("thorough", help="thorough mode: dots.mocr beside the regular readers (Apple silicon)",
                       description="Thorough mode also reads every OCR page with dots.mocr, a vision-language "
                                   "OCR model, on a Mac with Apple silicon. It needs the pdfintake[thorough] "
                                   "libraries and a one-time model download, and is off until enabled.")
    tsub = p.add_subparsers(dest="thorough_command", required=True, metavar="ACTION", parser_class=_Parser)
    for name, helptext in (("status", "whether this machine can run it, what is installed, and the setting"),
                           ("enable", "download and check the model, read a test page, and turn the mode on"),
                           ("disable", "turn the mode off (the model stays unless --remove)"),
                           ("decline", "record that you don't want it (enable still works later)")):
        q = tsub.add_parser(name, help=helptext)
        if name == "disable":
            q.add_argument("--remove", action="store_true", help="also delete the downloaded model")
        q.add_argument("--json", action="store_true")
        q.set_defaults(func=_cmd_thorough)

    # passwords ------------------------------------------------------------------------------------
    p = sub.add_parser("passwords", help="the saved passwords list")
    wsub = p.add_subparsers(dest="passwords_command", required=True, metavar="ACTION", parser_class=_Parser)
    q = wsub.add_parser("add", help="append the password(s) in a file to the saved list",
                        description="Append each line of FILE to the saved passwords list, creating it as a private "
                                    "file when needed. FILE is read once and left as it is; delete it afterwards.")
    q.add_argument("--from-file", required=True, metavar="FILE", help="one password per line")
    q.add_argument("--passwords-file", metavar="FILE",
                   help="the list to add to (default: the config folder's 'passwords' file)")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=_cmd_passwords_add)

    # kit ------------------------------------------------------------------------------------------
    _register_kit(sub)
    return parser


def _register_kit(sub: Any) -> None:
    """The kit group from ``kit.cli.register``; a stand-in that names the extra when that can't load."""
    try:
        from .kit import cli as kit_cli
    except ImportError:
        kit_cli = None
    if kit_cli is not None:
        kit_cli.register(sub)
        kit_parser = sub.choices.get("kit")
        if kit_parser is not None:
            for parser in _all_parsers(kit_parser):
                if parser.get_default("func") is not None:
                    parser.set_defaults(func=_kit_guarded(parser.get_default("func")))
        return
    p = sub.add_parser("kit", help="build and check a language pack (needs pdfintake[kit])")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=lambda args: _kit_missing(args))


def _all_parsers(parser: argparse.ArgumentParser) -> list[argparse.ArgumentParser]:
    out = [parser]
    for action in parser._actions:                       # noqa: SLF001 - argparse has no public walk
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for child in action.choices.values():
                out.extend(_all_parsers(child))
    return out


def _kit_extra_present() -> bool:
    """Whether every module of the [kit] extra can be imported (found, not imported)."""
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in KIT_EXTRA_MODULES)


def _kit_guarded(func: Any) -> Any:
    def run(args: argparse.Namespace) -> int:
        if not _kit_extra_present():
            return _kit_missing(args)
        return int(func(args))

    return run


def _kit_missing(args: argparse.Namespace) -> int:
    """Exit 3 with one JSON object, in the kit's own shape, naming the extra to install."""
    name = getattr(args, "kit_command", None)
    _emit({"ok": False, "command": f"kit {name}" if name else "kit", "exit": EXIT_SETUP,
           "error": {"code": "kit_extra_missing", "message": KIT_EXTRA_MESSAGE}})
    return EXIT_SETUP


# --------------------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``pdfintake`` command and return its exit code (the console script exits with it)."""
    from .fsutil import utf8_stdio

    utf8_stdio()
    args_list = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    head = args_list[: args_list.index("--")] if "--" in args_list else args_list
    want_json = "--json" in head
    parser = build_parser()
    try:
        if args_list and args_list[0] == "run":
            # Inputs may come before and after options (`run a.pdf --smaller b.pdf`), which plain
            # parse_args refuses for a "*" positional; the run parser alone can parse intermixed.
            args = parser.get_default("_run_parser").parse_intermixed_args(args_list[1:])
            args.command = "run"
        else:
            args = parser.parse_args(args_list)
    except _ArgError as exc:
        return _argument_error(exc, want_json, _command_of(head))
    except SystemExit as exc:                            # --help and --version print and end here
        code = exc.code
        if code is None:
            return EXIT_OK
        return code if isinstance(code, int) else EXIT_USAGE
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _err("interrupted; nothing more was written.")
        return EXIT_INTERRUPTED


def _command_of(head: Sequence[str]) -> str | None:
    for word in head:
        if not word.startswith("-"):
            return word
    return None


_UNRECOGNIZED = "unrecognized arguments: "
_INVALID_CHOICE = "invalid choice: "
_CHOOSE_FROM = " (choose from "
_INVALID_VALUE = re.compile(r"invalid [\w ]+ value: ")
VALUE_PLACEHOLDER = "<the value given>"


def _scrub_values(message: str) -> str:
    """argparse's message with every value the caller typed replaced by a placeholder.

    Options and argparse's own lists (``choose from ...``) stay, so the message still says what was
    wrong. Values go: a mistyped flag may have been followed by something that shouldn't be repeated
    into a log, such as a password (decided 2026-11-05).
    """
    if message.startswith(_UNRECOGNIZED):
        words = message[len(_UNRECOGNIZED):].split(" ")
        flags = [w.split("=", 1)[0] for w in words if w.startswith("-")]
        others = len(words) - len(flags)
        message = _UNRECOGNIZED + (" ".join(flags) if flags else "")
        if others:
            message += (" and " if flags else "") + f"{others} other value{'s' if others > 1 else ''}"
        return message
    at = message.find(_INVALID_CHOICE)
    if at >= 0:
        start = at + len(_INVALID_CHOICE)
        end = message.rfind(_CHOOSE_FROM)            # argparse's list comes last, after the value
        tail = message[end:] if end >= start else ""
        return message[:start] + VALUE_PLACEHOLDER + tail
    found = _INVALID_VALUE.search(message)
    if found:                                        # "invalid int value: 'two'": the value runs to the end
        return message[:found.end()] + VALUE_PLACEHOLDER
    return message


def _argument_error(exc: _ArgError, want_json: bool, command: str | None) -> int:
    message = _scrub_values(exc.message)
    message = message[:1].upper() + message[1:]
    if want_json:
        out: dict[str, Any] = {"exit": EXIT_USAGE, "error": {"code": "bad_arguments", "message": message}}
        if command in ("run", "apply"):
            out = {"exit": EXIT_USAGE, "records": [], "error": out["error"]}
        _emit(out)
    else:
        _err(f"{exc.prog}: {message}. See '{exc.prog} --help'.")
    return EXIT_USAGE


# --------------------------------------------------------------------------- output helpers

def _emit(obj: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    sys.stdout.flush()


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(text: str) -> None:
    sys.stderr.write(f"{PROG}: {text}\n")
    sys.stderr.flush()


def _shown(value: Any) -> str:
    """A path or name for a terminal: control characters and lone surrogates escaped, so a file name
    can't move the cursor or end a line."""
    text = str(value).encode("utf-8", "backslashreplace").decode("utf-8")
    return "".join(ch if ch.isprintable() or ch == " " else ch.encode("unicode_escape").decode("ascii")
                   for ch in text)


def _error_json(exc: PdfIntakeError) -> dict[str, Any]:
    out: dict[str, Any] = {"code": exc.code, "message": exc.message}
    problems = exc.detail.get("problems")
    if isinstance(problems, list) and all(isinstance(p, str) for p in problems):
        out["problems"] = problems[:20]
    return out


def _fail(args: argparse.Namespace, exc: PdfIntakeError, *, records: bool = False,
          extra: Mapping[str, Any] | None = None) -> int:
    """Report an error that stopped the whole command; return its exit code."""
    code = exc.exit_code
    if getattr(args, "json", False):
        out: dict[str, Any] = {"exit": code}
        if records:
            out["records"] = []
        out["error"] = _error_json(exc)
        if isinstance(exc, Refused):
            out["refusal"] = {"code": exc.code, "message": exc.message}
        if extra:
            out.update(extra)
        _emit(out)
    else:
        _err(exc.message)
    return code


def _unexpected(args: argparse.Namespace, exc: BaseException, *, records: bool = False) -> int:
    """A bug or an unreadable file outside any record: exit 1, naming only the kind of error.

    The message isn't repeated: it may quote a library's text about a file, and nothing a command
    prints may carry a password or document text.
    """
    name = type(exc).__name__ if type(exc).__name__.isidentifier() else "error"
    return _fail(args, PdfIntakeError("internal_error", f"pdfintake hit an unexpected {name}; nothing more was "
                                                        "written. Please report it with the command you ran.",
                                      kind=name), records=records)


# --------------------------------------------------------------------------- run and apply

def _read_paths_from(path: Path) -> list[Path]:
    """The paths in a ``--paths-from`` list: one UTF-8 path per line, in order.

    Only the line break is removed (``\\n`` or ``\\r\\n``), so spaces at either end stay part of the
    name; empty lines are skipped and a leading byte-order mark is ignored. A relative path is taken
    from the current folder, as on the command line. The list must be a regular file: a FIFO, a
    device or ``-`` is never read, so a run can't wait on standard input (DESIGN §12).
    """
    shown = _shown(path)
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        raise UsageError("paths_from_missing", f"The list of paths {shown} doesn't exist or can't be read.",
                         path=shown) from None
    if not stat.S_ISREG(st.st_mode):
        raise UsageError("paths_from_not_a_file", f"The list of paths {shown} isn't a regular file.", path=shown)
    if st.st_size > MAX_PATHS_FROM_BYTES:
        raise UsageError("paths_from_too_large", f"The list of paths {shown} is larger than 16 MB.", path=shown)
    try:
        with open(path, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):   # swapped for something else since the stat
                raise UsageError("paths_from_not_a_file", f"The list of paths {shown} isn't a regular file.",
                                 path=shown)
            raw = fh.read(MAX_PATHS_FROM_BYTES + 1)
    except OSError:
        raise UsageError("paths_from_missing", f"The list of paths {shown} can't be read.", path=shown) from None
    if len(raw) > MAX_PATHS_FROM_BYTES:
        raise UsageError("paths_from_too_large", f"The list of paths {shown} is larger than 16 MB.", path=shown)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise UsageError("paths_from_not_utf8", f"The list of paths {shown} isn't UTF-8 text.", path=shown) from None
    if text.startswith("﻿"):
        text = text[1:]
    out: list[Path] = []
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        if line:
            out.append(Path(line))
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        from . import flow

        inputs = [Path(p) for p in args.inputs]
        for listed in args.paths_from:
            inputs.extend(_read_paths_from(Path(listed)))
        opts = flow.RunOptions(
            provenance=args.provenance or "ordinary",
            out=Path(args.out) if args.out is not None else None,
            outdir=Path(args.outdir) if args.outdir is not None else None,
            lang=_lang_codes(args.lang, "--lang"),
            force_lang=_lang_codes(args.force_lang, "--force-lang"),
            fetch=args.fetch,
            smaller=bool(args.smaller),
            overwrite=bool(args.overwrite),
            redo_ocr=bool(args.redo_ocr),
            keep_existing_layer=bool(args.keep_existing_layer),
            separate=bool(args.separate),
            plan_only=bool(args.plan),
            workers=args.workers if args.workers is not None else flow.DEFAULT_WORKERS,
            passwords_file=Path(args.passwords_file) if args.passwords_file is not None else None,
            thorough=args.thorough,
        )
        records = flow.run(inputs, opts)
        return _report_records(args, records)           # inside the try: a bad record is still one JSON object
    except PdfIntakeError as exc:
        return _fail(args, exc, records=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001 - one JSON object, whatever happened
        return _unexpected(args, exc, records=True)


def _cmd_apply(args: argparse.Namespace) -> int:
    try:
        from . import flow

        records = flow.apply(args.record_id, choice=args.choice, lang=args.lang,
                             passwords_file=Path(args.passwords_file) if args.passwords_file is not None else None)
        return _report_records(args, records)
    except PdfIntakeError as exc:
        return _fail(args, exc, records=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001
        return _unexpected(args, exc, records=True)


def _report_records(args: argparse.Namespace, records: Sequence[Any]) -> int:
    """Print the records and return the process exit. Everything that can raise (the exit, the JSON)
    is worked out before the first byte is printed, so a failure here leaves room for one error object."""
    from . import record

    code = record.overall_exit(records) if records else EXIT_OK
    if args.json:
        _emit({"exit": code, "records": [record.to_json(r) for r in records]})
    else:
        for rec in records:
            _print_record(rec)
    return code


def _size(n: Any) -> str:
    try:
        value = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def _why(rec: Any) -> str:
    refusal = getattr(rec, "refusal", None)
    if isinstance(refusal, Mapping) and refusal.get("message"):
        return str(refusal["message"])
    for step in reversed(getattr(rec, "steps", []) or []):
        detail = step.get("detail") if isinstance(step, Mapping) else None
        if isinstance(step, Mapping) and step.get("status") == "failed" and isinstance(detail, Mapping):
            return str(detail.get("message") or detail.get("code") or "")
    return ""


def _print_record(rec: Any) -> None:
    """A few lines a person can read: what was written (or why not), then warnings, notes and offers."""
    inputs = (rec.input or {}).get("paths") or []
    first = Path(str(inputs[0])).name if inputs else "?"
    more = f" (+{len(inputs) - 1} more)" if len(inputs) > 1 else ""
    sizes = ((rec.input or {}).get("seal") or {}).get("sizes") or []
    out = rec.output
    if rec.status in ("done", "resolved", "plan") and isinstance(out, Mapping):
        kind = " (searchable copy)" if out.get("kind") == "searchable_copy" else ""
        before = _size(sum(sizes)) if sizes else "?"
        _say(f"{rec.status}: {_shown(first)}{more} -> {_shown(out.get('path'))}{kind} "
             f"[{before} -> {_size(out.get('bytes'))}]  record {rec.id}")
    elif rec.status in ("done", "resolved", "plan"):
        _say(f"{rec.status}: {_shown(first)}{more}, nothing written  record {rec.id}")
    else:
        why = _why(rec)
        _say(f"{rec.status} (exit {rec.exit}): {_shown(first)}{more}" + (f": {_shown(why)}" if why else "")
             + f"  record {rec.id}")
    for w in rec.warnings or []:
        page = f" (page {w.page})" if getattr(w, "page", None) else ""
        _say(f"  warning {w.code}{page}: {_shown(w.message)}")
    notes = [n.code for n in (rec.notes or []) if n.code != "input_unchanged"]
    if notes:
        _say("  notes: " + ", ".join(dict.fromkeys(notes)))
    for offer in rec.offers or []:
        ids = " | ".join(o.id for o in offer.options)
        _say(f"  offer {offer.kind}: {_shown(offer.summary)}  (pdfintake apply {rec.id} {offer.apply_flag} {ids})")


# --------------------------------------------------------------------------- doctor

def _cmd_doctor(args: argparse.Namespace) -> int:
    try:
        from . import doctor

        rep = doctor.report(canary=bool(args.canary), check_network=bool(args.check_network))
        code = doctor.exit_code(rep)
    except PdfIntakeError as exc:
        return _fail(args, exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001
        return _unexpected(args, exc)
    if args.json:
        _emit({"exit": code, **{k: v for k, v in rep.items() if k != "exit"}})
        return code
    plat = rep.get("platform") or {}
    _say(f"pdfintake {rep.get('pdfintake')} on {plat.get('key')} ({rep.get('os')}), Python {rep.get('python')}")
    engines = rep.get("engines") or {}
    if isinstance(engines, Mapping):
        for name, info in engines.items():
            if isinstance(info, Mapping):
                state = "ok" if info.get("available") else (info.get("state") or info.get("reason") or "unavailable")
                version = f" {info.get('version')}" if info.get("version") else ""
                _say(f"  engine {name}:{version} {state}")
    tho = rep.get("thorough") or {}
    if isinstance(tho, Mapping) and tho.get("available"):
        state = "on" if tho.get("on") else (tho.get("setting") or "not set up")
        _say(f"  thorough mode: {state}" + ("" if tho.get("ready") else " (see 'pdfintake thorough status')"))
    base = rep.get("base_pack") or {}
    _say(f"  base pack: {'ok ' + str(base.get('version')) if base.get('ok') else base.get('message', 'missing')}")
    for pack in rep.get("packs") or []:
        _say(f"  pack {pack.get('pack')}: {'ok' if pack.get('ok') else '; '.join(pack.get('problems') or [])}")
    _say(f"  cache: {_shown(rep.get('cache'))}")
    for problem in rep.get("problems") or []:
        _say(f"  problem: {_shown(problem)}")
    _say("ready" if code == EXIT_OK else "not ready (exit 3)")
    return code


# --------------------------------------------------------------------------- packs

def _entry_json(entry: Any, installed: bool | None) -> dict[str, Any]:
    dist = entry.dist
    return {"id": entry.id, "version": entry.version, "title": entry.title, "base": bool(entry.base),
            "scripts": list(entry.scripts), "languages": list(entry.langid_codes),
            "download_bytes": None if dist.kind == "installed" else int(dist.bytes),
            "installed_bytes": int(entry.installed_bytes), "installed": installed}


def _cmd_packs(args: argparse.Namespace) -> int:
    action = args.packs_command
    try:
        from . import platforms
        from .langpacks import index as index_mod
        from .langpacks import store
        from .ocr import engines
        from . import integrity

        if action == "path":
            out: dict[str, Any] = {"cache": str(platforms.cache_root()),
                                   "packs": str(platforms.cache_root() / store.PACKS),
                                   "tessdata": str(platforms.cache_root() / store.TESSDATA)}
            return _packs_out(args, EXIT_OK, out, [f"cache: {_shown(out['cache'])}", f"packs: {_shown(out['packs'])}"])
        idx = index_mod.load()
        if action in ("list", "show"):
            ok_ids = {v.pack for v in store.verify() if v.ok} if action == "list" else set()
            if action == "list":
                rows = [_entry_json(e, e.id in ok_ids) for e in idx.entries]
                lines = [f"{r['id']:>4} {r['title']} {r['version']}: "
                         f"{'installed' if r['installed'] else 'available'}" for r in rows]
                return _packs_out(args, EXIT_OK, {"packs": rows}, lines)
            entry = _entry(idx, args.pack_id)
            result = store.verify(entry.id)[0]
            row = _entry_json(entry, result.ok)
            row["problems"] = list(result.problems)
            return _packs_out(args, EXIT_OK, {"pack": row},
                              [f"{row['id']} {row['title']} {row['version']}: "
                               f"{'installed' if row['installed'] else 'not installed'}"])
        if action == "fetch":
            entry = _entry(idx, args.pack_id)
            engines.warm_up()
            with integrity.no_subprocess():
                pack = store.ensure(entry, allow_network=True)
            out = {"pack": {"id": pack.manifest.id, "version": pack.manifest.version, "path": str(pack.root)}}
            return _packs_out(args, EXIT_OK, out, [f"{pack.manifest.id} {pack.manifest.version}: installed and checked"])
        if action == "verify":
            if args.pack_id is not None:
                _entry(idx, args.pack_id)
            results = store.verify(args.pack_id)
            rows = [{"pack": v.pack, "ok": v.ok, "problems": list(v.problems)} for v in results]
            code = EXIT_OK if all(v.ok for v in results) else EXIT_SETUP
            lines = [f"{r['pack']}: {'ok' if r['ok'] else '; '.join(r['problems'])}" for r in rows]
            return _packs_out(args, code, {"packs": rows}, lines)
        if action == "remove":
            removed = store.remove(args.pack_id)
            return _packs_out(args, EXIT_OK, {"pack": args.pack_id, "removed": removed},
                              [f"{args.pack_id}: {'removed' if removed else 'was not installed'}"])
    except PdfIntakeError as exc:
        return _fail(args, exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001
        return _unexpected(args, exc)
    raise AssertionError(action)                         # argparse allows nothing else


def _entry(idx: Any, pack_id: str) -> Any:
    entry = idx.get(pack_id)
    if entry is None:
        known = ", ".join(e.id for e in idx.entries)
        raise UsageError("unknown_pack", f"This version of pdfintake has no pack {_shown(pack_id)!r} "
                                         f"(it knows {known}).", pack=_shown(pack_id))
    return entry


def _packs_out(args: argparse.Namespace, code: int, out: Mapping[str, Any], lines: Sequence[str]) -> int:
    if args.json:
        _emit({"exit": code, **out})
    else:
        for line in lines:
            _say(line)
    return code


# --------------------------------------------------------------------------- record show

def _cmd_record_show(args: argparse.Namespace) -> int:
    try:
        from . import record

        rec = record.load(args.record_id)
        if args.json:
            _emit({"exit": EXIT_OK, "record": record.to_json(rec)})
        else:
            _print_record(rec)
        return EXIT_OK
    except PdfIntakeError as exc:
        return _fail(args, exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001 - an undecodable or unopenable record file
        return _unexpected(args, exc)


# --------------------------------------------------------------------------- export

def _cmd_export(args: argparse.Namespace) -> int:
    """Build the export in the cache, then hand it to ``flow.commit``, the one writer."""
    kind = args.export_command
    try:
        from . import flow, fsutil, integrity
        from .pdf import export

        src = Path(os.path.abspath(Path(args.input).expanduser()))
        dest = Path(os.path.abspath(Path(args.out).expanduser()))
        fsutil.guard_output([src], dest, overwrite=bool(args.overwrite))   # exit 2 before any work
        seal = integrity.InputSeal.take([src])
        staging = fsutil.work_dir("staging", f"export-{secrets.token_hex(6)}")
        built = staging / "export.pdf"
        try:
            with integrity.no_subprocess():
                if kind == "clean":
                    rep = export.export_clean(src, built, provenance=args.provenance,
                                              strip_dates=bool(args.strip_dates))
                else:
                    rep = export.export_flatten(src, built, provenance=args.provenance)
                ref = flow.commit(built, dest, seal, None, args.provenance, adds_layer=False, inputs=[src],
                                  overwrite=bool(args.overwrite))
        finally:
            fsutil.prune(staging, older_than_days=0)      # the built file; the empty folder goes with the next prune
    except PdfIntakeError as exc:
        return _fail(args, exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001
        return _unexpected(args, exc)
    warnings = [{"code": w, "message": export.WARNING_MESSAGES.get(w, w)} for w in rep.warnings]
    out = {"exit": EXIT_OK, "kind": kind,
           "output": {"path": str(ref.path), "sha256": ref.sha256, "bytes": ref.bytes},
           "removed": list(rep.removed), "kept": list(rep.kept), "warnings": warnings,
           "notes": [{"code": "input_unchanged", "message": "The input file was not modified."}]}
    if args.json:
        _emit(out)
    else:
        _say(f"export {kind}: {_shown(src.name)} -> {_shown(ref.path)} [{_size(ref.bytes)}]")
        for w in warnings:
            _say(f"  warning {w['code']}: {w['message']}")
    return EXIT_OK


# --------------------------------------------------------------------------- thorough

class _Progress:
    """Download progress on standard error: a small file once it's done, a large one every 5%."""

    def __init__(self) -> None:
        self.shown: dict[str, int] = {}

    def __call__(self, name: str, done: int, total: int) -> None:
        pct = 100 * done // max(1, total)
        if total < 50_000_000 and done < total:
            return
        last = self.shown.get(name, -5)
        if pct >= last + 5 or (done == total and last < 100):
            self.shown[name] = pct
            _err(f"downloading {name}: {pct}% of {_size(total)}")


def _cmd_thorough(args: argparse.Namespace) -> int:
    action = args.thorough_command
    try:
        from . import integrity, thorough
        from .ocr import engines

        if action == "status":
            out = thorough.status()
        elif action == "enable":
            engines.warm_up()
            with integrity.no_subprocess():
                out = thorough.enable(progress=None if args.json else _Progress())
        elif action == "disable":
            out = thorough.disable(remove=bool(args.remove))
        else:
            out = thorough.decline()
    except PdfIntakeError as exc:
        return _fail(args, exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001
        return _unexpected(args, exc)
    if args.json:
        _emit({"exit": EXIT_OK, **out})
        return EXIT_OK
    if not out.get("available"):
        _say(f"thorough mode: not available here ({out.get('why_not')})")
        return EXIT_OK
    model = out.get("model") or {}
    libs = out.get("libraries") or {}
    state = "on" if out.get("on") else (out.get("setting") or "not decided")
    _say(f"thorough mode: {state}")
    _say(f"  libraries: {'installed' if not libs.get('missing') else 'missing ' + ', '.join(libs['missing'])}")
    if libs.get("install"):
        _say(f"  install them with: {libs['install']}")
    _say(f"  model {model.get('repo')}: " + ("installed" if model.get("installed")
                                            else f"not installed ({_size(model.get('download_bytes'))} to download)"))
    _say(f"  about {out.get('seconds_per_page')} s more a page")
    return EXIT_OK


# --------------------------------------------------------------------------- passwords

def _cmd_passwords_add(args: argparse.Namespace) -> int:
    """Append to the saved list. Nothing about the passwords is printed: not a count, not a line."""
    try:
        from . import passwords

        target = Path(args.passwords_file) if args.passwords_file is not None else passwords.default_path()
        passwords.add(Path(args.from_file), to=target)
    except PdfIntakeError as exc:
        return _fail(args, exc)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:                             # noqa: BLE001
        return _unexpected(args, exc)
    if args.json:
        _emit({"exit": EXIT_OK, "passwords_file": str(target)})
    else:
        _say(f"saved to {_shown(target)}; delete {_shown(args.from_file)} now.")
    return EXIT_OK


__all__ = ["main", "build_parser"]
