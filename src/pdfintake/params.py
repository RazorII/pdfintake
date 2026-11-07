"""One validator for engine, rule, fill and repair parameters.

Every registered member declares its parameters as a ``ParamTable``. A pack's ``params`` object
is checked against that table when the pack loads, so a bad pack is rejected with every problem
named (and its path) before anything runs.

What each kind accepts, and which ``Param`` attributes apply to it:

==============  =====================================================  ==========================
kind            value                                                  attributes that apply
==============  =====================================================  ==========================
``int``         an integer (never a bool)                              choices, min, max
``float``       a finite number (never a bool); returned as float      choices, min, max
``bool``        true or false                                          (none)
``str``         a string; needs ``pattern`` or ``choices``             pattern, choices, max_len
``str_list``    a list of distinct strings; needs pattern or choices   pattern, choices, max_len
                (``min``/``max`` bound the item count; ``max_len``     (per item), min, max
                bounds each item)
``use_ref``     the id of an engine use in the same reader set         (none)
``use_list``    a list of distinct use ids                             min, max (item count)
``asset_ref``   a model key of this pack, or ``en:<key>`` of the base  choices (the model kinds
                pack                                                   it may name)
``script``      a script name this core knows                          choices
``char_table``  at most 16 keys, each exactly one character; each      (none: the limits are
                value a list of at most 8 distinct literal strings of  fixed)
                1 to 3 characters
==============  =====================================================  ==========================

Patterns use ``re.fullmatch``. No free text reaches core from a pack: ``str`` and ``str_list``
must carry a pattern or a closed set of choices, and a ``char_table`` holds literals that
consumers compile with ``re.escape`` only. A table that breaks these rules is refused when it is
defined (``ValueError``), because that is a bug in core, not in a pack.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from .errors import PackError

ParamKind = Literal["int", "float", "bool", "str", "str_list", "use_ref", "use_list", "asset_ref", "script", "char_table"]
PARAM_KINDS: tuple[str, ...] = get_args(ParamKind)

CHAR_TABLE_MAX_KEYS = 16
CHAR_TABLE_MAX_VALUES = 8
CHAR_TABLE_MAX_VALUE_LEN = 3

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,39}")
_OPTIONAL_ATTRS = ("choices", "min", "max", "pattern", "max_len")
_APPLIES: dict[str, frozenset[str]] = {
    "int": frozenset({"choices", "min", "max"}),
    "float": frozenset({"choices", "min", "max"}),
    "bool": frozenset(),
    "str": frozenset({"choices", "pattern", "max_len"}),
    "str_list": frozenset({"choices", "pattern", "max_len", "min", "max"}),
    "use_ref": frozenset(),
    "use_list": frozenset({"min", "max"}),
    "asset_ref": frozenset({"choices"}),
    "script": frozenset({"choices"}),
    "char_table": frozenset(),
}
_LIST_KINDS = frozenset({"str_list", "use_list"})
_CONTEXT_KINDS = frozenset({"use_ref", "use_list", "asset_ref", "script"})


@dataclass(frozen=True)
class Param:
    name: str
    kind: ParamKind
    required: bool = False
    default: Any = None
    choices: tuple[Any, ...] | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None
    max_len: int | None = None

    def __post_init__(self) -> None:
        label = f"Param {self.name!r}"
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"{label}: a parameter name is 1-40 characters of a-z, 0-9 and _, starting with a letter")
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"{label}: unknown kind {self.kind!r}; kinds are {', '.join(PARAM_KINDS)}")
        label = f"Param {self.name!r} ({self.kind})"
        for attr in _OPTIONAL_ATTRS:
            if getattr(self, attr) is not None and attr not in _APPLIES[self.kind]:
                raise ValueError(f"{label}: {attr} does not apply to this kind")
        if self.choices is not None:
            if isinstance(self.choices, (str, bytes)) or not isinstance(self.choices, Sequence):
                raise ValueError(f"{label}: choices must be a tuple")
            object.__setattr__(self, "choices", tuple(self.choices))
            if not self.choices:
                raise ValueError(f"{label}: choices must not be empty")
            for c in self.choices:
                if self.kind == "int" and not _is_int(c):
                    raise ValueError(f"{label}: choice {c!r} is not an integer")
                if self.kind == "float" and not _is_number(c):
                    raise ValueError(f"{label}: choice {c!r} is not a number")
                if self.kind in ("str", "str_list", "asset_ref", "script") and not isinstance(c, str):
                    raise ValueError(f"{label}: choice {c!r} is not a string")
        if self.kind in ("str", "str_list") and self.pattern is None and self.choices is None:
            raise ValueError(f"{label}: needs a pattern or a choices tuple (no free text from a pack)")
        if self.pattern is not None:
            if not isinstance(self.pattern, str):
                raise ValueError(f"{label}: pattern must be a string")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"{label}: pattern doesn't compile: {exc}") from None
        for attr in ("min", "max"):
            bound = getattr(self, attr)
            if bound is None:
                continue
            if not _is_number(bound) or not math.isfinite(bound):
                raise ValueError(f"{label}: {attr} must be a finite number")
            if self.kind in _LIST_KINDS and (not _is_int(bound) or bound < 0):
                raise ValueError(f"{label}: {attr} counts items, so it must be a whole number >= 0")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"{label}: min is above max")
        if self.max_len is not None and (not _is_int(self.max_len) or self.max_len < 1):
            raise ValueError(f"{label}: max_len must be a whole number >= 1")
        if self.required and self.default is not None:
            raise ValueError(f"{label}: a required parameter has no default")
        if self.default is not None and self.kind not in _CONTEXT_KINDS:
            problems: list[str] = []
            _check(self, self.default, _NO_CONTEXT, "default", problems)
            if problems:
                raise ValueError(f"{label}: its default is invalid: {'; '.join(problems)}")


@dataclass(frozen=True)
class ValidationContext:
    use_ids: frozenset[str]
    asset_keys: frozenset[str]
    asset_kinds: Mapping[str, str]
    scripts: frozenset[str]
    where: str


_NO_CONTEXT = ValidationContext(frozenset(), frozenset(), {}, frozenset(), "")


class ParamTable:
    """The parameters one registered member takes."""

    def __init__(self, *params: Param) -> None:
        seen: set[str] = set()
        for p in params:
            if not isinstance(p, Param):
                raise ValueError(f"ParamTable takes Param objects, not {type(p).__name__}")
            if p.name in seen:
                raise ValueError(f"ParamTable: parameter {p.name!r} is defined twice")
            seen.add(p.name)
        self.params: tuple[Param, ...] = tuple(params)
        self._by_name: dict[str, Param] = {p.name: p for p in params}

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __repr__(self) -> str:
        return f"ParamTable({', '.join(p.name for p in self.params)})"

    def validate(self, raw: Mapping[str, Any], ctx: ValidationContext) -> dict[str, Any]:
        """The checked parameters, with the default filled in for every absent optional one that has one.

        An absent optional parameter without a default is left out, never set to None: members
        read it with ``params.get(name)``. Validation is idempotent (``validate(validate(x))``
        equals ``validate(x)``), so validated params can be checked again or written back into a
        manifest.

        Raises PackError("invalid_params") listing every problem at once; each problem starts with
        its path (``ctx.where`` plus the parameter), and the list is in ``detail["problems"]``.
        """
        problems: list[str] = []
        out: dict[str, Any] = {}
        if not isinstance(raw, Mapping):
            problems.append(f"{ctx.where or 'params'}: expected an object of parameters, got {_type_name(raw)}")
        else:
            for key in raw:
                if not isinstance(key, str) or key not in self._by_name:
                    takes = ", ".join(self._by_name) or "no parameters"
                    problems.append(f"{_path(ctx.where, key)}: unknown parameter (this member takes: {takes})")
            for p in self.params:
                path = _path(ctx.where, p.name)
                if p.name in raw:
                    before = len(problems)
                    value = _check(p, raw[p.name], ctx, path, problems)
                    if len(problems) == before:
                        out[p.name] = value
                elif p.required:
                    problems.append(f"{path}: required parameter is missing")
                elif p.default is not None:
                    out[p.name] = _copy_default(p)
                # An optional parameter with no default stays absent (members read it with
                # params.get), so validate(validate(x)) == validate(x) and the result can be
                # written back into a manifest, whose params allow no null.
        if problems:
            raise PackError(
                "invalid_params",
                "invalid parameters: " + "; ".join(problems),
                problems=problems,
                where=ctx.where,
            )
        return out

    def describe(self) -> list[dict[str, Any]]:
        """One plain dict per parameter, in definition order (for docs/pack-schema.md and ``kit init``)."""
        rows: list[dict[str, Any]] = []
        for p in self.params:
            rows.append({
                "name": p.name,
                "kind": p.kind,
                "required": p.required,
                "default": _copy_default(p),
                "choices": list(p.choices) if p.choices is not None else None,
                "min": p.min,
                "max": p.max,
                "pattern": p.pattern,
                "max_len": p.max_len,
            })
        return rows


def _copy_default(p: Param) -> Any:
    value = p.default
    if value is None:
        return None
    if p.kind in _LIST_KINDS:
        return list(value)
    if p.kind == "char_table":
        return {k: list(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "a boolean"
    if isinstance(v, (int, float)):
        return "a number"
    if isinstance(v, str):
        return "a string"
    if isinstance(v, Mapping):
        return "an object"
    if isinstance(v, (list, tuple)):
        return "an array"
    return f"a {type(v).__name__}"


def _path(where: str, key: Any) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{where}.{key}" if where else key
    token = f"[{json.dumps(key, ensure_ascii=False) if isinstance(key, str) else repr(key)}]"
    return f"{where}{token}" if where else token


def _bad_chars(s: str) -> bool:
    return any(unicodedata.category(ch) in ("Cc", "Cs") for ch in s)


def _bounds(p: Param, value: float, path: str, problems: list[str], what: str = "") -> None:
    if p.min is not None and value < p.min:
        problems.append(f"{path}: {what}must be at least {_fmt(p.min)}")
    if p.max is not None and value > p.max:
        problems.append(f"{path}: {what}must be at most {_fmt(p.max)}")


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def _check_str(p: Param, v: Any, path: str, problems: list[str]) -> bool:
    if not isinstance(v, str):
        problems.append(f"{path}: expected a string, got {_type_name(v)}")
        return False
    ok = True
    if p.max_len is not None and len(v) > p.max_len:
        problems.append(f"{path}: longer than {p.max_len} characters")
        ok = False
    if p.pattern is not None and not re.fullmatch(p.pattern, v):
        problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} doesn't match {p.pattern}")
        ok = False
    if p.choices is not None and v not in p.choices:
        problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} isn't one of {', '.join(map(str, p.choices))}")
        ok = False
    return ok


def _check_list(p: Param, v: Any, path: str, problems: list[str]) -> list[Any] | None:
    if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple)):
        problems.append(f"{path}: expected an array, got {_type_name(v)}")
        return None
    _bounds(p, len(v), path, problems, "the number of items ")
    seen: set[Any] = set()
    for i, item in enumerate(v):
        if isinstance(item, str):
            if item in seen:
                problems.append(f"{path}[{i}]: {json.dumps(item, ensure_ascii=False)} is listed twice")
            seen.add(item)
    return list(v)


def _check(p: Param, v: Any, ctx: ValidationContext, path: str, problems: list[str]) -> Any:
    """Append problems for value ``v`` of param ``p``; return the value to hand to the member."""
    if v is None:
        problems.append(f"{path}: null is not allowed")
        return None
    kind = p.kind
    if kind == "int":
        if not _is_int(v):
            problems.append(f"{path}: expected an integer, got {_type_name(v)}")
            return None
        _bounds(p, v, path, problems)
        if p.choices is not None and v not in p.choices:
            problems.append(f"{path}: {v} isn't one of {', '.join(map(str, p.choices))}")
        return v
    if kind == "float":
        if not _is_number(v):
            problems.append(f"{path}: expected a number, got {_type_name(v)}")
            return None
        if not math.isfinite(v):
            problems.append(f"{path}: must be a finite number")
            return None
        _bounds(p, v, path, problems)
        if p.choices is not None and v not in p.choices:
            problems.append(f"{path}: {v} isn't one of {', '.join(map(str, p.choices))}")
        return float(v)
    if kind == "bool":
        if not isinstance(v, bool):
            problems.append(f"{path}: expected true or false, got {_type_name(v)}")
            return None
        return v
    if kind == "str":
        _check_str(p, v, path, problems)
        return v
    if kind == "str_list":
        items = _check_list(p, v, path, problems)
        if items is not None:
            for i, item in enumerate(items):
                _check_str(p, item, f"{path}[{i}]", problems)
        return items
    if kind == "use_ref":
        if not isinstance(v, str):
            problems.append(f"{path}: expected a use id (a string), got {_type_name(v)}")
        elif v not in ctx.use_ids:
            problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} names no engine use in this reader set "
                            f"(uses: {', '.join(sorted(ctx.use_ids)) or 'none'})")
        return v
    if kind == "use_list":
        items = _check_list(p, v, path, problems)
        if items is not None:
            for i, item in enumerate(items):
                if not isinstance(item, str):
                    problems.append(f"{path}[{i}]: expected a use id (a string), got {_type_name(item)}")
                elif item not in ctx.use_ids:
                    problems.append(f"{path}[{i}]: {json.dumps(item, ensure_ascii=False)} names no engine use in "
                                    f"this reader set (uses: {', '.join(sorted(ctx.use_ids)) or 'none'})")
        return items
    if kind == "asset_ref":
        if not isinstance(v, str):
            problems.append(f"{path}: expected a model key (a string), got {_type_name(v)}")
        elif v not in ctx.asset_keys:
            problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} names no model of this pack or the base pack")
        elif p.choices is not None:
            kind_of = ctx.asset_kinds.get(v)
            if kind_of not in p.choices:
                problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} is a {kind_of or 'model of unknown kind'}; "
                                f"this parameter needs {' or '.join(p.choices)}")
        return v
    if kind == "script":
        if not isinstance(v, str):
            problems.append(f"{path}: expected a script name, got {_type_name(v)}")
        elif v not in ctx.scripts:
            problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} is not a script this core knows")
        elif p.choices is not None and v not in p.choices:
            problems.append(f"{path}: {json.dumps(v, ensure_ascii=False)} isn't one of {', '.join(p.choices)}")
        return v
    if kind == "char_table":
        return _check_char_table(v, path, problems)
    raise AssertionError(kind)  # unreachable: Param refuses unknown kinds


def _check_char_table(v: Any, path: str, problems: list[str]) -> dict[str, list[str]] | None:
    if not isinstance(v, Mapping):
        problems.append(f"{path}: expected an object mapping one character to its variants, got {_type_name(v)}")
        return None
    if len(v) > CHAR_TABLE_MAX_KEYS:
        problems.append(f"{path}: has {len(v)} keys; at most {CHAR_TABLE_MAX_KEYS}")
    out: dict[str, list[str]] = {}
    for key, values in v.items():
        kpath = f"{path}[{json.dumps(key, ensure_ascii=False) if isinstance(key, str) else repr(key)}]"
        if not isinstance(key, str) or len(key) != 1:
            problems.append(f"{kpath}: a key must be exactly one character")
        elif _bad_chars(key):
            problems.append(f"{kpath}: a key must not be a control character")
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            problems.append(f"{kpath}: expected an array of literal strings, got {_type_name(values)}")
            continue
        if len(values) > CHAR_TABLE_MAX_VALUES:
            problems.append(f"{kpath}: has {len(values)} values; at most {CHAR_TABLE_MAX_VALUES}")
        seen: set[str] = set()
        for i, item in enumerate(values):
            ipath = f"{kpath}[{i}]"
            if not isinstance(item, str):
                problems.append(f"{ipath}: expected a literal string, got {_type_name(item)}")
                continue
            if not 1 <= len(item) <= CHAR_TABLE_MAX_VALUE_LEN:
                problems.append(f"{ipath}: a value is 1 to {CHAR_TABLE_MAX_VALUE_LEN} characters, "
                                f"{json.dumps(item, ensure_ascii=False)} has {len(item)}")
            elif _bad_chars(item):
                problems.append(f"{ipath}: a value must not contain control characters")
            if item in seen:
                problems.append(f"{ipath}: {json.dumps(item, ensure_ascii=False)} is listed twice")
            seen.add(item)
        if isinstance(key, str):
            out[key] = [s for s in values if isinstance(s, str)]
    return out
