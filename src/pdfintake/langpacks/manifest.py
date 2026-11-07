"""The pack manifest: a strict, hand-written loader for schema 1.

A language pack is data only: a manifest plus pinned model, font and canary files. ``parse`` turns
a manifest's bytes into a frozen ``Manifest``; ``check`` lists every problem with a parsed
manifest at once, each starting with its JSON path (``$.readers.default.engines[0].params.psm``).

The loader does two things.

1. **The shape.** It walks ``manifest.schema.json`` (JSON Schema draft 2020-12, shipped beside
   this file) itself, so the schema is the single statement of the shape and jsonschema stays a
   test-only dependency. The walker understands exactly the keywords the schema uses and refuses
   a schema with any other, so a schema change can't be silently ignored. One deliberate
   difference from jsonschema: an anchored ``^...$`` pattern must match the whole string, so a
   trailing newline never passes (Python's ``$`` would let it through).
2. **The vocabulary.** Every name is checked against the running core: engines, pick rules and
   their versions, fills, repairs, normalisers, scripts and platform keys. Every ``params`` object
   passes its member's ``ParamTable``. Model references resolve to a model of the right engine
   and kind, in this pack or (``en:``) in the base pack. Sources must come from the allowlist.
   Model file names never end in a code suffix (``.py``, ``.pth``, ``.so`` and the like), and
   file names are compared ignoring case, because the store and the tessdata directory may sit
   on a case-insensitive file system. Every ``canary.must_find`` string holds a character
   other than whitespace.
   Fill steps must serve every script of the pack, a pick's corroborators are never readers, the
   default reader set covers every supported platform, and ``requires_core`` must not be newer
   than this core.

Nothing in a manifest is ever imported or executed, and no pack text reaches the user except
through patterns (ids, codes) or core templates.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, Literal

from .. import __version__
from ..errors import PackError, SetupError
from ..ocr import langplan as _langplan
from ..params import ValidationContext
from ..types import Role

SCHEMA_VERSIONS: tuple[int, ...] = (1,)       # the manifest schemas this core reads
BASE_PACK_ID = "en"                            # the base pack, shipped with the install
BASE_PREFIX = "en:"                            # a model reference into the base pack
MAX_MANIFEST_BYTES = 1 << 20

# The allowlist of model and font sources. The schema holds the same two patterns
# (tests/test_manifest.py checks they are identical); the loader adds: no "..", no query, no fragment.
SOURCE_PATTERNS: Mapping[Literal["model", "font"], re.Pattern[str]] = MappingProxyType({
    "model": re.compile(
        r"^https://(github\.com/tesseract-ocr/tessdata(_best|_fast)?/raw/[0-9a-f]{40}/[A-Za-z0-9_.+-]+"
        r"|github\.com/RapidAI/[A-Za-z0-9_.-]+/(raw/[0-9a-f]{40}|releases/download/[A-Za-z0-9_.+-]+)/[A-Za-z0-9_/.+-]+"
        r"|www\.modelscope\.cn/models/RapidAI/RapidOCR/resolve/v[0-9]+\.[0-9]+\.[0-9]+/[A-Za-z0-9_/.+-]+)$"),
    "font": re.compile(
        r"^https://github\.com/notofonts/[A-Za-z0-9_.-]+/(raw/[0-9a-f]{40}|releases/download/[A-Za-z0-9_.+-]+)"
        r"/[A-Za-z0-9_/.+-]+$"),
})

# Defaults for the optional detect fields (the function-word page rule). The ratio is the exact
# third, 3 x hits >= rival hits (decided 2026-11-05). It is taken from ocr.langplan, which applies the
# rule, so the value has one source (langplan imports nothing from langpacks at run time).
FW_MIN_HITS_DEFAULT = 3
FW_RATIO_DEFAULT: float = _langplan.FW_RATIO_DEFAULT
PROBE_MIN_LINE_CHARS_DEFAULT = 2
TRAINEDDATA = "traineddata"                    # the model kind that shares one tessdata directory

# A model file is data. A file name with a code suffix is refused, so nothing a pack ships can be
# picked up as code by Python or a platform's library loader (decided 2026-11-05). The list is
# datawheel.CODE_SUFFIXES (tests/test_manifest.py keeps the two equal), plus a versioned .so.
CODE_SUFFIXES: tuple[str, ...] = (".py", ".pyc", ".pyo", ".pyd", ".pyw", ".pyz", ".pth", ".so", ".dylib", ".dll",
                                  ".exe", ".com", ".bat", ".cmd", ".ps1", ".sh", ".msi", ".jar")
_VERSIONED_SO_RE = re.compile(r"\.so(\.[0-9]+)+$")


def is_code_name(name: str) -> bool:
    """Whether a file name ends in a code suffix (compared ignoring case), ``libx.so.1`` included."""
    low = name.casefold()
    return low.endswith(CODE_SUFFIXES) or _VERSIONED_SO_RE.search(low) is not None


# --------------------------------------------------------------------------- the frozen manifest

@dataclass(frozen=True)
class EngineUse:
    id: str
    engine: str
    role: Role
    params: Mapping[str, Any]                  # validated: defaults filled in
    models: tuple[str, ...]                    # asset references, "en:" for the base pack


@dataclass(frozen=True)
class RuleUse:
    rule: str
    version: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class StepUse:
    name: str
    version: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ReaderSet:
    engines: tuple[EngineUse, ...]
    pick: RuleUse
    fallback: Literal["checker", "none"]


@dataclass(frozen=True)
class Asset:
    key: str
    file: str
    engine: str
    kind: str
    bytes: int
    sha256: str
    source: str
    licence: str
    notice: str | None


@dataclass(frozen=True)
class FontAsset:
    key: str
    file: str
    covers: tuple[str, ...]
    bytes: int
    sha256: str
    source: str
    licence: str


@dataclass(frozen=True)
class CanaryAsset:
    file: str
    bytes: int
    sha256: str
    dpi: int
    must_find: tuple[str, ...]


@dataclass(frozen=True)
class MustSurvive:
    name: str
    chars: str
    unit: Literal["word", "trigram", "char"]
    floor: float


@dataclass(frozen=True)
class Probe:
    use: EngineUse
    min_chars: int
    min_conf: float
    min_line_chars: int


@dataclass(frozen=True)
class Detect:
    osd_scripts: tuple[str, ...]
    langid_codes: tuple[str, ...]
    close_group: tuple[str, ...]
    function_words: frozenset[str]
    rivals: frozenset[str]
    fw_min_hits: int
    fw_ratio: float
    probe: Probe | None


@dataclass(frozen=True)
class Manifest:
    schema: int
    id: str
    version: str
    requires_core: str
    base: bool
    title: str
    languages: tuple[dict[str, str], ...]
    scripts: tuple[str, ...]
    detect: Detect
    readers: Mapping[str, ReaderSet]
    normalize: tuple[str, ...]
    repairs: tuple[StepUse, ...]
    fills: tuple[StepUse, ...]
    models: tuple[Asset, ...]
    fonts: tuple[FontAsset, ...]
    canary: CanaryAsset | None
    must_survive: tuple[MustSurvive, ...]
    bench: Mapping[str, Any]
    raw_sha256: str

    def readers_for(self, platform_key: str) -> ReaderSet:
        """The reader set for this platform: its own entry, else ``default``."""
        rs = self.readers.get(platform_key)
        return rs if rs is not None else self.readers["default"]


# --------------------------------------------------------------------------- public entry points

def check(data: Mapping[str, Any], *, base: Manifest | None = None, core_version: str = __version__) -> list[str]:
    """Every problem with a parsed manifest, each starting with its JSON path; [] when it is valid.

    ``base`` is the base pack's manifest. A non-base pack needs it to resolve ``en:`` model
    references and to keep Tesseract file names unique against the base pack.
    """
    return _merge(_schema_problems(data), _Semantic(data, base, core_version).run())


def parse(data: bytes, *, base: Manifest | None = None, core_version: str = __version__) -> Manifest:
    """The manifest in ``data`` (UTF-8 JSON bytes), checked; PackError("invalid_manifest") listing every problem."""
    raw = _load_json(data, "manifest", "invalid_manifest", MAX_MANIFEST_BYTES)
    sem = _Semantic(raw, base, core_version)
    problems = _merge(_schema_problems(raw), sem.run())
    if problems:
        pack = raw.get("id") if isinstance(raw.get("id"), str) else None
        label = f"pack {pack!r}" if pack and _ID_RE.fullmatch(pack) else "the pack manifest"
        raise PackError("invalid_manifest",
                        f"{label} is invalid ({len(problems)} problem{'s' if len(problems) != 1 else ''}): "
                        + "; ".join(problems),
                        problems=problems, pack=pack)
    return sem.build(hashlib.sha256(bytes(data)).hexdigest())


# --------------------------------------------------------------------------- JSON input

_ID_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]{2,8})?")


def _load_json(data: Any, what: str, code: str, cap: int) -> dict[str, Any]:
    """Strict JSON: bytes only, UTF-8 without BOM, no duplicate keys, no NaN or Infinity, an object at the top."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PackError(code, f"the {what} must be given as bytes, not {type(data).__name__}", problems=[])
    data = bytes(data)
    if len(data) > cap:
        raise PackError(code, f"the {what} is {len(data)} bytes; at most {cap}", problems=[])
    if data.startswith(b"\xef\xbb\xbf"):
        raise PackError(code, f"the {what} starts with a byte-order mark; write plain UTF-8", problems=[])
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackError(code, f"the {what} isn't UTF-8 ({exc.reason} at byte {exc.start})", problems=[]) from None

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"the field {k!r} appears twice in one object")
            out[k] = v
        return out

    def no_constant(name: str) -> Any:
        raise ValueError(f"{name} is not JSON")

    try:
        raw = json.loads(text, object_pairs_hook=pairs, parse_constant=no_constant)
    except (ValueError, RecursionError) as exc:
        raise PackError(code, f"the {what} isn't valid JSON: {exc}", problems=[]) from None
    if not isinstance(raw, dict):
        raise PackError(code, f"the {what} must be a JSON object, not {_type_name(raw)}", problems=[])
    return raw


# --------------------------------------------------------------------------- the schema walker

_SCHEMA_KEYWORDS = frozenset({
    "$schema", "$id", "title", "$defs", "$ref", "type", "const", "enum", "pattern", "minLength", "maxLength",
    "minItems", "maxItems", "uniqueItems", "items", "required", "additionalProperties", "properties",
    "propertyNames", "maxProperties", "minimum", "maximum",
})


class _Schema:
    """A validator for the subset of draft 2020-12 the pack schemas use; every problem gets its path."""

    def __init__(self, root: Mapping[str, Any]) -> None:
        self.root = root
        self._patterns: dict[str, re.Pattern[str]] = {}
        self._audit(root, "#")

    def _audit(self, node: Any, where: str) -> None:
        if not isinstance(node, Mapping):
            raise ValueError(f"schema node {where} is not an object")
        unknown = set(node) - _SCHEMA_KEYWORDS
        if unknown:
            raise ValueError(f"schema node {where} uses keywords this loader doesn't implement: {sorted(unknown)}")
        for key in ("items", "propertyNames"):
            if key in node:
                self._audit(node[key], f"{where}/{key}")
        if isinstance(node.get("additionalProperties"), Mapping):
            self._audit(node["additionalProperties"], f"{where}/additionalProperties")
        for group in ("properties", "$defs"):
            for name, sub in (node.get(group) or {}).items():
                self._audit(sub, f"{where}/{group}/{name}")
        if "$ref" in node:
            self._resolve(node["$ref"])
        if "pattern" in node:
            self._pattern(node["pattern"])

    def _resolve(self, ref: str) -> Mapping[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"only local $refs are supported, not {ref!r}")
        node: Any = self.root
        for part in ref[2:].split("/"):
            node = node[part]
        return node

    def _pattern(self, pattern: str) -> re.Pattern[str]:
        compiled = self._patterns.get(pattern)
        if compiled is None:
            compiled = self._patterns[pattern] = re.compile(pattern)
        return compiled

    def _matches(self, pattern: str, value: str) -> bool:
        compiled = self._pattern(pattern)
        if pattern.startswith("^") and pattern.endswith("$") and not pattern.endswith("\\$"):
            return compiled.fullmatch(value) is not None
        return compiled.search(value) is not None

    def problems(self, value: Any, schema: Mapping[str, Any] | None = None, path: str = "$") -> list[str]:
        out: list[str] = []
        self._walk(value, self.root if schema is None else schema, path, out)
        return out

    def _walk(self, v: Any, s: Mapping[str, Any], path: str, out: list[str]) -> None:
        if "$ref" in s:
            self._walk(v, self._resolve(s["$ref"]), path, out)
        if "type" in s:
            types = s["type"] if isinstance(s["type"], list) else [s["type"]]
            if not any(_is_type(v, t) for t in types):
                out.append(f"{path}: expected {' or '.join(_ARTICLE[t] for t in types)}, got {_type_name(v)}")
                return
        if "const" in s and not _json_eq(v, s["const"]):
            out.append(f"{path}: must be {json.dumps(s['const'])}")
        if "enum" in s and not any(_json_eq(v, e) for e in s["enum"]):
            out.append(f"{path}: {_dump(v)} isn't one of {', '.join(_dump(e) for e in s['enum'])}")
        if isinstance(v, str):
            if "pattern" in s and not self._matches(s["pattern"], v):
                out.append(f"{path}: {_dump(v)} doesn't match {s['pattern']}")
            if "minLength" in s and len(v) < s["minLength"]:
                out.append(f"{path}: shorter than {s['minLength']} character{'s' if s['minLength'] != 1 else ''}")
            if "maxLength" in s and len(v) > s["maxLength"]:
                out.append(f"{path}: longer than {s['maxLength']} characters")
        if _is_type(v, "number"):
            if "minimum" in s and v < s["minimum"]:
                out.append(f"{path}: must be at least {_num(s['minimum'])}")
            if "maximum" in s and v > s["maximum"]:
                out.append(f"{path}: must be at most {_num(s['maximum'])}")
        if _is_type(v, "array"):
            self._array(v, s, path, out)
        if _is_type(v, "object"):
            self._object(v, s, path, out)

    def _array(self, v: Sequence[Any], s: Mapping[str, Any], path: str, out: list[str]) -> None:
        if "minItems" in s and len(v) < s["minItems"]:
            out.append(f"{path}: has {len(v)} item{'s' if len(v) != 1 else ''}; at least {s['minItems']}")
        if "maxItems" in s and len(v) > s["maxItems"]:
            out.append(f"{path}: has {len(v)} items; at most {s['maxItems']}")
        if s.get("uniqueItems"):
            for j in range(len(v)):
                for i in range(j):
                    if _json_eq(v[i], v[j]):
                        out.append(f"{path}[{j}]: repeats item {i}")
                        break
        if "items" in s:
            for i, item in enumerate(v):
                self._walk(item, s["items"], f"{path}[{i}]", out)

    def _object(self, v: Mapping[str, Any], s: Mapping[str, Any], path: str, out: list[str]) -> None:
        if "maxProperties" in s and len(v) > s["maxProperties"]:
            out.append(f"{path}: has {len(v)} fields; at most {s['maxProperties']}")
        for key in s.get("required", ()):
            if key not in v:
                out.append(f"{_join(path, key)}: required field is missing")
        props = s.get("properties", {})
        extra = s.get("additionalProperties", True)
        for key, item in v.items():
            kpath = _join(path, key)
            if "propertyNames" in s:
                for p in self.problems(key, s["propertyNames"], kpath):
                    out.append(p.replace(f"{kpath}: ", f"{kpath}: the field name ", 1))
            if key in props:
                self._walk(item, props[key], kpath, out)
            elif extra is False:
                allowed = ", ".join(props) if props else "none"
                out.append(f"{kpath}: unknown field (allowed here: {allowed})")
            elif isinstance(extra, Mapping):
                self._walk(item, extra, kpath, out)


@functools.lru_cache(maxsize=1)
def _manifest_schema() -> _Schema:
    text = resources.files(__package__).joinpath("manifest.schema.json").read_text(encoding="utf-8")
    return _Schema(json.loads(text))


_CODE_SUFFIX_PROBLEM = "ends in a code suffix"


def _merge(schema_problems: list[str], loader_problems: list[str]) -> list[str]:
    """Both lists, each problem once. The schema's ``noCodeSuffix`` pattern names the six suffixes
    §11.2 lists; the loader's ``CODE_SUFFIXES`` is a deliberate superset (decided 2026-11-05), so a
    name such as ``tha.sh`` or ``libtha.so.1`` passes the schema and is refused here. Where both fire
    on one path, the loader's plain words stand and the schema's regex text is dropped."""
    pattern = (_manifest_schema().root.get("$defs", {}).get("noCodeSuffix") or {}).get("pattern")
    worded = {p.split(": ", 1)[0] for p in loader_problems if _CODE_SUFFIX_PROBLEM in p}
    if pattern and worded:
        schema_problems = [p for p in schema_problems
                           if not (p.endswith(f" doesn't match {pattern}") and p.split(": ", 1)[0] in worded)]
    return _dedupe(schema_problems + loader_problems)


@functools.lru_cache(maxsize=1)
def _schema_refuses_blank_must_find() -> bool:
    """Whether the schema itself already refuses a whitespace-only ``must_find`` string, so the
    loader's own check would only repeat it."""
    schema = _manifest_schema()
    try:
        items = schema.root["$defs"]["canary"]["properties"]["must_find"]["items"]
    except (KeyError, TypeError):
        return False
    return all(schema.problems(blank, items) for blank in (" ", "\t\n", "\u3000"))


def _schema_problems(data: Any) -> list[str]:
    """The problems jsonschema would report against manifest.schema.json (in this loader's words)."""
    return _manifest_schema().problems(data)


_ARTICLE = {"object": "an object", "array": "an array", "string": "a string", "integer": "an integer",
            "number": "a number", "boolean": "true or false", "null": "null"}


def _is_type(v: Any, t: str) -> bool:
    if t == "object":
        return isinstance(v, Mapping)
    if t == "array":
        return isinstance(v, (list, tuple))
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "null":
        return v is None
    if isinstance(v, bool):
        return False
    if t == "integer":
        return isinstance(v, int) or (isinstance(v, float) and math.isfinite(v) and v.is_integer())
    if t == "number":
        return isinstance(v, int) or (isinstance(v, float) and math.isfinite(v))
    raise ValueError(f"unknown JSON type {t!r}")


def _json_eq(a: Any, b: Any) -> bool:
    """JSON equality: booleans never equal numbers, 1 equals 1.0, containers compare deeply."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if _is_type(a, "number") and _is_type(b, "number"):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_json_eq(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


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


def _dump(v: Any) -> str:
    try:
        text = json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(v)
    return text if len(text) <= 80 else text[:77] + "..."


def _num(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def _join(path: str, key: Any) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[{_dump(key) if isinstance(key, str) else repr(key)}]"


def _dedupe(problems: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in problems:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _semver(text: Any) -> tuple[int, int, int] | None:
    m = re.match(r"(?:>=)?(\d+)\.(\d+)\.(\d+)", text) if isinstance(text, str) else None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def source_problems(kind: Literal["model", "font"], source: Any) -> list[str]:
    """Why a model or font ``source`` URL is refused ([] when it is allowed)."""
    if not isinstance(source, str):
        return ["must be a URL string"]
    out: list[str] = []
    if "?" in source:
        out.append("has a query string")
    if "#" in source:
        out.append("has a fragment")
    if ".." in source:
        out.append("contains '..'")
    if SOURCE_PATTERNS[kind].fullmatch(source) is None:
        out.append("isn't from an allowed source (see SOURCE_PATTERNS)")
    return out


# --------------------------------------------------------------------------- the vocabulary checks

def _obj(v: Any) -> Mapping[str, Any] | None:
    return v if isinstance(v, Mapping) else None


def _arr(v: Any) -> Sequence[Any]:
    return v if isinstance(v, (list, tuple)) else ()


def _strs(v: Any) -> list[str]:
    return [x for x in _arr(v) if isinstance(x, str)]


def _freeze(v: Any) -> Any:
    if isinstance(v, Mapping):
        return MappingProxyType({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


@dataclass(frozen=True)
class _Ref:
    engine: str
    kind: str


class _Semantic:
    """The checks beyond the schema. Defensive: a malformed part (already reported by the schema
    walk) is skipped, never a crash, so every problem in the manifest is listed in one pass."""

    def __init__(self, data: Mapping[str, Any], base: Manifest | None, core_version: str) -> None:
        self.data = data if isinstance(data, Mapping) else {}
        self.base = base
        self.core_version = core_version
        self.problems: list[str] = []
        self.params: dict[str, dict[str, Any]] = {}       # path -> validated params, for build()
        self._engines: dict[str, Any] = {}

    # ---- helpers

    def add(self, path: str, message: str) -> None:
        self.problems.append(f"{path}: {message}")

    def engine(self, name: Any, path: str) -> Any | None:
        if not isinstance(name, str):
            return None
        if name in self._engines:
            return self._engines[name]
        from ..ocr import engines
        member: Any = None
        try:
            member = engines.get(name)
        except PackError as exc:
            self.add(path, exc.message)
        except SetupError as exc:
            self.add(path, f"engine {name!r} can't be loaded by this core: {exc.message}")
        self._engines[name] = member
        return member

    def is_base(self) -> bool:
        return self.data.get("base") is True

    # ---- the run

    def run(self) -> list[str]:
        from .. import platforms
        from ..ocr import fills, scripts, textfix

        self.platform_keys = frozenset(p.key for p in platforms.PLATFORMS)
        self.supported = tuple(p.key for p in platforms.PLATFORMS if p.supported)
        self.scripts = frozenset(scripts.SCRIPTS)
        self.fills = fills.FILLS
        self.repairs = textfix.REPAIRS
        self.normalizers = textfix.NORMALIZERS

        d = self.data
        self._version_and_base()
        self._names_known("$.scripts", d.get("scripts"), "script")
        self._languages()
        self._assets()
        self._detect()
        self._readers()
        self._normalize()
        self._steps("repairs")
        self._steps("fills")
        self._unique("$.must_survive", d.get("must_survive"), "name")
        self._canary()
        bench = _obj(d.get("bench"))
        for key in (_obj(bench.get("platforms")) or {}) if bench else {}:
            if key not in self.platform_keys:
                self.add(_join("$.bench.platforms", key), f"{key!r} is not a platform this core knows "
                                                          f"({', '.join(sorted(self.platform_keys))})")
        return self.problems

    def _version_and_base(self) -> None:
        d = self.data
        schema = d.get("schema")
        if _is_type(schema, "integer") and int(schema) not in SCHEMA_VERSIONS:
            self.add("$.schema", f"this core reads manifest schema {', '.join(map(str, SCHEMA_VERSIONS))}")
        need, have = _semver(d.get("requires_core")), _semver(self.core_version)
        if need is not None and have is not None and need > have:
            self.add("$.requires_core", f"this pack needs pdfintake {'.'.join(map(str, need))} or newer; "
                                        f"this is {self.core_version}")
        pack_id = d.get("id")
        if self.is_base() and pack_id != BASE_PACK_ID:
            self.add("$.base", f"only the base pack ({BASE_PACK_ID!r}) may set base: true")
        if pack_id == BASE_PACK_ID and not self.is_base():
            self.add("$.base", f"the {BASE_PACK_ID!r} pack is the base pack and must set base: true")
        if not self.is_base() and self.base is not None and not self.base.base:
            self.add("$", f"the base manifest given for this check is pack {self.base.id!r}, which isn't the base pack")

    def _canary(self) -> None:
        """Every ``must_find`` string holds a character other than whitespace: canary.py removes
        whitespace before it compares, so a blank string would find nothing (decided 2026-11-05)."""
        canary = _obj(self.data.get("canary"))
        if canary is None or _schema_refuses_blank_must_find():
            return
        for i, item in enumerate(_arr(canary.get("must_find"))):
            if isinstance(item, str) and item and not item.split():
                self.add(f"$.canary.must_find[{i}]", "must hold a character other than whitespace")

    def _names_known(self, path: str, values: Any, what: str) -> None:
        for i, name in enumerate(_arr(values)):
            if isinstance(name, str) and name not in self.scripts:
                self.add(f"{path}[{i}]", f"{name!r} is not a {what} this core knows")

    def _unique(self, path: str, items: Any, field: str, *, fold: bool = False) -> set[str]:
        """The values of ``field`` across ``items``, each repeat reported. With ``fold`` (file
        names), values are compared casefolded and the folded values are returned."""
        seen: dict[str, str] = {}
        for i, item in enumerate(_arr(items)):
            value = (_obj(item) or {}).get(field)
            if not isinstance(value, str):
                continue
            key = value.casefold() if fold else value
            first = seen.get(key)
            if first == value:
                self.add(f"{path}[{i}].{field}", f"{value!r} is used twice")
            elif first is not None:
                self.add(f"{path}[{i}].{field}", f"{value!r} is used twice, as {first!r} before it "
                                                 "(file names are compared ignoring case)")
            seen.setdefault(key, value)
        return set(seen)

    def _languages(self) -> None:
        langs = _arr(self.data.get("languages"))
        self._unique("$.languages", langs, "tag")
        roles = [(_obj(x) or {}).get("role") for x in langs]
        if langs and "primary" not in roles:
            self.add("$.languages", "no language has role primary")

    # ---- models and fonts

    def _assets(self) -> None:
        d = self.data
        models = _arr(d.get("models"))
        self._unique("$.models", models, "key")
        self._unique("$.models", models, "file", fold=True)
        base_tessdata = {a.file.casefold(): a.file for a in self.base.models
                         if a.kind == TRAINEDDATA} if self.base is not None and not self.is_base() else {}
        self.local: dict[str, _Ref] = {}
        for i, m in enumerate(models):
            m = _obj(m)
            if m is None:
                continue
            path = f"$.models[{i}]"
            for reason in source_problems("model", m.get("source")) if "source" in m else ():
                if not reason.startswith("isn't from"):
                    self.add(f"{path}.source", reason)
            eng = self.engine(m.get("engine"), f"{path}.engine")
            kind = m.get("kind")
            if eng is not None and isinstance(kind, str) and kind not in eng.MODEL_KINDS:
                self.add(f"{path}.kind", f"engine {eng.name!r} takes no {kind!r} model "
                                         f"(it takes: {', '.join(sorted(eng.MODEL_KINDS)) or 'none'})")
            if isinstance(m.get("key"), str) and isinstance(m.get("engine"), str) and isinstance(kind, str):
                self.local.setdefault(m["key"], _Ref(m["engine"], kind))
            file = m.get("file")
            if isinstance(file, str) and is_code_name(file):
                self.add(f"{path}.file", f"{file!r} {_CODE_SUFFIX_PROBLEM}; a model file is data, so its name "
                                         f"never ends in {', '.join(CODE_SUFFIXES)} or a versioned .so")
            clash = base_tessdata.get(file.casefold()) if kind == TRAINEDDATA and isinstance(file, str) else None
            if clash is not None:
                same = "" if clash == file else f" ({clash!r} there; file names are compared ignoring case)"
                self.add(f"{path}.file", f"{file!r} is a Tesseract model of the base pack{same}; Tesseract "
                                         "models share one directory, so their file names must be unique")
        self.base_refs: dict[str, _Ref] = {}
        if self.base is not None and not self.is_base():
            self.base_refs = {BASE_PREFIX + a.key: _Ref(a.engine, a.kind) for a in self.base.models}
        fonts = _arr(d.get("fonts"))
        self._unique("$.fonts", fonts, "key")
        self._unique("$.fonts", fonts, "file", fold=True)
        for i, f in enumerate(fonts):
            f = _obj(f)
            if f is None:
                continue
            for reason in source_problems("font", f.get("source")) if "source" in f else ():
                if not reason.startswith("isn't from"):
                    self.add(f"$.fonts[{i}].source", reason)
            self._names_known(f"$.fonts[{i}].covers", f.get("covers"), "script")

    def _resolve(self, ref: str, path: str) -> _Ref | None:
        if ref.startswith(BASE_PREFIX):
            if self.is_base():
                self.add(path, f"the base pack names its own models without {BASE_PREFIX!r}")
                return None
            if self.base is None:
                self.add(path, f"{ref!r} names a model of the base pack, but no base manifest was given to check it")
                return None
            found = self.base_refs.get(ref)
            if found is None:
                self.add(path, f"the base pack has no model {ref[len(BASE_PREFIX):]!r}")
            return found
        found = self.local.get(ref)
        if found is None:
            self.add(path, f"this pack has no model {ref!r}")
        return found

    def _ctx(self, use_ids: frozenset[str], where: str) -> ValidationContext:
        kinds = {k: r.kind for k, r in self.local.items()}
        kinds.update({k: r.kind for k, r in self.base_refs.items()})
        return ValidationContext(use_ids, frozenset(kinds), kinds, self.scripts, where)

    # ---- engine uses, reader sets, probe

    def _use(self, use: Mapping[str, Any], path: str, use_ids: frozenset[str]) -> None:
        name = use.get("engine")
        eng = self.engine(name, f"{path}.engine")
        refs = _strs(use.get("models"))
        seen: set[str] = set()
        for j, ref in enumerate(_arr(use.get("models"))):
            if not isinstance(ref, str):
                continue
            mpath = f"{path}.models[{j}]"
            if ref in seen:
                self.add(mpath, f"{ref!r} is listed twice")
            seen.add(ref)
            found = self._resolve(ref, mpath)
            if found is not None and isinstance(name, str) and found.engine != name:
                self.add(mpath, f"{ref!r} is a model of engine {found.engine!r}, not of {name!r}")
        params = _obj(use.get("params"))
        if eng is None or params is None:
            return
        where = f"{path}.params"
        try:
            self.params[where] = eng.PARAMS.validate(params, self._ctx(use_ids, where))
        except PackError as exc:
            self.problems.extend(exc.detail.get("problems") or [exc.message])
            return
        for p in eng.PARAMS:
            if p.kind == "asset_ref" and isinstance(params.get(p.name), str) and params[p.name] not in refs:
                self.add(f"{where}.{p.name}", f"{params[p.name]!r} isn't among this use's models")

    def _readers(self) -> None:
        readers = _obj(self.data.get("readers")) or {}
        self.reader_ids: dict[str, frozenset[str]] = {}
        for key, rs in readers.items():
            rpath = _join("$.readers", key)
            if key != "default" and key not in self.platform_keys:
                self.add(rpath, f"{key!r} is not a platform this core knows ({', '.join(sorted(self.platform_keys))})")
            rs = _obj(rs)
            if rs is None:
                continue
            uses = [(i, u) for i, u in enumerate(_arr(rs.get("engines"))) if isinstance(u, Mapping)]
            ids: list[str] = [u["id"] for _, u in uses if isinstance(u.get("id"), str)]
            self._unique(f"{rpath}.engines", rs.get("engines"), "id")
            use_ids = frozenset(ids)
            self.reader_ids[key] = use_ids
            roles = {u["id"]: u.get("role") for _, u in uses if isinstance(u.get("id"), str)}
            role_list = [u.get("role") for _, u in uses]
            if uses and "reader" not in role_list:
                self.add(f"{rpath}.engines", "no engine use has role reader")
            if role_list.count("checker") > 1:
                self.add(f"{rpath}.engines", "more than one engine use has role checker")
            for i, u in uses:
                upath = f"{rpath}.engines[{i}]"
                if u.get("role") == "probe":
                    self.add(f"{upath}.role", "a probe belongs in detect.probe, not in a reader set")
                self._use(u, upath, use_ids)
                eng = self._engines.get(u.get("engine")) if isinstance(u.get("engine"), str) else None
                if eng is None:
                    continue
                if key == "default":
                    missing = [p for p in self.supported if p not in eng.platforms]
                    if missing:
                        self.add(f"{upath}.engine", f"engine {eng.name!r} doesn't run on {', '.join(missing)}; "
                                                    "the default reader set must work on every supported platform")
                elif key in self.platform_keys and key not in eng.platforms:
                    self.add(f"{upath}.engine", f"engine {eng.name!r} doesn't run on {key}")
            if rs.get("fallback") == "checker" and uses and "checker" not in role_list:
                self.add(f"{rpath}.fallback", "fallback is checker, but no engine use has role checker")
            self._pick(_obj(rs.get("pick")), f"{rpath}.pick", use_ids, roles)

    def _pick(self, pick: Mapping[str, Any] | None, path: str, use_ids: frozenset[str],
              roles: Mapping[str, Any]) -> None:
        if pick is None or not isinstance(pick.get("rule"), str) or not _is_type(pick.get("version"), "integer"):
            return
        from ..ocr import rules
        try:
            rule = rules.get(pick["rule"], int(pick["version"]))
        except PackError as exc:
            self.add(f"{path}.rule", exc.message)
            return
        except SetupError as exc:
            self.add(f"{path}.rule", f"pick rule {pick['rule']!r} can't be loaded by this core: {exc.message}")
            return
        params = _obj(pick.get("params"))
        if params is not None:
            where = f"{path}.params"
            try:
                self.params[where] = rule.PARAMS.validate(params, self._ctx(use_ids, where))
            except PackError as exc:
                self.problems.extend(exc.detail.get("problems") or [exc.message])
            for j, uid in enumerate(_arr(params.get("corroborators"))):
                if isinstance(uid, str) and roles.get(uid) == "reader":
                    self.add(f"{where}.corroborators[{j}]", f"{uid!r} is a reader; a corroborator never competes, "
                                                            "so it must be a checker")
        scripts_ok = getattr(rule, "scripts", None)
        pack_scripts = _strs(self.data.get("scripts"))
        if scripts_ok is not None and pack_scripts and pack_scripts[0] not in scripts_ok:
            self.add(f"{path}.rule", f"pick rule {rule.name!r} serves {', '.join(sorted(scripts_ok))}, "
                                     f"not this pack's primary script {pack_scripts[0]}")

    def _detect(self) -> None:
        detect = _obj(self.data.get("detect"))
        if detect is None:
            return
        self._names_known("$.detect.osd_scripts", detect.get("osd_scripts"), "script")
        probe = _obj(detect.get("probe"))
        use = _obj(probe.get("use")) if probe else None
        if use is not None:
            path = "$.detect.probe.use"
            if use.get("role") not in (None, "probe"):
                self.add(f"{path}.role", "the detect probe's role must be probe")
            ids = frozenset([use["id"]]) if isinstance(use.get("id"), str) else frozenset()
            self._use(use, path, ids)
            # A probe runs during detection on every platform and has no per-platform alternative,
            # so it is held to the default reader set's coverage rule.
            eng = self._engines.get(use.get("engine")) if isinstance(use.get("engine"), str) else None
            if eng is not None:
                missing = [p for p in self.supported if p not in eng.platforms]
                if missing:
                    self.add(f"{path}.engine", f"engine {eng.name!r} doesn't run on {', '.join(missing)}; "
                                               "the detect probe must work on every supported platform")

    def _normalize(self) -> None:
        for i, name in enumerate(_arr(self.data.get("normalize"))):
            if isinstance(name, str) and name not in self.normalizers:
                self.add(f"$.normalize[{i}]", f"{name!r} is not a normaliser this core knows "
                                              f"({', '.join(sorted(self.normalizers))})")

    def _steps(self, group: Literal["repairs", "fills"]) -> None:
        table = self.repairs if group == "repairs" else self.fills
        what = "repair" if group == "repairs" else "fill"
        steps = _arr(self.data.get(group))
        self._unique(f"$.{group}", steps, "name")
        pack_scripts = set(_strs(self.data.get("scripts")))
        for i, st in enumerate(steps):
            st = _obj(st)
            if st is None or not isinstance(st.get("name"), str) or not _is_type(st.get("version"), "integer"):
                continue
            path = f"$.{group}[{i}]"
            key = (st["name"], int(st["version"]))
            member = table.get(key)
            if member is None:
                versions = sorted(v for n, v in table if n == key[0])
                if versions:
                    self.add(f"{path}.version", f"{what} {key[0]!r} has no version {key[1]}; this core registers "
                                                f"version {', '.join(map(str, versions))}")
                else:
                    known = ", ".join(f"{n} v{v}" for n, v in table) or "none"
                    self.add(f"{path}.name", f"{key[0]!r} is not a {what} this core knows ({known})")
                continue
            ok = getattr(member, "scripts_ok", None)
            if ok is not None and pack_scripts - set(ok):
                self.add(f"{path}.name", f"{what} {key[0]!r} serves only {', '.join(sorted(ok))}; this pack's "
                                         f"scripts include {', '.join(sorted(pack_scripts - set(ok)))}")
            raw = st.get("params", {})
            if not isinstance(raw, Mapping):
                continue
            where = f"{path}.params"
            sets = self.reader_ids if group == "repairs" else {"": frozenset()}
            for rkey, ids in (sets.items() if sets else [("", frozenset())]):
                try:
                    self.params.setdefault(where, member.PARAMS.validate(raw, self._ctx(ids, where)))
                except PackError as exc:
                    found = exc.detail.get("problems") or [exc.message]
                    note = f" (in reader set {rkey})" if rkey and len(sets) > 1 else ""
                    self.problems.extend(p + note for p in found)

    # ---- the frozen result (only called after a clean run)

    def build(self, raw_sha256: str) -> Manifest:
        d = self.data

        def params_at(where: str, raw: Any) -> Mapping[str, Any]:
            return _freeze(self.params.get(where, dict(raw or {})))

        def use(u: Mapping[str, Any], path: str) -> EngineUse:
            return EngineUse(u["id"], u["engine"], u["role"], params_at(f"{path}.params", u.get("params")),
                             tuple(u["models"]))

        readers: dict[str, ReaderSet] = {}
        for key, rs in d["readers"].items():
            rpath = _join("$.readers", key)
            engines = tuple(use(u, f"{rpath}.engines[{i}]") for i, u in enumerate(rs["engines"]))
            pk = rs["pick"]
            readers[key] = ReaderSet(engines, RuleUse(pk["rule"], int(pk["version"]),
                                                      params_at(f"{rpath}.pick.params", pk.get("params"))),
                                     rs["fallback"])

        det = d["detect"]
        langid = det.get("langid") or {}
        fw = det.get("function_words") or {}
        probe = det.get("probe")
        detect = Detect(
            osd_scripts=tuple(det["osd_scripts"]),
            langid_codes=tuple(langid.get("codes", ())),
            close_group=tuple(langid.get("close_group", ())),
            function_words=frozenset(fw.get("words", ())),
            rivals=frozenset(fw.get("rivals", ())),
            fw_min_hits=int(fw.get("min_hits", FW_MIN_HITS_DEFAULT)),
            fw_ratio=float(fw.get("ratio", FW_RATIO_DEFAULT)),
            probe=None if probe is None else Probe(
                use(probe["use"], "$.detect.probe.use"), int(probe["min_chars"]), float(probe["min_conf"]),
                int(probe.get("min_line_chars", PROBE_MIN_LINE_CHARS_DEFAULT))),
        )

        def steps(group: str) -> tuple[StepUse, ...]:
            return tuple(StepUse(s["name"], int(s["version"]), params_at(f"$.{group}[{i}].params", s.get("params")))
                         for i, s in enumerate(d[group]))

        canary = d.get("canary")
        return Manifest(
            schema=int(d["schema"]), id=d["id"], version=d["version"], requires_core=d["requires_core"],
            base=d.get("base") is True, title=d["title"],
            languages=tuple(dict(x) for x in d["languages"]), scripts=tuple(d["scripts"]),
            detect=detect, readers=MappingProxyType(readers), normalize=tuple(d["normalize"]),
            repairs=steps("repairs"), fills=steps("fills"),
            models=tuple(Asset(m["key"], m["file"], m["engine"], m["kind"], int(m["bytes"]), m["sha256"],
                               m["source"], m["licence"], m.get("notice")) for m in d["models"]),
            fonts=tuple(FontAsset(f["key"], f["file"], tuple(f["covers"]), int(f["bytes"]), f["sha256"],
                                  f["source"], f["licence"]) for f in d["fonts"]),
            canary=None if canary is None else CanaryAsset(canary["file"], int(canary["bytes"]), canary["sha256"],
                                                           int(canary["dpi"]), tuple(canary["must_find"])),
            must_survive=tuple(MustSurvive(m["name"], m["chars"], m["unit"], float(m["floor"]))
                               for m in d["must_survive"]),
            bench=_freeze(d["bench"]),
            raw_sha256=raw_sha256,
        )
