"""Shared data passed between modules: engines -> pick rules -> fills -> the layer writer.

Coordinates. A ``Quad`` is four points in pixels of the shared page render: origin top-left,
y down, clockwise from the top-left corner. Engines convert their own boxes (Vision's normalized
bottom-left boxes, Tesseract's bounding boxes) before returning; boxes are never in points.

JSON. ``to_jsonable`` turns any of these into plain JSON values (dataclass -> object, Path -> str,
tuple -> array). Document text is left out unless the caller passes ``include_text=True``:
each text field is replaced by its character count (``"chars"``), so a record built from these
objects never holds what a page says.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    import PIL.Image

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]   # px of the shared render, origin top-left, y down, clockwise from top-left
Role = Literal["reader", "checker", "probe"]
ReadStatus = Literal["ok", "empty", "failed", "timeout", "unavailable"]
FillKind = Literal["line", "word", "number", "id", "link"]

ROLES: tuple[str, ...] = get_args(Role)
READ_STATUSES: tuple[str, ...] = get_args(ReadStatus)
FILL_KINDS: tuple[str, ...] = get_args(FillKind)


@dataclass(frozen=True)
class PageImage:
    """One rendered page on disk (a PNG with its dpi), the only thing engine workers ever get."""

    index: int
    path: Path
    width: int
    height: int
    dpi: float
    cropbox: tuple[float, float, float, float]            # pt, page user space of the work PDF

    def open(self) -> "PIL.Image.Image":
        """Read the PNG (only the PNG plugin may open it). Never touches pdfium."""
        from PIL import Image

        img = Image.open(self.path, formats=["PNG"])
        try:
            img.load()
        except BaseException:
            img.close()
            raise
        if img.size != (self.width, self.height):
            size = img.size
            img.close()
            raise ValueError(
                f"{self.path}: the PNG is {size[0]}x{size[1]} px but the page image says {self.width}x{self.height}"
            )
        return img


@dataclass(frozen=True)
class Word:
    text: str
    conf: float
    box: Quad


@dataclass(frozen=True)
class Line:
    text: str
    conf: float
    box: Quad
    words: tuple[Word, ...] = ()


@dataclass
class EngineRead:
    use_id: str
    engine: str
    spec: str
    role: Role
    status: ReadStatus
    lines: list[Line]
    seconds: float
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        """The read's text, one line per ``Line``."""
        return "\n".join(line.text for line in self.lines)


@dataclass
class Pick:
    rule: str
    version: int
    chosen: str | None
    lines: list[Line]
    scores: dict[str, float]
    fallback: bool
    reason: str
    # Warning codes the rule decided and the pipeline can't derive from the fields above
    # (ocr.rules.PICK_WARNING_CODES, today only "agreement_low"); the pipeline copies them
    # onto the page. fallback_used and no_layer_readers_failed come from fallback and chosen.
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Fill:
    kind: FillKind
    text: str
    conf: float
    box: Quad
    source: str
    near: str | None = None


@dataclass(frozen=True)
class OsdResult:
    script: str | None
    script_conf: float
    orient_deg: int        # as tesserocr reports it
    orient_conf: float


@dataclass(frozen=True)
class ScriptCall:
    name: str | None
    conf: float
    by: Literal["osd", "vote", "hint", "probe", "trigger", "none"]
    eng_mean_conf: float | None


@dataclass(frozen=True)
class LangCall:
    tag: str
    prob: float
    by: Literal["langid", "function_words", "document", "hint", "forced", "pack_script"]
    windows: int


@dataclass
class PageOCR:
    index: int
    script: ScriptCall
    languages: list[LangCall]
    pack: str | None
    composition: str
    reads: list[EngineRead]
    pick: Pick
    fills: list[Fill]
    repairs: dict[str, int]
    must_survive: dict[str, dict[str, int]]
    warnings: list[str]


@dataclass(frozen=True)
class LayerLine:
    text: str
    box: Quad


@dataclass(frozen=True)
class LayerPage:
    index: int
    render_w: int
    render_h: int
    lines: tuple[LayerLine, ...]


# Fields that hold document text, per class. to_jsonable leaves them out unless asked;
# a "text" field is replaced by "chars", its length.
_TEXT_FIELDS: tuple[tuple[type, frozenset[str]], ...] = (
    (Word, frozenset({"text"})),
    (Line, frozenset({"text"})),
    (Fill, frozenset({"text", "near"})),
    (LayerLine, frozenset({"text"})),
)


def to_jsonable(obj: Any, *, include_text: bool = False) -> Any:
    """Plain JSON values for ``obj``: dataclasses -> dict, Path -> str, tuple -> list, set -> sorted list.

    Document text (``Word.text``, ``Line.text``, ``Fill.text`` and ``Fill.near``, ``LayerLine.text``)
    is replaced by a ``"chars"`` count unless ``include_text`` is true. Bytes and other unknown types
    raise TypeError rather than being dumped by accident.
    """
    return _convert(obj, include_text)


def _hidden_fields(obj: Any) -> frozenset[str]:
    for cls, names in _TEXT_FIELDS:
        if isinstance(obj, cls):
            return names
    return frozenset()


def _convert(obj: Any, include_text: bool) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return _convert(obj.value, include_text)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return float(obj)
    if isinstance(obj, PurePath):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        hidden = frozenset() if include_text else _hidden_fields(obj)
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.name in hidden:
                if f.name == "text":
                    out["chars"] = len(value)
                continue
            out[f.name] = _convert(value, include_text)
        return out
    if isinstance(obj, Mapping):
        return {_key(k): _convert(v, include_text) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v, include_text) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_convert(v, include_text) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"to_jsonable can't convert a {type(obj).__name__}")


def _key(key: Any) -> Any:
    if key is None or isinstance(key, (str, bool)):
        return key
    if isinstance(key, enum.Enum):
        return _key(key.value)
    if isinstance(key, numbers.Integral):
        return int(key)
    if isinstance(key, numbers.Real):
        return float(key)
    if isinstance(key, PurePath):
        return str(key)
    raise TypeError(f"to_jsonable can't use a {type(key).__name__} as an object key")
