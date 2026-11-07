"""Protocols for the registered kinds of core code that aren't engines or pick rules.

Normalisers, repairs, fills and language identifiers live in explicit tables in core
(``ocr/textfix.py``, ``ocr/fills.py``, ``ocr/detect.py``); packs name them, never import them.
Engines and pick rules have their own protocols and registries in ``ocr/engines`` and ``ocr/rules``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from .params import ParamTable
from .types import EngineRead, Fill, Line, OsdResult, PageImage

if TYPE_CHECKING:
    import PIL.Image


@runtime_checkable
class Normalizer(Protocol):
    name: ClassVar[str]

    def __call__(self, text: str) -> str: ...


@runtime_checkable
class Repair(Protocol):
    name: ClassVar[str]
    version: ClassVar[int]
    PARAMS: ClassVar[ParamTable]

    def repair(self, lines: list[Line], checker: EngineRead | None, params: Mapping[str, Any], *,
               languages: Sequence[str] = ()) -> tuple[list[Line], dict[str, int]]: ...


@runtime_checkable
class FillStep(Protocol):
    name: ClassVar[str]
    version: ClassVar[int]
    PARAMS: ClassVar[ParamTable]
    scripts_ok: ClassVar[frozenset[str] | None]

    def fill(self, chosen: Sequence[Line], checker: EngineRead, page: PageImage,
             params: Mapping[str, Any], already: Sequence[Fill]) -> list[Fill]: ...


@runtime_checkable
class LanguageIdentifier(Protocol):
    name: ClassVar[str]

    def classify(self, text: str) -> list[tuple[str, float]]: ...   # ISO 639-1/3 codes, probabilities, best first


@runtime_checkable
class OsdSession(Protocol):
    def detect(self, image: "PIL.Image.Image", dpi: float) -> OsdResult | None: ...   # None on too little text
