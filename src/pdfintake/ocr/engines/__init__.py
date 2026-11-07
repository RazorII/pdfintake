"""The engine protocol and the engine registry.

The registry is an explicit table: no entry points, no plugin discovery, and no import driven by
pack data. A pack names an engine; ``get`` resolves the name through ``_REGISTRY`` and imports the
engine's module only then (``warm_up`` imports them all up front, before ``flow`` installs the
no-subprocess guard). Adding an engine means adding a row here, writing its module, and passing
``tests/contract/test_engine_contract.py``.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

from ...errors import PackError, SetupError
from ...params import ParamTable
from ...types import EngineRead, PageImage, Role

if TYPE_CHECKING:
    from ...langpacks.manifest import Asset


@dataclass(frozen=True)
class Availability:
    ok: bool
    state: Literal["available", "not_on_platform", "missing_model", "unsupported_language", "broken"]
    detail: str


ModelPaths = Mapping[str, Path]        # asset key -> verified file


@dataclass(frozen=True)
class ContractCase:
    """What an engine hands the contract suite so it can be exercised on this machine.

    ``params`` are raw, as a pack would write them (the suite validates them with the engine's
    PARAMS). ``models`` maps asset keys to files and ``model_kinds`` gives each key's manifest kind.
    ``page`` writes a page the engine can read into the folder it is given; without one, the suite
    uses its own synthetic page and doesn't require text. ``unsupported_params``, when given, ask for
    a language the engine can't read, so ``availability`` must answer ``unsupported_language``.
    ``platform_key`` defaults to the platform the suite runs on.
    """

    params: Mapping[str, Any]
    models: ModelPaths = field(default_factory=dict)
    model_kinds: Mapping[str, str] = field(default_factory=dict)
    role: Role = "reader"
    page: Callable[[Path], PageImage] | None = None
    unsupported_params: Mapping[str, Any] | None = None
    platform_key: str | None = None
    timeout_s: float = 120.0


@runtime_checkable
class EngineSession(Protocol):
    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead: ...   # never raises for content; EngineFailure for engine errors

    def close(self) -> None: ...


@runtime_checkable
class Engine(Protocol):
    name: ClassVar[str]
    api: ClassVar[int]
    PARAMS: ClassVar[ParamTable]
    MODEL_KINDS: ClassVar[frozenset[str]]
    platforms: ClassVar[frozenset[str]]
    session_per_thread: ClassVar[bool]

    def version(self) -> str | None: ...

    def availability(self, params: Mapping[str, Any], models: ModelPaths, platform_key: str) -> Availability: ...

    def open(self, use_id: str, role: Role, params: Mapping[str, Any], models: ModelPaths) -> EngineSession: ...

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence["Asset"]) -> list[dict[str, Any]]: ...   # for the kit

    def contract_case(self) -> ContractCase | None: ...   # params + models usable in CI for the contract suite


_REGISTRY: dict[str, str] = {
    "tesseract": "pdfintake.ocr.engines.tesseract:ENGINE",
    "vision":    "pdfintake.ocr.engines.vision:ENGINE",
    "rapidocr":  "pdfintake.ocr.engines.rapid:ENGINE",
}

_lock = threading.RLock()
_loaded: dict[tuple[str, str], Any] = {}                                # (name, "module:attr") -> engine
_failed: dict[tuple[str, str], tuple[str, str, dict[str, Any]]] = {}    # (name, "module:attr") -> (code, message, detail)


def names() -> tuple[str, ...]:
    """Every registered engine name, in registry order."""
    return tuple(_REGISTRY)


def get(name: str) -> Engine:
    """The engine registered as ``name``, importing its module on first use.

    PackError("unknown_engine") for a name this core doesn't register; SetupError when the
    registered module can't be loaded or doesn't provide an engine of that name.
    """
    try:
        target = _REGISTRY[name]
    except (KeyError, TypeError):
        known = names()
        raise PackError(
            "unknown_engine",
            f"unknown engine {name!r}: this version of pdfintake knows {', '.join(known)}",
            engine=name, known=list(known),
        ) from None
    return _load(name, target)


def warm_up() -> None:
    """Import every engine module now. A module that fails is remembered, not retried, and ``get``
    raises its SetupError later; so no engine import ever happens under the no-subprocess guard."""
    for name, target in list(_REGISTRY.items()):
        try:
            _load(name, target)
        except SetupError:
            pass


def _load(name: str, target: Any) -> Any:
    if not isinstance(target, str):
        engine = target                      # a test-only override: the member object itself
        _check_name(name, engine, repr(engine))
        return engine
    key = (name, target)
    with _lock:
        if key in _loaded:
            return _loaded[key]
        if key in _failed:
            code, message, detail = _failed[key]
            raise SetupError(code, message, **detail)
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failure = ("engine_import_failed",
                       f"engine {name!r} could not be loaded from {module_name}: {type(exc).__name__}: {exc}",
                       {"engine": name, "module": module_name})
            _failed[key] = failure
            raise SetupError(failure[0], failure[1], **failure[2]) from exc
        engine = getattr(module, attr or "ENGINE", None)
        if engine is None:
            failure = ("engine_import_failed", f"engine {name!r}: {module_name} has no {attr or 'ENGINE'}",
                       {"engine": name, "module": module_name})
            _failed[key] = failure
            raise SetupError(failure[0], failure[1], **failure[2])
        _check_name(name, engine, target)
        _loaded[key] = engine
        return engine


def _check_name(name: str, engine: Any, where: str) -> None:
    actual = getattr(engine, "name", None)
    if actual != name:
        raise SetupError("engine_name_mismatch",
                         f"engine registered as {name!r} ({where}) calls itself {actual!r}",
                         engine=name, actual=actual)


def _clear_cache() -> None:
    """Tests only: forget every loaded and failed engine module."""
    with _lock:
        _loaded.clear()
        _failed.clear()
