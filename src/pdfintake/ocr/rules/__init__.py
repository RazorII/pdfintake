"""The pick-rule protocol and the pick-rule registry.

A pick rule chooses one read per page from the reader set's reads. Rules are registered by name
and version: a rule whose behaviour changes gets a new version, and the old version stays
registered while any published pack uses it. The registry is an explicit table resolved lazily;
nothing is discovered and no pack data drives an import. Every registered rule must pass
``tests/contract/test_rule_contract.py``.

Warnings. A rule reports a warning the pipeline can't work out on its own in ``Pick.warnings``,
using only codes in ``PICK_WARNING_CODES`` (each one is also a reader warning in the record's
closed ``WARNING_CODES``). Today that is ``agreement_low``: ``primary_with_floor`` below its floor
with ``fallback="none"`` keeps the primary and adds it. The pipeline copies these codes onto the
page. It derives ``fallback_used`` (``pick.fallback``) and ``no_layer_readers_failed``
(``pick.chosen is None``) itself, so a rule never puts those in ``Pick.warnings``.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from ...errors import PackError, SetupError
from ...params import ParamTable
from ...types import EngineRead, Pick


@runtime_checkable
class PickRule(Protocol):
    name: ClassVar[str]
    version: ClassVar[int]
    PARAMS: ClassVar[ParamTable]
    scripts: ClassVar[frozenset[str] | None]

    def pick(self, reads: Mapping[str, EngineRead], params: Mapping[str, Any], *, fallback: Literal["checker", "none"],
             checker_id: str | None) -> Pick: ...


PICK_WARNING_CODES: frozenset[str] = frozenset({"agreement_low"})

_REGISTRY: dict[tuple[str, int], str] = {
    ("corroborated_trigrams", 1): "pdfintake.ocr.rules.corroborated:RULE_V1",
    ("primary_with_floor", 1):    "pdfintake.ocr.rules.primary:FLOOR_V1",
    ("single", 1):                "pdfintake.ocr.rules.primary:SINGLE_V1",
}

_lock = threading.RLock()
_loaded: dict[tuple[str, int, str], Any] = {}                          # (name, version, "module:attr") -> rule
_failed: dict[tuple[str, int, str], tuple[str, str, dict[str, Any]]] = {}


def names() -> tuple[tuple[str, int], ...]:
    """Every registered (name, version), in registry order."""
    return tuple(_REGISTRY)


def get(name: str, version: int) -> PickRule:
    """The rule registered as (``name``, ``version``), importing its module on first use.

    PackError("unknown_rule") for a name this core doesn't register, PackError("unknown_rule_version")
    for a registered name without that version; SetupError when the module can't provide the rule.
    """
    registered = names()
    versions = sorted(v for n, v in registered if n == name)
    if not versions:
        known = ", ".join(f"{n} v{v}" for n, v in registered)
        raise PackError(
            "unknown_rule",
            f"unknown pick rule {name!r}: this version of pdfintake knows {known}",
            rule=name, version=version, known=[list(k) for k in registered],
        )
    if not isinstance(version, int) or isinstance(version, bool):
        raise PackError("unknown_rule_version",
                        f"pick rule {name!r}: version must be a whole number, not {version!r}",
                        rule=name, version=version, known_versions=versions)
    try:
        target = _REGISTRY[(name, version)]
    except KeyError:
        raise PackError(
            "unknown_rule_version",
            f"pick rule {name!r} has no version {version}; this version of pdfintake registers "
            f"version {', '.join(map(str, versions))}",
            rule=name, version=version, known_versions=versions,
        ) from None
    return _load(name, version, target)


def _load(name: str, version: int, target: Any) -> Any:
    if not isinstance(target, str):
        rule = target                        # a test-only override: the member object itself
        _check_identity(name, version, rule, repr(rule))
        return rule
    key = (name, version, target)
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
            failure = ("rule_import_failed",
                       f"pick rule {name!r} v{version} could not be loaded from {module_name}: {type(exc).__name__}: {exc}",
                       {"rule": name, "version": version, "module": module_name})
            _failed[key] = failure
            raise SetupError(failure[0], failure[1], **failure[2]) from exc
        rule = getattr(module, attr, None) if attr else None
        if rule is None:
            failure = ("rule_import_failed", f"pick rule {name!r} v{version}: {module_name} has no {attr or 'attribute'}",
                       {"rule": name, "version": version, "module": module_name})
            _failed[key] = failure
            raise SetupError(failure[0], failure[1], **failure[2])
        _check_identity(name, version, rule, target)
        _loaded[key] = rule
        return rule


def _check_identity(name: str, version: int, rule: Any, where: str) -> None:
    actual = (getattr(rule, "name", None), getattr(rule, "version", None))
    if actual != (name, version):
        raise SetupError("rule_name_mismatch",
                         f"pick rule registered as {name!r} v{version} ({where}) calls itself {actual[0]!r} v{actual[1]!r}",
                         rule=name, version=version, actual=list(actual))


def _clear_cache() -> None:
    """Tests only: forget every loaded and failed rule module."""
    with _lock:
        _loaded.clear()
        _failed.clear()
