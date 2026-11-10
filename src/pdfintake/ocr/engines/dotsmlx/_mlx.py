"""The one place this package imports mlx. Written for pdfintake (MIT); nothing here is ported.

mlx runs only on Apple silicon and isn't a pdfintake dependency, but ``tests/test_hygiene.py``
imports every module in ``src/`` on every OS. So the other modules take ``mx`` and ``nn`` from
here: the real ones when mlx is installed, otherwise stand-ins that are just enough for a module
to import (``nn.Module`` as a base class, ``mx.compile`` as a decorator). Anything else on a
stand-in raises ``ModuleNotFoundError``, and ``load()`` checks ``AVAILABLE`` before it builds
anything, so a missing mlx is a clear error, never a half-working model.
"""

from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    AVAILABLE = True
except ImportError:                      # not Apple silicon, or mlx isn't installed
    AVAILABLE = False

    def _missing(name: str) -> ModuleNotFoundError:
        return ModuleNotFoundError(f"mlx is needed for dots.mocr (mlx.{name}); it runs only on Apple silicon",
                                   name="mlx")

    class _Absent:
        def __init__(self, label: str) -> None:
            self._label = label

        def __getattr__(self, name: str) -> Any:
            raise _missing(f"{self._label}.{name}")

    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _missing("nn.Module")

    class _NN(_Absent):
        Module = _Module

    class _MX(_Absent):
        @staticmethod
        def compile(fun: Any = None, **kwargs: Any) -> Any:
            return fun if fun is not None else (lambda f: f)

    mx = _MX("core")                     # type: ignore[assignment]
    nn = _NN("nn")                       # type: ignore[assignment]

__all__ = ["AVAILABLE", "mx", "nn"]
