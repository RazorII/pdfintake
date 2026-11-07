"""Exit codes and the exception hierarchy.

Every error pdfintake raises on purpose is a ``PdfIntakeError``. It carries a short machine
``code`` (for example ``"unknown_engine"``), a core-worded ``message`` and keyword ``detail``,
and its class decides the process exit code. The exit-code table in the README and in the
skill must equal the ``EXIT_*`` constants here.

A run never asks (DESIGN §12, decided 2026-11-05): there is no "decision needed" exit. The old
exit 10 is retired; what a person could do next travels as offers on an exit-0 record. With
several records the process exits with the highest-priority code, 2 > 3 > 1 > 4 > 5 > 0.
"""

from __future__ import annotations

from typing import Any

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_SETUP, EXIT_REFUSED, EXIT_STALE = 0, 1, 2, 3, 4, 5
# Kit commands only (DESIGN §3.3): a must-survive or metric floor failed; the re-bench differs from
# the claim beyond tolerance; the privacy guard tripped in `kit private`.
EXIT_KIT_FLOOR, EXIT_KIT_DRIFT, EXIT_KIT_PRIVACY = 6, 7, 8


class PdfIntakeError(Exception):
    """Base class. ``exit_code`` is per class; ``code`` and ``detail`` are per instance."""

    exit_code: int = EXIT_FAILED
    code: str

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail: dict[str, Any] = dict(detail)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.code!r}, {self.message!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        # The default reduce passes only the message to __init__, which needs code and message.
        return (_rebuild, (type(self), self.code, self.message, self.detail))


def _rebuild(cls: type[PdfIntakeError], code: str, message: str, detail: dict[str, Any]) -> PdfIntakeError:
    return cls(code, message, **detail)


class UsageError(PdfIntakeError):
    """Bad arguments, an output that exists or would hit an input, an unknown record id (exit 2)."""

    exit_code = EXIT_USAGE


class SetupError(PdfIntakeError):
    """Base pack missing or broken, unsupported platform, broken model install (exit 3)."""

    exit_code = EXIT_SETUP


class PackError(SetupError):
    """A manifest, index or store problem, or a name no core registry knows (exit 3)."""


class IntegrityError(PackError):
    """A size or hash mismatch; the message names the file (exit 3)."""


class Refused(PdfIntakeError):
    """XFA, detectors disagree, unreadable XMP, a damaged file; nothing written (exit 4).

    Signed, certified, PDF/A, encrypted, counterparty and evidence files get a searchable copy
    instead (DESIGN §10), and a file no saved password opens is skipped, not refused (§12.1).
    """

    exit_code = EXIT_REFUSED


class StaleRecord(PdfIntakeError):
    """The input, output or candidate changed since the record was written (exit 5)."""

    exit_code = EXIT_STALE


class InputChanged(StaleRecord):
    """The input seal broke during a run or before ``apply`` (exit 5)."""


class VerifyFailed(PdfIntakeError):
    """A verify check failed; nothing was written. ``detail["checks"]`` holds the failed Checks (exit 1)."""

    exit_code = EXIT_FAILED

    @property
    def checks(self) -> list[Any]:
        return list(self.detail.get("checks", []))


class LosslessUnavailable(PdfIntakeError):
    """An input pdfintake can't place without losing data; the message names the file and the reason (exit 1)."""

    exit_code = EXIT_FAILED


class EngineFailure(PdfIntakeError):
    """An OCR engine error. The pipeline records it on the page; it never reaches the CLI."""

    exit_code = EXIT_FAILED
