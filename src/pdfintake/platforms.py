"""Which platform this is, whether pdfintake supports it, and where its cache lives.

A platform is one row of ``PLATFORMS``. Adding one is a new row here, a CI runner and a README
line; every pack has a ``default`` reader set, so no pack needs an edit.

The platform is read from ``sysconfig.get_platform()`` (the tag pip and uv match wheels against),
so it describes the running interpreter, not the machine underneath: an x86_64 Python under
Rosetta is ``darwin-x86_64``. Only a multi-architecture macOS tag (``universal2``) needs the
kernel's answer, taken from ``os.uname()``. Nothing here starts a process: of the ``platform``
module only ``mac_ver`` is used, because ``platform.machine()`` and friends can shell out.
"""

from __future__ import annotations

import os
import platform as _platform
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from .errors import SetupError


@dataclass(frozen=True)
class Platform:
    key: str
    sys_platform: str
    machines: tuple[str, ...]
    supported: bool
    ci_runners: tuple[str, ...]
    min_os: str
    notes: str


PLATFORMS: tuple[Platform, ...] = (
    Platform("darwin-arm64",  "darwin", ("arm64",),           True,  ("macos-15", "macos-latest"), "macOS 15", ""),
    Platform("win-amd64",     "win32",  ("AMD64", "x86_64"),  True,  ("windows-latest",), "Windows 10", ""),
    Platform("linux-x86_64",  "linux",  ("x86_64",),          True,  ("ubuntu-24.04", "ubuntu-latest"), "manylinux", ""),
    Platform("linux-aarch64", "linux",  ("aarch64", "arm64"), True,  ("ubuntu-24.04-arm",), "manylinux", "CI row optional"),
    Platform("darwin-x86_64", "darwin", ("x86_64",),          False, (), "", "pikepdf 10.17 has no Intel-Mac wheel"),
    Platform("win-arm64",     "win32",  ("ARM64",),           False, (), "", "tesserocr has no win_arm64 wheel"),
)

UNLISTED_NOTE = "pdfintake has no build for this platform"
BITS_NOTE = "pdfintake needs a 64-bit Python"

# macOS tags that name several architectures at once; the kernel says which one is running.
_MULTI_ARCH = {"universal2", "universal", "intel", "fat", "fat3", "fat64"}
# The operating-system part of a sysconfig tag, as sys.platform spells it.
_OS_OF_TAG = {"macosx": "darwin", "darwin": "darwin", "win": "win32", "linux": "linux"}


def _uname_machine() -> str:
    uname = getattr(os, "uname", None)
    if uname is None:
        return ""
    try:
        return uname().machine
    except OSError:
        return ""


def _platform_for(tag: str, machine: str = "", *, bits: int = 64) -> Platform:
    """The ``PLATFORMS`` row for a sysconfig platform tag, or an unsupported row naming why.

    ``machine`` is the kernel's machine name, used only when the tag names several architectures.
    """
    tag = tag.strip()
    if tag == "win32":                                   # 32-bit Windows Python
        os_name, arch = "win32", "x86"
    else:
        parts = tag.split("-")
        head = parts[0].lower()
        os_name = _OS_OF_TAG.get(head, head)
        arch = parts[-1] if len(parts) > 1 else ""
        if os_name == "darwin" and arch.lower() in _MULTI_ARCH:
            arch = machine or arch
    for row in PLATFORMS:
        if row.sys_platform == os_name and arch.lower() in {m.lower() for m in row.machines}:
            if bits != 64:
                break
            return row
    key = f"{'win' if os_name == 'win32' else os_name}-{arch.lower() or 'unknown'}"
    note = BITS_NOTE if bits != 64 else UNLISTED_NOTE
    return Platform(key, os_name, (arch,) if arch else (), False, (), "", note)


def current() -> Platform:
    """This interpreter's platform. An unlisted one comes back with ``supported=False``."""
    bits = 64 if sys.maxsize > 2**32 else 32
    return _platform_for(sysconfig.get_platform(), _uname_machine(), bits=bits)


def require_supported() -> Platform:
    """The current platform, or SetupError naming the reason it isn't supported."""
    plat = current()
    if not plat.supported:
        reason = plat.notes or UNLISTED_NOTE
        raise SetupError("unsupported_platform", f"pdfintake doesn't run on {plat.key}: {reason}.",
                         platform=plat.key, reason=reason)
    return plat


def macos_version() -> tuple[int, int] | None:
    """(major, minor) of the running macOS, or None anywhere else or when it can't be read."""
    if sys.platform != "darwin":
        return None
    release = _platform.mac_ver()[0]
    parts = release.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        return None
    return major, minor


def cache_root() -> Path:
    """Where pdfintake keeps packs, records, renders and staging. Never created here.

    ``PDFINTAKE_CACHE`` wins everywhere (made absolute). Otherwise: ``~/Library/Caches/pdfintake``
    on macOS, ``%LOCALAPPDATA%\\pdfintake\\Cache`` on Windows, and ``$XDG_CACHE_HOME/pdfintake`` or
    ``~/.cache/pdfintake`` elsewhere (a relative ``XDG_CACHE_HOME`` is ignored, as the XDG spec says).
    """
    override = os.environ.get("PDFINTAKE_CACHE")
    if override:
        return Path(os.path.abspath(os.path.expanduser(override)))
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "pdfintake"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else home / "AppData" / "Local"
        return base / "pdfintake" / "Cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / "pdfintake"
    return home / ".cache" / "pdfintake"
