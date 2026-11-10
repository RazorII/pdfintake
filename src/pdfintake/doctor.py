"""``pdfintake doctor``: is this install able to run, and if not, what exactly is missing (DESIGN §4.1).

``report`` gathers, without changing anything outside the cache:

* the platform (and whether pdfintake supports it), Python and pdfintake's versions, the OS;
* each registered OCR engine: whether it runs on this platform, and its version;
* Apple Vision's revision 3 languages on this Mac (macOS only);
* thorough mode (``thorough.status()``): whether this machine can run it, what is installed, the
  setting. Its engine, ``dots``, is optional: not installed is its own state, never ``broken``;
* the pack index, the English base pack (``store.base()``: version, RECORD, every hash) and every
  installed pack, verified (``store.verify()``);
* the cache folder, and the saved passwords file: where it is, whether it exists, and whether a
  run will use it. The answer is ``passwords.check``, which runs the very checks
  ``passwords.candidates`` makes for every run (owner, mode, size, UTF-8, at most 100 passwords)
  and returns only a reason code, so doctor can't call a file fine that every run refuses
  (decided 2026-11-05) and never holds a password (decided 2026-11-06). Only the reason code is
  reported, never a password or a count;
* with ``canary``, every installed pack's canary read with every engine of its reader set;
* with ``check_network``, whether the pack host answers (one HTTPS request, nothing downloaded).

``exit_code`` is 3 (SETUP) when pdfintake can't run here: an unsupported platform, no pack index,
or a missing or broken base pack (a failed base canary included). Anything else is reported and
exits 0. Every message is core's or comes from a ``PdfIntakeError``; no file content is reported.
Like ``run``, ``report`` imports every engine first and then holds ``integrity.no_subprocess()``.
"""

from __future__ import annotations

import os
import platform as _platform
import sys
from collections.abc import Mapping
from typing import Any

from . import __version__, integrity, passwords, platforms
from .errors import EXIT_OK, EXIT_SETUP, PdfIntakeError

NETWORK_URL = "https://github.com/RazorII/pdfintake"
_PASSWORDS_PROBLEM = ("Runs won't use the saved passwords file as it is, so every password-protected file will fail "
                      "with exit 2 until it is fixed; run one to see the reason.")
_PASSWORDS_PROBLEMS = {
    "mode": "The saved passwords file isn't private, so every password-protected file will fail with exit 2 until it "
            "is (chmod 600 on macOS and Linux).",
    "owner": "The saved passwords file belongs to another user, so every password-protected file will fail with "
             "exit 2 until it is replaced by one of your own.",
    "too_many": "The saved passwords file holds more than 100 passwords, the most pdfintake tries, so every "
                "password-protected file will fail with exit 2 until some are removed.",
    "unreadable": "The saved passwords file can't be read, so every password-protected file will fail with exit 2 "
                  "until it can (chmod 600 on macOS and Linux).",
}
NETWORK_TIMEOUT_S = 15.0
VISION_REVISION = 3
VISION_LEVEL = "accurate"


def report(*, canary: bool = False, check_network: bool = False) -> dict[str, Any]:
    """Everything ``pdfintake doctor`` prints, as JSON values; ``problems`` lists what blocks a run."""
    from .ocr import engines

    engines.warm_up()
    with integrity.no_subprocess():
        return _report(canary=canary, check_network=check_network)


def exit_code(rep: Mapping[str, Any]) -> int:
    """3 when pdfintake can't run here (``rep["blocking"]``), else 0."""
    return EXIT_SETUP if rep.get("blocking") else EXIT_OK


# --------------------------------------------------------------------------- the parts

def _error(exc: PdfIntakeError) -> dict[str, Any]:
    return {"ok": False, "code": exc.code, "message": exc.message}


def _os_name() -> str:
    """A short OS name and version, read without starting a process."""
    if sys.platform == "darwin":
        release = _platform.mac_ver()[0]
        return f"macOS {release}" if release else "macOS"
    if sys.platform == "win32":
        try:
            v = sys.getwindowsversion()
            return f"Windows {v.major}.{v.minor}.{v.build}"
        except AttributeError:                         # pragma: no cover
            return "Windows"
    if sys.platform.startswith("linux"):
        try:
            info = _platform.freedesktop_os_release()
        except OSError:
            return "Linux"
        return str(info.get("PRETTY_NAME") or info.get("NAME") or "Linux")[:60]
    return sys.platform


_THOROUGH_ENGINE = "dots"


def _thorough() -> dict[str, Any]:
    from . import thorough

    try:
        return thorough.status()
    except Exception as exc:                           # noqa: BLE001 - doctor reports, never crashes
        return {"available": False, "code": type(exc).__name__, "message": "thorough mode's state couldn't be read"}


def _engines(platform_key: str) -> dict[str, Any]:
    from .ocr import engines

    out: dict[str, Any] = {}
    for name in engines.names():
        try:
            engine = engines.get(name)
        except PdfIntakeError as exc:
            out[name] = {"available": False, "state": "broken", "code": exc.code, "message": exc.message}
            continue
        on_platform = platform_key in getattr(engine, "platforms", frozenset())
        try:
            version = engine.version()
        except PdfIntakeError as exc:
            out[name] = {"available": False, "state": "broken", "code": exc.code, "message": exc.message}
            continue
        except Exception as exc:                       # noqa: BLE001 - doctor reports, never crashes
            out[name] = {"available": False, "state": "broken", "code": type(exc).__name__,
                         "message": f"{name} couldn't report its version"}
            continue
        if not on_platform:
            out[name] = {"available": False, "state": "not_on_platform", "version": version}
        elif not version and name == _THOROUGH_ENGINE:
            out[name] = {"available": False, "state": "not_installed", "version": None,
                         "message": "thorough mode's reader, an optional extra; see 'pdfintake thorough status'"}
        elif not version:
            out[name] = {"available": False, "state": "broken", "version": None,
                         "message": f"{name} isn't installed or can't be loaded here"}
        else:
            out[name] = {"available": True, "state": "available", "version": version}
    return out


def _vision() -> dict[str, Any] | None:
    if sys.platform != "darwin":
        return None
    try:
        from .ocr.engines import vision

        langs = vision.supported_languages(VISION_REVISION, VISION_LEVEL)
    except PdfIntakeError as exc:
        return {"revision": VISION_REVISION, "level": VISION_LEVEL, **_error(exc)}
    except Exception as exc:                           # noqa: BLE001
        return {"revision": VISION_REVISION, "level": VISION_LEVEL, "ok": False, "code": type(exc).__name__,
                "message": "Apple Vision's languages couldn't be read"}
    return {"revision": VISION_REVISION, "level": VISION_LEVEL, "ok": True, "languages": list(langs)}


def _passwords_file() -> dict[str, Any]:
    """Where the saved passwords file is, whether it is there, and whether runs will use it.

    ``private`` is True only when ``passwords.check`` finds nothing a run would refuse (the checks
    every run applies, through the same code); otherwise ``reason`` is that check's reason code
    (``mode``, ``owner``, ``too_many`` …). No password is read into doctor.
    """
    path = passwords.default_path()
    if not os.path.lexists(path):
        return {"path": str(path), "exists": False, "private": None, "reason": None}
    reason = passwords.check(path)
    return {"path": str(path), "exists": True, "private": reason is None, "reason": reason}


def _network() -> dict[str, Any]:
    import ssl
    import urllib.error
    import urllib.request

    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(cafile=certifi.where())
    except (ImportError, OSError):
        pass
    req = urllib.request.Request(NETWORK_URL, method="HEAD", headers={"User-Agent": f"pdfintake/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_S, context=context) as resp:
            return {"ok": True, "url": NETWORK_URL, "status": int(resp.status)}
    except urllib.error.HTTPError as exc:
        return {"ok": exc.code < 500, "url": NETWORK_URL, "status": int(exc.code)}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "url": NETWORK_URL, "error": type(exc).__name__,
                "message": "github.com couldn't be reached; a network setting may block downloads from github.com "
                           "or release-assets.githubusercontent.com"}


def _canaries(packs: Mapping[str, Any], base: Any, platform_key: str) -> tuple[list[dict[str, Any]], list[str]]:
    from .langpacks import canary

    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for pid in sorted(packs):
        pack = packs[pid]
        if pack.manifest.canary is None:
            rows.append({"pack": pid, "use_id": None, "engine": None, "ok": True, "skipped": True,
                         "reason": "this pack has no canary"})
            continue
        try:
            results = canary.check(pack, base, platform_key)
        except PdfIntakeError as exc:
            rows.append({"pack": pid, "use_id": None, "engine": None, **_error(exc)})
            failed.append(pid)
            continue
        for r in results:
            rows.append({"pack": pid, "use_id": r.use_id, "engine": r.engine, "ok": bool(r.ok),
                         "missing": len(r.missing), "seconds": round(float(r.seconds), 3),
                         "cached": bool(r.cached), "skipped": canary.skipped(r),
                         "error": None if r.error is None else str(r.error)[:200]})
            if not r.ok and pid not in failed:
                failed.append(pid)
    return rows, failed


def _report(*, canary: bool, check_network: bool) -> dict[str, Any]:
    from .langpacks import index as index_mod
    from .langpacks import store

    plat = platforms.current()
    problems: list[str] = []
    blocking = False
    rep: dict[str, Any] = {
        "pdfintake": __version__,
        "python": _platform.python_version(),
        "os": _os_name(),
        "platform": {"key": plat.key, "supported": plat.supported, "notes": plat.notes},
        "cache": str(platforms.cache_root()),
    }
    if not plat.supported:
        blocking = True
        problems.append(f"pdfintake doesn't run on {plat.key}: {plat.notes or platforms.UNLISTED_NOTE}.")
    rep["engines"] = _engines(plat.key)
    rep["vision"] = _vision()
    rep["thorough"] = _thorough()

    try:
        index_mod.load()
        rep["index"] = {"ok": True}
    except PdfIntakeError as exc:
        rep["index"] = _error(exc)
        blocking = True
        problems.append(exc.message)

    base = None
    if rep["index"]["ok"]:
        try:
            base = store.base()
            rep["base_pack"] = {"ok": True, "id": base.manifest.id, "version": base.manifest.version}
        except PdfIntakeError as exc:
            rep["base_pack"] = _error(exc)
            blocking = True
            problems.append(exc.message)
        try:
            rep["packs"] = [{"pack": v.pack, "ok": v.ok, "problems": list(v.problems)} for v in store.verify()]
        except PdfIntakeError as exc:
            rep["packs"] = []
            problems.append(exc.message)
        for v in rep["packs"]:
            if not v["ok"] and (base is None or v["pack"] != base.manifest.id):
                problems.extend(f"{v['pack']}: {p}" for p in v["problems"])
    else:
        rep["base_pack"] = {"ok": False, "code": "index_missing", "message": "the pack index can't be read"}
        rep["packs"] = []
    rep["passwords_file"] = _passwords_file()
    if rep["passwords_file"]["exists"] and not rep["passwords_file"]["private"]:
        problems.append(_PASSWORDS_PROBLEMS.get(rep["passwords_file"]["reason"], _PASSWORDS_PROBLEM))

    rep["canaries"] = None
    if canary and base is not None:
        try:
            installed = store.installed()
        except PdfIntakeError as exc:
            installed = {base.manifest.id: base}
            problems.append(exc.message)
        rows, failed = _canaries(installed, base, plat.key)
        rep["canaries"] = rows
        for pid in failed:
            if pid == base.manifest.id:
                blocking = True
                problems.append("The English base pack failed its self-check on this computer.")
            else:
                problems.append(f"The {pid} language pack failed its self-check on this computer.")
    rep["network"] = _network() if check_network else None
    if rep["network"] is not None and not rep["network"]["ok"]:
        problems.append(rep["network"].get("message") or "the pack host couldn't be reached")
    rep["problems"] = problems
    rep["blocking"] = blocking
    rep["exit"] = EXIT_SETUP if blocking else EXIT_OK
    return rep
