"""The RapidOCR engine: rapidocr 3.x on onnxruntime, with every model path explicit.

A pack names the recognition model with the ``rec`` parameter (an ``onnx_rec`` asset) and lists
the detection model (and, for ``use_cls``, the angle classifier) among the use's models. The other
two roles are told apart by the models themselves, never by their names: onnxruntime opens each
file once (cached by path, size and mtime) and its output shape says what it is. A detector
outputs a one-channel probability map (rank 4), the classifier two angle scores (rank 2), and a
recogniser a sequence of character scores (rank 3) with its character list embedded in the model.

Nothing is ever downloaded, and no model is ever picked for us. Given no path, rapidocr falls back
to the default models bundled in its wheel (a different detector and recogniser from the pinned
ones, loaded without any network call) or fetches others from modelscope.cn, and it fetches a
character list when the recogniser carries none. So this module always passes the detector,
classifier and recogniser paths, never turns the classifier on without a classifier path, and
refuses a recogniser without an embedded character list (``broken``) before rapidocr sees it.
Images go to rapidocr as arrays, never as paths (rapidocr fetches a path that looks like a URL).

Boxes come back in pixels of the page image the engine was given: rapidocr maps its internal
resize back, and this module clamps each quad to the image and orders it clockwise from the
top-left corner. Text is passed raw; normalisers run in the pipeline.

``open_voter`` is the orientation voter (``pdf/orient.vote``): the detector plus the Latin
recogniser, classifier off, so text turned 180 degrees stays unreadable and loses the vote. It
takes the two by their base-pack keys (``rapid-det`` and ``rapid-rec-latin``, with or without the
``en:`` prefix) and ignores every other entry, so a caller may pass the base pack's whole model map.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ...errors import EngineFailure
from ...params import Param, ParamTable
from ...types import EngineRead, Line, PageImage, Quad, Role
from . import Availability, ContractCase, ModelPaths

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image

    from ...langpacks.manifest import Asset

# numpy and onnxruntime load with the module, so ``engines.warm_up()`` pays for them early.
# rapidocr's own class is imported on first open: it pulls in requests, and urllib3 binds a
# loopback socket at import to probe IPv6, which the import-hygiene test counts as network.
# Neither import starts a process. onnxruntime's telemetry is already off: the package's
# ``__init__`` sets ORT_DISABLE_TELEMETRY before anything can import onnxruntime. When a host
# program imported onnxruntime before pdfintake, that came too late, so the events the API can
# still stop are stopped here as well.
try:
    import numpy as _np
    import onnxruntime as _ort
    _IMPORT_ERROR: str | None = None
    try:
        _ort.disable_telemetry_events()
    except Exception:  # pragma: no cover - an onnxruntime without the call has nothing to stop
        pass
except Exception as _exc:  # pragma: no cover - only on a broken install
    _np = None  # type: ignore[assignment]
    _ort = None  # type: ignore[assignment]
    _IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

_rapidocr_lock = threading.Lock()
_rapidocr_class: Any = None


def _rapidocr() -> Any:
    """rapidocr's RapidOCR class, imported once."""
    global _rapidocr_class
    with _rapidocr_lock:
        if _rapidocr_class is None:
            from rapidocr import RapidOCR

            _rapidocr_class = RapidOCR
        return _rapidocr_class

NAME = "rapidocr"
PLATFORM_KEYS = frozenset({"darwin-arm64", "win-amd64", "linux-x86_64", "linux-aarch64"})
VOTER_USE_ID = "orient_vote"
VOTER_DET_KEY = "rapid-det"                # the base pack's detector (packs/en)
VOTER_REC_KEY = "rapid-rec-latin"          # the base pack's Latin recogniser (packs/en)
BASE_PREFIX = "en:"
# onnxruntime threads per session. Its default (every core) oversubscribes when the pipeline's
# workers (up to 4) read at once: four parallel A4 reads took 8.3 s wall with the default and
# 2.2 s with 4 threads each, on an 18-core machine (2026-11-05); the reads were identical at
# every thread count. So each session gets a quarter of the cores, 1 to 4.
INTRA_OP_THREADS = max(1, min(4, (os.cpu_count() or 4) // 4))
ModelKind = Literal["onnx_det", "onnx_cls", "onnx_rec"]


# --------------------------------------------------------------------------- model inspection

@dataclass(frozen=True)
class _ModelInfo:
    kind: ModelKind | None
    has_keys: bool
    problem: str | None


_inspect_lock = threading.Lock()
_inspected: dict[tuple[str, int, int], _ModelInfo] = {}


def _inspect(path: Path) -> _ModelInfo:
    """What one ONNX file is, from its output shape and metadata. Cached by path, size and mtime."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    with _inspect_lock:
        cached = _inspected.get(key)
    if cached is not None:
        return cached
    info = _inspect_uncached(path, st.st_size)
    with _inspect_lock:
        _inspected[key] = info
    return info


def _inspect_uncached(path: Path, size: int) -> _ModelInfo:
    if size == 0:
        return _ModelInfo(None, False, f"{path.name} is empty (0 bytes)")
    try:
        opts = _ort.SessionOptions()
        opts.log_severity_level = 4
        session = _ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        outputs = session.get_outputs()
        meta = session.get_modelmeta().custom_metadata_map
    except Exception as exc:
        return _ModelInfo(None, False, f"onnxruntime can't load {path.name}: {type(exc).__name__}: {_short(exc)}")
    if not outputs:
        return _ModelInfo(None, False, f"{path.name} has no outputs")
    shape = list(outputs[0].shape or [])
    kind: ModelKind | None = None
    if len(shape) == 4 and shape[1] == 1:
        kind = "onnx_det"
    elif len(shape) == 2:
        kind = "onnx_cls"
    elif len(shape) == 3:
        kind = "onnx_rec"
    if kind is None:
        return _ModelInfo(None, False, f"{path.name} is not a RapidOCR detector, classifier or recogniser "
                                       f"(output shape {shape})")
    has_keys = bool(meta.get("character", "").strip())
    return _ModelInfo(kind, has_keys, None)


def _short(exc: BaseException, limit: int = 200) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- resolving the models

@dataclass(frozen=True)
class _Resolved:
    det: Path
    cls: Path | None
    rec: Path
    rec_ref: str


class _Unresolved(Exception):
    def __init__(self, state: Literal["missing_model", "broken"], detail: str) -> None:
        super().__init__(detail)
        self.state = state
        self.detail = detail


def _lookup(models: ModelPaths, ref: str) -> tuple[str, Path] | None:
    """The entry for ``ref``, whether the caller kept the ``en:`` prefix of a base-pack ref or not."""
    for key in (ref, ref[len(BASE_PREFIX):] if ref.startswith(BASE_PREFIX) else BASE_PREFIX + ref):
        if key in models:
            return key, Path(models[key])
    return None


def _check_present(models: ModelPaths) -> None:
    for key in sorted(models):
        path = Path(models[key])
        if not path.is_file():
            raise _Unresolved("missing_model", f"model {key} is missing (no file at its path)")


def _usable(key: str, path: Path) -> _ModelInfo:
    info = _inspect(path)
    if info.problem is not None:
        raise _Unresolved("broken", f"model {key}: {info.problem}")
    return info


def _resolve(params: Mapping[str, Any], models: ModelPaths, *, voter: bool = False) -> _Resolved:
    if _IMPORT_ERROR is not None:
        raise _Unresolved("broken", f"rapidocr or onnxruntime can't be imported: {_IMPORT_ERROR}")
    if voter:
        # Only the two base-pack keys count; anything else in the map (osd, eng traineddata, the
        # classifier, another recogniser) is not the voter's and is never opened.
        picked: dict[str, Path] = {}
        for ref, role in ((VOTER_DET_KEY, "text-detection model"), (VOTER_REC_KEY, "Latin recogniser")):
            hit = _lookup(models, ref)
            if hit is None:
                raise _Unresolved("missing_model", f"the orientation voter needs the {role} {ref}; "
                                                   "it isn't among the models given")
            picked[hit[0]] = hit[1]
        models = picked
        params = {"rec": hit[0], "use_cls": False}             # hit: the recogniser, looked up last
    _check_present(models)
    use_cls = bool(params.get("use_cls", False))
    rec_ref = params.get("rec")
    if not isinstance(rec_ref, str) or not rec_ref:
        raise _Unresolved("broken", "the rec parameter (the recognition model) is missing")
    found = _lookup(models, rec_ref)
    if found is None:
        raise _Unresolved("missing_model", f"model {rec_ref} (rec) is missing")
    rec_key, rec_path = found
    info = _usable(rec_key, rec_path)
    if info.kind != "onnx_rec":
        raise _Unresolved("broken", f"model {rec_key} is a {info.kind}, not a recognition model")
    if not info.has_keys:
        raise _Unresolved("broken", f"model {rec_key} carries no character list; rapidocr would download one, "
                                    "so it is refused")
    dets: list[tuple[str, Path]] = []
    clss: list[tuple[str, Path]] = []
    for key in sorted(models):
        if key == rec_key:
            continue
        path = Path(models[key])
        kind = _usable(key, path).kind
        if kind == "onnx_det":
            dets.append((key, path))
        elif kind == "onnx_cls":
            clss.append((key, path))
    if not dets:
        raise _Unresolved("missing_model", "no text-detection model is among this use's models")
    if len(dets) > 1:
        raise _Unresolved("broken", f"{len(dets)} text-detection models were given ({', '.join(k for k, _ in dets)})")
    if len(clss) > 1:
        raise _Unresolved("broken", f"{len(clss)} angle classifiers were given ({', '.join(k for k, _ in clss)})")
    cls_path = clss[0][1] if clss else None
    if use_cls and not voter and cls_path is None:
        raise _Unresolved("missing_model", "use_cls is on but no angle-classifier model is among this use's models")
    return _Resolved(det=dets[0][1], cls=cls_path, rec=rec_path, rec_ref=rec_ref)


# --------------------------------------------------------------------------- the session

def _to_bgr(img: "PIL.Image.Image") -> "np.ndarray":
    """An 8-bit BGR array of ``img``: alpha composited over white, 16-bit samples cut to 8."""
    from PIL import Image

    mode = img.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        arr = _np.asarray(img)
        if mode == "I":
            arr = _np.clip(arr, 0, 65535)
        gray = (arr.astype(_np.uint32) >> 8).astype(_np.uint8)
        rgb = _np.repeat(gray[:, :, None], 3, axis=2)
    else:
        if mode in ("RGBA", "LA", "PA", "La", "RGBa") or (mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white, rgba).convert("RGB")
        elif mode != "RGB":
            img = img.convert("RGB")
        rgb = _np.asarray(img)
    return _np.ascontiguousarray(rgb[:, :, ::-1])


def _quad(points: Any, width: int, height: int) -> Quad:
    pts = [(min(max(float(p[0]), 0.0), float(width)), min(max(float(p[1]), 0.0), float(height)))
           for p in list(points)[:4]]
    if len(pts) != 4:
        raise ValueError("a box needs four points")
    area = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
    if area < 0:                                   # anticlockwise on screen (y down): reverse
        pts = [pts[0], pts[3], pts[2], pts[1]]
    start = min(range(4), key=lambda i: (pts[i][0] + pts[i][1], pts[i][1]))
    pts = pts[start:] + pts[:start]
    return (pts[0], pts[1], pts[2], pts[3])


def _conf(value: Any) -> float:
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(c):
        return 0.0
    return min(max(c, 0.0), 1.0)


class _Session:
    """One rapidocr instance. Reads are serialised; a read that runs out of time is reported as
    ``timeout`` and finishes in the background, holding the session until it does."""

    def __init__(self, use_id: str, role: Role, resolved: _Resolved, *, use_cls: bool, min_score: float,
                 spec: str) -> None:
        self.use_id = use_id
        self.role = role
        self.spec = spec
        self._use_cls = bool(use_cls and resolved.cls is not None)
        self._rec_ref = resolved.rec_ref
        self._lock = threading.Lock()
        self._closed = False
        params: dict[str, Any] = {
            "Global.use_cls": self._use_cls,
            "Global.text_score": float(min_score),
            "Global.log_level": "critical",
            "Global.return_word_box": False,
            "Det.model_path": str(resolved.det),
            "Rec.model_path": str(resolved.rec),
            "EngineConfig.onnxruntime.intra_op_num_threads": INTRA_OP_THREADS,
        }
        if resolved.cls is not None:
            params["Cls.model_path"] = str(resolved.cls)
        try:
            self._ocr = _rapidocr()(params=params)
            # Load the models now, so a bad file fails here and reads time only the OCR.
            for loader in ("_load_det_model", "_load_rec_model") + (("_load_cls_model",) if self._use_cls else ()):
                fn = getattr(self._ocr, loader, None)
                if fn is not None:
                    fn()
        except Exception as exc:
            raise EngineFailure("rapidocr_open_failed",
                                f"rapidocr could not load its models: {type(exc).__name__}: {_short(exc)}",
                                engine=NAME, use_id=use_id) from exc

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        start = time.perf_counter()

        def result(status: str, lines: list[Line], error: str | None = None,
                   meta: dict[str, Any] | None = None) -> EngineRead:
            return EngineRead(self.use_id, NAME, self.spec, self.role, status, lines,  # type: ignore[arg-type]
                              time.perf_counter() - start, error, meta or {"version": ENGINE.version()})

        if self._closed:
            raise EngineFailure("rapidocr_closed", "this rapidocr session is closed", engine=NAME, use_id=self.use_id)
        if not timeout_s > 0:
            return result("timeout", [], "no time left for the read")
        # Lock.acquire and Thread.join refuse waits above TIMEOUT_MAX (an infinite budget included).
        wait_s = min(float(timeout_s), threading.TIMEOUT_MAX)
        try:
            with page.open() as img:
                arr = _to_bgr(img)
        except Exception as exc:
            return result("failed", [], f"the page image could not be read: {type(exc).__name__}: {_short(exc)}")
        if not self._lock.acquire(timeout=wait_s):
            return result("timeout", [], "the session was still busy with a read that ran out of time")
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["out"] = self._ocr(arr, use_cls=self._use_cls)
            except BaseException as exc:  # reported by the caller
                box["error"] = exc
            finally:
                self._lock.release()

        worker = threading.Thread(target=work, name=f"rapidocr-{self.use_id}", daemon=True)
        worker.start()
        worker.join(max(0.0, wait_s - (time.perf_counter() - start)))
        if worker.is_alive():
            return result("timeout", [], f"rapidocr took longer than {timeout_s:g} s")
        if "error" in box:
            exc = box["error"]
            return result("failed", [], f"rapidocr failed: {type(exc).__name__}: {_short(exc)}")
        lines = self._lines(box.get("out"), page.width, page.height)
        meta = {"version": ENGINE.version(), "rec": self._rec_ref, "cls": self._use_cls,
                "mean_conf": round(sum(ln.conf for ln in lines) / len(lines), 4) if lines else None}
        return result("ok" if lines else "empty", lines, None, meta)

    @staticmethod
    def _lines(out: Any, width: int, height: int) -> list[Line]:
        txts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        boxes = getattr(out, "boxes", None)
        lines: list[Line] = []
        for i, text in enumerate(txts):
            if not isinstance(text, str) or not text.strip():
                continue
            if boxes is None or i >= len(boxes) or boxes[i] is None:
                continue                              # a line with no box can't be placed
            conf = _conf(scores[i] if i < len(scores) else 0.0)
            lines.append(Line(text, conf, _quad(boxes[i], width, height)))
        return lines

    def close(self) -> None:
        self._closed = True
        self._ocr = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- the engine

def _spec(rec_ref: str, use_cls: bool, min_score: float) -> str:
    return f"{NAME}:{rec_ref}:cls{int(bool(use_cls))}:min{min_score:g}"


class RapidOcrEngine:
    name: ClassVar[str] = NAME
    api: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable(
        Param("rec", "asset_ref", required=True, choices=("onnx_rec",)),
        Param("use_cls", "bool", default=False),
        Param("min_score", "float", default=0.5, min=0.0, max=1.0),
    )
    MODEL_KINDS: ClassVar[frozenset[str]] = frozenset({"onnx_det", "onnx_cls", "onnx_rec"})
    platforms: ClassVar[frozenset[str]] = PLATFORM_KEYS
    session_per_thread: ClassVar[bool] = True

    def version(self) -> str | None:
        try:
            return importlib.metadata.version("rapidocr")
        except importlib.metadata.PackageNotFoundError:
            return None

    def availability(self, params: Mapping[str, Any], models: ModelPaths, platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform",
                                f"rapidocr is not supported on {platform_key} "
                                f"(supported: {', '.join(sorted(self.platforms))})")
        try:
            _resolve(params, models)
        except _Unresolved as exc:
            return Availability(False, exc.state, exc.detail)
        return Availability(True, "available", "")

    def open(self, use_id: str, role: Role, params: Mapping[str, Any], models: ModelPaths) -> _Session:
        try:
            resolved = _resolve(params, models)
        except _Unresolved as exc:
            raise EngineFailure("rapidocr_unavailable", f"rapidocr can't open: {exc.detail}",
                                engine=NAME, use_id=use_id, state=exc.state) from None
        use_cls = bool(params.get("use_cls", False))
        min_score = float(params.get("min_score", 0.5))
        return _Session(use_id, role, resolved, use_cls=use_cls, min_score=min_score,
                        spec=_spec(resolved.rec_ref, use_cls, min_score))

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence["Asset"]) -> list[dict[str, Any]]:
        """One candidate per RapidOCR recogniser among ``assets``, classifier off (the kit benches them)."""
        keys = sorted({a.key for a in assets if getattr(a, "engine", None) == NAME and getattr(a, "kind", None) == "onnx_rec"})
        return [{"rec": key} for key in keys]

    def contract_case(self) -> ContractCase | None:
        """The base pack's detector and Latin recogniser with a made-up English page; None when the
        base pack isn't installed or verified here."""
        try:
            from ...langpacks import store

            base = store.base()
            det = rec = None
            for asset in base.manifest.models:
                if asset.engine != NAME:
                    continue
                if asset.kind == "onnx_det" and det is None:
                    det = asset
                elif asset.kind == "onnx_rec" and (rec is None or asset.key == VOTER_REC_KEY):
                    rec = asset
            if det is None or rec is None:
                return None
            det_path, rec_path = Path(base.asset_path(det.key)), Path(base.asset_path(rec.key))
        except Exception:
            return None
        return _contract_case(det_path, rec_path, det_key=det.key, rec_key=rec.key)


ENGINE = RapidOcrEngine()


def open_voter(models: ModelPaths) -> _Session:
    """The orientation voter: the detector and the Latin recogniser from ``models``, classifier off.

    The two are taken by key, ``rapid-det`` and ``rapid-rec-latin``, each with or without the base
    pack's ``en:`` prefix; every other entry (the classifier, another recogniser, Tesseract's osd
    and eng) is ignored and never opened, so the base pack's whole model map is a fair argument.
    EngineFailure (``missing_model``) when either key is absent, ``broken`` when its file isn't the
    model it should be.
    """
    try:
        resolved = _resolve({"use_cls": False}, models, voter=True)
    except _Unresolved as exc:
        raise EngineFailure("rapidocr_unavailable", f"the orientation voter can't open: {exc.detail}",
                            engine=NAME, use_id=VOTER_USE_ID, state=exc.state) from None
    return _Session(VOTER_USE_ID, "checker", resolved, use_cls=False, min_score=0.5,
                    spec=f"{NAME}:voter:{resolved.rec_ref}")


# --------------------------------------------------------------------------- the contract case

CONTRACT_LINES = ("The example invoice lists seven items", "Total amount 4521 units", "Reference AB123456")


def _contract_page(folder: Path) -> PageImage:
    """A made-up English page drawn with Pillow's built-in font, as a 300 dpi PNG."""
    from PIL import Image, ImageDraw, ImageFont

    width, height, dpi = 1500, 900, 300.0
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=44)
    for i, text in enumerate(CONTRACT_LINES):
        draw.text((110, 140 + i * 230), text, fill="black", font=font)
    path = Path(folder) / "rapidocr-contract.png"
    img.save(path, format="PNG", dpi=(dpi, dpi))
    return PageImage(index=0, path=path, width=width, height=height, dpi=dpi,
                     cropbox=(0.0, 0.0, width * 72.0 / dpi, height * 72.0 / dpi))


def _contract_case(det: Path, rec: Path, *, det_key: str = "rapid-det", rec_key: str = VOTER_REC_KEY) -> ContractCase | None:
    from PIL import features

    page = _contract_page if features.check("freetype2") else None
    return ContractCase(
        params={"rec": rec_key},
        models={det_key: det, rec_key: rec},
        model_kinds={det_key: "onnx_det", rec_key: "onnx_rec"},
        role="checker",
        page=page,
        timeout_s=120.0,
    )


__all__ = ["ENGINE", "open_voter", "RapidOcrEngine", "NAME", "VOTER_DET_KEY", "VOTER_REC_KEY"]
