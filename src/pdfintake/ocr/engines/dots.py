"""dots.mocr as an OCR engine: thorough mode's reader on Apple silicon (DESIGN §14).

dots.mocr is a vision-language model that reads a whole page into layout elements (a box, a
category and the text). It runs in-process on the Mac's GPU through ``mlx``, with the model code in
``dotsmlx`` (the package beside this module). No pack lists it: thorough mode adds it to pass 2 of
every OCR page, with the pinned model files from ``thorough.model_paths()``, and it never reads a
page alone (DESIGN §14.1). What shapes this module:

* **One GPU.** A session is shared by every thread (``session_per_thread`` False), the loaded model
  is shared by every session on the same folder, and reads run one at a time. A read's time limit
  starts when it gets the model, so a page waiting behind another page's read doesn't time out.
* **The image.** The render is cut to a 1,600 px long side, or to a 1.8 MP budget when that keeps
  more (a receipt keeps its width), with Lanczos, never upscaled: dots reads a page that size
  faster and no worse than at full resolution.
* **The prompt** is dots.ocr's own layout prompt (``prompt_layout_all_en`` in the public
  rednote-hilab/dots.ocr repository, ``dots_ocr/utils/prompts.py``), decoded greedily to at most
  ``MAX_TOKENS`` tokens.
* **Loops.** On a dense page the model can get stuck repeating one element, or one phrase inside an
  element, until the token limit. The read stops as soon as the output ends in such a loop
  (``LoopWatch``), keeps every element that closed before the cut, and drops the repeats.
* **No confidences.** The model gives none, so every line gets ``conf`` 1.0. Only thorough mode's
  fills and the ``corroborated_trigrams`` pick see these lines, and neither reads ``conf``.

Each element's text (Markdown, a table as HTML, a formula as LaTeX) is flattened to plain lines; an
element of several lines has its box cut into that many bands, one per line. Boxes come back in the
pixels of the image the model saw and are scaled to the render's.

``mlx`` and ``tokenizers`` come with the extra ``pdfintake[thorough]`` and are imported only when a
session opens, so this module imports anywhere.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ...errors import EngineFailure
from ...params import ParamTable
from ...types import EngineRead, Line, PageImage, Quad, Role
from . import Availability, ContractCase, ModelPaths

if TYPE_CHECKING:
    import PIL.Image

    from ...langpacks.manifest import Asset
    from .dotsmlx import DotsModel

__all__ = ["ENGINE", "MAX_PIXELS", "MAX_SIDE", "MAX_TOKENS", "PROMPT", "DotsEngine", "DotsSession", "LoopWatch",
           "canary", "elements_to_lines", "read_elements", "target_size"]

PLATFORM_KEYS = frozenset({"darwin-arm64"})
LIBRARIES = ("mlx", "tokenizers")
# The files the model is loaded from; the folder's other pinned files (the model card, the chat
# templates, the vocabulary) aren't read.
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "model.safetensors", "model.safetensors.index.json")

MAX_SIDE = 1600                 # the long side the render is cut to ...
MAX_PIXELS = 1_800_000          # ... unless this pixel budget keeps more
MAX_TOKENS = 6000
DEFAULT_TIMEOUT_S = 300.0
_ERROR_CHARS = 200

# dots.ocr's layout prompt, word for word (rednote-hilab/dots.ocr, dots_ocr/utils/prompts.py,
# "prompt_layout_all_en"; MIT).
PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


# --------------------------------------------------------------------------- the image

def target_size(width: int, height: int, max_side: int = MAX_SIDE, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """The size the model gets: the long side cut to ``max_side``, unless the ``max_pixels`` budget
    keeps more. Never larger than the image. An A4-shaped page at 1,600 px is about 1.8 MP, so only
    a page longer or wider than A4 gets more than 1,600 px on its long side."""
    scale = max_side / max(width, height)
    if max_pixels > 0:
        scale = max(scale, math.sqrt(max_pixels / (width * height)))
    if scale >= 1.0:
        return width, height
    return max(1, int(width * scale)), max(1, int(height * scale))


def _model_image(img: PIL.Image.Image) -> PIL.Image.Image:
    from PIL import Image

    if img.mode != "RGB":
        img = img.convert("RGB")
    size = target_size(img.width, img.height)
    return img if size == img.size else img.resize(size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- the model's output

def _body(raw: str) -> str:
    """The output without a Markdown code fence around it."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    return re.sub(r"```$", "", text).strip()


def complete_elements(raw: str) -> tuple[list[dict[str, Any]], bool]:
    """(elements, closed): every element object of the output's JSON array that is complete, in
    order, up to the first one that isn't. ``closed`` is True only when the array itself closed."""
    text = _body(raw)
    pos = text.find("[")
    if pos == -1:
        return [], False
    decoder = json.JSONDecoder()
    pos += 1
    out: list[dict[str, Any]] = []
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text):
            return out, False
        if text[pos] == "]":
            return out, True
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except ValueError:
            return out, False
        if isinstance(obj, dict):
            out.append(obj)


def _key(element: Mapping[str, Any]) -> str:
    return " ".join(str(element.get("text") or "").split()).casefold()


REPEAT_COPIES = 3          # a repeated block at the end counts as a loop from this many copies on
REPEAT_BLOCK = 3           # and may be up to this many elements long
STOP_ELEMENTS = 6          # a read stops once the repeated elements at its end reach this many


def trailing_repeat(elements: Sequence[Mapping[str, Any]]) -> tuple[int, int, int] | None:
    """(start, block, copies) when the list ends in one block of ``block`` elements repeated
    ``copies`` times (texts compared without case and spacing; boxes ignored, since a stuck model
    moves them a little each time), at least ``REPEAT_COPIES`` copies; ``elements[start:]`` is the
    run. Elements without text never make a loop."""
    for block in range(1, REPEAT_BLOCK + 1):
        if len(elements) < block * REPEAT_COPIES:
            break
        tail = [_key(e) for e in elements[-block:]]
        if not all(tail):
            continue
        copies, at = 1, len(elements) - block
        while at - block >= 0 and [_key(e) for e in elements[at - block:at]] == tail:
            copies += 1
            at -= block
        if copies >= REPEAT_COPIES:
            return at, block, copies
    return None


PHRASE_SPAN = 200          # a phrase loop covers at least this many characters at the end ...
PHRASE_LONGEST = 400       # ... repeating one phrase of at most this many characters ...
PHRASE_LETTERS = 3         # ... that holds at least this many letters outside markup


def repeated_phrase(text: str) -> bool:
    """Whether the text ends in one phrase written over and over: a loop inside a single element,
    which never closes, so ``trailing_repeat`` can't see it. The phrase must hold letters, because
    dot leaders, rules and empty table cells repeat on real pages."""
    n = len(text)
    for period in range(2, PHRASE_LONGEST + 1):
        span = max(PHRASE_SPAN, 2 * period)
        if n < span + period:
            break
        if text[-1] != text[-1 - period] or text[-span:] != text[-span - period:-period]:
            continue
        unit = text[-period:]
        if sum(c.isalpha() for c in re.sub(r"<[^>]*>", "", unit + unit)) >= 2 * PHRASE_LETTERS:
            return True
    return False


class LoopWatch:
    """Fed the whole output each time it grows; says when it ends in a loop, and which kind
    (``stopped``: ``loop`` for repeated elements, ``phrase`` for a phrase inside one)."""

    def __init__(self) -> None:
        self.stopped: str | None = None
        self._seen = 0

    def __call__(self, text: str) -> bool:
        if self.stopped is not None:
            return True
        new, self._seen = text[self._seen:], len(text)
        if "}" in new:                            # only a closing brace can end an element
            elements, closed = complete_elements(text)
            run = None if closed else trailing_repeat(elements)
            if run is not None and run[1] * run[2] >= STOP_ELEMENTS:
                self.stopped = "loop"
                return True
        if repeated_phrase(text):
            self.stopped = "phrase"
            return True
        return False


def read_elements(raw: str, cut: bool) -> tuple[list[dict[str, Any]], int]:
    """(elements, repeats dropped). A whole array that wasn't ``cut`` (by a loop stop or the token
    limit) is taken as it is; otherwise the elements that closed before the cut, with a trailing
    repeated block kept once."""
    elements, closed = complete_elements(raw)
    if closed and not cut:
        return elements, 0
    run = trailing_repeat(elements)
    if run is None:
        return elements, 0
    start, block, _ = run
    return elements[:start + block], len(elements) - (start + block)


_TAG = re.compile(r"<[^>]+>")
_ROW_END = re.compile(r"</tr\s*>|<br\s*/?>", re.IGNORECASE)
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_ENTITIES = (("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&amp;", "&"))


def flatten(text: str, category: str) -> list[str]:
    """An element's text as plain lines: table rows and line breaks become lines, tags, LaTeX
    commands and light Markdown go, entities are decoded. Nothing else is changed: the pipeline runs
    the pack's normalisers on every engine's lines alike."""
    t = _ROW_END.sub("\n", text)
    t = _TAG.sub(" ", t)
    for entity, char in _ENTITIES:
        t = t.replace(entity, char)
    if category == "Formula":
        t = _LATEX_COMMAND.sub(" ", t)
        t = re.sub(r"[${}^_]", " ", t)
    t = re.sub(r"[#*`|>]+", " ", t)
    out = []
    for line in t.split("\n"):
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            out.append(line)
    return out


def _clamp(v: float, hi: float) -> float:
    return min(max(v, 0.0), float(hi))


def elements_to_lines(elements: Sequence[Mapping[str, Any]], seen: tuple[int, int],
                      page: tuple[int, int]) -> list[Line]:
    """The elements' text as lines with boxes in the render's pixels. ``seen`` is the size of the
    image the model read (its boxes are in those pixels), ``page`` the render's. An element of
    several lines has its box cut into as many bands, top to bottom; an element without a usable box
    or text is left out."""
    sx, sy = page[0] / seen[0], page[1] / seen[1]
    out: list[Line] = []
    for e in elements:
        text = e.get("text")
        bbox = e.get("bbox")
        if not isinstance(text, str) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox):
            continue
        lines = flatten(text, str(e.get("category") or ""))
        if not lines:
            continue
        x0, x1 = sorted((_clamp(bbox[0] * sx, page[0]), _clamp(bbox[2] * sx, page[0])))
        y0, y1 = sorted((_clamp(bbox[1] * sy, page[1]), _clamp(bbox[3] * sy, page[1])))
        band = (y1 - y0) / len(lines)
        for i, line in enumerate(lines):
            top, bottom = round(y0 + i * band, 1), round(y0 + (i + 1) * band, 1)
            left, right = round(x0, 1), round(x1, 1)
            box: Quad = ((left, top), (right, top), (right, bottom), (left, bottom))
            out.append(Line(line, 1.0, box))
    return out


# --------------------------------------------------------------------------- the model, loaded once

_models_lock = threading.Lock()
_models: dict[str, DotsModel] = {}
_gpu = threading.Lock()                 # one read at a time, whichever session asks


def _on_apple_silicon() -> bool:
    import os

    return sys.platform == "darwin" and hasattr(os, "uname") and os.uname().machine == "arm64"


def _missing_libraries() -> list[str]:
    out = []
    for name in LIBRARIES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(name)
    return out


def _folder(models: ModelPaths) -> Path:
    """The one folder every required file is in. EngineFailure when one is missing or elsewhere."""
    missing = [name for name in REQUIRED_FILES if name not in models or not Path(models[name]).is_file()]
    if missing:
        raise EngineFailure("missing_model", f"the dots.mocr model is missing {', '.join(missing)}")
    folders = {Path(models[name]).parent for name in REQUIRED_FILES}
    if len(folders) != 1:
        raise EngineFailure("missing_model", "the dots.mocr model's files aren't in one folder")
    return folders.pop()


def _load(folder: Path) -> DotsModel:
    key = str(folder.resolve())
    with _models_lock:
        model = _models.get(key)
        if model is None:
            from . import dotsmlx

            try:
                model = dotsmlx.load(folder)
            except ModuleNotFoundError as exc:
                raise EngineFailure("dots_unavailable", f"dots.mocr needs {exc.name or 'mlx'}; install "
                                    "pdfintake[thorough]") from None
            except (OSError, ValueError, KeyError) as exc:
                raise EngineFailure("dots_unavailable",
                                    f"dots.mocr couldn't be loaded: {type(exc).__name__}: {exc}"[:_ERROR_CHARS]) from None
            _models[key] = model
        return model


def _reset_for_tests() -> None:
    """Tests only: forget every loaded model."""
    with _models_lock:
        _models.clear()


# --------------------------------------------------------------------------- the engine

class DotsSession:
    """Reads pages with one loaded model. Thread-safe: reads take turns on the GPU."""

    def __init__(self, engine: DotsEngine, use_id: str, role: Role, model: DotsModel | None,
                 unavailable: str | None) -> None:
        self.engine = engine
        self.use_id = use_id
        self.role = role
        self.model = model
        self.unavailable = unavailable          # why every read is "unavailable", or None
        self.closed = False

    def read(self, page: PageImage, *, timeout_s: float) -> EngineRead:
        if self.closed:
            raise EngineFailure("session_closed", "this dots.mocr session was closed")
        start = time.perf_counter()
        meta: dict[str, Any] = {}

        def result(status: str, lines: list[Line], error: str | None = None) -> EngineRead:
            version = self.engine.version() if status != "unavailable" else None
            if version is not None:
                meta["version"] = version
            return EngineRead(self.use_id, self.engine.name, "dots:mocr-8bit:layout", self.role,
                              status, lines, time.perf_counter() - start, error, meta)  # type: ignore[arg-type]

        if self.unavailable is not None or self.model is None:
            return result("unavailable", [], self.unavailable or "dots.mocr isn't loaded")
        try:
            limit = float(timeout_s)
        except (TypeError, ValueError):
            limit = DEFAULT_TIMEOUT_S
        if not limit > 0:
            return result("timeout", [], "no time left for the read")
        if not Path(page.path).is_file():
            return result("failed", [], "the page image is missing")
        try:
            with page.open() as img:
                image = _model_image(img)
                if image is img:                 # not resized or converted: keep it past the close
                    image = img.copy()
        except Exception as exc:
            return result("failed", [], f"the page image can't be read: {type(exc).__name__}"[:_ERROR_CHARS])
        return self._read_image(image, (page.width, page.height), limit, meta, result)

    def _read_image(self, image: PIL.Image.Image, page_size: tuple[int, int], limit: float,
                    meta: dict[str, Any], result: Callable[..., EngineRead]) -> EngineRead:
        from .dotsmlx.preprocess import smart_resize

        assert self.model is not None
        cfg = self.model.config.image
        try:
            seen_h, seen_w = smart_resize(image.height, image.width, cfg.patch_size * cfg.merge_size,
                                          cfg.min_pixels, cfg.max_pixels)
        except ValueError as exc:
            return result("failed", [], f"dots.mocr can't read a page this shape: {exc}"[:_ERROR_CHARS])
        meta["size"] = [seen_w, seen_h]
        watch = LoopWatch()
        with _gpu:
            deadline = time.monotonic() + limit
            try:
                out = self.model.read(image, PROMPT, MAX_TOKENS, stop=watch, deadline=deadline)
            except Exception as exc:  # an mlx or tokenizer error: this page only
                return result("failed", [], f"dots.mocr: {type(exc).__name__}: {exc}"[:_ERROR_CHARS])
        meta["tokens"] = out.tokens
        meta["stopped"] = watch.stopped or out.stopped
        if out.stopped == "deadline":
            return result("timeout", [], f"dots.mocr gave no complete read within {limit:g} s")
        elements, dropped = read_elements(out.text, cut=out.stopped != "eos")
        meta["elements"] = len(elements)
        if dropped:
            meta["repeats_dropped"] = dropped
        lines = elements_to_lines(elements, (seen_w, seen_h), page_size)
        return result("ok" if lines else "empty", lines)

    def close(self) -> None:
        self.closed = True


class DotsEngine:
    name: ClassVar[str] = "dots"
    api: ClassVar[int] = 1
    PARAMS: ClassVar[ParamTable] = ParamTable()
    MODEL_KINDS: ClassVar[frozenset[str]] = frozenset()     # its model is thorough mode's, not a pack's
    platforms: ClassVar[frozenset[str]] = PLATFORM_KEYS
    session_per_thread: ClassVar[bool] = False

    def version(self) -> str | None:
        """"dots.mocr-8bit 6d9e750, mlx 0.32.3"; None where mlx isn't installed."""
        if not _on_apple_silicon():
            return None
        try:
            mlx = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError:
            return None
        from ... import thorough

        return f"dots.mocr-8bit {thorough.REVISION[:7]}, mlx {mlx}"

    def availability(self, params: Mapping[str, Any], models: ModelPaths, platform_key: str) -> Availability:
        if platform_key not in self.platforms:
            return Availability(False, "not_on_platform", "dots.mocr runs only on a Mac with Apple silicon")
        if not _on_apple_silicon():
            return Availability(False, "not_on_platform", "dots.mocr runs only on a Mac with Apple silicon")
        try:
            _folder(models)
        except EngineFailure as exc:
            return Availability(False, "missing_model", exc.message)
        missing = _missing_libraries()
        if missing:
            return Availability(False, "broken", f"dots.mocr needs {', '.join(missing)} (pdfintake[thorough])")
        return Availability(True, "available", "")

    def open(self, use_id: str, role: Role, params: Mapping[str, Any], models: ModelPaths) -> DotsSession:
        """A session on the model in ``models``' folder, loaded once per process. Off Apple silicon
        its reads are ``unavailable``. EngineFailure when the model or the libraries are missing."""
        if not _on_apple_silicon():
            return DotsSession(self, use_id, role, None, "dots.mocr runs only on a Mac with Apple silicon")
        return DotsSession(self, use_id, role, _load(_folder(models)), None)

    def candidate_params(self, lang_tag: str, script: str, assets: Sequence[Asset]) -> list[dict[str, Any]]:
        return []                                # thorough mode adds it; no pack lists it

    def contract_case(self) -> ContractCase | None:
        """The canary page with the installed thorough-mode model; None when it isn't installed."""
        if not _on_apple_silicon() or _missing_libraries():
            return None
        from ... import thorough

        if thorough.model_problems():
            return None
        try:
            models = thorough.model_paths()
        except Exception:
            return None
        return ContractCase(params={}, models={name: models[name] for name in REQUIRED_FILES}, page=_canary_page,
                            platform_key="darwin-arm64", timeout_s=DEFAULT_TIMEOUT_S)


ENGINE = DotsEngine()


# --------------------------------------------------------------------------- the canary

CANARY_TEXT = "Sample invoice 4521"


def _canary_image() -> PIL.Image.Image:
    """A letter-shaped page with one line of made-up text near the top, in Pillow's own font."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(img).text((80, 120), CANARY_TEXT, fill="black", font=ImageFont.load_default(size=36))
    return img


def _canary_page(folder: Path) -> PageImage:
    img = _canary_image()
    path = Path(folder) / "dots-canary.png"
    img.save(path, format="PNG", dpi=(100, 100))
    return PageImage(index=0, path=path, width=img.width, height=img.height, dpi=100.0,
                     cropbox=(0.0, 0.0, img.width * 0.72, img.height * 0.72))


def canary(models: ModelPaths) -> str | None:
    """Load the model and read the made-up canary page; None when it reads back, else what went wrong."""
    if not _on_apple_silicon():
        return "dots.mocr runs only on a Mac with Apple silicon"
    try:
        session = DotsSession(ENGINE, "canary", "reader", _load(_folder(models)), None)
    except EngineFailure as exc:
        return exc.message
    meta: dict[str, Any] = {}

    def result(status: str, lines: list[Line], error: str | None = None) -> EngineRead:
        return EngineRead("canary", "dots", "dots:canary", "reader", status, lines, 0.0, error, meta)  # type: ignore[arg-type]

    image = _canary_image()
    read = session._read_image(image, image.size, DEFAULT_TIMEOUT_S, meta, result)
    session.close()
    text = read.text().casefold()
    if read.status != "ok":
        return f"the read came back {read.status}" + (f" ({read.error})" if read.error else "")
    if "4521" not in text or "invoice" not in text:
        return "the test page's text didn't come back"
    return None
