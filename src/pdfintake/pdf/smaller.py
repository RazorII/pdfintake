"""The smaller copy: recompress the images worth it, score every change, offer it only if it pays.

This runs after the lossless file is written and verified, on a copy in staging. It never touches
anything but image streams: every content stream, form (the text layer included), font, the
document Info and the XMP stay byte for byte as they were.

Which images are candidates (``find``):

* **R1**: a lossless (Flate, LZW, RunLength or unfiltered) continuous-tone image of at least
  1 megapixel that is stored at more than 4 bits per pixel: a photo or a colour scan that
  lossless compression can't shrink. It is re-encoded as JPEG at its own size.
* **R2**: any candidate image shown at more than 400 ppi: it is downsampled to 300 ppi, but never
  below 2,000 px on its long side, then encoded as JPEG. An image whose soft mask carries
  ``/Matte`` is never resized (ISO 32000's soft-mask image rules in 11.6.5 require such a mask
  to keep its parent's size), so it can only be an R1 candidate, at its own size.
* Never: 1, 2, 4 or 16-bit images, CCITT, JBIG2 or JPX streams, Indexed, CMYK, Separation,
  DeviceN, Lab and Cal colour spaces, stencil masks, images with a ``/Mask`` or a ``/Decode``
  array, and images used as another image's soft mask. Only images placed by page content
  (through forms at any depth) are considered.

How a candidate is built (``build``): each image is encoded with Pillow at every quality in
``qualities``, lowest first, and the first one whose worst tile scores at least ``bar``
(``quality.METRIC_ID``) is kept, provided it is smaller than the stream it replaces. Otherwise
the image stays lossless and is listed in ``kept_lossless`` with the reason; that includes an
image that can't be scored (smaller than 8 px across at the inspection resolution). The image keeps its
``/ColorSpace``, ``/SMask``, ``/Metadata`` and every other key; only the data, ``/Filter``,
``/DecodeParms`` and (for R2) ``/Width`` and ``/Height`` change.

Pages in ``ImageCandidate.page`` are 0-based indexes; the ``page`` numbers in the report's
``pages`` and ``kept_lossless`` rows are 1-based, as in the decision record (DESIGN §3.2).
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .. import quality

if TYPE_CHECKING:
    import pikepdf
    from PIL import Image as PILImage

    from .meta import MetaDiff

R1_MIN_PIXELS = 1_000_000
R1_MIN_STORED_BPP = 4.0
R2_MIN_PPI = 400.0
R2_TARGET_PPI = 300.0
R2_MIN_LONG_SIDE = 2000

_LOSSLESS_FILTERS = frozenset({"/FlateDecode", "/LZWDecode", "/RunLengthDecode", "/ASCIIHexDecode", "/ASCII85Decode"})
_NEVER_FILTERS = frozenset({"/CCITTFaxDecode", "/JBIG2Decode", "/JPXDecode", "/Crypt"})
_MAX_FORM_DEPTH = 16

Matrix = tuple[float, float, float, float, float, float]
_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class ImageCandidate:
    page: int                                   # 0-based index of the first page that shows it
    xobject: tuple[int, int]                    # (object number, generation)
    rule: Literal["R1", "R2"]
    source: dict[str, Any]                      # filter, bytes, width, height, bpc, colorspace, ppi
    target_ppi: float | None                    # R2 only: the ppi after downsampling


@dataclass
class SmallerReport:
    bytes_before: int
    bytes_after: int
    pages: list[dict[str, Any]]                 # {page, rule, from, to, score}
    kept_lossless: list[dict[str, Any]]         # {page, reason}
    text_layer_identical: bool
    metadata: MetaDiff
    min_score: float | None


# --------------------------------------------------------------------------- page walk

def _mul(m: Matrix, n: Matrix) -> Matrix:
    """m then n, in PDF's row-vector convention (``cm`` sets CTM = m x CTM)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)


def _matrix(raw: Any) -> Matrix | None:
    try:
        vals = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    return vals if len(vals) == 6 and all(math.isfinite(v) for v in vals) else None  # type: ignore[return-value]


def _names(raw: Any) -> list[str]:
    import pikepdf

    if raw is None:
        return []
    if isinstance(raw, pikepdf.Array):
        return [str(x) for x in raw]
    return [str(raw)]


def _get(obj: Any, key: str) -> Any:
    try:
        return obj.get(key)
    except Exception:          # noqa: BLE001 - a damaged dictionary is just "not there"
        return None


def _walk(pdf: pikepdf.Pdf) -> Iterator[tuple[int, Any, float]]:
    """Yield (page index, image stream, ppi) for every image placed by page content."""
    import pikepdf

    for index, page in enumerate(pdf.pages):
        unit = 1.0
        uu = _get(page.obj, "/UserUnit")
        if uu is not None:
            try:
                unit = float(uu) or 1.0
            except (TypeError, ValueError):
                unit = 1.0
        resources = _get(page.obj, "/Resources")
        if resources is None:
            resources = page.resources
        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception:      # noqa: BLE001 - unparseable content: nothing to recompress safely
            continue
        yield from _walk_ops(ops, resources, _IDENTITY, index, unit, 0, set())


def _walk_ops(ops: Any, resources: Any, ctm: Matrix, index: int, unit: float, depth: int,
              forms: set[tuple[int, int]]) -> Iterator[tuple[int, Any, float]]:
    import pikepdf

    stack: list[Matrix] = []
    xobjects = _get(resources, "/XObject") if resources is not None else None
    for operands, operator in ops:
        op = str(operator)
        if op == "q":
            stack.append(ctm)
        elif op == "Q":
            if stack:
                ctm = stack.pop()
        elif op == "cm":
            m = _matrix(operands)
            if m is not None:
                ctm = _mul(m, ctm)
        elif op == "Do" and xobjects is not None and operands:
            xo = _get(xobjects, str(operands[0]))
            if not isinstance(xo, pikepdf.Stream):
                continue
            subtype = str(_get(xo, "/Subtype"))
            if subtype == "/Image":
                ppi = _ppi(ctm, xo, unit)
                if ppi is not None:
                    yield index, xo, ppi
            elif subtype == "/Form" and depth < _MAX_FORM_DEPTH:
                key = xo.objgen
                if key in forms:
                    continue
                fm = _matrix(_get(xo, "/Matrix")) or _IDENTITY
                inner = _get(xo, "/Resources")
                try:
                    form_ops = pikepdf.parse_content_stream(xo)
                except Exception:  # noqa: BLE001
                    continue
                yield from _walk_ops(form_ops, inner if inner is not None else resources, _mul(fm, ctm),
                                     index, unit, depth + 1, forms | {key})


def _ppi(ctm: Matrix, image: Any, unit: float) -> float | None:
    try:
        w, h = int(_get(image, "/Width")), int(_get(image, "/Height"))
    except (TypeError, ValueError):
        return None
    a, b, c, d, _, _ = ctm
    x_in = math.hypot(a, b) * unit / 72.0
    y_in = math.hypot(c, d) * unit / 72.0
    if w <= 0 or h <= 0 or x_in <= 0 or y_in <= 0:
        return None
    return min(w / x_in, h / y_in)


# --------------------------------------------------------------------------- candidates

def _colorspace(image: Any) -> str | None:
    """'DeviceGray', 'DeviceRGB' or 'ICCBased' (one or three components), else None."""
    import pikepdf

    cs = _get(image, "/ColorSpace")
    if isinstance(cs, pikepdf.Name):
        return {"/DeviceGray": "DeviceGray", "/DeviceRGB": "DeviceRGB"}.get(str(cs))
    if isinstance(cs, pikepdf.Array) and len(cs) == 2 and str(cs[0]) == "/ICCBased":
        try:
            n = int(cs[1].get("/N"))
        except (TypeError, ValueError, AttributeError):
            return None
        return "ICCBased" if n in (1, 3) else None
    return None


def _mask_streams(pdf: pikepdf.Pdf) -> set[tuple[int, int]]:
    import pikepdf

    out: set[tuple[int, int]] = set()
    for obj in pdf.objects:
        if isinstance(obj, pikepdf.Stream) and str(_get(obj, "/Subtype")) == "/Image":
            for key in ("/SMask", "/Mask"):
                m = _get(obj, key)
                if isinstance(m, pikepdf.Stream) and m.is_indirect:
                    out.add(m.objgen)
    return out


def _has_matte(image: Any) -> bool:
    """True when the image's soft mask carries /Matte: the mask must then keep the image's size."""
    import pikepdf

    mask = _get(image, "/SMask")
    return isinstance(mask, pikepdf.Stream) and _get(mask, "/Matte") is not None


def r2_size(width: int, height: int, ppi: float) -> tuple[int, int, float] | None:
    """The R2 size for an image shown at ``ppi``: (width, height, ppi after), or None.

    300 ppi, but never below 2,000 px on the long side; None when that isn't smaller.
    """
    if ppi <= R2_MIN_PPI:
        return None
    long_side = max(width, height)
    new_long = max(round(long_side * R2_TARGET_PPI / ppi), R2_MIN_LONG_SIDE)
    if new_long >= long_side:
        return None
    f = new_long / long_side
    nw, nh = max(1, round(width * f)), max(1, round(height * f))
    return nw, nh, ppi * f


def _qualifies(image: Any, masks: set[tuple[int, int]]) -> dict[str, Any] | None:
    """The image's source facts if it may ever be recompressed, else None."""
    if image.objgen in masks or image.objgen == (0, 0):
        return None
    if _get(image, "/ImageMask") is True or _get(image, "/Mask") is not None or _get(image, "/Decode") is not None:
        return None
    if _get(image, "/SMaskInData") not in (None, 0):
        return None
    try:
        bpc = int(_get(image, "/BitsPerComponent"))
        w, h = int(_get(image, "/Width")), int(_get(image, "/Height"))
    except (TypeError, ValueError):
        return None
    if bpc != 8 or w <= 0 or h <= 0:
        return None
    filters = _names(_get(image, "/Filter"))
    if any(f in _NEVER_FILTERS for f in filters):
        return None
    if "/DCTDecode" in filters:
        if filters[-1] != "/DCTDecode":
            return None
        kind = "lossy"
    elif all(f in _LOSSLESS_FILTERS for f in filters):
        kind = "lossless"
    else:
        return None
    cs = _colorspace(image)
    if cs is None:
        return None
    try:
        raw = len(image.read_raw_bytes())
    except Exception:          # noqa: BLE001
        return None
    return {"filter": filters[-1].lstrip("/") if filters else "none", "bytes": raw, "width": w, "height": h,
            "bpc": bpc, "colorspace": cs, "kind": kind}


def find(pdf: Path) -> list[ImageCandidate]:
    """Every image worth trying in a smaller copy, by R1 or R2, in page then object order."""
    import pikepdf

    with pikepdf.open(pdf) as doc:
        masks = _mask_streams(doc)
        seen: dict[tuple[int, int], tuple[int, float, Any]] = {}
        for index, image, ppi in _walk(doc):
            key = image.objgen
            if key == (0, 0):
                continue
            if key in seen:
                first, low, obj = seen[key]
                seen[key] = (first, min(low, ppi), obj)   # the largest placement decides
            else:
                seen[key] = (index, ppi, image)
        out: list[ImageCandidate] = []
        for key, (index, ppi, image) in seen.items():
            facts = _qualifies(image, masks)
            if facts is None:
                continue
            w, h = facts["width"], facts["height"]
            source = {k: v for k, v in facts.items() if k != "kind"}
            source["ppi"] = round(ppi, 3)
            r2 = None if _has_matte(image) else r2_size(w, h, ppi)
            if r2 is not None:
                out.append(ImageCandidate(index, key, "R2", source, round(r2[2], 3)))
            elif (facts["kind"] == "lossless" and w * h >= R1_MIN_PIXELS
                  and facts["bytes"] * 8 / (w * h) > R1_MIN_STORED_BPP):
                out.append(ImageCandidate(index, key, "R1", source, None))
    out.sort(key=lambda c: (c.page, c.xobject))
    return out


# --------------------------------------------------------------------------- build

def _encode(img: PILImage.Image, q: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, subsampling=2, optimize=True, progressive=False)
    return buf.getvalue()


def _decode_source(image: Any) -> PILImage.Image | None:
    """The image's own colour samples, as stored: no soft mask, no /Decode, no colour conversion."""
    import pikepdf
    from PIL import Image

    cs = _colorspace(image)
    try:
        w, h = int(image.Width), int(image.Height)
        filters = _names(_get(image, "/Filter"))
        if filters and filters[-1] == "/DCTDecode":
            if len(filters) == 1:
                data = bytes(image.read_raw_bytes())
            else:                                   # e.g. ASCII85 around the JPEG: undo those only
                data = bytes(image.read_bytes(pikepdf.StreamDecodeLevel.generalized))
            img = Image.open(io.BytesIO(data), formats=["JPEG"])
            img.load()
        else:
            channels = 1 if cs == "DeviceGray" else 3 if cs == "DeviceRGB" else int(image.ColorSpace[1].N)
            mode = "L" if channels == 1 else "RGB"
            data = bytes(image.read_bytes())
            if len(data) < w * h * channels:
                return None
            img = Image.frombytes(mode, (w, h), data[:w * h * channels])
    except Exception:          # noqa: BLE001 - anything we can't read exactly stays lossless
        return None
    if img.mode not in ("L", "RGB") or img.size != (w, h):
        return None
    return img


def _stream_bytes(obj: Any) -> bytes:
    return bytes(obj.read_raw_bytes())


def _content_fingerprint(pdf: pikepdf.Pdf) -> list[tuple[int, str, bytes]]:
    """Raw bytes of every page content stream and every form reachable from a page."""
    import pikepdf

    out: list[tuple[int, str, bytes]] = []
    for index, page in enumerate(pdf.pages):
        contents = _get(page.obj, "/Contents")
        streams = list(contents) if isinstance(contents, pikepdf.Array) else [contents]
        for s in streams:
            if isinstance(s, pikepdf.Stream):
                out.append((index, "contents", _stream_bytes(s)))
        todo = [_get(page.obj, "/Resources")]
        seen: set[tuple[int, int]] = set()
        while todo:
            res = todo.pop()
            xobjects = _get(res, "/XObject") if res is not None else None
            if xobjects is None:
                continue
            for name in sorted(xobjects.keys()):
                xo = _get(xobjects, name)
                if isinstance(xo, pikepdf.Stream) and str(_get(xo, "/Subtype")) == "/Form":
                    if xo.is_indirect:
                        if xo.objgen in seen:
                            continue
                        seen.add(xo.objgen)
                    out.append((index, str(name), _stream_bytes(xo)))
                    todo.append(_get(xo, "/Resources"))
    return out


def _save(pdf: pikepdf.Pdf, dest: Path) -> None:
    import pikepdf

    pdf.save(dest, deterministic_id=True, object_stream_mode=pikepdf.ObjectStreamMode.generate,
             compress_streams=False, stream_decode_level=pikepdf.StreamDecodeLevel.none,
             recompress_flate=False, fix_metadata_version=False)


def _fmt(score: float) -> str:
    return f"{score:.1f}"


def build(src: Path, dest: Path, cands: Sequence[ImageCandidate], *, bar: float = 90.0,
          qualities: tuple[int, ...] = (95, 92, 90, 88)) -> SmallerReport:
    """Write the candidate copy of ``src`` to ``dest`` (in staging) and report what changed.

    Deterministic: the same input and candidates give the same bytes.
    """
    import pikepdf
    from PIL import Image

    from . import meta, verify

    src, dest = Path(src), Path(dest)
    if src.resolve() == dest.resolve():
        raise ValueError("build writes a new file; dest must differ from src")
    if not qualities or any(not (1 <= int(q) <= 100) for q in qualities):
        raise ValueError("qualities must be JPEG qualities between 1 and 100")
    order = sorted({int(q) for q in qualities})

    pages: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    with pikepdf.open(src) as pdf:
        before_fp = _content_fingerprint(pdf)
        done: set[tuple[int, int]] = set()
        for cand in sorted(cands, key=lambda c: (c.page, c.xobject)):
            if cand.xobject in done:
                continue
            done.add(cand.xobject)
            page_no = cand.page + 1
            try:
                obj = pdf.get_object(cand.xobject)
            except Exception:  # noqa: BLE001
                obj = None
            if not isinstance(obj, pikepdf.Stream) or str(_get(obj, "/Subtype")) != "/Image":
                kept.append({"page": page_no, "reason": "the image is no longer in the file"})
                continue
            raw_len = len(obj.read_raw_bytes())
            ref = _decode_source(obj)
            if ref is None:
                kept.append({"page": page_no, "reason": "the image could not be decoded exactly"})
                continue
            ppi = float(cand.source.get("ppi") or 0.0)
            if ppi <= 0:
                kept.append({"page": page_no, "reason": "the image's resolution is unknown"})
                continue
            work, dist_ppi = ref, ppi
            if cand.rule == "R2":
                if _has_matte(obj):
                    kept.append({"page": page_no, "reason": "its soft mask carries /Matte, so it can't be resized"})
                    continue
                size = r2_size(ref.width, ref.height, ppi)
                if size is None:
                    kept.append({"page": page_no, "reason": "the image is not above 400 ppi"})
                    continue
                work = ref.resize(size[:2], Image.Resampling.LANCZOS)
                # The long side's exact ratio: each axis is then within half a pixel of the same
                # area, which a thin image's rounded short side wouldn't be.
                dist_ppi = size[2]
            chosen: tuple[int, bytes, float] | None = None
            best = -math.inf
            try:
                for q in order:
                    data = _encode(work, q)
                    with Image.open(io.BytesIO(data), formats=["JPEG"]) as dec:
                        dec.load()
                        a, b = quality.at_inspection(ref, dec, ppi, dist_ppi)
                    score = quality.worst_tile(a, b).score
                    best = max(best, score)
                    if score >= bar:
                        chosen = (q, data, score)
                        break
            except ValueError as exc:          # too small to score, or a geometry we can't compare
                kept.append({"page": page_no, "reason": f"the image could not be scored ({exc})"})
                continue
            if chosen is None:
                kept.append({"page": page_no, "reason": f"scored {_fmt(best)}, below the bar of {bar:g}"})
                continue
            q, data, score = chosen
            if len(data) >= raw_len:
                kept.append({"page": page_no, "reason": f"the JPEG would not be smaller ({len(data)} vs {raw_len} bytes)"})
                continue
            obj.write(data, filter=pikepdf.Name.DCTDecode)
            obj.Width = work.width
            obj.Height = work.height
            obj.BitsPerComponent = 8
            pages.append({"page": page_no, "rule": cand.rule,
                          "from": {"filter": cand.source.get("filter", "unknown"), "bytes": raw_len},
                          "to": {"filter": "DCTDecode", "q": q, "bytes": len(data)},
                          "score": round(score, 2)})
        after_fp = _content_fingerprint(pdf)
        _save(pdf, dest)

    with pikepdf.open(dest) as out:
        saved_fp = _content_fingerprint(out)
    streams_same = before_fp == after_fp == saved_fp
    layer = verify.text_layer_identical(src, dest)
    metadata = meta.diff(meta.snapshot(src), meta.snapshot(dest))
    return SmallerReport(
        bytes_before=src.stat().st_size,
        bytes_after=dest.stat().st_size,
        pages=pages,
        kept_lossless=kept,
        text_layer_identical=bool(streams_same and getattr(layer, "ok", False)),
        metadata=metadata,
        min_score=min((p["score"] for p in pages), default=None),
    )


def worth_offering(rep: SmallerReport, *, min_pct: float = 30.0, min_bytes: int = 0) -> bool:  # decided 2026-11-05: no byte floor
    """True when the copy changed something, kept the layer and metadata, and saves enough.

    Enough means at least ``min_pct`` percent of the lossless file and at least ``min_bytes``
    bytes (no byte floor by default). The bar on each changed image was applied in ``build``.
    """
    if not rep.pages or not rep.text_layer_identical or rep.bytes_before <= 0:
        return False
    md = rep.metadata
    if md is not None and (list(getattr(md, "changed", []) or []) or list(getattr(md, "dropped", []) or [])):
        return False
    saved = rep.bytes_before - rep.bytes_after
    if saved <= 0 or saved < min_bytes:
        return False
    return saved * 100 >= min_pct * rep.bytes_before
