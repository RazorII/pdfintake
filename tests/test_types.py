"""Shared data formats: the per-page OCR structures and their JSON, the errors, the manifest schema,
and the fixture builders that make synthetic inputs for every other test."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import io
import json
import pickle
import pkgutil
import re
from collections import Counter
from importlib import resources
from pathlib import Path

import pytest

import pdfintake
from fixtures import build
from pdfintake import errors
from pdfintake.types import (FILL_KINDS, READ_STATUSES, ROLES, EngineRead, Fill, LangCall, LayerLine, LayerPage, Line,
                             OsdResult, PageImage, PageOCR, Pick, ScriptCall, Word, to_jsonable)

CANARY = "zqx-canary-7781-not-for-records"


def _page_ocr(text: str) -> PageOCR:
    read = build.engine_read("tesseract", [text, "a second made-up line"], spec="tesseract:eng:psm3", with_words=True)
    pick = Pick("single", 1, "tesseract", list(read.lines), {"tesseract": 1.0}, False, "the only reader")
    fill = Fill("id", text, 0.97, build.quad(1300, 120, 1420, 170), "rapidocr", near=text)
    return PageOCR(index=0, script=ScriptCall("Latin", 0.0, "osd", 91.0), languages=[LangCall("en", 0.99, "langid", 4)],
                   pack="en", composition="script", reads=[read], pick=pick, fills=[fill], repairs={"diacritic_word": 0},
                   must_survive={"double-acute": {"tesseract": 3, "layer": 3}}, warnings=[])


# --------------------------------------------------------------------------- types and JSON


def test_literal_sets():
    assert ROLES == ("reader", "checker", "probe")
    assert READ_STATUSES == ("ok", "empty", "failed", "timeout", "unavailable")
    assert FILL_KINDS == ("line", "word", "number", "id", "link")


def test_engine_read_json_shape():
    read = EngineRead("tesseract", "tesseract", "tesseract:tha+eng:psm3", "reader", "ok",
                      [Line("made-up line", 0.88, build.quad(112, 204, 905, 251))], 1.63,
                      meta={"mean_conf": 0.81, "version": "5.5.3"})
    data = to_jsonable(read, include_text=True)
    assert data == {
        "use_id": "tesseract", "engine": "tesseract", "spec": "tesseract:tha+eng:psm3", "role": "reader",
        "status": "ok", "seconds": 1.63, "error": None, "meta": {"mean_conf": 0.81, "version": "5.5.3"},
        "lines": [{"text": "made-up line", "conf": 0.88,
                   "box": [[112.0, 204.0], [905.0, 204.0], [905.0, 251.0], [112.0, 251.0]], "words": []}],
    }
    json.dumps(data)
    assert build.read_from_json(json.loads(json.dumps(data))) == read
    assert read.text() == "made-up line"


def test_pick_and_fill_json_shapes():
    pick = Pick("corroborated_trigrams", 1, "vision", [], {"vision": 299, "tesseract": 231}, False,
                "most corroborated Thai trigrams")
    assert to_jsonable(pick) == {"rule": "corroborated_trigrams", "version": 1, "chosen": "vision", "lines": [],
                                 "scores": {"vision": 299, "tesseract": 231}, "fallback": False,
                                 "reason": "most corroborated Thai trigrams", "warnings": []}
    low = Pick("primary_with_floor", 1, "tesseract", [], {"tesseract": 0.3}, False, "below the floor", ["agreement_low"])
    assert to_jsonable(low)["warnings"] == ["agreement_low"]
    fill = Fill("number", "4410", 0.97, build.quad(1300, 120, 1420, 170), "rapidocr")
    assert to_jsonable(fill, include_text=True) == {
        "kind": "number", "text": "4410", "conf": 0.97,
        "box": [[1300.0, 120.0], [1420.0, 120.0], [1420.0, 170.0], [1300.0, 170.0]], "source": "rapidocr", "near": None}


def test_document_text_never_survives_unless_asked():
    ocr = _page_ocr(CANARY)
    plain = json.dumps(to_jsonable(ocr), ensure_ascii=False)
    assert CANARY not in plain
    data = to_jsonable(ocr)
    assert data["reads"][0]["lines"][0] == {"chars": len(CANARY), "conf": 0.9, "box": data["reads"][0]["lines"][0]["box"],
                                            "words": data["reads"][0]["lines"][0]["words"]}
    assert "text" not in data["fills"][0] and "near" not in data["fills"][0]
    layer = LayerPage(0, 1000, 1400, (LayerLine(CANARY, build.quad(1, 1, 50, 20)),))
    assert CANARY not in json.dumps(to_jsonable(layer))
    assert CANARY in json.dumps(to_jsonable(ocr, include_text=True), ensure_ascii=False)


def test_to_jsonable_conversions():
    assert to_jsonable(Path("/tmp/example.pdf")) == str(Path("/tmp/example.pdf"))
    assert to_jsonable((1, (2.5, "x"))) == [1, [2.5, "x"]]
    assert to_jsonable(frozenset({"b", "a"})) == ["a", "b"]
    assert to_jsonable(Counter({"U+4E2D": 1})) == {"U+4E2D": 1}
    assert to_jsonable({1: True, "k": None}) == {1: True, "k": None}
    assert to_jsonable(OsdResult("Thai", 48.2, 0, 12.5)) == {"script": "Thai", "script_conf": 48.2, "orient_deg": 0,
                                                             "orient_conf": 12.5}
    with pytest.raises(TypeError):
        to_jsonable(b"raw bytes")
    with pytest.raises(TypeError):
        to_jsonable({("a", 1): 2})


def test_frozen_types_are_frozen():
    w = Word("x", 0.5, build.quad(0, 0, 1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.text = "y"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        LayerLine("x", w.box).text = "y"  # type: ignore[misc]


def test_page_image_opens_its_png(tmp_path: Path):
    page = build.page_image(tmp_path, image=build.text_like(320, 200), dpi=200, index=3)
    assert (page.index, page.width, page.height, page.dpi) == (3, 320, 200, 200.0)
    assert page.cropbox == (0.0, 0.0, 320 * 72 / 200, 200 * 72 / 200)
    with page.open() as img:
        assert img.size == (320, 200) and img.format == "PNG"
        assert round(img.info["dpi"][0]) == 200
    wrong = dataclasses.replace(page, width=321)
    with pytest.raises(ValueError, match="321x200"):
        wrong.open()
    jpeg = tmp_path / "not-a-png.png"
    jpeg.write_bytes(build.jpeg_bytes(build.blank(10, 10)))
    with pytest.raises(Exception):
        PageImage(0, jpeg, 10, 10, 72.0, (0, 0, 10, 10)).open()


# --------------------------------------------------------------------------- errors


def test_exit_codes():
    assert (errors.EXIT_OK, errors.EXIT_FAILED, errors.EXIT_USAGE, errors.EXIT_SETUP, errors.EXIT_REFUSED,
            errors.EXIT_STALE) == (0, 1, 2, 3, 4, 5)
    assert (errors.EXIT_KIT_FLOOR, errors.EXIT_KIT_DRIFT, errors.EXIT_KIT_PRIVACY) == (6, 7, 8)


def test_no_decision_exit():
    # DESIGN §12.5: a run never asks, so exit 10 and EXIT_DECISION are gone for good.
    assert not hasattr(errors, "EXIT_DECISION")
    codes = {v for k, v in vars(errors).items() if k.startswith("EXIT_")}
    assert codes == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    classes = _error_classes()
    assert errors.Refused in classes and errors.IntegrityError in classes
    assert all(cls.exit_code in codes and cls.exit_code != 10 for cls in classes)


def test_no_source_file_sets_exit_10():
    # The class walk above sees only the modules that import here; this reads every file in src/.
    findings = {str(p.relative_to(SRC)): f for p in _source_files() if (f := _exit_ten_findings(p.read_text("utf-8")))}
    assert findings == {}, f"exit 10 is retired (DESIGN §12.5): {findings}"


@pytest.mark.parametrize("code", [
    "class Waiting(PdfIntakeError):\n    exit_code = 10\n",
    "class Waiting(PdfIntakeError):\n    exit_code: int = 10\n",
    "from .errors import EXIT_DECISION\n",
    "class Waiting(PdfIntakeError):\n    exit_code = errors.EXIT_DECISION\n",
    "class Waiting(PdfIntakeError):\n    exit_code = ASK\n",
    "def f(err):\n    err.exit_code = 10\n",
    "import sys\ndef f():\n    sys.exit(10)\n",
    "def f():\n    raise SystemExit(10)\n",
    "def f(err):\n    setattr(err, 'exit_code', 10)\n",
])
def test_the_exit_10_scan_catches_planted_code(code):
    assert _exit_ten_findings(code)


@pytest.mark.parametrize("code", [
    "class Fine(PdfIntakeError):\n    exit_code = EXIT_REFUSED\n",
    "class Fine(PdfIntakeError):\n    exit_code: int = errors.EXIT_KIT_DRIFT\n",
    "class Fine(PdfIntakeError):\n    exit_code = 4\n",
    "def f():\n    return 10\n",                       # a 10 that isn't an exit
    "PAGES = 10\n",
    '"""Exit 10 is retired."""\n',
])
def test_the_exit_10_scan_allows_ordinary_code(code):
    assert _exit_ten_findings(code) == []


@pytest.mark.parametrize("stmt, excused", [
    ("import pdfintake.no_such_module_yet", True),          # a planned module not on disk
    ("from pdfintake import no_such_module_yet", True),
    ("from pdfintake.errors import NoSuchName", False),     # a missing name in a real module
    ("import no_such_third_party_module", False),
])
def test_the_class_walk_excuses_only_a_missing_sibling(stmt, excused):
    with pytest.raises(ImportError) as info:
        exec(stmt, {})
    assert _missing_sibling(info.value) is excused


SRC = Path(__file__).resolve().parents[1] / "src"
_EXIT_NAMES = {k: v for k, v in vars(errors).items() if k.startswith("EXIT_")}


def _source_files() -> list[Path]:
    return sorted(p for p in (SRC / "pdfintake").rglob("*.py") if "__pycache__" not in p.parts)


def _exit_value(node: ast.AST) -> int | None:
    """The exit code an expression names, or None when it names nothing this test can check."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
    return _EXIT_NAMES.get(name) if name else None


def _exit_ten_findings(source: str) -> list[str]:
    """Every place in ``source`` that sets exit 10, names EXIT_DECISION or sets an exit_code this test can't resolve."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        where = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name) and node.id == "EXIT_DECISION" or (
                isinstance(node, ast.Attribute) and node.attr == "EXIT_DECISION") or (
                isinstance(node, ast.alias) and node.name == "EXIT_DECISION"):
            found.append(f"line {where}: EXIT_DECISION")
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            targets, value = [node.target], node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
              and len(node.args) == 3 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "exit_code"):
            targets, value = [ast.Name("exit_code")], node.args[2]
        if value is not None and any((t.id if isinstance(t, ast.Name) else getattr(t, "attr", None)) == "exit_code"
                                     for t in targets):
            code = _exit_value(value)
            if code is None or code == 10 or code not in _EXIT_NAMES.values():
                found.append(f"line {where}: exit_code = {ast.unparse(value)}")
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if fname in ("exit", "_exit", "SystemExit") and _exit_value(node.args[0]) == 10:
                found.append(f"line {where}: {fname}(10)")
    return found


def _missing_sibling(exc: ImportError) -> bool:
    """True when the import failed only because a pdfintake module isn't on disk yet."""
    name = exc.name or ""
    m = re.match(r"cannot import name '(\w+)'", str(exc))
    if m and not isinstance(exc, ModuleNotFoundError):
        if not SRC.joinpath(*name.split("."), "__init__.py").is_file():
            return False                                    # a missing name in a module, not a missing module
        name = f"{name}.{m.group(1)}"
    if name != "pdfintake" and not name.startswith("pdfintake."):
        return False
    base = SRC.joinpath(*name.split("."))
    return not (base.with_suffix(".py").is_file() or (base / "__init__.py").is_file())


def _error_classes() -> list[type[errors.PdfIntakeError]]:
    """Every PdfIntakeError subclass, after importing every module in the package so none is missed."""
    for mod in pkgutil.walk_packages(pdfintake.__path__, "pdfintake."):
        if mod.name.endswith(".__main__"):                  # runs the CLI when imported
            continue
        try:
            importlib.import_module(mod.name)
        except ImportError as exc:
            # A sibling another unit hasn't built yet (DESIGN §2.1); the source scan still reads this file.
            if not _missing_sibling(exc):
                raise
    found, todo = [], [errors.PdfIntakeError]
    while todo:
        cls = todo.pop()
        found.append(cls)
        todo.extend(cls.__subclasses__())
    return found


@pytest.mark.parametrize("cls, code, parents", [
    (errors.PdfIntakeError, 1, ()),
    (errors.UsageError, 2, ()),
    (errors.SetupError, 3, ()),
    (errors.PackError, 3, (errors.SetupError,)),
    (errors.IntegrityError, 3, (errors.PackError, errors.SetupError)),
    (errors.Refused, 4, ()),
    (errors.StaleRecord, 5, ()),
    (errors.InputChanged, 5, (errors.StaleRecord,)),
    (errors.VerifyFailed, 1, ()),
    (errors.LosslessUnavailable, 1, ()),
    (errors.EngineFailure, 1, ()),
])
def test_error_classes(cls, code, parents):
    err = cls("some_code", "Something went wrong with example.pdf", path="example.pdf")
    assert isinstance(err, errors.PdfIntakeError) and isinstance(err, Exception)
    assert all(isinstance(err, p) for p in parents)
    assert err.exit_code == code and cls.exit_code == code
    assert err.code == "some_code" and err.message == str(err) == "Something went wrong with example.pdf"
    assert err.detail == {"path": "example.pdf"}
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is cls and back.code == err.code and back.message == err.message and back.detail == err.detail


def test_verify_failed_carries_checks():
    err = errors.VerifyFailed("verify_failed", "2 checks failed", checks=["a", "b"])
    assert err.checks == ["a", "b"]
    assert errors.VerifyFailed("verify_failed", "x").checks == []


# --------------------------------------------------------------------------- the manifest schema


def _schema() -> dict:
    text = resources.files("pdfintake.langpacks").joinpath("manifest.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def _manifest(**overrides) -> dict:
    sha = "0" * 64
    m = {
        "schema": 1, "id": "xx", "version": "1.0.0", "requires_core": ">=0.1.0", "title": "Example",
        "languages": [{"tag": "xx", "iso639_3": "xxx", "role": "primary"}],
        "scripts": ["Latin"],
        "detect": {"osd_scripts": ["Latin"], "langid": {"codes": ["xx"]}},
        "readers": {"default": {
            "engines": [{"id": "tesseract", "engine": "tesseract", "role": "reader", "params": {"lang": "xxx", "psm": 3},
                         "models": ["xxx", "en:eng"]}],
            "pick": {"rule": "single", "version": 1, "params": {"reader": "tesseract"}},
            "fallback": "none"}},
        "normalize": ["nfc"], "repairs": [], "fills": [{"name": "numbers", "version": 1}],
        "models": [{"key": "xxx", "file": "xxx.traineddata", "engine": "tesseract", "kind": "traineddata", "bytes": 1000,
                    "sha256": sha, "licence": "Apache-2.0", "notice": "NOTICE-tessdata",
                    "source": "https://github.com/tesseract-ocr/tessdata_best/raw/" + "a" * 40 + "/xxx.traineddata"}],
        "fonts": [], "must_survive": [{"name": "marks", "chars": "áé", "unit": "word", "floor": 0.9}],
        "bench": {"kit": "1.0", "corpus": "none", "platforms": {}},
    }
    m.update(overrides)
    return m


def test_schema_is_valid_draft_2020_12():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_manifest())


@pytest.mark.parametrize("source, ok", [
    ("https://github.com/tesseract-ocr/tessdata_best/raw/" + "a" * 40 + "/xxx.traineddata", True),
    ("https://github.com/tesseract-ocr/tessdata/raw/" + "b" * 40 + "/xxx.traineddata", True),
    ("https://github.com/RapidAI/RapidOCR/releases/download/v3.8.0/onnx/rec.onnx", True),
    ("https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/rec/th_rec.onnx", True),
    ("https://raw.example.com/xxx.traineddata", False),
    ("http://github.com/tesseract-ocr/tessdata_best/raw/" + "a" * 40 + "/xxx.traineddata", False),
    ("https://github.com/tesseract-ocr/tessdata_best/raw/main/xxx.traineddata", False),
    ("https://github.com/tesseract-ocr/tessdata_best/raw/" + "a" * 40 + "/xxx.traineddata?x=1", False),
    ("https://github.com/someone/tessdata_best/raw/" + "a" * 40 + "/xxx.traineddata", False),
])
def test_schema_holds_the_model_source_allowlist(source, ok):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(_schema())
    m = _manifest()
    m["models"][0]["source"] = source
    assert validator.is_valid(m) is ok


def test_schema_holds_the_font_source_allowlist():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(_schema())
    font = {"key": "noto-sans", "file": "NotoSans-Regular.ttf", "covers": ["Latin"], "bytes": 10, "sha256": "0" * 64,
            "licence": "OFL-1.1", "source": "https://github.com/notofonts/latin-greek-cyrillic/releases/download/v1/NotoSans-Regular.ttf"}
    assert validator.is_valid(_manifest(fonts=[font]))
    assert not validator.is_valid(_manifest(fonts=[dict(font, source="https://fonts.example.com/NotoSans-Regular.ttf")]))
    assert not validator.is_valid(_manifest(fonts=[dict(font, licence="Apache-2.0")]))


def test_schema_rejects_unknown_fields_and_bad_ids():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(_schema())
    assert not validator.is_valid(_manifest(extra=True))
    assert not validator.is_valid(_manifest(id="XX"))
    assert not validator.is_valid(_manifest(schema=2))
    assert not validator.is_valid(_manifest(title="lower case title"))
    missing = _manifest()
    del missing["bench"]
    assert not validator.is_valid(missing)


# --------------------------------------------------------------------------- fixture builders


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_image_builders_are_deterministic():
    for make in (lambda: build.noise(64, 48, seed=3), lambda: build.photo_like(80, 60, seed=1),
                 lambda: build.text_like(300, 200, seed=2), lambda: build.blank(10, 10, mode="CMYK")):
        assert make().tobytes() == make().tobytes()
    assert build.noise(32, 32, seed=1).tobytes() != build.noise(32, 32, seed=2).tobytes()
    assert _sha(build.png_bytes(build.text_like(), dpi=300)) == _sha(build.png_bytes(build.text_like(), dpi=300))
    assert _sha(build.jpeg_bytes(build.photo_like(64, 64), orientation=6)) == _sha(build.jpeg_bytes(build.photo_like(64, 64), orientation=6))


def test_jpeg_orientation_tag():
    from PIL import Image

    with Image.open(io.BytesIO(build.jpeg_bytes(build.photo_like(40, 30), orientation=6))) as img:
        assert img.getexif()[0x0112] == 6


@pytest.mark.parametrize("color_type, depth, interlace", [(2, 16, False), (0, 16, False), (0, 1, False), (3, 4, False),
                                                          (6, 8, False), (2, 8, True), (4, 16, True)])
def test_raw_png_round_trips(color_type, depth, interlace):
    from PIL import Image

    rows = build.png_samples(13, 7, color_type=color_type, bit_depth=depth, seed=5)
    data = build.raw_png(13, 7, color_type=color_type, bit_depth=depth, rows=rows, dpi=300, interlace=interlace)
    assert data == build.raw_png(13, 7, color_type=color_type, bit_depth=depth, seed=5, dpi=300, interlace=interlace)
    with Image.open(io.BytesIO(data), formats=["PNG"]) as img:
        img.load()
        assert img.size == (13, 7)
        if (color_type, depth) == (0, 16):
            assert [img.getpixel((x, 2)) for x in range(13)] == rows[2]
        if (color_type, depth) == (2, 8):
            assert list(img.getpixel((4, 3))) == rows[3][12:15]
    with pytest.raises(ValueError):
        build.raw_png(2, 2, color_type=2, bit_depth=4)


def test_pdf_builders_are_deterministic_and_readable(tmp_path: Path):
    import pikepdf
    import pypdfium2 as pdfium

    spec = [build.PageSpec(texts=(build.TextSpec("Visible sample text"), build.TextSpec("Hidden sample", y=600, mode=3))),
            build.PageSpec(width=842, height=595, rotate=90, images=(build.ImageSpec(build.photo_like(60, 40)),)),
            build.PageSpec(images=(build.ImageSpec(build.jpeg_bytes(build.photo_like(40, 40)), 10, 10, 100, 100),),
                           texts=(build.TextSpec("In a form (with parens)"),), in_form=True,
                           cropbox=(10, 10, 600, 780))]
    a, b = build.pdf_bytes(spec), build.pdf_bytes(spec)
    assert a == b
    with pikepdf.open(io.BytesIO(a)) as pdf:
        assert len(pdf.pages) == 3
        assert int(pdf.pages[1].obj.Rotate) == 90
        assert "/Fm0" in pdf.pages[2].obj.Resources.XObject
    doc = pdfium.PdfDocument(a)
    try:
        text = doc[0].get_textpage().get_text_bounded()
        assert "Visible sample text" in text and "Hidden sample" in text
        assert "In a form (with parens)" in doc[2].get_textpage().get_text_bounded()
    finally:
        doc.close()
    assert build.write_pdf(tmp_path / "two.pdf", 2).stat().st_size > 0
    with pytest.raises(ValueError):
        build.pdf_bytes([build.PageSpec(texts=(build.TextSpec(build.THAI_SAMPLE),))])


def test_box_font_is_deterministic_and_covers_what_it_claims(tmp_path: Path):
    from fontTools.ttLib import TTFont

    chars = "Abc 4521" + build.THAI_SAMPLE
    a, b = build.box_font(chars), build.box_font(chars)
    assert a == b
    font = TTFont(io.BytesIO(a))
    cmap = font.getBestCmap()
    assert set(map(ord, chars)) <= set(cmap)
    assert ord("Z") not in cmap
    hmtx = font["hmtx"]
    assert hmtx[cmap[ord("A")]][0] == 600
    assert hmtx[cmap[0x0E49]][0] == 0          # a Thai tone mark: zero advance
    disguised = TTFont(io.BytesIO(build.box_font("Abc", family="Noto Sans Thai")))
    assert disguised["name"].getDebugName(1) == "Noto Sans Thai"
    assert 0x0E01 not in disguised.getBestCmap()


def test_fpdf_text_pdf_with_a_box_font(tmp_path: Path):
    import pypdfium2 as pdfium

    font = build.write_box_font(tmp_path / "box.ttf", "Abc " + build.THAI_SAMPLE)
    out1 = build.fpdf_text_pdf(tmp_path / "a.pdf", ["Abc", build.THAI_SAMPLE], font)
    out2 = build.fpdf_text_pdf(tmp_path / "b.pdf", ["Abc", build.THAI_SAMPLE], font)
    assert out1.read_bytes() == out2.read_bytes()
    doc = pdfium.PdfDocument(out1)
    try:
        text = doc[0].get_textpage().get_text_bounded()
    finally:
        doc.close()
    assert "Abc" in text and build.THAI_SAMPLE in text
    import pikepdf

    hidden = build.fpdf_text_pdf(tmp_path / "c.pdf", ["Abc"], font, invisible=True)
    with pikepdf.open(hidden) as pdf:
        modes = [int(ops[0]) for ops, op in pikepdf.parse_content_stream(pdf.pages[0]) if str(op) == "Tr"]
    assert modes and set(modes) == {3}


def test_synthetic_reads():
    read = build.engine_read("vision", ["first made-up line", "second"], role="reader")
    assert read.status == "ok" and len(read.lines) == 2 and read.text() == "first made-up line\nsecond"
    assert read.lines[1].box[0][1] > read.lines[0].box[2][1]          # laid out top to bottom
    assert build.engine_read("vision", []).status == "empty"
    failed = build.engine_read("vision", status="failed", error="made-up error")
    assert failed.lines == [] and failed.error == "made-up error"
    words = build.line("two words", with_words=True).words
    assert [w.text for w in words] == ["two", "words"]
    assert build.engine_read("x", ["same"]) == build.engine_read("x", ["same"])
