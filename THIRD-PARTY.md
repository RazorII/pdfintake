# Third-party software and data

pdfintake's own code is MIT licensed. Nothing below is copied into this repository except the code listed under *Code ported from elsewhere*: the libraries are installed from PyPI when you install pdfintake, the models and fonts come in data wheels built from their pinned upstream files, and thorough mode's model is downloaded from Hugging Face when you turn the mode on. Licence ids are SPDX identifiers, read from each package's metadata.

## Runtime dependencies

| Package | Licence | What pdfintake uses it for |
|---|---|---|
| `certifi` | MPL-2.0 | CA certificates for the language-pack download |
| `fonttools` | MIT | glyph widths and coverage for the text layer; subsetting inside fpdf2 |
| `fpdf2` | LGPL-3.0-only | writing the invisible text layer (pinned to 2.9.0, used unmodified) |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | image arithmetic and the quality score |
| `onnxruntime` | MIT | runs RapidOCR's models |
| `opencv-python-headless` | Apache-2.0 | image handling inside RapidOCR; its wheels bundle codec libraries with their own licences |
| `pdfintake-data-en` | Apache-2.0 AND OFL-1.1 | the English data: Tesseract `osd` and `eng`, RapidOCR's detection and Latin models, Noto Sans, and a test image (built by this project, see below) |
| `pikepdf` | MPL-2.0 | reading and writing PDFs; bundles qpdf (Apache-2.0) |
| `pillow` | MIT-CMU | decoding and, where unavoidable, encoding images |
| `pillow-heif` | BSD-3-Clause | decoding HEIC photos. That is the licence of its source code; its binary wheels are GPL-2.0, its project says, because they bundle libheif and libde265 (LGPL-3.0) and x265 (GPL-2.0). uv installs it from PyPI on your computer, and this project never redistributes it |
| `py3langid` | BSD-3-Clause | telling languages apart on Latin-script pages |
| `pyobjc-framework-Vision` | MIT | Apple Vision OCR (macOS only) |
| `pypdfium2` | Apache-2.0 OR BSD-3-Clause | rendering pages and reading their text and images; its wheels bundle PDFium and the libraries it builds with (FreeType, ICU, Little CMS, libjpeg-turbo, OpenJPEG, libpng, zlib and others), each under its own licence, listed in the wheel |
| `rapidocr` | Apache-2.0 | text detection, a second OCR reading, and the page-orientation vote |
| `tesserocr` | MIT | Tesseract OCR in process; bundles Tesseract (Apache-2.0) and Leptonica (BSD-2-Clause), and on Windows further libraries (ICU, OpenSSL, libcurl, libssh2, libarchive, libxml2 and others) |

## Installed with them

| Package | Licence | Pulled in by |
|---|---|---|
| `charset-normalizer` | MIT | requests |
| `colorlog` | MIT | rapidocr |
| `flatbuffers` | Apache-2.0 | onnxruntime |
| `idna` | BSD-3-Clause | requests |
| `lxml` | BSD-3-Clause | pikepdf |
| `omegaconf` | BSD-3-Clause | rapidocr |
| `packaging` | Apache-2.0 OR BSD-2-Clause | pikepdf, onnxruntime |
| `protobuf` | BSD-3-Clause | onnxruntime |
| `pyclipper` | MIT | rapidocr |
| `pyyaml` | MIT | rapidocr, omegaconf |
| `requests` | Apache-2.0 | rapidocr |
| `shapely` | BSD-3-Clause | rapidocr |
| `six` | MIT | rapidocr |
| `tqdm` | MPL-2.0 AND MIT | rapidocr |
| `urllib3` | MIT | requests |
| `pyobjc-core` | MIT | pyobjc-framework-Vision (macOS only) |
| `pyobjc-framework-Cocoa` | MIT | pyobjc-framework-Vision (macOS only) |
| `pyobjc-framework-CoreML` | MIT | pyobjc-framework-Vision (macOS only) |
| `pyobjc-framework-Quartz` | MIT | pyobjc-framework-Vision (macOS only) |

The licences of the native libraries bundled inside the tesserocr, pypdfium2 and opencv-python-headless wheels are being read in full before the first release; this file will list each one.

## Models and fonts

The data wheels (`pdfintake-data-en` and `pdfintake-lang-<id>`) carry only files downloaded from these upstream projects at pinned commits or releases, each with its SHA-256, plus the upstream LICENSE and NOTICE files:

| Data | Licence | Upstream |
|---|---|---|
| Tesseract models (`tessdata`, `tessdata_best`, `tessdata_fast`) | Apache-2.0 | [github.com/tesseract-ocr](https://github.com/tesseract-ocr) |
| RapidOCR / PaddleOCR models (ONNX) | Apache-2.0 | [github.com/RapidAI](https://github.com/RapidAI) |
| Noto Sans, Noto Sans Thai | OFL-1.1 | [github.com/notofonts](https://github.com/notofonts) |

What each wheel carries, as its pack's `manifest.json` pins it:

| Wheel | Models and fonts |
|---|---|
| `pdfintake-data-en` (built in) | Tesseract `osd` (tessdata_best) and `eng` (tessdata_fast); RapidOCR's detection, direction and Latin recognition models; Noto Sans |
| `pdfintake-lang-de` | Tesseract `deu` (tessdata_best) |
| `pdfintake-lang-es`, `pdfintake-lang-fr`, `pdfintake-lang-it`, `pdfintake-lang-pt` | none of their own: they read with the English data's RapidOCR Latin model |
| `pdfintake-lang-hu` | Tesseract `hun` (tessdata_best) |
| `pdfintake-lang-pl` | Tesseract `pol` (tessdata_best) |
| `pdfintake-lang-th` | Tesseract `tha` (tessdata_best); RapidOCR's Thai recognition model; Noto Sans Thai |
| `pdfintake-lang-vi` | Tesseract `vie` (tessdata_best) |

The test image in each wheel is rendered by this project from made-up text and is MIT licensed like the code.

Each pack's `corpus.json` names the Wikipedia article revisions (CC-BY-SA-4.0) and the Noto fonts (OFL-1.1) its benchmark pages were rendered from. The kit downloads them when it benchmarks a pack; no article text and no benchmark page is stored in this repository or in a wheel.

## Code ported from elsewhere

- `src/pdfintake/quality.py` is a line-by-line numpy port of code from [libjxl](https://github.com/libjxl/libjxl) 0.12.0, BSD-3-Clause, copyright the JPEG XL Project Authors. The file keeps libjxl's copyright notice and licence text in its header. It ports from these libjxl sources:
  - `tools/ssimulacra2.cc`: the SSIMULACRA 2 score, `MakePositiveXYB`, `Downsample`, the maps and the weights
  - `tools/gauss_blur.cc`: the recursive (IIR) Gaussian, `CreateRecursiveGaussian` and `FastGaussian`
  - `lib/jxl/enc_xyb.cc`: `LinearRGBToXYB`
  - `lib/jxl/cms/transfer_functions-inl.h`: the sRGB curve, `TF_SRGB::DisplayFromEncoded`
  - `lib/jxl/base/fast_math-inl.h`: the fast cube root, `CubeRootAndAdd`
  - `lib/jxl/cms/opsin_params.h`: the opsin absorbance matrix and bias
  - `lib/jxl/base/matrix_ops.h`: `Inv3x3Matrix`

  It also refuses images under 8 × 8 px, as `tools/ssimulacra2_main.cc` does; no code is taken from that file.

- `src/pdfintake/ocr/engines/dotsmlx/` runs dots.mocr for thorough mode on plain `mlx`. Its model code is ported from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.4, MIT, Copyright (c) 2025 Prince Canuma; the package's `__init__.py` carries mlx-vlm's licence text, and each module's header names the files it ports. `preprocess.py` ports the Qwen2-VL image processor's resize and patching from [transformers](https://github.com/huggingface/transformers) 5.18, Apache-2.0, Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team, and carries that licence's header. The model's architecture, its chat template and the layout prompt in `ocr/engines/dots.py` are [dots.ocr](https://github.com/rednote-hilab/dots.ocr)'s, MIT, Copyright (c) 2025 rednote-hilab.

## Thorough mode

Installed only with `pdfintake[thorough]`, and only on a Mac with Apple silicon:

| Package | Licence | What pdfintake uses it for |
|---|---|---|
| `mlx` | MIT | runs dots.mocr on the Mac's GPU |
| `mlx-metal` | MIT | mlx's Metal backend (pulled in by mlx) |
| `tokenizers` | Apache-2.0 | dots.mocr's tokenizer, read from the model folder's `tokenizer.json` |

`tokenizers` pulls in `huggingface-hub` (Apache-2.0) and with it `hf-xet` (Apache-2.0), `httpx` (BSD-3-Clause), `httpcore` (BSD-3-Clause), `h11` (MIT), `anyio` (MIT), `click` (BSD-3-Clause), `filelock` (MIT), `fsspec` (BSD-3-Clause) and `typing-extensions` (PSF-2.0). pdfintake never calls `huggingface-hub`: it downloads the model itself (`pdfintake thorough enable`).

The model is [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) by rednote-hilab, MIT, as the 8-bit MLX conversion [mlx-community/dots.mocr-8bit](https://huggingface.co/mlx-community/dots.mocr-8bit) (MIT), pinned at revision `6d9e750ba44cf99e3c608cfdf47e59df209dce13` with every file's SHA-256 in `src/pdfintake/thorough.py`. It isn't in this repository or in any wheel: its 4.4 GB are over GitHub's limit for a release file. The model card comes down with it, for its licence line.

## Test and kit extras

Installed only with `pdfintake[test]`, `pdfintake[kit]` or `pdfintake[kit-thai]`, never by a normal install:

| Package | Licence | Extra |
|---|---|---|
| `pytest` | MIT | test |
| `jsonschema` | MIT | test |
| `pyhanko` | MIT | test (signs test files with a key made at test time) |
| `uharfbuzz` | Apache-2.0 | test, kit (shaping Thai text for test pages) |
| `pythainlp` | Apache-2.0 | kit-thai |

## Never used

Ghostscript, jbig2enc, OCRmyPDF and PyMuPDF are not used, not installed and never called.
