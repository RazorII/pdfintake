# Contributing

Thanks for helping. pdfintake has three seams, and each kind of change goes through one of them:

| You want to | It's a | Start here |
|---|---|---|
| Add or improve a language | **pack PR**: data only, touching only `packs/<id>/**` | [docs/add-a-language.md](docs/add-a-language.md) |
| Add an OCR engine | code PR | [docs/add-an-engine.md](docs/add-an-engine.md) |
| Add a way of choosing between engines' reads | code PR | [docs/add-a-pick-rule.md](docs/add-a-pick-rule.md) |
| Support another operating system or processor | code PR | [docs/add-a-platform.md](docs/add-a-platform.md) |
| Fix a bug, or anything else | code PR | an issue first, for anything bigger than a small fix |

A language never needs code. If adding one seems to need a code change (a new engine parameter, a new fill or repair), that part is a separate code PR, and the pack PR waits for a release that has it (`requires_core` in the manifest).

If you work with Claude, the repo's [CLAUDE.md](CLAUDE.md) points it to the right guide, and [docs/add-a-language.md](docs/add-a-language.md) is written for it to follow step by step.

## Set up

```
git clone https://github.com/<you>/pdfintake && cd pdfintake
scripts/dev-install.sh
pdfintake doctor --json
pytest
```

`scripts/dev-install.sh` builds the English data wheel from its pinned upstream files and installs pdfintake in editable mode with the `test` and `kit` extras. It needs [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.14.

## Ground rules

These keep the repo publishable and every user's files safe. CI enforces most of them.

- **No real documents, ever.** Tests build their images, PDFs, fonts and OCR reads in code, from made-up text. Example values come from the reserved ranges: `example.com` addresses, 555-0100 to 555-0199 phone numbers, the payment networks' published test cards, names such as Jane Doe. No scans, no screenshots of real paperwork, not even your own.
- **No binaries in the repo.** Models, fonts and test images live in the data wheels, built from pinned upstream files. Pack folders hold JSON and licence texts only.
- **No other programs.** Nothing in `src/` may start a process: no `subprocess`, `os.system`, `os.popen` or `multiprocessing`. `tests/test_hygiene.py` checks this statically, and a runtime guard catches the dependencies. Ghostscript, `ocrmypdf`, a command-line `tesseract` and `jpegtran` are therefore out, even as fallbacks.
- **No PyMuPDF** (`fitz`, `pymupdf`), for licence reasons.
- **Lossless means lossless.** No change may re-encode a JPEG, reduce bit depth, convert colour or drop transparency, except the one encode of a format a PDF can't carry, and the smaller copy the user said yes to.
- **One writer.** Only `flow.commit` and `flow.apply` write into a user's folders, after the refusal and input checks. `tests/test_write_paths.py` enforces it.
- **Core writes every word.** Questions, option labels and warnings are worded in core. A pack string that can reach a record is limited to a short pattern (ids, language codes).
- **No document text in records.** Records hold counts, hashes, scores and paths.
- **Imports that only exist on one OS** (Apple Vision) **or in an extra** (`uharfbuzz`, `pythainlp`) go inside the function that needs them, so every module imports everywhere.
- Open text files with `encoding="utf-8"`.
- **Generated files aren't edited by hand.** `src/pdfintake/langpacks/index.json` comes from `tools/build_index.py`, `tools/published-packs.json` from `scripts/release.sh` (through `tools/build_index.py --record-release`), and `src/pdfintake/ocr/calibration.json` from the maintainer's kit measurement. `tools/canary-env.txt` is maintainer-owned: it pins the exact library versions every pack's canary is drawn with.

## Pull requests

- Use the matching template: **language pack** or **code**. A pack PR that touches anything outside `packs/<id>/**` is a code PR.
- **Pack PRs** are re-benchmarked by CI on Linux, Windows and macOS against the numbers the pack claims, with its downloads checked against the allowed sources. A result outside tolerance fails; an override needs a reason and gets a maintainer review.
- **Code PRs** run the full suite on all three operating systems. A new engine, pick rule, fill or repair must pass its contract suite.
- Every PR is reviewed by the maintainer before it merges.
- Corpus texts for a pack must allow it: public domain, CC0, CC BY, CC BY-SA, or government text the law excludes from copyright. They're fetched by revision id at bench time and never committed.

By contributing you agree that your contribution is licensed under the MIT licence, like the rest of the code. Pack data keeps its upstream licence (Apache-2.0 models, OFL-1.1 fonts).

## Questions

Ask in [Discussions](https://github.com/RazorII/pdfintake/discussions/categories/q-a). Security problems go through [SECURITY.md](SECURITY.md), not an issue.
