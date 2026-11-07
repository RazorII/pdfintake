# CLAUDE.md

You're helping someone contribute to pdfintake: a command that turns scans, phone photos and PDFs into small, searchable PDFs, losslessly. This file sends you to the right guide. Read the guide in full before you change anything, and follow it step by step.

## Route the request

| The person wants to | Read and follow |
|---|---|
| Add a language, or improve one | [docs/add-a-language.md](docs/add-a-language.md) |
| Add an OCR engine | [docs/add-an-engine.md](docs/add-an-engine.md) |
| Add a pick rule (how the engines' reads are combined) | [docs/add-a-pick-rule.md](docs/add-a-pick-rule.md) |
| Support another platform | [docs/add-a-platform.md](docs/add-a-platform.md) |
| Anything else | [CONTRIBUTING.md](CONTRIBUTING.md) |

**Adding a language is the common case, and it needs no code.** It's a pack: a manifest plus pinned upstream models and fonts, chosen by running the kit (`pdfintake kit ...`), which benchmarks the shipped code. Every step in `docs/add-a-language.md` is one command with JSON output and documented exit codes. Don't hand-pick engines or hand-write the manifest's numbers: the kit chooses, and CI re-benches what it chose.

## Stop and ask the person

- A corpus text whose licence you can't classify as public domain, CC0, CC BY, CC BY-SA, or government text excluded from copyright.
- Any override of what `pdfintake kit choose` picked.
- Which characters must survive OCR in a language you aren't sure about.
- Any test or CI failure you can't explain.
- Any change outside the folder the guide names (for a pack, `packs/<id>/**`).

## Never

- Commit a real document, a scan, a screenshot of paperwork, or any personal data. Tests build everything in code from made-up text; example values are `example.com` addresses, 555-01xx phone numbers and names like Jane Doe.
- Commit a binary (model, font, image, PDF). They live in data wheels built from pinned upstream files.
- Make a run ask anything. Nothing in `src/` may prompt, read standard input for an answer, or wait on one: every choice comes from a flag or a default, and what a person might still want goes into the record's `offers`. The Claude skill doesn't ask during a batch either.
- Use a real password anywhere, or let one reach argv, the environment, a record, a log or an error message. Test passwords are invented strings in fixtures. pdfintake reads passwords only from a saved, private passwords file, and never touches a keychain or password manager.
- Start a process from `src/` (`subprocess`, `os.system`, `os.popen`, `multiprocessing`), or import PyMuPDF.
- Hand-edit `src/pdfintake/langpacks/index.json`, `tools/published-packs.json` or `src/pdfintake/ocr/calibration.json`. They're generated: `index.json` by `tools/build_index.py`, `published-packs.json` by `scripts/release.sh` (through `tools/build_index.py --record-release`), `calibration.json` by the maintainer's kit measurement. Don't change `tools/canary-env.txt` either: the maintainer owns it, and it pins the exact library versions every canary is drawn with.
- Push, tag, release or publish anything. The person and the maintainer do that.

## Run the tests

```
scripts/dev-install.sh      # once
pytest                      # all of it
pytest tests/contract       # the engine and pick-rule contracts
```

Tests that need a model, Apple Vision or the network skip themselves with a reason when it isn't there. A skip isn't a pass: say which tests skipped and why.
