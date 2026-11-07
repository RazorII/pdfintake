<!--
A code PR: anything outside packs/<id>/**. Open it with this template (add ?template=code.md to the
compare URL). An engine, pick rule or platform has its own guide in docs/; follow it.
-->

## What and why

## How it was tested

<!-- Which tests you added or ran, on which operating systems. Say which tests skipped and why: a skip isn't a pass. -->

## Checklist

- [ ] Tests build every fixture in code from made-up values (example.com, 555-01xx numbers, names like Jane Doe). No real document, scan or personal data, and no binary file.
- [ ] Nothing in `src/` starts a process (`subprocess`, `os.system`, `os.popen`, `multiprocessing`) or imports PyMuPDF; `tests/test_hygiene.py` passes.
- [ ] A pick rule, fill or repair whose behaviour changed has a new `version`; the old one stays registered while a published pack uses it.
- [ ] A new engine, rule, fill or repair has a `ParamTable` and passes its contract suite in `tests/contract/`.
- [ ] Every word a user or the skill sees comes from core templates; no new pack string reaches a record unless a pattern limits it.
- [ ] A new warning or note code is in `record.py` and explained in the skill; a new exit code is in `errors.py`, the README and the skill.
- [ ] A new platform is one row in `platforms.PLATFORMS`, a CI runner in `.github/workflows/test.yml` and a README row.
- [ ] `src/pdfintake/langpacks/index.json` and `src/pdfintake/ocr/calibration.json` weren't edited by hand.
