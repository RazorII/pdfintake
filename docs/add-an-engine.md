# Add an OCR engine

An engine is registered core code: a module in `src/pdfintake/ocr/engines/` plus one row in the engine registry. Packs then name it in their reader sets. Nothing is discovered at run time, and no pack data ever decides what gets imported.

Before you start, open an issue saying which engine, its licence, how it installs on each platform, and where its models come from. Two things are maintainer decisions, because they change what every user downloads or trusts: a new dependency, and a new model source (the allowed sources are part of the frozen manifest schema).

## 1. Write the module

Create `src/pdfintake/ocr/engines/<name>.py` and expose one object, `ENGINE`, that satisfies the `Engine` protocol in `src/pdfintake/ocr/engines/__init__.py`:

| Member | What it is |
|---|---|
| `name` | the registry name, matching `^[A-Za-z][A-Za-z0-9_]{0,39}$` |
| `api` | `1` |
| `PARAMS` | a `ParamTable` (`src/pdfintake/params.py`) listing every parameter a pack may set. A `str` or `str_list` parameter needs a `pattern` or `choices`: no free text reaches core from a pack. A model is an `asset_ref` whose `choices` name the model kinds it accepts. |
| `MODEL_KINDS` | the model kinds it reads, from the schema's list (`traineddata`, `onnx_det`, `onnx_cls`, `onnx_rec`, `rec_keys`). A new kind is a schema change: raise it in the issue. |
| `platforms` | the platform keys it runs on, from `platforms.PLATFORMS` |
| `session_per_thread` | `True` if a session can't be shared between threads; the pipeline then opens one per worker thread and language |
| `version()` | the engine's version string, or `None` when it isn't installed |
| `availability(params, models, platform_key)` | an `Availability` with one of `available`, `not_on_platform`, `missing_model`, `unsupported_language`, `broken`, and a short reason. It never raises, and it runs before any read: an engine that silently returns junk for an unsupported language must catch that here. |
| `open(use_id, role, params, models)` | an `EngineSession` |
| `candidate_params(lang_tag, script, assets)` | the parameter sets the kit should benchmark for a language, or `[]` |
| `contract_case()` | a `ContractCase` the contract suite can run on this machine, or `None` when it can't (no model cached, wrong platform) |

An `EngineSession` has `read(page, *, timeout_s)` and `close()`. `read` returns an `EngineRead` (`src/pdfintake/types.py`):

- **Boxes** are four-point quads in pixels of the shared render (`PageImage`): origin top-left, y down, clockwise from the top-left corner. Convert the engine's own coordinates inside the module, never in points.
- **Confidences** are in `[0, 1]`.
- **Text is raw.** Don't normalise it; the pipeline runs the pack's normalisers on every engine alike.
- **Status**: `ok`, `empty` (nothing found, which is not an error), `timeout` when `timeout_s` runs out, `failed` with `error` saying why, and `unavailable` only for "not on this platform or language". `unavailable` is never a failure.
- `read` never raises for anything a page contains. An engine error becomes `EngineFailure`, which the pipeline records on the page.

Rules every engine module follows:

- **Import the engine library lazily**, inside the functions that need it, when it exists only on one OS or might be missing. `tests/test_hygiene.py` imports every module on every OS without the optional extras.
- **No processes, no network.** No `subprocess`, `os.system`, `os.popen` or `multiprocessing`, and no downloads: every model path comes from the `models` mapping, which the store has already checked against the pack's SHA-256. If the library would fetch a model on its own, give it explicit paths so it can't.
- **Deterministic**: the same page and parameters give the same read, and two sessions in two threads give identical reads.
- **Thread-safe** under the pipeline's rules: workers get only `PageImage` objects (PNG files on disk), never a pdfium handle.

## 2. Register it

Add one row to `_REGISTRY` in `src/pdfintake/ocr/engines/__init__.py`:

```python
"<name>": "pdfintake.ocr.engines.<name>:ENGINE",
```

The registry imports the module only when a pack asks for the engine, or when `warm_up()` imports every engine before a run installs its no-subprocess guard. A module that fails to import becomes a named setup error, not a crash.

## 3. Pass the contract

```
pytest tests/contract/test_engine_contract.py
```

Every registered engine with a contract case on this machine must:

- return quads inside the image, clockwise, and confidences in `[0, 1]`;
- give `empty` for a blank page, not an exception;
- give identical reads from two sessions in parallel threads, and from the same input twice;
- honour the timeout, with `timeout` as the status, never an exception;
- report a missing model as `missing_model`;
- never report `unavailable` as `failed`.

Then write `tests/test_<name>.py` for what only your engine does (its coordinate conversion, its language gate, its version string). Build the test pages in code with `tests/fixtures/build.py`. A test that needs a real model skips with a reason when the model isn't there.

## 4. Add the dependency

- Add it to `pyproject.toml`, with a platform marker if it exists only on some platforms, and a version range you've tested.
- Add a row to [THIRD-PARTY.md](../THIRD-PARTY.md) with its SPDX licence id, and rows for anything new it pulls in.
- Check it installs with `uv tool install` on every supported platform. CI runs the suite on macOS, Windows and Linux.

## 5. Use it in a pack

An engine does nothing until a pack lists it. That's a separate pack PR, made with the kit ([add-a-language.md](add-a-language.md)): `pdfintake kit candidates` asks every registered engine for its `candidate_params`, and `kit bench` and `kit choose` decide whether it earns a place in the reader set. A pack that uses the engine sets `requires_core` to the first release that registers it.

## Open the PR

Use the **code** pull-request template. Say which platforms you tested on, and paste the contract suite's output.
