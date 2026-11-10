# The pack manifest (schema 1)

A language pack is data only: a `manifest.json`, plus the model, font and canary files it pins, shipped in a data-only wheel. pdfintake never imports or runs anything from a pack. This page describes what a manifest may contain and what the loader checks. To make a pack, follow [add-a-language.md](add-a-language.md): the kit writes the manifest's numbers, so you rarely edit one by hand.

The shape is fixed by [`src/pdfintake/langpacks/manifest.schema.json`](../src/pdfintake/langpacks/manifest.schema.json) (JSON Schema draft 2020-12). The loader is `pdfintake.langpacks.manifest`: `parse(data, base=...)` returns a frozen `Manifest`, and `check(data, base=...)` returns every problem at once, each starting with its JSON path:

```
$.readers.default.engines[0].params.psm: 7 isn't one of 3, 4, 6
$.fills[2].name: fill 'words' serves only Latin; this pack's scripts include Thai
$.models[0].source: has a query string
```

## Compatibility policy

- Schema 1 fields are never removed or renamed, and their meaning never changes.
- A new optional field means schema 2. Each core version lists the schemas it reads (`manifest.SCHEMA_VERSIONS`; this core reads 1).
- Registry names are vocabulary, not schema. The loader checks them against the running core, and `requires_core` gates them: a pack that uses a rule, fill or engine added in core 0.3.0 says `"requires_core": ">=0.3.0"`.
- A rule, fill or repair whose behaviour changes gets a new `version`. The old version stays registered while any published pack uses it, and a CI test enforces that.

## Top-level fields

| Field | Required | What it holds |
|---|---|---|
| `schema` | yes | Always `1`. |
| `id` | yes | The pack id: an ISO 639 code, optionally with a variant (`th`, `hu`, `sr-latn`). |
| `version` | yes | The pack's own version, `MAJOR.MINOR.PATCH`. |
| `requires_core` | yes | `>=MAJOR.MINOR.PATCH`: the oldest pdfintake that understands every name the pack uses. |
| `base` | no | `true` only on the base pack, `en`, which ships with the install. |
| `title` | yes | The language's English name (`Portuguese`), at most 40 letters, spaces, brackets and hyphens. |
| `languages` | yes | 1 to 4 `{tag, iso639_3, role}`; at least one has role `primary`. Tags are unique. |
| `scripts` | yes | 1 to 4 script names this core knows, the primary script first. |
| `detect` | yes | How a page is recognised as this pack's: see below. |
| `readers` | yes | Reader sets by platform key; `default` is required. See below. |
| `normalize` | yes | Normaliser names, applied in order to every engine's read. |
| `repairs` | yes | Repair steps `{name, version, params}` run on the picked lines. |
| `fills` | yes | Fill steps `{name, version, params}`; they always run in core's fixed order. |
| `models` | yes | Every model file: `{key, file, engine, kind, bytes, sha256, source, licence, notice?}`. |
| `fonts` | yes | Layer fonts: `{key, file, covers, bytes, sha256, source, licence}` (`OFL-1.1`). |
| `canary` | no | `{file, bytes, sha256, dpi, must_find}`: a rendered line every engine use of every reader set must find, checkers included (`kit wheel` reads it with each). Each `must_find` string holds a character other than whitespace. Every published pack needs one. |
| `must_survive` | yes | Characters the bench and the pipeline count: `{name, chars, unit, floor}`. |
| `bench` | yes | The numbers the kit measured (`kit`, `corpus`, `platforms`, optional `engine_versions`, `private`). |

### `detect`

- `osd_scripts`: the Tesseract OSD scripts that point to this pack.
- `langid`: `{codes, close_group?}`: language identifier codes for this pack, and the close group (languages that always need the user's answer).
- `function_words`: `{words, rivals?, min_hits?, ratio?}` for a function-word page score (defaults: `min_hits` 3, `ratio` 1/3, the exact third: the pack's words must reach a third of the rival words; a manifest that writes it out gives `0.3333333333333333`).
- `probe`: `{use, min_chars, min_conf, min_line_chars?}`: a cheap engine read that trips on this pack's script. The `use` has role `probe`. `min_line_chars` defaults to 2. A line counts only when the pack's script is at least `detect.PROBE_LINE_MIN_SHARE` (0.5) of the line's letters, so the stray characters a probe reads into Latin text don't add up.

### Reader sets

`readers` maps `default`, or a platform key such as `darwin-arm64`, to a reader set. A platform without its own entry uses `default`.

```json
{"engines": [{"id": "tesseract", "engine": "tesseract", "role": "reader",
              "params": {"lang": "hun", "psm": 3}, "models": ["hun"]},
             {"id": "rapidocr", "engine": "rapidocr", "role": "checker",
              "params": {"rec": "en:rapid-rec-latin"},
              "models": ["en:rapid-det", "en:rapid-cls", "en:rapid-rec-latin"]}],
 "pick": {"rule": "primary_with_floor", "version": 1, "params": {"primary": "tesseract", "checker": "rapidocr"}},
 "fallback": "none"}
```

- Each engine use has a unique `id` and a role: `reader` competes for the page; `checker` doesn't compete but feeds fills and repairs. A set has at least one reader and at most one checker.
- `models` lists asset references: a key of this pack's `models`, or `en:<key>` for a model of the base pack.
- `pick` names a registered pick rule and version. `fallback` is `checker` (keep the checker's read, named, when every reader fails) or `none` (no layer for that page).

## What the loader checks beyond the schema

- **Names exist.** Engines, pick rules and their versions, fills, repairs and their versions, normalisers, script names and platform keys must be registered in this core.
- **Parameters.** Every `params` object passes that member's parameter table (below). `use_ref` and `use_list` parameters name uses in the same reader set; a repair's `source` must name a use in every reader set. An `asset_ref` names a model of the kind the parameter needs, and that model must be among the use's `models`.
- **Models.** Every model a use references belongs to that use's engine, and every model's `kind` is one its engine takes. Model keys and files are unique within the pack. Tesseract model files share one directory, so a pack's `.traineddata` file names never repeat one of the base pack's, and are unique across the whole index.
- **File names.** File names are compared ignoring case (`Tha.traineddata` repeats `tha.traineddata`), because the store may sit on a case-insensitive file system. A model file is data, so its name never ends in a code suffix: `.py`, `.pyc`, `.pyo`, `.pyd`, `.pyw`, `.pyz`, `.pth`, `.so` (or a versioned `.so.1`), `.dylib`, `.dll`, `.exe`, `.com`, `.bat`, `.cmd`, `.ps1`, `.sh`, `.msi` or `.jar` (`manifest.CODE_SUFFIXES`, the same list the data wheel builder refuses). The schema's `noCodeSuffix` pattern names only six of these (`.py`, `.pyd`, `.pth`, `.so`, `.dylib`, `.dll`), so a file such as `tha.sh` or `libtha.so.1` passes a plain JSON Schema validator and is still refused by the loader. The loader is the gate every pack goes through, so it may be stricter than the schema but never looser.
- **Sources.** Every `source` matches the allowlist (below), and has no `..`, no query string and no fragment.
- **Fills.** A fill step must serve every script of the pack: `words` works on spaced Latin only, so a Thai pack can't list it.
- **Pick parameters.** A pick's `corroborators` are never uses with role `reader`.
- **Versions and the base pack.** `requires_core` is not newer than this core. Only `en` may set `base: true`, and `en` must. The base pack refers to its own models without `en:`; any other pack's `en:` references are checked against the base manifest.
- **Platform coverage.** The `default` reader set uses no engine that is missing on a supported platform. A platform's own reader set uses only engines that run there. The `detect.probe` use is held to the same rule as the `default` set: a probe runs during detection on every platform and has no per-platform alternative, so a Mac-only engine such as `vision` can't be the probe.
- **Canary.** Each `canary.must_find` string holds a character other than whitespace. The canary removes whitespace before it compares, so a blank string would find nothing.
- **Strict JSON.** UTF-8 without a byte-order mark, at most 1 MiB, no repeated keys, no `NaN` or `Infinity`. A pattern anchored with `^...$` must match the whole string, so a trailing newline never passes.

## The source allowlist

Models come only from these upstreams, pinned to a commit or a release:

- `https://github.com/tesseract-ocr/tessdata`, `tessdata_best` or `tessdata_fast`, at `raw/<40-hex commit>/<file>`;
- `https://github.com/RapidAI/<repo>/raw/<40-hex commit>/...` or `.../releases/download/<tag>/...`;
- `https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v<x.y.z>/...`.

Fonts come only from `https://github.com/notofonts/<repo>/raw/<40-hex commit>/...` or `.../releases/download/<tag>/...`.

The patterns are `manifest.SOURCE_PATTERNS`, identical to the schema's (a test keeps them equal).

## The pack index

`src/pdfintake/langpacks/index.json` is generated by `tools/build_index.py`, never edited by hand, and ships inside the core wheel. `pdfintake.langpacks.index.load()` reads it strictly:

```json
{"index_version": 1, "core_min": "0.1.0",
 "packs": [{"id": "pt", "version": "1.0.0", "base": false, "title": "Portuguese", "scripts": ["Latin"],
            "langid_codes": ["pt"], "detect": {"osd_scripts": ["Latin"], "langid": {"codes": ["pt"]}, "probe": false},
            "manifest_sha256": "<sha256 of packs/pt/manifest.json>",
            "dist": {"kind": "fetch", "distribution": "pdfintake-lang-pt",
                     "url": "https://github.com/RazorII/pdfintake/releases/download/<tag>/pdfintake_lang_pt-1.0.0-py3-none-any.whl",
                     "sha256": "<sha256 of the wheel>", "bytes": 27572},
            "installed_bytes": 26149}]}
```

- `detect` is the manifest's `detect` block with `probe` reduced to `true` or `false` (`index.detect_copy`), so detection can use a pack's OSD scripts and function words before the pack is fetched. `langid_codes` equals `detect.langid.codes`.
- Exactly one entry is the base pack: `en`, `kind: "installed"`, distribution `pdfintake-data-en`, `url: null`.
- Every other entry is `kind: "fetch"`, distribution `pdfintake-lang-<id>`. Its `url` is the GitHub release download: HTTPS on `github.com`, a release asset under `/RazorII/pdfintake/releases/download/`, with no user name, port, query, fragment, `..` or `%`, only RFC 3986 unreserved characters plus `:` and `/`, ending in the wheel's own file name (`index.url_problems`).
- pdfintake downloads only from the hosts in `index.ALLOWED_HOSTS`: `github.com`, and `release-assets.githubusercontent.com`, where GitHub serves release files. github.com answers a release download with a `302` to a signed address there, which returns the file (checked on public release assets on 2026-11-05). An entry never names that signed address: it carries a query string and expires.
- Those rules are for the entry's `url` only. A redirect met during a fetch is followed when it stays on HTTPS on a host in `ALLOWED_HOSTS` with no user name or port (`index.redirect_problems`); its query string is allowed, because that is how GitHub signs the address.
- `core_min` is not newer than this core, and pack ids are unique.

## Parameters of the registered members

The authoritative tables are each member's `PARAMS` in the code (`ParamTable.describe()` prints them); this is the list for this core.

### Engines

**`tesseract`**. Model kinds: traineddata. Platforms: darwin-arm64, linux-aarch64, linux-x86_64, win-amd64.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `lang` | str | yes |  | pattern `[A-Za-z][A-Za-z0-9]{1,15}(?:_[A-Za-z0-9]{1,8}){0,2}(?:\+[A-Za-z][A-Za-z0-9]{1,15}(?:_[A-Za-z0-9]{1,8}){0,2}){0,3}`; ≤ 80 chars |
| `psm` | int | no | `3` | one of `3`, `4`, `6` |
| `timeout_s` | float | no | `300.0` | ≥ 10.0; ≤ 600.0 |

**`vision`**. Model kinds: none. Platforms: darwin-arm64.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `languages` | str_list | yes |  | ≥ 1; ≤ 8; pattern `[a-z]{2,3}(?:-[A-Z][a-z]{3})?(?:-(?:[A-Z]{2}\|[0-9]{3}))?`; ≤ 16 chars |
| `revision` | int | no | `3` | one of `1`, `2`, `3` |
| `level` | str | no | `accurate` | one of `accurate`, `fast` |
| `language_correction` | bool | no | `True` |  |
| `timeout_s` | float | no | `120.0` | ≥ 10.0; ≤ 600.0 |

**`rapidocr`**. Model kinds: onnx_cls, onnx_det, onnx_rec. Platforms: darwin-arm64, linux-aarch64, linux-x86_64, win-amd64.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `rec` | asset_ref | yes |  | one of `onnx_rec` |
| `use_cls` | bool | no | `False` |  |
| `min_score` | float | no | `0.5` | ≥ 0.0; ≤ 1.0 |

**`dots`**. Model kinds: none. Platforms: darwin-arm64. Thorough mode's reader (dots.mocr), which a pack can't name: thorough mode adds it to every OCR page itself, with its own pinned model, when the user turns the mode on. No parameters.

### Pick rules

**`corroborated_trigrams` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `script` | script | yes |  |  |
| `tie_order` | use_list | no |  |  |
| `corroborators` | use_list | no | `[]` |  |

**`primary_with_floor` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `primary` | use_ref | yes |  |  |
| `checker` | use_ref | yes |  |  |
| `floor` | float | no | `0.5` | ≥ 0.0; ≤ 1.0 |
| `min_words` | int | no | `20` | ≥ 0; ≤ 100000 |
| `word_conf` | float | no | `0.8` | ≥ 0.0; ≤ 1.0 |
| `min_len` | int | no | `4` | ≥ 1; ≤ 40 |

**`single` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `reader` | use_ref | yes |  |  |

### Fills

**`lines` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `min_score` | float | no | `0.9` | ≥ 0.0; ≤ 1.0 |
| `min_alnum` | int | no | `8` | ≥ 1; ≤ 200 |
| `ratio` | float | no | `0.45` | ≥ 0.0; ≤ 1.0 |

**`words` v`1`**. Scripts: Latin only.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `min_score` | float | no | `0.8` | ≥ 0.0; ≤ 1.0 |
| `min_letters` | int | no | `4` | ≥ 2; ≤ 24 |

**`numbers` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `min_digits` | int | no | `3` | ≥ 1; ≤ 32 |
| `min_score` | float | no | `0.5` | ≥ 0.0; ≤ 1.0 |

**`ids` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `min_len` | int | no | `6` | ≥ 2; ≤ 64 |
| `max_len` | int | no | `16` | ≥ 2; ≤ 64 |
| `min_score` | float | no | `0.9` | ≥ 0.0; ≤ 1.0 |
| `near_edits` | int | no | `3` | ≥ 0; ≤ 8 |

**`links` v`1`**.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `min_score` | float | no | `0.5` | ≥ 0.0; ≤ 1.0 |

### Repairs

**`diacritic_word` v`1`**. Each checker word holding a `table` letter replaces its partner on the picked line: the word on the same text row that differs from it only in those letters.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `source` | use_ref | yes |  |  |
| `table` | char_table | yes |  |  |

**`diacritic_word` v`2`**. v1, limited to where the other language really is: it runs only on a page whose detected languages include one of `languages`, takes words only from a checker line holding one of the `context` words, and never takes a checker word holding a `never` letter. The hu pack sets `languages` to `de`, `context` to the German function words its detection lists as rivals, and `never` to the letters Hungarian writes and German doesn't. Counts in the record: `fixed`, `unpartnered`, `refused` (held out by `never`) and `no_context`.

| Parameter | Kind | Required | Default | Allowed |
|---|---|---|---|---|
| `source` | use_ref | yes |  |  |
| `table` | char_table | yes |  |  |
| `languages` | str_list | yes |  | 1 to 4 language codes, `[a-z]{2,3}` |
| `never` | str | no | `""` | up to 40 letters |
| `context` | str_list | no |  | 1 to 64 words of letters, up to 20 each, matched ignoring case |

### Normalisers

`nfc` (Unicode NFC) and `thai_sara_am` (Thai sara am put back together). Normalisers take no parameters.
