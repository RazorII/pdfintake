# Add a pick rule

A pick rule decides which engine's read becomes a page's text layer. A pack's reader set names one rule by name and version, with its parameters. Rules are registered core code in `src/pdfintake/ocr/rules/`; packs only choose among them.

The rules today:

| Rule | Version | What it does |
|---|---|---|
| `corroborated_trigrams` | 1 | the reader whose letter trigrams the other reads confirm most often wins; corroborators confirm but never win |
| `primary_with_floor` | 1 | the primary reader wins while it agrees with the checker above a floor; below it, the `fallback` decides |
| `single` | 1 | one reader, kept as it is |

## Versions never change meaning

A rule's behaviour is fixed by its `(name, version)`. If you change what an existing rule chooses, even slightly, that's a **new version**: register `("<name>", 2)` beside version 1. Published packs keep using version 1 until a pack PR moves them, and version 1 stays registered while any published pack uses it. A fix that doesn't change any choice (a crash, a clearer reason string) can stay in the same version.

## 1. Write the rule

Add a module in `src/pdfintake/ocr/rules/`, or a new object in an existing one, that satisfies the `PickRule` protocol in `src/pdfintake/ocr/rules/__init__.py`:

| Member | What it is |
|---|---|
| `name` | matching `^[A-Za-z][A-Za-z0-9_]{0,39}$` |
| `version` | a whole number, 1 to 99 |
| `PARAMS` | a `ParamTable` (`src/pdfintake/params.py`). Point at engine uses with `use_ref` and `use_list`; the loader checks they exist in the same reader set. |
| `scripts` | the scripts it may be used for, or `None` for any |
| `pick(reads, params, *, fallback, checker_id)` | a `Pick` (`src/pdfintake/types.py`) |

`reads` maps each use id to its `EngineRead`. The returned `Pick` has the rule's name and version, the chosen use id (or `None`), the chosen read's lines unchanged, a score per use, whether the fallback was used, and a short `reason` in plain English.

What every rule must do:

- **Be deterministic**, whatever order the reads arrive in. Break ties by a fixed order (`tie_order`, when the rule takes one).
- **No reader working:** with `fallback="checker"`, keep the checker's read, name it, and set `fallback` to true; with `fallback="none"`, choose nothing, so the page gets no layer.
- **Exactly one reader working:** keep it, uncorroborated.
- **Only a reader wins.** A corroborator or a checker never wins while a reader works, and nothing but a reader wins under `fallback="none"`.
- **Always give a reason.**
- **Refuse unknown parameter keys** (the `ParamTable` does this for you).
- **Warnings** go in `Pick.warnings`, only with codes from `PICK_WARNING_CODES`. The pipeline works out `fallback_used` and `no_layer_readers_failed` on its own, so don't add those. A new warning code is a core change: it needs core wording in `record.py` and an explanation in the skill.

A rule never reads images, never calls an engine and never touches the network. It works only on the reads it's given.

## 2. Register it

Add one row to `_REGISTRY` in `src/pdfintake/ocr/rules/__init__.py`:

```python
("<name>", 1): "pdfintake.ocr.rules.<module>:<OBJECT>",
```

## 3. Pass the contract

Add a scenario for the rule to `SCENARIOS` in `tests/contract/test_rule_contract.py`: which use ids are readers, which is the checker, and how to write the rule's parameters for them. A registered rule without a scenario fails the suite, so no rule can skip it. Then:

```
pytest tests/contract/test_rule_contract.py
```

Write `tests/test_rules_<module>.py` for the cases only your rule handles. Build the reads with `tests/fixtures/build.py` from made-up text, never from a real document's OCR.

## 4. Use it in a pack

`pdfintake kit bench` tries every compatible rule with every candidate reader set, so a new rule is benchmarked against the existing ones on the next pack bench ([add-a-language.md](add-a-language.md)). A pack that uses it sets `requires_core` to the first release that registers it.

## Open the PR

Use the **code** pull-request template, and say which pages your rule gets right that the existing ones don't, with bench numbers if you have them.
