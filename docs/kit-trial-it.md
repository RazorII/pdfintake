# Kit trial: Italian (`it`)

A contributor's Claude, who wrote none of the kit, added the Italian pack by following `CLAUDE.md` and `docs/add-a-language.md` step by step, without reading the kit's source, its tests or any design notes. Every command, its exit code, and every place the doc was wrong, unclear or missing something is below, with what was done instead.

- Machine: macOS 27.2 on Apple silicon (`darwin-arm64`), Apple Vision available (revision 3).
- Engines measured: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Every kit command ran from a shared, already set-up clone's `.venv` (step 1 was done beforehand), as `pdfintake kit <cmd> it --packs <repo>/packs --work <work> --json`, where `<work>` is a kit work folder outside the repo.
- Four other packs were benched on the same Mac at the same time (load average about 70 during the bench). That changes the seconds per page, not the recall numbers; see item 6.
- Stop points met: none. `kit choose` exited 0 and its pick was kept as is. The must-survive characters were given by the person: `accents=àèéìíòóùúÀÈÉÌÍÒÓÙÚ:word:0.9`. Every corpus text is Wikipedia, CC-BY-SA-4.0, on a neutral topic.

## Result

- Chosen on `darwin-arm64`: **RapidOCR alone** (Latin recogniser from the base pack), pick `single`. Worst-condition word recall 0.9361; accents recall 1.000 clean, 1.000 scan and 0.955 small.
- Runner-up as reported: Vision (`vision2`) + RapidOCR with `corroborated_trigrams` (set 52), margin 0.0018.
- `kit choose` dropped `ita.traineddata`: the chosen set reads with the base pack's models only, so the pack pins no model of its own and the wheel is 16,081 bytes.
- `kit verify it --download`: exit 0, all four checks `ok`, no warnings. `kit verify it --download --rebench`: exit 0, all four checks `ok`, every recall number identical to the claim (`drift: []`, `floors: []`).
- Files: `packs/it/manifest.json`, `corpus.json`, `bench.json` (what `kit report` says to commit). Also in the folder and listed under `leave_out`: `manifest.draft.json` and `NOTICE-tessdata` (the notice of the dropped Tesseract model).

## Steps

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | `pdfintake doctor --json` | **3** | 1 s |
| 2 | `pdfintake kit init it --script Latin --must-survive 'accents=àèéìíòóùúÀÈÉÌÍÒÓÙÚ:word:0.9'` | 0 | 1 s |
| 3 | `pdfintake kit corpus it --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072,Q4022 --font <NotoSans-Regular> --font <NotoSerif-Regular>` | 0 | 13 s |
| 4 | `pdfintake kit render it` | 0 | 7 s |
| 5 | `pdfintake kit candidates it` | 0 | 3 s |
| 5 | `pdfintake kit bench it` (background) | 0 | **about 43 min** (contended) |
| 6 | `pdfintake kit choose it` | 0 | 1 s |
| 7 | `kit private` | skipped | optional; the trial has no contributor PDFs |
| 8 | `pdfintake kit wheel it --must-find città --must-find perché --must-find così --must-find È --must-find 4521` | **1** | 1 s |
| 8 | `pdfintake kit wheel it --must-find città --must-find perché --must-find così --must-find 4521` | 0 | 1 s |
| 8 | `pdfintake kit verify it --download` | 0 | under 1 s |
| 8 | `pdfintake kit verify it --download --rebench` (background) | 0 | 46 s |
| 9 | `pdfintake kit report it` | 0 | 1 s |

The corpus is eight Italian Wikipedia articles, each title checked in the `pinned` list: Acqua (Q283), Sole (Q525), Luna (Q405), Fotosintesi clorofilliana (Q11982), Scacchi (Q718), Pane (Q7802), Vulcano (Q8072) and Fiume (Q4022). The fonts are the doc's two example URLs, `NotoSans-Regular.ttf` and `NotoSerif-Regular.ttf` from `notofonts.github.io` at commit `28b15b4…`.

`kit init` pinned `ita.traineddata` from `tessdata_best` (8,863,635 bytes) and reported `vision_languages` `[["it-IT", "en-US"]]`, `next: []`. `kit render` drew 24 images: 4 prose, 2 form, 1 microprint and 1 mixed-script page, each in the three conditions; the mixed-script page's second script is Greek, drawn by the two Latin fonts, as the doc says. `kit candidates` listed 54 sets (`total` 54, `refused_by_loader` 0, `cut` 0, `kept` 54); `kit bench` read all 24 pages with each of them, printing one progress line per set on standard error.

## Where the doc was wrong, unclear or missing something

Each item gives the step, what happened, and what was done instead. "Doc" means `docs/add-a-language.md`.

### Step 1: set up

1. **`doctor` exits 3 with a problem the doc routes to a reinstall the trial couldn't do.** The problem was `base_pack` `hash_mismatch`: the installed English data wheel's `manifest.json` has a different SHA-256 from the one the clone's generated pack index pins (`"blocking": true`). The doc's carve-out covers only `index_missing`; for anything else it says to run `scripts/dev-install.sh` again and, if it stays, that's stop point 4. In this trial step 1 had been done for us in a shared environment that must not be changed, so reinstalling wasn't an option. *Did instead:* went on to step 2, as the doc allows for `index_missing`, since the kit takes the base pack from `packs/en`. Every later kit command exited 0, so the kit doesn't depend on this check either. *Fix:* say which `doctor` problems block the kit and which don't. A stale base-pack hash (the index regenerated after the English wheel was built) looks like the same "doesn't matter to the kit" case as `index_missing`.

### Step 5: candidates and bench

2. **The same use id names two different engine settings in one bench.** In set 0 (the draft's own set), `tesseract` is Tesseract with the `ita` model. In sets 1 to 7 and 38 to 43, `tesseract` is Tesseract with the base pack's `eng` model, and the Italian model is `tesseract2`. The doc does say set 0 keeps the draft's own ids and that a numbered id is the same engine with other parameters. But a table of results reads as if "tesseract alone" scores 0.08 on accents, which looks like a broken Italian model. Only the `params` say it's English. *Did instead:* read each set's `params` before interpreting a row. *Fix:* give the candidates ids that don't collide with set 0's, or have the progress line and the bench JSON name the model (`tesseract[ita]`, `tesseract[eng]`).
3. **The bench took twice as long as the doc says, and its time depends on load.** The doc gives about 20 seconds per set and 10 to 20 minutes for 30 to 60 sets. Here 54 sets took about 43 minutes, 39 to 85 s per set, while four other packs were benched on the same machine. The kit's own first line estimated 59 minutes. Running it in the background, as the doc says, was enough. *Fix:* none needed for the procedure. One line would help: run one bench at a time, because the seconds per page it records go into the manifest (item 6).

### Step 6: choose

4. **"Runner-up" isn't defined, and the one reported isn't the next-best score.** `kit choose` reported `chosen: 29` (RapidOCR alone, worst-condition word recall 0.9361) with `runner_up: 52` (Vision `vision2` + RapidOCR, 0.9343) and `margin: 0.0018`. Set 43 (Tesseract `eng` + RapidOCR with Vision as corroborator) scored higher than the chosen set, 0.9377. Sets 30 to 37 (RapidOCR as primary with a second engine) scored exactly the same as set 29. That fits the documented policy: 43 is within the 0.005 tie window and has more engines, so the one-engine set wins. But the doc never says how the runner-up is picked: it isn't the second-best score, and it isn't the closest tie either. So a reviewer reading "margin 0.002" in the report can't tell that a set scored higher. *Fix:* define `runner_up` and `margin` in step 6 (for instance, "the best set with a different engine composition, and how far below the chosen set it scored"). Or report the best tied set, with its score, next to the runner-up.
5. **No override question arose.** Exit 0; the pick was kept. Noted only because stop point 2 is about this step.

### Step 8: wheel and verify

6. **The seconds per page in the manifest came from a loaded machine.** The bench recorded 9.915 s per page for the chosen set, and the manifest and report carry that number. The re-bench, with less load, measured 6.19 s; the Spanish trial measured 2.25 s for the same single-RapidOCR set. `verify --rebench` doesn't compare speed, so this passes. The pick wasn't affected either: the speed tie-break comes after engine count, and set 29 is the only one-engine set above the floors. *Fix:* see item 3. Re-benching on a quiet machine before the PR would put a truer number in the manifest. For this pack, the maintainer's re-bench will.
7. **`--must-find` takes at most four words, and the doc doesn't say so.** The doc says to repeat `--must-find` for each word. Passing five words (`città`, `perché`, `così`, `È`, `4521`) exited 1 with `kit_invalid_manifest`, "`$.canary.must_find: has 5 items; at most 4`". That's three words with the language's special characters plus `4521`. *Did instead:* dropped `È` and ran it again (exit 0; RapidOCR found all four). *Fix:* say "up to four words, `4521` included" in step 8, or have the CLI refuse with exit 2 (bad arguments) and that message before it renders anything.

### Step 9: the report

8. **The report body's "Models and fonts" table is empty.** Because the chosen set uses only the base pack's models and fonts, the table renders its header row and nothing else. *Fix:* write "none of its own: reads with the base pack's `en:rapid-*` models" instead of an empty table.
9. **The report names the must-survive group but not its characters.** The body says "Must survive: `accents` (word, floor 0.9)". The checklist asks a reviewer to confirm "the must-survive characters were confirmed by someone who reads the language", but the characters aren't in the body. *Fix:* print the characters beside the group name.
10. **The report doesn't list the unbenched platforms.** `kit choose` listed `win-amd64`, `linux-x86_64` and `linux-aarch64` under `unbenched`; the report body shows only `darwin-arm64` and doesn't say the others read with `default` and have no numbers yet. *Fix:* one line in the body naming the unbenched platforms.

### Followed as written, no issue

- Step 2's `vision_languages` shape, step 3's neutral-topic guidance, example font URLs and the note that the mixed-script page needs no extra font, step 5's `refused_by_loader` and `floors_ok`, step 6's `next` (it named `kit wheel`), `dropped_models` and `unbenched`, step 8's verify checks and `warnings`, and step 9's `commit` and `leave_out` all matched what the commands did.

## The bench on `darwin-arm64`

Word recall and must-survive (`accents`) recall per condition. The floor is 0.9 in every condition; a set below it in any condition is dropped. In set 0, `tesseract` reads with `ita`; in every other set `tesseract` reads with the base pack's `eng` and `tesseract2` with `ita` (item 2). `vision` has language correction on, `vision2` off. The seconds per page were measured while four other benches ran (item 6).

| # | Reader set | Pick | Words clean | Words scan | Words small | Accents clean | Accents scan | Accents small | Floors | s/page |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tesseract + rapidocr | primary_with_floor | 0.9822 | 0.9804 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 8.40 |
| 1 | tesseract | single | 0.9470 | 0.9427 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 2.61 |
| 2 | tesseract + vision | single | 0.9470 | 0.9427 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 2.23 |
| 3 | tesseract + vision | primary_with_floor | 0.9470 | 0.9427 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 1.87 |
| 4 | tesseract + vision2 | single | 0.9470 | 0.9427 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 1.43 |
| 5 | tesseract + vision2 | primary_with_floor | 0.9470 | 0.9427 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 1.26 |
| 6 | tesseract + rapidocr | single | 0.9472 | 0.9429 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 7.25 |
| 7 | tesseract + rapidocr | primary_with_floor | 0.9472 | 0.9429 | 0.8275 | 0.0809 | 0.0809 | 0.0673 | below | 10.92 |
| 8 | tesseract2 | single | 0.9820 | 0.9802 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 5.68 |
| 9 | tesseract2 + vision | single | 0.9820 | 0.9802 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 3.29 |
| 10 | tesseract2 + vision | primary_with_floor | 0.9820 | 0.9802 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 3.77 |
| 11 | tesseract2 + vision2 | single | 0.9820 | 0.9802 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 4.21 |
| 12 | tesseract2 + vision2 | primary_with_floor | 0.9820 | 0.9802 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 4.01 |
| 13 | tesseract2 + rapidocr | single | 0.9822 | 0.9804 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 9.17 |
| 14 | tesseract2 + rapidocr | primary_with_floor | 0.9822 | 0.9804 | 0.8970 | 1.0000 | 0.9961 | 0.8973 | below | 13.53 |
| 15 | vision | single | 0.9642 | 0.9521 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 0.88 |
| 16 | vision + tesseract | single | 0.9642 | 0.9521 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 1.66 |
| 17 | vision + tesseract | primary_with_floor | 0.9642 | 0.9521 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 1.48 |
| 18 | vision + tesseract2 | single | 0.9642 | 0.9521 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 3.86 |
| 19 | vision + tesseract2 | primary_with_floor | 0.9642 | 0.9521 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 7.41 |
| 20 | vision + rapidocr | single | 0.9644 | 0.9523 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 7.48 |
| 21 | vision + rapidocr | primary_with_floor | 0.9644 | 0.9523 | 0.8969 | 0.8872 | 0.8663 | 0.8225 | below | 8.28 |
| 22 | vision2 | single | 0.9584 | 0.9496 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 0.58 |
| 23 | vision2 + tesseract | single | 0.9584 | 0.9496 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 2.61 |
| 24 | vision2 + tesseract | primary_with_floor | 0.9584 | 0.9496 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 1.89 |
| 25 | vision2 + tesseract2 | single | 0.9584 | 0.9496 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 5.60 |
| 26 | vision2 + tesseract2 | primary_with_floor | 0.9584 | 0.9496 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 6.45 |
| 27 | vision2 + rapidocr | single | 0.9586 | 0.9498 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 8.79 |
| 28 | vision2 + rapidocr | primary_with_floor | 0.9586 | 0.9498 | 0.8911 | 0.8795 | 0.8702 | 0.8225 | below | 9.61 |
| 29 | rapidocr | single | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 9.91 |
| 30 | rapidocr + tesseract | single | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 12.34 |
| 31 | rapidocr + tesseract | primary_with_floor | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 10.60 |
| 32 | rapidocr + tesseract2 | single | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 13.48 |
| 33 | rapidocr + tesseract2 | primary_with_floor | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 15.89 |
| 34 | rapidocr + vision | single | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 8.88 |
| 35 | rapidocr + vision | primary_with_floor | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 8.94 |
| 36 | rapidocr + vision2 | single | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 8.41 |
| 37 | rapidocr + vision2 | primary_with_floor | 0.9869 | 0.9831 | 0.9361 | 1.0000 | 1.0000 | 0.9554 | pass | 8.33 |
| 38 | tesseract + vision | corroborated_trigrams | 0.9470 | 0.9465 | 0.8675 | 0.0809 | 0.2324 | 0.2246 | below | 2.60 |
| 39 | tesseract + vision + rapidocr | corroborated_trigrams | 0.9754 | 0.9716 | 0.8969 | 0.8349 | 0.8469 | 0.8225 | below | 10.11 |
| 40 | tesseract + vision2 | corroborated_trigrams | 0.9470 | 0.9465 | 0.8689 | 0.0809 | 0.2324 | 0.3580 | below | 2.47 |
| 41 | tesseract + vision2 + rapidocr | corroborated_trigrams | 0.9750 | 0.9714 | 0.8911 | 0.8349 | 0.8469 | 0.8225 | below | 10.63 |
| 42 | tesseract + rapidocr | corroborated_trigrams | 0.9508 | 0.9567 | 0.9043 | 0.2324 | 0.3855 | 0.3409 | below | 9.23 |
| 43 | tesseract + rapidocr + vision | corroborated_trigrams | 0.9869 | 0.9831 | 0.9377 | 1.0000 | 1.0000 | 0.9554 | pass | 11.47 |
| 44 | tesseract2 + vision | corroborated_trigrams | 0.9812 | 0.9802 | 0.8975 | 0.9880 | 0.9961 | 0.8391 | below | 5.65 |
| 45 | tesseract2 + vision + rapidocr | corroborated_trigrams | 0.9822 | 0.9810 | 0.8969 | 1.0000 | 0.9961 | 0.8225 | below | 15.16 |
| 46 | tesseract2 + vision2 | corroborated_trigrams | 0.9812 | 0.9787 | 0.8911 | 0.9880 | 0.9961 | 0.8225 | below | 3.91 |
| 47 | tesseract2 + vision2 + rapidocr | corroborated_trigrams | 0.9822 | 0.9797 | 0.8911 | 1.0000 | 0.9961 | 0.8225 | below | 10.63 |
| 48 | tesseract2 + rapidocr | corroborated_trigrams | 0.9863 | 0.9819 | 0.9351 | 1.0000 | 1.0000 | 0.9554 | pass | 12.68 |
| 49 | tesseract2 + rapidocr + vision | corroborated_trigrams | 0.9828 | 0.9831 | 0.9351 | 1.0000 | 1.0000 | 0.9554 | pass | 11.31 |
| 50 | vision + rapidocr | corroborated_trigrams | 0.9652 | 0.9812 | 0.9343 | 0.8872 | 1.0000 | 0.9388 | below | 7.17 |
| 51 | vision + rapidocr + tesseract | corroborated_trigrams | 0.9818 | 0.9812 | 0.8973 | 0.9880 | 1.0000 | 0.8225 | below | 6.70 |
| 52 | vision2 + rapidocr | corroborated_trigrams | 0.9818 | 0.9810 | 0.9343 | 0.9880 | 1.0000 | 0.9388 | pass | 5.55 |
| 53 | vision2 + rapidocr + tesseract | corroborated_trigrams | 0.9818 | 0.9810 | 0.8915 | 0.9880 | 1.0000 | 0.8225 | below | 6.13 |

Sets 29 to 37 (RapidOCR alone, or RapidOCR as the primary with a second engine) read identically. Tesseract's `ita` model is just below the accents floor on the small (110 dpi) pages, 0.897. Tesseract's `eng` model loses almost every accented word, 0.07 to 0.08. Vision falls below the floor in every condition, 0.82 to 0.89.

## Selection (`kit choose`, policy `maximin-v1`)

```json
{"darwin-arm64": {"chosen": 29, "runner_up": 52, "margin": 0.0018, "policy": "maximin-v1", "override_reason": null}}
```

`dropped_models`: `ita.traineddata`. `unbenched`: `win-amd64`, `linux-x86_64`, `linux-aarch64`. `next`: run `kit wheel`.

## `kit wheel`, `kit verify`

`kit wheel`: wheel `pdfintake_lang_it-1.0.0-py3-none-any.whl`, 16,081 bytes; canary `canary-it.png` (12,192 bytes, 300 dpi) with `città`, `perché`, `così` and `4521`, all found by RapidOCR.

`kit verify it --download`: exit 0. Checks: manifest `ok`, download `ok`, layer `ok`, wheel `ok`; `warnings: []`.

`kit verify it --download --rebench`: exit 0, all four checks `ok`, `warnings: []`. Re-bench on `darwin-arm64`, tolerance 0.01 (no Vision in the set), every value identical to the claim; `drift: []`, `floors: []`:

| Measured | clean | scan | small |
|---|---|---|---|
| words | 0.9869 | 0.9831 | 0.9361 |
| trigrams | 0.9993 | 0.9988 | 0.9566 |
| digits | 0.9761 | 0.9872 | 0.9282 |
| accents | 1.0000 | 1.0000 | 0.9554 |

Worst page 0.6225, as claimed. Speed 6.19 s per page against the claimed 9.915 (item 6).

## Tests

The whole suite, run once in the work tree (which holds the generated pack index and, at the time, the other new launch packs being built alongside this one): 3127 passed, 36 skipped, 1 failed. The failure is `tests/test_release.py::test_index_regenerates_identically`: `build_index.py --check` reports "version 1.0.0 has no wheel in dist" for `es`, `fr`, `it` and `pt`. That is the failure step 9 says to expect for a new pack, so it isn't stop point 4. The 36 skips: 20 end-to-end run tests and one Vision test need the English data package installed in the test environment, which it isn't (`base_pack_missing`); 4 sandbox smoke tests run only on a non-editable install; the pack network test runs only with `PDFINTAKE_NETWORK_TESTS=1`; the pyproject pin test skips while the pin is the placeholder; one SIGHUP test skips because a handler is already set; and 8 contract and record cases skip because they don't apply to the case under test (no corroborators, no tie order, no model files, valid values).

## `kit report it` body

`commit`: `packs/it/bench.json`, `packs/it/corpus.json`, `packs/it/manifest.json`. `leave_out`: `packs/it/NOTICE-tessdata`, `packs/it/manifest.draft.json`. No `missing` notices.

### Pack `it` 1.0.0: Italian

- Languages: it (primary)
- Scripts: Latin
- Requires core: `>=0.1.0`

**Reader sets**

- `default`: rapidocr (rapidocr, reader); pick `single` v1, fallback `none`

**Models and fonts**

| File | Kind | Bytes | Licence | Source |
|---|---|---|---|---|

- Canary: `canary-it.png`, 12192 bytes
- Must survive: `accents` (word, floor 0.9)

**Corpus**

- `public-v1`: 8 texts, 2 fonts, pages {"form": 2, "microprint": 1, "mixed_script": 1, "prose": 4}
  - Acqua: wikipedia `it` revision 152992062, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=152992062&action=raw
  - Sole: wikipedia `it` revision 152894518, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=152894518&action=raw
  - Luna: wikipedia `it` revision 153180897, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=153180897&action=raw
  - Fotosintesi clorofilliana: wikipedia `it` revision 152412937, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=152412937&action=raw
  - Scacchi: wikipedia `it` revision 152656469, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=152656469&action=raw
  - Pane: wikipedia `it` revision 152571357, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=152571357&action=raw
  - Vulcano: wikipedia `it` revision 153042642, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=153042642&action=raw
  - Fiume: wikipedia `it` revision 151257848, CC-BY-SA-4.0: https://it.wikipedia.org/w/index.php?oldid=151257848&action=raw

**Bench**

- Kit 1.0, corpus `public-v1@dd83571921cb07c4262d3b1d41271b212be75e3cb335f62e95ff196e093d3b8c`
- Engines: rapidocr 3.10.0, tesseract 5.5.3 (tesserocr 2.12.0), vision rev3 (macOS 27.2)

`darwin-arm64`: worst page 0.623, 9.915 s/page, policy `maximin-v1`, margin 0.002

| Metric | clean | scan | small |
|---|---|---|---|
| digits | 0.976 | 0.987 | 0.928 |
| must-accents | 1.000 | 1.000 | 0.955 |
| trigrams | 0.999 | 0.999 | 0.957 |
| words | 0.987 | 0.983 | 0.936 |

**Checklist**

- [ ] The PR changes only `packs/it/**`.
- [ ] No binaries: no model, font, image or PDF.
- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.
- [ ] `pdfintake kit verify it --download` passes.
- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.
- [ ] The must-survive characters were confirmed by someone who reads the language.
