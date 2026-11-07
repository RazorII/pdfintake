# Kit trial: French (`fr`)

A contributor's Claude, who wrote none of the kit, added the French pack by following `CLAUDE.md` and `docs/add-a-language.md` step by step, without using the kit's source, its tests or any design notes to get past a step. Every command, its exit code, and every place the doc was wrong, unclear or missing something is below, with what was done instead.

- Machine: macOS 27.2 on Apple silicon (`darwin-arm64`), Apple Vision available (revision 3).
- Engines measured: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Step 1 was a shared bench environment set up beforehand: a clean clone with its own `.venv` (editable install, English data wheel built). Every kit command ran as `<bench>/.venv/bin/pdfintake kit <cmd> fr --packs <repo>/packs --work <work> --json`, where `<work>` is a kit work folder outside the repo. Other packs were being benched on the same Mac at the same time (load average 19 to 88), so every time and every seconds-per-page figure below is slower than an idle machine would give.
- Stop points met: none. `kit choose` exited 0 and its pick was kept as is. The must-survive characters were given by the person: `accents=àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ:word:0.9`. Every corpus text is Wikipedia, CC-BY-SA-4.0, on a neutral topic.

## Result

- Chosen on `darwin-arm64`: **RapidOCR alone** (Latin recogniser from the base pack), pick `single` (set 29). Worst-condition word recall 0.9521; accents recall 0.995 clean, 0.990 scan and 0.960 small.
- Runner-up: RapidOCR + Tesseract (`eng`), `primary_with_floor` (set 31), same numbers, margin 0.000.
- `dropped_models`: `fra.traineddata`. The manifest pins no model and no font of its own; the wheel is 23,008 bytes.
- `kit verify fr --download`: exit 0, all four checks `ok`, in the populated work folder and in a fresh one. `kit verify fr --download --rebench`: exit 0, re-bench identical to the claim (no drift, no floor missed).
- Files: `packs/fr/manifest.json`, `corpus.json`, `bench.json` (to commit, per `kit report`); `manifest.draft.json` and `NOTICE-tessdata` stay in the folder and are left out.

## Steps

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | (bench environment already set up) `pdfintake doctor --json` | **3** | 1 s |
| 2 | `pdfintake kit init fr --script Latin --must-survive 'accents=àâçéèêëîïôùûüÿœæÀÂÇÉÈÊËÎÏÔÙÛÜŸŒÆ:word:0.9'` | 0 | 1 s |
| 3 | `pdfintake kit corpus fr --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072,Q4022 --font <NotoSans-Regular> --font <NotoSerif-Regular>` | 0 | 12 s |
| 4 | `pdfintake kit render fr` | 0 | 8 s |
| 5 | `pdfintake kit candidates fr` | 0 | 3 s |
| 5 | `pdfintake kit bench fr` (background, `nohup`) | 0 | **about 36 min** (54 sets, loaded machine) |
| 6 | `pdfintake kit choose fr` | 0 | 1 s |
| 7 | `kit private` | skipped | optional; the trial has no contributor PDFs |
| 8 | `pdfintake kit wheel fr --must-find Photosynthèse --must-find forêt --must-find garçon --must-find maïs --must-find cœur --must-find 4521` | **1** | 1 s |
| 8 | `pdfintake kit wheel fr --must-find Photosynthèse --must-find garçon --must-find cœur --must-find 4521` | 0 | 3 s |
| 8 | `pdfintake kit verify fr --download` | 0 | 1 s |
| 8 | `pdfintake kit verify fr --download` with a fresh, empty `--work` | 0 | 2 s |
| 8 | `pdfintake kit verify fr --download --rebench` (background) | 0 | 55 s |
| 9 | `pdfintake kit report fr` | 0 | 1 s |

The corpus is eight French Wikipedia articles, each title as `kit corpus` reported it: Eau (Q283), Soleil (Q525), Lune (Q405), Photosynthèse (Q11982), Échecs (Q718), Pain (Q7802), Volcan (Q8072) and Rivière (Q4022). The fonts are the doc's two example URLs, `NotoSans-Regular.ttf` and `NotoSerif-Regular.ttf` from `notofonts.github.io` at commit `28b15b4…`.

`kit render` drew 24 images: 4 prose, 2 form, 1 microprint and 1 mixed-script page, each in the three conditions. The mixed-script page's second script was Greek, drawn by the two Latin fonts, as the doc says. `kit candidates` listed 54 sets (`total` 54, `refused_by_loader` 0, `cut` 0, `kept` 54); `kit bench` read all 24 pages with each of them, every set `ok`.

## Where the doc was wrong, unclear or missing something

Each item gives the step, what happened, and what was done instead. "Doc" means `docs/add-a-language.md`.

### Step 1: set up

1. **`doctor` exited 3 for a reason the doc only routes to "run `dev-install.sh` again".** The problem was not `index_missing` but `base_pack` with code `hash_mismatch`: the installed `pdfintake_data_en/manifest.json` has a different SHA-256 from the one the pack index pins (`"blocking": true`). The doc says any problem other than `index_missing` means re-running `scripts/dev-install.sh`, and stop point 4 if it stays. *Did instead:* this trial was told the environment was already set up and must not be changed, so `dev-install.sh` was not re-run; went on to step 2, on the strength of the doc's own remark that the kit takes the base pack from `packs/en`. Every kit command afterwards exited 0. A likely cause is that `packs/en/manifest.json` changed in the tree after the bench environment's English wheel and index were built, which a contributor would meet after pulling a change to the English pack. *Fix:* say which `doctor` problems block the kit and which don't (the kit apparently doesn't need the installed base pack to match the index), or have `doctor` name the kit's own readiness separately.

### Step 2: start the draft

2. **`next` was empty right after `kit init`.** The doc says "`next` lists anything still missing", and the corpus was still empty (`corpus.json` has no texts, no fonts). A reader taking `next` literally would think nothing is left. *Did instead:* followed the doc's step order. *Fix:* have `kit init`'s `next` name `kit corpus` while the corpus has no texts or fewer than two fonts per script, or say in the doc that `next` covers only the manifest draft.

### Step 5: candidates and bench

3. **The bare engine id doesn't always mean the draft's parameters.** In set 0 (the draft's own set) `tesseract` reads with `fra`; in sets 1 to 7 the same id `tesseract` reads with `eng`, and the draft's `fra` reading appears as `tesseract2`. The doc says set 0 keeps "the draft's own ids" and a numbered id is "the same engine with its next set of parameters", but not that the un-numbered id outside set 0 may be a different parameter set (here, the base pack's English model). Reading the bench table, "tesseract + rapidocr" in set 0 and in sets 6 and 7 are different reader sets. *Did instead:* read each set's `params` in the candidates JSON. *Fix:* one sentence: outside set 0, ids are numbered per distinct parameter set, so always read `params`; or give set 0's uses the same numbered ids as the other sets.
4. **The time estimate assumes an idle machine.** The doc says about 20 seconds per set, 10 to 20 minutes for 30 to 60 sets. With other benches running, the 54 sets took about 36 minutes (18 to 72 s per set), and the kit's own estimate after set 1 said 49 minutes. That's fine for a background run, but the seconds per page written into `bench.json` and the manifest (7.25 s for RapidOCR alone, against 2.25 s in the Spanish trial on the same Mac) are the loaded machine's, and the tie-break uses them. *Fix:* say to bench on an otherwise idle machine, since `seconds_per_page` is recorded and used to break ties.

### Step 8: wheel and verify

5. **`--must-find` takes at most four words, and the doc doesn't say so.** The doc says "Repeat `--must-find` for each word". Six words (five French words and 4521) gave exit 1 with `kit_invalid_manifest`: "the manifest with its canary wouldn't load: $.canary.must_find: has 6 items; at most 4". The doc's step 8 only mentions `kit_canary_failed` for exit 1. *Did instead:* kept three words that between them carry `è`, `ç` and `œ`, plus 4521: `Photosynthèse`, `garçon`, `cœur`, `4521`. *Fix:* say "up to four words, 4521 included", and have the kit check the count as a bad argument (exit 2) before it renders.
6. **In a fresh work folder, `--download` fetched one file.** `kit verify fr --download` with an empty `--work` took 2 seconds and left only the base pack's `NotoSans-Regular.ttf` (431,364 bytes) and the re-rendered `canary-fr.png`. The French pack pins no model or font (`kit choose` dropped `fra.traineddata`), and RapidOCR's models come from the base pack, so this matches the doc's wording ("every model and font the pack pins, and the base pack's fonts"). Not a doc error; noted because a reader might expect the base pack's models to be fetched too.

### Step 9: report

7. **The report's "Models and fonts" table is empty.** The pack pins nothing of its own, so the body shows the table's header and no rows. A reviewer can't tell an empty table from a rendering fault. *Fix:* print a line such as "none of its own: reads with the base pack's RapidOCR models" when the lists are empty.
8. **`NOTICE-tessdata` was written by `kit init` and then left out.** `kit report` lists it under `leave_out`, which the doc covers ("a notice for a model `kit choose` dropped"), so this is consistent. Noted only because the folder ends the trial with two files that must not be committed (`manifest.draft.json`, `NOTICE-tessdata`), and nothing ignores them.

### What went as written

- `kit corpus` printed each text's `title`, and every Wikidata id led to the intended article (Q4022 is "Rivière").
- The mixed-script page used Greek with no extra font, as the doc says.
- `kit choose`'s `next` named `kit wheel` while the manifest had no canary.
- `kit verify` ran all four checks with `--download` in a fresh work folder, and `warnings` was empty.

## The bench on `darwin-arm64`

Word recall and must-survive (`accents`) recall per condition. The floor is 0.9 in every condition; a set below it in any condition is dropped. Engine ids with a 2 are the same engine with other parameters: `tesseract` reads with `eng` except in set 0, where it reads with `fra`; `tesseract2` reads with `fra`; `vision` has language correction on and `vision2` off. Seconds per page were measured on a loaded machine.

| # | Reader set | Pick | Words clean | Words scan | Words small | Accents clean | Accents scan | Accents small | Floors | s/page |
|---|---|---|---|---|---|---|---|---|---|---|
{'rapidocr': '3.10.0', 'tesseract': '5.5.3 (tesserocr 2.12.0)', 'vision': 'rev3 (macOS 27.2)'}
| 0 | tesseract + rapidocr | primary_with_floor | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 7.93 |
| 1 | tesseract | single | 0.9191 | 0.9149 | 0.7994 | 0.6621 | 0.6639 | 0.5447 | below | 2.56 |
| 2 | tesseract + vision | single | 0.9191 | 0.9149 | 0.8010 | 0.6621 | 0.6639 | 0.5447 | below | 2.53 |
| 3 | tesseract + vision | primary_with_floor | 0.9191 | 0.9149 | 0.8010 | 0.6621 | 0.6639 | 0.5447 | below | 1.95 |
| 4 | tesseract + vision2 | single | 0.9191 | 0.9149 | 0.8010 | 0.6621 | 0.6639 | 0.5447 | below | 1.47 |
| 5 | tesseract + vision2 | primary_with_floor | 0.9191 | 0.9149 | 0.8010 | 0.6621 | 0.6639 | 0.5447 | below | 1.33 |
| 6 | tesseract + rapidocr | single | 0.9191 | 0.9149 | 0.8010 | 0.6621 | 0.6639 | 0.5447 | below | 7.15 |
| 7 | tesseract + rapidocr | primary_with_floor | 0.9191 | 0.9149 | 0.8010 | 0.6621 | 0.6639 | 0.5447 | below | 10.97 |
| 8 | tesseract2 | single | 0.9779 | 0.9740 | 0.8673 | 0.9978 | 0.9904 | 0.8654 | below | 3.08 |
| 9 | tesseract2 + vision | single | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 2.59 |
| 10 | tesseract2 + vision | primary_with_floor | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 1.85 |
| 11 | tesseract2 + vision2 | single | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 1.61 |
| 12 | tesseract2 + vision2 | primary_with_floor | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 1.73 |
| 13 | tesseract2 + rapidocr | single | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 5.01 |
| 14 | tesseract2 + rapidocr | primary_with_floor | 0.9779 | 0.9740 | 0.8689 | 0.9978 | 0.9904 | 0.8654 | below | 5.00 |
| 15 | vision | single | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 0.76 |
| 16 | vision + tesseract | single | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 2.07 |
| 17 | vision + tesseract | primary_with_floor | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 2.29 |
| 18 | vision + tesseract2 | single | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 3.11 |
| 19 | vision + tesseract2 | primary_with_floor | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 3.36 |
| 20 | vision + rapidocr | single | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 7.23 |
| 21 | vision + rapidocr | primary_with_floor | 0.9535 | 0.9515 | 0.9035 | 0.9196 | 0.9481 | 0.8811 | below | 6.46 |
| 22 | vision2 | single | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 0.60 |
| 23 | vision2 + tesseract | single | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 1.48 |
| 24 | vision2 + tesseract | primary_with_floor | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 1.46 |
| 25 | vision2 + tesseract2 | single | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 1.65 |
| 26 | vision2 + tesseract2 | primary_with_floor | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 1.74 |
| 27 | vision2 + rapidocr | single | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 6.42 |
| 28 | vision2 + rapidocr | primary_with_floor | 0.9473 | 0.9500 | 0.8995 | 0.9151 | 0.9454 | 0.8804 | below | 6.96 |
| 29 | rapidocr | single | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 7.25 |
| 30 | rapidocr + tesseract | single | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 9.61 |
| 31 | rapidocr + tesseract | primary_with_floor | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 5.60 |
| 32 | rapidocr + tesseract2 | single | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 8.32 |
| 33 | rapidocr + tesseract2 | primary_with_floor | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 12.58 |
| 34 | rapidocr + vision | single | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 10.13 |
| 35 | rapidocr + vision | primary_with_floor | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 10.68 |
| 36 | rapidocr + vision2 | single | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 8.55 |
| 37 | rapidocr + vision2 | primary_with_floor | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 8.56 |
| 38 | tesseract + vision | corroborated_trigrams | 0.9241 | 0.9149 | 0.8616 | 0.6958 | 0.6639 | 0.6677 | below | 2.71 |
| 39 | tesseract + vision + rapidocr | corroborated_trigrams | 0.9682 | 0.9615 | 0.9035 | 0.9486 | 0.9312 | 0.8811 | below | 10.01 |
| 40 | tesseract + vision2 | corroborated_trigrams | 0.9241 | 0.9181 | 0.8581 | 0.6958 | 0.6879 | 0.6694 | below | 2.18 |
| 41 | tesseract + vision2 + rapidocr | corroborated_trigrams | 0.9655 | 0.9615 | 0.8995 | 0.9318 | 0.9312 | 0.8804 | below | 10.88 |
| 42 | tesseract + rapidocr | corroborated_trigrams | 0.9299 | 0.9314 | 0.9074 | 0.6958 | 0.7367 | 0.7445 | below | 9.93 |
| 43 | tesseract + rapidocr + vision | corroborated_trigrams | 0.9845 | 0.9768 | 0.9521 | 0.9954 | 0.9896 | 0.9602 | pass | 13.25 |
| 44 | tesseract2 + vision | corroborated_trigrams | 0.9773 | 0.9735 | 0.9033 | 0.9978 | 0.9881 | 0.8810 | below | 4.63 |
| 45 | tesseract2 + vision + rapidocr | corroborated_trigrams | 0.9773 | 0.9746 | 0.9041 | 0.9978 | 0.9904 | 0.8834 | below | 12.87 |
| 46 | tesseract2 + vision2 | corroborated_trigrams | 0.9769 | 0.9741 | 0.8987 | 0.9978 | 0.9881 | 0.8791 | below | 2.65 |
| 47 | tesseract2 + vision2 + rapidocr | corroborated_trigrams | 0.9771 | 0.9753 | 0.8995 | 0.9966 | 0.9904 | 0.8816 | below | 11.30 |
| 48 | tesseract2 + rapidocr | corroborated_trigrams | 0.9830 | 0.9751 | 0.9496 | 0.9978 | 0.9896 | 0.9578 | pass | 11.37 |
| 49 | tesseract2 + rapidocr + vision | corroborated_trigrams | 0.9783 | 0.9763 | 0.9504 | 0.9990 | 0.9908 | 0.9602 | pass | 10.10 |
| 50 | vision + rapidocr | corroborated_trigrams | 0.9765 | 0.9748 | 0.9441 | 0.9954 | 0.9801 | 0.9579 | pass | 9.27 |
| 51 | vision + rapidocr + tesseract | corroborated_trigrams | 0.9771 | 0.9750 | 0.9447 | 0.9954 | 0.9812 | 0.9590 | pass | 7.00 |
| 52 | vision2 + rapidocr | corroborated_trigrams | 0.9744 | 0.9748 | 0.9437 | 0.9954 | 0.9801 | 0.9568 | pass | 7.63 |
| 53 | vision2 + rapidocr + tesseract | corroborated_trigrams | 0.9788 | 0.9750 | 0.9447 | 0.9954 | 0.9812 | 0.9590 | pass | 9.44 |

Tesseract's `fra` model (sets 0 and 8 to 14) falls just under the accents floor on the small (110 dpi) pages, 0.865; its `eng` model loses most accents (0.54 to 0.66). Vision falls below the floor on the small pages, 0.880 to 0.881. Every set with RapidOCR as the primary (29 to 37) reads identically: the second engine changed nothing on these pages. Seven `corroborated_trigrams` sets also pass the floors, none with a better worst-condition word recall than RapidOCR alone.

## Selection (`kit choose`, policy `maximin-v1`)

```json
{"darwin-arm64": {"chosen": 29, "runner_up": 31, "margin": 0.0, "policy": "maximin-v1", "override_reason": null}}
```

`unbenched`: `win-amd64`, `linux-x86_64`, `linux-aarch64` (one machine; left to CI and the maintainer, as step 5 says). `dropped_models`: `fra.traineddata`. `canary`: `none`, and `next` named `kit wheel`.

## `kit verify fr --download --rebench`

Exit 0. Checks: manifest `ok`, download `ok`, layer `ok`, wheel `ok`; `warnings` empty. Re-bench on `darwin-arm64`, tolerance 0.01 (no Vision in the set): every metric in every condition (words, trigrams, digits and accents; the table in the report body below) is identical to the claim; worst page 0.723 as claimed; 6.93 s per page against the claimed 7.25. `drift: []`, `floors: []`.

`kit wheel`: wheel 23,008 bytes (SHA-256 `6aade0ae…`); canary `canary-fr.png` (19,089 bytes, 300 dpi) with `Photosynthèse`, `garçon`, `cœur` and `4521`, all found by RapidOCR.

## Tests

The whole suite, run once in the work tree (which holds the maintainer's generated pack index) with the test venv: 3123 passed, 36 skipped, 1 failed, in 6 min 53 s on the loaded machine. The failure is `tests/test_release.py::test_index_regenerates_identically`: `build_index.py --check` reports "fr: version 1.0.0 has no wheel in dist" (and the same for `es`). Step 9 of the doc says this one failure is expected in a pack PR. The skips: the network pack tests (`PDFINTAKE_NETWORK_TESTS` unset), the sandbox smoke tests (non-editable install only), the end-to-end run tests and one Vision test that need the English data package, which the test venv doesn't have, plus contract and record cases that don't apply.

## `kit report fr` body

### Pack `fr` 1.0.0: French

- Languages: fr (primary)
- Scripts: Latin
- Requires core: `>=0.1.0`

**Reader sets**

- `default`: rapidocr (rapidocr, reader); pick `single` v1, fallback `none`

**Models and fonts**

| File | Kind | Bytes | Licence | Source |
|---|---|---|---|---|

- Canary: `canary-fr.png`, 19089 bytes
- Must survive: `accents` (word, floor 0.9)

**Corpus**

- `public-v1`: 8 texts, 2 fonts, pages {"form": 2, "microprint": 1, "mixed_script": 1, "prose": 4}
  - Eau: wikipedia `fr` revision 241030901, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=241030901&action=raw
  - Soleil: wikipedia `fr` revision 241025844, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=241025844&action=raw
  - Lune: wikipedia `fr` revision 240787057, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=240787057&action=raw
  - Photosynthèse: wikipedia `fr` revision 238282939, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=238282939&action=raw
  - Échecs: wikipedia `fr` revision 241017049, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=241017049&action=raw
  - Pain: wikipedia `fr` revision 240824437, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=240824437&action=raw
  - Volcan: wikipedia `fr` revision 240478850, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=240478850&action=raw
  - Rivière: wikipedia `fr` revision 236817793, CC-BY-SA-4.0: https://fr.wikipedia.org/w/index.php?oldid=236817793&action=raw

**Bench**

- Kit 1.0, corpus `public-v1@809b4abfa793e9450044de80ddb90dca937f6c8800e649ba436b143a0466fd9e`
- Engines: rapidocr 3.10.0, tesseract 5.5.3 (tesserocr 2.12.0), vision rev3 (macOS 27.2)

`darwin-arm64`: worst page 0.723, 7.246 s/page, policy `maximin-v1`, margin 0.000

| Metric | clean | scan | small |
|---|---|---|---|
| digits | 1.000 | 1.000 | 0.978 |
| must-accents | 0.995 | 0.990 | 0.960 |
| trigrams | 1.000 | 0.999 | 0.977 |
| words | 0.985 | 0.977 | 0.952 |

**Checklist**

- [ ] The PR changes only `packs/fr/**`.
- [ ] No binaries: no model, font, image or PDF.
- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.
- [ ] `pdfintake kit verify fr --download` passes.
- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.
- [ ] The must-survive characters were confirmed by someone who reads the language.

