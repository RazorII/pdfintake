# Kit trial: Portuguese (`pt`)

A contributor's Claude, who wrote none of the kit, added the Portuguese pack by following `CLAUDE.md` and `docs/add-a-language.md` step by step, without reading the kit's source, its tests or any design notes to get past a step. Every command, its exit code, and every place the doc was wrong, unclear or missing something is below, with what was done instead.

- Machine: macOS 27.2 on Apple silicon (`darwin-arm64`), Apple Vision available (revision 3).
- Engines measured: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Step 1 was done beforehand: a shared clone of the repo with its own `.venv` (`<bench>` below). Every kit command ran as `<bench>/.venv/bin/pdfintake kit <cmd> pt --packs <repo>/packs --work <work> --json`, so the pack landed in the repo while downloads, pages and wheels stayed in a work folder outside it.
- Four other kit benches (`fr`, `it`, `pl`, `vi`) ran on the same Mac at the same time as this one (load average about 78). That slowed the bench and inflates every `seconds_per_page` below; recall numbers aren't affected (the re-bench reproduced them exactly).
- Stop points met: none. `kit choose` exited 0 and its pick was kept as is. The must-survive characters were given by the person: `accents=áâãàçéêíóôõúÁÂÃÀÇÉÊÍÓÔÕÚ:word:0.9`. Every corpus text is Wikipedia, CC-BY-SA-4.0, on a neutral topic.

## Result

- Chosen on `darwin-arm64`: **RapidOCR alone** (Latin recogniser from the base pack), pick `single` (set 29). Worst-condition word recall 0.9511; accents recall 0.998 clean, 0.997 scan and 0.950 small.
- Runner-up: set 37, RapidOCR + Vision (`vision2`, language correction off) with `primary_with_floor`, same numbers, margin 0.000. Ten sets tie at 0.9511; RapidOCR alone wins on fewer engines.
- `dropped_models`: `por.traineddata` (Tesseract's Portuguese model), so the pack pins no model or font of its own and the wheel is 26,453 bytes.
- `kit verify pt --download`: exit 0, all four checks `ok`, no warnings. With `--rebench`: exit 0, re-bench identical to the claim (no drift, no floor missed).
- Files to commit (per `kit report`): `packs/pt/manifest.json`, `corpus.json`, `bench.json`. Left out: `manifest.draft.json`, `NOTICE-tessdata` (its model was dropped).

## Steps

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | done beforehand (shared bench clone and venv) | n/a | n/a |
| 1 | `pdfintake doctor --json` | **3** | 1 s |
| 2 | `pdfintake kit init pt --script Latin --must-survive 'accents=áâãàçéêíóôõúÁÂÃÀÇÉÊÍÓÔÕÚ:word:0.9'` | 0 | 3 s |
| 3 | `pdfintake kit corpus pt --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072,Q4022 --font <NotoSans-Regular> --font <NotoSerif-Regular>` | 0 | 12 s |
| 4 | `pdfintake kit render pt` | 0 | 7 s |
| 5 | `pdfintake kit candidates pt` | 0 | 1 s |
| 5 | `pdfintake kit bench pt` (background, `nohup`) | 0 | **42 min** (11:59 to 12:41, under four other benches) |
| 6 | `pdfintake kit choose pt` | 0 | under 1 s |
| 7 | `kit private` | skipped | optional; the trial has no contributor PDFs |
| 8 | `pdfintake kit wheel pt --must-find Fotossíntese --must-find informações --must-find lâmpada --must-find vovô --must-find pão --must-find 4521` | **1** | 1 s |
| 8 | `pdfintake kit wheel pt --must-find Fotossíntese --must-find informações --must-find lâmpada --must-find 4521` | 0 | 2 s |
| 8 | `pdfintake kit verify pt --download` | 0 | 1 s |
| 8 | `pdfintake kit verify pt --download --rebench` | 0 | 47 s |
| 9 | `pdfintake kit report pt` | 0 | 1 s |

The corpus is eight Portuguese Wikipedia articles: Água (Q283), Sol (Q525), Lua (Q405), Fotossíntese (Q11982), Xadrez (Q718), Pão (Q7802), Vulcão (Q8072) and Rio (Q4022, the article on rivers in general). `kit corpus` printed every title, so each id was checked against the article meant. The fonts are `NotoSans-Regular.ttf` and `NotoSerif-Regular.ttf` from `github.com/notofonts/notofonts.github.io/raw/28b15b4b43b7bed62b5cf6e6b0b5ff5846270535/fonts/<family>/unhinted/ttf/`, exactly as the doc's example.

`kit render` drew 24 pages: 4 prose, 2 form, 1 microprint and 1 mixed-script page, each in the three conditions. The mixed-script page's second script is Greek, as step 3 says. `kit candidates` listed 54 sets (`total` 54, `refused_by_loader` 0, `cut` 0, `kept` 54); `kit bench` read all 24 pages with each of them, every set `ok`, 20 of the 54 with `floors_ok: true`.

## Where the doc was wrong, unclear or missing something

Each item gives the step, what happened, and what was done instead. "Doc" means `docs/add-a-language.md`.

### Step 1: set up

1. **`doctor` exited 3 with a base-pack `hash_mismatch`, not `index_missing`.** `base_pack` came back `"ok": false, "code": "hash_mismatch"`: the installed `pdfintake_data_en/manifest.json` has SHA-256 `d784…f307` while the pack index pins `18ad…223d`. `index` was `ok`. The doc's route for this is right ("any other problem: run `scripts/dev-install.sh` again"), but this trial's environment forbade touching the shared venv. The cause is outside the doc: the shared clone's source (with a newer `index.json`) had been synced from the repo, but its `packs/en` and installed English data wheel are older than the repo's `packs/en/manifest.json`. *Did instead:* went on without re-running `dev-install.sh`, because the doc says the kit takes the base pack from `packs/en` (here the repo's), and every kit command worked. *Gap:* the doc says "the kit doesn't read the index, and takes the base pack from `packs/en`" only under `index_missing`. A contributor who pulls a commit that changes `packs/en` meets this same `hash_mismatch`; the doc could say plainly that re-running `dev-install.sh` fixes it, and whether the kit can proceed meanwhile.

### Step 2: start the draft

2. **`kit init` also writes `NOTICE-tessdata`**, which step 2 doesn't mention (it lists only `manifest.draft.json` and `corpus.json`). Step 9 explains notice files later, and `kit report` put this one under `leave_out` once `kit choose` dropped the model. Harmless. *Fix:* name it in step 2.
3. **`next` was empty although the corpus was.** `kit init` printed `"next": []` with an empty `corpus.json`. The doc says "`next` lists anything still missing"; the corpus is still missing at that point. *Fix:* have `next` name `kit corpus`, or say `next` covers only the draft itself.
4. **No guidance for a language with regional variants.** Portuguese is written as European and Brazilian Portuguese; `vision_languages` came back `[["pt-BR", "en-US"]]` (Vision has no `pt-PT` on this Mac) and `pt.wikipedia.org` mixes both. The doc mentions variant ids only as `sr-latn`. *Did instead:* used `pt` for the whole language, which matches Tesseract's single `por` model and the Wikipedia edition. *Fix:* one line on when a variant deserves its own pack id.

### Step 5: candidates and bench

5. **The bare id `tesseract` means a different model in set 0 than in the other sets.** In set 0 (the draft's own set) `tesseract` reads with `lang: por`; in sets 1 to 7, 16, 17, 23, 24, 30, 31, 38 to 43, 51 and 53, `tesseract` reads with `lang: eng` (the base pack's model), and `tesseract2` is the Portuguese one. The progress line says only `set 2/54 (tesseract, single) ok`, so the numbers read as "Tesseract's Portuguese model keeps 18% of the accents", which is wrong: Tesseract `por` alone (set 8) keeps 100%, 99.3% and 90.1%. The doc says set 0 has "the draft's own ids" and a numbered id is "the same engine with its next set of parameters", but not that the bare id outside set 0 is the engine's first proposal (here English). *Did instead:* read each set's `params` in `packs/pt/bench.json`. *Fix:* say so in step 5, and put the model (`lang`, `rec`) or the variant's distinguishing parameter in the progress line.
6. **The bench's JSON doesn't say which set is which.** Its `sets` carry `index`, `score`, `floors_ok`, `below_floor`, `worst_page` and `seconds_per_page`, but no engines or parameters, so judging the result means opening `packs/pt/bench.json` (each set's `reader_set`) or the candidates file in the work folder. The doc doesn't say where to look. *Fix:* name the file in step 5, or add the engine list to each set in the JSON.
7. **The time estimate doesn't hold on a busy machine, and the doc doesn't say to bench on an idle one.** The doc's "about 20 seconds per set" gave 18 min; it took 42 min (about 47 s per set) with four other benches running. The progress line's "about N min left" swung from 70 min (after set 1) to 20 min (set 6) and then stayed at 13 to 25 min for half an hour. The recall numbers didn't suffer (the re-bench reproduced them exactly), but `seconds_per_page` did: the manifest claims 9.648 s/page for RapidOCR, the quieter re-bench measured 6.497, and Spanish measured 2.25 alone. Speed is `maximin-v1`'s last tie-break and goes into the PR body, so here it chose the runner-up (set 37 over sets 30 to 36, all with the same recall and two engines each) on contended timings. *Fix:* "bench on an otherwise idle machine; speed is a tie-break and is published."

### Step 6: choose

No doc issue. The output matched the doc: `selections`, `dropped_models` (`por.traineddata`), `unbenched` (`win-amd64`, `linux-x86_64`, `linux-aarch64`) and a `next` naming `kit wheel`. RapidOCR runs everywhere, so `default` is the Mac's pick and no per-platform entry was needed.

### Step 8: wheel and verify

8. **`--must-find` takes at most four entries, `4521` included, and the doc doesn't say so.** The first `kit wheel` with five words plus `4521` exited 1: `kit_invalid_manifest`, "`$.canary.must_find`: has 6 items; at most 4". The doc says only "Repeat `--must-find` for each word". *Did instead:* kept three words (`Fotossíntese`, `informações`, `lâmpada`, which carry í, ç, õ, ã and â) plus `4521`; RapidOCR found all four. *Fix:* "at most four, `4521` included".
9. **The re-bench's `claimed` and `measured` have different shapes.** `claimed.metrics` holds lists `[clean, scan, small]` and the must-survive groups as `must-<name>`; `measured.metrics` holds objects keyed by condition and the groups under a separate `must`. Comparing by eye means knowing the list order. Minor. *Fix:* the same shape on both sides, or a line in step 8.

### Step 9: report

10. **The report's "Models and fonts" table is empty.** With `por.traineddata` dropped, the pack pins no model or font of its own, and the body prints the table's header row and nothing else. A reviewer can't tell an empty table from a broken one, and the base-pack models the reader set uses (`en:rapid-det`, `en:rapid-cls`, `en:rapid-rec-latin`) appear nowhere in the body. *Fix:* print "none of its own; uses the base pack's `en:rapid-*`" in that case.
11. **The expected test failure matched the doc.** The whole suite gave 1 failed, 3128 passed, 35 skipped; the failure is `tests/test_release.py::test_index_regenerates_identically` with `pt: version 1.0.0 has no wheel in dist`, as step 9 predicts (it also names `es`, `fr` and `it`, packs other builders added to the same tree).

### What the doc got right this time

- The example Wikidata ids, the font URLs and "two Latin fonts for a Latin pack" worked as written; the mixed-script page came out Greek with no extra font.
- `kit corpus` printed each text's `title`; the `candidates` counts added up (`refused_by_loader` 0, `kept` equal to `count`); `floors_ok` and `below_floor` were where step 5 says; `choose`'s `next` named `kit wheel`; `verify --download` ran all four checks in one second with nothing skipped.

## The bench on `darwin-arm64`

Every set read all 24 pages. Words is word recall per condition (the primary metric for a spaced script); Accents is the must-survive recall (floor 0.9 in every condition). `tesseract (eng)` is the base pack's English model and `tesseract (por)` the Portuguese one (item 5). Seconds per page are inflated by four concurrent benches (item 7).

| Set | Engines | Pick | Words clean | Words scan | Words small | Accents clean | Accents scan | Accents small | Floors | Worst page | s/page |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tesseract (por) + rapidocr | primary_with_floor | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 11.72 |
| 1 | tesseract (eng) | single | 0.8363 | 0.8306 | 0.7360 | 0.181 | 0.183 | 0.162 | below | 0.086 | 1.20 |
| 2 | tesseract (eng) + vision (correction on) | single | 0.8363 | 0.8306 | 0.7360 | 0.181 | 0.183 | 0.162 | below | 0.086 | 1.51 |
| 3 | tesseract (eng) + vision (correction on) | primary_with_floor | 0.8363 | 0.8306 | 0.7360 | 0.181 | 0.183 | 0.162 | below | 0.086 | 1.95 |
| 4 | tesseract (eng) + vision2 (correction off) | single | 0.8363 | 0.8306 | 0.7360 | 0.181 | 0.183 | 0.162 | below | 0.086 | 2.49 |
| 5 | tesseract (eng) + vision2 (correction off) | primary_with_floor | 0.8363 | 0.8306 | 0.7360 | 0.181 | 0.183 | 0.162 | below | 0.086 | 2.65 |
| 6 | tesseract (eng) + rapidocr | single | 0.8363 | 0.8306 | 0.7361 | 0.181 | 0.183 | 0.162 | below | 0.086 | 10.61 |
| 7 | tesseract (eng) + rapidocr | primary_with_floor | 0.8363 | 0.8306 | 0.7361 | 0.181 | 0.183 | 0.162 | below | 0.086 | 6.21 |
| 8 | tesseract2 (por) | single | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 2.91 |
| 9 | tesseract2 (por) + vision (correction on) | single | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 3.37 |
| 10 | tesseract2 (por) + vision (correction on) | primary_with_floor | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 4.13 |
| 11 | tesseract2 (por) + vision2 (correction off) | single | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 4.04 |
| 12 | tesseract2 (por) + vision2 (correction off) | primary_with_floor | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 4.69 |
| 13 | tesseract2 (por) + rapidocr | single | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 11.39 |
| 14 | tesseract2 (por) + rapidocr | primary_with_floor | 0.9809 | 0.9772 | 0.8971 | 1.000 | 0.993 | 0.901 | ok | 0.351 | 9.37 |
| 15 | vision (correction on) | single | 0.9731 | 0.9643 | 0.9232 | 0.969 | 0.966 | 0.890 | below | 0.533 | 0.76 |
| 16 | vision (correction on) + tesseract (eng) | single | 0.9731 | 0.9643 | 0.9232 | 0.969 | 0.966 | 0.890 | below | 0.533 | 1.44 |
| 17 | vision (correction on) + tesseract (eng) | primary_with_floor | 0.9731 | 0.9643 | 0.9232 | 0.969 | 0.966 | 0.890 | below | 0.533 | 1.53 |
| 18 | vision (correction on) + tesseract2 (por) | single | 0.9731 | 0.9644 | 0.9232 | 0.969 | 0.966 | 0.890 | below | 0.533 | 6.54 |
| 19 | vision (correction on) + tesseract2 (por) | primary_with_floor | 0.9731 | 0.9644 | 0.9232 | 0.969 | 0.966 | 0.890 | below | 0.533 | 7.82 |
| 20 | vision (correction on) + rapidocr | single | 0.9731 | 0.9644 | 0.9233 | 0.969 | 0.966 | 0.890 | below | 0.533 | 8.37 |
| 21 | vision (correction on) + rapidocr | primary_with_floor | 0.9731 | 0.9644 | 0.9233 | 0.969 | 0.966 | 0.890 | below | 0.533 | 5.89 |
| 22 | vision2 (correction off) | single | 0.9692 | 0.9623 | 0.9072 | 0.967 | 0.963 | 0.876 | below | 0.413 | 0.56 |
| 23 | vision2 (correction off) + tesseract (eng) | single | 0.9692 | 0.9623 | 0.9072 | 0.967 | 0.963 | 0.876 | below | 0.413 | 1.85 |
| 24 | vision2 (correction off) + tesseract (eng) | primary_with_floor | 0.9692 | 0.9623 | 0.9072 | 0.967 | 0.963 | 0.876 | below | 0.413 | 2.19 |
| 25 | vision2 (correction off) + tesseract2 (por) | single | 0.9692 | 0.9624 | 0.9072 | 0.967 | 0.963 | 0.876 | below | 0.413 | 6.08 |
| 26 | vision2 (correction off) + tesseract2 (por) | primary_with_floor | 0.9692 | 0.9624 | 0.9072 | 0.967 | 0.963 | 0.876 | below | 0.413 | 6.82 |
| 27 | vision2 (correction off) + rapidocr | single | 0.9692 | 0.9624 | 0.9073 | 0.967 | 0.963 | 0.876 | below | 0.414 | 9.35 |
| 28 | vision2 (correction off) + rapidocr | primary_with_floor | 0.9692 | 0.9624 | 0.9073 | 0.967 | 0.963 | 0.876 | below | 0.414 | 9.68 |
| 29 | rapidocr | single | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 9.65 |
| 30 | rapidocr + tesseract (eng) | single | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 11.64 |
| 31 | rapidocr + tesseract (eng) | primary_with_floor | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 8.93 |
| 32 | rapidocr + tesseract2 (por) | single | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 12.70 |
| 33 | rapidocr + tesseract2 (por) | primary_with_floor | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 15.41 |
| 34 | rapidocr + vision (correction on) | single | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 10.11 |
| 35 | rapidocr + vision (correction on) | primary_with_floor | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 7.77 |
| 36 | rapidocr + vision2 (correction off) | single | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 8.75 |
| 37 | rapidocr + vision2 (correction off) | primary_with_floor | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 7.41 |
| 38 | tesseract (eng) + vision (correction on) | corroborated_trigrams | 0.8363 | 0.8306 | 0.7919 | 0.181 | 0.183 | 0.181 | below | 0.533 | 2.19 |
| 39 | tesseract (eng) + vision (correction on) + rapidocr | corroborated_trigrams | 0.9731 | 0.9608 | 0.9233 | 0.969 | 0.892 | 0.890 | below | 0.533 | 10.60 |
| 40 | tesseract (eng) + vision2 (correction off) | corroborated_trigrams | 0.8363 | 0.8306 | 0.7770 | 0.181 | 0.183 | 0.173 | below | 0.413 | 2.22 |
| 41 | tesseract (eng) + vision2 (correction off) + rapidocr | corroborated_trigrams | 0.9692 | 0.9606 | 0.9073 | 0.967 | 0.892 | 0.876 | below | 0.414 | 10.77 |
| 42 | tesseract (eng) + rapidocr | corroborated_trigrams | 0.8363 | 0.8306 | 0.8158 | 0.181 | 0.183 | 0.239 | below | 0.707 | 8.72 |
| 43 | tesseract (eng) + rapidocr + vision (correction on) | corroborated_trigrams | 0.9884 | 0.9830 | 0.9511 | 0.998 | 0.997 | 0.950 | ok | 0.724 | 11.57 |
| 44 | tesseract2 (por) + vision (correction on) | corroborated_trigrams | 0.9733 | 0.9778 | 0.9201 | 0.975 | 0.993 | 0.889 | below | 0.533 | 4.73 |
| 45 | tesseract2 (por) + vision (correction on) + rapidocr | corroborated_trigrams | 0.9811 | 0.9782 | 0.9224 | 0.996 | 0.992 | 0.889 | below | 0.533 | 15.10 |
| 46 | tesseract2 (por) + vision2 (correction off) | corroborated_trigrams | 0.9809 | 0.9778 | 0.9050 | 1.000 | 0.993 | 0.881 | below | 0.413 | 3.76 |
| 47 | tesseract2 (por) + vision2 (correction off) + rapidocr | corroborated_trigrams | 0.9811 | 0.9782 | 0.9070 | 0.996 | 0.992 | 0.879 | below | 0.414 | 10.60 |
| 48 | tesseract2 (por) + rapidocr | corroborated_trigrams | 0.9809 | 0.9790 | 0.9437 | 1.000 | 0.993 | 0.948 | ok | 0.724 | 12.34 |
| 49 | tesseract2 (por) + rapidocr + vision (correction on) | corroborated_trigrams | 0.9832 | 0.9817 | 0.9455 | 0.998 | 0.993 | 0.948 | ok | 0.724 | 11.04 |
| 50 | vision (correction on) + rapidocr | corroborated_trigrams | 0.9741 | 0.9820 | 0.9244 | 0.969 | 0.996 | 0.894 | below | 0.533 | 7.28 |
| 51 | vision (correction on) + rapidocr + tesseract (eng) | corroborated_trigrams | 0.9813 | 0.9818 | 0.9232 | 0.994 | 0.995 | 0.890 | below | 0.533 | 6.73 |
| 52 | vision2 (correction off) + rapidocr | corroborated_trigrams | 0.9818 | 0.9820 | 0.9084 | 0.996 | 0.996 | 0.881 | below | 0.413 | 5.48 |
| 53 | vision2 (correction off) + rapidocr + tesseract (eng) | corroborated_trigrams | 0.9813 | 0.9816 | 0.9072 | 0.994 | 0.995 | 0.876 | below | 0.413 | 6.11 |

## Selection (`kit choose`, policy `maximin-v1`)

- Dropped for a floor: 34 sets. They are the sets that read with Tesseract `eng` (sets 1 to 7, 38, 40, 42: accents about 0.18), the sets led by Vision (15 to 28: Vision keeps 0.890 of the accents on `small` with language correction on, 0.876 with it off), and the corroborated sets with Vision in them (39, 41, 44 to 47, 50 to 53: `small` accents 0.876 to 0.894, and 0.892 on `scan` for 39 and 41).
- Highest worst-condition word recall among the 20 that pass: 0.9511 (`small`), shared by ten sets: RapidOCR alone (29), RapidOCR with any second engine (30 to 37), and Tesseract `eng` + RapidOCR + Vision corroborated (43).
- Tie-break: fewer engines, so set 29, RapidOCR alone. Runner-up set 37, margin 0.000.
- Tesseract `por` alone (set 8) is the best set that doesn't use RapidOCR: 0.8971 on `small`, accents 0.901, just over the floor.

## `kit wheel` and `kit verify`

- `kit wheel pt`: wheel `pdfintake_lang_pt-1.0.0-py3-none-any.whl`, 26,453 bytes; canary `canary-pt.png`, 22,537 bytes at 300 dpi, `must_find` `Fotossíntese`, `informações`, `lâmpada`, `4521`; RapidOCR found all four.
- `kit verify pt --download`: exit 0; `manifest`, `download`, `layer` and `wheel` all `ok`; `warnings` empty.
- `kit verify pt --download --rebench`: exit 0 in 47 s; tolerance 0.01; measured (clean, scan, small) words 0.9884, 0.9830, 0.9511; trigrams 0.9993, 0.9987, 0.9690; digits 0.900, 1.000, 1.000; accents 0.9984, 0.9973, 0.9504, worst page 0.7235: identical to the claim. `drift` and `floors` empty. Seconds per page 6.497 against the claimed 9.648 (item 7; speed isn't compared).

## Tests

The whole suite, once, for information, with the test venv: **1 failed, 3128 passed, 35 skipped** in 211 s. The one failure is the expected `tests/test_release.py::test_index_regenerates_identically` (item 11). The skips, with their reasons:

- 20 in `tests/e2e/test_run_e2e.py` and 1 in `tests/test_vision.py`: the English data package `pdfintake-data-en` isn't installed in the test venv (`base_pack_missing`).
- 4 in `tests/e2e/test_sandbox_smoke.py`: run only on a non-editable install (the CI sandbox job).
- 1 in `tests/e2e/test_packs_network.py`: needs `PDFINTAKE_NETWORK_TESTS=1` (the CI network job).
- 1 in `tests/test_release.py`: the pyproject pin is still the placeholder.
- 5 in `tests/contract` and 3 in `tests/test_record.py`: parametrised cases that don't apply (no model files for Vision, rules without corroborators or tie order, valid or null values).

## `kit report pt` body

### Pack `pt` 1.0.0: Portuguese

- Languages: pt (primary)
- Scripts: Latin
- Requires core: `>=0.1.0`

**Reader sets**

- `default`: rapidocr (rapidocr, reader); pick `single` v1, fallback `none`

**Models and fonts**

| File | Kind | Bytes | Licence | Source |
|---|---|---|---|---|

- Canary: `canary-pt.png`, 22537 bytes
- Must survive: `accents` (word, floor 0.9)

**Corpus**

- `public-v1`: 8 texts, 2 fonts, pages {"form": 2, "microprint": 1, "mixed_script": 1, "prose": 4}
  - Água: wikipedia `pt` revision 72968719, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=72968719&action=raw
  - Sol: wikipedia `pt` revision 73197256, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73197256&action=raw
  - Lua: wikipedia `pt` revision 73289764, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73289764&action=raw
  - Fotossíntese: wikipedia `pt` revision 73032270, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73032270&action=raw
  - Xadrez: wikipedia `pt` revision 73370098, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73370098&action=raw
  - Pão: wikipedia `pt` revision 73084145, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73084145&action=raw
  - Vulcão: wikipedia `pt` revision 73354359, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73354359&action=raw
  - Rio: wikipedia `pt` revision 73342842, CC-BY-SA-4.0: https://pt.wikipedia.org/w/index.php?oldid=73342842&action=raw

**Bench**

- Kit 1.0, corpus `public-v1@2bbd2d2fc5bcdbe88bb2fa2a6675d2a6f632fc38a1a5e78e3b363f46c7f04f64`
- Engines: rapidocr 3.10.0, tesseract 5.5.3 (tesserocr 2.12.0), vision rev3 (macOS 27.2)

`darwin-arm64`: worst page 0.724, 9.648 s/page, policy `maximin-v1`, margin 0.000

| Metric | clean | scan | small |
|---|---|---|---|
| digits | 0.900 | 1.000 | 1.000 |
| must-accents | 0.998 | 0.997 | 0.950 |
| trigrams | 0.999 | 0.999 | 0.969 |
| words | 0.988 | 0.983 | 0.951 |

**Checklist**

- [ ] The PR changes only `packs/pt/**`.
- [ ] No binaries: no model, font, image or PDF.
- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.
- [ ] `pdfintake kit verify pt --download` passes.
- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.
- [ ] The must-survive characters were confirmed by someone who reads the language.

