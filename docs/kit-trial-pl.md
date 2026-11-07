# Kit trial: Polish (`pl`)

A kit trial after the Spanish one. A contributor's Claude, who wrote none of the kit, added the Polish pack by following `CLAUDE.md` and `docs/add-a-language.md` step by step, without reading the kit's source, its tests or any design notes. Every command, its exit code, and every place the doc was wrong, unclear or missing something is below, with what was done instead.

- Machine: macOS 27.2 on Apple silicon (`darwin-arm64`), Apple Vision available (revision 3, `pl-PL` among its languages).
- Engines measured: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Step 1 was not run here: the trial used a shared, already set-up clone with its own `.venv`. Every kit command ran as `<clone>/.venv/bin/pdfintake kit <cmd> pl --packs packs --work <work> --json`, where `<work>` is a kit work folder outside the repo.
- The must-survive characters were given by the person: `ogonek-kreska=ąćęłńóśźżĄĆĘŁŃÓŚŹŻ:word:0.9`. Every corpus text is Wikipedia, CC-BY-SA-4.0, on a neutral topic.

## Result: stopped at `kit choose`, exit 6

*Since the trial:* the kit now counts a must-survive character only in words an engine actually read, and its policy is `maximin-v2`. On a fresh bench under those rules `kit choose pl` passes with Tesseract `pol` + a RapidOCR checker, pick `single` (words 0.9751 clean, 0.9696 scan, 0.8630 small; `ogonek-kreska` 1.0, 1.0, 0.9966), and Polish ships. What follows is the trial as it ran.

`kit choose pl` exited 6 (`kit_floor`): **no reader set keeps the `ogonek-kreska` characters at or above 0.9 in every condition.** As the doc says (step 6), that goes to the person, and an override is stop point 2, so the trial stopped there. Steps 7 to 9 (`kit wheel`, `kit verify`, `kit report`) were not run, and `packs/pl/` has no `manifest.json`.

The numbers the person needs:

- **Closest sets.** The best worst-condition must-survive recall is 0.8836: set 44, Tesseract `pol` + Vision with `corroborated_trigrams` (must-survive 0.9965 clean, 0.9641 scan, 0.8836 small; words 0.9783 clean, 0.9652 scan and 0.8878 small). Next, set 49, Tesseract `pol` + RapidOCR + Vision with `corroborated_trigrams` (must-survive 0.9991 clean, 0.8990 scan and 0.8833 small; words 0.9798 clean, 0.9389 scan and 0.9111 small). Then Tesseract `pol` alone, set 8 (must-survive 0.9991 clean, 0.9913 scan and 0.8817 small; words 0.9798 clean, 0.9718 scan and 0.8666 small; 2.67 s per page), and every set with Tesseract `pol` as the primary reads the same.
- **One page decides it.** For Tesseract `pol` (set 8), every page is at or above 0.9 except `microprint-0-small`: words 0.116, must-survive 0.072. The same microprint page reads 1.000 at 300 dpi and 0.930 at 200 dpi. The 110 dpi microprint page alone pulls the `small` condition from above the floor to 0.8817. Set 44 misses on that page (0.117) and on `form-1-scan` (0.800).
- **Vision alone** (set 15): 0.9258 clean, 0.9393 scan, 0.8443 small, with misses on both form pages and the microprint page.
- **RapidOCR alone** (set 29, the Latin recogniser from the base pack): 0.5762 clean, 0.4274 scan, 0.5048 small, below the floor on all 24 pages. Every set where RapidOCR is the primary reads identically to it.
- **Tesseract with the English model** (sets 1 to 7, 38, 40 and 42): 0.000 in every condition, as expected for a model without these letters.

What the person can decide (stop point 2 or a code change, per step 6): accept a floor below 0.9 for `small` or for this language, override with `--override darwin-arm64=<index> --override-reason "<why>"`, or treat it as an engine gap (RapidOCR's Latin recogniser reads these letters at about 0.5). This trial makes none of those choices.

## Steps

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | `pdfintake doctor --json` (in the shared clone; `dev-install.sh` not re-run) | **3** | 2 s |
| 2 | `pdfintake kit init pl --script Latin --must-survive 'ogonek-kreska=ąćęłńóśźżĄĆĘŁŃÓŚŹŻ:word:0.9'` | 0 | 2 s |
| 3 | `pdfintake kit corpus pl --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072,Q4022 --font <NotoSans-Regular> --font <NotoSerif-Regular>` | 0 | 12 s |
| 4 | `pdfintake kit render pl` | 0 | 7 s |
| 5 | `pdfintake kit candidates pl` | 0 | under 1 s |
| 5 | `pdfintake kit bench pl` (background, `nohup`) | 0 | **40 min 26 s** |
| 6 | `pdfintake kit choose pl` | **6** | 1 s |
| 7 | `kit private` | not run | optional; no contributor PDFs, and the trial had stopped |
| 8 | `kit wheel`, `kit verify --download`, `kit verify --download --rebench` | not run | stopped at step 6 |
| 9 | `kit report pl` | not run | stopped at step 6 |

The fonts are `NotoSans-Regular.ttf` and `NotoSerif-Regular.ttf`, the two URLs given in step 3 of the doc.

The corpus is eight Polish Wikipedia articles, each title checked in the `pinned` list: Woda (Q283, revision 80265543), Słońce (Q525, 80610142), Księżyc (Q405, 81124335), Fotosynteza (Q11982, 81118481), Szachy (Q718, 80808204), Chleb (Q7802, 81126828), Wulkan (Q8072, 77520754) and Rzeka (Q4022, 80625384).

`kit render` drew 24 pages: 4 prose, 2 form, 1 microprint and 1 mixed-script page, each in the three conditions. The mixed-script page's second script is Greek, as the doc says. `kit candidates` listed 54 sets (`total` 54, `refused_by_loader` 0, `cut` 0, `kept` 54); `kit bench` read all 24 pages with each, and every set was `"ok": true`, `"floors_ok": false`.

## Where the doc was wrong, unclear or missing something

Each item gives the step, what happened, and what was done instead. "Doc" means `docs/add-a-language.md`.

### Step 1: set up

1. **`doctor` exited 3 with a problem the doc routes to `dev-install.sh`, which this trial could not re-run.** `base_pack` reported `hash_mismatch`: the installed English data wheel's `manifest.json` has a different SHA-256 from the one the pack index pins. That isn't `index_missing`, so the doc's route is "run `scripts/dev-install.sh` again; if it stays, stop point 4". The trial ran in a shared clone whose environment it must not change. The cause is explainable (the clone's English data wheel was built before `packs/en/manifest.json` last changed), so it isn't an unexplained failure. *Did instead:* went on to step 2, since the doc says the kit takes the base pack from `packs/en`; every kit command after it ran without a setup error (no exit 3). *Fix:* say in step 1 that a `base_pack` `hash_mismatch` after pulling new commits means the English data wheel is stale and `dev-install.sh` rebuilds it, and whether the kit is affected at all.

### Step 2: start the draft

2. **`next` is empty although the corpus is empty.** `kit init` printed `"next": []` while `corpus.json` held no texts and no fonts. The doc says "`next` lists anything still missing". *Fix:* have `next` name `kit corpus` after `init`, or say `next` only covers the draft itself.
3. **`kit init` also writes `NOTICE-tessdata`.** The doc says `init` writes the draft and an empty `corpus.json`; the folder also got `NOTICE-tessdata` (10,838 bytes). Harmless, and step 9 explains notices, but step 2 doesn't mention it. *Fix:* one clause in step 2.

### Step 5: candidates and bench

4. **The same use id means two different readers.** In set 0, `tesseract` reads with `pol` (the draft's own ids). In sets 1 to 7, 38 to 43 and 51/53, `tesseract` reads with `eng`, and `tesseract2` is `pol`. The doc does say set 0 keeps the draft's own ids, but the progress lines (`set 1/54 (tesseract + rapidocr, ...)`, `set 2/54 (tesseract, single)`) and the `kit choose` error show only the ids, so a reader can't tell a Polish Tesseract from an English one without opening the candidates file. *Fix (kit):* give set 0 the same numbering as the others, or show the model (`tesseract:pol`) in progress lines and messages.
5. **The progress lines count from 1, the JSON from 0.** `kit bench: set 1/54` is the set with `"index": 0`. *Fix:* use the index in the progress line, or say so in the doc.
6. **The bench took twice the doc's upper estimate.** The doc says 30 to 60 sets take 10 to 20 minutes. 54 sets took 40 min 26 s here, with the machine's load average between 60 and 99 from other work, and some sets at 90 to 105 s each. The first progress line said "about 70 min left" (set 0 included the model download), and the estimate then sat at 13 to 17 minutes for over ten minutes. Running it in the background as the doc says was the right call. *Fix:* say the estimate assumes an idle machine, and that the first set includes the model download.
7. **Sets with Tesseract's English model are benched for a Polish pack.** Ten of the 54 sets read Polish with `eng` only and score 0.000 on the must-survive characters. They cost about 8 minutes of the bench and, through item 8, fill the `kit choose` error. Not a doc error; noted for the kit.

### Step 6: choose

8. **The exit-6 message points away from the real gap.** `kit choose` listed eight problems, all `clean` or `scan`, the lowest values first: `ogonek-kreska clean 0.000 < 0.9`, `clean 0.576`, `clean 0.644`, `clean 0.645`, `clean 0.667`, `scan 0.000`, `scan 0.024`, `scan 0.336`. None of them is `small`, and none names a set. Yet the sets that come closest fail only in `small` (0.8836, 0.8833, 0.8817). A person reading the error alone would think Polish OCR fails on clean pages. The doc says "report the numbers to the person", but the only place the closest sets show is each set's `below_floor` in `kit bench`'s output, or `packs/pl/bench.json`. *Did instead:* read the `below_floor` lists and the per-page numbers in `packs/pl/bench.json` to build the summary above. *Fix (kit):* on exit 6, report the closest set(s) per platform with their index, reader set and per-condition misses.
9. **No route for "the floor is too strict".** Step 6 says the floors may be too strict, but no step says how to change a floor once the corpus is pinned and benched: whether `kit init --force` with a new `--must-survive` keeps `corpus.json` and `bench.json`, and whether a new floor needs a re-bench or only a new `kit choose`. *Fix:* one line in step 6 on how the person's new floor is applied, and what has to run again.
10. **How a condition's must-survive recall is aggregated isn't said.** In set 8, 23 of 24 pages are at or above 0.9 and one 110 dpi microprint page is at 0.072, and the `small` condition comes out at 0.8817. So the condition figure pools the pages (or weights them), and one page can decide the pack. The doc doesn't say how the per-condition number is computed, which matters when the person decides whether a floor is too strict. *Fix:* one sentence in step 5 or 6.

### Not a doc issue, noted for the kit

- **RapidOCR's Latin recogniser reads Polish letters poorly.** RapidOCR alone keeps 0.43 to 0.58 of the `ogonek-kreska` words per condition, below the floor on every page, including clean prose. On Spanish the same recogniser kept 0.93 to 1.00 of the accented words. Every set where RapidOCR is the primary reads identically to RapidOCR alone, and it drags `corroborated_trigrams` sets down where it takes part.
- **Lost numbers in the corpus text.** The mixed-script page's truth reads "istniało ponad piekarń", with the number before "piekarń" missing (probably a template dropped when the wikitext is flattened). It is missing on the page and in the truth alike, so recall isn't affected. The Spanish trial saw the same with spaces. *Added after the trial:* the kit's page-truth tokeniser (`render._words`) lost this number, not the corpus; fixed in the next kit change.
- **Digit recall is 0.8667 in every condition for every Tesseract- and Vision-led set**, and 1.000 for RapidOCR. The prose pages carry very few digits (`prose-1` has only `Chandrayaan-1,`), so one missed token moves the figure a lot.

## The bench on `darwin-arm64`

Word recall and must-survive (`ogonek-kreska`) recall per condition. The floor is 0.9 in every condition; a set below it in any condition is dropped. `tesseract` is `pol` in set 0 and `eng` everywhere else; `tesseract2` is `pol` (item 4). `vision2` is Vision without language correction. Seconds per page were measured on a heavily loaded machine (item 6).

| # | Reader set | Pick | Words clean | Words scan | Words small | Ogonek-kreska clean | Ogonek-kreska scan | Ogonek-kreska small | Floors | s/page |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tesseract + rapidocr | primary_with_floor | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 11.47 |
| 1 | tesseract | single | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 1.90 |
| 2 | tesseract + vision | single | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 1.70 |
| 3 | tesseract + vision | primary_with_floor | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 1.62 |
| 4 | tesseract + vision2 | single | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 1.96 |
| 5 | tesseract + vision2 | primary_with_floor | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 2.64 |
| 6 | tesseract + rapidocr | single | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 10.22 |
| 7 | tesseract + rapidocr | primary_with_floor | 0.6758 | 0.6664 | 0.5896 | 0.0000 | 0.0000 | 0.0000 | below | 8.19 |
| 8 | tesseract2 | single | 0.9798 | 0.9718 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 2.67 |
| 9 | tesseract2 + vision | single | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 2.60 |
| 10 | tesseract2 + vision | primary_with_floor | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 3.26 |
| 11 | tesseract2 + vision2 | single | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 3.45 |
| 12 | tesseract2 + vision2 | primary_with_floor | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 3.53 |
| 13 | tesseract2 + rapidocr | single | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 7.24 |
| 14 | tesseract2 + rapidocr | primary_with_floor | 0.9798 | 0.9734 | 0.8666 | 0.9991 | 0.9913 | 0.8817 | below | 11.36 |
| 15 | vision | single | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 0.90 |
| 16 | vision + tesseract | single | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 3.12 |
| 17 | vision + tesseract | primary_with_floor | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 1.94 |
| 18 | vision + tesseract2 | single | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 2.79 |
| 19 | vision + tesseract2 | primary_with_floor | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 2.72 |
| 20 | vision + rapidocr | single | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 6.45 |
| 21 | vision + rapidocr | primary_with_floor | 0.9591 | 0.9528 | 0.8807 | 0.9258 | 0.9393 | 0.8443 | below | 7.20 |
| 22 | vision2 | single | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 0.59 |
| 23 | vision2 + tesseract | single | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 3.38 |
| 24 | vision2 + tesseract | primary_with_floor | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 3.81 |
| 25 | vision2 + tesseract2 | single | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 6.53 |
| 26 | vision2 + tesseract2 | primary_with_floor | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 3.52 |
| 27 | vision2 + rapidocr | single | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 5.17 |
| 28 | vision2 + rapidocr | primary_with_floor | 0.9577 | 0.9524 | 0.8687 | 0.9228 | 0.9317 | 0.8322 | below | 5.52 |
| 29 | rapidocr | single | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 7.18 |
| 30 | rapidocr + tesseract | single | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 13.57 |
| 31 | rapidocr + tesseract | primary_with_floor | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 14.02 |
| 32 | rapidocr + tesseract2 | single | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 15.64 |
| 33 | rapidocr + tesseract2 | primary_with_floor | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 12.63 |
| 34 | rapidocr + vision | single | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 6.23 |
| 35 | rapidocr + vision | primary_with_floor | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 8.09 |
| 36 | rapidocr + vision2 | single | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 6.96 |
| 37 | rapidocr + vision2 | primary_with_floor | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 10.21 |
| 38 | tesseract + vision | corroborated_trigrams | 0.6758 | 0.6664 | 0.6217 | 0.0000 | 0.0000 | 0.0147 | below | 3.69 |
| 39 | tesseract + vision + rapidocr | corroborated_trigrams | 0.9591 | 0.7567 | 0.8807 | 0.9258 | 0.3431 | 0.8443 | below | 12.16 |
| 40 | tesseract + vision2 | corroborated_trigrams | 0.6758 | 0.6664 | 0.6127 | 0.0000 | 0.0000 | 0.0109 | below | 3.55 |
| 41 | tesseract + vision2 + rapidocr | corroborated_trigrams | 0.9577 | 0.7567 | 0.8687 | 0.9228 | 0.3361 | 0.8322 | below | 8.19 |
| 42 | tesseract + rapidocr | corroborated_trigrams | 0.6758 | 0.6816 | 0.6440 | 0.0000 | 0.0240 | 0.0106 | below | 8.91 |
| 43 | tesseract + rapidocr + vision | corroborated_trigrams | 0.8544 | 0.8025 | 0.8079 | 0.5762 | 0.4274 | 0.5048 | below | 11.30 |
| 44 | tesseract2 + vision | corroborated_trigrams | 0.9783 | 0.9652 | 0.8878 | 0.9965 | 0.9641 | 0.8836 | below | 6.21 |
| 45 | tesseract2 + vision + rapidocr | corroborated_trigrams | 0.9798 | 0.9718 | 0.8800 | 0.9991 | 0.9822 | 0.8489 | below | 14.32 |
| 46 | tesseract2 + vision2 | corroborated_trigrams | 0.9786 | 0.9726 | 0.8768 | 0.9965 | 0.9913 | 0.8715 | below | 3.64 |
| 47 | tesseract2 + vision2 + rapidocr | corroborated_trigrams | 0.9798 | 0.9710 | 0.8690 | 0.9991 | 0.9774 | 0.8368 | below | 14.25 |
| 48 | tesseract2 + rapidocr | corroborated_trigrams | 0.8841 | 0.8136 | 0.8467 | 0.6674 | 0.4857 | 0.6546 | below | 6.17 |
| 49 | tesseract2 + rapidocr + vision | corroborated_trigrams | 0.9798 | 0.9389 | 0.9111 | 0.9991 | 0.8990 | 0.8833 | below | 9.64 |
| 50 | vision + rapidocr | corroborated_trigrams | 0.8782 | 0.8073 | 0.8457 | 0.6451 | 0.4607 | 0.6503 | below | 7.05 |
| 51 | vision + rapidocr + tesseract | corroborated_trigrams | 0.8544 | 0.8025 | 0.8192 | 0.5762 | 0.4274 | 0.5465 | below | 6.64 |
| 52 | vision2 + rapidocr | corroborated_trigrams | 0.8774 | 0.8089 | 0.8325 | 0.6444 | 0.4607 | 0.6003 | below | 6.09 |
| 53 | vision2 + rapidocr + tesseract | corroborated_trigrams | 0.8544 | 0.8025 | 0.8192 | 0.5762 | 0.4274 | 0.5465 | below | 9.13 |

The microprint page per condition, word recall and must-survive recall:

| Set | Words clean | Words scan | Words small | Ogonek-kreska clean | Ogonek-kreska scan | Ogonek-kreska small |
|---|---|---|---|---|---|---|
| 8, Tesseract `pol` | 0.998 | 0.947 | 0.116 | 1.000 | 0.930 | 0.072 |
| 29, RapidOCR | 0.810 | 0.670 | 0.472 | 0.537 | 0.192 | 0.085 |

## `kit choose pl`

Exit 6.

```json
{"ok": false, "command": "kit choose", "error": {"code": "kit_floor", "message": "no reader set on darwin-arm64 keeps every must-survive unit above its floor: ogonek-kreska clean 0.000 < 0.9; ogonek-kreska clean 0.576 < 0.9; ogonek-kreska clean 0.644 < 0.9; ogonek-kreska clean 0.645 < 0.9; ogonek-kreska clean 0.667 < 0.9; ogonek-kreska scan 0.000 < 0.9; ogonek-kreska scan 0.024 < 0.9; ogonek-kreska scan 0.336 < 0.9"}}
```

(`problems` repeats the eight items of `message`.)

## Tests

The whole suite, run once at the end in the work tree, which also holds other packs being built at the same time: 3124 passed, 36 skipped, 1 failed, in 4 min 40 s. The failure is `tests/test_release.py::test_index_regenerates_identically`, the one step 9 expects: `build_index.py --check` reports that `es`, `fr`, `it` and `pt` have no wheel in `dist`. `pl` isn't among them, because it has no `manifest.json`. The skips: 12 because the test environment has no English data package installed, 4 sandbox smoke tests that run only on a non-editable install, 1 network test that runs only with `PDFINTAKE_NETWORK_TESTS=1`, 1 `th` test that needs the English data package, the placeholder pyproject pin, and parameter cases that don't apply to a rule or engine.

## `kit report pl` body

Not produced: the trial stopped at step 6, so `kit wheel`, `kit verify` and `kit report` were not run. `packs/pl/` holds `manifest.draft.json`, `corpus.json`, `bench.json` and `NOTICE-tessdata`, and no `manifest.json`.
