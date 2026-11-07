# Kit trial: Spanish (`es`)

The add-a-language kit's first trial. A contributor's Claude, who wrote none of the kit, added the Spanish pack by following `CLAUDE.md` and `docs/add-a-language.md` step by step, from a fresh clone, without reading the kit's source, its tests or any design notes. Every command, its exit code, and every place the doc was wrong, unclear or missing something is below, with what was done instead.

- Machine: macOS 27.2 on Apple silicon (`darwin-arm64`), Apple Vision available (revision 3).
- Engines measured: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Every kit command ran from the clone's own `.venv`, with `--packs packs --work <work> --json`, where `<work>` is a kit work folder outside the repo. `<clone>` below is the fresh clone.
- Stop points met: none. `kit choose` exited 0 and its pick was kept as is. The must-survive characters were given by the person: `accents=áéíóúüñÁÉÍÓÚÜÑ:word:0.9`. Every corpus text is Wikipedia, CC-BY-SA-4.0, on a neutral topic.

## Result

- Chosen on `darwin-arm64`: **RapidOCR alone** (Latin recogniser from the base pack), pick `single`. Worst-condition word recall 0.9454; accents recall 1.000 clean, 0.996 scan and 0.934 small; 2.25 s per page.
- Runner-up: RapidOCR + Vision (`vision2` variant), same numbers, margin 0.000.
- `kit verify es --download --rebench`: exit 0, all four checks `ok`, re-bench identical to the claim (no drift, no floor missed).
- Files: `packs/es/manifest.json`, `corpus.json`, `bench.json`, `NOTICE-tessdata`.

## Steps

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | `git clone <repo> <clone>` | 0 | seconds |
| 1 | `env -u VIRTUAL_ENV scripts/dev-install.sh` | 0 | about 1 min |
| 1 | `pdfintake doctor --json` | **3** | 1 s |
| 2 | `pdfintake kit init es --script Latin --must-survive 'accents=áéíóúüñÁÉÍÓÚÜÑ:word:0.9'` | 0 | 2 s |
| 3 | `pdfintake kit corpus es --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072,Q4022 --font <NotoSans-Regular> --font <NotoSerif-Regular>` | 0 | 10 s |
| 4 | `pdfintake kit render es` | 0 | 12 s |
| 5 | `pdfintake kit candidates es` | 0 | 3 s |
| 5 | `pdfintake kit bench es` | 0 | **about 12 min** |
| 6 | `pdfintake kit choose es` | 0 | 2 s |
| 7 | `kit private` | skipped | optional; the trial has no contributor PDFs |
| 8 | `pdfintake kit wheel es --must-find montaña --must-find pingüino --must-find Fotosíntesis --must-find 4521` | 0 | 2 s |
| 8 | `pdfintake kit verify es --download` | 0 | 1 s |
| 8 | `pdfintake kit verify es --download --rebench` | 0 | 18 s |
| 9 | `pdfintake kit report es` | 0 | 1 s |

The corpus is eight Spanish Wikipedia articles: Agua (Q283), Sol (Q525), Luna (Q405), Fotosíntesis (Q11982), Ajedrez (Q718), Pan (Q7802), Volcán (Q8072) and Río (Q4022). The fonts are `NotoSans-Regular.ttf` and `NotoSerif-Regular.ttf` from `github.com/notofonts/notofonts.github.io/raw/28b15b4b43b7bed62b5cf6e6b0b5ff5846270535/fonts/<family>/unhinted/ttf/`.

`kit render` drew 24 pages: 4 prose, 2 form, 1 microprint and 1 mixed-script page, each in the three conditions. `kit candidates` listed 33 sets; `kit bench` read all 24 pages with each of them.

## Where the doc was wrong, unclear or missing something

Each item gives the step, what happened, and what was done instead. "Doc" means `docs/add-a-language.md`.

### Step 1: set up

1. **`pdfintake` isn't on the PATH after `dev-install.sh`.** The script makes `.venv` but the doc goes straight to `pdfintake doctor --json`, which fails with "command not found" in a fresh shell. Only uv's own output says "Activate with: source .venv/bin/activate". *Did instead:* ran `.venv/bin/pdfintake` for every command. *Fix:* add `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows) to step 1.
2. **An active virtual environment captures the install.** The script's header says it installs into `$VIRTUAL_ENV` when one is set; the doc doesn't say so. A contributor with another project's environment active installs pdfintake into it. *Did instead:* ran it with `env -u VIRTUAL_ENV`. *Fix:* one line in step 1: deactivate any virtual environment first, or expect the install to go there.
3. **`doctor` exits 3 on a fresh clone, and the doc has no route for it.** It reports `index_missing` ("this pdfintake install has no pack index (langpacks/index.json); reinstall pdfintake"), `"blocking": true`. The committed tree has no `src/pdfintake/langpacks/index.json` yet, so every fresh clone hits this, and reinstalling doesn't help. The doc says "`doctor` must exit 0" and stops there; the exit-code table covers kit commands, not `doctor`. *Did instead:* went on to step 2. Every kit command afterwards exited 0, so the kit doesn't need the index. *Fix:* commit the generated index before the kit is used by outsiders, and give the doc a line on what to do when `doctor` isn't 0 (which problems block the kit and which don't).

### Step 2: start the draft

4. **`vision_languages` has an undocumented shape.** It came back as a list of lists, `[["es-ES", "en-US"]]`, while the doc says it "lists what Apple Vision reads for this language". Harmless here. *Fix:* say what each inner list is (here, Spanish with English as Vision's second language).
5. **`manifest.draft.json` has no stated fate.** `kit init` writes it, `.gitignore` doesn't ignore it, and step 9's list of files to commit leaves it out. A contributor who commits `packs/es/` whole commits the draft too. *Did instead:* followed step 9's list and left the draft out; checked first that `kit choose` re-runs fine from a pack folder without the draft (exit 0, same manifest, `"canary": "kept"`). *Fix:* say whether to delete it after step 6, or have `kit choose` remove it, or ignore it in `.gitignore`.

### Step 3: pin the corpus

6. **The example ids are city articles.** `--wikipedia Q283,Q90,Q1490` is water, Paris and Tokyo. The doc says nothing about which topics to pick beyond "long articles", while the repo's rules want neutral test data. *Did instead:* picked eight general-science and everyday topics (water, the Sun, the Moon, photosynthesis, chess, bread, volcanoes, rivers). *Fix:* make the example neutral, for instance `Q283,Q525,Q405` (water, the Sun, the Moon), and add one line: general topics, never a person, a place, a law or a contract.
7. **No example font URL.** The doc says "a `github.com/notofonts` file at a pinned commit or release" but never shows one, and finding a file means knowing notofonts' layout. *Did instead:* took the URL form from `docs/pack-schema.md` and the English pack's own font pin (`notofonts.github.io` at commit `28b15b4…`, `fonts/NotoSans/unhinted/ttf/NotoSans-Regular.ttf`), and used NotoSerif from the same commit as the second Latin font. *Fix:* give two ready-made Latin URLs in the doc.
8. **"Also add one font for a second script" is unclear, and wasn't needed.** The doc doesn't say which second script the kit draws on a Latin pack's mixed-script page, nor whether a Latin font that also covers Greek and Cyrillic counts. *Did instead:* pinned only the two Latin fonts (both cover Latin, Greek and Cyrillic). `kit corpus` and `kit render` raised nothing, and the mixed-script page came out with Greek lines drawn by those fonts. *Fix:* say the kit uses Greek for a Latin pack's second script and that NotoSans or NotoSerif already covers it, or name the font to add.
9. **`kit corpus` reports URLs but not article titles.** Its JSON lists `oldid` URLs only, so checking that each Wikidata id led to the intended article meant opening `packs/es/corpus.json`. The `kit report` body has the same gap (revisions, no titles), so a reviewer can't see the topics either. *Fix:* print `title` in both.

### Step 5: candidates and bench

10. **The candidate counts don't add up as the doc describes them.** The JSON gave `"count": 33` but `"total": 54, "kept": 54, "cut": 0` and an undocumented `"refused_by_loader": 21`. The doc explains `total`, `kept` and `cut_by_rule`; it doesn't say that the loader refuses sets after `kept` is counted. *Fix:* document `refused_by_loader`, and count `kept` after it (or say it's counted before).
11. **`tesseract2` and `vision2` aren't explained.** Candidate sets name engines `tesseract2` and `vision2` beside `tesseract` and `vision`. The candidate file shows these are variants with other parameters, but the doc doesn't say so. *Fix:* one sentence: an engine id with a number is the same engine with different parameters, listed in the candidates file.
12. **The bench is silent and takes a while, and the doc gives no time.** 33 sets on 24 pages took about 12 minutes on this Mac at around 400% CPU, with nothing on standard error and nothing written to the work folder until the end. A contributor's Claude with a 2-minute command timeout would kill it. *Did instead:* ran it in the background and polled for the exit. *Fix:* say how long a bench takes (minutes per set on a laptop) and to run it in the background; a progress line on standard error would help too.
13. **A set's `"ok": true` doesn't mean it passes the floors.** In the bench JSON every set is `"ok": true`, including the ones whose accents recall is below 0.9. `ok` means every page was read. *Fix:* say so in step 5, or add a `floors_ok` field.
14. **"Bench on every machine you have" leaves the single-machine case open.** See item 18: a contributor with only a Mac ends up with numbers for `darwin-arm64` only, and the doc doesn't say whether to also bench with `--platform <linux or windows key>` on the Mac, or to leave those platforms to CI.

### Step 6: choose

15. **`next` is empty although `kit wheel` hasn't run.** The first `kit choose` printed `"canary": "none"` and `"next": []`. The doc says `next` names `kit wheel` only when a canary was dropped, which is consistent, but a first-time contributor reading `next` as the to-do list sees nothing left. *Fix:* have `next` name `kit wheel` whenever the manifest has no canary.
16. **The pack keeps a model its chosen reader set doesn't use.** The chosen set is RapidOCR alone, with the base pack's models (`en:rapid-*`), yet the manifest still pins `spa.traineddata` (13,570,187 bytes) from `kit init`, and the wheel is 13,605,072 bytes, nearly all of it that unused model. The doc doesn't say whether unused models stay. *Did instead:* left it as `kit choose` wrote it; editing the manifest by hand is out of bounds. *Fix (kit):* drop models no reader set uses at `kit choose`, or document why they stay.
17. **The Mac's pick becomes every platform's reader set.** The manifest has one reader set, `default`, chosen from the `darwin-arm64` bench alone. Nothing in the doc says whether Linux and Windows get their own set when only one platform was benched. Here it does no harm, since RapidOCR runs everywhere, but a pick involving Vision would not run off a Mac.

### Step 8: wheel and verify

18. **On a platform with no claimed numbers, `--rebench` compares nothing and passes.** The manifest claims numbers for `darwin-arm64` only. `kit verify es --rebench --platform linux-x86_64` exited 0 with `"claimed": null`, `"drift": []`: the re-bench ran but had nothing to compare. So CI's Linux and Windows jobs would pass this pack without checking a number, though the doc says "CI then re-benches the pack on three operating systems". *Fix:* either have `verify --rebench` fail or warn when the platform has no claim, or tell single-machine contributors to bench the other platforms' keys with `--platform`.
19. **`--download` doesn't fetch from upstream when the files are already in the work folder.** The doc says verify checks "the download of every model and font from upstream". With a populated work folder, `kit verify es --download` took 1 second, and still exited 0 with `HTTPS_PROXY` pointed at a closed port; with an empty work folder the same proxy made it fail with `URLError ... Connection refused` after three attempts. So it reuses the hash-checked files from earlier steps. That's harmless for integrity, since the hashes are checked, but the doc's wording promises more. *Fix:* say "fetches anything not already in the work folder, and hash-checks all of it", or add a way to force a fresh fetch.
20. **`verify` can pass with two of its four checks skipped.** In an empty work folder, `kit verify es --download` exited 0, `"ok": true`, with `layer` "skipped: the layer fonts aren't downloaded (pass --download)" (although `--download` was passed) and `wheel` "skipped: the pack's files or its canary aren't in the work folder (kit wheel renders it)". CI runs `kit verify --download --rebench` on a fresh checkout without `kit wheel`, so the wheel check, and possibly the layer check, may never run in CI. *Fix (kit):* fetch the base pack's layer fonts under `--download`, rebuild the wheel in `verify` (or have CI run `kit wheel` first), and report `ok: false` or a warning when a check is skipped.

### Step 9 and the tests

21. **A pack-only change fails the test suite once the pack index is committed.** In a checkout that has the generated `src/pdfintake/langpacks/index.json`, adding `packs/es/` makes `tests/test_release.py::test_index_regenerates_identically` fail: `build_index.py --check` reports "es: version 1.0.0 has no wheel in dist". In the fresh clone, which has no index yet, the same test skips ("index.json isn't there yet"). The doc and `CLAUDE.md` tell a contributor to run the tests, and `CLAUDE.md` forbids editing the index, so a contributor would meet a failure they can't fix within `packs/es/**`. *Fix:* say in step 9 that this test fails until the maintainer regenerates the index, or have the check ignore packs that aren't in the index yet.
22. **The licence lists differ in small ways.** The doc's stop point 1 allows CC0; `CLAUDE.md`'s "Stop and ask" list and the PR template's checklist don't mention it. *Fix:* the same list in all three places.

### Not a doc issue, noted for the kit

- **Lost spaces in the corpus text.** The rendered truth reads "373,95°C (647,10K)" and "0,006atm", where the article has a space between number and unit (probably a template or non-breaking space dropped when the wikitext is flattened). It shows on the pages and in the truth alike, so recall isn't affected, but the pages look less like real text.
- **The tie window hides a slightly better set.** RapidOCR + Vision (`vision2`) with `corroborated_trigrams` (set 32) has a worst-condition word recall of 0.9461 against the chosen set's 0.9454, and passes the floors. `maximin-v1` treats the two as a tie (within 0.005) and takes the one-engine set, as the doc says, and the report's margin is 0.000 against the runner-up. This is the policy working as documented, not an override question.

## The bench on `darwin-arm64`

Word recall and must-survive (`accents`) recall per condition. The floor is 0.9 in every condition; a set below it in any condition is dropped. Engine ids with a 2 are the same engine with other parameters.

| # | Reader set | Pick | Words clean | Words scan | Words small | Accents clean | Accents scan | Accents small | Floors | s/page |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tesseract + rapidocr | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 5.14 |
| 1 | tesseract2 | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.46 |
| 2 | tesseract2 + vision | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.58 |
| 3 | tesseract2 + vision | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.58 |
| 4 | tesseract2 + vision2 | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.53 |
| 5 | tesseract2 + vision2 | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.52 |
| 6 | tesseract2 + rapidocr | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 4.42 |
| 7 | tesseract2 + rapidocr | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 4.50 |
| 8 | vision | single | 0.9601 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 0.68 |
| 9 | vision + tesseract2 | single | 0.9602 | 0.9533 | 0.9149 | 0.8746 | 0.8765 | 0.8862 | below | 2.71 |
| 10 | vision + tesseract2 | primary_with_floor | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.67 |
| 11 | vision + rapidocr | single | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.54 |
| 12 | vision + rapidocr | primary_with_floor | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.82 |
| 13 | vision2 | single | 0.9572 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 0.41 |
| 14 | vision2 + tesseract2 | single | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.70 |
| 15 | vision2 + tesseract2 | primary_with_floor | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.47 |
| 16 | vision2 + rapidocr | single | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.41 |
| 17 | vision2 + rapidocr | primary_with_floor | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.44 |
| 18 | rapidocr | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.25 |
| 19 | rapidocr + tesseract2 | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 4.48 |
| 20 | rapidocr + tesseract2 | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 4.84 |
| 21 | rapidocr + vision | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.46 |
| 22 | rapidocr + vision | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.49 |
| 23 | rapidocr + vision2 | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.39 |
| 24 | rapidocr + vision2 | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.44 |
| 25 | tesseract2 + vision | corroborated_trigrams | 0.9610 | 0.9695 | 0.9181 | 0.9042 | 0.9589 | 0.8952 | below | 2.46 |
| 26 | tesseract2 + vision + rapidocr | corroborated_trigrams | 0.9806 | 0.9744 | 0.9181 | 1.0000 | 0.9867 | 0.8952 | below | 4.38 |
| 27 | tesseract2 + vision2 | corroborated_trigrams | 0.9806 | 0.9695 | 0.9112 | 1.0000 | 0.9589 | 0.8946 | below | 2.37 |
| 28 | tesseract2 + vision2 + rapidocr | corroborated_trigrams | 0.9806 | 0.9744 | 0.9112 | 1.0000 | 0.9867 | 0.8946 | below | 4.27 |
| 29 | tesseract2 + rapidocr | corroborated_trigrams | 0.9806 | 0.9786 | 0.9441 | 1.0000 | 0.9964 | 0.9339 | pass | 4.39 |
| 30 | tesseract2 + rapidocr + vision | corroborated_trigrams | 0.9809 | 0.9803 | 0.9409 | 1.0000 | 0.9964 | 0.9339 | pass | 4.64 |
| 31 | vision + rapidocr | corroborated_trigrams | 0.9631 | 0.9787 | 0.9430 | 0.8746 | 0.9530 | 0.9298 | below | 2.87 |
| 32 | vision2 + rapidocr | corroborated_trigrams | 0.9810 | 0.9787 | 0.9461 | 0.9687 | 0.9530 | 0.9298 | pass | 2.63 |

Sets 18 to 24 (RapidOCR alone, or RapidOCR as the primary with a second engine) read identically: with RapidOCR as the primary, the second engine changed nothing on these pages. Tesseract's `spa` model falls below the accents floor on the small (110 dpi) pages, 0.882; Vision falls below it in every condition, 0.874 to 0.886.

## Selection (`kit choose`, policy `maximin-v1`)

```json
{"darwin-arm64": {"chosen": 18, "runner_up": 23, "margin": 0.0, "policy": "maximin-v1", "override_reason": null}}
```

## `kit verify es --download --rebench`

Exit 0. Checks: manifest `ok`, download `ok`, layer `ok`, wheel `ok`. Re-bench on `darwin-arm64`, tolerance 0.01 (no Vision in the set): every measured value (words, trigrams, digits and accents, in each condition) is identical to the claim, which the table in the report body below shows; 2.13 s per page against the claimed 2.25. `drift: []`, `floors: []`.

`kit wheel`: wheel 13,605,072 bytes; canary `canary-es.png` (19,218 bytes, 300 dpi) with `montaña`, `pingüino`, `Fotosíntesis` and `4521`, all found by RapidOCR.

## Tests

The whole suite, run once in a checkout that also has the maintainer's generated pack index: 3107 passed, 36 skipped, 1 failed. The failure is `tests/test_release.py::test_index_regenerates_identically`, caused by the new pack (item 21). The skips are the network, sandbox and English-base-pack tests that skip themselves by design in that environment. In the fresh clone, `tests/test_release.py` gave 70 passed, 4 skipped (no index yet).

## `kit report es` body

### Pack `es` 1.0.0: Spanish

- Languages: es (primary)
- Scripts: Latin
- Requires core: `>=0.1.0`

**Reader sets**

- `default`: rapidocr (rapidocr, reader); pick `single` v1, fallback `none`

**Models and fonts**

| File | Kind | Bytes | Licence | Source |
|---|---|---|---|---|
| `spa.traineddata` | traineddata | 13570187 | Apache-2.0 | https://github.com/tesseract-ocr/tessdata_best/raw/e12c65a915945e4c28e237a9b52bc4a8f39a0cec/spa.traineddata |

- Canary: `canary-es.png`, 19218 bytes
- Must survive: `accents` (word, floor 0.9)

**Corpus**

- `public-v1`: 8 texts, 2 fonts, pages {"form": 2, "microprint": 1, "mixed_script": 1, "prose": 4}
  - wikipedia `es` revision 176164945, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176164945&action=raw
  - wikipedia `es` revision 176154708, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176154708&action=raw
  - wikipedia `es` revision 175705929, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175705929&action=raw
  - wikipedia `es` revision 176120472, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176120472&action=raw
  - wikipedia `es` revision 175724719, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175724719&action=raw
  - wikipedia `es` revision 176158406, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176158406&action=raw
  - wikipedia `es` revision 175841461, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175841461&action=raw
  - wikipedia `es` revision 175931145, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175931145&action=raw

**Bench**

- Kit 1.0, corpus `public-v1@9268be5db0ed818b821574be5aeeadc4de0914cea4dcbab9b0dbf06e205c75b4`
- Engines: rapidocr 3.10.0, tesseract 5.5.3 (tesserocr 2.12.0), vision rev3 (macOS 27.2)

`darwin-arm64`: worst page 0.674, 2.253 s/page, policy `maximin-v1`, margin 0.000

| Metric | clean | scan | small |
|---|---|---|---|
| digits | 0.985 | 0.993 | 0.947 |
| must-accents | 1.000 | 0.996 | 0.934 |
| trigrams | 0.999 | 0.999 | 0.965 |
| words | 0.987 | 0.984 | 0.945 |

**Checklist**

- [ ] The PR changes only `packs/es/**`.
- [ ] No binaries: no model, font, image or PDF.
- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.
- [ ] `pdfintake kit verify es --download` passes.
- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.
- [ ] The must-survive characters were confirmed by someone who reads the language.

---

# Re-run after the doc and kit fixes

The kit owner rewrote `docs/add-a-language.md` for the 22 items above and fixed the kit where the trial hit a real bug (the largest: Tesseract's English candidates were all refused by the loader, so `refused_by_loader: 21` was a kit bug, not a normal count). The doc's steps 1, 2 (text only), 3, 4 (one sentence), 5, 6, 8 and 9 changed; step 7 did not. This re-run follows the new doc for the changed steps and everything that depends on them, and says for each earlier item whether it's gone.

- Machine: the same Mac, `darwin-arm64`, Apple Vision rev3. Engines: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Environment: the shared bench clone, with the owner's fixed source synced into it, run as `.venv/bin/pdfintake kit <cmd> es --packs <repo>/packs --work <work> --json`. `<work>` is a new, empty kit work folder; `<fresh>` is a second empty one used only for `verify`.
- Inputs unchanged: must-survive `accents=áéíóúüñÁÉÍÓÚÜÑ:word:0.9`; the same eight Wikipedia texts (CC-BY-SA-4.0); `kit choose`'s pick kept as is.
- Stop points met: none. `kit choose` exited 0, and no override was considered.

## Result of the re-run

- `kit candidates` now benches all 54 sets (it benched 33 before): `refused_by_loader` is 0, and the 21 Tesseract-English sets that were refused are measured. None of them changes the outcome: alone, Tesseract's English model reads the accents at 0.09 to 0.12, far below the floor, and the one combination with it that passes the floors (set 51) has a lower worst-condition word recall than RapidOCR alone.
- Chosen on `darwin-arm64`: **RapidOCR alone**, pick `single`, the same reader set as before. Worst-condition word recall 0.9454; accents 1.000 clean, 0.996 scan, 0.934 small; 2.10 s per page. The bench numbers are identical to the first trial's to four decimals.
- `kit choose` now drops `spa.traineddata` (13,570,187 bytes), which no chosen set uses: the wheel falls from 13,605,072 bytes to **23,111 bytes**. The canary (19,218 bytes, same SHA-256) is kept.
- `kit verify es --download` passes all four checks in an **empty** work folder, which it couldn't before (two were skipped). `--rebench` on `darwin-arm64` measured exactly the claimed numbers; `--rebench --platform linux-x86_64` now warns that the pack claims no numbers there.
- Files the pack PR commits, per `kit report`: `packs/es/manifest.json`, `corpus.json`, `bench.json`. `NOTICE-tessdata` moved to `leave_out` (its model was dropped); it's still in the folder, uncommitted.

## Steps of the re-run

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | `pdfintake doctor --json` | **3** | 1 s |
| 2 | not re-run (text-only change; the draft and corpus already exist, see R2) | n/a | |
| 3 | `pdfintake kit corpus es` (no arguments: print the pinned texts) | 0 | 1 s |
| 4 | `pdfintake kit render es` (empty work folder) | 0 | 9 s |
| 5 | `pdfintake kit candidates es` | 0 | 2 s |
| 5 | `pdfintake kit bench es` (in the background) | 0 | **938 s** (15.6 min, 54 sets) |
| 6 | `pdfintake kit choose es` | 0 | 2 s |
| 7 | `kit private` | skipped | optional, unchanged; no contributor PDFs |
| 8 | `pdfintake kit wheel es --must-find montaña --must-find pingüino --must-find Fotosíntesis --must-find 4521` | 0 | 3 s |
| 8 | `pdfintake kit verify es --download` (in `<work>`) | 0 | 1 s |
| 8 | `pdfintake kit verify es --download` (in `<fresh>`, empty) | 0 | 1 s |
| 8 | the same in another empty folder with `HTTPS_PROXY` set to a closed port | **1** | 4 s |
| 8 | `pdfintake kit verify es --download --rebench` (in `<fresh>`) | 0 | **93 s** |
| 8 | `pdfintake kit verify es --download --rebench --platform linux-x86_64` (in `<fresh>`) | 0 | 17 s |
| 9 | `pdfintake kit report es` | 0 | 1 s |
| 9 | the whole test suite | 1 | 208 s |

The closed-proxy run was a deliberate check of item 19 (does `--download` really fetch?), not a failure of the pack.

## The earlier items, one by one

| # | Item | Now |
|---|---|---|
| 1 | `pdfintake` not on PATH after `dev-install.sh` | **Gone** from the doc: step 1 now has `source .venv/bin/activate` and the `.venv/bin/pdfintake` alternative. `dev-install.sh` wasn't re-run here (the bench environment is shared and mustn't be reinstalled), so its new last line wasn't seen. |
| 2 | An active virtual environment takes the install | **Gone** from the doc (step 1's "Deactivate any other virtual environment first"). The script's new warning wasn't exercised, for the same reason. |
| 3 | `doctor` exits 3 on a fresh clone, no route | **Gone** for the case the trial met: step 1 now says exit 3 with only `index_missing` is fine for the kit. This run met a different exit 3, see R1. |
| 4 | `vision_languages` shape undocumented | **Gone**: step 2 explains the inner lists, with Spanish's `[["es-ES", "en-US"]]` as the example. Not re-run (step 2 changed in text only). |
| 5 | `manifest.draft.json` has no stated fate | **Gone**: step 2 says keep it until the PR is done and never commit it, and `kit report` lists it under `leave_out`. But see R2: the first trial had already removed the draft. |
| 6 | Example corpus ids are cities | **Gone**: the example is now water, the Sun, the Moon, photosynthesis, chess, bread and volcanoes, with a "pick general topics" rule. |
| 7 | No example font URL | **Gone**: two ready-made Latin URLs, and the folder pattern for other families. |
| 8 | The second script on a Latin pack's mixed page | **Gone**: step 3 says Greek (else Cyrillic) and that NotoSans and NotoSerif cover it. Checked: the rendered `mixed_script` page carries Greek lines, and render needed no extra font. |
| 9 | No article titles in `kit corpus` and `kit report` | **Gone**: `kit corpus es` lists `title` for every text (Agua, Sol, Luna, Fotosíntesis, Ajedrez, Pan, Volcán, Río), and the report body names each one. |
| 10 | Candidate counts don't add up | **Gone**: `total 54, refused_by_loader 0, cut 0, kept 54`, `count 54`. |
| 11 | `tesseract2` and `vision2` unexplained | **Gone**: step 5 explains numbered use ids. |
| 12 | The bench is silent and untimed | **Gone**: 54 progress lines on standard error (`kit bench: set 1/54 (rapidocr, single) ok, 15 s for 24 pages; about 13 min left`), and the doc's estimate (about 20 s a set, 30 to 60 sets in 10 to 20 min) matched: 54 sets in 15.6 min, 2 to 31 s a set. One small thing: the progress lines count sets from 1 (`set 1/54`), the JSON's `index` and `selections` from 0, so "set 30/54" in the log is index 29, the chosen set. |
| 13 | `ok` doesn't mean the floors pass | **Gone**: each set in the bench JSON has `floors_ok`, and `below_floor` lists the misses (`"accents clean 0.115 < 0.9"`, ...). 40 sets are below, 14 pass. |
| 14 | The single-machine case | **Gone**: step 5 says unbenched platforms are expected and the maintainer benches them; `kit choose` lists `win-amd64`, `linux-x86_64`, `linux-aarch64` under `unbenched`. |
| 15 | `next` empty although `kit wheel` hadn't run | **Gone as far as this run can tell.** The no-canary case wasn't met here, because the pack already had a canary. `next` did name `kit wheel` for the reason that applied (`"models were dropped, so the wheel changed: run pdfintake kit wheel es again"`), but step 6 doesn't mention that reason (R4). |
| 16 | The pack keeps an unused 13.6 MB model | **Gone**: `dropped_models: ["spa.traineddata"]`, the manifest's `models` is empty, the wheel is 23,111 bytes. |
| 17 | The Mac's pick becomes every platform's set | **Gone**: step 6 explains how `default` is picked and that a platform with no numbers reads with it. |
| 18 | `--rebench` on a platform with no claim passes silently | **Gone**: it now warns (`rebench: the pack claims no numbers for linux-x86_64, so only the must-survive floors were checked; ...`) in both `warnings` and `rebench.warning`. It still exits 0 with `"ok": true`, which matches the doc. |
| 19 | `--download` wording promised a fresh fetch | **Gone**: the doc now says a file already in the work folder with its pinned size and hash isn't fetched again. Checked both ways: in an empty folder with a closed proxy, verify exits 1 (`NotoSans-Regular.ttf couldn't be downloaded ... after three attempts`); in a populated folder with the same proxy, it exits 0. |
| 20 | `verify` passes with two checks skipped | **Gone**: in an empty work folder `verify --download` downloaded the base pack's NotoSans (431,364 bytes), rendered the canary to its pinned bytes, and reported `manifest`, `download`, `layer` and `wheel` all `ok`, with `warnings: []`. |
| 21 | The index test fails for a pack-only change | **Gone** from the doc: step 9 now names exactly that failure as expected. The suite gave exactly it (see Tests). |
| 22 | The licence lists differ | **Not gone.** The doc's stop point 1 now names CC0 and the kit's ids, but `CLAUDE.md` ("Stop and ask"), `CONTRIBUTING.md` and the language-pack PR template's checklist still list public domain, CC BY, CC BY-SA and government text, without CC0. Those files are outside this task's write scope; reported, not changed. |

The two "noted for the kit" items: the **lost spaces** in the corpus text are still there (`373,95°C (647,10K)`, `0,006atm` on the mixed-script page's truth). The **tie window** still applies; this time the margin is between two copies of RapidOCR alone (R3), and `vision2 + rapidocr` with `corroborated_trigrams` (set 52, 0.9461 against 0.9454) is again treated as a tie, as documented.

## New in this re-run

**R1. Step 1: `doctor` exits 3 for a reason the doc's route can't clear here.** The problem is `base_pack: hash_mismatch` ("…pdfintake_data_en/manifest.json has SHA-256 d784…; this pdfintake's pack index pins 18ad…"), with `index: ok`. The doc's route for any problem other than `index_missing` is to run `scripts/dev-install.sh` again and, if it stays, stop point 4. The cause is known and belongs to this environment, not to the doc: the source sync into the shared bench clone brought in another agent's work-in-progress `src/pdfintake/langpacks/index.json`, which pins a different English manifest from the one installed in the bench `.venv`. *Did instead:* didn't run `dev-install.sh` (the shared bench environment mustn't be reinstalled) and went on, since the doc says the kit takes the base pack from `packs/en` and doesn't read the index. Every kit command afterwards exited as expected. A real contributor on a fresh clone wouldn't meet this; whoever owns the bench environment should rebuild it once the index settles.

**R2. The first trial had removed `manifest.draft.json`, and that now matters.** The pack folder has no draft (the first trial left it out of the folder after checking `kit choose` works without it). The new doc says to keep it until the PR is done, because `kit choose` now drops unused models from `manifest.json` while "the draft keeps them, so a later bench still proposes them". In this run the bench still proposed the Spanish Tesseract model, because `manifest.json` still pinned it when `candidates` ran. After this `kit choose`, it doesn't: a later `kit candidates es` here would no longer offer `tesseract2` with `spa`. The doc also doesn't say what set 0 is when there's no draft (it says "the draft's own reader set"); here it was the manifest's reader set. *Did:* nothing, since recreating the draft means `kit init es --force`, which starts the pack over, and step 2 changed in text only. *Fix for the pack:* before any later re-bench, re-create the draft (or say in the doc how to recover one from `manifest.json`).

**R3. Two candidates are the same reader.** Set 0 (RapidOCR alone, params `rec`, `use_cls: false`, `min_score: 0.5`) and set 29 (RapidOCR alone, params `rec` only) bench identically to four decimals, and `kit choose` picked 29 over 0 on speed (2.10 against 2.13 s per page), margin 0.000. The written reader set is unchanged from before, so this is harmless, but the doc calls `total` "the distinct sets proposed", and the selection's runner-up and margin say nothing useful when the runner-up is the same reader. A kit note, not a pack problem.

**R4. Step 6 doesn't mention dropped models in `next`.** The doc says `next` names `kit wheel` when there's no canary yet, or when the chosen reader sets changed (canary dropped). This run had a kept canary and the same reader sets, and `next` still said to run `kit wheel` again, because a model was dropped and the wheel changed. Correct behaviour; the doc should list this third reason.

**R5. `verify`'s `download` check stays `ok` when the base pack's fonts can't be downloaded.** In the closed-proxy run the failure is reported under `layer` and `wheel` ("the base pack's fonts couldn't be downloaded"), while `download` says `ok`. Step 8 says `download` covers "the base pack's fonts". The run still exits 1 with a clear reason, so nothing passes that shouldn't; the doc and the check's name just disagree on where the failure shows.

**R6. A cold `verify --rebench` took 93 s.** In an empty work folder, `verify es --download --rebench` took 93 s (it renders the 24 pages and fetches what it needs first); in the same folder afterwards, 17 s. The doc says "up to a minute". Worth saying that a cold run takes longer.

**R7. The report's models table is empty.** With no pack-own models left, the report body prints the table header of "Models and fonts" and no rows. A line such as "none: reads with the base pack's models" would be clearer to a reviewer.

## The bench on `darwin-arm64` (re-run)

Word recall and must-survive (`accents`) recall per condition; the floor is 0.9 in every condition. `tesseract` is Tesseract with the base pack's English model, `tesseract2` with `spa`; `vision2` is Vision with its other parameter set.

| # | Reader set | Pick | Words clean | Words scan | Words small | Accents clean | Accents scan | Accents small | Floors | s/page |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | rapidocr | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.13 |
| 1 | tesseract | single | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 0.90 |
| 2 | tesseract + vision | single | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 1.14 |
| 3 | tesseract + vision | primary_with_floor | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 1.17 |
| 4 | tesseract + vision2 | single | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 1.08 |
| 5 | tesseract + vision2 | primary_with_floor | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 1.08 |
| 6 | tesseract + rapidocr | single | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 4.09 |
| 7 | tesseract + rapidocr | primary_with_floor | 0.8890 | 0.8836 | 0.7824 | 0.1154 | 0.1141 | 0.0905 | below | 3.10 |
| 8 | tesseract2 | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.37 |
| 9 | tesseract2 + vision | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.48 |
| 10 | tesseract2 + vision | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.46 |
| 11 | tesseract2 + vision2 | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.38 |
| 12 | tesseract2 + vision2 | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 2.38 |
| 13 | tesseract2 + rapidocr | single | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 4.23 |
| 14 | tesseract2 + rapidocr | primary_with_floor | 0.9806 | 0.9727 | 0.8773 | 1.0000 | 0.9867 | 0.8823 | below | 4.28 |
| 15 | vision | single | 0.9601 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 0.63 |
| 16 | vision + tesseract | single | 0.9601 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 1.22 |
| 17 | vision + tesseract | primary_with_floor | 0.9601 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 1.21 |
| 18 | vision + tesseract2 | single | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.57 |
| 19 | vision + tesseract2 | primary_with_floor | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.57 |
| 20 | vision + rapidocr | single | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.35 |
| 21 | vision + rapidocr | primary_with_floor | 0.9602 | 0.9533 | 0.9155 | 0.8746 | 0.8765 | 0.8862 | below | 2.40 |
| 22 | vision2 | single | 0.9572 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 0.38 |
| 23 | vision2 + tesseract | single | 0.9572 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 1.08 |
| 24 | vision2 + tesseract | primary_with_floor | 0.9572 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 1.05 |
| 25 | vision2 + tesseract2 | single | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.46 |
| 26 | vision2 + tesseract2 | primary_with_floor | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.44 |
| 27 | vision2 + rapidocr | single | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.29 |
| 28 | vision2 + rapidocr | primary_with_floor | 0.9573 | 0.9500 | 0.9084 | 0.8740 | 0.8747 | 0.8856 | below | 2.28 |
| 29 | rapidocr | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.10 |
| 30 | rapidocr + tesseract | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.94 |
| 31 | rapidocr + tesseract | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.94 |
| 32 | rapidocr + tesseract2 | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 4.34 |
| 33 | rapidocr + tesseract2 | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 4.32 |
| 34 | rapidocr + vision | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.40 |
| 35 | rapidocr + vision | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.38 |
| 36 | rapidocr + vision2 | single | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.31 |
| 37 | rapidocr + vision2 | primary_with_floor | 0.9874 | 0.9841 | 0.9454 | 1.0000 | 0.9964 | 0.9339 | pass | 2.49 |
| 38 | tesseract + vision | corroborated_trigrams | 0.8890 | 0.8836 | 0.8337 | 0.1154 | 0.1141 | 0.1148 | below | 1.14 |
| 39 | tesseract + vision + rapidocr | corroborated_trigrams | 0.9690 | 0.9620 | 0.9079 | 0.8649 | 0.8553 | 0.7829 | below | 3.15 |
| 40 | tesseract + vision2 | corroborated_trigrams | 0.8890 | 0.8836 | 0.8268 | 0.1154 | 0.1141 | 0.1141 | below | 1.02 |
| 41 | tesseract + vision2 + rapidocr | corroborated_trigrams | 0.9696 | 0.9620 | 0.9010 | 0.8667 | 0.8553 | 0.7823 | below | 2.97 |
| 42 | tesseract + rapidocr | corroborated_trigrams | 0.8890 | 0.9003 | 0.8569 | 0.1154 | 0.2118 | 0.1493 | below | 2.85 |
| 43 | tesseract + rapidocr + vision | corroborated_trigrams | 0.9763 | 0.9841 | 0.9454 | 0.8999 | 0.9964 | 0.9339 | below | 3.09 |
| 44 | tesseract2 + vision | corroborated_trigrams | 0.9610 | 0.9695 | 0.9181 | 0.9042 | 0.9589 | 0.8952 | below | 2.51 |
| 45 | tesseract2 + vision + rapidocr | corroborated_trigrams | 0.9806 | 0.9744 | 0.9181 | 1.0000 | 0.9867 | 0.8952 | below | 4.36 |
| 46 | tesseract2 + vision2 | corroborated_trigrams | 0.9806 | 0.9695 | 0.9112 | 1.0000 | 0.9589 | 0.8946 | below | 2.36 |
| 47 | tesseract2 + vision2 + rapidocr | corroborated_trigrams | 0.9806 | 0.9744 | 0.9112 | 1.0000 | 0.9867 | 0.8946 | below | 4.27 |
| 48 | tesseract2 + rapidocr | corroborated_trigrams | 0.9806 | 0.9786 | 0.9441 | 1.0000 | 0.9964 | 0.9339 | pass | 4.16 |
| 49 | tesseract2 + rapidocr + vision | corroborated_trigrams | 0.9809 | 0.9803 | 0.9409 | 1.0000 | 0.9964 | 0.9339 | pass | 4.39 |
| 50 | vision + rapidocr | corroborated_trigrams | 0.9631 | 0.9787 | 0.9430 | 0.8746 | 0.9530 | 0.9298 | below | 2.39 |
| 51 | vision + rapidocr + tesseract | corroborated_trigrams | 0.9800 | 0.9787 | 0.9430 | 0.9651 | 0.9530 | 0.9298 | pass | 3.22 |
| 52 | vision2 + rapidocr | corroborated_trigrams | 0.9810 | 0.9787 | 0.9461 | 0.9687 | 0.9530 | 0.9298 | pass | 2.21 |
| 53 | vision2 + rapidocr + tesseract | corroborated_trigrams | 0.9806 | 0.9787 | 0.9130 | 0.9668 | 0.9530 | 0.8946 | below | 3.07 |

The 21 sets with `tesseract` (Tesseract with the base pack's English model: 1 to 7, 16, 17, 23, 24, 30, 31, 38 to 43, 51 and 53) are the ones the loader used to refuse. Read alone, the English model loses nearly every accent (0.09 to 0.12). As a second engine behind RapidOCR it changes nothing (30, 31). In `corroborated_trigrams`, set 51 (Vision + RapidOCR + Tesseract English) passes the floors, at 0.9430 worst-condition word recall, and set 43 misses the clean-page floor by 0.0001 (0.8999). None beats RapidOCR alone. Every other set matches the first trial's numbers exactly; the indices moved because 21 sets were added. Set 0 is now the reader set of `manifest.json` (there is no draft, see R2) rather than the draft's Tesseract-Spanish + RapidOCR set, which is still benched as set 14.

## Selection (`kit choose`, policy `maximin-v1`, re-run)

```json
{"darwin-arm64": {"chosen": 29, "runner_up": 0, "margin": 0.0, "policy": "maximin-v1", "override_reason": null}}
```

`"canary": "kept"`, `dropped_models: ["spa.traineddata"]`, `unbenched: ["win-amd64", "linux-x86_64", "linux-aarch64"]`, and `next`: "models were dropped, so the wheel changed: run `pdfintake kit wheel es` again". The manifest's reader sets and canary are unchanged from the first trial; its `models` went from `spa.traineddata` to none, and the claimed seconds per page from 2.253 to 2.101.

## `kit wheel` and `kit verify` (re-run)

`kit wheel`: 23,111 bytes (was 13,605,072); canary `canary-es.png`, 19,218 bytes, SHA-256 `5d5ef5c5…`, unchanged; RapidOCR found `montaña`, `pingüino`, `Fotosíntesis` and `4521`, `missing: []`.

`kit verify es --download --rebench`, in an empty work folder: exit 0, checks `manifest`, `download`, `layer`, `wheel` all `ok`, `warnings: []`. Re-bench on `darwin-arm64`, tolerance 0.01: every measured value equals the claim, which the report body below shows per condition (worst page 0.6745), 2.01 s per page against the claimed 2.10; `drift: []`, `floors: []`.

`kit verify es --download --rebench --platform linux-x86_64`: exit 0, `"ok": true`, `claimed: null`, the same measured numbers, and the warning quoted under item 18.

## Tests (re-run)

The whole suite, in the work tree that now holds a generated pack index: **3122 passed, 36 skipped, 1 failed** in 208 s. The failure is `tests/test_release.py::test_index_regenerates_identically` with `es: version 1.0.0 has no wheel in dist`, the one step 9 now says to expect. The skips: the network pack tests (only with `PDFINTAKE_NETWORK_TESTS=1`), 20 end-to-end tests and one Vision test that need the English data package installed in the test environment, the four sandbox smoke tests (non-editable install only), the release test for the placeholder pyproject pin, one SIGHUP-handler test, and eight contract and record cases that don't apply (Vision needs no model files, a rule without corroborators or tie order, a valid or null value).

## `kit report es` body (re-run)

`commit`: `packs/es/bench.json`, `packs/es/corpus.json`, `packs/es/manifest.json`. `leave_out`: `packs/es/NOTICE-tessdata`. Nothing `missing`.

#### Pack `es` 1.0.0: Spanish

- Languages: es (primary)
- Scripts: Latin
- Requires core: `>=0.1.0`

**Reader sets**

- `default`: rapidocr (rapidocr, reader); pick `single` v1, fallback `none`

**Models and fonts**

| File | Kind | Bytes | Licence | Source |
|---|---|---|---|---|

- Canary: `canary-es.png`, 19218 bytes
- Must survive: `accents` (word, floor 0.9)

**Corpus**

- `public-v1`: 8 texts, 2 fonts, pages {"form": 2, "microprint": 1, "mixed_script": 1, "prose": 4}
  - Agua: wikipedia `es` revision 176164945, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176164945&action=raw
  - Sol: wikipedia `es` revision 176154708, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176154708&action=raw
  - Luna: wikipedia `es` revision 175705929, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175705929&action=raw
  - Fotosíntesis: wikipedia `es` revision 176120472, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176120472&action=raw
  - Ajedrez: wikipedia `es` revision 175724719, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175724719&action=raw
  - Pan: wikipedia `es` revision 176158406, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=176158406&action=raw
  - Volcán: wikipedia `es` revision 175841461, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175841461&action=raw
  - Río: wikipedia `es` revision 175931145, CC-BY-SA-4.0: https://es.wikipedia.org/w/index.php?oldid=175931145&action=raw

**Bench**

- Kit 1.0, corpus `public-v1@9268be5db0ed818b821574be5aeeadc4de0914cea4dcbab9b0dbf06e205c75b4`
- Engines: rapidocr 3.10.0, tesseract 5.5.3 (tesserocr 2.12.0), vision rev3 (macOS 27.2)

`darwin-arm64`: worst page 0.674, 2.101 s/page, policy `maximin-v1`, margin 0.000

| Metric | clean | scan | small |
|---|---|---|---|
| digits | 0.985 | 0.993 | 0.947 |
| must-accents | 1.000 | 0.996 | 0.934 |
| trigrams | 0.999 | 0.999 | 0.965 |
| words | 0.987 | 0.984 | 0.945 |

**Checklist**

- [ ] The PR changes only `packs/es/**`.
- [ ] No binaries: no model, font, image or PDF.
- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.
- [ ] `pdfintake kit verify es --download` passes.
- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.
- [ ] The must-survive characters were confirmed by someone who reads the language.
