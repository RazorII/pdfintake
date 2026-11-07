# Kit trial: Vietnamese (`vi`)

A contributor's Claude, who wrote none of the kit, added the Vietnamese pack by following `CLAUDE.md` and `docs/add-a-language.md` step by step, without reading the kit's source, its tests or any design notes. Every command, its exit code, and every place the doc was wrong, unclear or missing something is below, with what was done instead.

- Machine: macOS 27.2 on Apple silicon (`darwin-arm64`), Apple Vision available (revision 3).
- Engines measured: RapidOCR 3.10.0, Tesseract 5.5.3 (tesserocr 2.12.0), Vision rev3.
- Environment: a shared bench clone of the repo with its own `.venv` (step 1 was done before this trial). Every kit command ran as `.venv/bin/pdfintake kit <cmd> vi --packs <repo>/packs --work <work> --json`, where `<work>` is a kit work folder outside the repo; `<fresh>` is a second, empty work folder used only for the re-bench.
- Stop points met: none. `kit choose` exited 0 and its pick was kept as is; no override was considered. The must-survive characters were given by the person:
  - `letters=ăâđêôơưĂÂĐÊÔƠƯ:word:0.9` (the seven Vietnamese base letters, both cases);
  - `tones=àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ:word:0.85` (every lower-case vowel with a tone mark, including the stacked ones).
- Every corpus text is Wikipedia, CC-BY-SA-4.0, on a neutral topic.

## Result

*Since the trial:* under `maximin-v2` and the corrected page truth, `kit choose vi` picks Tesseract `vie` + Vision (language correction on) with RapidOCR as checker and corroborator (worst-condition words 0.9020, digits 0.9713), winning the tie window on digit recall. The canary's words changed to `không nguyên thành 4521`, because RapidOCR's Latin model can't write ă, ơ, ư or đ and every use must read the canary. What follows is the trial as it ran.

- **Which engine reads Vietnamese's stacked diacritics: Tesseract's `vie` model.** Alone (set 8) it keeps the letters at 0.9997 clean, 0.9963 scan and 0.9046 small, and the tones at 0.9998, 0.9922 and 0.8921. Vision comes second: 0.948, 0.985 and 0.888 for the letters (below the 0.9 floor on small pages) and 0.914, 0.947 and 0.863 for the tones, with language correction on or off. RapidOCR's Latin recogniser can't read Vietnamese: letters about 0.33 and tones about 0.29 in every condition, and word recall about 0.50. Tesseract's English model reads none of it (letters 0.000, tones 0.005).
- Chosen on `darwin-arm64`: **Tesseract `vie` + Vision (language correction off), pick `corroborated_trigrams`** (set 46). Worst-condition word recall 0.8881; letters 0.983, 0.986 and 0.903, tones 0.957, 0.967 and 0.905; 2.37 s per page.
- Runner-up: the same pair with Vision's language correction on (set 44), worst-condition word recall 0.8899, 2.45 s per page. The two are within the 0.005 tie window and both use two engines, so the faster one won. The reported margin is **-0.0018** (see item 4).
- `default` (every other platform): the draft's own set, Tesseract `vie` as primary with RapidOCR as checker, pick `primary_with_floor`, because the Mac's pick uses Vision, which runs only on a Mac. On this Mac it is set 0: worst-condition word recall 0.8752, letters 0.9046 and tones 0.8921 on small pages, floors kept.
- `kit verify vi --download --rebench` in an empty work folder: exit 0, all four checks `ok`, re-bench identical to the claim (no drift, no floor missed).
- Files the pack PR commits, per `kit report`: `packs/vi/manifest.json`, `corpus.json`, `bench.json`, `NOTICE-tessdata`. `manifest.draft.json` is in `leave_out`.

## Steps

| Step | Command | Exit | Time |
|---|---|---|---|
| 1 | done before the trial (shared bench clone); `pdfintake doctor --json` | **3** | under 1 s |
| 2 | `pdfintake kit init vi --script Latin --must-survive 'letters=ăâđêôơưĂÂĐÊÔƠƯ:word:0.9' --must-survive 'tones=àáảãạ…ỳýỷỹỵ:word:0.85'` (the full `tones` list is above) | 0 | 1 s |
| 3 | `pdfintake kit corpus vi --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072,Q4022 --font <NotoSans-Regular> --font <NotoSerif-Regular>` | 0 | 12 s |
| 4 | `pdfintake kit render vi` | 0 | 7 s |
| 5 | `pdfintake kit candidates vi` | 0 | under 1 s |
| 5 | `pdfintake kit bench vi`, first run, in the background, while four other kit benches ran on the same Mac | killed after set 10 | about 57 min |
| 5 | `pdfintake kit bench vi`, second run, in the background, alone on the Mac | 0 | **841 s** (14 min, 54 sets) |
| 6 | `pdfintake kit choose vi` | 0 | under 1 s |
| 7 | `kit private` | skipped | optional; the trial has no contributor PDFs |
| 8 | `pdfintake kit wheel vi --must-find nước --must-find Đất --must-find Trăng --must-find hợp --must-find 4521` | **1** | 1 s |
| 8 | `pdfintake kit wheel vi --must-find nước --must-find Đất --must-find hợp --must-find 4521` | 0 | 1 s |
| 8 | `pdfintake kit verify vi --download` (in `<work>`) | 0 | 1 s |
| 8 | `pdfintake kit verify vi --download --rebench` (in `<fresh>`, empty) | 0 | 38 s |
| 8 | `pdfintake kit verify vi --download --rebench --platform linux-x86_64` (in `<fresh>`) | 0 | 30 s |
| 9 | `pdfintake kit report vi` | 0 | under 1 s |
| 9 | the whole test suite | see *Tests* | |

The corpus is eight Vietnamese Wikipedia articles: Nước (water, Q283), Mặt Trời (the Sun, Q525), Mặt Trăng (the Moon, Q405), Quang hợp (photosynthesis, Q11982), Cờ vua (chess, Q718), Bánh mì (bread, Q7802), Núi lửa (volcano, Q8072) and Sông (river, Q4022). `kit corpus` printed each title, so each id was checked against the article it led to. The fonts are the doc's two ready-made URLs, `NotoSans-Regular.ttf` and `NotoSerif-Regular.ttf` from `github.com/notofonts/notofonts.github.io` at commit `28b15b4…`.

`kit render` drew 24 pages: 4 prose, 2 form, 1 microprint and 1 mixed-script page, each in the three conditions. A visual check of a clean prose page showed the stacked marks (ặ, ộ, ữ, ở) drawn correctly. `kit candidates` listed 54 sets (`total` 54, `refused_by_loader` 0, `cut` 0, `kept` 54), and `kit bench` read all 24 pages with each of them.

## Where the doc was wrong, unclear or missing something

Each item gives the step, what happened, and what was done instead. "Doc" means `docs/add-a-language.md`.

### Step 1: set up

1. **`doctor` exits 3 for a reason the doc doesn't route.** In the shared bench clone, `doctor` reported no `index_missing`: `index` was `ok`, and `base_pack` failed with code `hash_mismatch`, saying the installed English data wheel's `manifest.json` has one SHA-256 and the clone's pack index pins another. The doc says to run `scripts/dev-install.sh` again for any problem other than `index_missing`, and that a problem that stays is stop point 4. Step 1 was already done for this trial and its environment was not mine to reinstall. *Did instead:* went on to step 2. Every kit command afterwards exited 0, which fits the doc's note that the kit takes the base pack from `packs/en` rather than from the installed wheel. *Fix:* say in step 1 whether a `base_pack` problem blocks the kit (here it didn't), the same way the doc already does for `index_missing`. The environment issue itself is likely a pack index generated after `packs/en/manifest.json` changed, with the English wheel not rebuilt.

### Step 5: candidates and bench

2. **Benches running side by side can wedge Apple Vision, and nothing says so.** The first bench ran while four other kit benches ran on the same Mac (load average about 84 on 18 cores). Its sets with Vision took 249 to 744 seconds each, where the sets without Vision took 9 to 73 seconds. Once the other benches had finished (load average about 2), the next Vision set still took 736 seconds: about 31 seconds per page, with the bench process at 0% CPU and its four Vision threads blocked in Vision's own task queue. A one-page Vision read with the same languages in a fresh process took 0.4 seconds, and eight pages on four threads 1.5 seconds. Every set still printed `ok`, so the progress lines gave no sign that anything was wrong beyond the time; the forecast went from "about 64 min left" to "about 340 min left". *Did instead:* killed the first bench after set 10 and started it again alone; the second run's Vision sets took 7 to 19 seconds, and the whole bench 14 minutes. *Fix:* say in step 5 to run one bench at a time on a machine, and that a Vision set taking minutes rather than seconds means Vision is stuck in that process: stop the bench and start it again. A per-set warning when one engine's seconds per page jumps tenfold would catch it in the kit.
3. **The bare id `tesseract` means a different model from one set to the next.** In set 0 (the draft's own set) `tesseract` reads with `vie`; in sets 1 to 7 and 38 to 43 `tesseract` reads with `eng`, and `tesseract2` is the one that reads `vie`. The doc does say that set 0 keeps the draft's ids and that a set's `params` name the model, but the progress lines (`set 2/54 (tesseract, single)`) and `kit report`'s reader-set lines show only ids, so from them alone you can't tell an English Tesseract set from a Vietnamese one, and the chosen `darwin-arm64` set calls `vie` `tesseract2` while `default` calls it `tesseract`. *Did instead:* read the `params` of each set in the candidates file (the doc's "Each set lists its engine uses" is about that file; the command's JSON gives only the counts and the file's path). *Fix:* name the model in the progress line (`tesseract[vie]`), and say in step 5 that the sets are in the file named by `file`.

### Step 6: choose

4. **The doc doesn't say what a negative margin means.** `kit choose` reported `"chosen": 46, "runner_up": 44, "margin": -0.0018`, and `kit report` prints "margin -0.002". The runner-up has the higher worst-condition word recall (0.8899 against 0.8881); the two tie within 0.005, both have two engines, and set 46 is faster (2.37 against 2.45 s per page), which is the tie-break the doc describes. So the policy worked as written, but a reviewer reading "margin -0.002" can't tell that from a mistake. *Fix:* say in step 6 that the margin is the chosen set's worst-condition metric minus the runner-up's, and that it is negative when the pick was won on a tie-break.
5. **The report shows no numbers for the reader set most platforms get.** The manifest's `default` is the draft's own set (Tesseract `vie` + RapidOCR checker), and Linux and Windows read with it, yet `kit report`'s **Bench** section shows only the `darwin-arm64` set. Its numbers exist (it is set 0 of the Mac's bench, and `verify --rebench --platform linux-x86_64` measured the same values), but a reviewer of the PR can't see them. *Fix:* when `default` is a set that was benched on some platform, show its numbers in the report and say on which platform they were measured.

### Step 8: wheel and verify

6. **`--must-find` takes at most four words, and the doc doesn't say so.** The doc says "Repeat `--must-find` for each word"; five gave exit 1, `kit_invalid_manifest`: "the manifest with its canary wouldn't load: $.canary.must_find: has 5 items; at most 4". *Did instead:* dropped `Trăng` and kept `nước`, `Đất`, `hợp` and `4521`, which still cover ư, ơ with a tone, Đ, â with a tone and a dot below under a horn. *Fix:* say "up to four words, `4521` included".
7. **Which reader set reads the canary is unclear when the manifest has two.** The doc says `wheel` reads the canary "with every engine of the reader set". `kit wheel` read it with `tesseract2` and `vision2` (the Mac's set) only; the `default` set's engines weren't listed in `canary_reads`. Its reader is the same Tesseract `vie` model, so no gap showed here, but a `default` set with a different reader would ship a canary it was never checked against on this machine. *Fix:* say which set's engines read the canary (the current platform's, it appears), or read it with every set the manifest holds whose engines run here.

## Not a doc issue, noted for the kit and the maintainer

- **The chosen set reads digits worse on scans.** Set 46 has digit recall 0.9963 clean, **0.8159** scan and 0.8984 small, where Tesseract `vie` with a RapidOCR checker (set 0, the `default`) has 0.9899, 0.9825 and 0.9677. `maximin-v1` ranks by word recall only, so this is the policy working as documented, not an override question; it may be worth a look before Vietnamese ships, since form pages carry numbers.
- **A checker changes a `single` pick's result.** Tesseract `vie` alone (set 8) has word recall 0.9753 clean, 0.9719 scan and 0.8719 small, and digit recall 0.9836 clean, 0.9762 scan and 0.8429 small; the same reader with a Vision or RapidOCR checker under the same `single` rule (sets 9 to 14) has 0.9786, 0.9752 and 0.8752 and digits up to 0.9937, 0.9825 and 0.9344. With `single` the checker shouldn't change what is read. In the Spanish trial the same comparison was identical, so this may be specific to this pack's pages; not investigated, since the trial doesn't read the kit's source.
- **Vision's tag for Vietnamese is `vi-VT`.** `kit init`'s `vision_languages` gave `[["vi-VT", "en-US"]]`, and `doctor` lists `vi-VT` among Vision's languages. That's Apple's own tag (BCP 47 would be `vi-VN`); it works, as the bench shows.
- **Lost template text in the corpus.** The rendered truth reads "Pliny cha (sống vào năm đến và Claudius Ptolemaeus (khoảng năm giải thích …", where the article has years in templates that were dropped when the wikitext was flattened, leaving unclosed brackets. It shows on the pages and in the truth alike, so recall isn't affected; it's the same kind of loss as the Spanish trial's missing spaces between numbers and units. *Added after the trial:* the kit's page-truth tokeniser (`render._words`) lost these numbers and brackets, not the corpus; fixed in the next kit change.
- **Worst page.** Every Tesseract `vie` set has a worst page near 0.18 and the chosen set 0.33. The page isn't named in the report.

## The bench on `darwin-arm64`

Word recall and must-survive recall per condition (second run). Floors: `letters` 0.9 and `tones` 0.85 in every condition; a set below either in any condition is dropped. The model or Vision option is in brackets; `tesseract` is `eng` except in set 0.

| # | Reader set | Pick | Words clean | Words scan | Words small | Letters clean | Letters scan | Letters small | Tones clean | Tones scan | Tones small | Floors | s/page |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | tesseract (vie) + rapidocr | primary_with_floor | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 4.03 |
| 1 | tesseract (eng) | single | 0.2496 | 0.2491 | 0.2358 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 0.87 |
| 2 | tesseract (eng) + vision (corr on) | single | 0.2496 | 0.2492 | 0.2358 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 1.13 |
| 3 | tesseract (eng) + vision (corr on) | primary_with_floor | 0.2496 | 0.2492 | 0.2358 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 1.11 |
| 4 | tesseract (eng) + vision2 (corr off) | single | 0.2496 | 0.2492 | 0.2358 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 1.01 |
| 5 | tesseract (eng) + vision2 (corr off) | primary_with_floor | 0.2496 | 0.2492 | 0.2358 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 1.01 |
| 6 | tesseract (eng) + rapidocr | single | 0.2496 | 0.2492 | 0.2359 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 2.75 |
| 7 | tesseract (eng) + rapidocr | primary_with_floor | 0.2496 | 0.2492 | 0.2359 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0045 | 0.0042 | below | 2.83 |
| 8 | tesseract2 (vie) | single | 0.9753 | 0.9719 | 0.8719 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 2.35 |
| 9 | tesseract2 (vie) + vision (corr on) | single | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 2.49 |
| 10 | tesseract2 (vie) + vision (corr on) | primary_with_floor | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 2.47 |
| 11 | tesseract2 (vie) + vision2 (corr off) | single | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 2.37 |
| 12 | tesseract2 (vie) + vision2 (corr off) | primary_with_floor | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 2.37 |
| 13 | tesseract2 (vie) + rapidocr | single | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 4.18 |
| 14 | tesseract2 (vie) + rapidocr | primary_with_floor | 0.9786 | 0.9752 | 0.8752 | 0.9997 | 0.9963 | 0.9046 | 0.9998 | 0.9922 | 0.8921 | pass | 4.20 |
| 15 | vision (corr on) | single | 0.9327 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 0.61 |
| 16 | vision (corr on) + tesseract (eng) | single | 0.9327 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 1.19 |
| 17 | vision (corr on) + tesseract (eng) | primary_with_floor | 0.9327 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 1.17 |
| 18 | vision (corr on) + tesseract2 (vie) | single | 0.9328 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 2.54 |
| 19 | vision (corr on) + tesseract2 (vie) | primary_with_floor | 0.9328 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 2.53 |
| 20 | vision (corr on) + rapidocr | single | 0.9328 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 2.15 |
| 21 | vision (corr on) + rapidocr | primary_with_floor | 0.9328 | 0.9568 | 0.8789 | 0.9482 | 0.9851 | 0.8876 | 0.9138 | 0.9474 | 0.8629 | below | 2.19 |
| 22 | vision2 (corr off) | single | 0.9336 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 0.37 |
| 23 | vision2 (corr off) + tesseract (eng) | single | 0.9336 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 1.07 |
| 24 | vision2 (corr off) + tesseract (eng) | primary_with_floor | 0.9336 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 1.05 |
| 25 | vision2 (corr off) + tesseract2 (vie) | single | 0.9337 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 2.42 |
| 26 | vision2 (corr off) + tesseract2 (vie) | primary_with_floor | 0.9337 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 2.40 |
| 27 | vision2 (corr off) + rapidocr | single | 0.9337 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 2.06 |
| 28 | vision2 (corr off) + rapidocr | primary_with_floor | 0.9337 | 0.9570 | 0.8775 | 0.9517 | 0.9851 | 0.8876 | 0.9141 | 0.9476 | 0.8621 | below | 2.09 |
| 29 | rapidocr | single | 0.5042 | 0.4992 | 0.4885 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 1.92 |
| 30 | rapidocr + tesseract (eng) | single | 0.5042 | 0.4992 | 0.4885 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.79 |
| 31 | rapidocr + tesseract (eng) | primary_with_floor | 0.5042 | 0.4992 | 0.4885 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.76 |
| 32 | rapidocr + tesseract2 (vie) | single | 0.5042 | 0.4992 | 0.4886 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 4.17 |
| 33 | rapidocr + tesseract2 (vie) | primary_with_floor | 0.5042 | 0.4992 | 0.4886 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 4.15 |
| 34 | rapidocr + vision (corr on) | single | 0.5042 | 0.4992 | 0.4886 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.19 |
| 35 | rapidocr + vision (corr on) | primary_with_floor | 0.5042 | 0.4992 | 0.4886 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.19 |
| 36 | rapidocr + vision2 (corr off) | single | 0.5042 | 0.4992 | 0.4886 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.09 |
| 37 | rapidocr + vision2 (corr off) | primary_with_floor | 0.5042 | 0.4992 | 0.4886 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.05 |
| 38 | tesseract (eng) + vision (corr on) | corroborated_trigrams | 0.2496 | 0.2491 | 0.2763 | 0.0000 | 0.0000 | 0.0291 | 0.0045 | 0.0045 | 0.0395 | below | 1.13 |
| 39 | tesseract (eng) + vision (corr on) + rapidocr | corroborated_trigrams | 0.6474 | 0.5668 | 0.6803 | 0.6024 | 0.4933 | 0.6397 | 0.5720 | 0.4705 | 0.6189 | below | 2.93 |
| 40 | tesseract (eng) + vision2 (corr off) | corroborated_trigrams | 0.2496 | 0.2491 | 0.2745 | 0.0000 | 0.0000 | 0.0291 | 0.0045 | 0.0045 | 0.0383 | below | 1.00 |
| 41 | tesseract (eng) + vision2 (corr off) + rapidocr | corroborated_trigrams | 0.6481 | 0.6562 | 0.6786 | 0.6053 | 0.6154 | 0.6397 | 0.5720 | 0.5874 | 0.6180 | below | 2.85 |
| 42 | tesseract (eng) + rapidocr | corroborated_trigrams | 0.2496 | 0.2491 | 0.2603 | 0.0000 | 0.0000 | 0.0071 | 0.0045 | 0.0045 | 0.0200 | below | 2.72 |
| 43 | tesseract (eng) + rapidocr + vision (corr on) | corroborated_trigrams | 0.4775 | 0.4992 | 0.4886 | 0.3124 | 0.3285 | 0.3171 | 0.2641 | 0.2910 | 0.2812 | below | 2.94 |
| 44 | tesseract2 (vie) + vision (corr on) | corroborated_trigrams | 0.9654 | 0.9641 | 0.8899 | 0.9801 | 0.9861 | 0.9034 | 0.9565 | 0.9674 | 0.9065 | pass | 2.45 |
| 45 | tesseract2 (vie) + vision (corr on) + rapidocr | corroborated_trigrams | 0.9681 | 0.9639 | 0.8899 | 0.9789 | 0.9861 | 0.9034 | 0.9738 | 0.9573 | 0.8870 | pass | 4.25 |
| 46 | tesseract2 (vie) + vision2 (corr off) | corroborated_trigrams | 0.9659 | 0.9641 | 0.8881 | 0.9830 | 0.9861 | 0.9034 | 0.9565 | 0.9674 | 0.9053 | pass | 2.37 |
| 47 | tesseract2 (vie) + vision2 (corr off) + rapidocr | corroborated_trigrams | 0.9686 | 0.9639 | 0.8861 | 0.9818 | 0.9861 | 0.9005 | 0.9738 | 0.9573 | 0.8835 | pass | 4.18 |
| 48 | tesseract2 (vie) + rapidocr | corroborated_trigrams | 0.5042 | 0.5387 | 0.5263 | 0.3324 | 0.4035 | 0.3921 | 0.2929 | 0.3847 | 0.3789 | below | 4.08 |
| 49 | tesseract2 (vie) + rapidocr + vision (corr on) | corroborated_trigrams | 0.9786 | 0.9752 | 0.8792 | 0.9997 | 0.9963 | 0.8821 | 0.9998 | 0.9922 | 0.8905 | below | 4.27 |
| 50 | vision (corr on) + rapidocr | corroborated_trigrams | 0.5042 | 0.4992 | 0.5246 | 0.3324 | 0.3285 | 0.3921 | 0.2929 | 0.2910 | 0.3594 | below | 2.18 |
| 51 | vision (corr on) + rapidocr + tesseract (eng) | corroborated_trigrams | 0.5042 | 0.4992 | 0.4885 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 3.03 |
| 52 | vision2 (corr off) + rapidocr | corroborated_trigrams | 0.5042 | 0.4992 | 0.5246 | 0.3324 | 0.3285 | 0.3921 | 0.2929 | 0.2910 | 0.3594 | below | 2.08 |
| 53 | vision2 (corr off) + rapidocr + tesseract (eng) | corroborated_trigrams | 0.5042 | 0.4992 | 0.4885 | 0.3324 | 0.3285 | 0.3171 | 0.2929 | 0.2910 | 0.2812 | below | 2.90 |

Twelve sets keep both floors: every set with Tesseract `vie` as the primary or sole reader (0, 8 to 14) and Tesseract `vie` corroborated by Vision (44 to 47). Vision alone falls below the letters floor on small pages (0.888). Any set that starts from RapidOCR's Latin recogniser stays near 0.33, whatever follows it. Tesseract `vie` corroborated by Vision trades a little word recall on clean and scanned pages (0.966 against 0.979) for more on small pages (0.888 against 0.875), which is what `maximin-v1` ranks by.

## Selection (`kit choose`, policy `maximin-v1`)

```json
{"darwin-arm64": {"chosen": 46, "runner_up": 44, "margin": -0.0018, "policy": "maximin-v1", "override_reason": null}}
```

`dropped_models`: none (the one model, `vie.traineddata`, is used by both reader sets). `unbenched`: `win-amd64`, `linux-x86_64`, `linux-aarch64`. `next` named `kit wheel`.

## `kit wheel` and `kit verify`

- `kit wheel`: wheel 12,463,954 bytes; canary `canary-vi.png` (11,134 bytes, 300 dpi) with `nước`, `Đất`, `hợp` and `4521`, all found by `tesseract2` and by `vision2`.
- `kit verify vi --download` in `<work>`: exit 0, checks manifest, download, layer and wheel all `ok`, no warnings.
- `kit verify vi --download --rebench` in `<fresh>` (empty): exit 0, all four checks `ok`, no warnings. Re-bench on `darwin-arm64`, tolerance 0.02 (the set has Vision): every measured value (words, trigrams, digits, letters and tones, in each condition) equals the claim; worst page 0.3255 as claimed; 2.37 s per page against the claimed 2.37. `drift: []`, `floors: []`.
- `kit verify vi --download --rebench --platform linux-x86_64` in `<fresh>`: exit 0, all four checks `ok`. It re-benched `default` and checked the floors only, with the warning "the pack claims no numbers for linux-x86_64, so only the must-survive floors were checked". Measured: words 0.9786, 0.9752 and 0.8752, letters 0.9997, 0.9963 and 0.9046, tones 0.9998, 0.9922 and 0.8921, the same as set 0 of the Mac's bench.

## Tests

The whole suite, run once at the end in the work tree (which holds the generated pack index and, besides `vi`, the other new packs being built at the same time): **3130 passed, 36 skipped, 1 failed**, in 219 s.

- The failure is `tests/test_release.py::test_index_regenerates_identically`: `build_index.py --check` reports "vi: version 1.0.0 has no wheel in dist", and the same for `es`, `fr`, `it` and `pt`. This is the one failure step 9 says to expect in a pack PR, so it isn't stop point 4.
- The skips, by reason:
  - 12 in `tests/e2e/test_run_e2e.py` and 1 in `tests/test_vision.py`: the test environment has no English data package installed (`base_pack_missing`);
  - 4 in `tests/e2e/test_sandbox_smoke.py`: they run only on a non-editable install (CI's sandbox job);
  - 1 in `tests/e2e/test_packs_network.py`: network tests run only with `PDFINTAKE_NETWORK_TESTS=1` (CI's network job);
  - 1 in `tests/test_release.py`: the pyproject pin is still the placeholder;
  - the rest are contract and unit cases that don't apply (a rule that takes no `tie_order` or no corroborators, Vision needing no model files, a signal handler already set, two record-schema cases).

## `kit report vi` body

#### Pack `vi` 1.0.0: Vietnamese

- Languages: vi (primary)
- Scripts: Latin
- Requires core: `>=0.1.0`

**Reader sets**

- `default`: tesseract (tesseract, reader), rapidocr (rapidocr, checker); pick `primary_with_floor` v1, fallback `none`
- `darwin-arm64`: tesseract2 (tesseract, reader), vision2 (vision, reader); pick `corroborated_trigrams` v1, fallback `none`

**Models and fonts**

| File | Kind | Bytes | Licence | Source |
|---|---|---|---|---|
| `vie.traineddata` | traineddata | 12435550 | Apache-2.0 | https://github.com/tesseract-ocr/tessdata_best/raw/e12c65a915945e4c28e237a9b52bc4a8f39a0cec/vie.traineddata |

- Canary: `canary-vi.png`, 11134 bytes
- Must survive: `letters` (word, floor 0.9), `tones` (word, floor 0.85)

**Corpus**

- `public-v1`: 8 texts, 2 fonts, pages {"form": 2, "microprint": 1, "mixed_script": 1, "prose": 4}
  - Nước: wikipedia `vi` revision 75422415, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75422415&action=raw
  - Mặt Trời: wikipedia `vi` revision 75671654, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75671654&action=raw
  - Mặt Trăng: wikipedia `vi` revision 75647291, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75647291&action=raw
  - Quang hợp: wikipedia `vi` revision 75051386, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75051386&action=raw
  - Cờ vua: wikipedia `vi` revision 75694353, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75694353&action=raw
  - Bánh mì: wikipedia `vi` revision 74936685, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=74936685&action=raw
  - Núi lửa: wikipedia `vi` revision 75664555, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75664555&action=raw
  - Sông: wikipedia `vi` revision 75668503, CC-BY-SA-4.0: https://vi.wikipedia.org/w/index.php?oldid=75668503&action=raw

**Bench**

- Kit 1.0, corpus `public-v1@1d0680d44adb2b7f7d42281454f42d8e6c4bff511740fb965db37ad0de6d60e7`
- Engines: rapidocr 3.10.0, tesseract 5.5.3 (tesserocr 2.12.0), vision rev3 (macOS 27.2)

`darwin-arm64`: worst page 0.326, 2.368 s/page, policy `maximin-v1`, margin -0.002

| Metric | clean | scan | small |
|---|---|---|---|
| digits | 0.996 | 0.816 | 0.898 |
| must-letters | 0.983 | 0.986 | 0.903 |
| must-tones | 0.957 | 0.967 | 0.905 |
| trigrams | 0.979 | 0.982 | 0.921 |
| words | 0.966 | 0.964 | 0.888 |

**Checklist**

- [ ] The PR changes only `packs/vi/**`.
- [ ] No binaries: no model, font, image or PDF.
- [ ] Every corpus text has an allowed licence, pinned by revision id and hash.
- [ ] `pdfintake kit verify vi --download` passes.
- [ ] The reader set is the one `pdfintake kit choose` picked, or the override is explained above.
- [ ] The must-survive characters were confirmed by someone who reads the language.
