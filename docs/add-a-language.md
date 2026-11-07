# Add a language

These are instructions for Claude, helping a contributor add a language to pdfintake. Follow them in order. Every step is one command; it prints one JSON object and exits with a code from the table below. Read the JSON after every command before you go on.

A language is a **pack**: `packs/<id>/` holds a manifest, a corpus list, the bench numbers and the notices its models need, and nothing else. The kit picks the engines and writes the numbers by benchmarking the code `pdfintake run` itself uses, so you never hand-pick a reader set or type a number into a manifest. A pack PR touches only `packs/<id>/**`; anything else makes it a code PR (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

Below, `xx` is the pack id: the language's ISO 639 code (`es`, `pl`, or `sr-latn` for a variant; see step 2 on variants).

## Exit codes of every kit command

| Exit | Meaning | What you do |
|---|---|---|
| 0 | done, `"ok": true` | go to the next step |
| 1 | the command failed; `error` or `problems` in the JSON say why (for `kit verify`, also any failed check, even when the re-bench drifted too) | fix what it names, run the step again |
| 2 | bad arguments (including more `--must-find` words than a canary holds, or canary libraries at other versions than `tools/canary-env.txt` pins, step 8), or a kit file made by an older kit (steps 5 and 6) | correct the command, or do what the message says |
| 3 | setup: the English base pack or the `[kit]` extra is missing | run `scripts/dev-install.sh` again |
| 6 | a must-survive floor failed, or Tesseract has no model for the language | see the step |
| 7 | `kit verify --rebench` measured numbers outside tolerance of what the pack claims, and nothing else failed | see step 8 |
| 8 | `kit private` would have written something other than numbers; nothing was written | see step 7 |

Every command takes `--packs DIR` (default `./packs`), `--work DIR` (default the kit folder in pdfintake's cache, where downloads, rendered pages and wheels go) and `--json` (the output is JSON either way). Standard output is the one JSON object; a long command prints progress lines on standard error. No kit command prompts or reads standard input: every choice is a flag.

Most steps take seconds. Two take longer, so run them in the background and wait for them to exit rather than under a short command timeout: `kit bench` (step 5, about 15 minutes for the usual 54 sets on an idle laptop) and `kit verify --rebench` (step 8: about 15 to 40 seconds in a work folder that already holds the pages and models, and up to about a minute and a half in an empty one, because it renders the pages and fetches the models first).

**Uses are named with their model.** Wherever the kit shows a reader set to a person (progress lines, messages, `kit report`), each engine use is its id with the model or setting that tells it apart: `tesseract[vie]`, `tesseract[eng]`, `rapidocr[rapid-rec-latin]`, `vision2[vi-VT, no correction]`. Read the brackets, not the bare id: the same id can name different models in different sets (step 5).

## When to stop and get the person's decision

These four are for the contributor building the pack, never for a pdfintake run. At each one, stop, tell the person exactly what you found, and go on only once they have decided:

1. **A corpus licence you can't classify** as public domain, CC0, CC BY, CC BY-SA, or government text the law excludes from copyright (step 3). These are the licences the kit accepts (`public-domain`, `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `government-public`).
2. **An override** of the set `kit choose` picked, or **a change to a must-survive floor** after benching (step 6).
3. **The must-survive characters** of a language you aren't sure about: the person says which characters must come through OCR (step 2).
4. **A setup, test or CI failure you can't explain** (steps 1, 8 and 9).

## 1. Set up

```
git clone https://github.com/<you>/pdfintake && cd pdfintake
scripts/dev-install.sh
source .venv/bin/activate
pdfintake doctor --json
```

- `dev-install.sh` builds the English data wheel from its pinned upstream files and installs pdfintake, with the `kit` and `test` extras, into `.venv` in the clone. It needs [uv](https://docs.astral.sh/uv/).
- **Deactivate any other virtual environment first.** An environment that is active when the script runs takes the install, even another project's; the script says so on standard error.
- `pdfintake` is on the PATH only inside `.venv`: the script's last line says how to get there (`source .venv/bin/activate`; on Windows `.venv\Scripts\activate`). Without activating, run `.venv/bin/pdfintake` instead.
- `doctor` exits 0 when everything a run needs is in place, and its `engines` say which OCR engines this machine has (`"available": true`); the kit benches only those. **The kit needs less than a run does**: it takes the base pack from `packs/en` in the clone (and the base pack's files into its work folder) whenever the installed English data wheel can't be used, so it doesn't need the pack index either. So:
  - **Problems only in `index` and `base_pack` don't block the kit.** Go on to step 2. The two you'll meet:
    - `index_missing`: this checkout has no generated pack index yet.
    - `hash_mismatch` in `base_pack`: the installed English data wheel is older than `packs/en` (you pulled a commit that changed the English pack). Running `scripts/dev-install.sh` again rebuilds the wheel and clears it; `pdfintake run` needs that, the kit doesn't.
  - **Any other problem** (an unsupported platform, a missing `[kit]` extra): run `scripts/dev-install.sh` again. If it stays, that's stop point 4.
  - A kit command that exits 3 is a setup problem the kit itself can't work around; it names it.

## 2. Start the draft

```
pdfintake kit init xx --script Latin --must-survive acute=áéíóúñ:word:0.9
```

- It writes `packs/xx/manifest.draft.json`, an empty `packs/xx/corpus.json` and `packs/xx/NOTICE-tessdata`, and pins Tesseract's `tessdata_best` model for the language: a fixed commit, its size and SHA-256.
- `NOTICE-tessdata` is the Apache-2.0 notice the wheel ships for the pack's own Tesseract model: it names exactly the models the pack pins, each beside its pinned source. It follows `manifest.json` once there is one: when `kit choose` drops the model (step 6), it removes the notice, and when a model comes back it writes it again. Never edit it by hand; `kit verify` fails a notice that names other models. A non-Latin pack with `--font` also gets `OFL.txt`, copied from `packs/en`: replace its first line (the copyright holder of the English pack's fonts) with the one in your font's own `OFL.txt`, as `next` reminds you.
- `manifest.draft.json` is the kit's working copy: `kit candidates` and `kit bench` read it while it exists, and `kit choose` writes `manifest.json` from it. Keep it until the PR is done, and never commit it (step 9 lists what to commit).
- **A lost draft** (deleted, or never kept): `pdfintake kit init xx --from-manifest` rebuilds it from `manifest.json`. It keeps the manifest's readers, languages, fills, fonts and must-survive groups, leaves out the canary and the bench claims as a new draft does, and pins the language's Tesseract model again if `kit choose` dropped it, so the next bench proposes it again. Add `--force` when a draft is there; `manifest.json` itself isn't touched. The base pack (`en`) never has a draft: the kit reads and writes its `manifest.json` directly, so `kit init en --from-manifest` exits 2 (`kit_usage`).
- `--script` is the language's script, one of the names this core knows (`Latin`, `Thai`, `Greek`, `Cyrillic`, `Hebrew`, `Arabic`, `Devanagari`, `Han`, `Hiragana`, `Katakana`, `Hangul`).
- `--must-survive NAME=CHARS[:word|trigram|char[:FLOOR]]` names the characters the bench counts and the floor they must reach (default `word` and 0.9). Use `trigram` for a script written without spaces. Repeat it for several groups, each with its own floor. If you aren't sure which characters matter in this language, that's stop point 3.
- For a language the kit doesn't know, add `--iso639-3 <Tesseract's model name> --title <English name>`.
- A non-Latin script needs a layer font: add `--font <URL>` with a file from `github.com/notofonts` at a pinned commit or release (step 3 shows the URL form).
- The JSON's `vision_languages` lists the language lists Apple Vision would be called with on this Mac, one inner list each, the pack's language first and Vision's fallback after it: `[["es-ES", "en-US"]]` for Spanish. Vision's tag can differ from the usual one (`vi-VT` for Vietnamese); that's Apple's own and it works. The list is empty off a Mac and when Vision doesn't read the language.
- `next` lists everything still missing before you can bench, including the corpus while it has no texts or fonts (it names `kit corpus`, step 3).
- **Exit 6** means Tesseract has no model for the language. pdfintake can't read it with the engines it has: stop and tell the person.
- Run it again with `--force` to start the draft over; `corpus.json` and `bench.json` are kept.
- **Regional variants.** One pack serves a language when one Tesseract model and one set of must-survive characters fit all its written forms: `pt` covers European and Brazilian Portuguese (Tesseract has one `por`, and Wikipedia one edition). Give a variant its own id (`sr-latn`, `pt-br`) only when its written form differs enough to need another model, script or must-survive group. Vision may offer only one variant (`pt-BR`, not `pt-PT`); `vision_languages` shows which, and the bench measures it on whatever text you pin.

## 3. Pin the corpus

```
pdfintake kit corpus xx --wikipedia Q283,Q525,Q405,Q11982,Q718,Q7802,Q8072 --font https://github.com/notofonts/notofonts.github.io/raw/28b15b4b43b7bed62b5cf6e6b0b5ff5846270535/fonts/NotoSans/unhinted/ttf/NotoSans-Regular.ttf --font https://github.com/notofonts/notofonts.github.io/raw/28b15b4b43b7bed62b5cf6e6b0b5ff5846270535/fonts/NotoSerif/unhinted/ttf/NotoSerif-Regular.ttf
```

- `--wikipedia` takes Wikidata item ids. For each one the kit finds the article in the pack's language and pins its current revision id and the SHA-256 of that revision. Wikipedia text is CC-BY-SA-4.0. Pick five to ten items with long articles in the language.
- **Pick general topics**: science, nature, everyday things, games, arts. The ids above are water, the Sun, the Moon, photosynthesis, chess, bread and volcanoes. Never an article about a person, a country, a city or another place, a company, a law or a contract.
- The JSON's `pinned` lists every text with its `title`: check that each id led to the article you meant. `kit corpus xx` with no arguments prints it again without pinning anything; passing an id that is already pinned pins its current revision as a second text when the article has changed since.
- `--font` pins a font the bench pages are drawn with: a `github.com/notofonts` file at a pinned commit or release, as in the two URLs above (NotoSans and NotoSerif from the `notofonts.github.io` repository at a fixed commit). Give **at least two fonts for every script in the pack's `scripts`**:
  - a Latin pack: two Latin fonts, such as the two above;
  - a non-Latin pack: two for its script and two Latin ones, because its `scripts` lists Latin too. The same repository has a folder per family, at `fonts/<Family>/unhinted/ttf/<Family>-Regular.ttf`; use the families named for the script (a sans and a serif).

  `kit render` and `kit verify --rebench` stop with exit 1 (`kit_fonts_too_few`) when a script has fewer.
- **The mixed-script page needs no extra font.** Its second script is Latin for a non-Latin pack, and Greek (else Cyrillic) for a Latin pack; NotoSans and NotoSerif cover Latin, Greek and Cyrillic.
- `--url <https URL> --licence <licence> --title <title>` pins a plain-text file. The licence must be one of `public-domain`, `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `government-public`; any other is refused with exit 2. A text whose licence you can't classify is stop point 1. The file must come from a host the kit fetches corpus texts from, because CI fetches it again: Wikipedia, Wikisource, Project Gutenberg (`www.gutenberg.org`) or EUR-Lex (`eur-lex.europa.eu`). Any other host is refused with exit 2, and `kit verify` fails a `corpus.json` that names one; adding a host is a code PR.
- `--pages prose=4,form=2,microprint=1,mixed_script=1` sets how many pages of each kind to render (these are the defaults).
- The texts themselves are never committed: they're fetched by revision at bench time. Wikipedia's templates are flattened to text: those that are part of the sentence (a number's unit, a date, a no-break wrapper) keep their text, the rest (references, maintenance tags, infoboxes, language tags) are dropped; a page and its truth always hold exactly the same text.

## 4. Render the bench pages

```
pdfintake kit render xx
```

It draws each page kind with the pinned fonts (HarfBuzz shaping) and renders it three ways: `clean` at 300 dpi, `scan` at 200 dpi with a seeded scanner look, `small` at 110 dpi. The default pages make 24 images. The pages and their text go to the work folder, never into the repo. The kit always renders with the same seed, so the same corpus gives the same pages, and CI's re-bench re-renders exactly the pages your numbers came from.

- Exit 1 with `kit_fonts_too_few` names the script that needs another font: pin one in step 3 and render again.
- Exit 1 with `corpus_drift` means the text at a pinned revision changed upstream. Pin that item again in step 3 (`kit corpus xx --wikipedia <item>` pins its current revision), remove the old entry from `texts` in `packs/xx/corpus.json`, then render, bench and choose again.

## 5. List and bench the candidates

```
pdfintake kit candidates xx
pdfintake kit bench xx
```

- `candidates` asks every engine registered for this platform what it would read the language with, combines them with every pick rule that serves the script, and writes the list to the file its JSON names in `file` (in the work folder). Its JSON's `candidates` block says how the list was made:
  - `total`: the distinct sets proposed;
  - `refused_by_loader`: sets the manifest loader would refuse, never benched. It's normally 0. A pack-level repair whose `source` names a use (`diacritic_word` with `source: rapidocr`) is the one expected exception: the loader refuses every set without a use of that id, so those sets are counted here and not benched. Any other non-zero number is a kit bug worth reporting on the issue tracker;
  - `merged` and `merged_sets`: sets that are the same set once the loader fills in the defaults, whatever their use ids (the draft's own `tesseract` set and the generated `tesseract2` one, a set that only leaves out a default such as `timeout_s`, or one that lists a use's models in another order). Use ids count only where they change what a run does: `corroborated_trigrams` ranks the readers its `tie_order` leaves out by id, so two such sets whose ids sort differently stay apart. Each is benched once, under the lower index; `merged_sets` maps that index to the indices the others had before merging, and the set's entry in the JSON, `bench.json`, `kit choose` and the PR body name them (`merged`);
  - `cut` and `cut_by_rule`: sets dropped because each pick rule keeps at most 40 (`max_per_rule`), so no rule crowds out another;
  - `kept`: the sets that are benched, the same number as `count` (`total` less the refused, merged and cut ones).

  A candidates list made by an older kit (one without `merged_sets`) makes `kit bench` exit 2 with `kit_candidates_stale`: run `kit candidates` again.
- Each set in the JSON lists its use ids (`engines`), the same uses labelled with their model (`uses`) and each use's `params`. Set 0 is the draft's own reader set for this platform, with the draft's own ids (or `manifest.json`'s set when there's no draft). In the other sets, an engine that proposes several parameter sets gets its id numbered from the second on: `tesseract` is its first proposal and `tesseract2` its next. So **the bare id can name a different model in set 0 than elsewhere**: on a Vietnamese pack, `tesseract` in set 0 reads with `vie`, while `tesseract` in sets 1 to 7 reads with the base pack's `eng` and `tesseract2` is `vie`. The labels (`tesseract[vie]`, `tesseract[eng]`) tell them apart.
- `bench` downloads the pinned models (size and SHA-256 checked; `--offline` skips the download), reads every page with every set through `pipeline.read_pages`, the function `pdfintake run` calls, and writes `packs/xx/bench.json`: per platform and set, word, trigram and digit recall per condition, the must-survive recall, the worst page and the seconds per page. Each set in `bench.json` holds its whole `reader_set`.
- **What the numbers count.** The bench scores the text a run would put in the layer: the picked lines, then the checker's fills (the pack's `fills`, such as `numbers` and `ids`). So a set with a checker can score above the same reader alone even under the `single` rule, which picks the reader's lines unchanged: the checker's numbers and codes go into the layer as fills. Tesseract `vie` alone and with a RapidOCR checker read the same lines, and the checker's number fills lift digit recall on the small microprint page from 0.13 to 0.84.
- **A metric's condition figure is the mean of its pages.** For each condition, word, trigram and digit recall are the plain mean of the per-page recalls over that condition's pages, so one page that is barely read, such as the 110 dpi microprint page, pulls the whole condition down. A must-survive group is counted differently, over only what an engine read (step 6). `bench.json` keeps each page's numbers under the set's `pages`, which is where to look when a condition misses by a little.
- **Bench one pack at a time, on an otherwise idle machine.** `seconds_per_page` is recorded, published in the PR and the manifest, and it breaks ties in step 6; benches running side by side make every set slower, and can stall Apple Vision inside the bench process (sets with Vision taking minutes instead of seconds while the process sits at 0% CPU). If a Vision set takes minutes, stop the bench and start it again alone.
- **How long it takes**: on an idle Apple-silicon laptop, about 15 to 20 seconds per set for the default 24 pages, so the usual 54 sets take about 15 minutes; the first set also downloads the models, so its line and the first estimate run long. With other benches running, expect two to three times that. It prints one line per set on standard error (`kit bench: set 3 (4 of 54): tesseract2[vie] + rapidocr[rapid-rec-latin], primary_with_floor; ok, 21 s for 24 pages; about 18 min left`): `set 3` is the set's `index` in the JSON and `bench.json`, counted from 0, and `4 of 54` is how far the bench is. Run it in the background and wait for it to exit.
- In its JSON, a set's `ok` means every page was read. Whether the set keeps the must-survive characters is `floors_ok`: false when its must-survive figure is below a floor in any condition where the group is measurable (step 6), with the misses in `below_floor`. `kit choose` drops those sets. Each set also carries its `uses` and `pick`, so the JSON says which set is which.
- Bench on every machine you have (macOS adds Apple Vision). Each platform's numbers merge into `bench.json`. A supported platform you have no machine for gets no numbers: `kit choose` lists it under `unbenched`, CI checks it against the must-survive floors only where it runs on it, and the maintainer benches it on GitHub's runners before the pack ships. That's expected for a contributor with one machine.
- `--platform KEY` names another platform's key for its reader set; `--current-only` benches only the draft's own set.

## 6. Choose

```
pdfintake kit choose xx
```

It applies the policy `maximin-v2` per platform:
- drop every set with a must-survive figure below its floor in any condition where the group is measurable (below);
- keep the highest worst-condition primary metric (word recall for spaced scripts, trigram recall otherwise);
- break a tie (within 0.005 of the best) by the higher worst-condition digit recall, then by fewer engines, then by speed, then by the earlier set. Digit recall comes first because speed alone once picked a set that read numbers on scans at 0.82 over one at 0.98. The worst condition is the lowest of the conditions whose pages hold digits.

**What a must-survive figure counts.** The floor is there to catch an engine that reads a word and drops its accent or tone mark, not a page no engine can read at all. So only characters inside text an engine read count:
- a `word` group: the truth's words and the read's words are folded (accents and other marks removed, case ignored, and ł, đ, ø, ß, æ, œ and ı written as l, d, o, ss, ae, oe and i) and lined up. A truth word holding one of the group's characters counts only when the read has a word in its place with the same folded form. Of the group's characters in those words, the figure is the share the read kept exactly: `ő` read as `ö`, or `Ő` read as `ő`, is lost;
- a `trigram` or `char` group (a script without spaces, such as Thai): the text is cut into letters with their marks, and the letters alone (the skeleton) are lined up. A group character counts only inside a run of at least 4 letters whose skeleton the read matches, and it survived when the read has it on the same letter.

A condition's figure adds up all its pages: the characters that survived, divided by those inside read words (or runs). A condition where fewer than 20 of the group's characters were inside read words (or runs) is **not measurable**: it has no figure, its floor doesn't apply there, and `bench.json` still records its counts. For each set and group, `bench.json`'s `must_coverage` holds per condition `truth` (the group's characters in the truth), `read` (those inside read words), `survived` and `measurable`, and the PR body shows read / truth per condition. A group measurable in no condition passes, as a group the corpus never reached always has, but `kit choose` names it under `warnings` and the PR body flags it: tell the person, since its floor was never checked.

It writes `packs/xx/manifest.json` with the chosen reader sets and the `bench` block, and reports:
- `selections`: per platform, the chosen set's index and `chosen_set` (its labelled uses), with `chosen_merged` when other sets were merged into it, the runner-up (`runner_up`, `runner_up_set`), both scores, the margin, `ahead`, the chosen set's `must_coverage`, and `tie_break` when the tie-break decided:
  - **The runner-up** is the set the policy would have chosen without the chosen one: the next set in the policy's order (score, then the tie-break) that **reads differently**. Two sets read alike when every number the bench measured for them is the same, page by page: a second copy of one reader, or a reader with a checker that changed nothing. Those are one reader as far as the bench can tell, so neither is the other's runner-up. The runner-up isn't always the next-best score: a set that scored higher but lost the tie-break comes first only if the policy would take it next.
  - **The margin** is the chosen set's score minus the runner-up's. It's **negative** when the runner-up scored higher but lost on the tie-break (digit recall, then fewer engines, then speed), or when an override chose a lower-scoring set, and 0.0 when nothing else reads differently. The PR body explains this beside the number.
  - **`ahead`** lists every passing set that scored higher than the chosen one, with its score, so you see them whatever the runner-up is. Without an override those are the sets that lost on the tie-break; with one, they also include any set the override passed over.
  - **`tie_break`** is there when the tie-break decided: a set within 0.005 of the best score that reads differently lost, although the score alone, or fewer engines and speed alone, would have put it first. That is a set that scored higher, or one that had fewer engines, or as many and was faster, and lost on digit recall. Each such set is under `lost`, with its score, worst-condition digit recall (`digits`, `digits_cond`), engines, speed and `decided_by` (`digits`, `engines`, `speed` or `order`); the chosen set's own figures are under `chosen`, and a `summary` sentence the PR body repeats. A set in the window that the chosen one beats on score, engines and speed alike isn't listed: no order would have picked it.
- `warnings`: the must-survive groups the chosen set couldn't measure in any condition.
- The reader sets in the manifest: `default` is the first chosen set, taking Linux, then Windows, then macOS, that runs on every platform; a platform whose choice differs (a set with Vision on macOS, say) gets its own entry. A platform with no numbers reads with `default`. If no chosen set runs everywhere, `default` stays the draft's own set.
- `dropped_models`: models no chosen reader set uses. They leave `manifest.json` and the wheel, so nobody downloads a model that's never read; the draft keeps them, so a later bench still proposes them. `notice` says what happened to `NOTICE-tessdata` (`written`, `removed`, `unchanged` or `none`): it always names exactly the Tesseract models `manifest.json` keeps.
- `unbenched`: the supported platforms `bench.json` has no numbers for (see step 5).
- `next`: what to run next. It names `kit wheel` in three cases: the manifest has no canary yet (always so the first time); the chosen reader sets changed, so the canary was dropped (`"canary": "dropped"`); or models were dropped, so the wheel changed though the canary was kept.

Running it again later (after a re-bench, say) keeps the canary `kit wheel` pinned and the private block `kit private` wrote, unless the chosen reader sets changed. Exit 2 with `kit_bench_stale` means `bench.json` was made by an older kit, which counted must-survive characters differently, or that the manifest names a must-survive group `bench.json` has no numbers for (it was added after the bench): run `kit candidates` and `kit bench` again. Exit 2 with `kit_policy` means the manifest keeps numbers for a platform that were chosen under an older policy and `bench.json` has none for that platform: bench on it, then choose again. `kit verify` fails a manifest whose selection was made under another policy than `maximin-v2` and says to run `kit choose` again.

**Exit 6** means no set keeps the must-survive characters above their floors. The error names the sets that came closest, closest first (the smallest largest shortfall), each with its labelled uses, its pick rule and exactly where it misses: in `closest`, each miss is a group, a condition, the value and the floor (`ogonek small 0.8836 < 0.9`), and `reads_alike` lists the other sets that read exactly the same. Report those to the person, with the per-page numbers from `bench.json` when one page decides it (step 5). The person decides between three things (stop point 2):
- **a lower floor** for that group: rebuild the draft with the new floor, then run `kit choose xx` again. The bench stores recall numbers, not pass or fail, so a new floor needs no new bench. A change to the characters, or a new group, does: run `kit bench` before `kit choose`. **`--must-survive` replaces every group the draft has**, so repeat it once for every group, each with its characters and floor, the changed one with its new floor and the others as they were: a group you leave out is gone, and `kit choose` then passes without it (nothing flags a removed group). On a pack with a `letters` and a `tones` group:
  - when the pack has a `manifest.json`, prefer `pdfintake kit init xx --from-manifest --force --must-survive 'letters=CHARS:word:0.9' --must-survive 'tones=CHARS:word:0.85'`: it keeps everything else from the manifest;
  - without one, plain `pdfintake kit init xx --force` starts the draft from its flags alone, so pass `--script` again, every `--font` again (or the layer fonts are lost), `--iso639-3` and `--title` if step 2 needed them, and every group;
- **an override**, `--override PLATFORM=INDEX --override-reason "<why>"`: CI flags every override for the maintainer's review;
- **an engine gap**: the language needs another engine or model, which is a code change.

## 7. Optional: a private bench

```
pdfintake kit private xx --folder <a folder of the contributor's own born-digital PDFs>
```

Only pages with real text (not scans) are used, because their own text is the truth. Nothing but a page count and number triples (mean, worst, best per metric) is written, to `packs/xx/bench.private.json` and the manifest. A must-survive group is counted as in step 6, and a page counts for it only with at least 20 of the group's characters inside read words; when no page has that many, the pages are added up into one figure, and with fewer than 20 in all the group gets no triple. Nothing else leaves the machine. **Exit 8** means the result held something other than numbers and nothing was written; that is a kit bug: report it on the issue tracker.

## 8. Build the wheel and verify

```
pdfintake kit wheel xx --must-find <a word with the language's special characters> --must-find 4521
pdfintake kit verify xx --download
```

- `wheel` renders the canary line (the words every reader must find on every install), pins it in the manifest, reads the canary with every engine of **every reader set the pack names**, this platform's first (`--platform` picks which key counts as this one), and builds the data-only wheel in the work folder. An engine that can't run on this computer (Apple Vision on Linux, say) is skipped, not failed. Each entry of `canary_reads` names the platform key it was read for and the use with its label. Exit 1 with `kit_canary_failed` names each use that missed it, with its platform key, and the error's `canary_reads` lists what each use found and missed.
- **Every use reads the canary, checkers and corroborators included**, because a run reads the canary with its own platform's set and a single miss breaks the pack there. When a checker's recogniser can't write some of the language's marks (RapidOCR's Latin model has no ă, ơ or ư, and misreads ł, ż, ą and ź), choose words it does read: the vi, pl and hu canaries carry only the marks every one of their engines reads. The bench, not the canary, measures the rest.
- **When `wheel` fails, `manifest.json` is left as it was**, and a canary a use missed leaves no wheel. Pick other words, or fix what the error names, and run `wheel` again.
- **`--must-find` takes at most four entries, `4521` included**, each 1 to 40 characters: so three words of the language and `4521`. Choose the words that carry the most special characters between them. More is a bad argument: exit 2 before anything is downloaded or rendered.
- **`wheel` draws the canary only with the library versions `tools/canary-env.txt` pins** (fpdf2, uharfbuzz, Pillow, pypdfium2, pikepdf and fonttools). `dev-install.sh` and the pack-publish workflow draw every canary again in an environment built from that file, and a canary drawn with other versions can come out with other bytes, so its pin wouldn't reproduce. When a version here differs, or the file is missing, `wheel` exits 2 with `kit_canary_env` before anything is drawn and names each library. Install exactly those versions into the environment you run the kit from (`uv pip install -r tools/canary-env.txt` inside `.venv`) and run `wheel` again. Never edit the file: the maintainer owns it.
- `verify` runs four checks and reports each in `checks` as `ok`, `failed` or `skipped: <why>`:
  1. `manifest`: the manifest against the schema, the core's registries and the source allowlist, and each notice the pack's models name (it must name exactly those model files, each with its pinned source). A notice no model uses goes into `warnings`: don't commit it;
  2. `download`: every model and font the pack pins, fetched from upstream with `--download`. A file already in the work folder with its pinned size and SHA-256 isn't fetched again; every file's hash is checked either way. The base pack's fonts are fetched too, but a failure to fetch them shows under `layer` and `wheel`, which need them, not under `download`;
  3. `layer`: every must-survive character and the canary's words searchable in a PDF written with the pack's fonts;
  4. `wheel`: the canary re-renders to its pinned bytes, and the wheel builds and holds data only.
- With `--download` all four run, even in a fresh work folder: where `kit wheel` hasn't run (as in CI), `verify` renders the canary from the manifest and uses it only when its bytes equal the pin. Without `--download`, a check whose files aren't in the work folder is skipped. `ok` is true when nothing failed, and `warnings` lists every check that was skipped, so read `checks` and `warnings`, not only `ok`.
- CI runs `pdfintake kit verify xx --download --rebench --json` on Linux, Windows and macOS. `--rebench` renders the pages again and benches this platform's reader set again, then compares it with the numbers the manifest claims for that platform: 0.01 absolute tolerance, 0.02 for a set with Vision, and the must-survive floors exactly. In its JSON, `claimed.metrics` holds each metric as `[clean, scan, small]` (must-survive groups as `must-<name>`), while `measured` keeps the bench's shape (`metrics`, `must` and `must_coverage`, keyed by condition). A must-survive group that wasn't measurable in some condition when you chose has no `must-<name>` claim, so the re-bench checks it against its floors only, in every condition. A claimed group that falls under 20 read characters in a condition on the re-bench shows as `claimed …, not measured`, which is a difference (exit 7). **Exit 7** is a difference beyond tolerance, **exit 6** a floor missed, but only when the re-bench findings are the only problems: a failed check (download, layer or wheel) is exit 1 whatever the re-bench found. On a platform the manifest claims no numbers for, the re-bench checks the floors only, and its `warning` (and `warnings`) say so; `--platform KEY` re-benches another platform's set here, such as `default` from a Mac. Speed isn't compared. You can run it locally too.

## 9. Open the pull request

```
pdfintake kit report xx
```

- Its `body` is the PR text: the manifest summary with each use labelled, the pack's fills and repairs with their parameters, the models (or a line saying the pack pins none, with the base pack's models its readers use), the must-survive groups with their characters, the corpus with each text's title, the bench numbers per platform with the selection's margin and what its sign means, the set each platform reads with (and the sets merged into it), the must-survive coverage per condition (read / truth characters, with any condition that isn't measurable marked and a warning for a group measured nowhere), what decided a tie-break with its digit figures, the `default` set's numbers when no benched platform reads with it, and the platforms not benched yet. Open the PR from the **language-pack** template and paste the body into it.
- Its `commit` lists exactly the files to commit: the manifest, `corpus.json`, `bench.json`, the optional `bench.private.json`, and the notice files the pack's wheel carries. `leave_out` lists the other files in `packs/xx/`, such as `manifest.draft.json`: don't commit them. A `missing` notice is one the wheel needs and the folder lacks: run `kit choose xx` again, which writes `NOTICE-tessdata` (copy any other notice from `packs/en/`).
- Never commit a model, font, image or PDF.
- If you run the test suite: `tests/test_release.py::test_index_regenerates_identically` checks the generated pack index against `packs/`. A pack the index doesn't hold yet is added by the maintainer when the pack is published; the test passes and prints a line for it (`note: xx 1.0.0 isn't in the index yet ...`), so nothing is wrong. Any other failure is stop point 4.

CI then re-benches the pack on three operating systems. A failure you can't explain is stop point 4.
