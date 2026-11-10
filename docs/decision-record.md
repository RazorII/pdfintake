# The decision record

Every `pdfintake run` and `pdfintake apply` writes one **record** per output document. The record is the only interface between the CLI and anything that drives it, the Claude skill included. This page describes record version 1. The code that writes and checks it is `src/pdfintake/record.py`.

**A run never waits for an answer.** pdfintake doesn't ask anything during a run, and no record holds a question. Every choice comes from the flags the run was given or a default. What a person might still want to do afterwards (read a page again with another language, save a smaller copy) is listed in the record's `offers`, to report once after the whole batch. Nothing has to be done about an offer for a run to finish, and `run` never exits 10.

## Where records live

- Each record is saved to `<cache>/records/<id>.json`, written atomically. The cache is `PDFINTAKE_CACHE` when set, otherwise the platform's cache folder (`pdfintake doctor` prints it).
- `run --json` and `apply --json` print the same records on stdout as `{"exit": N, "records": [...]}`.
- `pdfintake record show <id> --json` prints a saved record again.

## Five rules

1. **A record never holds document text.** It holds counts, hashes, scores, codes and paths, never what a page says. OCR results are reduced to counts before they reach a record, and `record.validate` refuses any object key named `text`, `lines`, `words`, `near` or ending in `_text` anywhere in it. Characters left out of a text layer are named as code points (`"dropped": {"U+4E2D": 1}`), never as the characters themselves. File paths are kept (PRIVACY.md says so).
2. **Every vocabulary is closed.** Statuses, offer kinds, apply flags, option ids, output kinds, copy reasons, warning codes, note codes and usage codes come from fixed tables in core. A new code is a core change, never something a language pack can add.
3. **An offer is acted on with ids only.** If a person asks for it later, an offer is acted on with

   ```
   pdfintake apply <record id> <offer.apply_flag> <option id> --json
   ```

   The record id matches `^r-[0-9]{8}-[0-9a-f]{8}$`, the apply flag is `--choice` or `--lang`, and the option id matches `^(smaller|thorough|none|[a-z]{2,3})$`. No file name or other text from a record ever goes into the command. `apply` accepts a record id and nothing else: a path, `../x` or an id with a trailing newline is refused with exit 2.
4. **Nothing waits.** A record's exit says how the run went, never that something is waiting: a run that wrote every file, or skipped one that no saved password opened, exits 0 whatever offers it carries.
5. **A record never holds a password.** No field can take one:
   - `options` leaves out every run option whose name contains `password` (the saved passwords file's path included); a re-run reads the saved passwords again itself;
   - `record.validate` refuses any object key, anywhere, whose name contains `password` (a count keyed by a code such as `skipped_no_password` is not a password field);
   - every offer and every message is fixed core text with no slot for a value;
   - `record.save(rec, forbid=passwords)` refuses a record in which one of the saved passwords the run tried appears in a value that could carry it: step, verify and refusal text, a message that isn't core's default wording, a reader's error, or a key that isn't a plain name. The error says only how many places it found, and no problem `validate` reports ever repeats a password or the text around it.
   - The scan leaves out what existed before a password was read or can only hold a fixed value: the record's `id`, `parent` and `created` time, `history[]` times, flags, answers and children (each while it matches its own pattern), `tool`, `options`, `input` and `output` (paths, file names, hashes, kinds, copy reasons), `language.packs`, the smaller copy's `candidate`, `written` and `metric`, closed codes, any SHA-256 digest or record id, an offer of a known kind (core writes its words), a warning or note in core's default wording, and keys that are snake_case names or `U+XXXX` code points. In the text it does scan, an occurrence inside one of the record's own paths, file names, id or time doesn't count, and neither does a whole word that core's own wording uses (`password`, `copy`). So a PIN that is also the year, or account digits that are also in the file's name, never make a record impossible to save.
   - A password under four characters isn't scanned for, because it can't be told apart from the record's own words (`ok`, `pdf`); the points above still keep it out.

## Fields

A record has exactly these top-level fields, in this order. A field with nothing to say is `null`, `{}` or `[]`, never missing.

| Field | Type | Meaning |
|---|---|---|
| `record_version` | int | 1 |
| `id` | str | `r-YYYYMMDD-<8 hex>`; the date is the UTC day the record was made |
| `parent` | str or null | the record whose offer this `apply` acted on |
| `tool` | object | `version`, `platform` (a key such as `darwin-arm64`), `os`, `python`, `engines` (every registered engine's name → version string, or `"unavailable: <reason>"`; `dots` is thorough mode's reader, listed whether or not the mode ran) |
| `created` | str | UTC ISO 8601, for example `2026-11-05T09:30:00Z` |
| `status` | enum | `done`, `refused`, `failed`, `plan`, `resolved` (an offer on this record was acted on with `apply`; `history[]` says what was done) |
| `exit` | int | this record's exit code (below) |
| `provenance` | enum or null | `ordinary`, `counterparty`, `evidence`: from `--provenance`, which comes from the user's own words, otherwise `ordinary`. It is never asked. Null only when the run stopped before it read its options |
| `options` | object | the run options the run used, with `out` and `outdir` as strings or null (`apply` re-runs from these, to the same place). `thorough` is `true` for `--thorough`, `false` for `--no-thorough` and `null` when the saved setting decided; the `ocr` step's `detail.thorough` says whether the mode actually read the document |
| `input` | object | `paths[]`, `kind`, `pages`, `seal` (`sha256[]`, `sizes[]`, `mtimes_ns[]`, one entry per path; null before the inputs are sealed) |
| `output` | object or null | `path`, `sha256`, `bytes` and `kind`, always: `searchable` for an ordinary output, `searchable_copy` for a copy beside an original pdfintake doesn't change (below); null when nothing was written: nothing was worth writing, the file was refused, or no saved password opened it |
| `refusal` | object or null | `code` (a hard refusal code, below), `message`, `detectors` (`walk`, `raw`), `repaired` |
| `steps[]` | array | one row per step: `step`, `status` (`ok`, `skipped`, `changed`, `clear`, `failed`), `detail` |
| `verify[]` | array | each check: `name`, `ok`, `detail`, `page` |
| `language` | object | `plan` (`confident`, `uncertain`, `hinted`, `forced`, `not_needed`), `packs[]`, `canaries[]`, `detection`. `uncertain` means detection wasn't sure, so the likeliest language read the page (warning `language_uncertain`, and a `language` offer) |
| `pages[]` | array | per OCR'd page: `page`, `class`, `dpi`, `orientation`, `script`, `languages`, `pack`, `composition`, `readers[]`, `pick` (no lines), `fills` (counts by kind), `repairs`, `must_survive`, `retention`, `warnings[]` (codes). In thorough mode `readers[]` includes `dots` (its seconds and status) and the fills it added count under their kind |
| `shrink` | object | the lossless shrink's report |
| `metadata` | object | what happened to the metadata: `kept`, `changed`, `dropped` (names below) |
| `smaller` | object or null | the smaller copy's measurements: `offered`, `candidate`, `saving`, `metric`, `bar`, `min_score`, `pages[]`, `kept_lossless[]`, `text_layer_identical`, `metadata`, `applied`, `written` |
| `offers[]` | array | what a person could do next, never waited on: each `{kind, apply_flag, summary, options[]}`, an option being `{id, label, description}` (below) |
| `warnings[]` | array | `{code, message, page}` with `code` from the warning codes below |
| `notes[]` | array | `{code, message, page}` with `code` from the note codes below |
| `history[]` | array | `apply` actions on this record: `at`, `flag`, `answer`, `result`, `child` |

Page numbers in a record count from 1.

### `steps[]` details

A step's `detail` holds that step's own facts (counts, page numbers, flags, pack ids, codes) and, like the rest of the record, never document text (rule 1). Two steps say whether they reused the input's cache (`<cache>/work/`, see PRIVACY.md), so a re-run or an `apply --lang` shows where its result came from:

- `detect`: `pages`, how many pages detection covered, and `cached`, true when its result came from the input's cache, so no page was rendered or read again for it.
- `orientation`: `baked` (pages whose content was redrawn at the angle the checks found plus any rotation flag they had, the flag then set to 0), `cleared` (pages whose rotation flag was reset to 0 because their content was already upright), `kept` (pages a turn was called for but that keep their annotations, so they weren't turned: warning `not_turned_annotations`), `voter` and `osd` (whether each orientation check's reader opened; true on a cache hit, since only a pass with both open is stored), and `cached`, true when the checks' results came from the input's cache.

`cached` is always false for a password-protected or edit-locked input, which is never cached.

### `output` of a searchable copy

A file from the other side, an evidence original, or a signed, certified, PDF/A, PDF/UA-needing-a-layer or encrypted file is never changed. pdfintake writes a lossless searchable copy beside it, `<name> (searchable copy).pdf` (or in `--outdir`), exit 0. Its `output` reads:

```json
{"path": "/Users/example/Desktop/Lease (searchable copy).pdf", "sha256": "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "bytes": 1820417,
 "kind": "searchable_copy",
 "original": {"name": "Lease.pdf", "sha256": "9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f"},
 "copy_reasons": ["signed", "provenance_counterparty"]}
```

- `original.name` is the original's file name only, never its folder (`record.validate` refuses a name with `/` or `\` in it); `original.sha256` is its SHA-256 as sealed.
- `copy_reasons` holds at least one code from `refusals.COPYABLE`: `signed`, `certified`, `usage_rights`, `timestamped`, `pdfa`, `pdfua_needs_layer`, `encrypted`, `provenance_counterparty`, `provenance_evidence`. Each appears once, in the order of `refusals.CODES`. `record.validate` reads `COPYABLE` from `refusals` when it runs, so the two can't drift, and it never accepts a hard code (`xfa`, `xmp_unreadable`, `detectors_disagree`, `damaged`): those still refuse the file (exit 4).
- Only a `searchable_copy` carries `original` and `copy_reasons`.
- A copy never replaces a file: `--overwrite` with a copy is the usage error `copy_never_overwrites` (exit 2). No smaller copy is offered for a copy.
- A file that needs a password to open and isn't opened by any saved password is skipped, not copied (below).
- Its notes say what the copy doesn't carry: `signature_not_carried`, `conformance_claim_dropped` and `copy_unlocked`.

### `metadata` names

`kept`, `changed` and `dropped` list the parts of the metadata by name:

- `Info` appears whole only when it is unchanged, in `kept`. `XMP` always appears whole: in `kept`, `changed` or `dropped`.
- When the Info dictionaries differ, `Info` isn't listed; each Info key is listed on its own as `Info/<Key>` (for example `Info/Producer` in `changed`, `Info/Author` in `kept`, and every key in `dropped` when the output has no Info at all). A key that only the output has counts as changed.
- A part neither file has isn't listed. The PDF header version isn't metadata here: `shrink` records it.

A fresh record says `failed` with exit 1 until the run records its outcome, so a record left behind by a crash never claims success.

## Offers

**A run never waits for an answer.** `offers` lists what a person *could* do next with a written output. pdfintake never stops for them, the skill never asks about them during or before a batch, and a run's exit doesn't depend on them. Report them once, after the whole batch. If the person later asks for one, run the `apply` command above.

| Kind | Apply flag | Option ids | When |
|---|---|---|---|
| `language` | `--lang` | another pack's language code | pdfintake chose a language for some pages without asking, and another pack could read them again. Four cases: detection wasn't sure, so the likeliest language read them (warning `language_uncertain`); the language or writing system detection found has no pack yet, so English read them (`no_pack_for_language`); the pack they need isn't installed and the run didn't download anything, so English read them (`pack_not_installed`); or that pack's download or check failed this run, so English read them (`pack_broken`). The summary says which. Each option reads those pages again with that pack; one that isn't installed downloads it once. |
| `smaller` | `--choice` | `smaller` | a smaller copy would save at least 30% (no byte floor) with every changed page scoring at or above the bar. Only the lossless file was written. `smaller` saves `<output stem> (smaller).pdf` next to it. The staged candidate is kept for 14 days; after that `apply` rebuilds it. Never offered for a searchable copy, and not offered when `run --smaller` already wrote the copy. |
| `thorough` | `--choice` | `thorough` | this Mac (Apple silicon) can read in thorough mode, and the person hasn't chosen yet. The summary says what the mode needs (the `pdfintake[thorough]` libraries and a one-time 4.4 GB download from Hugging Face), what it costs (about 17 s more a page) and what it gives (numbers and lines in other scripts the regular readers miss, and on Thai pages a read the other readers back up; now and then a wrong extra word or number in the invisible text). `thorough` reads the document again with dots.mocr beside the regular readers, once the mode is set up with `pdfintake thorough enable` (exit 3 naming the setup otherwise). Not offered once the mode is on, off or turned down, nor off Apple silicon. |

An offer reads, for example:

```json
{"kind": "language", "apply_flag": "--lang",
 "summary": "Page 2 was read as Croatian, the likeliest language, but pdfintake wasn't sure. Page 2 can be read again as Bosnian or Serbian if you ask.",
 "options": [
   {"id": "bs", "label": "Bosnian (download 9 MB)", "description": "Reads page 2 again with the Bosnian language pack, downloading it once (9 MB)."},
   {"id": "sr", "label": "Serbian", "description": "Reads page 2 again with the Serbian language pack, already installed."}]}
```

- Every offer is about a written output: a record without one carries none, and it offers a smaller copy at most once.
- The summary, labels and descriptions are core's words. Show them as they are.
- Provenance and passwords are never offers or questions: `--provenance` comes from the user's own words (otherwise `ordinary`), and passwords come from the saved list.

### Passwords

pdfintake never asks for a password. It tries the passwords the user saved once (one per line in `--passwords-file PATH`, or the default saved passwords file), in file order, and uses the first that opens the file. The file must belong to the user and be readable by them alone, or pdfintake won't read it (usage error `passwords_file_unsafe`, exit 2). pdfintake never records, logs or reports which password worked, and never writes one into a record.

- **A saved password opens the file:** it gets a searchable copy, saved without a password so it can be searched (`output.kind` `searchable_copy`, copy reason `encrypted`, note `copy_unlocked`). The original keeps its password.
- **The saved passwords file can't be used:** usage error `passwords_file_unsafe` (exit 2); the usage codes below list every reason.
- **No saved password opens it:** the file is skipped. Nothing is written, the original is untouched, and the record says `skipped_no_password`, with exit 0. The batch carries on. List every skipped file once, after the batch, with the offer to save a password.
- **Locked only against editing** (an owner password, no open password): no password is needed; the file gets a searchable copy saved without encryption, with note `copy_unlocked`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | done; read the warnings and the offers. A file skipped because no saved password opened it is exit 0 too (`skipped_no_password`) |
| 1 | failed (the record says why) |
| 2 | usage: bad arguments, an input that is missing (`input_missing`), isn't a regular file (`input_not_file`: a folder, a pipe or a device) or isn't a PDF or an image pdfintake reads (`unsupported_input`), each in that input's own record; an unknown record id, a flag that doesn't fit the offer, `--overwrite` with a searchable copy (`copy_never_overwrites`), an output that is open in another program on Windows (`output_in_use`), a saved passwords file pdfintake won't use (`passwords_file_unsafe`) |
| 3 | setup: base pack missing or broken, unsupported platform, broken model install; for `apply --choice thorough`, thorough mode not set up here yet (`thorough_not_ready`, the message names `pdfintake thorough enable`) |
| 4 | refused; nothing written. For `run`: a hard refusal code (below); signed, certified, PDF/A, encrypted, counterparty and evidence files get a searchable copy instead, exit 0. For `apply --choice smaller`: the original, read again, has a hard code, or it is kept as a searchable copy, which never gets a smaller version (`no_smaller_for_a_copy`) |
| 5 | stale: an input, output or candidate changed since the record; run again |

With several records, the process exits with the highest-priority code among them, in the order 2, 3, 1, 4, 5, 0. Read every record, whatever the process exit.

A `run` or `apply` stopped by SIGTERM or SIGHUP (a closed terminal, a service manager stopping it) removes its work files and any half-written output, and the process exits with 128 plus the signal number: 143 for SIGTERM, 129 for SIGHUP. It prints no records then; run the same files again.

## Refusal codes

A refused file's `refusal.code` is a hard code from `refusals.HARD`. Nothing is written and the original is untouched (exit 4):

- `xfa`: an XFA form, whose pages the form draws when it is viewed, so its content can't be checked.
- `xmp_unreadable`: the XMP metadata is there but can't be read, so pdfintake can't tell whether the file is archival or otherwise protected.
- `detectors_disagree`: the object walk and the raw byte scan don't name the same signatures, certification, encryption or archival markers (for example, signature data left in an earlier version of the file), so pdfintake doesn't guess.
- `damaged`: the file fails the structure checks before any work. pikepdf can't open it at all, it was cut short or lost its end (no trailer and no cross-reference stream left), or a file that needed repair couldn't be read through. A file qpdf repairs from a broken cross-reference table (a bad `startxref`, a missing xref table) and then reads in full is not damaged: it is processed, with the warning `input_repaired`. A file with no trailer and no cross-reference stream is damaged even when qpdf rebuilds it and reads all of it. When a damaged file's detectors also disagree, the code is `damaged`.

The copy reasons (`signed`, `certified`, `usage_rights`, `timestamped`, `pdfa`, `pdfua_needs_layer`, `encrypted`, `provenance_counterparty`, `provenance_evidence`) never refuse: they put the file in copy mode (`output` of a searchable copy, above).

`apply --choice smaller` checks the original again before it writes. A hard code refuses it as above, and `no_smaller_for_a_copy` means the file is kept as a searchable copy, which never gets a smaller version. That refusal stops the command before any record is written: `apply --json` prints `{"exit": 4, "records": [], "error": {...}, "refusal": {"code": ..., "message": ...}}`, and the record the offer came from is unchanged.

## Warning codes

A warning is something the user should hear about. Show every one, always; a degraded reader is never hidden.

| Group | Codes |
|---|---|
| readers | `reader_failed`, `reader_timeout`, `fallback_used`, `agreement_low`, `no_layer_readers_failed`, `vision_language_unsupported`, `thorough_failed`, `thorough_unavailable` |
| language | `mixed_language_page`, `mixed_script_page`, `mixed_script_no_pack`, `pack_broken`, `language_uncertain`, `no_pack_for_language`, `pack_not_installed` |
| pages | `existing_layer_replaced`, `layer_shrank`, `existing_layer_kept`, `image_text_not_read`, `garbled_text`, `clip_text_kept` |
| layer | `layer_dropped_chars` |
| orientation | `orientation_disagree`, `not_turned_annotations` |
| input | `input_repaired`, `contains_jbig2`, `revisions_dropped`, `objects_dropped`, `render_capped`, `metadata_format_limit` |
| export | `exporting_a_copy` (export only: the export was made from a searchable copy, not from the original) |

`layer_dropped_chars` reads "Some characters couldn't be written into the text layer." It counts characters with no glyph in the layer's fonts and characters of lines whose box couldn't be placed.

The language warnings that replace what used to be a question:

- `language_uncertain`: detection wasn't sure, so the page was read with the likeliest language. The message names it and the other candidates, and the record carries a `language` offer.
- `no_pack_for_language`: the page's language has no pack yet, so it was read with English. The message names the language, or the writing system when detection found only that ("text in the Ethiopic writing system"); point to `docs/add-a-language.md` once, after the batch. When another installed or published pack could read the page, the record also carries a `language` offer.
- `pack_not_installed`: the page's pack isn't installed and the run didn't download anything (`--fetch never`), so it was read with English. The record carries a `language` offer to read it again with that pack, which downloads it once if the person asks.
- `pack_broken`: a language pack failed its check (its download, hash check or canary). When the base pack read a page instead, the message names the language and the record carries a `language` offer to read the page again with that pack, which fetches and checks it again.

A page can carry `language_uncertain` together with `no_pack_for_language`, `pack_not_installed` or `pack_broken`: detection wasn't sure, and its likeliest language couldn't be used. Its `language_uncertain` message then names the likeliest language and says what the page was read as instead.

## Note codes

A note is information, reported in one summary line:

- `input_unchanged`, `nothing_written`, `vision_not_on_platform`, `pack_fetched`, `mixed_script_unchecked`, `rotate_kept`.
- `render_scaled_down`: a page was rendered smaller so that no side passes 32767 pixels, the OCR engine's limit.
- For a searchable copy: `signature_not_carried` ("The signed original is the one that counts; this copy is for finding it."), `conformance_claim_dropped` (the copy no longer claims PDF/A or PDF/UA), and `copy_unlocked` ("The copy has no password, so anyone with the file can read it."). A copy is always saved without encryption, so it can be searched; always report `copy_unlocked` in one line.
- `skipped_no_password`: the file needs a password to open and no saved password opened it, so nothing was written and the file is untouched (exit 0).

## Usage codes

These usage errors (exit 2) have codes of their own, with core's wording in `USAGE_MESSAGES` in `record.py`. In a record the code is in the failed step's `detail.code`, with the message beside it:

- `copy_never_overwrites`: `--overwrite` was given for a searchable copy. A copy never replaces an existing file.
- `output_in_use` (Windows): the output to be replaced (`--overwrite`, or `apply` replacing its own output) is open in another program, and still was after pdfintake retried for two seconds. The message names the file. Nothing was written; close the file there and run again.
- `passwords_file_unsafe`: pdfintake won't use the saved passwords file as it is. The message names the file and the one reason it was turned down, never a password. The reasons `passwords.py` reports:
  - it doesn't exist (a `--passwords-file` that isn't there; a missing default file just means there are no saved passwords);
  - it isn't a regular file;
  - it can't be read;
  - other users can read or change it (on macOS and Linux any group or other permission bit; 0600 and 0400 pass);
  - it belongs to another user;
  - it isn't inside the user's folder (Windows, where that folder's permissions are what keep it private);
  - it is larger than 128 KB;
  - it holds more than 100 different passwords, the most pdfintake tries;
  - it already holds 100, so `passwords add` can't append another;
  - it isn't UTF-8 text.

  `passwords.py` words each reason; `USAGE_MESSAGES` holds core's fallback wording for the code, true whatever the reason.
- `input_missing`: an input doesn't exist or can't be read. That input gets its own record with exit 2, and the rest of the batch carries on.
- `input_not_file`: an input isn't a regular file: a folder, a pipe (FIFO) or a device such as `/dev/stdin`. pdfintake never opens it, because reading one would make the run wait. Its own record, exit 2; the batch carries on.
- `unsupported_input`: an input isn't a PDF or an image pdfintake can read. Its own record, exit 2; the batch carries on.

Other usage errors (an output name already taken, an output that would land on an input, a bad flag) have a code too, and their message says what to change.

Each code's message is written in core (`WARNING_MESSAGES`, `NOTE_MESSAGES` and `USAGE_MESSAGES` in `record.py`). Show the message the record carries; don't rebuild it from the code.

## Example

A finished record with a `smaller` offer (made-up values, shortened):

```json
{
  "record_version": 1, "id": "r-20261105-1a2b3c4d", "parent": null,
  "tool": {"version": "0.1.0", "platform": "darwin-arm64", "os": "macOS 15.4", "python": "3.12.7",
           "engines": {"tesseract": "5.5.3 (tesserocr 2.12.0)", "vision": "rev3 (macOS 15.4)", "rapidocr": "3.10.0"}},
  "created": "2026-11-05T09:30:00Z", "status": "done", "exit": 0, "provenance": "ordinary",
  "options": {"provenance": "ordinary", "fetch": "auto", "smaller": false},
  "input": {"paths": ["/Users/example/Desktop/Receipt.pdf"], "kind": "pdf", "pages": 3,
            "seal": {"sha256": ["9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f"], "sizes": [4813220], "mtimes_ns": [1100000000000000000]}},
  "output": {"path": "/Users/example/Desktop/Receipt (searchable).pdf", "sha256": "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "bytes": 4302118,
             "kind": "searchable"},
  "refusal": null, "steps": [], "verify": [], "language": {"plan": "confident"}, "pages": [],
  "shrink": {}, "metadata": {"kept": ["Info", "XMP"], "changed": [], "dropped": []}, "smaller": {"offered": true, "bar": 90, "min_score": 92.4},
  "offers": [{"kind": "smaller", "apply_flag": "--choice",
    "summary": "A smaller copy could be saved as well: 72% smaller (4.3 MB to 1.2 MB). Its most detailed part scores 92 out of 100; at 90 or above, a difference shows only when you flick between the two.",
    "options": [
      {"id": "smaller", "label": "Save the smaller copy too",
       "description": "Saves Receipt (searchable) (smaller).pdf next to it. Receipt (searchable).pdf and Receipt.pdf stay as they are."}]}],
  "warnings": [], "notes": [{"code": "input_unchanged", "message": "The input file was not modified.", "page": null}],
  "history": []
}
```

The run is finished as it stands. If the person later asks for the smaller copy, the command is `pdfintake apply r-20261105-1a2b3c4d --choice smaller --json`.

## Compatibility

Record version 1 fields are never removed or renamed, and their meaning never changes. A new offer kind, warning code or note code is added in core and listed here and in the skill; a reader that meets a code it doesn't know shows the record's message as it is.
