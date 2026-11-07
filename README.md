# pdfintake

[![test](https://github.com/RazorII/pdfintake/actions/workflows/test.yml/badge.svg)](https://github.com/RazorII/pdfintake/actions/workflows/test.yml)

Turn scans, phone photos and PDFs into small, searchable PDFs, on your own computer.

```
pdfintake run 'Receipt.pdf'
```

- **Images become a PDF without losing anything.** A JPEG goes in byte for byte. A PNG or TIFF keeps its bit depth, colour space and transparency. A sideways photo is turned by the page, not by re-encoding its pixels. HEIC, AVIF and lossy WebP, which a PDF can't hold as they are, are encoded once.
- **Pages that are only pictures of text get an invisible OCR text layer**, so you can search and copy them. Pages that already carry real text are never read again, so printed words can't be overwritten.
- **The file is shrunk losslessly**, and never made bigger.
- **A smaller copy is made only when you ask for it** (`--smaller`), and only where it's worth it: at least 30% smaller, with its most detailed part still scoring 90 or more out of 100 on the SSIMULACRA2 scale. It's saved beside the full-quality file. Without the flag, the record says which files could be made smaller, so you can ask later.
- **Every build is checked before it's written**: each page renders the same as the input, every image is identical, the text layer is all there and on the page, and the file's structure is sound. If any check fails, nothing is written.
- **The input is never modified.** pdfintake fingerprints it before it starts and checks it again before it writes.
- **A run never stops to ask.** Every choice comes from a flag or a default, so a batch of hundreds of files runs in the background and finishes.

English is built in, and French, German, Hungarian, Italian, Polish, Portuguese, Spanish, Thai and Vietnamese are language packs, downloaded the first time a document needs one. Anyone can add a language with the kit in this repo ([docs/add-a-language.md](docs/add-a-language.md)).

## Why

Scanned paperwork piles up as PDFs you can't search, phone photos you can't search either, and files far bigger than they need to be. The usual fixes trade one problem for another: OCR tools that re-encode every image, "compressors" that quietly blur the small print, and converters that turn a sideways photo by recompressing it. pdfintake does the three jobs without losing anything, and takes the one step that does lose a little only when you ask for it.

## Install

You need Python 3.11 to 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv tool install git+https://github.com/RazorII/pdfintake
pdfintake doctor
```

`pdfintake doctor` checks that the OCR engines, the English data and the cache work, and says what's wrong if they don't. If the shell says `pdfintake` isn't found, uv put it in `~/.local/bin`, which isn't on your PATH yet: `uv tool update-shell` fixes that.

pipx and pip work too, with an explicit Python 3.11 to 3.14 (`pipx install --python python3.12 git+https://github.com/RazorII/pdfintake`). To update later: `uv tool upgrade pdfintake`.

### Platforms

| Platform | Machine | Supported | Minimum OS | Notes |
|---|---|---|---|---|
| `darwin-arm64` | Mac with Apple silicon (Apple Vision available) | yes | macOS 15 | |
| `win-amd64` | 64-bit Windows on Intel or AMD | yes | Windows 10 | |
| `linux-x86_64` | 64-bit Linux on Intel or AMD | yes | manylinux | |
| `linux-aarch64` | 64-bit Linux on ARM | yes | manylinux | CI row optional |
| `darwin-x86_64` | Mac with an Intel processor | no | | pikepdf 10.17 has no Intel-Mac wheel |
| `win-arm64` | Windows on ARM | no | | tesserocr has no win_arm64 wheel |

Three limits come from libraries pdfintake depends on: pikepdf sets the macOS floor (15, Apple silicon), onnxruntime has no Python 3.15 build yet (so Python stays below 3.15; a weekly CI job watches for it), and tesserocr has no Windows ARM build.

## Use

```
pdfintake run 'scan.pdf'                                 # writes "scan (searchable).pdf" beside it
pdfintake run 'p1.jpg' 'p2.jpg' 'p3.jpg'                 # one PDF, named after the first image
pdfintake run 'p1.jpg' 'p2.jpg' --separate               # one PDF per image
pdfintake run 'lease.pdf' --provenance counterparty      # from the other side: a searchable copy beside it
pdfintake run 'a.pdf' 'b.pdf' --smaller                  # also a smaller copy where it's worth it
pdfintake run 'scan.pdf' --outdir 'out' --json           # machine-readable records
pdfintake run 'scan.pdf' --plan                          # what it would do and download; writes nothing
pdfintake passwords add --from-file 'pw.txt'             # save a password for later runs
pdfintake apply r-20261105-1a2b3c4d --choice smaller     # act on a run's offer later: the smaller copy
pdfintake apply r-20261105-5e6f7a8b --lang hr            # act on a run's offer later: read again as Croatian
pdfintake doctor --canary                                # test every engine on a known image
pdfintake packs list                                     # language packs, installed or available
pdfintake record show r-20261105-1a2b3c4d                # a saved record again
pdfintake export clean 'scan (searchable).pdf' -o 'share.pdf' --provenance ordinary
pdfintake --version
```

**Output names.** `<name> (searchable).pdf` when a text layer was added, `<name> (compact).pdf` when the file was only shrunk, `<name> (smaller).pdf` for the smaller copy, and `<name> (searchable copy).pdf` for an original pdfintake won't change (below). When neither a layer nor a 1% saving comes out of it, nothing is written and pdfintake says why. An existing file is never replaced unless you pass `--overwrite`.

**Inputs.** Images given together become one PDF, in the order given (`--separate` for one each). PDFs are never merged, and a run takes either PDFs or images, not both. `--paths-from FILE` reads one UTF-8 path per line, for names that are awkward to quote. Only regular files are read: a folder, a pipe or a device such as `/dev/stdin` is never opened, so it can't make a run wait (its record says `input_not_file`, exit 2, and the rest of the batch carries on).

**Languages.** pdfintake detects each page's writing system and language. `--lang CODE` sets the language of pages it was unsure about, `--force-lang CODE` sets every page, and `none` means no text layer for those pages. When detection isn't sure, the likeliest language reads the page, and the record says so and offers to read it again as another. `--fetch auto` (the default) downloads every pack a document needs, without asking. `--fetch never` works offline and reads pages whose pack isn't installed with English, with a warning.

**Existing text layers.** A page whose only text is an invisible layer from another tool has that layer replaced (`--keep-existing-layer` leaves it). A page pdfintake already read is left alone (`--redo-ocr` reads it again).

### Nothing is asked during a run

pdfintake never prompts and never waits for an answer. It writes a JSON **record** for every document ([docs/decision-record.md](docs/decision-record.md)), and what you might still want to do is listed in the record's `offers`, to act on later or never:

- **Language**: some pages could be read again with another language pack: `pdfintake apply <record id> --lang <code>`.
- **Smaller copy**: a smaller copy would save at least 30% and still look the same: `pdfintake apply <record id> --choice smaller`.

Everything else is a flag:

- **`--provenance counterparty` or `evidence`** for papers from the other side of a contract, deal or dispute, or originals you keep as evidence. Without it the run is `ordinary`, which loses nothing: no run ever changes an input.
- **Passwords** come from a saved list, never from a question (below).

### Password-protected files

pdfintake tries the passwords you saved, one per line, in a private file: `passwords` in pdfintake's config folder, or the file you name with `--passwords-file`. It reads that file only when it belongs to you and nobody else can read or change it (mode 0600 on macOS and Linux; inside your user folder on Windows), and tries at most 100 passwords. `pdfintake passwords add --from-file FILE` appends a password to the saved list and creates it as a private file if needed. A password never goes on the command line or into the environment, and never into a record, a log or an error message.

A file none of the saved passwords opens is skipped: nothing is written, the file is left as it is, and the record notes `skipped_no_password` with exit 0, so the rest of the batch carries on. Save the password and run that file again.

### Files pdfintake won't change

Some originals must stay exactly as they are: papers from the other side of a deal or dispute, originals you keep as evidence, signed and certified PDFs, PDF/A and PDF/UA files, and password-protected files. pdfintake never changes those. It writes a separate, lossless `<name> (searchable copy).pdf` beside each one instead, and offers no smaller version. The record's `output.kind` says `searchable_copy` and lists why in `copy_reasons`.

- **Signed files.** A signature can't survive an added text layer, so the copy shows the stamp as it was drawn but carries no signature. The signed original is the one that counts; the copy is for finding it.
- **PDF/A and PDF/UA files.** An added layer breaks those standards, so the copy no longer claims them. The original still does.
- **Password-protected files.** Opened with a saved password (above), or without one when the file is only locked against editing. The copy is saved without a password or edit lock, so Spotlight and other search tools can read it, and so can anyone who has the file. The original keeps its protection.
- **Names.** A searchable copy never replaces an existing file, so `--overwrite` doesn't work for one (exit 2, `copy_never_overwrites`).
- **Sharing.** A copy can be exported, with a warning that it isn't the original. For evidence, send the original.

XFA forms are still refused, because their pages are drawn only when they're viewed, and so is a file the two independent detectors disagree about, or one too damaged to check (`damaged`: it won't open, it was cut short, or it couldn't be read through after a repair). A file that only needed a small repair is processed, with a warning.

### Exit codes

| Exit code | Meaning |
|---|---|
| 0 | Done (read the warnings and offers anyway), including a file skipped because no saved password opened it |
| 1 | A file failed, including a failed check or an input pdfintake can't place without losing data |
| 2 | Bad arguments, an output that exists or would hit an input, mixed PDFs and images, an input that is missing or isn't a regular file (`input_not_file`: a folder, a pipe or a device), `--overwrite` for a searchable copy (`copy_never_overwrites`), an output open in another program on Windows (`output_in_use`), a saved passwords file pdfintake won't use (`passwords_file_unsafe`); for `apply`, an unknown record or a flag that doesn't fit the offer |
| 3 | Setup: the English data missing or broken, an unsupported platform, a broken model |
| 4 | Refused, nothing written: for `run`, an XFA form, metadata that can't be read, a damaged file, or one the two detectors disagree about; for `apply --choice smaller`, the same checks on the original again, or a file kept as a searchable copy, which never gets a smaller version (`no_smaller_for_a_copy`) |
| 5 | Stale: the input changed during the run, or since the record |
| 6 | Kit only: a must-survive or metric floor failed |
| 7 | Kit only: the re-bench differs from the pack's claim beyond tolerance |
| 8 | Kit only: the privacy guard tripped in `kit private` |

With several files, each output gets its own record, and the process exits with the most important code among them (2, then 3, 1, 4, 5, 0).

Stopped by SIGTERM or SIGHUP (a closed terminal, a service manager stopping it), `run` and `apply` remove their work files and any half-written output, then exit with 128 plus the signal number (143 for SIGTERM, 129 for SIGHUP). Run the same files again.

## How it reads text

- **Engines.** Tesseract (through tesserocr, in process), RapidOCR on onnxruntime, and on a Mac, Apple Vision. Each language pack says which engines read its pages on each platform and how their reads are combined, chosen by benchmark, not by hand.
- **Language packs are data only**: models, fonts and a test image, with a SHA-256 for every file. The pack list and each pack's fingerprint ship inside pdfintake, and every file is checked every time it's used. Packs are attached to this repo's GitHub releases, which are immutable: once published, a release's files and tag can't change.
- **No other programs.** While it runs, pdfintake never starts another process (no Ghostscript, no `ocrmypdf`, no command-line `tesseract`), and a runtime guard enforces it.
- **Apple Vision** is tested on every release on a real Mac, because GitHub's macOS machines may not run it.

## What it doesn't do (yet)

- **Sign or certify** an exported copy. That's planned for a later version.
- **Read pictures on pages that also carry printed text** (a photo with a caption, a stamped header). Such pages are left as they are, with a warning.
- **Repair a broken font map** in a born-digital PDF whose text comes out garbled. It's detected and reported.
- **Perspective-correct or deskew** a phone photo.
- Hold 16-bit colour images with transparency, 16-bit colour TIFF, or 12-bit and lossless JPEG exactly. Those fail with a clear message rather than lose data.

## Use it with Claude

pdfintake comes with a Claude plugin. Its skill tells Claude when to use pdfintake, how to run a whole batch without stopping to ask, and how to report it all in one summary at the end, every warning included. It needs somewhere Claude can run commands: Claude Code on macOS, Windows or Linux, or claude.ai's own sandbox.

```
claude plugin marketplace add RazorII/claude-plugins
claude plugin install pdfintake@razorii
```

The plugin installs pdfintake on first use if it isn't there yet, pinned to the release the plugin belongs to. What it runs and downloads is listed in [claude-plugin/README.md](claude-plugin/README.md).

Claude Code doesn't update plugins from this marketplace on its own until you let it: in `/plugin`, open **Marketplaces**, pick **razorii** and choose **Enable auto-update**. Without that, update by hand with `claude plugin update pdfintake@razorii`.

## Development

```
git clone https://github.com/RazorII/pdfintake && cd pdfintake
scripts/dev-install.sh
pytest
```

`scripts/dev-install.sh` builds the English data wheel from its pinned upstream sources and installs pdfintake in editable mode with the test and kit extras. The tests build their own images, PDFs and fonts in code, so the repo carries no binaries and no real documents. CI runs the suite on macOS, Windows and Linux.

Adding a language, an engine, a pick rule or a platform: [CONTRIBUTING.md](CONTRIBUTING.md).

## Questions and bugs

Ask questions in [Discussions](https://github.com/RazorII/pdfintake/discussions/categories/q-a). Report bugs in [Issues](https://github.com/RazorII/pdfintake/issues), with the `--json` record if you can (records hold file paths, counts and hashes, never document text). Security problems: [SECURITY.md](SECURITY.md).

## Privacy

pdfintake collects nothing and sends nothing anywhere; your files are read and written on your own computer. It goes online only to install and to download a language pack. See [PRIVACY.md](PRIVACY.md).

## License

MIT. The libraries and data pdfintake uses keep their own licences: [THIRD-PARTY.md](THIRD-PARTY.md).
