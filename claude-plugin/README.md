# pdfintake plugin for Claude

Turns scans, phone photos and PDFs into small, searchable PDFs, on your own computer. Several photos become one PDF without losing any quality, every page that's only a picture of text gets an invisible text layer so you can search and copy it, and the file is shrunk losslessly. A smaller, slightly compressed copy is made only when you ask for one. Your original file is never changed.

It works through a whole pile of files without stopping to ask you anything. Claude reads what you asked for, runs the batch, and gives you one short summary at the end: what was written, which files it skipped and why, and what it could still do if you want.

English is built in, and French, German, Hungarian, Italian, Polish, Portuguese, Spanish, Thai and Vietnamese are language packs, downloaded the first time a document needs one.

## Requirements

- A place where Claude can run commands: Claude Code on macOS, Windows or Linux, or claude.ai's own sandbox.
- Python 3.11 to 3.14. uv picks a suitable Python on its own.
- A Mac with Apple silicon and macOS 15 or newer, 64-bit Windows 10 or newer on an Intel or AMD processor, or 64-bit Linux (x86_64 or ARM). Intel Macs and Windows on ARM aren't supported yet, because two of the libraries pdfintake needs have no build for them.
- Internet access the first time, to install pdfintake and its libraries, and whenever a document needs a language pack that isn't downloaded yet.

## What the plugin runs and connects to

- **The plugin is one skill**: instructions that tell Claude how to use the `pdfintake` command. It has no hooks, no MCP servers and no background processes.
- **It runs one command, `pdfintake`**, MIT licensed, source at [github.com/RazorII/pdfintake](https://github.com/RazorII/pdfintake). If it isn't installed, Claude installs the release this plugin belongs to, pinned to its tag (`git+https://github.com/RazorII/pdfintake@v0.2.0`), with uv.
- **pdfintake does all its work on your computer.** OCR runs locally with Tesseract, RapidOCR and, on a Mac, Apple Vision. pdfintake never starts another program while it works, so no outside PDF or OCR tool ever touches your files.
- **It connects to these sites and no others:**
  - `github.com`: to install the command, and to download the English data and any language pack, which are files attached to pdfintake's own GitHub releases.
  - `release-assets.githubusercontent.com`: where GitHub serves those release files from (github.com redirects there).
  - `pypi.org` and `files.pythonhosted.org`: the Python libraries pdfintake needs (pikepdf, Pillow, Tesseract's Python binding and others), installed once by uv.

  No telemetry, no analytics, no other servers ([privacy policy](https://github.com/RazorII/pdfintake/blob/main/PRIVACY.md)).
- **Every download is checked.** Each language pack has a SHA-256 fingerprint built into pdfintake, and each file inside it is checked again every time it's used. GitHub's immutable releases mean a published pack can't be swapped later. A pack is data only (models, fonts and a test image) and is never run as code.
- **A language pack is downloaded only when a document needs it**, without asking, so a batch never stops. Say "don't download anything" and pdfintake stays offline, reading those pages with English instead. A pack is kept in pdfintake's cache, so it's downloaded once per computer. In claude.ai each new chat starts in a fresh sandbox, so pdfintake and its libraries (about 150 to 180 MB) are installed again there.
- **It writes only new files**, next to the original or in the folder you name: `<name> (searchable).pdf`, `<name> (compact).pdf`, `<name> (smaller).pdf` if you asked for smaller files, and `<name> (searchable copy).pdf` for originals pdfintake won't change (papers from the other side of a deal, evidence originals, signed, PDF/A and password-protected files). It never replaces an existing file unless you say so, and a searchable copy never replaces one at all.
- **Password-protected files.** Nobody is asked for a password during a run. pdfintake tries the passwords you saved before, in a file on your computer that only you can read. A file none of them opens is skipped and left exactly as it is, and Claude lists the skipped files once at the end. Give Claude the password then and it saves it for next time: it writes it to a temporary file, never into a command, has pdfintake add it to the saved list, and deletes the temporary file. A password you type in the chat stays in the chat. The searchable copy has no password, so search tools can read it, and so can anyone with the file. The original keeps its protection.

## Examples

- "Make this scan searchable: `scan-0042.pdf`." Claude writes `scan-0042 (searchable).pdf` beside it and tells you the size before and after and anything that didn't read cleanly.
- "Combine these four phone photos of the invoice into one PDF." Claude turns them into one upright, searchable PDF in the order you gave, with the photos' quality untouched.
- "Make everything in this folder searchable." Claude runs the whole folder in one go and reports once at the end: how many files were written, which ones need a password it doesn't have, and how many could be made smaller.
- "These letters are from my landlord: make them searchable." Claude leaves each original exactly as received and writes `<name> (searchable copy).pdf` beside it.
- "Make these small enough to email." Claude shrinks them losslessly and also saves a `(smaller).pdf` copy of each one where that saves a lot while still looking the same.
- "These are my own scans, and they're in Vietnamese." Claude reads them with the Vietnamese pack, downloading it first if needed, and says so in one line.
- "Give me a copy of this receipt I can share." Claude makes a clean copy with the personal metadata removed.
- "Here's the password for my bank statements." Claude saves it, runs the statements it skipped, and writes each `statement (searchable copy).pdf` without a password, telling you so in one line. The originals stay locked.

## Install

```
claude plugin marketplace add RazorII/claude-plugins
claude plugin install pdfintake@razorii
```

Source, issues and the full documentation: [github.com/RazorII/pdfintake](https://github.com/RazorII/pdfintake). MIT licensed.
