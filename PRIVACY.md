# Privacy

pdfintake and its Claude plugin collect no data.

- **Your files stay on your computer.** pdfintake reads them, runs OCR and writes the new PDF there. No page, image or piece of text ever leaves it.
- **pdfintake sends nothing anywhere.** No telemetry, no analytics, no crash reports, no account.
- **The only network traffic is downloading pdfintake and its data.** It connects to these hosts and no others:
  - `github.com`: installing the command from [github.com/RazorII/pdfintake](https://github.com/RazorII/pdfintake), and downloading the English data and the language packs, which are files attached to that repository's releases.
  - `release-assets.githubusercontent.com`: where GitHub serves those release files from.
  - `pypi.org` and `files.pythonhosted.org`: the Python libraries pdfintake needs, installed by uv.

  GitHub and the Python Package Index see what any download shows them: your IP address and the file you asked for. Their own privacy policies cover that ([GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement), [PyPI](https://policies.python.org/pypi.org/Privacy-Notice/)).
- **A language pack is downloaded only when a document needs it**, without asking, so a batch never stops. `--fetch never` keeps pdfintake fully offline once it's installed. The request names only the pack, never anything about your document.
- **What pdfintake keeps on your computer.** In its cache folder (`pdfintake doctor` prints where):
  - a **record** of each run, in `records/`. It holds the input and output **file paths**, their sizes and SHA-256 fingerprints, page counts, the languages and readers used, scores, warnings, the options the run used, and any offer you later acted on. **It never holds document text or a password**: OCR results are reduced to counts before the record is saved.
  - working files in `work/`. A run's own working files are deleted when the run ends. What stays is a cache for each input, so running it again, or reading its pages again with another language, doesn't redo the slow first pass:
    - pictures of its pages;
    - what detection found, **which includes the text its first reading pass read on each page**;
    - the two orientation checks' results: the turns they weighed, their scores and the name of the writing system they saw, never any text;
    - for a PDF built from images, the date it was first built, and nothing else.
  - a staged smaller copy in `staging/`.
  - downloaded language packs in `packs/`, with `tessdata/` (the pack files Tesseract reads) and `canary/` (each pack's self-check results, from the pack's own test image, never from your documents).

  The cache in `work/` and the staged copies in `staging/` are deleted 14 days after they were last written, the next time `pdfintake run` starts. If a run is killed before it can clean up, the next run removes the working files it left (on Windows, the 14-day clean-up does).

  **A password-protected or edit-locked file is never cached.** Its pictures, reads and votes would come from its unlocked pages, so all of its working files, the unlocked copy included, are deleted as soon as its run ends.

  Delete the cache folder to remove all of it.
- **Passwords.** pdfintake never asks for a password. It tries the ones you saved, one per line in a private file: `passwords` in pdfintake's config folder, or the file you name with `--passwords-file`. It won't read that file unless only you can read and change it (mode 0600 on macOS and Linux; inside your user folder on Windows). A password never goes on the command line or into the environment, and pdfintake never puts one in a record, a log or an error message, or says which one opened a file. A file none of them opens is skipped and left exactly as it is.
  - **Saving a password** (`pdfintake passwords add --from-file FILE`) appends it to the saved file, creating that file as private if it isn't there yet. That file is the only place pdfintake ever writes a password. It doesn't delete the file you added from: that's yours, or Claude's, to remove.
  - **The searchable copy has no password**, so Spotlight and other search tools can read it, and so can anyone who has the file. The original is never changed and keeps its protection.
  - While it works, pdfintake keeps an unlocked working copy in its cache (in a folder only you can open), and deletes it when the run ends, whether it worked or not.
  - pdfintake never opens a keychain or password manager. If you keep your passwords in one, a script of your own can write them to a private temporary file for each batch, pass `--passwords-file`, and delete it afterwards.
  - **Through Claude**, a password you type in the chat stays in that conversation, under Anthropic's privacy policy (below). Claude writes it to a temporary file, never into a command, saves it with `pdfintake passwords add`, and deletes the temporary file straight away.
- **The language kit.** `pdfintake kit private` benchmarks a pack on a folder of your own documents. The file it writes holds numbers only (it refuses to write anything else), and nothing from that folder leaves your computer unless you choose to share those numbers.
- **Claude itself is covered by Anthropic's privacy policy.** When you use pdfintake through Claude, what Claude does with your conversation, including any file you share in it, falls under [Anthropic's privacy policy](https://www.anthropic.com/legal/privacy), not this one.

Questions go to [Discussions](https://github.com/RazorII/pdfintake/discussions/categories/q-a).
