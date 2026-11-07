# Security

Please report a security problem privately, not in a public issue: open the repository's **Security** tab and choose **Report a vulnerability**. Only the maintainer sees the report, and you'll get a reply there.

pdfintake runs on your own computer and sends nothing anywhere (see [PRIVACY.md](PRIVACY.md)). It does open files from anywhere, and it downloads language packs, so these are the things worth reporting:

- **A crafted PDF or image** that makes pdfintake, or a library it uses, run code, start a process, read or write outside the input and the output it names, change the input, or hang or exhaust memory despite its limits (images over 200 megapixels are refused by name, and page renders are capped at 60 megapixels).
- **A way past the refusals and the copies**: an original that pdfintake changes or overwrites instead of writing a searchable copy beside it, an XFA file it doesn't refuse, or a copy that still carries signature material.
- **A way past the pack chain of trust**: a language pack file used without its SHA-256 matching, a download followed off the allowed hosts, a pack member outside its manifest, a path that escapes the cache (`..` or an absolute path), or a `.py`, `.pth` or native library inside a data wheel that gets past the checks.
- **A pack whose data can do more than data should**: a model, font or test image that makes Tesseract, onnxruntime, fontTools, fpdf2 or Pillow misbehave, or pack text that reaches Claude or the user without core's wording.
- **Document text in a record.**
- **A password that leaks**: one that shows up in a process list, the environment, a record, a log or an error message, or is written by pdfintake anywhere but the saved passwords file when you ask it to add one; a saved passwords file that pdfintake reads although others can read or change it, or creates with wider access than 0600; a temporary password file the Claude skill leaves behind; and an unlocked working copy of a password-protected file left in the cache after its run ends.
- **A run that stops to ask**: anything in `pdfintake run` that prompts, reads standard input for an answer, or waits, since a run is meant to work through a batch in the background.
- **A file name or record field that ends up inside a command** the Claude skill builds. Acting on an offer is meant to be built from ids only.

One gap is known and documented: `uv tool install` doesn't check the `#sha256=` pin on the English data wheel. pdfintake checks every file of that wheel itself before using any of them, and GitHub's immutable releases stop a published file from being replaced, but a file that runs when Python starts would run before those checks. A report that widens this is welcome.

Fixes go out as a new release. A published release is never edited: GitHub's immutable releases lock its files and its tag, so a fix is always the next version, and the plugin's skill installs that version.
