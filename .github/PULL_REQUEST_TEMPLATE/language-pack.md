<!--
A language pack PR. Open it with this template (add ?template=language-pack.md to the compare URL).
It may change only packs/<id>/**. Anything else makes it a code PR: use code.md instead.
Follow docs/add-a-language.md; paste the output of `pdfintake kit report <id>` below.
-->

## Pack

- Language and pack id:
- New pack, or a change to an existing one (which version):

## `pdfintake kit report` output

<!-- Paste it whole. It holds the manifest summary, the bench numbers per platform, the selection and its margin, the must-survive coverage per condition, and what decided a tie-break. -->

## Checklist

- [ ] The PR changes only `packs/<id>/**`.
- [ ] No binaries: no model, font, image or PDF. The manifest pins them by URL, size and SHA-256 from the allowed upstream hosts.
- [ ] Every corpus text has a licence `docs/add-a-language.md` allows (public domain, CC0, CC BY, CC BY-SA, or government text excluded from copyright), pinned by revision id and hash. Nothing from a real person's documents.
- [ ] `pdfintake kit verify <id>` passes: schema, registries, source allowlist, the selection policy (`maximin-v2`), upstream download with hash checks, the layer round trip, the wheel and its canary.
- [ ] The reader set is the one `pdfintake kit choose` picked. If it isn't, the override reason is in the manifest and explained here (CI flags it for review).
- [ ] The must-survive characters were confirmed by someone who reads the language:
- [ ] Every must-survive group was measurable in at least one condition on every benched platform, or the report's warning is explained here.
- [ ] Optional `kit private` numbers, if any, came from born-digital pages only and hold numbers only.

## Anything the kit or the docs got wrong

<!-- Where docs/add-a-language.md was unclear or a command misbehaved. This is how the kit improves. -->
