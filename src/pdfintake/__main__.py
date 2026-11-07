"""``python -m pdfintake``: the same entry point as the ``pdfintake`` command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
