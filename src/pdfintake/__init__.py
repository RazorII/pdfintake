"""pdfintake: turn scans, photos and PDFs into small, searchable PDFs."""

import os as _os

# The only version source: hatchling reads it, and scripts/release.sh greps this exact line.
__version__ = "0.3.0"

# onnxruntime's official builds (macOS, Linux and Windows) send usage telemetry to Microsoft by
# default, and keep a device id and an event queue under the user's home, from the moment
# onnxruntime is imported (measured with 1.31.1 on 2026-11-07). pdfintake reads documents on this
# machine and sends nothing anywhere, so it turns that off for its whole process before anything
# can import onnxruntime: onnxruntime reads the variable once, when it is imported, and setting it
# later changes nothing. The value is forced, not defaulted, so a "0" left in the environment
# can't turn it back on.
_os.environ["ORT_DISABLE_TELEMETRY"] = "1"
