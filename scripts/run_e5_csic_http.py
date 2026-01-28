"""Deprecated wrapper for the CSIC 2010 HTTP experiment.

Paper numbering was re-anchored to E1/E2/E3. The CSIC experiment is now E2:
  python scripts/run_e2_csic_http.py ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    scripts_dir = Path(__file__).resolve().parent
    target = scripts_dir / "run_e2_csic_http.py"
    raise SystemExit(subprocess.call([sys.executable, str(target)] + sys.argv[1:]))
