#!/usr/bin/env python3
"""E2 (main): CSIC 2010 HTTP end-to-end, token-fair only.

Thin wrapper around run_e2_csic_http.py with paper-friendly defaults.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def main() -> None:
    args = list(sys.argv[1:])
    if not _has_flag(args, "--data-dir"):
        args = ["--data-dir", "data/csic2010"] + args
    if not _has_flag(args, "--outdir"):
        args = ["--outdir", "paper/e2"] + args
    if not _has_flag(args, "--budget-modes"):
        # Main paper uses token-fair only; step-fair is in appendix_e2.
        args = ["--budget-modes", "token"] + args

    script = Path(__file__).resolve().parent / "run_e2_csic_http.py"
    cmd = [sys.executable, str(script)] + args
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

