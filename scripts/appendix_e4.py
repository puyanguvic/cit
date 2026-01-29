#!/usr/bin/env python3
"""Appendix E4: vocab-budget frontier sweep."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def main() -> None:
    args = list(sys.argv[1:])
    if not _has_flag(args, "--outdir"):
        args = ["--outdir", "paper/appendix_e4"] + args

    script = Path(__file__).resolve().parent / "run_e4_frontier.py"
    cmd = [sys.executable, str(script)] + args
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

