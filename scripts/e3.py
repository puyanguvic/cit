#!/usr/bin/env python3
"""E3 (main): Pareto slice via model-size scaling.

Thin wrapper around run_e3_pareto_prefix.py with paper-friendly defaults.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def main() -> None:
    args = list(sys.argv[1:])
    if not _has_flag(args, "--outdir"):
        args = ["--outdir", "paper/e3"] + args
    if not _has_flag(args, "--full-finetune"):
        # Probe-only collapses on random backbones; paper numbers assume full FT.
        args = ["--full-finetune"] + args

    script = Path(__file__).resolve().parent / "run_e3_pareto_prefix.py"
    cmd = [sys.executable, str(script)] + args
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

