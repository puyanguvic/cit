#!/usr/bin/env python3
"""Appendix E1: off-the-shelf tokenizer scan (public tokenizers).

Thin wrapper around run_e0_tokenizer_playground.py with paper-friendly defaults.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def main() -> None:
    args = list(sys.argv[1:])
    if not _has_flag(args, "--dataset"):
        args = ["--dataset", "csic2010"] + args
    if not _has_flag(args, "--split"):
        args = ["--split", "test"] + args
    if not _has_flag(args, "--data-dir"):
        args = ["--data-dir", "data/csic2010"] + args
    if not _has_flag(args, "--max-len"):
        args = ["--max-len", "512"] + args
    if not _has_flag(args, "--outdir"):
        args = ["--outdir", "paper/appendix_e1"] + args

    script = Path(__file__).resolve().parent / "run_e0_tokenizer_playground.py"
    cmd = [sys.executable, str(script)] + args
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

