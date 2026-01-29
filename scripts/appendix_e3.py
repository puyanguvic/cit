#!/usr/bin/env python3
"""Appendix E3: UCI tabular benchmarks (Adult / Credit-G)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)

def _strip_flag(argv: list[str], name: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == name:
            skip_next = True
            continue
        if a.startswith(name + "="):
            continue
        out.append(a)
    return out


def _run(dataset: str, extra_args: list[str]) -> None:
    script = Path(__file__).resolve().parent / "run_e2_uci.py"
    cmd = [sys.executable, str(script), "--dataset", dataset] + extra_args
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = list(sys.argv[1:])
    base_outdir = "paper/appendix_e3"
    for i, a in enumerate(args):
        if a == "--outdir" and i + 1 < len(args):
            base_outdir = args[i + 1]
            break
        if a.startswith("--outdir="):
            base_outdir = a.split("=", 1)[1]
            break

    # Avoid overwriting results by dataset; keep a clean subfolder per dataset.
    stripped = _strip_flag(args, "--outdir")
    _run("adult", ["--outdir", f"{base_outdir}/adult"] + stripped)
    _run("credit-g", ["--outdir", f"{base_outdir}/credit-g"] + stripped)


if __name__ == "__main__":
    main()
