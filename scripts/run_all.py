#!/usr/bin/env python3
"""One-click runner for CIT paper experiments.

Runs E1/E2/E3/E4 and (optionally) plots frontier figures.
All outputs are saved under results/<outdir>/seedK/...

Examples:
  python scripts/run_all.py --device cuda --seed 0
  python scripts/run_all.py --only e4 --vocabs 256,512,1024,2048,4096 --plot

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(pyfile: str, args: list[str]):
    cmd = [sys.executable, str(SCRIPTS_DIR / pyfile)] + args
    print("\n[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outroot", type=str, default="paper", help="Root folder under results/ for this full run")
    ap.add_argument("--vocabs", type=str, default="256,512,1024,2048,4096", help="Vocab sweep list for E4")
    ap.add_argument("--plot", action="store_true", help="Generate plots where applicable (E4)")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of experiments to run: e1,e2,e3,e4,e5 (default: run all)",
    )
    args = ap.parse_args()

    only = [x.strip().lower() for x in args.only.split(",") if x.strip()]
    def enabled(k: str) -> bool:
        return (not only) or (k in only)

    common = ["--device", args.device, "--seed", str(args.seed)]

    # E1
    if enabled("e1"):
        run("run_e1_synth.py", common + ["--outdir", f"{args.outroot}/e1_synth"])

    # E2 (Adult + Credit-G)
    if enabled("e2"):
        run("run_e2_uci.py", common + ["--dataset", "adult", "--outdir", f"{args.outroot}/e2_uci"])
        run("run_e2_uci.py", common + ["--dataset", "credit-g", "--outdir", f"{args.outroot}/e2_uci"])

    # E3
    if enabled("e3"):
        run(
            "run_e3_pareto_prefix.py",
            common
            + [
                "--outdir",
                f"{args.outroot}/e3_pareto",
                "--full-finetune",
                "--max-len",
                "512",
                "--model-families",
                "mini,small",
                "--total-tokens",
                "5000000",
            ],
        )

    # E4 + plots
    if enabled("e4"):
        e4_args = common + ["--vocabs", args.vocabs, "--outdir", f"{args.outroot}/e4_frontier"]
        if args.plot:
            e4_args.append("--plot")
        run("run_e4_frontier.py", e4_args)

    # E5 (end-to-end on public phishing-email dataset; auto-download via HF datasets)
    if enabled("e5"):
        run(
            "run_e5_end2end_phish.py",
            common
            + [
                "--outdir",
                f"{args.outroot}/e5_end2end_phish",
                "--max_len",
                "512",
                "--model_families",
                "mini",
                "--total_tokens",
                "5000000",
            ],
        )

    print("\n[OK] All requested experiments finished.")
    print(f"Results root: results/{args.outroot}/seed{args.seed} (and per-experiment subfolders)")

if __name__ == "__main__":
    main()
