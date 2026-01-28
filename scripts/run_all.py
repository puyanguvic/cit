#!/usr/bin/env python3
"""One-click runner for CIT paper experiments.

Runs the main-paper E1/E2/E3 experiments (and optionally appendix runs).
All outputs are saved under results/<outdir>/seedK/...

Examples:
  python scripts/run_all.py --device cuda --seed 0
  python scripts/run_all.py --only e2 --plot

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
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seed list (overrides --seed), e.g., 0,1,2",
    )
    ap.add_argument("--outroot", type=str, default="paper", help="Root folder under results/ for this full run")
    ap.add_argument("--vocabs", type=str, default="256,512,1024,2048,4096", help="Vocab sweep list (appendix frontier)")
    ap.add_argument("--plot", action="store_true", help="Generate plots where applicable (CSIC/frontier)")
    ap.add_argument(
        "--e2-data-dir",
        "--e5-data-dir",
        dest="e2_data_dir",
        type=str,
        default="data/csic2010",
        help="Data directory for E2 (CSIC 2010 HTTP).",
    )
    ap.add_argument(
        "--auto-download",
        dest="auto_download",
        action="store_true",
        help="Auto-download CSIC 2010 data for E2 (default).",
    )
    ap.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Disable auto-download for E2.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of runs: e0,e1,e2,e3,uci,frontier (default: run all main-paper E1/E2/E3)",
    )
    ap.set_defaults(auto_download=True)
    args = ap.parse_args()

    only = [x.strip().lower() for x in args.only.split(",") if x.strip()]

    # Backward-compatible aliases:
    # - old 'e5' == CSIC HTTP (now paper E2)
    # - old 'e4' == frontier sweep (appendix)
    alias = {"e5": "e2", "e4": "frontier"}
    only = [alias.get(x, x) for x in only]

    def enabled(k: str) -> bool:
        if not only:
            return k in {"e1", "e2", "e3"}
        return k in only

    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        if not seeds:
            raise ValueError("Empty --seeds list")
    else:
        seeds = [int(args.seed)]

    for seed in seeds:
        common_seed = ["--seed", str(seed)]
        common_device_seed = ["--device", args.device, "--seed", str(seed)]

        # E0 (tokenizer playground scan; CPU-only)
        if enabled("e0"):
            e0_args = common_seed + ["--outdir", f"{args.outroot}/e0_tokenizer_playground"]
            # Align with CSIC dataset used in E2 by default.
            e0_args += ["--dataset", "csic2010", "--data-dir", args.e2_data_dir]
            if args.auto_download:
                e0_args.append("--auto-download")
            run("run_e0_tokenizer_playground.py", e0_args)

        # E1
        if enabled("e1"):
            run("run_e1_synth.py", common_device_seed + ["--outdir", f"{args.outroot}/e1_synth"])

        # E2 (CSIC 2010 HTTP) + optional auto-download
        if enabled("e2"):
            e2_args = common_device_seed + ["--data-dir", args.e2_data_dir, "--outdir", f"{args.outroot}/e2_csic_http"]
            if args.auto_download:
                e2_args.append("--auto-download")
            if args.plot:
                e2_args.append("--plot")
            run("run_e2_csic_http.py", e2_args)

        # E3 (Adult: accuracy/latency Pareto slice)
        if enabled("e3"):
            run(
                "run_e3_pareto_prefix.py",
                common_device_seed
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

        # Appendix: utility on tabular datasets (legacy script name)
        if enabled("uci"):
            run("run_e2_uci.py", common_device_seed + ["--dataset", "adult", "--outdir", f"{args.outroot}/appendix_uci"])
            run("run_e2_uci.py", common_device_seed + ["--dataset", "credit-g", "--outdir", f"{args.outroot}/appendix_uci"])

        # Appendix: vocabulary-budget frontier sweep
        if enabled("frontier"):
            frontier_args = common_device_seed + ["--vocabs", args.vocabs, "--outdir", f"{args.outroot}/appendix_frontier"]
            if args.plot:
                frontier_args.append("--plot")
            run("run_e4_frontier.py", frontier_args)

    print("\n[OK] All requested experiments finished.")
    if len(seeds) == 1:
        print(f"Results root: results/{args.outroot}/<experiment>/seed{seeds[0]}")
    else:
        print(f"Results root: results/{args.outroot}/<experiment>/seedK for K in {seeds}")

if __name__ == "__main__":
    main()
