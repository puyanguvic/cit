#!/usr/bin/env python3
"""One-click runner for CIT paper experiments.

Runs the main-paper E1/E2/E3 experiments (and optionally appendix runs).
All outputs are saved under results/<outdir>/seedK/...

Examples:
  python scripts/run_all.py --device cuda --seed 0
  python scripts/run_all.py --only e2 --plot
  python scripts/run_all.py --only appendix_e1,appendix_e4 --plot

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(pyfile: str, args: list[str]):
    cmd = [sys.executable, str(SCRIPTS_DIR / pyfile)] + args
    print("\n[RUN]", " ".join(cmd), flush=True)
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
        "--paper",
        dest="paper",
        action="store_true",
        help="Export paper-ready tables/figures after running (default).",
    )
    ap.add_argument(
        "--no-paper",
        dest="paper",
        action="store_false",
        help="Disable paper artifact export.",
    )
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
        help=(
            "Comma-separated subset of runs: e1,e2,e3,appendix_e1,appendix_e2,appendix_e3,appendix_e4 "
            "(legacy aliases: e0->appendix_e1, uci->appendix_e3, frontier/e4->appendix_e4). "
            "Default: run main-paper e1,e2,e3."
        ),
    )
    ap.set_defaults(auto_download=True, paper=True)
    args = ap.parse_args()

    raw_only = [x.strip().lower() for x in args.only.split(",") if x.strip()]
    only = list(raw_only)

    # Backward-compatible aliases.
    alias = {
        # old names
        "e5": "e2",
        "e0": "appendix_e1",
        "uci": "appendix_e3",
        "frontier": "appendix_e4",
        "e4": "appendix_e4",
        # convenience
        "appendix": "appendix",
        "all": "all",
    }
    only = [alias.get(x, x) for x in only]

    def _expand_only(xs: list[str]) -> list[str]:
        allowed = {"e1", "e2", "e3", "appendix_e1", "appendix_e2", "appendix_e3", "appendix_e4"}
        if not xs:
            return []
        if "all" in xs:
            return sorted(allowed)
        if "appendix" in xs:
            xs = [x for x in xs if x != "appendix"] + [x for x in sorted(allowed) if x.startswith("appendix_")]
        # Deduplicate, keep stable order
        out: list[str] = []
        for x in xs:
            if x in allowed and x not in out:
                out.append(x)
        return out

    only = _expand_only(only)
    if raw_only and not only:
        raise ValueError(f"Unknown --only entries: {raw_only}")

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

    want_plots = bool(args.plot or args.paper)

    for seed in seeds:
        common_seed = ["--seed", str(seed)]
        common_device_seed = ["--device", args.device, "--seed", str(seed)]

        # E1 (main)
        if enabled("e1"):
            run("e1.py", common_device_seed + ["--outdir", f"{args.outroot}/e1"])

        # E2 (main; token-fair only) + optional auto-download
        if enabled("e2"):
            e2_args = common_device_seed + ["--data-dir", args.e2_data_dir, "--outdir", f"{args.outroot}/e2"]
            if args.auto_download:
                e2_args.append("--auto-download")
            if want_plots:
                e2_args.append("--plot")
            run("e2.py", e2_args)

        # E3 (main; Pareto slice)
        if enabled("e3"):
            run(
                "e3.py",
                common_device_seed
                + [
                    "--outdir",
                    f"{args.outroot}/e3",
                    "--max-len",
                    "512",
                    "--model-families",
                    "mini,small",
                    "--total-tokens",
                    "5000000",
                ],
            )

        # Appendix E1: tokenizer playground scan (public tokenizers), CPU-only
        if enabled("appendix_e1"):
            e0_args = common_seed + ["--outdir", f"{args.outroot}/appendix_e1", "--data-dir", args.e2_data_dir]
            if args.auto_download:
                e0_args.append("--auto-download")
            run("appendix_e1.py", e0_args)

        # Appendix E2: CSIC step-fair variant (optional)
        if enabled("appendix_e2"):
            e2_step_args = common_device_seed + ["--data-dir", args.e2_data_dir, "--outdir", f"{args.outroot}/appendix_e2"]
            if args.auto_download:
                e2_step_args.append("--auto-download")
            if want_plots:
                e2_step_args.append("--plot")
            run("appendix_e2.py", e2_step_args)

        # Appendix E3: UCI benchmarks
        if enabled("appendix_e3"):
            run("appendix_e3.py", common_device_seed + ["--outdir", f"{args.outroot}/appendix_e3"])

        # Appendix E4: vocabulary-budget frontier sweep
        if enabled("appendix_e4"):
            frontier_args = common_device_seed + ["--vocabs", args.vocabs, "--outdir", f"{args.outroot}/appendix_e4"]
            if want_plots:
                frontier_args.append("--plot")
            run("appendix_e4.py", frontier_args)

    if args.paper:
        run("export_paper_artifacts.py", ["--run-root", args.outroot])
    print("\n[OK] All requested experiments finished.")
    if len(seeds) == 1:
        print(f"Results root: results/{args.outroot}/<experiment>/seed{seeds[0]}")
    else:
        print(f"Results root: results/{args.outroot}/<experiment>/seedK for K in {seeds}")

if __name__ == "__main__":
    main()
