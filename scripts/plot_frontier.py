"""
Plot empirical interface frontier curves from results.csv.

This script reads a run directory (e.g., results/e4_frontier/seed0) produced by
run_e4_frontier.py and saves publication-ready plots (PDF + PNG) into the same
directory.

Outputs:
  - frontier_acc_vs_len.{pdf,png}
  - frontier_acc_vs_distortion.{pdf,png}
  - frontier_len_vs_distortion.{pdf,png}
  - frontier_tokenize_ms_vs_len.{pdf,png}   (optional systems proxy)

Notes:
  - We deliberately do NOT hardcode colors (matplotlib defaults) to keep styling neutral.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "NA")), []).append(row)
    return out


def plot_scatter(groups: dict[str, list[dict]], *, x_key: str, y_key: str, title: str, xlabel: str, ylabel: str, outpath: Path):
    plt.figure()
    for name, rows in sorted(groups.items()):
        xs = [to_float(r.get(x_key)) for r in rows]
        ys = [to_float(r.get(y_key)) for r in rows]
        # Sort by x to make trend easier to see
        pairs = sorted(zip(xs, ys), key=lambda t: (t[0], t[1]))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        plt.plot(xs, ys, marker="o", linestyle="-", label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.savefig(outpath.with_suffix(".png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory, e.g., results/e4_frontier/seed0")
    ap.add_argument("--results_csv", type=str, default="results.csv", help="CSV filename inside run_dir")
    ap.add_argument("--group_key", type=str, default="tokenizer", help="Grouping key for legend (default: tokenizer)")
    ap.add_argument("--prefix", type=str, default="", help="Optional prefix for output filenames (e.g., token_fair)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / args.results_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")

    prefix = args.prefix.strip()
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    rows = read_rows(csv_path)
    groups = group_by(rows, args.group_key)

    # Common keys produced by run_e4_frontier.py
    plot_scatter(
        groups,
        x_key="avg_len",
        y_key="acc",
        title="Accuracy vs. Rate (avg token length)",
        xlabel="avg_len (tokens/sample)",
        ylabel="accuracy",
        outpath=run_dir / f"{prefix}frontier_acc_vs_len",
    )

    if any("distortion_hat" in r for r in rows):
        plot_scatter(
            groups,
            x_key="distortion_hat",
            y_key="acc",
            title="Accuracy vs. Surrogate Distortion",
            xlabel="distortion_hat (prefix KL surrogate)",
            ylabel="accuracy",
            outpath=run_dir / f"{prefix}frontier_acc_vs_distortion",
        )
        plot_scatter(
            groups,
            x_key="distortion_hat",
            y_key="avg_len",
            title="Rate vs. Surrogate Distortion",
            xlabel="distortion_hat (prefix KL surrogate)",
            ylabel="avg_len (tokens/sample)",
            outpath=run_dir / f"{prefix}frontier_len_vs_distortion",
        )

    # Systems proxy plots (if present)
    if any("tokenize_ms_per_sample" in r for r in rows):
        plot_scatter(
            groups,
            x_key="avg_len",
            y_key="tokenize_ms_per_sample",
            title="Tokenization Time vs. Rate",
            xlabel="avg_len (tokens/sample)",
            ylabel="tokenize_ms_per_sample",
            outpath=run_dir / f"{prefix}frontier_tokenize_ms_vs_len",
        )

    print(f"[OK] Wrote plots under: {run_dir}")


if __name__ == "__main__":
    main()
