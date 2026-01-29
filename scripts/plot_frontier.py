"""Plot publication-ready frontier curves from a results CSV.

This script is intentionally lightweight: it only reads a CSV and emits a few
PDF/PNG plots, so it can be re-run after experiments without re-training.

We use a fixed, colorblind-friendly palette for tokenizers to keep styling
consistent across the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get(key, "NA")), []).append(row)
    return out


TOKENIZER_COLORS: dict[str, str] = {
    "BPE": "#0072B2",
    "Bytes": "#E69F00",
    "CIT": "#009E73",
    "Unigram": "#D55E00",
    "WordPiece": "#CC79A7",
}


MODEL_MARKERS: dict[str, str] = {
    "tiny": "^",
    "small": "s",
    "base": "D",
    "mini": "o",
}


VOCAB_MARKERS: dict[int, str] = {
    256: "o",
    512: "s",
    1024: "^",
    2048: "D",
    4096: "P",
}


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _finite(vals: Iterable[float]) -> list[float]:
    out = []
    for v in vals:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        out.append(float(v))
    return out


def _set_limits(ax, xs: list[float], ys: list[float]) -> None:
    xs_f = _finite(xs)
    ys_f = _finite(ys)
    if xs_f:
        x0, x1 = min(xs_f), max(xs_f)
        pad = 0.03 * max(1e-9, (x1 - x0))
        ax.set_xlim(x0 - pad, x1 + pad)
    if ys_f:
        y0, y1 = min(ys_f), max(ys_f)
        pad = 0.06 * max(1e-9, (y1 - y0))
        ax.set_ylim(y0 - pad, y1 + pad)


def _tokenizer_color(name: str) -> str:
    return TOKENIZER_COLORS.get(name, "#333333")


def plot_frontier(
    rows: list[dict],
    *,
    group_key: str,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
) -> None:
    _apply_style()

    # Heuristic: for CSIC frontiers we have both acc and drift_acc; show both
    # with filled/hollow markers and a connecting segment (robustness gap).
    has_drift = any("drift_acc" in r for r in rows)
    has_model = any("model" in r for r in rows)
    show_drift_gap = bool(has_drift and has_model and x_key == "avg_len" and y_key == "acc" and group_key == "tokenizer")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    all_xs: list[float] = []
    all_ys: list[float] = []

    groups = group_by(rows, group_key)
    group_names = sorted(groups.keys())
    # Prefer a stable tokenizer order when available.
    group_names = sorted(group_names, key=lambda n: (0, n) if n in TOKENIZER_COLORS else (1, n))

    if show_drift_gap:
        # Build model order for consistent markers.
        models = sorted({str(r.get("model", "")) for r in rows if str(r.get("model", ""))})
        model_handles: list[Line2D] = []
        for m in models:
            marker = MODEL_MARKERS.get(m, "o")
            model_handles.append(Line2D([0], [0], marker=marker, color="black", linestyle="None", label=m))

        tok_handles: list[Line2D] = []
        for tok in group_names:
            tok_handles.append(
                Line2D([0], [0], marker="o", color=_tokenizer_color(tok), linestyle="None", label=tok)
            )

        # Test vs drift semantics legend.
        sem_handles = [
            Line2D([0], [0], marker="o", color="black", markerfacecolor="black", linestyle="None", label="test"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", label="drift"),
        ]

        for tok in group_names:
            color = _tokenizer_color(tok)
            tok_rows = groups[tok]
            for r in tok_rows:
                x = to_float(r.get(x_key))
                y_test = to_float(r.get("acc"))
                y_drift = to_float(r.get("drift_acc"))
                m = str(r.get("model", ""))
                marker = MODEL_MARKERS.get(m, "o")

                all_xs.extend([x])
                all_ys.extend([y_test, y_drift])

                ax.plot([x, x], [y_drift, y_test], color=color, alpha=0.25, linewidth=1.8, solid_capstyle="round")
                ax.scatter([x], [y_test], s=42, marker=marker, color=color, edgecolors="white", linewidths=0.6, zorder=3)
                ax.scatter([x], [y_drift], s=42, marker=marker, facecolors="none", edgecolors=color, linewidths=1.3, zorder=3)

        fig.legend(
            handles=tok_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=min(5, len(tok_handles)),
            frameon=False,
        )
        ax.legend(handles=sem_handles + model_handles, loc="lower left", frameon=False, ncol=1)
    else:
        # Generic: line+markers per group.
        style_key = None
        if any("vocab_size" in r for r in rows):
            style_key = "vocab_size"
        elif any("model" in r for r in rows):
            style_key = "model"

        show_vocab_lines = bool(style_key == "vocab_size" and not (x_key == "avg_len" and y_key == "tokenize_ms_per_sample"))

        vocab_sizes: list[int] = []
        if style_key == "vocab_size":
            for r in rows:
                try:
                    vocab_sizes.append(int(float(r.get("vocab_size", 0))))
                except Exception:
                    continue
            vocab_sizes = sorted({v for v in vocab_sizes if v > 0})

        for name in group_names:
            color = _tokenizer_color(name)
            rs = groups[name]

            # Sort by x for a sensible visual trend.
            pairs = []
            for r in rs:
                x = to_float(r.get(x_key))
                y = to_float(r.get(y_key))
                pairs.append((x, y, r))
                all_xs.append(x)
                all_ys.append(y)
            if style_key == "vocab_size":
                def _vkey(t):
                    r = t[2]
                    try:
                        return int(float(r.get("vocab_size", 0)))
                    except Exception:
                        return 0
                pairs.sort(key=_vkey)
            else:
                pairs.sort(key=lambda t: (t[0], t[1]))

            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]

            # If we have many points (e.g., vocab sweep), draw a line; otherwise just scatter.
            draw_line = len(pairs) >= 3 and style_key == "vocab_size"
            if style_key == "vocab_size" and x_key == "avg_len" and y_key == "tokenize_ms_per_sample":
                draw_line = False
            if draw_line:
                ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.9, label=name, zorder=2)

            # Scatter points with optional marker by style_key.
            for x, y, r in pairs:
                marker = "o"
                if style_key == "model":
                    marker = MODEL_MARKERS.get(str(r.get("model", "")), "o")
                elif style_key == "vocab_size":
                    try:
                        marker = VOCAB_MARKERS.get(int(float(r.get("vocab_size", 0))), "o")
                    except Exception:
                        marker = "o"
                ax.scatter([x], [y], s=42, marker=marker, color=color, edgecolors="white", linewidths=0.6, zorder=3)

        tok_handles: list[Line2D] = []
        for name in group_names:
            tok_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=_tokenizer_color(name),
                    linestyle="-" if show_vocab_lines else "None",
                    label=name,
                    linewidth=2.0,
                )
            )
        leg1 = ax.legend(handles=tok_handles, loc="best", frameon=False)
        if style_key == "vocab_size" and vocab_sizes:
            ax.add_artist(leg1)
            vocab_handles: list[Line2D] = []
            for v in vocab_sizes:
                marker = VOCAB_MARKERS.get(v, "o")
                vocab_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="black",
                        linestyle="None",
                        label=str(v),
                        markersize=7,
                    )
                )
            ax.legend(handles=vocab_handles, title="Vocab", loc="lower right", frameon=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_limits(ax, all_xs, all_ys)
    if show_drift_gap:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    else:
        fig.tight_layout()
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    # Common keys produced by run_e4_frontier.py / run_e2_csic_http.py
    plot_frontier(
        rows,
        group_key=args.group_key,
        x_key="avg_len",
        y_key="acc",
        xlabel="avg_len (tokens/sample)",
        ylabel="accuracy",
        outpath=run_dir / f"{prefix}frontier_acc_vs_len",
    )

    if any("distortion_hat" in r for r in rows):
        plot_frontier(
            rows,
            group_key=args.group_key,
            x_key="distortion_hat",
            y_key="acc",
            xlabel=r"$\hat{\Delta}$ (prefix KL surrogate)",
            ylabel="accuracy",
            outpath=run_dir / f"{prefix}frontier_acc_vs_distortion",
        )
        plot_frontier(
            rows,
            group_key=args.group_key,
            x_key="distortion_hat",
            y_key="avg_len",
            xlabel=r"$\hat{\Delta}$ (prefix KL surrogate)",
            ylabel="avg_len (tokens/sample)",
            outpath=run_dir / f"{prefix}frontier_len_vs_distortion",
        )

    # Systems proxy plots (if present)
    if any("tokenize_ms_per_sample" in r for r in rows):
        plot_frontier(
            rows,
            group_key=args.group_key,
            x_key="avg_len",
            y_key="tokenize_ms_per_sample",
            xlabel="avg_len (tokens/sample)",
            ylabel="tokenization time (ms/sample)",
            outpath=run_dir / f"{prefix}frontier_tokenize_ms_vs_len",
        )

    print(f"[OK] Wrote plots under: {run_dir}")


if __name__ == "__main__":
    main()
