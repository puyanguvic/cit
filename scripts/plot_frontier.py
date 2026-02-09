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
    show_drift_gap = bool(
        has_drift
        and has_model
        and y_key == "acc"
        and group_key == "tokenizer"
        and x_key in {"avg_len", "total_tokens"}
    )

    compact_names = {"frontier_len_vs_distortion", "frontier_tokenize_ms_vs_len"}
    is_compact = outpath.name in compact_names
    fig, ax = plt.subplots(figsize=(3.45, 2.65) if is_compact else (6.6, 3.2))

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

        # Small horizontal offsets per-model to avoid overplotting when multiple
        # models share the same avg_len (tokenization is model-independent).
        xs_unique = sorted({to_float(r.get(x_key)) for r in rows})
        xs_unique = [x for x in xs_unique if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        xs_unique = sorted({float(x) for x in xs_unique})
        min_sep = 1.0
        if len(xs_unique) >= 2:
            diffs = [b - a for a, b in zip(xs_unique, xs_unique[1:]) if b > a]
            if diffs:
                min_sep = float(min(diffs))
        dx = 0.06 * min_sep
        model_offsets: dict[str, float] = {}
        if models:
            center = 0.5 * (len(models) - 1)
            for i, m in enumerate(models):
                model_offsets[m] = (i - center) * dx

        # If there is a big empty x-gap (e.g., Bytes is far right), use a broken
        # x-axis to increase density for the main cluster.
        use_broken_x = False
        break_left_max: float | None = None
        break_right_min: float | None = None
        if x_key == "avg_len" and len(xs_unique) >= 4:
            gaps = [(xs_unique[i + 1] - xs_unique[i], i) for i in range(len(xs_unique) - 1)]
            gap, idx = max(gaps, key=lambda t: t[0])
            span = xs_unique[-1] - xs_unique[0]
            if span > 0 and (gap / span) >= 0.45:
                use_broken_x = True
                break_left_max = xs_unique[idx]
                break_right_min = xs_unique[idx + 1]

        tok_handles: list[Line2D] = [
            Line2D([0], [0], marker="o", color=_tokenizer_color(tok), linestyle="None", label=tok) for tok in group_names
        ]

        # Test vs drift semantics legend.
        sem_handles = [
            Line2D([0], [0], marker="o", color="black", markerfacecolor="black", linestyle="None", label="test"),
            Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", label="drift"),
        ]

        def _draw_on(ax) -> None:
            # For token-budget sweeps, connect points across budgets so trends are
            # visible (otherwise many points share the same x under avg_len).
            if x_key == "total_tokens":
                for tok in group_names:
                    color = _tokenizer_color(tok)
                    is_highlight = tok == "CIT"
                    for m in models:
                        pts_test: list[tuple[float, float]] = []
                        pts_drift: list[tuple[float, float]] = []
                        for r in groups[tok]:
                            if str(r.get("model", "")) != m:
                                continue
                            x0 = to_float(r.get(x_key))
                            y_test = to_float(r.get("acc"))
                            y_drift = to_float(r.get("drift_acc"))
                            if x0 is None or y_test is None or y_drift is None:
                                continue
                            x = x0 + model_offsets.get(m, 0.0)
                            pts_test.append((x, y_test))
                            pts_drift.append((x, y_drift))
                        if len(pts_test) >= 2:
                            pts_test.sort(key=lambda t: t[0])
                            ax.plot(
                                [p[0] for p in pts_test],
                                [p[1] for p in pts_test],
                                color=color,
                                alpha=0.50 if is_highlight else 0.18,
                                linewidth=2.2 if is_highlight else 1.1,
                                zorder=1,
                            )
                        if len(pts_drift) >= 2:
                            pts_drift.sort(key=lambda t: t[0])
                            ax.plot(
                                [p[0] for p in pts_drift],
                                [p[1] for p in pts_drift],
                                color=color,
                                alpha=0.32 if is_highlight else 0.12,
                                linewidth=1.8 if is_highlight else 0.9,
                                linestyle="--",
                                zorder=1,
                            )

            for tok in group_names:
                color = _tokenizer_color(tok)
                tok_rows = groups[tok]
                is_highlight = tok == "CIT"
                for r in tok_rows:
                    x0 = to_float(r.get(x_key))
                    y_test = to_float(r.get("acc"))
                    y_drift = to_float(r.get("drift_acc"))
                    m = str(r.get("model", ""))
                    marker = MODEL_MARKERS.get(m, "o")
                    x = x0 + model_offsets.get(m, 0.0)

                    all_xs.append(x0)
                    all_ys.extend([y_test, y_drift])

                    gap_alpha = 0.55 if is_highlight else 0.18
                    gap_lw = 2.6 if is_highlight else 1.6
                    point_s = 58 if is_highlight else 42
                    z = 4 if is_highlight else 3
                    test_edge = "black" if is_highlight else "white"
                    test_lw = 0.9 if is_highlight else 0.6
                    drift_edge = "black" if is_highlight else color
                    drift_lw = 1.5 if is_highlight else 1.3

                    ax.plot([x, x], [y_drift, y_test], color=color, alpha=gap_alpha, linewidth=gap_lw, solid_capstyle="round")
                    ax.scatter([x], [y_test], s=point_s, marker=marker, color=color, edgecolors=test_edge, linewidths=test_lw, zorder=z)
                    ax.scatter([x], [y_drift], s=point_s, marker=marker, facecolors="none", edgecolors=drift_edge, linewidths=drift_lw, zorder=z)

                    # Optional error bars (e.g., when plotting mean/std summaries).
                    yerr_test = to_float(r.get("acc_std"))
                    yerr_drift = to_float(r.get("drift_acc_std"))
                    if yerr_test is not None and yerr_test > 0.0:
                        ax.errorbar(
                            [x],
                            [y_test],
                            yerr=[yerr_test],
                            fmt="none",
                            ecolor=color,
                            elinewidth=0.9,
                            capsize=2,
                            alpha=0.55 if is_highlight else 0.35,
                            zorder=z - 0.5,
                        )
                    if yerr_drift is not None and yerr_drift > 0.0:
                        ax.errorbar(
                            [x],
                            [y_drift],
                            yerr=[yerr_drift],
                            fmt="none",
                            ecolor=color,
                            elinewidth=0.9,
                            capsize=2,
                            alpha=0.45 if is_highlight else 0.28,
                            zorder=z - 0.5,
                        )

            if "CIT" in groups:
                # Light annotation near the best CIT test point.
                best_row = None
                best_y = None
                for r in groups["CIT"]:
                    x = to_float(r.get(x_key))
                    y = to_float(r.get("acc"))
                    if x is None or y is None:
                        continue
                    if best_y is None or y > best_y:
                        best_row = r
                        best_y = y
                if best_row is not None and best_y is not None:
                    x0 = to_float(best_row.get(x_key))
                    m = str(best_row.get("model", ""))
                    x = x0 + model_offsets.get(m, 0.0)
                    ax.annotate(
                        "CIT",
                        xy=(x, best_y),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=9,
                        weight="bold",
                        color=_tokenizer_color("CIT"),
                    )

        if use_broken_x and break_left_max is not None and break_right_min is not None:
            # Switch to a broken-axis layout.
            plt.close(fig)
            fig, (ax_l, ax_r) = plt.subplots(
                1,
                2,
                sharey=True,
                figsize=(7.2, 3.2),
                gridspec_kw={"width_ratios": [3.2, 1.1], "wspace": 0.05},
            )
            _apply_style()
            _draw_on(ax_l)
            _draw_on(ax_r)

            left_min = xs_unique[0]
            right_max = xs_unique[-1]
            pad = 0.15 * min_sep
            ax_l.set_xlim(left_min - pad, break_left_max + pad)
            ax_r.set_xlim(break_right_min - 0.30 * min_sep, right_max + 0.18 * min_sep)

            # Avoid scientific offset formatting on broken-axis panels.
            try:
                import matplotlib.ticker as mticker

                fmt = mticker.ScalarFormatter(useOffset=False)
                fmt.set_scientific(False)
                for ax_ in (ax_l, ax_r):
                    ax_.xaxis.set_major_formatter(fmt)
                    ax_.xaxis.get_offset_text().set_visible(False)
            except Exception:
                pass

            ax_l.spines.right.set_visible(False)
            ax_r.spines.left.set_visible(False)
            ax_r.tick_params(labelleft=False)
            ax_r.yaxis.set_ticks_position("right")

            d = 0.012
            kwargs = dict(color="black", clip_on=False, linewidth=0.8)
            ax_l.plot((1 - d, 1 + d), (-d, +d), transform=ax_l.transAxes, **kwargs)
            ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_l.transAxes, **kwargs)
            ax_r.plot((-d, +d), (-d, +d), transform=ax_r.transAxes, **kwargs)
            ax_r.plot((-d, +d), (1 - d, 1 + d), transform=ax_r.transAxes, **kwargs)

            fig.legend(
                handles=tok_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),
                ncol=min(5, len(tok_handles)),
                frameon=False,
            )
            ax_l.legend(handles=sem_handles + model_handles, loc="lower left", frameon=False, ncol=1)

            ax_l.set_ylabel(ylabel)
            fig.supxlabel(xlabel)
            _set_limits(ax_l, [], all_ys)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        else:
            _draw_on(ax)
            fig.legend(
                handles=tok_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.06),
                ncol=min(5, len(tok_handles)),
                frameon=False,
            )
            ax.legend(handles=sem_handles + model_handles, loc="lower left", frameon=False, ncol=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            _set_limits(ax, all_xs, all_ys)
            if x_key == "total_tokens":
                try:
                    import matplotlib.ticker as mticker

                    ax.set_xscale("log")
                    xs_pos = [x for x in all_xs if isinstance(x, float) and x > 0 and not math.isnan(x) and not math.isinf(x)]
                    if xs_pos:
                        ax.set_xlim(min(xs_pos) / 1.15, max(xs_pos) * 1.15)

                    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))

                    def _fmt_tokens(x, _pos=None) -> str:
                        try:
                            x = float(x)
                        except Exception:
                            return ""
                        if x >= 1_000_000:
                            v = x / 1_000_000.0
                            if abs(v - round(v)) < 1e-6:
                                return f"{int(round(v))}M"
                            return f"{v:.1f}M"
                        if x >= 1_000:
                            v = x / 1_000.0
                            if abs(v - round(v)) < 1e-6:
                                return f"{int(round(v))}k"
                            return f"{v:.1f}k"
                        return str(int(round(x)))

                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_tokens))
                    ax.xaxis.get_offset_text().set_visible(False)
                except Exception:
                    pass
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(outpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    else:
        # Generic: line+markers per group.
        style_key = None
        if any("vocab_size" in r for r in rows):
            style_key = "vocab_size"
        elif any("model" in r for r in rows):
            style_key = "model"

        show_vocab_lines = bool(style_key == "vocab_size")

        vocab_sizes: list[int] = []
        if style_key == "vocab_size":
            for r in rows:
                try:
                    vocab_sizes.append(int(float(r.get("vocab_size", 0))))
                except Exception:
                    continue
            vocab_sizes = sorted({v for v in vocab_sizes if v > 0})

        # A compact, more readable design for the appendix rate--distortion plot.
        is_rate_distortion = bool(style_key == "vocab_size" and x_key == "avg_len" and y_key == "distortion_hat")

        if is_rate_distortion:
            # Trajectory per tokenizer, marker shapes denote vocab size. Directly
            # label curves at the largest vocab to avoid a crowded legend in a
            # half-column figure.
            label_offsets = {
                "CIT": (6, -2),
                "BPE": (6, 6),
                "WordPiece": (6, -8),
                "Unigram": (6, 6),
                "Bytes": (6, 6),
            }

            for name in group_names:
                color = _tokenizer_color(name)
                is_highlight = name == "CIT"
                rs = groups[name]

                pairs: list[tuple[int, float, float, dict]] = []
                for r in rs:
                    try:
                        v = int(float(r.get("vocab_size", 0)))
                    except Exception:
                        v = 0
                    x = to_float(r.get(x_key))
                    y = to_float(r.get(y_key))
                    pairs.append((v, x, y, r))
                    all_xs.append(x)
                    all_ys.append(y)
                pairs = [p for p in pairs if p[0] > 0]
                pairs.sort(key=lambda t: t[0])

                xs = [p[1] for p in pairs]
                ys = [p[2] for p in pairs]
                if len(pairs) >= 2:
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linewidth=2.4 if is_highlight else 1.6,
                        alpha=0.95 if is_highlight else 0.70,
                        zorder=2,
                    )

                for v, x, y, _r in pairs:
                    marker = VOCAB_MARKERS.get(v, "o")
                    ax.scatter(
                        [x],
                        [y],
                        s=58 if is_highlight else 46,
                        marker=marker,
                        color=color,
                        edgecolors="black" if is_highlight else "white",
                        linewidths=0.9 if is_highlight else 0.7,
                        zorder=3 if is_highlight else 2.5,
                    )

                # Label the largest-vocab point.
                if pairs:
                    v_last, x_last, y_last, _r_last = pairs[-1]
                    dx, dy = label_offsets.get(name, (6, 6))
                    ax.annotate(
                        name,
                        xy=(x_last, y_last),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=9,
                        weight="bold" if is_highlight else "normal",
                        color=color,
                    )

            # Legend: vocab shapes only (tokenizers are labeled on-plot).
            vocab_handles: list[Line2D] = []
            for v in vocab_sizes:
                vocab_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=VOCAB_MARKERS.get(v, "o"),
                        color="black",
                        linestyle="None",
                        label=str(v),
                        markersize=6.5,
                    )
                )
            ax.legend(handles=vocab_handles, title="Vocab", loc="upper right", frameon=False, borderaxespad=0.2)

            # "Better" direction cue: lower length + lower distortion.
            try:
                ax.annotate(
                    "better",
                    xy=(0.07, 0.12),
                    xytext=(0.22, 0.26),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#666666"},
                    fontsize=9,
                    color="#666666",
                )
            except Exception:
                pass
        else:
            for name in group_names:
                color = _tokenizer_color(name)
                is_highlight = name == "CIT"
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

                # For vocab sweeps, draw a light trajectory line to show the
                # progression as vocab grows (even for tokenize-time plots).
                if style_key == "vocab_size" and len(pairs) >= 2:
                    ls = "--" if (x_key == "avg_len" and y_key == "tokenize_ms_per_sample") else "-"
                    lw = 1.8 if ls == "--" else 2.2
                    alpha = 0.95 if is_highlight else 0.75
                    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, alpha=alpha, label=name, zorder=2)

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
                    s = 58 if is_highlight else 42
                    edge = "black" if is_highlight else "white"
                    lw = 0.9 if is_highlight else 0.6
                    z = 4 if is_highlight else 3
                    ax.scatter([x], [y], s=s, marker=marker, color=color, edgecolors=edge, linewidths=lw, zorder=z)

        if not is_rate_distortion:
            tok_handles: list[Line2D] = []
            tok_ls = "--" if (style_key == "vocab_size" and x_key == "avg_len" and y_key == "tokenize_ms_per_sample") else "-"
            for name in group_names:
                tok_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color=_tokenizer_color(name),
                        linestyle=tok_ls if show_vocab_lines else "None",
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
            x_key="avg_len",
            y_key="distortion_hat",
            xlabel="avg_len (tokens/sample)",
            ylabel=r"$\hat{\Delta}$ (prefix KL surrogate)",
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
