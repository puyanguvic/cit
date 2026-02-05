#!/usr/bin/env python3
"""Export publication-ready tables/figures for inserting into a paper.

This script reads experiment outputs under results/<run-root>/ and writes:
  - Main-paper artifacts into: results/<run-root>/paper_artifacts/main/
  - Appendix artifacts into:   results/<run-root>/paper_artifacts/appendix/

It copies any existing plots produced by other scripts (e.g., plot_frontier.py),
and generates a small number of missing plots (currently: E3 Pareto).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKENIZER_COLORS: dict[str, str] = {
    "BPE": "#0072B2",
    "Bytes": "#E69F00",
    "CIT": "#009E73",
    "Unigram": "#D55E00",
    "WordPiece": "#CC79A7",
}

BACKBONE_MARKERS: dict[str, str] = {
    "tiny": "o",
    "mini": "o",
    "small": "s",
    "base": "^",
}


def _apply_mpl_style() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

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


def _tokenizer_color(name: str) -> str:
    return TOKENIZER_COLORS.get(name, "#333333")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _to_float(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _seed_dirs(exp_dir: Path) -> list[Path]:
    if not exp_dir.exists():
        return []
    out = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]
    def _key(p: Path) -> tuple[int, str]:
        suf = p.name[len("seed") :]
        try:
            return (int(suf), p.name)
        except Exception:
            return (10**9, p.name)

    return sorted(out, key=_key)


def _collect_many(csv_paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in csv_paths:
        for r in _read_csv(p):
            r["_source"] = str(p)
            rows.append(r)
    return rows


def _seed_csvs_recursive(root: Path, csv_name: str) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob(csv_name):
        if p.is_file() and p.parent.is_dir() and p.parent.name.startswith("seed"):
            out.append(p)
    return sorted(out, key=lambda p: str(p))


def _group_stats(rows: list[dict], *, group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    buckets: dict[tuple[str, ...], list[dict]] = {}
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in group_keys)
        buckets.setdefault(key, []).append(r)

    out_rows: list[dict] = []
    for key, rs in sorted(buckets.items(), key=lambda kv: kv[0]):
        out: dict[str, object] = {k: key[i] for i, k in enumerate(group_keys)}
        out["n"] = len(rs)
        for mk in metric_keys:
            vals = [_to_float(r.get(mk)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                out[f"{mk}_mean"] = None
                out[f"{mk}_std"] = None
                continue
            out[f"{mk}_mean"] = statistics.mean(vals)
            out[f"{mk}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out_rows.append(out)
    return out_rows


def _latex_escape(s: object) -> str:
    text = str(s)
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def _fmt_mean_std(mean: float | None, std: float | None, *, n: int, digits: int, kind: str) -> str:
    if mean is None:
        return ""
    if kind == "int":
        return str(int(round(mean)))
    if kind == "pct":
        mean = 100.0 * float(mean)
        std = 0.0 if std is None else 100.0 * float(std)
    if n > 1 and std is not None and float(std) > 0.0:
        return f"${mean:.{digits}f} \\\\pm {std:.{digits}f}$"
    return f"{mean:.{digits}f}"


@dataclass(frozen=True)
class Col:
    key: str
    header: str
    align: str = "l"
    metric: str | None = None
    digits: int = 3
    kind: str = "float"  # float | int | pct


def _write_latex_table(
    rows: list[dict],
    *,
    columns: list[Col],
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    align = "".join(c.align for c in columns)
    # Headers/captions/labels may intentionally include LaTeX, so keep them raw.
    header_cells = [str(c.header) for c in columns]

    lines: list[str] = []
    lines.append("% Auto-generated by scripts/export_paper_artifacts.py")
    lines.append("% Requires: \\usepackage{booktabs}")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    for r in rows:
        n = int(r.get("n", 1) or 1)
        cells: list[str] = []
        for c in columns:
            if c.metric:
                mean = r.get(f"{c.metric}_mean")
                std = r.get(f"{c.metric}_std")
                cells.append(_fmt_mean_std(_to_float(mean), _to_float(std), n=n, digits=c.digits, kind=c.kind))
            else:
                cells.append(_latex_escape(r.get(c.key, "")))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _first_existing_dir(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _pick_dir_with_seed_csv(candidates: list[Path], csv_name: str) -> Path | None:
    """Pick the first candidate that actually contains at least one seed CSV."""
    for d in candidates:
        if not d.exists():
            continue
        for sd in _seed_dirs(d):
            if (sd / csv_name).exists():
                return d
    # Fallback: any existing directory (even if empty/incomplete).
    return _first_existing_dir(*candidates)


def _pick_dir_with_seed_csv_recursive(candidates: list[Path], csv_name: str) -> Path | None:
    """Pick the first candidate that contains a seed CSV anywhere under it."""
    for d in candidates:
        if not d.exists():
            continue
        if _seed_csvs_recursive(d, csv_name):
            return d
    return _first_existing_dir(*candidates)


def _pick_seed_dir(exp_dir: Path) -> Path | None:
    seeds = _seed_dirs(exp_dir)
    return seeds[0] if seeds else None


def _plot_e3_pareto(csv_path: Path, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        print("[warn] matplotlib not available; skipping E3 plots.")
        return

    if not csv_path.exists():
        return

    _apply_mpl_style()
    rows = _read_csv(csv_path)
    if not rows:
        return

    def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(str(r.get(key, "NA")), []).append(r)
        return out

    # Stable ordering for consistent aesthetics across runs.
    tokenizers = sorted({str(r.get("tokenizer", "")) for r in rows if str(r.get("tokenizer", ""))})
    tokenizers = sorted(tokenizers, key=lambda n: (0, n) if n in TOKENIZER_COLORS else (1, n))
    backbones = sorted({str(r.get("backbone", "")) for r in rows if str(r.get("backbone", ""))})

    # Build quick lookup by (tokenizer, backbone).
    by_key: dict[tuple[str, str], dict] = {}
    for r in rows:
        tok = str(r.get("tokenizer", ""))
        bb = str(r.get("backbone", ""))
        if tok and bb:
            by_key[(tok, bb)] = r

    def _scatter(ax, x_key: str, y_key: str, *, xlabel: str, ylabel: str) -> None:
        x_log = x_key == "p95_latency_ms"
        xs_all: list[float] = []
        ys_all: list[float] = []

        # Optional connectors between backbones for each tokenizer (helps reduce
        # the "floating points" feeling when there are only a few backbones).
        for tok in tokenizers:
            pts: list[tuple[float, float]] = []
            for bb in backbones:
                r = by_key.get((tok, bb))
                if r is None:
                    continue
                x = _to_float(r.get(x_key))
                y = _to_float(r.get(y_key))
                if x is None or y is None:
                    continue
                pts.append((float(x), float(y)))
            if len(pts) >= 2:
                pts = sorted(pts, key=lambda t: t[0])
                is_highlight = tok == "CIT"
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=_tokenizer_color(tok),
                    alpha=0.55 if is_highlight else 0.22,
                    linewidth=2.0 if is_highlight else 1.1,
                    zorder=1,
                )

        for tok in tokenizers:
            color = _tokenizer_color(tok)
            for bb in backbones:
                r = by_key.get((tok, bb))
                if r is None:
                    continue
                x = _to_float(r.get(x_key))
                y = _to_float(r.get(y_key))
                if x is None or y is None:
                    continue
                xs_all.append(float(x))
                ys_all.append(float(y))
                marker = BACKBONE_MARKERS.get(bb, "o")
                is_highlight = tok == "CIT"
                ax.scatter(
                    [float(x)],
                    [float(y)],
                    s=62 if is_highlight else 46,
                    marker=marker,
                    color=color,
                    edgecolors="black" if is_highlight else "white",
                    linewidths=0.9 if is_highlight else 0.6,
                    alpha=1.0 if is_highlight else 0.85,
                    zorder=4 if is_highlight else 3,
                )

        if xs_all:
            x0, x1 = min(xs_all), max(xs_all)
            if x_log and x0 > 0:
                ax.set_xscale("log")
                ax.set_xlim(x0 / 1.12, x1 * 1.12)
            else:
                pad = 0.03 * max(1e-9, (x1 - x0))
                ax.set_xlim(x0 - pad, x1 + pad)
        if ys_all:
            y0, y1 = min(ys_all), max(ys_all)
            pad = 0.06 * max(1e-9, (y1 - y0))
            ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.98, 0.04, "better ↖", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#333333")

        if x_key in {"avg_len", "p95_latency_ms"}:
            r = by_key.get(("CIT", "mini"))
            if r is not None:
                x = _to_float(r.get(x_key))
                y = _to_float(r.get(y_key))
                if x is not None and y is not None:
                    ax.annotate(
                        "CIT",
                        xy=(float(x), float(y)),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=9,
                        weight="bold",
                        color=_tokenizer_color("CIT"),
                    )

    have_params = any("params_m" in r for r in rows)

    # Triptych (publication-ready).
    fig, axes = plt.subplots(1, 3 if have_params else 2, figsize=(10.4 if have_params else 7.0, 3.1))
    try:
        axes_list = list(axes)  # type: ignore[arg-type]
    except TypeError:
        axes_list = [axes]  # type: ignore[list-item]
    axes = axes_list

    _scatter(
        axes[0],
        "avg_len",
        "acc",
        xlabel="avg_len (tokens/sample)",
        ylabel="accuracy",
    )
    axes[0].set_title("Rate")

    _scatter(
        axes[1],
        "p95_latency_ms",
        "acc",
        xlabel="p95 latency (ms)",
        ylabel="accuracy",
    )
    axes[1].set_title("Latency")

    if have_params and len(axes) >= 3:
        _scatter(
            axes[2],
            "params_m",
            "acc",
            xlabel="params (M)",
            ylabel="accuracy",
        )
        axes[2].set_title("Parameters")

    tok_handles = [
        Line2D([0], [0], marker="o", color=_tokenizer_color(tok), linestyle="None", label=tok) for tok in tokenizers
    ]
    bb_handles = [
        Line2D([0], [0], marker=BACKBONE_MARKERS.get(bb, "o"), color="black", linestyle="None", label=bb)
        for bb in backbones
    ]

    leg1 = fig.legend(
        handles=tok_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=min(5, len(tok_handles)),
        frameon=False,
    )
    fig.add_artist(leg1)
    fig.legend(handles=bb_handles, loc="upper right", bbox_to_anchor=(0.995, 1.15), frameon=False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e3_pareto_triptych.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "e3_pareto_triptych.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Keep per-metric panels for convenience/backward compatibility.
    def _single(name: str, x_key: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6.6, 3.2))
        _scatter(ax, x_key, "acc", xlabel=xlabel, ylabel="accuracy")
        leg1 = ax.legend(
            handles=tok_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=min(5, len(tok_handles)),
            frameon=False,
        )
        ax.add_artist(leg1)
        ax.legend(handles=bb_handles, loc="lower right", frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    _single("e3_pareto_acc_vs_len", "avg_len", "avg_len (tokens/sample)")
    _single("e3_pareto_acc_vs_latency", "p95_latency_ms", "p95 latency (ms)")
    if have_params:
        _single("e3_pareto_acc_vs_params", "params_m", "params (M)")


def _read_curve_csv(path: Path) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            tok = _to_float(row.get("seen_tokens"))
            val = _to_float(row.get("val_acc"))
            if tok is None or val is None:
                continue
            xs.append(int(tok))
            ys.append(float(val))
    return xs, ys


def _t95_tokens(xs: list[int], ys: list[float]) -> int | None:
    if not xs or not ys:
        return None
    best = max(ys)
    if best <= 0.0:
        return None
    target = 0.95 * best
    # xs are non-decreasing by construction; find first time reaching target.
    for x, y in sorted(zip(xs, ys), key=lambda t: t[0]):
        if y >= target:
            return int(x)
    return None


def _plot_e2_learning_curves_mean(
    seed_dirs: list[Path],
    *,
    mode: str,
    model_kind: str,
    total_tokens: int,
    out_dir: Path,
    name: str,
) -> bool:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] numpy/matplotlib not available; skipping E2 learning-curve plot.")
        return False

    if not seed_dirs:
        return False

    _apply_mpl_style()

    # Build a fixed token grid for averaging.
    total_tokens = int(total_tokens)
    grid = np.linspace(0.0, float(total_tokens), num=21)

    # Collect per-tokenizer curves across seeds.
    curves: dict[str, list[tuple[list[int], list[float]]]] = {}
    for sd in seed_dirs:
        curves_dir = sd / "curves" / mode
        if not curves_dir.exists():
            continue
        for p in curves_dir.glob(f"curves_{model_kind}_*.csv"):
            tok = p.stem.split("_", 2)[-1]
            xs, ys = _read_curve_csv(p)
            if xs and ys:
                curves.setdefault(tok, []).append((xs, ys))

    if not curves:
        return False

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    tok_names = sorted(curves.keys(), key=lambda n: (0, n) if n in TOKENIZER_COLORS else (1, n))
    for tok in tok_names:
        runs = curves[tok]
        vals = []
        for xs, ys in runs:
            # Interpolate onto grid (use endpoints outside range).
            xs_np = np.asarray(xs, dtype=np.float64)
            ys_np = np.asarray(ys, dtype=np.float64)
            order = np.argsort(xs_np)
            xs_np = xs_np[order]
            ys_np = ys_np[order]
            vals.append(np.interp(grid, xs_np, ys_np, left=float(ys_np[0]), right=float(ys_np[-1])))
        y_mean = np.mean(np.stack(vals, axis=0), axis=0)
        ax.plot(grid / 1e6, y_mean, linestyle="-", linewidth=2.2, color=_tokenizer_color(tok), label=tok)

    ax.set_xlabel("seen tokens (M)")
    ax.set_ylabel("validation accuracy")
    ax.set_xlim(0.0, float(total_tokens) / 1e6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(5, len(tok_names)), frameon=False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=str, default="paper", help="Folder under results/ (default: paper)")
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Optional override output dir (default: results/<run-root>/paper_artifacts)",
    )
    args = ap.parse_args()

    run_root = Path("results") / args.run_root
    outroot = Path(args.outdir) if args.outdir else (run_root / "paper_artifacts")
    main_tables_dir = outroot / "main" / "tables"
    main_figs_dir = outroot / "main" / "figures"
    appendix_tables_dir = outroot / "appendix" / "tables"
    appendix_figs_dir = outroot / "appendix" / "figures"

    # Clear old outputs to avoid stale artifacts lingering across runs.
    _reset_dir(main_tables_dir)
    _reset_dir(main_figs_dir)
    _reset_dir(appendix_tables_dir)
    _reset_dir(appendix_figs_dir)
    for legacy in [outroot / "tables", outroot / "figures"]:
        if legacy.exists():
            shutil.rmtree(legacy)

    # ---- Tables ----
    # Main paper: E1 / E2 / E3
    wrote_tables = 0
    wrote_figs = 0

    # E1 (synthetic)
    e1_dir = _pick_dir_with_seed_csv([run_root / "e1", run_root / "e1_synth"], "results.csv")
    if e1_dir is None:
        e1_csvs = []
    else:
        e1_csvs = [p / "results.csv" for p in _seed_dirs(e1_dir) if (p / "results.csv").exists()]
    if e1_csvs:
        e1_rows = _collect_many(e1_csvs)
        e1_sum = _group_stats(
            e1_rows,
            group_keys=["tokenizer"],
            metric_keys=["test_acc", "drift_acc", "rate_E_len", "distortion_hat"],
        )
        _write_latex_table(
            e1_sum,
            columns=[
                Col("tokenizer", "Tokenizer", align="l"),
                Col("test_acc", "Test acc.", align="r", metric="test_acc", digits=3),
                Col("drift_acc", "Drift acc.", align="r", metric="drift_acc", digits=3),
                Col("rate_E_len", "Rate $\\mathbb{E}[|Z|]$", align="r", metric="rate_E_len", digits=1),
                Col("distortion_hat", "Distortion $\\hat{\\Delta}$", align="r", metric="distortion_hat", digits=3),
            ],
            out_path=main_tables_dir / "e1_synth.tex",
            caption="E1 (synthetic) summary across seeds.",
            label="tab:e1_synth",
        )
        wrote_tables += 1

    # E2 (CSIC HTTP): token-fair + step-fair
    e2_main_dir = _pick_dir_with_seed_csv([run_root / "e2", run_root / "e2_csic_http"], "results_token_fair.csv")
    e2_step_dir = _pick_dir_with_seed_csv([run_root / "appendix_e2"], "results_step_fair.csv")

    e2_token_csvs: list[Path] = []
    if e2_main_dir is not None:
        e2_token_csvs = [
            p / "results_token_fair.csv" for p in _seed_dirs(e2_main_dir) if (p / "results_token_fair.csv").exists()
        ]

    e2_step_csvs: list[Path] = []
    if e2_step_dir is not None:
        e2_step_csvs = [
            p / "results_step_fair.csv" for p in _seed_dirs(e2_step_dir) if (p / "results_step_fair.csv").exists()
        ]
    if e2_token_csvs:
        e2_token_rows = _collect_many(e2_token_csvs)
        e2_token_sum = _group_stats(
            e2_token_rows,
            group_keys=["model", "tokenizer"],
            metric_keys=["acc", "drift_acc", "avg_len", "p95_len"],
        )
        _write_latex_table(
            e2_token_sum,
            columns=[
                Col("model", "Model", align="l"),
                Col("tokenizer", "Tokenizer", align="l"),
                Col("acc", "Acc.", align="r", metric="acc", digits=3),
                Col("drift_acc", "Drift acc.", align="r", metric="drift_acc", digits=3),
                Col("avg_len", "avg\\_len", align="r", metric="avg_len", digits=1),
                Col("p95_len", "p95\\_len", align="r", metric="p95_len", digits=0, kind="int"),
            ],
            out_path=main_tables_dir / "e2_csic_token_fair.tex",
            caption="E2 (CSIC 2010 HTTP) token-fair summary across seeds.",
            label="tab:e2_csic_token_fair",
        )
        wrote_tables += 1

        # Convergence diagnostics from learning curves (token-fair).
        seed_dirs = _seed_dirs(e2_main_dir) if e2_main_dir is not None else []
        conv_rows: list[dict] = []
        for sd in seed_dirs:
            seed_s = sd.name[len("seed") :]
            try:
                seed = int(seed_s)
            except Exception:
                seed = 0
            curves_dir = sd / "curves" / "token_fair"
            if not curves_dir.exists():
                continue
            for p in curves_dir.glob("curves_*.csv"):
                parts = p.stem.split("_")
                if len(parts) < 3 or parts[0] != "curves":
                    continue
                model_kind = parts[1]
                tok = "_".join(parts[2:])
                xs, ys = _read_curve_csv(p)
                t95 = _t95_tokens(xs, ys)
                if t95 is None:
                    continue
                conv_rows.append(
                    {
                        "model": model_kind,
                        "tokenizer": tok,
                        "t95_tokens_m": float(t95) / 1_000_000.0,
                        "seed": int(seed),
                    }
                )
        if conv_rows:
            conv_sum = _group_stats(
                conv_rows,
                group_keys=["model", "tokenizer"],
                metric_keys=["t95_tokens_m"],
            )
            _write_latex_table(
                conv_sum,
                columns=[
                    Col("model", "Model", align="l"),
                    Col("tokenizer", "Tokenizer", align="l"),
                    Col("t95_tokens_m", "$t_{95}$ (M tokens)", align="r", metric="t95_tokens_m", digits=2),
                ],
                out_path=main_tables_dir / "e2_csic_convergence.tex",
                caption="E2 (CSIC 2010 HTTP) token-fair convergence: tokens to reach 95\\% of best validation accuracy (from learning curves).",
                label="tab:e2_csic_convergence",
            )
            wrote_tables += 1

        # Learning curve figure (mean across seeds, token-fair, Tiny).
        try:
            total_tokens = int(
                max((_to_float(r.get("total_tokens")) or 0.0) for r in e2_token_rows)  # type: ignore[arg-type]
            )
        except Exception:
            total_tokens = 0
        if total_tokens > 0 and seed_dirs:
            if _plot_e2_learning_curves_mean(
                seed_dirs,
                mode="token_fair",
                model_kind="tiny",
                total_tokens=total_tokens,
                out_dir=main_figs_dir,
                name="e2_csic_token_fair_learning_curves_tiny",
            ):
                wrote_figs += 2

    if e2_step_csvs:
        e2_step_rows = _collect_many(e2_step_csvs)
        e2_step_sum = _group_stats(
            e2_step_rows,
            group_keys=["model", "tokenizer"],
            metric_keys=["acc", "drift_acc", "avg_len", "p95_len"],
        )
        _write_latex_table(
            e2_step_sum,
            columns=[
                Col("model", "Model", align="l"),
                Col("tokenizer", "Tokenizer", align="l"),
                Col("acc", "Acc.", align="r", metric="acc", digits=3),
                Col("drift_acc", "Drift acc.", align="r", metric="drift_acc", digits=3),
                Col("avg_len", "avg\\_len", align="r", metric="avg_len", digits=1),
                Col("p95_len", "p95\\_len", align="r", metric="p95_len", digits=0, kind="int"),
            ],
            out_path=appendix_tables_dir / "e2_csic_step_fair.tex",
            caption="Appendix: E2 (CSIC 2010 HTTP) step-fair summary across seeds.",
            label="tab:appendix_e2_csic_step_fair",
        )
        wrote_tables += 1

    # E3 (Pareto slice)
    e3_dir = _pick_dir_with_seed_csv([run_root / "e3", run_root / "e3_pareto"], "results.csv")
    e3_csvs = [] if e3_dir is None else [p / "results.csv" for p in _seed_dirs(e3_dir) if (p / "results.csv").exists()]
    if e3_csvs:
        e3_rows = _collect_many(e3_csvs)
        e3_sum = _group_stats(
            e3_rows,
            group_keys=["backbone", "tokenizer"],
            metric_keys=["acc", "avg_len", "p95_len", "distortion_hat", "p95_latency_ms", "params_m"],
        )
        _write_latex_table(
            e3_sum,
            columns=[
                Col("backbone", "Backbone", align="l"),
                Col("tokenizer", "Tokenizer", align="l"),
                Col("params_m", "Params (M)", align="r", metric="params_m", digits=2),
                Col("acc", "Acc.", align="r", metric="acc", digits=3),
                Col("avg_len", "avg\\_len", align="r", metric="avg_len", digits=1),
                Col("p95_len", "p95\\_len", align="r", metric="p95_len", digits=0, kind="int"),
                Col("distortion_hat", "Distortion $\\hat{\\Delta}$", align="r", metric="distortion_hat", digits=3),
                Col("p95_latency_ms", "p95 latency (ms)", align="r", metric="p95_latency_ms", digits=2),
            ],
            out_path=main_tables_dir / "e3_pareto.tex",
            caption="E3 (Pareto slice) summary across seeds.",
            label="tab:e3_pareto",
        )
        wrote_tables += 1

    # ---- Appendix tables ----
    # E0 (tokenizer playground scan)
    e0_dir = _pick_dir_with_seed_csv([run_root / "appendix_e1", run_root / "e0_tokenizer_playground"], "results.csv")
    e0_csvs = [] if e0_dir is None else [p / "results.csv" for p in _seed_dirs(e0_dir) if (p / "results.csv").exists()]
    if e0_csvs:
        e0_rows = _collect_many(e0_csvs)
        e0_sum = _group_stats(
            e0_rows,
            group_keys=["dataset", "split", "tokenizer"],
            metric_keys=["avg_len", "p95_len", "p99_len", "trunc_rate_at_max_len", "tokenize_ms_per_sample"],
        )
        _write_latex_table(
            e0_sum,
            columns=[
                Col("dataset", "Dataset", align="l"),
                Col("split", "Split", align="l"),
                Col("tokenizer", "Tokenizer", align="l"),
                Col("avg_len", "avg\\_len", align="r", metric="avg_len", digits=1),
                Col("p95_len", "p95\\_len", align="r", metric="p95_len", digits=0, kind="int"),
                Col("p99_len", "p99\\_len", align="r", metric="p99_len", digits=0, kind="int"),
                Col("trunc_rate_at_max_len", "trunc\\_rate@L$_{\\max}$ (\\%)", align="r", metric="trunc_rate_at_max_len", digits=1, kind="pct"),
                Col("tokenize_ms_per_sample", "tokenize ms/sample", align="r", metric="tokenize_ms_per_sample", digits=2),
            ],
            out_path=appendix_tables_dir / "e0_tokenizer_playground.tex",
            caption="Appendix: tokenizer length/truncation scan (E0).",
            label="tab:appendix_e0_tokenizer_playground",
        )
        wrote_tables += 1

    # Frontier sweep (appendix)
    frontier_dir = _pick_dir_with_seed_csv(
        [run_root / "appendix_e4", run_root / "appendix_frontier", run_root / "e4_frontier"], "results.csv"
    )
    if frontier_dir is not None:
        frontier_csvs = [p / "results.csv" for p in _seed_dirs(frontier_dir) if (p / "results.csv").exists()]
        if frontier_csvs:
            frontier_rows = _collect_many(frontier_csvs)
            frontier_sum = _group_stats(
                frontier_rows,
                group_keys=["vocab_size", "tokenizer"],
                metric_keys=["acc", "avg_len", "p95_len", "distortion_hat", "tokenize_ms_per_sample"],
            )
            _write_latex_table(
                frontier_sum,
                columns=[
                    Col("vocab_size", "Vocab", align="r"),
                    Col("tokenizer", "Tokenizer", align="l"),
                    Col("acc", "Acc.", align="r", metric="acc", digits=3),
                    Col("avg_len", "avg\\_len", align="r", metric="avg_len", digits=1),
                    Col("p95_len", "p95\\_len", align="r", metric="p95_len", digits=0, kind="int"),
                    Col("distortion_hat", "Distortion $\\hat{\\Delta}$", align="r", metric="distortion_hat", digits=3),
                    Col("tokenize_ms_per_sample", "tokenize ms/sample", align="r", metric="tokenize_ms_per_sample", digits=2),
                ],
                out_path=appendix_tables_dir / "frontier.tex",
                caption="Appendix: vocab-budget frontier sweep.",
                label="tab:appendix_frontier",
            )
            wrote_tables += 1

    # UCI appendix
    uci_dir = _pick_dir_with_seed_csv_recursive([run_root / "appendix_e3", run_root / "appendix_uci", run_root / "e2_uci"], "results.csv")
    if uci_dir is not None:
        uci_csvs = _seed_csvs_recursive(uci_dir, "results.csv")
        if uci_csvs:
            uci_rows = _collect_many(uci_csvs)
            uci_sum = _group_stats(
                uci_rows,
                group_keys=["dataset", "tokenizer"],
                metric_keys=["acc", "avg_len", "p95_len", "distortion_hat"],
            )
            _write_latex_table(
                uci_sum,
                columns=[
                    Col("dataset", "Dataset", align="l"),
                    Col("tokenizer", "Tokenizer", align="l"),
                    Col("acc", "Acc.", align="r", metric="acc", digits=3),
                    Col("avg_len", "avg\\_len", align="r", metric="avg_len", digits=1),
                    Col("p95_len", "p95\\_len", align="r", metric="p95_len", digits=0, kind="int"),
                    Col("distortion_hat", "Distortion $\\hat{\\Delta}$", align="r", metric="distortion_hat", digits=3),
                ],
                out_path=appendix_tables_dir / "uci.tex",
                caption="Appendix: UCI benchmarks summary across seeds.",
                label="tab:appendix_uci",
            )
            wrote_tables += 1

    # ---- Figures ----
    # Copy existing plots (when present) for stable LaTeX paths.
    copied = 0

    # E2 figures (main)
    e2_main_seed = _pick_seed_dir(e2_main_dir) if e2_main_dir is not None else None
    if e2_main_seed is not None:
        to_copy_main = [
            ("token_fair_frontier_acc_vs_len.pdf", "e2_csic_token_fair_frontier_acc_vs_len.pdf"),
            ("token_fair_frontier_acc_vs_len.png", "e2_csic_token_fair_frontier_acc_vs_len.png"),
        ]
        for src_name, dst_name in to_copy_main:
            if _copy_if_exists(e2_main_seed / src_name, main_figs_dir / dst_name):
                copied += 1
                wrote_figs += 1

    # E2 step-fair figures (appendix)
    e2_step_seed = _pick_seed_dir(e2_step_dir) if e2_step_dir is not None else None
    if e2_step_seed is not None:
        to_copy_app = [
            ("step_fair_frontier_acc_vs_len.pdf", "e2_csic_step_fair_frontier_acc_vs_len.pdf"),
            ("step_fair_frontier_acc_vs_len.png", "e2_csic_step_fair_frontier_acc_vs_len.png"),
        ]
        for src_name, dst_name in to_copy_app:
            if _copy_if_exists(e2_step_seed / src_name, appendix_figs_dir / dst_name):
                copied += 1
                wrote_figs += 1

    # E3 figures (main): generated
    e3_seed = _pick_seed_dir(e3_dir) if e3_dir is not None else None
    if e3_seed is not None and (e3_seed / "results.csv").exists():
        _plot_e3_pareto(e3_seed / "results.csv", main_figs_dir)
        wrote_figs += 2  # pdf+png pairs per plot; approximate for reporting

    # Frontier figures (appendix)
    if frontier_dir is not None:
        frontier_seed = _pick_seed_dir(frontier_dir)
        if frontier_seed is not None:
            to_copy_app = [
                ("frontier_acc_vs_len.pdf", "frontier_acc_vs_len.pdf"),
                ("frontier_acc_vs_len.png", "frontier_acc_vs_len.png"),
                ("frontier_acc_vs_distortion.pdf", "frontier_acc_vs_distortion.pdf"),
                ("frontier_acc_vs_distortion.png", "frontier_acc_vs_distortion.png"),
                ("frontier_len_vs_distortion.pdf", "frontier_len_vs_distortion.pdf"),
                ("frontier_len_vs_distortion.png", "frontier_len_vs_distortion.png"),
                ("frontier_tokenize_ms_vs_len.pdf", "frontier_tokenize_ms_vs_len.pdf"),
                ("frontier_tokenize_ms_vs_len.png", "frontier_tokenize_ms_vs_len.png"),
            ]
            for src_name, dst_name in to_copy_app:
                if _copy_if_exists(frontier_seed / src_name, appendix_figs_dir / dst_name):
                    copied += 1
                    wrote_figs += 1

    # ---- Manifest ----
    manifest_lines: list[str] = []
    manifest_lines.append("# Paper artifacts (auto-generated)")
    manifest_lines.append("")
    manifest_lines.append("## Main paper (3 experiments)")
    manifest_lines.append("- E1 (synthetic): `main/tables/e1_synth.tex`")
    manifest_lines.append(
        "- E2 (CSIC HTTP): `main/tables/e2_csic_token_fair.tex`, `main/tables/e2_csic_convergence.tex` + `main/figures/e2_csic_token_fair_learning_curves_tiny.pdf`"
    )
    manifest_lines.append(
        "- E3 (Pareto slice): `main/tables/e3_pareto.tex` + `main/figures/e3_pareto_triptych.pdf` (and per-metric `e3_pareto_*.pdf`)"
    )
    manifest_lines.append("")
    manifest_lines.append("## Appendix")
    manifest_lines.append("- Tokenizer scan (E0): `appendix/tables/e0_tokenizer_playground.tex` (if present)")
    manifest_lines.append("- E2 step-fair variant: `appendix/tables/e2_csic_step_fair.tex` + `appendix/figures/e2_csic_step_fair_frontier_acc_vs_len.pdf` (if present)")
    manifest_lines.append("- UCI benchmarks: `appendix/tables/uci.tex` (if present)")
    manifest_lines.append("- Vocab frontier sweep: `appendix/tables/frontier.tex` + `appendix/figures/frontier_*.pdf` (if present)")
    manifest_lines.append("")
    manifest_lines.append("## Notes")
    manifest_lines.append("- Tables require `\\usepackage{booktabs}`.")
    manifest_lines.append("- If multiple seeds are present, tables show mean ± std; otherwise they show the single-seed value.")
    (outroot / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {wrote_tables} tables under: {outroot}")
    print(f"[OK] Wrote figures under: {outroot} (copied {copied} existing files)")


if __name__ == "__main__":
    main()
