#!/usr/bin/env python3
"""Run multi-budget sweeps to increase figure density.

This script runs E2 and/or E3 over a list of token budgets (total non-pad encoder
tokens), aggregates results across seeds, and writes a high-density scaling plot.

Typical usage (re-run after changing code / plots):

  python scripts/run_token_budget_sweep.py --exp e2 --budgets 1M,2M,5M,10M --seeds 0,1,2 --device cuda
  python scripts/run_token_budget_sweep.py --exp e3 --budgets 1M,2M,5M,10M --seeds 0 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCRIPTS_DIR = Path(__file__).resolve().parent


def _run(pyfile: str, args: list[str], *, dry_run: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / pyfile)] + args
    print("[RUN]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(cmd)


def _parse_count(s: str) -> int:
    raw = s.strip().replace("_", "")
    if not raw:
        raise ValueError("Empty token-count entry")

    mult = 1
    if raw[-1] in {"k", "K", "m", "M", "b", "B"}:
        suf = raw[-1]
        raw = raw[:-1].strip()
        mult = {"k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000, "b": 1_000_000_000, "B": 1_000_000_000}[suf]

    try:
        val = float(raw)
    except Exception as e:
        raise ValueError(f"Invalid token-count entry: {s!r}") from e

    out = int(round(val * mult))
    if out <= 0:
        raise ValueError(f"Non-positive token count: {s!r}")
    return out


def _parse_count_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty list")
    out: list[int] = []
    for p in parts:
        v = _parse_count(p)
        if v not in out:
            out.append(v)
    return out


def _fmt_count(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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


def _group_mean_std(rows: list[dict], *, group_keys: list[str], metric_keys: list[str]) -> list[dict]:
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
                out[mk] = ""
                out[f"{mk}_std"] = ""
                continue
            out[mk] = statistics.mean(vals)
            out[f"{mk}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out_rows.append(out)
    return out_rows


def _summarize_e2(run_root: Path) -> tuple[list[dict], list[dict]]:
    sweep_root = run_root / "e2_budget_sweep"
    csvs: list[Path] = []
    for budget_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()]) if sweep_root.exists() else []:
        for sd in _seed_dirs(budget_dir):
            p = sd / "results_token_fair.csv"
            if p.exists():
                csvs.append(p)

    rows = _collect_many(csvs)
    summary = _group_mean_std(
        rows,
        group_keys=["dataset", "model", "tokenizer", "budget_mode", "total_tokens"],
        metric_keys=["acc", "drift_acc", "avg_len", "p95_len", "total_steps"],
    )
    return rows, summary


def _summarize_e3(run_root: Path) -> tuple[list[dict], list[dict]]:
    sweep_root = run_root / "e3_budget_sweep"
    csvs: list[Path] = []
    for budget_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()]) if sweep_root.exists() else []:
        for sd in _seed_dirs(budget_dir):
            p = sd / "results.csv"
            if p.exists():
                csvs.append(p)

    rows = _collect_many(csvs)
    summary = _group_mean_std(
        rows,
        group_keys=["tokenizer", "backbone", "total_tokens"],
        metric_keys=["acc", "avg_len", "p95_len", "distortion_hat", "p95_latency_ms", "params_m"],
    )
    return rows, summary


def _plot_e2(summary_rows: list[dict], *, outpath: Path) -> None:
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from plot_frontier import plot_frontier  # type: ignore
    except Exception:
        print("[warn] Could not import scripts/plot_frontier.py; skipping E2 sweep plot.")
        return

    if not summary_rows:
        return

    plot_frontier(
        summary_rows,
        group_key="tokenizer",
        x_key="total_tokens",
        y_key="acc",
        xlabel="token budget (non-pad encoder tokens, log scale)",
        ylabel="accuracy",
        outpath=outpath,
    )


def _plot_e3(summary_rows: list[dict], *, outpath: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import matplotlib.ticker as mticker
    except Exception:
        print("[warn] matplotlib not available; skipping E3 sweep plot.")
        return

    if not summary_rows:
        return

    # Use the same color palette as plot_frontier/exporter when available.
    tok_colors = {
        "BPE": "#0072B2",
        "Bytes": "#E69F00",
        "CIT": "#009E73",
        "Unigram": "#D55E00",
        "WordPiece": "#CC79A7",
    }

    def _tok_color(tok: str) -> str:
        return tok_colors.get(tok, "#333333")

    backbones = sorted({str(r.get("backbone", "")) for r in summary_rows if str(r.get("backbone", ""))})
    tokenizers = sorted({str(r.get("tokenizer", "")) for r in summary_rows if str(r.get("tokenizer", ""))})
    tokenizers = sorted(tokenizers, key=lambda n: (0, n) if n in tok_colors else (1, n))

    fig, axes = plt.subplots(1, max(1, len(backbones)), figsize=(10.4 if len(backbones) > 1 else 6.6, 3.1), sharey=True)
    try:
        axes_list = list(axes)  # type: ignore[arg-type]
    except TypeError:
        axes_list = [axes]  # type: ignore[list-item]

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

    for ax, bb in zip(axes_list, backbones, strict=False):
        bb_rows = [r for r in summary_rows if str(r.get("backbone", "")) == bb]
        for tok in tokenizers:
            pts = []
            for r in bb_rows:
                if str(r.get("tokenizer", "")) != tok:
                    continue
                x = _to_float(r.get("total_tokens"))
                y = _to_float(r.get("acc"))
                yerr = _to_float(r.get("acc_std"))
                if x is None or y is None:
                    continue
                pts.append((x, y, 0.0 if yerr is None else yerr))
            if not pts:
                continue
            pts.sort(key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            yerrs = [p[2] for p in pts]
            is_highlight = tok == "CIT"
            ax.plot(
                xs,
                ys,
                color=_tok_color(tok),
                linewidth=2.2 if is_highlight else 1.2,
                alpha=0.95 if is_highlight else 0.75,
                marker="o",
                markersize=5.5,
            )
            if any(e > 0.0 for e in yerrs):
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    fmt="none",
                    ecolor=_tok_color(tok),
                    elinewidth=0.9,
                    capsize=2,
                    alpha=0.55 if is_highlight else 0.35,
                )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_tokens))
        ax.xaxis.get_offset_text().set_visible(False)
        ax.set_xlabel("token budget (non-pad encoder tokens, log scale)")
        ax.set_title(bb)
        ax.grid(True, alpha=0.25)
        ax.text(0.98, 0.04, "better ↖", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#333333")

    axes_list[0].set_ylabel("accuracy")

    tok_handles = [
        Line2D([0], [0], marker="o", color=_tok_color(tok), linestyle="-", label=tok, linewidth=2.0) for tok in tokenizers
    ]
    fig.legend(
        handles=tok_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=min(5, len(tok_handles)),
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="e2", choices=["e2", "e3", "all"])
    ap.add_argument("--budgets", type=str, default="1M,2M,5M,10M", help="Comma-separated token budgets.")
    ap.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds (e.g., 0,1,2).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--outroot", type=str, default="paper", help="Root folder under results/ for sweep outputs.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute.")
    ap.add_argument("--summarize-only", action="store_true", help="Skip running; just aggregate + plot.")
    ap.add_argument("--plot", action="store_true", help="Write sweep plots after aggregation.")

    # E2 knobs (defaults match the main paper)
    ap.add_argument("--e2-data-dir", type=str, default="data/csic2010")
    ap.add_argument("--e2-vocab", type=int, default=2048)
    ap.add_argument("--e2-max-len", type=int, default=512)
    ap.add_argument("--e2-models", type=str, default="tiny,small")
    ap.add_argument("--e2-budget-modes", type=str, default="token", help="Comma-separated: token,step (default token).")
    ap.add_argument("--auto-download", dest="auto_download", action="store_true")
    ap.add_argument("--no-auto-download", dest="auto_download", action="store_false")
    ap.set_defaults(auto_download=True)

    # E3 knobs (defaults match the main paper)
    ap.add_argument("--e3-vocab", type=int, default=2048)
    ap.add_argument("--e3-max-len", type=int, default=512)
    ap.add_argument("--e3-model-families", type=str, default="mini,small")
    ap.add_argument("--e3-full-finetune", dest="e3_full_finetune", action="store_true")
    ap.add_argument("--e3-probe-only", dest="e3_full_finetune", action="store_false")
    ap.set_defaults(e3_full_finetune=True)

    args = ap.parse_args()

    budgets = _parse_count_list(args.budgets)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("Empty --seeds list")

    run_root = Path("results") / args.outroot
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        if args.exp in {"e2", "all"}:
            for total_tokens in budgets:
                tag = f"tok{_fmt_count(total_tokens)}"
                for seed in seeds:
                    e2_args = [
                        "--data-dir",
                        args.e2_data_dir,
                        "--outdir",
                        f"{args.outroot}/e2_budget_sweep/{tag}",
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--vocab",
                        str(args.e2_vocab),
                        "--max-len",
                        str(args.e2_max_len),
                        "--models",
                        args.e2_models,
                        "--total-tokens",
                        str(total_tokens),
                        "--budget-modes",
                        args.e2_budget_modes,
                    ]
                    if args.auto_download:
                        e2_args.append("--auto-download")
                    _run("run_e2_csic_http.py", e2_args, dry_run=bool(args.dry_run))

        if args.exp in {"e3", "all"}:
            for total_tokens in budgets:
                tag = f"tok{_fmt_count(total_tokens)}"
                for seed in seeds:
                    e3_args = [
                        "--outdir",
                        f"{args.outroot}/e3_budget_sweep/{tag}",
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--vocab",
                        str(args.e3_vocab),
                        "--max-len",
                        str(args.e3_max_len),
                        "--model-families",
                        args.e3_model_families,
                        "--total-tokens",
                        str(total_tokens),
                    ]
                    if args.e3_full_finetune:
                        e3_args.append("--full-finetune")
                    _run("run_e3_pareto_prefix.py", e3_args, dry_run=bool(args.dry_run))

    # Aggregation (can run even in --summarize-only mode).
    if args.exp in {"e2", "all"}:
        e2_rows, e2_sum = _summarize_e2(run_root)
        e2_root = run_root / "e2_budget_sweep"
        _write_csv(e2_rows, e2_root / "all_results_token_fair.csv")
        _write_csv(e2_sum, e2_root / "summary_token_fair.csv")
        if args.plot:
            _plot_e2(e2_sum, outpath=e2_root / "token_fair_acc_vs_tokens")

    if args.exp in {"e3", "all"}:
        e3_rows, e3_sum = _summarize_e3(run_root)
        e3_root = run_root / "e3_budget_sweep"
        _write_csv(e3_rows, e3_root / "all_results.csv")
        _write_csv(e3_sum, e3_root / "summary.csv")
        if args.plot:
            _plot_e3(e3_sum, outpath=e3_root / "acc_vs_tokens")

    print(f"[OK] Wrote sweep outputs under: {run_root}")


if __name__ == "__main__":
    main()

