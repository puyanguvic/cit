#!/usr/bin/env python3
"""Summarize main-paper results across seeds.

This script reads per-seed CSVs under results/<run_root>/ and writes
mean/std summaries (grouped by tokenizer/model) to results/<run_root>/summary/.

Default run_root assumes the paper runner:
  python scripts/run_all.py --outroot paper ...
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable


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
    return sorted(out, key=lambda p: p.name)


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
                out[f"{mk}_mean"] = ""
                out[f"{mk}_std"] = ""
                continue
            out[f"{mk}_mean"] = statistics.mean(vals)
            out[f"{mk}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out_rows.append(out)
    return out_rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _collect_many(csv_paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in csv_paths:
        for r in _read_csv(p):
            r["_source"] = str(p)
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=str, default="paper", help="Folder under results/ (default: paper)")
    ap.add_argument("--outdir", type=str, default="", help="Optional override output dir (default: results/<run-root>/summary)")
    args = ap.parse_args()

    run_root = Path("results") / args.run_root
    outdir = Path(args.outdir) if args.outdir else (run_root / "summary")

    # E1
    e1_dir = run_root / "e1_synth"
    e1_csvs = [p / "results.csv" for p in _seed_dirs(e1_dir) if (p / "results.csv").exists()]
    e1_rows = _collect_many(e1_csvs)
    e1_sum = _group_stats(
        e1_rows,
        group_keys=["tokenizer"],
        metric_keys=["test_acc", "drift_acc", "rate_E_len", "distortion_hat", "oracle_acc", "oracle_drift_acc"],
    )
    _write_csv(e1_sum, outdir / "e1_summary.csv")

    # E2 (token-fair + step-fair)
    e2_dir = run_root / "e2_csic_http"
    e2_token_csvs = [p / "results_token_fair.csv" for p in _seed_dirs(e2_dir) if (p / "results_token_fair.csv").exists()]
    e2_step_csvs = [p / "results_step_fair.csv" for p in _seed_dirs(e2_dir) if (p / "results_step_fair.csv").exists()]
    e2_token_rows = _collect_many(e2_token_csvs)
    e2_step_rows = _collect_many(e2_step_csvs)

    e2_token_sum = _group_stats(
        e2_token_rows,
        group_keys=["dataset", "model", "tokenizer", "budget_mode"],
        metric_keys=["acc", "drift_acc", "avg_len", "p95_len", "total_steps", "total_tokens"],
    )
    _write_csv(e2_token_sum, outdir / "e2_token_fair_summary.csv")

    e2_step_sum = _group_stats(
        e2_step_rows,
        group_keys=["dataset", "model", "tokenizer", "budget_mode"],
        metric_keys=["acc", "drift_acc", "avg_len", "p95_len", "total_steps", "total_tokens"],
    )
    _write_csv(e2_step_sum, outdir / "e2_step_fair_summary.csv")

    # E3
    e3_dir = run_root / "e3_pareto"
    e3_csvs = [p / "results.csv" for p in _seed_dirs(e3_dir) if (p / "results.csv").exists()]
    e3_rows = _collect_many(e3_csvs)
    e3_sum = _group_stats(
        e3_rows,
        group_keys=["backbone", "tokenizer"],
        metric_keys=["acc", "avg_len", "p95_len", "distortion_hat", "p95_latency_ms"],
    )
    _write_csv(e3_sum, outdir / "e3_summary.csv")

    print(f"[OK] Wrote summaries under: {outdir}")


if __name__ == "__main__":
    main()

