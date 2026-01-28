"""E0: Off-the-shelf tokenizer scan (tokenizer playground, local).

Motivation:
  The paper argues that canonical NLP tokenizers (e.g., BERT WordPiece, GPT BPE)
  behave poorly when used as drop-in interfaces for structured/semi-structured
  streams. This script quantifies that effect by running *pretrained* public
  tokenizers on the paper's serialized datasets and reporting token-length
  distributions (avg/P95/P99, truncation rates, etc.).

This is inspired by:
  https://huggingface.co/spaces/Xenova/the-tokenizer-playground

Example:
  python scripts/run_e0_tokenizer_playground.py --dataset csic2010 --data-dir data/csic2010 --auto-download
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from cit.utils.artifacts import save_json, save_run_metadata


def _quantiles(lengths: List[int], qs: Tuple[float, ...]) -> Dict[str, float]:
    if not lengths:
        return {f"p{int(q*100):02d}": 0.0 for q in qs}
    arr = np.asarray(lengths, dtype=np.float32)
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(q*100):02d}"] = float(np.quantile(arr, q))
    return out


def _safe_int(x: float) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _encode_lengths(tok, texts: List[str]) -> Tuple[List[int], float]:
    """Return per-sample token lengths and avg ms/sample."""
    t0 = time.perf_counter()
    # Fast path: batch encode if available.
    try:
        out = tok(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        ids = out["input_ids"]
    except Exception:
        ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    dt = time.perf_counter() - t0
    lens = [len(x) for x in ids]
    ms = (dt * 1000.0) / max(1, len(texts))
    return lens, float(ms)


def _load_texts(args) -> Tuple[str, str, List[str], Optional[List[int]]]:
    """Return (dataset_name, split_name, texts, labels_or_none)."""
    if args.dataset == "csic2010":
        from cit.data.http_csic import load_csic2010_http

        # Optional auto-download (mirrors E2).
        if args.auto_download:
            data_dir = Path(args.data_dir)
            have_dir = data_dir.exists()
            has_data = False
            if have_dir:
                for name in ["csic2010.csv", "csic2010.tsv", "data.csv", "data.tsv"]:
                    if (data_dir / name).exists():
                        has_data = True
                        break
                if not has_data:
                    for pat in ["normal.txt", "normalTraffic*.txt", "anomalous.txt", "anomalousTraffic*.txt"]:
                        if list(data_dir.glob(pat)):
                            has_data = True
                            break
            if not have_dir or not has_data:
                setup_script = Path(__file__).parent / "setup_csic2010_lexr.py"
                subprocess.check_call([sys.executable, str(setup_script), "--out", str(data_dir)])

        Xtr, ytr, Xva, yva, Xte, yte = load_csic2010_http(
            args.data_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            seed=args.seed,
            include_raw=bool(args.include_raw),
        )
        split = args.split
        if split == "train":
            return "csic2010", split, Xtr, ytr
        if split == "val":
            return "csic2010", split, Xva, yva
        if split == "test":
            return "csic2010", split, Xte, yte
        if split == "all":
            return "csic2010", split, Xtr + Xva + Xte, ytr + yva + yte
        raise ValueError(f"Unknown split: {split}")

    if args.dataset in {"adult", "credit-g"}:
        from cit.data.uci import load_adult, load_german_credit
        from cit.data.serialize import SerializeCfg, serialize_df

        ds = load_adult(args.seed) if args.dataset == "adult" else load_german_credit(args.seed)
        ser_cfg = SerializeCfg()
        Xtr = serialize_df(ds.X_train, ser_cfg)
        Xte = serialize_df(ds.X_test, ser_cfg)
        if args.split == "train":
            return args.dataset, args.split, Xtr, ds.y_train.tolist()
        if args.split == "test":
            return args.dataset, args.split, Xte, ds.y_test.tolist()
        if args.split == "all":
            return args.dataset, args.split, Xtr + Xte, ds.y_train.tolist() + ds.y_test.tolist()
        raise ValueError("UCI datasets support split=train,test,all")

    raise ValueError(f"Unknown dataset: {args.dataset}")


def _parse_tokenizer_specs(spec: str) -> List[Tuple[str, str]]:
    """Parse comma-separated list of tokenizer specs.

    Accepts either:
      - a Hugging Face repo id: bert-base-cased
      - an explicit name=repo_id pair: grok1=Xenova/grok-1-tokenizer
    """
    out: List[Tuple[str, str]] = []
    for part in spec.split(","):
        s = part.strip()
        if not s:
            continue
        if "=" in s:
            name, repo = s.split("=", 1)
            out.append((name.strip(), repo.strip()))
        else:
            out.append((s, s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="csic2010",
        choices=["csic2010", "adult", "credit-g"],
        help="Dataset to tokenize (default: csic2010).",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="paper/e0_tokenizer_playground")

    # CSIC options
    ap.add_argument("--data-dir", type=str, default="data/csic2010")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--include-raw", action="store_true")
    ap.add_argument("--auto-download", action="store_true")

    # Tokenizers
    ap.add_argument(
        "--tokenizers",
        type=str,
        default="bert-base-cased,gpt2,gpt3=Xenova/text-davinci-003,gpt35=Xenova/gpt-3.5-turbo,gpt4=Xenova/gpt-4,grok1=Xenova/grok-1-tokenizer",
        help="Comma-separated list of tokenizer repo ids or name=repo_id pairs.",
    )
    ap.add_argument(
        "--hf-cache",
        type=str,
        default="results/hf_cache/transformers",
        help="HF cache directory (must be writable; default is under results/).",
    )
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--local-files-only", action="store_true", help="Do not hit the network (requires cached tokenizers).")

    # Measurement
    ap.add_argument("--max-samples", type=int, default=5000, help="Cap number of samples for speed (0=all).")
    ap.add_argument("--max-len", type=int, default=512, help="Truncation length for reporting truncation rates.")
    args = ap.parse_args()

    # Keep this script torch-free (tokenizer-only) to avoid CUDA init noise.
    import random

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    os.environ["PYTHONHASHSEED"] = str(int(args.seed))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Ensure cache writes stay inside the workspace.
    cache_dir = Path(args.hf_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.parent))

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e0_tokenizer_playground", args=vars(args))

    dataset_name, split_name, texts, labels = _load_texts(args)
    if args.max_samples and args.max_samples > 0:
        texts = texts[: int(args.max_samples)]
        if labels is not None:
            labels = labels[: int(args.max_samples)]

    # Lazy import to keep script import-light.
    from transformers import AutoTokenizer

    specs = _parse_tokenizer_specs(args.tokenizers)
    rows: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    for display_name, repo_id in specs:
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "split": split_name,
            "n": len(texts),
            "tokenizer": display_name,
            "hf_id": repo_id,
            "ok": 0,
        }
        try:
            tok = AutoTokenizer.from_pretrained(
                repo_id,
                cache_dir=str(cache_dir),
                trust_remote_code=bool(args.trust_remote_code),
                local_files_only=bool(args.local_files_only),
            )
            # This scan intentionally measures long sequences; avoid noisy warnings
            # about exceeding model_max_length for tokenizers with small defaults.
            try:
                tok.model_max_length = int(1e9)
            except Exception:
                pass
            row["tok_class"] = tok.__class__.__name__
            row["is_fast"] = int(getattr(tok, "is_fast", False))
            row["vocab_size"] = int(getattr(tok, "vocab_size", len(tok)))

            lens, ms = _encode_lengths(tok, texts)
            row["avg_len"] = float(np.mean(lens)) if lens else 0.0
            row["max_len"] = int(max(lens)) if lens else 0
            row.update({k + "_len": _safe_int(v) for k, v in _quantiles(lens, (0.50, 0.90, 0.95, 0.99)).items()})
            row["trunc_rate_at_max_len"] = float(sum(1 for x in lens if x > int(args.max_len))) / float(max(1, len(lens)))
            row["tokenize_ms_per_sample"] = float(ms)
            row["ok"] = 1
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            errors[display_name] = err
            row["error"] = err
        rows.append(row)

    # Save results
    out_csv = outdir / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows, "errors": errors}, outdir / "results.json")

    # Short Markdown summary for copy-paste into the paper draft.
    md_lines = [
        f"# E0 tokenizer scan: {dataset_name} ({split_name}, n={len(texts)})",
        "",
        f"| tokenizer | avg_len | p95_len | p99_len | max_len | trunc@{int(args.max_len)} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if not int(r.get("ok", 0)):
            md_lines.append(f"| {r['tokenizer']} | - | - | - | - | - |")
            continue
        md_lines.append(
            "| {tok} | {avg:.1f} | {p95} | {p99} | {mx} | {tr:.2%} |".format(
                tok=r["tokenizer"],
                avg=float(r.get("avg_len", 0.0)),
                p95=int(r.get("p95_len", 0)),
                p99=int(r.get("p99_len", 0)),
                mx=int(r.get("max_len", 0)),
                tr=float(r.get("trunc_rate_at_max_len", 0.0)),
                max_len=int(args.max_len),
            )
        )
    (outdir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[wrote] {out_csv}")
    print(f"[wrote] {outdir / 'summary.md'}")
    if errors:
        print("[warn] Some tokenizers failed to load:")
        for k, v in errors.items():
            print("  -", k, "->", v)


if __name__ == "__main__":
    main()
