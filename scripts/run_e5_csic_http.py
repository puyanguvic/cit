"""E5: End-to-end training on a structured public HTTP dataset (CSIC 2010).

This experiment is meant to validate CIT under a *controlled* end-to-end setup:
we train the same encoder family from scratch for each tokenizer, holding
architecture and token-budget constant.

The script expects the user to place CSIC 2010 raw files under --data-dir.
See cit.data.http_csic.load_csic2010_http for supported layouts.

Example:
  python scripts/run_e5_csic_http.py --data-dir data/csic2010 --device cuda --seed 0
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple

from cit.data.http_csic import load_csic2010_http
from cit.data.drift import drift_reorder_kv_groups, drift_whitespace
from cit.tokenizers.metrics import estimate_rate
from cit.tokenizers.hf_baselines import train_bpe, train_wordpiece, train_unigram
from cit.tokenizers.cit_contract import Contract, apply_contract
from cit.tokenizers.cit_induction import InductionCfg, train_cit
from cit.tokenizers.runtime import tokenize_longest_match
from cit.models.encoder import TinyEncoder, SmallEncoder, BaseEncoder
from cit.models.train import train_compute_matched
from cit.models.eval import evaluate
from cit.utils.artifacts import save_json, save_run_metadata, save_vocab_json
from cit.utils.seed import set_seed


def hf_encode(tok, texts: List[str]) -> List[List[int]]:
    return [tok.encode(t).ids for t in texts]


def build_drift_texts(texts: List[str], seed: int) -> List[str]:
    """Build a realistic robustness slice.

    For HTTP-like records, fully shuffling *all* serialized fields is often too
    destructive (it can effectively change semantics). Here we use gentler,
    record-aware operators: reorder query/body/header groups and normalize
    whitespace.
    """
    out = []
    for i, x in enumerate(texts):
        y = drift_reorder_kv_groups(x, seed=seed + i)
        y = drift_whitespace(y)
        out.append(y)
    return out


def make_model(kind: str, vocab_size: int, n_classes: int, max_len: int):
    if kind == "tiny":
        return TinyEncoder(vocab_size=vocab_size, n_classes=n_classes, max_len=max_len)
    if kind == "small":
        return SmallEncoder(vocab_size=vocab_size, n_classes=n_classes, max_len=max_len)
    if kind == "base":
        return BaseEncoder(vocab_size=vocab_size, n_classes=n_classes, max_len=max_len)
    raise ValueError(f"Unknown model kind: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing CSIC 2010 raw files")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--models",
        type=str,
        default="tiny,small",
        help="Comma-separated model sizes to run: tiny,small,base",
    )
    ap.add_argument(
        "--total-tokens",
        type=int,
        default=10_000_000,
        help="Encoder-token budget per (tokenizer, model) run (token-fair training).",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="paper/e5_csic_http")
    ap.add_argument(
        "--include-raw",
        action="store_true",
        help="Include the raw HTTP request field in serialization (off by default for tokenizer-centric runs).",
    )
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument(
        "--auto-download",
        action="store_true",
        help="If set, download and prepare the CSIC 2010 CSVs from lexr.ai when data-dir is missing.",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e5_csic_http", args=vars(args))

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
            print(f"[auto-download] Preparing CSIC 2010 dataset under {data_dir} ...")
            subprocess.check_call([sys.executable, str(setup_script), "--out", str(data_dir)])

    Xtr, ytr, Xva, yva, Xte, yte = load_csic2010_http(
        args.data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        include_raw=bool(args.include_raw),
    )
    n_classes = len(set(ytr))

    # Class-imbalance handling (important for HTTP anomaly detection): compute
    # inverse-frequency weights so training does not collapse to majority class.
    counts = [0] * n_classes
    for yy in ytr:
        if 0 <= int(yy) < n_classes:
            counts[int(yy)] += 1
    total = max(1, sum(counts))
    class_weights = [float(total / max(1, c)) for c in counts]

    # Train tokenizers on training serialization.
    bpe = train_bpe(Xtr, vocab_size=args.vocab)
    wp = train_wordpiece(Xtr, vocab_size=args.vocab)
    uni = train_unigram(Xtr, vocab_size=args.vocab)

    contract = Contract()
    cit_vocab, cit_contract = train_cit(
        Xtr,
        ytr,
        contract,
        InductionCfg(vocab_size=args.vocab, seed=args.seed),
        log_path=str(outdir / "tokenizers" / "cit" / "induction_log.jsonl"),
    )

    # Save tokenizer artifacts.
    tok_dir = outdir / "tokenizers"
    (tok_dir / "bpe").mkdir(parents=True, exist_ok=True)
    (tok_dir / "wordpiece").mkdir(parents=True, exist_ok=True)
    (tok_dir / "unigram").mkdir(parents=True, exist_ok=True)
    (tok_dir / "cit").mkdir(parents=True, exist_ok=True)
    bpe.save(str(tok_dir / "bpe" / "tokenizer.json"))
    wp.save(str(tok_dir / "wordpiece" / "tokenizer.json"))
    uni.save(str(tok_dir / "unigram" / "tokenizer.json"))
    save_vocab_json(cit_vocab, tok_dir / "cit" / "vocab.json")
    save_json(cit_contract, tok_dir / "cit" / "contract.json")

    # Drift slice (role/boundary stress under the same serializer markers).
    Xte_drift = build_drift_texts(Xte, seed=args.seed)

    encoders: List[Tuple[str, Callable[[List[str]], List[List[int]]]]]
    encoders = [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("Unigram", lambda t: hf_encode(uni, t)),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab) for s in t]),
    ]

    model_kinds = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for model_kind in model_kinds:
        for tok_name, enc in encoders:
            tr_ids = enc(Xtr)
            va_ids = enc(Xva)
            te_ids = enc(Xte)
            te_drift_ids = enc(Xte_drift)

            pad_id = 0
            train_pairs = list(zip(tr_ids, ytr))
            val_pairs = list(zip(va_ids, yva))
            test_pairs = list(zip(te_ids, yte))
            drift_pairs = list(zip(te_drift_ids, yte))

            model = make_model(model_kind, vocab_size=args.vocab, n_classes=n_classes, max_len=args.max_len)

            # Conservative LR scaling for larger models to avoid collapse.
            lr_eff = float(args.lr)
            if model_kind == "base":
                lr_eff = min(lr_eff, 2e-4)

            log_path = outdir / "train_logs" / f"{model_kind}_{tok_name}.jsonl"
            model = train_compute_matched(
                model,
                train_pairs,
                pad_id,
                args.max_len,
                total_tokens=args.total_tokens,
                device=args.device,
                batch_size=args.batch_size,
                lr=lr_eff,
                probe_only=False,
                grad_clip=float(args.grad_clip),
                class_weights=class_weights,
                warmup_tokens=max(50_000, int(args.total_tokens // 20)),
                log_path=str(log_path),
                eval_pairs=val_pairs,
                eval_every_tokens=max(50_000, args.total_tokens // 50),
            )

            acc = evaluate(model, test_pairs, pad_id, args.max_len, device=args.device)
            drift_acc = evaluate(model, drift_pairs, pad_id, args.max_len, device=args.device)

            avg_len = estimate_rate(te_ids)
            p95 = sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]

            row = {
                "dataset": "csic2010",
                "model": model_kind,
                "tokenizer": tok_name,
                "acc": float(acc),
                "drift_acc": float(drift_acc),
                "avg_len": float(avg_len),
                "p95_len": int(p95),
                "vocab": int(args.vocab),
                "max_len": int(args.max_len),
                "total_tokens": int(args.total_tokens),
                "seed": int(args.seed),
            }
            rows.append(row)
            print(
                f"csic2010\t{model_kind}\t{tok_name}\tacc={acc:.4f}\tdrift_acc={drift_acc:.4f}\tavg_len={avg_len:.1f}\tp95_len={p95}"
            )

    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows}, outdir / "results.json")
    print(f"[wrote] {outdir / 'results.csv'}")


if __name__ == "__main__":
    main()
