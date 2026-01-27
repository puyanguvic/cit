"""E5: End-to-end controlled training on a public phishing-email dataset.

Goal:
  Validate whether CIT improves *end-to-end* detection training under a strict
  controlled setting:
    - same encoder family (TinyEncoder)
    - same parameterization / depth / width
    - same training schedule and optimizer
    - only tokenizer differs

Dataset:
  Hugging Face: zefang-liu/phishing-email-dataset (~18.6k rows)

Outputs (all under results/...):
  - results.csv : aggregate metrics (acc, avg_len, p95_len, p95_ms, etc.)
  - curves_*.csv : step-wise training curves for loss/acc
  - plots/*.png (if --plot) : learning curves + length/speed summaries
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cit.data.hf_phishing import load_hf_phishing_email_dataset
from cit.models.encoder import TinyEncoder
from cit.models.train import train_seqcls, tokenize_pairs
from cit.tokenizers.baselines import build_bpe_tokenizer, build_wordpiece_tokenizer, build_unigram_tokenizer
from cit.tokenizers.cit import build_cit_tokenizer
from cit.tokenizers.metrics import estimate_rate


def measure_p95_infer_ms(model, pairs, *, device: str, batch_size: int = 32, n_batches: int = 50, max_len: int = 256) -> float:
    """Rudimentary p95 inference latency (ms) over tokenized (ids,label) pairs."""
    import time
    import torch

    rng = np.random.default_rng(0)
    model.eval()
    model.to(device)

    def make_batch(batch_pairs):
        ids = [p[0][:max_len] for p in batch_pairs]
        L = max(len(s) for s in ids) if ids else 1
        L = min(L, max_len)
        x = torch.zeros((len(ids), L), dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for i, s in enumerate(ids):
            s = s[:L]
            if len(s):
                x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                m[i, : len(s)] = 1
        return x, m

    # Warmup
    for _ in range(5):
        batch = [pairs[i] for i in rng.integers(0, len(pairs), size=min(batch_size, len(pairs)))]
        x, m = make_batch(batch)
        x, m = x.to(device), m.to(device)
        with torch.no_grad():
            _ = model(x, m)

    times = []
    if device.startswith("cuda") and torch.cuda.is_available():
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(n_batches):
            batch = [pairs[i] for i in rng.integers(0, len(pairs), size=min(batch_size, len(pairs)))]
            x, m = make_batch(batch)
            x, m = x.to(device), m.to(device)
            starter.record()
            with torch.no_grad():
                _ = model(x, m)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    else:
        for _ in range(n_batches):
            batch = [pairs[i] for i in rng.integers(0, len(pairs), size=min(batch_size, len(pairs)))]
            x, m = make_batch(batch)
            x, m = x.to(device), m.to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x, m)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return float(np.percentile(times, 95)) if times else float("nan")


def _ensure_outdir(outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    return outdir


def _tokenizer_suite(vocab_size: int, seed: int):
    # Keep baselines simple and deterministic for comparability.
    return {
        "BPE": lambda corpus: build_bpe_tokenizer(corpus=corpus, vocab_size=vocab_size, seed=seed),
        "WordPiece": lambda corpus: build_wordpiece_tokenizer(corpus=corpus, vocab_size=vocab_size, seed=seed),
        "Unigram": lambda corpus: build_unigram_tokenizer(corpus=corpus, vocab_size=vocab_size, seed=seed),
        "CIT": lambda corpus: build_cit_tokenizer(corpus=corpus, vocab_size=vocab_size, seed=seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="paper/e5_end2end_phish")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument(
        "--total_tokens",
        type=int,
        default=5_000_000,
        help="Training budget measured in *non-pad encoder tokens* (token-fair).",
    )
    ap.add_argument(
        "--eval_every_tokens",
        type=int,
        default=200_000,
        help="How often to evaluate/log, in seen encoder tokens.",
    )
    ap.add_argument(
        "--model_families",
        type=str,
        default="mini",
        help="Comma-separated model families (mini,small,medium).",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    outdir = _ensure_outdir(Path("results") / args.outdir / f"seed{args.seed}")

    # Load dataset (auto-download via HF datasets)
    train_ex, test_ex = load_hf_phishing_email_dataset(seed=args.seed, max_samples=args.max_samples)
    train_texts = [x.text for x in train_ex]
    test_texts = [x.text for x in test_ex]
    corpus = train_texts  # tokenizer training corpus
    y_train = [x.label for x in train_ex]
    y_test = [x.label for x in test_ex]

    suite = _tokenizer_suite(args.vocab_size, args.seed)
    rows: List[Dict[str, float]] = []
    families = parse_model_families(args.model_families)

    def tokenize_texts(texts: List[str], labels: List[int], tok) -> List[Tuple[List[int], int]]:
        pairs: List[Tuple[List[int], int]] = []
        for t, y in zip(texts, labels):
            ids = tok.encode(t).ids[: args.max_len]
            pairs.append((ids, int(y)))
        return pairs

    for tok_name, builder in suite.items():
        tok = builder(corpus)

        # Token length stats (rate proxy)
        lens = [len(tok.encode(t).ids) for t in test_texts]
        avg_len = float(np.mean(lens))
        p95_len = float(np.percentile(lens, 95))
        rate = float(estimate_rate(test_texts, tok))

        # Prepare tokenized train/test pairs (ids,label)
        train_pairs = tokenize_texts(train_texts, y_train, tok)
        test_pairs = tokenize_texts(test_texts, y_test, tok)

        for fam in families:
            fam_cfg = get_family_cfg(fam)
            model = TinyEncoder(
                vocab_size=tok.get_vocab_size(),
                n_classes=2,
                dropout=0.1,
                max_len=args.max_len,
                **fam_cfg,
            )

            # Log training at token-aligned intervals.
            curve_rows: List[dict] = []

            def _on_log(rec: dict):
                curve_rows.append(dict(rec))

            model = train_compute_matched(
                model,
                train_pairs,
                pad_id=0,
                max_len=args.max_len,
                total_tokens=args.total_tokens,
                device=args.device,
                batch_size=args.batch_size,
                lr=args.lr,
                probe_only=False,
                eval_pairs=test_pairs,
                eval_every_tokens=args.eval_every_tokens,
                on_log=_on_log,
                log_path=str(outdir / f"train_{tok_name}_{fam}.jsonl"),
            )

            test_acc = float(evaluate(model, test_pairs, pad_id=0, max_len=args.max_len, device=args.device))

            # Training accuracy (subsample for speed)
            tr_sub = train_pairs[: min(2000, len(train_pairs))]
            train_acc = float(evaluate(model, tr_sub, pad_id=0, max_len=args.max_len, device=args.device))

            p95_ms = float(
                measure_p95_infer_ms(
                    model,
                    test_pairs,
                    device=args.device,
                    batch_size=min(args.batch_size, 32),
                    max_len=args.max_len,
                )
            )

            wall_s = float(curve_rows[-1]["wall_s"]) if curve_rows else float("nan")

            # Save curve
            curves_path = outdir / f"curves_{tok_name}_{fam}.csv"
            with curves_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["seen_tokens", "train_loss", "train_acc", "val_acc", "wall_s"])
                for r in curve_rows:
                    w.writerow(
                        [
                            int(r.get("seen_tokens", 0)),
                            float(r.get("train_loss", float("nan"))),
                            float(r.get("train_acc", float("nan"))),
                            float(r.get("val_acc", float("nan"))),
                            float(r.get("wall_s", float("nan"))),
                        ]
                    )

            rows.append(
                {
                    "tokenizer": tok_name,
                    "model_family": fam,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "gen_gap": float(train_acc - test_acc),
                    "avg_len": avg_len,
                    "p95_len": p95_len,
                    "rate": rate,
                    "p95_ms": p95_ms,
                    "train_wall_s": wall_s,
                }
            )

            print(
                f"{tok_name:10s} {fam:6s} test_acc={test_acc:.4f} avg_len={avg_len:.1f} p95_len={p95_len:.0f} p95_ms={p95_ms:.3f}"
            )

    # Save summary
    out_csv = outdir / "results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot if requested
    if args.plot:
        import matplotlib.pyplot as plt

        # Learning curves: val_acc vs seen_tokens
        for r in rows:
            tok_name = r["tokenizer"]
            fam = r["model_family"]
            curves_path = outdir / f"curves_{tok_name}_{fam}.csv"
            data = np.genfromtxt(curves_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if data.size == 0:
                continue
            plt.figure()
            plt.plot(data["seen_tokens"], data["val_acc"], label="val_acc")
            plt.xlabel("seen encoder tokens")
            plt.ylabel("val_acc")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "plots" / f"valacc_{tok_name}_{fam}.png")
            plt.close()

    print(f"[wrote] {out_csv}")


if __name__ == "__main__":
    main()
