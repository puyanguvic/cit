"""E3: System Pareto slice via model-size scaling.

We vary encoder size (Small/Base) and evaluate accuracy vs. token length.
For a lightweight public benchmark, we use OpenML Adult.

This is intentionally minimal: it produces a CSV that you can directly copy
into the paper's Pareto table/plot.
"""

from __future__ import annotations

import argparse
import csv
import time

import torch

from cit.utils.seed import set_seed
from cit.data.uci import load_adult
from cit.data.serialize import SerializeCfg, serialize_df
from cit.tokenizers.hf_baselines import train_bpe, train_wordpiece
from cit.tokenizers.cit_contract import Contract, apply_contract
from cit.tokenizers.cit_induction import train_cit, InductionCfg
from cit.tokenizers.runtime import tokenize_longest_match
from cit.models.encoder import TinyEncoder
from cit.models.train import train_compute_matched
from cit.models.eval import evaluate


def hf_encode(tok, texts):
    return [tok.encode(t).ids for t in texts]


def measure_latency(model, sample_batch, device: str, iters: int = 50) -> float:
    """Return p95 forward latency in milliseconds."""
    x, m = sample_batch
    x = x.to(device)
    m = m.to(device)
    model = model.to(device)
    model.eval()

    # warmup
    for _ in range(10):
        _ = model(x, m)

    times = []
    if device.startswith("cuda") and torch.cuda.is_available():
        for _ in range(iters):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            _ = model(x, m)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x, m)
            times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return float(times[int(0.95 * len(times))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--total-tokens", type=int, default=2_000_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="e3_pareto.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    ds = load_adult(args.seed)
    n_classes = len(set(ds.y_train))

    ser_cfg = SerializeCfg()
    Xtr = serialize_df(ds.X_train, ser_cfg)
    Xte = serialize_df(ds.X_test, ser_cfg)

    # tokenizers: keep to 2 for speed in demo
    bpe = train_bpe(Xtr, vocab_size=args.vocab)
    wp = train_wordpiece(Xtr, vocab_size=args.vocab)

    contract = Contract()
    cit_vocab, cit_contract = train_cit(Xtr, ds.y_train.tolist(), contract, InductionCfg(vocab_size=args.vocab, seed=args.seed))

    tokenizers = [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab) for s in t]),
    ]

    # model sizes
    backbones = [
        ("Small", dict(d_model=96, n_layers=2, n_heads=3)),
        ("Base", dict(d_model=128, n_layers=4, n_heads=4)),
    ]

    rows = []
    for tok_name, enc in tokenizers:
        tr_ids = enc(Xtr)
        te_ids = enc(Xte)
        pad_id = 0
        train_pairs = list(zip(tr_ids, ds.y_train.tolist()))
        test_pairs = list(zip(te_ids, ds.y_test.tolist()))

        avg_len = sum(len(s) for s in te_ids) / len(te_ids)
        p95 = sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]

        for bb_name, bb_cfg in backbones:
            model = TinyEncoder(vocab_size=args.vocab, n_classes=n_classes, max_len=args.max_len, **bb_cfg)
            model = train_compute_matched(model, train_pairs, pad_id, args.max_len, args.total_tokens, device=args.device)
            acc = evaluate(model, test_pairs, pad_id, args.max_len, device=args.device)

            # latency on a small batch
            # build a batch of 32 samples
            batch_ids = te_ids[:32]
            L = min(args.max_len, max(len(s) for s in batch_ids))
            x = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
            m = torch.zeros((len(batch_ids), L), dtype=torch.long)
            for i, s in enumerate(batch_ids):
                s = s[:L]
                x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                m[i, : len(s)] = 1
            p95_ms = measure_latency(model, (x, m), device=args.device)

            rows.append(
                {
                    "tokenizer": tok_name,
                    "backbone": bb_name,
                    "acc": acc,
                    "avg_len": avg_len,
                    "p95_len": p95,
                    "p95_latency_ms": p95_ms,
                }
            )
            print(tok_name, bb_name, "acc", acc, "p95_ms", p95_ms)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
