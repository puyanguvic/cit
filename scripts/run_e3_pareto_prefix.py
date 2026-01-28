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
from pathlib import Path

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
from cit.utils.artifacts import save_json, save_vocab_json, save_run_metadata
from cit.tokenizers.metrics import DistortionCfg, estimate_surrogate_distortion, estimate_rate


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
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--total-tokens",
        type=int,
        default=5_000_000,
        help="Training budget measured in *non-pad encoder tokens* (token-fair).",
    )
    ap.add_argument(
        "--model-families",
        type=str,
        default="mini,small",
        help="Comma-separated model families (mini,small,medium).",
    )
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        help="Full fine-tuning of the encoder. Recommended for E3; probe-only on a random backbone collapses.",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="e3_pareto.csv")
    ap.add_argument("--outdir", type=str, default="runs/e3_pareto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"

    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e3_pareto", args=vars(args))
    ds = load_adult(args.seed)
    n_classes = len(set(ds.y_train))

    ser_cfg = SerializeCfg()
    Xtr = serialize_df(ds.X_train, ser_cfg)
    Xte = serialize_df(ds.X_test, ser_cfg)

    # tokenizers: keep to 2 for speed in demo
    bpe = train_bpe(Xtr, vocab_size=args.vocab)
    wp = train_wordpiece(Xtr, vocab_size=args.vocab)

    contract = Contract()
    cit_vocab, cit_contract = train_cit(
        Xtr,
        ds.y_train.tolist(),
        contract,
        InductionCfg(vocab_size=args.vocab, seed=args.seed),
        log_path=str(outdir / "tokenizers" / "cit" / "induction_log.jsonl"),
    )

    # save tokenizer artifacts
    tok_dir = outdir / "tokenizers"
    (tok_dir / "bpe").mkdir(parents=True, exist_ok=True)
    (tok_dir / "wordpiece").mkdir(parents=True, exist_ok=True)
    (tok_dir / "cit").mkdir(parents=True, exist_ok=True)
    bpe.save(str(tok_dir / "bpe" / "tokenizer.json"))
    wp.save(str(tok_dir / "wordpiece" / "tokenizer.json"))
    save_vocab_json(cit_vocab, tok_dir / "cit" / "vocab.json")
    save_json(cit_contract, tok_dir / "cit" / "contract.json")

    tokenizers = [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab) for s in t]),
    ]

    # model families
    from cit.models.family import parse_model_families, get_family_cfg

    fams = parse_model_families(args.model_families)
    backbones = [(name, get_family_cfg(name)) for name in fams]

    rows = []
    dcfg = DistortionCfg(sample_size=2000, seed=args.seed, show_progress=True)
    for tok_name, enc in tokenizers:
        tr_ids = enc(Xtr)
        te_ids = enc(Xte)
        pad_id = 0
        train_pairs = list(zip(tr_ids, ds.y_train.tolist()))
        test_pairs = list(zip(te_ids, ds.y_test.tolist()))

        avg_len = sum(len(s) for s in te_ids) / len(te_ids)
        p95 = sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]

        # interface distortion (surrogate)
        if tok_name == "CIT":
            enc_pref = lambda s: tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab)
        else:
            base_tok = {"BPE": bpe, "WordPiece": wp}[tok_name]
            enc_pref = lambda s, _tok=base_tok: _tok.encode(s).ids
        dist = estimate_surrogate_distortion(Xtr, ds.y_train.tolist(), encode_prefix=enc_pref, vocab_size=args.vocab, cfg=dcfg)

        for bb_name, bb_cfg in backbones:
            model = TinyEncoder(vocab_size=args.vocab, n_classes=n_classes, max_len=args.max_len, **bb_cfg)
            model = train_compute_matched(
                model,
                train_pairs,
                pad_id,
                args.max_len,
                args.total_tokens,
                device=args.device,
                probe_only=not args.full_finetune,
            )
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
                    "distortion_hat": float(dist),
                    "p95_latency_ms": p95_ms,
                }
            )
            print(tok_name, bb_name, "acc", acc, "p95_ms", p95_ms)

    out_csv = Path(args.out)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # canonical copy inside outdir
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows}, outdir / "results.json")


if __name__ == "__main__":
    main()
