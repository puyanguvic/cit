"""E4: Empirical interface frontier across vocab budgets.

This script sweeps vocabulary budgets for multiple tokenizers (BPE/WordPiece/CIT)
under a *drop-in* encoder constraint, and saves the rate–distortion / accuracy
trade-off curves needed for the paper.

Outputs (under results/--outdir/seedK):
  - results.csv / results.json: per-(vocab, tokenizer) metrics
  - frontier.csv: Pareto frontier points (maximize acc, minimize avg_len)
  - tokenizers/vocabXXXX/<tok>/... : tokenizer artifacts for reproducibility
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
from cit.tokenizers.metrics import DistortionCfg, estimate_surrogate_distortion


def hf_encode(tok, texts):
    return [tok.encode(t).ids for t in texts]


def _parse_int_list(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Empty --vocabs list")
    return out


def _pareto_frontier(points: list[dict], *, x_key: str, y_key: str) -> list[dict]:
    """Return non-dominated points for minimizing x and maximizing y."""
    # Sort by x asc, then y desc
    pts = sorted(points, key=lambda d: (float(d[x_key]), -float(d[y_key])))
    frontier = []
    best_y = float("-inf")
    for d in pts:
        y = float(d[y_key])
        if y > best_y:
            frontier.append(d)
            best_y = y
    return frontier


def measure_tokenize_time(encoder_fn, texts: list[str], iters: int = 3) -> float:
    """Rough per-sample tokenize time in milliseconds."""
    # run a few repeats to smooth python noise
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = encoder_fn(texts)
        times.append((time.perf_counter() - t0) * 1000.0)
    # average per-sample
    return float(sum(times) / len(times) / max(1, len(texts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabs", type=str, default="256,512,1024,2048,4096")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--total-tokens", type=int, default=2_000_000)
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        help="Full fine-tuning of the encoder (much slower). Default is probe-only.",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--outdir", type=str, default="runs/e4_frontier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true", help="If set, run plot_frontier.py after saving CSVs")
    # model backbone
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    # distortion surrogate settings
    ap.add_argument("--dist-samples", type=int, default=2000)
    args = ap.parse_args()

    vocabs = _parse_int_list(args.vocabs)
    set_seed(args.seed)

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"

    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e4_frontier", args={**vars(args), "vocabs": vocabs})

    ds = load_adult(args.seed)
    n_classes = len(set(ds.y_train))

    ser_cfg = SerializeCfg()
    Xtr = serialize_df(ds.X_train, ser_cfg)
    Xte = serialize_df(ds.X_test, ser_cfg)

    # For tokenize-time measurement: keep a small, stable subset
    t_measure = Xte[:1000]

    rows: list[dict] = []
    dcfg = DistortionCfg(sample_size=args.dist_samples, seed=args.seed)

    for vocab_size in vocabs:
        print(f"\n=== vocab_size={vocab_size} ===")
        # Train tokenizers for this budget
        bpe = train_bpe(Xtr, vocab_size=vocab_size)
        wp = train_wordpiece(Xtr, vocab_size=vocab_size)

        contract = Contract()
        cit_vocab, cit_contract = train_cit(
            Xtr,
            ds.y_train.tolist(),
            contract,
            InductionCfg(vocab_size=vocab_size, seed=args.seed),
            log_path=str(outdir / "tokenizers" / f"vocab{vocab_size:04d}" / "cit" / "induction_log.jsonl"),
        )

        # Save tokenizer artifacts grouped by budget for reproducibility
        tok_root = outdir / "tokenizers" / f"vocab{vocab_size:04d}"
        (tok_root / "bpe").mkdir(parents=True, exist_ok=True)
        (tok_root / "wordpiece").mkdir(parents=True, exist_ok=True)
        (tok_root / "cit").mkdir(parents=True, exist_ok=True)
        bpe.save(str(tok_root / "bpe" / "tokenizer.json"))
        wp.save(str(tok_root / "wordpiece" / "tokenizer.json"))
        save_vocab_json(cit_vocab, tok_root / "cit" / "vocab.json")
        save_json(cit_contract, tok_root / "cit" / "contract.json")

        tokenizers = [
            ("BPE", lambda t, _tok=bpe: hf_encode(_tok, t)),
            ("WordPiece", lambda t, _tok=wp: hf_encode(_tok, t)),
            ("CIT", lambda t, _v=cit_vocab, _c=cit_contract: [tokenize_longest_match(apply_contract(s, _c), _v) for s in t]),
        ]

        for tok_name, enc in tokenizers:
            t0_tok = time.perf_counter()
            tr_ids = enc(Xtr)
            te_ids = enc(Xte)
            tok_time_s = time.perf_counter() - t0_tok

            pad_id = 0
            train_pairs = list(zip(tr_ids, ds.y_train.tolist()))
            test_pairs = list(zip(te_ids, ds.y_test.tolist()))

            avg_len = sum(len(s) for s in te_ids) / len(te_ids)
            p95 = sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]

            # interface distortion (surrogate)
            if tok_name == "CIT":
                enc_pref = lambda s, _v=cit_vocab, _c=cit_contract: tokenize_longest_match(apply_contract(s, _c), _v)
            else:
                base_tok = {"BPE": bpe, "WordPiece": wp}[tok_name]
                enc_pref = lambda s, _tok=base_tok: _tok.encode(s).ids
            dist = estimate_surrogate_distortion(
                Xtr, ds.y_train.tolist(), encode_prefix=enc_pref, vocab_size=vocab_size, cfg=dcfg
            )

            # model train/eval
            model = TinyEncoder(
                vocab_size=vocab_size,
                n_classes=n_classes,
                max_len=args.max_len,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
            )
            t0_train = time.perf_counter()
            model = train_compute_matched(
                model,
                train_pairs,
                pad_id,
                args.max_len,
                args.total_tokens,
                device=args.device,
                probe_only=not args.full_finetune,
            )
            train_time_s = time.perf_counter() - t0_train
            acc = evaluate(model, test_pairs, pad_id, args.max_len, device=args.device)

            # tokenize-time microbench
            tok_ms = measure_tokenize_time(lambda tt: enc(tt), t_measure)

            row = {
                "vocab_size": vocab_size,
                "tokenizer": tok_name,
                "acc": float(acc),
                "avg_len": float(avg_len),
                "p95_len": int(p95),
                "distortion_hat": float(dist),
                "tokenize_ms_per_sample": float(tok_ms),
                "encode_total_s": float(tok_time_s),
                "train_time_s": float(train_time_s),
            }
            rows.append(row)
            print(f"{tok_name:9s} acc={acc:.4f} avg_len={avg_len:.1f} dist={float(dist):.4f} tok_ms={tok_ms:.4f}")

    # Save all points
    out_csv = outdir / "results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows}, outdir / "results.json")

    # Frontier (global across tokenizers/budgets): maximize acc, minimize avg_len
    frontier = _pareto_frontier(rows, x_key="avg_len", y_key="acc")
    out_f = outdir / "frontier.csv"
    with out_f.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(frontier)
    save_json({"frontier": frontier}, outdir / "frontier.json")


# Optional: auto-plot for paper figures
if args.plot:
    import subprocess, sys
    script = Path(__file__).parent / "plot_frontier.py"
    subprocess.check_call([sys.executable, str(script), "--run_dir", str(outdir)])
    print("\nSaved:", out_csv)
    print("Saved:", out_f)


if __name__ == "__main__":
    main()
