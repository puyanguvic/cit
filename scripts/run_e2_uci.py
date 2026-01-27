"""E2: Drop-in utility under compute-matched training on public tabular datasets.

This script uses OpenML UCI-style datasets, serializes each row into a
structured record string, trains tokenizers (BPE/WP/Unigram/CIT), and then
trains a TinyEncoder under a fixed encoder-token budget.

Example:
  python scripts/run_e2_uci.py --dataset adult --vocab 2048 --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cit.utils.artifacts import save_json, save_vocab_json, save_run_metadata
from cit.tokenizers.metrics import DistortionCfg, estimate_surrogate_distortion, estimate_rate

from cit.utils.seed import set_seed
from cit.data.uci import load_adult, load_german_credit
from cit.data.serialize import SerializeCfg, serialize_df
from cit.tokenizers.hf_baselines import train_bpe, train_wordpiece, train_unigram
from cit.tokenizers.cit_contract import Contract, apply_contract
from cit.tokenizers.cit_induction import train_cit, InductionCfg
from cit.tokenizers.runtime import tokenize_longest_match
from cit.models.encoder import TinyEncoder
from cit.models.train import train_compute_matched
from cit.models.eval import evaluate


def hf_encode(tok, texts):
    return [tok.encode(t).ids for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "credit-g"], default="adult")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--total-tokens", type=int, default=2_000_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs/e2_uci")
    ap.add_argument("--out", type=str, default="", help="Optional CSV output path (legacy)")
    args = ap.parse_args()

    set_seed(args.seed)

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"

    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e2_uci", args=vars(args))

    ds = load_adult(args.seed) if args.dataset == "adult" else load_german_credit(args.seed)
    n_classes = len(set(ds.y_train))

    ser_cfg = SerializeCfg()
    Xtr = serialize_df(ds.X_train, ser_cfg)
    Xte = serialize_df(ds.X_test, ser_cfg)

    # apply contract inputs for CIT training (but baselines use raw serialization)
    contract = Contract()
    Xtr_c = [apply_contract(x, contract) for x in Xtr]

    # tokenizers
    bpe = train_bpe(Xtr, vocab_size=args.vocab)
    wp = train_wordpiece(Xtr, vocab_size=args.vocab)
    uni = train_unigram(Xtr, vocab_size=args.vocab)

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
    (tok_dir / "unigram").mkdir(parents=True, exist_ok=True)
    (tok_dir / "cit").mkdir(parents=True, exist_ok=True)
    bpe.save(str(tok_dir / "bpe" / "tokenizer.json"))
    wp.save(str(tok_dir / "wordpiece" / "tokenizer.json"))
    uni.save(str(tok_dir / "unigram" / "tokenizer.json"))
    save_vocab_json(cit_vocab, tok_dir / "cit" / "vocab.json")
    save_json(cit_contract, tok_dir / "cit" / "contract.json")

    encoders = [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("Unigram", lambda t: hf_encode(uni, t)),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab) for s in t]),
    ]

    dcfg = DistortionCfg(sample_size=2000, seed=args.seed)

    rows = []
    for name, enc in encoders:
        tr_ids = enc(Xtr)
        te_ids = enc(Xte)
        pad_id = 0
        train_pairs = list(zip(tr_ids, ds.y_train.tolist()))
        test_pairs = list(zip(te_ids, ds.y_test.tolist()))

        model = TinyEncoder(vocab_size=args.vocab, n_classes=n_classes, max_len=args.max_len)
        model = train_compute_matched(model, train_pairs, pad_id, args.max_len, args.total_tokens, device=args.device)
        acc = evaluate(model, test_pairs, pad_id, args.max_len, device=args.device)

        avg_len = estimate_rate(te_ids)
        p95 = sorted([len(s) for s in te_ids])[int(0.95 * len(te_ids))]

        # interface distortion (surrogate)
        if name == "CIT":
            enc_pref = lambda s: tokenize_longest_match(apply_contract(s, cit_contract), cit_vocab)
        else:
            base_tok = {"BPE": bpe, "WordPiece": wp, "Unigram": uni}[name]
            enc_pref = lambda s, _tok=base_tok: _tok.encode(s).ids
        dist = estimate_surrogate_distortion(Xtr, ds.y_train.tolist(), encode_prefix=enc_pref, vocab_size=args.vocab, cfg=dcfg)
        row = {
            "dataset": args.dataset,
            "tokenizer": name,
            "acc": acc,
            "avg_len": avg_len,
            "p95_len": p95,
            "distortion_hat": float(dist),
            "vocab": args.vocab,
            "max_len": args.max_len,
            "total_tokens": args.total_tokens,
            "seed": args.seed,
        }
        rows.append(row)
        print(f"{args.dataset}\t{name}\tacc={acc:.4f}\tavg_len={avg_len:.1f}\tp95_len={p95}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[wrote] {outp}")

    # always write a canonical results file
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"rows": rows}, outdir / "results.json")
    print(f"[wrote] {outdir / 'results.csv'}")


if __name__ == "__main__":
    main()
