import argparse
import csv
from pathlib import Path
import sys

from tqdm import tqdm

from cit.utils.seed import set_seed
from cit.utils.artifacts import save_json, save_vocab_json, save_run_metadata
from cit.data.synthetic import SynthConfig, make_synth_dataset
from cit.data.drift import make_variants
from cit.tokenizers.hf_baselines import train_bpe, train_wordpiece, train_unigram
from cit.tokenizers.cit_contract import Contract, apply_contract
from cit.tokenizers.cit_induction import train_cit, InductionCfg
from cit.tokenizers.runtime import tokenize_longest_match
from cit.models.encoder import TinyEncoder
from cit.models.train import train_compute_matched
from cit.models.eval import evaluate
from cit.tokenizers.metrics import (
    DistortionCfg,
    estimate_prefix_oracle_accuracy_split,
    estimate_surrogate_distortion,
    estimate_rate,
)


def _token_label_purity(token_seqs, labels, min_count: int = 20, topk: int = 50):
    """Simple leakage audit: find tokens that strongly predict labels."""
    counts = {}
    for seq, y in zip(token_seqs, labels):
        for tid in seq:
            d = counts.get(tid)
            if d is None:
                d = {}
                counts[tid] = d
            d[y] = d.get(y, 0) + 1
    rows = []
    for tid, d in counts.items():
        total = sum(d.values())
        if total < min_count:
            continue
        best_label, best_cnt = max(d.items(), key=lambda kv: kv[1])
        rows.append(
            {
                "token_id": int(tid),
                "count": int(total),
                "best_label": int(best_label),
                "purity": float(best_cnt / total),
            }
        )
    rows.sort(key=lambda r: (r["purity"], r["count"]), reverse=True)
    return rows[:topk]


def hf_encode(tok, texts):
    return [tok.encode(t).ids for t in texts]


def _encode_utf8_bytes_single(s: str) -> list[int]:
    # Reserve 0 for pad_id in the encoder; keep byte ids in [1,256].
    return [b + 1 for b in s.encode("utf-8", errors="replace")]


def encode_utf8_bytes(texts: list[str]) -> list[list[int]]:
    return [_encode_utf8_bytes_single(t) for t in texts]


def _parse_cit_variants(raw: str) -> list[str]:
    out: list[str] = []
    alias = {
        "full": "full",
        "cit": "full",
        "no_typed": "no_typed",
        "no-typed": "no_typed",
        "notyped": "no_typed",
        "no_contract": "no_contract",
        "no-contract": "no_contract",
        "nocontract": "no_contract",
    }
    for part in raw.split(","):
        key = alias.get(part.strip().lower())
        if key and key not in out:
            out.append(key)
    return out


def _make_cit_prefix_encoder(vocab: dict[str, int], contract: Contract, apply_contract_text: bool):
    def _enc(s: str) -> list[int]:
        x = apply_contract(s, contract) if apply_contract_text else s
        return tokenize_longest_match(x, vocab)

    return _enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs/e1_synth")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--total-tokens", type=int, default=3_000_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-drift-samples",
        type=int,
        default=200,
        help="Number of test examples to sample for drift evaluation (each expanded into multiple variants).",
    )
    ap.add_argument(
        "--cit-variants",
        type=str,
        default="full,no_typed,no_contract",
        help="Comma-separated CIT variants: full,no_typed,no_contract",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    user_out = Path(args.outdir)
    if user_out.is_absolute():
        user_out = Path(user_out.name)
    outdir = (Path("results") / user_out) / f"seed{args.seed}"

    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e1_synth", args=vars(args))

    cfg_tr = SynthConfig(n_fields=20, signal_vocab=20, signal_pos="middle", seed=args.seed)
    cfg_te = SynthConfig(n_fields=20, signal_vocab=20, signal_pos="middle", seed=args.seed + 1)
    Xtr, ytr = make_synth_dataset(cfg_tr, 5000)
    Xte, yte = make_synth_dataset(cfg_te, 2000)
    n_classes = cfg_tr.signal_vocab

    # baselines
    bpe = train_bpe(Xtr, vocab_size=args.vocab)
    wp = train_wordpiece(Xtr, vocab_size=args.vocab)
    uni = train_unigram(Xtr, vocab_size=args.vocab)

    tok_dir = outdir / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)

    # CIT full + ablations
    cit_variants = _parse_cit_variants(args.cit_variants)
    if not cit_variants:
        raise ValueError("Empty --cit-variants list")
    cit_specs = {
        "full": {
            "name": "CIT",
            "artifact_dir": "cit",
            "contract": Contract(min_id_len=12, min_num_len=6),
            "apply_contract_text": True,
            "enforce_equals_boundary": True,
        },
        "no_typed": {
            "name": "CIT_no_typed",
            "artifact_dir": "cit_no_typed",
            "contract": Contract(min_id_len=10**9, min_num_len=10**9),
            "apply_contract_text": True,
            "enforce_equals_boundary": True,
        },
        "no_contract": {
            "name": "CIT_no_contract",
            "artifact_dir": "cit_no_contract",
            "contract": Contract(min_id_len=10**9, min_num_len=10**9),
            "apply_contract_text": False,
            "enforce_equals_boundary": False,
        },
    }
    cit_entries = []
    for variant in cit_variants:
        spec = cit_specs[variant]
        variant_dir = tok_dir / spec["artifact_dir"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        cit_vocab, cit_contract = train_cit(
            Xtr,
            ytr,
            spec["contract"],
            InductionCfg(
                vocab_size=args.vocab,
                lambda_dist=1.0,
                seed=args.seed,
                apply_contract_text=bool(spec["apply_contract_text"]),
                enforce_equals_boundary=bool(spec["enforce_equals_boundary"]),
            ),
            log_path=str(variant_dir / "induction_log.jsonl"),
        )
        save_vocab_json(cit_vocab, variant_dir / "vocab.json")
        save_json(cit_contract, variant_dir / "contract.json")
        cit_entries.append(
            {
                "name": str(spec["name"]),
                "vocab": cit_vocab,
                "contract": cit_contract,
                "apply_contract_text": bool(spec["apply_contract_text"]),
            }
        )

    # save baseline tokenizer artifacts
    (tok_dir / "bpe").mkdir(parents=True, exist_ok=True)
    (tok_dir / "wordpiece").mkdir(parents=True, exist_ok=True)
    (tok_dir / "unigram").mkdir(parents=True, exist_ok=True)
    bpe.save(str(tok_dir / "bpe" / "tokenizer.json"))
    wp.save(str(tok_dir / "wordpiece" / "tokenizer.json"))
    uni.save(str(tok_dir / "unigram" / "tokenizer.json"))

    # interface metrics (rate + surrogate distortion + oracle prefix predictability)
    #
    # Note: the synthetic signal is placed mid-record. To make the interface probes
    # meaningful, ensure the raw-prefix lengths actually reach the signal field;
    # otherwise the teacher sees no label information and the probes become noise.
    dcfg = DistortionCfg(
        sample_size=2000,
        seed=args.seed,
        show_progress=True,
        char_prefix_ts=(1024, 2048),
    )

    runs = []
    audit_dir = outdir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_specs = [
        {
            "name": "BPE",
            "encode_batch": lambda t: hf_encode(bpe, t),
            "encode_prefix": lambda s, _tok=bpe: _tok.encode(s).ids,
        },
        {
            "name": "WordPiece",
            "encode_batch": lambda t: hf_encode(wp, t),
            "encode_prefix": lambda s, _tok=wp: _tok.encode(s).ids,
        },
        {
            "name": "Unigram",
            "encode_batch": lambda t: hf_encode(uni, t),
            "encode_prefix": lambda s, _tok=uni: _tok.encode(s).ids,
        },
        {
            "name": "Bytes",
            "encode_batch": encode_utf8_bytes,
            "encode_prefix": _encode_utf8_bytes_single,
        },
    ]
    for entry in cit_entries:
        enc_pref = _make_cit_prefix_encoder(
            entry["vocab"], entry["contract"], bool(entry["apply_contract_text"])
        )
        tokenizer_specs.append(
            {
                "name": entry["name"],
                "encode_batch": lambda t, _f=enc_pref: [_f(x) for x in t],
                "encode_prefix": enc_pref,
            }
        )

    for spec in tqdm(
        tokenizer_specs,
        desc="tokenizers",
        leave=True,
        disable=not sys.stderr.isatty(),
    ):
        name = spec["name"]
        enc = spec["encode_batch"]
        enc_pref = spec["encode_prefix"]

        tr_ids = enc(Xtr)
        te_ids = enc(Xte)
        purity = _token_label_purity(tr_ids, ytr, min_count=20, topk=50)
        with (audit_dir / f"token_purity_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["token_id", "count", "best_label", "purity"])
            w.writeheader()
            w.writerows(purity)

        pad_id = 0
        max_len = int(args.max_len)
        total_tokens = int(args.total_tokens)  # match
        train_ds = list(zip(tr_ids, ytr))
        test_ds = list(zip(te_ids, yte))
        model = TinyEncoder(vocab_size=args.vocab, n_classes=n_classes)
        model = train_compute_matched(
            model,
            train_ds,
            pad_id,
            max_len,
            total_tokens,
            device=args.device,
            batch_size=int(args.batch_size),
            lr=float(args.lr),
        )
        acc = evaluate(model, test_ds, pad_id, max_len, device=args.device)

        rate = estimate_rate(te_ids)
        dist = estimate_surrogate_distortion(Xtr, ytr, encode_prefix=enc_pref, vocab_size=args.vocab, cfg=dcfg)
        oracle_acc = estimate_prefix_oracle_accuracy_split(
            Xtr,
            ytr,
            Xte,
            yte,
            encode_prefix=enc_pref,
            cfg=dcfg,
        )

        # robustness: average over variants
        drift_accs = []
        drift_texts = []
        drift_labels = []
        for i in tqdm(
            range(min(args.n_drift_samples, len(Xte))),
            desc=f"drift_{name}",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):  # sample for speed
            vs = make_variants(Xte[i], seed=i)
            drift_texts.extend(vs)
            drift_labels.extend([yte[i]] * len(vs))
            v_ids = enc(vs)
            v_ds = list(zip(v_ids, [yte[i]] * len(vs)))
            drift_accs.append(evaluate(model, v_ds, pad_id, max_len, device=args.device))
        drift_acc = sum(drift_accs) / len(drift_accs)
        oracle_drift_acc = estimate_prefix_oracle_accuracy_split(
            Xtr,
            ytr,
            drift_texts,
            drift_labels,
            encode_prefix=enc_pref,
            cfg=dcfg,
        )

        runs.append(
            {
                "tokenizer": name,
                "test_acc": float(acc),
                "drift_acc": float(drift_acc),
                "rate_E_len": float(rate),
                "distortion_hat": float(dist),
                "oracle_acc": float(oracle_acc),
                "oracle_drift_acc": float(oracle_drift_acc),
                "seed": int(args.seed),
                "vocab": int(args.vocab),
            }
        )
        print(name, "test_acc", acc, "drift_acc", drift_acc, "rate", rate, "dist", dist)

    save_json({"runs": runs}, outdir / "results.json")
    with (outdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)
    print(f"[wrote] {outdir / 'results.csv'}")


if __name__ == "__main__":
    main()
