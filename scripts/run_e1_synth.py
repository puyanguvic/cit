import argparse
from pathlib import Path
import json

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
from cit.tokenizers.metrics import DistortionCfg, estimate_surrogate_distortion, estimate_rate

def hf_encode(tok, texts):
    return [tok.encode(t).ids for t in texts]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs/e1_synth")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir) / f"seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(outdir, exp_name="e1_synth", args=vars(args))

    cfg = SynthConfig(n_fields=20, signal_vocab=20, signal_pos="middle", seed=0)
    Xtr, ytr = make_synth_dataset(cfg, 5000)
    Xte, yte = make_synth_dataset(cfg, 2000)
    n_classes = cfg.signal_vocab

    # baselines
    bpe = train_bpe(Xtr, vocab_size=args.vocab)
    wp  = train_wordpiece(Xtr, vocab_size=args.vocab)
    uni = train_unigram(Xtr, vocab_size=args.vocab)

    # CIT
    contract = Contract(min_id_len=12, min_num_len=6)
    cit_art, cit_contract = train_cit(
        Xtr,
        ytr,
        contract,
        InductionCfg(vocab_size=args.vocab, lambda_dist=1.0, seed=args.seed),
        log_path=str(outdir / "tokenizers" / "cit" / "induction_log.jsonl"),
    )

    # save tokenizer artifacts
    tok_dir = outdir / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "bpe").mkdir(parents=True, exist_ok=True)
    (tok_dir / "wordpiece").mkdir(parents=True, exist_ok=True)
    (tok_dir / "unigram").mkdir(parents=True, exist_ok=True)
    (tok_dir / "cit").mkdir(parents=True, exist_ok=True)
    bpe.save(str(tok_dir / "bpe" / "tokenizer.json"))
    wp.save(str(tok_dir / "wordpiece" / "tokenizer.json"))
    uni.save(str(tok_dir / "unigram" / "tokenizer.json"))
    save_vocab_json(cit_art, tok_dir / "cit" / "vocab.json")
    save_json(contract, tok_dir / "cit" / "contract.json")

    # interface metrics (rate + surrogate distortion)
    dcfg = DistortionCfg(sample_size=2000, seed=args.seed)

    runs = []
    for name, enc in [
        ("BPE", lambda t: hf_encode(bpe, t)),
        ("WordPiece", lambda t: hf_encode(wp, t)),
        ("Unigram", lambda t: hf_encode(uni, t)),
        ("CIT", lambda t: [tokenize_longest_match(apply_contract(s, cit_contract), cit_art) for s in t]),
    ]:
        tr_ids = enc(Xtr)
        te_ids = enc(Xte)
        pad_id = 0
        max_len = 128
        total_tokens = 3_000_000  # match
        train_ds = list(zip(tr_ids, ytr))
        test_ds  = list(zip(te_ids, yte))
        model = TinyEncoder(vocab_size=args.vocab, n_classes=n_classes)
        model = train_compute_matched(model, train_ds, pad_id, max_len, total_tokens, device=args.device)
        acc = evaluate(model, test_ds, pad_id, max_len, device=args.device)

        rate = estimate_rate(te_ids)
        # distortion: use the same encoder function applied to *raw prefixes*
        if name == "CIT":
            enc_pref = lambda s: tokenize_longest_match(apply_contract(s, cit_contract), cit_art)
        else:
            # tokenizers.Tokenizer encodes full string; for prefixes we just encode prefix strings.
            base_tok = {"BPE": bpe, "WordPiece": wp, "Unigram": uni}[name]
            enc_pref = lambda s, _tok=base_tok: _tok.encode(s).ids
        dist = estimate_surrogate_distortion(Xtr, ytr, encode_prefix=enc_pref, vocab_size=args.vocab, cfg=dcfg)

        # robustness: average over variants
        drift_accs = []
        for i in range(200):  # sample for speed
            vs = make_variants(Xte[i], seed=i)
            v_ids = enc(vs)
            v_ds = list(zip(v_ids, [yte[i]]*len(vs)))
            drift_accs.append(evaluate(model, v_ds, pad_id, max_len, device=args.device))
        drift_acc = sum(drift_accs) / len(drift_accs)
        runs.append(
            {
                "tokenizer": name,
                "test_acc": float(acc),
                "drift_acc": float(drift_acc),
                "rate_E_len": float(rate),
                "distortion_hat": float(dist),
            }
        )
        print(name, "test_acc", acc, "drift_acc", drift_acc, "rate", rate, "dist", dist)

    save_json({"runs": runs}, outdir / "results.json")

if __name__ == "__main__":
    main()

