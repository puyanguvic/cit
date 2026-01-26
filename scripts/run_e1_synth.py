from cit.utils.seed import set_seed
from cit.data.synthetic import SynthConfig, make_synth_dataset
from cit.data.drift import make_variants
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
    set_seed(0)

    cfg = SynthConfig(n_fields=20, signal_vocab=20, signal_pos="middle", seed=0)
    Xtr, ytr = make_synth_dataset(cfg, 5000)
    Xte, yte = make_synth_dataset(cfg, 2000)
    n_classes = cfg.signal_vocab

    # baselines
    bpe = train_bpe(Xtr, vocab_size=2048)
    wp  = train_wordpiece(Xtr, vocab_size=2048)
    uni = train_unigram(Xtr, vocab_size=2048)

    # CIT
    contract = Contract(min_id_len=12, min_num_len=6)
    cit_art, cit_contract = train_cit(Xtr, ytr, contract, InductionCfg(vocab_size=2048, lambda_dist=1.0, seed=0))

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
        model = TinyEncoder(vocab_size=2048, n_classes=n_classes)
        model = train_compute_matched(model, train_ds, pad_id, max_len, total_tokens, device="cuda")
        acc = evaluate(model, test_ds, pad_id, max_len, device="cuda")

        # robustness: average over variants
        drift_accs = []
        for i in range(200):  # sample for speed
            vs = make_variants(Xte[i], seed=i)
            v_ids = enc(vs)
            v_ds = list(zip(v_ids, [yte[i]]*len(vs)))
            drift_accs.append(evaluate(model, v_ds, pad_id, max_len, device="cuda"))
        runs.append((name, acc, sum(drift_accs)/len(drift_accs)))
        print(name, "test_acc", acc, "drift_acc", sum(drift_accs)/len(drift_accs))

if __name__ == "__main__":
    main()

