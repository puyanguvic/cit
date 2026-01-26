from dataclasses import dataclass
from typing import List, Tuple, Set
import random

from .cit_contract import apply_contract, Contract
from .runtime import build_artifact, tokenize_longest_match
from .probes import estimate_ce
from .hf_baselines import SPECIALS

@dataclass
class InductionCfg:
    vocab_size: int = 2048
    min_freq: int = 20
    max_cand_len: int = 12
    lambda_dist: float = 1.0
    seed: int = 0
    max_rounds: int = 200  # to keep build time bounded in MVP

def _collect_candidates(texts: List[str], min_freq: int, max_len: int) -> List[str]:
    # naive contiguous char spans (MVP). You can replace with boundary-respecting spans later.
    from collections import Counter
    cnt = Counter()
    for s in texts:
        n = len(s)
        for i in range(n):
            for l in range(2, min(max_len, n-i)+1):
                cnt[s[i:i+l]] += 1
    return [c for c,f in cnt.items() if f >= min_freq]

def train_cit(
    raw_texts: List[str],
    y: List[int],
    contract: Contract,
    cfg: InductionCfg,
    val_split: float = 0.2,
) -> Tuple[object, object]:
    rng = random.Random(cfg.seed)
    # contract
    texts = [apply_contract(t, contract) for t in raw_texts]

    # split
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    n_val = int(len(idx)*val_split)
    val_idx = set(idx[:n_val])
    tr_texts = [texts[i] for i in idx[n_val:]]
    tr_y = [y[i] for i in idx[n_val:]]
    va_texts = [texts[i] for i in idx[:n_val]]
    va_y = [y[i] for i in idx[:n_val]]

    # base vocab: specials + single chars
    charset: Set[str] = set("".join(tr_texts))
    vocab_list = SPECIALS + sorted(list(charset))
    vocab_list = vocab_list[:cfg.vocab_size]  # initial cap
    art = build_artifact(vocab_list)

    # candidates
    cands = _collect_candidates(tr_texts, cfg.min_freq, cfg.max_cand_len)
    # remove any that are already in vocab
    cands = [c for c in cands if c not in art.vocab]

    def tok_batch(txts):
        return [tokenize_longest_match(s, art) for s in txts]

    # baseline CE (distortion proxy)
    tr_ids = tok_batch(tr_texts)
    va_ids = tok_batch(va_texts)
    base_ce = estimate_ce(tr_ids, tr_y, va_ids, va_y, vocab_size=len(art.vocab), seed=cfg.seed)

    # greedy add tokens
    rounds = min(cfg.max_rounds, cfg.vocab_size - len(vocab_list))
    for _ in range(rounds):
        best = None
        best_score = -1e18

        # sample subset to keep it fast
        sample = cands if len(cands) <= 200 else rng.sample(cands, 200)

        # current rate
        cur_len = sum(len(ids) for ids in va_ids) / max(1, len(va_ids))

        for c in sample:
            # try add
            new_vocab = art.inv_vocab + [c]
            new_art = build_artifact(new_vocab)
            # rate gain (on val)
            new_va_ids = [tokenize_longest_match(s, new_art) for s in va_texts]
            new_len = sum(len(ids) for ids in new_va_ids) / max(1, len(new_va_ids))
            gain = cur_len - new_len

            # distortion increment via probe CE
            new_tr_ids = [tokenize_longest_match(s, new_art) for s in tr_texts]
            new_ce = estimate_ce(new_tr_ids, tr_y, new_va_ids, va_y, vocab_size=len(new_art.vocab), seed=cfg.seed)
            delta = new_ce - base_ce  # <=0 good

            score = gain - cfg.lambda_dist * delta
            if score > best_score:
                best_score = score
                best = (c, new_art, new_va_ids, new_ce)

        if best is None:
            break
        c, art, va_ids, base_ce = best
        if c in cands:
            cands.remove(c)
        if len(art.vocab) >= cfg.vocab_size:
            break

    return art, contract

