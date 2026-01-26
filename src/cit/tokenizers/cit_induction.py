from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from .cit_contract import Contract, apply_contract, extract_contract_markers
from .runtime import tokenize_longest_match


@dataclass
class InductionCfg:
    vocab_size: int = 2048
    min_freq: int = 5
    max_token_len: int = 16
    max_texts_for_candidates: int = 20000
    max_total_candidates: int = 80000
    score_sample_size: int = 4000
    # Prefix budgets used by the distortion probe (token-prefix lengths)
    prefix_ks: Tuple[int, ...] = (16, 32, 64, 128)
    lambda_dist: float = 1.0
    seed: int = 0


def _collect_candidates(texts: List[str], cfg: InductionCfg) -> Dict[str, int]:
    """Collect substring candidates from plain segments.

    We treat '<...>' atoms and '=' as hard boundaries and never propose candidates
    that cross them.
    """
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(min(len(texts), cfg.max_texts_for_candidates))
    if len(texts) > len(idx):
        idx = rng.choice(len(texts), size=len(idx), replace=False)

    counts: Dict[str, int] = {}
    for i in idx:
        t = texts[int(i)]
        # whitespace boundaries
        for chunk in t.split():
            # hard boundary atoms are not broken
            if chunk.startswith("<") and chunk.endswith(">"):
                counts[chunk] = counts.get(chunk, 0) + 1
                continue
            # do not cross '='; split into subchunks
            for sub in chunk.split("="):
                if not sub:
                    continue
                L = len(sub)
                for a in range(L):
                    for b in range(a + 1, min(L, a + cfg.max_token_len) + 1):
                        s = sub[a:b]
                        counts[s] = counts.get(s, 0) + 1
        # also keep separators as tokens if present
        if "=" in t:
            counts["="] = counts.get("=", 0) + t.count("=")

        if len(counts) >= cfg.max_total_candidates:
            break

    # filter by min_freq
    counts = {k: v for k, v in counts.items() if v >= cfg.min_freq}
    return counts


def _init_vocab(texts: List[str], cfg: InductionCfg) -> Dict[str, int]:
    vocab: Dict[str, int] = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }
    # contract markers (typed symbols, record markers)
    markers = set()
    for t in texts[: min(len(texts), cfg.max_texts_for_candidates)]:
        markers |= extract_contract_markers(t)
    for m in sorted(markers):
        if m not in vocab:
            vocab[m] = len(vocab)
    # structural tokens
    for tok in ["=", "<SEP>", "<REC>", "<END>"]:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def _estimate_distortion(
    texts: List[str],
    labels: np.ndarray,
    vocab: Dict[str, int],
    sample_size: int,
    prefix_ks: Tuple[int, ...],
    seed: int,
) -> float:
    """Lightweight prefix-aligned interface distortion proxy.

    We approximate directed semantic distortion using a cheap probe that predicts
    labels from *tokenized prefixes*. Concretely, we average probe cross-entropy
    over a small set of token-prefix budgets (prefix_ks).

    Note: this is an interface diagnostic, not a deployed model.
    """
    from .probes import estimate_prefix_ce

    rng = np.random.default_rng(seed)
    n = len(texts)
    m = min(sample_size, n)
    idx = rng.choice(n, size=m, replace=False)

    token_ids = [tokenize_longest_match(texts[int(i)], vocab) for i in idx]
    y = labels[idx].tolist()

    # simple deterministic split
    split = int(0.8 * len(token_ids))
    tr_ids, va_ids = token_ids[:split], token_ids[split:]
    y_tr, y_va = y[:split], y[split:]
    if len(va_ids) == 0:
        va_ids, y_va = tr_ids, y_tr

    return estimate_prefix_ce(
        tr_ids,
        y_tr,
        va_ids,
        y_va,
        vocab_size=max(vocab.values()) + 1,
        prefix_ks=prefix_ks,
        seed=seed,
    )


def train_cit(raw_texts: List[str], labels: List[int], contract: Contract, cfg: InductionCfg) -> Tuple[Dict[str, int], Contract]:
    """Train CIT vocabulary with greedy gain--distortion selection.

    This is a compact reference implementation suitable for experiments.
    """
    rng = np.random.default_rng(cfg.seed)
    y = np.asarray(labels, dtype=np.int64)

    # apply contract
    texts = [apply_contract(t, contract) for t in raw_texts]

    vocab = _init_vocab(texts, cfg)
    candidates = _collect_candidates(texts, cfg)

    # current distortion estimate
    base_dist = _estimate_distortion(texts, y, vocab, cfg.score_sample_size, cfg.prefix_ks, cfg.seed)

    # greedily grow
    remaining = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)

    pbar = tqdm(total=cfg.vocab_size - len(vocab), desc="train_cit", leave=False)
    while len(vocab) < cfg.vocab_size and remaining:
        # sample a small batch of candidates to score (speed)
        batch = remaining[: 512]
        remaining = remaining[512:]

        best = None
        best_score = -1e9
        best_dist = None

        for tok, freq in batch:
            if tok in vocab:
                continue
            # gain proxy: freq weighted by length saved
            gain = freq * max(1, len(tok) - 1)
            # estimate new distortion on-the-fly using a small sample
            tmp_vocab = vocab.copy()
            tmp_vocab[tok] = len(tmp_vocab)
            dist = _estimate_distortion(
                texts,
                y,
                tmp_vocab,
                max(512, cfg.score_sample_size // 4),
                cfg.prefix_ks,
                int(rng.integers(1e9)),
            )
            delta = dist - base_dist
            score = gain - cfg.lambda_dist * delta
            if score > best_score:
                best_score = score
                best = tok
                best_dist = dist

        if best is None:
            continue

        vocab[best] = len(vocab)
        base_dist = float(best_dist)
        pbar.update(1)

    pbar.close()
    return vocab, contract
