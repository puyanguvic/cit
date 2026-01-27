"""Lightweight probes used as *interface diagnostics*.

These probes are intentionally simple (bag-of-tokens logistic regression) so that:
  (i) they are cheap to train during induction,
 (ii) they act as a tokenizer/interface diagnostic rather than a strong model,
(iii) they remain deterministic and easy to reproduce.

We provide both full-sequence and prefix-aligned estimates (avg over prefix lengths)
to better match the paper's prefix formulation.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def _truncate(ids: Sequence[int], k: int | None) -> Sequence[int]:
    return ids if k is None else ids[:k]


def featurize_token_ids(token_ids: Sequence[Sequence[int]], vocab_size: int) -> np.ndarray:
    """Simple bag-of-tokens feature map."""
    X = np.zeros((len(token_ids), vocab_size), dtype=np.float32)
    for i, ids in enumerate(token_ids):
        for t in ids:
            if 0 <= t < vocab_size:
                X[i, t] += 1.0
    return X


def estimate_ce(
    token_ids_train: Sequence[Sequence[int]],
    y_train: Sequence[int],
    token_ids_val: Sequence[Sequence[int]],
    y_val: Sequence[int],
    vocab_size: int,
    seed: int = 0,
) -> float:
    """Teacher-aligned cross-entropy proxy using a simple probe."""
    Xtr = featurize_token_ids(token_ids_train, vocab_size)
    Xva = featurize_token_ids(token_ids_val, vocab_size)
    clf = LogisticRegression(
        max_iter=200,
        random_state=seed,
    )
    clf.fit(Xtr, y_train)
    p = clf.predict_proba(Xva)
    return float(log_loss(y_val, p))


def estimate_prefix_ce(
    token_ids_train: Sequence[Sequence[int]],
    y_train: Sequence[int],
    token_ids_val: Sequence[Sequence[int]],
    y_val: Sequence[int],
    vocab_size: int,
    prefix_ks: Iterable[int] = (16, 32, 64, 128),
    seed: int = 0,
) -> float:
    """Prefix-aligned CE proxy.

    Returns the average CE over a small set of token-prefix budgets.
    This is a practical surrogate for the paper's prefix-aligned distortion.
    """
    ces: List[float] = []
    for k in prefix_ks:
        tr = [_truncate(ids, k) for ids in token_ids_train]
        va = [_truncate(ids, k) for ids in token_ids_val]
        ces.append(estimate_ce(tr, y_train, va, y_val, vocab_size=vocab_size, seed=seed))
    return float(np.mean(ces))
