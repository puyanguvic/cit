"""Tokenizer/interface metrics used by the paper.

This module intentionally keeps the *interface* measurements lightweight and
deterministic, so they can be run during vocabulary induction and reported in
experiments.

Key quantities in the draft:
  - Rate: expected token length R(\tau)=E[|Z|]
  - Surrogate directed semantic distortion \hat\Delta(\tau): a prefix-aligned,
    teacher--student discrepancy under the deployed prefix-emitting execution.

We estimate distortion via a cheap teacher/student probe pair:
  - Teacher: character n-gram logistic regression on raw prefixes.
  - Student: bag-of-tokens logistic regression on tokenized prefixes.
and report average KL(teacher || student) over a grid of (raw-prefix length,
token-prefix budget).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def estimate_rate(token_ids: Sequence[Sequence[int]]) -> float:
    """Expected token length."""
    if not token_ids:
        return 0.0
    return float(np.mean([len(x) for x in token_ids]))


def _safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


@dataclass(frozen=True)
class DistortionCfg:
    # Raw-character prefix lengths (t in X_{1:t}).
    char_prefix_ts: Tuple[int, ...] = (64, 128, 256)
    # Token-prefix budgets (k in first k tokens of Z^{(t)}).
    token_prefix_ks: Tuple[int, ...] = (16, 32, 64, 128)
    # Training/validation sizes for the probes.
    sample_size: int = 4000
    seed: int = 0
    # Char n-gram features (kept small for speed).
    ngram_range: Tuple[int, int] = (3, 5)
    max_features: int = 20000


def estimate_surrogate_distortion(
    raw_texts: Sequence[str],
    labels: Sequence[int],
    *,
    encode_prefix: Callable[[str], List[int]],
    vocab_size: int,
    cfg: DistortionCfg,
) -> float:
    """Estimate a deployment-aligned, prefix-averaged teacher--student KL.

    encode_prefix must implement the deployed prefix-emitting semantics:
        encode_prefix(raw_prefix_string) -> token_ids

    We compute for each raw prefix length t and token budget k:
        KL( teacher(raw[:t]) || student(trunc_k( tau(raw[:t]) )) )
    and average over (t,k) and examples.
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(raw_texts)
    if n == 0:
        return 0.0
    m = min(cfg.sample_size, n)
    idx = rng.choice(n, size=m, replace=False)
    texts = [raw_texts[int(i)] for i in idx]
    y = np.asarray([labels[int(i)] for i in idx], dtype=np.int64)

    # deterministic split
    split = max(1, int(0.8 * m))
    tr_texts, va_texts = texts[:split], texts[split:]
    y_tr, y_va = y[:split], y[split:]
    if len(va_texts) == 0:
        va_texts, y_va = tr_texts, y_tr

    kls: List[float] = []

    for t in cfg.char_prefix_ts:
        tr_pref = [s[:t] for s in tr_texts]
        va_pref = [s[:t] for s in va_texts]

        # teacher on raw prefixes
        vec = CountVectorizer(
            analyzer="char",
            ngram_range=cfg.ngram_range,
            max_features=cfg.max_features,
        )
        Xtr = vec.fit_transform(tr_pref)
        Xva = vec.transform(va_pref)
        teacher = LogisticRegression(
            max_iter=200,
            random_state=int(cfg.seed),
        )
        teacher.fit(Xtr, y_tr)
        p_teacher = teacher.predict_proba(Xva)

        # student on tokenized prefixes, evaluated under token-prefix budgets
        # We train the student once per t (full prefix), then evaluate under ks.
        tr_tok = [encode_prefix(s[:t]) for s in tr_texts]
        va_tok = [encode_prefix(s[:t]) for s in va_texts]

        # bag-of-tokens
        Xtr_s = np.zeros((len(tr_tok), vocab_size), dtype=np.float32)
        Xva_s = np.zeros((len(va_tok), vocab_size), dtype=np.float32)
        for i, ids in enumerate(tr_tok):
            for tid in ids:
                if 0 <= tid < vocab_size:
                    Xtr_s[i, tid] += 1.0
        for i, ids in enumerate(va_tok):
            for tid in ids:
                if 0 <= tid < vocab_size:
                    Xva_s[i, tid] += 1.0

        student = LogisticRegression(
            max_iter=200,
            random_state=int(cfg.seed) + 13,
        )
        student.fit(Xtr_s, y_tr)

        # Evaluate KL under token-prefix budgets by recomputing features with truncation.
        for k in cfg.token_prefix_ks:
            Xva_k = np.zeros_like(Xva_s)
            for i, ids in enumerate(va_tok):
                for tid in ids[:k]:
                    if 0 <= tid < vocab_size:
                        Xva_k[i, tid] += 1.0
            p_student = student.predict_proba(Xva_k)
            kls.append(float(np.mean(_safe_kl(p_teacher, p_student))))

    return float(np.mean(kls)) if kls else 0.0
