r"""Tokenizer/interface metrics used by the paper.

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
from inspect import signature
import sys
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def _make_logreg(**kwargs) -> LogisticRegression:
    # Filter kwargs for sklearn version compatibility (e.g., multi_class removed).
    sig = signature(LogisticRegression)
    kwargs.pop("n_jobs", None)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return LogisticRegression(**filtered)


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
    # Progress display
    show_progress: bool = False
    progress_desc: str = "distortion"


def estimate_prefix_oracle_accuracy(
    raw_texts: Sequence[str],
    labels: Sequence[int],
    *,
    encode_prefix: Callable[[str], List[int]],
    cfg: DistortionCfg,
) -> float:
    """Estimate an interface-only prefix predictability score.

    We measure how well a deterministic prefix-emitting interface preserves
    task-relevant distinctions *without* learning a parametric model:
    for each (char prefix length t, token budget k), we build a lookup table
    from observed token prefixes to the majority label in the training split,
    and evaluate it on a held-out split.

    This serves as a simple, training-free proxy for prefix semantic collapse:
    if many different semantic states collapse to the same token prefix, the
    majority-label oracle will approach chance.
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(raw_texts)
    if n == 0:
        return 0.0
    m = min(cfg.sample_size, n)
    idx = rng.choice(n, size=m, replace=False)
    texts = [raw_texts[int(i)] for i in idx]
    y = np.asarray([labels[int(i)] for i in idx], dtype=np.int64)

    # deterministic split (aligned with estimate_surrogate_distortion)
    split = max(1, int(0.8 * m))
    tr_texts, va_texts = texts[:split], texts[split:]
    y_tr, y_va = y[:split], y[split:]
    if len(va_texts) == 0:
        va_texts, y_va = tr_texts, y_tr

    return estimate_prefix_oracle_accuracy_split(
        tr_texts,
        y_tr.tolist(),
        va_texts,
        y_va.tolist(),
        encode_prefix=encode_prefix,
        cfg=cfg,
    )


def estimate_prefix_oracle_accuracy_split(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    eval_texts: Sequence[str],
    eval_labels: Sequence[int],
    *,
    encode_prefix: Callable[[str], List[int]],
    cfg: DistortionCfg,
) -> float:
    """Oracle prefix accuracy using an explicit train/eval split.

    This variant is useful for experiments where we want an interface-only
    predictability score on a true held-out set (e.g., test set or drifted
    variants), while keeping the computation lightweight via sampling.
    """
    rng = np.random.default_rng(cfg.seed)

    n_tr = len(train_texts)
    n_ev = len(eval_texts)
    if n_tr == 0 or n_ev == 0:
        return 0.0

    m_tr = min(cfg.sample_size, n_tr)
    m_ev = min(cfg.sample_size, n_ev)
    tr_idx = rng.choice(n_tr, size=m_tr, replace=False)
    ev_idx = rng.choice(n_ev, size=m_ev, replace=False)

    tr_texts = [train_texts[int(i)] for i in tr_idx]
    ev_texts = [eval_texts[int(i)] for i in ev_idx]
    y_tr = np.asarray([train_labels[int(i)] for i in tr_idx], dtype=np.int64)
    y_ev = np.asarray([eval_labels[int(i)] for i in ev_idx], dtype=np.int64)

    # Fallback label when a prefix key is unseen in training.
    try:
        global_major = int(np.bincount(y_tr).argmax())
    except Exception:
        global_major = int(y_tr[0]) if len(y_tr) else 0

    accs: List[float] = []
    ts_iter = cfg.char_prefix_ts
    if cfg.show_progress and sys.stderr.isatty():
        ts_iter = tqdm(
            ts_iter,
            desc="oracle_acc",
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )
    for t in ts_iter:
        tr_tok = [encode_prefix(s[:t]) for s in tr_texts]
        ev_tok = [encode_prefix(s[:t]) for s in ev_texts]

        for k in cfg.token_prefix_ks:
            # key -> {label -> count}
            counts: dict[tuple[int, ...], dict[int, int]] = {}
            for ids, yy in zip(tr_tok, y_tr):
                key = tuple(ids[:k])
                d = counts.get(key)
                if d is None:
                    d = {}
                    counts[key] = d
                lab = int(yy)
                d[lab] = d.get(lab, 0) + 1

            correct = 0
            denom = max(1, len(y_ev))
            for ids, yy in zip(ev_tok, y_ev):
                key = tuple(ids[:k])
                d = counts.get(key)
                if not d:
                    pred = global_major
                else:
                    pred = max(d.items(), key=lambda kv: kv[1])[0]
                correct += int(pred == int(yy))
            accs.append(float(correct) / float(denom))

    return float(np.mean(accs)) if accs else 0.0


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

    ts_iter = cfg.char_prefix_ts
    if cfg.show_progress and sys.stderr.isatty():
        ts_iter = tqdm(
            ts_iter,
            desc=cfg.progress_desc,
            leave=False,
            dynamic_ncols=True,
            disable=False,
        )
    for t in ts_iter:
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
        teacher = _make_logreg(
            max_iter=1000,
            tol=1e-3,
            n_jobs=1,
            random_state=int(cfg.seed),
            multi_class="auto",
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

        student = _make_logreg(
            max_iter=1000,
            tol=1e-3,
            n_jobs=1,
            random_state=int(cfg.seed) + 13,
            multi_class="auto",
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
