from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from datasets import load_dataset


@dataclass
class PhishEmailExample:
    text: str
    label: int  # 1 = phishing, 0 = safe


def load_hf_phishing_email_dataset(
    *,
    seed: int = 0,
    max_samples: Optional[int] = None,
) -> Tuple[List[PhishEmailExample], List[PhishEmailExample]]:
    """Load a small public phishing email dataset from Hugging Face.

    Dataset: zefang-liu/phishing-email-dataset (single split 'train').
    Fields: 'Email Text' and 'Email Type' (Safe Email / Phishing Email).

    Returns:
        train, test lists of examples (stratified-ish by shuffling then split).
    """
    ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    # Convert to python lists (keeps it lightweight; dataset is ~18.6k rows).
    texts = [str(x) for x in ds["Email Text"]]
    types = [str(x) for x in ds["Email Type"]]

    labels: List[int] = []
    for t in types:
        if t.strip().lower().startswith("phishing"):
            labels.append(1)
        else:
            labels.append(0)

    # Deterministic shuffle + split
    import numpy as np

    n = len(texts)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    if max_samples is not None:
        idx = idx[: int(max_samples)]

    # 80/20 split
    n2 = len(idx)
    n_train = int(0.8 * n2)
    tr_idx = idx[:n_train]
    te_idx = idx[n_train:]

    train = [PhishEmailExample(texts[i], labels[i]) for i in tr_idx]
    test = [PhishEmailExample(texts[i], labels[i]) for i in te_idx]
    return train, test
