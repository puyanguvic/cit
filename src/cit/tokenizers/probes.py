import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def featurize_token_ids(token_ids, vocab_size: int):
    # bag-of-tokens (simple & fast)
    X = np.zeros((len(token_ids), vocab_size), dtype=np.float32)
    for i, ids in enumerate(token_ids):
        for t in ids:
            if 0 <= t < vocab_size:
                X[i, t] += 1.0
    return X

def estimate_ce(token_ids_train, y_train, token_ids_val, y_val, vocab_size: int, seed: int = 0):
    Xtr = featurize_token_ids(token_ids_train, vocab_size)
    Xva = featurize_token_ids(token_ids_val, vocab_size)
    clf = LogisticRegression(
        max_iter=200,
        n_jobs=1,
        random_state=seed,
        multi_class="auto",
    )
    clf.fit(Xtr, y_train)
    p = clf.predict_proba(Xva)
    return float(log_loss(y_val, p))

