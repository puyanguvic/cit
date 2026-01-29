from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import openml
except Exception as e:  # pragma: no cover
    openml = None


_KNOWN_OPENML_DATASETS: dict[str, dict[str, int | str]] = {
    # Stable OpenML IDs for the datasets used in the paper. Keeping an explicit
    # map lets us run in offline/restricted environments by reading cached
    # parquet files without needing to resolve names via the OpenML API.
    "adult": {"id": 179, "target": "class"},
    "credit-g": {"id": 31, "target": "class"},
}


@dataclass
class UCIDataset:
    name: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


def _require_openml() -> None:
    if openml is None:
        raise ImportError(
            "openml is required for UCI loaders. Install with: pip install openml"
        )


def _openml_cache_root() -> Path:
    # OpenML honors OPENML_CACHE_DIRECTORY; otherwise it defaults to ~/.cache/openml
    return Path(os.environ.get("OPENML_CACHE_DIRECTORY", str(Path.home() / ".cache" / "openml")))


def _cached_openml_parquet(dataset_id: int) -> Path | None:
    root = _openml_cache_root()
    pq = root / "org" / "openml" / "www" / "datasets" / str(dataset_id) / f"dataset_{dataset_id}.pq"
    return pq if pq.exists() else None


def load_openml_classification(
    name: str,
    test_size: float = 0.2,
    seed: int = 0,
    max_rows: int | None = None,
) -> UCIDataset:
    """Load an OpenML dataset by name and return a stratified split.

    We use OpenML to keep datasets public and easy to reproduce.
    """
    _require_openml()

    dataset_id: int | None = None
    target_col: str | None = None
    if name in _KNOWN_OPENML_DATASETS:
        dataset_id = int(_KNOWN_OPENML_DATASETS[name]["id"])
        target_col = str(_KNOWN_OPENML_DATASETS[name]["target"])

    # Prefer reading cached parquet directly when available. This avoids
    # OpenML's get_data() path, which may attempt to write cache files and can
    # fail in restricted/sandboxed environments.
    if dataset_id is not None:
        pq = _cached_openml_parquet(dataset_id)
        if pq is not None:
            df = pd.read_parquet(pq)
            if target_col not in df.columns:
                raise ValueError(f"Cached OpenML dataset {dataset_id} missing target column: {target_col}")
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            ds = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    else:
        ds = openml.datasets.get_dataset(name)
        X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

    if max_rows is not None:
        X = X.iloc[:max_rows].copy()
        y = y.iloc[:max_rows].copy()

    # normalize label to integer classes
    y = pd.Categorical(y).codes.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return UCIDataset(
        name=name,
        X_train=X_train.reset_index(drop=True),
        y_train=y_train,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test,
    )


def load_adult(seed: int = 0) -> UCIDataset:
    # OpenML canonical name: "adult"
    return load_openml_classification("adult", seed=seed)


def load_german_credit(seed: int = 0) -> UCIDataset:
    # OpenML canonical name: "credit-g"
    return load_openml_classification("credit-g", seed=seed)
