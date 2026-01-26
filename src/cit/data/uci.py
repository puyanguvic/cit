from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import openml
except Exception as e:  # pragma: no cover
    openml = None


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

    ds = openml.datasets.get_dataset(name)
    X, y, categorical_indicator, attribute_names = ds.get_data(
        dataset_format="dataframe", target=ds.default_target_attribute
    )

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
