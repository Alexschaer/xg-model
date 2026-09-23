"""Measures of how good predicted probabilities are."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def log_loss(labels: ArrayLike, probabilities: ArrayLike) -> float:
    """Return the average log loss of predicted probabilities."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(
    labels: ArrayLike, probabilities: ArrayLike, groups: int = 10
) -> pd.DataFrame:
    """Compare predicted and observed goal rates in groups of equal size."""
    data = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=float),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    data["group"] = pd.qcut(data["probability"], groups, duplicates="drop")
    return (
        data.groupby("group", observed=True)
        .agg(
            predicted=("probability", "mean"),
            observed=("label", "mean"),
            shots=("label", "size"),
        )
        .reset_index(drop=True)
    )
