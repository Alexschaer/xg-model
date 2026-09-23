"""Measures of how good predicted probabilities are."""

import numpy as np
from numpy.typing import ArrayLike


def log_loss(labels: ArrayLike, probabilities: ArrayLike) -> float:
    """Return the average log loss of predicted probabilities."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
