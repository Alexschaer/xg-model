"""Rescale features so that gradient descent treats them equally."""

from typing import Self

import numpy as np


class Standardizer:
    """Rescale features to mean 0 and standard deviation 1."""

    def fit(self, features: np.ndarray) -> Self:
        """Learn mean and standard deviation from the training data."""
        self.mean = features.mean(axis=0)
        std = features.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Rescale features with the learned mean and standard deviation."""
        return (features - self.mean) / self.std
