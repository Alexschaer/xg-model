"""Logistic regression trained with gradient descent, implemented from scratch."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike


def sigmoid(z: ArrayLike) -> np.ndarray:
    """Map any real number to a probability between 0 and 1."""
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=float)))


def log_loss(labels: ArrayLike, probabilities: ArrayLike) -> float:
    """Return the average log loss of predicted probabilities."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


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


class LogisticRegression:
    """Predict probabilities with a weighted sum passed through a sigmoid."""

    def __init__(self, learning_rate: float = 0.1, iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, features: ArrayLike, labels: ArrayLike) -> Self:
        """Learn weights and bias with gradient descent on the log loss."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0
        self.loss_history: list[float] = []

        for _ in range(self.iterations):
            p = self.predict_proba(x)
            error = p - y
            self.weights -= self.learning_rate * (x.T @ error) / len(y)
            self.bias -= self.learning_rate * error.mean()
            self.loss_history.append(log_loss(y, p))

        return self

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        """Return the predicted goal probability for every shot."""
        x = np.asarray(features, dtype=float)
        return sigmoid(x @ self.weights + self.bias)
