"""Logistic regression trained with gradient descent, implemented from scratch."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from xg_model.modelling.metrics import log_loss


def sigmoid(z: ArrayLike) -> np.ndarray:
    """Map any real number to a probability between 0 and 1."""
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=float)))


class LogisticRegression:
    """Predict probabilities with a weighted sum passed through a sigmoid."""

    @classmethod
    def from_parameters(cls, weights: ArrayLike, bias: float) -> Self:
        """Create an already trained model from stored parameters."""
        model = cls()
        model.weights = np.asarray(weights, dtype=float)
        model.bias = float(bias)
        return model

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
