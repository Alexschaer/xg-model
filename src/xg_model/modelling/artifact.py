"""A trained xG model that can be saved, loaded and used for predictions."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from xg_model.modelling.logistic_regression import LogisticRegression
from xg_model.modelling.prepare import CATEGORICAL
from xg_model.modelling.scaling import Standardizer

MODEL_PATH = Path("models/xg_model.json")
ONE_HOT_PREFIXES = tuple(f"{column}_" for column in CATEGORICAL)


@dataclass
class XgModel:
    """Everything needed to turn features into goal probabilities."""

    feature_names: list[str]
    scaler: Standardizer
    model: LogisticRegression

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return the goal probability for every row of features."""
        aligned = self._align(features).to_numpy(dtype=float)
        return self.model.predict_proba(self.scaler.transform(aligned))

    def _align(self, features: pd.DataFrame) -> pd.DataFrame:
        """Order the columns as in training; absent categories count as baseline."""
        missing = [name for name in self.feature_names if name not in features]
        required = [name for name in missing if not name.startswith(ONE_HOT_PREFIXES)]
        if required:
            raise ValueError(f"Missing features: {required}")
        return features.reindex(columns=self.feature_names, fill_value=0)

    def save(self, path: Path = MODEL_PATH) -> None:
        """Write all parameters to a JSON file."""
        parameters = {
            "feature_names": self.feature_names,
            "mean": self.scaler.mean.tolist(),
            "std": self.scaler.std.tolist(),
            "weights": self.model.weights.tolist(),
            "bias": self.model.bias,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parameters, indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> Self:
        """Read a model written by save."""
        parameters = json.loads(path.read_text())
        return cls(
            feature_names=parameters["feature_names"],
            scaler=Standardizer.from_parameters(parameters["mean"], parameters["std"]),
            model=LogisticRegression.from_parameters(
                parameters["weights"], parameters["bias"]
            ),
        )
