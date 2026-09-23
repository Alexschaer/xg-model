"""Translate model weights into effects that can be read in football terms."""

import numpy as np
import pandas as pd

from xg_model.modelling.logistic_regression import LogisticRegression, sigmoid
from xg_model.modelling.scaling import Standardizer

REFERENCE_CHANCE = 0.10


def feature_effects(
    model: LogisticRegression, scaler: Standardizer, feature_names: list[str]
) -> pd.DataFrame:
    """Return the effect of every feature, sorted by importance."""
    per_unit = model.weights / scaler.std
    reference_logit = np.log(REFERENCE_CHANCE / (1 - REFERENCE_CHANCE))
    effects = pd.DataFrame(
        {
            "feature": feature_names,
            "standardized_weight": model.weights,
            "odds_ratio": np.exp(per_unit),
            "chance_from_10_percent": sigmoid(reference_logit + per_unit),
        }
    )
    order = effects["standardized_weight"].abs().sort_values(ascending=False).index
    return effects.loc[order].reset_index(drop=True)
