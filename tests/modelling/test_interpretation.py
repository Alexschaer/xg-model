import numpy as np
import pytest

from xg_model.modelling.interpretation import feature_effects
from xg_model.modelling.logistic_regression import LogisticRegression
from xg_model.modelling.scaling import Standardizer


def trained(weights, std):
    model = LogisticRegression()
    model.weights = np.array(weights)
    model.bias = 0.0
    scaler = Standardizer()
    scaler.mean = np.zeros(len(std))
    scaler.std = np.array(std)
    return model, scaler


def test_weights_are_converted_back_to_original_units():
    model, scaler = trained([2.0], [2.0])

    effects = feature_effects(model, scaler, ["distance"])

    assert effects["odds_ratio"].iloc[0] == pytest.approx(np.e)


def test_feature_without_effect_keeps_reference_chance():
    model, scaler = trained([0.0], [1.0])

    effects = feature_effects(model, scaler, ["minute"])

    assert effects["chance_from_10_percent"].iloc[0] == pytest.approx(0.10)


def test_positive_weight_raises_chance():
    model, scaler = trained([1.0], [1.0])

    effects = feature_effects(model, scaler, ["through_ball"])

    assert effects["chance_from_10_percent"].iloc[0] > 0.10


def test_effects_are_sorted_by_absolute_weight():
    model, scaler = trained([0.1, -0.9, 0.5], [1.0, 1.0, 1.0])

    effects = feature_effects(model, scaler, ["a", "b", "c"])

    assert list(effects["feature"]) == ["b", "c", "a"]
