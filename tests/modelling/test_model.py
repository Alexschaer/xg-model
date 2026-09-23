import math

import numpy as np
import pytest

from xg_model.modelling.logistic_regression import LogisticRegression, sigmoid
from xg_model.modelling.metrics import log_loss
from xg_model.modelling.scaling import Standardizer


def test_sigmoid_of_zero_is_one_half():
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_stays_between_zero_and_one_for_extreme_values():
    result = sigmoid(np.array([-1000.0, 1000.0]))

    assert result[0] == pytest.approx(0)
    assert result[1] == pytest.approx(1)


def test_sigmoid_is_symmetric():
    assert sigmoid(2) + sigmoid(-2) == pytest.approx(1)


def test_log_loss_of_perfect_predictions_is_zero():
    assert log_loss([0, 1], [0.0, 1.0]) == pytest.approx(0, abs=1e-10)


def test_log_loss_of_coin_flip_is_log_two():
    assert log_loss([0, 1], [0.5, 0.5]) == pytest.approx(math.log(2))


def test_standardizer_centres_and_scales_training_data():
    train = np.array([[1.0, 10.0], [3.0, 30.0]])

    result = Standardizer().fit(train).transform(train)

    assert result.mean(axis=0) == pytest.approx([0, 0])
    assert result.std(axis=0) == pytest.approx([1, 1])


def test_standardizer_handles_constant_columns():
    train = np.array([[5.0], [5.0]])

    result = Standardizer().fit(train).transform(train)

    assert result == pytest.approx(np.array([[0.0], [0.0]]))


def test_standardizer_applies_training_statistics_to_new_data():
    scaler = Standardizer().fit(np.array([[0.0], [2.0]]))

    assert scaler.transform(np.array([[4.0]]))[0, 0] == pytest.approx(3)


def test_model_without_information_predicts_base_rate():
    features = np.zeros((100, 1))
    labels = np.array([1] * 25 + [0] * 75)

    model = LogisticRegression(learning_rate=0.5).fit(features, labels)

    assert model.predict_proba(features) == pytest.approx(np.full(100, 0.25), abs=1e-3)


def test_model_learns_group_rates_from_binary_feature():
    features = np.array([[-1.0]] * 50 + [[1.0]] * 50)
    labels = np.array([1] * 10 + [0] * 40 + [1] * 40 + [0] * 10)

    model = LogisticRegression(learning_rate=1.0, iterations=2000).fit(features, labels)

    low, high = model.predict_proba(np.array([[-1.0], [1.0]]))
    assert low == pytest.approx(0.2, abs=0.01)
    assert high == pytest.approx(0.8, abs=0.01)


def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    labels = (features[:, 0] + rng.normal(size=200) > 0).astype(int)

    model = LogisticRegression().fit(features, labels)

    assert model.loss_history[-1] < model.loss_history[0]
