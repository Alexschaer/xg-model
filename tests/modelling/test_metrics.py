import pytest

from xg_model.modelling.metrics import calibration_table


def test_calibration_table_compares_predicted_and_observed_rates():
    probabilities = [0.1] * 10 + [0.9] * 10
    labels = [1] + [0] * 9 + [1] * 9 + [0]

    table = calibration_table(labels, probabilities, groups=2)

    assert list(table["predicted"]) == pytest.approx([0.1, 0.9])
    assert list(table["observed"]) == pytest.approx([0.1, 0.9])
    assert list(table["shots"]) == [10, 10]


def test_calibration_table_groups_have_equal_size():
    probabilities = [i / 100 for i in range(100)]
    labels = [0] * 100

    table = calibration_table(labels, probabilities, groups=4)

    assert list(table["shots"]) == [25, 25, 25, 25]
