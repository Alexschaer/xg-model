import pandas as pd
import pytest

from xg_model.modelling.artifact import XgModel
from xg_model.modelling.logistic_regression import LogisticRegression
from xg_model.modelling.scaling import Standardizer

FEATURES = pd.DataFrame({"distance": [8.0, 20.0], "body_part_Head": [0, 1]})


def small_model():
    return XgModel(
        feature_names=["distance", "body_part_Head"],
        scaler=Standardizer.from_parameters([10.0, 0.1], [5.0, 0.3]),
        model=LogisticRegression.from_parameters([-1.0, -0.5], -2.0),
    )


def test_loaded_model_predicts_the_same_as_saved_model(tmp_path):
    model = small_model()
    path = tmp_path / "model.json"

    model.save(path)
    loaded = XgModel.load(path)

    assert loaded.predict(FEATURES) == pytest.approx(model.predict(FEATURES))


def test_prediction_does_not_depend_on_column_order():
    model = small_model()
    reordered = FEATURES[["body_part_Head", "distance"]]

    assert model.predict(reordered) == pytest.approx(model.predict(FEATURES))


def test_missing_category_column_counts_as_baseline():
    model = small_model()
    without_column = pd.DataFrame({"distance": [8.0]})
    with_zero = pd.DataFrame({"distance": [8.0], "body_part_Head": [0]})

    assert model.predict(without_column) == pytest.approx(model.predict(with_zero))


def test_missing_numeric_feature_raises_an_error():
    with pytest.raises(ValueError, match="distance"):
        small_model().predict(pd.DataFrame({"body_part_Head": [1]}))
