import pandas as pd
import pytest

from xg_model.features import add_geometry


def test_add_geometry_adds_distance_and_angle():
    table = pd.DataFrame({"x": [108.0], "y": [40.0]})

    result = add_geometry(table)

    assert result["distance"].iloc[0] == pytest.approx(12)
    assert result["angle"].iloc[0] > 0


def test_add_geometry_does_not_modify_input():
    table = pd.DataFrame({"x": [108.0], "y": [40.0]})

    add_geometry(table)

    assert list(table.columns) == ["x", "y"]
