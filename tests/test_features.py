import math

import pandas as pd
import pytest

from xg_model.features import add_geometry, add_team_strength


def test_add_geometry_adds_distance_and_angle():
    table = pd.DataFrame({"x": [108.0], "y": [40.0]})

    result = add_geometry(table)

    assert result["distance"].iloc[0] == pytest.approx(12)
    assert result["angle"].iloc[0] > 0


def test_add_geometry_does_not_modify_input():
    table = pd.DataFrame({"x": [108.0], "y": [40.0]})

    add_geometry(table)

    assert list(table.columns) == ["x", "y"]


def test_add_team_strength_uses_rating_of_shooting_team():
    table = pd.DataFrame({"match_id": [1, 1], "is_home": [True, False]})
    ratings = {1: (1600.0, 1500.0)}

    result = add_team_strength(table, ratings)

    assert list(result["team_rating"]) == [1600.0, 1500.0]
    assert list(result["rating_difference"]) == [100.0, -100.0]


def test_add_team_strength_leaves_unrated_matches_empty():
    table = pd.DataFrame({"match_id": [2], "is_home": [True]})

    result = add_team_strength(table, {})

    assert math.isnan(result["rating_difference"].iloc[0])
