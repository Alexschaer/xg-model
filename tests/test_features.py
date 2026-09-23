import math

import pandas as pd
import pytest

from xg_model.features import add_geometry, add_team_strength, build_features

SHOT = {
    "index": 5,
    "period": 1,
    "minute": 10,
    "team": {"id": 10, "name": "Home FC"},
    "player": {"name": "Striker"},
    "location": [108.0, 40.0],
    "play_pattern": {"name": "Regular Play"},
    "shot": {
        "body_part": {"name": "Right Foot"},
        "type": {"name": "Open Play"},
        "technique": {"name": "Normal"},
        "outcome": {"name": "Goal"},
        "statsbomb_xg": 0.3,
    },
}

RECORD = {
    "competition": {
        "competition_id": 1,
        "season_id": 1,
        "competition_name": "Test League",
        "season_name": "2016",
        "competition_gender": "male",
    },
    "match": {
        "match_id": 1,
        "match_date": "2016-01-01",
        "kick_off": None,
        "home_team": {"home_team_id": 10},
        "away_team": {"away_team_id": 20},
        "home_score": 1,
        "away_score": 0,
    },
    "shots": [SHOT],
    "own_goals": [],
}


def test_build_features_combines_all_steps():
    table = build_features([RECORD])

    assert len(table) == 1
    expected = {"distance", "defenders_in_cone", "score_difference", "team_rating"}
    assert expected <= set(table.columns)


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
