import math

import pandas as pd

from xg_model.modelling.prepare import (
    fill_missing,
    filter_shots,
    one_hot,
    prepare,
    season_start,
)

BASE = {
    "period": 1,
    "season": "2015/2016",
    "has_freeze_frame": True,
    "distance": 12.0,
    "angle": 0.6,
    "defenders_in_cone": 1.0,
    "nearest_opponent_distance": 2.0,
    "goalkeeper_distance": 1.0,
    "minute": 10,
    "score_difference": 0,
    "rating_difference": 0.0,
    "assist_length": 10.0,
    "under_pressure": False,
    "first_time": False,
    "one_on_one": False,
    "open_goal": False,
    "aerial_won": False,
    "assisted": True,
    "assist_cross": False,
    "assist_through_ball": False,
    "assist_cut_back": False,
    "goalkeeper_visible": True,
    "goalkeeper_in_cone": True,
    "is_home": True,
    "body_part": "Right Foot",
    "shot_type": "Open Play",
    "technique": "Normal",
    "play_pattern": "Regular Play",
    "assist_height": "Ground Pass",
    "gender": "male",
    "is_goal": False,
}


def table(*changes):
    return pd.DataFrame([{**BASE, **change} for change in changes])


def test_season_start():
    assert season_start("2015/2016") == 2015
    assert season_start("2022") == 2022


def test_filter_shots_removes_unwanted_shots():
    shots = table(
        {},
        {"period": 5},
        {"shot_type": "Penalty"},
        {"season": "1990"},
        {"has_freeze_frame": False},
    )

    assert len(filter_shots(shots)) == 1


def test_fill_missing_rating_becomes_zero_and_is_flagged():
    result = fill_missing(table({"rating_difference": math.nan}))

    assert result["rating_difference"].iloc[0] == 0.0
    assert not result["has_rating"].iloc[0]


def test_fill_missing_keeps_known_rating():
    result = fill_missing(table({"rating_difference": 80.0}))

    assert result["rating_difference"].iloc[0] == 80.0
    assert result["has_rating"].iloc[0]


def test_one_hot_drops_most_common_category():
    shots = table(
        {"body_part": "Right Foot"},
        {"body_part": "Right Foot"},
        {"body_part": "Head"},
    )

    assert list(one_hot(shots, "body_part").columns) == ["body_part_Head"]


def test_prepare_returns_only_numbers_without_gaps():
    features, labels = prepare(table({}, {"is_goal": True, "body_part": "Head"}))

    assert features.notna().all().all()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)
    assert list(labels) == [0, 1]


def test_prepare_drops_rows_with_remaining_gaps():
    features, labels = prepare(table({}, {"nearest_opponent_distance": math.nan}))

    assert len(features) == len(labels) == 1
