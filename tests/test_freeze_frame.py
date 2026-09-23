import math

import pytest

from xg_model.freeze_frame import freeze_frame_columns

PENALTY_SPOT = [108.0, 40.0]


def player(x, y, teammate=False, position="Center Back"):
    return {
        "location": [x, y],
        "teammate": teammate,
        "position": {"name": position},
    }


def test_missing_freeze_frame_is_marked():
    columns = freeze_frame_columns(PENALTY_SPOT, None)

    assert columns["has_freeze_frame"] is False
    assert math.isnan(columns["defenders_in_cone"])
    assert columns["goalkeeper_visible"] is False


def test_counts_defenders_in_cone():
    frame = [player(114, 40), player(114, 30)]

    columns = freeze_frame_columns(PENALTY_SPOT, frame)

    assert columns["defenders_in_cone"] == 1


def test_ignores_teammates():
    frame = [player(114, 40, teammate=True)]

    columns = freeze_frame_columns(PENALTY_SPOT, frame)

    assert columns["defenders_in_cone"] == 0
    assert math.isnan(columns["nearest_opponent_distance"])


def test_goalkeeper_is_not_counted_as_defender():
    frame = [player(119, 40, position="Goalkeeper")]

    columns = freeze_frame_columns(PENALTY_SPOT, frame)

    assert columns["defenders_in_cone"] == 0
    assert columns["goalkeeper_visible"] is True
    assert columns["goalkeeper_distance"] == pytest.approx(1)
    assert columns["goalkeeper_in_cone"] is True


def test_nearest_opponent_distance():
    frame = [player(110, 40), player(100, 40)]

    columns = freeze_frame_columns(PENALTY_SPOT, frame)

    assert columns["nearest_opponent_distance"] == pytest.approx(2)


def test_missing_goalkeeper():
    frame = [player(114, 40)]

    columns = freeze_frame_columns(PENALTY_SPOT, frame)

    assert columns["goalkeeper_visible"] is False
    assert math.isnan(columns["goalkeeper_distance"])
    assert columns["goalkeeper_in_cone"] is False
