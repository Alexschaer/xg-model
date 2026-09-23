import math

import pandas as pd
import pytest

from xg_model.geometry import distance, distance_to_goal, goal_angle


def test_distance_classic_triangle():
    assert distance(0, 0, 3, 4) == 5


def test_distance_same_point_is_zero():
    assert distance(1, 1, 1, 1) == 0


def test_distance_is_symmetric():
    assert distance(1, 2, 4, 6) == pytest.approx(distance(4, 6, 1, 2))


def test_distance_to_goal_from_penalty_spot():
    assert distance_to_goal(108, 40) == pytest.approx(12)


def test_goal_angle_from_penalty_spot():
    assert goal_angle(108, 40) == pytest.approx(2 * math.atan(4 / 12))


def test_goal_angle_is_symmetric_around_goal_centre():
    assert goal_angle(100, 30) == pytest.approx(goal_angle(100, 50))


def test_goal_angle_shrinks_with_distance():
    assert goal_angle(90, 40) < goal_angle(110, 40)


def test_goal_angle_on_goal_line_outside_posts_is_zero():
    assert goal_angle(120, 20) == pytest.approx(0)


def test_geometry_works_on_whole_columns():
    xs = pd.Series([108.0, 120.0])
    ys = pd.Series([40.0, 40.0])

    assert list(distance_to_goal(xs, ys)) == pytest.approx([12, 0])
