import pytest

from xg_model.geometry import distance


def test_distance_classic_triangle():
    assert distance(0, 0, 3, 4) == 5


def test_distance_same_point_is_zero():
    assert distance(1, 1, 1, 1) == 0


def test_distance_is_symmetric():
    assert distance(1, 2, 4, 6) == pytest.approx(distance(4, 6, 1, 2))
