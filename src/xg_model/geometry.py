"""Geometry on the StatsBomb pitch (120 x 80, attacking towards x = 120)."""

import numpy as np
from numpy.typing import ArrayLike

GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
LEFT_POST_Y = 36.0
RIGHT_POST_Y = 44.0


def distance(x1: ArrayLike, y1: ArrayLike, x2: ArrayLike, y2: ArrayLike) -> ArrayLike:
    """Return the Euclidean distance between two points."""
    return np.hypot(np.subtract(x2, x1), np.subtract(y2, y1))


def distance_to_goal(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Return the distance from a location to the centre of the goal."""
    return distance(x, y, GOAL_X, GOAL_CENTER_Y)


def goal_angle(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Return the angle in radians under which the goal mouth is seen."""
    to_left_post = np.arctan2(np.subtract(LEFT_POST_Y, y), np.subtract(GOAL_X, x))
    to_right_post = np.arctan2(np.subtract(RIGHT_POST_Y, y), np.subtract(GOAL_X, x))
    return np.abs(to_right_post - to_left_post)
