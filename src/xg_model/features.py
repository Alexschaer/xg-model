"""Turn the raw shots table into model features."""

import pandas as pd

from xg_model.geometry import distance_to_goal, goal_angle


def add_geometry(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the table with distance and angle to goal."""
    return table.assign(
        distance=distance_to_goal(table["x"], table["y"]),
        angle=goal_angle(table["x"], table["y"]),
    )
