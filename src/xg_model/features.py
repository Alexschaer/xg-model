"""Turn the raw shots table into model features."""

import math

import pandas as pd

from xg_model.geometry import distance_to_goal, goal_angle


def add_geometry(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the table with distance and angle to goal."""
    return table.assign(
        distance=distance_to_goal(table["x"], table["y"]),
        angle=goal_angle(table["x"], table["y"]),
    )


def add_team_strength(table: pd.DataFrame, ratings: dict) -> pd.DataFrame:
    """Return a copy of the table with the Elo ratings of both teams."""
    missing = (math.nan, math.nan)
    home = table["match_id"].map(lambda match_id: ratings.get(match_id, missing)[0])
    away = table["match_id"].map(lambda match_id: ratings.get(match_id, missing)[1])

    team = home.where(table["is_home"], away)
    opponent = away.where(table["is_home"], home)
    return table.assign(
        team_rating=team,
        opponent_rating=opponent,
        rating_difference=team - opponent,
    )
