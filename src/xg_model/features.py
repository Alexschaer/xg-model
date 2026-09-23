"""Turn the raw shots table into model features."""

import math
from pathlib import Path

import pandas as pd

from xg_model.features.geometry import distance_to_goal, goal_angle
from xg_model.features.strength import match_ratings
from xg_model.features.table import build_shots_table, load_records


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


FEATURES_PATH = Path("data/shots.parquet")


def build_features(records: list[dict]) -> pd.DataFrame:
    """Return the complete feature table, one row per shot."""
    table = build_shots_table(records)
    table = add_geometry(table)
    return add_team_strength(table, match_ratings(records))


if __name__ == "__main__":  # pragma: no cover
    build_features(load_records()).to_parquet(FEATURES_PATH)
