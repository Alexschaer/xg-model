"""Prepare the feature table for model training and prediction."""

import numpy as np
import pandas as pd

from xg_model.features.game_state import PENALTY_SHOOTOUT_PERIOD

FIRST_SEASON = 2003
MAX_OPPONENT_DISTANCE = 8.0

NUMERIC = [
    "distance",
    "log_distance",
    "angle",
    "defenders_in_cone",
    "nearest_opponent_distance",
    "goalkeeper_distance",
    "minute",
    "score_difference",
    "rating_difference",
    "assist_length",
]
FLAGS = [
    "under_pressure",
    "first_time",
    "one_on_one",
    "open_goal",
    "aerial_won",
    "assisted",
    "assist_cross",
    "assist_through_ball",
    "assist_cut_back",
    "any_defender_in_cone",
    "goalkeeper_visible",
    "goalkeeper_in_cone",
    "is_home",
    "has_rating",
]
CATEGORICAL = [
    "body_part",
    "shot_type",
    "technique",
    "play_pattern",
    "assist_height",
    "gender",
]


def season_start(season: str) -> int:
    """Return the year a season started, e.g. 2015 for '2015/2016'."""
    return int(season[:4])


def filter_shots(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only shots that the model should learn from."""
    keep = (
        (table["period"] != PENALTY_SHOOTOUT_PERIOD)
        & (table["shot_type"] != "Penalty")
        & (table["season"].map(season_start) >= FIRST_SEASON)
        & table["has_freeze_frame"]
    )
    return table[keep]


def fill_missing(table: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with neutral ones, keeping track of what was missing."""
    return table.assign(
        has_rating=table["rating_difference"].notna(),
        rating_difference=table["rating_difference"].fillna(0.0),
        goalkeeper_distance=table["goalkeeper_distance"].fillna(0.0),
    )


def add_transformations(table: pd.DataFrame) -> pd.DataFrame:
    """Add reshaped versions of features whose effect is not linear.

    Space to the nearest opponent is capped: beyond a few yards a shooter is
    unmarked, and more space barely helps. The cap also stops the linear model
    from extrapolating far beyond the distances it was trained on.
    """
    return table.assign(
        log_distance=np.log1p(table["distance"]),
        any_defender_in_cone=table["defenders_in_cone"] > 0,
        nearest_opponent_distance=table["nearest_opponent_distance"].clip(
            upper=MAX_OPPONENT_DISTANCE
        ),
    )


def complete(table: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps and add reshaped features, as needed before encoding."""
    return add_transformations(fill_missing(table))


def encode(table: pd.DataFrame) -> pd.DataFrame:
    """Turn a completed table into numbers, with a 0/1 column for every category."""
    return pd.concat(
        [
            table[NUMERIC].astype(float),
            table[FLAGS].astype(int),
            pd.get_dummies(table[CATEGORICAL], dtype=int),
        ],
        axis=1,
    )


def baseline_columns(table: pd.DataFrame) -> list[str]:
    """Return the encoded column of the most common value of every category."""
    return [f"{column}_{table[column].mode()[0]}" for column in CATEGORICAL]


def prepare(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model inputs and the goal labels for training."""
    table = complete(filter_shots(table)).dropna(subset=NUMERIC)
    features = encode(table).drop(columns=baseline_columns(table))
    return features, table["is_goal"].astype(int)
