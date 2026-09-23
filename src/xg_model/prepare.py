"""Prepare the feature table for model training."""

import pandas as pd

from xg_model.context import PENALTY_SHOOTOUT_PERIOD

FIRST_SEASON = 2003

NUMERIC = [
    "distance",
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


def one_hot(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Encode a category column as 0/1 columns, dropping the most common one."""
    baseline = table[column].mode()[0]
    dummies = pd.get_dummies(table[column], prefix=column, dtype=int)
    return dummies.drop(columns=f"{column}_{baseline}")


def prepare(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model inputs and the goal labels."""
    table = fill_missing(filter_shots(table)).dropna(subset=NUMERIC)
    features = pd.concat(
        [
            table[NUMERIC].astype(float),
            table[FLAGS].astype(int),
            *[one_hot(table, column) for column in CATEGORICAL],
        ],
        axis=1,
    )
    return features, table["is_goal"].astype(int)
