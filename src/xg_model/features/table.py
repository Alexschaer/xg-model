"""Build the shot table: one row per shot with all model features."""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from xg_model.data.ingest import DATA_DIR
from xg_model.features.freeze_frame import freeze_frame_columns
from xg_model.features.game_state import game_state_columns
from xg_model.features.geometry import distance_to_goal, goal_angle
from xg_model.features.strength import Ratings, match_ratings

Record = dict[str, Any]

FEATURES_PATH = Path("data/shots.parquet")


def load_records(data_dir: Path = DATA_DIR) -> list[Record]:
    """Read all stored match records."""
    return [json.loads(path.read_text()) for path in sorted(data_dir.glob("*.json"))]


def key_pass_columns(key_pass: dict[str, Any] | None) -> dict[str, Any]:
    """Describe the pass that led to a shot, or mark the shot as unassisted."""
    if key_pass is None:
        return {
            "assisted": False,
            "assist_height": "None",
            "assist_length": 0.0,
            "assist_cross": False,
            "assist_through_ball": False,
            "assist_cut_back": False,
        }

    details = key_pass["pass"]
    technique = details.get("technique", {}).get("name")
    is_through_ball = details.get("through_ball", False) or technique == "Through Ball"
    return {
        "assisted": True,
        "assist_height": details["height"]["name"],
        "assist_length": details["length"],
        "assist_cross": details.get("cross", False),
        "assist_through_ball": is_through_ball,
        "assist_cut_back": details.get("cut_back", False),
    }


def shot_row(record: Record, shot: dict[str, Any]) -> dict[str, Any]:
    """Flatten one shot and its match context into a single table row."""
    details = shot["shot"]
    return {
        "match_id": record["match"]["match_id"],
        "competition": record["competition"]["competition_name"],
        "season": record["competition"]["season_name"],
        "gender": record["competition"]["competition_gender"],
        "period": shot["period"],
        "minute": shot["minute"],
        "team": shot["team"]["name"],
        "player": shot["player"]["name"],
        "x": shot["location"][0],
        "y": shot["location"][1],
        "body_part": details["body_part"]["name"],
        "shot_type": details["type"]["name"],
        "technique": details["technique"]["name"],
        "play_pattern": shot["play_pattern"]["name"],
        "under_pressure": shot.get("under_pressure", False),
        "first_time": details.get("first_time", False),
        "one_on_one": details.get("one_on_one", False),
        "open_goal": details.get("open_goal", False),
        "aerial_won": details.get("aerial_won", False),
        **key_pass_columns(shot.get("key_pass")),
        **freeze_frame_columns(shot["location"], details.get("freeze_frame")),
        **game_state_columns(record, shot),
        "is_goal": details["outcome"]["name"] == "Goal",
        "statsbomb_xg": details["statsbomb_xg"],
    }


def build_shots_table(records: list[Record]) -> pd.DataFrame:
    """Return a table with one row per shot."""
    rows = [shot_row(record, shot) for record in records for shot in record["shots"]]
    return pd.DataFrame(rows)


def add_geometry(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the table with distance and angle to goal."""
    return table.assign(
        distance=distance_to_goal(table["x"], table["y"]),
        angle=goal_angle(table["x"], table["y"]),
    )


def add_team_strength(table: pd.DataFrame, ratings: Ratings) -> pd.DataFrame:
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


def build_features(records: list[Record]) -> pd.DataFrame:
    """Return the complete feature table, one row per shot."""
    table = build_shots_table(records)
    table = add_geometry(table)
    return add_team_strength(table, match_ratings(records))


if __name__ == "__main__":  # pragma: no cover
    build_features(load_records()).to_parquet(FEATURES_PATH)
