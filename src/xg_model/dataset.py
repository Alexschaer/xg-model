"""Load stored match records and turn them into a table of shots."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from xg_model.pipeline import DATA_DIR

Record = dict[str, Any]


def load_records(data_dir: Path = DATA_DIR) -> list[Record]:
    """Read all stored match records."""
    return [json.loads(path.read_text()) for path in sorted(data_dir.glob("*.json"))]


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
        "is_goal": details["outcome"]["name"] == "Goal",
        "statsbomb_xg": details["statsbomb_xg"],
    }


def build_shots_table(records: list[Record]) -> pd.DataFrame:
    """Return a table with one row per shot."""
    rows = [shot_row(record, shot) for record in records for shot in record["shots"]]
    return pd.DataFrame(rows)
