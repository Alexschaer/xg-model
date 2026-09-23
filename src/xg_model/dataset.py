"""Load stored match records and turn them into a table of shots."""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from xg_model.freeze_frame import freeze_frame_columns
from xg_model.pipeline import DATA_DIR

Record = dict[str, Any]


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
        "is_goal": details["outcome"]["name"] == "Goal",
        "statsbomb_xg": details["statsbomb_xg"],
    }


def build_shots_table(records: list[Record]) -> pd.DataFrame:
    """Return a table with one row per shot."""
    rows = [shot_row(record, shot) for record in records for shot in record["shots"]]
    return pd.DataFrame(rows)
