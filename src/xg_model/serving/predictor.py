"""Turn a described shot situation into an xG value, reusing the training code."""

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from xg_model.features.freeze_frame import freeze_frame_columns
from xg_model.features.geometry import Point
from xg_model.features.table import add_geometry, key_pass_columns
from xg_model.modelling.artifact import XgModel
from xg_model.modelling.prepare import complete, encode

GOALKEEPER_ON_LINE: Point = (119.0, 40.0)
TYPICAL_ASSIST_LENGTH = 15.0


@dataclass(frozen=True)
class Situation:
    """Everything that is known about a shot at the moment it is taken."""

    x: float
    y: float
    body_part: str = "Right Foot"
    technique: str = "Normal"
    shot_type: str = "Open Play"
    play_pattern: str = "Regular Play"
    assist: str = "None"
    through_ball: bool = False
    cross: bool = False
    cut_back: bool = False
    defenders: tuple[Point, ...] = ()
    goalkeeper: Point = GOALKEEPER_ON_LINE
    under_pressure: bool = False
    first_time: bool = False
    one_on_one: bool = False
    open_goal: bool = False
    aerial_won: bool = False
    minute: int = 45
    score_difference: int = 0
    is_home: bool = True
    gender: str = "male"


def opponents(situation: Situation) -> list[dict[str, Any]]:
    """Describe defenders and goalkeeper like a StatsBomb freeze frame."""
    defenders = [
        {
            "location": list(position),
            "teammate": False,
            "position": {"name": "Center Back"},
        }
        for position in situation.defenders
    ]
    goalkeeper = {
        "location": list(situation.goalkeeper),
        "teammate": False,
        "position": {"name": "Goalkeeper"},
    }
    return [*defenders, goalkeeper]


def key_pass(situation: Situation) -> dict[str, Any] | None:
    """Describe the assist like a StatsBomb pass, or None without assist."""
    if situation.assist == "None":
        return None
    return {
        "pass": {
            "height": {"name": situation.assist},
            "length": TYPICAL_ASSIST_LENGTH,
            "cross": situation.cross,
            "through_ball": situation.through_ball,
            "cut_back": situation.cut_back,
        }
    }


def situation_row(situation: Situation) -> dict[str, Any]:
    """Build one row with the same columns as the training table."""
    location = [situation.x, situation.y]
    return {
        "x": situation.x,
        "y": situation.y,
        "body_part": situation.body_part,
        "technique": situation.technique,
        "shot_type": situation.shot_type,
        "play_pattern": situation.play_pattern,
        "gender": situation.gender,
        "minute": situation.minute,
        "score_difference": situation.score_difference,
        "rating_difference": math.nan,
        "is_home": situation.is_home,
        "under_pressure": situation.under_pressure,
        "first_time": situation.first_time,
        "one_on_one": situation.one_on_one,
        "open_goal": situation.open_goal,
        "aerial_won": situation.aerial_won,
        **key_pass_columns(key_pass(situation)),
        **freeze_frame_columns(location, opponents(situation)),
    }


def expected_goals(model: XgModel, situation: Situation) -> float:
    """Return the goal probability of a situation."""
    table = add_geometry(pd.DataFrame([situation_row(situation)]))
    return float(model.predict(encode(complete(table)))[0])
