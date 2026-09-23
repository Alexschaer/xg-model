"""Features describing the positions of all players at the moment of a shot."""

import math
from typing import Any

from xg_model.geometry import (
    GOAL_X,
    LEFT_POST_Y,
    RIGHT_POST_Y,
    Point,
    distance,
    distance_to_goal,
    in_triangle,
)

LEFT_POST = (GOAL_X, LEFT_POST_Y)
RIGHT_POST = (GOAL_X, RIGHT_POST_Y)

Player = dict[str, Any]


def freeze_frame_columns(
    location: list[float], freeze_frame: list[Player] | None
) -> dict[str, Any]:
    """Summarise the positions of all players when the shot was taken."""
    if freeze_frame is None:
        return {
            "has_freeze_frame": False,
            "defenders_in_cone": math.nan,
            "nearest_opponent_distance": math.nan,
            "goalkeeper_visible": False,
            "goalkeeper_distance": math.nan,
            "goalkeeper_in_cone": False,
        }

    shooter: Point = (location[0], location[1])

    def in_cone(player: Player) -> bool:
        x, y = player["location"]
        return in_triangle((x, y), shooter, LEFT_POST, RIGHT_POST)

    def distance_to_shooter(player: Player) -> float:
        x, y = player["location"]
        return float(distance(shooter[0], shooter[1], x, y))

    opponents = [player for player in freeze_frame if not player["teammate"]]
    goalkeepers = [p for p in opponents if p["position"]["name"] == "Goalkeeper"]
    outfield = [p for p in opponents if p["position"]["name"] != "Goalkeeper"]
    goalkeeper = goalkeepers[0] if goalkeepers else None

    return {
        "has_freeze_frame": True,
        "defenders_in_cone": sum(in_cone(player) for player in outfield),
        "nearest_opponent_distance": min(
            (distance_to_shooter(player) for player in opponents),
            default=math.nan,
        ),
        "goalkeeper_visible": goalkeeper is not None,
        "goalkeeper_distance": (
            float(distance_to_goal(*goalkeeper["location"]))
            if goalkeeper
            else math.nan
        ),
        "goalkeeper_in_cone": in_cone(goalkeeper) if goalkeeper else False,
    }