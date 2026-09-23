"""Features describing the match situation when a shot was taken."""

from typing import Any

Record = dict[str, Any]
Event = dict[str, Any]

PENALTY_SHOOTOUT_PERIOD = 5


def goals_in_match(record: Record) -> list[tuple[int, int]]:
    """Return (event index, scoring team id) for every goal outside a shootout."""
    shot_goals = [
        (shot["index"], shot["team"]["id"])
        for shot in record["shots"]
        if shot["shot"]["outcome"]["name"] == "Goal"
        and shot["period"] != PENALTY_SHOOTOUT_PERIOD
    ]
    own_goals = [(goal["index"], goal["team"]["id"]) for goal in record["own_goals"]]
    return shot_goals + own_goals


def game_state_columns(record: Record, shot: Event) -> dict[str, Any]:
    """Describe score and venue from the shooting team's perspective."""
    team_id = shot["team"]["id"]
    earlier_scorers = [
        scorer for index, scorer in goals_in_match(record) if index < shot["index"]
    ]
    goals_for = sum(scorer == team_id for scorer in earlier_scorers)
    goals_against = len(earlier_scorers) - goals_for
    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "score_difference": goals_for - goals_against,
        "is_home": team_id == record["match"]["home_team"]["home_team_id"],
    }
