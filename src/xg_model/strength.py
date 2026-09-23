"""Team strength as Elo ratings, computed from match results in time order."""

from collections import defaultdict
from typing import Any

Match = dict[str, Any]
Ratings = dict[int, tuple[float, float]]

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
MIN_TEAMS = 4


def expected_score(rating: float, opponent_rating: float) -> float:
    """Return the expected result (0 to 1) for a team against an opponent."""
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def actual_score(goals: int, opponent_goals: int) -> float:
    """Return 1 for a win, 0.5 for a draw and 0 for a loss."""
    if goals > opponent_goals:
        return 1.0
    if goals == opponent_goals:
        return 0.5
    return 0.0


def elo_before_matches(matches: list[Match], k_factor: float = K_FACTOR) -> Ratings:
    """Return the (home, away) ratings of every match before kick-off."""
    ratings: dict[int, float] = {}
    before: Ratings = {}

    in_time_order = sorted(
        matches, key=lambda match: (match["match_date"], match["kick_off"] or "")
    )
    for match in in_time_order:
        home = match["home_team"]["home_team_id"]
        away = match["away_team"]["away_team_id"]
        home_rating = ratings.get(home, INITIAL_RATING)
        away_rating = ratings.get(away, INITIAL_RATING)
        before[match["match_id"]] = (home_rating, away_rating)

        result = actual_score(match["home_score"], match["away_score"])
        change = k_factor * (result - expected_score(home_rating, away_rating))
        ratings[home] = home_rating + change
        ratings[away] = away_rating - change

    return before


def is_complete_league(matches: list[Match]) -> bool:
    """Return whether every team has met every other team at least once."""
    teams = {match["home_team"]["home_team_id"] for match in matches} | {
        match["away_team"]["away_team_id"] for match in matches
    }
    pairings = len(teams) * (len(teams) - 1) / 2
    return len(teams) >= MIN_TEAMS and len(matches) >= pairings


def match_ratings(records: list[dict[str, Any]]) -> Ratings:
    """Return pre-match Elo ratings for all matches of complete leagues."""
    seasons: dict[tuple[int, int], list[Match]] = defaultdict(list)
    for record in records:
        competition = record["competition"]
        key = (competition["competition_id"], competition["season_id"])
        seasons[key].append(record["match"])

    ratings: Ratings = {}
    for matches in seasons.values():
        if is_complete_league(matches):
            ratings.update(elo_before_matches(matches))
    return ratings
