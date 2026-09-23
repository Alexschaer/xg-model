from xg_model.context import game_state_columns, goals_in_match

HOME = 10
AWAY = 20


def shot(index, team, outcome="Saved", period=1):
    return {
        "index": index,
        "period": period,
        "team": {"id": team},
        "shot": {"outcome": {"name": outcome}},
    }


def own_goal(index, benefiting_team):
    return {"index": index, "team": {"id": benefiting_team}}


def record(shots, own_goals=()):
    return {
        "match": {"home_team": {"home_team_id": HOME}},
        "shots": list(shots),
        "own_goals": list(own_goals),
    }


def test_goals_in_match_includes_shot_goals_and_own_goals():
    match = record([shot(1, HOME, "Goal"), shot(2, AWAY)], [own_goal(3, AWAY)])

    assert goals_in_match(match) == [(1, HOME), (3, AWAY)]


def test_goals_in_match_ignores_penalty_shootout():
    match = record([shot(1, HOME, "Goal", period=5)])

    assert goals_in_match(match) == []


def test_score_before_first_goal_is_level():
    first = shot(1, HOME)
    match = record([first, shot(2, HOME, "Goal")])

    columns = game_state_columns(match, first)

    assert columns["score_difference"] == 0


def test_score_counts_only_earlier_goals_from_shooters_view():
    later = shot(5, AWAY)
    match = record([shot(1, HOME, "Goal"), shot(2, HOME, "Goal"), later])

    columns = game_state_columns(match, later)

    assert columns["goals_for"] == 0
    assert columns["goals_against"] == 2
    assert columns["score_difference"] == -2


def test_own_goal_counts_for_benefiting_team():
    later = shot(5, AWAY)
    match = record([later], [own_goal(3, AWAY)])

    assert game_state_columns(match, later)["goals_for"] == 1


def test_is_home():
    home_shot = shot(1, HOME)
    away_shot = shot(2, AWAY)
    match = record([home_shot, away_shot])

    assert game_state_columns(match, home_shot)["is_home"] is True
    assert game_state_columns(match, away_shot)["is_home"] is False