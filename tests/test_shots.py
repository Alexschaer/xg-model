from xg_model.shots import extract_own_goals, extract_shots

PASS = {"id": "p", "type": {"name": "Pass"}}
SHOT = {"id": "s", "type": {"name": "Shot"}, "shot": {"key_pass_id": "p"}}
SHOT_WITHOUT_ASSIST = {"id": "t", "type": {"name": "Shot"}, "shot": {}}
OWN_GOAL_FOR = {"id": "o", "type": {"name": "Own Goal For"}}
OWN_GOAL_AGAINST = {"id": "x", "type": {"name": "Own Goal Against"}}


def test_extract_shots_keeps_only_shots():
    result = extract_shots([SHOT, PASS])

    assert [event["id"] for event in result] == ["s"]


def test_extract_shots_attaches_key_pass():
    result = extract_shots([PASS, SHOT])

    assert result[0]["key_pass"] == PASS


def test_extract_shots_without_assist_has_no_key_pass():
    result = extract_shots([SHOT_WITHOUT_ASSIST])

    assert result[0]["key_pass"] is None


def test_extract_shots_without_shots_returns_empty_list():
    assert extract_shots([PASS]) == []


def test_extract_shots_does_not_modify_input():
    extract_shots([SHOT, PASS])

    assert "key_pass" not in SHOT


def test_extract_own_goals_counts_each_own_goal_once():
    result = extract_own_goals([OWN_GOAL_FOR, OWN_GOAL_AGAINST, PASS])

    assert result == [OWN_GOAL_FOR]