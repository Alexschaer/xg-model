import json

from xg_model.dataset import build_shots_table, key_pass_columns, load_records, shot_row

RECORD = {
    "competition": {
        "competition_name": "Premier League",
        "season_name": "2015/2016",
        "competition_gender": "male",
    },
    "match": {"match_id": 1},
}


def make_shot(outcome="Saved", **extra):
    shot = {
        "period": 1,
        "minute": 5,
        "team": {"name": "Chelsea"},
        "player": {"name": "Eden Hazard"},
        "location": [100.0, 40.0],
        "play_pattern": {"name": "Regular Play"},
        "shot": {
            "body_part": {"name": "Right Foot"},
            "type": {"name": "Open Play"},
            "technique": {"name": "Normal"},
            "outcome": {"name": outcome},
            "statsbomb_xg": 0.1,
        },
    }
    return {**shot, **extra}


def test_shot_row_flattens_shot_and_context():
    row = shot_row(RECORD, make_shot())

    assert row["match_id"] == 1
    assert row["gender"] == "male"
    assert row["player"] == "Eden Hazard"
    assert (row["x"], row["y"]) == (100.0, 40.0)
    assert row["body_part"] == "Right Foot"


def test_shot_row_marks_goals():
    assert shot_row(RECORD, make_shot(outcome="Goal"))["is_goal"] is True
    assert shot_row(RECORD, make_shot(outcome="Saved"))["is_goal"] is False


def test_shot_row_defaults_missing_flags_to_false():
    row = shot_row(RECORD, make_shot())

    assert row["under_pressure"] is False
    assert row["first_time"] is False


def test_shot_row_reads_flags_when_present():
    row = shot_row(RECORD, make_shot(under_pressure=True))

    assert row["under_pressure"] is True


def test_build_shots_table_has_one_row_per_shot():
    records = [
        {**RECORD, "shots": [make_shot(), make_shot()]},
        {**RECORD, "shots": [make_shot()]},
    ]

    table = build_shots_table(records)

    assert len(table) == 3


def test_load_records_reads_all_files(tmp_path):
    (tmp_path / "1.json").write_text(json.dumps({"match": {"match_id": 1}}))
    (tmp_path / "2.json").write_text(json.dumps({"match": {"match_id": 2}}))

    records = load_records(tmp_path)

    assert [record["match"]["match_id"] for record in records] == [1, 2]


def make_pass(**extra):
    details = {"height": {"name": "Ground Pass"}, "length": 15.0}
    return {"pass": {**details, **extra}}


def test_key_pass_columns_without_assist():
    columns = key_pass_columns(None)

    assert columns["assisted"] is False
    assert columns["assist_height"] == "None"


def test_key_pass_columns_describes_pass():
    columns = key_pass_columns(make_pass())

    assert columns["assisted"] is True
    assert columns["assist_height"] == "Ground Pass"
    assert columns["assist_length"] == 15.0
    assert columns["assist_cross"] is False


def test_key_pass_columns_detects_cross():
    assert key_pass_columns(make_pass(cross=True))["assist_cross"] is True


def test_key_pass_columns_detects_cut_back():
    assert key_pass_columns(make_pass(cut_back=True))["assist_cut_back"] is True


def test_key_pass_columns_detects_through_ball_as_flag():
    columns = key_pass_columns(make_pass(through_ball=True))

    assert columns["assist_through_ball"] is True


def test_key_pass_columns_detects_through_ball_as_technique():
    columns = key_pass_columns(make_pass(technique={"name": "Through Ball"}))

    assert columns["assist_through_ball"] is True


def test_shot_row_includes_shot_flags_and_assist():
    row = shot_row(RECORD, make_shot(key_pass=make_pass(cross=True)))

    assert row["one_on_one"] is False
    assert row["assisted"] is True
    assert row["assist_cross"] is True
