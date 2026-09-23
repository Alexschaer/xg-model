import json

from xg_model import pipeline

COMPETITION = {
    "competition_id": 43,
    "season_id": 106,
    "competition_name": "FIFA World Cup",
    "season_name": "2022",
}
MATCH = {"match_id": 1}
EVENTS = [
    {"id": "a", "type": {"name": "Shot"}, "shot": {}},
    {"id": "b", "type": {"name": "Pass"}},
    {"id": "c", "type": {"name": "Own Goal For"}},
]


def fake_sources(monkeypatch, fetched_events):
    monkeypatch.setattr(pipeline, "fetch_competitions", lambda: [COMPETITION])
    monkeypatch.setattr(pipeline, "fetch_matches", lambda c, s: [MATCH])

    def fake_fetch_events(match_id):
        fetched_events.append(match_id)
        return EVENTS

    monkeypatch.setattr(pipeline, "fetch_events", fake_fetch_events)


def test_download_all_matches_stores_complete_record(monkeypatch, tmp_path):
    fake_sources(monkeypatch, [])

    pipeline.download_all_matches(tmp_path)

    record = json.loads((tmp_path / "1.json").read_text())
    assert record["competition"] == COMPETITION
    assert record["match"] == MATCH
    assert [shot["id"] for shot in record["shots"]] == ["a"]
    assert [goal["id"] for goal in record["own_goals"]] == ["c"]


def test_download_all_matches_skips_existing_matches(monkeypatch, tmp_path):
    fetched_events = []
    fake_sources(monkeypatch, fetched_events)
    (tmp_path / "1.json").write_text("{}")

    pipeline.download_all_matches(tmp_path)

    assert fetched_events == []


def test_download_all_matches_leaves_no_temporary_files(monkeypatch, tmp_path):
    fake_sources(monkeypatch, [])

    pipeline.download_all_matches(tmp_path)

    assert list(tmp_path.glob("*.tmp")) == []