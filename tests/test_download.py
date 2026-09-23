import io

from xg_model import download


def test_fetch_json_builds_url_and_parses_response(monkeypatch):
    requested_urls = []

    def fake_urlopen(url):
        requested_urls.append(url)
        return io.BytesIO(b'[{"id": 1}]')

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)

    result = download.fetch_json("some/file.json")

    assert result == [{"id": 1}]
    assert requested_urls == [f"{download.BASE_URL}/some/file.json"]


def test_fetch_competitions_uses_correct_path(monkeypatch):
    requested_paths = []
    monkeypatch.setattr(download, "fetch_json", requested_paths.append)

    download.fetch_competitions()

    assert requested_paths == ["competitions.json"]


def test_fetch_matches_uses_correct_path(monkeypatch):
    requested_paths = []
    monkeypatch.setattr(download, "fetch_json", requested_paths.append)

    download.fetch_matches(43, 106)

    assert requested_paths == ["matches/43/106.json"]


def test_fetch_events_uses_correct_path(monkeypatch):
    requested_paths = []
    monkeypatch.setattr(download, "fetch_json", requested_paths.append)

    download.fetch_events(3869685)

    assert requested_paths == ["events/3869685.json"]