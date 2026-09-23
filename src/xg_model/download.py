"""Download StatsBomb open data from GitHub."""

import json
import urllib.request
from typing import Any

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def fetch_json(path: str) -> Any:
    """Download a JSON file from the StatsBomb open data repository."""
    url = f"{BASE_URL}/{path}"
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def fetch_competitions() -> list[dict[str, Any]]:
    """Return all available competitions and seasons."""
    return fetch_json("competitions.json")


def fetch_matches(competition_id: int, season_id: int) -> list[dict[str, Any]]:
    """Return all matches of one competition season."""
    return fetch_json(f"matches/{competition_id}/{season_id}.json")


def fetch_events(match_id: int) -> list[dict[str, Any]]:
    """Return all events of one match."""
    return fetch_json(f"events/{match_id}.json")
