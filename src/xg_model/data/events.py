"""Extract relevant events from StatsBomb event data."""

from typing import Any

Event = dict[str, Any]


def extract_shots(events: list[Event]) -> list[Event]:
    """Return all shot events, each with its key pass attached."""
    events_by_id = {event["id"]: event for event in events}
    return [
        {**event, "key_pass": events_by_id.get(event["shot"].get("key_pass_id"))}
        for event in events
        if event["type"]["name"] == "Shot"
    ]


def extract_own_goals(events: list[Event]) -> list[Event]:
    """Return all own goal events, credited to the benefiting team."""
    return [event for event in events if event["type"]["name"] == "Own Goal For"]
