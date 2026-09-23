"""Download StatsBomb match data and store the parts needed for the model."""

import json
from pathlib import Path

from xg_model.download import fetch_competitions, fetch_events, fetch_matches
from xg_model.shots import extract_own_goals, extract_shots

DATA_DIR = Path("data/matches")


def download_all_matches(data_dir: Path = DATA_DIR) -> None:
    """Download all matches, skipping matches already stored."""
    data_dir.mkdir(parents=True, exist_ok=True)

    for competition in fetch_competitions():
        matches = fetch_matches(
            competition["competition_id"], competition["season_id"]
        )
        print(
            f"{competition['competition_name']} {competition['season_name']}: "
            f"{len(matches)} matches"
        )

        for match in matches:
            target = data_dir / f"{match['match_id']}.json"
            if target.exists():
                continue

            events = fetch_events(match["match_id"])
            record = {
                "competition": competition,
                "match": match,
                "shots": extract_shots(events),
                "own_goals": extract_own_goals(events),
            }

            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(record))
            temporary.replace(target)


if __name__ == "__main__":  # pragma: no cover
    download_all_matches()