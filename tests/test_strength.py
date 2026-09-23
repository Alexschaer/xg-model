import pytest

from xg_model.strength import (
    INITIAL_RATING,
    actual_score,
    elo_before_matches,
    expected_score,
    is_complete_league,
    match_ratings,
)


def match(match_id, date, home, away, home_score, away_score):
    return {
        "match_id": match_id,
        "match_date": date,
        "kick_off": "15:00:00.000",
        "home_team": {"home_team_id": home},
        "away_team": {"away_team_id": away},
        "home_score": home_score,
        "away_score": away_score,
    }


def round_robin(teams):
    matches = []
    for i, home in enumerate(teams):
        for away in teams[i + 1 :]:
            matches.append(match(len(matches), "2016-01-01", home, away, 1, 0))
    return matches


def test_expected_score_is_even_for_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_favours_stronger_team():
    assert expected_score(1700, 1500) > 0.5


def test_expected_scores_of_both_teams_add_up_to_one():
    assert expected_score(1600, 1450) + expected_score(1450, 1600) == pytest.approx(1)


def test_actual_score():
    assert actual_score(2, 1) == 1.0
    assert actual_score(1, 1) == 0.5
    assert actual_score(0, 3) == 0.0


def test_first_match_starts_at_initial_rating():
    ratings = elo_before_matches([match(1, "2016-01-01", 10, 20, 2, 0)])

    assert ratings[1] == (INITIAL_RATING, INITIAL_RATING)


def test_winner_gains_what_loser_loses():
    matches = [
        match(1, "2016-01-01", 10, 20, 2, 0),
        match(2, "2016-01-08", 10, 20, 0, 0),
    ]

    home, away = elo_before_matches(matches)[2]

    assert home > INITIAL_RATING
    assert home - INITIAL_RATING == pytest.approx(INITIAL_RATING - away)


def test_matches_are_processed_in_time_order():
    later = match(2, "2016-01-08", 10, 20, 0, 0)
    earlier = match(1, "2016-01-01", 10, 20, 2, 0)

    ratings = elo_before_matches([later, earlier])

    assert ratings[1] == (INITIAL_RATING, INITIAL_RATING)
    assert ratings[2][0] > INITIAL_RATING


def test_complete_league():
    assert is_complete_league(round_robin([1, 2, 3, 4]))


def test_incomplete_league():
    assert not is_complete_league(round_robin([1, 2, 3, 4])[:3])


def test_single_final_is_not_a_league():
    assert not is_complete_league([match(1, "2016-01-01", 10, 20, 1, 0)])


def test_match_ratings_only_covers_complete_leagues():
    league = [
        {"competition": {"competition_id": 1, "season_id": 1}, "match": m}
        for m in round_robin([1, 2, 3, 4])
    ]
    final = [
        {
            "competition": {"competition_id": 2, "season_id": 1},
            "match": match(99, "2016-05-01", 10, 20, 1, 0),
        }
    ]

    ratings = match_ratings(league + final)

    assert 0 in ratings
    assert 99 not in ratings
