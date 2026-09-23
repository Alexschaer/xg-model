import pandas as pd

from xg_model.modelling.split import holdout_mask

MATCH_IDS = pd.Series([match for match in range(100) for _ in range(3)])


def test_shots_of_one_match_stay_together():
    mask = holdout_mask(MATCH_IDS)

    per_match = mask.groupby(MATCH_IDS).nunique()
    assert (per_match == 1).all()


def test_share_of_test_matches():
    mask = holdout_mask(MATCH_IDS, test_share=0.2)

    assert MATCH_IDS[mask].nunique() == 20


def test_same_seed_gives_same_split():
    assert holdout_mask(MATCH_IDS, seed=1).equals(holdout_mask(MATCH_IDS, seed=1))


def test_different_seed_gives_different_split():
    assert not holdout_mask(MATCH_IDS, seed=1).equals(holdout_mask(MATCH_IDS, seed=2))


def test_split_does_not_depend_on_row_order():
    shuffled = MATCH_IDS.sample(frac=1, random_state=0)

    original_test = set(MATCH_IDS[holdout_mask(MATCH_IDS)])
    shuffled_test = set(shuffled[holdout_mask(shuffled)])

    assert original_test == shuffled_test
