"""Split shots into training and test sets, keeping each match together."""

import numpy as np
import pandas as pd

TEST_SHARE = 0.2
SEED = 42


def test_mask(
    match_ids: pd.Series, test_share: float = TEST_SHARE, seed: int = SEED
) -> pd.Series:
    """Return True for every shot whose match belongs to the test set."""
    matches = np.sort(match_ids.unique())
    rng = np.random.default_rng(seed)
    test_count = round(len(matches) * test_share)
    test_matches = rng.choice(matches, size=test_count, replace=False)
    return match_ids.isin(test_matches)
