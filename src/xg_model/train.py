"""Train the xG model and compare it with a baseline and StatsBomb's xG."""

import numpy as np
import pandas as pd

from xg_model.features import FEATURES_PATH
from xg_model.model import LogisticRegression, Standardizer, log_loss
from xg_model.prepare import prepare
from xg_model.split import test_mask


def main() -> None:  # pragma: no cover
    """Train on the training matches and evaluate on the test matches."""
    table = pd.read_parquet(FEATURES_PATH)
    features, labels = prepare(table)
    is_test = test_mask(table.loc[features.index, "match_id"]).to_numpy()
    statsbomb = table.loc[features.index, "statsbomb_xg"].to_numpy()

    x = features.to_numpy(dtype=float)
    y = labels.to_numpy()
    x_train, y_train = x[~is_test], y[~is_test]
    x_test, y_test = x[is_test], y[is_test]

    scaler = Standardizer().fit(x_train)
    model = LogisticRegression().fit(scaler.transform(x_train), y_train)
    predictions = model.predict_proba(scaler.transform(x_test))

    base_rate = y_train.mean()
    print(f"Training loss: {model.loss_history[0]:.4f} -> {model.loss_history[-1]:.4f}")
    print()
    print("Test log loss (lower is better):")
    baseline = np.full(len(y_test), base_rate)
    print(f"  Baseline, always {base_rate:.3f}: {log_loss(y_test, baseline):.4f}")
    print(f"  Own model:              {log_loss(y_test, predictions):.4f}")
    print(f"  StatsBomb xG:           {log_loss(y_test, statsbomb[is_test]):.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
