"""Train the xG model and evaluate it against a baseline and StatsBomb's xG."""

from pathlib import Path

import numpy as np
import pandas as pd

from xg_model.features.table import FEATURES_PATH
from xg_model.modelling.artifact import MODEL_PATH, XgModel
from xg_model.modelling.interpretation import feature_effects
from xg_model.modelling.logistic_regression import LogisticRegression
from xg_model.modelling.metrics import calibration_table, log_loss
from xg_model.modelling.plots import calibration_plot, loss_curve
from xg_model.modelling.prepare import prepare
from xg_model.modelling.scaling import Standardizer
from xg_model.modelling.split import holdout_mask

FIGURES_DIR = Path("docs/images")
ITERATIONS = 3000


def main() -> None:  # pragma: no cover
    """Train on the training matches and evaluate on the test matches."""
    table = pd.read_parquet(FEATURES_PATH)
    features, labels = prepare(table)
    is_test = holdout_mask(table.loc[features.index, "match_id"]).to_numpy()
    statsbomb = table.loc[features.index, "statsbomb_xg"].to_numpy()

    x = features.to_numpy(dtype=float)
    y = labels.to_numpy()
    x_train, y_train = x[~is_test], y[~is_test]
    x_test, y_test = x[is_test], y[is_test]

    scaler = Standardizer().fit(x_train)
    model = LogisticRegression(iterations=ITERATIONS).fit(
        scaler.transform(x_train), y_train
    )
    predictions = model.predict_proba(scaler.transform(x_test))

    history = model.loss_history
    print(f"Training loss: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"Decrease over the last 100 iterations: {history[-100] - history[-1]:.6f}")
    print()

    base_rate = y_train.mean()
    baseline = np.full(len(y_test), base_rate)
    print("Test log loss (lower is better):")
    print(f"  Baseline, always {base_rate:.3f}: {log_loss(y_test, baseline):.4f}")
    print(f"  Own model:              {log_loss(y_test, predictions):.4f}")
    print(f"  StatsBomb xG:           {log_loss(y_test, statsbomb[is_test]):.4f}")
    print()

    own_calibration = calibration_table(y_test, predictions)
    statsbomb_calibration = calibration_table(y_test, statsbomb[is_test])
    print("Calibration of own model:")
    print(own_calibration.round(3).to_string(index=False))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    loss_curve(history).savefig(FIGURES_DIR / "loss_curve.png", dpi=150)
    calibration_plot(
        {"Own model": own_calibration, "StatsBomb": statsbomb_calibration}
    ).savefig(FIGURES_DIR / "calibration.png", dpi=150)
    print(f"\nFigures saved to {FIGURES_DIR}/")
    effects = feature_effects(model, scaler, list(features.columns))
    print("\nFeature effects, most important first:")
    print(effects.round(3).to_string(index=False))
    XgModel(list(features.columns), scaler, model).save()
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
