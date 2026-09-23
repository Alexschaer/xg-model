import pandas as pd

from xg_model.modelling.plots import calibration_plot, loss_curve


def test_loss_curve_draws_one_point_per_iteration():
    figure = loss_curve([0.7, 0.5, 0.4])

    (line,) = figure.axes[0].get_lines()
    assert len(line.get_xdata()) == 3


def test_calibration_plot_draws_diagonal_and_one_line_per_model():
    table = pd.DataFrame({"predicted": [0.1, 0.5], "observed": [0.1, 0.4]})

    figure = calibration_plot({"A": table, "B": table})

    labels = [line.get_label() for line in figure.axes[0].get_lines()]
    assert labels == ["Perfect calibration", "A", "B"]
