"""Figures for evaluating the model."""

import pandas as pd
from matplotlib.figure import Figure


def loss_curve(loss_history: list[float]) -> Figure:
    """Show how the training loss develops over the iterations."""
    figure = Figure(figsize=(7, 4))
    axes = figure.subplots()
    axes.plot(range(1, len(loss_history) + 1), loss_history)
    axes.set_xscale("log")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Training log loss")
    axes.set_title("Convergence of gradient descent")
    figure.tight_layout()
    return figure


def calibration_plot(tables: dict[str, pd.DataFrame]) -> Figure:
    """Compare predicted and observed goal rates for one or more models."""
    figure = Figure(figsize=(6, 6))
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for name, table in tables.items():
        axes.plot(table["predicted"], table["observed"], marker="o", label=name)
    axes.set_xlabel("Predicted goal probability")
    axes.set_ylabel("Observed goal rate")
    axes.set_title("Calibration on test matches")
    axes.legend()
    figure.tight_layout()
    return figure
