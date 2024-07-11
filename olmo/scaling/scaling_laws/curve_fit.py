import csv
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy


@dataclass
class CurveFitConfig:
    path: str
    """
    Path containing the W&B downloaded data and metadata.
    """

    keys: List[str]
    """
    The metrics for computing the scaling law predictions.
    """

    dot_size: float
    """
    Plotting parameter.
    """

    title: str
    """
    Plot title.
    """

    train_step_min: Optional[int] = None
    """
    Lower bound for the training period used to compute scaling coefficients.
    """

    train_step_max: Optional[int] = None
    """
    Upper bound for the training period used to compute scaling coefficients.
    """

    eval_step_max: Optional[int] = None
    """
    Upper bound for the prediction validation period using the scaling coefficients.
    Lower bound is `train_step_max`. 
    """

    final_loss_tokens: Optional[int] = None
    """
    The step at which to make the final prediction.
    """

    outlier_threshold: Optional[float] = None
    """
    This parameter can be tuned based on the curves to discount outliers.
    """


# Power Law functions


def openai_fit(x, a, b, c):
    return (a / x + c) ** b


def chinchilla_fit(x, a, b, c):
    return a * x**b + c


def chinchilla_contaminated_fit(x, a, b, c, d):
    return (a * x**b + c) * (1 - x / d)


def get_data(config: CurveFitConfig):
    train_xs, train_ys = [], []
    eval_xs, eval_ys = [], []

    with open(config.path) as file_ref:
        reader = csv.DictReader(file_ref)
        for r, row in enumerate(reader):
            x = float(row["throughput/total_tokens"])
            y = np.mean([float(row[key]) for key in config.keys])
            batch_size = int(row["batch_size_in_tokens"])
            if config.outlier_threshold is not None and y > config.outlier_threshold:  # remove outliers
                continue
            if config.train_step_min is not None and x <= config.train_step_min * batch_size:
                continue
            if config.train_step_max is None or x <= config.train_step_max * batch_size:
                train_xs.append(x)
                train_ys.append(y)
            elif config.eval_step_max is None or x <= config.eval_step_max * batch_size:
                eval_xs.append(x)
                eval_ys.append(y)

    return train_xs, train_ys, eval_xs, eval_ys


def get_coefficients(train_xs, train_ys, fitting_func, p0):
    coeffs = scipy.optimize.curve_fit(fitting_func, train_xs, train_ys, p0=p0, maxfev=50000)[0]
    coeffs_string = ", ".join([chr(ord("a") + i) + f" = {coeffs[i]:.2f}" for i in range(len(coeffs))])
    print(f"{fitting_func.__name__}: {coeffs_string}")
    return coeffs


def plot_scaling(
    train_xs, train_ys, eval_xs, fitting_func, final_loss_tokens, p0=[1e16, 0.1, 0], predict=False, **plot_kwargs
):
    coefficients = get_coefficients(train_xs, train_ys, fitting_func, p0=p0)

    plt.plot(
        train_xs + eval_xs,
        fitting_func(np.array(train_xs + eval_xs), *coefficients),
        **plot_kwargs,
    )

    if predict:
        final_ce_loss = chinchilla_fit(final_loss_tokens, *coefficients)
        plt.plot(final_loss_tokens, final_ce_loss, "x", color=plot_kwargs.get("color", "red"))

        plt.text(
            0.2,
            0.63,
            f"Predicted CE Loss = y(x = {final_loss_tokens:.2g}) = {final_ce_loss:.2f}",
            fontsize=10,
            transform=plt.gca().transAxes,
        )

        plt.text(
            0.2,
            0.56,
            f"Predicted PPL = e^{final_ce_loss:.2f} = {np.exp(final_ce_loss):.2f}",
            fontsize=10,
            transform=plt.gca().transAxes,
        )
