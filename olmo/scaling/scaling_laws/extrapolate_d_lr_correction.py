import csv

import matplotlib.pyplot as plt
import numpy as np

from .extrapolate_d import ExtrapolateDConfig
from .utils import chinchilla_fit, get_coefficients_huber


def get_data_at_n(config: ExtrapolateDConfig):
    train_ds, train_hs, train_ys = [], [], []
    eval_ds, eval_hs, eval_ys = [], [], []

    with open(config.path) as file_ref:
        reader = csv.DictReader(file_ref)
        for r, row in enumerate(reader):
            d = float(row["throughput/total_tokens"])
            h = float(row["optim/learning_rate_group0"]) / float(row["learning_rate_peak"])
            y = np.mean([float(row[key]) for key in config.keys])
            batch_size = int(row["batch_size_in_tokens"])
            if config.outlier_threshold is not None and y > config.outlier_threshold:  # remove outliers
                continue
            if config.train_step_min is not None and d <= config.train_step_min * batch_size:
                continue
            if config.train_step_max is None or d <= config.train_step_max * batch_size:
                train_ds.append(d)
                train_hs.append(h)
                train_ys.append(y)
            elif config.eval_step_max is None or d <= config.eval_step_max * batch_size:
                eval_ds.append(d)
                eval_hs.append(h)
                eval_ys.append(y)

    return train_ds, train_hs, train_ys, eval_ds, eval_hs, eval_ys


def plot_d_scaling_at_n(
    train_ds,
    train_hs,
    train_ys,
    eval_ds,
    eval_hs,
    fitting_func,
    grad_func,
    final_loss_tokens,
    p0,
    predict=False,
    **plot_kwargs,
):
    train_dhs = [[d, h] for d, h in zip(train_ds, train_hs)]
    eval_dhs = [[d, h] for d, h in zip(eval_ds, eval_hs)]
    coefficients = get_coefficients_huber(train_dhs, train_ys, fitting_func, grad_func, p0=p0, bounds=None)

    plt.plot(
        train_ds + eval_ds,
        [fitting_func([d, h], coefficients) for [d, h] in train_dhs + eval_dhs],
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
