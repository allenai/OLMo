import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_lr_minus_fit,
    get_coefficients_huber,
    grad_chinchilla_n_d_lr_minus_fit,
)
from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    get_ax,
    get_data_by_name,
    parse_args,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_name = get_data_by_name(configs, args.keys, min_step=5000)

    sns.set_style("whitegrid")

    num_axs = 5
    fig, axs = plt.subplots(1, num_axs, figsize=(num_axs * 4, 3))

    train_ndhs, train_ys = [], []
    for name, data in data_by_name.items():
        config = configs[name]
        if config.mode == "train":
            train_ndhs += [[n, d, h] for n, d, h in zip(data["ns"], data["ds"], data["hs"])]
            train_ys += data["ys"]

    # fit the parameters
    coefficients = get_coefficients_huber(
        train_ndhs,
        train_ys,
        chinchilla_n_d_lr_minus_fit,
        grad_chinchilla_n_d_lr_minus_fit,
        p0=[4.0, 4.0, 0.3, 0.3, 0.5, 0.0],
        bounds=[(None, None), (None, None), (0, None), (0, None), (0, None), (None, None)],
    )
    a, b, alpha, beta, E, F = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "ns": data["ns"],
            "ds": data["ds"],
            "ys": [
                chinchilla_n_d_lr_minus_fit([n, d, h], coefficients)
                for n, d, h in zip(data["ns"], data["ds"], data["hs"])
            ],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        ax.scatter(
            data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=10, alpha=0.25
        )

    # plot the fitted curve
    for name, data in predicted_data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        ax.plot(
            data["ds"],
            data["ys"],
            color=config.color,
            linestyle="--",
            linewidth=1.5,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )

    # annotate the error
    for name, data in data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        pred_data = predicted_data_by_name[name]
        rel_errors = [np.abs((pred_y - y) / y) for y, pred_y in zip(data["ys"], pred_data["ys"])]
        rel_error = np.mean(rel_errors)
        ax.annotate(
            f"{rel_error:.2%}",
            xy=(data["ds"][-1], pred_data["ys"][-1]),
            xycoords="data",
            xytext=(-4, 8),
            textcoords="offset points",
            fontsize=9,
            color=config.color,
        )

    for ax in axs:
        ax.legend(loc="upper right", ncols=1, fontsize=7)
        ax.set_xlabel("Tokens (D)")
    axs[0].set_ylabel("Loss")
    plt.suptitle(
        f"{args.key}\nL(N, D, H) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f} - {F:.2f} * H",
        fontsize=10,
    )
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
