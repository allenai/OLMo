import json

import matplotlib.pyplot as plt
import numpy as np

from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    chinchilla_n_d_lr_fit,
    get_coefficients_huber,
    get_data_by_name,
    grad_chinchilla_n_d_lr_fit,
    parse_args,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_name = get_data_by_name(configs, args.keys, min_step=500)

    plt.figure(figsize=(8, 6))

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
        chinchilla_n_d_lr_fit,
        grad_chinchilla_n_d_lr_fit,
        p0=[4.0, 15.0, 0.25, 0.7, 1.5, 0.05],
        bounds=[(None, None), (None, None), (0, None), (0, None), (0, None), (0, None)],
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
                chinchilla_n_d_lr_fit([n, d, h], coefficients)
                for n, d, h in zip(data["ns"], data["ds"], data["hs"])
            ],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        plt.scatter(data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=5.0)

    # plot the fitted curve
    for name, data in predicted_data_by_name.items():
        config = configs[name]
        if config.mode == "train":
            plt.plot(
                data["ds"],
                data["ys"],
                color=config.color,
                linestyle="--",
                linewidth=0.8,
                label=f"{config.label} (fitted)",
            )
        else:
            plt.plot(
                data["ds"],
                data["ys"],
                color=config.color,
                linestyle="--",
                linewidth=0.8,
                label=f"{config.label} (predicted)",
            )
    plt.text(
        x=0.20,
        y=0.45,
        s=f"L(n, d, h) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f} + {F:.2f} * h",
        fontsize=8,
        transform=plt.gca().transAxes,
    )

    plt.legend(loc="upper right", ncols=2, fontsize=6)
    plt.xlabel("Tokens (d)")
    plt.ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    plt.title("Fitting loss curves, with LR correction")
    plt.savefig(args.output_path, dpi=300)


if __name__ == "__main__":
    main()
