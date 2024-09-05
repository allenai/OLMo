import json

import matplotlib.pyplot as plt
import numpy as np

from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    chinchilla_n_d_lr_power_minus_fit,
    get_ax,
    get_coefficients_huber,
    get_data_by_name,
    grad_chinchilla_n_d_lr_power_minus_fit,
    parse_args,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_name = get_data_by_name(configs, args.keys, min_step=3000)

    num_axs = 5
    fig, axs = plt.subplots(1, num_axs, figsize=(num_axs * 8, 6))

    train_ns1hs, train_ys = [], []
    for name, data in data_by_name.items():
        config = configs[name]
        if config.mode == "train":
            train_ns1hs += [[n, s1, h] for n, s1, h in zip(data["ns"], data["s1s"], data["hs"])]
            train_ys += data["ys"]

    # fit the parameters
    coefficients = get_coefficients_huber(
        train_ns1hs,
        train_ys,
        chinchilla_n_d_lr_power_minus_fit,
        grad_chinchilla_n_d_lr_power_minus_fit,
        p0=[3.0, 6.0, 0.2, 0.4, 1.0, 0.05, 0.05],
        bounds=[(None, None), (None, None), (0, None), (0, None), (0, None), (0, None), (None, None)],
    )
    a, b, alpha, beta, E, F, gamma = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "ns": data["ns"],
            "ds": data["ds"],
            "ys": [
                chinchilla_n_d_lr_power_minus_fit([n, s1, h], coefficients)
                for n, s1, h in zip(data["ns"], data["s1s"], data["hs"])
            ],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        ax.scatter(data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=5.0)

    # plot the fitted curve
    for name, data in predicted_data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        if config.mode == "train":
            ax.plot(
                data["ds"],
                data["ys"],
                color=config.color,
                linestyle="--",
                linewidth=0.8,
                label=f"{config.label} (fitted)",
            )
        else:
            ax.plot(
                data["ds"],
                data["ys"],
                color=config.color,
                linestyle="--",
                linewidth=0.8,
                label=f"{config.label} (predicted)",
            )
    plt.text(
        x=0.40,
        y=0.90,
        s=f"L(n, s1, h) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / s1^{beta:.2f} + {E:.2f} - {F:.2f} * (1 - h) * n^{gamma:.2f}",
        fontsize=12,
        transform=fig.transFigure,
    )

    for ax in axs:
        ax.legend(loc="upper right", ncols=2, fontsize=10)
        ax.set_xlabel("Tokens (d)")
    axs[0].set_ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    plt.suptitle("Fitting loss curves, with LR power minus s1 correction")
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
