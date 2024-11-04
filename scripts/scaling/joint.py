import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    chinchilla_n_d_fit,
    get_ax,
    get_coefficients_huber,
    get_data_by_name,
    grad_chinchilla_n_d_fit,
    parse_args,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_name = get_data_by_name(configs, args.keys, min_step=3000)

    sns.set_style("whitegrid")

    num_axs = 5
    fig, axs = plt.subplots(1, num_axs, figsize=(num_axs * 6, 4.5))

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
        chinchilla_n_d_fit,
        grad_chinchilla_n_d_fit,
        p0=[4.0, 15.0, 0.25, 0.7, 1.5],
        bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)],
    )
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "ns": data["ns"],
            "ds": data["ds"],
            "ys": [chinchilla_n_d_fit([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        ax.scatter(
            data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=5, alpha=0.4
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
            linewidth=2.0,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )

    # annotate the error
    all_rel_errors = []
    for name, data in data_by_name.items():
        config = configs[name]
        ax = axs[get_ax(name)]
        pred_data = predicted_data_by_name[name]
        rel_errors = [np.abs((pred_y - y) / y) for y, pred_y in zip(data["ys"], pred_data["ys"])]
        all_rel_errors += rel_errors
        rel_error = np.mean(rel_errors)
        ax.annotate(
            f"err: {rel_error:.2%}",
            xy=(data["ds"][-1], pred_data["ys"][-1]),
            xycoords="data",
            xytext=(-10, 8),
            textcoords="offset points",
            fontsize=9,
            color=config.color,
        )
    axs[3].annotate(
        f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}\nAvg err: {np.mean(all_rel_errors):.2%}",
        xy=(0.15, 0.55),
        xycoords="axes fraction",
        fontsize=9,
    )
    plt.text(
        x=0.40,
        y=0.90,
        s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
        fontsize=12,
        transform=fig.transFigure,
    )

    for ax in axs:
        ax.legend(loc="upper right", ncols=2, fontsize=8)
        ax.set_xlabel("Tokens (D)")
    axs[0].set_ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    axs[3].set_ylabel("Loss")
    axs[3].set_title(args.key)
    plt.suptitle("Fitting loss curves")
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
