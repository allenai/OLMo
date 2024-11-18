import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_lr_minus_fit,
    sigmoid,
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

    coefficients = [3.5051796, 4.52225812, 0.25991131, 0.28089689, 0.57286154, 0.02209304]
    sigmoid_coeffs = [-0.77899618, 0.75179073, 12.64004912, 1.03518459]

    # make predictions
    predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "ns": data["ns"],
            "ds": data["ds"],
            "ys": [
                sigmoid(chinchilla_n_d_lr_minus_fit([n, d, h], coefficients), *sigmoid_coeffs)
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
            xytext=(-4, -12),
            textcoords="offset points",
            fontsize=9,
            color=config.color,
        )

    for ax in axs:
        ax.legend(loc="lower right", ncols=1, fontsize=7)
        ax.set_xlabel("Tokens (D)")
    axs[0].set_ylabel("Accuracy")
    plt.suptitle(
        f"{args.key.replace('-acc', '')}",
        fontsize=10,
    )
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
