import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    FinalConfig,
    chinchilla_flops_fit,
    chinchilla_fit,
    get_flops_data_by_name,
    get_coefficients_huber,
    get_coefficients,
    grad_chinchilla_flops_fit,
    parse_args,
)

MARKERS = ["s", "P", "p", "*"]


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: FinalConfig(**config) for name, config in configs.items()}

    data_by_name = get_flops_data_by_name(configs, args.keys, num_to_avg=args.num_to_avg)

    sns.set_style("whitegrid")

    plt.figure(figsize=(6, 4.5))

    train_fs, train_ys = [], []
    for name, data in data_by_name.items():
        config = configs[name]
        if config.mode == "train":
            train_fs += data["fs"]
            train_ys += data["ys"]

    # fit the parameters
    # coefficients = get_coefficients_huber(
    #     train_fs,
    #     train_ys,
    #     chinchilla_flops_fit,
    #     grad_chinchilla_flops_fit,
    #     p0=[-3.0, 0.09, 0.1],
    #     bounds=[(None, None), (None, None), (None, None)],
    #     max_iter=100000,
    # )

    coefficients = get_coefficients(train_fs, train_ys, chinchilla_fit, p0=[-3.0, 0.09, 0.1])

    a, b, E = coefficients

    # make predictions
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "fs": data["fs"],
            "ys": [chinchilla_flops_fit(flops, coefficients) for flops in data["fs"]],
        }
        fs = np.linspace(min(data["fs"]), max(data["fs"]), 100)
        plotted_predicted_data_by_name[name] = {
            "fs": fs,
            "ys": [chinchilla_flops_fit(flops, coefficients) for flops in fs],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        # plt.scatter(data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=10)
        for i, (f, y) in enumerate(zip(data["fs"], data["ys"])):
            plt.scatter(f, y, color=config.color, marker=MARKERS[i], s=50)

        predicted_data = predicted_data_by_name[name]
        for f, y, y_pred in zip(data["fs"], data["ys"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            plt.annotate(
                f"{rel_error * 100:+.1f}%",
                (f, y),
                textcoords="offset points",
                xytext=(6, 6),
                ha="center",
                fontsize=8,
                color=config.color,
            )

    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]
        plt.plot(
            data["fs"],
            data["ys"],
            color=config.color,
            linestyle="--",
            linewidth=2.0,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )
    plt.text(
        x=0.20,
        y=0.55,
        s=f"L(F) = {a:.2f} F ^ {b:.2f} + {E:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )

    plt.xscale("log")
    plt.legend(loc="upper right", ncols=1, fontsize=10)
    plt.xlabel("Flops (F)")
    plt.ylabel("Loss")
    plt.title(args.key)
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")

    # y_1b_3T = chinchilla_flops_fit([1176832000, 3e12], coefficients)
    # print(f"Predicted final loss for 1b-3T: {y_1b_3T:.3f}")
    # y_7b_2T = chinchilla_flops_fit([6682316800, 2e12], coefficients)
    # print(f"Predicted final loss for 7b-2T: {y_7b_2T:.3f}")
    # y_7b_3T = chinchilla_flops_fit([6682316800, 3e12], coefficients)
    # print(f"Predicted final loss for 7b-3T: {y_7b_3T:.3f}")
    # y_13b_5T = chinchilla_flops_fit([13e9, 5e12], coefficients)
    # print(f"Predicted final loss for 13b-5T: {y_13b_5T:.3f}")


if __name__ == "__main__":
    main()
