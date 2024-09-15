import json

import matplotlib.pyplot as plt
import numpy as np

from olmo.scaling.scaling_laws.utils import (
    FinalConfig,
    chinchilla_n_d_fit,
    get_coefficients_huber,
    get_final_data_by_name,
    grad_chinchilla_n_d_fit,
    parse_args,
)


def main():
    args = parse_args()

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: FinalConfig(**config) for name, config in configs.items()}

    data_by_name = get_final_data_by_name(configs, args.keys, num_to_avg=args.num_to_avg)

    plt.figure()

    train_nds, train_ys = [], []
    for name, data in data_by_name.items():
        config = configs[name]
        if config.mode == "train":
            train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
            train_ys += data["ys"]

    # fit the parameters
    coefficients = get_coefficients_huber(
        train_nds,
        train_ys,
        chinchilla_n_d_fit,
        grad_chinchilla_n_d_fit,
        p0=[3.0, 6.0, 0.1, 0.2, 1.0],
        bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)],
    )
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}
    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "ds": data["ds"],
            "ys": [chinchilla_n_d_fit([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
        }
        ds = np.linspace(min(data["ds"]), max(data["ds"]), 100)
        ns = [data["ns"][0]] * len(ds)
        plotted_predicted_data_by_name[name] = {
            "ds": ds,
            "ys": [chinchilla_n_d_fit([n, d], coefficients) for n, d in zip(ns, ds)],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        plt.scatter(data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=5.0)

        predicted_data = predicted_data_by_name[name]
        for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            plt.annotate(
                f"{rel_error * 100:+.1f}%",
                (d, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
                color=config.color,
            )

    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
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
        x=0.25,
        y=0.50,
        s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )

    plt.legend(loc="upper right", ncols=2)
    plt.xlabel("Tokens (d)")
    plt.ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    plt.title("Fitting final loss")
    plt.savefig(args.output_path, dpi=300)

    y_1b_3T = chinchilla_n_d_fit([1176832000, 3e12], coefficients)
    print(f"Predicted final loss for 1b-3T: {y_1b_3T:.3f}")
    y_7b_2T = chinchilla_n_d_fit([6682316800, 2e12], coefficients)
    print(f"Predicted final loss for 7b-2T: {y_7b_2T:.3f}")
    y_7b_3T = chinchilla_n_d_fit([6682316800, 3e12], coefficients)
    print(f"Predicted final loss for 7b-3T: {y_7b_3T:.3f}")
    y_13b_5T = chinchilla_n_d_fit([13e9, 5e12], coefficients)
    print(f"Predicted final loss for 13b-5T: {y_13b_5T:.3f}")


if __name__ == "__main__":
    main()
