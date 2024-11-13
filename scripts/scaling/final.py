import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
    chinchilla_n_d_fit_e,
    get_coefficients,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_final_data_by_name,
    prettify,
    tasks,
)

MARKERS = ["s", "P", "p", "*"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--keys", nargs="+", default=[], help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument(
        "--num_to_avg", type=int, default=1, help="Number of final ckpts to average (for final loss fitting)"
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    parser.add_argument(
        "-a", "--accuracy", action="store_true", default=False, help="Predict accuracy metrics directly"
    )
    args = parser.parse_args()

    return args


def fit_step1(data_by_name, is_accuracy: bool = False):
    train_nds, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
            train_ys += data["ys"]

    # fit the parameters

    if is_accuracy:
        p0 = [1.0, 1.0, -0.01, -0.5, 0.1]
        coefficients = get_coefficients(train_nds, train_ys, chinchilla_n_d_fit_e, p0=p0)

    else:
        p0 = [3.0, 6.0, 0.1, 0.2, 1.0]
        bounds = [(0, None), (0, None), (0, None), (None, None), (None, None)]

        coefficients = get_coefficients_huber(
            train_nds,
            train_ys,
            chinchilla_n_d_fit,
            grad_chinchilla_n_d_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
        )

    return coefficients


def predict_step1(data_by_name, coefficients):
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

        if data["mode"] == "eval":
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error)


def str_chinchilla_n_d_fit(coefficients):
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)
    return f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}"


def plot_step1(
    configs,
    data_by_name,
    predicted_data_by_name,
    plotted_predicted_data_by_name,
    task_name,
    fit_str,
    is_accuracy=False,
    ax=plt.gca(),
):
    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        for i, (d, y) in enumerate(zip(data["ds"], data["ys"])):
            ax.scatter(d, y, color=config.color, marker=MARKERS[i], s=50)

        predicted_data = predicted_data_by_name[name]
        for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            ax.annotate(
                f"{prettify(rel_error)}",
                (d, y),
                textcoords="offset points",
                xytext=(6, 6),
                ha="center",
                fontsize=8,
                color=config.color,
            )

    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]
        ax.plot(
            data["ds"],
            data["ys"],
            color=config.color,
            linestyle="--",
            linewidth=2.0,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )
    ax.text(
        x=0.20,
        y=0.25,
        s=fit_str,
        fontsize=10,
        transform=ax.transAxes,
    )

    ax.legend(loc="upper right", ncols=1, fontsize=10)
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel("Accuracy" if is_accuracy else "Loss")
    ax.set_title(task_name)


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    if len(args.keys) == 1 and args.keys[0] == "all":
        args.keys = tasks.keys()

    sns.set_style("whitegrid")

    num_tasks = len(args.keys)
    fig, axes = plt.subplots(num_tasks, 1, figsize=(6, 4.5 * num_tasks), squeeze=False)

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        task = tasks[task_name]
        keys = task.get_accuracy_keys() if args.accuracy else task.get_loss_keys()
        data_by_name = get_final_data_by_name(configs, keys, num_to_avg=args.num_to_avg)

        coefficients = fit_step1(data_by_name, args.accuracy)

        # make predictions

        predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_step1(
            data_by_name, coefficients
        )

        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"

        plot_step1(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_chinchilla_n_d_fit(coefficients),
            args.accuracy,
            axes[i][0],
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(args.output_path, dpi=300)

    print(results)

    # y_1b_3T = chinchilla_n_d_fit([1176832000, 3e12], coefficients)
    # print(f"Predicted final loss for 1b-3T: {y_1b_3T:.3f}")
    # y_7b_2T = chinchilla_n_d_fit([6682316800, 2e12], coefficients)
    # print(f"Predicted final loss for 7b-2T: {y_7b_2T:.3f}")
    # y_7b_3T = chinchilla_n_d_fit([6682316800, 3e12], coefficients)
    # print(f"Predicted final loss for 7b-3T: {y_7b_3T:.3f}")
    # y_13b_5T = chinchilla_n_d_fit([13e9, 5e12], coefficients)
    # print(f"Predicted final loss for 13b-5T: {y_13b_5T:.3f}")


if __name__ == "__main__":
    main()
