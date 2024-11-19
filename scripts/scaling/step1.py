# python scripts/scaling/step1.py -k main -c scripts/scaling/final.json -o figure/peteish-final/step1_main.png
# python scripts/scaling/step1.py -k core_small_avg -c scripts/scaling/final.json -o figure/peteish-final/step1_core_small_avg.png

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
    chinchilla_n_d_negated_fit,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
    grad_chinchilla_n_d_negated_fit,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_task_sets,
    prettify,
)

MARKERS = ["s", "P", "p", "*", "o"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument(
        "-y", "--y_metric", default="rc_bpb", choices=["rc_bpb", "rc_acc"], help="Metric to predict"
    )
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=False, help="Path to write output figure")
    args = parser.parse_args()

    if not args.keys:
        args.keys = ["main"]
    args.keys = get_task_sets(args.keys)

    return args


def fit_step1(data_by_name, y_metric):
    train_nds, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
            train_ys += data["ys"]

    if y_metric == "rc_bpb":
        p0 = [3.0, 6.0, 0.1, 0.2, 1.0]
        bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
        coefficients, cov = get_coefficients_huber(
            train_nds,
            train_ys,
            chinchilla_n_d_fit,
            grad_chinchilla_n_d_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
            return_cov=True,
        )
    elif y_metric == "rc_acc":
        p0 = [2.0, 2.0, 0.2, 0.2, 1.0]
        bounds = [(0, None), (0, None), (0, None), (None, None), (None, None)]
        coefficients, cov = get_coefficients_huber(
            train_nds,
            train_ys,
            chinchilla_n_d_negated_fit,
            grad_chinchilla_n_d_negated_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
            return_cov=True
        )
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    return coefficients, cov


def predict_step1(data_by_name, coefficients, y_metric):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
    dmax = 1.2 * max([max(data["ds"]) for data in data_by_name.values()])

    if y_metric == "rc_bpb":
        func = chinchilla_n_d_fit
    elif y_metric == "rc_acc":
        func = chinchilla_n_d_negated_fit
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "ds": data["ds"],
            "ys": [func([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
        }
        ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        ns = [data["ns"][0]] * len(ds)
        plotted_predicted_data_by_name[name] = {
            "ds": ds,
            "ys": [func([n, d], coefficients) for n, d in zip(ns, ds)],
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
    y_metric,
    ax=plt.gca(),
):
    # plot the actual and predicted data
    unsigned_rel_errors = []
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        for i, (d, y) in enumerate(zip(data["ds"], data["ys"])):
            ax.scatter(
                d,
                y,
                color=config.color,
                marker=MARKERS[i] if config.mode == "train" else "o",
                s=50,
            )

        for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                unsigned_rel_errors.append(np.abs(rel_error))
            else:
                ax.annotate(
                    f"{prettify(rel_error)}",
                    (d, y),
                    textcoords="offset points",
                    xytext=(3, 3),
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=config.color,
                )
    avg_unsigned_rel_error = np.mean(unsigned_rel_errors)

    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]
        ax.plot(
            data["ds"],
            data["ys"],
            color=config.color,
            linestyle="--",
            linewidth=1.5,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=8)
    ax.set_xlabel("Tokens (D)")
    if y_metric == "rc_bpb":
        ax.set_ylabel("Task loss")
    elif y_metric == "rc_acc":
        ax.set_ylabel("Task RC accuracy")
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")
    ax.set_title(
        f"{task_name}\n{fit_str}\navg rel error on fitting = {avg_unsigned_rel_error * 100:.2f}%",
        fontsize=9,
    )


def main():
    args = parse_args()
    configs = get_final_configs(args.config_path)

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols

    if args.output_path:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.75 * num_cols, 3.25 * num_rows), squeeze=False)

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric=args.y_metric, moving_avg=args.moving_avg
        )

        # fit the parameters
        coefficients, cov = fit_step1(data_by_name, args.y_metric)

        # make predictions
        predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_step1(
            data_by_name, coefficients, y_metric=args.y_metric
        )
        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"

        if args.output_path:
            plot_step1(configs,
                data_by_name,
                predicted_data_by_name,
                plotted_predicted_data_by_name,
                task_name,
                str_chinchilla_n_d_fit(coefficients),
                args.y_metric,
                axes[i // num_cols][i % num_cols],
            )

    if args.output_path:
        fig.tight_layout()
        fig.savefig(args.output_path, dpi=300)

    print(results)


if __name__ == "__main__":
    main()
