# python scripts/scaling/step1.py -k main -c scripts/scaling/final.json -o figure/peteish-final/step1_main.png
# python scripts/scaling/step1.py -k core_small_avg -c scripts/scaling/final.json -o figure/peteish-final/step1_core_small_avg.png

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import get_coefficients
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


def chinchilla_flops_fit(x, a, b, E):
    # return ax**b + E
    return a * np.pow(x, b) + E


def fit_step1(data_by_name, y_metric):
    train_fs, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_fs += data["fs"]
            train_ys += data["ys"]

    if y_metric == "rc_bpb":
        p0 = [2.0, -0.3, 0.1]
        bounds = ([0, -np.inf, -np.inf], [np.inf, 0, np.inf])
        coefficients, cov = get_coefficients(
            train_fs,
            train_ys,
            chinchilla_flops_fit,
            p0,
            bounds=bounds,
            disp=False,
            return_cov=True,
        )
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    return coefficients, cov


def predict_step1(configs, data_by_name, coefficients, y_metric):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    unsigned_rel_errors = []

    fmin = 0.8 * min([min(data["fs"]) for data in data_by_name.values()])
    fmax = 1.2 * max([max(data["fs"]) for data in data_by_name.values()])

    if y_metric == "rc_bpb":
        func = chinchilla_flops_fit
    elif y_metric == "rc_acc":
        func = chinchilla_flops_fit
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "fs": data["fs"],
            "ys": [func(f, *coefficients) for f in data["fs"]],
        }
        fs = np.exp(np.linspace(np.log(fmin), np.log(fmax), 100))
        plotted_predicted_data_by_name[name] = {
            "fs": fs,
            "ys": [func(f, *coefficients) for f in fs],
        }

        if configs[name].mode == "eval":
            predicted_data = predicted_data_by_name[name]
            for f, y, y_pred in zip(data["fs"], data["ys"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y
        else:
            predicted_data = predicted_data_by_name[name]
            for f, y, y_pred in zip(data["fs"], data["ys"], predicted_data["ys"]):
                rel_error_t = (y_pred - y) / y
                unsigned_rel_errors.append(np.abs(rel_error_t))

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error), unsigned_rel_errors


def str_chinchilla_flops_fit(coefficients):
    a, b, E = coefficients
    return f"L(F) = {a:.2f}F^{b:.2f} + {E:.2f}"


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

        for i, (f, y) in enumerate(zip(data["fs"], data["ys"])):
            ax.scatter(
                f,
                y,
                color=config.color,
                marker=MARKERS[i] if config.mode == "train" else "o",
                s=50,
            )

        for f, y, y_pred in zip(data["fs"], data["ys"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                unsigned_rel_errors.append(np.abs(rel_error))
            else:
                ax.annotate(
                    f"{prettify(rel_error)}",
                    (f, y),
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
            data["fs"],
            data["ys"],
            color=config.color,
            linestyle="--",
            linewidth=1.5,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=8)
    ax.set_xlabel("Flops (F)")
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

    fitting_error = 0

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
        (
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            (y, y_pred, rel_error),
            unsigned_rel_errors,
        ) = predict_step1(configs, data_by_name, coefficients, y_metric=args.y_metric)

        avg_unsigned_rel_error = np.mean(unsigned_rel_errors)
        fitting_error += avg_unsigned_rel_error

        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)} | {prettify(avg_unsigned_rel_error)}"

        if args.output_path:
            plot_step1(
                configs,
                data_by_name,
                predicted_data_by_name,
                plotted_predicted_data_by_name,
                task_name,
                str_chinchilla_flops_fit(coefficients),
                args.y_metric,
                axes[i // num_cols][i % num_cols],
            )

    if args.output_path:
        fig.tight_layout()
        fig.savefig(args.output_path, dpi=300)

    print(results)
    print("Total fitting error: ", prettify(fitting_error / num_tasks))


if __name__ == "__main__":
    main()
