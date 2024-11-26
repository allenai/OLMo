# python scripts/scaling/single_step.py -k v2_main -c scripts/scaling/final.json -o figure/peteish-moreeval/single_step_main.pdf --moving_avg 5

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    combined_fit,
    get_coefficients_huber,
    grad_combined_fit,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

MARKERS = ["s", "P", "p", "*", "o"]
FONTSIZE = 9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    args.keys = get_task_sets(args.keys)

    return args


def fit_single_step(data_by_name, task_name):
    train_nds, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
            train_ys += data["ys"]

    p0 = [3.0, 5.0, 0.2, 0.3, 0.0, tasks[task_name].task_minimum - 1.0, 1.0]
    bounds = [(0, 10), (0, 10), (0, 1), (0, 1), (-10, 10), (-0.9999, 0), (0, 1)]
    coefficients = get_coefficients_huber(
        train_nds,
        train_ys,
        combined_fit,
        grad_combined_fit,
        p0=p0,
        bounds=bounds,
        max_iter=1000000,
        disp=False,
    )

    return coefficients


def predict_single_step(data_by_name, coefficients):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
    dmax = 1.5 * max([max(data["ds"]) for data in data_by_name.values()])

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "ds": data["ds"],
            "ys": [combined_fit([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
        }
        ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        ns = [data["ns"][0]] * len(ds)
        plotted_predicted_data_by_name[name] = {
            "ds": ds,
            "ys": [combined_fit([n, d], coefficients) for n, d in zip(ns, ds)],
        }

        if data["mode"] == "eval":
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error)


def str_combined_fit(coefficients):
    a, b, alpha, beta, E, p, q = coefficients
    A, B = np.exp(a), np.exp(b)
    return (
        f"Acc(N, D) = {p:.2f} / (1 + e^-({A:.2f} / N^{alpha:.2f} \n + {B:.2f} / D^{beta:.2f} + {E:.2f})) + {q:.2f}"
    )


def plot_single_step(
    configs,
    data_by_name,
    predicted_data_by_name,
    plotted_predicted_data_by_name,
    task_name,
    fit_str,
    ax=plt.gca(),
):
    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]
        ax.plot(
            data["ds"],
            data["ys"],
            color=config.color,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f'{config.label} (fitted)' if config.mode == "train" else None,
        )

    # plot the actual and predicted data
    unsigned_rel_errors = []
    num_eval_annotation = 0
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        for i, (d, y) in enumerate(zip(data["ds"], data["ys"])):
            ax.scatter(
                d,
                y,
                color=config.color,
                marker=MARKERS[i] if config.mode == "train" else "o",
                s=50 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )

        for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                unsigned_rel_errors.append(np.abs(rel_error))
            else:
                ax.scatter(
                    d,
                    y_pred,
                    color=config.color,
                    marker="x",
                    s=20,
                    label=f"{config.label} (predicted)",
                )
                ax.annotate(
                    f"{abs(rel_error * 100):.1f}%",
                    (d, y_pred),
                    textcoords="offset points",
                    xytext=(10, -5 + 10 * num_eval_annotation),
                    ha="left",
                    va="bottom",
                    fontsize=FONTSIZE,
                    color=config.color,
                )
                num_eval_annotation += 1
    avg_unsigned_rel_error = np.mean(unsigned_rel_errors)

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
    ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    ax.set_title(
        f"{tasks[task_name].display_name} ({avg_unsigned_rel_error * 100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )


def main():
    args = parse_args()
    configs = get_final_configs(args.config_path)

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.75 * num_cols, 2.25 * num_rows), squeeze=False)

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_step1_data_by_name(configs, task_name, y_metric="rc_acc", moving_avg=args.moving_avg)

        # fit the parameters
        coefficients = fit_single_step(data_by_name, task_name)

        # make predictions
        predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_single_step(
            data_by_name, coefficients
        )
        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"

        plot_single_step(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_combined_fit(coefficients),
            axes[i // num_cols][i % num_cols],
        )

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    # delete x-axis labels for all but the bottom row
    for i in range(num_cols):
        for j in range(num_rows):
            if j != num_rows - 1:
                axes[j][i].set_xlabel("")
            if i != 0:
                axes[j][i].set_ylabel("")

            axes[j][i].legend().remove()

    fig.tight_layout(w_pad=0.01)
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=10,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1.07),
        handletextpad=0.3,
        columnspacing=0.7,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")

    print(results)


if __name__ == "__main__":
    main()
