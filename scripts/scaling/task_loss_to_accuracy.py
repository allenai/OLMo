import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import get_coefficients, sigmoid
from olmo.scaling.scaling_laws.utils import (
    get_downstream_data_by_name,
    get_final_configs,
    prettify,
    tasks,
)

MARKERS = ["s", "P", "p", "*"]

# MARKERS = {"1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}


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
    args = parser.parse_args()

    return args


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
        data_by_name = get_downstream_data_by_name(configs, task_name)

        train_xs, train_ys = [], []
        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "train":
                train_xs += data["xs"]
                train_ys += data["ys"]

        # add ideal points (these are not plotted) # TODO: should we plot?
        train_xs.append(0.0001)
        train_ys.append(tasks[task_name].task_maximum)
        train_xs.append(2.6)  # TODO: make task-specific
        train_ys.append(tasks[task_name].task_minimum)

        # fit the parameters

        coefficients = get_coefficients(
            train_xs, train_ys, sigmoid, p0=[tasks[task_name].task_minimum - 1.0, 0.9, 3.0, 1.0]
        )

        L, x0, k, b = coefficients

        # make predictions
        predicted_data_by_name = {}
        plotted_predicted_data_by_name = {}
        for name, data in data_by_name.items():
            config = configs[name]
            predicted_data_by_name[name] = {
                "xs": data["xs"],
                "ys": [sigmoid(x, *coefficients) for x in data["xs"]],
            }
            xs = np.linspace(min(data["xs"]), max(data["xs"]), 100)
            plotted_predicted_data_by_name[name] = {
                "xs": xs,
                "ys": [sigmoid(x, *coefficients) for x in xs],
            }

        ax = axes[i][0]

        # plot the actual data
        for name, data in data_by_name.items():
            config = configs[name]
            # plt.scatter(data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=10)
            for i, (x, y) in enumerate(zip(data["xs"], data["ys"])):
                ax.scatter(x, y, color=config.color, marker="o", s=10)

            if config.mode == "eval":
                predicted_data = predicted_data_by_name[name]
                for x, y, y_pred in zip(data["xs"], data["ys"], predicted_data["ys"]):
                    rel_error = (y_pred - y) / y
                    ax.annotate(
                        f"{prettify(rel_error)}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(6, 6),
                        ha="center",
                        fontsize=8,
                        color=config.color,
                    )

                results += (
                    f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"
                )

        # plot the fitted curve
        for name, data in plotted_predicted_data_by_name.items():
            config = configs[name]
            ax.plot(
                data["xs"],
                data["ys"],
                color=config.color,
                linestyle="--",
                linewidth=2.0,
                label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
            )
        ax.text(
            x=0.20,
            y=0.55,
            s=f"σ(L, x0, k, b) = {L:.2f} / (1 + e^(-({k:.2f}(x - {x0:.2f})))) + {b:.2f}",
            fontsize=10,
            transform=plt.gca().transAxes,
        )

        ax.legend(loc="upper right", ncols=1, fontsize=10)
        ax.set_xlabel("Task Loss (x)")
        ax.set_ylabel("Task accuracy")
        ax.set_title(task_name)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(args.output_path, dpi=300)

    print(results)

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
