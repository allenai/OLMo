import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
    get_coefficients,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
    sigmoid,
)
from olmo.scaling.scaling_laws.utils import (
    get_step2_data_by_name,
    get_final_configs,
    get_task_sets,
    prettify,
    tasks,
)

# MARKERS = ["s", "P", "p", "*"]

MARKERS = {"1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--keys", nargs="+", default=[], help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument(
        "--num_to_avg", type=int, default=1, help="Number of final ckpts to average (for final loss fitting)"
    )
    parser.add_argument(
        "--moving_avg",
        type=int,
        default=1,
        help="Number of final ckpts to keep moving average over (for loss to accuracy fitting)",
    )
    parser.add_argument(
        "--skip_perc",
        type=float,
        default=0.0,
        help="Percentage of intermediate ckpts to skip from the beginning (for loss to accuracy fitting)",
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    return args


def str_chinchilla_n_d_fit(coefficients):
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)
    return f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}"


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    args.keys = get_task_sets(args.keys)

    sns.set_style("whitegrid")

    num_tasks = len(args.keys)
    fig, axes = plt.subplots(num_tasks, 3, figsize=(6 * 3, 4.5 * num_tasks), squeeze=False)

    results = "Task Name | Loss Error | Accuracy Error | Stacked Accuracy Error"

    for r, task_name in enumerate(args.keys):
        loss_error = None
        acc_error = None
        cum_acc_error = None

        # Step 1

        # keys = task.get_loss_keys()
        # data_by_name = get_final_data_by_name(configs, keys, num_to_avg=args.num_to_avg)
        data_by_name = get_step2_data_by_name(configs, task_name, moving_avg=args.moving_avg, last_n_points=1)

        train_nds, train_xs = [], []
        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "train":
                train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
                train_xs += data["xs"]

        ## Step 1: fit the parameters

        p0 = [3.0, 6.0, 0.1, 0.2, 1.0]
        bounds = [(0, None), (0, None), (0, None), (None, None), (None, None)]

        coefficients = get_coefficients_huber(
            train_nds,
            train_xs,
            chinchilla_n_d_fit,
            grad_chinchilla_n_d_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
        )

        ## Step 1: make predictions
        predicted_data_by_name = {}
        plotted_predicted_data_by_name = {}
        for name, data in data_by_name.items():
            predicted_data_by_name[name] = {
                "ds": data["ds"],
                "xs": [chinchilla_n_d_fit([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
            }
            ds = np.linspace(min(data["ds"]), max(data["ds"]), 100)
            ns = [data["ns"][0]] * len(ds)
            plotted_predicted_data_by_name[name] = {
                "ds": ds,
                "xs": [chinchilla_n_d_fit([n, d], coefficients) for n, d in zip(ns, ds)],
            }

            config = configs[name]
            if config.mode == "eval":
                predicted_data = predicted_data_by_name[name]
                for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["xs"]):
                    rel_error = (y_pred - y) / y
                    loss_error = rel_error

        ## Step 1: plot

        ax = axes[r][0]

        # plot the actual data
        for name, data in data_by_name.items():
            config = configs[name]
            for i, (d, y, length) in enumerate(zip(data["ds"], data["xs"], data["ls"])):
                ax.scatter(d, y, color=config.color, marker=MARKERS.get(length, "*"), s=50)

            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["ys"], predicted_data["xs"]):
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
                data["xs"],
                color=config.color,
                linestyle="--",
                linewidth=2.0,
                label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
            )
        ax.text(
            x=0.20,
            y=0.25,
            s=str_chinchilla_n_d_fit(coefficients),
            fontsize=10,
            transform=ax.transAxes,
        )

        ax.set_xscale("log")
        ax.legend(loc="upper right", ncols=1, fontsize=10)
        ax.set_xlabel("Tokens (D)")
        ax.set_ylabel("Loss")
        ax.set_title(task_name)

        # Step 2

        data_by_name = get_step2_data_by_name(
            configs, task_name, moving_avg=args.moving_avg, skip_perc=args.skip_perc
        )

        # Add row for predicted loss from step 1
        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "eval":
                predicted_data = predicted_data_by_name[name]  # step1 predictions
                data["xs"] += predicted_data["xs"]
                data["ys"] += data["ys"]
                data["ds"] += data["ds"]

        train_xs, train_ys = [], []
        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "train":
                train_xs += data["xs"]
                train_ys += data["ys"]

        # add ideal points
        min_ideal_point = (max(train_xs), tasks[task_name].task_minimum)
        max_ideal_point = (0.0, tasks[task_name].task_maximum)

        train_xs.append(min_ideal_point[0])
        train_ys.append(min_ideal_point[1])
        train_xs.append(max_ideal_point[0])
        train_ys.append(max_ideal_point[1])

        ## Step 2: fit the parameters

        coefficients, cov = get_coefficients(
            train_xs,
            train_ys,
            sigmoid,
            p0=[tasks[task_name].task_minimum - 1.0, 0.9, 3.0, 1.0],
            bounds=([-np.inf, 0.0, 0.0, 0.0], [0.0, np.inf, np.inf, 1.0]),
            disp=False,
            return_cov=True,
        )

        L, x0, k, b = coefficients

        ## Step 2: make predictions
        # predicted_data_by_name = {}
        # plotted_predicted_data_by_name = {}
        # for name, data in data_by_name.items():
        #     config = configs[name]
        #     predicted_data_by_name[name] = {
        #         "xs": data["xs"],
        #         "ys": [sigmoid(x, *coefficients) for x in data["xs"]],
        #     }

        #     # include ideal points
        #     # max_ideal_point will have smaller loss value
        #     xs = np.linspace(
        #         min(min(data["xs"]), max_ideal_point[0]), max(max(data["xs"]), min_ideal_point[0]), 100
        #     )

        #     plotted_predicted_data_by_name[name] = {
        #         "xs": xs,
        #         "ys": [sigmoid(x, *coefficients) for x in xs],
        #     }

        # make predictions
        predicted_data_by_name = {}
        for name, data in data_by_name.items():
            config = configs[name]
            predicted_data_by_name[name] = {
                "xs": data["xs"],
                "ys": [sigmoid(x, *coefficients) for x in data["xs"]],
            }
        xmin = 0.9 * min(min(data["xs"]) for data in data_by_name.values())
        xmax = max(max(data["xs"]) for data in data_by_name.values())
        xs = np.linspace(xmin, xmax, 100)
        plotted_predicted_data = {
            "xs": xs,
            "ys": [sigmoid(x, *coefficients) for x in xs],
        }

        ## Step 2: plot

        ax = axes[r][1]

        # plot the actual data
        for name, data in data_by_name.items():
            config = configs[name]
            predicted_data = predicted_data_by_name[name]

            if config.mode == "train":
                ax.scatter(data["xs"], data["ys"], color=config.color, marker="o", s=10)
            else:
                predicted_data = predicted_data_by_name[name]
                for i, (x, y, y_pred) in enumerate(zip(data["xs"], data["ys"], predicted_data["ys"])):
                    rel_error = (y_pred - y) / y
                    if i == 0:
                        label = "using actual loss"
                        marker = "o"
                        acc_error = rel_error
                    else:
                        label = "using step1 loss"
                        marker = "x"
                        cum_acc_error = rel_error
                    label = f"{config.label} ({label}): {prettify(rel_error)}"
                    ax.scatter(x, y, color=config.color, marker=marker, s=10, label=label)

        # # plot ideal points
        # ax.scatter(min_ideal_point[0], min_ideal_point[1], color="grey", marker="^", s=20)
        # ax.scatter(max_ideal_point[0], max_ideal_point[1], color="grey", marker="^", s=20)

        # plot the fitted curve
        ax.plot(
            plotted_predicted_data["xs"],
            plotted_predicted_data["ys"],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        ax.text(
            x=0.20,
            y=0.55,
            s=f"σ(L, x0, k, b) = {L:.2f} / (1 + e^(-({k:.2f}(x - {x0:.2f})))) + {b:.2f}",
            fontsize=10,
            transform=ax.transAxes,
        )

        ax.legend(loc="upper right", ncols=1, fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.set_xlabel("Task Loss (x)")
        ax.set_ylabel("Task accuracy")
        ax.set_title(task_name)

        # Stacked plot

        ax = axes[r][2]

        # plot the actual data
        for name, data in data_by_name.items():
            config = configs[name]
            ax.scatter(data["ds"], data["ys"], color=config.color, marker="o", s=10, label=config.label)

        ax.set_xscale("log")
        ax.legend(loc="upper left", ncols=1, fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.set_xlabel("Tokens (D)")
        ax.set_ylabel("Task accuracy")
        ax.set_title(task_name)

        # Append results
        results += f"\n{task_name} | {prettify(loss_error)} | {prettify(acc_error)} | {prettify(cum_acc_error)}"

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
