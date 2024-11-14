import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import get_coefficients, sigmoid
from olmo.scaling.scaling_laws.utils import (
    get_downstream_data_by_name,
    get_final_configs,
    get_task_sets,
    prettify,
    tasks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument(
        "--moving_avg",
        type=int,
        default=1,
        help="Number of final ckpts to keep moving average over (for loss to accuracy fitting)",
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    args.keys = get_task_sets(args.keys)

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(3, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3 * num_rows), squeeze=False)

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_downstream_data_by_name(configs, task_name, moving_avg=args.moving_avg)

        train_xs, train_ys = [], []
        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "train":
                train_xs += data["xs"]
                train_ys += data["ys"]

            else:
                data["xs"] = data["xs"][-1:]
                data["ys"] = data["ys"][-1:]

        # add ideal points (these are not plotted)
        train_xs.append(0.0)
        train_ys.append(tasks[task_name].task_maximum)
        train_xs.append(max(train_xs))
        train_ys.append(tasks[task_name].task_minimum)

        # fit the parameters
        coefficients, cov = get_coefficients(
            train_xs,
            train_ys,
            sigmoid,
            p0=[tasks[task_name].task_minimum - 1.0, 0.9, 3.0, 1.0],
            bounds=([-np.inf, 0.0, 0.0, 0.0], [0.0, np.inf, np.inf, np.inf]),
            disp=False,
            return_cov=True,
        )
        a, x0, k, b = coefficients

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

        # Compute standard errors for prediction
        # Compute the Jacobian matrix of partial derivatives with respect to parameters
        jacobian = np.zeros((len(plotted_predicted_data["xs"]), len(coefficients)))
        for j, x_val in enumerate(plotted_predicted_data["xs"]):
            # Partial derivatives
            jacobian[j, 0] = 1 / (1 + np.exp(-k * (x_val - x0)))
            jacobian[j, 1] = a * k * np.exp(-k * (x_val - x0)) / (1 + np.exp(-k * (x_val - x0))) ** 2
            jacobian[j, 2] = a * (x_val - x0) * np.exp(-k * (x_val - x0)) / (1 + np.exp(-k * (x_val - x0))) ** 2
            jacobian[j, 3] = 1

        # Compute standard errors for predictions
        intermediate = np.sum(jacobian @ cov @ jacobian.T, axis=1)
        # TODO: DANGER, this approximation may be bad.
        std_errors = np.sqrt(intermediate.clip(min=0.0))
        # std_errors = np.sqrt(np.abs(intermediate))

        # Compute prediction intervals
        plotted_y_lower = plotted_predicted_data["ys"] - 1.96 * std_errors
        plotted_y_upper = plotted_predicted_data["ys"] + 1.96 * std_errors

        ax = axes[i // num_cols][i % num_cols]

        # plot the actual and predicted data
        unsigned_rel_errs = []
        for name, data in data_by_name.items():
            config = configs[name]
            predicted_data = predicted_data_by_name[name]

            ax.scatter(
                data["xs"],
                data["ys"],
                color=config.color,
                marker="o",
                s=10,
                label=f"{config.label} ({'fitted' if config.mode == 'train' else 'predicted'})",
            )
            for x, y, y_pred in zip(data["xs"], data["ys"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

                if config.mode == "train":
                    unsigned_rel_errs.append(abs(rel_error))
                else:
                    ax.annotate(
                        f"{prettify(rel_error)}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(3, 3),
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color=config.color,
                    )
                    results += (
                        f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"
                    )
        avg_unsigned_rel_err = np.mean(unsigned_rel_errs)

        # plot the fitted curve
        ax.plot(
            plotted_predicted_data["xs"],
            plotted_predicted_data["ys"],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        ax.fill_between(
            plotted_predicted_data["xs"], plotted_y_lower, plotted_y_upper, color="pink", alpha=0.3
        )  # , label="95% Prediction Interval")

        ax.legend(loc="upper right", ncols=1, fontsize=8)
        ax.set_xlabel("Task loss")
        ax.set_ylabel("Task accuracy")

        ax.set_ylim([0, 1.0])
        ax.set_title(
            f"{task_name}\nσ(L) = {a:.2f} / (1 + e^(-{k:.2f}(L - {x0:.2f}))) + {b:.2f}\navg unsigned rel error on fitting = {avg_unsigned_rel_err * 100:.2f}%",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(args.output_path, dpi=300)

    print(results)


if __name__ == "__main__":
    main()
