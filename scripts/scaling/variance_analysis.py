import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    get_downstream_data_by_name,
    get_final_configs,
    get_task_sets,
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
    parser.add_argument("--last_n_points", type=int, default=10, help="Number of ckpts to compute variance over")
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


def inset_zoom_step1(ax, axins, x_vals, y_vals):
    x_width, y_width = (max(x_vals) - min(x_vals)), (max(y_vals) - min(y_vals))
    window_size = 0.2
    x_max = max(x_vals) + x_width * (window_size / 2)
    y_min = y_vals[-1] - y_width * (window_size / 2)
    x_min = x_max - x_width * window_size
    y_max = y_min + y_width * window_size
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    ax.indicate_inset_zoom(axins, edgecolor="black")


def inset_zoom_step2(ax, axins, x, y):
    x_width, y_width = 0.2, 0.05
    # x_width, y_width = 0.25, 0.1
    x_max = x + x_width
    x_min = x - x_width
    y_min = y - y_width
    y_max = y + y_width
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    ax.indicate_inset_zoom(axins, edgecolor="black")


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    args.keys = get_task_sets(args.keys)

    sns.set_style("whitegrid")

    num_tasks = len(args.keys)
    fig, axes = plt.subplots(num_tasks, 2, figsize=(6 * 2, 4.5 * num_tasks), squeeze=False)

    results = "Task Name | Task Loss Std Dev | Task Loss Coeff of Var | Accuracy Std Dev | Accuracy Coeff of Var"

    num_tasks = len(args.keys)

    loss_coeffs = {}
    acc_coeffs = {}

    for r, task_name in enumerate(args.keys):
        data_by_name = get_downstream_data_by_name(configs, task_name, moving_avg=args.moving_avg)
        last_n_points = args.last_n_points

        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "eval":  # we are assuming that we have results of intermediate steps here
                ds = data["ds"][-last_n_points:]
                xs = data["xs"][-last_n_points:]
                ys = data["ys"][-last_n_points:]

                loss_std_dev = np.std(xs)
                loss_coeff_of_var = loss_std_dev / np.mean(xs)
                acc_std_dev = np.std(ys)
                acc_coeff_of_var = acc_std_dev / np.mean(ys)

                results += f"\n{task_name} | {loss_std_dev:.5f} | {loss_coeff_of_var:.5f} | {acc_std_dev:.5f} | {acc_coeff_of_var:.5f}"

                loss_coeffs[task_name] = loss_coeff_of_var
                acc_coeffs[task_name] = acc_coeff_of_var

                # Step 1

                ax = axes[r][0]

                axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35])  # bottom right

                for ax_ in [ax, axins]:
                    ax_.scatter(
                        data["ds"][:-last_n_points],
                        data["xs"][:-last_n_points],
                        color=config.color,
                        alpha=0.3,
                        marker="o",
                        s=10,
                    )
                    ax_.scatter(ds, xs, color="blue", alpha=0.3, marker="o", s=10)

                inset_zoom_step1(ax, axins, data["ds"], xs)

                # ax.set_xscale("log")
                # ax.legend(loc="upper right", ncols=1, fontsize=10)
                ax.set_xlabel("Tokens (D)")
                ax.set_ylabel("Loss")
                ax.set_title(task_name)

                # Step 2

                ax = axes[r][1]

                axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35])  # bottom right

                for ax_ in [ax, axins]:
                    ax_.scatter(
                        data["xs"][:-last_n_points],
                        data["ys"][:-last_n_points],
                        color=config.color,
                        alpha=0.3,
                        marker="o",
                        s=10,
                    )
                    ax_.scatter(xs, ys, color="blue", alpha=0.3, marker="o", s=10)

                inset_zoom_step2(ax, axins, xs[-1], ys[-1])

                # ax.legend(loc="upper right", ncols=1, fontsize=10)
                ax.set_xlabel("Task Loss")
                ax.set_ylabel("Accuracy")
                ax.set_title(task_name)
                break

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(args.output_path, dpi=300)

    print(results)

    mean_loss_coeff = np.mean(list(loss_coeffs.values()))
    mean_acc_coeff = np.mean(list(acc_coeffs.values()))
    epsilon = 0.001

    print(
        f"avg loss coeff: {mean_loss_coeff}. tasks above threshold: ",
        [key for key, val in loss_coeffs.items() if val > mean_loss_coeff + epsilon],
    )
    print(
        f"avg acc coeff: {mean_acc_coeff}. tasks above threshold: ",
        [key for key, val in acc_coeffs.items() if val > mean_acc_coeff + epsilon],
    )


if __name__ == "__main__":
    main()
