import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step2_data_by_name,
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

    results = "Task Name | Absolute Std Dev (Loss) | Relative Std Dev (Loss)| Absolute Std Dev (Accuracy) | Relative Std Dev (Accuracy"

    num_tasks = len(args.keys)

    loss_std_devs = {}
    acc_std_devs = {}
    loss_coeffs = {}
    acc_coeffs = {}

    for r, task_name in enumerate(args.keys):
        data_by_name = get_step2_data_by_name(configs, task_name, moving_avg=args.moving_avg)
        last_n_points = args.last_n_points

        for name, data in data_by_name.items():
            config = configs[name]
            if config.mode == "eval":  # we are assuming that we have results of intermediate steps here
                total_points = len(data["ds"])
                start_point = int(np.ceil(0.3 * total_points))
                ds = data["ds"][-last_n_points:]
                xs = data["xs"][-last_n_points:]
                ys = data["ys"][-last_n_points:]

                loss_std_dev = np.std(xs)
                loss_coeff_of_var = loss_std_dev / np.mean(xs)
                acc_std_dev = np.std(ys)
                acc_coeff_of_var = acc_std_dev / np.mean(ys)

                # results += f"\n{task_name} | {loss_std_dev:.5f} | {loss_coeff_of_var:.5f} | {acc_std_dev:.5f} | {acc_coeff_of_var:.5f}"
                # results += f"\n{task_name.replace('_', ' ').replace('5shot', '5-shot')} & {round(loss_coeff_of_var, 3)} & {round(acc_coeff_of_var, 3)} \\\\"

                results += f"\n{task_name.replace('_', ' ').replace('5shot', '5-shot')} & "
                results += f"{round(loss_std_dev, 3):.4f} & {round(loss_coeff_of_var, 3) * 100:.1f}\\% & "
                results += f"{round(acc_std_dev, 3):.4f} & {round(acc_coeff_of_var, 3) * 100:.1f}\\% \\\\"

                loss_std_devs[task_name] = loss_std_dev
                acc_std_devs[task_name] = acc_std_dev
                loss_coeffs[task_name] = loss_coeff_of_var
                acc_coeffs[task_name] = acc_coeff_of_var

                # Step 1

                ax = axes[r][0]

                # axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35])  # bottom right

                for ax_ in [ax]:  # , axins]:
                    ax_.scatter(
                        data["ds"][start_point:-last_n_points],
                        data["xs"][start_point:-last_n_points],
                        color="teal",
                        alpha=0.3,
                        marker="o",
                        s=10,
                    )
                    ax_.scatter(ds, xs, color="orange", marker="o", s=10)

                # inset_zoom_step1(ax, axins, data["ds"], xs)

                # ax.set_xscale("log")
                # ax.legend(loc="upper right", ncols=1, fontsize=10)
                ax.set_xlabel("Tokens (D)")
                ax.set_ylabel("Loss")
                ax.set_title(f"{task_name} coefficient of variance: {loss_coeff_of_var:.3f}")

                # Step 2

                ax = axes[r][1]

                # axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35])  # bottom right

                for ax_ in [ax]:  # , axins]:
                    ax_.scatter(
                        data["xs"][start_point:-last_n_points],
                        data["ys"][start_point:-last_n_points],
                        color="teal",
                        alpha=0.3,
                        marker="o",
                        s=10,
                    )
                    ax_.scatter(xs, ys, color="orange", marker="o", s=10)

                # inset_zoom_step2(ax, axins, xs[-1], ys[-1])

                # ax.legend(loc="upper right", ncols=1, fontsize=10)
                ax.set_xlabel("Task Loss")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"{task_name} coefficient of variance: {acc_coeff_of_var:.3f}")
                break

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(args.output_path, dpi=300)

    # print(results)

    mean_loss_sd = np.mean(list(loss_std_devs.values()))
    mean_acc_sd = np.mean(list(acc_std_devs.values()))
    mean_loss_coeff = np.mean(list(loss_coeffs.values()))
    mean_acc_coeff = np.mean(list(acc_coeffs.values()))
    epsilon = 0.0  # 0.001

    import pandas as pd

    loss_sd_df = (
        pd.DataFrame.from_dict(loss_std_devs, orient="index")
        .reset_index()
        .rename({0: "Loss SD", "index": "Task"}, axis=1)
    )
    loss_rsd_df = (
        pd.DataFrame.from_dict(loss_coeffs, orient="index")
        .reset_index()
        .rename({0: "Loss Rel SD (CV)", "index": "Task"}, axis=1)
    )

    acc_sd_df = (
        pd.DataFrame.from_dict(acc_std_devs, orient="index")
        .reset_index()
        .rename({0: "Accuracy SD", "index": "Task"}, axis=1)
    )
    acc_rsd_df = (
        pd.DataFrame.from_dict(acc_coeffs, orient="index")
        .reset_index()
        .rename({0: "Accuracy Rel SD (CV)", "index": "Task"}, axis=1)
    )

    loss_df = pd.merge(loss_sd_df, loss_rsd_df)
    acc_df = pd.merge(acc_sd_df, acc_rsd_df)
    df = pd.merge(loss_df, acc_df)
    df.to_csv("variance_analysis_1b.csv", index=False)

    df = df.sort_values(by="Loss SD", ascending=False)

    print(df)

    print(
        f"avg loss std dev: {mean_loss_sd:.4f}. tasks above threshold: ",
        [key for key, val in loss_std_devs.items() if val > mean_loss_sd + epsilon],
    )
    print(
        f"avg acc std dev: {mean_acc_sd:.4f}. tasks above threshold: ",
        [key for key, val in acc_std_devs.items() if val > mean_acc_sd + epsilon],
    )

    print(
        f"avg loss coeff: {mean_loss_coeff * 100:.3f}%. tasks above threshold: ",
        [key for key, val in loss_coeffs.items() if val > mean_loss_coeff + epsilon],
    )
    print(
        f"avg acc coeff: {mean_acc_coeff * 100:.3f}%. tasks above threshold: ",
        [key for key, val in acc_coeffs.items() if val > mean_acc_coeff + epsilon],
    )


if __name__ == "__main__":
    main()
