# python scripts/scaling/variance_analysis.py -k v2_main_variance -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance_all.png --last_n_points 10 --run_prediction
# python scripts/scaling/variance_analysis.py -k v2_main_variance -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance_all.png --last_n_points 10 --run_prediction --print_table_as_latex
# python scripts/scaling/variance_analysis.py -k v2_main_variance -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance_all.png --last_n_points 10
# python scripts/scaling/variance_analysis.py -k mmlu_avg_test_5shot openbookqa_test_5shot -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance.png --last_n_points 10

import argparse
import os
import sys

# Suppress matplot warnings from other curve-fitting functions
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from predict import main as predict_main
from step1 import main as step1_main
from step2 import main as step2_main

from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step2_data_by_name,
    get_task_sets,
    tasks,
)

warnings.filterwarnings("ignore", category=UserWarning, message="No artists with labels found to put in legend")

FONTSIZE = 9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--keys", nargs="+", default=[], help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument("--last_n_points", type=int, default=10, help="Number of ckpts to compute variance over")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    parser.add_argument(
        "--print_table_as_latex", action="store_true", help="Whether to print the analysis table in latex"
    )
    parser.add_argument(
        "--run_prediction",
        action="store_true",
        help="Also report prediction errors alongisde the std. dev. of the ladder model",
    )
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


def _plot_single_variance_analysis(config, ds, xs, ys, task_name, last_n_points, loss_coeff_of_var, acc_coeff_of_var, ax1, ax2):
    assert config.mode == "eval"  # we are assuming that we have results of intermediate steps here
    total_points = len(ds)
    start_point = int(np.ceil(0.3 * total_points))

    # Step 1
    for ax_ in [ax1]:
        ax_.scatter(
            ds[start_point:-last_n_points],
            xs[start_point:-last_n_points],
            color=config.color,
            marker="o",
            s=10,
            alpha=0.3,
            label=f"{config.label} (intermediate checkpoints)",
        )
        ax_.scatter(
            ds[-last_n_points:],
            xs[-last_n_points:],
            color="orange",
            marker="o",
            s=10,
            alpha=0.5,
            label=f"{config.label} (final {last_n_points} checkpoints)",
        )

    ax1.set_xscale("log")
    ax1.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax1.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
    ax1.set_ylabel("Task loss", fontsize=FONTSIZE)
    display_name = tasks[task_name].display_name if task_name in tasks else task_name
    ax1.set_title(
        f"{display_name}\n"
        + r"(Loss relative SD$_{10}$: "
        + f"{loss_coeff_of_var*100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax1.tick_params(axis="x", which="both", reset=True)
    ax1.tick_params(axis="x", which="minor", labelsize=0)

    # Step 2
    for ax_ in [ax2]:
        ax_.scatter(
            xs[start_point:-last_n_points],
            ys[start_point:-last_n_points],
            color=config.color,
            marker="o",
            s=10,
            alpha=0.3,
            label=f"{config.label} (intermediate checkpoints)",
        )
        ax_.scatter(
            xs[-last_n_points:],
            ys[-last_n_points:],
            color="orange",
            marker="o",
            s=10,
            alpha=0.5,
            label=f"{config.label} (final {last_n_points} checkpoints)",
        )

    ax2.legend(loc="upper right", ncols=1, fontsize=10)
    ax2.set_xlabel("Task loss", fontsize=FONTSIZE)
    ax2.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    display_name = tasks[task_name].display_name if task_name in tasks else task_name
    ax2.set_title(
        f"{display_name}\n"
        + r"(Accuracy relative SD$_{10}$: "
        + f"{acc_coeff_of_var*100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))


def plot_variance_analysis(config, variance_results, last_n_points):
    num_tasks = len(variance_results)

    if num_tasks < 4:
        n_groups = 1
        fig, axes = plt.subplots(
            num_tasks // n_groups, 2 * n_groups, figsize=(2.75 * 2 * n_groups, 2.5 * (num_tasks // n_groups))
        )
    else:
        # Create a figure with spacing between the two groups of tasks
        n_groups = 2
        fig = plt.figure(figsize=(2.75 * 2 * n_groups, 2.25 * num_tasks // n_groups))
        gs = fig.add_gridspec(
            (num_tasks // n_groups),
            (2 * n_groups) + 1,
            width_ratios=[1, 1, 0, 1, 1],
            wspace=0.4,
            hspace=0.4,
            left=0.05,
            right=0.97,
            bottom=0.05,
            top=0.94,
        )
        axes = []
        for row in range(num_tasks // n_groups):
            row_axes = []
            for col in [0, 1, 3, 4]:
                row_axes.append(fig.add_subplot(gs[row, col]))
            axes.append(row_axes)
        axes = np.array(axes)

    # Plot results
    for i, (task_name, results) in enumerate(variance_results.items()):
        ax1: plt.Axes = axes[(i * 2) // (2*n_groups)][(i * 2) % (2*n_groups)]
        ax2: plt.Axes = axes[(i * 2) // (2*n_groups)][((i * 2) % (2*n_groups))+1]

        _plot_single_variance_analysis(
            config,
            results["data"]["ds"], results["data"]["xs"], results["data"]["ys"],
            task_name,
            last_n_points,
            results['loss_coeff_of_var'],
            results['acc_coeff_of_var'],
            ax1, ax2
        )

    # Collect all unique handles and labels
    all_handles = []
    all_labels = []
    for row in axes:
        for ax in row:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in all_labels:
                    all_handles.append(handle)
                    all_labels.append(label)

    # Remove redundant labels / legends
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i != len(axes) - 1:
                ax.set_xlabel("")
            if ax.get_legend():
                ax.get_legend().remove()

    # Add shared legend
    legend = fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        ncol=2,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1),
        handletextpad=0.3,
        columnspacing=0.7,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    if num_tasks < 4:
        fig.tight_layout(h_pad=0.02, rect=[0, 0, 1, 0.95])

    df = pd.merge(
        pd.merge(
            pd.DataFrame.from_dict({t: r['loss_std_dev'] for t, r in variance_results.items()}, orient="index")
            .reset_index()
            .rename({0: "Loss SD", "index": "Task"}, axis=1),
            pd.DataFrame.from_dict({t: r['loss_coeff_of_var'] for t, r in variance_results.items()}, orient="index")
            .reset_index()
            .rename({0: "Loss Rel SD (CV)", "index": "Task"}, axis=1),
        ),
        pd.merge(
            pd.DataFrame.from_dict({t: r['acc_std_dev'] for t, r in variance_results.items()}, orient="index")
            .reset_index()
            .rename({0: "Accuracy SD", "index": "Task"}, axis=1),
            pd.DataFrame.from_dict({t: r['acc_coeff_of_var'] for t, r in variance_results.items()}, orient="index")
            .reset_index()
            .rename({0: "Accuracy Rel SD (CV)", "index": "Task"}, axis=1),
        ),
    )

    return fig, df


def compute_variance(configs, keys, last_n_points):
    variance_results = {}

    for r, task_name in enumerate(keys):
        data_by_name = get_step2_data_by_name(configs, task_name)

        # get only entry of data_by_name
        assert len(data_by_name) == 1, f'Can only compute variance on one model at a time, seeing: {data_by_name.keys()}'
        name = list(data_by_name.keys())[0]
        data = data_by_name[name]

        config = configs[name]

        ds = data["ds"][-last_n_points:]
        xs = data["xs"][-last_n_points:]
        ys = data["ys"][-last_n_points:]

        loss_std_dev = np.std(xs)
        loss_coeff_of_var = loss_std_dev / np.mean(xs)
        acc_std_dev = np.std(ys)
        acc_coeff_of_var = acc_std_dev / np.mean(ys)

        variance_results[task_name] = {
            'config': config,
            'data': data,
            'last_n_points': last_n_points,
            'loss_std_dev': loss_std_dev,
            'acc_std_dev': acc_std_dev,
            'loss_coeff_of_var': loss_coeff_of_var,
            'acc_coeff_of_var': acc_coeff_of_var,
        }

    return name, variance_results


def run_two_step_prediction(keys_key):
    """Use subprocesses to run each stage of the ladder model"""
    # Run predictions on 7B scale
    orig_argv = sys.argv
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # surpress printing

    # Run step 1 only
    step1_args = [
        "-k",
        keys_key,
        "-c",
        "scripts/scaling/final_7b_only.json",
        "-o",
        "/tmp/step1_main.pdf",
        "--moving_avg",
        "5",
    ]
    sys.argv = [sys.argv[0]] + step1_args
    step1_df = step1_main()

    # Run step 2 only
    step2_args = [
        "-k",
        keys_key,
        "-c",
        "scripts/scaling/final_7b_only.json",
        "-o",
        "/tmp/step2_main.pdf",
        "--skip_perc",
        "0.1",
        "--moving_avg",
        "5",
    ]
    sys.argv = [sys.argv[0]] + step2_args
    step2_df = step2_main()

    # Run stacked
    predict_args = [
        "-k",
        keys_key,
        "-c",
        "scripts/scaling/final.json",
        "--step2-config-path",
        "scripts/scaling/final_7b_only.json",
        "-o",
        "/tmp/chained_main.pdf",
        "-n",
        "6887575552",
        "-d",
        "3945065873408",
        "-t",
        "7B-4T",
        "--skip_perc",
        "0.1",
        "--moving_avg",
        "5",
    ]
    sys.argv = [sys.argv[0]] + predict_args
    predict_df = predict_main()

    # Restore original argv and printing
    sys.stdout.close()
    sys.stdout = original_stdout
    sys.argv = orig_argv

    # Rename columns
    step1_df = step1_df[["Task", "Rel Error"]].rename(columns={"Rel Error": "7B Loss Rel Error"})
    step2_df = step2_df[["Task", "Rel Error"]].rename(columns={"Rel Error": "7B Accuracy Rel Error"})
    predict_df = predict_df[["Task", "Rel Error"]].rename(columns={"Rel Error": "7B Stacked Rel Error"})

    return step1_df, step2_df, predict_df


def print_table(df, last_n_points, print_table_as_latex=False):
    """Print std. dev. and (optionally) prediction errors"""
    print(f"Standard deviation over last {last_n_points} checkpoints:")
    if print_table_as_latex:
        # Convert to %
        latex_df = df.copy()
        latex_df.iloc[:, 1:] = latex_df.iloc[:, 1:]

        # Add red cell color to values above the column-wise mean
        means = latex_df.iloc[:, 1:].abs().mean()

        def format_with_color(x, col, col_mean):
            if "CV" in col:
                return "\\cellcolor{red!30}" + f"{x * 100:.2f} \\%" if x > col_mean else f"{x * 100:.2f} \\%"
            elif "Error" in col:
                # return f'{x:.2f} \\%'
                return (
                    "\\cellcolor{red!30}" + f"{abs(x) * 100:.1f} \\%"
                    if abs(x) > col_mean
                    else f"{abs(x) * 100:.1f} \\%"
                )
            else:
                return "\\cellcolor{red!30}" + f"{x:.4f}" if x > col_mean else f"{x:.4f}"

        formatters = {}
        for col, mean in means.items():
            formatters[col] = lambda x, c=col, m=mean: format_with_color(x, c, m)

        # Print table as labex
        latex_table = latex_df.to_latex(index=False, formatters=formatters, escape=False)
        print(latex_table)
    else:
        # Add red cell color to values above the column-wise mean
        colored_df = df.copy()
        means = colored_df.iloc[:, 1:].mean()

        # Print table header
        table_str = colored_df.to_string(justify="left")
        header = table_str.split("\n")[0]
        print(header)

        # Add terminal colors
        for col in [col for col in df.columns[1:]]:
            if "CV" in col:
                mean = means[col]
                colored_df[col] = df[col].apply(
                    lambda x: f"\033[91m{x*100:>10.2f}%\033[0m" if x > mean else f"\033[92m{x*100:>10.2f}%\033[0m"
                )
            elif "Error" in col:
                colored_df[col] = df[col].apply(lambda x: f"\033[0m{abs(x)*100:>10.2f}%\033[0m")
            else:
                mean = means[col]
                colored_df[col] = df[col].apply(
                    lambda x: f"\033[91m{x:>10.4f}\033[0m" if x > mean else f"\033[92m{x:>10.4f}\033[0m"
                )

        print(colored_df.to_string(justify="left", header=False))


def print_correlation_table(df):
    """Compute pairwise pearson correlations between numeric columns"""
    from scipy import stats

    numeric_cols = df.select_dtypes(include=["float64"]).columns

    correlations = np.zeros((len(numeric_cols), len(numeric_cols)))
    p_values = np.zeros((len(numeric_cols), len(numeric_cols)))

    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            corr, p_val = stats.pearsonr(abs(df[col1]), abs(df[col2]))
            correlations[i, j] = corr
            p_values[i, j] = p_val

    correlations_df = pd.DataFrame(correlations, columns=numeric_cols, index=numeric_cols)
    p_values_df = pd.DataFrame(p_values, columns=numeric_cols, index=numeric_cols)

    print("\nPearson correlation (p-values):")
    corr_with_p = (
        correlations_df.apply(lambda x: x.map("{:.3f}".format)).astype(str)
        + " ("
        + p_values_df.apply(lambda x: x.map("{:.3f}".format)).astype(str)
        + ")"
    )
    corr_with_p = corr_with_p.where(np.tril(np.ones(corr_with_p.shape), k=-1).astype(bool), "")
    print(corr_with_p.to_string(na_rep=""))


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    # Render only two tasks for paper
    # args.keys = ['mmlu_avg_test_5shot', 'openbookqa_test_5shot']

    keys = get_task_sets(args.keys)
    keys_key = str(args.keys[0])

    sns.set_style("whitegrid")

    model_name, variance_results = compute_variance(configs, keys, args.last_n_points)
    fig, df = plot_variance_analysis(configs[model_name], variance_results, args.last_n_points)

    if args.run_prediction:
        step1_rel_errors, step2_rel_errors, predict_rel_errors = run_two_step_prediction(keys_key)

        # Merge with the existing df
        df = df.merge(step1_rel_errors, on="Task", how="left")
        df = df.merge(step2_rel_errors, on="Task", how="left")
        df = df.merge(predict_rel_errors, on="Task", how="left")

    df = df.sort_values(by="Loss SD", ascending=False, ignore_index=True)

    df["Task"] = df["Task"].map(lambda x: tasks[x].display_name)

    print_table(df, args.last_n_points, args.print_table_as_latex)

    if args.run_prediction:
        print_correlation_table(df)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        fig.savefig(args.output_path, dpi=300)
        df.to_csv(args.output_path.replace(".pdf", ".csv").replace(".png", ".csv"), index=False)


if __name__ == "__main__":
    main()
