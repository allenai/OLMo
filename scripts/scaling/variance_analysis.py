# python scripts/scaling/variance_analysis.py -k v2_main_variance -c scripts/scaling/final_variance.json -o figure/peteish-moreeval/variance.pdf --last_n_points 10

import argparse
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step2_data_by_name,
    get_task_sets,
    tasks
)
from step1 import main as step1_main
from step2 import main as step2_main
from predict import main as predict_main

# Suppress matplot warnings from other curve-fitting functions
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No artists with labels found to put in legend")

FONTSIZE = 9


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


def plot_variance_analysis(configs, keys, last_n_points, moving_avg):
    num_tasks = len(keys)
    
    if num_tasks < 4:
        n_groups = 1
        fig, axes = plt.subplots(num_tasks // n_groups, 2 * n_groups, figsize=(2.75 * 2 * n_groups, 2.25 * (num_tasks // n_groups)))
    else:
        # Create a figure with spacing between the two groups of tasks
        n_groups = 2
        fig = plt.figure(figsize=(2.25 * (num_tasks // n_groups), 2.75 * 2 * n_groups))
        gs = fig.add_gridspec(
            (num_tasks // n_groups),
            (2 * n_groups)+1,
            width_ratios=[1, 1, 0, 1, 1],
            wspace=0.4,
            hspace=0.32,
            left=0.05,
            right=0.97,
            bottom=0.05,
            top=0.94
        )
        axes = []
        for row in range(num_tasks // n_groups):
            row_axes = []
            for col in [0, 1, 3, 4]:
                row_axes.append(fig.add_subplot(gs[row, col]))
            axes.append(row_axes)
        axes = np.array(axes)

    loss_std_devs = {}
    acc_std_devs = {}
    loss_coeffs = {}
    acc_coeffs = {}

    for r, task_name in enumerate(keys):
        data_by_name = get_step2_data_by_name(configs, task_name, moving_avg=moving_avg)

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

                loss_std_devs[task_name] = loss_std_dev
                acc_std_devs[task_name] = acc_std_dev
                loss_coeffs[task_name] = loss_coeff_of_var
                acc_coeffs[task_name] = acc_coeff_of_var

                # Step 1
                ax: plt.Axes = axes[r // (num_tasks // (2 * n_groups))][(r % n_groups) * 2]

                inset_axis, no_legend = False, True

                if inset_axis:
                    if no_legend:
                        axins = ax.inset_axes([0.48, 0.48, 0.5, 0.5])
                    else:
                        axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35]) # bottom right
                    _axes = [ax, axins]
                else:
                    _axes = [ax]

                if inset_axis:
                    x, y = np.array(data["ds"]), np.array(data["xs"])
                    # Set the limits for the zoomed region
                    x_width, y_width = (max(x) - min(x)), (max(y) - min(y))
                    window_size = 0.2
                    x_max = max(x) + x_width * (window_size/2)
                    x_min = x_max - x_width * window_size
                    # y_min = y_pred - y_width * (window_size/2) # <- center on target/actual
                    y_min = y - y_width * (window_size/2) # <- center on prediction
                    y_max = y_min + y_width * window_size
                    axins.set_xlim(x_min, x_max)
                    # axins.set_ylim(y_min, y_max)
                    ax.indicate_inset_zoom(axins, edgecolor="black")

                for ax_ in _axes:
                    ax_.scatter(
                        data["ds"][start_point:-last_n_points],
                        data["xs"][start_point:-last_n_points],
                        color=config.color, marker="o", s=10, alpha=0.3,
                        # label=config.label
                        label=f'{config.label} (intermediate checkpoints)'
                    )
                    ax_.scatter(
                        ds, xs, 
                        color="orange", marker="o", s=10, alpha=0.5,
                        label=f'{config.label} (final {last_n_points} checkpoints)'
                    )

                ax.set_xscale("log")
                ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
                ax.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
                ax.set_ylabel("Task loss", fontsize=FONTSIZE)
                ax.set_title(
                    f"{tasks[task_name].display_name} " + r"(Loss SD$_{10}$: " + f'{loss_coeff_of_var:.4f})', 
                    fontsize=FONTSIZE,
                    fontweight="bold",
                )
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

                # Step 2
                ax: plt.Axes = axes[r // (num_tasks // (2 * n_groups))][((r % n_groups) * 2)+1]

                for ax_ in [ax]:  # , axins]:
                    ax_.scatter(
                        data["xs"][start_point:-last_n_points],
                        data["ys"][start_point:-last_n_points],
                        color=config.color, marker="o", s=10, alpha=0.3,
                        label=f'{config.label} (intermediate checkpoints)'
                    )
                    ax_.scatter(
                        xs, ys, 
                        color="orange", marker="o", s=10, alpha=0.5,
                        label=f'{config.label} (final {last_n_points} checkpoints)'
                    )

                ax.legend(loc="upper right", ncols=1, fontsize=10)
                ax.set_xlabel("Task loss", fontsize=FONTSIZE)
                ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
                ax.set_title(
                    f"{tasks[task_name].display_name} " + r"(Accuracy SD$_{10}$: " + f'{acc_coeff_of_var:.4f})', 
                    fontsize=FONTSIZE,
                    fontweight="bold",
                )
                ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                break

    # Collect all unique handles and labels
    all_handles = []
    all_labels = []
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
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
        all_handles, all_labels, 
        loc='upper center', ncol=2, fontsize=FONTSIZE, bbox_to_anchor=(0.5, 1), # 1
        handletextpad=0.3, columnspacing=0.7
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    if num_tasks < 4: fig.tight_layout(rect=[0, 0, 1, 0.95])

    df = pd.merge(
        pd.merge(
            pd.DataFrame.from_dict(loss_std_devs, orient="index").reset_index().rename({0: "Loss SD", "index": "Task"}, axis=1),
            pd.DataFrame.from_dict(loss_coeffs, orient="index").reset_index().rename({0: "Loss Rel SD (CV)", "index": "Task"}, axis=1)
        ),
        pd.merge(
            pd.DataFrame.from_dict(acc_std_devs, orient="index").reset_index().rename({0: "Accuracy SD", "index": "Task"}, axis=1),
            pd.DataFrame.from_dict(acc_coeffs, orient="index").reset_index().rename({0: "Accuracy Rel SD (CV)", "index": "Task"}, axis=1)
        )
    )

    return fig, df


def main():
    args = parse_args()
    keys_key = str(args.keys[0])

    configs = get_final_configs(args.config_path)

    keys = get_task_sets(args.keys)

    sns.set_style("whitegrid")

    # Render only two tasks for paper
    # args.keys = ['mmlu_avg_test_5shot', 'openbookqa_test_5shot']
    fig, df = plot_variance_analysis(configs, keys, args.last_n_points, args.moving_avg)

    # Run predictions on 7B scale    
    orig_argv = sys.argv
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w') # surpress printing

    # Run step 1 only
    step1_args = [
        "-k", keys_key,
        "-c", "scripts/scaling/final_variance_7b.json", 
        "-o", "/tmp/step1_main.pdf",
        "--moving_avg", "5"
    ]
    sys.argv = [sys.argv[0]] + step1_args
    step1_df = step1_main()

    # Run step 2 only
    step2_args = [
        "-k", keys_key,
        "-c", "scripts/scaling/final_variance_7b.json",
        "-o", "/tmp/step2_main.pdf",
        "--skip_perc", "0.1",
        "--moving_avg", "5"
    ]
    sys.argv = [sys.argv[0]] + step2_args
    step2_df = step2_main()

    # Run stacked
    predict_args = [
        "-k", keys_key,
        "-c", "scripts/scaling/final.json",
        "--step2-config-path", "scripts/scaling/final_variance_7b.json",
        "-o", "/tmp/chained_main.pdf",
        "-n", "6887575552",
        "-d", "3945065873408",
        "-t", "7B-4T",
        "--skip_perc", "0.1",
        "--moving_avg", "5"
    ]
    sys.argv = [sys.argv[0]] + predict_args
    predict_df = predict_main()

    # Restore original argv and printing
    sys.stdout.close()
    sys.stdout = original_stdout
    sys.argv = orig_argv

    # Merge with the existing df
    step1_rel_errors = step1_df[['Task', 'Rel Error']].rename(columns={'Rel Error': '7B Loss Rel Error'})
    step2_rel_errors = step2_df[['Task', 'Rel Error']].rename(columns={'Rel Error': '7B Accuracy Rel Error'})
    predict_rel_errors = predict_df[['Task', 'Rel Error']].rename(columns={'Rel Error': '7B Stacked Rel Error'})
    df = df.merge(step1_rel_errors, on='Task', how='left')
    df = df.merge(step2_rel_errors, on='Task', how='left')
    df = df.merge(predict_rel_errors, on='Task', how='left')

    df = df.sort_values(by="Loss SD", ascending=False, ignore_index=True)
    # df = df.sort_values(by="7B Stacked Rel Error", ascending=False, ignore_index=True)

    df['Task'] = df['Task'].map(lambda x: tasks[x].display_name)

    print(f'Standard deviation over last {args.last_n_points} checkpoints:')
    print_table_as_latex = False
    if print_table_as_latex:
        # Convert to %
        latex_df = df.copy()
        latex_df.iloc[:, 1:] = latex_df.iloc[:, 1:]

        # Add red cell color to values above the column-wise mean
        means = latex_df.iloc[:, 1:].abs().mean()
        def format_with_color(x, col, col_mean): 
            if 'CV' in col:
                return '\\cellcolor{red!30}' + f'{x * 100:.2f} \\%' if x > col_mean else f'{x * 100:.2f} \\%'
            elif 'Error' in col:
                # return f'{x:.2f} \\%'
                return '\\cellcolor{red!30}' + f'{abs(x) * 100:.1f} \\%' if abs(x) > col_mean else f'{abs(x) * 100:.1f} \\%'
            else:
                return '\\cellcolor{red!30}' + f'{x:.4f}' if x > col_mean else f'{x:.4f}'
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
        table_str = colored_df.to_string(justify='left')
        header = table_str.split('\n')[0]
        print(header)
        
        # Add terminal colors
        for col in [col for col in df.columns[1:]]:
            if "CV" in col:
                mean = means[col]
                colored_df[col] = df[col].apply(
                    lambda x: f"\033[91m{x*100:>10.2f}%\033[0m" if x > mean else f"\033[92m{x*100:>10.2f}%\033[0m"
                )
            elif 'Error' in col:
                colored_df[col] = df[col].apply(
                    lambda x: f"\033[0m{abs(x)*100:>10.2f}%\033[0m"
                )
            else:
                mean = means[col]
                colored_df[col] = df[col].apply(
                    lambda x: f"\033[91m{x:>10.4f}\033[0m" if x > mean else f"\033[92m{x:>10.4f}\033[0m"
                )
            
        print(colored_df.to_string(justify='left', header=False))

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        fig.savefig(args.output_path, dpi=300)
        df.to_csv(args.output_path.replace(".pdf", ".csv"), index=False)

    # Compute pairwise pearson correlations between numeric columns
    from scipy import stats
    numeric_cols = df.select_dtypes(include=['float64']).columns
    
    correlations = np.zeros((len(numeric_cols), len(numeric_cols)))
    p_values = np.zeros((len(numeric_cols), len(numeric_cols)))
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            corr, p_val = stats.pearsonr(abs(df[col1]), abs(df[col2]))
            correlations[i,j] = corr
            p_values[i,j] = p_val
    
    correlations_df = pd.DataFrame(correlations, columns=numeric_cols, index=numeric_cols)
    p_values_df = pd.DataFrame(p_values, columns=numeric_cols, index=numeric_cols)
    
    print("\nPearson correlation (p-values):")
    corr_with_p = correlations_df.apply(lambda x: x.map('{:.3f}'.format)).astype(str) + " (" + p_values_df.apply(lambda x: x.map('{:.3f}'.format)).astype(str) + ")"
    corr_with_p = corr_with_p.where(np.tril(np.ones(corr_with_p.shape), k=-1).astype(bool), '')
    print(corr_with_p.to_string(na_rep=''))


if __name__ == "__main__":
    main()

    # mean_loss_sd = np.mean(list(loss_std_devs.values()))
    # mean_acc_sd = np.mean(list(acc_std_devs.values()))
    # mean_loss_coeff = np.mean(list(loss_coeffs.values()))
    # mean_acc_coeff = np.mean(list(acc_coeffs.values()))

    # print(
    #     f"avg loss std dev: {mean_loss_sd:.4f}. tasks above threshold: ",
    #     [key for key, val in loss_std_devs.items() if val > mean_loss_sd],
    # )
    # print(
    #     f"avg acc std dev: {mean_acc_sd:.4f}. tasks above threshold: ",
    #     [key for key, val in acc_std_devs.items() if val > mean_acc_sd],
    # )

    # print(
    #     f"avg loss coeff: {mean_loss_coeff * 100:.3f}%. tasks above threshold: ",
    #     [key for key, val in loss_coeffs.items() if val > mean_loss_coeff],
    # )
    # print(
    #     f"avg acc coeff: {mean_acc_coeff * 100:.3f}%. tasks above threshold: ",
    #     [key for key, val in acc_coeffs.items() if val > mean_acc_coeff],
    # )