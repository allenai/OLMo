# python scripts/scaling/compute_vs_error_analysis.py --moving_avg 5 --skip_perc 0.1 -o step1_compute_error.pdf --vary flops --which_step step1 --do_average
# python scripts/scaling/compute_vs_error_analysis.py --moving_avg 5 --skip_perc 0.1 -o step2_compute_error.pdf --vary flops --which_step step2 --do_average
# python scripts/scaling/compute_vs_error_analysis.py --moving_avg 5 --skip_perc 0.1 -o stacked_compute_error.pdf --vary flops --which_step stacked --do_average

# python scripts/scaling/compute_vs_error_analysis.py --moving_avg 5 --skip_perc 0.1 -o stacked_N_error_all.pdf --vary N --which_step stacked
# python scripts/scaling/compute_vs_error_analysis.py --moving_avg 5 --skip_perc 0.1 -o stacked_xC_error_all.pdf --vary xC --which_step stacked

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from step1 import fit_step1, predict_step1
from step2 import fit_step2, predict_step2

from olmo.scaling.scaling_laws.fitting_functions import chinchilla_n_d_fit, sigmoid
from olmo.scaling.scaling_laws.utils import (
    FinalConfig,
    get_step1_data_by_name,
    get_step2_data_by_name,
    tasks,
)

MODELS = ["190M", "370M", "760M", "1B"]
CHINCHILLA_MULTIPLIERS = [1, 2, 5, 10]

COLOR_MAP = {"190M": "darkred", "370M": "darkorange", "760M": "darkgreen", "1B": "teal"}

TARGET_COLOR = "darkviolet"

MODEL_FLOPS = {
    "190M": 1903391232,
    "370M": 3443922944,
    "600M": 5180751744,
    "760M": 6373843968,
    "1B": 10109071360,
    "3B": 22970355200,
    "7B": 49412071424,
    "13B": 91335915520,
}

MODEL_PARAMS = {
    "190M": 190354176,
    "370M": 371262464,
    "600M": 597382464,
    "760M": 758220288,
    "1B": 1279395840,
    "3B": 3169537280,
    "7B": 6887575552,
    "13B": 13202396160,
}

TARGET_CONFIG = {
    "paths": ["scripts/scaling/data/peteish-moreeval/peteish7_eval_anneal.csv"],
    "mode": "eval",
    "n": 6887575552,
    "label": "7B-4T",
    "color": "darkviolet",
}


TASKS = [
    "mmlu_avg_test_5shot",
    "hellaswag_val_5shot",
    "arc_challenge_test_5shot",
    "arc_easy_test_5shot",
    "piqa_val_5shot",
    "csqa_val_5shot",
    "socialiqa_val_5shot",
    "openbookqa_test_5shot",
]


def predict_stacked(configs, data_by_name, step1_coefficients, step2_coefficients):
    dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
    dmax = 1.5 * max([max(data["ds"]) for data in data_by_name.values()])

    unsigned_rel_errors = []
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "ds": data["ds"],
            "xs": [
                sigmoid(chinchilla_n_d_fit([n, d], step1_coefficients), *step2_coefficients)
                for n, d in zip(data["ns"], data["ds"])
            ],
        }

        if config.mode == "eval":
            for d, e_y, e_y_pred in zip(data["ds"], data["xs"], predicted_data_by_name[name]["xs"]):
                rel_error = (e_y_pred - e_y) / e_y
        else:
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["xs"]):
                rel_error_t = (y_pred - y) / y
                unsigned_rel_errors.append(np.abs(rel_error_t))

        ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        ns = [data["ns"][0]] * len(ds)
        plotted_predicted_data_by_name[name] = {
            "ds": ds,
            "ys": [
                sigmoid(chinchilla_n_d_fit([n, d], step1_coefficients), *step2_coefficients)
                for n, d in zip(ns, ds)
            ],
        }

    return predicted_data_by_name, plotted_predicted_data_by_name, (e_y, e_y_pred, rel_error), unsigned_rel_errors


def run_all_steps(configs, moving_avg=1, skip_perc=0.0, which_error="pred_error"):
    step1_fitting_error = 0.0
    step2_fitting_error = 0.0
    stacked_fitting_error = 0.0

    step1_pred_error = 0.0
    step2_pred_error = 0.0
    stacked_pred_error = 0.0

    output = {}

    for task_name in TASKS:
        step1_data_by_name = get_step1_data_by_name(configs, task_name, y_metric="rc_bpb", moving_avg=moving_avg)

        step1_coefficients, cov = fit_step1(step1_data_by_name, y_metric="rc_bpb")

        a, b, (y, y_pred, step1_rel_error), step1_unsigned_rel_errors = predict_step1(
            configs, step1_data_by_name, step1_coefficients, y_metric="rc_bpb"
        )

        step1_avg_unsigned_rel_error = np.mean(step1_unsigned_rel_errors)
        step1_fitting_error += step1_avg_unsigned_rel_error

        step2_data_by_name = get_step2_data_by_name(
            configs,
            task_name,
            x_metric="rc_bpb",
            y_metric="rc_acc",
            moving_avg=moving_avg,
            skip_perc=skip_perc,
        )

        step2_coefficients, cov = fit_step2(step2_data_by_name, task_name, "rc_acc")

        a, b, (y, y_pred, step2_rel_error, _), step2_unsigned_rel_errors = predict_step2(
            configs, step2_data_by_name, step2_coefficients, cov, y_metric="rc_acc"
        )

        step2_avg_unsigned_rel_error = np.mean(step2_unsigned_rel_errors)
        step2_fitting_error += step2_avg_unsigned_rel_error


        step1_data_by_name = get_step1_data_by_name(configs, task_name, y_metric="rc_acc", moving_avg=args.moving_avg)

        a, b, (y, y_pred, stacked_rel_error), stacked_unsigned_rel_errors = predict_stacked(
            configs, step1_data_by_name, step1_coefficients, step2_coefficients
        )

        # if "hellaswag" in task_name:
        #     print(task_name, step1_rel_error, step2_rel_error, stacked_rel_error)

        stacked_avg_unsigned_rel_error = np.mean(stacked_unsigned_rel_errors)
        stacked_fitting_error += stacked_avg_unsigned_rel_error

        if which_error == "pred_error":
            output[task_name] = {
                "step1": np.abs(step1_rel_error),
                "step2": np.abs(step2_rel_error),
                "stacked": np.abs(stacked_rel_error),
            }
        else:
            output[task_name] = {
                "step1": np.abs(step1_avg_unsigned_rel_error),
                "step2": np.abs(step2_avg_unsigned_rel_error),
                "stacked": np.abs(stacked_avg_unsigned_rel_error),
            }

    return output


def plot_vary_n(N_df, output_path, which_step):
    FONTSIZE = 11
    sns.set_style("whitegrid")
    num_tasks = len(TASKS)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2.5 * num_rows), squeeze=False)

    for i, task_name in enumerate(N_df.columns):
        ax = axes[i // num_cols][i % num_cols]
        task_df = N_df[task_name]

        if which_step == "all":
            colors = {"step1": "lightgrey", "step2": "grey", "stacked": "darkgreen"}
            for step_ in ["step1", "step2", "stacked"]:
                ax.plot(
                    [MODEL_PARAMS[model] for model in task_df.index],
                    [x[step_] for x in task_df.values],
                    color=colors[step_],
                    linestyle="--",
                    linewidth=1.5,
                    label=step_ if step_ != "stacked" else "chained",
                )
            ax.set_ylabel(f"Prediction Error".title())
            ax.legend(loc="upper right", ncols=1, fontsize=8)

        else:
            ax.plot(
                [MODEL_PARAMS[model] for model in task_df.index],
                [x[which_step] for x in task_df.values],
                color="grey",
                linestyle="--",
                linewidth=1.5,
            )
            ax.set_ylabel(f"{which_step} prediction Error".title())

        ax.set_ylim([0, 1.0])
        # ax.set_xscale("log")
        ax.set_xlabel("(Upto) Model Size (N)")
        
        ax.set_title(f"{tasks[task_name].display_name}", fontsize=FONTSIZE, fontweight="bold")
        # ax.set_xticks([MODEL_FLOPS[model] for model in task_df.index], task_df.index)

    fig.tight_layout()

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    for i in range(num_cols):
        for j in range(num_rows):
            if j != num_rows - 1:
                axes[j][i].set_xlabel("")
            if i != 0:
                axes[j][i].set_ylabel("")

            axes[j][i].legend().remove()

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=10,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1.07),
        handletextpad=0.1,
        columnspacing=0.7,
    )

    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def run_predictions_vary_n(args, which_step="stacked"):
    output_per_N = {}

    configs = {TARGET_CONFIG["label"]: TARGET_CONFIG}

    for N in MODELS:
        paths = [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv" for mult in CHINCHILLA_MULTIPLIERS]
        configs[N] = {"paths": paths, "mode": "train", "n": MODEL_PARAMS[N], "label": N, "color": COLOR_MAP[N]}

        final_configs = {name: FinalConfig(**config) for name, config in configs.items()}

        if N == "1B":
            print(final_configs)

        output = run_all_steps(final_configs, moving_avg=args.moving_avg, skip_perc=args.skip_perc)

        # output_per_N[N] = {key: val[which_step] for key, val in output.items()}
        output_per_N[N] = output

    N_df = pd.DataFrame.from_dict(output_per_N).transpose()
    plot_vary_n(N_df, args.output_path, which_step)


def plot_vary_xC(xC_df, output_path, which_step):
    FONTSIZE = 11
    sns.set_style("whitegrid")
    num_tasks = len(TASKS)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2.5 * num_rows), squeeze=False)

    for i, task_name in enumerate(xC_df.columns):
        ax = axes[i // num_cols][i % num_cols]
        task_df = xC_df[task_name]

        if which_step == "all":
            colors = {"step1": "lightgrey", "step2": "grey", "stacked": "darkgreen"}
            for step_ in ["step1", "step2", "stacked"]:
                ax.plot(
                    CHINCHILLA_MULTIPLIERS,
                    [x[step_] for x in task_df.values],
                    color=colors[step_],
                    linestyle="--",
                    linewidth=1.5,
                    label=step_ if step_ != "stacked" else "chained",
                )
            ax.set_ylabel(f"Prediction Error".title())
            ax.legend(loc="upper right", ncols=1, fontsize=8)
        else:
            ax.plot(
                CHINCHILLA_MULTIPLIERS,
                [x[which_step] for x in task_df.values],
                color="grey",
                linestyle="--",
                linewidth=1.5,
            )
            ax.set_ylabel(f"{which_step} prediction Error".title())

        ax.set_ylim([0, 1.0])
        ax.set_xticks([0, 1, 2, 5, 10])
        ax.set_xlabel("Chinchilla Multiplier (xC)")
        ax.set_title(f"{tasks[task_name].display_name}", fontsize=FONTSIZE, fontweight="bold")
        # ax.set_xticks([MODEL_FLOPS[model] for model in task_df.index], task_df.index)

    fig.tight_layout()

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    for i in range(num_cols):
        for j in range(num_rows):
            if j != num_rows - 1:
                axes[j][i].set_xlabel("")
            if i != 0:
                axes[j][i].set_ylabel("")

            axes[j][i].legend().remove()

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=10,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1.07),
        handletextpad=0.1,
        columnspacing=0.7,
    )

    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def run_predictions_vary_xC(args, which_step="stacked"):
    output_per_xC = {}

    configs = {TARGET_CONFIG["label"]: TARGET_CONFIG}

    for mult in CHINCHILLA_MULTIPLIERS:
        for N in MODELS:
            if N in configs:
                configs[N]["paths"] += [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv"]
            else:
                paths = [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv"]
                configs[N] = {
                    "paths": paths,
                    "mode": "train",
                    "n": MODEL_PARAMS[N],
                    "label": N,
                    "color": COLOR_MAP[N],
                }

        final_configs = {name: FinalConfig(**config) for name, config in configs.items()}

        output = run_all_steps(final_configs, moving_avg=args.moving_avg, skip_perc=args.skip_perc)

        # output_per_xC[mult] = {key: val[which_step] for key, val in output.items()}
        output_per_xC[mult] = output

    xC_df = pd.DataFrame.from_dict(output_per_xC).transpose()
    plot_vary_xC(xC_df, args.output_path, which_step)


def plot_vary_flops(flops_df, output_path, which_step, do_average=False):
    FONTSIZE = 11
    sns.set_style("whitegrid")

    if not do_average:
        num_tasks = len(TASKS)
        num_cols = min(4, num_tasks)
        num_rows = (num_tasks + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2.5 * num_rows), squeeze=False)

        for i, task_name in enumerate(flops_df.columns):
            ax = axes[i // num_cols][i % num_cols]
            task_df = flops_df[task_name]

            if which_step == "all":
                colors = {"step1": "lightgrey", "step2": "grey", "stacked": "darkgreen"}
                for step_ in ["step1", "step2", "stacked"]:
                    ax.plot(
                        task_df.index,
                        [x[step_] for x in task_df.values],
                        color=colors[step_],
                        linestyle="--",
                        linewidth=1.5,
                        label=step_ if step_ != "stacked" else "chained",
                    )
                ax.set_ylabel(f"Prediction Error".title())
                ax.legend(loc="upper right", ncols=1, fontsize=8)
            else:
                ax.plot(
                    task_df.index,
                    [x[which_step] for x in task_df.values],
                    color="grey",
                    linestyle="--",
                    linewidth=1.5,
                )
                ax.set_ylabel(f"{which_step} prediction Error".title())

            ax.set_ylim([0, 1.0])
            # ax.set_xticks([0, 1, 2, 5, 10])
            ax.set_xscale("log")
            ax.set_xlabel("Total flops used for prediction")
            ax.set_title(f"{tasks[task_name].display_name}", fontsize=FONTSIZE, fontweight="bold")
            # ax.set_xticks([MODEL_FLOPS[model] for model in task_df.index], task_df.index)

    else:
        flops_df["average"] = flops_df.mean(axis=1)

        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(
            flops_df.index,
            flops_df["average"].values,
            color="grey",
            linestyle="--",
            linewidth=1.5,
        )

        # ax.set_ylim([0, 1.0])
        ax.set_xscale("log")
        ax.set_xlabel("Total flops used for prediction")
        ax.set_ylabel("Prediction Error")
        ax.set_title(f"Average {which_step} prediction error".title(), fontsize=FONTSIZE, fontweight="bold")

    fig.tight_layout()

    if not do_average:
        handles, labels = axes[-1][-1].get_legend_handles_labels()
        for i in range(num_cols):
            for j in range(num_rows):
                if j != num_rows - 1:
                    axes[j][i].set_xlabel("")
                if i != 0:
                    axes[j][i].set_ylabel("")

                axes[j][i].legend().remove()

        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=10,
            fontsize=FONTSIZE,
            bbox_to_anchor=(0.5, 1.07),
            handletextpad=0.1,
            columnspacing=0.7,
        )

        for handle in legend.legend_handles:
            handle.set_alpha(1.0)


    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def run_predictions_vary_flops(args, which_step="stacked", do_average=False):
    output_per_flops = {}

    configs = {TARGET_CONFIG["label"]: TARGET_CONFIG}

    all_flops = {}
    for N in MODELS:
        for mult in CHINCHILLA_MULTIPLIERS:
            all_flops[f"{N}-{mult}xC"] = MODEL_FLOPS[N] * (MODEL_PARAMS[N] * 20 * mult)

    sorted_flops = sorted(all_flops.items(), key=lambda item: item[1])

    cum_flops = 0

    for run_name, num_flops in sorted_flops:
        cum_flops = num_flops
        N, xC = run_name.replace("xC", "").split("-")
        mult = int(xC)
        if N in configs:
            configs[N]["paths"] += [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv"]
        else:
            paths = [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv"]
            configs[N] = {"paths": paths, "mode": "train", "n": MODEL_PARAMS[N], "label": N, "color": COLOR_MAP[N]}

        final_configs = {name: FinalConfig(**config) for name, config in configs.items()}

        output = run_all_steps(final_configs, moving_avg=args.moving_avg, skip_perc=args.skip_perc)

        # output_per_flops[cum_flops] = {key: val[which_step] for key, val in output.items()}
        output_per_flops[cum_flops] = output

        print(f"{cum_flops:.3e}", output["mmlu_avg_test_5shot"])

    flops_df = pd.DataFrame.from_dict(output_per_flops).transpose()

    plot_vary_flops(flops_df, args.output_path, which_step=which_step, do_average=do_average)


def run_predictions_vary_flops_step(args, which_step="stacked", do_average=False):
    output_per_flops = {}

    configs = {TARGET_CONFIG["label"]: TARGET_CONFIG}

    all_flops = {}
    for N in MODELS:
        for mult in CHINCHILLA_MULTIPLIERS:
            all_flops[f"{N}-{mult}xC"] = MODEL_FLOPS[N] * (MODEL_PARAMS[N] * 20 * mult)

    # sorted_flops = sorted(all_flops.items(), key=lambda item: item[1])
    sorted_flops = all_flops.items()

    cum_flops = 0

    for run_name, num_flops in sorted_flops:
        cum_flops = num_flops
        N, xC = run_name.replace("xC", "").split("-")
        mult = int(xC)
        if N in configs:
            configs[N]["paths"] += [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv"]
        else:
            paths = [f"scripts/scaling/data/peteish-moreeval/{N}-{mult}xC.csv"]
            configs[N] = {"paths": paths, "mode": "train", "n": MODEL_PARAMS[N], "label": N, "color": COLOR_MAP[N]}

        final_configs = {name: FinalConfig(**config) for name, config in configs.items()}
        print(final_configs)

        output = run_all_steps(final_configs, moving_avg=args.moving_avg, skip_perc=args.skip_perc)

        # output_per_flops[cum_flops] = {key: val[which_step] for key, val in output.items()}
        output_per_flops[cum_flops] = output

        print(f"{cum_flops:.3e}", output["mmlu_avg_test_5shot"])

    flops_df = pd.DataFrame.from_dict(output_per_flops).transpose()

    plot_vary_flops(flops_df, args.output_path, which_step=which_step, do_average=do_average)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument(
        "--skip_perc",
        type=float,
        default=0.0,
        help="Percentage of intermediate ckpts to skip from the beginning (for loss to accuracy fitting)",
    )
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")

    parser.add_argument("--vary", type=str, default="flops")
    parser.add_argument("--which_step", type=str, default="stacked")
    parser.add_argument("--do_average", action="store_true", default=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.vary == "N":
        run_predictions_vary_n(args, args.which_step)
    elif args.vary == "xC":
        run_predictions_vary_xC(args, args.which_step)
    elif args.vary == "flops":
        run_predictions_vary_flops(args, args.which_step, args.do_average)
    else:
        raise ValueError(f"vary = {args.vary} not recognized. Use one of [N, xC]")
