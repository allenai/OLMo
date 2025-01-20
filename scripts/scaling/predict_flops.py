# python scripts/scaling/predict_flops.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -o figure/peteish-moreeval/chained_flops_main.pdf -n 6887575552 -d 3945065873408 -t 7B-4T --skip_perc 0.1 --moving_avg 5

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from step1_flops import fit_step1
from step2 import fit_step2
from step2_mc import fit_step2 as fit_step2_mc

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_flops_fit,
    log_sigmoid,
    sigmoid,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_step2_data_by_name,
    get_task_sets,
    tasks,
)

MARKERS = ["s", "P", "p", "*", "o"]
FONTSIZE = 9

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--keys", nargs="+", default=[], help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument(
        "-x", "--x_metric", default="rc_bpb", choices=["rc_bpb", "c4", "rc_soft_log"], help="Metric as input"
    )
    parser.add_argument(
        "-y", "--y_metric", default="rc_acc", choices=["rc_acc", "mc_acc"], help="Metric to predict"
    )
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument(
        "--skip_perc",
        type=float,
        default=0.0,
        help="Percentage of intermediate ckpts to skip from the beginning (for loss to accuracy fitting)",
    )
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("--step2-config-path", type=str, default=None, help="Path to config file for step2")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    parser.add_argument("-n", "--n", type=int, required=True, help="Model size of the target model")
    parser.add_argument("-d", "--d", type=int, required=True, help="Data size of the target model")
    parser.add_argument(
        "-t", "--target-name", type=str, default=None, help="Path to the csv file of the target model"
    )
    parser.add_argument("--use_log_sigmoid", action="store_true", help="Use log sigmoid for fitting")
    args = parser.parse_args()

    args.keys = get_task_sets(args.keys)

    return args


def predict_chained_flops(data_by_name, step1_coefficients, step2_coefficients):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    fmin = 0.8 * min([min(data["fs"]) for data in data_by_name.values()])
    fmax = 1.5 * max([max(data["fs"]) for data in data_by_name.values()])

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "fs": data["fs"],
            "ys": [
                sigmoid(chinchilla_flops_fit(f, step1_coefficients), *step2_coefficients)
                for f in data["fs"]
            ],
        }
        fs = np.exp(np.linspace(np.log(fmin), np.log(fmax), 100))
        plotted_predicted_data_by_name[name] = {
            "fs": fs,
            "ys": [
                sigmoid(chinchilla_flops_fit(f, step1_coefficients), *step2_coefficients)
                for f in fs
            ],
        }

        if data["mode"] == "eval":
            predicted_data = predicted_data_by_name[name]
            for f, y, y_pred in zip(data["fs"], data["ys"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error)


def str_chained_fit(step1_coefficients, step2_coefficients):
    a, alpha, E = step1_coefficients
    A = np.exp(a)
    a, x0, k, b = step2_coefficients
    return f"L(F) = {A:.2f} / F^{alpha:.2f} + {E:.2f}; Acc(L) = {a:.2f} / (1 + e^(-{k:.2f}(L - {x0:.2f}))) + {b:.2f}"


def plot_chained(
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
            data["fs"],
            data["ys"],
            color=config.color,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"{config.label} (fitted)" if config.mode == "train" else None,
        )

    # plot the actual and predicted data
    num_eval_annotation = 0
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        for i, (f, y) in enumerate(zip(data["fs"], data["xs"])):
            ax.scatter(
                f,
                y,
                color=config.color,
                marker=MARKERS[i] if config.mode == "train" else "o",
                s=50 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )

        for f, y, y_pred in zip(data["fs"], data["xs"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                pass
            else:
                ax.scatter(
                    f,
                    y_pred,
                    color=config.color,
                    marker="x",
                    s=20,
                    label=f"{config.label} (predicted)",
                )
                ax.annotate(
                    f"{abs(rel_error * 100):.1f}%",
                    (f, y_pred),
                    textcoords="offset points",
                    xytext=(10, -5 + 10 * num_eval_annotation),
                    ha="left",
                    va="bottom",
                    fontsize=FONTSIZE,
                    color=config.color,
                )
                num_eval_annotation += 1

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("FLOPs (C)", fontsize=FONTSIZE)
    ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    display_name = tasks[task_name].display_name if task_name in tasks else task_name
    ax.set_title(
        f"{display_name}",
        fontsize=FONTSIZE,
        fontweight="bold",
    )


def main():
    args = parse_args()
    configs = get_final_configs(args.config_path)
    if args.step2_config_path:
        step2_configs = get_final_configs(args.step2_config_path)
    else:
        step2_configs = configs

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.75 * num_cols, 2.25 * num_rows), squeeze=False)

    results = {}
    results_str = "Task Name | Prediction | Actual | Rel Error"

    for r, task_name in enumerate(args.keys):
        step1_data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric=args.x_metric, moving_avg=args.moving_avg
        )
        step2_data_by_name = get_step2_data_by_name(
            step2_configs,
            task_name,
            x_metric=args.x_metric,
            y_metric=args.y_metric,
            moving_avg=args.moving_avg,
            skip_perc=args.skip_perc,
        )
        single_step_data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric="rc_acc", moving_avg=args.moving_avg
        )

        # fit the parameters
        step1_coefficients, _ = fit_step1(step1_data_by_name, y_metric=args.x_metric)
        if args.y_metric == "rc_acc":
            step2_coefficients, _ = fit_step2(step2_data_by_name, task_name, args.y_metric, args.use_log_sigmoid)
        elif args.y_metric == "mc_acc":
            step2_coefficients, _ = fit_step2_mc(
                step2_data_by_name, task_name, args.y_metric, args.use_log_sigmoid
            )
        else:
            raise ValueError(f"Invalid y_metric: {args.y_metric})")

        # make predictions
        predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_chained_flops(
            single_step_data_by_name, step1_coefficients, step2_coefficients
        )

        plot_chained(
            configs,
            single_step_data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_chained_fit(step1_coefficients, step2_coefficients),
            axes[r // num_cols][r % num_cols],
        )

        # make predictions
        if args.n == 6887575552:
            f = MODEL_FLOPS["7B"]
        elif args.n == 13202396160:
            f = MODEL_FLOPS["13B"]

        pred_loss = chinchilla_flops_fit(f * args.d, step1_coefficients)
        fit_fn = log_sigmoid if args.use_log_sigmoid else sigmoid
        pred_acc = fit_fn(pred_loss, *step2_coefficients)
        if args.target_name:
            data = step2_data_by_name[args.target_name]
            actual_acc = data["ys"][-1]
            rel_error = np.abs(pred_acc - actual_acc) / actual_acc
            results[task_name] = {"Actual": y, "Pred": y_pred, "Rel Error": rel_error}
            results_str += (
                f"\n{task_name} | {pred_acc * 100:.1f} | {actual_acc * 100:.1f} | {rel_error * 100:.1f}%"
            )
        else:
            results_str += f"\n{task_name} | {pred_acc * 100:.1f} | - | -"

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

    if args.output_path:
        fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
        df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename({"index": "Task"}, axis=1)
        df.to_csv(args.output_path.replace(".pdf", ".csv"), index=False)

    print(results_str)

    return df


if __name__ == "__main__":
    main()
