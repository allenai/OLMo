# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -o figure/peteish-moreeval/chained_main.pdf -n 6887575552 -d 3945065873408 -t 7B-4T --skip_perc 0.1 --moving_avg 5
# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -o figure/peteish-moreeval/chained_main.pdf -n 13202396160 -d 5000088518656 -t 13B-5T --skip_perc 0.1 --moving_avg 5
# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -o figure/peteish-moreeval/chained_c4_main.pdf -n 6887575552 -d 3945065873408 -t 7B-4T --skip_perc 0.1 --moving_avg 5 --x_metric c4
# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -o figure/peteish-moreeval/chained_c4_main.pdf -n 13202396160 -d 5000088518656 -t 13B-5T --skip_perc 0.1 --moving_avg 5 --x_metric c4
# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2_mc_7B.json -o figure/peteish-moreeval/chained_mc_7B_main.pdf -y mc_acc -n 6887575552 -d 3945065873408 -t 7B-4T --moving_avg 5
# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2_mc_13B.json -o figure/peteish-moreeval/chained_mc_13B_main.pdf -y mc_acc -n 13202396160 -d 5000088518656 -t 13B-5T --moving_avg 5
# python scripts/scaling/predict.py -k v2_main -c scripts/scaling/final.json --step2-config-path scripts/scaling/step2.json -o figure/peteish-moreeval/chained_taskce_main.pdf -n 6887575552 -d 3945065873408 -t 7B-4T --skip_perc 0.5 --x_metric rc_soft_log  --use_log_sigmoid

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from step1 import fit_step1
from step2 import fit_step2
from step2_mc import fit_step2 as fit_step2_mc

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
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

MARKERS = {"0.5xC": "D", "1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}
FONTSIZE = 9


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


def predict_chained(data_by_name, step1_coefficients, step2_coefficients, use_log_sigmoid=False):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
    dmax = 1.5 * max([max(data["ds"]) for data in data_by_name.values()])

    fit_fn = log_sigmoid if use_log_sigmoid else sigmoid

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "ds": data["ds"],
            "ys": [
                fit_fn(chinchilla_n_d_fit([n, d], step1_coefficients), *step2_coefficients)
                for n, d in zip(data["ns"], data["ds"])
            ],
        }
        ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        ns = [data["ns"][0]] * len(ds)
        plotted_predicted_data_by_name[name] = {
            "ds": ds,
            "ys": [
                fit_fn(chinchilla_n_d_fit([n, d], step1_coefficients), *step2_coefficients)
                for n, d in zip(ns, ds)
            ],
        }

        if data["mode"] == "eval":
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error)


def str_chained_fit(step1_coefficients, step2_coefficients, use_log_sigmoid=False):
    a, b, alpha, beta, E = step1_coefficients
    A, B = np.exp(a), np.exp(b)
    if use_log_sigmoid:
        a, x0, k = step2_coefficients
        return f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}; Acc(L) = 1 - {-a:.2f} * log(1 - 1/(1 + e^(-{k:.2f}(L - {x0:.2f})))"
    else:
        a, x0, k, b = step2_coefficients
        return f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}; Acc(L) = {a:.2f} / (1 + e^(-{k:.2f}(L - {x0:.2f}))) + {b:.2f}"


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
            data["ds"],
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

        for i, (d, y, l) in enumerate(zip(data["ds"], data["xs"], data["ls"])):
            ax.scatter(
                d,
                y,
                color=config.color,
                marker=MARKERS[l] if config.mode == "train" else "o",
                s=50 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )

        for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["ys"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                pass
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

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
    ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    ax.set_title(
        f"{tasks[task_name].display_name}",
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
        predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_chained(
            single_step_data_by_name, step1_coefficients, step2_coefficients, args.use_log_sigmoid
        )

        plot_chained(
            configs,
            single_step_data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_chained_fit(step1_coefficients, step2_coefficients, args.use_log_sigmoid),
            axes[r // num_cols][r % num_cols],
        )

        # make predictions
        pred_loss = chinchilla_n_d_fit([args.n, args.d], step1_coefficients)
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
