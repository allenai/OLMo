# python scripts/scaling/step1.py -k v2_main -c scripts/scaling/final.json -o figure/peteish-moreeval/step1_main.pdf --moving_avg 5
# python scripts/scaling/step1.py -k mmlu_avg_test_5shot -c scripts/scaling/final.json -o figure/peteish-moreeval/step1_c4_main.pdf --y_metric c4 --moving_avg 5
# python scripts/scaling/step1.py -k v2_main -c scripts/scaling/final.json -o figure/peteish-moreeval/step1_acc_main.pdf --y_metric rc_acc
# python scripts/scaling/step1.py -k v2_main -c scripts/scaling/final.json -o figure/peteish-moreeval/step1_taskce_main.pdf -y rc_soft_log

import argparse
import pandas as pd
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
    chinchilla_n_d_negated_fit,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
    grad_chinchilla_n_d_negated_fit,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

MARKERS = ["s", "P", "p", "*", "o"]
FONTSIZE = 9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument(
        "-y",
        "--y_metric",
        default="rc_bpb",
        choices=["rc_bpb", "rc_acc", "c4", "rc_soft_log"],
        help="Metric to predict",
    )
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=False, help="Path to write output figure")
    args = parser.parse_args()

    if not args.keys:
        args.keys = ["main"]
    args.keys = get_task_sets(args.keys)

    return args


def fit_step1(data_by_name, y_metric):
    train_nds, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
            train_ys += data["xs"]

    bounds: List[Tuple[Any, Any]]

    if y_metric == "rc_bpb" or y_metric == "c4" or y_metric == "rc_soft_log":
        p0 = [3.0, 6.0, 0.1, 0.2, 1.0]
        bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]
        # p0 = [3.0, 6.0, 0.25, 0.3, 1.0]
        # bounds = [(0, None), (0, None), (0.25, 0.4), (0.19, 0.31), (0, None)] # moving_avg=1
        # # bounds = [(0, None), (0, None), (0, 0.3), (0.25, 0.45), (0, None)] # moving_avg=10
        # # bounds = [(0, None), (0, None), (0.15, 0.4), (0.3, 0.33), (0, None)]
        coefficients, cov = get_coefficients_huber(
            train_nds,
            train_ys,
            chinchilla_n_d_fit,
            grad_chinchilla_n_d_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
            return_cov=True,
        )
    elif y_metric == "rc_acc":
        p0 = [2.0, 2.0, 0.2, 0.2, 1.0]
        bounds = [(0, None), (0, None), (0, None), (None, None), (None, None)]
        coefficients, cov = get_coefficients_huber(
            train_nds,
            train_ys,
            chinchilla_n_d_negated_fit,
            grad_chinchilla_n_d_negated_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
            return_cov=True,
        )
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    return coefficients, cov


def scale_data_by_name(data_by_name):
    dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
    dmax = 1.2 * max([max(data["ds"]) for data in data_by_name.values()])

    for name, data in data_by_name.items():
        data["ds"] = ((np.array(data["ds"]) - dmin) / (dmax - dmin)).tolist()
        # data["fs"] = np.log(np.array(data["fs"]).astype(float)).tolist()

    return data_by_name


def predict_step1(configs, data_by_name, coefficients, y_metric):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    unsigned_rel_errors = []

    dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
    dmax = 1.5 * max([max(data["ds"]) for data in data_by_name.values()])

    if y_metric == "rc_bpb" or y_metric == "c4" or y_metric == "rc_soft_log":
        func = chinchilla_n_d_fit
    elif y_metric == "rc_acc":
        func = chinchilla_n_d_negated_fit
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    y, y_pred, rel_error = 0.0, 0.0, 0.0

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "ds": data["ds"],
            "xs": [func([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
        }
        ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        ns = [data["ns"][0]] * len(ds)
        plotted_predicted_data_by_name[name] = {
            "ds": ds,
            "xs": [func([n, d], coefficients) for n, d in zip(ns, ds)],
        }

        if configs[name].mode == "eval":
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["xs"]):
                rel_error = (y_pred - y) / y
        else:
            predicted_data = predicted_data_by_name[name]
            for f, y, y_pred in zip(data["fs"], data["xs"], predicted_data["xs"]):
                rel_error_t = (y_pred - y) / y
                unsigned_rel_errors.append(np.abs(rel_error_t))

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error), unsigned_rel_errors


def str_chinchilla_n_d_fit(coefficients):
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)
    return f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}"


def plot_step1(
    configs,
    data_by_name,
    predicted_data_by_name,
    plotted_predicted_data_by_name,
    task_name,
    fit_str,
    y_metric,
    coefficients,
    cov,
    ax=plt.gca(),
):
    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]

        ax.plot(
            data["ds"],
            data["xs"],
            color=config.color,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f'{config.label} (fitted)' if config.mode == "train" else None,
        )

    # plot the actual and predicted data
    unsigned_rel_errors = []
    num_eval_annotation = 0
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        for i, (d, x) in enumerate(zip(data["ds"], data["xs"])):
            ax.scatter(
                d,
                x,
                color=config.color,
                marker=MARKERS[i] if config.mode == "train" else "o",
                s=50 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )

        for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["xs"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                unsigned_rel_errors.append(np.abs(rel_error))
            else:
                ax.scatter(
                    d,
                    y_pred,
                    color=config.color,
                    marker="x",
                    s=20,
                    label=f"{config.label} ({'predicted'})",
                )

                ax.annotate(
                    f"{abs(rel_error) * 100:.1f}%",
                    (d, y_pred),
                    textcoords="offset points",
                    xytext=(10, 1 - 10 * num_eval_annotation)
                    if y_metric == "rc_bpb"
                    else (-3, 5 * (-3 if num_eval_annotation % 2 == 0 else 1)),
                    ha="left",
                    va="bottom",
                    fontsize=FONTSIZE,
                    color=config.color,
                )
                num_eval_annotation += 1
    avg_unsigned_rel_error = np.mean(unsigned_rel_errors)

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
    y_label_name = {
        "rc_bpb": "Task loss",
        "rc_acc": "Task RC accuracy",
        "c4": "C4 loss",
        "rc_soft_log": "TaskCE",
    }[y_metric]
    ax.set_ylabel(y_label_name, fontsize=FONTSIZE)
    ax.set_title(
        f"{tasks[task_name].display_name} (Fitting error: {avg_unsigned_rel_error * 100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )


def main():
    args = parse_args()
    configs = get_final_configs(args.config_path)

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols

    fitting_error = 0

    if args.output_path:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.75 * num_cols, 2.25 * num_rows), squeeze=False)

    results = {}
    results_str = "Task Name | Actual Value | Predicted Value | Relative Error | Fitting Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric=args.y_metric, moving_avg=args.moving_avg
        )

        # fit the parameters
        coefficients, cov = fit_step1(data_by_name, args.y_metric)

        # make predictions
        (
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            (y, y_pred, rel_error),
            unsigned_rel_errors,
        ) = predict_step1(configs, data_by_name, coefficients, y_metric=args.y_metric)

        avg_unsigned_rel_error = np.mean(unsigned_rel_errors)
        fitting_error += avg_unsigned_rel_error

        results[task_name] = {"Actual": y, "Pred": y_pred, "Rel Error": rel_error, "Fit Error": avg_unsigned_rel_error}
        results_str += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)} | {prettify(avg_unsigned_rel_error)}"

        if args.output_path:
            plot_step1(
                configs,
                data_by_name,
                predicted_data_by_name,
                plotted_predicted_data_by_name,
                task_name,
                str_chinchilla_n_d_fit(coefficients),
                args.y_metric,
                coefficients,
                cov,
                axes[i // num_cols][i % num_cols],
            )

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
    if num_tasks > 1:
        legend = fig.legend(handles, labels, loc='upper center',
                            ncol=10, fontsize=FONTSIZE, bbox_to_anchor=(0.5, 1.07),
                            handletextpad=0.3, columnspacing=0.7)
    else:
        legend = fig.legend(handles, labels, loc='upper center',
                            ncol=1, fontsize=FONTSIZE, bbox_to_anchor=(1.3, 0.9),
                            handletextpad=0.1, columnspacing=0.7)

    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    if args.output_path:
        fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
        df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename({"index": "Task"}, axis=1)
        df.to_csv(args.output_path.replace(".pdf", ".csv"), index=False)

    print(results_str)
    print("Total fitting error: ", prettify(fitting_error / num_tasks))


if __name__ == "__main__":
    main()
