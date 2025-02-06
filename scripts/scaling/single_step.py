# python scripts/scaling/single_step.py -k v2_main -c scripts/scaling/final.json -o figure/peteish-moreeval/single_step_main.pdf --moving_avg 5

import argparse
import warnings
from scipy.optimize import OptimizeWarning

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    combined_fit,
    get_coefficients_huber,
    grad_combined_fit,
    sigmoid,
    # grad_sigmoid,
    sigmoid_fit,
    grad_sigmoid_fit,
    exponential_fit,
    # grad_exponential_fit,
    combined_flops_sigmoid_fit,
    grad_combined_flops_sigmoid_fit
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

MARKERS = {"0.5xC": "D", "1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}
FONTSIZE = 9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    args = parser.parse_args()

    args.keys = get_task_sets(args.keys)

    return args


def fit_single_step(data_by_name, task_name, use_flops=False):
    train_nds, train_fs, train_ys = [], [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_nds += [[n, d] for n, d in zip(data["ns"], data["ds"])]
            train_fs += data["fs"]
            train_ys += data["xs"]

    # p0 = [3.0, 5.0, 0.2, 0.3, 0.0, tasks[task_name].task_minimum - 1.0, 1.0]
    # p0 = [3.0, 5.0, 0.2, 0.3, 0.0, -1, 1.0]
    # p0 = [3.0, 5.0, 0.2, 0.3, 0.0, -1, 1.0]
    p0s = [
        [3.0, 5.0, 0.2, 0.3, 0.0, -1, 1.0],
    ]

    if use_flops:
        p0s = [
            # Default fitting setup
            # [10.0, 0.4, 0.5, -0.7, 0.6, 7.0, 1]

            # for ian's project (original full 5-param fits?)
            # [3.0, 5.0, 0.2, 0.3, 0.0, -1, 1.0],
            # [3.0, 5.0, 0.2, 0.3, 0.0, 1, 1.0],

            # ian sigmoid fits (copied)
            [10.0, 0.4, 0.5, -0.7, 0.6, 7.0, 1],
            [10.0, 0.4, 0.5, -0.7, 1.4, 3.2, 1.3], # piqa total_prob_per_char
            [10.0, 0.4, 0.5, -2.7, 1.4, 3.2, 1.3],
            [10.0, 0.4, 0.5, -0.09, 0.6, 6.5, 0.1], # arc_easy correct_prob
            [10.0, 0.4, 0.5, -0.64, -0.41, 12.7, 1.05], # csqa
            [10.0, 0.4, 0.5, -2.15771768e-02, 1.39273020e+00, -5.85129498e+01, 4.54084320e-01],
            [10.0, 0.4, 0.5, -1.60006552e+00, 1.36001561e+00, -4.95252012e+02, 4.10217043e-01]
        ]
    # bounds = [(0, 10), (0, 10), (0, 1), (0, 1), (-10, 10), (-0.9999, 0), (0, 1)]
    # bounds = [(0, None), (0, None), (0, None), (0, None), (None, None), (-1.0, 0), (0, 1)]
    bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    # bounds = [(None, None), (None, None), (None, None), (None, None)]
    
    try_idx = 0
    while try_idx < len(p0s):
        try_p0 = p0s[try_idx]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", OptimizeWarning)

            if use_flops:
                x, y = train_fs, train_ys
                fit_f, fit_grad = combined_flops_sigmoid_fit, grad_combined_flops_sigmoid_fit
            else:
                x, y = train_nds, train_ys
                fit_f, fit_grad = combined_fit, grad_combined_fit
                
            try:
                coefficients, cov = get_coefficients_huber(
                    x,
                    y,
                    fit_f,
                    fit_grad,
                    p0=try_p0,
                    bounds=bounds,
                    max_iter=1000000,
                    disp=False,
                    return_cov=True,
                )

                if cov.sum() > 1_000_000:
                    # print(cov.sum())
                    warnings.warn(f"Cov is {cov.sum()}", OptimizeWarning) 
                
                # Check if an OptimizeWarning was raised
                if not any(issubclass(warning.category, OptimizeWarning) for warning in w):
                    # print(task_name, coefficients)
                    return coefficients
            except RuntimeError as e:
                print(f'Optimization error for step 2 on {task_name}')
                pass
            
            try_idx += 1

    print(f'Failed to optimize single step 1 on {task_name}')
    return coefficients


def predict_single_step(data_by_name, coefficients, use_flops=False):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    if use_flops:
        dmin = 0.8 * min([min(data["fs"]) for data in data_by_name.values()])
        dmax = 1.5 * max([max(data["fs"]) for data in data_by_name.values()])
    else:
        dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
        dmax = 1.5 * max([max(data["ds"]) for data in data_by_name.values()])

    for name, data in data_by_name.items():
        if use_flops:
            predicted_data_by_name[name] = {
                "fs": data["fs"],
                "ys": [combined_flops_sigmoid_fit(f, coefficients) for f in data["fs"]],
            }
        else:
            predicted_data_by_name[name] = {
                "ds": data["ds"],
                "ys": [combined_fit([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
            }
        ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
        ns = [data["ns"][0]] * len(ds)
        if use_flops:
            plotted_predicted_data_by_name[name] = {
                "ds": ds,
                "ys": [combined_flops_sigmoid_fit(f, coefficients) for f in ds],
            }
        else:
            plotted_predicted_data_by_name[name] = {
                "ds": ds,
                "ys": [combined_fit([n, d], coefficients) for n, d in zip(ns, ds)],
            }

        if data["mode"] == "eval":
            predicted_data = predicted_data_by_name[name]
            for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["ys"]):
                rel_error = (y_pred - y) / y

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error)


def str_combined_fit(coefficients):
    a, b, alpha, beta, E, p, q = coefficients
    A, B = np.exp(a), np.exp(b)
    return (
        f"Acc(N, D) = {p:.2f} / (1 + e^-({A:.2f} / N^{alpha:.2f} \n + {B:.2f} / D^{beta:.2f} + {E:.2f})) + {q:.2f}"
    )


def plot_single_step(
    configs,
    data_by_name,
    predicted_data_by_name,
    plotted_predicted_data_by_name,
    task_name,
    fit_str,
    use_flops=False,
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
    unsigned_rel_errors = []
    num_eval_annotation = 0
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        if use_flops:
            xs, ys = data["fs"], data["xs"]
        else:
            xs, ys = data["ds"], data["xs"]

        # for i, (d, y, l) in enumerate(zip(data["ds"], data["xs"], data["ls"])):
        # for i, (d, y) in enumerate(zip(data["ds"], data["xs"])):
        for i, (d, y) in enumerate(zip(xs, ys)):
            ax.scatter(
                d,
                y,
                color=config.color,
                # marker=MARKERS[l] if config.mode == "train" else "o",
                marker=MARKERS['5xC'] if config.mode == "train" else "o",
                s=50 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )

        # for d, y, y_pred in zip(data["ds"], data["xs"], predicted_data["ys"]):
        for d, y, y_pred in zip(xs, ys, predicted_data["ys"]):
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
    avg_unsigned_rel_error = np.mean(unsigned_rel_errors)

    ax.set_xscale("log")
    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("Tokens (D)", fontsize=FONTSIZE)
    ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    display_name = tasks[task_name].display_name if task_name in tasks else task_name
    ax.set_title(
        f"{display_name} ({avg_unsigned_rel_error * 100:.2f}%)",
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
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.75 * num_cols, 2.25 * num_rows), squeeze=False)

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_step1_data_by_name(configs, task_name, y_metric="rc_acc", moving_avg=args.moving_avg)

        for name, data in data_by_name.items():
            data["ys"] = data["xs"]  # to deal with refactor

        # fit the parameters
        coefficients = fit_single_step(data_by_name, task_name)

        # make predictions
        predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error) = predict_single_step(
            data_by_name, coefficients
        )
        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)}"

        plot_single_step(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_combined_fit(coefficients),
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

    print(results)


if __name__ == "__main__":
    main()
