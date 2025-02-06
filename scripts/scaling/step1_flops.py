# python scripts/scaling/step1_flops.py -k v2_main -c scripts/scaling/final.json -o figure/peteish-moreeval/step1_flops_main.pdf --moving_avg 5

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_flops_fit,
    chinchilla_flops_negated_fit,
    get_coefficients_huber,
    grad_chinchilla_flops_fit,
    grad_chinchilla_flops_negated_fit,
    chinchilla_flops_2_param_fit,
    grad_chinchilla_flops_2_param_fit,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

MARKERS = ["s", "P", "p", "*", "o"]
FONTSIZE = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
    parser.add_argument(
        "-y", "--y_metric", default="rc_bpb", choices=["rc_bpb", "rc_acc"], help="Metric to predict"
    )
    parser.add_argument("--moving_avg", type=int, default=1, help="Moving average for bpb loss")
    parser.add_argument("-c", "--config-path", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output-path", type=str, required=False, help="Path to write output figure")
    args = parser.parse_args()

    if not args.keys:
        args.keys = ["main"]
    args.keys = get_task_sets(args.keys)

    return args


def fit_step1(data_by_name, y_metric, use_two_param=False):
    train_fs, train_xs = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_fs += data["fs"]
            train_xs += data["xs"]

    if y_metric == "rc_bpb":
        # p0 = [3.0, 0.1, 1.0]

        if use_two_param:
            p0 = [10.0, 0.4] # for 2 param fit
            bounds = [(0, None), (0, 1.0)]
        else:
            p0 = [10.0, 0.4, 0.5] # changed for Ian's project!
            bounds = [(0, None), (0, 1.0), (0, None)]

        if use_two_param:
            fit_f, fit_grad = chinchilla_flops_2_param_fit, grad_chinchilla_flops_2_param_fit
        else:
            fit_f, fit_grad = chinchilla_flops_fit, grad_chinchilla_flops_fit

        coefficients, cov = get_coefficients_huber(
            train_fs,
            train_xs,
            fit_f,
            fit_grad,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
            return_cov=True,
        )
    elif y_metric == "rc_acc":
        # p0 = [2.0, 2.0, 0.2, 0.2, 1.0]
        p0 = [-1.0, 0.1, 0.5] # changed for Ian's project!
        # bounds = [(0, None), (0, 1.0), (0, None)]
        bounds = [(None, None), (None, None), (None, None)]
        coefficients, cov = get_coefficients_huber(
            train_fs,
            train_xs,
            chinchilla_flops_negated_fit,
            grad_chinchilla_flops_negated_fit,
            p0=p0,
            bounds=bounds,
            max_iter=1000000,
            disp=False,
            return_cov=True,
        )
        raise RuntimeError('hi its david dont be here, not sure about correctness')
        # print(coefficients)
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    return coefficients, cov


def scale_data_by_name(data_by_name):
    fmin = 0.8 * min([min(data["fs"]) for data in data_by_name.values()])
    fmax = 1.2 * max([max(data["fs"]) for data in data_by_name.values()])

    for name, data in data_by_name.items():
        data["fs"] = ((np.array(data["fs"]) - fmin) / (fmax - fmin)).tolist()
        # data["fs"] = np.log(np.array(data["fs"]).astype(float)).tolist()

    return data_by_name


def predict_step1(configs, data_by_name, coefficients, y_metric, use_two_param=False):
    predicted_data_by_name = {}
    plotted_predicted_data_by_name = {}

    unsigned_rel_errors = []
    rel_error = float('inf')

    fmin = 0.8 * min([min(data["fs"]) for data in data_by_name.values()])
    fmax = 1.5 * max([max(data["fs"]) for data in data_by_name.values()])

    if y_metric == "rc_bpb":
        if use_two_param:
            func = chinchilla_flops_2_param_fit
        else:
            func = chinchilla_flops_fit
    elif y_metric == "rc_acc":
        func = chinchilla_flops_fit
        raise RuntimeError('david says no')
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")

    for name, data in data_by_name.items():
        predicted_data_by_name[name] = {
            "fs": data["fs"],
            "xs": [func(f, coefficients) for f in data["fs"]],
        }
        fs = np.exp(np.linspace(np.log(fmin), np.log(fmax), 100))
        plotted_predicted_data_by_name[name] = {
            "fs": fs,
            "xs": [func(f, coefficients) for f in fs],
        }

        if configs[name].mode == "eval":
            predicted_data = predicted_data_by_name[name]
            for f, y, y_pred in zip(data["fs"], data["xs"], predicted_data["xs"]):
                rel_error = (y_pred - y) / y
        else:
            predicted_data = predicted_data_by_name[name]
            for f, y, y_pred in zip(data["fs"], data["xs"], predicted_data["xs"]):
                rel_error_t = (y_pred - y) / y
                unsigned_rel_errors.append(np.abs(rel_error_t))

    return predicted_data_by_name, plotted_predicted_data_by_name, (y, y_pred, rel_error), unsigned_rel_errors


def str_chinchilla_flops_fit(coefficients):
    if len(coefficients) == 3:
        a, alpha, E = coefficients
        A = np.exp(a)
        return f"L(F) = {A:.2f} / F^{alpha:.2f} + {E:.2f}"
    elif len(coefficients) == 2:
        a, alpha = coefficients
        A = np.exp(a)
        return f"L(F) = {A:.2f} / F^{alpha:.2f}"


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
    # fmin = min(min(data["fs"]) for data in plotted_predicted_data_by_name.values())
    # fmax = max(max(data["fs"]) for data in plotted_predicted_data_by_name.values())
    # fs = np.linspace(fmin, fmax, 100)
    # plotted_predicted_data = {
    #     "fs": fs,
    #     "ys": [chinchilla_flops(f, *coefficients) for f in fs],
    # }

    # std_errors = get_std_errors(
    #     plotted_predicted_data["fs"],
    #     plotted_predicted_data["ys"],
    #     coefficients,
    #     cov,
    #     chinchilla_flops_fit,
    #     grad_chinchilla_flops_fit,
    # )

    # # Compute prediction intervals
    # plotted_y_lower = plotted_predicted_data["ys"] - 1.96 * std_errors
    # plotted_y_upper = plotted_predicted_data["ys"] + 1.96 * std_errors

    # ax.fill_between(plotted_predicted_data["fs"], plotted_y_lower, plotted_y_upper, color="pink", alpha=0.3)
    
    # plot the fitted curve
    for name, data in plotted_predicted_data_by_name.items():
        config = configs[name]
        ax.plot(
            data["fs"],
            data["xs"],
            # color="black",
            color=config.color,
            linestyle="-",
            alpha=0.5,
            # linewidth=1.5,
            linewidth=1,
            # label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )
        break

    # plot the actual and predicted data
    unsigned_rel_errors = []
    num_eval_annotation = 0
    for size_idx, (name, data) in enumerate(data_by_name.items()):
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        # print(data)
        # print(predicted_data)

        for i, (f, y) in enumerate(zip(data["fs"], data["xs"])):
            ax.scatter(
                f,
                y,
                color=config.color,
                marker=MARKERS[size_idx] if config.mode == "train" else "o",
                # s=50 if config.mode == "train" else 20,
                s=20 if config.mode == "train" else 20,
                label=f"{config.label} (target)" if config.mode == "eval" else None,
            )
            # print(f, y)

        for f, y, y_pred in zip(data["fs"], data["xs"], predicted_data["xs"]):
            rel_error = (y_pred - y) / y
            if config.mode == "train":
                unsigned_rel_errors.append(np.abs(rel_error))
            else:
                ax.scatter(
                    f,
                    y_pred,
                    color=config.color,
                    marker="x",
                    s=20,
                    label=f"{config.label} ({'predicted'})",
                )
                # ax.annotate(
                #     f"{abs(100 * rel_error):.1f}%",
                #     (f, y_pred),
                #     textcoords="offset points",
                #     xytext=(10, 1 - 10 * num_eval_annotation),
                #     ha="left",
                #     va="bottom",
                #     fontsize=FONTSIZE,
                #     color=config.color,
                # )
                num_eval_annotation += 1
            # print(f, y)
    avg_unsigned_rel_error = np.mean(unsigned_rel_errors)

    ax.set_xscale("log")
    ax.legend(loc="lower left", ncols=1, fontsize=FONTSIZE)
    ax.set_xlabel("Flops (C)", fontsize=FONTSIZE)
    if y_metric == "rc_bpb":
        ax.set_ylabel("Task loss", fontsize=FONTSIZE)
    elif y_metric == "rc_acc":
        ax.set_ylabel("Task RC accuracy", fontsize=FONTSIZE)
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")
    display_name = tasks[task_name].display_name if task_name in tasks else task_name
    ax.set_title(
        f"{display_name} ({avg_unsigned_rel_error * 100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )
    # ax.text(0.95, 0.95, fit_str, transform=ax.transAxes, ha='right', va='top')


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

    results = "Task Name | Actual Value | Predicted Value | Relative Error"

    for i, task_name in enumerate(args.keys):
        data_by_name = get_step1_data_by_name(
            configs, task_name, y_metric=args.y_metric, moving_avg=args.moving_avg
        )

        # data_by_name = scale_data_by_name(data_by_name)

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

        results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)} | {prettify(avg_unsigned_rel_error)}"

        if args.output_path:
            plot_step1(
                configs,
                data_by_name,
                predicted_data_by_name,
                plotted_predicted_data_by_name,
                task_name,
                str_chinchilla_flops_fit(coefficients),
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

    if args.output_path:
        fig.savefig(args.output_path, dpi=300, bbox_inches="tight")

    print(results)
    print("Total fitting error: ", prettify(fitting_error / num_tasks))


if __name__ == "__main__":
    main()
