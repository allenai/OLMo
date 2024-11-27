# python scripts/scaling/step2.py -k v2_main -c scripts/scaling/step2.json -o figure/peteish-moreeval/step2_main.pdf --skip_perc 0.1 --moving_avg 5
# python scripts/scaling/step2.py -k v2_main -c scripts/scaling/step2.json -o figure/peteish-moreeval/step2_c4_main.pdf --x_metric c4 --skip_perc 0.1 --moving_avg 5
# python scripts/scaling/step2.py -k v2_main -c scripts/scaling/step2.json -o figure/peteish-moreeval/step2_taskce_main.pdf --skip_perc 0.5 --use_log_sigmoid --x_metric rc_soft_log

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    get_coefficients,
    get_std_errors,
    grad_log_sigmoid_fit,
    grad_sigmoid_fit,
    log_sigmoid,
    log_sigmoid_fit,
    sigmoid,
    sigmoid_fit,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step2_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

FONTSIZE = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keys", nargs="+", default=[], help="Key(s) for tasks")
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
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Path to write output figure")
    parser.add_argument("--use_log_sigmoid", action="store_true", help="Use log sigmoid instead")
    args = parser.parse_args()

    return args


def fit_step2(data_by_name, task_name, y_metric, use_log_sigmoid=False):
    train_xs, train_ys = [], []
    for name, data in data_by_name.items():
        if data["mode"] == "train":
            train_xs += data["xs"]
            train_ys += data["ys"]
        else:
            data["xs"] = data["xs"][-1:]
            data["ys"] = data["ys"][-1:]

    # add ideal points (these are not plotted)
    if not use_log_sigmoid:
        train_xs.append(0.0)
        train_ys.append(tasks[task_name].task_maximum)
        # train_xs.append(max(train_xs))
        # train_ys.append(tasks[task_name].task_minimum)

    # fit the parameters
    if use_log_sigmoid:
        coefficients, cov = get_coefficients(
            train_xs,
            train_ys,
            log_sigmoid,
            p0=[-0.1, 0.9, 3.0],
            bounds=([-np.inf, 0.0, 0.0], [0.0, np.inf, np.inf]),
            disp=False,
            return_cov=True,
        )
    else:
        coefficients, cov = get_coefficients(
            train_xs,
            train_ys,
            sigmoid,
            p0=[tasks[task_name].task_minimum - 1.0, 0.9, 3.0, tasks[task_name].task_maximum],
            bounds=([-1.0, 0.0, 0.0, 0.0], [0.0, np.inf, np.inf, 1.0]),
            disp=False,
            return_cov=True,
        )

    return coefficients, cov


def predict_step2(configs, data_by_name, coefficients, cov, y_metric, use_log_sigmoid=False):
    predict_fn = log_sigmoid if use_log_sigmoid else sigmoid
    fit_fn = log_sigmoid_fit if use_log_sigmoid else sigmoid_fit
    grad_fit_fn = grad_log_sigmoid_fit if use_log_sigmoid else grad_sigmoid_fit

    all_rel_errors = []

    predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "xs": data["xs"],
            "ys": [predict_fn(x, *coefficients) for x in data["xs"]],
        }
        if config.mode == "eval":
            for x, y, y_pred in zip(data["xs"], data["ys"], predicted_data_by_name[name]["ys"]):
                rel_error = (y_pred - y) / y
                std_error = get_std_errors([x], [y_pred], coefficients, cov, fit_fn, grad_fit_fn)  # [0]
                delta_error = 1.96 * std_error

                all_rel_errors.append(rel_error)

    xmin = min(min(data["xs"]) for data in data_by_name.values())
    xmax = max(max(data["xs"]) for data in data_by_name.values())
    xmin = xmin - 0.2 * (xmax - xmin)

    xs = np.linspace(xmin, xmax, 100)
    plotted_predicted_data = {
        "xs": xs,
        "ys": [predict_fn(x, *coefficients) for x in xs],
    }

    return predicted_data_by_name, plotted_predicted_data, (y, y_pred, rel_error, delta_error), all_rel_errors


def plot_step2(
    configs,
    data_by_name,
    predicted_data_by_name,
    plotted_predicted_data,
    task_name,
    fit_str,
    x_metric,
    y_metric,
    coefficients,
    cov,
    use_log_sigmoid=False,
    ax=plt.gca(),
):
    fit_fn = log_sigmoid_fit if use_log_sigmoid else sigmoid_fit
    grad_fit_fn = grad_log_sigmoid_fit if use_log_sigmoid else grad_sigmoid_fit

    std_errors = get_std_errors(
        plotted_predicted_data["xs"],
        plotted_predicted_data["ys"],
        coefficients,
        cov,
        fit_fn,
        grad_fit_fn,
    )

    # Compute prediction intervals
    plotted_y_lower = plotted_predicted_data["ys"] - 1.96 * std_errors
    plotted_y_upper = plotted_predicted_data["ys"] + 1.96 * std_errors
    unsigned_rel_errs = []

    num_eval_annotation = 0
    eval_num = 0
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data = predicted_data_by_name[name]

        if config.mode == "train":
            ax.scatter(
                data["xs"],
                data["ys"],
                color=config.color,
                marker="o" if config.mode == "train" else "x",
                s=10,
                edgecolors="none" if config.mode == "train" else None,
                alpha=0.5 if config.mode == "train" else 1.0,
                label=f"{config.label} ({'fitted' if config.mode == 'train' else 'target'})",
            )
        for i, (x, y, y_pred) in enumerate(zip(data["xs"], data["ys"], predicted_data["ys"])):

            rel_error = (y_pred - y) / y

            if config.mode == "train":
                unsigned_rel_errs.append(abs(rel_error))
            else:
                if i == 0:
                    ax.scatter(
                        x,
                        y,
                        color=config.color,
                        marker="x",
                        s=20,
                        label=f"{config.label} ({'target'})",
                    )
                    ax.scatter(
                        x,
                        y_pred,
                        color=config.color,
                        marker="o",
                        s=20,
                        label=f"{config.label} ({'predicted'})",
                    )
                    ax.annotate(
                        f"{np.abs(rel_error) * 100:.1f}%",
                        (x, y),
                        textcoords="offset points",
                        xytext=(8 - 40 * num_eval_annotation, -7 + eval_num * 2),
                        ha="left",
                        va="bottom",
                        fontsize=FONTSIZE,
                        color=config.color,
                    )
                    num_eval_annotation += 1
                else:
                    pass
    avg_unsigned_rel_err = np.mean(unsigned_rel_errs)

    # plot the fitted curve
    ax.plot(
        plotted_predicted_data["xs"],
        plotted_predicted_data["ys"],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    ax.fill_between(plotted_predicted_data["xs"], plotted_y_lower, plotted_y_upper, color="pink", alpha=0.3)

    ax.legend(loc="upper right", ncols=1, fontsize=FONTSIZE)
    x_label_name = {
        "rc_bpb": "Task loss",
        "c4": "C4 loss",
        "rc_soft_log": "TaskCE",
    }[x_metric]
    ax.set_xlabel(x_label_name, fontsize=FONTSIZE)

    y_label_name = {
        "rc_acc": "Task RC accuracy",
        "mc_acc": "Task MC accuracy",
    }[y_metric]
    ax.set_ylabel(y_label_name, fontsize=FONTSIZE)

    # ax.set_ylim([0, 1.0])
    ax.set_title(
        f"{tasks[task_name].display_name} (Fitting error: {avg_unsigned_rel_err * 100:.2f}%)",
        fontsize=FONTSIZE,
        fontweight="bold",
    )


def str_sigmoid(coefficients, use_log_sigmoid=False):
    if use_log_sigmoid:
        a, x0, k = coefficients

        # def log_sigmoid(x, a, x0, k):
        #     y = np.log(1 - 1/(1 + np.exp(-k * (x - x0))))
        #     o = (-a) * y + 1
        #     return o

        return f"Acc(L) = 1 - {-a:.2f} * log(1 - 1/(1 + e^(-{k:.2f}(L - {x0:.2f})))"
    else:
        a, x0, k, b = coefficients
        return f"Acc(L) = {a:.2f} / (1 + e^(-{k:.2f}(L - {x0:.2f}))) + {b:.2f}"


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    args.keys = get_task_sets(args.keys)

    sns.set_style("whitegrid")
    num_tasks = len(args.keys)
    num_cols = min(4, num_tasks)
    num_rows = (num_tasks + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.75 * num_cols, 2.5 * num_rows), squeeze=False)

    results = {}
    results_str = "Task Name | Actual Value | Predicted Value | Relative Error"

    rel_errors = []
    for i, task_name in enumerate(args.keys):
        data_by_name = get_step2_data_by_name(
            configs,
            task_name,
            x_metric=args.x_metric,
            y_metric=args.y_metric,
            moving_avg=args.moving_avg,
            skip_perc=args.skip_perc,
        )

        coefficients, cov = fit_step2(data_by_name, task_name, args.y_metric, use_log_sigmoid=args.use_log_sigmoid)
        # a, x0, k, b = coefficients

        # make predictions
        (
            predicted_data_by_name,
            plotted_predicted_data,
            (y, y_pred, rel_error, delta_error),
            all_rel_errors,
        ) = predict_step2(
            configs, data_by_name, coefficients, cov, y_metric=args.y_metric, use_log_sigmoid=args.use_log_sigmoid
        )
        rel_errors += all_rel_errors

        str_formula = str_sigmoid(coefficients, use_log_sigmoid=args.use_log_sigmoid)
        results[task_name] = {"Actual": y, "Pred": y_pred, "Rel Error": rel_error}
        results_str += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)} | {str_formula}"

        # plot the actual and predicted data
        ax = axes[i // num_cols][i % num_cols]

        plot_step2(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data,
            task_name,
            str_formula,
            args.x_metric,
            args.y_metric,
            coefficients,
            cov,
            use_log_sigmoid=args.use_log_sigmoid,
            ax=ax,
        )

    print(f"Mean relative error: {np.mean(np.abs(rel_errors)) * 100:.2f}%")

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
                            handletextpad=0.1,columnspacing=0.7)
    else:
        legend = fig.legend(handles, labels, loc='upper center',
                            ncol=1, fontsize=FONTSIZE, bbox_to_anchor=(1.4, 0.8),
                            handletextpad=0.1,columnspacing=0.7)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename({"index": "Task"}, axis=1)
    
    if args.output_path:
        fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
        df.to_csv(args.output_path.replace(".pdf", ".csv"), index=False)

    print(results_str)
    
    return df

if __name__ == "__main__":
    main()
