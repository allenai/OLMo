# python scripts/scaling/stacked.py -k v2_main -c scripts/scaling/step2.json -o figure/peteish-moreeval/stacked_main.pdf --skip_perc 0.1 --moving_avg 5
# python scripts/scaling/stacked.py -k hellaswag_val_5shot -c scripts/scaling/step2.json -o figure/peteish-moreeval/figure1.pdf --skip_perc 0.1 --moving_avg 5

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from step1 import fit_step1, plot_step1, predict_step1, str_chinchilla_n_d_fit
from step2 import fit_step2, plot_step2, predict_step2, str_sigmoid

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
    get_coefficients,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
    sigmoid,
)
from olmo.scaling.scaling_laws.utils import (
    get_final_configs,
    get_step1_data_by_name,
    get_step2_data_by_name,
    get_task_sets,
    prettify,
    tasks,
)

# MARKERS = ["s", "P", "p", "*"]

MARKERS = {"1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}
FONTSIZE = 11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--keys", nargs="+", default=[], help="For avg metrics. Use one of [all-val-lm, all-bpb]"
    )
    parser.add_argument(
        "--num_to_avg", type=int, default=1, help="Number of final ckpts to average (for final loss fitting)"
    )
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
    parser.add_argument("--target_n", type=str, required=False, default="")
    parser.add_argument("--target_d", type=str, required=False, default="")
    args = parser.parse_args()

    return args


def str_chinchilla_n_d_fit(coefficients):
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)
    return f"L(N, D) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}"


# These are updated with actual Peteish count
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


_number_unit_re = re.compile(r"^([0-9]+)([a-zA-Z]+)$")
_run_name_re = re.compile(r"^([^-]+)-([^-]+)-([^-]+)$")


def parse_size(size: str) -> int:
    return MODEL_PARAMS[size]


def parse_length(length: str, model_size: int) -> int:
    length_in_tokens, length_unit = _number_unit_re.match(length.strip().upper()).groups()  # type: ignore
    length_in_tokens = int(length_in_tokens)
    if length_unit == "C" or length_unit == "XC":
        length_in_tokens *= 20 * model_size
    elif length_unit == "K":
        length_in_tokens *= 1000
    elif length_unit == "M":
        length_in_tokens *= 1000000
    elif length_unit == "B":
        length_in_tokens *= 1000000000
    elif length_unit == "T":
        length_in_tokens *= 1000000000000
    else:
        raise ValueError(f"Could not parse length '{length}'")
    return length_in_tokens


def main():
    args = parse_args()

    configs = get_final_configs(args.config_path)

    args.keys = get_task_sets(args.keys)

    sns.set_style("whitegrid")

    if args.target_n:
        pred_n = parse_size(args.target_n)
        pred_d = parse_length(args.target_d, pred_n)

    num_tasks = len(args.keys)
    num_cols = 3
    num_rows = num_tasks
    fig, axes = plt.subplots(num_tasks, num_cols, figsize=(2.75 * num_cols, 2.25 * num_rows), squeeze=False)

    # results = "Task Name | Loss Error | Accuracy Error | Stacked Accuracy Error"

    results = {}

    for r, task_name in enumerate(args.keys):
        task_results = {}

        # Step 1

        data_by_name = get_step1_data_by_name(configs, task_name, y_metric="rc_bpb", moving_avg=args.moving_avg)

        # fit the parameters
        coefficients, cov = fit_step1(data_by_name, "rc_bpb")

        # make predictions
        (
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            (y, y_pred, rel_error),
            unsigned_rel_errors,
        ) = predict_step1(configs, data_by_name, coefficients, y_metric="rc_bpb")

        avg_unsigned_rel_error = np.mean(unsigned_rel_errors)
        # fitting_error += avg_unsigned_rel_error

        # results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)} | {prettify(avg_unsigned_rel_error)}"

        ax = axes[r][0]
        plot_step1(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            str_chinchilla_n_d_fit(coefficients),
            "rc_bpb",
            coefficients,
            cov,
            ax=ax,
        )

        ax.set_title("Step 1", fontweight="bold", fontsize=FONTSIZE)

        step1_coefficients = coefficients

        # Step 2

        data_by_name = get_step2_data_by_name(
            configs,
            task_name,
            x_metric="rc_bpb",
            y_metric="rc_acc",
            moving_avg=args.moving_avg,
            skip_perc=args.skip_perc,
        )

        # # Add row for predicted loss from step 1
        # for name, data in data_by_name.items():
        #     config = configs[name]
        #     if config.mode == "eval":
        #         predicted_data = predicted_data_by_name[name]  # step1 predictions
        #         data["xs"] += predicted_data["xs"]
        #         data["ys"] += data["ys"]
        #         data["ds"] += data["ds"]
        #         data["row_type"] = ["actual", "step1"]

        coefficients, cov = fit_step2(data_by_name, task_name, "rc_acc", use_log_sigmoid=False)
        step2_coefficients = coefficients

        # make predictions
        (
            predicted_data_by_name,
            plotted_predicted_data,
            (y, y_pred, rel_error, delta_error),
            all_rel_errors,
        ) = predict_step2(configs, data_by_name, coefficients, cov, y_metric="rc_acc", use_log_sigmoid=False)
        # rel_errors += all_rel_errors

        str_formula = str_sigmoid(coefficients, use_log_sigmoid=False)
        # results += f"\n{task_name} | {prettify(y, False)} | {prettify(y_pred, False)} | {prettify(rel_error)} | {str_formula}"

        # plot the actual and predicted data
        ax = axes[r][1]

        plot_step2(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data,
            task_name,
            str_formula,
            "rc_bpb",
            "rc_acc",
            coefficients,
            cov,
            use_log_sigmoid=False,
            ax=ax,
        )

        ax.set_title("Step 2", fontweight="bold", fontsize=FONTSIZE)

        # Stacked plot

        ax = axes[r][2]

        data_by_name = get_step1_data_by_name(configs, task_name, y_metric="rc_acc", moving_avg=args.moving_avg)

        dmin = 0.8 * min([min(data["ds"]) for data in data_by_name.values()])
        dmax = 1.5 * max([max(data["ds"]) for data in data_by_name.values()])

        for name, data in data_by_name.items():
            predicted_data_by_name[name] = {
                "ds": data["ds"],
                "xs": [
                    sigmoid(chinchilla_n_d_fit([n, d], step1_coefficients), *step2_coefficients)
                    for n, d in zip(data["ns"], data["ds"])
                ],
            }
            ds = np.exp(np.linspace(np.log(dmin), np.log(dmax), 100))
            ns = [data["ns"][0]] * len(ds)
            plotted_predicted_data_by_name[name] = {
                "ds": ds,
                "xs": [
                    sigmoid(chinchilla_n_d_fit([n, d], step1_coefficients), *step2_coefficients)
                    for n, d in zip(ns, ds)
                ],
            }

        plot_step1(
            configs,
            data_by_name,
            predicted_data_by_name,
            plotted_predicted_data_by_name,
            task_name,
            "",
            "rc_acc",
            coefficients,
            cov,
            ax,
        )

        ax.set_title("Chained", fontweight="bold", fontsize=FONTSIZE)

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    # delete x-axis labels for all but the bottom row
    for i in range(num_cols):
        for j in range(num_rows):
            axes[j][i].legend().remove()

    fig.tight_layout()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=FONTSIZE,
        bbox_to_anchor=(0.5, 1.25),
        handletextpad=0.3,
        columnspacing=0.7,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    # fig.subplots_adjust(top=0.88)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")

    # print(results_str)


if __name__ == "__main__":
    main()
