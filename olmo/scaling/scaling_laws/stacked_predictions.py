import csv
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from olmo.scaling.scaling_laws.utils import (
    FinalConfig,
    chinchilla_n_d_fit,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
)

# Fitting functions
from olmo.util import StrEnum


def sigmoid(x, L, x0, k, b):
    o = L / (1 + np.exp(-k * (x - x0))) + b
    return o


def reverse_sigmoid(y, L, x0, k, b):
    return x0 - 1 / k * np.log((L / (y - b)) - 1)


# Error with using huber fit; possibly due to incorrect bounds (try later).

# def sigmoid_fit(x, p):
#     return p[0] / (1 + np.exp(-p[2] * (x - p[1]))) + p[3]

# def grad_sigmoid_fit(x, p):
#     grad_L = 1 / (1 + np.exp(-p[2] * (x - p[1])))
#     grad_x0 = p[0] * p[2] * sigmoid_fit(x, p) * (1 - sigmoid_fit(x, p))
#     grad_k = p[0] * (x - p[1]) * sigmoid_fit(x, p) * (1 - sigmoid_fit(x, p))
#     grad_b = 1
#     return [grad_L, grad_x0, grad_k, grad_b]

# # fit the parameters
# coefficients = get_coefficients_huber(
#     train_nds,
#     train_ys,
#     sigmoid_fit,
#     grad_sigmoid_fit,
#     p0=[-1.3, 0.5, 3, 0.3],
#     bounds=None, #[(None, 0), (None, None), (None, None), (None, None)],
# )


BASELINE_BY_TASK_NAME = {
    "HellaSwag-0shot": 0.25,
    "MMLU-Var": 0.25,
    "HellaSwag-5shot": 0.25,
    "ARC-Easy-5shot": 0.25,
    "ARC-Challenge-5shot": 0.25,
    "PiQA-5shot": 0.5,
    "Winogrande-5shot": 0.5,
    "OpenbookQA-5shot": 0.25,
    "SciQ-0shot": 0.25,
    "Copa-0shot": 0.5,
    "CSQA-5shot": 0.2,
    "SocialIQA-5shot": 1 / 3,
}


def get_all_data_by_name(configs, keys) -> Dict:
    data_by_name: Dict = defaultdict(lambda: defaultdict(lambda: []))
    for name, config in configs.items():
        for path in config.paths:
            with open(path) as file_ref:
                reader = csv.DictReader(file_ref)
                rows = [row for row in reader]
                for row in rows:
                    y = np.mean([float(row[key]) for key in keys])
                    data_by_name[name][path].append(y)
    return data_by_name


def size_length_from_path(path):
    name = path.split("/")[-1].strip(".csv")
    return name.split("-")[:2]


def get_dataframe_from_configs(
    x_dict: Dict[str, Dict],
    y_dict: Dict[str, Dict],
    configs: Dict[str, FinalConfig],
) -> pd.DataFrame:
    df = pd.DataFrame()
    xs = []
    ys = []
    params = []
    sizes = []
    lengths = []
    modes = []
    runs = []
    colors = []
    for name, path_dict in x_dict.items():
        config = configs[name]
        for path in path_dict:
            size, length = size_length_from_path(path)
            run_name = f"{size}-{length}"
            x_data = x_dict[name][path]
            y_data = y_dict[name][path]
            xs += x_data
            ys += y_data
            params += [config.n for _ in range(len(x_data))]
            sizes += [size for _ in range(len(x_data))]
            lengths += [length for _ in range(len(x_data))]
            modes += [config.mode for _ in range(len(x_data))]
            runs += [run_name for _ in range(len(x_data))]
            colors += [config.color for _ in range(len(x_data))]

    df["x"] = xs
    df["y"] = ys
    df["params"] = params
    df["size"] = sizes
    df["length"] = lengths
    df["mode"] = modes
    df["run"] = runs
    df["color"] = colors
    return df


def get_predicted_error(df):
    eval_row = df[df["mode"] == "eval"].iloc[-1]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    # rel_error = f"{rel_error * 100:+.1f}%"
    return rel_error


def fit_step1(df: pd.DataFrame):
    df = df.dropna()

    # Fit
    train_nds = list(df[df["mode"] == "train"][["params", "x"]].itertuples(index=False, name=None))
    train_ys = df[df["mode"] == "train"]["y"]

    # fit the parameters
    coefficients = get_coefficients_huber(
        train_nds,
        train_ys,
        chinchilla_n_d_fit,
        grad_chinchilla_n_d_fit,
        p0=[3.0, 6.0, 0.1, 0.2, 1.0],
        bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)],
        disp=False,
    )

    df["predicted_y"] = df.apply(lambda x: chinchilla_n_d_fit([x.params, x.x], coefficients), axis=1)
    return df, coefficients


def predict_step1(n: int, d: int, coefficients: List[float]):
    return chinchilla_n_d_fit([n, d], coefficients)


def plot_step1(
    df, coefficients, ax, x_label=None, y_label=None, title="Fitting final score", do_label=True, logscale=False
):
    # a, b, alpha, beta, E = coefficients
    # A, B = np.exp(a), np.exp(b)

    eval_row = df[df["mode"] == "eval"].iloc[-1]
    x = eval_row["x"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    run_name = eval_row["run"]

    for label in df["size"].unique():
        adf = df[df["size"] == label]
        ax.scatter(
            adf["x"], adf["y"], color="white", edgecolors=adf["color"], s=7.0, label=label if do_label else None
        )

    ax.scatter(x, y, marker="x", color="blue", label=f"actual ({run_name})= {y:0.4f}" if do_label else None, s=50)
    ax.scatter(
        x,
        y_pred,
        marker="^",
        color="black",
        label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None,
        s=50,
    )
    ax.annotate(
        f"{eval_row['run']}: {rel_error * 100:+.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(10, 5),
        ha="center",
        fontsize=10,
        color="brown",
    )

    for params in df["params"].unique():
        plotted_xs = np.linspace(df[df["params"] == params]["x"].max(), df[df["params"] == params]["x"].min(), 100)
        plotted_ys = [chinchilla_n_d_fit([params, x_val], coefficients) for x_val in plotted_xs]

        ax.plot(
            plotted_xs,
            plotted_ys,
            color="black",
            linestyle="--",
            linewidth=0.8,
        )

    # ax.text(
    #     x=0.25,
    #     y=0.50,
    #     s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
    #     fontsize=10,
    #     transform=ax.transAxes,
    # )

    if do_label:
        ax.legend(loc="upper right", ncols=1)

    if logscale:
        ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def fit_step2(df: pd.DataFrame, baseline: float, add_ideal_points: bool = True):
    df = df.dropna()

    # Fit

    train_xs = df[df["mode"] == "train"]["x"]
    train_ys = df[df["mode"] == "train"]["y"]

    if add_ideal_points:
        train_xs = pd.concat([pd.Series([0.0001]), train_xs, pd.Series([2.6])], ignore_index=True)
        train_ys = pd.concat([pd.Series([1.0]), train_ys, pd.Series([baseline])], ignore_index=True)

    coefficients, pcov = curve_fit(sigmoid, train_xs, train_ys, p0=[baseline - 1.0, 0.9, 3.0, 1.0], maxfev=1000000)

    df["predicted_y"] = df["x"].apply(lambda x: sigmoid(x, *coefficients))

    return df, coefficients


def predict_step2(bpb_loss: float, coefficients: List[float]):
    return sigmoid(bpb_loss, *coefficients)


def plot_step2(
    df,
    coefficients,
    ax,
    x_label=None,
    y_label=None,
    title="Fitting final score",
    add_ideal_points=True,
    do_label=True,
):
    eval_row = df[df["mode"] == "eval"].iloc[-1]
    x = eval_row["x"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    run_name = eval_row["run"]

    for label in df["size"].unique():
        adf = df[df["size"] == label]
        ax.scatter(
            adf["x"], adf["y"], color="white", edgecolors=adf["color"], s=7.0, label=label if do_label else None
        )

    ax.scatter(x, y, marker="x", color="blue", label=f"actual ({run_name}) = {y:0.4f}" if do_label else None, s=50)
    ax.scatter(
        x,
        y_pred,
        marker="^",
        color="black",
        label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None,
        s=50,
    )
    ax.annotate(
        f"{eval_row['run']}: {rel_error * 100:+.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(30, 5),
        ha="center",
        fontsize=10,
        color="brown",
    )

    if add_ideal_points:
        plotted_xs = np.linspace(max(2.6, df["x"].max()), 0.01, 100)
    else:
        plotted_xs = np.linspace(df["x"].max(), df["x"].min(), 100)
    plotted_ys = [sigmoid(x_val, *coefficients) for x_val in plotted_xs]

    ax.plot(
        plotted_xs,
        plotted_ys,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )

    # L, x0, k, b = coefficients
    # print(f"σ(L, x0, k, b) \n = {L:.2f} / (1 + e^(-({k:.2f}(x - {x0:.2f})))) + {b:.2f}")
    # ax.text(
    #     x=0.25,
    #     y=0.50,
    #     s=f"σ(L, x0, k, b) \n = {L:.2f} / (1 + e^(-({k:.2f}(x - {x0:.2f})))) + {b:.2f}",
    #     fontsize=10,
    #     transform=plt.gca().transAxes,
    # )

    if do_label:
        ax.legend(loc="upper right", ncols=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_stacked(
    df, step2_df, ax, x_label=None, y_label=None, title=None, do_label=True, do_grey=False, logscale=False
):
    mode_colors = {"train": "grey", "eval": "lightgrey"}

    for label in df["size"].unique():
        adf = df[df["size"] == label]
        ax.scatter(
            adf["x"],
            adf["y"],
            color="white",
            edgecolors=adf["mode"].apply(lambda x: mode_colors[x]) if do_grey else adf["color"],
            s=7.0,
            label=label,
        )

    step2_df = (
        pd.merge(df.reset_index()[["index", "x"]], step2_df, left_on="index", right_on="level_1", how="inner")
        .rename({"x_x": "tokens", "x_y": "x"}, axis=1)
        .drop("index", axis=1)
        .drop("level_1", axis=1)
    )
    eval_row = step2_df[step2_df["mode"] == "eval"].iloc[-1]
    x = eval_row["tokens"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y

    ax.scatter(x, y, marker="x", color="blue", label=f"actual = {y:0.4f}", s=100)
    ax.scatter(x, y_pred, marker="^", color="black", label=f"predicted = {y_pred:0.4}", s=100)
    ax.annotate(
        f"{eval_row['run']}: {rel_error * 100:+.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(30, -30),
        ha="center",
        fontsize=10,
        color="brown",
    )

    if do_label:
        ax.legend(loc="lower right", ncols=1)

    if logscale:
        ax.set_xscale("log")
    ax.set_xlabel(x_label or "tokens")
    ax.set_ylabel(y_label or "accuracy")
    ax.set_title(title or "stacked prediction")


class DownstreamPredictionFeatures(StrEnum):
    raw = "raw"
    moving_average = "moving_average"  # TODO: how to specify window size?
    exponential_moving_average = "exponential_moving_average"  # TODO: how to specify alpha.


def apply_moving_average(step_df, column: str, window: int = 20):
    return step_df.groupby("run")[column].transform(lambda x: x.rolling(window=window).mean())


def apply_exponential_moving_average(step_df, column: str, alpha: float = 0.5):
    return step_df.groupby("run")[column].transform(lambda x: x.ewm(alpha=alpha).mean())


def get_downstream_predictions(
    configs: Dict[str, FinalConfig],
    tasks: Dict,
    feature_type: DownstreamPredictionFeatures = DownstreamPredictionFeatures.raw,
    use_last_n_points_step1: int = 1,
    use_last_n_percentage: float = 1.0,
    *,
    save_figures: Optional[str] = None,
    target_n_d: Optional[Tuple[int, int]] = None,
    **feature_kwargs,
):
    assert 0.0 <= use_last_n_percentage <= 1.0
    do_plot = save_figures is not None

    if do_plot:
        rows = len(tasks.keys())
        fig, axes = plt.subplots(rows, 3, figsize=(20, 5 * rows))

    no_error = target_n_d is not None

    if not no_error:
        target = [run_name for run_name in configs if configs[run_name].mode == "eval"][0]
        step1_error: Dict = {target: {}}
        stacked_error: Dict = {target: {}}
    else:
        target = "_".join([str(x) for x in target_n_d])

    step1_predictions: Dict = {target: {}}
    stacked_predictions: Dict = {target: {}}

    for i, (task_name, task) in enumerate(tasks.items()):
        tokens = get_all_data_by_name(configs, ["throughput/total_tokens"])
        bpb_loss = get_all_data_by_name(configs, task["bpb"])
        downstream_loss = get_all_data_by_name(configs, task["score"])

        step1_df = get_dataframe_from_configs(tokens, bpb_loss, configs)

        if feature_type == DownstreamPredictionFeatures.moving_average:
            step1_df["y"] = apply_moving_average(step1_df, "y", **feature_kwargs)
        elif feature_type == DownstreamPredictionFeatures.exponential_moving_average:
            step1_df["y"] == apply_exponential_moving_average(step1_df, "y", **feature_kwargs)

        step1_df = step1_df.groupby("run").apply(lambda rows: rows.iloc[-use_last_n_points_step1:], include_groups=False).reset_index()
        step1_df, coefficients = fit_step1(step1_df)

        if not no_error:
            target_n_d = [
                step1_df[step1_df["mode"] == "eval"].params.iloc[0],
                step1_df[step1_df["mode"] == "eval"].x.iloc[0],
            ]

        step1_predictions[target][task_name] = predict_step1(*target_n_d, coefficients)

        if do_plot:
            plot_step1(
                step1_df,
                coefficients,
                axes[i][0],
                x_label="tokens",
                y_label="task loss",
                title=f"predicting task_loss ({task_name})",
                do_label=True,
                logscale=True,
            )

        step2_df = get_dataframe_from_configs(bpb_loss, downstream_loss, configs)

        step2_df = (
            step2_df.groupby("run")
            .apply(lambda x: x.iloc[-int(np.ceil(use_last_n_percentage * len(x))) :], include_groups=False)
            .reset_index()
        )

        if feature_type == DownstreamPredictionFeatures.moving_average:
            step2_df["x"] = apply_moving_average(step2_df, "x", **feature_kwargs)
        elif feature_type == DownstreamPredictionFeatures.exponential_moving_average:
            step2_df["x"] == apply_exponential_moving_average(step2_df, "x", **feature_kwargs)

        last_match_idx = step2_df.loc[step2_df["mode"] == "eval"].tail(1).index
        step2_df.loc[last_match_idx, "x"] = step1_predictions[target][task_name]

        step2_df, coefficients = fit_step2(step2_df, tasks[task_name]["baseline"])

        stacked_predictions[target][task_name] = predict_step2(step1_predictions[target][task_name], coefficients)

        if do_plot:
            plot_step2(
                step2_df,
                coefficients,
                axes[i][1],
                x_label="task loss",
                y_label="task accuracy",
                title=f"predicting task_accuracy ({task_name})",
                do_label=True,
            )

        if not no_error:
            step1_error[target][task_name] = get_predicted_error(step1_df)
            stacked_error[target][task_name] = get_predicted_error(step2_df)

        if do_plot:
            df = get_dataframe_from_configs(tokens, downstream_loss, configs)
            plot_stacked(
                df,
                step2_df,
                axes[i][2],
                title=f"Stacked predictions using {feature_type} ({feature_kwargs})",
                do_label=True,
                do_grey=False,
                logscale=True,
            )

    if do_plot:
        fig.suptitle("Combined 2-step downstream predictions", fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.savefig(save_figures, dpi=300)
        # plt.close()

    if not no_error:
        return step1_predictions, stacked_predictions, step1_error, stacked_error
    else:
        return step1_predictions, stacked_predictions
