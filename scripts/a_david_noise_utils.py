from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ladder_peteish as ladder
from olmo.scaling.scaling_laws.utils import get_coefficients_huber, chinchilla_n_d_fit, grad_chinchilla_n_d_fit
from typing import Optional

from collections import defaultdict
import csv
from scipy.optimize import curve_fit

COLOR_MAP = {
    "190M": "darkred",
    "370M": "darkorange",
    "600M": "goldenrod",
    "760M": "darkgreen",
    "1B": "teal",
    "7B": "darkmagenta",
}

MMLU_NAMES = ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]

DEV_TASKS = {
    "HellaSwag-5shot": {
        "bpb": ["eval/downstream_bpb/hellaswag_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/hellaswag_rc_5shot_len_norm"],
    },
    "ARC-Challenge-5shot": {
        "bpb": ["eval/downstream_bpb/arc_challenge_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/arc_challenge_rc_5shot_len_norm"],
    },
}

ALL_TASKS = {
    "MMLU-Var": {
        "bpb": [f"eval/downstream_bpb/{n}_var_bpb_bpb" for n in MMLU_NAMES],
        "score": [f"eval/downstream/{n}_var_len_norm" for n in MMLU_NAMES],
        "x_label": "mmlu_var_bpb",
        "y_label": "mmlu_var_score",
    },
    "MMLU-Stem": {
        "bpb": [f"eval/downstream_bpb/mmlu_stem_var_bpb_bpb"],
        "score": [f"eval/downstream/mmlu_stem_var_len_norm"],
        "x_label": "mmlu_var_bpb",
        "y_label": "mmlu_var_score",
    },
    "MMLU-Humanities": {
        "bpb": [f"eval/downstream_bpb/mmlu_humanities_var_bpb_bpb"],
        "score": [f"eval/downstream/mmlu_humanities_var_len_norm"],
        "x_label": "mmlu_var_bpb",
        "y_label": "mmlu_var_score",
    },
    "MMLU-Social-Science": {
        "bpb": [f"eval/downstream_bpb/mmlu_social_sciences_var_bpb_bpb"],
        "score": [f"eval/downstream/mmlu_social_sciences_var_len_norm"],
        "x_label": "mmlu_var_bpb",
        "y_label": "mmlu_var_score",
    },
    "MMLU-Other": {
        "bpb": [f"eval/downstream_bpb/mmlu_other_var_bpb_bpb"],
        "score": [f"eval/downstream/mmlu_other_var_len_norm"],
        "x_label": "mmlu_var_bpb",
        "y_label": "mmlu_var_score",
    },
    "HellaSwag-5shot": {
        "bpb": ["eval/downstream_bpb/hellaswag_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/hellaswag_rc_5shot_len_norm"],
    },
    "ARC-Easy-5shot": {
        "bpb": ["eval/downstream_bpb/arc_easy_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/arc_easy_rc_5shot_acc"],
    },
    "ARC-Challenge-5shot": {
        "bpb": ["eval/downstream_bpb/arc_challenge_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/arc_challenge_rc_5shot_len_norm"],
    },
    # "ARC-Challenge-0shot": {
    #     "bpb": ["eval/downstream_bpb/arc_challenge_rc_0shot_bpb_bpb"],
    #     "score": ["eval/downstream/arc_challenge_rc_0shot_len_norm"],
    # },
    "PiQA-5shot": {
        "bpb": ["eval/downstream_bpb/piqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/piqa_rc_5shot_len_norm"],
    },
    "Winogrande-5shot": {
        "bpb": ["eval/downstream_bpb/winogrande_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/winogrande_rc_5shot_acc"],
    },
    "OpenbookQA-5shot": {
        "bpb": ["eval/downstream_bpb/openbookqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/openbookqa_rc_5shot_len_norm"],
    },
    "SciQ-0shot": {
        "bpb": ["eval/downstream_bpb/sciq_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/sciq_rc_0shot_acc"],
    },
    "CSQA-5shot": {
        "bpb": ["eval/downstream_bpb/csqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/csqa_rc_5shot_len_norm"],
    },
    "SocialIQA-5shot": {
        "bpb": ["eval/downstream_bpb/socialiqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/socialiqa_rc_5shot_len_norm"],
    },
    "BoolQ-5shot": {
        "bpb": ["eval/downstream_bpb/boolq_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/boolq_rc_5shot_acc"],
    },
    # "HellaSwag-0shot": {
    #     "bpb": ["eval/downstream_bpb/hellaswag_rc_0shot_bpb_bpb"],
    #     "score": ["eval/downstream/hellaswag_rc_0shot_len_norm"],
    # },
    # "Copa-0shot": {
    #     "bpb": ["eval/downstream_bpb/copa_rc_0shot_bpb_bpb"],
    #     "score": ["eval/downstream/copa_rc_0shot_acc"],
    # },
}

ZERO_SHOT_TASKS = {
    "HellaSwag-0shot": {
        "bpb": ["eval/downstream_bpb/hellaswag_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/hellaswag_rc_0shot_len_norm"],
    },
    "ARC-Easy-0shot": {
        "bpb": ["eval/downstream_bpb/arc_easy_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/arc_easy_rc_0shot_acc"],
    },
    "ARC-Challenge-0shot": {
        "bpb": ["eval/downstream_bpb/arc_challenge_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/arc_challenge_rc_0shot_len_norm"],
    },
    "PiQA-0shot": {
        "bpb": ["eval/downstream_bpb/piqa_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/piqa_rc_0shot_len_norm"],
    },
    "Winogrande-0shot": {
        "bpb": ["eval/downstream_bpb/winogrande_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/winogrande_rc_0shot_acc"],
    },
    "OpenbookQA-0shot": {
        "bpb": ["eval/downstream_bpb/openbookqa_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/openbookqa_rc_0shot_len_norm"],
    },
    "CSQA-0shot": {
        "bpb": ["eval/downstream_bpb/csqa_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/csqa_rc_0shot_len_norm"],
    },
    "SocialIQA-0shot": {
        "bpb": ["eval/downstream_bpb/socialiqa_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/socialiqa_rc_0shot_len_norm"],
    },
    "BoolQ-0shot": {
        "bpb": ["eval/downstream_bpb/boolq_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/boolq_rc_0shot_acc"],
    },
}

# TASKS = DEV_TASKS # Dev tasks are for quickly prototyping notebooks
TASKS = ALL_TASKS 

BASELINE_BY_TASK_NAME = {
    'HellaSwag-0shot': 0.25,
    'MMLU-Var': 0.25,
    'MMLU-Stem': 0.25,
    'MMLU-Humanities': 0.25,
    'MMLU-Social-Science': 0.25,
    'MMLU-Other': 0.25,
    'HellaSwag-5shot': 0.25,
    'ARC-Easy-5shot': 0.25,
    'ARC-Challenge-5shot': 0.25,
    'PiQA-5shot': 0.5,
    'Winogrande-5shot': 0.5,
    'OpenbookQA-5shot': 0.25,
    'SciQ-0shot': 0.25,
    'Copa-0shot': 0.5,
    'CSQA-5shot': 0.2,
    'SocialIQA-5shot': 1 / 3,
    'BoolQ-5shot': 0.5,

    'HellaSwag-0shot': 0.25,
    'ARC-Easy-0shot': 0.25,
    'ARC-Challenge-0shot': 0.25,
    'PiQA-0shot': 0.5,
    'Winogrande-0shot': 0.5,
    'OpenbookQA-0shot': 0.25,
    'CSQA-0shot': 0.2,
    'SocialIQA-0shot': 1 / 3,
    'BoolQ-0shot': 0.5,
}

# We only include ce loss and the 6 dolma sets, as these are the sets we can include in the paper
ce_columns = [
    'eval/c4_en-validation/CrossEntropyLoss',
    'eval/dolma_books-validation/CrossEntropyLoss',
    'eval/dolma_common-crawl-validation/CrossEntropyLoss',
    'eval/dolma_pes2o-validation/CrossEntropyLoss',
    'eval/dolma_reddit-validation/CrossEntropyLoss',
    'eval/dolma_stack-validation/CrossEntropyLoss',
    'eval/dolma_wiki-validation/CrossEntropyLoss',
]

N_LAST_CKPTS = 20

pd.options.mode.chained_assignment = None


def sigmoid(x, L, x0, k, b):
    o = L / (1 + np.exp(- k * (x - x0))) + b
    return (o)


# def reverse_sigmoid(y, L, x0, k, b):
#     return x0 - 1/k * np.log((L / (y - b)) -1)


def get_gflops(run_name: str, length_in_tokens: Optional[int] = None):
    run_name, size, length = get_name_size_length(run_name)
    length_in_tokens = length_in_tokens or ladder.parse_length(length, ladder.parse_size(size))
    flops = ladder.MODEL_GFLOPS[size]
    return flops * length_in_tokens / 1e9


def get_params(run_name: str):
    run_name, size, length = get_name_size_length(run_name)
    params = ladder.MODEL_PARAMS[size]
    return params


def get_all_data_by_name(configs, keys):
    data_by_name = defaultdict(lambda: defaultdict(lambda: []))
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
    #wandb/amberish-rulebased/150M-1xC.csv
    name = path.split("/")[-1].strip(".csv")
    return name.split("-")[:2]


def get_dataframe(configs, x_dict, y_dict):
    data = []
    for name, path_dict in x_dict.items():
        config = configs[name]
        for path in path_dict:
            size, length = size_length_from_path(path)
            run_name = f"{size}-{length}"
            x_data = x_dict[name][path]
            y_data = y_dict[name][path]
            
            for x, y in zip(x_data, y_data):
                data.append({
                    "x": x,
                    "y": y,
                    "params": config.n,
                    "size": size,
                    "length": length,
                    "mode": config.mode,
                    "run": run_name,
                    "color": config.color,
                    "last_n_percent": 0.02  # Consider using config.use_last_n_percentage if available
                })
    
    df = pd.DataFrame(data)
    return df


def get_predicted_error(df: pd.DataFrame):
    """ Get relative error of a single prediction point """
    eval_row = df[df["mode"]=="eval"].iloc[-1]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    return rel_error


def get_last_n_predicted_error(df: pd.DataFrame, full_df: pd.DataFrame):
    """ Get relative error using last N checkpoints """
    eval_row = df[df["mode"]=="eval"].iloc[-1]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y

    N = N_LAST_CKPTS
    eval_row_lastn = full_df[full_df["mode"]=="eval"].iloc[-N:]
    y_lastn = eval_row_lastn["y"]
    y_lastn_mean = eval_row_lastn["y"].mean()
    rel_error = (y_pred - y) / y
    rel_error_lastn_mean = (y_pred - y_lastn_mean) / y_lastn_mean
    
    # add # std dev from pred target
    y_lastn_std = np.std(y_lastn)
    z_score = (y_pred - y_lastn_mean) / y_lastn_std

    y_lastn_std_uniform = (y_lastn.max() - y_lastn.min()) / 12**(1/2)
    y_lastn_std_score = (y_pred - y_lastn_mean) / y_lastn_std_uniform
    # y_lastn_std_score = (y_pred - ((y_lastn.max() + y_lastn.min())/2)) / y_lastn_std_uniform

    return {
        "y": y,
        "y_lastn": y_lastn.tolist(),
        "y_pred": y_pred,
        "y_lastn_std": y_lastn_std,
        "y_lastn_z_score": z_score,
        "y_lastn_std_uniform": y_lastn_std_uniform,
        "y_lastn_std_score": y_lastn_std_score,
        "rel_error": rel_error,
        "rel_error_lastn_mean": rel_error_lastn_mean
    }


def fit_step1(df: pd.DataFrame):
    df = df.dropna()

    train_nds = list(df[df["mode"]=="train"][["params", "x"]].itertuples(index=False, name=None))
    train_ys = df[df["mode"]=="train"]["y"]

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


def predict_step1(n, d, coefficients):
    return chinchilla_n_d_fit([n, d], coefficients)


def fit_step2(df: pd.DataFrame, baseline, add_ideal_points=True):
    df = df.dropna()

    train_xs = df[df["mode"]=="train"]["x"]
    train_ys = df[df["mode"]=="train"]["y"]

    if add_ideal_points:
        train_xs = pd.concat([pd.Series([0.01]), train_xs, pd.Series([2.6])], ignore_index=True)
        train_ys = pd.concat([pd.Series([1.0]), train_ys, pd.Series([baseline])], ignore_index=True)
        # train_xs = pd.concat([pd.Series([0.01]), train_xs], ignore_index=True)
        # train_ys = pd.concat([pd.Series([1.0]), train_ys], ignore_index=True)

    coefficients, pcov = curve_fit(sigmoid, train_xs, train_ys, p0=[baseline - 1.0, 0.9, 3.0, 1.0], maxfev=1000000)
    df["predicted_y"] = df["x"].apply(lambda x: sigmoid(x, *coefficients))

    return df, coefficients


def predict_step2(bpb_loss, coefficients):
    return sigmoid(bpb_loss, *coefficients)


def plot_step1(df: pd.DataFrame, coefficients, ax: plt.Axes, x_label=None, y_label=None, title="Fitting final score", do_label=True, full_df=None, inset_axis=True):
    eval_row = df[df["mode"]=="eval"].iloc[-1]
    x = eval_row["x"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    run_name = eval_row["run"]

    no_legend = True

    # average last n checkpoints of target scale
    if full_df is not None:
        N = N_LAST_CKPTS
        eval_row_lastn = full_df[full_df["mode"]=="eval"].iloc[-N:]
        x_lastn = eval_row_lastn["x"]
        y_lastn = eval_row_lastn["y"]
        x_lastn_mean = eval_row_lastn["x"].iloc[-1]
        y_lastn_mean = eval_row_lastn["y"].mean()
        rel_error_lastn_mean = (y_pred - y_lastn_mean) / y_lastn_mean

    # create inset ax
    if inset_axis:
        if no_legend:
            axins = ax.inset_axes([0.48, 0.48, 0.5, 0.5])
        else:
            # axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35]) # top right
            # axins = ax.inset_axes([0.2, 0.63, 0.35, 0.35]) # top left
            axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35]) # bottom right
        axes = [ax, axins]
    else:
        axes = [ax]

    for label in df["size"].unique():
        adf = df[df["size"]==label]
        for _ax in axes:
            _ax.scatter(adf["x"], adf["y"], color="white", edgecolors=adf["color"], s=7.0, label=label if do_label else None)
        # ax.scatter(adf["x"], adf["y"], color=adf["color"], s=0.5, label=label if do_label else None)

    for _ax in axes:
        _ax.scatter(x, y, marker="x", color="blue", label=f"actual ({run_name})= {y:0.4f}" if do_label else None, s=50)
        _ax.scatter(x, y_pred, marker="^", color="black", label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None, s=50)
        _ax.annotate( 
            f"{eval_row['run']}: {rel_error * 100:+.1f}%", (x, y), textcoords="offset points", 
            xytext=(6, 3), ha="left", fontsize=8, color="blue"
        )
        if full_df is not None:
            _ax.annotate( 
                f"{eval_row['run']}: {rel_error_lastn_mean * 100:+.1f}%", (x, y_lastn_mean), textcoords="offset points", 
                xytext=(6, -9), ha="left", fontsize=8, color="green"
            )
            # adjust_text_util(_ax)

    for _ax in axes:
        for params in df["params"].unique():
            plotted_xs = np.linspace(df[df["params"]==params]["x"].max(), df[df["params"]==params]["x"].min(), 100)
            plotted_ys = [chinchilla_n_d_fit([params, x_val], coefficients) for x_val in plotted_xs]

            _ax.plot(plotted_xs, plotted_ys, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    if full_df is not None:
        # show all target scale checkpoints (exclude the first 30)
        eval_row_all_points = full_df[full_df["mode"]=="eval"][30:]
        x_all = eval_row_all_points["x"]
        y_all = eval_row_all_points["y"]
        for _ax in axes:
            _ax.scatter(x_all, y_all, marker="x", color="blue", s=1, alpha=0.3)

        for _ax in axes:
            # on average of last n checkpoints
            _ax.scatter(x_lastn, y_lastn, color="green", s=1, zorder=10)
            _ax.scatter(x_lastn_mean, y_lastn_mean, marker="x", color="green", s=50, zorder=10)

        # # Add ±1 std dev for last N points
        # y_lastn_mean_std = eval_row_lastn["y"].std()
        # width = 0.01e11
        # for _ax in [ax, axins]:
        #     _ax.fill_between(
        #         [x_lastn_mean-width, x_lastn_mean+width], 
        #         [y_lastn_mean-y_lastn_mean_std, y_lastn_mean-y_lastn_mean_std],
        #         [y_lastn_mean+y_lastn_mean_std, y_lastn_mean+y_lastn_mean_std],
        #         color="green",
        #         alpha=0.2,
        #         zorder=9
        #     )
            
    # a, b, alpha, beta, E = coefficients
    # A, B = np.exp(a), np.exp(b)
    # ax.text(
    #     x=0.25,
    #     y=0.50,
    #     s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
    #     fontsize=10,
    #     transform=ax.transAxes,
    # )

    if do_label and not no_legend:
        ax.legend(loc="upper right", ncols=1, fontsize=8)

    if inset_axis:
        # Set the limits for the zoomed region
        x_width, y_width = (max(df["x"]) - min(df["x"])), (max(df["y"]) - min(df["y"]))
        window_size = 0.2
        x_max = max(df["x"]) + x_width * (window_size/2)
        x_min = x_max - x_width * window_size
        # y_min = y_pred - y_width * (window_size/2) # <- center on target/actual
        y_min = y - y_width * (window_size/2) # <- center on prediction
        y_max = y_min + y_width * window_size
        axins.set_xlim(x_min, x_max)
        axins.set_ylim(y_min, y_max)
        ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_step2(df: pd.DataFrame, coefficients, ax: plt.Axes, x_label=None, y_label=None, title="Fitting final score", add_ideal_points=True, do_label=True, full_df=None, x_pred=None):    
    eval_row = df[df["mode"]=="eval"].iloc[-1]
    x = eval_row["x"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    run_name = eval_row["run"]

    no_legend = True

    # We 
    y_pred_x_pred = predict_step2(x_pred, coefficients)

    # average last n checkpoints of target scale
    if full_df is not None:
        N = N_LAST_CKPTS
        eval_row_lastn = full_df[full_df["mode"]=="eval"].iloc[-N:]
        x_lastn = eval_row_lastn["x"]
        y_lastn = eval_row_lastn["y"]
        # x_lastn_mean = eval_row_lastn["x"].iloc[-1]
        x_lastn_mean = eval_row_lastn["x"].mean() # <- avg of task loss AND task accuracy
        y_lastn_mean = eval_row_lastn["y"].mean()
        y_lastn_pred = predict_step2(x_lastn_mean, coefficients) # <- re-predict with new x
        rel_error_lastn_mean = (y_lastn_pred - y_lastn_mean) / y_lastn_mean

    # create inset ax
    if no_legend:
        axins = ax.inset_axes([0.48, 0.48, 0.5, 0.5])
    else:
        # axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35]) # top right
        # axins = ax.inset_axes([0.2, 0.63, 0.35, 0.35]) # top left
        axins = ax.inset_axes([0.63, 0.33, 0.35, 0.35]) # bottom right

    for _ax in [ax, axins]:
    # for _ax in [ax]:
        for label in df["size"].unique():
            adf = df[df["size"]==label]
            adf = adf[adf["mode"]=="train"]
            _ax.scatter(adf["x"], adf["y"], color="white", edgecolors=adf["color"], s=7.0, label=label if do_label else None)

    for _ax in [ax, axins]:
        _ax.scatter(x_pred, y_pred_x_pred, marker="^", color="black", s=50)
        _ax.scatter(x, y, marker="x", color="blue", label=f"actual ({run_name}) = {y:0.4f}" if do_label else None, s=50)
        _ax.scatter(x, y_pred, marker="^", color="blue", label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None, s=50)
        _ax.scatter(x_lastn_mean, y_lastn_pred, marker="^", color="green", label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None, s=50)
        _ax.annotate( 
            f"{eval_row['run']}: {rel_error * 100:+.1f}%", (x, y), textcoords="offset points", 
            xytext=(6, 3), ha="left", fontsize=8, color="blue"
        )
        if full_df is not None:
            _ax.annotate( 
                f"{eval_row['run']}: {rel_error_lastn_mean * 100:+.1f}%", (x, y_lastn_mean), textcoords="offset points", 
                xytext=(6, -9), ha="left", fontsize=8, color="green"
            )

    if add_ideal_points:
        plotted_xs = np.linspace(max(2.6, df["x"].max()), 0.01, 100)
    else:
        plotted_xs = np.linspace(df["x"].max(), df["x"].min(), 100)
    plotted_ys = [sigmoid(x_val, *coefficients) for x_val in plotted_xs]

    if full_df is not None:
        # show all target scale checkpoints (exclude the first 30)
        eval_row_all_points = full_df[full_df["mode"]=="eval"][30:]
        x_all = eval_row_all_points["x"]
        y_all = eval_row_all_points["y"]
        for _ax in [ax, axins]:
            # on average of last n checkpoints
            _ax.scatter(x_lastn, y_lastn, color="green", s=1, zorder=10)
            _ax.scatter(x_lastn_mean, y_lastn_mean, marker="x", color="green", s=50, zorder=10)
        axins.scatter(x_all, y_all, marker="x", color="blue", s=1, alpha=0.2)

    for _ax in [ax, axins]:
        _ax.plot(
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

    if do_label and not no_legend:
        ax.legend(loc="upper right", ncols=1)

    # Set the limits for the zoomed region
    x_width, y_width = 0.2, 0.05
    # x_width, y_width = 0.25, 0.1
    x_max = x + x_width
    x_min = x - x_width
    # y_min = y_pred - y_width # <- center on target/actual
    y_min = y - y_width # <- center on prediction
    y_max = y + y_width
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_stacked(df: pd.DataFrame, step2_df: pd.DataFrame, ax: plt.Axes, x_label=None, y_label=None, title=None, do_label=True, full_df=None, do_grey=False):
    mode_colors = {
        "train": "grey",
        "eval": "lightgrey"
    }
    
    for label in df["size"].unique():
        adf = df[df["size"]==label]
        ax.scatter(
            adf["x"], adf["y"],
            color="white", s=7.0, label=label,
            #edgecolors=adf["size"].apply(lambda x: color_map[x]),
            edgecolors=adf["mode"].apply(lambda x: mode_colors[x]) if do_grey else adf["color"],
        )

    step2_df["tokens"] = df["x"]
    eval_row = step2_df[step2_df["mode"]=="eval"].iloc[-1]
    x = eval_row["tokens"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y

    # average last n checkpoints of target scale
    if full_df is not None:
        N = N_LAST_CKPTS
        eval_row_lastn = df[df["mode"]=="eval"].iloc[-N:]
        x_lastn = eval_row_lastn["x"]
        # x_lastn = df[df["size"]==label]
        y_lastn = eval_row_lastn["y"]
        x_lastn_mean = eval_row_lastn["x"].iloc[-1]
        y_lastn_mean = eval_row_lastn["y"].mean()
        rel_error_lastn_mean = (y_pred - y_lastn_mean) / y_lastn_mean
    
        # plot last N points
        ax.scatter(x_lastn, y_lastn, color="green", s=7.0,)
    
    ax.scatter(x, y, marker="x", color="blue", label=f"actual = {y:0.4f}", s=100)
    ax.scatter(x, y_lastn_mean, marker="x", color="green", label=f"mean actual = {y_lastn_mean:0.4f}", s=100)
    ax.scatter(x, y_pred, marker="^", color="black", label=f"predicted = {y_pred:0.4}", s=100)
    ax.annotate(f"{eval_row['run']}: {rel_error * 100:+.1f}%", (x, y), textcoords="offset points", xytext=(30, -30), ha="right", fontsize=10, color="blue")
    if full_df is not None:
        ax.annotate(f"{eval_row['run']}: {rel_error_lastn_mean * 100:+.1f}%", (x, y_lastn_mean), textcoords="offset points", xytext=(30, -30), ha="right", fontsize=10, color="green")

    if do_label:
        ax.legend(loc="lower right", ncols=1)
    ax.set_xlabel(x_label or "tokens")
    ax.set_ylabel(y_label or f"accuracy") # ({task_name})
    ax.set_title(title or "stacked prediction")


def run_stacked(all_configs, tasks=TASKS, limit_ckpts=None, smoothing=None, render_plot=True):
    if render_plot:
        rows = len(tasks.keys())
        fig, axes = plt.subplots(rows, 3, figsize=(20, 5 * rows))

    step1_error = {}
    step2_error = {}
    stacked_error = {}

    for configs in all_configs:
        target = [run_name for run_name in configs if configs[run_name].mode == "eval"][0]
        step1_error[target] = {}
        step2_error[target] = {}
        stacked_error[target] = {}

        for i, (task_name, task) in enumerate(tasks.items()):
            tokens = get_all_data_by_name(configs, ["throughput/total_tokens"])
            bpb_loss = get_all_data_by_name(configs, task['bpb'])
            downstream_loss = get_all_data_by_name(configs, task['score'])
        
            df = get_dataframe(configs, tokens, downstream_loss)
            
            step1_df = get_dataframe(configs, tokens, bpb_loss)
            full_step1_df = step1_df.copy()

            if smoothing == 'moving':
                # trick: moving avg
                step1_df["y"] = step1_df.groupby('run')['y'].transform(lambda x: x.rolling(window=20).mean())
            elif smoothing == 'ema':
                EMA_ALPHA = 0.5
                # trick: do the exponential average
                step1_df["y"] = step1_df.groupby('run')['y'].transform(lambda x: x.ewm(alpha=EMA_ALPHA).mean())

            step1_df = step1_df.groupby('run').apply(lambda rows: rows.iloc[-1], include_groups=False).reset_index()
            step1_df, coefficients = fit_step1(step1_df)

            # Only plot for the final prediction
            do_plot = "1B-10xC" in configs
            inset_axis = "Winogrande" not in task_name

            if do_plot and render_plot:
                plot_step1(
                    step1_df,
                    coefficients,
                    axes[i][0],
                    x_label="tokens",
                    y_label="task loss",
                    title=f"predicting task_loss ({task_name})",
                    do_label=True,
                    full_df=full_step1_df,
                    inset_axis=inset_axis
                )
            
            # step1_error[target][task_name] = get_predicted_error(step1_df)
            step1_error[target][task_name] = get_last_n_predicted_error(step1_df, full_step1_df)

            step2_df = get_dataframe(configs, bpb_loss, downstream_loss)
            full_step2_df = step2_df.copy()

            if smoothing == 'moving':
                # trick: moving avg
                step2_df["x"] = step2_df.groupby('run')['x'].transform(lambda x: x.rolling(window=20).mean())
            elif smoothing == 'ema':
                # trick: do the exponential average
                step2_df["x"] = step2_df.groupby('run')['x'].transform(lambda x: x.ewm(alpha=EMA_ALPHA).mean())

            if limit_ckpts == 'last_n':
                # trick: use last n% of points
                LAST_N_PERCENT = 0.02
                step2_df = step2_df.groupby('run').apply(lambda x: x.iloc[-int(np.ceil(LAST_N_PERCENT*len(x))):], include_groups=False).reset_index()
            elif limit_ckpts == 'final':
                # trick: only use final checkpoint for curve fitting
                # step2_df = step2_df.groupby('run').apply(lambda rows: rows.iloc[:-1], include_groups=False).reset_index()

                # trick: use average of last 10 checkpoints for curve fitting
                step2_df = step2_df.groupby('run').apply(lambda rows: rows.iloc[:-10], include_groups=False).reset_index()
                step2_df["x"] = step2_df.groupby('run')['x'].transform(lambda x: x.mean())
                step2_df["y"] = step2_df.groupby('run')['y'].transform(lambda x: x.mean())

            # Extract the prediction for the task loss
            last_match_idx = step2_df.loc[step2_df["mode"]=="eval"].tail(1).index
            step_1_pred = step1_df[step1_df["mode"]=="eval"].predicted_y.values[0]
        
            step2_df, coefficients = fit_step2(step2_df, BASELINE_BY_TASK_NAME[task_name])
            if do_plot and render_plot:
                plot_step2(
                    step2_df,
                    coefficients,
                    axes[i][1], x_label="task loss",
                    y_label="task accuracy",
                    title=f"predicting task_accuracy ({task_name})",
                    do_label=True, 
                    full_df=full_step2_df,
                    x_pred=step_1_pred
                )

            # Calculate error using the gold task loss
            step2_error[target][task_name] = get_last_n_predicted_error(step2_df, full_step2_df)
            
            # Now, we use the predicted task loss in step 1 for the ACTUAL model loss, such that our final prediction
            # predicts the loss AND accuracy
            step2_df.loc[last_match_idx, 'x'] = step_1_pred
            step2_df, coefficients = fit_step2(step2_df, BASELINE_BY_TASK_NAME[task_name])

            # stacked_error[target][task_name] = get_predicted_error(step2_df)
            stacked_error[target][task_name] = get_last_n_predicted_error(step2_df, full_step2_df)

            if do_plot:
                if limit_ckpts == 'last_n':
                    # trick: use last n% of points
                    LAST_N_PERCENT = 0.02
                    df = df.groupby('run').apply(lambda x: x.iloc[-int(np.ceil(LAST_N_PERCENT*len(x))):], include_groups=False).reset_index()
                elif limit_ckpts == 'final':
                    df = df.groupby('run').apply(lambda rows: rows.iloc[-1], include_groups=False).reset_index()

                if do_plot and render_plot:
                    plot_stacked(df, step2_df, axes[i][2], title="Stacked prediction using raw bpb", do_label=True, full_df=full_step2_df, do_grey=True)

    if render_plot:
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

    return step1_error, step2_error, stacked_error


def get_name_size_length(run_name: str):
    run_name = run_name.split("/")[-1]
    size, length = run_name.split("-")[-2:]
    return run_name, size, length


def prettify(rel_error):
    return f"{rel_error * 100:+.1f}%"


def round_str(n):
    return f"{n:.1f}"


def sci_str(n):
    return f"{(10**2)*n:0.3f}e-02"


def print_error_table(stacked_error: dict[str, dict]):
    from IPython.display import display, Markdown

    targets = stacked_error.keys()

    mkdn = """| Task | """ + ''.join([f'Stacked error ({str(t)}) |' for t in targets])
    mkdn += """\n| --- |""" +  len(targets) * """ --- |"""

    for task in TASKS:
        mkdn += f"\n| {task} |" 
        for target in targets:
            mkdn += f" {prettify(stacked_error[target][task]['rel_error'])} |"

    mkdn += "\n| **Avg signed error** | "
    for target in targets:
        errors = [x['rel_error'] for x in stacked_error[target].values()]
        mkdn += f"**{prettify(np.mean(errors))}** |"

        
    mkdn += "\n| **Avg unsigned error** | "
    for target in targets:
        errors = [x['rel_error'] for x in stacked_error[target].values()]
        mkdn += f"**{prettify(np.mean(np.abs(errors)))}** |"
    
    # print(mkdn)
    display(Markdown(mkdn))


def print_step_error_table(step1_error: dict[str, dict]=None, step2_error: dict[str, dict]=None, stacked_error: dict[str, dict]=None, entry_value: str='rel_error'):
    from IPython.display import display, Markdown

    targets = step1_error.keys()
    tasks = next(iter(step1_error.values())).keys()

    if entry_value == 'y_lastn_std':
        label = 'std dev'
        format = sci_str
    elif entry_value == 'y_lastn_z_score':
        label = 'z-score'
        format = round_str
    elif entry_value == 'y_lastn_std_uniform':
        label = 'uniform std dev'
        format = sci_str
    elif entry_value == 'y_lastn_std_score':
        label = 'standardized score'
        format = round_str
    else:
        label = 'error'
        format = prettify

    mkdn = """| Task | """
    n_cols = 0
    if step1_error is not None:
        n_cols += 1
        mkdn += ''.join([f'Step 1 {label} ({str(t)}) |' for t in targets])
    if step2_error is not None:
        n_cols += 1
        mkdn += ''.join([f'Step 2 ONLY {label} ({str(t)}) |' for t in targets])
    if stacked_error is not None:
        n_cols += 1
        mkdn += ''.join([f'Stacked {label} ({str(t)}) |' for t in targets])
    mkdn += """\n| --- |""" + n_cols * len(targets) * """ --- |"""

    for task in tasks:
        mkdn += f"\n| {task} |" 
        for target in targets:
            for _error_dict in [step1_error, step2_error, stacked_error]:
                if _error_dict is not None:
                    mkdn += f" {format(_error_dict[target][task][entry_value])} |"

    EXCLUDED = ['BoolQ', 'Challenge', 'MMLU-Stem', 'MMLU-Humanities', 'MMLU-Social-Science', 'MMLU-Other']

    # mkdn += f"\n| **Avg signed {label}** | "
    # for target in targets:
    #     for _error_dict in [step1_error, step2_error, stacked_error]:
    #         if _error_dict is not None:
    #             errors = [t[entry_value] for name, t in _error_dict[target].items() if not any([substr in name for substr in EXCLUDED])]
    #             mkdn += f"**{format(np.mean(errors))}** |"

    mkdn += f"\n| **Avg unsigned {label}** (excl. BoolQ, ARC-c) | "
    for target in targets:
        for _error_dict in [step1_error, step2_error, stacked_error]:
            if _error_dict is not None:
                errors = [t[entry_value] for name, t in _error_dict[target].items() if not any([substr in name for substr in EXCLUDED])]
                mkdn += f"**{format(np.mean(np.abs(errors)))}** |"
    
    # print(mkdn)
    display(Markdown(mkdn))


def print_results_table(results, columns=None):
    from IPython.display import display, Markdown

    if columns is None:
        columns = ["Task", "Predicted Task Loss", "Actual Task Loss", "Signed Relative Error"]

    def format_value_with_ci(value, ci):
        # return f"{prettify(value)} ({prettify(ci[0])}, {prettify(ci[1])})"
        return f"{prettify(value)} ± {(100*abs(ci[0] - ci[1])):.1f}%"

    def value_with_ci(value, ci):
        # return f"{value:.3f} ({ci[0]:.3f}, {ci[1]:.3f})"
        return f"{value:.3f} ± {(abs(ci[0] - ci[1])):.3f}"

    def generate_table_row(task, result):
        pred = value_with_ci(result['predictions']['mean'], result['predictions']['ci'])
        actual = value_with_ci(result['actuals']['mean'], result['actuals']['ci'])
        diff = format_value_with_ci(result['difference']['mean'], result['difference']['ci'])
        return f"| {task} | {pred} | {actual} | {diff} |"

    def generate_table_header(columns):
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        return header + "\n" + separator

    table = generate_table_header(columns)
    
    for task, result in results.items():
        table += "\n" + generate_table_row(task, result)
    
    display(Markdown(table))


def plot_std_dev(step1_error, stacked_error, plot_zero_shot=False, all_configs=None):
    import matplotlib.pyplot as plt

    EXCLUDED = ['BoolQ']

    fig, ax = plt.subplots(figsize=(8, 5))

    # few shot
    tasks = [name for name, task in step1_error['1B-10xC'].items() if not any([substr in name for substr in EXCLUDED])]
    step_1_std_dev = [task['y_lastn_std_uniform'] for name, task in step1_error['1B-10xC'].items() if not any([substr in name for substr in EXCLUDED])]
    stacked_std_dev = [task['y_lastn_std_uniform'] for name, task in stacked_error['1B-10xC'].items() if not any([substr in name for substr in EXCLUDED])]

    ax.scatter(step_1_std_dev, stacked_std_dev, s=5, marker="x", color='b')
    for i, task in enumerate(tasks): ax.text(step_1_std_dev[i], stacked_std_dev[i], task, fontsize=6)

    # zero shot
    if plot_zero_shot:
        assert all_configs is not None
        zero_shot_step1_error, zero_shot_step2_error, zero_shot_stacked_error = run_stacked(all_configs, tasks=ZERO_SHOT_TASKS, render_plot=False)
        # print_step_error_table(zero_shot_step1_error, zero_shot_step2_error, zero_shot_stacked_error, entry_value='rel_error_lastn_mean')

        tasks = [name for name, task in zero_shot_step1_error['1B-10xC'].items() if not any([substr in name for substr in EXCLUDED])]
        step_1_std_dev = [task['y_lastn_std_uniform'] for name, task in zero_shot_step1_error['1B-10xC'].items() if not any([substr in name for substr in EXCLUDED])]
        stacked_std_dev = [task['y_lastn_std_uniform'] for name, task in zero_shot_stacked_error['1B-10xC'].items() if not any([substr in name for substr in EXCLUDED])]

        ax.scatter(step_1_std_dev, stacked_std_dev, s=5, marker="x", color='r')
        for i, task in enumerate(tasks): ax.text(step_1_std_dev[i], stacked_std_dev[i], task, fontsize=6)

    ax.set_xlabel("Task Loss Standard Deviation ")
    ax.set_ylabel("Task Accuracy Standard Deviation")
    ax.set_title(f"Std. dev. of last {N_LAST_CKPTS} checkpoints (1B-10xC)")

    texts = ax.texts
    from adjustText import adjust_text
    # _ = adjust_text(ax.texts)