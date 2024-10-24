from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ladder_peteish as ladder
from olmo.scaling.scaling_laws.utils import get_coefficients_huber, chinchilla_n_d_fit, grad_chinchilla_n_d_fit
from typing import Optional

from collections import defaultdict
import csv
from scipy.optimize import curve_fit
from scipy import stats

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
    "ARC-Easy-5shot": {
        "bpb": ["eval/downstream_bpb/arc_easy_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/arc_easy_rc_5shot_acc"],
    },
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
    }


    # "HellaSwag-0shot": {
    #     "bpb": ["eval/downstream_bpb/hellaswag_rc_0shot_bpb_bpb"],
    #     "score": ["eval/downstream/hellaswag_rc_0shot_len_norm"],
    # },

    # "BoolQ-5shot": {
    #     "bpb": ["eval/downstream_bpb/boolq_rc_5shot_bpb_bpb"],
    #     "score": ["eval/downstream/boolq_rc_5shot_acc"],
    # },

    # "Copa-0shot": {
    #     "bpb": ["eval/downstream_bpb/copa_rc_0shot_bpb_bpb"],
    #     "score": ["eval/downstream/copa_rc_0shot_acc"],
    # },
}

TASKS = DEV_TASKS | ALL_TASKS # Dev tasks are for quickly prototyping notebooks

BASELINE_BY_TASK_NAME = {
    'HellaSwag-0shot': 0.25,
    'MMLU-Var': 0.25,
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

    coefficients, pcov = curve_fit(sigmoid, train_xs, train_ys, p0=[baseline - 1.0, 0.9, 3.0, 1.0], maxfev=1000000)
    df["predicted_y"] = df["x"].apply(lambda x: sigmoid(x, *coefficients))

    return df, coefficients


def predict_step2(bpb_loss, coefficients):
    return sigmoid(bpb_loss, *coefficients)


def plot_step1(df: pd.DataFrame, coefficients, ax: plt.Axes, x_label=None, y_label=None, title="Fitting final score", do_label=True):
    eval_row = df[df["mode"]=="eval"].iloc[-1]
    x = eval_row["x"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    run_name = eval_row["run"]

    for label in df["size"].unique():
        adf = df[df["size"]==label]
        ax.scatter(adf["x"], adf["y"], color="white", edgecolors=adf["color"], s=7.0, label=label if do_label else None)
        # ax.scatter(adf["x"], adf["y"], color=adf["color"], s=0.5, label=label if do_label else None)

    ax.scatter(x, y, marker="x", color="blue", label=f"actual ({run_name})= {y:0.4f}" if do_label else None, s=50)
    ax.scatter(x, y_pred, marker="^", color="black", label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None, s=50)
    # ax.annotate(
    #     f"{eval_row['run']}: {rel_error * 100:+.1f}%",
    #     (x, y),
    #     textcoords="offset points",
    #     xytext=(10, 5),
    #     ha="center",
    #     fontsize=10,
    #     color="brown",
    # )

    for params in df["params"].unique():
        plotted_xs = np.linspace(df[df["params"]==params]["x"].max(), df[df["params"]==params]["x"].min(), 100)
        plotted_ys = [chinchilla_n_d_fit([params, x_val], coefficients) for x_val in plotted_xs]

        ax.plot(
            plotted_xs,
            plotted_ys,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5
        )

    # a, b, alpha, beta, E = coefficients
    # A, B = np.exp(a), np.exp(b)
    # ax.text(
    #     x=0.25,
    #     y=0.50,
    #     s=f"L(n, d) = {A:.2f} / n^{alpha:.2f} + {B:.2f} / d^{beta:.2f} + {E:.2f}",
    #     fontsize=10,
    #     transform=ax.transAxes,
    # )

    if do_label:
        ax.legend(loc="upper right", ncols=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_step2(df: pd.DataFrame, coefficients, ax: plt.Axes, x_label=None, y_label=None, title="Fitting final score", add_ideal_points=True, do_label=True):    
    eval_row = df[df["mode"]=="eval"].iloc[-1]
    x = eval_row["x"]
    y = eval_row["y"]
    y_pred = eval_row["predicted_y"]
    rel_error = (y_pred - y) / y
    run_name = eval_row["run"]

    for label in df["size"].unique():
        adf = df[df["size"]==label]
        ax.scatter(adf["x"], adf["y"], color="white", edgecolors=adf["color"], s=7.0, label=label if do_label else None)

    ax.scatter(x, y, marker="x", color="blue", label=f"actual ({run_name}) = {y:0.4f}" if do_label else None, s=50)
    ax.scatter(x, y_pred, marker="^", color="black", label=f"predicted ({run_name}) = {y_pred:0.4}" if do_label else None, s=50)
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


def plot_stacked(df: pd.DataFrame, step2_df: pd.DataFrame, ax: plt.Axes, x_label=None, y_label=None, title=None, do_label=True, do_grey=False):
    mode_colors = {
        "train": "grey",
        "eval": "lightgrey"
    }
    
    for label in df["size"].unique():
        adf = df[df["size"]==label]
        ax.scatter(
            adf["x"],
            adf["y"],
            color="white",
            #edgecolors=adf["size"].apply(lambda x: color_map[x]),
            edgecolors=adf["mode"].apply(lambda x: mode_colors[x]) if do_grey else adf["color"],
            s=7.0,
            label=label
        )

    step2_df["tokens"] = df["x"]
    eval_row = step2_df[step2_df["mode"]=="eval"].iloc[-1]
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
    ax.set_xlabel(x_label or "tokens")
    ax.set_ylabel(y_label or f"accuracy") # ({task_name})
    ax.set_title(title or "stacked prediction")


def get_name_size_length(run_name: str):
    run_name = run_name.split("/")[-1]
    size, length = run_name.split("-")[-2:]
    return run_name, size, length


def prettify(rel_error):
    return f"{rel_error * 100:+.1f}%"


def print_error_table(stacked_error: dict[str, dict]):
    from IPython.display import display, Markdown

    targets = stacked_error.keys()

    mkdn = """| Task | """ + ''.join([f'Stacked error ({str(t)}) |' for t in targets])
    mkdn += """\n| --- |""" +  len(targets) * """ --- |"""

    for task in TASKS:
        mkdn += f"\n| {task} |" 
        for target in targets:
            mkdn += f" {prettify(stacked_error[target][task])} |"

    mkdn += "\n| **Avg signed error** | "
    for target in targets:
        mkdn += f"**{prettify(np.mean(list(stacked_error[target].values())))}** |"

        
    mkdn += "\n| **Avg unsigned error** | "
    for target in targets:
        mkdn += f"**{prettify(np.mean(np.abs(list(stacked_error[target].values()))))}** |"
    
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