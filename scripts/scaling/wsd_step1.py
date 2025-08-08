# python scripts/scaling/wsd_step1.py -c scripts/scaling/const.json -k hellaswag_val_5shot -o figure/peteish-moreeval-wsd/wsd_step1_hellaswag.png
# python scripts/scaling/wsd_step1.py -c scripts/scaling/const.json -k mmlu_avg_test_5shot -o figure/peteish-moreeval-wsd/wsd_step1_mmlu.png

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from olmo.scaling.scaling_laws.fitting_functions import (
    chinchilla_n_d_fit,
    get_coefficients_huber,
    grad_chinchilla_n_d_fit,
)
from olmo.scaling.scaling_laws.utils import (
    ExtrapolateNConfig,
    get_ax,
    get_data_by_name,
    parse_args,
)


MARKERS = {"0.5xC": "D", "1xC": "s", "2xC": "P", "5xC": "p", "10xC": "*"}
D_START_STEP_BY_NAME = {
    "190m-0.5xC": 72625 * 0.9 * 0.05,
    "190m-1xC": 72625 * 0.9 * 0.1,
    "190m-2xC": 72625 * 0.9 * 0.2,
    "190m-5xC": 72625 * 0.9 * 0.5,
    "190m-10xC": 72625 * 0.9 * 1.0,
    "370m-0.5xC": 94427 * 0.9 * 0.05,
    "370m-1xC": 94427 * 0.9 * 0.1,
    "370m-2xC": 94427 * 0.9 * 0.2,
    "370m-5xC": 94427 * 0.9 * 0.5,
    "370m-10xC": 94427 * 0.9 * 1.0,
    "760m-0.5xC": 115706 * 0.9 * 0.05,
    "760m-1xC": 115706 * 0.9 * 0.1,
    "760m-2xC": 115706 * 0.9 * 0.2,
    "760m-5xC": 115706 * 0.9 * 0.5,
    "760m-10xC": 115706 * 0.9 * 1.0,
    "1.3b-0.5xC": 162694 * 0.9 * 0.05,
    "1.3b-1xC": 162694 * 0.9 * 0.1,
    "1.3b-2xC": 162694 * 0.9 * 0.2,
    "1.3b-5xC": 162694 * 0.9 * 0.5,
    "1.3b-10xC": 162694 * 0.9 * 1.0,
    "3.2b-0.5xC": 201524 * 0.9 * 0.05,
    "3.2b-1xC": 201524 * 0.9 * 0.1,
    "3.2b-2xC": 201524 * 0.9 * 0.2,
    "3.2b-5xC": 201524 * 0.9 * 0.5,
    "3.2b-10xC": 201524 * 0.9 * 1.0,
}


def main():
    args = parse_args()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    with open(args.config_path) as f:
        configs = json.load(f)
        configs = {name: ExtrapolateNConfig(**config) for name, config in configs.items()}

    data_by_name = get_data_by_name(configs, args.keys, min_step=5000)

    # gather data
    train_ndhs, train_ys = [], []
    for name, data in data_by_name.items():
        config = configs[name]
        if config.mode == "train":
            train_ndhs += [[n, d, h] for n, d, h in zip(data["ns"], data["ds"], data["hs"])]
            train_ys += data["ys"]

    # fit the parameters
    coefficients = get_coefficients_huber(
        train_ndhs,
        train_ys,
        chinchilla_n_d_fit,
        grad_chinchilla_n_d_fit,
        p0=[4.0, 4.0, 0.3, 0.3, 0.5],
        bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)],
    )
    a, b, alpha, beta, E = coefficients
    A, B = np.exp(a), np.exp(b)

    # make predictions
    predicted_data_by_name = {}
    for name, data in data_by_name.items():
        config = configs[name]
        predicted_data_by_name[name] = {
            "ns": data["ns"],
            "ds": data["ds"],
            "ys": [chinchilla_n_d_fit([n, d], coefficients) for n, d in zip(data["ns"], data["ds"])],
        }

    # plot the actual data
    for name, data in data_by_name.items():
        config = configs[name]
        ax.scatter(
            data["ds"], data["ys"], color="white", edgecolors=config.color, label=config.label, s=5, alpha=0.25
        )

    # plot the fitted curve
    for name, data in predicted_data_by_name.items():
        config = configs[name]
        ax.plot(
            data["ds"],
            data["ys"],
            color=config.color,
            linestyle="--",
            linewidth=1.0,
            label=f'{config.label} ({"fitted" if config.mode == "train" else "predicted"})',
        )



    # decay
    decay_configs = {}
    for name, config in configs.items():
        for length in ['0.5xC', '1xC', '2xC', '5xC', '10xC']:
            decay_name = f'{name}-{length}'
            decay_path = config.path.replace("const", "decay").replace("10xC", length)
            decay_config = copy.deepcopy(config)
            decay_config.path = decay_path
            decay_config.label = decay_name
            decay_configs[decay_name] = decay_config

    decay_data_by_name = {}
    for name, config in decay_configs.items():
        decay_data_by_name = {
            **decay_data_by_name,
            **get_data_by_name({name: config}, args.keys, min_step=int(D_START_STEP_BY_NAME[name])),
        }

    # gather data
    delta_by_name = {}
    for name, data in decay_data_by_name.items():
        delta = data['ys'][0] - data['ys'][-1]
        delta_by_name[name] = delta

    # fit the parameters
    delta = np.mean([
        delta
        for name, delta in delta_by_name.items()
        if decay_configs[name].mode == "train"
    ])

    # # make predictions
    # final_prediction_by_name = {}
    # for name, config in decay_configs.items():
    #     n = config.n
    #     d = decay_data_by_name[name]["ds"][-1]
    #     y = chinchilla_n_d_fit([n, d], coefficients)
    #     y = y - delta
    #     final_prediction_by_name[name] = y

    # plot the actual and predicted data
    for name, data in decay_data_by_name.items():
        config = decay_configs[name]

        # plot the actual data
        d = data["ds"][-1]
        y = data["ys"][-1]
        l = name.split("-")[-1]
        ax.scatter(d, y, color=config.color, marker=MARKERS[l], s=10)

        # plot the predicted decay line
        d_start = data["ds"][0]
        d_end = data["ds"][-1]
        y_start = chinchilla_n_d_fit([config.n, d_start], coefficients)
        y_end = y_start - delta
        ax.scatter(d, y_end, color="black", marker="x", s=10)
        ax.plot([d_start, d_end], [y_start, y_end], color="black", linestyle="--", linewidth=1.0)

        rel_error = (y_end - y) / y
        ax.annotate(
            f"{abs(rel_error) * 100:.1f}%",
            (d, y_end),
            textcoords="offset points",
            xytext=(4, 0),
            ha="left",
            va="center",
            fontsize=7,
            color=config.color,
        )

    ax.legend(loc="upper right", ncols=2, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Tokens (D)")
    ax.set_ylabel(f"CE loss, {args.key if args.key != '' else args.keys}")
    # ax.set_title(args.key, fontsize=10)
    # plt.suptitle(
    #     f"{args.key}\nL(N, D, H) = {A:.2f} / N^{alpha:.2f} + {B:.2f} / D^{beta:.2f} + {E:.2f}",
    #     fontsize=8,
    # )
    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
