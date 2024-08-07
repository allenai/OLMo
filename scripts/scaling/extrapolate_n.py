import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.extrapolate_n import (
    ExtrapolateNConfig,
    get_data_at_d,
    plot_n_scaling_at_d,
)
from olmo.scaling.scaling_laws.utils import chinchilla_fit, validation

VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

CONFIGS = {
    '20m': {
        'path': 'wandb/tiny-olmo-20M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 21266432,
        'label': '20m',
        'color': 'darkred',
    },
    '60m': {
        'path': 'wandb/tiny-olmo-60M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 59310080,
        'label': '60m',
        'color': 'darkorange',
    },
    '150m': {
        'path': 'wandb/tiny-olmo-150M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 151879680,
        'label': '150m',
        'color': 'gold',
    },
    # '300m': {
    #     'path': 'wandb/tiny-olmo-300M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
    #     'keys': VAL_KEYS,
    #     'mode': 'train',
    #     'n': 319980544,
    #     'label': '300m',
    #     'color': 'darkgreen',
    # },
    '700m': {
        'path': 'wandb/tiny-olmo-700M-rms-norm-adam-eps-1e-8-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 681297408,
        'label': '700m',
        'color': 'teal',
    },
    '1b': {
        'path': 'wandb/amberish1.csv',
        'keys': VAL_KEYS,
        'mode': 'eval',
        'n': 1176832000,
        'label': '7b',
        'color': 'darkblue',
    },
    '7b': {
        'path': 'wandb/amberish7.csv',
        'keys': VAL_KEYS,
        'mode': 'eval',
        'n': 6682316800,
        'label': '7b',
        'color': 'darkviolet',
    },
}
# CONFIGS = {
#     '150m': {
#         'path': 'wandb/baseline-150M-1xC_val-all.csv',
#         'keys': VAL_KEYS,
#         'mode': 'train',
#         'n': 151898880,
#         'label': '150m',
#         'color': 'gold',
#     },
#     '300m': {
#         'path': 'wandb/baseline-300M-1xC_val-all.csv',
#         'keys': VAL_KEYS,
#         'mode': 'train',
#         'n': 319980544,
#         'label': '300m',
#         'color': 'darkgreen',
#     },
#     '700m': {
#         'path': 'wandb/baseline-750M-1xC_val-all.csv',
#         'keys': VAL_KEYS,
#         'mode': 'train',
#         'n': 681297408,
#         'label': '750m',
#         'color': 'teal',
#     },
#     '1b': {
#         'path': 'wandb/baseline-1B-1xC_val-all.csv',
#         'keys': VAL_KEYS,
#         'mode': 'eval',
#         'n': 1176832000,
#         'label': '1b',
#         'color': 'darkblue',
#     },
# }


def fit_curves(
    configs: Dict[str, ExtrapolateNConfig], output_path: PathOrStr, d: int,
):
    train_ns, train_ys, eval_ns, eval_ys = get_data_at_d(configs, d)

    plt.figure()

    # Plot actual points
    plt.scatter(train_ns, train_ys, color="blue", s=20.0, edgecolors="none")
    legends = ["Actual Points (train)"]
    if len(eval_ns) > 0:
        plt.scatter(eval_ns, eval_ys, color="green", s=20.0, edgecolors="none")
        legends.append("Actual Points (eval)")

    plot_n_scaling_at_d(
        train_ns,
        train_ys,
        eval_ns,
        chinchilla_fit,
        p0=[1e2, -0.3, 2.0],
        color="red",
        linewidth=1.0,
    )
    legends.append("Fitted Curve, y = a * x^b + c (Chinchilla)")

    plt.legend(legends, loc="upper right")

    plt.xlabel("Model size (n)")
    plt.ylabel("CE Loss")
    plt.title(f"Extrapolate across model size @ tokens = {d}")
    plt.savefig(f"{output_path}/extrapolate_n_at_{d}.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output folder")
    parser.add_argument("-d", "--d", type=int, required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    configs = {name: ExtrapolateNConfig(**config) for name, config in CONFIGS.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(configs, args.output_path, args.d)


if __name__ == "__main__":
    main()
