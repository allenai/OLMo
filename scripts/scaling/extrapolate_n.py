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
from olmo.scaling.scaling_laws.utils import validation, chinchilla_fit


VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

CONFIGS = {
    '20m': {
        'path': 'wandb/tiny-olmo-20M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 21266432,
        'label': '20m',
        'color': 'red',
    },
    '60m': {
        'path': 'wandb/tiny-olmo-60M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 59310080,
        'label': '60m',
        'color': 'orange',
    },
    '150m': {
        'path': 'wandb/tiny-olmo-150M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 151879680,
        'label': '150m',
        'color': 'yellow',
    },
    # '300m': {
    #     'path': '../hc-law/wandb/ananya-300m-lr6e-4_val-all.csv',
    #     'keys': VAL_KEYS,
    #     'mode': 'train',
    #     'n': 319980544,
    #     'label': '300m',
    #     'color': 'blue',
    # },
    '700m': {
        'path': 'wandb/tiny-olmo-700M-rms-norm-adam-eps-1e-8-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'n': 758564352,
        'label': '700m',
        'color': 'green',
    },
    '7b': {
        'path': 'wandb/amberish7.csv',
        'keys': VAL_KEYS,
        'mode': 'eval',
        'n': 7000000000, # approximate
        'label': '7b',
        'color': 'cyan',
    }
}


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
        p0=[1e5, -0.5, 2.0],
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
