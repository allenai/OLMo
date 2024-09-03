import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.extrapolate_n import (
    ExtrapolateNConfig,
    get_data_forall_d,
    plot_n_scaling_forall_d,
)
from olmo.scaling.scaling_laws.utils import chinchilla_fit, validation

VAL_KEYS = [f"eval/{val}/CrossEntropyLoss" for val in validation]

CONFIGS = {
    "20m": {
        "path": "wandb/tiny-olmo-20M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv",
        "keys": VAL_KEYS,
        "mode": "train",
        "n": 21266432,
        "label": "20m",
        "color": "darkred",
    },
    "60m": {
        "path": "wandb/tiny-olmo-60M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv",
        "keys": VAL_KEYS,
        "mode": "train",
        "n": 59310080,
        "label": "60m",
        "color": "darkorange",
    },
    "150m": {
        "path": "wandb/tiny-olmo-150M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv",
        "keys": VAL_KEYS,
        "mode": "train",
        "n": 151879680,
        "label": "150m",
        "color": "gold",
    },
    # '300m': {
    #     'path': 'wandb/tiny-olmo-300M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
    #     'keys': VAL_KEYS,
    #     'mode': 'train',
    #     'n': 319980544,
    #     'label': '300m',
    #     'color': 'darkgreen',
    # },
    "700m": {
        "path": "wandb/tiny-olmo-700M-rms-norm-adam-eps-1e-8-emb-wd_val-all.csv",
        "keys": VAL_KEYS,
        "mode": "train",
        "n": 681297408,
        "label": "700m",
        "color": "teal",
    },
    "1b": {
        "path": "wandb/amberish1.csv",
        "keys": VAL_KEYS,
        "mode": "eval",
        "n": 1176832000,
        "label": "1b",
        "color": "darkblue",
    },
    "7b": {
        "path": "wandb/amberish7.csv",
        "keys": VAL_KEYS,
        "mode": "eval",
        "n": 6682316800,
        "label": "7b",
        "color": "darkviolet",
    },
}


def fit_curves(
    configs: Dict[str, ExtrapolateNConfig],
    output_path: PathOrStr,
):
    data_by_d, data_by_n = get_data_forall_d(configs)

    plt.figure()

    plot_n_scaling_forall_d(
        data_by_d,
        data_by_n,
        configs,
        chinchilla_fit,
        p0=[1e5, -0.5, 2.0],
    )

    plt.legend(loc="upper right")

    plt.xlabel("Tokens (d)")
    plt.ylabel("CE Loss")
    plt.title(f"Extrapolate across model size at all ckpts")
    plt.savefig(f"{output_path}/extrapolate_n_forall_d.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output folder")

    return parser.parse_args()


def main():
    args = parse_args()
    configs = {name: ExtrapolateNConfig(**config) for name, config in CONFIGS.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(configs, args.output_path)


if __name__ == "__main__":
    main()
