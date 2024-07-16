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
from olmo.scaling.scaling_laws.utils import validation, chinchilla_fit

VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

# Mapping from N (model size; non-embedding parameter count) to config
CONFIG_BY_N = {
    21266432: {
        'path': 'wandb/tiny-olmo-20M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'label': '20m',
        'color': 'red',
    },
    59310080: {
        'path': 'wandb/tiny-olmo-60M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'label': '60m',
        'color': 'orange',
    },
    151879680: {
        'path': 'wandb/tiny-olmo-150M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'train',
        'label': '150m',
        'color': 'yellow',
    },
    # 319980544: {
    #     'path': '../hc-law/wandb/ananya-300m-lr6e-4_val-all.csv',
    #     'keys': VAL_KEYS,
    #     'mode': 'eval',
    #     'label': '300m',
    #     'color': 'blue',
    # },
    758564352: {
        'path': 'wandb/tiny-olmo-700M-rms-norm-adam-eps-1e-8-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'mode': 'eval',
        'label': '700m',
        'color': 'green',
    },
}


def fit_curves(
    config_by_n: Dict[int, ExtrapolateNConfig], output_path: PathOrStr,
):
    data_by_d, data_by_n = get_data_forall_d(config_by_n)

    plt.figure()

    plot_n_scaling_forall_d(
        data_by_d,
        data_by_n,
        config_by_n,
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
    config_by_n = {n: ExtrapolateNConfig(**config) for n, config in CONFIG_BY_N.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(config_by_n, args.output_path)


if __name__ == "__main__":
    main()
