import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.curve_fit import (
    chinchilla_fit,
)
from olmo.scaling.scaling_laws.extrapolate_n_forall_d import (
    CurveFitConfig,
    get_data_forall_d,
    plot_n_scaling_forall_d,
)
from olmo.scaling.scaling_laws.utils import validation

VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

CONFIG_BY_N = {
    21266432: {
        'path': '../hc-law/wandb/ananya-20m_val-all.csv',
        'keys': VAL_KEYS,
        'n': 21266432,
        'mode': 'train',
        'label': '20m',
        'color': 'red',
    },
    59310080: {
        'path': '../hc-law/wandb/ananya-60m_val-all.csv',
        'keys': VAL_KEYS,
        'n': 59310080,
        'mode': 'train',
        'label': '60m',
        'color': 'orange',
    },
    151879680: {
        'path': '../hc-law/wandb/ananya-150m_val-all.csv',
        'keys': VAL_KEYS,
        'n': 151879680,
        'mode': 'train',
        'label': '150m',
        'color': 'yellow',
    },
    319980544: {
        'path': '../hc-law/wandb/ananya-300m-lr6e-4_val-all.csv',
        'keys': VAL_KEYS,
        'n': 319980544,
        'mode': 'eval',
        'label': '300m',
        'color': 'blue',
    },
    758564352: {
        'path': '../hc-law/wandb/ananya-700m-lr6e-4_val-all.csv',
        'keys': VAL_KEYS,
        'n': 758564352,
        'mode': 'eval',
        'label': '700m',
        'color': 'green',
    },
}


def fit_curves(
    config_by_n: Dict[int, CurveFitConfig], output_path: PathOrStr,
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
    config_by_n = {n: CurveFitConfig(**config) for n, config in CONFIG_BY_N.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(config_by_n, args.output_path)


if __name__ == "__main__":
    main()
