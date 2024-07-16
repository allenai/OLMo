import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.curve_fit import (
    chinchilla_fit,
)
from olmo.scaling.scaling_laws.extrapolate_n import (
    CurveFitConfig,
    get_data_at_d,
    plot_n_scaling_at_d,
)
from olmo.scaling.scaling_laws.utils import validation

VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

CONFIG_BY_N = {
    21266432: {
        'path': '../hc-law/wandb/ananya-20m_val-all.csv',
        'keys': VAL_KEYS,
        'n': 21266432,
        'mode': 'train',
    },
    59310080: {
        'path': '../hc-law/wandb/ananya-60m_val-all.csv',
        'keys': VAL_KEYS,
        'n': 59310080,
        'mode': 'train',
    },
    151879680: {
        'path': '../hc-law/wandb/ananya-150m_val-all.csv',
        'keys': VAL_KEYS,
        'n': 151879680,
        'mode': 'train',
    },
    319980544: {
        'path': '../hc-law/wandb/ananya-300m-lr6e-4_val-all.csv',
        'keys': VAL_KEYS,
        'n': 319980544,
        'mode': 'train',
    },
    758564352: {
        'path': '../hc-law/wandb/ananya-700m-lr6e-4_val-all.csv',
        'keys': VAL_KEYS,
        'n': 758564352,
        'mode': 'eval',
    },
}


def fit_curves(
    config_by_n: Dict[int, CurveFitConfig], output_path: PathOrStr, d: int,
):
    train_ns, train_ys, eval_ns, eval_ys = get_data_at_d(config_by_n, d)

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
    configs = {n: CurveFitConfig(**config) for n, config in CONFIG_BY_N.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(configs, args.output_path, args.d)


if __name__ == "__main__":
    main()
