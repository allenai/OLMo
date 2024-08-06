import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.extrapolate_d import (
    ExtrapolateDConfig,
)
from olmo.scaling.scaling_laws.extrapolate_d_lr_correction import (
    get_data_at_n,
    plot_d_scaling_at_n,
)
from olmo.scaling.scaling_laws.utils import (
    validation,
    chinchilla_d_lr_fit,
    grad_chinchilla_d_lr_fit,
)


VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

CONFIGS = {
    'ananya-20m_val-all': {
        'path': 'wandb/tiny-olmo-20M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'train_step_min': 5000,
        'train_step_max': 405000,
        'eval_step_max': 405000,
        'final_loss_tokens': 1698693120000,
        'outlier_threshold': None,
        'dot_size': 5.0,
        'title': 'ananya 20m, val-all',
    },
    'ananya-60m_val-all': {
        'path': 'wandb/tiny-olmo-60M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'train_step_min': 5000,
        'train_step_max': 405000,
        'eval_step_max': 405000,
        'final_loss_tokens': 1698693120000,
        'outlier_threshold': None,
        'dot_size': 5.0,
        'title': 'ananya 60m, val-all',
    },
    'ananya-150m_val-all': {
        'path': 'wandb/tiny-olmo-150M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'train_step_min': 5000,
        'train_step_max': 405000,
        'eval_step_max': 405000,
        'final_loss_tokens': 1698693120000,
        'outlier_threshold': None,
        'dot_size': 5.0,
        'title': 'ananya 150m, val-all',
    },
    # 'ananya-300m_val-all': {
    #     'path': 'wandb/tiny-olmo-300M-rms-norm-adam-eps-1e-8-lr-6e-4-emb-wd_val-all.csv',
    #     'keys': VAL_KEYS,
    #     'train_step_min': 5000,
    #     'train_step_max': 405000,
    #     'eval_step_max': 405000,
    #     'final_loss_tokens': 1698693120000,
    #     'outlier_threshold': None,
    #     'dot_size': 5.0,
    #     'title': 'ananya 300m, val-all',
    # },
    'ananya-700m_val-all': {
        'path': 'wandb/tiny-olmo-700M-rms-norm-adam-eps-1e-8-emb-wd_val-all.csv',
        'keys': VAL_KEYS,
        'train_step_min': 5000,
        'train_step_max': 405000,
        'eval_step_max': 405000,
        'final_loss_tokens': 1698693120000,
        'outlier_threshold': None,
        'dot_size': 5.0,
        'title': 'ananya 700m, val-all',
    },
    'amberish1_val-all': {
        'path': 'wandb/amberish1.csv',
        'keys': VAL_KEYS,
        'train_step_min': 5000,
        'train_step_max': 537605,
        'eval_step_max': 537605,
        'final_loss_tokens': 2252341248000,
        'outlier_threshold': None,
        'dot_size': 5.0,
        'title': 'amberish1, val-all',
    },
    'amberish7_val-all': {
        'path': 'wandb/amberish7.csv',
        'keys': VAL_KEYS,
        'train_step_min': 5000,
        'train_step_max': 478000,
        'eval_step_max': 478000,
        'final_loss_tokens': 2004877312000,
        'outlier_threshold': None,
        'dot_size': 5.0,
        'title': 'amberish7, val-all',
    },
}


def fit_curves(
    configs: Dict[str, ExtrapolateDConfig], output_path: PathOrStr,
):
    for name, config in configs.items():
        train_ds, train_hs, train_ys, eval_ds, eval_hs, eval_ys = get_data_at_n(config)

        plt.figure()

        # Plot actual points
        legends = ["Actual Points (train)"]
        plt.scatter(train_ds, train_ys, color="blue", alpha=0.5, s=config.dot_size, edgecolors="none")
        if len(eval_ds) > 0:
            plt.scatter(eval_ds, eval_ys, color="green", alpha=0.5, s=config.dot_size, edgecolors="none")
            legends.append("Actual Points (eval)")

        plot_d_scaling_at_n(
            train_ds,
            train_hs,
            train_ys,
            eval_ds,
            eval_hs,
            chinchilla_d_lr_fit,
            grad_chinchilla_d_lr_fit,
            config.final_loss_tokens,
            p0=[4.0, 0.8, 2.5, 0.1],
            predict=False,
            color="magenta",
            linewidth=1.0,
        )
        legends.append("Fitted Curve, y = B / d^beta + E + F * h")

        plt.legend(legends, loc="upper right")

        plt.xlabel("Tokens")
        plt.ylabel("CE Loss")
        plt.title(config.title)
        plt.savefig(f"{output_path}/extrapolate_d_lr_{name}.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output folder")

    return parser.parse_args()


def main():
    args = parse_args()
    configs = {key: ExtrapolateDConfig(**value) for key, value in CONFIGS.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(configs, args.output_path)


if __name__ == "__main__":
    main()
