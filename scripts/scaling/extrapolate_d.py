import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.extrapolate_d import (
    ExtrapolateDConfig,
    get_data_at_n,
    plot_d_scaling_at_n,
)
from olmo.scaling.scaling_laws.utils import (
    validation,
    chinchilla_contaminated_fit,
    chinchilla_fit,
    openai_fit,
)
from olmo.util import StrEnum


VAL_KEYS = [f'eval/{val}/CrossEntropyLoss' for val in validation]

# CONFIGS = {
#     "mup-olmo-128-train": {
#         "path": "wandb_outputs/mup-olmo-128-train.csv",
#         "keys": ["train/CrossEntropyLoss"],  # validation
#         "train_step_min": 150,
#         "train_step_max": 550,
#         "eval_step_max": 750,
#         "final_loss_tokens": 4194304000,
#         "outlier_threshold": None,
#         "dot_size": 5.0,
#         "title": "mup-OLMo-128M, train loss",
#     },
#     "mup-olmo-256-train": {
#         "path": "wandb_outputs/mup-olmo-256-train.csv",
#         "keys": ["train/CrossEntropyLoss"],
#         "train_step_min": 150,
#         "train_step_max": 550,
#         "eval_step_max": 750,
#         "final_loss_tokens": 4194304000,
#         "outlier_threshold": None,
#         "dot_size": 5.0,
#         "title": "mup-OLMo-256M, train loss",
#     },
# }
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
# CONFIGS = {
#     'amber': {
#         'path': 'wandb/amber.csv',
#         'keys': ['eval/all/CrossEntropyLoss'],
#         'train_step_min': 1,
#         'train_step_max': 177,
#         'eval_step_max': 354,
#         'final_loss_tokens': 354,
#         'outlier_threshold': None,
#         'dot_size': 5.0,
#         'title': 'amber',
#     }
# }


class CurveFitMode(StrEnum):
    default: str = "default"

    contaminated: str = "contaminated"

    all: str = "all"


def fit_curves(
    configs: Dict[str, ExtrapolateDConfig], output_path: PathOrStr, mode: CurveFitMode = CurveFitMode.default
):
    for name, config in configs.items():
        train_ds, train_ys, eval_ds, eval_ys = get_data_at_n(config)

        plt.figure()

        # Plot actual points
        legends = ["Actual Points (train)"]
        plt.scatter(train_ds, train_ys, color="blue", alpha=0.5, s=config.dot_size, edgecolors="none")
        if len(eval_ds) > 0:
            plt.scatter(eval_ds, eval_ys, color="green", alpha=0.5, s=config.dot_size, edgecolors="none")
            legends.append("Actual Points (eval)")

        if mode in [CurveFitMode.default, CurveFitMode.all]:
            plot_d_scaling_at_n(
                train_ds,
                train_ys,
                eval_ds,
                openai_fit,
                config.final_loss_tokens,
                p0=[1e16, 0.1, 0],
                color="orange",
                linewidth=1.0,
            )
            legends.append("Fitted Curve, y = (a / x + c)^b (OpenAI)")

            plot_d_scaling_at_n(
                train_ds,
                train_ys,
                eval_ds,
                chinchilla_fit,
                config.final_loss_tokens,
                p0=[1e5, -0.5, 2.0],
                predict=True,
                color="red",
                linewidth=1.0,
            )
            legends.append("Fitted Curve, y = a * x^b + c (Chinchilla)")
            legends.append("Predicted Point (Chinchilla)")

        if mode in [CurveFitMode.contaminated, CurveFitMode.all]:
            plot_d_scaling_at_n(
                train_ds,
                train_ys,
                eval_ds,
                chinchilla_contaminated_fit,
                config.final_loss_tokens,
                p0=[1e5, -0.5, 2.0, 0.0],
                color="magenta",
                linewidth=1.0,
            )
            legends.append("Fitted Curve, y = (a * x^b + c) * (1 - x/d)")

        plt.legend(legends, loc="upper right")

        plt.xlabel("Tokens")
        plt.ylabel("CE Loss")
        plt.title(config.title)
        plt.savefig(f"{output_path}/extrapolate_d_{mode}_{name}.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output folder")
    parser.add_argument(
        "-m", "--mode", type=str, required=False, default="default", help="Options: [default, contaminated, all]"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configs = {key: ExtrapolateDConfig(**value) for key, value in CONFIGS.items()}

    os.makedirs(args.output_path, exist_ok=True)
    fit_curves(configs, args.output_path, args.mode)


if __name__ == "__main__":
    main()
