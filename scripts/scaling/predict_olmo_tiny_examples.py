import argparse
from typing import Dict

import matplotlib.pyplot as plt

from olmo.aliases import PathOrStr
from olmo.scaling.scaling_laws.curve_fit import (
    CurveFitConfig,
    CurveFitMode,
    chinchilla_contaminated_fit,
    chinchilla_fit,
    get_data,
    openai_fit,
    plot_scaling,
)

validation = [
    "c4_en-validation",
    "dolma_books-validation",
    "dolma_common-crawl-validation",
    "dolma_pes2o-validation",
    "dolma_reddit-validation",
    "dolma_stack-validation",
    "dolma_wiki-validation",
    "ice-validation",
    "m2d2_s2orc-validation",
    "pile-validation",
    "wikitext_103-validation",
]

v3_validation = [
    "v3-small-c4_en-validation",
    "v3-small-dolma_books-validation",
    "v3-small-dolma_common-crawl-validation",
    "v3-small-dolma_pes2o-validation",
    "v3-small-dolma_reddit-validation",
    "v3-small-dolma_stack-validation",
    "v3-small-dolma_wiki-validation",
    "v3-small-ice-validation",
    "v3-small-m2d2_s2orc-validation",
    #'v3-small-pile-validation',
    "v3-small-wikitext_103-validation",
]

downstream = [
    "hellaswag_len_norm",
    "winogrande_acc",
    "piqa_len_norm",
    "social_iqa_len_norm",
    "openbook_qa_len_norm",
    "commonsense_qa_len_norm",
    "boolq_acc",
    "copa_acc",
    "arc_easy_acc",
    "arc_challenge_len_norm",
    "sciq_acc",
    "mmlu_social_sciences_var_len_norm",
    "mmlu_humanities_var_len_norm",
    "mmlu_other_var_len_norm",
    "mmlu_stem_mc_5shot_test_len_norm",
    "mmlu_humanities_mc_5shot_len_norm",
    "mmlu_social_sciences_mc_5shot_len_norm",
    "mmlu_stem_var_len_norm",
    "mmlu_other_mc_5shot_test_len_norm",
    "mmlu_humanities_mc_5shot_test_len_norm",
    "mmlu_stem_mc_5shot_len_norm",
    "mmlu_social_sciences_mc_5shot_test_len_norm",
    "mmlu_other_mc_5shot_len_norm",
]


CONFIGS = {
    "mup-olmo-128-train": {
        "path": "wandb_outputs/mup-olmo-128-train.csv",
        "keys": ["train/CrossEntropyLoss"],  # validation
        "train_step_min": 150,
        "train_step_max": 550,
        "eval_step_max": 750,
        "final_loss_tokens": 4194304000,
        "outlier_threshold": None,
        "dot_size": 5.0,
        "title": "mup-OLMo-128M, train loss",
    },
    "mup-olmo-256-train": {
        "path": "wandb_outputs/mup-olmo-256-train.csv",
        "keys": ["train/CrossEntropyLoss"],
        "train_step_min": 150,
        "train_step_max": 550,
        "eval_step_max": 750,
        "final_loss_tokens": 4194304000,
        "outlier_threshold": None,
        "dot_size": 5.0,
        "title": "mup-OLMo-256M, train loss",
    },
}


def fit_curves(
    configs: Dict[str, CurveFitConfig], output_path: PathOrStr, mode: CurveFitMode = CurveFitMode.default
):
    for name, config in configs.items():
        train_xs, train_ys, eval_xs, eval_ys = get_data(config)

        plt.figure()

        # Plot actual points
        legends = ["Actual Points (train)"]
        plt.scatter(train_xs, train_ys, color="blue", alpha=0.5, s=config.dot_size, edgecolors="none")
        if len(eval_xs) > 0:
            plt.scatter(eval_xs, eval_ys, color="green", alpha=0.5, s=config.dot_size, edgecolors="none")
            legends.append("Actual Points (eval)")

        if mode in [CurveFitMode.default, CurveFitMode.all]:
            plot_scaling(
                train_xs,
                train_ys,
                eval_xs,
                openai_fit,
                config.final_loss_tokens,
                p0=[1e16, 0.1, 0],
                color="orange",
                linewidth=1.0,
            )
            legends.append("Fitted Curve, y = (a / x + c)^b (OpenAI)")

            plot_scaling(
                train_xs,
                train_ys,
                eval_xs,
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
            plot_scaling(
                train_xs,
                train_ys,
                eval_xs,
                chinchilla_contaminated_fit,
                config.final_loss_tokens,
                color="magenta",
                linewidth=1.0,
            )
            legends.append("Fitted Curve, y = (a * x^b + c) * (1 - x/d)")

        plt.legend(legends, loc="upper right")

        plt.xlabel("Tokens")
        plt.ylabel("CE Loss")
        plt.title(config.title)
        plt.savefig(f"{output_path}/{mode}_{name}.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", type=str, required=True, help="Output file")
    parser.add_argument(
        "-m", "--mode", type=str, required=False, default="default", help="Options: [default, contaminated, all]"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configs = {key: CurveFitConfig(**value) for key, value in CONFIGS.items()}

    fit_curves(configs, args.output_path, args.mode)


if __name__ == "__main__":
    main()
