import argparse
import json

import numpy as np

from olmo.scaling.scaling_laws.stacked_predictions import (
    DownstreamPredictionFeatures,
    get_downstream_predictions,
)
from olmo.scaling.scaling_laws.utils import FinalConfig

# We only include ce loss and the 6 dolma sets, as these are the sets we can include in the paper
ce_columns = [
    "eval/c4_en-validation/CrossEntropyLoss",
    "eval/dolma_books-validation/CrossEntropyLoss",
    "eval/dolma_common-crawl-validation/CrossEntropyLoss",
    "eval/dolma_pes2o-validation/CrossEntropyLoss",
    "eval/dolma_reddit-validation/CrossEntropyLoss",
    "eval/dolma_stack-validation/CrossEntropyLoss",
    "eval/dolma_wiki-validation/CrossEntropyLoss",
]

mmlu_names = ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]

# TODO: missing mmlu 5shot
tasks = {
    "MMLU-Var": {
        "bpb": [f"eval/downstream_bpb/{n}_var_bpb_bpb" for n in mmlu_names],
        "score": [f"eval/downstream/{n}_var_len_norm" for n in mmlu_names],
        "x_label": "mmlu_var_bpb",
        "y_label": "mmlu_var_score",
    },
    "HellaSwag-5shot": {
        "bpb": ["eval/downstream_bpb/hellaswag_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/hellaswag_rc_5shot_len_norm"],
    },
    "ARC-Easy-5shot": {
        "bpb": ["eval/downstream_bpb/arc_easy_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/arc_easy_rc_5shot_acc"],
    },
    "ARC-Challenge-5shot": {
        "bpb": ["eval/downstream_bpb/arc_challenge_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/arc_challenge_rc_5shot_len_norm"],
    },
    "PiQA-5shot": {
        "bpb": ["eval/downstream_bpb/piqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/piqa_rc_5shot_len_norm"],
    },
    "Winogrande-5shot": {
        "bpb": ["eval/downstream_bpb/winogrande_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/winogrande_rc_5shot_acc"],
    },
    "OpenbookQA-5shot": {
        "bpb": ["eval/downstream_bpb/openbookqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/openbookqa_rc_5shot_len_norm"],
    },
    "SciQ-0shot": {
        "bpb": ["eval/downstream_bpb/sciq_rc_0shot_bpb_bpb"],
        "score": ["eval/downstream/sciq_rc_0shot_acc"],
    },
    "CSQA-5shot": {
        "bpb": ["eval/downstream_bpb/csqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/csqa_rc_5shot_len_norm"],
    },
    "SocialIQA-5shot": {
        "bpb": ["eval/downstream_bpb/socialiqa_rc_5shot_bpb_bpb"],
        "score": ["eval/downstream/socialiqa_rc_5shot_len_norm"],
    },
}


def prettify(rel_error):
    return f"{rel_error * 100:+.1f}%"


def make_parser():
    parser = argparse.ArgumentParser(
        description="Get downstream predictions for a target model, based on model ladder outputs."
    )
    # TODO: Give an example.

    parser.add_argument("config_path", help="Path to config specifying the input and target model runs.")

    parser.add_argument(
        "--save_figures",
        type=str,
        help="Use this to specify a png path for saving the plots for fitted curves. If not specified, plots will not be created",
        default=None,
    )

    parser.add_argument(
        "--use_last_n_percentage",
        type=float,
        default=1.0,
        help="Optionally limit the number of training points to last n percentage",
    )

    parser.add_argument(
        "--feature_type",
        type=str,
        default=DownstreamPredictionFeatures.raw,
        help="{raw, moving_average, exponential_moving_average}",
    )

    parser.add_argument("--feature_kwargs", type=str, default="{}", help="Eg. {'window': 20}")

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # if args.save_figures is not None:
    #     os.makedirs(args.save_figures, exist_ok=True)

    with open(args.config_path) as f:
        configs = json.load(f)
    configs = {name: FinalConfig(**config) for name, config in configs.items()}

    feature_kwargs = json.loads(args.feature_kwargs)

    _, stacked_error = get_downstream_predictions(
        configs,
        tasks,
        args.feature_type,
        args.use_last_n_percentage,
        save_figures=args.save_figures,
        **feature_kwargs,
    )

    mkdn = """| Task | Stacked error |\n| --- | --- |"""

    for task in tasks:
        mkdn += f"\n| {task} |"
        for target in stacked_error:
            mkdn += f"{prettify(stacked_error[target][task])} |"

    mkdn += "\n| **Avg signed error** | "
    for target in stacked_error:
        mkdn += f"**{prettify(np.mean(list(stacked_error[target].values())))}** |"

    mkdn += "\n| **Avg unsigned error** | "
    for target in stacked_error:
        mkdn += f"**{prettify(np.mean(np.abs(list(stacked_error[target].values()))))}** |"
    print(mkdn)


if __name__ == "__main__":
    main()
