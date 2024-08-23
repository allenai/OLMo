"""
Convenience script to compare the model state of 2 unsharded OLMo checkpoints.

The intended main usage of this script is to find which parameters of 2 models
are identical.

Example usage (Aug 2024):
python scripts/compare_model_state.py \
    --base_model_path test_model/step0-unsharded \
    --compare_model_path test_model/step0-unsharded
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def compare_model_param(
    base_model: Any,
    compare_model: Any,
    param_name: str,
    *,
    norm_order: int = 1,
    include_non_tensor_state: bool = False,
    verbose: bool = False,
):
    base_value = base_model[param_name]
    compare_value = compare_model[param_name]

    if isinstance(base_value, torch.Tensor):
        if verbose or base_value.dtype != compare_value.dtype:
            logger.info("%s param dtypes: %s %s", param_name, base_value.dtype, compare_value.dtype)
        if verbose or base_value.shape != compare_value.shape:
            logger.info("%s param shapes: %s %s", param_name, base_value.shape, compare_value.shape)
        if (
            norm_diff := torch.linalg.vector_norm((compare_value - base_value).float(), ord=norm_order).item()
        ) != 0.0 or verbose:
            logger.info("%s param norm diff: %.6f", param_name, norm_diff)
    elif include_non_tensor_state:
        logger.info("%s params: %s %s", param_name, base_value, compare_value)
    else:
        logger.info("Base output is type %s, skipping", type(base_value))


def compare_model_state(
    base_model_folder: Path,
    compare_model_folder: Path,
    *,
    norm_order: int = 1,
    include_non_tensor_state: bool = False,
    verbose: bool = False,
):
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info("Loading base model")
    base_model = torch.load(str(base_model_folder / "model.pt"), map_location=map_location)
    logger.info("Loading compare model")
    compare_model = torch.load(str(compare_model_folder / "model.pt"), map_location=map_location)
    logger.info("Loading complete")

    base_model_params = set(base_model.keys())
    compare_model_params = set(compare_model.keys())

    base_only_model_params = base_model_params - compare_model_params
    if len(base_only_model_params) > 0:
        logger.info("Base-only model params: %s", ", ".join(base_only_model_params))

    compare_only_model_params = compare_model_params - base_model_params
    if len(compare_only_model_params) > 0:
        logger.info("Compare-only model params: %s", ", ".join(compare_only_model_params))

    common_params = base_model_params.intersection(compare_model_params)
    for param_key in sorted(common_params):
        compare_model_param(
            base_model,
            compare_model,
            param_key,
            norm_order=norm_order,
            include_non_tensor_state=include_non_tensor_state,
            verbose=verbose,
        )


def main():
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "base_model_path",
        type=Path,
        help="Path where the base (i.e. reference) model is stored",
    )
    parser.add_argument(
        "compare_model_path",
        type=Path,
        help="Path where the compare (a.k.a new, different) model is stored",
    )
    parser.add_argument(
        "--norm_order",
        type=int,
        default=1,
        help="Order of the norm used for comparing model states",
    )
    parser.add_argument(
        "--skip_non_tensor_state",
        action="store_false",
        dest="include_non_tensor_state",
        help="If set, do not compare a model state that is not a tensor",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, show extra information",
    )

    args = parser.parse_args()
    compare_model_state(
        args.base_model_path,
        args.compare_model_path,
        norm_order=args.norm_order,
        include_non_tensor_state=args.include_non_tensor_state,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
