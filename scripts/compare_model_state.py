from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Any, List

import torch


logger = logging.getLogger(__name__)


def _get_module_names(checkpoint_traces_folder: Path) -> List[str]:
    module_names = []
    for trace_file in checkpoint_traces_folder.iterdir():
        trace_file_name = trace_file.name
        if trace_file_name.endswith("_input.pt"):
            module_name = trace_file_name.removesuffix("_input.pt")
        elif trace_file_name.endswith("_output.pt"):
            module_name = trace_file_name.removesuffix("_output.pt")
        else:
            logger.warning("Cannot get parameter from file %s, skipping", trace_file_name)

        module_names.append(module_name)

    return module_names


def compare_model_param(
    base_model: Any,
    compare_model: Any,
    param_name: str,
    *,
    include_non_tensor_state: bool = False,
    verbose: bool = False,
):
    base_value = base_model[param_name]
    compare_value = compare_model[param_name]

    if isinstance(base_value, torch.Tensor):
        if verbose or base_value.dtype != compare_value.dtype:
            logger.info("%s input dtypes: %s %s", param_name, base_value.dtype, compare_value.dtype)
        if verbose or base_value.shape != compare_value.shape:
            logger.info("%s input shapes: %s %s", param_name, base_value.shape, compare_value.shape)
        if (norm_diff := torch.linalg.vector_norm((compare_value - base_value).float()).item()) != 0.0 or verbose:
            logger.info("%s input norm diff: %.6f", param_name, norm_diff)
    elif include_non_tensor_state:
        logger.info("%s params: %s %s", param_name, base_value, compare_value)
    else:
        logger.info("Base output is type %s, skipping", type(base_value))


def compare_model_state(
    base_model_folder: Path,
    compare_model_folder: Path,
    *,
    include_non_tensor_state: bool = False,
    verbose: bool = False,
):
    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_model = torch.load(str(base_model_folder / "model.pt"), map_location=map_location)
    compare_model = torch.load(str(compare_model_folder / "model.pt"), map_location=map_location)

    base_model_params = set(_get_module_names(base_model.keys()))
    compare_model_params = set(_get_module_names(compare_model.keys()))

    base_only_model_params = base_model_params - compare_model_params
    if len(base_only_model_params) > 0:
        logger.info("Base-only model params: %s", ", ".join(base_only_model_params))

    compare_only_model_params = compare_model_params - base_model_params
    if len(compare_only_model_params) > 0:
        logger.info("Compare-only model params: %s", ", ".join(compare_only_model_params))

    common_params = base_only_model_params.intersection(compare_only_model_params)
    for param_key in sorted(common_params):
        compare_model_param(
            base_model,
            compare_model,
            param_key,
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
        include_non_tensor_state=args.include_non_tensor_state,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
