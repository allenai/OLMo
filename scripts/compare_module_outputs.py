"""
Script for comparing collected outputs of OLMo submodules from 2
different training run steps (of the same or different runs).

This script is useful for identifying where model activations start to differ
within 2 forward passes that should yield identical results. In turn, detecting
regressions can be a lot quicker/easier. 

This script requires that traces containing submodule outputs have been collected
during training. The traces can be saved using
`--module_outputs_save_steps=[<list of step>]`. Be mindful that the saving takes
a lot of storage and is very slow, so collect traces sparingly. If comparing 2
training runs starting from the same checkpoint, a viable approach is to collect
the 2 steps after training resumes. The first step can be used to detect issues
in the forward pass, while if only the second step shows discrepancies then the
backward pass may be the cause of any issues.

Example usage (Aug 2024):
```
python scripts/compare_module_outputs.py test_model/traces/step10 test_model_2/traces/step10
```

If this model produces no output stating diffs (without `--verbose`), then the
outputs between the 2 models are identical. If `mis-matching wte elements: ...`
shows a non-zero value, then the input data of the 2 forward passes being compared
is likely different.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List

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


def compare_module_output(
    base_traces_folder: Path,
    compare_traces_folder: Path,
    module_name: str,
    *,
    include_non_tensor_outputs: bool = True,
    verbose: bool = False,
):
    base_module_input_path = base_traces_folder / f"{module_name}_input.pt"
    base_module_output_path = base_traces_folder / f"{module_name}_output.pt"
    compare_module_input_path = compare_traces_folder / f"{module_name}_input.pt"
    compare_module_output_path = compare_traces_folder / f"{module_name}_output.pt"

    map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    base_input = torch.load(str(base_module_input_path), map_location=map_location)
    compare_input = torch.load(str(compare_module_input_path), map_location=map_location)

    if verbose or base_input.dtype != compare_input.dtype:
        logger.info("%s input dtypes: %s %s", module_name, base_input.dtype, compare_input.dtype)
    if verbose or base_input.shape != compare_input.shape:
        logger.info("%s input shapes: %s %s", module_name, base_input.shape, compare_input.shape)
    if (norm_diff := torch.linalg.vector_norm((compare_input - base_input).float()).item()) != 0.0 or verbose:
        logger.info("%s input norm diff: %.6f", module_name, norm_diff)
    if "wte" in module_name:
        logger.info(
            "%s mis-matching wte elements: %d",
            module_name,
            torch.sum(torch.logical_not(torch.eq(base_input, compare_input))),
        )

    base_output = torch.load(str(base_module_output_path), map_location=map_location)
    compare_output = torch.load(str(compare_module_output_path), map_location=map_location)

    if isinstance(base_output, torch.Tensor):
        if verbose or base_output.dtype != compare_output.dtype:
            logger.info("%s output dtypes: %s %s", module_name, base_output.dtype, compare_output.dtype)
        if (
            norm_diff := torch.linalg.vector_norm((compare_output - base_output).float()).item()
        ) != 0.0 or verbose:
            logger.info("%s output norm diff: %.6f", module_name, norm_diff)
    elif include_non_tensor_outputs:
        logger.info("%s outputs: %s %s", module_name, base_output, compare_output)
    else:
        if verbose:
            logger.info("Base output is type %s, skipping", type(base_output))


def compare_module_outputs(
    base_traces_folder: Path,
    compare_traces_folder: Path,
    *,
    include_non_tensor_outputs: bool = True,
    verbose: bool = False,
):
    base_modules = set(_get_module_names(base_traces_folder))
    compare_modules = set(_get_module_names(compare_traces_folder))

    base_only_modules = base_modules - compare_modules
    if len(base_only_modules) > 0:
        logger.info("Base-only modules: %s", ", ".join(base_only_modules))

    compare_only_modules = compare_modules - base_modules
    if len(compare_only_modules) > 0:
        logger.info("Compare-only modules: %s", ", ".join(compare_only_modules))

    common_modules = base_modules.intersection(compare_modules)
    for module_name in sorted(common_modules):
        compare_module_output(
            base_traces_folder,
            compare_traces_folder,
            module_name,
            include_non_tensor_outputs=include_non_tensor_outputs,
            verbose=verbose,
        )


def main():
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "base_model_traces_path",
        type=Path,
        help="Path where output traces of the base (i.e. reference) model are stored",
    )
    parser.add_argument(
        "compare_model_traces_path",
        type=Path,
        help="Path where output traces of the compare (a.k.a new, different) model are stored",
    )
    parser.add_argument(
        "--include_non_tensor_outputs",
        action="store_true",
        dest="include_non_tensor_outputs",
        help="If set, compare module outputs that are not tensors",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, show extra information",
    )

    args = parser.parse_args()
    compare_module_outputs(
        args.base_model_traces_path,
        args.compare_model_traces_path,
        include_non_tensor_outputs=args.include_non_tensor_outputs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
