from argparse import ArgumentParser
import logging
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


def compare_module_output(base_traces_folder: Path, compare_traces_folder: Path, module_name: str):
    base_module_input_path = base_traces_folder / f"{module_name}_input.pt"
    base_module_output_path = base_traces_folder / f"{module_name}_output.pt"
    compare_module_input_path = compare_traces_folder / f"{module_name}_input.pt"
    compare_module_output_path = compare_traces_folder / f"{module_name}_output.pt"

    map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    base_input = torch.load(str(base_module_input_path), map_location=map_location)
    compare_input = torch.load(str(compare_module_input_path), map_location=map_location)
    logger.info("Input dtypes: %.2f", base_input.dtype, compare_input.dtype)
    logger.info("Input norm diff: %.2f", torch.linalg.vector_norm((compare_input - base_input).float()))

    base_output = torch.load(str(base_module_output_path), map_location=map_location)
    compare_output = torch.load(str(compare_module_output_path), map_location=map_location)

    if isinstance(base_output, torch.Tensor):
        logger.info("Output dtypes: %.2f", base_output.dtype, compare_output.dtype)
        logger.info("Output norm diff: %.2f", torch.linalg.vector_norm(compare_output - base_output))
    else:
        logger.info("Outputs: %s %s", base_output, compare_output)


def compare_model_outputs(base_traces_folder: Path, compare_traces_folder: Path):
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
        compare_module_output(base_traces_folder, compare_traces_folder, module_name)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "base_model_traces_path",
        type=Path,
        help="Path where traces of the base (i.e. reference) model are stored",
    )
    parser.add_argument(
        "compare_model_traces_path",
        type=Path,
        help="Path where traces of the compare (a.k.a new, different) model are stored",
    )

    args = parser.parse_args()
    compare_model_outputs(args.base_model_traces_path, args.compare_model_traces_path)


if __name__ == "__main__":
    main()
