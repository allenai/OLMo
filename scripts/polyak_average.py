from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import List

import torch

from olmo.model import OLMo


logger = logging.getLogger(__name__)


def polyak_average(
    run_folder: Path,
    steps: List[int],
    save_folder: Path,
    *,
    verbose: bool = False,
):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if len(steps) <= 0:
        raise ValueError("At least 1 step must be provided")

    checkpoint_paths = []
    for step in steps:
        if (path := run_folder / f"step{step}-unsharded").is_dir():
            checkpoint_paths.append(path)
        elif (path := run_folder / f"step{step}").is_dir():
            checkpoint_paths.append(path)
        else:
            raise ValueError(f"No checkpoint found for step {step}")

    averaged_model = None
    for checkpoint_path in checkpoint_paths:
        if verbose:
            logger.info("Loading model at %s", checkpoint_path)

        model = OLMo.from_checkpoint(checkpoint_path, device=device_name)

        if averaged_model is None:
            averaged_model = model
        else:
            with torch.no_grad():
                for param_name, param in model.named_parameters():
                    averaged_param = averaged_model.get_parameter(param_name)
                    averaged_param.add_(param)

    assert averaged_model is not None
    with torch.no_grad():
        for param_name, param in averaged_model.named_parameters():
            averaged_param = averaged_model.get_parameter(param_name)
            averaged_param.div_(len(checkpoint_paths))

    save_path = save_folder / "model.pt"
    if verbose:
        logger.info("Saving to %s", save_path)
    torch.save(model.state_dict(), str(save_path))


def main():
    logging.basicConfig(encoding="utf-8", level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "run_path",
        type=Path,
        help="Path of a run with multiple checkpoints",
    )
    parser.add_argument(
        "save_path",
        type=Path,
        help="Path where averaged model should be saved",
    )
    parser.add_argument(
        "steps",
        nargs="+",
        type=int,
        default=[],
        help="Steps which should be incorporated in the average",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, show extra information",
    )

    args = parser.parse_args()
    polyak_average(
        args.run_path,
        args.steps,
        args.save_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
