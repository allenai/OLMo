"""
Soup OLMo checkpoints.

Author: Luca Soldaini (@soldni)
Email:  lucas@allenai.org

"""


import argparse
import logging
from enum import Enum
from pathlib import Path

import torch
from tqdm import tqdm

from olmo.checkpoint import build_sharded_checkpointer
from olmo.config import TrainConfig
from olmo.safetensors_util import safetensors_file_to_state_dict


def get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class SoupType(Enum):
    uniform = "uniform"


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.exists() and path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)

    if (path / "model.pt").exists():
        return torch.load(path / "model.pt", map_location="cpu", weights_only=True)

    if (path / "model.safetensors").exists():
        safetensors_file_to_state_dict(path / "model.safetensors")

    if (path / "model").exists() and (config_path := (path / "config.yaml")).exists():
        train_config = TrainConfig.load(config_path)
        checkpointer = build_sharded_checkpointer(train_config)
        model_state, _, _ = checkpointer.unshard_checkpoint(
            load_path=str(path), load_optimizer_state=False, load_trainer_state=False
        )
        return model_state

    raise FileNotFoundError(f"Could not find checkpoint in {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Soup OLMo checkpoints")
    parser.add_argument(
        "-c",
        "--checkpoints",
        type=Path,
        required=True,
        nargs="+",
        help="Path to checkpoint(s) to soup",
    )
    parser.add_argument(
        "-s",
        "--soup-type",
        type=SoupType,
        default=SoupType.uniform,
        help=f"Methods for checkpoint souping. Choose from: {', '.join(SoupType.__members__.keys())}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to save the souped checkpoint",
    )
    opts = parser.parse_args()
    return opts


def main():
    logger = get_logger()
    args = parse_args()

    checkpoint_average: dict[str, torch.Tensor] = {}

    for path in tqdm(args.checkpoints, desc="Loading checkpoints", position=0):
        state_dict = load_checkpoint(path)

        if len(checkpoint_average) == 0:
            # initialize checkpoint_average with zeros
            checkpoint_average = {k: torch.zeros_like(v) for k, v in state_dict.items()}

        if any(k not in state_dict for k in checkpoint_average.keys()) or any(
            k not in checkpoint_average for k in state_dict.keys()
        ):
            raise ValueError(f"Checkpoint {path} has different keys")

        for k in tqdm(state_dict, desc="Summing checkpoints", position=1):
            if state_dict[k].shape != checkpoint_average[k].shape:
                raise ValueError(f"Checkpoint {path} has different shape for key {k}")
            checkpoint_average[k] += state_dict[k] / len(args.checkpoints)

        # free memory
        del state_dict

    logger.info(f"Saving averaged checkpoint to {args.output}")
    # save the averaged checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_average, args.output / "model.pt")

    logger.info("Copying config.yaml")
    # copy the config file
    if (config_path := args.checkpoints[0] / "config.yaml").exists():
        with open(config_path, "r") as src_f, open(args.output / "config.yaml", "w") as dst_f:
            dst_f.write(src_f.read())
    logger.info("Done!")


if __name__ == "__main__":
    main()
