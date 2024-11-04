"""
Soups OLMo checkpoints.

Example usage:

```bash
    python scripts/olmo_soup.py -c \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-moremath-dclm07-fw2-se-flan/step11931 \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-moremath-dclm07-fw2-se-flan-seed2/step11931 \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-moremath-dclm07-fw2-se-flan-seed3/step11931 \
        -o /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-anneal-from-928646-50B-nowup-moremath-dclm07-fw2-se-flan-soup/step11931
```

Author: Luca Soldaini (@soldni)

"""  # noqa


import argparse
from enum import Enum
from pathlib import Path

import torch
from tqdm import tqdm

from olmo.checkpoint import build_sharded_checkpointer
from olmo.config import TrainConfig
from olmo.safetensors_util import safetensors_file_to_state_dict


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
    parser = argparse.ArgumentParser(description="Soup OLMo checkponts")
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

    print(f"Saving averaged checkpoint to {args.output}")
    # save the averaged checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_average, args.output / "model.pt")

    print("Copying config.yaml")
    # copy the config file
    if (config_path := args.checkpoints[0] / "config.yaml").exists():
        config_path.rename(args.output / "config.yaml")
    print("Done!")


if __name__ == "__main__":
    main()
