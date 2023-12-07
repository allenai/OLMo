import logging
import shutil
from pathlib import Path
from typing import Union

import torch

from olmo.checkpoint import (
    Checkpointer,
    LocalShardedCheckpointer,
    TorchLegacyShardedCheckpointer,
)
from olmo.config import ShardedCheckpointerType, TrainConfig

logger = logging.getLogger(__name__)


def unshard(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sharded_checkpoint_type: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy,
    model_only: bool = False,
) -> None:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig.load(input_dir / "config.yaml", validate_paths=False)
    checkpointer: Checkpointer
    if sharded_checkpoint_type == ShardedCheckpointerType.torch_legacy:
        checkpointer = TorchLegacyShardedCheckpointer(config)
    elif sharded_checkpoint_type == ShardedCheckpointerType.local:
        checkpointer = LocalShardedCheckpointer(config)
    else:
        raise NotImplementedError(sharded_checkpoint_type)

    model_state_dict, optim_state_dict, trainer_state_dict = checkpointer.unshard_checkpoint(
        input_dir,
        load_optimizer_state=not model_only,
        load_trainer_state=not model_only,
    )

    # model
    model_output = str(output_dir / "model.pt")
    logger.info("Saving model state to %s", model_output)
    torch.save(model_state_dict, model_output)
    del model_state_dict

    if not model_only:
        # optimizer
        optim_output = str(output_dir / "optim.pt")
        logger.info("Saving optimizer state to %s", optim_output)
        torch.save(optim_state_dict, optim_output)
        del optim_state_dict

        # trainer
        train_output = str(output_dir / "train.pt")
        logger.info("Saving everything else to %s", train_output)
        torch.save(trainer_state_dict, train_output)
        del trainer_state_dict

    logger.info("Copying config.yaml to %s", output_dir)
    shutil.copy(input_dir / "config.yaml", output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="unshard.py", description="Unshard sharded checkpoints on CPU")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--type",
        choices=list(ShardedCheckpointerType),
        default=ShardedCheckpointerType.torch_legacy,
        help="""The sharded checkpoint type.""",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    unshard(
        args.input_dir,
        args.output_dir,
        sharded_checkpoint_type=args.type,
        model_only=args.model_only,
    )


if __name__ == "__main__":
    main()
