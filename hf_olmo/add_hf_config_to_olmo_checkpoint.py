import argparse
import logging
import os

from hf_integration.configuration_olmo import OLMoConfig
from olmo import Olmo

logger = logging.getLogger(__name__)


def write_config(checkpoint_dir: str):
    # save config as HF config
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    model = Olmo.from_checkpoint(checkpoint_dir)

    config_kwargs = model.config.asdict()
    config_kwargs["use_cache"] = True
    config = OLMoConfig(**config_kwargs)

    logger.info(f"Saving HF-compatible config to {os.path.join(checkpoint_dir, 'config.json')}")
    config.save_pretrained(checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Location of OLMo checkpoint.",
    )

    args = parser.parse_args()
    write_config(checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
