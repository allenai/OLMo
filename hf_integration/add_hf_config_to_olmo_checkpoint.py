import argparse

from hf_integration.configuration_olmo import OLMoConfig
from olmo import Olmo


def write_config(checkpoint_dir: str):
    # save config as HF config
    # TODO: add logging
    model = Olmo.from_checkpoint(checkpoint_dir)
    config = OLMoConfig(**model.config.asdict())
    config.save_pretrained(checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, making it easier to load weights as HF models"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Location of OLMo checkpoint.",
    )

    args = parser.parse_args()
    write_config(
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
