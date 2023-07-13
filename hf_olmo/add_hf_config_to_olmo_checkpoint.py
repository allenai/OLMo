import argparse
import logging
import os
import shutil

from hf_olmo.configuration_olmo import OLMoConfig
from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast
from olmo import ModelConfig

logger = logging.getLogger(__name__)


def write_config(checkpoint_dir: str):
    # save config as HF config
    from cached_path import cached_path

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))
    model_config = ModelConfig.load(config_path, key="model")
    config_kwargs = model_config.asdict()
    config_kwargs["use_cache"] = True
    config = OLMoConfig(**config_kwargs)

    logger.info(f"Saving HF-compatible config to {os.path.join(checkpoint_dir, 'config.json')}")
    config.save_pretrained(checkpoint_dir)

    tokenizer = OLMoTokenizerFast.from_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def download_remote_checkpoint_and_add_hf_config(checkpoint_dir: str, local_dir: str):
    from cached_path import cached_path

    model_name = os.path.basename(checkpoint_dir)
    local_model_path = os.path.join(local_dir, model_name)
    os.makedirs(local_model_path, exist_ok=True)

    model_files = ["model.pt", "config.yaml"]  # , "optim.pt", "other.pt"]
    for filename in model_files:
        final_location = os.path.join(local_model_path, filename)
        if not os.path.exists(final_location):
            remote_file = os.path.join(checkpoint_dir, filename)
            logger.debug(f"Downloading file {filename}")
            cached_file = cached_path(remote_file)
            shutil.copy(cached_file, final_location)
            logger.debug(f"File at {final_location}")
        else:
            logger.info(f"File already present at {final_location}")

    write_config(local_model_path)
    return local_model_path


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
