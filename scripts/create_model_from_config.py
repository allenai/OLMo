import os

import torch

from hf_olmo.convert_olmo_to_hf import convert_checkpoint, fix_tokenizer
from olmo import ModelConfig, OLMo, TrainConfig


def save_model(config_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    config = TrainConfig.load(config_path)

    config.model.init_device = "cpu"
    model = OLMo(config.model, init_params=True)

    config.save(os.path.join(output_path, "config.yaml"))
    torch.save(model.state_dict(), os.path.join(output_path, "model.pt"))

    fix_tokenizer(checkpoint_dir=output_path)
    convert_checkpoint(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a test model with random weights given a config path")
    parser.add_argument("-c", "--config-path", required=True)
    parser.add_argument("-o", "--output-path", required=True)

    args = parser.parse_args()
    save_model(args.config_path, args.output_path)
