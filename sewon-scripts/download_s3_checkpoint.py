import argparse
import logging
import os
import re
import shutil
import tempfile
from hashlib import md5
from typing import Iterable, Optional
from urllib.parse import urlparse

import torch
from omegaconf import OmegaConf as om
from tqdm import tqdm

from hf_olmo.configuration_olmo import OLMoConfig
from hf_olmo.modeling_olmo import OLMoForCausalLM
from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast
from olmo import ModelConfig, Tokenizer, TrainConfig
from olmo.checkpoint import build_sharded_checkpointer
from olmo.util import _get_s3_client

logger = logging.getLogger(__name__)

HF_FILENAMES = {
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
}


def longest_common_prefix(strs: Iterable[str]) -> str:
    """
    Finds the longest common prefix among a list of strings.
    """
    if not strs:
        return ""

    # Find the shortest string in the list
    shortest_str = min(strs, key=len)

    for i, char in enumerate(shortest_str):
        for other_str in strs:
            if other_str[i] != char:
                return shortest_str[:i]

    return shortest_str


def write_config(checkpoint_dir: str):
    # save config as HF config

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    config_path = os.path.join(checkpoint_dir, "config.yaml")
    model_config = ModelConfig.load(config_path, key="model")
    config_kwargs = model_config.asdict()
    config_kwargs["use_cache"] = True
    config = OLMoConfig(**config_kwargs)

    logger.info(f"Saving HF-compatible config to {os.path.join(checkpoint_dir, 'config.json')}")
    config.save_pretrained(checkpoint_dir)


def write_model(checkpoint_dir: str, ignore_olmo_compatibility: bool = False):
    # For device_map = "auto", etc. the models are loaded in a way that start_prefix is not computed correctly.
    # So, we explicitly store the model with the expected prefix.

    old_model_path = os.path.join(checkpoint_dir, "model.pt")
    new_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    state_dict = torch.load(old_model_path, map_location="cpu")

    # this takes care of the case where the model was saved with a different prefix,
    # typically due to unsharding.
    common_prefix = longest_common_prefix(state_dict.keys())
    new_state_dict = {
        key.replace(common_prefix, f"{OLMoForCausalLM.base_model_prefix}.transformer."): val
        for key, val in state_dict.items()
    }
    torch.save(new_state_dict, new_model_path)

    if ignore_olmo_compatibility:
        os.remove(old_model_path)


def write_tokenizer(checkpoint_dir: str):
    tokenizer_raw = Tokenizer.from_checkpoint(checkpoint_dir)
    tokenizer = OLMoTokenizerFast(
        tokenizer_object=tokenizer_raw.base_tokenizer,
        truncation=tokenizer_raw.truncate_direction,
        max_length=tokenizer_raw.truncate_to,
        eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False),
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.pad_token_id = tokenizer_raw.pad_token_id
    tokenizer.eos_token_id = tokenizer_raw.eos_token_id

    tokenizer.save_pretrained(checkpoint_dir)


def convert_checkpoint(checkpoint_dir: str, ignore_olmo_compatibility: bool = False):
    print("Converting checkpoint to HF format...")
    write_config(checkpoint_dir)

    print("Saving model to checkpoint...")
    write_model(checkpoint_dir, ignore_olmo_compatibility=ignore_olmo_compatibility)

    print("Saving tokenizer to checkpoint...")
    write_tokenizer(checkpoint_dir)

    # Cannot remove it before writing the tokenizer
    if ignore_olmo_compatibility:
        os.remove(os.path.join(checkpoint_dir, "config.yaml"))


def fix_tokenizer(checkpoint_dir: str, tokenizer_name_or_path: Optional[str] = None):
    path = os.path.join(checkpoint_dir, "config.yaml")
    conf = om.load(path)

    print("Saving tokenizer to checkpoint...")

    tokenizer_name_or_path = str(tokenizer_name_or_path or conf["tokenizer"]["identifier"])  # pyright: ignore

    try:
        if os.path.exists(tokenizer_name_or_path):
            Tokenizer.from_file(tokenizer_name_or_path)
        else:
            Tokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception as e:
        # the tokenizer is not valid
        logger.error(f"Invalid tokenizer: {tokenizer_name_or_path}. Error: {e}")
        raise e

    conf["tokenizer"]["identifier"] = tokenizer_name_or_path  # pyright: ignore

    if tokenizer_name_or_path == "allenai/gpt-neox-olmo-dolma-v1_5" or tokenizer_name_or_path.endswith(
        "allenai_eleuther-ai-gpt-neox-20b-pii-special.json"
    ):
        conf["model"]["eos_token_id"] = 50279  # pyright: ignore

    om.save(conf, path)


def download_s3_directory(bucket_name, prefix, local_dir, ignore):
    # Create S3 client
    s3_client = _get_s3_client("s3")

    re_ignore = re.compile(ignore) if ignore else None

    # List objects within the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    # Create a list to hold all the files to download
    files_to_download = []
    for page in pages:
        for obj in page.get("Contents", []):
            if re_ignore and re_ignore.search(obj["Key"]):
                continue
            files_to_download.append(obj["Key"])

    # Initialize the progress bar
    for s3_key in tqdm(files_to_download, desc="Downloading files"):
        # Construct the full local path
        local_file_path = os.path.join(local_dir, os.path.relpath(s3_key, prefix))
        local_file_dir = os.path.dirname(local_file_path)

        # Ensure local directory exists
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        # Download the file
        s3_client.download_file(bucket_name, s3_key, local_file_path)


def main():
    checkpoint_dir = args.checkpoint_dir
    parsed_dir = urlparse(checkpoint_dir)

    assert parsed_dir.scheme in ["s3", ""], "Only s3 and local paths are supported."

    if os.path.exists(checkpoint_dir):
        return checkpoint_dir

    local_dir = os.path.join("local_checkpoints", "/".join(checkpoint_dir.split("/")[-2:]))
    print ("Saving to", local_dir)
    
    if not os.path.exists(local_dir):
        try:
            os.makedirs(local_dir, exist_ok=True)
            print(f"Downloading checkpoint to {local_dir}...")
            download_s3_directory(
                bucket_name=parsed_dir.netloc,
                prefix=parsed_dir.path.lstrip("/"),
                local_dir=local_dir,
            )
        except Exception as e:
            logger.error(f"Error downloading checkpoint: {e}")
            shutil.rmtree(local_dir)
            raise e

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, and creates pytorch_model.bin, "
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Location of OLMo checkpoint.",
        required=True,
    )
    args = parser.parse_args()
    main(args)