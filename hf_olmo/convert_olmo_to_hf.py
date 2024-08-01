import argparse
from contextlib import contextmanager
from hashlib import md5
import logging
import os
import shutil
import tempfile
from typing import Generator, Optional
from urllib.parse import urlparse


import boto3
import huggingface_hub
from olmo.checkpoint import OlmoCoreCheckpointer
import torch
from tqdm import tqdm
from omegaconf import OmegaConf as om

from hf_olmo.configuration_olmo import OLMoConfig
from hf_olmo.modeling_olmo import OLMoForCausalLM
from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast
from olmo import ModelConfig, Tokenizer, TrainConfig

logger = logging.getLogger(__name__)


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

    state_dict = torch.load(old_model_path)
    new_state_dict = {f"{OLMoForCausalLM.base_model_prefix}.{key}": val for key, val in state_dict.items()}
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
    write_config(checkpoint_dir)
    write_model(checkpoint_dir, ignore_olmo_compatibility=ignore_olmo_compatibility)
    write_tokenizer(checkpoint_dir)

    # Cannot remove it before writing the tokenizer
    if ignore_olmo_compatibility:
        os.remove(os.path.join(checkpoint_dir, "config.yaml"))


def fix_tokenizer(checkpoint_dir: str, tokenizer_name_or_path: Optional[str] = None):
    path = os.path.join(checkpoint_dir, "config.yaml")
    conf = om.load(path)

    tokenizer_name_or_path = str(tokenizer_name_or_path or conf["tokenizer"]["identifier"])  # pyright: ignore

    try:
        Tokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise e

    conf["tokenizer"]["identifier"] = tokenizer_name_or_path  # pyright: ignore

    if tokenizer_name_or_path == "allenai/gpt-neox-olmo-dolma-v1_5":
        conf["model"]["eos_token_id"] = 50279  # pyright: ignore

    om.save(conf, path)


def download_s3_directory(bucket_name, prefix, local_dir):
    # Create S3 client
    s3_client = boto3.client("s3")

    # List objects within the given prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    # Create a list to hold all the files to download
    files_to_download = []
    for page in pages:
        for obj in page.get("Contents", []):
            files_to_download.append(obj["Key"])

    # Initialize the progress bar
    with tqdm(total=len(files_to_download), desc="Downloading files") as pbar:
        for s3_key in files_to_download:
            # Construct the full local path
            local_file_path = os.path.join(local_dir, os.path.relpath(s3_key, prefix))
            local_file_dir = os.path.dirname(local_file_path)

            # Ensure local directory exists
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            # Download the file
            s3_client.download_file(bucket_name, s3_key, local_file_path)

            # Update the progress bar
            pbar.update(1)


@contextmanager
def make_local_checkpoint(checkpoint_dir: str) -> Generator[str, None, None]:
    parsed_dir = urlparse(checkpoint_dir)

    assert parsed_dir.scheme in ["s3", ""], "Only s3 and local paths are supported."

    if os.path.exists(checkpoint_dir):
        yield checkpoint_dir
        return

    temp_dir = os.path.join(tempfile.gettempdir(), md5(checkpoint_dir.encode()).hexdigest())
    if os.path.exists(temp_dir):
        yield temp_dir
        return
    try:
        os.makedirs(temp_dir, exist_ok=True)
        download_s3_directory(parsed_dir.netloc, parsed_dir.path[1:], temp_dir)
    except Exception as e:
        logger.error(f"Error downloading checkpoint: {e}")
        shutil.rmtree(temp_dir)
        raise e

    yield temp_dir


@contextmanager
def upload_local_checkpoint(local_checkpoint_dir: str, destination_dir: str) -> Generator[None, None, None]:
    yield

    if destination_dir == local_checkpoint_dir:
        return
    elif (parsed_url := urlparse(destination_dir)).scheme == "s3":
        s3_bucket_name = parsed_url.netloc
        s3_prefix = parsed_url.path[1:]

        local_paths = [
            os.path.join(root, post_fn) for root, _, files in os.walk(local_checkpoint_dir) for post_fn in files
        ]
        dest_paths = [
            os.path.join(s3_prefix, os.path.relpath(local_path, local_checkpoint_dir))
            for local_path in local_paths
        ]

        s3_client = boto3.client("s3")
        for local_path, dest_path in tqdm(
            zip(local_paths, dest_paths), desc="Uploading files", total=len(local_paths)
        ):
            s3_client.upload_file(local_path, s3_bucket_name, dest_path)
    elif parsed_url.scheme == "":
        shutil.copytree(local_checkpoint_dir, destination_dir)
    else:
        raise ValueError(f"Unsupported destination: {destination_dir}. Only s3 and local paths are supported.")


def maybe_unshard(checkpoint_dir: str):
    if os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
        return

    train_config = TrainConfig.load(os.path.join(checkpoint_dir, "config.yaml"))
    checkpointer = OlmoCoreCheckpointer(train_config)
    model_state, _, _ = checkpointer.unshard_checkpoint(
        load_path=checkpoint_dir, load_optimizer_state=False, load_trainer_state=False
    )
    torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, and creates pytorch_model.bin, "
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Location of OLMo checkpoint.",
        required=True,
    )

    parser.add_argument(
        "--destination-dir",
        help="Location to save the converted checkpoint; default is the same as the checkpoint-dir.",
        default=None,
    )

    parser.add_argument(
        "--ignore-olmo-compatibility",
        action="store_true",
        help="Ignore compatibility with the olmo codebase. "
        "This will remove files that are needed specifically for olmo codebase, eg. config.yaml, etc.",
    )
    parser.add_argument(
        "--logger-level",
        default="warning",
        help="Set the logger level.",
    )

    parser.add_argument(
        "--tokenizer",
        help="Override the tokenizer to use for the checkpoint.",
    )
    parser.add_argument(
        "--keep-olmo-artifacts",
        action="store_true",
        help="Keep olmo-specific artifacts in the checkpoint.",
    )

    args = parser.parse_args()

    args.destination_dir = args.destination_dir or args.checkpoint_dir
    logging.basicConfig()
    logger.setLevel(logging.getLevelName(args.logger_level.upper()))

    with make_local_checkpoint(args.checkpoint_dir) as local_checkpoint_dir, upload_local_checkpoint(
        local_checkpoint_dir, args.destination_dir
    ):
        args.checkpoint_dir = local_checkpoint_dir
        maybe_unshard(local_checkpoint_dir)

        fix_tokenizer(checkpoint_dir=local_checkpoint_dir, tokenizer_name_or_path=args.tokenizer)
        convert_checkpoint(args.checkpoint_dir, args.ignore_olmo_compatibility)

        if not args.keep_olmo_artifacts:
            os.remove(os.path.join(local_checkpoint_dir, "config.yaml"))
            os.remove(os.path.join(local_checkpoint_dir, "model.pt"))
            shutil.rmtree(os.path.join(local_checkpoint_dir, "optim"), ignore_errors=True)
            shutil.rmtree(os.path.join(local_checkpoint_dir, "model"), ignore_errors=True)
            shutil.rmtree(os.path.join(local_checkpoint_dir, "train"), ignore_errors=True)


if __name__ == "__main__":
    main()
