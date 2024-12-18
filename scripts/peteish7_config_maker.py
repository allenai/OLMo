import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, MutableMapping, Tuple
from urllib.parse import urlparse

import boto3
import yaml
from tabulate import tabulate
from tqdm.auto import tqdm

# ===================================================
# =                  S3 HELPERS                     =
# ===================================================


def get_single_s3_size(s3_uri: str, s3_client=None) -> int:
    # Gets the size in bytes of an individual s3 path
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc
    # Remove leading slash and handle edge cases
    object_key = parsed.path.lstrip("/")
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return response["ContentLength"]
    except Exception as e:
        if hasattr(e, "response") and e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(f"The object {object_key} does not exist in bucket {bucket_name}.")
        else:
            raise


def get_batch_s3_size(s3_uris: List[str]):
    # Faster way to get size in bytes for a lot of s3 paths: maps s3_uri -> size
    s3_client = boto3.client("s3")

    def partial_size(s3_uri: str):
        size = get_single_s3_size(s3_uri, s3_client=s3_client)
        return s3_uri, size

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(partial_size, uri) for uri in s3_uris]
        results = []
        for future in tqdm(futures, total=len(futures)):
            results.append(future.result())

    # Convert results to dictionary
    sizes: Dict[str, int] = {}
    for s3_uri, size in results:
        sizes[s3_uri] = sizes.get(s3_uri, 0) + size
    # sizes = dict(results)
    return sizes


def list_s3_paths(s3_uri: str, extension: str = ".npy") -> List[Tuple[str, int]]:
    """
    Lists all paths in an S3 bucket with given prefix and extension, along with their sizes.

    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Prefix to filter objects (e.g., 'data/')
        extension (str): File extension to filter (e.g., '.csv')

    Returns:
        List[Tuple[str, int]]: List of tuples containing (path, size in bytes)
    """
    parsed = urlparse(s3_uri)
    bucket_name = parsed.netloc

    # Remove leading slash and handle edge cases
    prefix = parsed.path.lstrip("/")

    s3_client = boto3.client("s3")

    # Ensure prefix ends with '/' if it's meant to be a directory
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Ensure extension starts with '.'
    if not extension.startswith("."):
        extension = "." + extension

    paths_and_sizes = []
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        # Paginate through results to handle large buckets
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(extension):
                    paths_and_sizes.append((key, obj["Size"]))

        return paths_and_sizes

    except Exception as e:
        print(f"Error listing objects: {str(e)}")
        return []


# =================================================================
# =                Other config-specific helpers                  =
# =================================================================

BASE_YAML_STR = """run_name: REPLACE_RUN_NAME_HERE
seed: 7201
dry_run: false

wandb:
  name: ${run_name}
  project: olmo-medium
  group: ${run_name}

model:
  d_model: 4096
  n_heads: 32
  n_layers: 32
  mlp_hidden_size: 22016
  weight_tying: false
  alibi: false
  rope: true
  rope_theta: 500000
  flash_attention: true
  attention_dropout: 0.0
  include_bias: false
  block_type: sequential
  layer_norm_type: rms
  layer_norm_with_affine: true
  layer_norm_eps: 1e-6
  bias_for_layer_norm: false
  attention_layer_norm: true
  attention_layer_norm_with_affine: true
  norm_after: true
  activation_type: swiglu
  residual_dropout: 0.0
  embedding_dropout: 0.0
  max_sequence_length: 4096
  vocab_size: 100278
  embedding_size: 100352
  eos_token_id: 100257
  pad_token_id: 100277
  init_device: meta
  init_fn: normal
  init_std: 0.02
  init_cutoff_factor: 3

softmax_auxiliary_loss: true
auxiliary_loss_multiplier: 1e-5
fused_loss: true

compile: null

optimizer:
  name: adamw
  learning_rate: 0.000061499
  weight_decay: 0.1
  eps: 1e-8
  decay_norm_and_bias: true
  decay_embeddings: false
  betas:
  - 0.9
  - 0.95
  metrics_log_interval: 1

scheduler:
  name: linear_with_warmup
  t_warmup: 0
  alpha_f: 0

tokenizer:
  identifier: tokenizers/allenai_dolma2.json
  truncate_direction: right

save_folder: /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/${run_name}
save_overwrite: false

save_interval: 1000
save_interval_ephemeral: 250
save_num_checkpoints_to_keep: -1
sharded_checkpointer: olmo_core

save_interval_unsharded: null
save_num_unsharded_checkpoints_to_keep: -1

load_path: /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646

restore_dataloader: false
no_pre_train_checkpoint: true

max_duration: 50e9T
stop_at: 11931                  # round(50e9 / (1024 * 4096)) + 10
global_train_batch_size: 1024
device_train_microbatch_size: 2

precision: amp_bf16

fsdp:
  wrapping_strategy: by_block_and_size
  precision: mixed

activation_checkpointing: one_in_four

max_grad_norm: 1.0
max_grad_norm_ratio: null

speed_monitor:
  window_size: 1

gen1_gc_interval: 1

eval_interval: 1000
eval_subset_num_batches: -1
device_eval_batch_size: ${device_train_microbatch_size}
evaluators:
  # - label: all-small-ppl-validation
  #   data:
  #     num_workers: 0
  #     drop_last: true
  #     # generate_doc_lengths: true
  #     memmap_dtype: uint32
  #     datasets:
  #       c4_en-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy
  #       dolma_books-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_books/val/part-0-00000.npy
  #       dolma_common-crawl-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_common-crawl/val/part-0-00000.npy
  #       dolma_pes2o-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_pes2o/val/part-0-00000.npy
  #       dolma_reddit-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_reddit/val/part-0-00000.npy
  #       dolma_stack-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_stack/val/part-0-00000.npy
  #       dolma_wiki-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/dolma_wiki/val/part-0-00000.npy
  #       ice-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/ice/val/part-0-00000.npy
  #       m2d2_s2orc-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/m2d2_s2orc/val/part-0-00000.npy
  #       pile-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/pile/val/part-0-00000.npy
  #       wikitext_103-validation:
  #         - /weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/wikitext_103/val/part-0-00000.npy

  ##########################
  # Downstream evaluations #
  ##########################
  - label: piqa
    type: downstream

  - label: hellaswag
    type: downstream

  - label: winogrande
    type: downstream

  - label: openbook_qa
    type: downstream

  - label: boolq
    type: downstream

  - label: sciq
    type: downstream

  - label: arc_easy
    type: downstream

  - label: arc_challenge
    type: downstream

  - label: copa
    type: downstream

  #- label: rte
  #  type: downstream

  #- label: commitment_bank
  #  type: downstream

  #- label: sst2
  #  type: downstream

  - label: commonsense_qa
    type: downstream

  - label: social_iqa
    type: downstream

  - label: mmlu_stem_var
    type: downstream

  - label: mmlu_humanities_var
    type: downstream

  - label: mmlu_social_sciences_var
    type: downstream

  - label: mmlu_other_var
    type: downstream

  - label: mmlu_stem_mc_5shot
    type: downstream

  - label: mmlu_humanities_mc_5shot
    type: downstream

  - label: mmlu_social_sciences_mc_5shot
    type: downstream

  - label: mmlu_other_mc_5shot
    type: downstream

  - label: mmlu_stem_mc_5shot_test
    type: downstream

  - label: mmlu_humanities_mc_5shot_test
    type: downstream

  - label: mmlu_social_sciences_mc_5shot_test
    type: downstream

  - label: mmlu_other_mc_5shot_test
    type: downstream

  - label: basic_arithmetic
    type: downstream

  - label: trivia_qa_wiki_ppl
    type: downstream

  - label: natural_qs_open_ppl
    type: downstream

  - label: arc_easy_ppl
    type: downstream

data:
  pad_direction: right
  # generate_doc_lengths: true
  num_workers: 32
  drop_last: true
  pin_memory: true
  prefetch_factor: 8
  persistent_workers: true
  memmap_dtype: uint32
  timeout: 0
  instance_filter:
    repetition_max_period: 13
    repetition_min_period: 1
    repetition_max_count: 32
  paths:"""


def human_format_number(num, decimal_places=2):
    """
    Format a number using K for thousands, M for millions, B for billions, T for trillions.

    Args:
        num: Number to format
        decimal_places: Number of decimal places to show (default: 2)

    Examples:
        format_number(999) => '999'
        format_number(1000) => '1.00K'
        format_number(1500) => '1.50K'
        format_number(1000000) => '1.00M'
        format_number(1500000000) => '1.50B'
    """
    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num < 1000:
        return f"{sign}{abs_num}"

    suffixes = ["", "K", "M", "B", "T"]
    magnitude = 0

    while abs_num >= 1000 and magnitude < len(suffixes) - 1:
        abs_num /= 1000
        magnitude += 1

    # Format with specified decimal places
    formatted = f"{abs_num:.{decimal_places}f}"

    return f"{sign}{formatted}{suffixes[magnitude]}"


def get_token_strs(token_source, bytes_per_token=4):
    if isinstance(token_source, str):
        s3_source = token_source
        ratio = 1.0
    else:
        s3_source, ratio = token_source

    paths_and_sizes = list_s3_paths(s3_source)
    parsed = urlparse(s3_source)
    bucket_name = parsed.netloc
    paths_and_sizes = [("s3://%s/%s" % (bucket_name, p), s) for p, s in paths_and_sizes]
    random.shuffle(paths_and_sizes)
    total_tokens = sum(_[1] for _ in paths_and_sizes) // bytes_per_token
    target_tokens = total_tokens * ratio

    paths_to_add = []
    tokens_to_add = 0
    for p, s in paths_and_sizes:
        paths_to_add.append(p)
        tokens_to_add += s // bytes_per_token
        if tokens_to_add >= target_tokens:
            break
    lines_to_add = ["#SOURCE: %s (%sT)" % (s3_source, human_format_number(tokens_to_add))]
    for p in paths_to_add:
        lines_to_add.append("- %s" % p)
    return lines_to_add


def add_paths(token_sources, output_yaml_file):
    # Adds things to the yaml file.
    # Token sources is a list of either... s3_uri: str | (s3_uri: str, fraction: float)
    # Also I'm not bothering with pyyaml, just appending to the base config (which will be included)
    # ^this is a very crude stone-age tool, don't @ me
    assert output_yaml_file.startswith("peteish7-weka-anneal-from-928646-50B-")
    assert output_yaml_file.endswith(".yaml")

    base_config_str = BASE_YAML_STR.replace(
        "REPLACE_RUN_NAME_HERE", os.path.splitext(os.path.basename(output_yaml_file))[0]
    )

    lines_to_add = []
    for source in token_sources:
        lines_to_add.extend(get_token_strs(source))
    true_lines_to_add = ["\n    %s" % line for line in lines_to_add]
    output_str = base_config_str + "".join(true_lines_to_add)
    with open(output_yaml_file, "w") as f:
        f.write(output_str)


def examine_config(yaml_file, bytes_per_token=4):
    """
    Groups the token sources by their dirname and computes sizes and how much data was taken total.
    Prints out rows of:
    (token source, total_tokens, percentage_taken, tokens_taken)
    """

    print("Getting tokens per input file...")
    # Step 1: collect all paths of tokens
    with open(yaml_file, "r") as f:
        yaml_content = yaml.safe_load(f)
    paths = yaml_content.get("data", {}).get("paths", [])
    paths_to_tokens = {k: v // bytes_per_token for k, v in get_batch_s3_size(paths).items()}

    # Step 2: Gather all sources, count tokens taken
    print("Grouping output files into groups...")
    groups = set(_read_path_comments(yaml_file))

    def get_group(s3_uri):
        for g in groups:
            if s3_uri.startswith(g):
                return g
        raise Exception("UNKNOWN GROUP FOR %s" % s3_uri)

    tokens_taken: MutableMapping[str, int] = defaultdict(int)
    for p, tok in paths_to_tokens.items():
        tokens_taken[get_group(p)] += tok

    # Step 3: count total tokens per group
    print("Getting total group sizes...")
    total_tokens = {}
    for g in tqdm(groups):
        paths_and_sizes = list_s3_paths(g)
        total_tokens[g] = sum(_[1] for _ in paths_and_sizes) // bytes_per_token
    # Step 4: get ratios of percentage taken
    ratios = {
        g: "%.04f" % (tokens_taken[g] / total_tokens[g]) for g in groups
    }  # .04f here (ranging from 0.00 to 1.00)

    # Step 5: actually print the outputs
    rows = sorted([(g, total_tokens[g], ratios[g], tokens_taken[g]) for g in groups])
    print("Put this in your spreadsheet!")
    print(tabulate(rows, headers=["paths", "total_tokens", "percentage taken", "tokens taken"]))


def _read_path_comments(yaml_file):
    # This is helpful for examining paths
    lines = open(yaml_file, "r").readlines()
    path_sources = []
    seen_paths = False
    for line in lines:
        if not seen_paths and line.strip() != "paths:":
            continue
        elif line.strip() == "paths:":
            seen_paths = True
        elif line.strip().startswith("#"):
            path_sources.append(line.strip().split(" ")[1])
        else:
            pass
    return path_sources


# =================================================
# =                     MAIN                      =
# =================================================


if __name__ == "__main__":
    """
    Use this interactively like `python -i peteish7_config_maker.py`, since tuples are weird to pass
    [ or load all these modules into a jupyter notebook ]
    Usage example:
        MATH_TOKENS = ['s3://ai2-llm/preprocessed/personahub_math_v2_79975/',   # uses 100% of this dataset
                       's3://ai2-llm/preprocessed/basic_math_mj/dolma2-tokenizer', # uses 100% of this dataset
                       ('s3://ai2-llm/preprocessed/gsm8k-synth/resample_v1_6x/allenai/dolma2-tokenizer/', 0.5) # uses 50% of this dataset
                      ]
        OUTPUT_YAML = 'peteish7-weka-anneal-from-928646-50B-test_math.yaml'
        add_paths(MATH_TOKENS, OUTPUT_YAML)

        # and then you can populate the spreadsheet with the output of examine_config
        print(examine_config(OUTPUT_YAML))
    """
