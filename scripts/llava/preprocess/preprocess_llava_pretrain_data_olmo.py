"""
Script for preparing the LLaVA pretraining data for aligning visual features with an OLMo model.
"""

import sys
import json
import logging
from os.path import join
from time import perf_counter
from pathlib import Path
from omegaconf import OmegaConf as om
import multiprocessing
import math
from functools import partial
from rich.progress import track

from olmo import Tokenizer
from olmo.mm_data.object_store import ObjectStore, FileStore
from olmo.mm_data.data_store import MMStorageConfig, write_data_file
from olmo.mm_data.structure_index import VectorizedIndexer, get_index_file as get_structure_idx_file
from olmo.mm_data.preprocess import ImageFile, Masked, preprocess_example
from olmo.util import prepare_cli_environment


logger = logging.getLogger(__name__)


def parse_examples(data_path, image_dir, tokenizer, use_image=True):
    with open(data_path) as f:
        list_data_dict = json.load(f)
    # The caption is the second conversation in the list of conversations.
    input_examples = []
    for data_dict in list_data_dict:
        conv = data_dict['conversations'][1]['value'].strip()
        if use_image:
            example = [
                Masked(tokenizer.eos_token),
                ImageFile(join(image_dir, data_dict['image'])),
                conv + tokenizer.eos_token,
            ]
        else:
            example = [Masked(tokenizer.eos_token), conv + tokenizer.eos_token]
        input_examples.append(example)
    return input_examples


def preprocess(example, tokenizer: Tokenizer, object_store: ObjectStore, data_config: MMStorageConfig):
    document = preprocess_example(
        example, tokenizer, object_store, data_config, add_bos_token=False, add_eos_token=False,
    )
    return document


def main(config_file):
    args = om.load(config_file)

    # Tokenizer
    tokenizer = Tokenizer.from_file(
        args.tokenizer_file,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
    )

    # Parse pretraining data
    t0 = perf_counter()
    logger.info("Parsing pretraining data...")
    input_examples = parse_examples(args.data_file, args.image_dir, tokenizer, args.use_image)
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    data_config = MMStorageConfig()
    object_store = None
    if args.use_image:
        source_dir = args.source_dir
        Path(source_dir).mkdir(parents=True, exist_ok=True)
        object_store = FileStore(source_dir)
    # Pre-processing examples by tokenizing the text and storing the images
    t0 = perf_counter()
    logger.info("Preprocessing examples by tokenizing the text and storing the images...")
    _preprocess = partial(preprocess, tokenizer=tokenizer, object_store=object_store, data_config=data_config)
    if args.num_proc > 1:
        with multiprocessing.Pool(args.num_proc) as pool:
            documents = pool.map(_preprocess, input_examples)
    else:
        documents = [_preprocess(input_example) for input_example in input_examples]
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    # Build data/index files
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_data_files = math.ceil(len(documents) / args.split_size)
    basic_indexer = VectorizedIndexer()
    data_files = []
    t0 = perf_counter()
    logger.info(f"Starting {num_data_files} data/index files build")
    for i in track(range(num_data_files)):
        data_file = join(output_dir, f"data_{i:03d}.bin")
        iterator = write_data_file(documents[i*args.split_size:(i+1)*args.split_size], data_file, data_config)
        index_file = get_structure_idx_file(data_file)
        basic_indexer.write_index(index_file, iterator)
        data_files.append(data_file)
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

if __name__ == '__main__':
    prepare_cli_environment()

    try:
        config_file = sys.argv[1]
    except IndexError:
        raise RuntimeError(f"Usage: {sys.argv[0]} [CONFIG_PATH]")
    main(config_file)
