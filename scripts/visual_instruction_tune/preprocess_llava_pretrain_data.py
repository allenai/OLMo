"""
Script for preparing the LLaVA pretraining data for aligning visual features with an OLMo model.
"""

import sys
import random
import json
import logging
from os.path import join
from time import perf_counter
from pathlib import Path
from omegaconf import OmegaConf as om
import multiprocessing
import math

from olmo import Tokenizer
from olmo.mm_data.object_store import FileStore
from olmo.mm_data.data_store import MMStorageConfig, write_data_file
from olmo.mm_data.image_token_size import FixedNumberOfToken, AnyResImageTokenizer
from olmo.mm_data.sequence_index import get_idx_file as get_sequence_idx_file
from olmo.mm_data.structure_index import VectorizedIndexer, get_index_file as get_structure_idx_file
from olmo.mm_data.preprocess import ImageFile, preprocess_example
from olmo.util import prepare_cli_environment


logger = logging.getLogger(__name__)

def get_sizer(v_cfg):
    if v_cfg.anyres:
        return AnyResImageTokenizer(
            v_cfg.image_width,
            v_cfg.image_height,
            v_cfg.patch_width,
            v_cfg.patch_height,
            v_cfg.possible_resolutions,
            v_cfg.n_queries,
        )
    else:
        n_tokens = v_cfg.n_queries if v_cfg.n_queries is not None else (v_cfg.image_width // v_cfg.patch_width) * (v_cfg.image_height // v_cfg.patch_height)
        return FixedNumberOfToken(n_tokens)


def parse_examples(data_path, image_dir, use_image=True):
    with open(data_path) as f:
        list_data_dict = json.load(f)
    # Construct a list of [ImageFile, caption].
    # The caption is the second conversation in the list of conversations.
    input_examples = []
    for data_dict in list_data_dict:
        example = []
        if use_image:
            example += [ImageFile(join(image_dir, data_dict['image']))]
        example += [data_dict['conversations'][1]['value']]
        input_examples.append(example)
    return input_examples


def main(config_file):
    args = om.load(config_file)

    # Parse pretraining data
    t0 = perf_counter()
    logger.info("Parsing pretraining data...")
    input_examples = parse_examples(args.data_file, args.image_dir, args.use_image)
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    # Tokenizer
    tokenizer = Tokenizer.from_file(
        args.tokenizer_file,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
    )
    data_config = MMStorageConfig()
    object_store = None
    if args.use_image:
        source_dir = args.source_dir
        Path(source_dir).mkdir(parents=True, exist_ok=True)
        object_store = FileStore(source_dir)
    # Pre-processing examples by tokenizing the text and storing the images
    t0 = perf_counter()
    logger.info("Preprocessing examples by tokenizing the text and storing the images...")
    if args.num_proc > 1:
        with multiprocessing.Pool(args.num_proc) as pool:
            documents = pool.starmap(
                preprocess_example,
                [(input_example, tokenizer, object_store, data_config) for input_example in input_examples])
    else:
        documents = [preprocess_example(input_example, tokenizer, object_store, data_config) for input_example in input_examples]
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    # Shuffle the documents
    seed = args.seed
    t0 = perf_counter()
    logger.info(f"Starting document shuffle {seed}")
    random.seed(seed)
    random.shuffle(documents)
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    # Build data/index files
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_data_files = math.ceil(len(documents) / args.split_size)
    basic_indexer = VectorizedIndexer()
    data_files = []
    t0 = perf_counter()
    logger.info(f"Starting {num_data_files} data/index files build")
    for i in range(num_data_files):
        data_file = join(output_dir, f"data_s{seed}_{i:03d}.bin")
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
