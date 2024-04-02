"""
Script for preparing the LLaVA data for training a multimodal OLMo model.
"""

import sys
import json
import logging
from os.path import join
from typing import Union, Optional, List, cast, Tuple
from time import perf_counter
from pathlib import Path
from omegaconf import OmegaConf as om, DictConfig
import multiprocessing
import math
from functools import partial
from rich import progress, traceback
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer

from olmo import Tokenizer
from olmo.aliases import PathOrStr
from olmo.mm_data.object_store import ObjectStore, FileStore
from olmo.mm_data.data_store import MMStorageConfig, write_data_file, Document, ImageChunk, TextChunk
from olmo.mm_data.structure_index import VectorizedIndexer, get_index_file as get_structure_idx_file
from olmo.mm_data.preprocess import ImageFile, Masked, InputExample, preprocess_example
from olmo.mm_data.conversation import DEFAULT_IMAGE_TOKEN, conv_templates, Conversation
from olmo.util import prepare_cli_environment, clean_opt
from hf_olmo import *


logger = logging.getLogger(__name__)


def parse_pretrain_examples(
    data_path: PathOrStr,
    image_dir: PathOrStr,
    tokenizer: Tokenizer,
    eos_token: Optional[str] = None
) -> List[InputExample]:
    with open(data_path) as f:
        list_data_dict = json.load(f)
    # The caption is the second sentence in the list of conversations.
    input_examples = []
    eos_token = eos_token or tokenizer.eos_token
    for data_dict in list_data_dict:
        conv = data_dict['conversations'][1]['value'].strip()
        example = [
            Masked(tokenizer.bos_token),
            ImageFile(join(image_dir, data_dict['image'])),
            conv + eos_token,
        ]
        input_examples.append(example)
    return input_examples


def parse_instruct_tune_examples(
    data_path: PathOrStr,
    image_dir: PathOrStr,
    tokenizer: Tokenizer,
    conv_cfg: Conversation,
    add_system_message: bool,
) -> List[InputExample]:
    with open(data_path) as f:
        list_data_dict = json.load(f)
    roles = {"human": conv_cfg.roles[0], "gpt": conv_cfg.roles[1]}
    seps = [conv_cfg.sep, conv_cfg.sep2]
    input_examples = []
    for data_dict in list_data_dict:
        conversations = data_dict["conversations"]
        example = []
        if roles[conversations[0]["from"]] != conv_cfg.roles[0]:
            # Skip the first one if it is not from human
            conversations = conversations[1:]
        assert len(conversations) % 2 == 0, "Conversations must be in pairs"
        for i in range(0, len(conversations), 2):
            if i == 0 and add_system_message:
                start_token = tokenizer.bos_token + conv_cfg.system + seps[0]
            elif i == 0:
                start_token = tokenizer.bos_token
            else:
                start_token = ""
            sentence1 = conversations[i]
            sentence2 = conversations[i+1]
            role1, role2 = roles[sentence1["from"]], roles[sentence2["from"]]
            assert role1 == conv_cfg.roles[0], f"First role should be {conv_cfg.roles[0]}"
            assert role2 == conv_cfg.roles[1], f"Second role should be {conv_cfg.roles[1]}"
            value1, value2 = sentence1["value"], sentence2["value"]
            if DEFAULT_IMAGE_TOKEN in value1:
                value1 = value1.replace(DEFAULT_IMAGE_TOKEN, '')
                example += [
                    Masked(start_token + role1 + conv_cfg.role_sep),
                    ImageFile(join(image_dir, data_dict['image'])),
                    Masked(value1.strip() + seps[0] + role2 + conv_cfg.role_sep),
                ]
                if value2:
                    example += [value2.strip() + seps[1]]
            else:
                example += [
                    Masked(
                        start_token + role1 + conv_cfg.role_sep + value1.strip() + seps[0] + role2 + conv_cfg.role_sep
                    ),
                ]
                if value2:
                    example += [value2.strip() + seps[1]]
        input_examples.append(example)
    return input_examples


def preprocess(
    ex: Tuple[int, InputExample],
    tokenizer: Tokenizer,
    object_store: ObjectStore,
    data_config: MMStorageConfig,
) -> Tuple[int, Document]:
    i, example = ex
    try:
        document = preprocess_example(
            example,
            tokenizer,
            object_store,
            data_config,
            add_bos_token=False,
            add_eos_token=False,
        )
    except Exception as e:
        logger.error(f"Error processing example: {e}")
        document = None
    return i, document


def truncate_document(doc: Tuple[int, Document]) -> Tuple[int, Document]:
    i, document = doc
    masked_indices = [i for i in range(len(document)) if isinstance(document[i], TextChunk) and document[i].is_masked()]
    if len(masked_indices) > 255:
        logger.error("Truncate document: too many masked text chunks")
        document = document[:masked_indices[255]]
    return i, document


def main(args: DictConfig):

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        use_fast='vicuna' not in args.model_name,
    )
    tokenizer.add_bos_token = False
    tokenizer = cast(Tokenizer, tokenizer)

    if 'vicuna' in args.model_name:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    
    conv_cfg = conv_templates[args.conv_version]

    # Parse pretraining data
    t0 = perf_counter()
    if args.conv_version == "plain":
        logger.info("Parsing pretraining data...")
        input_examples = parse_pretrain_examples(
            args.data_file, args.image_dir, tokenizer, eos_token="\n" if 'vicuna' in args.model_name else None)
    else:
        logger.info("Parsing instruct tuning data...")
        input_examples = parse_instruct_tune_examples(
            args.data_file,
            args.image_dir,
            tokenizer,
            conv_cfg,
            args.add_system_message,
        )
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    data_config = MMStorageConfig()
    object_store = None
    source_dir = args.source_dir
    Path(source_dir).mkdir(parents=True, exist_ok=True)
    object_store = FileStore(source_dir)

    # Pre-processing examples by tokenizing the text and storing the images
    t0 = perf_counter()
    logger.info("Preprocessing examples by tokenizing the text and storing the images...")
    _preprocess = partial(
        preprocess,
        tokenizer=tokenizer,
        object_store=object_store,
        data_config=data_config,
    )
    if args.num_proc > 1:
        with multiprocessing.Pool(args.num_proc) as pool:
            documents = list(progress.track(
                pool.imap(_preprocess, list(enumerate(input_examples))),
                total=len(input_examples),
                description="Preprocessing examples...",
            ))
    else:
        documents = []
        for ex in progress.track(enumerate(input_examples), description="Preprocessing examples..."):
            documents.append(_preprocess(ex))
    documents = sorted(documents, key=lambda x: x[0])
    documents = [document for i, document in documents if document is not None]
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    # Truncate too long documents
    t0 = perf_counter()
    logger.info("Truncating too long documents if any...")
    if args.num_proc > 1:
        with multiprocessing.Pool(args.num_proc) as pool:
            truncated_documents = list(progress.track(
                pool.imap(truncate_document, list(enumerate(documents))),
                total=len(documents),
                description="Truncating documents...",
            ))
    else:
        truncated_documents = []
        for doc in progress.track(enumerate(documents), description="Truncating documents..."):
            truncated_documents.append(truncate_document(doc))
    truncated_documents = sorted(truncated_documents, key=lambda x: x[0])
    documents = [document for i, document in truncated_documents if document is not None]
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

    # Build data/index files
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_data_files = math.ceil(len(documents) / args.split_size)
    basic_indexer = VectorizedIndexer()
    data_files = []
    t0 = perf_counter()
    logger.info(f"Starting {num_data_files} data/index files build")
    for i in progress.track(range(num_data_files)):
        data_file = join(output_dir, f"data_{i:03d}.bin")
        iterator = write_data_file(documents[i*args.split_size:(i+1)*args.split_size], data_file, data_config)
        index_file = get_structure_idx_file(data_file)
        basic_indexer.write_index(index_file, iterator)
        data_files.append(data_file)
    logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")

if __name__ == '__main__':
    prepare_cli_environment()
    traceback.install()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise RuntimeError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")
    args = om.load(yaml_path)
    args_list = [clean_opt(s) for s in args_list]
    if args_list:
        args = om.merge(args, om.from_dotlist(args_list))
    main(args)
