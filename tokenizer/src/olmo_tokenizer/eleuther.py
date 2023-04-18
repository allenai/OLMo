# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Assumes a dataset of jsonl files in the same format as the neox training set.
"""

from contextlib import ExitStack
import gzip
import itertools
import os
from tempfile import NamedTemporaryFile
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC, NFC

from glob import glob
import json
import argparse

from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)


def load_jsonl(input_path, quiet=True) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    mode = 'rb' if input_path.endswith('.gz') else 'r'

    with ExitStack() as stack:
        stream = stack.enter_context(open_file_for_read(input_path, mode=mode))
        if input_path.endswith('.gz'):
            stream = stack.enter_context(gzip.open(stream, mode='rt'))

        for line in stream:
            data.append(json.loads(line.rstrip("\n|\r")))   # type: ignore

    if not quiet:
        print("Loaded {} records from {}".format(len(data), input_path))
    return data


def json_iterator(input_dir, text_key="text"):
    if isinstance(input_dir, str):
        input_dir = [input_dir]

    all_jsonls = itertools.chain.from_iterable(recursively_list_files(d) for d in input_dir)
    for j in all_jsonls:
        data = load_jsonl(j)
        for doc in data:
            yield doc[text_key]


def train_tokenizer(
    input_dir: str, save_path: str, tokenizer_type: str = "BPE", vocab_size: int = 52000
):
    """
    Trains a tokenizer on all the json files in `input_dir` and saves it to `save_path`

    :param input_dir: input directory containing jsonl files
    :param save_path: path to save tokenizer to
    :param tokenizer_type: type of tokenizer to train.
    :param vocab_size: int, size of tokenizer's vocab
    :return:
    """

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
    tokenizer = Tokenizer(model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )   # type: ignore
    tokenizer.decoder = decoders.ByteLevel()    # type: ignore
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore
    tokenizer.normalizer = NFC()   # type: ignore

    # And then train
    trainer = trainers.BpeTrainer(  # type: ignore
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>"]
    )
    tokenizer.train_from_iterator(json_iterator(input_dir), trainer)

    with NamedTemporaryFile(delete=False) as f:
        tokenizer.save(f.name)
        f_name = f.name

    print(f"Tokenizer saved at {f.name}")

    with open_file_for_write(save_path, 'w') as f:
        with open(f_name, "r") as f2:
            f.write(f2.read())

    os.remove(f_name)

    print(f"Tokenizer uploaded to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="script for training a multilingual "
        "HF tokenizer on CC dumps with upweighting for low resource languages"
    )
    parser.add_argument(
        "-i", "--json_input_dir",
        type=str,
        nargs="+",
        help="Path to folder containing tokenizer training data in jsonl format",
        required=True,
    )
    parser.add_argument(
        "-o", "--tokenizer_output_path",
        type=str,
        help="Path to which your trained tokenizer will be saved (should end in .json)",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        help="type of tokenizer to train, currently only BPE is supported",
        choices=["BPE"],
        default="BPE",
    )
    parser.add_argument(
        "-v",
        "--vocab_size",
        help="vocabulary size of tokenizer, default=64k",
        type=int,
        default=64_000,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_tokenizer(
        args.json_input_dir,
        save_path=args.tokenizer_output_path,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
    )
