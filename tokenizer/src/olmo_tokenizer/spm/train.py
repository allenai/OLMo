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

import gzip
import itertools
import json
import multiprocessing
import os
import random
import unicodedata
from contextlib import ExitStack
from functools import partial
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Union

import sentencepiece as spm
import springs as sp
from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)


def is_gzip_file(path: str) -> bool:
    return path.endswith(".gz") or path.endswith(".gzip")


def load_jsonl(input_path, quiet=False) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    mode = "rb" if is_gzip_file(input_path) else "r"

    with ExitStack() as stack:
        stream = stack.enter_context(open_file_for_read(input_path, mode=mode))

        if is_gzip_file(input_path):
            stream = stack.enter_context(gzip.open(stream, mode="rt"))

        for line in stream:
            data.append(json.loads(line.rstrip("\n|\r")))  # type: ignore

    print(f"Loaded {len(data):,} records from {input_path}") if not quiet else None
    return data


def load_text(input_path, quiet=False) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    mode = "rb" if is_gzip_file(input_path) else "r"

    with ExitStack() as stack:
        stream = stack.enter_context(open_file_for_read(input_path, mode=mode))

        if is_gzip_file(input_path):
            stream = stack.enter_context(gzip.open(stream, mode="rt"))

        data = [{"text": line} for line in stream]

    print(f"Loaded {len(data):,} records from {input_path}") if not quiet else None
    return data


def json_iterator(
    input_dir: Union[str, List[str]],
    text_key="text",
    normalization: Literal["NFC", "NFKC", "NFD", "NFKD", None] = "NFC",
    data_format: Literal["jsonl", "text"] = "jsonl",
    sample: Optional[float] = None,
    quiet: bool = False,
):
    if isinstance(input_dir, str):
        input_dir = [input_dir]

    all_jsonls = sorted(itertools.chain.from_iterable(recursively_list_files(d) for d in input_dir))
    random.shuffle(all_jsonls)
    total_cnt = 0

    for j in all_jsonls:
        current_cnt = 0

        if sample is not None and sample >= 1 and total_cnt >= sample:
            break

        data = load_jsonl(j, quiet=quiet) if data_format == "jsonl" else load_text(j, quiet=quiet)

        for doc in data:
            if sample is not None:
                if sample < 1.0 and random.random() > sample:
                    continue
                elif sample >= 1.0 and total_cnt >= sample:
                    break

            text = doc[text_key]
            if normalization is not None:
                text = unicodedata.normalize(normalization, text)

            # replace null characters with spaces
            text = text.replace("\x00", " ")

            yield text
            total_cnt += 1
            current_cnt += 1

        print(f"Sampled {current_cnt:,} records from {j}") if (not quiet and sample is not None) else None

    print(f"Processed {total_cnt:,} total records") if not quiet else None


@sp.dataclass
class ModelConfig:
    vocab_size: int = 64_000
    seed_sentencepiece_size: int = 100_000
    max_sentence_length: int = 10_000
    split_digits: bool = True
    remove_extra_whitespaces: bool = False
    normalization_rule_name: str = "identity"
    model_type: str = sp.field(default="unigram", help="Choose between unigram (default), bpe, word, or char.")
    allow_whitespace_only_pieces: bool = True
    byte_fallback: bool = True
    num_threads: int = multiprocessing.cpu_count() - 1
    train_extremely_large_corpus: bool = False
    input_sentence_size: int = 0
    shuffle_input_sentence: bool = False


@sp.dataclass
class TrainConfig:
    input_dir: Optional[str] = None
    input_dirs: Optional[List[str]] = None
    save_path: str = sp.MISSING
    normalization: Union[str, None] = sp.field(
        default="NFC", help="Choose between NFC (default), NFKC, NFD, NFKD, or None."
    )
    model: ModelConfig = ModelConfig()
    tabs_indent_for_code: bool = True
    tabs_indent_max_depth: int = 8
    space_indent_for_code: bool = True
    space_indent_max_depth: int = 8
    sample: Optional[float] = None
    random_seed: int = 42
    data_format: str = sp.field(default="jsonl", help="Choose between jsonl (default) or text.")


@sp.cli(TrainConfig)
def train_tokenizer(config: TrainConfig):
    # set the seed for reproducibility
    random.seed(config.random_seed)

    assert (
        config.input_dir is not None or config.input_dirs is not None
    ), "Must specify either input_dir or input_dirs"
    assert config.normalization in [
        "NFC",
        "NFKC",
        "NFD",
        "NFKD",
        None,
    ], "Normalization must be one of NFC, NFKC, NFD, NFKD, or None"
    assert (
        config.normalization is None or config.model.normalization_rule_name == "identity"
    ), "If external normalization is used, normalization_rule_name must be 'identity'"
    assert config.data_format in ["jsonl", "text"], "Data format must be one of jsonl or text"

    if config.tabs_indent_for_code or config.space_indent_for_code:
        assert (
            config.model.allow_whitespace_only_pieces is True and config.model.remove_extra_whitespaces is False
        ), (
            "If you want to allow tabs or spaces, you must also allow whitespace only pieces and not "
            "remove extra whitespaces (allow_whitespace_only_pieces=True, remove_extra_whitespaces=False)"
        )

    _json_iterator = partial(
        json_iterator,
        normalization=config.normalization,  # pyright: ignore
        sample=config.sample,
        data_format=config.data_format,  # pyright: ignore
    )

    user_defined_symbols: List[str] = sorted(
        set(
            [
                *(
                    ["\t" * i for i in range(2, config.tabs_indent_max_depth + 1)]
                    if config.tabs_indent_for_code
                    else []
                ),
                *(
                    [" " * 2 * i for i in range(2, config.space_indent_max_depth + 1)]
                    if config.space_indent_for_code
                    else []
                ),
                *(
                    [" " * 4 * i for i in range(2, config.space_indent_max_depth + 1)]
                    if config.space_indent_for_code
                    else []
                ),
                *(["\n", "\r", "\t"] if config.model.allow_whitespace_only_pieces else []),
            ]
        )
    )

    print("Dry run for iterating over the data...", end=" ")
    i = 0
    for _ in _json_iterator(config.input_dir or config.input_dirs):  # pyright: ignore
        i += 1
        if i == 10:
            break
    print(f"Successfully dry-run on {i} lines.")

    with TemporaryDirectory() as tmp_dir:
        spm.SentencePieceTrainer.Train(
            sentence_iterator=_json_iterator(config.input_dir or config.input_dirs),
            # pyright: ignore
            model_prefix=os.path.join(tmp_dir, "ai2_llm"),
            user_defined_symbols=user_defined_symbols,
            # random_seed=config.random_seed,
            **sp.to_dict(config.model),  # pyright: ignore
        )

        with ExitStack() as stack:
            for fn in os.listdir(tmp_dir):
                _, extension = fn.rsplit(".", -1)
                src = stack.enter_context(open_file_for_read(os.path.join(tmp_dir, fn), "rb"))
                dst = stack.enter_context(open_file_for_write(f"{config.save_path}.{extension}", "wb"))
                dst.write(src.read())
                stack.pop_all().close()

            dst = stack.enter_context(open_file_for_write(f"{config.save_path}.yaml", "w"))
            dst.write(sp.to_yaml(config))


if __name__ == "__main__":
    train_tokenizer()
