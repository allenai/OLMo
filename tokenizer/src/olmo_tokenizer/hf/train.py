import gzip
import itertools
import json
import multiprocessing
import os
import random
from string import punctuation
import unicodedata
from contextlib import ExitStack
from functools import partial
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Literal, Optional, Union

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
import springs as sp
from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)
import tqdm


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
    quiet: bool = True,
    min_sentence_length: int = 0,
):
    if isinstance(input_dir, str):
        input_dir = [input_dir]

    all_jsonls = sorted(itertools.chain.from_iterable(recursively_list_files(d) for d in input_dir))
    random.shuffle(all_jsonls)
    total_cnt = 0

    files_pbar = tqdm.tqdm(total=len(all_jsonls), desc="Source files", position=0, unit="f", unit_scale=True)
    records_pbar = tqdm.tqdm(total=0, desc="Records", position=1, unit="r", unit_scale=True)

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
            text = text.replace("\x00", " ").strip()

            if min_sentence_length > 0 and len(text) >= min_sentence_length:
                yield text

            total_cnt += 1
            current_cnt += 1
            records_pbar.update(1)

        print(f"Sampled {current_cnt:,} records from {j}") if (not quiet and sample is not None) else None
        files_pbar.update(1)

    print(f"Processed {total_cnt:,} total records") if not quiet else None
    print('\n' * 2)


@sp.dataclass
class TrainConfig:
    input_dir: Optional[str] = None
    input_dirs: Optional[List[str]] = None
    save_path: str = sp.MISSING
    normalization: Union[str, None] = sp.field(
        default="NFD", help="Choose between NFD, NFKD, NFC, or NFKC"
    )
    vocab_size: int = 64_000
    model: str = sp.field(default="Unigram", help="Choose between Unigram (default) or BPE.")
    # seed_sentencepiece_size: int = 100_000
    split_digits: bool = True
    remove_extra_whitespaces: bool = False
    num_threads: int = multiprocessing.cpu_count() - 1
    # tabs_indent_for_code: bool = True
    # tabs_indent_max_depth: int = 8
    # space_indent_for_code: bool = True
    # space_indent_max_depth: int = 8
    min_sentence_length: int = 16
    by_sentence: bool = False
    # max_sentence_length: int = 10_000
    sample: Optional[float] = None
    random_seed: int = 42
    data_format: str = sp.field(default="jsonl", help="Choose between jsonl (default) or text.")
    debug: bool = False


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
        "NFKD"
    ], "Normalization must be one of NFC, NFKC, NFD, NFKD"
    assert config.data_format in ["jsonl", "text"], "Data format must be one of jsonl or text"

    assert config.model in ['BPE', 'Unigram'], "Model must be one of BPE or Unigram"

    _json_iterator = partial(
        json_iterator,
        normalization=config.normalization,  # pyright: ignore
        sample=config.sample,
        data_format=config.data_format,  # pyright: ignore
        min_sentence_length=config.min_sentence_length,  # pyright: ignore
    )

    # Initialize tokenizer object.
    if config.model == "BPE":
        model = models.BPE(byte_fallback=True)
    else:
        model = models.Unigram(None)

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = getattr(normalizers, config.normalization)()     # type: ignore

    if config.model == "BPE":
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore
            [
                # Split on all punctuation.
                pre_tokenizers.Split(
                    pattern=Regex(" ?[[:punct:]]"),
                    behavior="isolated",
                    invert=False,
                ),
                # Split up digits.
                pre_tokenizers.Split(
                    pattern=Regex(" ?\\d"),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            ]
        )
        tokenizer.decoder = decoders.Sequence([     # type: ignore
            decoders.ByteFallback(),
            decoders.ByteLevel(add_prefix_space=False, use_regex=True),
        ])

        trainer = trainers.BpeTrainer(    # type: ignore
            vocab_size=config.vocab_size - 2,
            special_tokens=[],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore
            [
                pre_tokenizers.Metaspace(add_prefix_space=False),
                pre_tokenizers.Split(
                    pattern=Regex("▁?[[:punct:]]"),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.Split(
                    pattern=Regex("▁?\\d"),
                    behavior="isolated",
                    invert=False,
                )
            ]
        )
        tokenizer.decoder = decoders.Metaspace()  # type: ignore

        trainer = trainers.UnigramTrainer(  # type: ignore
            vocab_size=config.vocab_size - 2,
            special_tokens=[],
            initial_alphabet=["▁", *(chr(x) for x in range(256))]
        )

    print("\nDry run for iterating over the data...")
    i = 0
    for t in _json_iterator(config.input_dir or config.input_dirs):  # pyright: ignore
        if i == 10:
            break
        print(repr(t[:40]) + '...')
        i += 1
    if i == 0:
        raise ValueError("No data found. Please check your input path.")
    else:
        print(f"Successfully dry-run on {i} lines.")

    tokenizer.train_from_iterator(
        _json_iterator(config.input_dir or config.input_dirs),  # pyright: ignore
        trainer=trainer
    )

    with NamedTemporaryFile(delete=False) as f:
        tokenizer.save(f.name)
        f_name = f.name

    print(f"Tokenizer saved at {f.name}")

    with open_file_for_write(config.save_path + '.json', "w") as f, open(f_name, "r") as g:
        f.write(g.read())

    with open_file_for_write(config.save_path + "_config.yaml", "w") as f:
        f.write(sp.to_yaml(config))

    os.remove(f_name)

    print(f"Tokenizer uploaded to {config.save_path}")


if __name__ == "__main__":
    train_tokenizer()
