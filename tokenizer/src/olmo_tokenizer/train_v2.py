import gzip
import os
from queue import Queue
import re
from argparse import ArgumentParser
from time import sleep
from typing import Iterator, Union
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

import orjson
from smashed.utils.io_utils import (
    open_file_for_read,
    open_file_for_write,
    recursively_list_files,
)
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from tqdm import tqdm


def read_single(path, queue, lang):
    with open_file_for_read(path, mode="rb") as file_obj:
        with gzip.open(file_obj, mode="rt") as stream:
            for line in stream:
                data = orjson.loads(line)

                if lang == "en" and data.get("source", "") == "wiki":
                    # this excludes wiki, but not wiki-en or other
                    # sources
                    continue

                text = data.get("text", "").strip()
                if text:
                    queue.put(text)

    queue.put(None)


def emitter(queue: "Queue[Union[str, None]]", total: int):
    with tqdm(desc="Lines", unit=" l", unit_scale=True, position=2) as lines_pbar, \
            tqdm(desc="Files", unit=" f", unit_scale=True, position=1, total=total) as files_pbar:
        while True:
            if queue.empty():
                sleep(0.1)
                continue

            content = queue.get()
            if content is None:
                total -= 1
                files_pbar.update(1)
                if total == 0:
                    break

            yield content
            lines_pbar.update(1)


def data_it(base_path, lang) -> Iterator[str]:
    paths = list(recursively_list_files(base_path))

    with tqdm(desc="Lines", unit=" l", unit_scale=True, position=1) as lines_pbar, tqdm(
        total=len(paths), desc="Files", unit=" f", unit_scale=True, position=0
    ) as files_pbar:
        for path in paths:
            with open_file_for_read(path, mode="rb") as file_obj:
                with gzip.open(file_obj, mode="rt") as stream:
                    for line in stream:
                        data = orjson.loads(line)

                        if lang == "en" and data.get("source", "") == "wiki":
                            # this excludes wiki, but not wiki-en or other
                            # sources
                            continue

                        text = data.get("text", "").strip()
                        if text:
                            yield text
                            lines_pbar.update(1)

            files_pbar.update(1)


def main():
    ap = ArgumentParser()
    ap.add_argument("name", type=str, choices=("v2", "v2_small", "v2_tiny"))
    ap.add_argument("--bloom", action="store_true")
    ap.add_argument("--lang", type=str, default="all", choices=("all", "en"))
    ap.add_argument("--vocab_size", type=int, default=64000)
    opts = ap.parse_args()

    # These special tokens are not added to the tokenizer's vocabulary to ensure they
    # never appear in the training data. That is, there is no string form of
    # these tokens.
    EOS_TOKEN_ID = opts.vocab_size - 1
    PAD_TOKEN_ID = opts.vocab_size - 2

    # Initialize tokenizer object.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFD()  # type: ignore

    if opts.bloom:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore
            [
                # Split up digits.
                pre_tokenizers.Digits(individual_digits=True),
                # bloom-style split on regex
                pre_tokenizers.Split(
                    pattern=Regex(" ?[^(\\s|[.,!?…。，、।۔،])]+"),
                    behavior="isolated",
                    invert=False,
                ),
                # Finally, do the byte-level BPE things.
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    use_regex=False,
                ),
            ]
        )
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore
            [
                # Split up digits.
                pre_tokenizers.Digits(individual_digits=True),
                # Split on all punctuation.
                pre_tokenizers.Punctuation(),
                # Finally, do the byte-level BPE things.
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore

    # Initialize trainer.
    trainer = trainers.BpeTrainer(  # type: ignore
        # make room for special tokens which we don't want in the actual vocab
        vocab_size=opts.vocab_size - 2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
    )

    BASE_PATH = f"s3://ai2-llm/tokenizer/data/{opts.name}"
    DEST_PATH = "s3://ai2-llm/tokenizer/model"

    # paths = list(recursively_list_files(BASE_PATH))
    # with get_context("spawn").Manager() as manager:
    #     queue = manager.Queue()

    #     with ProcessPoolExecutor(max_workers=8) as executor:
    #         for path in paths:
    #             executor.submit(read_single, path, queue, opts.lang)

    #         tokenizer.train_from_iterator(emitter(queue, total=len(paths)), trainer=trainer)

    tokenizer.train_from_iterator(data_it(BASE_PATH, opts.lang), trainer=trainer)

    output_name = f"{opts.name}{'_bloom' if opts.bloom else ''}{'_en' if opts.lang == 'en' else ''}.json"
    LOCAL_PATH = str(os.path.expanduser(f"~/{output_name}"))
    print(f"Saving tokenizer to {LOCAL_PATH}...")
    tokenizer.save(LOCAL_PATH)

    REMOTE_PATH = f"{DEST_PATH}/{output_name}"
    print(f"Uploading tokenizer to {REMOTE_PATH}...")
    with open_file_for_write(REMOTE_PATH, mode="w") as file_obj:
        with open(LOCAL_PATH, mode="r") as stream:
            file_obj.write(stream.read())


if __name__ == "__main__":
    main()
