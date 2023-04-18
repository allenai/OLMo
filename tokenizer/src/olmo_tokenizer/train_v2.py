import gzip
import os
from argparse import ArgumentParser
from typing import Iterator, List

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
    trainers,
)
from tqdm import tqdm


def data_it(
    base_paths: List[str],
    lang: str = 'all',
    no_s2: bool = False,
    batch_size: int = 1000
) -> Iterator[List[str]]:
    paths = set()
    for bp in base_paths:
        paths.update(recursively_list_files(bp))
    paths = sorted(paths)

    batch: List[str] = []

    with tqdm(desc="Lines", unit=" l", unit_scale=True, position=1) as lines_pbar, tqdm(
        total=len(paths), desc="Files", unit=" f", unit_scale=True, position=0
    ) as files_pbar:
        for path in paths:
            with open_file_for_read(path, mode="rb") as file_obj:
                with gzip.open(file_obj, mode="rt") as stream:
                    for line in stream:
                        data = orjson.loads(line)

                        if lang == "en" and data.get("source", "") == "wiki":
                            # this excludes wiki, but not wiki-en or other sources
                            continue

                        if no_s2 and data.get("source", "") in {"s2", "s2orc", "s2ag"}:
                            # this excludes s2
                            continue

                        text = data.get("text", "").strip()
                        if text:
                            batch.append(text)

                        if len(batch) >= batch_size:
                            yield batch
                            lines_pbar.update(len(batch))
                            batch = []

            files_pbar.update(1)

    if batch:
        yield batch
        lines_pbar.update(len(batch))


DEFAULT_BASE_PATH = "s3://ai2-llm/tokenizer/data"


def main():
    ap = ArgumentParser()
    ap.add_argument("name", type=str, choices=("v2", "v2_small", "v2_tiny"))
    ap.add_argument('--normalization', choices=('nfd', 'nfc'), default='nfd')
    ap.add_argument("--bloom", action="store_true")
    ap.add_argument('--no-s2', action='store_true')
    ap.add_argument("--lang", type=str, default="all", choices=("all", "en"))
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--comment", type=str, default=None)
    ap.add_argument('-p', '--base-path', default=[DEFAULT_BASE_PATH], nargs='+')
    opts = ap.parse_args()

    # These special tokens are not added to the tokenizer's vocabulary to ensure they
    # never appear in the training data. That is, there is no string form of
    # these tokens.
    EOS_TOKEN_ID = opts.vocab_size - 1
    PAD_TOKEN_ID = opts.vocab_size - 2

    # Initialize tokenizer object.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = (
        normalizers.NFD() if opts.normalization == 'nfd' else normalizers.NFC()     # type: ignore
    )

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

    base_path: List[str]
    if len(opts.base_path)  == 0 and opts.base_path[0] == DEFAULT_BASE_PATH:
        base_path = [DEFAULT_BASE_PATH.rstrip('/') + '/' + opts.name]
    else:
        base_path = opts.base_path

    DEST_PATH = "s3://ai2-llm/tokenizer/model"

    # paths = list(recursively_list_files(BASE_PATH))
    # with get_context("spawn").Manager() as manager:
    #     queue = manager.Queue()

    #     with ProcessPoolExecutor(max_workers=8) as executor:
    #         for path in paths:
    #             executor.submit(read_single, path, queue, opts.lang)

    #         tokenizer.train_from_iterator(emitter(queue, total=len(paths)), trainer=trainer)

    tokenizer.train_from_iterator(data_it(base_path, opts.lang, opts.no_s2), trainer=trainer)

    output_name = (
        opts.name +
        ('_bloom' if opts.bloom else '') +
        ('_en' if opts.lang == 'en' else '') +
        ('_nos2' if opts.no_s2 else '') +
        (f'_{opts.normalization}' if opts.normalization != 'nfd' else '') +
        (f'_{opts.comment}' if opts.comment else '') +
        ".json"
    )
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
