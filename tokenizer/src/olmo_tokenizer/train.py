import gzip
import os
from argparse import ArgumentParser
from typing import Iterator

from smashed.utils.io_utils import open_file_for_read, recursively_list_files
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tqdm import tqdm

ap = ArgumentParser()
ap.add_argument("name", type=str, default="v1")
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


def data_it(base_path=BASE_PATH) -> Iterator[str]:
    paths = list(recursively_list_files(base_path))

    with tqdm(desc="Lines", unit=" l", unit_scale=True, position=1) as lines_pbar, tqdm(
        total=len(paths), desc="Files", unit=" f", unit_scale=True, position=0
    ) as files_pbar:
        for path in paths:
            with open_file_for_read(path, mode="rb") as file_obj:
                with gzip.open(file_obj, mode="rt") as stream:
                    for line in stream:
                        yield str(line)
                        lines_pbar.update(1)

            files_pbar.update(1)


tokenizer.train_from_iterator(data_it(), trainer=trainer)
tokenizer.save(os.path.expanduser(f"~/{opts.name}.json"))
