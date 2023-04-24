import gzip
import itertools
import os
import random
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import Generator, List, Literal, Optional, Union, cast

import msgspec
import springs as sp
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


def is_gzip_file(path: str) -> bool:
    return path.endswith(".gz") or path.endswith(".gzip")


class JsonDoc(msgspec.Struct):
    text: str


class DataReader:
    def __init__(
        self,
        seed: int = 0,
        sample: Union[float, None] = None,
        quiet: bool = True,
        data_format: Literal["jsonl", "text"] = "jsonl",
        min_sentence_length: int = 0,
    ) -> None:
        self.seed = seed
        self.sample = sample
        self.quiet = quiet
        self.file_type = data_format
        self.cnt = 0
        self.min_sentence_length = min_sentence_length

    def increment(self) -> None:
        self.cnt += 1

    def should_sample(self) -> bool:
        if self.sample is None:
            return True
        elif self.sample < 1.0:
            return random.random() < self.sample
        else:
            return self.cnt < self.sample

    def read_json(self, s: str) -> JsonDoc:
        return msgspec.json.decode(s, type=JsonDoc)

    def read_text(self, s: str) -> JsonDoc:
        return JsonDoc(text=s)

    def post_process(self, text: str) -> Union[str, None]:
        text = text.replace("\x00", " ").strip()

        if self.min_sentence_length > 0 and len(text) >= self.min_sentence_length:
            return text

        return None

    def __call__(self, input_path: str) -> Generator[str, None, None]:
        mode = "rb" if is_gzip_file(input_path) else "r"

        fn = self.read_json if self.file_type == "jsonl" else self.read_text

        with ExitStack() as stack:
            stream = stack.enter_context(open_file_for_read(input_path, mode=mode))

            if is_gzip_file(input_path):
                stream = stack.enter_context(gzip.open(stream, mode="rt"))

            for line in stream:
                if not self.should_sample():
                    continue

                doc = fn(cast(str, line))
                if (post := self.post_process(doc.text)) is not None:
                    yield post

        self.print_done(input_path)

    def print_done(self, input_path: str) -> None:
        print(f"Loaded {self.cnt:,} records from {input_path}") if not self.quiet else None


def make_bpe_tokenizer(vocab_size: int, normalization: str = "NFC"):
    model = models.BPE(byte_fallback=True)
    tokenizer = Tokenizer(model)

    tokenizer.normalizer = getattr(normalizers, normalization)()  # type: ignore

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
    tokenizer.decoder = decoders.Sequence(
        [  # type: ignore
            decoders.ByteFallback(),
            decoders.ByteLevel(add_prefix_space=False, use_regex=True),     # type: ignore
        ]
    )

    trainer = trainers.BpeTrainer(  # type: ignore
        vocab_size=vocab_size - 2, special_tokens=[], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    return tokenizer, trainer


def make_unigram_tokenizer(vocab_size: int, normalization: str = "NFC"):
    model = models.Unigram(None)
    tokenizer = Tokenizer(model)

    tokenizer.normalizer = getattr(normalizers, normalization)()  # type: ignore

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
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)     # type: ignore

    trainer = trainers.UnigramTrainer(  # type: ignore
        vocab_size=vocab_size - 2,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # type: ignore
    )

    return tokenizer, trainer


def get_data_iterator(reader: DataReader, input_dir: Union[str, List[str]], seed: int = 0):
    if isinstance(input_dir, str):
        input_dir = [input_dir]

    random.seed(seed)
    all_files = sorted(itertools.chain.from_iterable(recursively_list_files(d) for d in input_dir))
    random.shuffle(all_files)

    for f in all_files:
        yield from reader(f)


@sp.dataclass
class TrainConfig:
    input_dir: Optional[str] = None
    input_dirs: Optional[List[str]] = None
    save_path: str = sp.MISSING
    normalization: Union[str, None] = sp.field(default="NFD", help="Choose between NFD, NFKD, NFC, or NFKC")
    vocab_size: int = 64_000
    model: str = sp.field(default="BPE", help="Choose between BPE (default) or Unigram.")

    min_sentence_length: int = 16
    sample: Optional[float] = None
    random_seed: int = 42

    data_format: str = sp.field(default="jsonl", help="Choose between jsonl (default) or text.")


@sp.cli(TrainConfig)
def train_tokenizer(config: TrainConfig):
    input_dir = config.input_dir or config.input_dirs

    assert input_dir is not None, "Must specify either input_dir or input_dirs"
    assert config.normalization in [
        "NFC",
        "NFKC",
        "NFD",
        "NFKD",
    ], "Normalization must be one of NFC, NFKC, NFD, NFKD"
    assert config.data_format in ["jsonl", "text"], "Data format must be one of jsonl or text"

    if config.model == "BPE":
        tokenizer, trainer = make_bpe_tokenizer(config.vocab_size, config.normalization)
    elif config.model == "Unigram":
        tokenizer, trainer = make_unigram_tokenizer(config.vocab_size, config.normalization)
    else:
        raise ValueError(f"Unknown model {config.model}")

    reader = DataReader(
        seed=config.random_seed,
        sample=config.sample,
        quiet=True,
        data_format=config.data_format,  # type: ignore
        min_sentence_length=config.min_sentence_length,
    )
    data_iterator = get_data_iterator(reader=reader, input_dir=input_dir, seed=config.random_seed)

    tokenizer.train_from_iterator(data_iterator, trainer=trainer)

    with NamedTemporaryFile(delete=False) as f:
        tokenizer.save(f.name)
        f_name = f.name

    print(f"Tokenizer saved at {f.name}")

    with open_file_for_write(config.save_path + ".json", "w") as f, open(f_name, "r") as g:
        f.write(g.read())  # type: ignore

    with open_file_for_write(config.save_path + "_config.yaml", "w") as f:
        f.write(sp.to_yaml(config))

    os.remove(f_name)

    print(f"Tokenizer uploaded to {config.save_path}")


if __name__ == "__main__":
    train_tokenizer()
