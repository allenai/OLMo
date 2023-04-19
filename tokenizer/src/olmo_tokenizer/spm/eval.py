import gzip
import itertools
import json
import os
import unicodedata
from contextlib import ExitStack
from functools import partial
from tempfile import TemporaryDirectory
from typing import List, Literal, Optional, Union

import sentencepiece as spm
import springs as sp
from cached_path import cached_path


@sp.dataclass
class SentencePieceEvalConfig:
    model_path: str = sp.MISSING
    normalization: Union[str, None] = sp.field(
        default="NFC", help="Choose between NFC (default), NFKC, NFD, NFKD, or None."
    )


@sp.cli(SentencePieceEvalConfig)
def eval_tokenizer(config: SentencePieceEvalConfig):
    # load sentencepiece tokenizer model
    sp_model = spm.SentencePieceProcessor()

    sp_model.Load(str(cached_path(config.model_path)))

    texts = [
        "This is a test.",
        "This is another test.\nI love tests!",
        "This is a test with 33 data points.\tAnd some tabs. And other things.",
        "This has spaces. And things. And more spaces.",
    ]

    for text in texts:
        if config.normalization:
            text = unicodedata.normalize(config.normalization, text)

        enc = sp_model.EncodeAsPieces(text)
        dec = sp_model.DecodePieces(enc)
        print("Encoded: ", enc)
        print("Decoded: ", dec)
        print("\n")


if __name__ == "__main__":
    eval_tokenizer()
