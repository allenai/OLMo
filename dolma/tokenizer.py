from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List, Optional, Union

from tokenizers import Tokenizer as BaseTokenizer

from .config import Config
from .util import StrEnum

__all__ = ["Tokenizer", "TruncationDirection"]


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around :class:`tokenizers.Tokenizer`.

    :param base_tokenizer: The :class:`tokenizers.Tokenizer` to use.
    :param config: The DOLMA config.
    :param truncate_to: Truncate when tokenizing to this number of token IDs.
    :param truncate_direction: The direction to truncate in. "right" means truncate the tokens
        on the right. "left" means truncate the tokens on the left. If ``truncate_to`` is null,
        this setting has no effect.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        config: Config,
        truncate_to: Optional[int] = None,
        truncate_direction: Union[str, TruncationDirection] = TruncationDirection.right,
    ):
        self.base_tokenizer = base_tokenizer
        self.config = config
        self.truncate_to = truncate_to
        self.truncate_direction = TruncationDirection(truncate_direction)
        assert self.config.vocab_size == self.base_tokenizer.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        return self.config.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @classmethod
    def from_pretrained(cls, identifier: str, config: Optional[Config] = None, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param config: The DOLMA config.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        if config is None:
            config = Config(
                vocab_size=base_tokenizer.get_vocab_size(), eos_token_id=base_tokenizer.get_vocab_size() - 1
            )
        return cls(base_tokenizer, config, **kwargs)

    def add_special_tokens(self, input_ids: List[int]) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if not input_ids or input_ids[-1] != self.eos_token_id:
            input_ids.append(self.eos_token_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1

    @contextmanager
    def _truncation(
        self, truncate_to: Optional[int], direction: Union[str, TruncationDirection] = TruncationDirection.right
    ) -> Generator["Tokenizer", None, None]:
        """
        A context manager to temporarily enable/disable truncation.
        """
        truncation = self.base_tokenizer.truncation

        try:
            if truncate_to is not None:
                self.base_tokenizer.enable_truncation(truncate_to, direction=str(direction))
            else:
                self.base_tokenizer.no_truncation()
            yield self
        finally:
            if truncation is None:
                self.base_tokenizer.no_truncation()
            else:
                self.base_tokenizer.enable_truncation(**truncation)

    def encode(self, input: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        with self._truncation(truncate_to, direction=self.truncate_direction):
            input_ids = self.base_tokenizer.encode(input).ids

        if add_special_tokens:
            input_ids = self.add_special_tokens(input_ids)

        return input_ids

    def encode_batch(self, inputs: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
        truncate_to = self.truncate_to
        if truncate_to is not None and add_special_tokens:
            truncate_to -= self.num_special_tokens_to_add(False)

        with self._truncation(truncate_to, direction=self.truncate_direction):
            batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = encoding.ids
            if add_special_tokens:
                input_ids = self.add_special_tokens(input_ids)
            all_input_ids.append(input_ids)

        return all_input_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.base_tokenizer.decode(token_ids)
