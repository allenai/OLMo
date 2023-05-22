"""

Filters.

@kylel, @soldni

"""

import regex
from tokenizers import Regex, pre_tokenizers

from ..core_tools.data_types import DocResult, Document, Span
from ..core_tools.registry import TaggerRegistry
from ..core_tools.taggers import BaseTagger


@TaggerRegistry.add("char_length_v1")
class CharLengthV1(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = len(doc.text)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("whitespace_tokenizer_v1")
class WhitespaceLengthV1(BaseTagger):
    WHITESPACE_REGEX = regex.compile(r"\w+|[^\w\s]+")

    def predict(self, doc: Document) -> DocResult:
        score = len(self.WHITESPACE_REGEX.split(doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])


@TaggerRegistry.add("olmo_pretokenizer_v1")
class OlmoPreTokenizerV1(BaseTagger):
    def __init__(self) -> None:
        self.pre_tokenizer = pre_tokenizers.Sequence(  # type: ignore
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

    def predict(self, doc: Document) -> DocResult:
        score = len(self.pre_tokenizer.pre_tokenize_str(doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])
