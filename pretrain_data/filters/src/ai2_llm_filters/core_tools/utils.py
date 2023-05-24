import re
import string
from typing import List

import blingfire

from .data_types import TextSlice


def make_variable_name(name: str, remove_multiple_underscores: bool = False) -> str:
    # use underscores for any non-valid characters in variable name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    if remove_multiple_underscores:
        # replace multiple underscores with a single underscore
        name = re.sub(r"__+", "_", name)

    if name[0] in string.digits:
        raise ValueError(f"Invalid variable name {name}")

    return name


def split_paragraphs(text: str) -> List[TextSlice]:
    """
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character, or a sequence of one or more characters, followed by the end of the string.
    """
    return [
        TextSlice(doc=text, start=match.start(), end=match.end())
        for match in re.finditer(r"([^\n]*\n|[^\n]+$)", text)
    ]


def split_sentences(text: str) -> List[TextSlice]:
    """
    Split a string into sentences.
    """
    _, offsets = blingfire.text_to_sentences_and_offsets(text)
    return [TextSlice(doc=text, start=start, end=end) for (start, end) in offsets]
