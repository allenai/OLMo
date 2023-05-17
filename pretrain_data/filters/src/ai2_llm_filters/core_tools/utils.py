import re
import string
from typing import List, NamedTuple


def make_variable_name(name: str) -> str:
    # use underscores for any non-valid characters in variable name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # replace multiple underscores with a single underscore
    name = re.sub(r"__+", "_", name)

    if name[0] in string.digits:
        raise ValueError(f"Invalid variable name {name}")

    return name


class Paragraph(NamedTuple):
    text: str
    start: int
    end: int


def split_paragraphs(text: str) -> List[Paragraph]:
    """
    Split a string into paragraphs.
    """
    return [
        Paragraph(text=match.group(0), start=match.start(), end=match.end())
        for match in re.finditer(r"[^\n]+(\n+|$)", text)
    ]
