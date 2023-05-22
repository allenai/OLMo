from .data_types import DocResult, Document, Span
from .registry import TaggerRegistry
from .taggers import BaseTagger

__all__ = [
    "BaseTagger",
    "DocResult",
    "Document",
    "Span",
    "TaggerRegistry",
]
