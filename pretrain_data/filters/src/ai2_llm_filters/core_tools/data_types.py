"""

Data types assumed by Filters.

@kylel, @soldni

"""

from typing import Any, Dict, List, Optional, Union

from msgspec import Struct


class InputSpec(Struct):
    id: str
    text: str
    source: str
    version: Optional[str] = None


class OutputSpec(Struct):
    source: str
    id: str
    attributes: Dict[str, List[Union[int, float]]]


class Document:
    __slots__ = "source", "version", "id", "text"

    def __init__(self, source: str, id: str, text: str, version: Optional[str] = None) -> None:
        self.source = source
        self.version = version
        self.id = id
        self.text = text

    @classmethod
    def from_json(cls, d: Dict) -> "Document":
        return Document(source=d["source"], version=d["version"], id=d["id"], text=d["text"])

    def to_json(self) -> Dict:
        return {"source": self.source, "version": self.version, "id": self.id, "text": self.text}

    def __str__(self) -> str:
        return (
            str(self.__class__.__name__)
            + f"(source={repr(self.source)},version={repr(self.version)},id={repr(self.id)},text={repr(self.text)})"
        )


class Span:
    __slots__ = "start", "end", "type", "score"

    def __init__(self, start: int, end: int, type: str, score: float = 1.0):
        self.start = start
        self.end = end
        self.type = type
        self.score = float(score)

    def mention(self, text: str, window: int = 0) -> str:
        return text[max(0, self.start - window) : min(len(text), self.end + window)]

    @classmethod
    def from_json(cls, di: Dict) -> "Span":
        return Span(start=di["start"], end=di["end"], type=di["type"], score=di["score"])

    def to_json(self, text: Optional[str] = None, window: int = 0) -> dict:
        span_repr = {"start": self.start, "end": self.end, "type": self.type, "score": self.score}
        if text is not None:
            span_repr["mention"] = self.mention(text=text, window=window)
        return span_repr

    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(start={self.start},end={self.end},type={repr(self.type)},score={self.score:.5f})"


class DocResult:
    __slots__ = "doc", "spans"

    def __init__(self, doc: Document, spans: List[Span]) -> None:
        self.doc = doc
        self.spans = spans

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "DocResult":
        return DocResult(
            doc=Document.from_json(d["doc"]),
            spans=[Span.from_json(span) for span in d["spans"]],
        )

    def to_json(self, with_doc: bool = False, window: int = 0) -> Dict[str, Any]:
        d: Dict[str, Any] = {"spans": [span.to_json(text=self.doc.text, window=window) for span in self.spans]}
        if with_doc:
            d["doc"] = self.doc.to_json()
        return d

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(doc={self.doc},spans=[{','.join(str(s) for s in self.spans)}])"


class TextSlice:
    """A slice of text from a document."""

    __slots__ = "doc", "start", "end"

    def __init__(self, doc: str, start: int, end: int):
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def text(self) -> str:
        return self.doc[self.start : self.end]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(text={repr(self.text)},start={self.start},end={self.end})"
