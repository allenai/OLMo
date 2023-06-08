"""

Data types assumed by Filters.

@kylel, @soldni

"""

from typing import Any, Dict, List, Optional, Tuple, Union

from msgspec import Struct


class Ai2LlmFatalError(Exception):
    """Fatal error. Abort the entire process"""

    pass


class Ai2LlmShardError(Exception):
    """Fail the shard and continue"""


class Ai2LlmRetryableFailure(Exception):
    """Retry if a shard throws this error"""

    pass


class InputSpec(Struct):
    id: str
    text: str
    source: str
    version: Optional[str] = None


class OutputSpec(Struct):
    id: str
    attributes: Dict[str, List[List[Union[int, float]]]]
    source: Optional[str] = None


class Document:
    __slots__ = "source", "version", "id", "text"

    def __init__(self, source: str, id: str, text: str, version: Optional[str] = None) -> None:
        self.source = source
        self.version = version
        self.id = id
        self.text = text

    @classmethod
    def from_spec(cls, spec: InputSpec) -> "Document":
        return Document(source=spec.source, version=spec.version, id=spec.id, text=spec.text)

    def to_spec(self) -> InputSpec:
        return InputSpec(source=self.source, version=self.version, id=self.id, text=self.text)

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
    __slots__ = "start", "end", "type", "score", "experiment", "tagger"

    def __init__(
        self,
        start: int,
        end: int,
        type: str,
        score: float = 1.0,
        experiment: Optional[str] = None,
        tagger: Optional[str] = None,
    ):
        self.start = start
        self.end = end
        self.type = type
        self.score = float(score)
        self.experiment = experiment
        self.tagger = tagger

    def mention(self, text: str, window: int = 0) -> str:
        return text[max(0, self.start - window) : min(len(text), self.end + window)]

    def select(self, doc: Document) -> str:
        return doc.text[self.start : self.end]

    @classmethod
    def from_spec(cls, attribute_name: str, attribute_value: List[Union[int, float]]) -> "Span":
        if "__" in attribute_name:
            # bff tagger has different name
            exp_name, tgr_name, attr_type = attribute_name.split("__", 2)
        else:
            exp_name = tgr_name = attr_type = attribute_name

        start, end, score = attribute_value
        return Span(
            start=int(start),
            end=int(end),
            type=attr_type,
            score=float(score),
            experiment=exp_name,
            tagger=tgr_name,
        )

    def to_spec(self) -> Tuple[str, List[Union[int, float]]]:
        assert self.experiment is not None, "Experiment name must be set to convert to spec"
        assert self.tagger is not None, "Tagger name must be set to convert to spec"
        return (
            f"{self.experiment}__{self.tagger}__{self.type}",
            [self.start, self.end, self.score],
        )

    def __len__(self) -> int:
        return self.end - self.start

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
    def from_spec(cls, doc: InputSpec, *attrs_groups: OutputSpec) -> "DocResult":
        spans: List[Span] = []
        for attrs in attrs_groups:
            assert doc.id == attrs.id, f"doc.id={doc.id} != attrs.id={attrs.id}"
            spans.extend(
                [
                    Span.from_spec(attribute_name=attr_name, attribute_value=attr_value)
                    for attr_name, attr_values in attrs.attributes.items()
                    for attr_value in attr_values
                ]
            )
        return DocResult(doc=Document.from_spec(doc), spans=spans)

    def to_spec(self) -> Tuple[InputSpec, OutputSpec]:
        doc_spec = self.doc.to_spec()
        attributes: Dict[str, List[List[Union[int, float]]]] = {}

        for span in self.spans:
            attr_name, attr_value = span.to_spec()
            attributes.setdefault(attr_name, []).append(attr_value)

        return doc_spec, OutputSpec(source=self.doc.source, id=self.doc.id, attributes=attributes)

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
