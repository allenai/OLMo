"""

Data types assumed by Filters.

@kylel, @soldni

"""

from typing import Dict, List


class Document:
    __slots__ = "source", "version", "id", "text"

    def __init__(self, source: str, version: str, id: str, text: str) -> None:
        self.source = source
        self.version = version
        self.id = id
        self.text = text

    def to_json(self) -> Dict:
        return {"source": self.source, "version": self.version, "id": self.id, "text": self.text}


class Span:
    __slots__ = "start", "end", "type"

    def __init__(self, start: int, end: int, type: str):
        self.start = start
        self.end = end
        self.type = type

    def mention(self, text: str, window: int = 0) -> str:
        return text[max(0, self.start - window) : min(len(text), self.end + window)]

    def to_json(self, text: str, window: int = 0) -> List:
        if text:
            return [self.start, self.end, self.type, self.mention(text=text, window=window)]
        else:
            return [self.start, self.end, self.type]


class DocResult:
    __slots__ = "doc", "spans", "score"

    def __init__(self, doc: Document, spans: List[Span], score: float) -> None:
        self.doc = doc
        self.spans = spans
        self.score = score

    def to_json(self, with_doc: bool = False, window: int = 0) -> Dict:
        d = {
            "score": self.score,
            "spans": [span.to_json(text=self.doc.text, window=window) for span in self.spans],
        }
        if with_doc:
            d["doc"] = self.doc.to_json()
        return d
