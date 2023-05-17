"""

Filters.

@kylel, @soldni

"""
from abc import abstractmethod
from typing import Any, Dict

from .data_types import DocResult, Document


class BaseTagger:
    @abstractmethod
    def train(self, trainfile: str):
        ...

    @abstractmethod
    def save(self, outdir: str):
        ...

    @abstractmethod
    def predict(self, doc: Document) -> DocResult:
        raise NotImplementedError

    def tag(self, row: Dict[str, Any]) -> Dict[str, list]:
        """Internal function that is used by the tagger to get data """
        doc = Document(source=row["source"], version=row["version"], id=row["id"], text=row["text"])
        doc_result = self.predict(doc)
        tagger_output: Dict[str, list] = {}
        for span in doc_result.spans:
            tagger_output.setdefault(span.type, []).append([span.start, span.end, span.score])
        return tagger_output
