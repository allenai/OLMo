"""

Filters.

@kylel, @soldni

"""
from abc import abstractmethod
from typing import Dict, List, Union

from .data_types import DocResult, Document, InputSpec


class BaseTagger:
    @classmethod
    def train(cls, *args, **kwargs):
        raise RuntimeError("This tagger does not support training")

    @classmethod
    def test(cls, *args, **kwargs):
        raise RuntimeError("This tagger does not support testing")

    @abstractmethod
    def predict(self, doc: Document) -> DocResult:
        raise NotImplementedError

    def tag(self, row: InputSpec) -> Dict[str, List[List[Union[int, float]]]]:
        """Internal function that is used by the tagger to get data"""
        doc = Document(source=row.source, version=row.version, id=row.id, text=row.text)
        doc_result = self.predict(doc)

        tagger_output: Dict[str, list] = {}
        for span in doc_result.spans:
            tagger_output.setdefault(span.type, []).append([span.start, span.end, round(float(span.score), 5)])
        return tagger_output
