"""

Filters.

@kylel, @soldni

"""
from abc import abstractmethod
from typing import Dict

from .data_types import DocResult, Document, InputSpec


class BaseTagger:
    @classmethod
    def environment_setup(cls) -> None:
        """Run any setup code for the tagger; this is called only once on startup, and not per process. You
        likely don't need this, but it is useful for things like downloading models, installing binaries, etc.
        """
        pass

    @classmethod
    def train(cls, *args, **kwargs):
        raise RuntimeError("This tagger does not support training")

    @classmethod
    def test(cls, *args, **kwargs):
        raise RuntimeError("This tagger does not support testing")

    @abstractmethod
    def predict(self, doc: Document) -> DocResult:
        raise NotImplementedError

    def tag(self, row: InputSpec) -> Dict[str, list]:
        """Internal function that is used by the tagger to get data"""
        doc = Document(source=row.source, version=row.version, id=row.id, text=row.text)
        doc_result = self.predict(doc)

        tagger_output: Dict[str, list] = {}
        for span in doc_result.spans:
            tagger_output.setdefault(span.type, []).append([span.start, span.end, round(span.score, 5)])
        return tagger_output
