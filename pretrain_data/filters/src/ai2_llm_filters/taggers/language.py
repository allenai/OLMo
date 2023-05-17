"""

Filters.

@kylel, @soldni

"""
from typing import List, Tuple

import cld3
import pycld2 as cld2
from cached_path import cached_path
from fasttext.FastText import _FastText

from ..core_tools.data_types import DocResult, Document, Span
from ..core_tools.registry import TaggerRegistry
from ..core_tools.taggers import BaseTagger
from ..core_tools.utils import split_paragraphs


@TaggerRegistry.add("cld3_en_doc_v1")
class Cld3LanguageTagger(BaseTagger):
    def _predict_text(self, text: str) -> Tuple[str, float]:
        pred = cld3.get_language(text)  # pyright: ignore
        score = pred.probability if pred.language == "en" else 0.0
        return "en", score

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type=lang, score=score)])


@TaggerRegistry.add("cld3_en_paragraph_v1")
class Cld3LanguageTaggerParagraph(Cld3LanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            spans.append(Span(start=paragraph.start, end=paragraph.end, type=lang, score=score))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("cld2_en_doc_v1")
class Cld2LanguageFilter(BaseTagger):
    def _predict_text(self, text: str) -> Tuple[str, float]:
        is_reliable, text_bytes_found, details = cld2.detect(text)
        score = max([d[2] for d in details if d[0] == "ENGLISH" and is_reliable] or [0.0])
        return "ENGLISH" if is_reliable else "UNKNOWN", score

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type=lang, score=score)])


@TaggerRegistry.add("cld2_en_paragraph_v1")
class Cld2LanguageFilterParagraph(Cld2LanguageFilter):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            spans.append(Span(start=paragraph.start, end=paragraph.end, type=lang, score=score))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("fasttext_en_doc_v1")
class FastTextLanguageFilter(BaseTagger):
    def __init__(self, model_path: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"):
        # we use this private attribute to avoid a warning from the fasttext library
        # see this comment:
        # https://github.com/facebookresearch/fastText/issues/1056#issuecomment-1278058705
        self.model = _FastText(model_path=str(cached_path(model_path)))

    def _predict_text(self, text: str) -> Tuple[str, float]:
        pred = self.model.predict(text.lower().replace("\n", " "))
        score = max([float(p) for p, l in zip(pred[1], pred[0]) if l == "__label__en"] or [0.0])
        return "en", score

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type=lang, score=score)])


@TaggerRegistry.add("fasttext_en_paragraph_v1")
class FastTextLanguageFilterParagraph(FastTextLanguageFilter):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            spans.append(Span(start=paragraph.start, end=paragraph.end, type=lang, score=score))
        return DocResult(doc=doc, spans=spans)
