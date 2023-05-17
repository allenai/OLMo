"""

Filters.

@kylel, @soldni

"""
from typing import List, Tuple

import cld3
import pycld2 as cld2

from ..core_tools.data_types import DocResult, Document, Span, TextSlice
from ..core_tools.ft_tagger import BaseFastTextTagger, Prediction
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


@TaggerRegistry.add("ft_lang_id_en_doc_v1")
class FastTextEnglishLanguageDocumentTagger(BaseFastTextTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Prediction:
        pred = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip(), k=-1)
        for label, score in zip(*pred):
            if label == "__label__en":
                return Prediction(label=label, score=score)
        return Prediction(label="__label__en", score=0.0)


@TaggerRegistry.add("ft_lang_id_en_paragraph_v1")
class FastTextEnglishLanguageParagraphTagger(FastTextEnglishLanguageDocumentTagger):
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)
