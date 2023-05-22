"""

Filters.

@kylel, @soldni

"""
from typing import Iterable, List, Tuple

import cld3
import pycld2 as cld2
import regex
from unidecode import unidecode

from ..core_tools.data_types import DocResult, Document, Span, TextSlice
from ..core_tools.ft_tagger import BaseFastTextTagger, Prediction
from ..core_tools.registry import TaggerRegistry
from ..core_tools.taggers import BaseTagger
from ..core_tools.utils import split_paragraphs


@TaggerRegistry.add("cld3_en_doc_v2")
class Cld3LanguageTagger(BaseTagger):
    def _predict_text(self, text: str) -> Tuple[str, float]:
        pred = cld3.get_language(text)  # pyright: ignore
        score = pred.probability if pred.language == "en" else 0.0
        return "en", score

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        positive_span = Span(start=0, end=len(doc.text), type=lang, score=score)
        negative_span = Span(start=0, end=len(doc.text), type=f"not_{lang}", score=1.0 - score)
        return DocResult(doc=doc, spans=[positive_span, negative_span])


@TaggerRegistry.add("cld3_en_paragraph_v2")
class Cld3LanguageTaggerParagraph(Cld3LanguageTagger):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("cld2_en_doc_v2")
class Cld2LanguageFilter(BaseTagger):
    RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")

    def _sanitize_input(self, text: str) -> str:
        return self.RE_BAD_CHARS.sub("", text)

    def _to_ascii_input(self, text: str) -> str:
        return unidecode(text)

    def _identity_fn(self, text: str) -> str:
        return text

    def _predict_text(self, text: str) -> Tuple[str, float]:
        details = []
        is_reliable = False
        for fn in (self._identity_fn, self._to_ascii_input, self._sanitize_input):
            try:
                is_reliable, _, details = cld2.detect(fn(text))
                break
            except cld2.error:
                ...

        score = max([d[2] for d in details if d[0] == "ENGLISH" and is_reliable] or [0])
        return "en", score / 100.0

    def predict(self, doc: Document) -> DocResult:
        lang, score = self._predict_text(doc.text)
        positive_span = Span(start=0, end=len(doc.text), type=lang, score=score)
        negative_span = Span(start=0, end=len(doc.text), type=f"not_{lang}", score=1.0 - score)
        return DocResult(doc=doc, spans=[positive_span, negative_span])


@TaggerRegistry.add("cld2_en_paragraph_v2")
class Cld2LanguageFilterParagraph(Cld2LanguageFilter):
    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []
        for paragraph in paragraphs:
            lang, score = self._predict_text(paragraph.text)  # pyright: ignore
            positive_span = Span(start=paragraph.start, end=paragraph.end, type=lang, score=score)
            negative_span = Span(start=paragraph.start, end=paragraph.end, type=f"not_{lang}", score=1.0 - score)
            spans.extend((positive_span, negative_span))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("ft_lang_id_en_doc_v2")
class FastTextEnglishLanguageDocumentTagger(BaseFastTextTagger):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        pred = self.classifier.predict(text_slice.text.lower().replace("\n", " ").strip(), k=-1)
        for label, score in zip(*pred):
            if label == "__label__en":
                return Prediction(label="en", score=score), Prediction(label="not_en", score=1.0 - score)
        return Prediction(label="en", score=0.0), Prediction(label="not_en", score=1.0)


@TaggerRegistry.add("ft_lang_id_en_paragraph_v2")
class FastTextEnglishLanguageParagraphTagger(FastTextEnglishLanguageDocumentTagger):
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.PARAGRAPH_LEVEL_TAGGER)
