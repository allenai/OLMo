"""

Filters.

@kylel, @soldni

"""
import os
import re
from abc import abstractmethod
from typing import List, Optional

# language id
import cld3
import pycld2 as cld2
from cached_path import cached_path
from fasttext.FastText import _FastText

# fasttext
import fasttext
import wget
from nltk.tokenize.punkt import PunktSentenceTokenizer

# pii
from presidio_analyzer import AnalyzerEngine

from .data_types import DocResult, Document, Span


class Filter:
    @abstractmethod
    def train(self, trainfile: str):
        raise NotImplementedError

    @abstractmethod
    def save(self, outdir: str):
        raise NotImplementedError

    @abstractmethod
    def predict(self, doc: Document) -> List[DocResult]:
        raise NotImplementedError


class Cld3LanguageFilter(Filter):
    def train(self, trainfile: str):
        pass

    def save(self, outdir: str):
        pass

    def predict(self, doc: Document) -> List[DocResult]:
        pred = cld3.get_language(doc.text)  # pyright: ignore
        return [
            DocResult(
                doc=doc, spans=[Span(start=0, end=len(doc.text), type=pred.language)], score=pred.probability
            )
        ]

    
class Cld2LanguageFilter(Filter):
    def train(self, trainfile: str):
        pass

    def save(self, outdir: str):
        pass

    def predict(self, doc: Document) -> List[DocResult]:
        is_reliable, text_bytes_found, details = cld2.detect(doc.text)
        return [
            DocResult(
                doc=doc,
                spans=[Span(start=0, end=len(doc.text), type=details[0].language_code)],
                score=details[0].percent,
            )
        ]
      
 
class FastTextLanguageFilter(Filter):
    MODEL_PATH = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        # we use this private attribute to avoid a warning from the fasttext library
        # see this comment:
        # https://github.com/facebookresearch/fastText/issues/1056#issuecomment-1278058705
        self.model = _FastText(model_path=str(cached_path(model_path)))

    def train(self, trainfile: str):
        pass

    def save(self, outdir: str):
        pass

    def predict(self, doc: Document) -> List[DocResult]:
        pred = self.model.predict(doc.text.lower().replace("\n", " "))
        lang = pred[0][0].split("__")[-1]  # pyright: ignore
        score = float(pred[1])
        return [DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type=lang)], score=score)]


class FastTextFilter(Filter):

    # Either provide path to a trained model file on S3 (<bucket>/<file>)
    # or provide paths to training and test data files
    # do_train flag controls whether classifier needs to be trained
    # do_test flag controls whether to check classifier accuracy on test data
    # level flag controls whether prediction should be run at document or sentence level
    # By default, model is loaded from a pretrained model file and run at document level
    def __init__(
        self,
        model_path: str = None,
        trainfile: str = None,
        testfile: str = None,
        savepath: str = None,
        do_train: bool = False,
        do_test: bool = False,
        level: str = "doc",
        sent_threshold: float = None,
    ) -> None:
        self.classifier = None
        if model_path is not None:
            if os.path.exists(model_path):
                file_name = model_path
            else:
                file_name = wget.download(model_path, out=savepath)
            self.classifier = fasttext.load_model(file_name)
        elif do_train:
            self.train(trainfile, savepath)
        else:
            raise NotImplementedError("Please either provide a path to a trained model or train a new model")
        if do_test:
            self.test(testfile)
        self.level = level
        if self.level == "sent":
            self.sent_tokenizer = PunktSentenceTokenizer()
            self.sent_threshold = sent_threshold

    def train(self, trainfile: str, savepath: str):
        if trainfile is None:
            raise ValueError("Please provide a file containing training data")
        classifier = fasttext.train_supervised(input=trainfile, epoch=100, wordNgrams=2)
        self.classifier = classifier
        if savepath is None:
            raise Warning("Model will not be saved since no save path was provided")
        else:
            classifier.save_model(savepath)

    def test(self, testfile: str):
        if testfile is None:
            raise ValueError("Please provide a file containing test data")
        if self.classifier is None:
            raise ValueError("Please either download or train a classifier model first")
        model_performance = self.classifier.test("jigsaw_nsfw/fasttext_final.test")
        print(model_performance)

    def predict(self, doc: Document) -> List[DocResult]:
        if self.level == "doc":
            # Perform prediction at document level
            # FastText complains about newlines so replace them with spaces
            text = doc.text.replace("\n", " ")
            pred_label, pred_probs = self.classifier.predict(text, k=-1)
            score = pred_probs[1]
            return DocResult(doc=doc, spans=[], score=score)
        elif self.level == "sent":
            # Perform prediction at sentence level
            filter_sents: List[Span] = []
            sent_span_indices = self.sent_tokenizer.span_tokenize(doc.text)
            filter_sentence_count = 0.0
            total_sentence_count = 0.0
            for start, end in sent_span_indices:
                total_sentence_count += 1
                sentence_text = doc.text[start:end].replace("\n", " ")
                pred_label, pred_probs = self.classifier.predict(sentence_text, k=-1)
                score = pred_probs[1]
                # Can tweak this and experiment with different thresholds
                if score > self.sent_threshold:
                    span = Span(start=start, end=end, type=f"{score:.4f}")
                    filter_sents.append(span)
                    filter_sentence_count += 1
            doc_score = float(filter_sentence_count) / total_sentence_count
            return DocResult(doc=doc, spans=filter_sents, score=doc_score)
        else:
            raise NotImplementedError("Please provide a valid granularity for prediction (doc or sent)")

class PiiFilter(Filter):
    EMAIL = "EMAIL_ADDRESS"
    PHONE = "PHONE_NUMBER"
    IP = "IP_ADDRESS"

    PRESIDIO = "presidio"
    REGEX = "regex"

    ENGLISH = "en"
    WINDOW = 100

    def __init__(
        self, method: Optional[str] = None, postprocess: Optional[bool] = None, window: Optional[int] = None
    ) -> None:
        # configs
        self.method = method if method else self.REGEX
        self.postprocess = postprocess if postprocess else True
        self.window = window if window else self.WINDOW

        # Regular expressions for different types of PII
        self.pii_type_to_regex = {
            self.EMAIL: re.compile("[.\\s@,?!;:)(]*([^\\s@]+@[^\\s@,?!;:)(]+?)[.\\s@,?!;:)(]?[\\s\n\r]"),
            self.PHONE: re.compile("\\s+\\(?(\\d{3})\\)?[-\\. ]*(\\d{3})[-. ]?(\\d{4})"),
            self.IP: re.compile(
                "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
            ),
        }
        self.url_regex = re.compile(
            "(?i)\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|"
            "(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]"
            "{};:'\".,<>?«»“”‘’]))"
        )

        # presidio
        if self.method == self.PRESIDIO:
            self.analyzer = AnalyzerEngine()

    def predict(self, doc: Document) -> List[DocResult]:
        """Main runner."""
        # extract
        if self.method == self.PRESIDIO:
            pii_spans = self._extract_pii_presidio(text=doc.text)
        elif self.method == self.REGEX:
            pii_spans = self._extract_pii_regex(text=doc.text)
        else:
            raise NotImplementedError
        # post process
        if self.postprocess:
            new_pii_spans = self._postprocess(text=doc.text, pii_spans=pii_spans, window=self.window)
        else:
            new_pii_spans = pii_spans
        # document-level score
        score = self._score(text=doc.text, pii_spans=new_pii_spans)
        return [DocResult(doc=doc, spans=new_pii_spans, score=score)]

    def _score(self, text: str, pii_spans: List[Span]) -> float:
        return len(pii_spans) * 1.0 / len(text.split())

    def _extract_pii_regex(self, text: str) -> List[Span]:
        pii_spans: List[Span] = []
        for pii_type, regex in self.pii_type_to_regex.items():
            for match in regex.finditer(text):
                start, end = match.span()
                pii_spans.append(Span(start=start, end=end, type=pii_type))
        return pii_spans

    def _extract_pii_presidio(self, text: str) -> List[Span]:
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=[self.EMAIL, self.PHONE, self.IP],
            language=self.ENGLISH,
        )
        pii_spans: List[Span] = []
        for res in analyzer_results:
            pii_spans.append(Span(start=res.start, end=res.end, type=res.entity_type))
        return pii_spans

    def _postprocess(self, text: str, pii_spans: List[Span], window: int) -> List[Span]:
        """Applies some rules to remove over-prediction of PII types."""
        new_pii_spans = []
        for pii_span in pii_spans:
            if pii_span.type == self.EMAIL:
                if self._is_email(text, pii_span):
                    new_pii_spans.append(pii_span)
                else:
                    pass

            elif pii_span.type == self.PHONE or pii_span.type == self.IP:
                context = pii_span.mention(text=text, window=window)
                # for both phone numbers & IP addresses, context shouldnt
                # contain these strings
                if "isbn" in context or "doi" in context or "#" in context:
                    pass
                elif pii_span.type == self.IP:
                    new_pii_spans.append(pii_span)
                elif pii_span.type == self.PHONE:
                    # for phone numbers, additionally shouldnt be URL
                    if self._contains_url(text=text):
                        pass
                    else:
                        new_pii_spans.append(pii_span)

            else:
                raise NotImplementedError(f"Unsupported PII type for Postprocess: {pii_span.type}")
        return new_pii_spans

    def _contains_url(self, text: str) -> bool:
        return len(self.url_regex.findall(text)) > 0

    def _is_email(self, text: str, pii_span: Span) -> bool:
        """
        Rules:
        (1) The email address besides the domain, cannot be only "("
        (2) There must be a "." in the domain
        """
        mention = pii_span.mention(text=text)
        addressee = mention.split("@")[0]
        domain = mention.split("@")[1]
        if addressee.strip() == "(" or "." not in domain:
            return False
        return True
