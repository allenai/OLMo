"""

Filters.

@kylel, @soldni

"""
import os
import re
from typing import List, Optional, Literal

# fasttext
import fasttext
import wget
from nltk.tokenize.punkt import PunktSentenceTokenizer

# pii
from presidio_analyzer import AnalyzerEngine

from ..core_tools.data_types import DocResult, Document, Span
from ..core_tools.taggers import BaseTagger
from ..core_tools.registry import TaggerRegistry


class BaseFastTextTagger(BaseTagger):
    # Either provide path to a trained model file on S3 (<bucket>/<file>)
    # or provide paths to training and test data files
    # do_train flag controls whether classifier needs to be trained
    # do_test flag controls whether to check classifier accuracy on test data
    # level flag controls whether prediction should be run at document or sentence level
    # By default, model is loaded from a pretrained model file and run at
    # document level
    def __init__(
        self,
        model_path: Optional[str] = None,
        trainfile: Optional[str] = None,
        testfile: Optional[str] = None,
        savepath: Optional[str] = None,
        do_train: bool = False,
        do_test: bool = False,
        level: str = "doc",
    ) -> None:
        self.classifier = None
        if model_path is not None:
            if os.path.exists(model_path):
                file_name = model_path
            else:
                file_name = wget.download(model_path, out=savepath)
            self.classifier = fasttext.load_model(file_name)
        elif do_train:
            assert trainfile is not None, "Please provide a path to save the train a model"
            if savepath:
                self.save(savepath)
        else:
            raise NotImplementedError("Please either provide a path to a trained model or train a new model")
        if do_test:
            assert testfile is not None, "Please provide a path to test data"
            self.test(testfile)


        assert level in ["doc", "sent"], "Please provide a valid level, either 'doc' or 'sent'"
        self.level = level

        if self.level == "sent":
            self.sent_tokenizer = PunktSentenceTokenizer()

    def save(self, outdir: str):
        assert self.classifier is not None, "Please either download or train a classifier model first"
        self.classifier.save_model(outdir)

    def train(self, trainfile: str):
        if trainfile is None:
            raise ValueError("Please provide a file containing training data")
        classifier = fasttext.train_supervised(input=trainfile, epoch=100, wordNgrams=2)
        self.classifier = classifier

    def test(self, testfile: str):
        if testfile is None:
            raise ValueError("Please provide a file containing test data")
        if self.classifier is None:
            raise ValueError("Please either download or train a classifier model first")
        model_performance = self.classifier.test("jigsaw_nsfw/fasttext_final.test")
        print(model_performance)

    def predict(self, doc: Document) -> DocResult:
        if self.classifier is None:
            raise ValueError("Please either download or train a classifier model first")

        spans: List[Span] = []

        if self.level == "doc":
            # Perform prediction at document level
            # FastText complains about newlines so replace them with spaces
            text = doc.text.replace("\n", " ")
            _, pred_probs = self.classifier.predict(text, k=-1)
            score = pred_probs[1]
            spans.append(Span(start=0, end=len(doc.text), type="ft_doc", score=score))

        elif self.level == "sent":
            # Perform prediction at sentence level
            sent_span_indices = self.sent_tokenizer.span_tokenize(doc.text)

            for start, end in sent_span_indices:
                sentence_text = doc.text[start:end].replace("\n", " ")
                _, pred_probs = self.classifier.predict(sentence_text, k=-1)
                score = pred_probs[1]
                span = Span(start=start, end=end, type="ft_sent", score=score)
                spans.append(span)
        else:
            raise NotImplementedError("Please provide a valid granularity for prediction (doc or sent)")

        return DocResult(doc=doc, spans=spans)



class BasePiiFilter(BaseTagger):
    EMAIL = "EMAIL_ADDRESS"
    PHONE = "PHONE_NUMBER"
    IP = "IP_ADDRESS"

    PRESIDIO = "presidio"
    REGEX = "regex"

    ENGLISH = "en"
    WINDOW = 100

    def __init__(
        self,
        method: str,
        postprocess: bool,
        window: int,
    ) -> None:
        assert method in [self.PRESIDIO, self.REGEX], \
            f"Please provide a valid method for filtering ({self.PRESIDIO} or {self.REGEX})"

        # configs
        self.method = method
        self.postprocess = postprocess
        self.window = window

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

    def predict(self, doc: Document) -> DocResult:
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
        new_pii_spans.append(Span(start=0, end=len(doc.text), type="doc", score=score))
        return DocResult(doc=doc, spans=new_pii_spans)

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


@TaggerRegistry.add("pii_presidio_v1")
class PiiPresidioV1(BasePiiFilter):
    def __init__(self):
        super().__init__(method=self.PRESIDIO, postprocess=True, window=self.WINDOW)


@TaggerRegistry.add("pii_regex_v1")
class PiiRegexV1(BasePiiFilter):
    def __init__(self):
        super().__init__(method=self.REGEX, postprocess=True, window=self.WINDOW)
