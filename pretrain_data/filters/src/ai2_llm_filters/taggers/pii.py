"""

Filters.

@kylel, @soldni

"""

try:
    import re2 as re
except ImportError:
    import re
else:
    re.set_fallback_notification(re.FALLBACK_WARNING)


from typing import List

from presidio_analyzer import AnalyzerEngine

from ..core_tools.data_types import DocResult, Document, Span, TextSlice
from ..core_tools.registry import TaggerRegistry
from ..core_tools.taggers import BaseTagger
from ..core_tools.utils import split_paragraphs


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
        assert method in [
            self.PRESIDIO,
            self.REGEX,
        ], f"Please provide a valid method for filtering ({self.PRESIDIO} or {self.REGEX})"

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


@TaggerRegistry.add("pii_regex_v2")
class PiiRegexV2(PiiRegexV1):
    def _score(self, text: str, pii_spans: List[Span]) -> float:
        try:
            score = len(pii_spans) * 1.0 / len(text.split())
        except ZeroDivisionError:
            score = -1.0
        return score


@TaggerRegistry.add("pii_regex_with_counts_fast_v2")
class FastPiiRegex(BaseTagger):
    EMAIL_KEY = "EMAIL_ADDRESS"
    PHONE_KEY = "PHONE_NUMBER"
    IP_KEY = "IP_ADDRESS"

    EMAIL_REGEX = "[.\\s@,?!;:)(]*([^\\s@]+@[^\\s@,?!;:)(]+?)[.\\s@,?!;:)(]?[\\s\n\r]"
    PHONE_REGEX = "\\s+\\(?(\\d{3})\\)?[-\\. ]*(\\d{3})[-. ]?(\\d{4})"
    IP_REGEX = "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    URL_REGEX = "(?i)\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:'\".,<>?«»“”‘’]))"  # noqa: E501

    def __init__(
        self,
        email_regex: str = EMAIL_REGEX,
        phone_regex: str = PHONE_REGEX,
        ip_regex: str = IP_REGEX,
        url_regex: str = URL_REGEX,
    ) -> None:
        self.email_regex = re.compile(email_regex)
        self.phone_regex = re.compile(phone_regex)
        self.ip_regex = re.compile(ip_regex)
        self.url_regex = re.compile(url_regex)

        self.pre_ip_regex = re.compile(r"\.[^\s]")
        self.pre_phone_regex = re.compile(r"\d")

    def _false_positive_identifiers(self, text: str) -> bool:
        return "isbn" in text or "doi" in text or "#" in text

    def _predict_email(self, slice: TextSlice) -> List[Span]:
        if "@" not in slice.text:
            return []

        spans = []
        for match in self.email_regex.finditer(slice.text):
            addressee, domain = match.group(1).split("@", 1)
            if addressee.strip() == "(" or "." not in domain:
                continue

            start, end = match.span()
            spans.append(Span(start=start + slice.start, end=end + slice.start, type=self.EMAIL_KEY))

        return spans

    def _predict_phone(self, slice: TextSlice) -> List[Span]:
        if not self.pre_phone_regex.search(slice.text):
            return []

        spans = []
        for match in self.phone_regex.finditer(slice.text):
            start, end = match.span()
            spans.append(Span(start=start + slice.start, end=end + slice.start, type=self.PHONE_KEY))

        return spans

    def _predict_ip(self, slice: TextSlice) -> List[Span]:
        if not self.pre_ip_regex.search(slice.text):
            return []

        spans = []
        for match in self.ip_regex.finditer(slice.text):
            if self._contains_url(match.group(0)):
                continue
            start, end = match.span()
            spans.append(Span(start=start + slice.start, end=end + slice.start, type=self.IP_KEY))

        return spans

    def _contains_url(self, text: str) -> bool:
        return self.url_regex.search(text) is not None

    def predict(self, doc: Document) -> DocResult:
        paragraphs = split_paragraphs(doc.text)
        spans: List[Span] = []

        for paragraph in paragraphs:
            spans.extend(self._predict_email(paragraph))
            spans.extend(self._predict_phone(paragraph))
            spans.extend(self._predict_ip(paragraph))

        # doc level score is the count of spans matching any of the PII types
        score = sum(1.0 for s in spans if s.type != "doc")
        spans.append(Span(start=0, end=len(doc.text), type="doc_count", score=score))

        try:
            # fraction of words that are PII
            score = sum(len(s) for s in spans) / len(doc.text)
        except ZeroDivisionError:
            # empty doc
            score = -1.0

        spans.append(Span(start=0, end=len(doc.text), type="doc_frac", score=score))
        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("pii_regex_with_counts_v2")
class PiiRegexWithCountV2(BasePiiFilter):
    def __init__(self):
        super().__init__(method=self.REGEX, postprocess=True, window=self.WINDOW)

    def predict(self, doc: Document) -> DocResult:
        doc_result = super().predict(doc=doc)
        count = sum(1 for s in doc_result.spans if s.type != "doc")
        doc_result.spans.append(Span(start=0, end=len(doc.text), type="doc_count", score=count))
        return doc_result
