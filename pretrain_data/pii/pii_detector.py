"""

PII detector module.

Thanks to @Abhilasha Ravichander. Refactored by @kylel.

"""

import re
from typing import Dict, List, Optional, Tuple

from presidio_analyzer import AnalyzerEngine

EMAIL = "EMAIL_ADDRESS"
PHONE = "PHONE_NUMBER"
IP = "IP_ADDRESS"
IBAN = "IBAN_CODE"

PRESIDIO = "presidio"
REGEX = "regex"

ENGLISH = "en"

WINDOW = 50


class Document:
    def __init__(self, source: str, version: str, id: str, text: str) -> None:
        self.source = source
        self.version = version
        self.id = id
        self.text = text

    def to_json(self) -> Dict:
        return {"source": self.source, "version": self.version, "id": self.id, "text": self.text}


class PiiSpan:
    def __init__(self, start: int, end: int, type: str):
        self.start = start
        self.end = end
        self.type = type

    def mention(self, text: str) -> str:
        return text[self.start : self.end]

    def context(self, text: str, window: int) -> str:
        return text[max(0, self.start - window) : min(len(text), self.end + window)]

    def to_json(self, context: str) -> List:
        if context:
            return [self.start, self.end, self.type, self.mention(text=context)]
        else:
            return [self.start, self.end, self.type]


class DocResult:
    def __init__(self, doc: Document, pii_spans: List[PiiSpan], score: float) -> None:
        self.doc = doc
        self.pii_spans = pii_spans
        self.score = score

    def to_json(self, with_doc: bool = False) -> Dict:
        d = {"pii": [pii_span.to_json(context=self.doc.text) for pii_span in self.pii_spans], "score": self.score}
        if with_doc:
            d["doc"] = self.doc.to_json()
        return d


class PiiDetector:
    def __init__(self) -> None:
        # Regular expressions for different types of PII
        self.email_regex = re.compile("[.\s@,?!;:)(]*([^\s@]+@[^\s@,?!;:)(]+?)[.\s@,?!;:)(]?[\s\n\r]")
        self.phone_regex = re.compile("\s+\(?(\d{3})\)?[-\. ]*(\d{3})[-. ]?(\d{4})")
        self.ip_regex = re.compile(
            "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        )
        self.url_regex = re.compile(
            "(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        )
        self.analyzer = AnalyzerEngine()

    def _contains_url(self, text: str) -> bool:
        return len(self.url_regex.findall(text)) > 0

    def _is_email(self, text: str, pii_span: PiiSpan) -> bool:
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

    def postprocess(self, text: str, pii_spans: List[PiiSpan], window: int) -> List[PiiSpan]:
        """Applies some rules to remove over-prediction of PII types."""
        new_pii_spans = []
        for pii_span in pii_spans:
            if pii_span.type == EMAIL and self._is_email(text, pii_span):
                new_pii_spans.append(pii_span)

            elif pii_span.type == PHONE or pii_span.type == IP:
                context = pii_span.context(text=text, window=window)
                # for both phone numbers & IP addresses, context shouldnt contain these strings
                if "isbn" in context or "doi" in context or "#" in context:
                    pass
                elif pii_span.type == IP:
                    new_pii_spans.append(pii_span)
                elif pii_span.type == PHONE:
                    # for phone numbers, additionally shouldnt be URL
                    if self._contains_url(text=text):
                        pass
                    else:
                        new_pii_spans.append(pii_span)

            elif pii_span.type == IBAN:
                new_pii_spans.append(pii_span)

            else:
                raise NotImplementedError
        return new_pii_spans

    def predict(
        self, doc: Document, method: str = PRESIDIO, do_postprocess: bool = False, window: int = WINDOW
    ) -> List[DocResult]:
        """Main runner."""
        # extract
        if method == PRESIDIO:
            pii_spans = self._extract_pii_presidio(text=doc.text)
        elif method == REGEX:
            pii_spans = self._extract_pii_regex(text=doc.text)
        else:
            raise NotImplementedError
        # post process
        if do_postprocess:
            new_pii_spans = self.postprocess(text=doc.text, pii_spans=pii_spans, window=window)
        else:
            new_pii_spans = pii_spans
        # document-level score
        score = self.score(text=doc.text, pii_spans=new_pii_spans)
        return DocResult(doc=doc, pii_spans=new_pii_spans, score=score)

    def score(self, text: str, pii_spans: List[PiiSpan]) -> float:
        return len(pii_spans) * 1.0 / len(text.split())

    def _extract_pii_regex(self, text: str) -> List[PiiSpan]:
        raise NotImplementedError("[kylel] Refactor WIP")
        # pii = []

        # for pii_type, regex_pattern in self.pattern_dict.items():
        #     # search for the pattern in the string
        #     matches = regex_pattern.findall(text.lower())
        #     # loop through the matches and print corresponding values from the dictionary
        #     for match in matches:
        #         if self.postprocess(text, match, pii_type):
        #             match = str("".join(match))
        #             pii_start = text.find(match)

        #             if pii_start == -1:
        #                 import pdb

        #                 pdb.set_trace()
        #             pii_end = pii_start + len(match)

        #             pii.append([pii_start, pii_end, pii_type, match])

        # return pii

    def _extract_pii_presidio(self, text: str) -> List[PiiSpan]:
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=[EMAIL, PHONE, IP, IBAN],
            language=ENGLISH,
        )
        pii_spans: List[PiiSpan] = []
        for res in analyzer_results:
            pii_spans.append(PiiSpan(start=res.start, end=res.end, type=res.entity_type))
        return pii_spans
