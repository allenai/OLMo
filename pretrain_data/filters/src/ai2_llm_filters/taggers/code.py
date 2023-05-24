"""

Code secrets.

@akshitab

"""
import logging
import re
from typing import Generator, List

from detect_secrets import SecretsCollection
from detect_secrets.core.scan import (
    PotentialSecret,
    _process_line_based_plugins,
    get_plugins,
)
from detect_secrets.settings import default_settings

from ..core_tools.data_types import DocResult, Document, Span
from ..core_tools.registry import TaggerRegistry
from ..core_tools.taggers import BaseTagger

logger = logging.getLogger(__name__)


def scan_code(code: str) -> Generator[PotentialSecret, None, None]:
    if not get_plugins():
        logger.error("No plugins to scan with!")
        return

    has_secret = False
    for lines in [code.splitlines()]:
        for secret in _process_line_based_plugins(
            lines=list(enumerate(lines, start=1)),
            filename="code_str.yml",
        ):
            has_secret = True
            yield secret

        if has_secret:
            break


class SecretsCollectionForStringInput(SecretsCollection):
    def scan_str(self, code_str: str):
        for secret in scan_code(code_str):
            self["code_str.yml"].add(secret)


def get_secrets(code: str):
    secrets = SecretsCollectionForStringInput()
    with default_settings():
        secrets.scan_str(code)

    return secrets


@TaggerRegistry.add("code_secrets_v1")
class CodeSecretsTagger(BaseTagger):
    @classmethod
    def _extract_code_secrets(cls, text: str) -> List[Span]:
        secrets_spans: List[Span] = []

        text_lines = text.splitlines()
        secrets = get_secrets(text)
        for _, secret in secrets:
            line_number = secret.line_number - 1
            span = secret.secret_value
            span_line = text_lines[line_number]
            line_start = text.find(span_line)
            start = line_start + span_line.find(span)
            end = start + len(span)
            assert text[start:end] == span
            secret_type = secret.type.replace(" ", "_")
            secrets_spans.append(Span(start=start, end=end, type=f"SECRET_{secret_type}"))  # , span])

        return secrets_spans

    def predict(self, doc: Document) -> DocResult:
        """Main runner."""
        spans = self._extract_code_secrets(doc.text)

        # document-level score
        score = self._score(text=doc.text, secrets_spans=spans)
        spans.append(Span(start=0, end=len(doc.text), type="doc", score=score))
        return DocResult(doc=doc, spans=spans)

    def _score(self, text: str, secrets_spans: List[Span]) -> float:
        try:
            score = len(secrets_spans) * 1.0 / len(text.split())
        except ZeroDivisionError:
            score = -1.0
        return score


@TaggerRegistry.add("code_copyright_comments_v1")
class CodeCopyrightTagger(BaseTagger):
    """
    Based on RedPajama code filtering.
    """

    def __init__(self):
        self.cpat = re.compile("copyright", re.IGNORECASE)
        self.pat = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")

    def _extract_copyright_spans(self, text: str) -> List[Span]:
        copyright_spans: List[Span] = []

        reg = self.pat.search(text)

        if reg:
            # found one, now see if it contains "copyright", if so strip it
            span = reg.span()
            sub = text[span[0] : span[1]]
            if self.cpat.search(sub):
                copyright_spans.append(Span(start=span[0], end=span[1], type="copyright_notice", score=1.0))
            return copyright_spans

        lines = text.split("\n")
        skip = 0
        # Greedy replace any file that begins with comment block, most
        # are copyright headers
        end = 0
        for k in range(len(lines)):
            if lines[k].startswith("//") or lines[k].startswith("#") or lines[k].startswith("--") or not lines[k]:
                skip = skip + 1
                if not lines[k]:
                    end += 1
                else:
                    end += len(lines[k])
            else:
                break

        if skip:
            copyright_spans.append(Span(start=0, end=end, type="comment_block", score=1.0))
        return copyright_spans

    def predict(self, doc: Document) -> DocResult:
        """Main runner."""
        spans = self._extract_copyright_spans(doc.text)

        # document-level score
        score = self._score(text=doc.text, copyright_spans=spans)
        spans.append(Span(start=0, end=len(doc.text), type="doc", score=score))
        return DocResult(doc=doc, spans=spans)

    def _score(self, text: str, copyright_spans: List[Span]) -> float:
        try:
            if len(copyright_spans) == 0:
                score = 0.0
            else:
                span = copyright_spans[0]
                # percentage of content affected
                score = (span.end - span.start + 1) * 1.0 / len(text)
        except ZeroDivisionError:
            score = -1.0
        return score
