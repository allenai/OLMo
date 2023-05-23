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


def extract_code_secrets(text: str):
    secrets_spans: List[List] = []

    secrets = get_secrets(text)
    for _, secret in secrets:
        line_number = secret.line_number - 1
        span = secret.secret_value
        span_line = text.splitlines()[line_number]
        line_start = text.find(span_line)
        start = line_start + span_line.find(span)
        end = start + len(span)
        assert text[start:end] == span
        secret_type = secret.type.replace(" ", "_")
        secrets_spans.append([start, end, f"SECRET_{secret_type}", span])

    return secrets_spans


@TaggerRegistry.add("code_secrets_v1")
class CodeSecretsFilter(BaseTagger):
    @classmethod
    def _extract_code_secrets(cls, text: str) -> List[Span]:
        secrets_spans: List[List] = []

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
            secrets_spans.append(Span(start=start, end=end, type=f"SECRET_{secret_type}")) #, span])

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
