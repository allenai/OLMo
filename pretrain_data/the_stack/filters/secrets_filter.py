import logging
from typing import Generator

from detect_secrets import SecretsCollection
from detect_secrets.core.scan import (
    PotentialSecret,
    _process_line_based_plugins,
    get_plugins,
)
from detect_secrets.settings import default_settings

logger = logging.getLogger(__name__)

THE_STR = "code_str.yml"


def scan_code(code: str) -> Generator[PotentialSecret, None, None]:
    if not get_plugins():
        logger.error("No plugins to scan with!")
        return

    has_secret = False
    for lines in [code.splitlines()]:
        for secret in _process_line_based_plugins(
            lines=list(enumerate(lines, start=1)),
            filename=THE_STR,
        ):
            has_secret = True
            yield secret

        if has_secret:
            break


class SecretsCollectionForStringInput(SecretsCollection):
    def scan_str(self, code_str: str):
        for secret in scan_code(code_str):
            self[THE_STR].add(secret)


def get_secrets(code: str):
    secrets = SecretsCollectionForStringInput()
    with default_settings():
        secrets.scan_str(code)

    return secrets
