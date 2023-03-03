import sys
from enum import Enum

from .exceptions import DolmaError

__all__ = ["StrEnum", "install_excepthook"]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    from rich import print
    from rich.traceback import Traceback

    if isinstance(value, DolmaError):
        print(f"[b red]{exctype.__name__}:[/] {value}", file=sys.stderr)
    else:
        print(Traceback.from_exception(exctype, value, traceback), file=sys.stderr)


def install_excepthook():
    sys.excepthook = excepthook
