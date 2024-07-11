from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import importlib_resources
from importlib_resources.abc import Traversable


def _get_data_traversable(data_rel_path: str) -> Traversable:
    return importlib_resources.files("olmo_data").joinpath(data_rel_path)


def is_data_dir(data_rel_path: str) -> bool:
    return _get_data_traversable(data_rel_path).is_dir()


def is_data_file(data_rel_path: str) -> bool:
    return _get_data_traversable(data_rel_path).is_file()


@contextmanager
def get_data_path(data_rel_path: str) -> Generator[Path, None, None]:
    try:
        with importlib_resources.as_file(_get_data_traversable(data_rel_path)) as path:
            yield path
    finally:
        pass
