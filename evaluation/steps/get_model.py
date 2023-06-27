from typing import Union
import os
import pathlib
from tango import step
from cached_path import cached_path
from hf_olmo.add_hf_config_to_olmo_checkpoint import write_config


def get_model(model_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    with cached_path()
    write_config(model_path)
    return model_path