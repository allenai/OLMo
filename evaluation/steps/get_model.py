import logging
import os
from typing import Union

from tango import step

from hf_olmo.add_hf_config_to_olmo_checkpoint import (
    download_remote_checkpoint_and_add_hf_config,
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.getenv("HOME"), "models")


@step("get-model-path", cacheable=True, version="002")
def get_model_path(model_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    if "olmo" in model_path:  # TODO: ugly. fix.
        local_model_path = download_remote_checkpoint_and_add_hf_config(
            checkpoint_dir=str(model_path), local_dir=MODEL_DIR
        )
    else:
        local_model_path = model_path

    return local_model_path
