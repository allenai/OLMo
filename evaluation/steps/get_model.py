import logging
import os
from typing import Union

from tango import step

from hf_olmo.add_hf_config_to_olmo_checkpoint import (
    download_remote_checkpoint_and_add_hf_config,
)

logger = logging.getLogger(__name__)


@step("get-model-path", cacheable=True, version="003")
def get_model_path(model_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    # TODO: ugly. fix.
    if "olmo" in str(model_path):
        try:
            model_dir = os.environ["GLOBAL_MODEL_DIR"]
        except KeyError:
            raise KeyError(
                "Please set `GLOBAL_MODEL_DIR` to some location locally accessible to your experiment run"
                ", like /net/nfs.cirrascale"
            )

        local_model_path = download_remote_checkpoint_and_add_hf_config(
            checkpoint_dir=str(model_path), local_dir=model_dir
        )
    else:
        local_model_path = model_path

    return local_model_path
