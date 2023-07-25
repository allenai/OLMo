import logging
import os
from typing import Optional, Union

from tango import step

from hf_olmo.add_hf_config_to_olmo_checkpoint import (
    download_remote_checkpoint_and_add_hf_config,
)

logger = logging.getLogger(__name__)


@step("get-model-path", cacheable=True, version="004")
def get_model_path(
    model_path: Union[str, os.PathLike],
    revision: Optional[str] = None,
) -> Union[str, os.PathLike]:
    # TODO: ugly. fix. Ideally, the model_path already has HF-olmo model.
    if "olmo" in str(model_path):
        try:
            model_dir = os.environ["GLOBAL_MODEL_DIR"]
        except KeyError:
            raise KeyError(
                "Please set `GLOBAL_MODEL_DIR` to some location locally accessible to your experiment run"
                ", like /net/nfs.cirrascale"
            )

        checkpoint_dir = str(model_path)
        if revision:
            checkpoint_dir += "/" + revision

        local_model_path = download_remote_checkpoint_and_add_hf_config(
            checkpoint_dir=checkpoint_dir, local_dir=model_dir
        )
    else:
        local_model_path = model_path
        # if revision:
        #     local_model_path += f",revision={revision}"
        # if trust_remote_code:
        #     local_model_path += f",trust_remote_code={trust_remote_code}"

    return local_model_path
