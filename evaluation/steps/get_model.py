import shutil
from typing import Union
import os
import logging
from tango import step

from cached_path import cached_path
from hf_olmo.add_hf_config_to_olmo_checkpoint import write_config

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.getenv("HOME"), "models")


@step("get-model", cacheable=True, version="001")
def get_model(model_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    # TODO: perhaps easier thing would be to create config.json and copy it to
    # whatever remote s3: or gs: location.
    model_name = os.path.basename(model_path)
    local_model_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(local_model_path, exist_ok=True)

    model_files = ["model.pt", "config.yaml"] # "optim.pt", "other.pt"]
    for filename in model_files:
        final_location = os.path.join(local_model_path, filename)
        if not os.path.exists(final_location):
            remote_file = os.path.join(model_path, filename)
            logger.debug(f"Downloading file {filename}")
            cached_file = cached_path(remote_file)
            shutil.copy(cached_file, final_location)
            logger.debug(f"File at {final_location}")
        else:
            logger.info(f"File already present at {final_location}")

    write_config(local_model_path)
    # TODO:  move this to hf_olmo
    return local_model_path

@step("env-location-check", version="003")
def env_location_check(env_var: str) -> bool:
    location = os.getenv(env_var, "None")
    print(location)
    print(os.path.exists(location))
    return os.path.exists(location)
