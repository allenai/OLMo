# from typing import Union
import logging
import os
# import pathlib
import logging
from tango import step
logger = logging.getLogger(__name__)
# from cached_path import cached_path
# from hf_olmo.add_hf_config_to_olmo_checkpoint import write_config
#
#
# def get_model(model_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
#     with cached_path()
#     write_config(model_path)
#     return model_path

@step("env-location-check", version="002")
def env_location_check(env_var: str) -> bool:
    location = os.getenv(env_var, "None")
    print(location)
    print(os.path.exists(location))
    return os.path.exists(location)
