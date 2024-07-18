from typing import List, Optional, Union

from mup import get_shapes, make_base_shapes

from olmo.config import ModelConfig
from olmo.model import OLMo


def load_mu_model(config: ModelConfig):
    config.use_mup = True
    model = OLMo(config, init_params=False)
    return model


def save_base_shapes(
    model_config: Union[str, ModelConfig], output_path: str, dims_to_scale: Optional[List] = None
):
    if isinstance(model_config, str):
        model_config = ModelConfig.load(model_config, key="model")

    if dims_to_scale is None:
        dims_to_scale = ["d_model"]

    print(f"saving base shapes at {output_path}")

    base_shapes = get_shapes(load_mu_model(model_config))

    # just need to change whatever dimension(s) we are scaling
    # currently only scaling width, but may scale depth also
    # width scaling by d_model, but can also be done based on num_heads, etc.

    for dim in dims_to_scale:
        setattr(model_config, dim, getattr(model_config, dim) * 2)

    delta_shapes = get_shapes(load_mu_model(model_config))
    make_base_shapes(base_shapes, delta_shapes, savefile=output_path)
