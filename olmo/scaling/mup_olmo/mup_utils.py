import logging
from typing import List, Optional, Union

from mup import apply_infshapes, get_shapes, make_base_shapes, zip_infshapes
import torch

from olmo.config import DistributedStrategy, FSDPWrapStrategy, ModelConfig
from olmo.model import OLMo
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP
from olmo.torch_util import get_default_device, get_local_rank


log = logging.getLogger(__name__)


def load_model(model_cfg: ModelConfig, distributed_strategy: Optional[DistributedStrategy] = None):
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = get_default_device()

    if distributed_strategy == DistributedStrategy.fsdp:
        model_cfg.init_device = "meta"
    else:
        model_cfg.init_device = "cuda" if torch.cuda.is_available() else "cpu"

    olmo_model = OLMo(model_cfg, init_params=False)

    infshapes = None
    if model_cfg.use_mup:
        infshapes = zip_infshapes(model_cfg.mup_base_shapes, olmo_model)
        # olmo_model.set_base_shapes()

    # for name, p in olmo_model.named_parameters():
    #     if not hasattr(p, "infshape"):
    #         log.info("DEBUG: unwrapped model. name %s missing infshapes", name)

    if distributed_strategy == DistributedStrategy.ddp:
        log.info("Wrapping model with DDP...")

        if not torch.cuda.is_available():
            raise RuntimeError("DDP cannot run without `cuda`.")

        model_cfg.init_device = "cuda"
        # move to cuda before calling ddp
        dist_model = DDP(olmo_model.to(device))
    elif distributed_strategy == DistributedStrategy.fsdp:
        # Wrap the model in FSDP.
        log.info("Wrapping model with FSDP...")

        if not torch.cuda.is_available():
            raise RuntimeError("FSDP cannot run without `cuda`.")

        wrap_policy = olmo_model.get_fsdp_wrap_policy(FSDPWrapStrategy.by_block)

        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=device)

        param_init_fn = dummy_init_fn

        # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
        dist_model = FSDP(
            olmo_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            auto_wrap_policy=wrap_policy,
            use_orig_params=True,  # needed for compile, mup and some of our optimizer/parameter metrics
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
        )
    else:
        if distributed_strategy is not None:
            raise NotImplementedError(distributed_strategy)

        dist_model = olmo_model

    # for name, p in dist_model.named_parameters():
    #     if not hasattr(p, "infshape"):
    #         log.info("DEBUG: wrapped model. name %s missing infshapes", name)
    # log.info("DEBUG: wrapped model. name %s, has_infshape %s", "transformer.ff_out (mureadout)", hasattr(dist_model.transformer.ff_out.weight, "infshape"))

    if infshapes is not None:
        apply_infshapes(dist_model, infshapes)
        olmo_model.reset_parameters()

    # for name, p in dist_model.named_parameters():
    #     if not hasattr(p, "infshape"):
    #         log.info("DEBUG: wrapped model. name %s missing infshapes", name)
    # log.info("DEBUG: wrapped model. name %s, has_infshape %s", "transformer.ff_out (mureadout)", hasattr(dist_model.transformer.ff_out.weight, "infshape"))

    log.info("Model:")
    log.info(dist_model)

    return dist_model


def load_mu_model(config: ModelConfig, distributed_strategy: Optional[DistributedStrategy] = None):
    config.use_mup = True
    return load_model(config, distributed_strategy=distributed_strategy)


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
