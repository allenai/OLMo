import argparse
import logging
import os
import re
from copy import deepcopy
from typing import Set, Tuple, Union

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import ShardingStrategy

from olmo import (
    ActivationType,
    DDPConfig,
    InitFnType,
    LayerNormType,
    ModelConfig,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TokenizerConfig,
    TrainConfig,
    WandbConfig,
)
from olmo.config import (
    DataConfig,
    DistributedStrategy,
    EvaluatorConfig,
    EvaluatorType,
    FSDPConfig,
    FSDPPrecision,
    FSDPWrapStrategy,
    InstanceFilterConfig,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
)
from olmo.data import named_data_mixes
from olmo.model import OLMo
from olmo.util import (
    add_cached_path_clients,
    find_latest_checkpoint,
    flatten_dict,
    prepare_cli_environment,
)

log = logging.getLogger("train")

MODEL_CONFIG_150M = ModelConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    mlp_ratio=8,
    weight_tying=False,
    alibi=False,
    rope=True,
    flash_attention=True,
    attention_dropout=0.0,
    attention_layer_norm=False,
    include_bias=False,
    layer_norm_type=LayerNormType.rms,
    layer_norm_with_affine=True,
    layer_norm_eps=1e-6,
    bias_for_layer_norm=False,
    attention_layer_norm_with_affine=False,
    activation_type=ActivationType.swiglu,
    residual_dropout=0.0,
    embedding_dropout=0.0,
    max_sequence_length=1024,
    vocab_size=50280,
    embedding_size=50304,
    eos_token_id=50279,
    pad_token_id=1,
    init_device="cpu",
    init_fn=InitFnType.normal,
    init_std=0.02,
    init_cutoff_factor=3,
)

MODEL_CONFIGS = {
    "150M": MODEL_CONFIG_150M,
    "300M": MODEL_CONFIG_150M.update_with(d_model=1024, n_heads=16, n_layers=16, mlp_ratio=8),
    "530M": MODEL_CONFIG_150M.update_with(d_model=1344, n_heads=16, n_layers=16, mlp_ratio=8),
    "750M": MODEL_CONFIG_150M.update_with(d_model=1536, n_heads=16, n_layers=16, mlp_ratio=8),
    "1B": MODEL_CONFIG_150M.update_with(d_model=2048, n_heads=16, n_layers=16, mlp_ratio=8),
    "7B": MODEL_CONFIG_150M.update_with(
        d_model=4096, n_heads=32, n_layers=32, mlp_ratio=0, mlp_hidden_size=22016, init_device="meta"
    ),
}

if __name__ == "__main__":

    base_config = MODEL_CONFIGS["300M"]
    for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        model_config = base_config.update_with(max_sequence_length=seq_len)
        model = OLMo(model_config)
        print(f"Number of forward FLOPs for seq_len={seq_len}: {model.num_fwd_flops}")
        print(f"Number of backward FLOPs for seq_len={seq_len}: {model.num_bck_flops}")