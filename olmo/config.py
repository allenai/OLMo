from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import ShardingStrategy

from .aliases import PathOrStr
from .exceptions import OlmoConfigurationError

__all__ = [
    "LogFilterType",
    "ActivationType",
    "BlockType",
    "CompilerConfig",
    "LayerNormType",
    "ModelConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "DataConfig",
    "EvaluatorConfig",
    "TokenizerConfig",
    "TrainConfig",
    "PaddingDirection",
    "TruncationDirection",
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "FSDPConfig",
    "CheckpointType",
]


C = TypeVar("C", bound="BaseConfig")


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class BaseConfig:
    @classmethod
    def _register_resolvers(cls):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = glob(path)
                if not matches:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            for path in paths:
                if Path(path).exists():
                    return path
            raise FileNotFoundError(", ".join(paths))

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OlmoConfigurationError(str(e))

    @classmethod
    def load(cls: Type[C], path: PathOrStr, overrides: Optional[List[str]] = None, key: Optional[str] = None) -> C:
        """Load from a YAML file."""
        cls._register_resolvers()
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OlmoConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out


class LogFilterType(StrEnum):
    rank0_only = "rank0_only"
    local_rank0_only = "local_rank0_only"


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """

    low_precision_rms = "low_precision_rms"
    """
    A low-precision version of RMSNorm.
    """


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"


@dataclass
class ModelConfig(BaseConfig):
    """
    OLMo (model) configuration.
    """

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    alibi: bool = False
    """
    If ``True``, use ALiBi embeddings. Mutually exclusive with ``rope``.
    """

    alibi_bias_max: float = 8.0
    """
    Maximum absolute value of ALiBi bias.
    """

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    flash_attention: bool = False
    """
    If ``True``, use ``FlashAttention``.
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    multi_query_attention: bool = False
    """
    Use the Multi-Query formulation of attention used in PaLM. This reduces the number of parameters
    and is more efficient during inference.
    """

    attention_layer_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    embedding_dropout: float = 0.1
    """
    The dropout probability for embeddings.
    """

    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    In PaLM, they got rid of all bias terms because they found that large
    models tend to have near 0 bias terms anyway.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    eos_token_id: int = 50256
    """
    The ID of the end-of-sentence special token.
    """

    pad_token_id: int = 50256
    """
    The ID of the token to use for padding. Defaults to the ID of the EOS token.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
    """

    init_std: float = 0.02
    """
    Standard deviation used when initializing parameters.
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """


class OptimizerType(StrEnum):
    lionw = "lionw"
    adam = "adam"
    adamw = "adamw"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.lionw
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    no_decay_norm_and_bias: bool = True
    """Do not apply weight decay to norms and biases."""

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    t_warmup: int = 100
    t_max: Optional[int] = None
    alpha_f: float = 0.1


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class DataConfig(BaseConfig):
    paths: List[str] = field(default_factory=lambda: [])
    pad_direction: PaddingDirection = PaddingDirection.right
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0


class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"


@dataclass
class EvaluatorConfig(BaseConfig):
    label: str
    type: EvaluatorType = EvaluatorType.lm
    data: DataConfig = field(default_factory=DataConfig)
    device_eval_batch_size: Optional[int] = None
    subset_num_batches: Optional[int] = None


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "gpt2"
    truncate_direction: TruncationDirection = TruncationDirection.right


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = "ai2-llm"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = None
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    backend: str = "inductor"
    """
    The backend to use.
    """


@dataclass
class FSDPConfig(BaseConfig):
    use_orig_params: bool = True
    """
    This must be ``True`` if using ``compile``.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"


@dataclass
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    The name of the run.
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """
    OLMo Model configuration.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When restarting, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When restarting, use this to fast-forward the dataloader beyond the last checkpoint.
    This can be useful when restarting due to a loss spike in order to skip the data that
    corresponded to the spike.
    """

    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 1000
    """
    How often (in terms of batches) to run evaluations.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    """
    Tokenizer configuration.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    save_interval: int = 1000
    """
    How often (in terms of batches) to save training state checkpoints that can be used for restarts.
    """

    save_interval_unsharded: Optional[int] = None
    """
    How often (if at all) to save the unsharded state to a single file.
    For large models it can be costly to save these, so it usually makes sense to save
    these less often than regular (sharded) training checkpoints.
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many checkpoints to keep.
    """

    save_num_unsharded_checkpoints_to_keep: int = -1
    """
    How many unsharded checkpoints to keep.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite any conflicting checkpoint files.
    """

    force_save_unsharded: bool = False
    """
    Save an unsharded checkpoint before training (even during a dry run).
    Use this option with `--load-path={PATH}` and `--dry_run` to convert a sharded
    checkpoint into an unsharded checkpoint.
    """

    load_path: Optional[str] = None
    """
    The path to a (sharded) training checkpoint to restore/resume from.
    """

    max_duration: int = 10000
    """
    Maximum number of batches to train for.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_batch_size: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``global_train_batch_size // world_size``.
    """

    device_train_microbatch_size: int = 16
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    device_eval_batch_size: int = 16
    """
    The number of evaluation instances passed to the model in a single forward pass on each device.
    """

    eval_subset_num_batches: int = -1
    """
    The number of batches to use for downstream evaluation from each dataset.
    """

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    device_train_grad_accum: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``device_train_batch_size // device_train_microbatch_size``.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradients to this value if set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    activation_checkpointing: bool = False
    """
    Use activation checkpointing on transformer blocks.
    """

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0.
    """

    time_limit: Optional[float] = 60 * 60 * 47.5
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    On LUMI we have 48 hours max per job, so we default to just under 48 hours to give us time
    to write out a final checkpoint.
    """

    save_data_indices: bool = False
    """
    If ``True``, write the indices of the examples in each batch for each rank to a tsv file in the save folder.
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")
