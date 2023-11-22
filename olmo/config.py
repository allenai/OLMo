from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .aliases import PathOrStr
from .exceptions import OlmoConfigurationError
from .util import StrEnum

__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "CompilerConfig",
    "LayerNormType",
    "InitFnType",
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
    "FSDPPrecision",
    "FSDPWrapStrategy",
    "FSDPConfig",
    "CheckpointType",
]

C = TypeVar("C", bound="BaseConfig")


class BaseConfig:
    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            from .util import is_url

            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        # Finds the latest checkpoint in a folder.
        def path_last_checkpoint(path) -> str:
            from .util import find_latest_checkpoint

            latest_checkpoint = find_latest_checkpoint(path)
            if latest_checkpoint is None:
                if validate_paths:
                    raise FileNotFoundError(f"Could not find a latest checkpoint at {path}")
                else:
                    return ""
            else:
                return str(latest_checkpoint)

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)
        om.register_new_resolver("path.last_checkpoint", path_last_checkpoint, replace=True)

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
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        cls._register_resolvers(validate_paths=validate_paths)
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

    amd_compatible = "amd_compatible"
    """
    LayerNorm implemented manually to work around an issue with ROCm.
    """


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


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
    This is only used when ``mlp_hidden_size`` is not set.
    """

    mlp_hidden_size: Optional[int] = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap groups
    of blocks together with a single FSDP wrapper during training.
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

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
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

    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
    to ``False``.
    """

    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
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

    bias_for_layer_norm: Optional[bool] = None
    """
    Whether or not to include bias parameters in layer norm.
    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in
    layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    scale_logits: bool = False
    """
    If ``True``, scale the output logits by ``1 / sqrt(d_model)``.
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

    weight_tying: bool = True
    """
    Whether to tie output linear weights to the input embedding.
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

    init_fn: InitFnType = InitFnType.normal
    """
    The weight initialization strategy.
    """

    init_std: float = 0.02
    """
    The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal".
    """

    init_cutoff_factor: Optional[float] = None
    """
    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal". Setting this to None means values are not cutoff.
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """


class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.lionw
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)

    no_decay_norm_and_bias: Optional[bool] = None
    """
    Deprecated. Use ``decay_norm_and_bias`` and ``decay_embeddings`` instead.
    """

    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False
    metrics_log_interval: Optional[int] = None
    """
    The interval with which to collect and log detailed parameter-specific metrics.
    This only applies when logging to W&B, since these metrics won't be logged to the console.
    If not set, defaults to the wandb `log_interval`.
    """

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"
    max_scheduler = "max_scheduler"
    constant = "constant"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    t_warmup: int = 100
    t_max: Optional[int] = None
    alpha_f: float = 0.1

    grad_clip_warmup_steps: Optional[int] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class DataConfig(BaseConfig):
    paths: Optional[List[str]] = None
    datasets: Optional[Dict[str, List[str]]] = None
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


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each OLMo block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """


@dataclass
class FSDPConfig(BaseConfig):
    use_orig_params: bool = True
    """
    This must be ``True`` if using ``compile`` or you want to track the parameter norm during training.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    wrapping_strategy: Optional[FSDPWrapStrategy] = None
    """
    The wrapping strategy to use. If ``None``, the default, the model is wrapped with a single top-level
    FSDP instance.
    """

    precision: FSDPPrecision = FSDPPrecision.pure


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"


class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    fine_grained = "fine_grained"


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

    remote_save_folder: Optional[str] = None
    """
    A folder in a cloud bucket to upload saved checkpoints to.
    """

    canceled_check_interval: int = 50
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
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

    no_pre_train_checkpoint: bool = False
    """
    Skip saving pre-train checkpoint.
    """

    load_path: Optional[str] = None
    """
    The path to a training checkpoint to restore/resume from.

    Note that you can make use of the "path.last_checkpoint" Omegaconfig YAML resolver here, which takes
    a local or remote directory and resolves to the latest checkpoint (sharded or unsharded) in that directory.
    For example,

    ```bash
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/v1_5-mix-run-001}'
    ```
    """

    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    """
    The sharded checkpointer type to use to load the initial checkpoint from ``load_path``.
    """

    reset_optimizer_state: bool = False
    """
    When this is set, we restore the model from a checkpoint (if given), but we leave the optimizer uninitialized.
    We also set a new learning rate schedule that does a new warmup, such that it intercepts the original learning
    curve (according to the current learning rate schedule settings), and continues from there.
    """

    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy
    """
    The name of the sharded checkpointer to use to save (sharded) checkpoints throughout training.
    """

    new_style_checkpoints: Optional[bool] = None
    """
    Deprecated. Use ``sharded_checkpointer`` instead.
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
    Clip gradient norms to this value if set.
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
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

    early_stopping_factor: Optional[float] = None

    save_data_indices: bool = True
    """
    Save training data indices from each batch for each worker.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step.
    """

    activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None
    """
    The activation checkpointing strategy to use.
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

    @property
    def fsdp_precision(self) -> MixedPrecision:
        if self.fsdp.precision == FSDPPrecision.pure:
            return MixedPrecision(
                param_dtype=self.autocast_precision,
                reduce_dtype=self.autocast_precision,
                buffer_dtype=self.autocast_precision,
            )
        elif self.fsdp.precision == FSDPPrecision.mixed:
            return MixedPrecision(
                param_dtype=self.autocast_precision,
                reduce_dtype=torch.float32,
                buffer_dtype=self.autocast_precision,
            )
        else:
            raise NotImplementedError(f"{self.fsdp.precision}")
