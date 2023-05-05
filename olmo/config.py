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

from .aliases import PathOrStr
from .exceptions import OlmoConfigurationError

__all__ = [
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

    @property
    def device(self) -> Optional[str]:
        if self.init_device == "meta" or self.init_device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return self.init_device


class OptimizerType(StrEnum):
    adamw = "adamw"
    decoupled_adamw = "decoupled_adamw"
    decoupled_lionw = "decoupled_lionw"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.decoupled_lionw
    learning_rate: Optional[float] = None
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    def __post_init__(self):
        self.betas = tuple(self.betas)


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    constant_with_warmup = "constant_with_warmup"
    linear_decay_with_warmup = "linear_decay_with_warmup"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    t_warmup: str = "100ba"
    alpha_f: float = 0.1


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class DataConfig(BaseConfig):
    paths: List[str] = field(default_factory=lambda: [])
    pad_direction: PaddingDirection = PaddingDirection.right
    num_workers: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = True
    timeout: int = 0


@dataclass
class EvaluatorConfig(BaseConfig):
    label: str
    data: DataConfig
    device_eval_microbatch_size: int
    metric_names: List[str]
    subset_num_batches: int = -1


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
    tags: Optional[List[str]] = None
    log_artifacts: bool = False
    rank_zero_only: bool = True


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
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    seed: int = 6198
    dry_run: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    algorithms: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    data: DataConfig = field(default_factory=DataConfig)
    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    eval_interval: Union[int, str] = "1ep"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    save_folder: str = "./"
    save_interval: Union[str, int] = "1ep"
    save_num_checkpoints_to_keep: int = -1
    save_overwrite: bool = False
    load_path: Optional[str] = None
    load_weights_only: bool = False
    max_duration: Union[str, int] = "10ep"
    global_train_batch_size: int = 512
    device_train_batch_size: Union[str, int] = "auto"
    device_train_microbatch_size: Union[str, int] = "auto"
    device_train_grad_accum: Union[str, int] = "auto"
    device_eval_batch_size: Optional[int] = None
    n_gpus: Optional[int] = None
    precision: Optional[str] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    wandb: Optional[WandbConfig] = None
    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    console_log_interval: Union[str, int] = "1ba"
    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    @property
    def device(self) -> Optional[str]:
        return self.model.device
