from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import torch

from .aliases import PathOrStr
from .exceptions import DolmaConfigurationError

__all__ = [
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "DataConfig",
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
    def new(cls: Type[C], overrides: Optional[List[str]] = None) -> C:
        from omegaconf import OmegaConf
        from omegaconf.errors import ConfigKeyError

        conf = OmegaConf.structured(cls)
        if overrides:
            try:
                conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
            except ConfigKeyError as e:
                raise DolmaConfigurationError(str(e))
        return cast(C, OmegaConf.to_object(conf))

    @classmethod
    def load(cls: Type[C], path: PathOrStr, overrides: Optional[List[str]] = None) -> C:
        """Load from a YAML file."""
        from omegaconf import OmegaConf
        from omegaconf.errors import ConfigKeyError

        schema = OmegaConf.structured(cls)
        try:
            conf = OmegaConf.merge(schema, OmegaConf.load(str(path)))
            if overrides:
                conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
        except ConfigKeyError as e:
            raise DolmaConfigurationError(str(e))
        return cast(C, OmegaConf.to_object(conf))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        from omegaconf import OmegaConf

        OmegaConf.save(config=self, f=str(path))

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore


@dataclass
class ModelConfig(BaseConfig):
    """
    DOLMA (model) configuration.
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
    The ratio of the inner MLP dimensionality to `d_model`.
    """

    alibi: bool = False
    """
    If `True`, use ALiBi embeddings.
    """

    alibi_bias_max: float = 8.0
    """
    Maximum absolute value of ALiBi bias.
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
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

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
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

    @property
    def device(self) -> Optional[str]:
        if self.init_device == "meta" or self.init_device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return self.init_device


@dataclass
class OptimizerConfig(BaseConfig):
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self):
        self.betas = tuple(self.betas)


@dataclass
class SchedulerConfig(BaseConfig):
    name: str = "cosine_with_warmup"
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
    prefetch_factor: int = 2
    persistent_workers: bool = True
    timeout: int = 0


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
    init_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class TrainConfig(BaseConfig):
    """
    DOLMA training configuration.
    """

    run_name: Optional[str] = None
    seed: int = 6198
    dry_run: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    algorithms: Optional[Dict[str, Dict[str, Any]]] = None
    data: DataConfig = field(default_factory=DataConfig)
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

    @property
    def device(self) -> Optional[str]:
        return self.model.device
