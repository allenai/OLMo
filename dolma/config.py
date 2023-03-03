from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar, cast

import torch

from .aliases import PathOrStr
from .exceptions import DolmaConfigurationError

__all__ = ["Config", "TrainConfig"]


C = TypeVar("C", bound="BaseConfig")


class BaseConfig:
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


@dataclass
class Config(BaseConfig):
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
class TrainConfig(BaseConfig):
    """
    DOLMA training configuration.
    """

    model: Config
