from dataclasses import dataclass
from typing import Optional

__all__ = ["Config"]


@dataclass
class Config:
    """
    DOLMA configuration.

    Note that the defaults for these attributes are equivalent to the base GPT2 model.
    """

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

    device: Optional[str] = None
    """
    The torch device to use, e.g. "cpu" or "cuda:0".
    """
