"""
STU (Spectral Transform Unit) implementation for OLMo.
Adapted from the Flash-STU architecture.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = [
    "STU",
    "OLMoSTUBlock",
    "get_spectral_filters",
    "get_hankel",
    "nearest_power_of_two",
]


def nearest_power_of_two(n: int, round_up: bool = True) -> int:
    """Find the nearest power of two to n."""
    if n <= 0:
        return 1
    log2_n = math.log2(n)
    if round_up:
        return 2 ** math.ceil(log2_n)
    else:
        return 2 ** math.floor(log2_n)


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    """
    Generate the Hankel matrix for spectral decomposition.
    
    Args:
        seq_len: Sequence length
        use_hankel_L: Whether to use the L variant of the Hankel matrix
    
    Returns:
        Hankel matrix as numpy array
    """
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)

    return Z


def get_spectral_filters(
    seq_len: int,
    num_eigh: int,
    use_hankel_L: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Compute spectral filters via eigendecomposition of the Hankel matrix.
    
    Args:
        seq_len: Sequence length
        num_eigh: Number of eigenvectors to keep (K)
        use_hankel_L: Whether to use the L variant
        device: Target device
        dtype: Target dtype
    
    Returns:
        Spectral filters tensor of shape (seq_len, num_eigh)
    """
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = np.linalg.eigh(Z)
    # Take the top K eigenvalues/vectors
    sigma, phi = sigma[-num_eigh:], phi[:, -num_eigh:]
    # Scale by fourth root of eigenvalues
    phi *= sigma ** 0.25
    return torch.tensor(phi, device=device, dtype=dtype)


def convolve(
    u: torch.Tensor,
    v: torch.Tensor,
    n: int,
    use_approx: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convolve input u with filters v using FFT.
    
    Args:
        u: Input tensor of shape (batch_size, seq_len, d_in) or (batch_size, seq_len, K, d_in)
        v: Filter tensor
        n: FFT size (padded sequence length)
        use_approx: Whether to use the approximation variant
    
    Returns:
        Tuple of (U_plus, U_minus) convolved tensors
    """
    bsz, seq_len, d_in = u.shape

    # Create sign pattern for alternating convolution
    sgn = torch.full((1, seq_len, 1), 1, device=u.device, dtype=u.dtype)
    sgn[:, 1::2] *= -1

    if use_approx:
        _, d_out = v.shape
        v = v.reshape(1, -1, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.reshape(1, -1, K, 1, 1).to(torch.float32)
        u = u.reshape(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    # FFT-based convolution
    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus.to(u.dtype), U_minus.to(u.dtype)


class STU(nn.Module):
    """
    Spectral Transform Unit (STU) module.
    
    This module performs spectral convolution using learned projections
    and eigendecomposed Hankel matrices.
    """

    def __init__(
        self,
        config: ModelConfig,
        phi: torch.Tensor,
        n: int,
        feature_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            config: Model configuration
            phi: Spectral filters (seq_len, num_eigh)
            n: Padded sequence length for FFT
            feature_dim: Optional override for the feature dimension handled by
                the STU (defaults to ``config.d_model``)
        """
        super().__init__()
        self.config = config
        # Register phi as a buffer so it moves with the model to different devices
        self.register_buffer('phi', phi, persistent=False)
        self.n = n
        self.K = config.stu_num_eigh
        self.d_model = feature_dim or config.d_model
        self.d_in = self.d_model
        self.d_out = self.d_model
        self.use_hankel_L = config.stu_use_hankel_L
        self.use_approx = config.stu_use_approx

        if self.use_approx:
            # Approximation: project inputs and filters separately
            self.M_inputs = nn.Parameter(
                torch.empty(self.d_in, self.d_out, device=config.init_device)
            )
            self.M_filters = nn.Parameter(
                torch.empty(self.K, self.d_in, device=config.init_device)
            )
        else:
            # Full version: project after convolution
            self.M_phi_plus = nn.Parameter(
                torch.empty(self.K, self.d_in, self.d_out, device=config.init_device)
            )
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.empty(self.K, self.d_in, self.d_out, device=config.init_device)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of STU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.use_approx:
            # Contract inputs and filters over K and d_in dimensions, then convolve
            x_proj = x @ self.M_inputs
            phi_proj = self.phi @ self.M_filters
            spectral_plus, spectral_minus = convolve(
                x_proj, phi_proj, self.n, self.use_approx
            )
        else:
            # Convolve first, then contract over K and d_in dimensions
            U_plus, U_minus = convolve(x, self.phi, self.n, self.use_approx)
            spectral_plus = torch.tensordot(
                U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
            )
            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(
                    U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
                )

        return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

    def reset_parameters(self):
        """Initialize STU parameters."""
        if self.use_approx:
            nn.init.xavier_normal_(self.M_inputs)
            nn.init.xavier_normal_(self.M_filters)
        else:
            nn.init.xavier_normal_(self.M_phi_plus)
            if not self.use_hankel_L:
                nn.init.xavier_normal_(self.M_phi_minus)


class OLMoSTUBlock(nn.Module):
    """
    OLMo-style block using STU instead of attention.
    
    This block follows the same structure as OLMoSequentialBlock:
    x -> LayerNorm -> STU -> Residual
    x -> LayerNorm -> MLP -> Residual
    """

    def __init__(
        self,
        layer_id: int,
        config: ModelConfig,
        phi: torch.Tensor,
        n: int,
    ):
        """
        Args:
            layer_id: Layer index
            config: Model configuration
            phi: Spectral filters
            n: Padded sequence length
        """
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )

        # Import here to avoid circular imports
        from .model import Activation, Dropout, LayerNormBase
        from .usp import UniversalSequencePreconditioner

        # Dropout
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms
        self.stu_norm = LayerNormBase.build(config, size=config.d_model)
        self.ff_norm = LayerNormBase.build(config, size=config.d_model)

        # STU module
        self.stu_mlp_enabled = config.stu_enable_mlp_sandwich

        # USP (Universal Sequence Preconditioning) module
        self.usp_enabled = config.usp_enable and self.stu_mlp_enabled
        if self.usp_enabled:
            self.usp = UniversalSequencePreconditioner(
                config=config,
                degree=config.usp_degree,
                lambda_init=config.usp_lambda_init,
                alpha_init=config.usp_alpha_init,
                apply_layer_norm=config.usp_apply_layer_norm,
                learnable_params=config.usp_learnable_params,
            )

        if self.stu_mlp_enabled:
            self.stu_mlp_hidden_size = (
                config.stu_mlp_hidden_size
                if config.stu_mlp_hidden_size is not None
                else (
                    config.mlp_hidden_size
                    if config.mlp_hidden_size is not None
                    else config.mlp_ratio * config.d_model
                )
            )
            self.stu_mlp_act = Activation.build(config)
            assert (self.stu_mlp_act.output_multiplier * self.stu_mlp_hidden_size) % 1 == 0
            self.stu_inner_dim = int(self.stu_mlp_act.output_multiplier * self.stu_mlp_hidden_size)

            self.stu_mlp_in_proj = nn.Linear(
                config.d_model,
                self.stu_mlp_hidden_size,
                bias=config.include_bias,
                device=config.init_device,
            )
            self.stu_mlp_out_proj = nn.Linear(
                self.stu_inner_dim,
                config.d_model,
                bias=config.include_bias,
                device=config.init_device,
            )
            self.stu_mlp_out_proj._is_residual = True  # type: ignore
            stu_feature_dim = self.stu_inner_dim
        else:
            self.stu_mlp_hidden_size = None
            self.stu_mlp_act = None
            self.stu_inner_dim = config.d_model
            self.stu_mlp_in_proj = nn.Identity()
            self.stu_mlp_out_proj = nn.Identity()
            stu_feature_dim = None

        self.stu = STU(config, phi, n, feature_dim=stu_feature_dim)

        # MLP (feed-forward)
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        self._activation_checkpoint_fn: Optional[callable] = None

    def reset_parameters(self):
        """Initialize block parameters."""
        from .initialization import init_normal

        self.stu_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        self.stu.reset_parameters()

        # Initialize USP if enabled
        if self.usp_enabled:
            # USP initialization is handled in the USP module itself
            pass

        # Initialize projections based on config
        if self.config.init_fn == "normal":
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == "mitchell":
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == "full_megatron":
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor

        init_normal(self.ff_proj, std, cutoff_factor)

        if self.stu_mlp_enabled:
            init_normal(self.stu_mlp_in_proj, std, cutoff_factor)

        # Output projection with layer-dependent std
        if self.config.init_fn == "mitchell":
            ff_out_std = 1 / math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1))
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == "full_megatron":
            ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            ff_out_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor

        init_normal(self.ff_out, ff_out_std, cutoff_factor)

        if self.stu_mlp_enabled:
            if self.config.init_fn == "mitchell":
                stu_out_std = 1 / math.sqrt(2 * self.stu_mlp_out_proj.in_features * (self.layer_id + 1))
                stu_cutoff = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == "full_megatron":
                stu_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
                stu_cutoff = self.config.init_cutoff_factor or 3.0
            else:
                stu_out_std = self.config.init_std
                stu_cutoff = self.config.init_cutoff_factor
            init_normal(self.stu_mlp_out_proj, stu_out_std, stu_cutoff)

    def set_activation_checkpointing(self, strategy, checkpoint_func=None):
        """Set activation checkpointing for this block."""
        from .config import ActivationCheckpointingStrategy
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            from .model import activation_checkpoint_function
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

        # Propagate activation checkpointing to USP module
        if self.usp_enabled:
            self.usp.set_activation_checkpointing(strategy, checkpoint_func)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass matching OLMoBlock interface.
        
        Note: STU blocks don't use attention_bias, layer_past, or cache,
        but we keep the signature for compatibility.
        """
        # STU path with residual connection
        if not self.config.norm_after:
            if self.usp_enabled:
                # USP handles layer normalization internally if configured
                if self._activation_checkpoint_fn is not None:
                    h = self._activation_checkpoint_fn(self.usp, x)
                else:
                    h = self.usp(x)
            else:
                # Standard layer normalization
                if self._activation_checkpoint_fn is not None:
                    h = self._activation_checkpoint_fn(self.stu_norm, x)
                else:
                    h = self.stu_norm(x)
        else:
            # For norm_after=True, we still apply USP if enabled, but without layer norm
            if self.usp_enabled:
                # Configure USP to not apply layer norm in this case
                if self._activation_checkpoint_fn is not None:
                    h = self._activation_checkpoint_fn(self.usp.filter, x)
                else:
                    h = self.usp.filter(x)
            else:
                h = x

        # Apply STU sandwich architecture
        stu_input = h
        if self.stu_mlp_enabled:
            # MLP input projection: d_model → stu_mlp_hidden_size
            if self._activation_checkpoint_fn is not None:
                stu_input = self._activation_checkpoint_fn(self.stu_mlp_in_proj, stu_input)
            else:
                stu_input = self.stu_mlp_in_proj(stu_input)

            # Nonlinearity (SwiGLU activation)
            if self._activation_checkpoint_fn is not None:
                stu_input = self._activation_checkpoint_fn(self.stu_mlp_act, stu_input)
            else:
                stu_input = self.stu_mlp_act(stu_input)

        if self._activation_checkpoint_fn is not None:
            stu_out = self._activation_checkpoint_fn(self.stu, stu_input)
        else:
            stu_out = self.stu(stu_input)

        if self.stu_mlp_enabled:
            if self._activation_checkpoint_fn is not None:
                stu_out = self._activation_checkpoint_fn(self.stu_mlp_out_proj, stu_out)
            else:
                stu_out = self.stu_mlp_out_proj(stu_out)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                stu_out = self._activation_checkpoint_fn(self.stu_norm, stu_out)
            else:
                stu_out = self.stu_norm(stu_out)

        x = x + self.dropout(stu_out)

        # Feed-forward path with residual connection
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)

        x = self.ff_out(x)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)
            else:
                x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        # Return None for cache to match OLMoBlock interface
        return x, None

    def reset_streaming_state(self):
        """Reset streaming state for inference (USP ring buffers)."""
        if self.usp_enabled:
            self.usp.reset_streaming_state()

    def get_usp_analysis(self, x: torch.Tensor) -> Optional[dict]:
        """Get USP analysis for debugging and monitoring."""
        if not self.usp_enabled:
            return None

        # Apply USP with analysis enabled
        _, analysis = self.usp(x, return_analysis=True)
        return analysis

