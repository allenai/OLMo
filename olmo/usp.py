"""
Universal Sequence Preconditioning (USP) for STU Sandwich Architecture.

This module implements causal polynomial preconditioning using monic Chebyshev polynomials
for spectral regularization of temporal sequences in the STU sandwich layers.

The key insight is that by applying a carefully designed causal filter before the nonlinearity,
we can suppress low-frequency modes that cause optimization instability while preserving
high-frequency dynamics essential for sequence modeling.

Mathematical Foundation:
- USP operator: G = I - λα Σ(i=1 to n) c_i S^i where S is the causal shift operator
- Chebyshev design: coefficients c_i from monic Chebyshev polynomial T̂_n(x) = 2^(1-n) T_n(x)
- Frequency response: G(ω) = 1 - λα C(e^(-iω)) with uniform spectral shrinkage
- State-space view: equivalent to left-multiplying state matrix A by q_n(A) = z^n p_n(z^(-1))
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = [
    "generate_monic_chebyshev_coefficients",
    "UniversalSequencePreconditioner",
    "CausalPolynomialFilter",
    "compute_operator_norm_bound",
    "analyze_frequency_response",
]


def generate_monic_chebyshev_coefficients(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate coefficients for the monic Chebyshev polynomial T̂_n(x) = 2^(1-n) T_n(x).

    The monic Chebyshev polynomial has the extremal property:
    max_{x ∈ [-1,1]} |T̂_n(x)| = 2^(1-n)

    which is the minimal possible uniform deviation among all degree-n monic polynomials.
    This ensures optimal uniform spectral shrinkage across the normalized frequency band.

    Args:
        n: Degree of the Chebyshev polynomial (n >= 1)
        dtype: Target data type for coefficients

    Returns:
        Tensor of shape (n+1,) containing coefficients [c_0, c_1, ..., c_n]
        where c_n = 1 (monic property) and p_n(x) = Σ c_i x^i

    Mathematical details:
    - T_0(x) = 1, T_1(x) = x, T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    - T̂_n(x) = 2^(1-n) T_n(x) ensures ||T̂_n||_{[-1,1]} = 2^(1-n)
    - Coefficients computed via recurrence relation for numerical stability
    """
    if n < 1:
        raise ValueError(f"Degree n must be >= 1, got {n}")

    # Initialize coefficient arrays for Chebyshev polynomials T_k(x)
    # We'll store coefficients for T_0, T_1, ..., T_n
    max_degree = n + 1
    coeffs = torch.zeros(max_degree, max_degree, dtype=dtype)

    # Base cases: T_0(x) = 1, T_1(x) = x
    coeffs[0, 0] = 1.0  # T_0(x) = 1
    if n >= 1:
        coeffs[1, 1] = 1.0  # T_1(x) = x

    # Recurrence: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    for k in range(2, n + 1):
        # 2x T_{k-1}(x): shift coefficients right and multiply by 2
        coeffs[k, 1:k+1] = 2.0 * coeffs[k-1, :k]
        # Subtract T_{k-2}(x)
        coeffs[k, :k-1] -= coeffs[k-2, :k-1]

    # Extract T_n coefficients and make monic: T̂_n(x) = 2^(1-n) T_n(x)
    t_n_coeffs = coeffs[n, :n+1].clone()
    monic_factor = 2.0 ** (1 - n) if n > 0 else 1.0
    monic_coeffs = monic_factor * t_n_coeffs

    # Ensure c_n = 1 (monic property) - this should be automatically satisfied
    # but we verify for numerical stability
    if abs(monic_coeffs[-1] - 1.0) > 1e-10:
        # Normalize to ensure monic property
        monic_coeffs = monic_coeffs / monic_coeffs[-1]

    # Verify coefficient magnitude bounds from Lemma 3.2: max |c_k| ≤ 2^(0.3n)
    max_coeff_magnitude = torch.max(torch.abs(monic_coeffs)).item()
    theoretical_bound = 2.0 ** (0.3 * n)
    if max_coeff_magnitude > theoretical_bound * 1.1:  # Allow 10% tolerance for numerical errors
        raise RuntimeError(f"Coefficient magnitude {max_coeff_magnitude:.4f} exceeds theoretical bound {theoretical_bound:.4f}")

    return monic_coeffs


def compute_operator_norm_bound(coeffs: torch.Tensor, lambda_val: float, alpha_val: float) -> float:
    """
    Compute the ℓ_∞ → ℓ_∞ operator norm bound for the USP operator.

    For the causal convolution G = I - λα Σ c_i S^i, we have:
    ||G||_{∞→∞} ≤ 1 + |λα| ||c||_1

    Args:
        coeffs: Polynomial coefficients [c_0, c_1, ..., c_n]
        lambda_val: Residual gate parameter λ
        alpha_val: Preconditioner strength α

    Returns:
        Upper bound on operator norm
    """
    c_norm_1 = torch.sum(torch.abs(coeffs[1:])).item()  # Skip c_0 = 1 since it's the identity part
    return 1.0 + abs(lambda_val * alpha_val) * c_norm_1


def analyze_frequency_response(
    coeffs: torch.Tensor,
    lambda_val: float,
    alpha_val: float,
    omega_points: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analyze the frequency response G(ω) = 1 - λα C(e^(-iω)) of the USP operator.

    Args:
        coeffs: Polynomial coefficients [c_0, c_1, ..., c_n]
        lambda_val: Residual gate parameter λ
        alpha_val: Preconditioner strength α
        omega_points: Number of frequency points to evaluate

    Returns:
        Tuple of (frequencies, response_magnitude) both of shape (omega_points,)
    """
    # Frequency grid [0, π]
    omega = torch.linspace(0, math.pi, omega_points, dtype=coeffs.dtype)

    # Compute C(e^(-iω)) = Σ(k=1 to n) c_k e^(-ikω)
    n = len(coeffs) - 1
    c_response = torch.zeros_like(omega, dtype=torch.complex64)

    for k in range(1, n + 1):  # Skip c_0 since it doesn't appear in C(z)
        c_response += coeffs[k] * torch.exp(-1j * k * omega)

    # Frequency response G(ω) = 1 - λα C(e^(-iω))
    g_response = 1.0 - lambda_val * alpha_val * c_response
    g_magnitude = torch.abs(g_response)

    return omega, g_magnitude


def validate_eigenvalue_bounds(
    eigenvalues: torch.Tensor,
    T: int,
    d_out: int,
    tolerance: float = 1.1
) -> Tuple[bool, float, float]:
    """
    Validate that eigenvalues satisfy the paper's complex bounds requirement.

    From Theorem 2.1 and 2.2: max_j |arg(λ_j)| ≤ 1/(32 log²(2T³/d_out))²

    This is critical for achieving dimension-independent regret bounds.

    Args:
        eigenvalues: Complex eigenvalues tensor
        T: Sequence length (horizon)
        d_out: Output dimension
        tolerance: Tolerance factor for the bound (default 1.1 = 10% slack)

    Returns:
        Tuple of (is_valid, max_arg, required_bound)
    """
    # Compute the required bound from the paper
    log_term = math.log(2 * T**3 / d_out)
    required_bound = 1.0 / (32 * log_term**2)**2

    # Compute the maximum argument (imaginary part angle)
    eigenvalue_args = torch.angle(eigenvalues)
    max_arg = torch.max(torch.abs(eigenvalue_args)).item()

    # Check if bounds are satisfied
    is_valid = max_arg <= tolerance * required_bound

    return is_valid, max_arg, required_bound


def compute_required_eigenvalue_bound(T: int, d_out: int) -> float:
    """
    Compute the required eigenvalue argument bound from the paper.

    From Theorems 2.1 and 2.2: max_j |arg(λ_j)| ≤ 1/(32 log²(2T³/d_out))²

    Args:
        T: Sequence length (horizon)
        d_out: Output dimension

    Returns:
        Maximum allowed |arg(λ_j)| value
    """
    if T <= 0 or d_out <= 0:
        raise ValueError(f"T and d_out must be positive, got T={T}, d_out={d_out}")

    log_term = math.log(2 * T**3 / d_out)
    if log_term <= 0:
        raise ValueError(f"Invalid parameters: log(2T³/d_out) = {log_term} ≤ 0")

    required_bound = 1.0 / (32 * log_term**2)**2
    return required_bound


class CausalPolynomialFilter(nn.Module):
    """
    Causal polynomial filter implementing G = I - λα Σ c_i S^i.

    This module performs depthwise causal convolution with a fixed polynomial kernel
    derived from monic Chebyshev polynomials. The filter acts independently on each
    channel, providing spectral preconditioning without inter-channel coupling.

    Key properties:
    - Strictly causal: (Sx)_t := x_{t-1} with (Sx)_1 := 0
    - Parameter-efficient: only 2 learnable scalars (λ, α) regardless of model width
    - Streaming-compatible: O(nd) memory and computation per token
    - Frequency-selective: suppresses low-frequency modes while preserving others
    """

    def __init__(
        self,
        degree: int,
        d_model: int,
        lambda_init: float = 0.5,
        alpha_init: float = 0.1,
        learnable_params: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize causal polynomial filter.

        Args:
            degree: Polynomial degree n (typically 2-4 for good balance)
            d_model: Model dimension (number of channels)
            lambda_init: Initial value for residual gate λ ∈ [0,1]
            alpha_init: Initial value for preconditioner strength α
            learnable_params: Whether λ and α are learnable parameters
            device: Target device
            dtype: Data type for computations
        """
        super().__init__()

        # Validate inputs
        if degree < 1:
            raise ValueError(f"Degree must be >= 1, got {degree}")
        if degree > 10:
            raise ValueError(f"Degree {degree} is too large, maximum recommended is 10")
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if not (0.0 <= lambda_init <= 1.0):
            raise ValueError(f"lambda_init must be in [0,1], got {lambda_init}")
        if not (-1.0 <= alpha_init <= 1.0):
            raise ValueError(f"alpha_init should be in [-1,1] for stability, got {alpha_init}")

        self.degree = degree
        self.d_model = d_model

        # Generate monic Chebyshev coefficients (fixed, not learned)
        coeffs = generate_monic_chebyshev_coefficients(degree, dtype=dtype)
        self.register_buffer('coeffs', coeffs.to(device=device))

        # Learnable gating parameters
        if learnable_params:
            self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=dtype, device=device))
            self.alpha_param = nn.Parameter(torch.tensor(alpha_init, dtype=dtype, device=device))
        else:
            self.register_buffer('lambda_param', torch.tensor(lambda_init, dtype=dtype, device=device))
            self.register_buffer('alpha_param', torch.tensor(alpha_init, dtype=dtype, device=device))

        # Ring buffer for streaming inference
        self.register_buffer('ring_buffer', torch.zeros(degree, d_model, dtype=dtype, device=device))
        self.register_buffer('buffer_idx', torch.tensor(0, dtype=torch.long, device=device))

    def forward(self, x: torch.Tensor, use_streaming: bool = False) -> torch.Tensor:
        """
        Apply causal polynomial filter to input sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            use_streaming: Whether to use streaming mode (for inference)

        Returns:
            Filtered tensor of same shape: G(x) = x - λα Σ c_i S^i x
        """
        # Validate input
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, d_model), got {x.dim()}D")

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(f"Input d_model {d_model} doesn't match filter d_model {self.d_model}")

        if seq_len < 1:
            raise ValueError(f"Sequence length must be >= 1, got {seq_len}")

        # Check device compatibility
        if x.device != self.coeffs.device:
            raise ValueError(f"Input device {x.device} doesn't match filter device {self.coeffs.device}")

        # Check for NaN/Inf in input
        if not torch.all(torch.isfinite(x)):
            raise ValueError("Input contains NaN or Inf values")

        if use_streaming:
            return self._forward_streaming(x)
        else:
            return self._forward_parallel(x)

    def _forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel (training) implementation using causal convolution."""
        batch_size, seq_len, d_model = x.shape

        # Construct causal convolution kernel: [1, -λα c_1, -λα c_2, ..., -λα c_n]
        lambda_alpha = self.lambda_param * self.alpha_param
        kernel = torch.zeros(self.degree + 1, dtype=x.dtype, device=x.device)
        kernel[0] = 1.0  # Identity component
        kernel[1:] = -lambda_alpha * self.coeffs[1:]  # Polynomial components

        # Apply causal convolution using F.conv1d
        # Input shape: (batch_size * d_model, 1, seq_len)
        # Kernel shape: (1, 1, degree + 1)
        x_reshaped = x.transpose(1, 2).contiguous()  # (B, D, T)
        x_flat = x_reshaped.view(batch_size * d_model, 1, seq_len)

        kernel_conv = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, degree + 1)

        # Pad input for causality
        padding = (self.degree, 0)  # Left padding only
        x_padded = F.pad(x_flat, padding)

        # Apply convolution
        y_flat = F.conv1d(x_padded, kernel_conv, groups=1)

        # Reshape back
        y_reshaped = y_flat.view(batch_size, d_model, seq_len)
        y = y_reshaped.transpose(1, 2).contiguous()  # (B, T, D)

        # Sanity check output
        if not torch.all(torch.isfinite(y)):
            raise RuntimeError("USP output contains NaN or Inf values - check parameters and input")

        return y

    def _forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        """Streaming (inference) implementation using ring buffer."""
        batch_size, seq_len, d_model = x.shape

        lambda_alpha = self.lambda_param * self.alpha_param
        output = torch.zeros_like(x)

        # Process each time step
        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]  # (batch_size, d_model)

            # Compute polynomial sum: Σ c_i x_{t-i}
            poly_sum = torch.zeros_like(x_t)

            # We need x_{t-1}, x_{t-2}, ..., x_{t-degree}
            # Ring buffer stores the last 'degree' inputs
            for i in range(1, self.degree + 1):
                # We want x_{t-i}. If we've seen at least i inputs, we have it
                total_inputs_seen = t  # inputs seen so far in this sequence
                if total_inputs_seen >= i:
                    # The input x_{t-i} is at position (buffer_idx - i) mod degree
                    # But buffer_idx points to where we'll write NEXT
                    # So the most recent input (from step t-1) is at (buffer_idx - 1) mod degree
                    history_pos = (self.buffer_idx - i) % self.degree
                    poly_sum += self.coeffs[i] * self.ring_buffer[history_pos]

            # Apply filter: y_t = x_t - λα * poly_sum
            y_t = x_t - lambda_alpha * poly_sum
            output[:, t, :] = y_t

            # Store current input in ring buffer for future use
            self.ring_buffer[self.buffer_idx] = x_t
            self.buffer_idx = (self.buffer_idx + 1) % self.degree

        # Sanity check output
        if not torch.all(torch.isfinite(output)):
            raise RuntimeError("USP streaming output contains NaN or Inf values - check parameters and input")

        return output

    def reset_streaming_state(self):
        """Reset ring buffer for streaming inference."""
        self.ring_buffer.zero_()
        self.buffer_idx.zero_()

    def get_effective_kernel(self) -> torch.Tensor:
        """Get the effective convolution kernel [1, -λα c_1, ..., -λα c_n]."""
        lambda_alpha = self.lambda_param * self.alpha_param
        kernel = torch.zeros(self.degree + 1, dtype=self.coeffs.dtype, device=self.coeffs.device)
        kernel[0] = 1.0
        kernel[1:] = -lambda_alpha * self.coeffs[1:]
        return kernel

    def get_operator_norm_bound(self) -> float:
        """Compute theoretical upper bound on operator norm."""
        return compute_operator_norm_bound(self.coeffs, self.lambda_param.item(), self.alpha_param.item())


class UniversalSequencePreconditioner(nn.Module):
    """
    Universal Sequence Preconditioning (USP) module for STU Sandwich integration.

    This is the main interface for applying USP to sequences. It combines the
    causal polynomial filter with optional layer normalization and provides
    methods for analysis and debugging.

    The mathematical formulation is:
    ũ = G(LN(h)) where G = I - λα Σ c_i S^i

    This ensures that both the nonlinearity φ and the STU core K_θ see spectrally
    preconditioned inputs, improving optimization stability and gradient conditioning.
    """

    def __init__(
        self,
        config: ModelConfig,
        degree: int = 3,
        lambda_init: float = 0.5,
        alpha_init: float = 0.1,
        apply_layer_norm: bool = True,
        learnable_params: bool = True,
    ):
        """
        Initialize Universal Sequence Preconditioner.

        Args:
            config: Model configuration containing d_model and other settings
            degree: Polynomial degree for Chebyshev filter (typically 2-4)
            lambda_init: Initial residual gate value
            alpha_init: Initial preconditioner strength
            apply_layer_norm: Whether to apply layer norm before filtering
            learnable_params: Whether λ and α are trainable
        """
        super().__init__()

        self.config = config
        self.degree = degree
        self.apply_layer_norm = apply_layer_norm

        # Layer normalization (if enabled)
        if apply_layer_norm:
            from .model import LayerNormBase
            self.layer_norm = LayerNormBase.build(config, size=config.d_model)
        else:
            self.layer_norm = nn.Identity()

        # Causal polynomial filter
        self.filter = CausalPolynomialFilter(
            degree=degree,
            d_model=config.d_model,
            lambda_init=lambda_init,
            alpha_init=alpha_init,
            learnable_params=learnable_params,
            device=config.init_device,
            dtype=getattr(config, 'init_dtype', torch.float32),
        )

        # Activation checkpointing support
        self._activation_checkpoint_fn: Optional[callable] = None

    def forward(
        self,
        x: torch.Tensor,
        use_streaming: bool = False,
        return_analysis: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Apply universal sequence preconditioning.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            use_streaming: Use streaming mode for inference
            return_analysis: Return additional analysis information

        Returns:
            Preconditioned tensor, optionally with analysis dict
        """
        # Apply layer normalization
        if self._activation_checkpoint_fn is not None:
            u = self._activation_checkpoint_fn(self.layer_norm, x)
        else:
            u = self.layer_norm(x)

        # Apply causal polynomial filter
        if self._activation_checkpoint_fn is not None:
            u_tilde = self._activation_checkpoint_fn(self.filter, u, use_streaming)
        else:
            u_tilde = self.filter(u, use_streaming=use_streaming)

        if return_analysis:
            analysis = self._compute_analysis(x, u, u_tilde)
            return u_tilde, analysis

        return u_tilde

    def _compute_analysis(self, x_orig: torch.Tensor, u_norm: torch.Tensor, u_tilde: torch.Tensor) -> dict:
        """Compute analysis metrics for debugging and monitoring."""
        with torch.no_grad():
            # Spectral properties
            operator_norm_bound = self.filter.get_operator_norm_bound()

            # Signal statistics
            orig_std = torch.std(x_orig, dim=(0, 1))  # Per-channel std
            norm_std = torch.std(u_norm, dim=(0, 1))
            filtered_std = torch.std(u_tilde, dim=(0, 1))

            # Filtering strength
            lambda_val = self.filter.lambda_param.item()
            alpha_val = self.filter.alpha_param.item()
            effective_strength = lambda_val * alpha_val

            # Frequency analysis (on first channel for efficiency)
            omega, response_mag = analyze_frequency_response(
                self.filter.coeffs, lambda_val, alpha_val, omega_points=256
            )

            return {
                'operator_norm_bound': operator_norm_bound,
                'lambda': lambda_val,
                'alpha': alpha_val,
                'effective_strength': effective_strength,
                'signal_std_orig': orig_std.mean().item(),
                'signal_std_norm': norm_std.mean().item(),
                'signal_std_filtered': filtered_std.mean().item(),
                'frequency_response_min': response_mag.min().item(),
                'frequency_response_max': response_mag.max().item(),
                'low_freq_suppression': 1.0 - response_mag[0].item(),  # Suppression at ω=0
            }

    def set_activation_checkpointing(self, strategy, checkpoint_func=None):
        """Set activation checkpointing for the USP module."""
        from .config import ActivationCheckpointingStrategy
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            from .model import activation_checkpoint_function
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    def reset_streaming_state(self):
        """Reset streaming state for inference."""
        self.filter.reset_streaming_state()

    def get_effective_transfer_function(self, omega_points: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the effective frequency response of the preconditioner."""
        lambda_val = self.filter.lambda_param.item()
        alpha_val = self.filter.alpha_param.item()
        return analyze_frequency_response(self.filter.coeffs, lambda_val, alpha_val, omega_points)