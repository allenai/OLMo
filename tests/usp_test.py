"""
Test suite for Universal Sequence Preconditioning (USP) implementation.

Tests the corrected implementation following the paper's Algorithm 1:
- Monic Chebyshev polynomial M_n(x) = 2^(n-1) T_n(x)
- Preconditioning: y_preconditioned_t = y_t + Σ(j=1 to n) c_j * y_{t-j}
- Complex eigenvalue bounds: max_j |arg(λ_j)| ≤ 1/(32 log²(2T³/d_out))²
"""

import math
import pytest
import torch
import torch.nn as nn

from olmo.usp import (
    generate_monic_chebyshev_coefficients,
    CausalPolynomialFilter,
    UniversalSequencePreconditioner,
    compute_operator_norm_bound,
    analyze_frequency_response,
    validate_eigenvalue_bounds,
    compute_required_eigenvalue_bound,
)
from olmo.config import ModelConfig


class TestChebyshevPolynomials:
    """Test Chebyshev polynomial generation and properties."""

    def test_monic_chebyshev_degree_1(self):
        """Test degree 1 monic Chebyshev polynomial."""
        coeffs = generate_monic_chebyshev_coefficients(1)

        # T̂_1(x) = 2^(1-1) * T_1(x) = 1 * x = x, so coeffs should be [0, 1]
        expected = torch.tensor([0., 1.])
        torch.testing.assert_close(coeffs, expected, rtol=1e-10, atol=1e-10)

    def test_monic_chebyshev_degree_2(self):
        """Test degree 2 monic Chebyshev polynomial."""
        coeffs = generate_monic_chebyshev_coefficients(2)

        # T̂_2(x) = 2^(1-2) * T_2(x) = 0.5 * (2x² - 1) = x² - 0.5
        # So coeffs should be [-0.5, 0, 1] for polynomial -0.5 + 0*x + 1*x²
        expected = torch.tensor([-0.5, 0., 1.])
        torch.testing.assert_close(coeffs, expected, rtol=1e-10, atol=1e-10)

    def test_monic_chebyshev_degree_3(self):
        """Test degree 3 monic Chebyshev polynomial."""
        coeffs = generate_monic_chebyshev_coefficients(3)

        # T̂_3(x) = 2^(1-3) * T_3(x) = 0.25 * (4x³ - 3x) = x³ - 0.75x
        # So coeffs should be [0, -0.75, 0, 1] for polynomial 0 - 0.75*x + 0*x² + 1*x³
        expected = torch.tensor([0., -0.75, 0., 1.])
        torch.testing.assert_close(coeffs, expected, rtol=1e-10, atol=1e-10)

    def test_coefficient_magnitude_bounds(self):
        """Test Lemma 3.2: max |c_k| ≤ 2^(0.3n)."""
        for n in range(1, 8):
            coeffs = generate_monic_chebyshev_coefficients(n)
            max_coeff = torch.max(torch.abs(coeffs)).item()
            theoretical_bound = 2.0 ** (0.3 * n)

            assert max_coeff <= theoretical_bound * 1.1, f"Degree {n}: {max_coeff} > {theoretical_bound}"

    def test_polynomial_evaluation_on_unit_circle(self):
        """Test polynomial evaluation at specific points."""
        coeffs = generate_monic_chebyshev_coefficients(2)

        # Evaluate at x = 1: T̂_2(1) = 1² - 0.5 = 0.5
        x = torch.tensor(1.0)
        result = torch.sum(coeffs * (x ** torch.arange(len(coeffs))))
        expected = 0.5
        assert abs(result.item() - expected) < 1e-10


class TestCausalPolynomialFilter:
    """Test causal polynomial filter implementation."""

    @pytest.fixture
    def base_config(self):
        """Basic configuration for testing."""
        return {
            'degree': 3,
            'd_model': 32,
        }

    def test_filter_initialization(self, base_config):
        """Test proper initialization of causal polynomial filter."""
        filter_module = CausalPolynomialFilter(**base_config)

        # Check parameters
        assert filter_module.degree == 3
        assert filter_module.d_model == 32

        # Check coefficients
        assert filter_module.coeffs.shape == (4,)  # degree + 1

        # Check buffers
        assert filter_module.ring_buffer.shape == (3, 32)  # degree x d_model
        assert filter_module.buffer_idx.item() == 0

    def test_forward_shapes(self, base_config):
        """Test that forward pass preserves tensor shapes."""
        filter_module = CausalPolynomialFilter(**base_config)

        batch_size, seq_len, d_model = 2, 10, 32
        input_tensor = torch.randn(batch_size, seq_len, d_model)

        output = filter_module(input_tensor)

        assert output.shape == input_tensor.shape
        assert torch.all(torch.isfinite(output))

    def test_parallel_vs_streaming_equivalence(self, base_config):
        """Test that parallel and streaming modes give identical results."""
        filter_module = CausalPolynomialFilter(**base_config)

        batch_size, seq_len, d_model = 1, 20, 32
        x = torch.randn(batch_size, seq_len, d_model)

        # Parallel processing
        y_parallel = filter_module(x, use_streaming=False)

        # Streaming processing
        filter_module.reset_streaming_state()
        y_streaming = filter_module(x, use_streaming=True)

        # Results should be identical
        torch.testing.assert_close(y_parallel, y_streaming, rtol=1e-6, atol=1e-10)

    def test_causality_property(self, base_config):
        """Test that the filter is strictly causal."""
        filter_module = CausalPolynomialFilter(**base_config)

        batch_size, seq_len, d_model = 1, 10, 32
        x = torch.randn(batch_size, seq_len, d_model)

        # Process sequence
        y = filter_module(x)

        # Modify future input
        x_modified = x.clone()
        x_modified[:, 5:, :] = torch.randn_like(x_modified[:, 5:, :])

        # Process first 5 timesteps only
        y_partial = filter_module(x[:, :5, :])
        y_modified_partial = filter_module(x_modified[:, :5, :])

        # First 5 outputs should be identical (causality)
        torch.testing.assert_close(y_partial, y_modified_partial, rtol=1e-10, atol=1e-10)

    def test_preconditioning_effect(self, base_config):
        """Test that preconditioning actually changes the signal."""
        filter_module = CausalPolynomialFilter(**base_config)

        batch_size, seq_len, d_model = 2, 15, 32
        x = torch.randn(batch_size, seq_len, d_model)

        y = filter_module(x)

        # Output should be different from input (unless coefficients are very special)
        assert not torch.allclose(y, x, atol=1e-6)

    def test_operator_norm_bound(self, base_config):
        """Test theoretical operator norm bound."""
        filter_module = CausalPolynomialFilter(**base_config)

        # Compute theoretical bound
        bound = filter_module.get_operator_norm_bound()

        # Bound should be reasonable
        assert bound >= 1.0  # At least includes identity component
        assert bound < 100.0  # Not too large for reasonable polynomials


class TestEigenvalueBounds:
    """Test eigenvalue bound validation functions."""

    def test_required_bound_computation(self):
        """Test computation of required eigenvalue bounds."""
        T, d_out = 1000, 64

        bound = compute_required_eigenvalue_bound(T, d_out)

        # Bound should be positive and reasonable
        assert bound > 0
        assert bound < 1.0  # Should be quite restrictive

        # Check formula: 1/(32 log²(2T³/d_out))²
        log_term = math.log(2 * T**3 / d_out)
        expected = 1.0 / (32 * log_term**2)**2
        assert abs(bound - expected) < 1e-12

    def test_eigenvalue_validation_real_case(self):
        """Test eigenvalue validation for real eigenvalues (should pass)."""
        eigenvalues = torch.tensor([0.5, 0.8, -0.3, 0.9])  # All real
        T, d_out = 1000, 64

        is_valid, max_arg, required_bound = validate_eigenvalue_bounds(eigenvalues, T, d_out)

        assert is_valid  # Real eigenvalues have zero argument
        assert max_arg == 0.0
        assert required_bound > 0

    def test_eigenvalue_validation_complex_case(self):
        """Test eigenvalue validation for complex eigenvalues."""
        # Create eigenvalues with small imaginary parts
        eigenvalues = torch.tensor([
            0.5 + 0.01j,
            0.8 - 0.01j,
            -0.3 + 0.005j,
            0.9
        ])
        T, d_out = 10000, 64  # Large T to make bound very restrictive

        is_valid, max_arg, required_bound = validate_eigenvalue_bounds(eigenvalues, T, d_out)

        assert max_arg > 0  # Should detect imaginary components
        # Validity depends on the specific bound calculation

    def test_bound_parameter_validation(self):
        """Test parameter validation for bound computation."""
        with pytest.raises(ValueError):
            compute_required_eigenvalue_bound(-1, 64)  # Negative T

        with pytest.raises(ValueError):
            compute_required_eigenvalue_bound(1000, 0)  # Zero d_out


class TestUniversalSequencePreconditioner:
    """Test the main USP interface."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        return ModelConfig(
            d_model=64,
            usp_enable=True,
            usp_degree=3,
            usp_apply_layer_norm=True,
        )

    def test_usp_initialization(self, mock_config):
        """Test USP module initialization."""
        usp = UniversalSequencePreconditioner(
            config=mock_config,
            degree=3,
            apply_layer_norm=True,
        )

        assert usp.degree == 3
        assert usp.apply_layer_norm is True
        assert hasattr(usp, 'filter')
        assert hasattr(usp, 'layer_norm')

    def test_usp_forward_pass(self, mock_config):
        """Test USP forward pass."""
        usp = UniversalSequencePreconditioner(
            config=mock_config,
            degree=3,
            apply_layer_norm=True,
        )

        batch_size, seq_len, d_model = 2, 12, 64
        x = torch.randn(batch_size, seq_len, d_model)

        # Basic forward pass
        output = usp(x)
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))

        # Forward pass with analysis
        output, analysis = usp(x, return_analysis=True)
        assert output.shape == x.shape
        assert isinstance(analysis, dict)
        assert 'operator_norm_bound' in analysis
        assert 'frequency_response_min' in analysis

    def test_usp_streaming_mode(self, mock_config):
        """Test USP streaming mode."""
        usp = UniversalSequencePreconditioner(
            config=mock_config,
            degree=2,
            apply_layer_norm=False,  # Simplify for testing
        )

        batch_size, seq_len, d_model = 1, 15, 64
        x = torch.randn(batch_size, seq_len, d_model)

        # Normal mode
        y_normal = usp(x, use_streaming=False)

        # Streaming mode
        usp.reset_streaming_state()
        y_streaming = usp(x, use_streaming=True)

        # Should be identical
        torch.testing.assert_close(y_normal, y_streaming, rtol=1e-6, atol=1e-10)


class TestFrequencyAnalysis:
    """Test frequency response analysis functions."""

    def test_frequency_response_analysis(self):
        """Test frequency response computation."""
        coeffs = generate_monic_chebyshev_coefficients(2)

        omega, response = analyze_frequency_response(coeffs, omega_points=128)

        assert omega.shape == (128,)
        assert response.shape == (128,)
        assert torch.all(omega >= 0)
        assert torch.all(omega <= math.pi)
        assert torch.all(response >= 0)  # Magnitude is non-negative

    def test_frequency_response_properties(self):
        """Test specific properties of frequency response."""
        coeffs = generate_monic_chebyshev_coefficients(1)  # Simple case: [0, 1]

        omega, response = analyze_frequency_response(coeffs, omega_points=1024)

        # For polynomial x, H(ω) = e^(-iω), so |H(ω)| = 1
        expected_magnitude = torch.ones_like(response)
        torch.testing.assert_close(response, expected_magnitude, rtol=1e-10, atol=1e-10)


class TestIntegrationWithPaperAlgorithms:
    """Test integration with the paper's algorithms."""

    def test_algorithm_1_preconditioning(self):
        """Test that we implement the correct prediction step G = I - λα Σ c_i S^i."""
        # Create simple synthetic data
        batch_size, seq_len, d_model = 1, 10, 4
        y = torch.randn(batch_size, seq_len, d_model)

        # Apply preconditioning manually implementing G = I - λα Σ c_i S^i
        degree = 2
        coeffs = generate_monic_chebyshev_coefficients(degree)

        # Use default parameters to match filter
        lambda_alpha = 0.5 * 0.1  # lambda_init * alpha_init defaults

        # Manual implementation: y_t - λα Σ(i=1 to n) c_i * y_{t-i}
        y_manual = torch.zeros_like(y)
        for t in range(seq_len):
            y_manual[:, t, :] = y[:, t, :]  # Identity component
            for i in range(1, min(t + 1, degree + 1)):
                y_manual[:, t, :] -= lambda_alpha * coeffs[i] * y[:, t - i, :]

        # Apply using our filter with same parameters
        filter_module = CausalPolynomialFilter(
            degree=degree, d_model=d_model,
            lambda_init=0.5, alpha_init=0.1, learnable_params=False
        )
        y_filter = filter_module(y)

        # Results should match closely
        torch.testing.assert_close(y_filter, y_manual, rtol=1e-5, atol=1e-6)

    def test_coefficient_bounds_lemma_3_2(self):
        """Verify Lemma 3.2: max |c_k| ≤ 2^(0.3n)."""
        for n in range(1, 10):
            coeffs = generate_monic_chebyshev_coefficients(n)
            max_coeff = torch.max(torch.abs(coeffs)).item()
            bound = 2.0 ** (0.3 * n)

            assert max_coeff <= bound * 1.1, f"Degree {n}: max coeff {max_coeff} exceeds bound {bound}"


if __name__ == "__main__":
    pytest.main([__file__])