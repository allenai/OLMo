# Universal Sequence Preconditioning (USP) for OLMo

## Overview

Universal Sequence Preconditioning (USP) is a novel spectral regularization technique for improving the optimization stability and performance of STU (Spectral Transform Unit) sandwich layers in the OLMo architecture. USP applies causal polynomial preconditioning using monic Chebyshev polynomials to suppress problematic low-frequency modes while preserving essential high-frequency dynamics.

## Key Benefits

- **Enhanced Training Stability**: Suppresses low-frequency modes that cause gradient instability
- **Improved Convergence**: Better optimization geometry through spectral preconditioning
- **Parameter Efficiency**: Only 2 learnable scalars (λ, α) regardless of model width
- **Streaming Compatible**: Efficient O(nd) implementation for real-time inference
- **Mathematically Principled**: Based on Chebyshev extremal theory for optimal spectral shaping

## Quick Start

### Installation

USP is integrated into the main OLMo codebase. Simply install OLMo as usual:

```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[all]
```

### Basic Usage

Enable USP in your model configuration:

```yaml
model:
  # Enable STU sandwich layers (required for USP)
  stu_enable_mlp_sandwich: true
  stu_layer_schedule: all  # or "alternating"

  # Enable USP preconditioning
  usp_enable: true
  usp_degree: 3                 # Polynomial degree (2-4 recommended)
  usp_lambda_init: 0.5          # Residual gate parameter
  usp_alpha_init: 0.1           # Preconditioning strength
  usp_learnable_params: true    # Make parameters trainable
  usp_apply_layer_norm: true    # Apply layer norm before USP
  usp_warmup_steps: 1000        # Freeze USP params for stability
```

### Training Example

```bash
# Train 150M model with USP-enabled STU sandwich layers
torchrun --nproc_per_node=8 scripts/train.py \
  configs/stu-ablations/OLMo-STU-USP-Sandwich-All-150M.yaml

# Train tiny 20M model for experimentation
python scripts/train.py \
  configs/tiny/OLMo-STU-USP-20M.yaml --save_overwrite
```

## Mathematical Foundation

### Core Formulation

USP applies a causal polynomial filter of the form:

```
G = I - λα Σ(i=1 to n) c_i S^i
```

Where:
- `S` is the causal shift operator: `(Sx)_t := x_{t-1}`
- `c_i` are monic Chebyshev polynomial coefficients
- `λ ∈ [0,1]` is the residual gate parameter
- `α` is the preconditioner strength
- `n` is the polynomial degree

### Integration with STU Sandwich

In the STU sandwich architecture:

```
Input → LayerNorm → USP → MLP_in → SwiGLU → STU_core → MLP_out → Output
```

The mathematical flow becomes:
```
h^(ℓ+1) = h^ℓ + [K_θ(φ(G(LN(h^ℓ)) W_in + b)) W_out + b]
```

### Frequency Response

The frequency response is:
```
G(ω) = 1 - λα C(e^(-iω))
```

Where the Chebyshev design ensures uniform spectral shrinkage with minimal deviation.

## Configuration Reference

### USP Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `usp_enable` | bool | `false` | - | Enable USP in STU sandwich layers |
| `usp_degree` | int | `3` | [1, 10] | Chebyshev polynomial degree |
| `usp_lambda_init` | float | `0.5` | [0, 1] | Initial residual gate parameter |
| `usp_alpha_init` | float | `0.1` | [-1, 1] | Initial preconditioner strength |
| `usp_learnable_params` | bool | `true` | - | Whether λ and α are trainable |
| `usp_apply_layer_norm` | bool | `true` | - | Apply layer norm before USP |
| `usp_warmup_steps` | int | `0` | ≥ 0 | Steps to freeze USP parameters |

### Recommended Settings

**For research/experimentation (20M models):**
```yaml
usp_degree: 2
usp_lambda_init: 0.3
usp_alpha_init: 0.05
usp_warmup_steps: 500
```

**For production training (150M+ models):**
```yaml
usp_degree: 3
usp_lambda_init: 0.5
usp_alpha_init: 0.1
usp_warmup_steps: 1000
```

**For aggressive preconditioning:**
```yaml
usp_degree: 4
usp_lambda_init: 0.7
usp_alpha_init: 0.15
usp_warmup_steps: 2000
```

## Performance Characteristics

### Computational Overhead

| Mode | Additional Cost | Memory | Notes |
|------|----------------|---------|-------|
| Training | Negligible | O(nd) buffers | Dominated by STU operations |
| Streaming | O(nd) per token | O(nd) ring buffer | Efficient for inference |
| Analysis | O(T log T) per block | - | For debugging/monitoring |

### Stability Properties

- **Operator Norm Bound**: `‖G‖ ≤ 1 + |λα| ‖c‖₁`
- **Lipschitz Constant**: Near 1-Lipschitz for small `|λα|`
- **Gradient Conditioning**: Improved via spectral regularization

## Available Configurations

### Pre-configured Models

```bash
# 150M parameter models
configs/stu-ablations/OLMo-STU-USP-Sandwich-All-150M.yaml
configs/stu-ablations/OLMo-STU-USP-Sandwich-Alternating-150M.yaml

# 20M parameter model for experimentation
configs/tiny/OLMo-STU-USP-20M.yaml
```

### Configuration Variants

- **All-STU**: Every layer uses STU with USP (`stu_layer_schedule: all`)
- **Alternating**: Alternate between attention and STU+USP layers (`stu_layer_schedule: alternating`)
- **Attention-Last**: STU+USP layers followed by attention layers (`stu_layer_schedule: attention_last`)

## Monitoring and Analysis

### Runtime Analysis

```python
# Get USP analysis during training
analysis = stu_block.get_usp_analysis(input_tensor)

print(f"Operator norm bound: {analysis['operator_norm_bound']:.4f}")
print(f"Low-frequency suppression: {analysis['low_freq_suppression']:.4f}")
print(f"Effective strength λα: {analysis['effective_strength']:.4f}")
```

### Parameter Monitoring

```python
# Monitor USP parameters during training
for layer_idx, block in enumerate(model.transformer.blocks):
    if hasattr(block, 'usp') and block.usp_enabled:
        lambda_val = block.usp.filter.lambda_param.item()
        alpha_val = block.usp.filter.alpha_param.item()
        print(f"Layer {layer_idx}: λ={lambda_val:.4f}, α={alpha_val:.4f}")
```

### Frequency Response Analysis

```python
# Plot frequency response for debugging
omega, response = block.usp.get_effective_transfer_function()
plt.plot(omega.numpy(), response.numpy())
plt.title("USP Frequency Response")
plt.xlabel("Frequency (rad)")
plt.ylabel("Magnitude")
plt.show()
```

## Troubleshooting

### Common Issues

**Training Instability**
```yaml
# Solution: Reduce preconditioning strength
usp_alpha_init: 0.05  # Lower from 0.1
usp_warmup_steps: 2000  # Increase warmup
```

**Slow Convergence**
```yaml
# Solution: Check parameter learning
usp_learnable_params: true  # Ensure parameters adapt
```

**Memory Issues**
```yaml
# Solution: Reduce polynomial degree
usp_degree: 2  # Lower from 3 for long sequences
```

**NaN/Inf Errors**
- Check input data for NaN values
- Verify `usp_alpha_init` is not too large
- Increase `usp_warmup_steps` for stability

### Debugging Commands

```bash
# Run with detailed logging
python scripts/train.py config.yaml --log-level DEBUG

# Test USP implementation
pytest tests/usp_test.py -v

# Check configuration validity
python -c "from olmo.config import ModelConfig; ModelConfig(usp_enable=True, stu_enable_mlp_sandwich=True)"
```

## Testing

### Run Test Suite

```bash
# Full test suite
pytest tests/usp_test.py -v

# Specific test categories
pytest tests/usp_test.py::TestChebyshevPolynomials -v
pytest tests/usp_test.py::TestCausalPolynomialFilter -v
pytest tests/usp_test.py::TestSTUUSPIntegration -v
```

### Test Coverage

- ✅ Chebyshev polynomial mathematical properties
- ✅ Causal filtering correctness and streaming equivalence
- ✅ STU integration and gradient flow
- ✅ Numerical stability and edge cases
- ✅ Configuration validation
- ✅ Device compatibility

## Experimental Results

### Ablation Studies

To reproduce the ablation studies:

```bash
# Baseline: STU without USP
torchrun --nproc_per_node=8 scripts/train.py \
  configs/stu-ablations/OLMo-STU-Sandwich-All-150M.yaml

# USP variant: STU with USP
torchrun --nproc_per_node=8 scripts/train.py \
  configs/stu-ablations/OLMo-STU-USP-Sandwich-All-150M.yaml

# Compare perplexity and downstream task performance
```

### Expected Improvements

- **Training Stability**: Reduced gradient variance, fewer training failures
- **Convergence Speed**: Faster loss decrease, better learning rate tolerance
- **Downstream Performance**: Improved task performance through better optimization

## Implementation Details

### File Structure

```
olmo/
├── usp.py                 # Core USP implementation
├── stu.py                 # STU blocks with USP integration
├── config.py              # Configuration parameters
└── __init__.py            # Module exports

tests/
└── usp_test.py           # Comprehensive test suite

configs/
├── stu-ablations/
│   ├── OLMo-STU-USP-Sandwich-All-150M.yaml
│   └── OLMo-STU-USP-Sandwich-Alternating-150M.yaml
└── tiny/
    └── OLMo-STU-USP-20M.yaml

docs/
└── USP_IMPLEMENTATION.md  # Technical documentation
```

### Key Classes

- `CausalPolynomialFilter`: Core USP operator with streaming support
- `UniversalSequencePreconditioner`: Main interface with layer norm integration
- `OLMoSTUBlock`: Modified STU block with USP integration

## Contributing

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[dev]

# Run tests
pytest tests/usp_test.py

# Run linting
ruff check olmo/usp.py
black olmo/usp.py
mypy olmo/usp.py
```

### Adding New Features

1. Add functionality to `olmo/usp.py`
2. Update configuration in `olmo/config.py`
3. Add tests to `tests/usp_test.py`
4. Update documentation

## References

1. **Chebyshev Polynomials**: Optimal uniform approximation theory
2. **Spectral Methods**: Sequence modeling via frequency domain analysis
3. **STU Architecture**: Spectral Transform Units for efficient attention alternatives
4. **Residual Networks**: Gradient flow and stability in deep networks

## Citation

If you use USP in your research, please cite:

```bibtex
@software{olmo_usp_2024,
  title={Universal Sequence Preconditioning for STU Sandwich Layers},
  author={OLMo Team},
  year={2024},
  url={https://github.com/allenai/OLMo}
}
```

## License

USP is released under the same license as OLMo. See the main repository for license details.

## Support

- **Documentation**: [USP Implementation Guide](docs/USP_IMPLEMENTATION.md)
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join the OLMo community for questions and collaboration

---

**Note**: USP requires STU sandwich layers (`stu_enable_mlp_sandwich: true`) to function. It provides the most benefit when used with the spectral transform architecture rather than standard attention layers.