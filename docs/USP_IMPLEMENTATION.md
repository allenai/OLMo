# Universal Sequence Preconditioning (USP) Implementation

## Overview

Universal Sequence Preconditioning (USP) is a novel technique for improving the optimization stability and spectral properties of STU (Spectral Transform Unit) sandwich layers. USP applies causal polynomial preconditioning using monic Chebyshev polynomials to suppress low-frequency modes that can cause training instabilities.

## Mathematical Foundation

### Core Formulation

The USP operator is defined as:
```
G = I - λα Σ(i=1 to n) c_i S^i
```

Where:
- `S` is the causal shift operator: `(Sx)_t := x_{t-1}` with `(Sx)_1 := 0`
- `c_i` are coefficients from the monic Chebyshev polynomial T̂_n(x) = 2^(1-n) T_n(x)
- `λ ∈ [0,1]` is the residual gate parameter
- `α` is the preconditioner strength parameter
- `n` is the polynomial degree

### Integration with STU Sandwich

In the STU sandwich architecture, USP is applied as:
```
h^(ℓ+1) = h^ℓ + [K_θ(φ(G(LN(h^ℓ)) W_in + b_in)) W_out + b_out]
```

Where:
- `G(LN(h^ℓ))` applies USP preconditioning after layer normalization
- `φ` is the SwiGLU nonlinearity
- `K_θ` is the STU core spectral operator
- `W_in, W_out` are the MLP sandwich projections

### Frequency Response

The frequency response is:
```
G(ω) = 1 - λα C(e^(-iω))
```

Where `C(z) = Σ(i=1 to n) c_i z^i`. The Chebyshev design ensures uniform spectral shrinkage with minimal alternating deviation.

## Implementation Architecture

### Key Components

1. **CausalPolynomialFilter**: Core USP operator with streaming support
2. **UniversalSequencePreconditioner**: Main interface with layer norm integration
3. **OLMoSTUBlock**: Modified STU block with USP integration
4. **Chebyshev coefficient generation**: Mathematical utilities

### File Structure

```
olmo/
├── usp.py                 # Main USP implementation
├── stu.py                 # Modified STU blocks with USP integration
├── config.py              # USP configuration parameters
└── tests/usp_test.py      # Comprehensive test suite

configs/
├── stu-ablations/
│   ├── OLMo-STU-USP-Sandwich-All-150M.yaml
│   └── OLMo-STU-USP-Sandwich-Alternating-150M.yaml
└── tiny/
    └── OLMo-STU-USP-20M.yaml
```

## Configuration Parameters

### USP-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `usp_enable` | bool | `false` | Enable USP in STU sandwich layers |
| `usp_degree` | int | `3` | Chebyshev polynomial degree (2-4 recommended) |
| `usp_lambda_init` | float | `0.5` | Initial residual gate parameter λ |
| `usp_alpha_init` | float | `0.1` | Initial preconditioner strength α |
| `usp_learnable_params` | bool | `true` | Whether λ and α are trainable |
| `usp_apply_layer_norm` | bool | `true` | Apply layer norm before USP |
| `usp_warmup_steps` | int | `0` | Steps to freeze USP parameters |

### Recommended Settings

**For 150M models:**
```yaml
usp_enable: true
usp_degree: 3
usp_lambda_init: 0.5
usp_alpha_init: 0.1
usp_learnable_params: true
usp_apply_layer_norm: true
usp_warmup_steps: 1000
```

**For tiny models (20M):**
```yaml
usp_enable: true
usp_degree: 2
usp_lambda_init: 0.3
usp_alpha_init: 0.05
usp_learnable_params: true
usp_apply_layer_norm: true
usp_warmup_steps: 500
```

## Key Features

### Streaming Compatibility

USP supports both parallel (training) and streaming (inference) modes:

- **Parallel mode**: Uses causal convolution with padding
- **Streaming mode**: Uses ring buffer for O(nd) memory and computation per token

### Gradient Conditioning

USP improves gradient conditioning through:

1. **Spectral regularization**: Suppresses problematic low-frequency modes
2. **Operator norm control**: Theoretical bound ||G||_∞→∞ ≤ 1 + |λα| ||c||_1
3. **Lipschitz properties**: Near-1-Lipschitz for small |λα|

### Mathematical Properties

1. **Causality**: Strictly causal with finite temporal support
2. **Parameter efficiency**: Only 2 learnable scalars regardless of model width
3. **Commutation**: G commutes with STU core K_θ in linear regime
4. **Extremal property**: Chebyshev design minimizes uniform deviation

## Usage Examples

### Basic Usage

```python
from olmo.config import ModelConfig
from olmo.stu import OLMoSTUBlock
from olmo.usp import UniversalSequencePreconditioner

# Configure model with USP
config = ModelConfig(
    d_model=768,
    stu_enable_mlp_sandwich=True,
    usp_enable=True,
    usp_degree=3,
    usp_lambda_init=0.5,
    usp_alpha_init=0.1,
)

# USP is automatically integrated in STU blocks
# when stu_enable_mlp_sandwich=True and usp_enable=True
```

### Analysis and Debugging

```python
# Get USP analysis for monitoring
analysis = stu_block.get_usp_analysis(input_tensor)
print(f"Operator norm bound: {analysis['operator_norm_bound']}")
print(f"Low-freq suppression: {analysis['low_freq_suppression']}")
print(f"Effective strength: {analysis['effective_strength']}")
```

### Streaming Inference

```python
# Reset streaming state before inference
stu_block.reset_streaming_state()

# Process tokens sequentially
for token_batch in input_stream:
    output = stu_block(token_batch)
```

## Performance Characteristics

### Computational Complexity

- **Training**: O(d T log T) per block (dominated by STU spectral ops)
- **Inference**: O(nd) additional cost per token
- **Memory**: O(nd) additional buffers for streaming

### Numerical Stability

- **Operator norm**: Bounded by 1 + |λα| ||c||_1
- **Gradient stability**: Improved conditioning via spectral regularization
- **Parameter sensitivity**: Robust to hyperparameter choices within recommended ranges

## Experimental Validation

### Test Coverage

The test suite (`tests/usp_test.py`) covers:

1. **Mathematical correctness**: Chebyshev properties, causality, streaming equivalence
2. **Integration testing**: STU block compatibility, gradient flow
3. **Numerical stability**: Edge cases, long sequences, extreme parameters
4. **Performance**: Memory usage, computational overhead

### Ablation Studies

Recommended ablation configurations:

1. **Baseline**: STU sandwich without USP
2. **USP-All**: All STU layers with USP (degree 3)
3. **USP-Alternating**: Alternating STU/attention with USP in STU layers
4. **Degree ablation**: USP with degrees 2, 3, 4
5. **Parameter sensitivity**: Different λ, α initial values

## Troubleshooting

### Common Issues

1. **Training instability**: Reduce `usp_alpha_init` or increase `usp_warmup_steps`
2. **Slow convergence**: Check that `usp_learnable_params=true`
3. **Memory errors**: Reduce `usp_degree` for very long sequences
4. **Gradient explosion**: Verify operator norm bounds are reasonable

### Debugging Tools

```python
# Monitor USP parameters during training
lambda_val = model.stu_blocks[0].usp.filter.lambda_param.item()
alpha_val = model.stu_blocks[0].usp.filter.alpha_param.item()
print(f"USP params: λ={lambda_val:.4f}, α={alpha_val:.4f}")

# Check frequency response
omega, response = model.stu_blocks[0].usp.get_effective_transfer_function()
plt.plot(omega, response)
plt.title("USP Frequency Response")
```

## Future Extensions

### Potential Enhancements

1. **Adaptive degree**: Learn polynomial degree during training
2. **Per-layer tuning**: Different USP parameters per layer
3. **Attention integration**: USP for key streams in hybrid architectures
4. **Hardware optimization**: CUDA kernels for streaming mode

### Research Directions

1. **Theoretical analysis**: Convergence guarantees, spectral gap properties
2. **Scale studies**: USP effectiveness at larger model sizes
3. **Task adaptation**: USP tuning for specific downstream tasks
4. **Architecture variants**: USP integration with other sequence models

## References

1. Chebyshev polynomials and approximation theory
2. Spectral methods for sequence modeling
3. STU (Spectral Transform Unit) architecture
4. Residual network optimization theory