# Depth vs. Width Ablation Study

## Hypothesis
The STU-Alternating models perform better partly because they have **fewer but wider layers** (shallower depth), which creates better optimization geometry at 150M scale.

## Experimental Design

### Original Models (Baseline)

#### Regular STU Models
| Model | Layers | d_model | n_heads | Architecture | Param Count |
|-------|--------|---------|---------|--------------|-------------|
| **Transformer-150M** | 12 | 768 | 12 | All attention | ~150M |
| **STU-All-150M** | 15 | 768 | 12 | All STU | ~193M |
| **STU-Alternating-150M** | 14 | 768 | 12 | 7 Attn + 7 STU | ~197M |

#### Sandwich STU Models
| Model | Layers | d_model | n_heads | Architecture | Param Count |
|-------|--------|---------|---------|--------------|-------------|
| **STU-Sandwich-All-150M** | 5 | 768 | 12 | All Sandwich-STU | ~196M |
| **STU-Sandwich-Alternating-150M** | 8 | 768 | 12 | 4 Attn + 4 Sandwich-STU | ~210M |

### New Ablation Models

#### Regular STU Ablations

##### 1. **STU-Deep-Alternating-150M** (Controls for Depth)
- **Layers**: 15 (same as STU-All)
- **d_model**: 704 (reduced from 768)
- **n_heads**: 11
- **Architecture**: Alternating (8 STU + 7 attention)
- **Purpose**: Tests if Alternating's advantage persists at **deeper** depth

##### 2. **STU-Wide-All-150M** (Controls for Width)
- **Layers**: 14 (same as STU-Alternating)
- **d_model**: 768 (same as original)
- **n_heads**: 12
- **Architecture**: All STU
- **Purpose**: Tests if All-STU improves at **shallower** depth

##### 3. **Transformer-Deep-150M** (Pure Transformer Baseline)
- **Layers**: 15
- **d_model**: 704
- **n_heads**: 11
- **Architecture**: All attention
- **Purpose**: Baseline for deep pure transformer

##### 4. **Transformer-Wide-150M** (Pure Transformer Baseline)
- **Layers**: 10
- **d_model**: 832
- **n_heads**: 13
- **Architecture**: All attention
- **Purpose**: Baseline for wide shallow transformer

#### Sandwich STU Ablations

##### 5. **STU-Sandwich-Deep-Alternating-150M** (Controls for Depth)
- **Layers**: 10 (increased from 8)
- **d_model**: 640 (reduced from 768)
- **n_heads**: 10
- **Architecture**: Alternating (5 Sandwich-STU + 5 attention)
- **Purpose**: Tests if Sandwich-Alternating's advantage persists at **deeper** depth

##### 6. **STU-Sandwich-Deep-All-150M** (Controls for Depth)
- **Layers**: 8 (increased from 5)
- **d_model**: 640 (reduced from 768)
- **n_heads**: 10
- **Architecture**: All Sandwich-STU
- **Purpose**: Tests if All-Sandwich improves when made deeper to match Alternating depth

##### 7. **STU-Sandwich-Wide-All-150M** (Controls for Width)
- **Layers**: 4 (reduced from 5)
- **d_model**: 896 (increased from 768)
- **n_heads**: 14
- **Architecture**: All Sandwich-STU
- **Purpose**: Tests if All-Sandwich improves when made **shallower and wider**

---

## Expected Results & Interpretations

### Scenario A: Depth is the dominant factor
If **shallower is better at 150M scale**:
- **STU-Wide-All-150M** (14 layers) should significantly outperform **STU-All-150M** (15 layers)
- **STU-Deep-Alternating-150M** (15 layers) should perform worse than **STU-Alternating-150M** (14 layers)
- **Transformer-Wide-150M** (10 layers) should outperform **Transformer-150M** (12 layers)

**Interpretation**: The optimization benefits of fewer gradient steps dominate at small scale.

---

### Scenario B: Width is the dominant factor
If **wider layers provide more capacity**:
- **STU-Deep-Alternating-150M** (d=704) should perform worse than **STU-Alternating-150M** (d=768)
- Width matters more than number of sequential transformations

**Interpretation**: Per-layer representational capacity matters more than depth.

---

### Scenario C: Alternating pattern is the dominant factor
If **mixing STU + attention** provides the real benefit:
- **STU-Deep-Alternating-150M** should still outperform both:
  - **STU-Wide-All-150M** (same depth, no alternating)
  - **STU-All-150M** (deeper, no alternating)
- The alternating pattern wins regardless of depth/width tradeoff

**Interpretation**: Architectural diversity (STU + attention) is the key factor.

---

### Scenario D: It's all optimization geometry
If the original results are purely about **gradient flow**:
- **Transformer-Wide-150M** should perform best among transformers
- **STU-Wide-All-150M** should approach **STU-Alternating-150M** performance
- Shallower models converge faster in learning curves

**Interpretation**: The depth/width tradeoff affects optimization stability more than representational capacity.

---

## Key Comparisons

### Isolating Depth Effect
Compare models with **same architecture** but different depths:
- **STU-Alternating-150M** (14L, d=768) vs **STU-Deep-Alternating-150M** (15L, d=704)
- **STU-All-150M** (15L, d=768) vs **STU-Wide-All-150M** (14L, d=768)

### Isolating Width Effect
Compare models with **same depth** but different widths:
- **STU-Alternating-150M** (14L, d=768) vs **STU-Wide-All-150M** (14L, d=768)
  - Tests: Does alternating pattern still win at same depth?

### Isolating Architecture Pattern
Compare **alternating vs. all** at different depths:
- **STU-Deep-Alternating-150M** vs **Transformer-Deep-150M** (both 15L, d=704)
- **STU-Wide-All-150M** vs **STU-Alternating-150M** (both 14L, d=768)

---

## Metrics to Track

1. **Final Cross-Entropy Loss** (ΔCE vs. baseline)
2. **Learning Curves** (convergence speed)
3. **Gradient Norm Statistics** (optimization stability)
4. **Downstream Task Performance**
5. **Training Time per Step** (computational efficiency)

---

## Predicted Outcome

Based on the hypothesis, I predict:
- **STU-Wide-All-150M** will perform **better** than STU-All-150M (depth matters)
- **STU-Deep-Alternating-150M** will perform **worse** than STU-Alternating-150M (confirming shallow is better)
- **Transformer-Wide-150M** will show that pure transformers also benefit from shallow+wide
- The alternating pattern will still provide additional gains beyond depth/width optimization

This will help us decompose the gain into:
1. **Optimization effect** (depth/width tradeoff)
2. **Architectural effect** (STU vs. attention)
3. **Diversity effect** (alternating pattern)

---

## Running the Experiments

### Regular STU Ablations
```bash
# Deep Alternating (15 layers, thinner)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-STU-Deep-Alternating-150M.yaml

# Wide All-STU (14 layers, all STU)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-STU-Wide-All-150M.yaml

# Deep Transformer baseline (15 layers)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-Transformer-Deep-150M.yaml

# Wide Transformer baseline (10 layers)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-Transformer-Wide-150M.yaml
```

### Sandwich STU Ablations
```bash
# Deep Sandwich-Alternating (10 layers, thinner)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-STU-Sandwich-Deep-Alternating-150M.yaml

# Deep Sandwich-All (8 layers, matching Sandwich-Alternating depth)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-STU-Sandwich-Deep-All-150M.yaml

# Wide Sandwich-All (4 layers, wider)
python scripts/train.py configs/stu-wide-depth-ablations/OLMo-STU-Sandwich-Wide-All-150M.yaml
```

---

## Notes

- All models use the same training setup (optimizer, scheduler, data, etc.)
- Same seed (6198) for reproducibility
- Parameter counts are approximate; exact counts should be verified
- Focus on relative ΔCE differences, not absolute values

