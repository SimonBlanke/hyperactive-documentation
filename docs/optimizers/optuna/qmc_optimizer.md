# QMC Optimizer

## Introduction

Quasi-Monte Carlo (QMC) Optimizer uses low-discrepancy sequences instead of random sampling. This provides better coverage of the parameter space compared to pure random sampling, often leading to faster convergence.

## Usage Example

```python
--8<-- "optimizers_optuna_qmc_optimizer_example.py"
```

## When to Use QMC Optimizer

**Best for:**
- Better space coverage than random search
- Integration and sampling problems
- When you want deterministic "random" sequences
- High-dimensional parameter spaces

**Parameters:**
- `qmc_type`: Type of QMC sequence ("sobol", "halton")
- `scramble`: Whether to scramble the sequence

## QMC Sequence Types

### Sobol Sequences
- Excellent space-filling properties
- Good for most optimization problems
- Default choice for QMC

### Halton Sequences  
- Simpler construction
- Can suffer from correlation in higher dimensions
- Good for lower-dimensional problems
