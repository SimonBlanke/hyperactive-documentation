# Random Optimizer

## Introduction

The Random Optimizer performs random sampling of the parameter space. While simple, random search is often surprisingly effective and serves as an important baseline for more sophisticated optimization algorithms. It's particularly useful for high-dimensional spaces and as a starting point for optimization studies.

## About the Implementation

Random search uniformly samples from the defined parameter space without using information from previous evaluations. Despite its simplicity, it has several advantages:

- **No bias**: Doesn't get stuck in local optima
- **Parallelizable**: All evaluations are independent
- **Robust**: Works consistently across different problem types
- **Fast**: Minimal computational overhead

## Parameters

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: The experiment object defining the optimization problem

### `seed`
- **Type**: `int` or `None`
- **Default**: `None`
- **Description**: Random seed for reproducible results

## Usage Example

```python
--8<-- "optimizers_optuna_random_optimizer_example.py"
```

## When to Use Random Search

**Best for:**
- **Baseline comparison**: Always run random search as a baseline
- **High-dimensional spaces**: Often competitive with sophisticated methods
- **Unknown problem structure**: When you don't know what algorithm to use
- **Parallel execution**: All evaluations can run simultaneously
- **Quick exploration**: Fast way to get a sense of the parameter space
- **Large search spaces**: Covers space more uniformly than grid search

**Consider alternatives if:**
- **Limited evaluation budget**: More sophisticated methods may find better solutions faster
- **Expensive evaluations**: Bayesian methods or TPE might be more sample-efficient
- **Known smooth landscapes**: Gradient-based or model-based methods might be better

## Comparison with Other Methods

| Aspect | Random Search | Grid Search | Bayesian Opt | TPE |
|--------|--------------|-------------|---------------|-----|
| Sample Efficiency | Low | Very Low | High | High |
| Computational Overhead | None | None | High | Medium |
| Parallelization | Perfect | Perfect | Limited | Limited |
| Parameter Types | All | All | Mostly Continuous | All |
| Consistency | High | High | Medium | Medium |

## Advanced Usage

### Reproducible Results

```python
--8<-- "optimizers_optuna_random_optimizer_example_2.py"
```

### Parallel Evaluation Strategy

```python
--8<-- "optimizers_optuna_random_optimizer_example_3.py"
```

## Mathematical Properties

Random search has several important theoretical properties:

### Convergence Rate
For a function with global optimum in the top $p$-quantile, random search needs $O(\frac{1}{p})$ samples to find a solution in that quantile with high probability.

### High-Dimensional Performance
Unlike grid search, random search performance doesn't degrade exponentially with dimensionality. It's particularly effective when only a subset of parameters significantly affect the objective.

### Coverage Properties
Random search provides uniform coverage of the parameter space, avoiding bias toward particular regions.

## Performance Tips

1. **Use as baseline**: Always compare other algorithms against random search
2. **Multiple runs**: Run several times with different seeds for robustness
3. **Large budgets**: Give random search sufficient evaluations (100+ for small spaces)
4. **Parameter space design**: Ensure parameter ranges are well-chosen
5. **Parallel execution**: Take advantage of perfect parallelizability

## Common Patterns

### Warm-up Phase

```python
--8<-- "optimizers_optuna_random_optimizer_example_4.py"
```

### Multi-Resolution Search

```python
--8<-- "optimizers_optuna_random_optimizer_example_5.py"
```

### Evaluation Budget Management

```python
--8<-- "optimizers_optuna_random_optimizer_example_6.py"
```

## Research Applications

Random search is extensively used in:

- **Neural architecture search**: Exploring architectural choices
- **Hyperparameter optimization**: Baseline for AutoML systems
- **Reinforcement learning**: Policy and algorithm hyperparameters
- **Scientific computing**: Parameter studies in simulations

## References

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
- Li, L., et al. (2016). Hyperband: A novel bandit-based approach to hyperparameter optimization.
- Optuna documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)