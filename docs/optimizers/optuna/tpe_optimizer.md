# TPE Optimizer

## Introduction

The Tree-structured Parzen Estimator (TPE) Optimizer is a Bayesian optimization algorithm that uses Parzen estimators to model the objective function. It's one of the most popular and effective algorithms for hyperparameter optimization, particularly well-suited for expensive function evaluations.

## About the Implementation

TPE works by building probabilistic models of the objective function using historical evaluation data. It separates the observations into "good" and "bad" groups based on a quantile, then models each group separately using Parzen estimators (kernel density estimation). The algorithm selects new points by maximizing the Expected Improvement criterion.

Key features:
- **Adaptive modeling**: Builds better models as more data is collected
- **Categorical support**: Handles mixed parameter spaces naturally
- **Robust performance**: Works well across many different optimization problems
- **Sample efficient**: Requires fewer evaluations than random search

## Parameters

### Common (via Base Optuna Adapter)
- `param_space` (dict): parameter space; tuples/lists treated as ranges/choices
- `n_trials` (int): number of trials to run
- `initialize` (dict | None): optional warm start/grid/vertices/random init
- `early_stopping` (int | None): stop if no improvement after N trials
- `max_score` (float | None): stop when reaching threshold
- `experiment` (BaseExperiment): the experiment to optimize

### `n_startup_trials`
- **Type**: `int`
- **Default**: `10`
- **Description**: Number of random trials before TPE starts. These initial random samples help build the initial model.

### `n_ei_candidates` 
- **Type**: `int`
- **Default**: `24`
- **Description**: Number of candidate points to evaluate when computing Expected Improvement

### `weights`
- **Type**: `callable | None`
- **Default**: `None`
- **Description**: Optional weighting function passed to Optuna's TPESampler.

### `random_state`
- **Type**: `int | None`
- **Default**: `None`
- **Description**: Seed for reproducibility (sets `seed` in the underlying TPESampler).

## Usage Example



## When to Use TPE

**Best for:**
- **Mixed parameter spaces**: Handles continuous, discrete, and categorical parameters
- **Moderate evaluation budgets**: Works well with 50-500 evaluations
- **Expensive function evaluations**: Sample-efficient compared to grid/random search
- **General-purpose optimization**: Robust across many problem types

**Consider alternatives if:**
- **Very high dimensions**: May struggle with >50 parameters
- **Very cheap evaluations**: Random search might be sufficient
- **Specific problem structure**: Specialized algorithms might be better

## Comparison with Other Algorithms

| Algorithm | Sample Efficiency | Parameter Types | Computational Cost |
|-----------|------------------|------------------|-------------------|
| TPE | High | All types | Medium |
| Random Search | Low | All types | Low |
| Bayesian Opt | High | Mostly continuous | High |
| Grid Search | Low | Discrete only | Low |

## Advanced Usage

### Custom Gamma Values



### Warm Starting



## Performance Tips

1. **Start with defaults**: TPE's default parameters work well for most problems
2. **Adjust gamma**: Use smaller gamma (0.1-0.2) for exploitation, larger (0.3-0.4) for exploration
3. **Scale startup trials**: Use 10-20 startup trials for most problems
4. **Parameter space design**: Keep parameter spaces reasonably sized (each dimension <100 values)

## References

- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization.
- Optuna Documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
