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

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: The experiment object defining the optimization problem

### `n_startup_trials`
- **Type**: `int`
- **Default**: `10`
- **Description**: Number of random trials before TPE starts. These initial random samples help build the initial model.

### `n_ei_candidates` 
- **Type**: `int`
- **Default**: `24`
- **Description**: Number of candidate points to evaluate when computing Expected Improvement

### `gamma`
- **Type**: `float`
- **Default**: `0.25`
- **Description**: Quantile for splitting observations into good/bad groups. Lower values create more selective "good" groups.

### `prior_weight`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Prior weight for the Parzen estimators

## Usage Example

```python
from hyperactive.opt.optuna import TPEOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load dataset
X, y = load_wine(return_X_y=True)

# Define search space
param_grid = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    cv=5
)

# Create TPE optimizer with custom parameters
optimizer = TPEOptimizer(
    experiment=experiment,
    n_startup_trials=15,  # More random trials for better initialization
    gamma=0.2  # More selective good/bad split
)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
```

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

```python
# More exploitation (smaller good group)
optimizer = TPEOptimizer(experiment=experiment, gamma=0.1)

# More exploration (larger good group) 
optimizer = TPEOptimizer(experiment=experiment, gamma=0.4)
```

### Warm Starting

```python
# For sequential optimization runs
optimizer = TPEOptimizer(
    experiment=experiment,
    n_startup_trials=5  # Fewer random trials if you have prior knowledge
)
```

## Performance Tips

1. **Start with defaults**: TPE's default parameters work well for most problems
2. **Adjust gamma**: Use smaller gamma (0.1-0.2) for exploitation, larger (0.3-0.4) for exploration
3. **Scale startup trials**: Use 10-20 startup trials for most problems
4. **Parameter space design**: Keep parameter spaces reasonably sized (each dimension <100 values)

## References

- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization.
- Optuna Documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)