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
from hyperactive.opt.optuna import RandomOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Define search space
param_grid = {
    "n_estimators": [50, 100, 150, 200, 300, 400, 500],
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 8, 12],
    "max_features": ["sqrt", "log2", None, 0.5, 0.7, 0.9]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    cv=5,
    scoring="neg_mean_squared_error"
)

# Create random optimizer
optimizer = RandomOptimizer(experiment=experiment, seed=42)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
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
# For reproducible random sampling
optimizer = RandomOptimizer(experiment=experiment, seed=42)

# Run multiple times with different seeds
seeds = [42, 123, 456, 789, 999]
results = []

for seed in seeds:
    optimizer = RandomOptimizer(experiment=experiment, seed=seed)
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    results.append((seed, best_params, score))

# Find best across all runs
best_run = max(results, key=lambda x: x[2])
print(f"Best result from seed {best_run[0]}: {best_run[2]}")
```

### Parallel Evaluation Strategy

```python
# Random search is naturally parallel
# You can run multiple optimizers simultaneously

import concurrent.futures
from functools import partial

def run_random_optimization(seed, experiment):
    optimizer = RandomOptimizer(experiment=experiment, seed=seed)
    return optimizer.solve()

# Run multiple random searches in parallel
seeds = range(10)
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(
        partial(run_random_optimization, experiment=experiment),
        seeds
    ))

# Evaluate all results
scores = [experiment.score(params)[0] for params in results]
best_idx = max(range(len(scores)), key=lambda i: scores[i])
best_params = results[best_idx]
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
# Use random search to explore, then switch to sophisticated methods
random_optimizer = RandomOptimizer(experiment=experiment, seed=42)
random_best = random_optimizer.solve()

# Use random results to inform more sophisticated search
# (This pattern requires manual implementation)
```

### Multi-Resolution Search

```python
# Coarse random search first
coarse_param_grid = {
    "n_estimators": [50, 100, 200, 500],  # Fewer options
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 10, 20]
}

coarse_experiment = SklearnCvExperiment(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=coarse_param_grid,
    X=X, y=y, cv=3  # Faster CV
)

coarse_optimizer = RandomOptimizer(experiment=coarse_experiment)
coarse_best = coarse_optimizer.solve()

# Then refine around promising regions
```

### Evaluation Budget Management

```python
# For limited budgets, track progress
class BudgetTracker:
    def __init__(self, max_evals=100):
        self.max_evals = max_evals
        self.current_evals = 0
        self.best_score = float('-inf')
        self.best_params = None
    
    def should_continue(self):
        return self.current_evals < self.max_evals
    
    def update(self, params, score):
        self.current_evals += 1
        if score > self.best_score:
            self.best_score = score
            self.best_params = params

# Use with manual evaluation loop if needed
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