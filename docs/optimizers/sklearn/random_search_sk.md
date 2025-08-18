# Random Search SK

## Introduction

Random Search SK provides direct integration with scikit-learn's RandomizedSearchCV, combining the efficiency of random sampling with sklearn's mature cross-validation infrastructure. This optimizer randomly samples parameter combinations from specified distributions.

## About the Implementation

This optimizer leverages sklearn's RandomizedSearchCV implementation, offering:

- **Native sklearn integration**: Uses sklearn's RandomizedSearchCV under the hood
- **Distribution sampling**: Can sample from probability distributions, not just discrete lists
- **Parallel execution**: Full support for sklearn's n_jobs parameter
- **Proven efficiency**: Often matches or outperforms grid search with fewer evaluations

## Parameters

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: The experiment object defining the optimization problem

### `n_iter`
- **Type**: `int`
- **Default**: `100`
- **Description**: Number of parameter combinations to evaluate

### `n_jobs`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of parallel jobs for cross-validation. -1 uses all processors.

### `cv`
- **Type**: `int` or cross-validation generator
- **Default**: `3`
- **Description**: Cross-validation strategy

### `random_state`
- **Type**: `int` or `None`
- **Default**: `None`
- **Description**: Random seed for reproducible results

### `verbose`
- **Type**: `int`
- **Default**: `0`
- **Description**: Verbosity level

## Usage Example

```python
from hyperactive.opt import RandomSearchSk
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from scipy.stats import randint, uniform

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Define search space with distributions
param_distributions = {
    "n_estimators": randint(50, 300),  # Random integers between 50-299
    "max_depth": randint(3, 20),       # Random integers between 3-19
    "min_samples_split": uniform(0.01, 0.19),  # Uniform float between 0.01-0.2
    "min_samples_leaf": uniform(0.01, 0.09),   # Uniform float between 0.01-0.1
    "max_features": ["sqrt", "log2", None]  # Discrete choices
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_distributions,  # Can use distributions
    X=X, y=y,
    cv=5,
    scoring="neg_mean_squared_error"
)

# Create random search optimizer
optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=200,  # Number of random samples
    n_jobs=-1,   # Use all available cores
    cv=5,
    random_state=42,
    verbose=1
)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
```

## When to Use Random Search SK

**Best for:**
- **Large parameter spaces**: More efficient than grid search for high-dimensional spaces
- **Mixed parameter types**: Handles continuous and discrete parameters naturally
- **Limited time budget**: Can stop after any number of evaluations
- **Distribution sampling**: When you want to sample from probability distributions
- **sklearn workflows**: Familiar interface for sklearn users

**Consider alternatives if:**
- **Small parameter spaces**: Grid search might be more thorough
- **Very expensive evaluations**: Bayesian methods might be more sample-efficient
- **Need sophisticated search**: TPE or Bayesian optimization might be better

## Parameter Distributions

### Discrete Uniform (Choice from List)

```python
param_distributions = {
    "kernel": ["linear", "rbf", "poly"],  # Choose randomly from list
    "degree": [2, 3, 4, 5],              # Choose randomly from integers
}
```

### Continuous Distributions

```python
from scipy.stats import uniform, loguniform, randint

param_distributions = {
    "C": loguniform(1e-3, 1e3),          # Log-uniform between 0.001 and 1000
    "gamma": loguniform(1e-4, 1e1),      # Log-uniform between 0.0001 and 10
    "tol": uniform(1e-5, 1e-3),          # Uniform between 0.00001 and 0.001
    "max_iter": randint(100, 2000),      # Random integers 100-1999
}
```

### Custom Distributions

```python
from scipy.stats import norm, expon

param_distributions = {
    "learning_rate": expon(scale=0.01),   # Exponential distribution
    "weight_decay": norm(0.001, 0.0005), # Normal distribution (mean=0.001, std=0.0005)
}
```

## Advanced Usage

### Budget Management

```python
# Quick exploration
optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=50,    # Fewer iterations for quick results
    n_jobs=-1,
    random_state=42
)

# Thorough search
optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=500,   # More iterations for thorough exploration
    n_jobs=-1,
    random_state=42
)
```

### Reproducible Results

```python
# Set random state for reproducibility
optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=100,
    random_state=42  # Fixed seed
)

# Multiple runs with different seeds
seeds = [42, 123, 456, 789]
results = []

for seed in seeds:
    optimizer = RandomSearchSk(
        experiment=experiment,
        n_iter=100,
        random_state=seed
    )
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    results.append((seed, best_params, score))

# Find best across all runs
best_result = max(results, key=lambda x: x[2])
```

### Progressive Search

```python
# Coarse search first
coarse_distributions = {
    "n_estimators": randint(50, 200),
    "max_depth": randint(3, 15),
    "learning_rate": uniform(0.01, 0.19)  # 0.01 to 0.2
}

coarse_optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=50,
    random_state=42
)
coarse_best = coarse_optimizer.solve()

# Then refine around promising regions
# (This requires manual implementation based on coarse_best results)
```

## Comparison with Grid Search

| Aspect | Random Search SK | Grid Search SK |
|--------|------------------|----------------|
| **Parameter Space** | Large/infinite | Small/finite |
| **Sampling** | Random from distributions | Systematic combinations |
| **Efficiency** | High for large spaces | Low for large spaces |
| **Completeness** | Probabilistic coverage | Complete coverage |
| **Stopping** | Any time | Must complete grid |
| **Distributions** | Supports continuous | Discrete values only |

## Performance Tips

### Parallel Execution

```python
# Maximize parallelization
optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=200,
    n_jobs=-1,      # Use all cores
    cv=3            # Balance CV folds with parallelization
)
```

### Memory Management

```python
# For memory-constrained environments
optimizer = RandomSearchSk(
    experiment=experiment,
    n_iter=100,
    n_jobs=2,       # Limit parallel jobs
    cv=3,           # Fewer CV folds
    verbose=0       # Reduce output
)
```

### Search Space Design

```python
# Effective parameter space design
from scipy.stats import loguniform, uniform, randint

param_distributions = {
    # Log scale for parameters spanning orders of magnitude
    "learning_rate": loguniform(1e-5, 1e-1),
    "regularization": loguniform(1e-6, 1e-2),
    
    # Linear scale for bounded parameters
    "dropout_rate": uniform(0.0, 0.5),
    
    # Integer ranges for discrete parameters
    "hidden_units": randint(32, 512),
    
    # Discrete choices for categorical parameters
    "activation": ["relu", "tanh", "sigmoid"]
}
```

## Common Use Cases

### Neural Network Hyperparameters

```python
from sklearn.neural_network import MLPRegressor

param_distributions = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
    "learning_rate_init": loguniform(1e-4, 1e-1),
    "alpha": loguniform(1e-6, 1e-2),
    "batch_size": randint(32, 256)
}
```

### Ensemble Method Tuning

```python
from sklearn.ensemble import GradientBoostingClassifier

param_distributions = {
    "n_estimators": randint(50, 300),
    "learning_rate": uniform(0.01, 0.3),
    "max_depth": randint(3, 10),
    "subsample": uniform(0.8, 0.2)  # 0.8 to 1.0
}
```

### SVM Optimization

```python
from sklearn.svm import SVC

param_distributions = {
    "C": loguniform(1e-2, 1e2),
    "gamma": loguniform(1e-4, 1e-1),
    "kernel": ["rbf", "linear", "poly"]
}
```

## Integration with Experiment Pipeline

```python
# Multi-stage optimization
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_distributions = {
    'scaler__with_mean': [True, False],
    'scaler__with_std': [True, False],
    'regressor__n_estimators': randint(50, 200),
    'regressor__max_depth': randint(5, 20)
}
```

## Best Practices

1. **Distribution choice**: Use log-uniform for parameters spanning orders of magnitude
2. **Sample size**: Use n_iter ≥ 10 × number of parameters as a rule of thumb
3. **Random seeds**: Set random_state for reproducible results
4. **Cross-validation**: Choose appropriate CV strategy for your data
5. **Parallel processing**: Leverage n_jobs for faster execution
6. **Progressive refinement**: Start broad, then narrow down promising regions

## References

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
- Scikit-learn RandomizedSearchCV documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- scipy.stats distributions: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)