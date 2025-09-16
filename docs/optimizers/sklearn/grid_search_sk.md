# GridSearchSk

## Introduction

`GridSearchSk` exhaustively evaluates a sklearn-style parameter grid using Hyperactive’s evaluation pipeline. It combines `sklearn.model_selection.ParameterGrid` with a Hyperactive `Experiment` (commonly `SklearnCvExperiment`).

## About the Implementation

- Uses `ParameterGrid` to enumerate candidate parameter sets
- Uses the provided `experiment` to evaluate each candidate (e.g., CV via `SklearnCvExperiment`)
- Parallelism controlled via Hyperactive backends (`backend`, `backend_params`)

## Parameters

### `param_grid`
- **Type**: `dict[str, list]`
- **Description**: Sklearn-style parameter grid to exhaustively evaluate

### `error_score`
- **Type**: `float`, default `np.nan`
- **Description**: Score to assign if evaluation raises

### `backend`
- **Type**: `{"None","loky","multiprocessing","threading","joblib","dask","ray"}`
- **Description**: Parallel backend used by Hyperactive

### `backend_params`
- **Type**: `dict` or `None`
- **Description**: Backend configuration (e.g., `{"n_jobs": -1}` for joblib)

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: Experiment used to evaluate candidates (e.g., `SklearnCvExperiment`)

## Usage Example

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import GridSearchSk

X, y = load_iris(return_X_y=True)
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

opt = GridSearchSk(
    param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    backend="joblib",
    backend_params={"n_jobs": -1},
    experiment=exp,
)
best_params = opt.solve()
```

## When to Use Grid Search SK

**Best for:**
- **Small parameter spaces**: When exhaustive search is feasible
- **Discrete parameters**: Works with any parameter type sklearn supports
- **Reproducible results**: Deterministic and systematic exploration
- **sklearn familiarity**: When you want sklearn's interface and behavior
- **Parallel resources**: When you can leverage multiple cores effectively

**Consider alternatives if:**
- **Large parameter spaces**: Combinatorial explosion makes it infeasible
- **Continuous parameters**: Random or Bayesian methods might be better
- **Limited time budget**: More efficient methods available
- **Expensive evaluations**: Sample-efficient methods preferred

## Comparison with Other Grid Search Methods

| Method | Backend | Parallelization | Notes |
|--------|---------|----------------|-------|
| GridSearchSk | Hyperactive | Backends via `backend` | Sklearn-style grid + Hyperactive experiments |
| GridSearch (GFO) | GFO | Internal to GFO | Grid traversal (step/direction), different algorithm |
| GridOptimizer (Optuna) | Optuna | Optuna | Uses Optuna’s grid sampler |

## Advanced Usage

### Custom Cross-Validation

Use a custom CV splitter in the experiment:

```python
from sklearn.model_selection import StratifiedKFold
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y, cv=StratifiedKFold(5, shuffle=True))
```

### Different Scoring Metrics

Provide a scorer string or callable to the experiment:

```python
from sklearn.metrics import f1_score
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y, scoring=f1_score)
```

### Verbose Output



## Performance Optimization

### Parallel Processing



### Memory Management



## Common Use Cases

### Classification Hyperparameter Tuning



### Regression Model Tuning



### Pipeline Optimization



## Integration Patterns

### With Hyperactive Experiments



### With OptCV Interface



## Best Practices

1. **Parameter space size**: Keep total combinations reasonable (<1000)
2. **Cross-validation**: Use appropriate CV strategy for your data
3. **Parallelization**: Use `backend_params` (e.g., `{"n_jobs": -1}` with joblib)
4. **Memory management**: Monitor memory usage with large grids
5. **Scoring metrics**: Choose appropriate metrics for your problem
6. **Reproducibility**: Set random seeds in estimators

## Limitations

- **Computational cost**: Scales multiplicatively with parameter options
- **Discrete parameters only**: Cannot optimize continuous ranges directly
- **No early stopping**: Evaluates all combinations regardless of results
- **Memory usage**: Can be memory-intensive for large grids

## References

- Scikit-learn GridSearchCV documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
