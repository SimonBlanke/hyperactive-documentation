# RandomSearchSk

## Introduction

`RandomSearchSk` randomly samples parameter combinations from specified distributions or lists using `sklearn.model_selection.ParameterSampler`, and evaluates them through a Hyperactive `Experiment` (e.g., `SklearnCvExperiment`).

## About the Implementation

- Uses `ParameterSampler` to draw candidate configurations
- Uses the provided `experiment` to evaluate each candidate (e.g., CV via `SklearnCvExperiment`)
- Parallelism controlled via Hyperactive backends (`backend`, `backend_params`)

## Parameters

### `param_distributions`
- **Type**: `dict[str, list | numpy.ndarray | scipy.stats.rv_frozen]`
- **Description**: Distributions or lists to sample from

### `n_iter`
- **Type**: `int`, default `10`
- **Description**: Number of sampled configurations

### `random_state`
- **Type**: `int | None`
- **Description**: Random seed for sampling

### `error_score`
- **Type**: `float`, default `np.nan`
- **Description**: Score to assign if evaluation raises

### `backend`
- **Type**: `{"None","loky","multiprocessing","threading","joblib","dask","ray"}`
- **Description**: Parallel backend used by Hyperactive

### `backend_params`
- **Type**: `dict` or `None`
- **Description**: Backend configuration

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: Experiment used to evaluate candidates (e.g., `SklearnCvExperiment`)

## Usage Example

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import RandomSearchSk

X, y = load_iris(return_X_y=True)
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

opt = RandomSearchSk(
    param_distributions={
        "C": [0.01, 0.1, 1, 10],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    n_iter=20,
    backend="threading",
    backend_params={"n_jobs": 2},
    experiment=exp,
)
best_params = opt.solve()
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



### Continuous Distributions



### Custom Distributions



## Advanced Usage

### Budget Management

Control evaluations via `n_iter`; use `backend_params` to scale out across cores.

### Reproducible Results

Set `random_state` on the optimizer and estimators.

### Progressive Search



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



### Memory Management



### Search Space Design



## Common Use Cases

### Neural Network Hyperparameters



### Ensemble Method Tuning



### SVM Optimization



## Integration with Experiment Pipeline



## Best Practices

1. **Distribution choice**: Use log-uniform for parameters spanning orders of magnitude
2. **Sample size**: Use n_iter ≥ 10 × number of parameters as a rule of thumb
3. **Random seeds**: Set random_state for reproducible results
4. **Cross-validation**: Choose appropriate CV strategy for your data
5. **Parallel processing**: Use `backend_params` (e.g., `{"n_jobs": -1}` with joblib)
6. **Progressive refinement**: Start broad, then narrow down promising regions

## References

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization.
- Scikit-learn RandomizedSearchCV documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- scipy.stats distributions: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
