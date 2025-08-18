# Grid Search SK

## Introduction

Grid Search SK provides direct integration with scikit-learn's GridSearchCV, offering the familiar sklearn interface while working within Hyperactive's architecture. This optimizer systematically searches through all combinations of specified parameter values.

## About the Implementation

This optimizer leverages sklearn's mature and optimized GridSearchCV implementation, providing:

- **Native sklearn integration**: Uses sklearn's GridSearchCV under the hood
- **Parallel execution**: Supports sklearn's n_jobs parameter for parallelization
- **Cross-validation**: Built-in cross-validation support
- **Familiar interface**: Standard sklearn parameter handling

## Parameters

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: The experiment object defining the optimization problem

### `n_jobs`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of parallel jobs for cross-validation. -1 uses all processors.

### `cv`
- **Type**: `int` or cross-validation generator
- **Default**: `3`
- **Description**: Cross-validation strategy

### `verbose`
- **Type**: `int`
- **Default**: `0`
- **Description**: Verbosity level for sklearn's GridSearchCV

## Usage Example

```python
from hyperactive.opt import GridSearchSk
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Define search space
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01],
    "kernel": ["rbf", "linear", "poly"]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=SVC(),
    param_grid=param_grid,
    X=X, y=y,
    cv=5
)

# Create grid search optimizer
optimizer = GridSearchSk(
    experiment=experiment,
    n_jobs=-1,  # Use all available cores
    cv=5,
    verbose=1
)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
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

| Method | Backend | Parallelization | Features | Use Case |
|--------|---------|----------------|----------|----------|
| GridSearchSk | sklearn | Excellent | CV, scoring | sklearn workflows |
| GridSearch (GFO) | GFO | Limited | Custom logic | General optimization |
| GridOptimizer (Optuna) | Optuna | Good | Optuna ecosystem | Optuna workflows |

## Advanced Usage

### Custom Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Stratified K-Fold for imbalanced datasets
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

optimizer = GridSearchSk(
    experiment=experiment,
    cv=cv_strategy,
    n_jobs=-1
)

# Time Series Split for temporal data
cv_strategy = TimeSeriesSplit(n_splits=5)

optimizer = GridSearchSk(
    experiment=experiment,
    cv=cv_strategy
)
```

### Different Scoring Metrics

```python
# Multiple scoring metrics
experiment = SklearnCvExperiment(
    estimator=SVC(),
    param_grid=param_grid,
    X=X, y=y,
    scoring='f1_weighted',  # F1 score for imbalanced data
    cv=5
)

# Custom scoring function
from sklearn.metrics import make_scorer, balanced_accuracy_score

custom_scorer = make_scorer(balanced_accuracy_score)

experiment = SklearnCvExperiment(
    estimator=SVC(),
    param_grid=param_grid,
    X=X, y=y,
    scoring=custom_scorer,
    cv=5
)
```

### Verbose Output

```python
# Different verbosity levels
optimizer = GridSearchSk(
    experiment=experiment,
    verbose=0  # Silent
)

optimizer = GridSearchSk(
    experiment=experiment,
    verbose=1  # Progress information
)

optimizer = GridSearchSk(
    experiment=experiment,
    verbose=2  # Detailed information for each fold
)
```

## Performance Optimization

### Parallel Processing

```python
# Use all available cores
optimizer = GridSearchSk(
    experiment=experiment,
    n_jobs=-1
)

# Use specific number of cores
optimizer = GridSearchSk(
    experiment=experiment,
    n_jobs=4  # Use 4 cores
)

# Sequential processing (for memory-constrained environments)
optimizer = GridSearchSk(
    experiment=experiment,
    n_jobs=1
)
```

### Memory Management

```python
# For large datasets or many parameter combinations
# Consider reducing CV folds or using simpler models first

# Lighter cross-validation
optimizer = GridSearchSk(
    experiment=experiment,
    cv=3,  # Fewer folds
    n_jobs=2  # Fewer parallel jobs to reduce memory usage
)
```

## Common Use Cases

### Classification Hyperparameter Tuning

```python
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    scoring='accuracy',
    cv=5
)
```

### Regression Model Tuning

```python
from sklearn.ensemble import GradientBoostingRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}

experiment = SklearnCvExperiment(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    scoring='neg_mean_squared_error',
    cv=5
)
```

### Pipeline Optimization

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Parameters for pipeline components
param_grid = {
    'scaler__with_mean': [True, False],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'linear']
}

experiment = SklearnCvExperiment(
    estimator=pipeline,
    param_grid=param_grid,
    X=X, y=y,
    cv=5
)
```

## Integration Patterns

### With Hyperactive Experiments

```python
# Use with different experiment types
from hyperactive.experiment.bench import Sphere

# Custom experiment
experiment = Sphere(dimensions=2, bounds=(-5, 5))

# Note: GridSearchSk works best with SklearnCvExperiment
# For other experiments, consider GFO GridSearch instead
```

### With OptCV Interface

```python
from hyperactive.integrations.sklearn import OptCV

# GridSearchSk through OptCV interface
opt_cv = OptCV(
    estimator=SVC(),
    optimizer=GridSearchSk(experiment=experiment),
    cv=5
)

opt_cv.fit(X, y)
predictions = opt_cv.predict(X_test)
```

## Best Practices

1. **Parameter space size**: Keep total combinations reasonable (<1000)
2. **Cross-validation**: Use appropriate CV strategy for your data
3. **Parallelization**: Leverage n_jobs for faster execution
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