# OptCV Interface

## Introduction

OptCV provides a drop-in replacement for scikit-learn's GridSearchCV and RandomizedSearchCV, offering the same familiar interface while enabling the use of any Hyperactive optimizer. This bridge allows sklearn users to leverage advanced optimization algorithms without changing their existing workflow.

## Design Philosophy

OptCV maintains complete compatibility with sklearn's search interfaces while providing:
- **Familiar API**: Same methods and attributes as GridSearchCV
- **Advanced optimizers**: Access to 25+ optimization algorithms  
- **Seamless integration**: Works with existing sklearn pipelines and workflows
- **Enhanced performance**: Often better results than grid/random search

## Class Signature

```python
from hyperactive.integrations.sklearn import OptCV

opt_cv = OptCV(
    estimator,           # sklearn estimator
    optimizer,           # Hyperactive optimizer  
    cv=5,               # cross-validation strategy
    scoring=None,        # scoring function
    refit=True,         # refit best estimator
    n_jobs=1            # parallel jobs
)
```

## Parameters

### `estimator`
- **Type**: sklearn estimator
- **Description**: The machine learning model to optimize
- **Examples**: Any sklearn classifier, regressor, or pipeline

### `optimizer`
- **Type**: Hyperactive optimizer instance
- **Description**: Configured Hyperactive optimizer with experiment
- **Note**: Must be initialized with a compatible experiment

### `cv`
- **Type**: `int` or sklearn CV object
- **Default**: `5`
- **Description**: Cross-validation strategy

### `scoring`
- **Type**: `str`, callable, or `None`
- **Default**: `None` (uses estimator's default scorer)
- **Description**: Scoring function for evaluation

### `refit`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to refit the best estimator on full dataset

### `n_jobs`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of parallel jobs for cross-validation

## Basic Usage

### Simple Classification Example

```python
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load and prepare data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter search space
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10]
}

# Create experiment for the optimizer
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5
)

# Create OptCV with Bayesian optimization
opt_cv = OptCV(
    estimator=RandomForestClassifier(random_state=42),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=5,
    scoring='f1_weighted',
    refit=True
)

# Use like sklearn's GridSearchCV
opt_cv.fit(X_train, y_train)

# Access results (same as GridSearchCV)
print("Best parameters:", opt_cv.best_params_)
print("Best score:", opt_cv.best_score_)

# Make predictions
y_pred = opt_cv.predict(X_test)
test_score = opt_cv.score(X_test, y_test)
print("Test score:", test_score)
```

### Regression Example

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes

# Load regression data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression parameter space
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 0.9, 1.0]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)

# Create OptCV
opt_cv = OptCV(
    estimator=GradientBoostingRegressor(random_state=42),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=5,
    scoring='neg_mean_squared_error'
)

# Fit and evaluate
opt_cv.fit(X_train, y_train)
print(f"Best MSE: {-opt_cv.best_score_:.4f}")

# Predict and evaluate
predictions = opt_cv.predict(X_test)
from sklearn.metrics import mean_squared_error
test_mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {test_mse:.4f}")
```

## Advanced Usage

### Different Optimizers Comparison

```python
from hyperactive.opt.gfo import RandomSearch, ParticleSwarmOptimizer
from hyperactive.opt.optuna import TPEOptimizer

# Create base experiment
base_experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5
)

# Test different optimizers
optimizers = {
    'Bayesian': BayesianOptimizer(experiment=base_experiment),
    'Random': RandomSearch(experiment=base_experiment),
    'TPE': TPEOptimizer(experiment=base_experiment),
    'PSO': ParticleSwarmOptimizer(experiment=base_experiment, population=20)
}

results = {}
for name, optimizer in optimizers.items():
    # Create separate OptCV for each optimizer
    opt_cv = OptCV(
        estimator=RandomForestClassifier(random_state=42),
        optimizer=optimizer,
        cv=5,
        scoring='accuracy'
    )
    
    opt_cv.fit(X_train, y_train)
    test_score = opt_cv.score(X_test, y_test)
    
    results[name] = {
        'cv_score': opt_cv.best_score_,
        'test_score': test_score,
        'best_params': opt_cv.best_params_
    }
    
    print(f"{name}: CV={opt_cv.best_score_:.4f}, Test={test_score:.4f}")

# Find best optimizer
best_optimizer = max(results.items(), key=lambda x: x[1]['test_score'])
print(f"Best optimizer: {best_optimizer[0]}")
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif)),
    ('classifier', SVC(random_state=42))
])

# Pipeline parameter grid
param_grid = {
    'scaler__with_mean': [True, False],
    'selector__k': [5, 10, 15],
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
    'classifier__kernel': ['rbf', 'linear']
}

# Create experiment for pipeline
experiment = SklearnCvExperiment(
    estimator=pipeline,
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5
)

# Optimize pipeline with OptCV
opt_cv = OptCV(
    estimator=pipeline,
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=5,
    scoring='f1_weighted'
)

opt_cv.fit(X_train, y_train)
print("Best pipeline parameters:", opt_cv.best_params_)
print("Pipeline test score:", opt_cv.score(X_test, y_test))
```

### Custom Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, GroupKFold

# Stratified CV for imbalanced data
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

opt_cv = OptCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=stratified_cv,
    scoring='f1_weighted'
)

# Group CV for grouped data
# groups = [...]  # Group labels for each sample
group_cv = GroupKFold(n_splits=5)

opt_cv_groups = OptCV(
    estimator=RandomForestClassifier(random_state=42),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=group_cv,
    scoring='accuracy'
)

# Note: For GroupKFold, you'll need to pass groups to fit()
# opt_cv_groups.fit(X_train, y_train, groups=train_groups)
```

## Integration with Existing Workflows

### Drop-in Replacement for GridSearchCV

```python
# Original GridSearchCV code
from sklearn.model_selection import GridSearchCV

# old_grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )

# Replace with OptCV using advanced optimizer
opt_cv = OptCV(
    estimator=RandomForestClassifier(random_state=42),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # For CV parallelization
)

# Same interface
opt_cv.fit(X_train, y_train)
print("Best params:", opt_cv.best_params_)
print("Best score:", opt_cv.best_score_)
y_pred = opt_cv.predict(X_test)
```

### With sklearn Model Selection

```python
from sklearn.model_selection import cross_val_score, validation_curve

# Use OptCV in sklearn's validation tools
cv_scores = cross_val_score(opt_cv, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Validation curves work too (though optimizer choice is more important)
param_range = [0.1, 1, 10, 100]
train_scores, test_scores = validation_curve(
    opt_cv, X, y, 
    param_name='estimator__C',  # Note: access through estimator
    param_range=param_range,
    cv=5
)
```

## Attributes (Same as GridSearchCV)

After fitting, OptCV provides the same attributes as sklearn's search objects:

### `best_params_`
```python
# Dictionary of best parameters found
print("Best parameters:", opt_cv.best_params_)
```

### `best_score_`
```python
# Best cross-validation score achieved
print("Best CV score:", opt_cv.best_score_)
```

### `best_estimator_`
```python
# Estimator fitted with best parameters (if refit=True)
best_model = opt_cv.best_estimator_
predictions = best_model.predict(X_test)
```

### `cv_results_` (Limited)
```python
# Note: OptCV provides limited cv_results_ compared to GridSearchCV
# since Hyperactive optimizers don't always track all evaluations
```

## Methods (Same as GridSearchCV)

### `fit(X, y)`
```python
# Fit the optimizer and find best parameters
opt_cv.fit(X_train, y_train)
```

### `predict(X)`
```python
# Predict using the best estimator (requires refit=True)
predictions = opt_cv.predict(X_test)
```

### `predict_proba(X)`
```python
# Predict probabilities (for classifiers that support it)
if hasattr(opt_cv, 'predict_proba'):
    probabilities = opt_cv.predict_proba(X_test)
```

### `score(X, y)`
```python
# Score using the best estimator
test_score = opt_cv.score(X_test, y_test)
```

### `decision_function(X)`
```python
# Decision function (for applicable estimators)
if hasattr(opt_cv, 'decision_function'):
    decisions = opt_cv.decision_function(X_test)
```

## Performance Considerations

### Memory Usage
```python
# OptCV can be more memory efficient than GridSearchCV
# since it doesn't store all parameter combinations

# For memory-constrained environments:
opt_cv = OptCV(
    estimator=estimator,
    optimizer=optimizer,
    cv=3,     # Fewer CV folds
    n_jobs=1  # Sequential processing
)
```

### Computational Efficiency
```python
# Bayesian optimization often finds better solutions with fewer evaluations
from time import time

start_time = time()
opt_cv.fit(X_train, y_train)
optimization_time = time() - start_time

print(f"Optimization completed in {optimization_time:.2f} seconds")
print(f"Best score: {opt_cv.best_score_:.4f}")
```

## Common Patterns

### Ensemble of Optimized Models

```python
# Train multiple models with different optimizers
models = {}
optimizers_config = {
    'bayes': BayesianOptimizer(experiment=experiment1),
    'tpe': TPEOptimizer(experiment=experiment2),
    'pso': ParticleSwarmOptimizer(experiment=experiment3, population=20)
}

for name, optimizer in optimizers_config.items():
    opt_cv = OptCV(
        estimator=RandomForestClassifier(random_state=42),
        optimizer=optimizer,
        cv=5
    )
    opt_cv.fit(X_train, y_train)
    models[name] = opt_cv.best_estimator_

# Create ensemble predictions
ensemble_predictions = []
for name, model in models.items():
    pred = model.predict_proba(X_test)
    ensemble_predictions.append(pred)

# Average predictions
import numpy as np
avg_predictions = np.mean(ensemble_predictions, axis=0)
final_predictions = np.argmax(avg_predictions, axis=1)
```

### Hyperparameter Analysis

```python
# Analyze parameter importance across different optimizers
param_importance = {}

for name, optimizer in optimizers.items():
    opt_cv = OptCV(estimator=estimator, optimizer=optimizer, cv=5)
    opt_cv.fit(X_train, y_train)
    
    for param, value in opt_cv.best_params_.items():
        if param not in param_importance:
            param_importance[param] = []
        param_importance[param].append(value)

# Analyze parameter stability
for param, values in param_importance.items():
    print(f"{param}: {set(values)}")  # Show unique values found
```

## Best Practices

1. **Optimizer Selection**: Choose optimizers based on your parameter space characteristics
2. **Refit Setting**: Set `refit=True` for production models
3. **Cross-Validation**: Use appropriate CV strategy for your data type
4. **Scoring Metrics**: Select metrics that align with business objectives
5. **Memory Management**: Monitor memory usage with large datasets
6. **Reproducibility**: Set random seeds in both estimators and optimizers

## Migration from sklearn

### From GridSearchCV
```python
# Before
grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy')

# After  
opt_cv = OptCV(estimator, BayesianOptimizer(experiment), cv=5, scoring='accuracy')
```

### From RandomizedSearchCV
```python
# Before
random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=100, cv=5)

# After
opt_cv = OptCV(estimator, TPEOptimizer(experiment), cv=5)
```

## References

- Scikit-learn GridSearchCV: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Scikit-learn Model Selection: [https://scikit-learn.org/stable/model_selection.html](https://scikit-learn.org/stable/model_selection.html)