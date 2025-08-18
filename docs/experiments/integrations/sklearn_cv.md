# Scikit-learn CV Experiment

## Introduction

The SklearnCvExperiment class provides seamless integration between Hyperactive optimizers and scikit-learn estimators with cross-validation. This experiment type handles the complexity of hyperparameter optimization with proper cross-validation, making it the go-to choice for most machine learning optimization tasks.

## Core Functionality

SklearnCvExperiment automatically:
- Manages cross-validation splits
- Handles parameter validation and conversion
- Supports all sklearn estimators and pipelines
- Provides comprehensive scoring options
- Enables parallel cross-validation execution

## Class Signature

```python
from hyperactive.experiment.integrations import SklearnCvExperiment

experiment = SklearnCvExperiment(
    estimator,           # sklearn estimator
    param_grid,          # parameter search space
    X, y,               # training data
    cv=5,               # cross-validation strategy
    scoring='accuracy',  # scoring metric
    n_jobs=1            # parallel jobs for CV
)
```

## Parameters

### `estimator`
- **Type**: sklearn estimator or pipeline
- **Description**: The machine learning model to optimize
- **Examples**: Any sklearn estimator (classifiers, regressors, clusterers)

### `param_grid`
- **Type**: `dict`
- **Description**: Dictionary defining the hyperparameter search space
- **Format**: `{"param_name": [value1, value2, ...]}`

### `X, y`
- **Type**: array-like
- **Description**: Training features and target values
- **Note**: Must be compatible with the chosen estimator

### `cv`
- **Type**: `int` or sklearn CV object
- **Default**: `5`
- **Description**: Cross-validation strategy
- **Options**: Integer (K-fold), or sklearn CV objects

### `scoring`
- **Type**: `str` or callable
- **Default**: Depends on estimator type
- **Description**: Scoring function for evaluation
- **Options**: sklearn scoring strings or custom scorer functions

### `n_jobs`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of parallel jobs for cross-validation
- **Options**: `1` (sequential), `-1` (all cores), or specific number

## Basic Usage Examples

### Classification

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import BayesianOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load and split data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter search space
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring='f1_weighted'  # Good for potentially imbalanced data
)

# Optimize
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Best CV score:", experiment.score(best_params)[0])

# Train final model with best parameters
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print("Test score:", test_score)
```

### Regression

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, make_scorer

# Load regression dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define search space for regression
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.8, 0.9, 1.0]
}

# Create regression experiment
experiment = SklearnCvExperiment(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring='neg_mean_squared_error',  # Standard regression metric
    n_jobs=-1  # Use all available cores
)

# Optimize
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Evaluate
final_model = GradientBoostingRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse:.4f}")
```

## Advanced Usage

### Custom Cross-Validation Strategies

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold
import numpy as np

# Stratified K-Fold for imbalanced datasets
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=stratified_cv,
    scoring='f1_weighted'
)

# Time Series Cross-Validation
ts_cv = TimeSeriesSplit(n_splits=5)

experiment_ts = SklearnCvExperiment(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    X=X_time_series, y=y_time_series,  # Time series data
    cv=ts_cv,
    scoring='neg_mean_absolute_error'
)

# Group K-Fold for grouped data
groups = np.array([...])  # Group labels for each sample
group_cv = GroupKFold(n_splits=5)

experiment_group = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_grouped, y=y_grouped,
    cv=group_cv,
    scoring='accuracy'
)
```

### Custom Scoring Functions

```python
from sklearn.metrics import make_scorer, precision_recall_fscore_support

# Custom scoring function
def custom_f1_score(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    return f1.mean()

custom_scorer = make_scorer(custom_f1_score)

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring=custom_scorer
)

# Multiple metrics evaluation (for analysis, not optimization)
from sklearn.metrics import classification_report

def evaluate_model_comprehensively(estimator, X_test, y_test):
    predictions = estimator.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    return report

# After optimization
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
comprehensive_results = evaluate_model_comprehensively(final_model, X_test, y_test)
```

### Pipeline Optimization

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SelectKBest, f_classif
from sklearn.svm import SVC

# Create preprocessing and model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', SVC(random_state=42))
])

# Define parameter grid for entire pipeline
param_grid = {
    # Scaler parameters
    'scaler__with_mean': [True, False],
    'scaler__with_std': [True, False],
    
    # Feature selection parameters
    'feature_selection__k': [5, 10, 15, 20],
    
    # Classifier parameters
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'classifier__kernel': ['rbf', 'linear']
}

# Create experiment with pipeline
experiment = SklearnCvExperiment(
    estimator=pipeline,
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

# Optimize pipeline
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best pipeline parameters:", best_params)
```

### Early Stopping and Budget Management

```python
# For algorithms that support early stopping
from sklearn.ensemble import GradientBoostingClassifier

# Include validation_fraction and n_iter_no_change in search
param_grid = {
    "n_estimators": [100, 200, 500, 1000],  # Large values for early stopping
    "learning_rate": [0.01, 0.05, 0.1],
    "validation_fraction": [0.1, 0.2],
    "n_iter_no_change": [5, 10, 15]
}

experiment = SklearnCvExperiment(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=3,  # Fewer folds due to early stopping
    scoring='f1_weighted'
)
```

## Common Patterns

### Multi-Algorithm Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Define algorithms and their search spaces
algorithms = {
    'Random Forest': {
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Gradient Boosting': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    'SVM': {
        'estimator': SVC(random_state=42),
        'param_grid': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
    }
}

# Compare algorithms
results = {}
for name, config in algorithms.items():
    experiment = SklearnCvExperiment(
        estimator=config['estimator'],
        param_grid=config['param_grid'],
        X=X_train, y=y_train,
        cv=5,
        scoring='f1_weighted'
    )
    
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    
    results[name] = {
        'best_params': best_params,
        'best_score': best_score
    }
    
    print(f"{name}: {best_score:.4f}")

# Find best algorithm
best_algorithm = max(results.items(), key=lambda x: x[1]['best_score'])
print(f"Best algorithm: {best_algorithm[0]} with score {best_algorithm[1]['best_score']:.4f}")
```

### Nested Cross-Validation

```python
# Proper evaluation with nested CV (conceptual - requires custom implementation)
from sklearn.model_selection import cross_val_score

def nested_cv_evaluation(estimator, param_grid, X, y, inner_cv=5, outer_cv=3):
    """
    Perform nested cross-validation for unbiased performance estimation
    """
    outer_scores = []
    
    # Outer CV loop
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in outer_cv_splitter.split(X, y):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Inner optimization
        experiment = SklearnCvExperiment(
            estimator=estimator,
            param_grid=param_grid,
            X=X_train_outer, y=y_train_outer,
            cv=inner_cv,
            scoring='f1_weighted'
        )
        
        optimizer = BayesianOptimizer(experiment=experiment)
        best_params = optimizer.solve()
        
        # Evaluate on outer test set
        final_model = estimator.__class__(**best_params)
        final_model.fit(X_train_outer, y_train_outer)
        outer_score = final_model.score(X_test_outer, y_test_outer)
        outer_scores.append(outer_score)
    
    return np.mean(outer_scores), np.std(outer_scores)

# Use nested CV
mean_score, std_score = nested_cv_evaluation(
    RandomForestClassifier(random_state=42),
    param_grid,
    X, y
)
print(f"Nested CV Score: {mean_score:.4f} ± {std_score:.4f}")
```

## Best Practices

1. **Data Splitting**: Always use proper train/validation/test splits
2. **CV Strategy**: Choose appropriate cross-validation for your data type
3. **Scoring Metrics**: Select metrics that align with your business objective
4. **Parameter Ranges**: Define reasonable parameter ranges based on domain knowledge
5. **Computational Resources**: Use n_jobs for parallel processing when available
6. **Reproducibility**: Set random seeds in estimators and CV splitters
7. **Evaluation**: Use nested CV for unbiased performance estimation

## Common Issues and Solutions

### Memory Issues
```python
# Reduce memory usage
experiment = SklearnCvExperiment(
    estimator=estimator,
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=3,     # Fewer folds
    n_jobs=2  # Fewer parallel jobs
)
```

### Imbalanced Datasets
```python
# Use stratified CV and appropriate metrics
from sklearn.model_selection import StratifiedKFold

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted'  # Or 'roc_auc' for binary classification
)
```

### Time Series Data
```python
# Use time series CV
from sklearn.model_selection import TimeSeriesSplit

experiment = SklearnCvExperiment(
    estimator=estimator,
    param_grid=param_grid,
    X=X_time_series, y=y_time_series,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_mean_absolute_error'
)
```

## Integration with Other Hyperactive Components

### With OptCV Interface
```python
from hyperactive.integrations.sklearn import OptCV

# Use SklearnCvExperiment through OptCV
opt_cv = OptCV(
    estimator=RandomForestClassifier(random_state=42),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=5
)

opt_cv.fit(X_train, y_train)
```

### With Different Optimizers
```python
# Compare different optimizers on the same experiment
from hyperactive.opt.gfo import RandomSearch, ParticleSwarmOptimizer
from hyperactive.opt.optuna import TPEOptimizer

optimizers = [
    BayesianOptimizer(experiment=experiment),
    RandomSearch(experiment=experiment),
    TPEOptimizer(experiment=experiment),
    ParticleSwarmOptimizer(experiment=experiment, population=20)
]

for optimizer in optimizers:
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    print(f"{optimizer.__class__.__name__}: {score:.4f}")
```

## References

- Scikit-learn User Guide: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- Cross-validation documentation: [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
- Model evaluation: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)