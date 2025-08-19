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
--8<-- "experiments_integrations_opt_cv_example.py"
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
--8<-- "experiments_integrations_opt_cv_example_2.py"
```

### Regression Example

```python
--8<-- "experiments_integrations_opt_cv_example_3.py"
```

## Advanced Usage

### Different Optimizers Comparison

```python
--8<-- "experiments_integrations_opt_cv_example_4.py"
```

### Pipeline Integration

```python
--8<-- "experiments_integrations_opt_cv_example_5.py"
```

### Custom Cross-Validation

```python
--8<-- "experiments_integrations_opt_cv_example_6.py"
```

## Integration with Existing Workflows

### Drop-in Replacement for GridSearchCV

```python
--8<-- "experiments_integrations_opt_cv_example_7.py"
```

### With sklearn Model Selection

```python
--8<-- "experiments_integrations_opt_cv_example_8.py"
```

## Attributes (Same as GridSearchCV)

After fitting, OptCV provides the same attributes as sklearn's search objects:

### `best_params_`
```python
--8<-- "experiments_integrations_opt_cv_example_9.py"
```

### `best_score_`
```python
--8<-- "experiments_integrations_opt_cv_example_10.py"
```

### `best_estimator_`
```python
--8<-- "experiments_integrations_opt_cv_example_11.py"
```

### `cv_results_` (Limited)
```python
--8<-- "experiments_integrations_opt_cv_example_12.py"
```

## Methods (Same as GridSearchCV)

### `fit(X, y)`
```python
--8<-- "experiments_integrations_opt_cv_example_13.py"
```

### `predict(X)`
```python
--8<-- "experiments_integrations_opt_cv_example_14.py"
```

### `predict_proba(X)`
```python
--8<-- "experiments_integrations_opt_cv_example_15.py"
```

### `score(X, y)`
```python
--8<-- "experiments_integrations_opt_cv_example_16.py"
```

### `decision_function(X)`
```python
--8<-- "experiments_integrations_opt_cv_example_17.py"
```

## Performance Considerations

### Memory Usage
```python
--8<-- "experiments_integrations_opt_cv_example_18.py"
```

### Computational Efficiency
```python
--8<-- "experiments_integrations_opt_cv_example_19.py"
```

## Common Patterns

### Ensemble of Optimized Models

```python
--8<-- "experiments_integrations_opt_cv_example_20.py"
```

### Hyperparameter Analysis

```python
--8<-- "experiments_integrations_opt_cv_example_21.py"
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
--8<-- "experiments_integrations_opt_cv_example_22.py"
```

### From RandomizedSearchCV
```python
--8<-- "experiments_integrations_opt_cv_example_23.py"
```

## References

- Scikit-learn GridSearchCV: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Scikit-learn Model Selection: [https://scikit-learn.org/stable/model_selection.html](https://scikit-learn.org/stable/model_selection.html)