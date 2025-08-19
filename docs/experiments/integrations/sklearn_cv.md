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
--8<-- "experiments_integrations_sklearn_cv_example.py"
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
--8<-- "experiments_integrations_sklearn_cv_example_2.py"
```

### Regression

```python
--8<-- "experiments_integrations_sklearn_cv_example_3.py"
```

## Advanced Usage

### Custom Cross-Validation Strategies

```python
--8<-- "experiments_integrations_sklearn_cv_example_4.py"
```

### Custom Scoring Functions

```python
--8<-- "experiments_integrations_sklearn_cv_example_5.py"
```

### Pipeline Optimization

```python
--8<-- "experiments_integrations_sklearn_cv_example_6.py"
```

### Early Stopping and Budget Management

```python
--8<-- "experiments_integrations_sklearn_cv_example_7.py"
```

## Common Patterns

### Multi-Algorithm Comparison

```python
--8<-- "experiments_integrations_sklearn_cv_example_8.py"
```

### Nested Cross-Validation

```python
--8<-- "experiments_integrations_sklearn_cv_example_9.py"
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
--8<-- "experiments_integrations_sklearn_cv_example_10.py"
```

### Imbalanced Datasets
```python
--8<-- "experiments_integrations_sklearn_cv_example_11.py"
```

### Time Series Data
```python
--8<-- "experiments_integrations_sklearn_cv_example_12.py"
```

## Integration with Other Hyperactive Components

### With OptCV Interface
```python
--8<-- "experiments_integrations_sklearn_cv_example_13.py"
```

### With Different Optimizers
```python
--8<-- "experiments_integrations_sklearn_cv_example_14.py"
```

## References

- Scikit-learn User Guide: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- Cross-validation documentation: [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
- Model evaluation: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)