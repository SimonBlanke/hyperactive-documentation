# OptCV Interface

## Introduction

OptCV provides a familiar sklearn-like interface for hyperparameter search while enabling the use of any Hyperactive optimizer. This bridge lets sklearn users leverage advanced optimization algorithms without changing their workflow significantly.

## Design Philosophy

OptCV aims for close compatibility with sklearn's search interfaces while providing:
- **Familiar API**: Similar methods and key attributes to GridSearchCV
- **Advanced optimizers**: Access to 25+ optimization algorithms  
- **Seamless integration**: Works with existing sklearn pipelines and workflows
- **Enhanced performance**: Often better results than grid/random search

## Class Signature



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

## Basic Usage

### Simple Classification Example



### Regression Example



## Advanced Usage

### Different Optimizers Comparison



### Pipeline Integration



### Custom Cross-Validation



## Integration with Existing Workflows

### Drop-in Replacement for GridSearchCV



### With sklearn Model Selection



## Attributes

After fitting, OptCV provides the following attributes:

### `best_params_`


### `best_score_`


### `best_estimator_`


Note: `cv_results_` is not provided by OptCV in v5.


## Methods (Same as GridSearchCV)

### `fit(X, y)`


### `predict(X)`


### `predict_proba(X)`


### `score(X, y)`


### `decision_function(X)`


## Performance Considerations

### Memory Usage


### Computational Efficiency


## Common Patterns

### Ensemble of Optimized Models



### Hyperparameter Analysis



## Best Practices

1. **Optimizer Selection**: Choose optimizers based on your parameter space characteristics
2. **Refit Setting**: Set `refit=True` for production models
3. **Cross-Validation**: Use appropriate CV strategy for your data type
4. **Scoring Metrics**: Select metrics that align with business objectives
5. **Memory Management**: Monitor memory usage with large datasets
6. **Reproducibility**: Set random seeds in both estimators and optimizers

## Migration from sklearn

### From GridSearchCV


### From RandomizedSearchCV


## References

- Scikit-learn GridSearchCV: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Scikit-learn Model Selection: [https://scikit-learn.org/stable/model_selection.html](https://scikit-learn.org/stable/model_selection.html)
