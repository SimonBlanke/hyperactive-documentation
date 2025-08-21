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



### Different Scoring Metrics



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