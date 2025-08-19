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
--8<-- "optimizers_sklearn_random_search_sk_example.py"
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
--8<-- "optimizers_sklearn_random_search_sk_example_2.py"
```

### Continuous Distributions

```python
--8<-- "optimizers_sklearn_random_search_sk_example_3.py"
```

### Custom Distributions

```python
--8<-- "optimizers_sklearn_random_search_sk_example_4.py"
```

## Advanced Usage

### Budget Management

```python
--8<-- "optimizers_sklearn_random_search_sk_example_5.py"
```

### Reproducible Results

```python
--8<-- "optimizers_sklearn_random_search_sk_example_6.py"
```

### Progressive Search

```python
--8<-- "optimizers_sklearn_random_search_sk_example_7.py"
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
--8<-- "optimizers_sklearn_random_search_sk_example_8.py"
```

### Memory Management

```python
--8<-- "optimizers_sklearn_random_search_sk_example_9.py"
```

### Search Space Design

```python
--8<-- "optimizers_sklearn_random_search_sk_example_10.py"
```

## Common Use Cases

### Neural Network Hyperparameters

```python
--8<-- "optimizers_sklearn_random_search_sk_example_11.py"
```

### Ensemble Method Tuning

```python
--8<-- "optimizers_sklearn_random_search_sk_example_12.py"
```

### SVM Optimization

```python
--8<-- "optimizers_sklearn_random_search_sk_example_13.py"
```

## Integration with Experiment Pipeline

```python
--8<-- "optimizers_sklearn_random_search_sk_example_14.py"
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