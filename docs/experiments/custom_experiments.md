# Custom Experiments

## Introduction

Creating custom experiments in Hyperactive allows you to optimize any objective function or complex system. By inheriting from `BaseExperiment`, you can define domain-specific optimization problems that go beyond standard machine learning hyperparameter tuning.

## BaseExperiment Overview

All experiments inherit from `BaseExperiment`, which provides:
- **Parameter space definition**: Define what parameters to optimize
- **Objective evaluation**: Implement your evaluation logic
- **Metadata handling**: Return additional information about evaluations
- **Tag system**: Specify experiment properties and requirements

## Basic Custom Experiment

```python
--8<-- "experiments_custom_experiments_example.py"
```

## Advanced Custom Experiment

```python
--8<-- "experiments_custom_experiments_example_2.py"
```

## Machine Learning Custom Experiment

```python
--8<-- "experiments_custom_experiments_example_3.py"
```

## Multi-Objective Custom Experiment

```python
--8<-- "experiments_custom_experiments_example_4.py"
```

## Simulation-Based Experiment

```python
--8<-- "experiments_custom_experiments_example_5.py"
```

## Experiment with External Data

```python
--8<-- "experiments_custom_experiments_example_6.py"
```

## Best Practices for Custom Experiments

### Error Handling
```python
--8<-- "experiments_custom_experiments_example_7.py"
```

### Parameter Validation
```python
--8<-- "experiments_custom_experiments_example_8.py"
```

### Caching and Memoization
```python
--8<-- "experiments_custom_experiments_example_9.py"
```

## Testing Custom Experiments

```python
--8<-- "experiments_custom_experiments_example_10.py"
```

## Integration with Different Optimizers

```python
--8<-- "experiments_custom_experiments_example_11.py"
```

## References

- Hyperactive base classes: BaseExperiment and BaseOptimizer documentation
- Python subprocess module: [https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html)
- Scikit-learn custom scorers: [https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions)
