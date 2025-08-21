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



## Advanced Custom Experiment



## Machine Learning Custom Experiment



## Multi-Objective Custom Experiment



## Simulation-Based Experiment



## Experiment with External Data



## Best Practices for Custom Experiments

### Error Handling


### Parameter Validation


### Caching and Memoization


## Testing Custom Experiments



## Integration with Different Optimizers



## References

- Hyperactive base classes: BaseExperiment and BaseOptimizer documentation
- Python subprocess module: [https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html)
- Scikit-learn custom scorers: [https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions)
