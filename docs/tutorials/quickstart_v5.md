# Quick Start Guide

This guide will get you up and running with Hyperactive quickly, covering the most common use cases.

## Installation

```bash
pip install hyperactive
```

## Basic Concepts

Hyperactive is built around two main concepts:

1. **Experiments**: Define your optimization problem
2. **Optimizers**: Choose how to solve the problem

## Your First Optimization

Let's start with a simple mathematical optimization problem:

```python
--8<-- "tutorials_quickstart_v5_example.py"
```

## Machine Learning Hyperparameter Optimization

The most common use case is optimizing ML model hyperparameters:

```python
--8<-- "tutorials_quickstart_v5_example_2.py"
```

## Using Different Optimization Algorithms

Hyperactive v5 provides 25+ optimization algorithms. Here's how to try different ones:

```python
--8<-- "tutorials_quickstart_v5_example_3.py"
```

## Sklearn-Compatible Interface

If you prefer the familiar scikit-learn interface:

```python
--8<-- "tutorials_quickstart_v5_example_4.py"
```

## Advanced Parameter Spaces

For more complex parameter spaces with different data types:

```python
--8<-- "tutorials_quickstart_v5_example_5.py"
```

## Working with Different Backends

Hyperactive supports multiple optimization backends:

### Optuna Backend

```python
--8<-- "tutorials_quickstart_v5_example_6.py"
```

### Scikit-learn Backend

```python
--8<-- "tutorials_quickstart_v5_example_7.py"
```

## Custom Optimization Problems

Create your own optimization experiment:

```python
--8<-- "tutorials_quickstart_v5_example_8.py"
```

## Performance Tips

### 1. Choose the Right Algorithm

- **Random Search**: Quick baseline, good for high-dimensional spaces
- **Bayesian Optimization**: Sample-efficient, great for expensive evaluations
- **Particle Swarm**: Good for continuous spaces, handles multi-modal functions
- **Grid Search**: Systematic, interpretable, good for discrete spaces

### 2. Algorithm-Specific Tips

```python
--8<-- "tutorials_quickstart_v5_example_9.py"
```

### 3. Cross-Validation Strategy

```python
--8<-- "tutorials_quickstart_v5_example_10.py"
```

### 4. Parallel Processing

```python
--8<-- "tutorials_quickstart_v5_example_11.py"
```

## Common Patterns

### Comparing Multiple Models

```python
--8<-- "tutorials_quickstart_v5_example_12.py"
```

### Early Stopping for Time-Limited Optimization

```python
--8<-- "tutorials_quickstart_v5_example_13.py"
```

## Next Steps

1. **Explore more algorithms**: Try different optimizers for your specific problem
2. **Custom experiments**: Create domain-specific optimization problems
3. **Advanced features**: Multi-objective optimization, constraint handling
4. **Integration**: Use with other ML frameworks like XGBoost, LightGBM
5. **Scaling**: Learn about distributed optimization for large-scale problems

For more advanced topics, check out:
- [Advanced Optimization Strategies](optimization_strategies.md)
- [Creating Custom Experiments](custom_experiments.md)
- [Multi-Objective Optimization](multi_objective.md)
- [Performance Optimization](performance_tips.md)