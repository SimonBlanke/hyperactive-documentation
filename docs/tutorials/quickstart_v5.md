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



## Machine Learning Hyperparameter Optimization

The most common use case is optimizing ML model hyperparameters:



## Using Different Optimization Algorithms

Hyperactive v5 provides 25+ optimization algorithms. Here's how to try different ones:



## Sklearn-Compatible Interface

If you prefer the familiar scikit-learn interface:



## Advanced Parameter Spaces

For more complex parameter spaces with different data types:



## Working with Different Backends

Hyperactive supports multiple optimization backends:

### Optuna Backend



### Scikit-learn Backend



## Custom Optimization Problems

Create your own optimization experiment:



## Performance Tips

### 1. Choose the Right Algorithm

- **Random Search**: Quick baseline, good for high-dimensional spaces
- **Bayesian Optimization**: Sample-efficient, great for expensive evaluations
- **Particle Swarm**: Good for continuous spaces, handles multi-modal functions
- **Grid Search**: Systematic, interpretable, good for discrete spaces

### 2. Algorithm-Specific Tips



### 3. Cross-Validation Strategy



### 4. Parallel Processing



## Common Patterns

### Comparing Multiple Models



### Early Stopping for Time-Limited Optimization



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