# Migration Guide: v4 to v5

This guide helps you migrate from Hyperactive v4 to v5. The v5 release introduces significant architectural changes that improve modularity and extensibility, but require updates to existing code.

## Overview of Changes

### Major Changes in v5

1. **No more `Hyperactive` class**: Individual optimizer classes are used directly
2. **Experiment-based approach**: Optimization problems are defined as experiment objects
3. **Backend-specific imports**: Optimizers are imported from specific backend modules
4. **New base class architecture**: Uses skbase framework for extensibility
5. **Enhanced integration layer**: Better support for ML frameworks

## Before You Start

### Check Your Dependencies

V5 has updated dependencies:

```bash
# Install v5
pip install hyperactive>=5.0.0

# Key new dependencies
pip install scikit-base
pip install gradient-free-optimizers>=1.2.4
```

## Basic API Migration

### v4 Pattern

Replace prior monolithic usage with experiment + optimizer composition.

### v5 Pattern

Example patterns with v5 components:

Sklearn cross‑validation + sklearn-style randomized search:

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import RandomSearchSk

X, y = load_iris(return_X_y=True)
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

opt = RandomSearchSk(
    param_distributions={"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01]},
    n_iter=20,
    experiment=exp,
)
best = opt.solve()
```

Optuna TPE with mixed parameter spaces:

```python
from hyperactive.opt.optuna import TPEOptimizer

space = {"C": (1e-2, 100), "kernel": ["rbf", "poly"], "gamma": (1e-6, 1e-1)}
opt = TPEOptimizer(param_space=space, n_trials=25, experiment=exp)
best = opt.solve()
```

## Migration Examples

### Example 1: Basic Hyperparameter Optimization

**v4 Code:**


**v5 Code:**
As above: combine `SklearnCvExperiment` with any optimizer (e.g., `RandomSearchSk`).

### Example 2: Custom Objective Function

**v4 Code:**


**v5 Code:**
Wrap a callable using `FunctionExperiment`:

```python
from hyperactive.experiment.func import FunctionExperiment
from hyperactive.opt.gfo import RandomSearch

def obj(x, y):
    return -(x**2 + y**2), {}

exp = FunctionExperiment(obj, parametrization="kwargs")
best = RandomSearch(experiment=exp).solve()
```

### Example 3: Multiple Search Runs

**v4 Code:**


**v5 Code:**


## Algorithm Name Mapping

Many algorithm names have changed between versions:

| v4 Name | v5 Import | v5 Class Name |
|---------|-----------|---------------|
| `"HillClimbing"` | `from hyperactive.opt.gfo import HillClimbing` | `HillClimbing` |
| `"BayesianOptimization"` | `from hyperactive.opt.gfo import BayesianOptimizer` | `BayesianOptimizer` |
| `"RandomSearch"` | `from hyperactive.opt.gfo import RandomSearch` | `RandomSearch` |
| `"GridSearch"` | `from hyperactive.opt.gfo import GridSearch` | `GridSearch` |
| `"SimulatedAnnealing"` | `from hyperactive.opt.gfo import SimulatedAnnealing` | `SimulatedAnnealing` |
| `"ParticleSwarm"` | `from hyperactive.opt.gfo import ParticleSwarmOptimizer` | `ParticleSwarmOptimizer` |
| `"GeneticAlgorithm"` | `from hyperactive.opt.gfo import GeneticAlgorithm` | `GeneticAlgorithm` |

### New Algorithms in v5

V5 adds many new algorithms not available in v4:



## Parameter Configuration

### v4 Parameter Passing



### v5 Parameter Passing



## Advanced Migration Patterns

### Memory and Warm Starting

**v4 Code:**


**v5 Code:**


### Parallel Processing

**v4 Code:**


**v5 Code:**
Use sklearn-style optimizers with backends:

```python
from hyperactive.opt import GridSearchSk as GridSearch

opt = GridSearch(
    param_grid={"C": [0.1, 1, 10]},
    backend="joblib",
    backend_params={"n_jobs": -1},
    experiment=exp,
)
best = opt.solve()
```

## Integration with ML Frameworks

### Scikit-learn Integration

**v4 Approach:**


**v5 Approach:**


## Common Migration Issues

### 1. Search Space Definition

**Problem:** v4 used various formats for search spaces.

**Solution:** v5 uses consistent dict format:


### 2. Result Access

**Problem:** Different result access patterns.

**Solution:**


### 3. Progress Monitoring

**Problem:** v5 doesn't have built-in progress bars.

**Solution:** Implement custom monitoring:


## Testing Your Migration

Create a simple test to verify your migration:



## Performance Considerations

V5 may have different performance characteristics:

1. **Startup time**: May be slightly higher due to skbase framework
2. **Memory usage**: Generally lower due to more efficient architecture  
3. **Evaluation speed**: Similar or faster for most use cases
4. **Extensibility**: Much better - easier to add custom components

## Getting Help

If you encounter issues during migration:

1. **Check examples**: Look at the example files in v5 repository
2. **Read documentation**: This documentation covers all new features
3. **Compare patterns**: Use this guide's before/after examples
4. **Community support**: Ask questions on GitHub issues

## Summary Checklist

- [ ] Update imports to use specific backend modules
- [ ] Convert objective functions to experiment classes
- [ ] Replace `Hyperactive()` with individual optimizer classes  
- [ ] Update algorithm names and parameter passing
- [ ] Test migrated code with simple examples
- [ ] Update any custom extensions to use new base classes
- [ ] Consider new v5 features like OptCV for sklearn integration

The migration requires some effort, but v5's improved architecture provides much better extensibility and cleaner interfaces for long-term use.
