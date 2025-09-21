# Hyperactive

## Introduction

Hyperactive is a comprehensive optimization and data collection toolbox for convenient and fast prototyping of computationally expensive models. It features a redesigned architecture based on the skbase framework, providing enhanced modularity, better extensibility, and an intuitive API.

## Key Features

- **25+ Optimization Algorithms** across multiple backends (GFO, Optuna, Scikit-learn)
- **Modular Architecture** with base classes for optimizers and experiments
- **Multiple Backend Support** - Choose the best optimization engine for your needs
- **Framework Integrations** - Native support for scikit-learn, sktime, and more
- **Extensible Design** - Easy to add custom optimizers and experiments
- **Tag-based Metadata System** - Rich algorithm properties and capabilities

## Installation

The most recent version of Hyperactive is available on PyPi:

```console
pip install hyperactive
```

Optional integrations and backends:

- Optuna adapters: `pip install optuna<5`
- Sktime integrations: `pip install hyperactive[sktime-integration]`
- Parallel backends (as needed): `pip install joblib dask ray`
  - Note: ray support depends on your Python version; see `pyproject.toml` constraints.

## Quick Start
Get productive in minutes with a minimal example.

1) Optimize a built-in benchmark (no data required)

```python
from hyperactive.experiment.bench import Ackley
from hyperactive.opt.gfo import RandomSearch

# Define the objective (2D Ackley benchmark)
exp = Ackley(d=2)

# Pick an optimizer and run
opt = RandomSearch(experiment=exp)
best_params = opt.solve()
print("Best parameters:", best_params)
```

2) Optimize a scikit-learn model with cross‑validation

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import GridSearchSk

X, y = load_iris(return_X_y=True)

# Wrap your estimator + data in an experiment
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

# Search the hyperparameter grid and get the best params
opt = GridSearchSk(param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                   experiment=exp)
best_params = opt.solve()
print("Best parameters:", best_params)
```

Next steps:
- Check the full Quickstart: `tutorials/quickstart_v5.md`
- Explore optimizers: `api/optimizers.md`
- Learn experiments and integrations: `api/experiments.md`


## Key Features

### Complete Architecture Design
- **Base Classes**: `BaseOptimizer` and `BaseExperiment` classes using skbase framework
- **Plugin Architecture**: Optimizers and experiments are extensible plugins
- **Tag System**: Rich metadata system for algorithm properties and capabilities

### Multiple Optimization Backend Support
- **Gradient-Free-Optimizers**: 16+ algorithms including Bayesian optimization, genetic algorithms, and more
- **Optuna Integration**: 8+ algorithms including TPE, CMA-ES, and multi-objective optimization
- **Scikit-learn Backend**: Direct integration with GridSearchCV and RandomizedSearchCV

### Integration Layer
- **OptCV**: Scikit-learn compatible hyperparameter optimization
- **Experiment Classes**: Specialized experiments for different ML frameworks
- **Built-in Benchmarks**: Standard optimization test functions (Ackley, Sphere, etc.)

### Developer Experience
- **Type Hints**: Full type annotation support
- **Comprehensive Documentation**: Complete API documentation and examples
- **Extension Templates**: Easy-to-use templates for custom optimizers and experiments

## Migration Guide

For users migrating from previous versions, Hyperactive introduces some changes in the API:

- **Individual optimizer classes**: Use specific optimizer classes directly
- **Experiment-based approach**: Define your optimization problem as an experiment
- **Backend-specific imports**: Import optimizers from their specific backend modules
- **Plugin architecture**: Extend functionality through base classes

For detailed migration guidance, see the [Migration Guide](tutorials/migration_guide.md).

## License

[![LICENSE](https://img.shields.io/github/license/SimonBlanke/Hyperactive?style=for-the-badge)](https://github.com/SimonBlanke/Hyperactive/blob/master/LICENSE)
