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

## Quick Start

```python
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load dataset
X, y = load_wine(return_X_y=True)

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10]
    },
    X=X, y=y
)

# Create optimizer
optimizer = BayesianOptimizer(experiment=experiment)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
```

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