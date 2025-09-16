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

Let's start with a simple function optimization using a callable:

```python
from hyperactive.experiment.func import FunctionExperiment
from hyperactive.opt.gfo import RandomSearch

def parabola(x):
    return -(x - 2.0) ** 2, {}

exp = FunctionExperiment(parabola, parametrization="dict")
opt = RandomSearch(experiment=exp)
best_params = opt.solve()
print(best_params)
```



## Machine Learning Hyperparameter Optimization

Optimize an sklearn model with cross‑validation:

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import RandomSearchSk

X, y = load_iris(return_X_y=True)
exp = SklearnCvExperiment(estimator=SVC(), X=X, y=y)

search = RandomSearchSk(
    param_distributions={
        "C": [0.01, 0.1, 1, 10],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    n_iter=20,
    experiment=exp,
)
best_params = search.solve()
```



## Using Different Optimization Algorithms

Hyperactive v5 provides 25+ algorithms. Swap the optimizer while keeping the experiment:

```python
from hyperactive.opt.gfo import HillClimbing, SimulatedAnnealing

opt = HillClimbing(experiment=exp)  # local search
best_local = opt.solve()

opt = SimulatedAnnealing(experiment=exp)  # global search
best_global = opt.solve()
```



## Sklearn-Compatible Interface

If you prefer the familiar scikit-learn interface:

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from hyperactive.integrations import OptCV
from hyperactive.opt import GridSearchSk as GridSearch

X, y = load_iris(return_X_y=True)
optimizer = GridSearch(param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]})

svc = OptCV(SVC(), optimizer)
svc.fit(X, y)
print(svc.best_params_)
```



## Advanced Parameter Spaces

For complex parameter spaces (continuous + categorical) using Optuna:

```python
from hyperactive.opt.optuna import TPEOptimizer
space = {
    "C": (1e-2, 1e2),
    "gamma": (1e-6, 1e1),
    "kernel": ["rbf", "poly"],
}
opt = TPEOptimizer(param_space=space, n_trials=25, experiment=exp)
best = opt.solve()
```



## Working with Different Backends

Hyperactive supports multiple parallel backends on sklearn-style searches:

### Optuna Backend

Optuna-based optimizers are configured via their own parameters (no `backend`): see TPE/GP docs.

### Scikit-learn Backend

Sklearn-style searches accept `backend` and `backend_params`:

```python
from hyperactive.opt import GridSearchSk as GridSearch

grid = GridSearch(
    param_grid={"C": [0.1, 1, 10]},
    backend="joblib",
    backend_params={"n_jobs": -1},
    experiment=exp,
)
best = grid.solve()
```

## Custom Optimization Problems

Create your own optimization experiment:

```python
from hyperactive.experiment import BaseExperiment

class MyExp(BaseExperiment):
    def _paramnames(self):
        return ["x", "y"]
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        return -(x**2 + y**2), {}

from hyperactive.opt.gfo import RandomSearch
exp2 = MyExp()
best = RandomSearch(experiment=exp2).solve()
```

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
3. **Advanced features**: Multi-objective optimization; constraint handling is typically implemented inside the experiment (e.g., by returning penalized scores). Hyperactive v5 does not have a first‑class constraints API.
4. **Integration**: Use with other ML frameworks like XGBoost, LightGBM
5. **Scaling**: Learn about distributed optimization for large-scale problems

For more advanced topics, check out:
- [Optimization Algorithms Overview](../optimizers/index.md)
- [Creating Custom Experiments](../experiments/custom_experiments.md)
