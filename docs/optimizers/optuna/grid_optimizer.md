# Grid Optimizer

## Introduction

The Grid Optimizer provides systematic grid search through Optuna's interface. It exhaustively searches through all combinations of specified parameter values, ensuring complete coverage of the parameter space.

## Usage Example

```python
from hyperactive.opt.optuna import GridOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Define search space
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "linear"]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=SVC(),
    param_grid=param_grid,
    X=X, y=y,
    cv=5
)

# Create grid optimizer
optimizer = GridOptimizer(experiment=experiment)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
```

## When to Use Grid Optimizer

**Best for:**
- Small parameter spaces
- Discrete parameter values
- Complete space exploration
- Reproducible results

**Avoid if:**
- Large parameter spaces (combinatorial explosion)
- Continuous parameters
- Limited evaluation budget
