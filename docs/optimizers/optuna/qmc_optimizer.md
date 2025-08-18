# QMC Optimizer

## Introduction

Quasi-Monte Carlo (QMC) Optimizer uses low-discrepancy sequences instead of random sampling. This provides better coverage of the parameter space compared to pure random sampling, often leading to faster convergence.

## Usage Example

```python
from hyperactive.opt.optuna import QMCOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load dataset
X, y = load_wine(return_X_y=True)

# Define search space
param_grid = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    cv=5
)

# Create QMC optimizer
optimizer = QMCOptimizer(
    experiment=experiment,
    qmc_type="sobol"
)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
```

## When to Use QMC Optimizer

**Best for:**
- Better space coverage than random search
- Integration and sampling problems
- When you want deterministic "random" sequences
- High-dimensional parameter spaces

**Parameters:**
- `qmc_type`: Type of QMC sequence ("sobol", "halton")
- `scramble`: Whether to scramble the sequence

## QMC Sequence Types

### Sobol Sequences
- Excellent space-filling properties
- Good for most optimization problems
- Default choice for QMC

### Halton Sequences  
- Simpler construction
- Can suffer from correlation in higher dimensions
- Good for lower-dimensional problems
