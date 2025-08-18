# NSGA-II Optimizer

## Introduction

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective evolutionary algorithm designed for problems with multiple conflicting objectives. It's particularly effective for Pareto optimization problems.

## Usage Example

```python
from hyperactive.opt.optuna import NSGAIIOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment

# Note: Multi-objective optimization requires special experiment setup
# This is a simplified example - actual multi-objective experiments
# would return multiple objectives

# Create optimizer
optimizer = NSGAIIOptimizer(
    experiment=experiment,
    population_size=50
)

# Run optimization
best_params = optimizer.solve()
```

## When to Use NSGA-II

**Best for:**
- Multi-objective optimization
- Pareto front exploration
- Conflicting objectives (accuracy vs speed, performance vs complexity)
- Population-based search

**Parameters:**
- `population_size`: Size of the population (default: 50)
- `mutation_prob`: Probability of mutation
- `crossover_prob`: Probability of crossover
