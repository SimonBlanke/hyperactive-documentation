# NSGA-III Optimizer

## Introduction

NSGA-III (Non-dominated Sorting Genetic Algorithm III) is an advanced multi-objective evolutionary algorithm designed for many-objective optimization problems (>3 objectives). It uses reference points to maintain diversity.

## Usage Example

```python
from hyperactive.opt.optuna import NSGAIIIOptimizer

# Create optimizer
optimizer = NSGAIIIOptimizer(
    experiment=experiment,
    population_size=50
)

# Run optimization
best_params = optimizer.solve()
```

## When to Use NSGA-III

**Best for:**
- Many-objective optimization (>3 objectives)
- Reference point-based optimization
- Maintaining diversity in high-objective spaces
- Complex Pareto fronts

**Parameters:**
- `population_size`: Size of the population
- `reference_points`: Custom reference points for diversity
