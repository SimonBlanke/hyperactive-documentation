# Sphere Function

## Introduction

The Sphere function is one of the simplest and most fundamental benchmark functions in optimization. It represents a convex, unimodal function with a single global minimum, making it ideal for testing basic convergence properties of optimization algorithms.

## Mathematical Definition

The Sphere function is defined as:

$$f(x) = \sum_{i=1}^d x_i^2$$

Where $d$ is the dimensionality of the problem.

## Properties

- **Global minimum**: $f(0, 0, ..., 0) = 0$
- **Search domain**: Typically $[-5.12, 5.12]^d$ or $[-10, 10]^d$
- **Unimodal**: Single global minimum, no local minima
- **Separable**: Each variable contributes independently
- **Convex**: Bowl-shaped surface
- **Differentiable**: Smooth everywhere
- **Scalable**: Can be used in any dimension

## Usage Example

```python
from hyperactive.experiment.bench import Sphere
from hyperactive.opt.gfo import HillClimbing

# Create 3D Sphere function
experiment = Sphere(dimensions=3, bounds=(-10, 10))

# Create optimizer
optimizer = HillClimbing(experiment=experiment)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])

# The best score should be close to 0 (global minimum)
```

## Characteristics for Algorithm Testing

### What the Sphere Function Tests

1. **Basic Convergence**: Can the algorithm find the global minimum?
2. **Convergence Rate**: How quickly does it converge?
3. **Precision**: How close can it get to the exact optimum?
4. **Scalability**: Performance vs dimensionality relationship
5. **Consistency**: Reliable performance across runs

### Expected Algorithm Performance

- **Gradient-based methods**: Excellent, direct path to optimum
- **Hill Climbing**: Very good, steady improvement
- **Random Search**: Slow but eventually successful
- **Population methods**: Good, may be overkill for this simple function
- **Bayesian Optimization**: Excellent, very sample-efficient

## Multi-Dimensional Scaling

```python
# Test algorithm scalability
from hyperactive.opt.gfo import BayesianOptimizer
import time

dimensions = [2, 5, 10, 20, 50]
results = {}

for dim in dimensions:
    experiment = Sphere(dimensions=dim, bounds=(-10, 10))
    optimizer = BayesianOptimizer(experiment=experiment)
    
    start_time = time.time()
    best_params = optimizer.solve()
    end_time = time.time()
    
    best_score = experiment.score(best_params)[0]
    results[dim] = {
        'score': best_score,
        'time': end_time - start_time
    }
    
    print(f"{dim}D: Score = {best_score:.6f}, Time = {results[dim]['time']:.2f}s")
```

## Convergence Analysis

```python
# Analyze convergence behavior
import matplotlib.pyplot as plt

class ConvergenceTracker:
    def __init__(self):
        self.scores = []
        self.evaluations = 0
    
    def track_evaluation(self, params, score):
        self.evaluations += 1
        self.scores.append(score)

# This would require custom experiment implementation for tracking
```

## Algorithm Comparison

```python
from hyperactive.opt.gfo import (
    HillClimbing, RandomSearch, SimulatedAnnealing
)

# Create experiment
experiment = Sphere(dimensions=5, bounds=(-10, 10))

# Compare algorithms on simple function
algorithms = {
    "Hill Climbing": HillClimbing(experiment=experiment),
    "Random Search": RandomSearch(experiment=experiment),
    "Simulated Annealing": SimulatedAnnealing(experiment=experiment)
}

results = {}
for name, optimizer in algorithms.items():
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    results[name] = best_score
    print(f"{name}: {best_score:.8f}")
```

## Parameter Space Exploration

```python
# Different search space sizes
bounds_sizes = [(-1, 1), (-5, 5), (-10, 10), (-50, 50)]

for bounds in bounds_sizes:
    experiment = Sphere(dimensions=2, bounds=bounds)
    optimizer = HillClimbing(experiment=experiment)
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    
    print(f"Bounds {bounds}: Best score = {best_score:.8f}")
    print(f"Best params: {best_params}")
```

## Custom Sphere Variations

```python
# You can create custom sphere-like functions
from hyperactive.base import BaseExperiment
import numpy as np

class WeightedSphere(BaseExperiment):
    def __init__(self, dimensions=2, bounds=(-10, 10), weights=None):
        super().__init__()
        self.dimensions = dimensions
        self.bounds = bounds
        self.weights = weights or np.ones(dimensions)
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        # Weighted sphere function
        result = -np.sum(self.weights * x**2)  # Negative for maximization
        return result, {"weights_used": self.weights.tolist()}

# Use custom sphere
weights = np.array([1, 2, 3])  # Different importance for each dimension
experiment = WeightedSphere(dimensions=3, bounds=(-5, 5), weights=weights)
optimizer = HillClimbing(experiment=experiment)
best_params = optimizer.solve()
```

## Theoretical Properties

### Gradient Information
The gradient of the sphere function is:
$$\nabla f(x) = 2x$$

This means:
- Gradient points directly away from the origin
- Magnitude increases linearly with distance from origin
- Steepest descent leads directly to the global minimum

### Hessian Matrix
The Hessian is:
$$H = 2I$$

Where $I$ is the identity matrix, indicating:
- Constant curvature in all directions
- No interaction between variables
- Well-conditioned optimization problem

## Performance Baselines

For the sphere function, expect these approximate performance characteristics:

- **Random Search**: $O(d)$ evaluations to get within reasonable distance
- **Hill Climbing**: $O(\log(\epsilon^{-1}))$ for precision $\epsilon$
- **Gradient Methods**: Quadratic convergence
- **Population Methods**: May require more evaluations but very reliable

## Common Mistakes

1. **Over-engineering**: Using complex algorithms on this simple function
2. **Insufficient precision**: Stopping optimization too early
3. **Wrong bounds**: Using bounds that don't include the global minimum
4. **Ignoring scaling**: Not accounting for dimension-dependent difficulty

## References

- De Jong, K. A. (1975). An analysis of the behavior of a class of genetic adaptive systems.
- Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems.
