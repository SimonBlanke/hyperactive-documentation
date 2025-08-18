# Ackley Function

## Introduction

The Ackley function is a widely used multimodal benchmark function for testing optimization algorithms. It features many local minima and a single global minimum, making it an excellent test for an algorithm's ability to avoid getting trapped in local optima.

## Mathematical Definition

The Ackley function is defined as:

$$f(x) = -a \exp\left(-b\sqrt{rac{1}{d}\sum_{i=1}^d x_i^2}ight) - \exp\left(rac{1}{d}\sum_{i=1}^d \cos(c x_i)ight) + a + e$$

Where:
- $a = 20$ (amplitude of exponential term)
- $b = 0.2$ (exponential decay factor)  
- $c = 2\pi$ (frequency of cosine term)
- $d$ = dimensionality
- $e$ = Euler's number

## Properties

- **Global minimum**: $f(0, 0, ..., 0) = 0$
- **Search domain**: Typically $[-32.768, 32.768]^d$ or $[-5, 5]^d$
- **Multimodal**: Many local minima
- **Non-separable**: Variables are coupled
- **Differentiable**: Smooth function
- **Scalable**: Can be used in any dimension

## Usage Example

```python
from hyperactive.experiment.bench import Ackley
from hyperactive.opt.gfo import BayesianOptimizer

# Create 2D Ackley function
experiment = Ackley(dimensions=2, bounds=(-5, 5))

# Create optimizer
optimizer = BayesianOptimizer(experiment=experiment)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])

# The best score should be close to 0 (global minimum)
```

## Characteristics for Algorithm Testing

### What the Ackley Function Tests

1. **Global vs Local Search**: Can the algorithm escape local minima?
2. **Exploitation**: Can it fine-tune around the global optimum?
3. **Scalability**: How does performance change with dimensionality?
4. **Robustness**: Consistent performance across runs?

### Expected Behavior

- **Random Search**: Explores broadly but may miss global optimum
- **Hill Climbing**: Often gets trapped in local minima
- **Simulated Annealing**: Can escape local minima with proper cooling
- **Bayesian Optimization**: Usually finds global optimum efficiently
- **Population Methods**: Good at avoiding local minima

## Multi-Dimensional Usage

```python
# Test scalability
dimensions = [2, 5, 10, 20]
results = {}

for dim in dimensions:
    experiment = Ackley(dimensions=dim, bounds=(-5, 5))
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    results[dim] = best_score
    print(f"{dim}D: Best score = {best_score}")
```

## Visualization

The Ackley function in 2D shows:
- A central valley leading to the global minimum at (0,0)
- Many small local minima scattered across the surface
- Exponential decay from the center
- High-frequency oscillations creating the local minima

## Algorithm Comparison Example

```python
from hyperactive.opt.gfo import (
    RandomSearch, HillClimbing, SimulatedAnnealing, 
    BayesianOptimizer, ParticleSwarmOptimizer
)

# Create experiment
experiment = Ackley(dimensions=5, bounds=(-5, 5))

# Test multiple algorithms
algorithms = {
    "Random Search": RandomSearch(experiment=experiment),
    "Hill Climbing": HillClimbing(experiment=experiment),
    "Simulated Annealing": SimulatedAnnealing(experiment=experiment),
    "Bayesian Optimization": BayesianOptimizer(experiment=experiment),
    "Particle Swarm": ParticleSwarmOptimizer(experiment=experiment, population=20)
}

results = {}
for name, optimizer in algorithms.items():
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    results[name] = best_score
    print(f"{name}: {best_score:.6f}")

# Best performing algorithm
best_algorithm = max(results.items(), key=lambda x: x[1])
print(f"\nBest: {best_algorithm[0]} with score {best_algorithm[1]:.6f}")
```

## Parameter Sensitivity

The Ackley function parameters affect difficulty:

```python
# Standard Ackley (easier)
experiment_easy = Ackley(dimensions=2, bounds=(-5, 5))

# Larger search space (harder)  
experiment_hard = Ackley(dimensions=2, bounds=(-32, 32))

# Higher dimensions (much harder)
experiment_very_hard = Ackley(dimensions=20, bounds=(-5, 5))
```

## References

- Ackley, D. H. (1987). A connectionist machine for genetic hillclimbing.
- Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems.
