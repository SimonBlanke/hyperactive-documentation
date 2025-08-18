# Parabola Function

## Introduction

The Parabola function is a simple quadratic benchmark function that provides a basic test case for optimization algorithms. It's similar to the Sphere function but with a different mathematical structure, offering insights into algorithm behavior on quadratic landscapes.

## Mathematical Definition

The Parabola function can be defined in various forms. A common implementation is:

$$f(x) = \sum_{i=1}^d (x_i - c_i)^2$$

Where $c_i$ are offset constants (often set to 0 for centered parabola).

## Properties

- **Global minimum**: $f(c_1, c_2, ..., c_d) = 0$ where $c_i$ are the offsets
- **Search domain**: Typically $[-10, 10]^d$ or similar
- **Unimodal**: Single global minimum
- **Separable**: Variables are independent  
- **Convex**: Bowl-shaped surface
- **Quadratic**: Second-order polynomial
- **Scalable**: Works in any dimension

## Usage Example

```python
from hyperactive.experiment.bench import Parabola
from hyperactive.opt.gfo import HillClimbing

# Create 2D Parabola function
experiment = Parabola(dimensions=2, bounds=(-5, 5))

# Create optimizer
optimizer = HillClimbing(experiment=experiment, epsilon=0.1)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])

# Should find minimum close to origin
```

## Algorithm Testing

### Convergence Speed Comparison

```python
from hyperactive.opt.gfo import (
    HillClimbing, RandomSearch, BayesianOptimizer
)
import time

# Create experiment
experiment = Parabola(dimensions=3, bounds=(-10, 10))

# Test different algorithms
algorithms = {
    "Hill Climbing": HillClimbing(experiment=experiment),
    "Random Search": RandomSearch(experiment=experiment),
    "Bayesian Opt": BayesianOptimizer(experiment=experiment)
}

for name, optimizer in algorithms.items():
    start_time = time.time()
    best_params = optimizer.solve()
    end_time = time.time()
    
    best_score = experiment.score(best_params)[0]
    print(f"{name}:")
    print(f"  Score: {best_score:.8f}")
    print(f"  Time: {end_time - start_time:.2f}s")
    print(f"  Params: {best_params}")
```

### Precision Testing

```python
# Test how close algorithms can get to the true optimum
experiment = Parabola(dimensions=2, bounds=(-5, 5))

optimizers = [
    HillClimbing(experiment=experiment, epsilon=0.01),  # Fine-grained
    HillClimbing(experiment=experiment, epsilon=0.1),   # Coarse-grained
    BayesianOptimizer(experiment=experiment)
]

for i, optimizer in enumerate(optimizers):
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    
    # Calculate distance from true optimum (assuming centered at origin)
    distance = sum(best_params[key]**2 for key in best_params)**0.5
    
    print(f"Optimizer {i+1}:")
    print(f"  Final score: {best_score:.8f}")
    print(f"  Distance from optimum: {distance:.8f}")
```

### Scalability Analysis

```python
# Test performance across dimensions
dimensions = [2, 5, 10, 20]

for dim in dimensions:
    experiment = Parabola(dimensions=dim, bounds=(-5, 5))
    optimizer = HillClimbing(experiment=experiment)
    
    start_time = time.time()
    best_params = optimizer.solve()
    end_time = time.time()
    
    best_score = experiment.score(best_params)[0]
    
    print(f"{dim}D Parabola:")
    print(f"  Score: {best_score:.6f}")
    print(f"  Time: {end_time - start_time:.2f}s")
```

## Custom Parabola Variations

```python
from hyperactive.base import BaseExperiment
import numpy as np

class OffsetParabola(BaseExperiment):
    """Parabola with custom center point"""
    
    def __init__(self, dimensions=2, bounds=(-10, 10), center=None):
        super().__init__()
        self.dimensions = dimensions
        self.bounds = bounds
        self.center = center or np.zeros(dimensions)
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        # Parabola centered at self.center
        result = -np.sum((x - self.center)**2)  # Negative for maximization
        return result, {"center": self.center.tolist()}

# Parabola centered at (2, 3)
experiment = OffsetParabola(
    dimensions=2, 
    bounds=(-5, 5), 
    center=np.array([2, 3])
)

optimizer = HillClimbing(experiment=experiment)
best_params = optimizer.solve()
print("Best params:", best_params)
print("Should be close to [2, 3]")
```

## Rotated Parabola

```python
class RotatedParabola(BaseExperiment):
    """Parabola with rotated axes (non-separable)"""
    
    def __init__(self, dimensions=2, bounds=(-10, 10)):
        super().__init__()
        self.dimensions = dimensions
        self.bounds = bounds
        # Create random rotation matrix
        np.random.seed(42)  # For reproducibility
        self.rotation_matrix = self._random_rotation_matrix(dimensions)
    
    def _random_rotation_matrix(self, n):
        """Generate random rotation matrix"""
        # For simplicity, using a basic rotation for 2D
        if n == 2:
            angle = np.pi / 4  # 45 degree rotation
            return np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
        else:
            return np.eye(n)  # Identity for higher dimensions
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        # Rotate the coordinates
        rotated_x = self.rotation_matrix @ x
        # Standard parabola in rotated space
        result = -np.sum(rotated_x**2)
        return result, {"rotation_applied": True}

# Test rotated parabola
experiment = RotatedParabola(dimensions=2, bounds=(-5, 5))
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Educational Use Cases

### Teaching Optimization Basics

```python
# Demonstrate different step sizes
step_sizes = [0.01, 0.1, 1.0]
experiment = Parabola(dimensions=2, bounds=(-10, 10))

for epsilon in step_sizes:
    optimizer = HillClimbing(experiment=experiment, epsilon=epsilon)
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    
    print(f"Step size {epsilon}:")
    print(f"  Final score: {best_score:.6f}")
    print(f"  Final params: {best_params}")
```

### Convergence Visualization

```python
# Track optimization progress (conceptual - would need custom implementation)
class TrackingParabola(BaseExperiment):
    def __init__(self, dimensions=2, bounds=(-10, 10)):
        super().__init__()
        self.dimensions = dimensions
        self.bounds = bounds
        self.evaluation_history = []
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        result = -np.sum(x**2)
        
        # Track evaluations
        self.evaluation_history.append({
            'params': params.copy(),
            'score': result
        })
        
        return result, {"evaluation_number": len(self.evaluation_history)}

# Use tracking experiment
experiment = TrackingParabola(dimensions=2, bounds=(-5, 5))
optimizer = HillClimbing(experiment=experiment)
best_params = optimizer.solve()

print(f"Total evaluations: {len(experiment.evaluation_history)}")
print(f"Final score: {experiment.evaluation_history[-1]['score']}")
```

## Performance Characteristics

For the Parabola function, typical performance expectations:

- **Hill Climbing**: Very efficient, direct path to minimum
- **Random Search**: Slow convergence, many evaluations needed
- **Simulated Annealing**: Good performance, some random exploration
- **Bayesian Optimization**: Excellent efficiency, few evaluations needed
- **Population methods**: Overkill for this simple function

## Common Applications

1. **Algorithm validation**: Quick sanity check for new optimizers
2. **Parameter tuning**: Test optimizer parameters on simple function
3. **Educational purposes**: Demonstrate optimization principles
4. **Baseline comparison**: Simple reference for performance
5. **Convergence analysis**: Study optimization dynamics

## References

- Fletcher, R. (2013). Practical methods of optimization.
- Nocedal, J., & Wright, S. (2006). Numerical optimization.
