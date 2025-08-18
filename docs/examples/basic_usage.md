# Basic Usage

## Introduction

This page demonstrates fundamental Hyperactive usage patterns, from simple optimization tasks to more complex scenarios. These examples provide a foundation for understanding how to structure optimization problems and choose appropriate algorithms.

## Simple Mathematical Function Optimization

### Optimizing a 2D Function

```python
from hyperactive.base import BaseExperiment
from hyperactive.opt.gfo import BayesianOptimizer
import numpy as np

class SimpleFunction(BaseExperiment):
    """Optimize a simple 2D mathematical function"""
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        # Optimize negative of sphere function (Hyperactive maximizes)
        result = -(x**2 + y**2)
        return result, {"function_value": result}

# Create and run optimization
experiment = SimpleFunction()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Best value:", experiment.score(best_params))
```

### Multi-Dimensional Optimization

```python
class HighDimensionalFunction(BaseExperiment):
    """Optimize a high-dimensional function"""
    
    def __init__(self, dimensions=10):
        super().__init__()
        self.dimensions = dimensions
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        # Extract parameter vector
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        
        # Rosenbrock function (classic optimization benchmark)
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        # Return negative for maximization
        return -result, {"dimensions": len(x), "norm": np.linalg.norm(x)}

# Run optimization
experiment = HighDimensionalFunction(dimensions=5)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Optimized Rosenbrock function:")
print("Best parameters:", best_params)
```

## Basic Machine Learning Optimization

### Simple Classifier Optimization

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load simple dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define search space
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=3,
    scoring="accuracy"
)

# Optimize
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Train and evaluate final model
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
test_accuracy = final_model.score(X_test, y_test)

print("Best hyperparameters:", best_params)
print("Test accuracy:", test_accuracy)
```

### Regression Example

```python
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston

# Load regression dataset (using Boston housing for simplicity)
try:
    X, y = load_boston(return_X_y=True)
except ImportError:
    # Alternative if Boston dataset is not available
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=506, n_features=13, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ridge regression parameter optimization
param_grid = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    "fit_intercept": [True, False],
    "solver": ["auto", "svd", "cholesky"]
}

experiment = SklearnCvExperiment(
    estimator=Ridge(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

# Evaluate
final_model = Ridge(**best_params, random_state=42)
final_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score
predictions = final_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Best parameters:", best_params)
print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")
```

## Comparing Different Optimizers

### Algorithm Performance Comparison

```python
from hyperactive.opt.gfo import RandomSearch, HillClimbing, ParticleSwarmOptimizer
from hyperactive.opt.optuna import TPEOptimizer

# Define a test function
class TestFunction(BaseExperiment):
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        x, y, z = params["x"], params["y"], params["z"]
        # Ackley function (multimodal test function)
        result = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
                 np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + \
                 np.e + 20
        return -result, {"original_value": result}  # Minimize Ackley

# Test different optimizers
experiment = TestFunction()
optimizers = {
    "Bayesian": BayesianOptimizer(experiment=experiment),
    "Random Search": RandomSearch(experiment=experiment),
    "Hill Climbing": HillClimbing(experiment=experiment),
    "TPE": TPEOptimizer(experiment=experiment),
    "Particle Swarm": ParticleSwarmOptimizer(experiment=experiment, population=20)
}

results = {}
for name, optimizer in optimizers.items():
    best_params = optimizer.solve()
    best_score = experiment.score(best_params)[0]
    results[name] = {"params": best_params, "score": best_score}
    print(f"{name}: {best_score:.6f}")

# Find best performer
best_result = max(results.items(), key=lambda x: x[1]["score"])
print(f"\nBest optimizer: {best_result[0]} with score {best_result[1]['score']:.6f}")
```

## Different Parameter Types

### Mixed Parameter Types

```python
class MixedParameterExperiment(BaseExperiment):
    """Experiment with different parameter types"""
    
    def _paramnames(self):
        return ["continuous", "discrete", "categorical", "boolean"]
    
    def _evaluate(self, params):
        # Continuous parameter (float)
        continuous_contrib = params["continuous"] ** 2
        
        # Discrete parameter (integer) 
        discrete_contrib = params["discrete"] * 0.1
        
        # Categorical parameter (string/choice)
        categorical_map = {"option_a": 1.0, "option_b": 1.5, "option_c": 0.8}
        categorical_contrib = categorical_map.get(params["categorical"], 0)
        
        # Boolean parameter (0 or 1)
        boolean_contrib = 0.5 if params["boolean"] > 0.5 else 0
        
        total_score = continuous_contrib + discrete_contrib + categorical_contrib + boolean_contrib
        
        return total_score, {
            "continuous_part": continuous_contrib,
            "discrete_part": discrete_contrib, 
            "categorical_part": categorical_contrib,
            "boolean_part": boolean_contrib
        }

# Define parameter space with bounds and choices
experiment = MixedParameterExperiment()

# Note: Parameter bounds and types are typically defined in the optimizer
# Here we show conceptual usage
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best mixed parameters:", best_params)
print("Parameter breakdown:", experiment.score(best_params)[1])
```

## Working with Parameter Constraints

### Conditional Parameters

```python
class ConditionalParameterExperiment(BaseExperiment):
    """Experiment with parameter dependencies"""
    
    def _paramnames(self):
        return ["algorithm_type", "param_a", "param_b", "shared_param"]
    
    def _evaluate(self, params):
        algorithm = params["algorithm_type"]
        shared = params["shared_param"]
        
        if algorithm == "linear":
            # Linear algorithm only uses param_a
            score = params["param_a"] * shared
        elif algorithm == "nonlinear":
            # Nonlinear algorithm uses param_b
            score = params["param_b"] ** 2 * shared
        else:
            # Default case
            score = shared
        
        return score, {"algorithm_used": algorithm}

# Usage with conditional logic
experiment = ConditionalParameterExperiment()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best conditional parameters:", best_params)
```

## Error Handling and Robustness

### Robust Optimization with Error Handling

```python
import random

class RobustExperiment(BaseExperiment):
    """Experiment with built-in error handling"""
    
    def _paramnames(self):
        return ["success_rate", "noise_level", "computation_param"]
    
    def _evaluate(self, params):
        try:
            # Simulate computation that might fail
            if random.random() > params["success_rate"]:
                raise ValueError("Computation failed randomly")
            
            # Simulate noisy computation
            base_result = params["computation_param"] ** 2
            noise = random.gauss(0, params["noise_level"])
            result = base_result + noise
            
            # Validate result
            if result < -1000:  # Invalid result
                return float('-inf'), {"error": "result_out_of_bounds"}
            
            return result, {"noise_added": noise, "base_result": base_result}
            
        except Exception as e:
            # Return poor score for failed evaluations
            return float('-inf'), {"error": str(e)}

# Run robust optimization
experiment = RobustExperiment()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Robust optimization result:", best_params)
```

## Basic Parallel Processing

### Concurrent Evaluations

```python
from hyperactive.opt.gfo import RandomSearch

class ParallelizableExperiment(BaseExperiment):
    """Experiment suitable for parallel evaluation"""
    
    def _paramnames(self):
        return ["param1", "param2", "param3"]
    
    def _evaluate(self, params):
        # Simulate computation time
        import time
        time.sleep(0.1)  # Small delay to simulate work
        
        # Simple objective function
        result = sum(p**2 for p in params.values())
        return -result, {"evaluation_time": 0.1}

# Using random search which can benefit from parallelization
experiment = ParallelizableExperiment()
optimizer = RandomSearch(experiment=experiment)

# Note: Actual parallel execution depends on optimizer implementation
best_params = optimizer.solve()
print("Parallel optimization result:", best_params)
```

## Progress Tracking and Monitoring

### Optimization with Progress Tracking

```python
class TrackableExperiment(BaseExperiment):
    """Experiment that tracks optimization progress"""
    
    def __init__(self):
        super().__init__()
        self.evaluation_count = 0
        self.best_score = float('-inf')
        self.history = []
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        self.evaluation_count += 1
        
        # Simple objective
        score = -(params["x"]**2 + params["y"]**2)
        
        # Track progress
        if score > self.best_score:
            self.best_score = score
            print(f"Evaluation {self.evaluation_count}: New best score {score:.4f}")
        
        self.history.append({
            "evaluation": self.evaluation_count,
            "params": params.copy(),
            "score": score
        })
        
        return score, {"evaluation_number": self.evaluation_count}
    
    def get_optimization_history(self):
        return self.history

# Run with progress tracking
experiment = TrackableExperiment()
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print(f"\nOptimization completed after {experiment.evaluation_count} evaluations")
print("Best parameters:", best_params)

# Analyze optimization history
history = experiment.get_optimization_history()
scores = [h["score"] for h in history]
print(f"Score progression: {scores[:5]} ... {scores[-5:]}")
```

## Basic Experiment Templates

### Template for New Experiments

```python
class ExperimentTemplate(BaseExperiment):
    """Template for creating new experiments"""
    
    def __init__(self, custom_parameter=None):
        super().__init__()
        self.custom_parameter = custom_parameter
        # Initialize any experiment-specific attributes
    
    def _paramnames(self):
        # Define the parameters to optimize
        return ["param1", "param2", "param3"]
    
    def _evaluate(self, params):
        try:
            # Extract parameters
            p1 = params["param1"]
            p2 = params["param2"] 
            p3 = params["param3"]
            
            # Implement your objective function
            objective_value = self._compute_objective(p1, p2, p3)
            
            # Return score and metadata
            metadata = {
                "computation_details": "example",
                "custom_metric": objective_value * 2
            }
            
            return objective_value, metadata
            
        except Exception as e:
            # Handle errors gracefully
            return float('-inf'), {"error": str(e)}
    
    def _compute_objective(self, p1, p2, p3):
        """Implement your specific objective function here"""
        # This is where your domain-specific logic goes
        return p1 + p2 * p3  # Example computation

# Example usage of template
experiment = ExperimentTemplate(custom_parameter="example")
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Template experiment result:", best_params)
```

## Quick Start Checklist

1. **Choose experiment type**:
   - `BaseExperiment` for custom objectives
   - `SklearnCvExperiment` for ML hyperparameter tuning

2. **Define parameter space**:
   - Implement `_paramnames()` method
   - Ensure parameter bounds are reasonable

3. **Implement objective function**:
   - Return (score, metadata) tuple
   - Handle errors gracefully
   - Consider if you're maximizing or minimizing

4. **Select optimizer**:
   - `BayesianOptimizer` for general purpose
   - `RandomSearch` for baseline comparison
   - Algorithm-specific optimizers for special cases

5. **Run optimization**:
   - Create experiment and optimizer instances
   - Call `optimizer.solve()`
   - Evaluate results

6. **Analyze results**:
   - Check best parameters and scores
   - Validate on independent test data
   - Consider multiple runs for statistical significance

## References

- Hyperactive base classes documentation
- Optimization algorithm selection guide
- Parameter space design best practices