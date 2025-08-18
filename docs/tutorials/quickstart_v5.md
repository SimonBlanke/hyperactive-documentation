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

Let's start with a simple mathematical optimization problem:

```python
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.bench import Sphere

# Create experiment - minimize sphere function in 2D
experiment = Sphere(dimensions=2, bounds=(-5, 5))

# Create optimizer  
optimizer = BayesianOptimizer(experiment=experiment)

# Run optimization
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
```

## Machine Learning Hyperparameter Optimization

The most common use case is optimizing ML model hyperparameters:

```python
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load dataset
X, y = load_wine(return_X_y=True)

# Define the hyperparameter search space
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy"
)

# Create optimizer
optimizer = BayesianOptimizer(experiment=experiment)

# Run optimization
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Best CV accuracy:", experiment.score(best_params)[0])
```

## Using Different Optimization Algorithms

Hyperactive v5 provides 25+ optimization algorithms. Here's how to try different ones:

```python
from hyperactive.opt.gfo import (
    RandomSearch, 
    BayesianOptimizer, 
    ParticleSwarmOptimizer
)

# Same experiment as above
experiment = SklearnCvExperiment(...)

# Try different optimizers
optimizers = {
    "Random Search": RandomSearch(experiment=experiment),
    "Bayesian Optimization": BayesianOptimizer(experiment=experiment),
    "Particle Swarm": ParticleSwarmOptimizer(experiment=experiment, population=20)
}

results = {}
for name, optimizer in optimizers.items():
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    results[name] = (best_params, score)
    print(f"{name}: {score:.4f}")

# Find best algorithm
best_algorithm = max(results.items(), key=lambda x: x[1][1])
print(f"\nBest algorithm: {best_algorithm[0]} with score {best_algorithm[1][1]:.4f}")
```

## Sklearn-Compatible Interface

If you prefer the familiar scikit-learn interface:

```python
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define search space
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
    "kernel": ["rbf", "linear"]
}

# Create experiment for the optimizer
experiment = SklearnCvExperiment(
    estimator=SVC(),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=3
)

# Create OptCV - works like GridSearchCV
opt_cv = OptCV(
    estimator=SVC(),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=3
)

# Fit and predict like any sklearn estimator
opt_cv.fit(X_train, y_train)
predictions = opt_cv.predict(X_test)

print("Best parameters:", opt_cv.best_params_)
print("Best CV score:", opt_cv.best_score_)
print("Test accuracy:", opt_cv.score(X_test, y_test))
```

## Advanced Parameter Spaces

For more complex parameter spaces with different data types:

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.neural_network import MLPClassifier

# Mixed parameter types
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "lbfgs"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "max_iter": [200, 300, 500]
}

experiment = SklearnCvExperiment(
    estimator=MLPClassifier(random_state=42),
    param_grid=param_grid,
    X=X, y=y,
    cv=3,
    scoring="f1_macro"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Working with Different Backends

Hyperactive supports multiple optimization backends:

### Optuna Backend

```python
from hyperactive.opt.optuna import TPEOptimizer, CmaEsOptimizer

# Tree-structured Parzen Estimator
tpe_optimizer = TPEOptimizer(experiment=experiment)
best_params_tpe = tpe_optimizer.solve()

# CMA-ES for continuous optimization
cma_optimizer = CmaEsOptimizer(experiment=experiment)
best_params_cma = cma_optimizer.solve()
```

### Scikit-learn Backend

```python
from hyperactive.opt import GridSearchSk, RandomSearchSk

# Direct sklearn integration
grid_optimizer = GridSearchSk(experiment=experiment)
random_optimizer = RandomSearchSk(experiment=experiment)

best_params_grid = grid_optimizer.solve()
best_params_random = random_optimizer.solve()
```

## Custom Optimization Problems

Create your own optimization experiment:

```python
from hyperactive.base import BaseExperiment
import numpy as np

class CustomObjective(BaseExperiment):
    def __init__(self):
        super().__init__()
    
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        # Your custom objective function
        x, y, z = params["x"], params["y"], params["z"]
        
        # Example: minimize sum of squares with constraint
        objective = -(x**2 + y**2 + z**2)  # Negative for maximization
        
        # Add penalty for constraint violation
        if x + y + z > 5:
            objective -= 1000  # Heavy penalty
        
        return objective, {"constraint_violation": x + y + z > 5}

# Use custom experiment
custom_exp = CustomObjective()
optimizer = BayesianOptimizer(experiment=custom_exp)
best_params = optimizer.solve()
```

## Performance Tips

### 1. Choose the Right Algorithm

- **Random Search**: Quick baseline, good for high-dimensional spaces
- **Bayesian Optimization**: Sample-efficient, great for expensive evaluations
- **Particle Swarm**: Good for continuous spaces, handles multi-modal functions
- **Grid Search**: Systematic, interpretable, good for discrete spaces

### 2. Algorithm-Specific Tips

```python
# For Bayesian optimization - adjust exploration
optimizer = BayesianOptimizer(experiment=experiment, xi=0.01)  # More exploitation
optimizer = BayesianOptimizer(experiment=experiment, xi=0.1)   # More exploration

# For population methods - tune population size
optimizer = ParticleSwarmOptimizer(experiment=experiment, population=50)
optimizer = GeneticAlgorithm(experiment=experiment, population=100)

# For local search - adjust step size
optimizer = HillClimbing(experiment=experiment, epsilon=0.1)  # Larger steps
```

### 3. Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

# Use stratified CV for imbalanced datasets
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    X=X, y=y,
    cv=cv,  # Use custom CV strategy
    scoring="f1_weighted"  # Better for imbalanced data
)
```

### 4. Parallel Processing

```python
# Enable parallel cross-validation
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    X=X, y=y,
    cv=5,
    n_jobs=-1  # Use all available cores
)
```

## Common Patterns

### Comparing Multiple Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

models_configs = {
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "SVM": {
        "estimator": SVC(),
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"]
        }
    }
}

results = {}
for name, config in models_configs.items():
    experiment = SklearnCvExperiment(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        X=X, y=y, cv=5
    )
    
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    
    results[name] = {"params": best_params, "score": score}
    print(f"{name}: {score:.4f}")
```

### Early Stopping for Time-Limited Optimization

```python
import time

class TimeLimitedExperiment(BaseExperiment):
    def __init__(self, base_experiment, time_limit):
        super().__init__()
        self.base_experiment = base_experiment
        self.time_limit = time_limit
        self.start_time = None
    
    def _paramnames(self):
        return self.base_experiment.paramnames()
    
    def _evaluate(self, params):
        if self.start_time is None:
            self.start_time = time.time()
        
        if time.time() - self.start_time > self.time_limit:
            # Return poor score to stop optimization
            return float('-inf'), {"timeout": True}
        
        return self.base_experiment._evaluate(params)

# Use time-limited experiment
base_exp = SklearnCvExperiment(...)
time_limited_exp = TimeLimitedExperiment(base_exp, time_limit=300)  # 5 minutes
optimizer = BayesianOptimizer(experiment=time_limited_exp)
```

## Next Steps

1. **Explore more algorithms**: Try different optimizers for your specific problem
2. **Custom experiments**: Create domain-specific optimization problems
3. **Advanced features**: Multi-objective optimization, constraint handling
4. **Integration**: Use with other ML frameworks like XGBoost, LightGBM
5. **Scaling**: Learn about distributed optimization for large-scale problems

For more advanced topics, check out:
- [Advanced Optimization Strategies](optimization_strategies.md)
- [Creating Custom Experiments](custom_experiments.md)
- [Multi-Objective Optimization](multi_objective.md)
- [Performance Optimization](performance_tips.md)