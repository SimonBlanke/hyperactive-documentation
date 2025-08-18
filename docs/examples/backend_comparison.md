# Backend Comparison

## Introduction

Hyperactive supports three major optimization backends: GFO (Gradient-Free-Optimizers), Optuna, and Scikit-learn. Each backend has unique strengths and is suited for different types of optimization problems. This page provides comprehensive comparisons and guidance for selecting the right backend for your use case.

## Backend Overview

### GFO (Gradient-Free-Optimizers) Backend
- **Strengths**: Diverse algorithm selection, fast execution, good for continuous optimization
- **Best for**: Mathematical function optimization, engineering problems, quick prototyping
- **Algorithms**: 15+ including Bayesian, PSO, Simulated Annealing, Hill Climbing

### Optuna Backend  
- **Strengths**: Advanced hyperparameter optimization, pruning, multi-objective support
- **Best for**: Machine learning hyperparameter tuning, complex search spaces, research
- **Algorithms**: TPE, CMA-ES, GP, NSGA-II/III, Grid, Random, QMC

### Scikit-learn Backend
- **Strengths**: Familiar interface, robust implementation, sklearn integration
- **Best for**: Simple hyperparameter searches, baseline comparisons, educational use
- **Algorithms**: GridSearchSk, RandomSearchSk

## Performance Comparison

### Mathematical Function Optimization

```python
from hyperactive.base import BaseExperiment
from hyperactive.opt.gfo import BayesianOptimizer as GFOBayesian, RandomSearch as GFORandomSearch
from hyperactive.opt.optuna import TPEOptimizer, CMAESOptimizer
from hyperactive.opt.sklearn import RandomSearchSk
import numpy as np
import time

class RosenbrockFunction(BaseExperiment):
    """Rosenbrock function benchmark"""
    
    def __init__(self, dimensions=5):
        super().__init__()
        self.dimensions = dimensions
    
    def _paramnames(self):
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        
        # Rosenbrock function
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        return -result, {"function_value": result}

# Test different backends on Rosenbrock function
experiment = RosenbrockFunction(dimensions=3)

# Define optimizers from different backends
optimizers = {
    "GFO Bayesian": GFOBayesian(experiment=experiment),
    "GFO Random": GFORandomSearch(experiment=experiment),
    "Optuna TPE": TPEOptimizer(experiment=experiment),
    "Optuna CMA-ES": CMAESOptimizer(experiment=experiment)
    # Note: Sklearn optimizers need different setup for this type of problem
}

results = {}

print("Rosenbrock Function Optimization Comparison:")
print("=" * 50)

for name, optimizer in optimizers.items():
    start_time = time.time()
    best_params = optimizer.solve()
    end_time = time.time()
    
    best_score = experiment.score(best_params)[0]
    function_value = -best_score  # Convert back to Rosenbrock value
    
    results[name] = {
        "best_params": best_params,
        "function_value": function_value,
        "time_taken": end_time - start_time,
        "score": best_score
    }
    
    print(f"{name}:")
    print(f"  Function value: {function_value:.6f}")
    print(f"  Time taken: {end_time - start_time:.3f}s")
    print(f"  Parameters: {best_params}")
    print()

# Find best performer
best_result = min(results.items(), key=lambda x: x[1]["function_value"])
print(f"Best performer: {best_result[0]} with value {best_result[1]['function_value']:.6f}")
```

### Machine Learning Hyperparameter Optimization

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define parameter space
param_grid = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Create base experiment
base_experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring="f1_weighted"
)

# Test ML-focused optimizers
ml_optimizers = {
    "GFO Bayesian": GFOBayesian(experiment=base_experiment),
    "GFO Random": GFORandomSearch(experiment=base_experiment),
    "Optuna TPE": TPEOptimizer(experiment=base_experiment),
    "Optuna GP": TPEOptimizer(experiment=base_experiment),  # Gaussian Process
    "Sklearn Random": RandomSearchSk(experiment=base_experiment)
}

ml_results = {}

print("\nMachine Learning Hyperparameter Optimization:")
print("=" * 50)

for name, optimizer in ml_optimizers.items():
    start_time = time.time()
    best_params = optimizer.solve()
    end_time = time.time()
    
    cv_score = base_experiment.score(best_params)[0]
    
    # Test on hold-out set
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    test_score = final_model.score(X_test, y_test)
    
    ml_results[name] = {
        "best_params": best_params,
        "cv_score": cv_score,
        "test_score": test_score,
        "time_taken": end_time - start_time
    }
    
    print(f"{name}:")
    print(f"  CV Score: {cv_score:.4f}")
    print(f"  Test Score: {test_score:.4f}")
    print(f"  Time: {end_time - start_time:.3f}s")
    print(f"  Best params: {best_params}")
    print()

# Best ML optimizer
best_ml = max(ml_results.items(), key=lambda x: x[1]["test_score"])
print(f"Best ML optimizer: {best_ml[0]} with test score {best_ml[1]['test_score']:.4f}")
```

## Algorithm-Specific Comparisons

### Bayesian Optimization Comparison

```python
# Compare Bayesian optimization implementations across backends
from hyperactive.opt.gfo import BayesianOptimizer as GFOBayesian
from hyperactive.opt.optuna import TPEOptimizer, GPOptimizer

class AckleyFunction(BaseExperiment):
    """Ackley function - challenging multimodal optimization"""
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        result = (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - 
                 np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 
                 np.e + 20)
        return -result, {"ackley_value": result}

experiment = AckleyFunction()

bayesian_optimizers = {
    "GFO Bayesian": GFOBayesian(experiment=experiment),
    "Optuna TPE": TPEOptimizer(experiment=experiment),
    "Optuna GP": GPOptimizer(experiment=experiment)
}

bayesian_results = {}

print("\nBayesian Optimization Backend Comparison:")
print("=" * 50)

for name, optimizer in bayesian_optimizers.items():
    # Run multiple trials for statistical significance
    trial_results = []
    
    for trial in range(5):
        start_time = time.time()
        best_params = optimizer.solve()
        end_time = time.time()
        
        score = experiment.score(best_params)[0]
        ackley_value = -score
        
        trial_results.append({
            "ackley_value": ackley_value,
            "time": end_time - start_time,
            "params": best_params
        })
    
    # Calculate statistics
    ackley_values = [r["ackley_value"] for r in trial_results]
    times = [r["time"] for r in trial_results]
    
    bayesian_results[name] = {
        "mean_ackley": np.mean(ackley_values),
        "std_ackley": np.std(ackley_values),
        "mean_time": np.mean(times),
        "best_ackley": min(ackley_values),
        "success_rate": np.mean([v < 0.1 for v in ackley_values])  # Success if close to global minimum
    }
    
    print(f"{name}:")
    print(f"  Mean Ackley: {bayesian_results[name]['mean_ackley']:.6f} ± {bayesian_results[name]['std_ackley']:.6f}")
    print(f"  Best Ackley: {bayesian_results[name]['best_ackley']:.6f}")
    print(f"  Mean time: {bayesian_results[name]['mean_time']:.3f}s")
    print(f"  Success rate: {bayesian_results[name]['success_rate']:.1%}")
    print()
```

### Random Search Comparison

```python
# Compare random search implementations
from hyperactive.opt.gfo import RandomSearch as GFORandomSearch
from hyperactive.opt.optuna import RandomOptimizer as OptunaRandom
from hyperactive.opt.sklearn import RandomSearchSk

# Use the same Ackley function
experiment = AckleyFunction()

random_optimizers = {
    "GFO Random": GFORandomSearch(experiment=experiment),
    "Optuna Random": OptunaRandom(experiment=experiment)
    # Note: Sklearn RandomSearch requires different setup for pure function optimization
}

random_results = {}

print("\nRandom Search Backend Comparison:")
print("=" * 40)

for name, optimizer in random_optimizers.items():
    start_time = time.time()
    best_params = optimizer.solve()
    end_time = time.time()
    
    score = experiment.score(best_params)[0]
    ackley_value = -score
    
    random_results[name] = {
        "ackley_value": ackley_value,
        "time": end_time - start_time,
        "params": best_params
    }
    
    print(f"{name}:")
    print(f"  Ackley value: {ackley_value:.6f}")
    print(f"  Time: {end_time - start_time:.3f}s")
    print()
```

## Use Case Recommendations

### When to Use GFO Backend

```python
# GFO is excellent for:
# 1. Mathematical function optimization
# 2. Engineering optimization problems  
# 3. When you need specific algorithms (PSO, Simulated Annealing, etc.)
# 4. Fast prototyping and experimentation

class EngineeringOptimizationExample(BaseExperiment):
    """Example engineering problem - beam design"""
    
    def _paramnames(self):
        return ["width", "height", "thickness"]
    
    def _evaluate(self, params):
        width = max(0.01, params["width"])
        height = max(0.01, params["height"])  
        thickness = max(0.001, min(width/2, height/2, params["thickness"]))
        
        # Weight (minimize)
        weight = width * height * thickness * 7850  # Steel density
        
        # Strength constraint
        moment_of_inertia = (width * height**3 - (width-2*thickness) * (height-2*thickness)**3) / 12
        max_stress = 1000 * height / (2 * moment_of_inertia)  # Simplified stress calc
        
        if max_stress > 250e6:  # Yield strength exceeded
            return float('-inf'), {"constraint_violation": True}
        
        return -weight, {"weight": weight, "stress": max_stress, "valid": True}

# GFO excels at this type of problem
engineering_experiment = EngineeringOptimizationExample()

# Test different GFO algorithms
from hyperactive.opt.gfo import (BayesianOptimizer, ParticleSwarmOptimizer, 
                                SimulatedAnnealing, HillClimbing)

gfo_algorithms = {
    "Bayesian": BayesianOptimizer(experiment=engineering_experiment),
    "PSO": ParticleSwarmOptimizer(experiment=engineering_experiment, population=20),
    "Simulated Annealing": SimulatedAnnealing(experiment=engineering_experiment),
    "Hill Climbing": HillClimbing(experiment=engineering_experiment)
}

print("\nGFO Backend - Engineering Optimization:")
print("=" * 40)

for name, optimizer in gfo_algorithms.items():
    best_params = optimizer.solve()
    result = engineering_experiment.score(best_params)
    
    if result[1]["valid"]:
        print(f"{name}: Weight = {result[1]['weight']:.2f}kg, Stress = {result[1]['stress']/1e6:.1f}MPa")
    else:
        print(f"{name}: Invalid solution")
```

### When to Use Optuna Backend

```python
# Optuna is excellent for:
# 1. Machine learning hyperparameter optimization
# 2. Complex search spaces with conditional parameters  
# 3. Multi-objective optimization
# 4. When you need pruning capabilities

from hyperactive.opt.optuna import NSGAIIOptimizer

class MLMultiObjectiveExample(BaseExperiment):
    """Multi-objective ML optimization - accuracy vs model complexity"""
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf"]
    
    def _evaluate(self, params):
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42
        )
        
        # Objective 1: Accuracy (maximize)
        accuracy = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy').mean()
        
        # Objective 2: Model simplicity (minimize complexity)
        complexity = params["n_estimators"] * (params["max_depth"] if params["max_depth"] > 0 else 10)
        simplicity = 1000 / (1 + complexity)  # Higher is simpler
        
        # Return tuple for multi-objective
        return (accuracy, simplicity), {
            "accuracy": accuracy,
            "complexity": complexity,
            "simplicity": simplicity
        }

# Multi-objective optimization with Optuna
multi_obj_experiment = MLMultiObjectiveExample(X_train, y_train)
nsga2_optimizer = NSGAIIOptimizer(experiment=multi_obj_experiment, population_size=20)

print("\nOptuna Backend - Multi-Objective Optimization:")
print("=" * 45)

best_params = nsga2_optimizer.solve()
result = multi_obj_experiment.score(best_params)

print(f"Best solution: {best_params}")
print(f"Accuracy: {result[1]['accuracy']:.4f}")
print(f"Simplicity: {result[1]['simplicity']:.2f}")
```

### When to Use Scikit-learn Backend

```python
# Scikit-learn backend is excellent for:
# 1. Simple hyperparameter optimization
# 2. Baseline comparisons
# 3. Educational purposes
# 4. When you need sklearn-compatible interface

from hyperactive.integrations.sklearn import OptCV
from sklearn.model_selection import GridSearchCV

# Simple parameter grid
simple_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# Compare Hyperactive sklearn backend with pure sklearn
sklearn_experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=simple_param_grid,
    X=X_train, y=y_train,
    cv=5,
    scoring="accuracy"
)

sklearn_optimizer = RandomSearchSk(experiment=sklearn_experiment)

# Traditional sklearn approach
traditional_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=simple_param_grid,
    cv=5,
    scoring="accuracy"
)

print("\nScikit-learn Backend Comparison:")
print("=" * 35)

# Hyperactive sklearn backend
start_time = time.time()
best_params_hyperactive = sklearn_optimizer.solve()
hyperactive_time = time.time() - start_time
hyperactive_score = sklearn_experiment.score(best_params_hyperactive)[0]

# Traditional sklearn
start_time = time.time()
traditional_grid_search.fit(X_train, y_train)
sklearn_time = time.time() - start_time

print(f"Hyperactive sklearn backend:")
print(f"  Best score: {hyperactive_score:.4f}")
print(f"  Time: {hyperactive_time:.3f}s")
print(f"  Parameters: {best_params_hyperactive}")

print(f"\nTraditional GridSearchCV:")
print(f"  Best score: {traditional_grid_search.best_score_:.4f}")
print(f"  Time: {sklearn_time:.3f}s")
print(f"  Parameters: {traditional_grid_search.best_params_}")
```

## Backend Selection Guide

### Decision Matrix

```python
def recommend_backend(problem_type, search_space_size, optimization_time, ml_focus, multi_objective):
    """
    Recommend best backend based on problem characteristics
    
    Parameters:
    - problem_type: 'mathematical', 'ml_hyperparams', 'engineering', 'business'
    - search_space_size: 'small', 'medium', 'large'
    - optimization_time: 'fast', 'medium', 'long'
    - ml_focus: bool - whether problem is ML-related
    - multi_objective: bool - whether multiple objectives need optimization
    """
    
    recommendations = []
    
    # GFO Backend scoring
    gfo_score = 0
    if problem_type in ['mathematical', 'engineering']:
        gfo_score += 3
    if search_space_size in ['small', 'medium']:
        gfo_score += 2
    if optimization_time == 'fast':
        gfo_score += 2
    if not multi_objective:
        gfo_score += 1
    
    # Optuna Backend scoring
    optuna_score = 0
    if ml_focus:
        optuna_score += 3
    if search_space_size in ['medium', 'large']:
        optuna_score += 2
    if multi_objective:
        optuna_score += 3
    if optimization_time in ['medium', 'long']:
        optuna_score += 1
    
    # Sklearn Backend scoring
    sklearn_score = 0
    if ml_focus:
        sklearn_score += 2
    if search_space_size == 'small':
        sklearn_score += 2
    if optimization_time == 'fast':
        sklearn_score += 1
    
    # Generate recommendations
    scores = [('GFO', gfo_score), ('Optuna', optuna_score), ('Scikit-learn', sklearn_score)]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores

# Example recommendations
test_cases = [
    {
        "name": "Mathematical Function Optimization",
        "problem_type": "mathematical",
        "search_space_size": "medium",
        "optimization_time": "fast",
        "ml_focus": False,
        "multi_objective": False
    },
    {
        "name": "Deep Learning Hyperparameter Tuning",
        "problem_type": "ml_hyperparams", 
        "search_space_size": "large",
        "optimization_time": "long",
        "ml_focus": True,
        "multi_objective": False
    },
    {
        "name": "Multi-Objective ML Optimization",
        "problem_type": "ml_hyperparams",
        "search_space_size": "medium", 
        "optimization_time": "medium",
        "ml_focus": True,
        "multi_objective": True
    },
    {
        "name": "Simple Baseline Comparison",
        "problem_type": "ml_hyperparams",
        "search_space_size": "small",
        "optimization_time": "fast", 
        "ml_focus": True,
        "multi_objective": False
    }
]

print("\nBackend Recommendation Guide:")
print("=" * 40)

for case in test_cases:
    recommendations = recommend_backend(
        case["problem_type"],
        case["search_space_size"], 
        case["optimization_time"],
        case["ml_focus"],
        case["multi_objective"]
    )
    
    print(f"\n{case['name']}:")
    print(f"  Characteristics: {case['problem_type']}, {case['search_space_size']} space, {case['optimization_time']} time")
    print(f"  Recommended backends:")
    for i, (backend, score) in enumerate(recommendations):
        print(f"    {i+1}. {backend} (score: {score})")
```

## Performance Benchmarks

### Computational Efficiency

```python
# Benchmark computational efficiency across backends
import matplotlib.pyplot as plt

def benchmark_efficiency():
    """Compare optimization efficiency across backends"""
    
    problem_sizes = [2, 5, 10, 20]  # Dimensions
    backends_to_test = ["GFO", "Optuna"]  # Skip sklearn for function optimization
    
    efficiency_results = {backend: [] for backend in backends_to_test}
    
    for dim in problem_sizes:
        print(f"Testing {dim}D optimization...")
        
        # Create test problem
        class TestProblem(BaseExperiment):
            def __init__(self, dimensions):
                super().__init__()
                self.dimensions = dimensions
            
            def _paramnames(self):
                return [f"x{i}" for i in range(self.dimensions)]
            
            def _evaluate(self, params):
                x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
                return -np.sum(x**2), {}  # Simple sphere function
        
        experiment = TestProblem(dim)
        
        # Test GFO
        start_time = time.time()
        gfo_optimizer = GFOBayesian(experiment=experiment)
        gfo_best = gfo_optimizer.solve()
        gfo_time = time.time() - start_time
        gfo_score = experiment.score(gfo_best)[0]
        
        efficiency_results["GFO"].append({
            "dimensions": dim,
            "time": gfo_time,
            "score": gfo_score
        })
        
        # Test Optuna
        start_time = time.time()
        optuna_optimizer = TPEOptimizer(experiment=experiment)
        optuna_best = optuna_optimizer.solve()
        optuna_time = time.time() - start_time
        optuna_score = experiment.score(optuna_best)[0]
        
        efficiency_results["Optuna"].append({
            "dimensions": dim,
            "time": optuna_time, 
            "score": optuna_score
        })
    
    # Print results
    print("\nEfficiency Benchmark Results:")
    print("=" * 40)
    
    for backend, results in efficiency_results.items():
        print(f"\n{backend} Backend:")
        for result in results:
            print(f"  {result['dimensions']}D: {result['time']:.3f}s, score: {result['score']:.4f}")
    
    return efficiency_results

# Run efficiency benchmark
# efficiency_data = benchmark_efficiency()
```

## Summary and Recommendations

### Quick Reference Guide

| Use Case | Recommended Backend | Alternative | Notes |
|----------|-------------------|-------------|--------|
| Mathematical Functions | GFO | Optuna | GFO has specialized algorithms |
| ML Hyperparameters | Optuna | GFO | Optuna has ML-specific features |
| Multi-Objective | Optuna | - | NSGA-II/III support |
| Quick Baselines | Scikit-learn | GFO | Familiar interface |
| Engineering Design | GFO | Optuna | Domain-specific algorithms |
| Large Search Spaces | Optuna | GFO | Better scaling |
| Real-time Optimization | GFO | - | Fastest execution |
| Research/Experimentation | Optuna | GFO | Advanced features |

### Best Practices by Backend

#### GFO Backend
```python
# Best practices for GFO
# 1. Choose algorithm based on problem characteristics
# 2. Use population-based methods (PSO) for multimodal problems
# 3. Use Bayesian optimization for expensive evaluations
# 4. Consider Simulated Annealing for discrete problems

gfo_best_practices = {
    "continuous_smooth": "BayesianOptimizer",
    "multimodal": "ParticleSwarmOptimizer",
    "discrete": "SimulatedAnnealing", 
    "fast_convergence": "HillClimbing",
    "global_search": "EvolutionStrategy"
}
```

#### Optuna Backend
```python
# Best practices for Optuna
# 1. Use TPE for general hyperparameter optimization
# 2. Use CMA-ES for continuous parameter spaces
# 3. Use NSGA-II for multi-objective problems
# 4. Consider pruning for expensive ML evaluations

optuna_best_practices = {
    "ml_hyperparams": "TPEOptimizer",
    "continuous_only": "CMAESOptimizer", 
    "multi_objective": "NSGAIIOptimizer",
    "neural_networks": "TPEOptimizer",  # with pruning
    "large_search_space": "QMCOptimizer"
}
```

#### Scikit-learn Backend
```python
# Best practices for Sklearn
# 1. Use for simple parameter grids
# 2. Good for educational purposes
# 3. Provides familiar sklearn interface
# 4. Best for small search spaces

sklearn_best_practices = {
    "simple_grids": "GridSearchSk",
    "random_sampling": "RandomSearchSk",
    "sklearn_compatibility": "OptCV interface",
    "baseline_comparison": "GridSearchSk"
}
```

## Conclusion

The choice of backend significantly impacts optimization performance and ease of use. Consider your specific requirements:

- **Problem domain**: Mathematical vs ML vs Engineering
- **Search space characteristics**: Size, dimensionality, parameter types
- **Computational constraints**: Time, resources, evaluations
- **Advanced features**: Multi-objective, pruning, conditional parameters
- **Integration needs**: Existing workflows, frameworks

Start with the recommended backend for your use case, but don't hesitate to experiment with alternatives. Each backend has unique strengths that may benefit your specific optimization problem.

## References

- GFO Documentation: [https://github.com/SimonBlanke/Gradient-Free-Optimizers](https://github.com/SimonBlanke/Gradient-Free-Optimizers)
- Optuna Documentation: [https://optuna.org/](https://optuna.org/)
- Scikit-learn Model Selection: [https://scikit-learn.org/stable/model_selection.html](https://scikit-learn.org/stable/model_selection.html)