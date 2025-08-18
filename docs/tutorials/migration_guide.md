# Migration Guide: v4 to v5

This guide helps you migrate from Hyperactive v4 to v5. The v5 release introduces significant architectural changes that improve modularity and extensibility, but require updates to existing code.

## Overview of Changes

### Major Changes in v5

1. **No more `Hyperactive` class**: Individual optimizer classes are used directly
2. **Experiment-based approach**: Optimization problems are defined as experiment objects
3. **Backend-specific imports**: Optimizers are imported from specific backend modules
4. **New base class architecture**: Uses skbase framework for extensibility
5. **Enhanced integration layer**: Better support for ML frameworks

## Before You Start

### Check Your Dependencies

V5 has updated dependencies:

```bash
# Install v5
pip install hyperactive>=5.0.0

# Key new dependencies
pip install scikit-base
pip install gradient-free-optimizers>=1.2.4
```

## Basic API Migration

### v4 Pattern

```python
# v4 Code
from hyperactive import Hyperactive

def objective_function(para):
    score = model.score(X, y)  # Your evaluation logic
    return score

search_space = {
    "param1": [1, 2, 3, 4, 5],
    "param2": [0.1, 0.01, 0.001],
}

hyper = Hyperactive()
hyper.add_search(
    objective_function,
    search_space,
    optimizer="BayesianOptimization",  # String name
    n_iter=100
)
hyper.run()

best_para = hyper.best_para()
```

### v5 Pattern

```python
# v5 Code
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment

# Define experiment instead of objective function
experiment = SklearnCvExperiment(
    estimator=model,
    param_grid={
        "param1": [1, 2, 3, 4, 5],
        "param2": [0.1, 0.01, 0.001],
    },
    X=X, y=y,
    cv=5
)

# Use optimizer class directly
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Migration Examples

### Example 1: Basic Hyperparameter Optimization

**v4 Code:**
```python
from hyperactive import Hyperactive
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective_function(para):
    model = RandomForestClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)
    return scores.mean()

search_space = {
    "n_estimators": range(10, 200, 10),
    "max_depth": range(3, 15),
}

hyper = Hyperactive()
hyper.add_search(objective_function, search_space, n_iter=50)
hyper.run()
```

**v5 Code:**
```python
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier

# Create experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": list(range(10, 200, 10)),
        "max_depth": list(range(3, 15)),
    },
    X=X_train, y=y_train,
    cv=3
)

# Use optimizer directly
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

### Example 2: Custom Objective Function

**v4 Code:**
```python
def complex_objective(para):
    # Your custom logic
    x, y = para["x"], para["y"]
    result = -(x**2 + y**2)  # Minimize sphere
    
    # Add constraint penalty
    if x + y > 5:
        result -= 100
    
    return result

search_space = {
    "x": np.arange(-10, 10, 0.1),
    "y": np.arange(-10, 10, 0.1),
}

hyper = Hyperactive()
hyper.add_search(complex_objective, search_space, optimizer="SimulatedAnnealing")
hyper.run()
```

**v5 Code:**
```python
from hyperactive.base import BaseExperiment
from hyperactive.opt.gfo import SimulatedAnnealing
import numpy as np

class CustomExperiment(BaseExperiment):
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        result = -(x**2 + y**2)  # Minimize sphere
        
        # Add constraint penalty
        if x + y > 5:
            result -= 100
        
        return result, {"constraint_violation": x + y > 5}

# Use custom experiment
experiment = CustomExperiment()
optimizer = SimulatedAnnealing(experiment=experiment)
best_params = optimizer.solve()
```

### Example 3: Multiple Search Runs

**v4 Code:**
```python
hyper = Hyperactive()
hyper.add_search(objective_func1, search_space1, optimizer="RandomSearch")
hyper.add_search(objective_func2, search_space2, optimizer="BayesianOptimization")
hyper.run()

# Access results
results1 = hyper.search_data[objective_func1]
results2 = hyper.search_data[objective_func2]
```

**v5 Code:**
```python
from hyperactive.opt.gfo import RandomSearch, BayesianOptimizer

# Create separate experiments
experiment1 = SklearnCvExperiment(...)  # Your first experiment
experiment2 = SklearnCvExperiment(...)  # Your second experiment

# Run optimizations separately
optimizer1 = RandomSearch(experiment=experiment1)
optimizer2 = BayesianOptimizer(experiment=experiment2)

best_params1 = optimizer1.solve()
best_params2 = optimizer2.solve()

# Access best parameters
results = {
    "experiment1": best_params1,
    "experiment2": best_params2
}
```

## Algorithm Name Mapping

Many algorithm names have changed between versions:

| v4 Name | v5 Import | v5 Class Name |
|---------|-----------|---------------|
| `"HillClimbing"` | `from hyperactive.opt.gfo import HillClimbing` | `HillClimbing` |
| `"BayesianOptimization"` | `from hyperactive.opt.gfo import BayesianOptimizer` | `BayesianOptimizer` |
| `"RandomSearch"` | `from hyperactive.opt.gfo import RandomSearch` | `RandomSearch` |
| `"GridSearch"` | `from hyperactive.opt.gfo import GridSearch` | `GridSearch` |
| `"SimulatedAnnealing"` | `from hyperactive.opt.gfo import SimulatedAnnealing` | `SimulatedAnnealing` |
| `"ParticleSwarm"` | `from hyperactive.opt.gfo import ParticleSwarmOptimizer` | `ParticleSwarmOptimizer` |
| `"GeneticAlgorithm"` | `from hyperactive.opt.gfo import GeneticAlgorithm` | `GeneticAlgorithm` |

### New Algorithms in v5

V5 adds many new algorithms not available in v4:

```python
# Optuna backend algorithms
from hyperactive.opt.optuna import (
    TPEOptimizer, CmaEsOptimizer, GPOptimizer,
    NSGAIIOptimizer, NSGAIIIOptimizer
)

# Additional GFO algorithms  
from hyperactive.opt.gfo import (
    DirectAlgorithm, LipschitzOptimizer, 
    ForestOptimizer, SpiralOptimization
)

# Scikit-learn integration
from hyperactive.opt import GridSearchSk, RandomSearchSk
```

## Parameter Configuration

### v4 Parameter Passing

```python
hyper.add_search(
    objective_function,
    search_space,
    optimizer="BayesianOptimization",
    n_iter=100,
    # Optimizer-specific parameters
    xi=0.01,
    gpr=some_gpr_object,
    # General parameters
    random_state=42,
    verbosity=["progress_bar", "print_results"]
)
```

### v5 Parameter Passing

```python
# Parameters are passed during optimizer initialization
optimizer = BayesianOptimizer(
    experiment=experiment,
    # Optimizer-specific parameters
    xi=0.01,
    gpr=some_gpr_object
)

# No built-in progress bars - you can implement your own
best_params = optimizer.solve()
```

## Advanced Migration Patterns

### Memory and Warm Starting

**v4 Code:**
```python
hyper = Hyperactive()
hyper.add_search(
    objective_function,
    search_space,
    memory=True,
    memory_warm_start=previous_results
)
```

**v5 Code:**
```python
# V5 doesn't have built-in memory, but you can implement it:
from hyperactive.opt.gfo import BayesianOptimizer

class MemoryExperiment(BaseExperiment):
    def __init__(self, base_experiment, memory_dict=None):
        super().__init__()
        self.base_experiment = base_experiment
        self.memory = memory_dict or {}
    
    def _paramnames(self):
        return self.base_experiment.paramnames()
    
    def _evaluate(self, params):
        # Create key from parameters
        key = tuple(sorted(params.items()))
        
        if key in self.memory:
            return self.memory[key]
        
        result = self.base_experiment._evaluate(params)
        self.memory[key] = result
        return result

# Use memory experiment
base_exp = SklearnCvExperiment(...)
memory_exp = MemoryExperiment(base_exp, previous_results)
optimizer = BayesianOptimizer(experiment=memory_exp)
```

### Parallel Processing

**v4 Code:**
```python
hyper = Hyperactive()
hyper.add_search(
    objective_function,
    search_space,
    n_jobs=4  # Parallel evaluation
)
```

**v5 Code:**
```python
# Use n_jobs in experiment for CV parallelization
experiment = SklearnCvExperiment(
    estimator=model,
    param_grid=param_grid,
    X=X, y=y,
    cv=5,
    n_jobs=4  # Parallel CV
)

# Or implement your own parallel experiment
import joblib
from hyperactive.base import BaseExperiment

class ParallelExperiment(BaseExperiment):
    def __init__(self, base_experiment, n_jobs=1):
        super().__init__()
        self.base_experiment = base_experiment
        self.n_jobs = n_jobs
    
    def _evaluate_batch(self, params_list):
        """Evaluate multiple parameter sets in parallel."""
        def eval_single(params):
            return self.base_experiment._evaluate(params)
        
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(eval_single)(params) for params in params_list
        )
        return results
```

## Integration with ML Frameworks

### Scikit-learn Integration

**v4 Approach:**
```python
def sklearn_objective(para):
    model = SomeSklearnModel(**para)
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()
```

**v5 Approach:**
```python
# Use built-in sklearn integration
from hyperactive.integrations.sklearn import OptCV

opt_cv = OptCV(
    estimator=SomeSklearnModel(),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=5
)

opt_cv.fit(X, y)
```

## Common Migration Issues

### 1. Search Space Definition

**Problem:** v4 used various formats for search spaces.

**Solution:** v5 uses consistent dict format:
```python
# v5 format
param_grid = {
    "continuous_param": [0.1, 0.5, 1.0, 2.0],  # Discrete values
    "categorical_param": ["option1", "option2", "option3"],
    "integer_param": [1, 5, 10, 20, 50]
}
```

### 2. Result Access

**Problem:** Different result access patterns.

**Solution:**
```python
# v4
best_params = hyper.best_para()
all_results = hyper.results_

# v5
best_params = optimizer.solve()
best_score = experiment.score(best_params)[0]

# Access best_params_ attribute after solve()
print("Best found:", optimizer.best_params_)
```

### 3. Progress Monitoring

**Problem:** v5 doesn't have built-in progress bars.

**Solution:** Implement custom monitoring:
```python
class MonitoredExperiment(BaseExperiment):
    def __init__(self, base_experiment):
        super().__init__()
        self.base_experiment = base_experiment
        self.eval_count = 0
        self.best_score = float('-inf')
    
    def _paramnames(self):
        return self.base_experiment.paramnames()
    
    def _evaluate(self, params):
        self.eval_count += 1
        result, metadata = self.base_experiment._evaluate(params)
        
        if result > self.best_score:
            self.best_score = result
            print(f"Evaluation {self.eval_count}: New best score {result:.4f}")
        
        return result, metadata
```

## Testing Your Migration

Create a simple test to verify your migration:

```python
def test_migration():
    """Test that v5 code produces reasonable results."""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from hyperactive.opt.gfo import RandomSearch
    from hyperactive.experiment.integrations import SklearnCvExperiment
    
    # Load test data
    X, y = load_iris(return_X_y=True)
    
    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={"n_estimators": [10, 50, 100]},
        X=X, y=y, cv=3
    )
    
    # Run optimization
    optimizer = RandomSearch(experiment=experiment)
    best_params = optimizer.solve()
    
    # Verify results
    assert best_params is not None
    assert "n_estimators" in best_params
    assert best_params["n_estimators"] in [10, 50, 100]
    
    print("Migration test passed!")

# Run test
test_migration()
```

## Performance Considerations

V5 may have different performance characteristics:

1. **Startup time**: May be slightly higher due to skbase framework
2. **Memory usage**: Generally lower due to more efficient architecture  
3. **Evaluation speed**: Similar or faster for most use cases
4. **Extensibility**: Much better - easier to add custom components

## Getting Help

If you encounter issues during migration:

1. **Check examples**: Look at the example files in v5 repository
2. **Read documentation**: This documentation covers all new features
3. **Compare patterns**: Use this guide's before/after examples
4. **Community support**: Ask questions on GitHub issues

## Summary Checklist

- [ ] Update imports to use specific backend modules
- [ ] Convert objective functions to experiment classes
- [ ] Replace `Hyperactive()` with individual optimizer classes  
- [ ] Update algorithm names and parameter passing
- [ ] Test migrated code with simple examples
- [ ] Update any custom extensions to use new base classes
- [ ] Consider new v5 features like OptCV for sklearn integration

The migration requires some effort, but v5's improved architecture provides much better extensibility and cleaner interfaces for long-term use.