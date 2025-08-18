# Base Classes

Hyperactive v5 is built on a foundation of base classes that provide the core functionality for optimization algorithms and experiments. These classes use the skbase framework and provide a consistent interface across all components.

## BaseOptimizer

The `BaseOptimizer` class is the foundation for all optimization algorithms in Hyperactive v5.

### Class Signature

```python
from hyperactive.base import BaseOptimizer

class BaseOptimizer(BaseObject):
    def __init__(self):
        # Initialize with experiment parameter
        pass
```

### Key Methods

#### `solve()`

Run the optimization search process to maximize the experiment's score.

**Returns:**
- `best_params` (dict): The best parameters found during optimization

**Example:**
```python
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment

# Create experiment and optimizer
experiment = SklearnCvExperiment(...)
optimizer = BayesianOptimizer(experiment=experiment)

# Run optimization
best_params = optimizer.solve()
```

#### `get_search_config()`

Get the search configuration parameters for the optimizer.

**Returns:**
- `dict`: Configuration dictionary with optimizer-specific parameters

#### `get_experiment()`

Get the associated experiment object.

**Returns:**
- `BaseExperiment`: The experiment to optimize parameters for

### Properties

#### `best_params_`

The best parameters found after calling `solve()`.

**Type:** `dict`

### Tags System

BaseOptimizer uses a comprehensive tag system to provide metadata about algorithm properties:

```python
_tags = {
    "object_type": "optimizer",
    "python_dependencies": None,
    "info:name": str,  # Algorithm name
    "info:local_vs_global": str,  # "local", "mixed", "global"
    "info:explore_vs_exploit": str,  # "explore", "exploit", "mixed"  
    "info:compute": str,  # "low", "middle", "high"
}
```

**Tag Meanings:**
- **local_vs_global**: Whether the algorithm focuses on local or global search
- **explore_vs_exploit**: Balance between exploration and exploitation
- **compute**: Computational cost category

## BaseExperiment

The `BaseExperiment` class defines the optimization problem and objective function.

### Class Signature

```python
from hyperactive.base import BaseExperiment

class BaseExperiment(BaseObject):
    def __init__(self):
        # Initialize experiment
        pass
```

### Key Methods

#### `evaluate(params)`

Evaluate the given parameters and return the objective value.

**Parameters:**
- `params` (dict): Parameters to evaluate

**Returns:**
- `float`: The objective value
- `dict`: Additional metadata about the evaluation

**Example:**
```python
# Custom experiment example
class MyExperiment(BaseExperiment):
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        # Your objective function
        score = -(params["x"]**2 + params["y"]**2)  # Minimize sphere function
        return score, {}

experiment = MyExperiment()
score, metadata = experiment.evaluate({"x": 1.0, "y": 2.0})
```

#### `score(params)`

Score the parameters with sign adjustment for maximization.

**Parameters:**
- `params` (dict): Parameters to score

**Returns:**
- `float`: Score adjusted so higher is always better
- `dict`: Additional metadata

#### `paramnames()`

Get the parameter names for the search space.

**Returns:**
- `list`: List of parameter names

### Tags System

BaseExperiment uses tags to specify optimization properties:

```python
_tags = {
    "object_type": "experiment",
    "python_dependencies": None,
    "property:randomness": str,  # "random" or "deterministic"
    "property:higher_or_lower_is_better": str,  # "higher", "lower", "mixed"
}
```

### Abstract Methods

When creating custom experiments, you must implement:

#### `_paramnames()`

Return the parameter names for the search space.

#### `_evaluate(params)`

Implement the actual objective function evaluation.

## Usage Patterns

### Basic Optimizer Usage

```python
from hyperactive.opt.gfo import RandomSearch
from hyperactive.experiment.bench import Sphere

# Create experiment with parameter space
experiment = Sphere(dimensions=2, bounds=(-5, 5))

# Create optimizer
optimizer = RandomSearch(experiment=experiment)

# Run optimization
best_params = optimizer.solve()
```

### Custom Experiment Creation

```python
from hyperactive.base import BaseExperiment

class CustomExperiment(BaseExperiment):
    def __init__(self, my_param=1.0):
        super().__init__()
        self.my_param = my_param
    
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        # Your custom objective function
        result = some_expensive_computation(params, self.my_param)
        return result, {"computation_time": time.time()}

# Use custom experiment
experiment = CustomExperiment(my_param=2.0)
optimizer = SomeOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

### Tag Inspection

```python
from hyperactive.opt.gfo import BayesianOptimizer

# Check optimizer properties
optimizer = BayesianOptimizer(experiment=experiment)
print("Algorithm type:", optimizer.get_tag("info:local_vs_global"))
print("Computational cost:", optimizer.get_tag("info:compute"))
print("Exploration style:", optimizer.get_tag("info:explore_vs_exploit"))
```