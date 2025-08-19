
## BaseOptimizer

The `BaseOptimizer` class is the foundation for all optimization algorithms in Hyperactive v5.

### Class Signature

```python
--8<-- "api_base_classes_example.py"
```

### Key Methods

#### `solve()`

Run the optimization search process to maximize the experiment's score.

**Returns:**
- `best_params` (dict): The best parameters found during optimization

**Example:**
```python
--8<-- "api_base_classes_example_2.py"
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
--8<-- "api_base_classes_example_3.py"
```

**Tag Meanings:**
- **local_vs_global**: Whether the algorithm focuses on local or global search
- **explore_vs_exploit**: Balance between exploration and exploitation
- **compute**: Computational cost category