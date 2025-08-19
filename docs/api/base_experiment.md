
## BaseExperiment

The `BaseExperiment` class defines the optimization problem and objective function.

### Class Signature

```python
--8<-- "api_base_classes_example_4.py"
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
--8<-- "api_base_classes_example_5.py"
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
--8<-- "api_base_classes_example_6.py"
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
--8<-- "api_base_classes_example_7.py"
```

### Custom Experiment Creation

```python
--8<-- "api_base_classes_example_8.py"
```

### Tag Inspection

```python
--8<-- "api_base_classes_example_9.py"
```