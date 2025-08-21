
## BaseExperiment

The `BaseExperiment` class defines the optimization problem and objective function.

### Class Signature



### Key Methods

#### `evaluate(params)`

Evaluate the given parameters and return the objective value.

**Parameters:**
- `params` (dict): Parameters to evaluate

**Returns:**
- `float`: The objective value
- `dict`: Additional metadata about the evaluation

**Example:**


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



### Abstract Methods

When creating custom experiments, you must implement:

#### `_paramnames()`

Return the parameter names for the search space.

#### `_evaluate(params)`

Implement the actual objective function evaluation.

## Usage Patterns

### Basic Optimizer Usage



### Custom Experiment Creation



### Tag Inspection

