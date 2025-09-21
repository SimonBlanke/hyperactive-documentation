
## BaseExperiment

The `BaseExperiment` class defines the optimization problem and objective function.

### Class Signature

```python
from skbase.base import BaseObject

class BaseExperiment(BaseObject):
    # ---- public API ----
    def paramnames(self) -> list[str] | None: ...
    def evaluate(self, params: dict) -> tuple[float, dict]: ...
    def score(self, params: dict) -> tuple[float, dict]: ...  # higher-is-better
    def __call__(self, params: dict) -> float: ...  # shorthand for score(...)[0]

    # ---- to implement in subclasses ----
    def _paramnames(self) -> list[str] | None: ...
    def _evaluate(self, params: dict) -> tuple[float, dict]: ...

    # ---- key tags ----
    # "object_type": "experiment"
    # "python_dependencies": None | list[str]
    # "property:randomness": "random" | "deterministic"
    # "property:higher_or_lower_is_better": "higher" | "lower" | "mixed"
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
from hyperactive.experiment.func import FunctionExperiment

def f(opt):  # maximize
    x = opt["x"]
    return -(x - 3)**2, {"x": x}

exp = FunctionExperiment(f)
value, meta = exp.evaluate({"x": 1})  # raw objective value
score, meta = exp.score({"x": 1})     # sign-adjusted (higher-is-better)
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
- `list[str] | None`: Parameter names. If `None`, any parameter keys are accepted.
- `__call__(params)` is provided as shorthand for `score(params)` and returns only the score (float).
- Score sign follows the tag `property:higher_or_lower_is_better`:
  - `"higher"`: score equals evaluate (higher-is-better)
  - `"lower"`: score is `-evaluate` (converted to higher-is-better)
  - `"mixed"`: not supported by default; override `score` for custom behavior

### Tags System

BaseExperiment uses tags to specify optimization properties:

- `object_type`: always `"experiment"`.
- `python_dependencies`: optional list of extra dependency names.
- `property:randomness`: `"deterministic"` or `"random"` (stochastic evaluation).
- `property:higher_or_lower_is_better`: controls the sign in `score(...)`.
  - `"higher"`: `score == evaluate`.
  - `"lower"`: `score == -evaluate`.
  - `"mixed"`: unsupported by default; override `score` for custom behavior.

Read/modify tags via `get_tag`/`set_tags`:

```python
hib = exp.get_tag("property:higher_or_lower_is_better", default="higher")
exp.set_tags(**{"property:randomness": "deterministic"})
```

### Abstract Methods

When creating custom experiments, you must implement:

#### `_paramnames()`

Return the parameter names for the search space.

#### `_evaluate(params)`

Implement the actual objective function evaluation.

## Usage Patterns

### Basic Optimizer Usage

Use any optimizer with an experiment. Below, a small benchmark wrapped by
`FunctionExperiment` is optimized using random search.

```python
from hyperactive.experiment.func import FunctionExperiment
from hyperactive.opt.gfo import RandomSearch

def sphere(opt):  # maximize
    x, y = opt["x"], opt["y"]
    return -(x**2 + y**2), {"r": (x**2 + y**2) ** 0.5}

exp = FunctionExperiment(sphere)
opt = RandomSearch(experiment=exp)
best = opt.solve()
```

### Custom Experiment Creation

Subclass `BaseExperiment` when you need full control over evaluation or metadata.

```python
from hyperactive.base import BaseExperiment

class ParabolaExp(BaseExperiment):
    def _paramnames(self): # optional
        return ["x"]

    def evaluate(self, params):  # maximize
        x = params["x"]
        return -(x - 2)**2, {"x": x}

exp = ParabolaExp()
value, meta = exp.evaluate({"x": 0.5})
```

### Tag Inspection

Inspect how an experiment is scored and whether it is stochastic:

```python
hib = exp.get_tag("property:higher_or_lower_is_better")  # "higher" | "lower"
rand = exp.get_tag("property:randomness")                # "random" | "deterministic"
```
