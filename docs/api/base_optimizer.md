
## BaseOptimizer

The `BaseOptimizer` class is the foundation for all optimization algorithms in Hyperactive v5.

### Class Signature

```python
class BaseOptimizer(BaseObject):
    def get_search_config(self) -> dict: ...
    def get_experiment(self) -> "BaseExperiment": ...
    def solve(self) -> dict: ...
    def _solve(self, experiment, *args, **kwargs) -> dict: ...  # to override
```

### Key Methods

#### `solve()`
Run the optimization to maximize the experiment’s score (higher-is-better convention). Sets `best_params_` and may set additional attributes depending on the optimizer.

**Returns:**
- `dict`: Best parameters found

#### `get_search_config()`
Returns the optimizer’s configuration, excluding the experiment object (useful to introspect/store settings).

**Returns:**
- `dict`: Optimizer-specific parameters

#### `get_experiment()`
Returns the associated experiment. If a plain callable was provided, it is wrapped as a `FunctionExperiment` automatically.

**Returns:**
- `BaseExperiment`: Experiment to optimize

### Properties

- `best_params_` (`dict`): Best parameters found (set by `solve()`)
- `best_score_` (`float`, optional): Signed score per higher-is-better convention (set by some optimizers, e.g., `GridSearchSk`, `RandomSearchSk`)
- `best_index_` (`int`, optional): Index of the best candidate in the evaluated sequence (set by some optimizers)

### Minimal Example

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt import GridSearchSk

exp = SklearnCvExperiment(...)
opt = GridSearchSk(param_grid={"C": [0.1, 1, 10]}, experiment=exp)
best = opt.solve()
```

### Tags System

BaseOptimizer uses tags to describe algorithm properties:

- `info:local_vs_global`: local | mixed | global
- `info:explore_vs_exploit`: explore | exploit | mixed
- `info:compute`: low | middle | high
