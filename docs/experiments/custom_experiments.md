# Custom Experiments

## Introduction

Creating custom experiments in Hyperactive allows you to optimize any objective function or complex system. By inheriting from `BaseExperiment`, you can define domain-specific optimization problems that go beyond standard machine learning hyperparameter tuning.

## BaseExperiment Overview

All experiments inherit from `BaseExperiment`, which provides:
- **Parameter space definition**: Define what parameters to optimize
- **Objective evaluation**: Implement your evaluation logic
- **Metadata handling**: Return additional information about evaluations
- **Tag system**: Specify experiment properties and requirements

## Basic Custom Experiment
Here's a minimal example showing how to define a custom experiment by subclassing `BaseExperiment`:

```python
from hyperactive.experiment import BaseExperiment

class MyParabolaExperiment(BaseExperiment):
    def __init__(self, a=1.0, b=0.0, c=0.0):
        self.a, self.b, self.c = a, b, c
        super().__init__()

    def _paramnames(self):
        return ["x"]

    def _evaluate(self, params):
        x = params["x"]
        value = -(self.a * x * x + self.b * x + self.c)  # maximize
        return value, {"x": x}

# usage
from hyperactive.opt.gfo import HillClimbing

exp = MyParabolaExperiment(a=1.0, b=0.0, c=0.0)
opt = HillClimbing(experiment=exp)
best = opt.solve()
```

## Advanced Custom Experiment
Multi-parameter spaces and metadata logging example:

```python
class WeightedSumExperiment(BaseExperiment):
    def __init__(self, w1=1.0, w2=0.5):
        self.w1, self.w2 = w1, w2
        super().__init__()

    def _paramnames(self):
        return ["x", "y"]

    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        value = -(self.w1 * x**2 + self.w2 * y**2)
        return value, {"norm": (x**2 + y**2) ** 0.5}
```

## Machine Learning Custom Experiment
If you need full control beyond the built-in integrations, wrap your ML workflow directly:

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone

class SVCExperiment(BaseExperiment):
    def __init__(self, estimator, X, y, scoring=None, cv=None):
        self.estimator, self.X, self.y = estimator, X, y
        self.cv = KFold(n_splits=3, shuffle=True) if cv is None else cv
        self.scoring = scoring
        super().__init__()

    def _paramnames(self):
        return list(self.estimator.get_params().keys())

    def _evaluate(self, params):
        est = clone(self.estimator).set_params(**params)
        scores = cross_val_score(est, self.X, self.y, scoring=self.scoring, cv=self.cv)
        return scores.mean(), {"cv_scores": scores}
```

## Multi-Objective Custom Experiment



## Simulation-Based Experiment



## Experiment with External Data



## FunctionExperiment (wrap a callable)

You can wrap a plain Python function without subclassing by using `FunctionExperiment`:

```python
from hyperactive.experiment.func import FunctionExperiment
from hyperactive.opt.gfo import RandomSearch

def parabola(x):
    return -(x - 3) ** 2, {}

exp = FunctionExperiment(parabola, parametrization="dict")
opt = RandomSearch(experiment=exp)
best = opt.solve()
```

`FunctionExperiment` also supports kwargs-style functions:

```python
def f(x, y):
    return -(x**2 + y**2), {}

exp = FunctionExperiment(f, parametrization="kwargs")
opt = RandomSearch(experiment=exp)
best = opt.solve()
```
## Best Practices for Custom Experiments

### Error Handling


### Parameter Validation


### Caching and Memoization


## Testing Custom Experiments



## Integration with Different Optimizers



## References

- Hyperactive base classes: BaseExperiment and BaseOptimizer documentation
- Python subprocess module: [https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html)
- Scikit-learn custom scorers: [https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions)
