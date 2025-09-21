# Experiments and Integrations

Hyperactive v5 introduces an experiment-based approach where optimization problems are defined as experiment objects. This provides better modularity and makes it easier to create reusable optimization setups.

## Experiment Types

### Built-in Benchmark Experiments

These experiments implement standard optimization test functions commonly used for algorithm benchmarking.

#### Ackley Function

The Ackley function is a widely used multimodal test function.



**Properties:**
- **Global minimum**: f(0, 0, ..., 0) = 0
- **Multimodal**: Many local minima
- **Separable**: No
- **Differentiable**: Yes

#### Sphere Function

Simple quadratic function for testing convergence.



**Properties:**
- **Global minimum**: f(0, 0, ..., 0) = 0  
- **Unimodal**: Single optimum
- **Separable**: Yes
- **Differentiable**: Yes

#### Parabola Function

Basic parabolic test function.



### Framework Integration Experiments  

These experiments integrate optimization with machine learning frameworks.

## Scikit-learn Integration

### SklearnCvExperiment

Cross-validation based hyperparameter optimization for scikit-learn estimators.



**Parameters:**
- **estimator**: Scikit-learn estimator to optimize
- **X, y**: Training data arrays
- **cv**: Cross-validation strategy (int or CV object). If `None`, defaults to `KFold(n_splits=3, shuffle=True)`.
- **scoring**: Scoring function or string. If `None`, uses the estimator’s default `score`.

Notes:
- Parallelization is controlled by the optimizer via `backend` and `backend_params` (e.g., `GridSearchSk`, `RandomSearchSk`), not by `SklearnCvExperiment`.
- Define your search space (e.g., grids, distributions) on the optimizer; the experiment only encapsulates data, estimator, CV, and scoring.

### OptCV - Scikit-learn Compatible Optimizer

For users who prefer the familiar scikit-learn interface, `OptCV` provides a drop-in replacement for `GridSearchCV` and `RandomizedSearchCV`.



**Key Features:**
- **Familiar Interface**: Drop-in replacement for GridSearchCV
- **Any Optimizer**: Use any Hyperactive optimizer
- **Refit Support**: Automatically refit best estimator
- **Full sklearn Compatibility**: Supports all sklearn estimator methods

## Time Series Integration

### SktimeForecastingExperiment

Specialized experiment for time series forecasting with sktime.



## Creating Custom Experiments

You can create custom experiments by inheriting from `BaseExperiment`.

### Basic Custom Experiment
Minimal subclass implementing `_paramnames` and `_evaluate`:

```python
from hyperactive.base import BaseExperiment

class WeightedParabola(BaseExperiment):
    def __init__(self, w=1.0):
        self.w = w
        super().__init__()

    def _paramnames(self):
        return ["x"]

    def _evaluate(self, params):  # maximize
        x = params["x"]
        value = -(self.w * (x - 2) ** 2)
        return value, {"x": x}

exp = WeightedParabola(w=0.5)
```

### Machine Learning Custom Experiment
Wrap a full ML workflow when built-in integrations are not sufficient:

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import clone
from hyperactive.base import BaseExperiment

class SVCExp(BaseExperiment):
    def __init__(self, estimator, X, y, scoring=None, cv=None):
        self.estimator, self.X, self.y = estimator, X, y
        self.scoring = scoring
        self.cv = KFold(n_splits=3, shuffle=True) if cv is None else cv
        super().__init__()

    def _paramnames(self):
        return list(self.estimator.get_params().keys())

    def _evaluate(self, params):
        est = clone(self.estimator).set_params(**params)
        scores = cross_val_score(est, self.X, self.y, scoring=self.scoring, cv=self.cv)
        return scores.mean(), {"cv_scores": scores}
```

### Experiment with External Dependencies
Experiments can call external libraries or services. Keep them robust and report
useful metadata:

```python
import time
from hyperactive.base import BaseExperiment

class SimulatedServiceExp(BaseExperiment):
    def _paramnames(self):
        return ["alpha", "beta"]

    def _evaluate(self, params):
        start = time.time()
        # ... call external code or simulate compute ...
        score = -(params["alpha"] - 1) ** 2 - (params["beta"] - 2) ** 2
        return score, {"elapsed_s": time.time() - start}
```

## Advanced Usage Patterns

### Multi-Objective Experiments
Hyperactive’s `BaseExperiment` is single-objective: `evaluate` returns one float.
If you have multiple objectives, use one of the following strategies:

- Scalarization: combine multiple metrics (e.g., weighted sum) into one score.
- Lexicographic or constrained scoring: encode secondary objectives in metadata
  and penalize the primary score when constraints are violated.
- Optuna multi-objective optimizers: available in Hyperactive’s Optuna backend
  (e.g., NSGA-II/III). Current experiments still return a scalar score; use
  scalarization to define the objective while tracking additional metrics in
  metadata for analysis.

### Experiment Composition
Compose experiments by delegating to inner experiments, or by building staged
pipelines. Example: a meta-experiment that adds a penalty term to any wrapped
experiment’s score.

```python
class PenalizedExperiment(BaseExperiment):
    def __init__(self, base_exp, penalty=0.0):
        self.base_exp = base_exp
        self.penalty = penalty
        super().__init__()

    def _paramnames(self):
        return self.base_exp.paramnames()

    def _evaluate(self, params):
        val, meta = self.base_exp.evaluate(params)
        return val - self.penalty, {**meta, "penalty": self.penalty}
```

## Best Practices

### Experiment Design
1. **Clear parameter names**: Use descriptive parameter names
2. **Proper bounds checking**: Validate parameters in `_evaluate`
3. **Error handling**: Return meaningful scores for invalid parameters  
4. **Metadata logging**: Include useful information in metadata dict
5. **Reproducibility**: Set random seeds where appropriate

### Performance Optimization
1. **Caching**: Cache expensive computations when possible
2. **Early stopping**: Return early for obviously poor parameters
3. **Parallel evaluation**: Use optimizer `backend`/`backend_params` for CV parallelization
4. **Memory management**: Clean up resources in long-running experiments

### Integration Guidelines
1. **Framework compatibility**: Follow framework conventions
2. **Type hints**: Use proper type annotations
3. **Documentation**: Document parameter spaces and expected behavior
4. **Testing**: Include unit tests for custom experiments
