# Sktime Classification

## Introduction

The sktime classification integration enables hyperparameter optimization of time series classifiers using proper cross‑validation and scoring. Use it via:

- Experiment: `SktimeClassificationExperiment`
- Estimator interface: `hyperactive.integrations.sktime.TSCOptCV`

## Key Features

- Works with sktime classifiers and pipelines
- Flexible CV strategy and scoring coercion
- Clean separation between optimizer and experiment

## Basic Usage

```python
from hyperactive.experiment.integrations import SktimeClassificationExperiment
from hyperactive.opt import RandomSearchSk

# prepare sktime data X, y and a classifier `clf`
exp = SktimeClassificationExperiment(estimator=clf, X=X, y=y, cv=3)
opt = RandomSearchSk(
    param_distributions={"param": ["a", "b"]},
    n_iter=20,
    experiment=exp,
)
best_params = opt.solve()
```

## With TSCOptCV

```python
from hyperactive.integrations.sktime import TSCOptCV
from hyperactive.opt import TPEOptimizer

tsc = TSCOptCV(
    estimator=clf,
    optimizer=TPEOptimizer(param_space={"param": (0.1, 1.0)}, n_trials=30),
    cv=3,
)
tsc.fit(X, y)
print(tsc.best_params_)
```

## Notes

- Choose scoring appropriate for your task (string or callable)
- For forecasting (not classification), use `SktimeForecastingExperiment`

