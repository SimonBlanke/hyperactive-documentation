# Sktime Forecasting

## Introduction

The Sktime Forecasting integration provides specialized optimization capabilities for time series forecasting models. This integration handles the unique challenges of time series data, including proper temporal cross-validation and forecasting-specific parameter spaces.

## Key Features

- **Time series cross-validation**: Proper temporal validation strategies
- **Forecasting model support**: Integration with sktime forecasters
- **Horizon optimization**: Multi-step ahead forecasting optimization
- **Seasonal parameter tuning**: Seasonal model parameter optimization

## Basic Usage

```python
from hyperactive.experiment.integrations import SktimeForecastingExperiment
from hyperactive.opt.gfo import BayesianOptimizer
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.datasets import load_airline

# Load time series data
y = load_airline()

# Define parameter space for ARIMA
param_grid = {
    "order": [(1,1,1), (2,1,1), (1,1,2), (2,1,2)],
    "seasonal_order": [(1,1,1,12), (0,1,1,12), (1,0,1,12)]
}

# Create forecasting experiment
experiment = SktimeForecastingExperiment(
    forecaster=ARIMA(),
    param_grid=param_grid,
    y=y,
    cv=5,  # Time series CV
    scoring="mean_squared_error"
)

# Optimize
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters:", best_params)
```

## Time Series Cross-Validation

```python
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_selection import SlidingWindowSplitter

# Sliding window cross-validation
cv = SlidingWindowSplitter(
    window_length=36,  # Training window
    step_length=12,    # Step between windows
    fh=[1, 2, 3, 6, 12]  # Forecasting horizons
)

experiment = SktimeForecastingExperiment(
    forecaster=ThetaForecaster(),
    param_grid={"sp": [12, 24], "deseasonalize": [True, False]},
    y=y,
    cv=cv,
    scoring="mean_absolute_percentage_error"
)
```

## Seasonal Model Optimization

```python
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

# Seasonal exponential smoothing
param_grid = {
    "trend": [None, "add", "mul"],
    "seasonal": [None, "add", "mul"], 
    "sp": [12, 24, 52],  # Seasonal periods
    "damped_trend": [True, False]
}

experiment = SktimeForecastingExperiment(
    forecaster=ExponentialSmoothing(),
    param_grid=param_grid,
    y=y,
    cv=SlidingWindowSplitter(window_length=100, step_length=24),
    scoring="mean_absolute_error"
)
```

## Multi-Step Forecasting

```python
# Optimize for different forecasting horizons
horizons = [1, 3, 6, 12]
results = {}

for h in horizons:
    cv = SlidingWindowSplitter(
        window_length=60,
        step_length=12,
        fh=list(range(1, h+1))  # 1 to h steps ahead
    )
    
    experiment = SktimeForecastingExperiment(
        forecaster=ARIMA(),
        param_grid=param_grid,
        y=y,
        cv=cv,
        scoring="mean_squared_error"
    )
    
    optimizer = BayesianOptimizer(experiment=experiment)
    best_params = optimizer.solve()
    results[h] = best_params
    
    print(f"Horizon {h}: {best_params}")
```

## References

- Sktime documentation: [https://www.sktime.org/](https://www.sktime.org/)
- Time series cross-validation: [https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.temporal_train_test_split.html](https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.forecasting.model_selection.temporal_train_test_split.html)
