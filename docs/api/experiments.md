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



### Machine Learning Custom Experiment



### Experiment with External Dependencies



## Advanced Usage Patterns

### Multi-Objective Experiments



### Experiment Composition



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
