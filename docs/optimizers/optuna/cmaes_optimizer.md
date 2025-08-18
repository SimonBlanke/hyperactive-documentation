# CMA-ES Optimizer

## Introduction

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a state-of-the-art evolutionary algorithm for continuous optimization problems. It's particularly effective for non-convex, multimodal optimization landscapes and is considered one of the best general-purpose continuous optimizers available.

## About the Implementation

CMA-ES maintains a multivariate normal distribution of candidate solutions and adapts both the mean and covariance matrix of this distribution based on the success of previous generations. The algorithm excels at handling:

- **Ill-conditioned problems**: Automatically adapts to the problem's scale and correlation structure
- **Multimodal landscapes**: Can escape local optima through its population-based approach
- **Noisy objectives**: Robust to evaluation noise through population sampling

Key features:
- **Scale-invariant**: Adapts to different parameter scales automatically
- **Rotation-invariant**: Learns parameter correlations and dependencies
- **Self-adaptive**: No manual parameter tuning required for most problems
- **Proven convergence**: Strong theoretical foundation with convergence guarantees

## Parameters

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: The experiment object defining the optimization problem

### `sigma0`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Initial step size (standard deviation). Should be roughly 1/4 to 1/2 of the search domain width.

### `population_size`
- **Type**: `int` or `None`
- **Default**: `None` (auto-calculated)
- **Description**: Population size. If None, uses `4 + floor(3 * log(n_dimensions))`

### `seed`
- **Type**: `int` or `None`
- **Default**: `None`
- **Description**: Random seed for reproducible results

## Usage Example

```python
from hyperactive.opt.optuna import CmaEsOptimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes

# Load dataset
X, y = load_diabetes(return_X_y=True)

# Define continuous search space (CMA-ES works best with continuous parameters)
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "epsilon": [0.001, 0.01, 0.1, 1.0],
    "gamma": [0.001, 0.01, 0.1, 1.0]  # assuming RBF kernel
}

# Create experiment
experiment = SklearnCvExperiment(
    estimator=SVR(kernel='rbf'),
    param_grid=param_grid,
    X=X, y=y,
    cv=5,
    scoring="neg_mean_squared_error"
)

# Create CMA-ES optimizer
optimizer = CmaEsOptimizer(
    experiment=experiment,
    sigma0=0.5,  # Initial step size
    population_size=20  # Custom population size
)

# Run optimization
best_params = optimizer.solve()
print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
```

## When to Use CMA-ES

**Best for:**
- **Continuous optimization**: Excels with real-valued parameters
- **Expensive evaluations**: Very sample-efficient for complex landscapes
- **Unknown problem structure**: Adapts automatically to problem characteristics
- **Multimodal problems**: Can escape local optima effectively
- **Medium to high dimensions**: Scales well up to 100+ dimensions

**Consider alternatives if:**
- **Discrete/categorical parameters**: Use TPE or other discrete optimizers
- **Very cheap evaluations**: Random search might be sufficient
- **Low dimensions (<5)**: Simpler methods might be faster
- **Large populations not feasible**: CMA-ES needs multiple evaluations per iteration

## Comparison with Other Algorithms

| Algorithm | Continuous Opt | Discrete Opt | Sample Efficiency | Scalability |
|-----------|---------------|--------------|-------------------|-------------|
| CMA-ES | Excellent | Poor | Very High | Good (to ~100D) |
| TPE | Good | Excellent | High | Good |
| Bayesian Opt | Excellent | Poor | High | Moderate |
| Random Search | Good | Good | Low | Excellent |

## Advanced Usage

### Custom Population Size

```python
# For high-dimensional problems
n_dimensions = len(param_grid)
custom_pop_size = 4 + int(3 * np.log(n_dimensions))

optimizer = CmaEsOptimizer(
    experiment=experiment,
    population_size=custom_pop_size * 2  # Larger population for difficult problems
)
```

### Step Size Tuning

```python
# For narrow search ranges
optimizer = CmaEsOptimizer(
    experiment=experiment,
    sigma0=0.1  # Smaller steps for fine-tuning
)

# For wide search ranges
optimizer = CmaEsOptimizer(
    experiment=experiment,
    sigma0=2.0  # Larger steps for exploration
)
```

### Reproducible Results

```python
optimizer = CmaEsOptimizer(
    experiment=experiment,
    seed=42  # For reproducible optimization runs
)
```

## Mathematical Background

CMA-ES maintains a multivariate normal distribution $\mathcal{N}(m, \sigma^2 C)$ where:

- $m$ is the mean (center of the search distribution)
- $\sigma$ is the step size (global scaling)
- $C$ is the covariance matrix (shape and orientation)

The algorithm updates these parameters based on the success of sampled points:

1. **Mean update**: Move toward successful solutions
2. **Step size adaptation**: Increase/decrease based on success rate
3. **Covariance adaptation**: Learn problem structure and dependencies

## Performance Tips

1. **Parameter scaling**: Ensure all parameters have similar scales (0-1 or similar ranges)
2. **Step size**: Start with sigma0 ≈ 0.25 * (parameter_range)
3. **Population size**: Use default unless you have specific requirements
4. **Evaluation budget**: CMA-ES typically needs 100+ evaluations to be effective
5. **Continuous parameters**: Works best when parameters can be treated as continuous

## Common Use Cases

### Neural Network Hyperparameters

```python
param_grid = {
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64, 128, 256],  # Can be treated as continuous
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
}
```

### Regression Model Tuning

```python
param_grid = {
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    "l1_ratio": [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0],
    "max_iter": [100, 500, 1000, 2000, 5000]
}
```

## References

- Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies.
- Hansen, N. (2016). The CMA evolution strategy: A tutorial.
- Optuna CMA-ES documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)