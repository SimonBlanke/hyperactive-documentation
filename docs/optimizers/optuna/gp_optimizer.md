# GP Optimizer

## Introduction

The Gaussian Process (GP) Optimizer implements Bayesian optimization using Gaussian processes as surrogate models. It's a sophisticated algorithm that builds a probabilistic model of the objective function and uses this model to make intelligent decisions about where to evaluate next.

## About the Implementation

Gaussian Process optimization maintains a probabilistic model of the objective function and uses acquisition functions to balance exploration and exploitation. The GP provides both predictions and uncertainty estimates, making it ideal for expensive function evaluations.

Key features:
- **Uncertainty quantification**: Provides confidence intervals for predictions
- **Sample efficiency**: Excellent for expensive evaluations
- **Principled exploration**: Uses uncertainty to guide search
- **Non-parametric**: Adapts to complex function shapes

## Parameters

### Common (via Base Optuna Adapter)
- `param_space` (dict): parameter space; tuples/lists treated as ranges/choices
- `n_trials` (int): number of trials to run
- `initialize` (dict | None): optional warm start/grid/vertices/random init
- `early_stopping` (int | None): stop if no improvement after N trials
- `max_score` (float | None): stop when reaching threshold
- `experiment` (BaseExperiment): the experiment to optimize

### `n_startup_trials`
- **Type**: `int`
- **Default**: `10`
- **Description**: Number of random trials before GP optimization starts

### `deterministic_objective`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether the objective function is deterministic (passes through to Optuna's GPSampler).

### `random_state`
- **Type**: `int | None`
- **Default**: `None`
- **Description**: Seed for reproducibility (sets `seed` in the underlying GPSampler).

## Usage Example



## When to Use GP Optimizer

**Best for:**
- **Expensive evaluations**: When each evaluation takes significant time/resources
- **Continuous parameters**: Works best with real-valued parameters  
- **Smooth objectives**: Most effective on smooth or moderately noisy functions
- **Low to moderate dimensions**: Typically <20 parameters
- **Sample-efficient optimization**: When you have limited evaluation budget

**Consider alternatives if:**
- **Many categorical parameters**: TPE might be better
- **High dimensions**: CMA-ES or TPE might scale better
- **Very noisy objectives**: More robust methods might be needed
- **Cheap evaluations**: Random search might be sufficient

## Acquisition Functions

Note: The specific acquisition function is handled internally by the Optuna sampler used by this adapter and is not user-configurable via this API. The following concepts are provided for background only.

### Expected Improvement (EI)
Balanced exploration-exploitation using improvement probability and magnitude.

### Lower Confidence Bound (LCB)
Conservative trade-off between mean prediction and uncertainty.

### Probability of Improvement (PI)
Focuses on points with high probability of improving over current best.

## Advanced Usage

### Custom Acquisition Function



### Startup Trials Tuning



## Comparison with Other Algorithms

| Algorithm | Sample Efficiency | Continuous | Categorical | Scalability | Uncertainty |
|-----------|------------------|------------|-------------|-------------|-------------|
| GP | Very High | Excellent | Limited | Poor (>20D) | Excellent |
| TPE | High | Good | Excellent | Good | Good |
| CMA-ES | High | Excellent | Poor | Good | None |
| Random | Low | Good | Good | Excellent | None |

## Mathematical Background

Gaussian Process regression assumes the objective function $f$ follows a GP prior:

$$f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))$$

where:
- $\mu(x)$ is the mean function (often assumed to be 0)
- $k(x, x')$ is the covariance (kernel) function

Given observations $\{(x_i, y_i)\}_{i=1}^n$, the posterior predictive distribution is:

$$f(x) | \mathcal{D} \sim \mathcal{N}(\mu_n(x), \sigma_n^2(x))$$

The acquisition function uses both $\mu_n(x)$ (predicted value) and $\sigma_n(x)$ (uncertainty) to select the next evaluation point.

## Performance Tips

1. **Parameter scaling**: Normalize parameters to similar scales (0-1)
2. **Startup trials**: Use 10-20% of total budget for random initialization
3. **Kernel choice**: Default RBF kernel works well for most smooth functions
4. **Batch evaluation**: GP optimization is inherently sequential
5. **Noise handling**: Add noise parameter if objective is noisy

## Common Use Cases

### Neural Network Hyperparameters



### Scientific Simulation Parameters



### Model Regularization



## Limitations

1. **Computational cost**: GP inference scales as O(n³) with number of observations
2. **Categorical parameters**: Not naturally handled (requires encoding)
3. **High dimensions**: Performance degrades beyond ~20 parameters
4. **Non-stationary functions**: Standard GP assumes stationarity
5. **Discrete parameters**: Requires careful handling

## Integration with Experimental Design

GP optimization naturally integrates with experimental design principles:

- **Sequential design**: Each evaluation informs the next
- **Uncertainty quantification**: Provides confidence in predictions
- **Active learning**: Focuses evaluations where learning is maximal
- **Robust optimization**: Can incorporate noise models

## References

- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms.
- Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes for machine learning.
- Optuna GP documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
