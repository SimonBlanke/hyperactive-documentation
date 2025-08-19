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

### `experiment`
- **Type**: `BaseExperiment`
- **Description**: The experiment object defining the optimization problem

### `n_startup_trials`
- **Type**: `int`
- **Default**: `10`
- **Description**: Number of random trials before GP optimization starts

### `acquisition_func`
- **Type**: `str`
- **Default**: `"ei"` (Expected Improvement)
- **Options**: `"ei"`, `"lcb"` (Lower Confidence Bound), `"pi"` (Probability of Improvement)
- **Description**: Acquisition function for selecting next evaluation point

## Usage Example

```python
--8<-- "optimizers_optuna_gp_optimizer_example.py"
```

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

### Expected Improvement (EI)
- **Best for**: Balanced exploration-exploitation
- **Formula**: $EI(x) = \sigma(x) \cdot \phi(Z) + (\mu(x) - f_{best}) \cdot \Phi(Z)$
- **Use when**: General-purpose optimization

### Lower Confidence Bound (LCB)
- **Best for**: Conservative optimization with uncertainty consideration
- **Formula**: $LCB(x) = \mu(x) - \kappa \cdot \sigma(x)$
- **Use when**: You want to avoid risky evaluations

### Probability of Improvement (PI)
- **Best for**: When you want high probability of improvement
- **Formula**: $PI(x) = \Phi(\frac{\mu(x) - f_{best}}{\sigma(x)})$
- **Use when**: Conservative improvement is preferred

## Advanced Usage

### Custom Acquisition Function

```python
--8<-- "optimizers_optuna_gp_optimizer_example_2.py"
```

### Startup Trials Tuning

```python
--8<-- "optimizers_optuna_gp_optimizer_example_3.py"
```

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

```python
--8<-- "optimizers_optuna_gp_optimizer_example_4.py"
```

### Scientific Simulation Parameters

```python
--8<-- "optimizers_optuna_gp_optimizer_example_5.py"
```

### Model Regularization

```python
--8<-- "optimizers_optuna_gp_optimizer_example_6.py"
```

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