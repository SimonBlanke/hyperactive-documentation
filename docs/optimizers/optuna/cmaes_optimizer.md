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



### Step Size Tuning



### Reproducible Results



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



### Regression Model Tuning



## References

- Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies.
- Hansen, N. (2016). The CMA evolution strategy: A tutorial.
- Optuna CMA-ES documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)