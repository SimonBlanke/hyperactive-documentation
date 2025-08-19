# Ackley Function

## Introduction

The Ackley function is a widely used multimodal benchmark function for testing optimization algorithms. It features many local minima and a single global minimum, making it an excellent test for an algorithm's ability to avoid getting trapped in local optima.

## Mathematical Definition

The Ackley function is defined as:

$$f(x) = -a \exp\left(-b\sqrt{rac{1}{d}\sum_{i=1}^d x_i^2}
ight) - \exp\left(rac{1}{d}\sum_{i=1}^d \cos(c x_i)
ight) + a + e$$

Where:
- $a = 20$ (amplitude of exponential term)
- $b = 0.2$ (exponential decay factor)  
- $c = 2\pi$ (frequency of cosine term)
- $d$ = dimensionality
- $e$ = Euler's number

## Properties

- **Global minimum**: $f(0, 0, ..., 0) = 0$
- **Search domain**: Typically $[-32.768, 32.768]^d$ or $[-5, 5]^d$
- **Multimodal**: Many local minima
- **Non-separable**: Variables are coupled
- **Differentiable**: Smooth function
- **Scalable**: Can be used in any dimension

## Usage Example

```python
--8<-- "experiments_benchmarks_ackley_example.py"
```

## Characteristics for Algorithm Testing

### What the Ackley Function Tests

1. **Global vs Local Search**: Can the algorithm escape local minima?
2. **Exploitation**: Can it fine-tune around the global optimum?
3. **Scalability**: How does performance change with dimensionality?
4. **Robustness**: Consistent performance across runs?

### Expected Behavior

- **Random Search**: Explores broadly but may miss global optimum
- **Hill Climbing**: Often gets trapped in local minima
- **Simulated Annealing**: Can escape local minima with proper cooling
- **Bayesian Optimization**: Usually finds global optimum efficiently
- **Population Methods**: Good at avoiding local minima

## Multi-Dimensional Usage

```python
--8<-- "experiments_benchmarks_ackley_example_2.py"
```

## Visualization

The Ackley function in 2D shows:
- A central valley leading to the global minimum at (0,0)
- Many small local minima scattered across the surface
- Exponential decay from the center
- High-frequency oscillations creating the local minima

## Algorithm Comparison Example

```python
--8<-- "experiments_benchmarks_ackley_example_3.py"
```

## Parameter Sensitivity

The Ackley function parameters affect difficulty:

```python
--8<-- "experiments_benchmarks_ackley_example_4.py"
```

## References

- Ackley, D. H. (1987). A connectionist machine for genetic hillclimbing.
- Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems.
