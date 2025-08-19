# Parabola Function

## Introduction

The Parabola function is a simple quadratic benchmark function that provides a basic test case for optimization algorithms. It's similar to the Sphere function but with a different mathematical structure, offering insights into algorithm behavior on quadratic landscapes.

## Mathematical Definition

The Parabola function can be defined in various forms. A common implementation is:

$$f(x) = \sum_{i=1}^d (x_i - c_i)^2$$

Where $c_i$ are offset constants (often set to 0 for centered parabola).

## Properties

- **Global minimum**: $f(c_1, c_2, ..., c_d) = 0$ where $c_i$ are the offsets
- **Search domain**: Typically $[-10, 10]^d$ or similar
- **Unimodal**: Single global minimum
- **Separable**: Variables are independent  
- **Convex**: Bowl-shaped surface
- **Quadratic**: Second-order polynomial
- **Scalable**: Works in any dimension

## Usage Example

```python
--8<-- "experiments_benchmarks_parabola_example.py"
```

## Algorithm Testing

### Convergence Speed Comparison

```python
--8<-- "experiments_benchmarks_parabola_example_2.py"
```

### Precision Testing

```python
--8<-- "experiments_benchmarks_parabola_example_3.py"
```

### Scalability Analysis

```python
--8<-- "experiments_benchmarks_parabola_example_4.py"
```

## Custom Parabola Variations

```python
--8<-- "experiments_benchmarks_parabola_example_5.py"
```

## Rotated Parabola

```python
--8<-- "experiments_benchmarks_parabola_example_6.py"
```

## Educational Use Cases

### Teaching Optimization Basics

```python
--8<-- "experiments_benchmarks_parabola_example_7.py"
```

### Convergence Visualization

```python
--8<-- "experiments_benchmarks_parabola_example_8.py"
```

## Performance Characteristics

For the Parabola function, typical performance expectations:

- **Hill Climbing**: Very efficient, direct path to minimum
- **Random Search**: Slow convergence, many evaluations needed
- **Simulated Annealing**: Good performance, some random exploration
- **Bayesian Optimization**: Excellent efficiency, few evaluations needed
- **Population methods**: Overkill for this simple function

## Common Applications

1. **Algorithm validation**: Quick sanity check for new optimizers
2. **Parameter tuning**: Test optimizer parameters on simple function
3. **Educational purposes**: Demonstrate optimization principles
4. **Baseline comparison**: Simple reference for performance
5. **Convergence analysis**: Study optimization dynamics

## References

- Fletcher, R. (2013). Practical methods of optimization.
- Nocedal, J., & Wright, S. (2006). Numerical optimization.
