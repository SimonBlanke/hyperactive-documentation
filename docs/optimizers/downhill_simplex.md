# Downhill Simplex Optimizer


## Introduction

The downhill simplex optimizer works by grouping `number of dimensions + 1`-positions into a simplex, which can explore the search-space by changing shape.
The simplex changes shape by reflecting, expanding, contracting or shrinking via
the `alpha`, `gamma`, `beta` or `sigma` parameters.


{% include 'optimizers/examples/downhill_simplex.md' %}

## About the implementation

The **downhill simplex** algorithm works in a completly different way from the other local
hill climbing based optimizers. It is much more complex, because there are 
reflecting-, expanding-, contracting- and shrinking-steps for each the iteration and the evaluation. This leads to a bigger and more complex source code than the hill climbing based algorithms.

In this implementation the downhill simplex algorithm has some similarities to population-based algorithms. It needs at least `number of dimensions + 1` initial positions to form a simplex in the search-space and the movement of the positions in the simplex are affected by each other.


  1. Choose `number of dimensions + 1` positions to build a simplex
  2. Sort simplex positions by their score:
    - $x_0$ is the best position
    - $x_{N-1}$ is the second worst
    - $x_N$ is the worst
    - ...
  3. Calculate the center position $m$ of all but the worst position
    - $m = \frac{1}{N}\sum^{N-1}_{i=0}x_i$
  4. Reflect ($\alpha$) the worst position at the center position
    - $r = m + \alpha (m - x_N)$
  5. If the reflected position is better than $x_0$:
    - $e=m+\gamma(m-x_N)$
    - $p$ is the better position of $e$ and $r$
    - $x_N \leftarrow p$
  6. If the reflected position is better than $x_{N-1}$:
    - $x_N \leftarrow r$
    - Go to step 2
  7. If $h$ is the better position of $x_N$ and $r$:
    - $c=h+\beta(m-h)$
  8. If c is better than $x_N$:
    - $x_N \leftarrow c$
  9.  Shrink the simplex
    - For each dimension $(i \in {1, ..., N})$
      - $x_i \leftarrow x_i + \sigma(x_o - x_i)$
  10. Go to step 2



## Parameters

{% include 'parameters/alpha.md' %}

{% include 'parameters/gamma.md' %}

{% include 'parameters/beta.md' %}

{% include 'parameters/sigma.md' %}
