# Parallel Tempering


## Introduction

**Parallel Tempering** initializes multiple simulated annealing searches with different 
temperatures and chooses to swap those temperatures with a probability based on 
its temperature and difference of current scores.


{% include 'optimizers/examples/parallel_tempering.md' %}

## About the implementation

The population of the parallel tempering optimizer consists of multiple initializations
of the simulated annealing optimizer class. Each of those receives a random starting temperature.
The algorithm calculates the probability of swapping temperatures
for every combination of annealer instances. 

$$
p = \min \left( 1, exp^{(\text{score}_i-\text{score}_j)(\frac{1}{T_i} - \frac{1}{T_j})} \right)
$$

The indices $i$ and $j$ correspond to the two simulated annealing optimizers.



## Parameters

{% include 'parameters/population.md' %}

{% include 'parameters/n_iter_swap.md' %}

{% include 'parameters/rand_rest_p.md' %}
