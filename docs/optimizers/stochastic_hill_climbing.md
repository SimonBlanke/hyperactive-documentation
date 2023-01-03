# Stochastic Hill Climbing


## Introduction

Stochastic hill climbing extends the normal hill climbing by a simple method against getting 
stuck in local optima. It has a parameter `p_accept` you can set, 
that determines the probability to 
accept worse solutions as a next position. 
This should enable the stochastic hill climbing to find better solutions in
a non-convex optimization problem over many iterations.

{% include 'optimizers/examples/stochastic_hill_climbing.md' %}

## About the implementation

The **stochastic hill climbing** inherits the behaviour of the regular hill climbing algorithm and
adds its additional functionality after the evaluation of the score is done. 
If the new score is not better than the previous one the algorithm starts the following calculation:

$$
\text{score}_{normalized} = \text{norm} * \frac{\text{score}_{current} - \text{score}_{new}}{\text{score}_{current} + \text{score}_{new}}
$$

$$
p = \exp^{-\text{score}_{normalized}}
$$

If $p$ is smaller than a random number between 0 and `p_accept` the new position gets accepted anyways.



## Parameters

{% include 'parameters/epsilon.md' %}

{% include 'parameters/distribution.md' %}

{% include 'parameters/n_neighbours.md' %}

{% include 'parameters/p_accept.md' %}


