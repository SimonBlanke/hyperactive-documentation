# Repulsing Hill Climbing


## Introduction

The **repulsing hill climbing** optimization algorithm inherits from the regular hill-climbing
algorithm and adds a way to escape local optima by increasing the step-size once to jump away from its current region.

{% include 'optimizers/examples/repulsing_hill_climbing.md' %}

## About the implementation

Similar to other hill climbing based algorithms the **repulsing hill climbing**
inherits the methods from the regular hill climbing and adds a functionality to
escape local optima. The repulsing hill climbing temporally increases `epsilon`
by multiplying it with the `repulsion_factor` for the next iteration. 

$$
\sigma = \text{dim-size} * \epsilon * \text{repulsion_factor}
$$


This way the algorithm *jumps* away from the current position to explore other regions of the search-space. So the repulsing hill climbing is different from stochastic hill climbing and simulated annealing in that it **always** activates its methods to espace local optima instead of activating them with a certain **probability**.




## Parameters

{% include 'parameters/epsilon.md' %}

{% include 'parameters/distribution.md' %}

{% include 'parameters/n_neighbours.md' %}

{% include 'parameters/repulsion_factor.md' %}

