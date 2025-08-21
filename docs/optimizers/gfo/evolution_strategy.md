# Evolution Strategy


## Introduction

Evolution strategy mutates and combines the best individuals of a population across a 
number of generations without transforming them into an array of bits 
(like genetic algorithms) but uses the real values of the positions.

{% include 'optimizers/gfo/examples/evolution_strategy.md' %}


## About the implementation

The mutation part of the evolution strategy optimizer is a regular hill climbing iteration step. The idea of a mutation corresponds to small changes of the current position of the optimizer. 
The crossover works by randomly combining the positions of some of the two best individuals in the population to get a new position. The individuals with the worst scores are removed to preserve the number of individuals in the `population`.

The `mutation_rate` and `crossover_rate` are weights that correspond to the probability of choosing one method. The algorithm decides what method to use in the following way:

$$
R = \text{random-float} (0 ... \text{total-rate})
$$

where:

$$
\text{total-rate} = \text{mutation-rate} + \text{crossover-rate}
$$




## Parameters

{% include 'parameters/population.md' %}

{% include 'parameters/mutation_rate.md' %}

{% include 'parameters/crossover_rate.md' %}

{% include 'parameters/rand_rest_p.md' %}
