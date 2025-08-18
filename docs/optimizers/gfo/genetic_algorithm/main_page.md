# Genetic Algorithm


## Introduction

A genetic algorithm mimics natural selection by evolving a population of candidate solutions through processes akin to biological reproduction and mutation. Each candidate is evaluated using a fitness function, and the best-performing solutions are selected to create new candidates through crossover and mutation. This iterative process enables the algorithm to explore and exploit the solution space, making it particularly useful for complex or discontinuous landscapes where gradient-based methods fail.

{% include 'optimizers/gfo/examples/genetic_algorithm.md' %}


## About the implementation

The genetic algorithm optimizer works similar to the evolution strategy, in that both do an mutation **or** crossover step in each iteration. The probability of doing each is determined by the mutation and crossover rate.


## Parameters

{% include 'parameters/population.md' %}

{% include 'optimizers/gfo/genetic_algorithm/mutation_rate.md' %}

{% include 'optimizers/gfo/genetic_algorithm/crossover_rate.md' %}

{% include 'optimizers/gfo/genetic_algorithm/n_parents.md' %}

{% include 'optimizers/gfo/genetic_algorithm/offspring.md' %}
