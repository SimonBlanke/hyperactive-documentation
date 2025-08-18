# Differential Evolution


## Introduction

Differential evolution is an optimization technique that iteratively improves a population of candidate solutions by combining and perturbing them based on their differences. It generates new candidates by adding the weighted difference between two population members to a third member, creating trial solutions that are evaluated for their fitness. If a trial solution is better than the target it replaces, ensuring continual improvement. This method is effective for optimizing complex, nonlinear, and multimodal functions where traditional gradient-based methods are impractical.

{% include 'optimizers/gfo/examples/differential_evolution.md' %}


## About the implementation

The differential evolution optimizer inherits from the evolutionary algorithm class. Therefore it is similar to the genetic algorithm or evolution strategy, like using a crossover step to mix individuals in the population by e.g. discrete recombination. But it sets itself apart in the iteration loop by always doing a mutation step and crossover step to mix the target- and mutant-vector.


## Parameters

{% include 'parameters/population.md' %}

{% include 'optimizers/gfo/differential_evolution/mutation_rate.md' %}

{% include 'optimizers/gfo/differential_evolution/crossover_rate.md' %}
