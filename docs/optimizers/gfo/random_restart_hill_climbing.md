# Random Restart Hill Climbing


## Introduction

The random restart hill climbing works by starting a hill climbing search and jumping to a random 
new position after `n_iter_restart` iterations. Those restarts should prevent the algorithm getting stuck in local optima.

{% include 'optimizers/gfo/examples/random_restart_hill_climbing.md' %}


## About the implementation

The random restart hill climbing inherits its behaviour from the regular hill climbing and 
expands it by jumping to a random position during the iteration step if the following criteria is meet:

$$
\text{iter}_i  \mathrm{\%}  \text{n_iter_restart} = 0
$$



## Parameters

{% include 'parameters/epsilon.md' %}

{% include 'parameters/distribution.md' %}

{% include 'parameters/n_neighbours.md' %}

{% include 'parameters/n_iter_restart.md' %}
