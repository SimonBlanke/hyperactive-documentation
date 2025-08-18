# Particle Swarm Optimization


## Introduction

Particle swarm optimization works by initializing a number of positions in the search space,
which move according to their own inertia, the attraction to their own known best positions
and the global best position.


{% include 'optimizers/gfo/examples/particle_swarm_optimization.md' %}

## About the implementation

The particle swarm optimizer initializes multiple particle classes, which inherit from
the hill-climbing class. In the current version all movement-calculations are contained in
the particle class. Particles have a velocity, which they maintain because of their `inertia` even without attraction to other particles via `cognitive_weight` or `social_weight`. 

The particles use the following parameters and vectors to move through the search space:

  - $\omega$ **=>** `inertia`
  - $c_k$ **=>** `cognitive_weight`
  - $c_s$ **=>** `social_weight`
  - $r_1$ $r_2$ **=>** random floats between 0 ... 1
  - $p_n$ **=>** current position of the particle
  - $p_{best}$ **=>** best position of the particle
  - $g_{best}$ **=>** best position of all particles



The velocity of a particle is calculated by the following equation:

$$
v_{n+1} = \omega \cdot v_n + c_k \cdot r_1 \cdot (p_{best}-p_n) + c_s \cdot r_2 \cdot (g_{best} - p_n)
$$



Since the particle is always moving for the same single timeframe we can just add the velocity and the current position of the particle to get the new position.



## Parameters

{% include 'parameters/population.md' %}

{% include 'parameters/inertia.md' %}

{% include 'parameters/cognitive_weight.md' %}

{% include 'parameters/social_weight.md' %}

{% include 'parameters/rand_rest_p.md' %}
