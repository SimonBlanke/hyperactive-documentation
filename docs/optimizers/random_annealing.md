# Random Annealing


## Introduction

The **random annealing** algorithm is based on hill climbing and derived on the regular simulated annealing algorithm. It takes the idea of a temperatur and annealing to change the step-size over time.

{% include 'optimizers/examples/random_annealing.md' %}


## About the implementation

It also has a `start_temp` and an `annealing_rate` but uses them differently. The `start_temp` gets changed by the `annealing_rate` similarly to simulated annealing.

$$
\text{start_temp} \leftarrow  \text{start_temp} * \text{annealing_rate}
$$


In the **random annealing** algorithm the `start_temp` acts as a factor that gets multiplied with `epsilon` to change the step-size of this algorithm over time. 

$$
\epsilon' = \epsilon * \text{start_temp}
$$

The idea is, that you want the algorithm to explore the search-space in the beginning of the optimization run and concentrate more on fine tuning the best results locally towards the end.



## Parameters

{% include 'parameters/epsilon.md' %}

{% include 'parameters/distribution.md' %}

{% include 'parameters/n_neighbours.md' %}

{% include 'parameters/start_temp_random_annealing.md' %}

{% include 'parameters/annealing_rate.md' %}

