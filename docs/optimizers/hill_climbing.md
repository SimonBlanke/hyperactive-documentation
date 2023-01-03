# Hill Climbing


## Introduction

Hill climbing is a very basic optimization technique, that explores the search space only localy. It starts at an initial point, which is often chosen randomly and continues to move to positions within its neighbourhood with a better solution. It has no method against getting stuck in local optima.


{% include 'optimizers/examples/hill_climbing.md' %}



## About the implementation

The hill climbing algorithm is saving the current position from the search space and finds new positions by sampling random positions around it with a gaussian `distribution`. This is done with a gaussian `distribution`. So if the position is further away the probability of getting selected is lower, but never zero. This makes the hill climbing more heuristic and a bit more adaptable. The positions found around its current positions are called *neighbours*. The hill climbing will sample and evaluate `n_neighbours` before moving its current position to the best of those neighbour. After moving the algorithm restarts sampling neighbours from its `distribution` around its new current position.

If the `distribution` is "normal" (default) the hill climbing algorithm will sample its neighbours with the normal distribution:

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} 
  \exp^{\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)}
$$

In this equation $\mu$ is the current position in the search space and $\sigma$ is calculated with the size of the search space in each dimension and the epsilon.

$$
\sigma = \text{dim-size} * \epsilon
$$

So the standard deviation of the gaussian distribution in each dimension is dependend on the size of the search space in the corresponding dimension. This improves the exploration if the sizes of the search space dimensions are differing from each other.


## Parameters

{% include 'parameters/epsilon.md' %}

{% include 'parameters/distribution.md' %}

{% include 'parameters/n_neighbours.md' %}

