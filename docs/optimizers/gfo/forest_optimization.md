# Forest Optimization

## Introduction

The forest-optimizer calculates the expected improvement of the position space with a 
tree-based model. This optimization technique is very similar to bayesian-optimization
in every part, except its surrogate model.


{% include 'optimizers/gfo/examples/forest_optimization.md' %}

## About the implementation

The forest-optimizer shares most of its code base with the bayesian-optimizer. Only its model to 
calculate the expected score $\mu$ and its standard deviation $\sigma$ is different. Tree based models do not 
calculate the standard deviation by them self. A modification is necessary to determine the
standard deviation from the impurity of the trees in the ensemble. This modification is taken from the paper *"Algorithm Runtime Prediction: Methods & Evaluation"* chapter *4.3.3*.



## Parameters

{% include 'parameters/xi.md' %}

{% include 'parameters/replacement.md' %}

{% include 'parameters/tree_regressor.md' %}

{% include 'parameters/max_sample_size.md' %}

{% include 'parameters/sampling.md' %}

{% include 'parameters/warm_start_smbo.md' %}

{% include 'parameters/rand_rest_p.md' %}
