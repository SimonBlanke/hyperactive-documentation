# Tree Structured Parzen Estimators


## Introduction

Tree of Parzen Estimators chooses new positions by calculating an acquisition function. 
It assesses all possible positions by calculating the ratio of their probability being among the best positions and the worst positions. Those probabilities are determined with a kernel density estimator, which is trained on already evaluated positions.


{% include 'optimizers/examples/tree_structured_parzen_estimators.md' %}

## About the implementation

Similar to other sequence-model-based optimization algorithms the positions and scores 
of previous evaluations are saved as features and targets to train a machine learning algorithm.
For the tree structured parzen estimators we use two separate kernel density estimators one receiving the best positions, while the other gets the worst positions to calculate the acquisition function. 

The separation of the position into best and worst is very simple. First we sort the positions by their scores. We split this list at the following index:

$$
i_{split} = \max( n_{samples} \gamma, 1)
$$

where:

- $\gamma$ **=>** `gamma_tpe`
- $n_{samples}$ **=>** the number of positions in the list



## Parameters

{% include 'parameters/gamma_tpe.md' %}

{% include 'parameters/max_sample_size.md' %}

{% include 'parameters/sampling.md' %}

{% include 'parameters/warm_start_smbo.md' %}

{% include 'parameters/rand_rest_p.md' %}
