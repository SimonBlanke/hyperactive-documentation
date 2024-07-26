# Bayesian Optimization


## Introduction

Bayesian optimization chooses new positions by calculating the expected 
improvement of every position in the search space based on a gaussian process 
that trains on already evaluated positions.

{% include 'optimizers/examples/bayesian_optimization.md' %}


## About the implementation

The bayesian optimizer collects the information about the position and score in each 
iteration. The gaussian process regressor fits to the position (features) and score (target),
and predicts the scores of all unknown positions. This is why the bayesian optimization needs
at least one initial position. The gaussian process returns the standard deviation 
in addition to the prediction (or mean), both of which are required to 
compute the acquisition function.
The position of the best predicted score
is evaluated next. The selected position and its true score is then collected, 
restarting the cycle. The acquisition function used in this algorithm is the expected improvement.  The expected improvement is calculated by the following equation:

$$
\text{expected improvement} = ( \mu - y_{sample, max} - \xi ) \cdot \varphi(Z) + \sigma \cdot \Phi(Z)
$$

where:

$$
\mu, \sigma = \text{surrogate-model.predict}(...)
$$

and:

- $y_{sample, max}$ **=>** best known score
- $\xi$ **=>** xi-parameter
- $\varphi$ **=>** Probability density function
- $\Phi$ **=>** Cumulative distribution function

The surrogate model used in bayesian optimization is the gassian process regressor. A crucial property of this model is, that it returns the uncertainty of the prediction $\sigma$ together with the predicted value $\mu$.


## Parameters

{% include 'parameters/xi.md' %}

{% include 'parameters/replacement.md' %}

{% include 'parameters/gpr.md' %}

{% include 'parameters/max_sample_size.md' %}

{% include 'parameters/sampling.md' %}

{% include 'parameters/warm_start_smbo.md' %}

{% include 'parameters/rand_rest_p.md' %}
