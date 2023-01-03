# DIRECT Algorithm


## Introduction

The DIRECT algorithm works by separating the search-space into smaller rectangle-shaped subspaces and evaluating their center positions. The algorithm decides which subspace to further separate by calculating an upper-bound within each subspace. The algorithm always splits the subspace-rectangles along the biggest edge. The upper-bound determines the highest potential improvement in each subspace. Therefore the DIRECT algorithm works similar to acquisition-function-based algorithms by predicting the score of new positions in the search-space based on known positions and scores.


{% include 'optimizers/examples/direct_algorithm.md' %}


## About the implementation

The DIRECT algorithm calculates the upper-bound based on the distance between the center position of the subspace and the furthest position:

$$
U = score_{c} + k \cdot || x_c-x_{furthest} ||_2
$$

where:

- $score_{c}$ **=>** score of center position
- $k$ **=>** is the slope parameter (hard-coded to 1 in v1.2)
- $|| x_c-x_{furthest} ||_2$ **=>** distance between the center position of the subspace and the furthest position from the center

Compared to other smbo the DIRECT algorithm does not need to do calculations based on all positions in the search-space but only a small subset. This makes this algorithm less computationally expensive, than bayesian- or lipschitz-optimization.