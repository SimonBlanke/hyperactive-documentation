# Grid Search


## Introduction

The grid search explores the search space by starting from a corner and progressing `step_size`-steps
per iteration. Increasing the `step_size` enables a more uniform exploration of the search space. 


{% include 'optimizers/examples/grid_search.md' %}


## About the implementation

The implementation of this grid-search was realized by
[Thomas Gak-Deluen](https://github.com/tgdn) and his team. The algorithm
works by choosing a direction in the beginning and moving through the search space
one 2-dimensional-plane at a time.



## Parameters

{% include 'parameters/step_size.md' %}


