# Powell's Method


## Introduction

This powell's method implementation works by optimizing each search space dimension at a 
time with a hill climbing algorithm. It works by setting the 
search space range for all dimensions except one to a single value. The hill climbing
algorithms searches the best position within this dimension. 


{% include 'optimizers/examples/powells_method.md' %}


## About the implementation

The powell's method implemented in Gradient-Free-Optimizers does only see one dimension at a time.
This differs from the original idea of creating (and searching through) 
one search-vector at a time, that spans through multiple dimensions.
After `iters_p_dim` iterations the next dimension is searched, while the 
search space range from the previously searched dimension is set to the best position,
This way the algorithm finds new best positions one dimension at a time.


## Parameters

{% include 'parameters/iters_p_dim.md' %}


