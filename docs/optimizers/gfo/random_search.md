# Random Search


## Introduction

The random search explores by choosing a new position at random after each iteration. 
Some random search implementations choose a new position within a large hypersphere around 
the current position. The implementation in hyperactive is purely random across the 
search space in each step.

{% include 'optimizers/gfo/examples/random_search.md' %}


## About the implementation

The random search is a very simple algorithm that has no parameters to change its behaviour.
In each iteration the random position is selected via random.choice 
from a list of possible positions.

