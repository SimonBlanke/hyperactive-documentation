# Custom Optimization Strategy


## How it works

Optimization strategies are designed to automatically pass useful data from one optimization algorithm to the next:

- The best parameter found in one optimization run ist automatically passed to the next. 
- The search-data if passed as memory-warm-start to all consecutive optimization runs.
- If an algorithm accepts warm-start-smbo as a parameter the search-data is also automatically passed.

Without optimization strategies the steps above can be manually done, but by chaining together the algorithms into strategies it is automatically done for you.



!!! example 
    ```python
    ...

    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    ...

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(RandomSearchOptimizer(), duration=0.5)
    opt_strat.add_optimizer(BayesianOptimizer(), duration=0.5)
    
    hyper = Hyperactive()
    hyper.add_search(..., optimizer=opt_strat)
    hyper.run()

    ...
    ```


## Templates


### Random search, hill climbing

This strategy starts with a long random search to explore the search-space and continues with hill climbing to locally find the best optimum. If there are no improvements during the first hill-climbing run the strategy will early stop and continue tuning the best results with a hill-climbing algorithm with smaller step size (small epsilon).

!!! example
    ```python
    from hyperactive.optimizers import RandomSearchOptimizer, HillClimbingOptimizer
    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(RandomSearchOptimizer(), duration=0.5)
    opt_strat.add_optimizer(HillClimbingOptimizer(), duration=0.5, early_stopping={"n_iter_no_change": 5})
    opt_strat.add_optimizer(HillClimbingOptimizer(epsilon=0.01), duration=0.5)
    ```


### Particle swarm optimizer, random search

In this strategy we try to resolve a shortcoming of the particle swarm optimization by resetting it with a random search. This helps the particles find new positions after they converge in the previous attempt.

!!! example 
    ```python
    from hyperactive.optimizers import ParticleSwarmOptimizer, RandomSearchOptimizer
    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(ParticleSwarmOptimizer(), duration=0.4, early_stopping={"n_iter_no_change": 5})
    opt_strat.add_optimizer(RandomSearchOptimizer(), duration=0.2)
    opt_strat.add_optimizer(ParticleSwarmOptimizer(), duration=0.4)
    ```


### Direct algorithm, hill climbing

This strategy is simple and uses light weight algorithms. The direct search is the computationally least expensive of the sequence-model-based algorithms. After the initial step the hill climbing will continue searching locally to get an even better result, while keeping the computational cost low.

!!! example
    ```python
    from hyperactive.optimizers import DirectAlgorithm, HillClimbingOptimizer
    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(DirectAlgorithm(), duration=0.6)
    opt_strat.add_optimizer(HillClimbingOptimizer(epsilon=0.01), duration=0.4)
    ```


### Random search, bayesian optimization, hill climbing

This strategy attempts to save computational time, while still getting the benefit of bayesian optimization. The random search will to most of the exploration that would normally be done by bayesian optimization in its first iterations. After the bayesian optimization step the strategy will continue tuning the best position with a hill climbing with small step size.

!!! example
    ```python
    from hyperactive.optimizers import RandomSearchOptimizer, BayesianOptimizer, HillClimbingOptimizer
    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(RandomSearchOptimizer(), duration=0.7)
    opt_strat.add_optimizer(BayesianOptimizer(), duration=0.1)
    opt_strat.add_optimizer(HillClimbingOptimizer(epsilon=0.01), duration=0.2)
    ```


### Direct algorithm, lipschitz optimizer, bayesian optimization, random annealing

This strategy starts with a direct search and fills the positions that the bayesian optimization would normally explore with the lipschitz optimizer. The bayesian optimization will use the known positions to find a good solution, while the random annealing will tune this position with a low step size, that decreases further over time.

!!! example
    ```python
    from hyperactive.optimizers import DirectAlgorithm, LipschitzOptimizer, BayesianOptimizer, RandomAnnealingOptimizer
    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(DirectAlgorithm(), duration=0.4)
    opt_strat.add_optimizer(LipschitzOptimizer(), duration=0.15)
    opt_strat.add_optimizer(BayesianOptimizer(), duration=0.15)
    opt_strat.add_optimizer(RandomAnnealingOptimizer(start_temp=0.3), duration=0.3)
    ```