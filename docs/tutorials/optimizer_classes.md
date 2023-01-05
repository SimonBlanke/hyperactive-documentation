# Optimizer Classes

Each of the following optimizer classes can be initialized and passed to the "add_search"-method via the "optimizer"-argument. During this initialization the optimizer class accepts only optimizer-specific-paramters (no random_state, initialize, ... ):

!!! example 

    ```python
    optimizer = HillClimbingOptimizer(epsilon=0.1, distribution="laplace", n_neighbours=4)

    # for the default parameters you can just write:
    optimizer = HillClimbingOptimizer()

    hyper = Hyperactive()
    # and pass it to Hyperactive:
    hyper.add_search(model, search_space, optimizer=optimizer, n_iter=100)
    hyper.run()
    ```

So the optimizer-classes are different from Gradient-Free-Optimizers. A more detailed explanation of the optimization-algorithms and the optimizer-specific-paramters can be found in the Optimization Tutorial.

- HillClimbingOptimizer
- StochasticHillClimbingOptimizer
- RepulsingHillClimbingOptimizer
- SimulatedAnnealingOptimizer
- DownhillSimplexOptimizer
- RandomSearchOptimizer
- GridSearchOptimizer
- RandomRestartHillClimbingOptimizer
- RandomAnnealingOptimizer
- PowellsMethod
- PatternSearch
- ParallelTemperingOptimizer
- ParticleSwarmOptimizer
- SpiralOptimization
- EvolutionStrategyOptimizer
- BayesianOptimizer
- LipschitzOptimizer
- DirectAlgorithm
- TreeStructuredParzenEstimators
- ForestOptimizer

