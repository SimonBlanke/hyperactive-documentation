!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import SimulatedAnnealingOptimizer

    ...

    optimizer = SimulatedAnnealingOptimizer(annealing_rate=0.999)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
