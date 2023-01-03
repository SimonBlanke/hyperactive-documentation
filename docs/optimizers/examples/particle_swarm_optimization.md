!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import ParticleSwarmOptimizer

    ...

    optimizer = ParticleSwarmOptimizer(population=5)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
