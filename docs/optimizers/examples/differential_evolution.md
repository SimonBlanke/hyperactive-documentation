!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import DifferentialEvolutionOptimizer

    ...

    optimizer = DifferentialEvolutionOptimizer(mutation_rate=0.8)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=100, optimizer=optimizer)
    hyper.run()
    ```
