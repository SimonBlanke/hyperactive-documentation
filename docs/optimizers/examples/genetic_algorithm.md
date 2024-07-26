!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import GeneticAlgorithmOptimizer

    ...

    optimizer = GeneticAlgorithmOptimizer(mutation_rate=0.4, n_parents=3)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=100, optimizer=optimizer)
    hyper.run()
    ```
