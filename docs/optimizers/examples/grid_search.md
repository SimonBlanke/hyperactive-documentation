!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import GridSearchOptimizer

    ...

    optimizer = GridSearchOptimizer(step_size=3)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
