!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import LipschitzOptimizer

    ...

    optimizer = LipschitzOptimizer()

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
