!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import DownhillSimplexOptimizer

    ...

    optimizer = DownhillSimplexOptimizer(alpha=0.5)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
