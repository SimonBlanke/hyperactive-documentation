!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import SpiralOptimization

    ...

    optimizer = SpiralOptimization(decay_rate=1.1)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
