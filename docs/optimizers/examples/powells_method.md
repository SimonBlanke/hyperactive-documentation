!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import PowellsMethod

    ...

    optimizer = PowellsMethod(iters_p_dim=15)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
