!!! example 

    ```python
    from hyperactive import Hyperactive
    from hyperactive.optimizers import PatternSearch

    ...

    optimizer = PatternSearch(n_positions=2)

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=50, optimizer=optimizer)
    hyper.run()
    ```
