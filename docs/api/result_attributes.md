# Result Methods

The methods shown on this page return results of the optimization run. Therefore they should be called after the `.run()`-method. Since hyperactive accepts multiple searches, each of which can have a different objective-function, the result methods need the objective-function as a parameter. If you run multiple searches with the same objective-function the search-data is automatically concatenated.

!!! example
    ```python
    from hyperactive import Hyperactive

    ...

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()

    best_para = hyper.best_para(objective_function)
    best_score = hyper.best_score(objective_function)
    search_data = hyper.search_data(objective_function)
    ```

## .best_para()

### `objective_function`

- **type:** `callable`

Parameter dictionary of the best score of the given objective_function found in the previous optimization run.


## .best_score()

### `objective_function`

- **type:** `callable`

Numerical value of the best score of the given objective_function found in the previous optimization run.


## .search_data()

### `objective_function`

- **type:** `callable`

The dataframe contains score and parameter information of the given objective_function found in the optimization run. If the parameter times is set to True the evaluation- and iteration- times are added to the dataframe.