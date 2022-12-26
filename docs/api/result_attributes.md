# Result Attributes

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