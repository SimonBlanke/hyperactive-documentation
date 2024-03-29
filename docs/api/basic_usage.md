# Basic Usage

## Hyperactive

The `Hyperactive`-class is the main component to run the optimization. You can import it directly from the `hyperactive`-package, then add one or more searches and run it. This would look as follows:

??? example "Import and use Hyperactive"
    ```python
    from hyperactive import Hyperactive

    ...

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=100)
    hyper.run()
    ```


The `Hyperactive`-class can receive the following parameters:

### `verbosity`

- **type:** `list`

The verbosity list determines what part of the optimization information will be printed in the command line.

- `progress_bar`
- `print_results`
- `print_times`

### `distribution`

- **type:** `str`

Determine, which distribution service you want to use. Each library uses different packages to pickle objects:

- `multiprocessing` uses pickle
- `joblib` uses dill
- `pathos` uses cloudpickle


### `n_processes`

- **type:** `str`, `int`

The maximum number of processes that are allowed to run simultaneously. If n_processes is of int-type there will only run n_processes-number of jobs simultaneously instead of all at once. So if n_processes=10 and n_jobs_total=35, then the schedule would look like this 10 - 10 - 10 - 5. This saves computational resources if there is a large number of n_jobs. If "auto", then n_processes is the sum of all n_jobs (from .add_search(...)).


## add_search

The `add_search`-method contains the following arguments:

### `objective_function`

- **type:** `callable`

The objective function defines the optimization problem. The optimization algorithm will try to maximize the numerical value that is returned by the objective function by trying out different parameters from the search space.

### `search_space`

- **type:** `dict`

Defines the space were the optimization algorithm can search for the best parameters for the given objective function.

### `n_iter`

- **type:** `int`

The number of iterations that will be performed during the optimization run. The entire iteration consists of the optimization-step, which decides the next parameter that will be evaluated and the evaluation-step, which will run the objective function with the chosen parameter and return the score.

### `optimizer`

- **type:** `object`

The 'optimizer'-argument accepts an instance of an optimization class or an optimization strategy that can be imported from Hyperactive. The "default" corresponds to the random search optimizer. The optimization classes are imported from `hyperactive.optimizers`, while the optimization strategies are imported from `hyperactive.optimizers.strategies`.

The imported optimization classes from hyperactive are different from the [optimization backend](https://github.com/SimonBlanke/Gradient-Free-Optimizers). They only accept optimizer-specific-parameters (no `search-space` or `initialize`). 

The following optimization classes can be imported and used:

- HillClimbingOptimizer
- StochasticHillClimbingOptimizer
- RepulsingHillClimbingOptimizer
- SimulatedAnnealingOptimizer
- DownhillSimplexOptimizer
- RandomSearchOptimizer
- GridSearchOptimizer
- RandomRestartHillClimbingOptimizer
- RandomAnnealingOptimizer
- PowellsMethod
- PatternSearch
- ParallelTemperingOptimizer
- ParticleSwarmOptimizer
- SpiralOptimization
- EvolutionStrategyOptimizer
- BayesianOptimizer
- LipschitzOptimizer
- DirectAlgorithm
- TreeStructuredParzenEstimators
- ForestOptimizer

The following optimization strategy can be imported and used:

- CustomOptimizationStrategy


??? example "Using an optimization algorithm"
    ```python
    ...

    from hyperactive.optimizers import HillClimbingOptimizer

    ...

    opt_hco = HillClimbingOptimizer(epsilon=0.08)

    hyper = Hyperactive()
    hyper.add_search(..., optimizer=opt_hco)
    hyper.run()

    ...
    ```


??? example "Using an optimization strategy"
    ```python
    ...

    from hyperactive.optimizers.strategies import CustomOptimizationStrategy

    ...

    opt_strat = CustomOptimizationStrategy()
    opt_strat.add_optimizer(RandomSearchOptimizer(), duration=0.5)
    opt_strat.add_optimizer(BayesianOptimizer(), duration=0.5)
    
    hyper = Hyperactive()
    hyper.add_search(..., optimizer=opt_strat)
    hyper.run()

    ...
    ```


### `n_jobs`

- **type:** `int`

Number of jobs to run in parallel. Those jobs are optimization runs that work independent from another (no information sharing). If n_jobs == -1 the maximum available number of cpu cores is used.

### `initialize`

- **type:** `dict`

The initialization dictionary automatically determines a number of parameters that will be evaluated in the first n iterations (n is the sum of the values in initialize). The initialize keywords are the following:


- `grid`:
    Initializes positions in a grid like pattern. Positions that cannot be put into a grid are randomly positioned. For very high dimensional search spaces (>30) this pattern becomes random.

- `vertices`:
    Initializes positions at the vertices of the search space. Positions that cannot be put into a new vertex are randomly positioned.

- `random`:
    Number of random initialized positions

- `warm_start`:
    List of parameter dictionaries that marks additional start points for the optimization run.

??? example "Use the `initialize`-parameter"
    ```python
    ... 
    search_space = {
        "x1": list(range(10, 150, 5)),
        "x2": list(range(2, 12)),
    }

    ws1 = {"x1": 10, "x2": 2}
    ws2 = {"x1": 15, "x2": 10}

    hyper = Hyperactive()
    hyper.add_search(
        model,
        search_space,
        n_iter=30,
        initialize={"grid": 4, "random": 10, "vertices": 4, "warm_start": [ws1, ws2]},
    )
    hyper.run()
    ```

### `constraints`

- **type:** `list`, `None`
- **default:** `None`

The constraints-argument accepts a list of functions. These functions contain the same argument as the objective-function to access the parameters from the search-space and returns a boolean value. With these parameters you can set new conditions and boundries for the search-space by returning `True` or `False` depending on the parameters from the argument. If the returning value is true the position in the search-space is valid, but if it is false the position is **not** inside the "valid area" of the search-space.

Optimization algorithms will never select a position inside the constrained area of the search-space to be evaluated inside the objective-function.


!!! example
    ```python
    ...

    search_space = {
        "x1": list(np.arange(-10, 31, 0.3)),
        "x2": list(np.arange(-10, 31, 0.3)),
    }

    def constraint_1(para):
        # only values in 'x1' higher than -5 are valid
        return para["x1"] > -5

    constraints_list = [constraint_1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function, search_space, n_iter=50, constraints=constraints_list
    )
    hyper.run()

    ...
    ```


### `pass_through`

- **type:** `dict`

The pass_through accepts a dictionary that contains information that will be passed to the objective-function argument. This information will not change during the optimization run, unless the user does so by himself (within the objective-function).

??? example "Use the `pass_through`-parameter"
    ```python
    ... 
    def objective_function(para):
        para.pass_through["stuff1"] # <--- this variable is 1
        para.pass_through["stuff2"] # <--- this variable is 2

        score = -para["x1"] * para["x1"]
        return score

    pass_through = {
    "stuff1": 1,
    "stuff2": 2,
    }

    hyper = Hyperactive()
    hyper.add_search(
        model,
        search_space,
        n_iter=30,
        pass_through=pass_through,
    )
    hyper.run()
    ```

### `callbacks`

- **type:** `dict`

The callbacks enables you to pass functions to hyperactive that are called every iteration during the optimization run. The function has access to the same argument as the objective-function. You can decide if the functions are called before or after the objective-function is evaluated via the keys of the callbacks-dictionary. The values of the dictionary are lists of the callback-functions. The following example should show they way to use callbacks:

??? example "Use the `callbacks`-parameter"
    ```python
    ...

    def callback_1(access):
    # do some stuff

    def callback_2(access):
    # do some stuff

    def callback_3(access):
    # do some stuff

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        callbacks={
        "after": [callback_1, callback_2],
        "before": [callback_3]
        },
    )
    hyper.run()
    ```

### `catch`

- **type:** `dict`

The catch parameter provides a way to handle exceptions that occur during the evaluation of the objective-function or the callbacks. It is a dictionary that accepts the exception class as a key and the score that is returned instead as the value. This way you can handle multiple types of exceptions and return different scores for each. In the case of an exception it often makes sense to return np.nan as a score. You can see an example of this in the following code-snippet:

??? example "Use the `catch`-parameter"
    ```python
    ...

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        catch={
        ValueError: np.nan,
        },
    )
    hyper.run()
    ```

### `max_score`

- **type:** `float`

Maximum score until the optimization stops. The score will be checked after each completed iteration.


### `early_stopping`

- **type:** `dict`

Stops the optimization run early if it did not achive any score-improvement within the last iterations. The early_stopping-parameter enables to set three parameters:

- `n_iter_no_change`: Non-optional int-parameter. This marks the last n iterations to look for an improvement over the iterations that came before n. If the best score of the entire run is within those last n iterations the run will continue (until other stopping criteria are met), otherwise the run will stop.

- `tol_abs`: Optional float-paramter. The score must have improved at least this absolute tolerance in the last n iterations over the best score in the iterations before n. This is an absolute value, so 0.1 means an imporvement of 0.8 -> 0.9 is acceptable but 0.81 -> 0.9 would stop the run.

- `tol_rel`: Optional float-paramter. The score must have imporved at least this relative tolerance (in percentage) in the last n iterations over the best score in the iterations before n. This is a relative value, so 10 means an imporvement of 0.8 -> 0.88 is acceptable but 0.8 -> 0.87 would stop the run.


### `random_state`

- **type:** `int`

Random state for random processes in the random, numpy and scipy module.

### `memory`

- **type:** `bool`, `str`

Whether or not to use the "memory"-feature. The memory is a dictionary, which gets filled with parameters and scores during the optimization run. If the optimizer encounters a parameter that is already in the dictionary it just extracts the score instead of reevaluating the objective function (which can take a long time). If memory is set to "share" and there are multiple jobs for the same objective function then the memory dictionary is automatically shared between the different processes.

### `memory_warm_start`

- **type:** `pandas dataframe`

Pandas dataframe that contains score and parameter information that will be automatically loaded into the memory-dictionary.


## run

The `run`-method has the following arguments:

### `max_time`

- **type:** `float`

Maximum number of seconds until the optimization stops. The time will be checked after each completed iteration.