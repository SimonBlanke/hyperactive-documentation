# Search Space

The search space defines what values the optimizer can select during the search. These selected values will be inside the objective function argument and can be accessed like in a dictionary. The values in each search space dimension should always be in a list. If you use np.arange you should put it in a list afterwards:


## Numerical dimensions

!!! example 
    ```python
    search_space = {
        "x1": list(np.arange(-100, 101, 1)),
        "x2": list(np.arange(-100, 101, 1)),
    }
    ```

A special feature of Hyperactive is shown in the next example. You can put not just numeric values into the search space dimensions, but also strings and functions. This enables a very high flexibility in how you can create your studies.

!!! example 
    ```python
    def func1():
    # do stuff
    return stuff
    

    def func2():
    # do stuff
    return stuff


    search_space = {
        "x": list(np.arange(-100, 101, 1)),
        "str": ["a string", "another string"],
        "function" : [func1, func2],
    }
    ```

## Categorical dimensions



If you want to put other types of variables (like numpy arrays, pandas dataframes, lists, ...) into the search space you can do that via functions:

!!! example 
    ```python
    def array1():
    return np.array([1, 2, 3])
    

    def array2():
    return np.array([3, 2, 1])


    search_space = {
        "x": list(np.arange(-100, 101, 1)),
        "str": ["a string", "another string"],
        "numpy_array" : [array1, array2],
    }
    ```

The functions contain the numpy arrays and returns them. This way you can use them inside the objective function.


