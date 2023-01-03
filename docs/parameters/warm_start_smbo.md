### `warm_start_smbo`

The `warm_start_smbo`-parameter is a pandas dataframe that contains search-data with the results from a previous optimization run. The dataframe containing the search-data could look like this:

|  x1 | x2  | score  |  
|---|---|---|
|  5 |  15 |  0.3 |  
|  10 | 12  |  0.7 |  
| ...  |  ... |  ... |  
| ...  |  ... |  ... |  

Where the corresponding search-space would look like this:

```python
search_space = {
  "x1": np.arange(0, 20),
  "x2": np.arange(0, 20),
}
```

Before passing the search-data to the optimizer make sure, that the columns match the search-space of the new optimization run. So you could not add another dimension ("x3") to the search-space and expect the warm-start to work. The dimensionality of the optimization must be preserved and fit the problem.

```python
opt = BayesianOptimization(warm_start_smbo=search_data)
```


  - **type**: pandas dataframe, None
  - **default**: None
  - **possible values**: -

---
