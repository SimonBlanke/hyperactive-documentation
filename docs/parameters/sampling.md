### `sampling`

The `sampling`-parameter is a second pass of randomly sampling. It samples from the list of all possible positions (not directly from the search-space). This might be necessary, because the `predict`-method of the surrogate model could overload the memory. 

  - **type**: dict
  - **default**: {'random': 1000000}
  - **typical range**: -

---