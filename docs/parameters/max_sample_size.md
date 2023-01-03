### `max_sample_size`

The `max_sample_size` is a first pass of randomly sampling, before all possible positions are generated for the sequence-model-based optimization. It samples the search space directly and takes effect if the search-space is very large:

```python
search_data = {
  "x1": np.arange(0, 1000, 0.01),
  "x2": np.arange(0, 1000, 0.01),
  "x3": np.arange(0, 1000, 0.01),
  "x4": np.arange(0, 1000, 0.01),
}
```

The `max_sample_size`-parameter is necessary to avoid a memory overload from generating all possible positions from the search-space. The search-space above corresponds to a list of $100000^4 = 100000000000000000000$ numpy arrays. This memory overload is expected for a sequence-model-based optimization algorithm, because the surrogate model has the job make a prediction for every position in the search-space to calculate the acquisition-function. The `max_sample_size`-parameter was introduced to provide a better out-of-the-box experience if using smb-optimizers.



  - **type**: int
  - **default**: 10000000
  - **typical range**: 1000000 ... 100000000

---