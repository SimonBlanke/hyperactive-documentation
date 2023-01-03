### `epsilon`

The *step-size* of the hill climbing algorithm. Increasing `epsilon` also increases the average step-size, because its proportional to the standard-deviation of the distribution of the hill-climbing-based algorithm. If `epsilon` is too large the newly selected positions will be at the edge of the search space. If its value is very low it might not find new positions.

  - **type**: float
  - **default**: 0.03
  - **typical range**: 0.01 ... 0.3

---