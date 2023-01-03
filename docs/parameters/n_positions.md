### `n_positions`

Number of positions that the pattern consists of. If the value of `n_positions` is large the  algorithm will take a lot of time to choose the next position to move to, but the choice will probably be a good one. It might be a prudent approach to increase `n_positions` of the search-space has a lot of dimensions, because there are more possible directions to move to.


  - **type**: int
  - **default**: 4
  - **typical range**: 2 ... 8

---