### `n_neighbours`

The number of positions the algorithm explores from its current postion before setting its current position to the best of those neighbour positions. If the value of `n_neighbours` is large the hill-climbing-based algorithm will take a lot of time to choose the next position to move to, but the choice will probably be a good one. It might be a prudent approach to increase `n_neighbours` of the search-space has a lot of dimensions, because there are more possible directions to move to.

  - **type**: int
  - **default**: 3
  - **typical range**: 1 ... 10

---