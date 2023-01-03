### `step_size`

The number of steps the grid search takes after each iteration. If this parameter is 
set to 3 the grid search won't select the next position, but the one it would normally
select after 3 iterations. This way we get a sparse grid after the first pass through
the search space. After the first pass is done the grid search starts at the beginning
and does a second pass with the same step size. And a third pass after that.

  - **type**: int
  - **default**: 1
  - **typical range**: 1 ... 1000

---