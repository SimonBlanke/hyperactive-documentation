### `start_temp`

The start temperatur is set to the given value at the start of the optimization run and gets changed by the `annealing_rate` over time. This `start_temp` is multiplied with `epsilon` to change the step-size of this hill-climbing-based algorithm over time. 

  - **type**: float
  - **default**: 1
  - **typical range**: 0.5 ... 1.5

---