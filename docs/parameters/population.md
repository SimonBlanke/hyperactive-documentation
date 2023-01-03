### `population`

Size of the population for population-based optimization algorithms. A member of the population is called *single optimizer*, *individual* or *particle* depending on the type of algorithm. Each member of the population is a separate optimizer class with information about the positions and scores of the optimizer and all methods to perform the iteration and evaluation steps.

All population based optimizers in this package calculate the new positions one member at a time. So if the optimizer performs 10 iterations and has a population size of 10, then each member of the population would move once to a new position. 


  - **type**: int
  - **default**: 10
  - **typical range**: 4 ... 25

---