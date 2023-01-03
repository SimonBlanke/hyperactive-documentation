### `gamma_tpe`

This parameter determines the separation of the explored positions into good and bad. It must be in the range between 0 and 1. A value of 0.2 means, that the best $20\%$ of the known positions are put into the list of best known positions, while the rest is put into the list of worst known positions.

  - **type**: float
  - **default**: 0.2
  - **typical range**: 0.05 ... 0.75

---