### `gpr`

The access to the surrogate model. To pass a surrogate model it must be similar to the following code: 



The `predict`-method returns only $\mu$ if `return_std=False` and returns $\mu$ and $\sigma$ if `return_std=True`. Note that you have to pass the instantiated class to the `gpr`-parameter:



  - **type**: class
  - **default**: -
  - **possible values**: -

---