### `gpr`

The access to the surrogate model. To pass a surrogate model it must be similar to the following code: 

```python
class GPR:
    def __init__(self):
        self.gpr = GaussianProcessRegressor()

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)
```

The `predict`-method returns only $\mu$ if `return_std=False` and returns $\mu$ and $\sigma$ if `return_std=True`. Note that you have to pass the instantiated class to the `gpr`-parameter:

```python
surrogate_model = GPR()
opt=BayesianOptimizer(gpr=surrogate_model)
```

  - **type**: class
  - **default**: -
  - **possible values**: -

---