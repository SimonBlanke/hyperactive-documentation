## HyperactiveSearchCV

The `HyperactiveSearchCV` class is used for hyperparameter tuning with cross-validation, utilizing Sklearn estimators. It is designed to integrate various optimization methods, allowing users to efficiently find the best hyperparameters.

You can import and use `HyperactiveSearchCV` as follows:

??? example "Import and use HyperactiveSearchCV"
    ```python
    from your_package import HyperactiveSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    estimator = RandomForestClassifier()
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 10, None],
        "min_samples_split": [2, 5, 10]
    }
    
    search_cv = HyperactiveSearchCV(estimator=estimator, params_config=param_grid)
    search_cv.fit(X_train, y_train)
    print(search_cv.best_params_)
    ```

The `HyperactiveSearchCV` class provides several parameters and methods, detailed below.

### Parameters

#### `estimator`
The Sklearn-compatible estimator (e.g., classifier or regressor) to be tuned.

- **type:** `SklearnBaseEstimator`

#### `params_config`
A dictionary defining the hyperparameter search space. Keys are parameter names, and values are lists of possible values.

- **type:** `Dict[str, list]`

#### `optimizer`
The optimizer used for hyperparameter search. You can specify either the name of a predefined optimizer or a custom optimizer.

- **type:** `Union[str, Type[RandomSearchOptimizer]]`
- **default:** `"default"`

#### `n_iter`
The number of parameter settings to be sampled during the optimization process.

- **type:** `int`
- **default:** `100`

#### `scoring`
The scoring method to evaluate the predictions on the test set. Can be a scoring function or a string referring to a Sklearn scoring method.

- **type:** `Callable | str | None`

#### `n_jobs`
Number of jobs to run in parallel.

- **type:** `int`
- **default:** `1`

#### `random_state`
Random seed for reproducibility of the optimization process.

- **type:** `int | None`

#### `refit`
Whether to refit the best estimator using the entire dataset after hyperparameter search.

- **type:** `bool`
- **default:** `True`

#### `cv`
The cross-validation splitting strategy to be used for evaluating the model performance.

- **type:** `int | "BaseCrossValidator" | Iterable | None`

### Methods

#### `fit`
Tunes the hyperparameters of the estimator using the provided training data.

- **Parameters:**
  - `X`: array-like or sparse matrix, shape `(n_samples, n_features)`
    - The training input samples.
  - `y`: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
    - The target values.
  - `**fit_params`: Additional fit parameters to pass to the estimator.

- **Returns:**
  - `self`: The fitted instance of the `HyperactiveSearchCV` class.

??? example "Using the fit method"
    ```python
    search_cv.fit(X_train, y_train)
    ```

#### `score`
Computes the performance of the best estimator on the input data.

- **Parameters:**
  - `X`: array-like or sparse matrix, shape `(n_samples, n_features)`
    - The input data on which the performance is evaluated.
  - `y`: array-like, shape `(n_samples,)`, optional
    - The true target values for scoring.
  - `**params`: Additional parameters for scoring.

- **Returns:**
  - `float`: The score of the best estimator on the input data.

??? example "Using the score method"
    ```python
    test_score = search_cv.score(X_test, y_test)
    print(test_score)
    ```

### Attributes

#### `best_params_`
Dictionary containing the best parameters found during the search.

#### `best_estimator_`
The estimator with the best parameters, refitted with the entire dataset if `refit=True`.

#### `best_score_`
The best score achieved by the estimator with the best hyperparameters.

#### `search_data_`
Data from the optimization search, which includes detailed information on each evaluated configuration.
