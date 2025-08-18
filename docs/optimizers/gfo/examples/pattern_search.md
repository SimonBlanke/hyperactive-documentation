!!! example 

    ```python
    from hyperactive.opt.gfo import PatternSearch
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.linear_model import Ridge
    from sklearn.datasets import load_diabetes

    # Load dataset
    X, y = load_diabetes(return_X_y=True)

    # Define search space
    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=Ridge(),
        param_grid=param_grid,
        X=X, y=y,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    # Create optimizer with custom pattern size
    optimizer = PatternSearch(
        experiment=experiment,
        pattern_size=0.2
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```