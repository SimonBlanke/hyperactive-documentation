!!! example 

    ```python
    from hyperactive.opt.gfo import DifferentialEvolution
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.svm import SVR
    from sklearn.datasets import load_boston

    # Load dataset (note: load_boston is deprecated, using for example)
    try:
        X, y = load_boston(return_X_y=True)
    except ImportError:
        # Fallback for newer sklearn versions
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X, y = data.data, data.target

    # Define search space
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "epsilon": [0.01, 0.1, 0.2]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=SVR(),
        param_grid=param_grid,
        X=X, y=y,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    # Create optimizer with custom mutation rate
    optimizer = DifferentialEvolution(
        experiment=experiment,
        mutation_rate=0.8,
        population=25
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
