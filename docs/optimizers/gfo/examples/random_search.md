!!! example 

    ```python
    from hyperactive.opt.gfo import RandomSearch
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_digits

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer
    optimizer = RandomSearch(experiment=experiment)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
