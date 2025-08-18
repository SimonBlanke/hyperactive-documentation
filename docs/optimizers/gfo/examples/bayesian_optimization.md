!!! example 

    ```python
    from hyperactive.opt.gfo import BayesianOptimizer
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_wine

    # Load dataset
    X, y = load_wine(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=5
    )

    # Create optimizer with custom parameters
    optimizer = BayesianOptimizer(experiment=experiment, xi=0.15)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
