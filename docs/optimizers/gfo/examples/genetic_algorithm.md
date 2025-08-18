!!! example 

    ```python
    from hyperactive.opt.gfo import GeneticAlgorithm
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_diabetes

    # Load dataset
    X, y = load_diabetes(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    # Create optimizer with custom parameters
    optimizer = GeneticAlgorithm(
        experiment=experiment,
        mutation_rate=0.4,
        population=30
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
