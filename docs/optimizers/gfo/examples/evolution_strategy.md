!!! example 

    ```python
    from hyperactive.opt.gfo import EvolutionStrategy
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import load_diabetes

    # Load dataset
    X, y = load_diabetes(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    # Create optimizer with evolution strategy parameters
    optimizer = EvolutionStrategy(
        experiment=experiment,
        population=20,
        mutation_rate=0.2
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```