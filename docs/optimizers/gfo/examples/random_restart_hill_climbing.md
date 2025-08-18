!!! example 

    ```python
    from hyperactive.opt.gfo import RandomRestartHillClimbing
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_wine

    # Load dataset
    X, y = load_wine(return_X_y=True)

    # Define search space
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=5
    )

    # Create optimizer with restart parameters
    optimizer = RandomRestartHillClimbing(
        experiment=experiment,
        n_iter_restart=25
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```