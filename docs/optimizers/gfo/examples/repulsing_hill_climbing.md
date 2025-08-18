!!! example 

    ```python
    from hyperactive.opt.gfo import RepulsingHillClimbing
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Define search space
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        X=X, y=y,
        cv=5
    )

    # Create optimizer with custom repulsion factor
    optimizer = RepulsingHillClimbing(
        experiment=experiment,
        repulsion_factor=3
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
