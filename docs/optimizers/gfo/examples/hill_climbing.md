!!! example 

    ```python
    from hyperactive.opt.gfo import HillClimbing
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Define search space
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
        "kernel": ["rbf", "linear"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=SVC(),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer with custom parameters
    optimizer = HillClimbing(experiment=experiment, epsilon=0.1)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
