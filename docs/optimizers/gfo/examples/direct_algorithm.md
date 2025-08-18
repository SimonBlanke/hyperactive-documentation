!!! example 

    ```python
    from hyperactive.opt.gfo import DirectAlgorithm
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.datasets import load_iris

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Define search space (limited due to GP complexity)
    param_grid = {
        "alpha": [1e-10, 1e-8, 1e-6, 1e-4],
        "n_restarts_optimizer": [0, 1, 2, 3]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=GaussianProcessClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer
    optimizer = DirectAlgorithm(experiment=experiment)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```