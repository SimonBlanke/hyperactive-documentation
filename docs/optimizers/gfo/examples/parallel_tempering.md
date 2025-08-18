!!! example 

    ```python
    from hyperactive.opt.gfo import ParallelTempering
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import BaggingClassifier
    from sklearn.datasets import load_digits

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [10, 20, 50, 100],
        "max_samples": [0.5, 0.7, 0.9, 1.0],
        "max_features": [0.5, 0.7, 0.9, 1.0]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=BaggingClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer with parallel tempering parameters
    optimizer = ParallelTempering(
        experiment=experiment,
        population=8,
        n_iter_swap=50
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```