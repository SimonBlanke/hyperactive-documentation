!!! example 

    ```python
    from hyperactive.opt.gfo import ForestOptimizer
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Define search space
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "linear", "poly"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=SVC(),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer with custom xi parameter
    optimizer = ForestOptimizer(
        experiment=experiment,
        xi=0.01
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
