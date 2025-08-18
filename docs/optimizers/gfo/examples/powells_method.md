!!! example 

    ```python
    from hyperactive.opt.gfo import PowellsMethod
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.datasets import load_digits

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
        "algorithm": ["SAMME", "SAMME.R"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=AdaBoostClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer
    optimizer = PowellsMethod(experiment=experiment)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```