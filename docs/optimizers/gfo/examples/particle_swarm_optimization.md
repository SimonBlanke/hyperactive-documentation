!!! example 

    ```python
    from hyperactive.opt.gfo import ParticleSwarmOptimizer
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import load_digits

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Define search space
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer with custom population size
    optimizer = ParticleSwarmOptimizer(experiment=experiment, population=20)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
