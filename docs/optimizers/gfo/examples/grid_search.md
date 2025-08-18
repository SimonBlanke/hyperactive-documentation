!!! example 

    ```python
    from hyperactive.opt.gfo import GridSearch
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.svm import SVC
    from sklearn.datasets import load_breast_cancer

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Define search space
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=SVC(),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer with custom step size
    optimizer = GridSearch(experiment=experiment, step_size=0.1)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
