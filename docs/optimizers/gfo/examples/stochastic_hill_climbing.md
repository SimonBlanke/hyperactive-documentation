!!! example 

    ```python
    from hyperactive.opt.gfo import StochasticHillClimbing
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Define search space
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_grid=param_grid,
        X=X, y=y,
        cv=5
    )

    # Create optimizer with custom parameters
    optimizer = StochasticHillClimbing(
        experiment=experiment,
        p_accept=0.1
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```