!!! example 

    ```python
    from hyperactive.opt.gfo import LipschitzOptimizer
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.linear_model import ElasticNet
    from sklearn.datasets import load_diabetes

    # Load dataset
    X, y = load_diabetes(return_X_y=True)

    # Define search space
    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=ElasticNet(random_state=42),
        param_grid=param_grid,
        X=X, y=y,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    # Create optimizer
    optimizer = LipschitzOptimizer(experiment=experiment)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```