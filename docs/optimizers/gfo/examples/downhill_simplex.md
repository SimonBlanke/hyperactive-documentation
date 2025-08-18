!!! example 

    ```python
    from hyperactive.opt.gfo import DownhillSimplexOptimizer
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import load_iris

    # Load dataset
    X, y = load_iris(return_X_y=True)

    # Create experiment (GaussianNB has limited hyperparameters)
    # Using a simple example for demonstration
    experiment = SklearnCvExperiment(
        estimator=GaussianNB(),
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        X=X, y=y,
        cv=5
    )

    # Create optimizer
    optimizer = DownhillSimplexOptimizer(experiment=experiment)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```