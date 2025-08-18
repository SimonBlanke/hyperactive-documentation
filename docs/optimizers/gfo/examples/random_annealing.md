!!! example 

    ```python
    from hyperactive.opt.gfo import SimulatedAnnealing
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import load_breast_cancer

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Define search space
    param_grid = {
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "penalty": ["l1", "l2", "elasticnet"],
        "learning_rate": ["constant", "optimal", "invscaling"]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=SGDClassifier(random_state=42, max_iter=1000),
        param_grid=param_grid,
        X=X, y=y,
        cv=5
    )

    # Create optimizer with annealing parameters
    optimizer = SimulatedAnnealing(
        experiment=experiment,
        start_temp=10.0,
        annealing_rate=0.98
    )

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```