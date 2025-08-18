!!! example 

    ```python
    from hyperactive.opt.gfo import SimulatedAnnealing
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import load_wine

    # Load dataset
    X, y = load_wine(return_X_y=True)

    # Define search space
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01, 0.1]
    }

    # Create experiment
    experiment = SklearnCvExperiment(
        estimator=MLPClassifier(random_state=42, max_iter=200),
        param_grid=param_grid,
        X=X, y=y,
        cv=3
    )

    # Create optimizer with custom annealing rate
    optimizer = SimulatedAnnealing(experiment=experiment, annealing_rate=0.999)

    # Run optimization
    best_params = optimizer.solve()
    print("Best parameters:", best_params)
    ```
