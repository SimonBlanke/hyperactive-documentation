# Scikit-learn–Styled Optimizers

Hyperactive provides scikit-learn–styled optimizers that use sklearn-style parameter grids and distributions while running through Hyperactive’s optimization and experiment abstractions. These are designed for users who prefer sklearn-like configuration but want to stay within the Hyperactive v5 architecture.

Key characteristics:

- Sklearn-style search spaces via `ParameterGrid` / `ParameterSampler`
- Evaluation via a Hyperactive `Experiment` (typically `SklearnCvExperiment`)
- Parallelism via Hyperactive backends (`backend`, `backend_params`)
- Results exposed on the optimizer (`best_params_`, `best_score_`, etc.)

