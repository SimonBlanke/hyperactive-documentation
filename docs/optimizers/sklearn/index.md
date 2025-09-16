# Scikit-learn–Styled Optimizers

Hyperactive provides sklearn-style search using `ParameterGrid` / `ParameterSampler`, with evaluation routed through Hyperactive experiments. This preserves familiar configuration while keeping the v5 architecture’s separation between search (optimizer) and evaluation (experiment).

Key characteristics:

- Sklearn-style search spaces via `ParameterGrid` / `ParameterSampler`
- Evaluation via a Hyperactive `Experiment` (typically `SklearnCvExperiment`)
- Parallelism via optimizer backends (`backend`, `backend_params`)
- Results exposed on the optimizer (`best_params_`, `best_score_`, etc.)

