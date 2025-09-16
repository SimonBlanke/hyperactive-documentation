# Optimization Algorithms

Hyperactive v5 provides 25+ optimization algorithms across three different backends. Each backend offers different strengths and is optimized for different use cases.

## Backend Overview

### Gradient-Free-Optimizers (GFO) Backend

The GFO backend provides 16 optimization algorithms with consistent interfaces and comprehensive parameter control.

**Import from:** `hyperactive.opt.gfo` (also re-exported in `hyperactive.opt`)

### Optuna Backend  

The Optuna backend provides 8 modern optimization algorithms with advanced features like multi-objective optimization.

**Import from:** `hyperactive.opt.optuna` (also re-exported in `hyperactive.opt`)

### Scikit-learn Backend

Sklearn-style search using `ParameterGrid` / `ParameterSampler`, with evaluation handled by Hyperactive experiments.

**Import from:** `hyperactive.opt.gridsearch` and `hyperactive.opt`

## Gradient-Free-Optimizers Backend

### Hill Climbing Variants

#### HillClimbing
Basic hill climbing with local search.



**Properties:**
- **Type**: Local search
- **Exploration**: Low
- **Computational Cost**: Low

#### StochasticHillClimbing
Hill climbing with random steps.



#### RepulsingHillClimbing
Hill climbing with repulsion mechanism to avoid local minima.



#### RandomRestartHillClimbing
Hill climbing with periodic random restarts.



### Annealing Methods

#### SimulatedAnnealing
Classical simulated annealing algorithm.



**Properties:**
- **Type**: Global search with cooling
- **Exploration**: High initially, decreases over time
- **Computational Cost**: Low

### Direct Methods

#### DownhillSimplexOptimizer
Nelder-Mead simplex method for derivative-free optimization.



#### PowellsMethod
Powell's conjugate direction method.



#### PatternSearch
Pattern search optimization.



### Sampling Methods

#### RandomSearch
Pure random sampling of the search space.



**Properties:**
- **Type**: Global search
- **Exploration**: Maximum
- **Computational Cost**: Low

#### GridSearch
Systematic grid search over parameter space.



### Advanced Methods

#### DirectAlgorithm
DIRECT (Dividing Rectangles) global optimization.



#### LipschitzOptimizer
Optimization using Lipschitz constraint.



### Population-Based Methods

#### ParticleSwarmOptimizer
Particle swarm optimization algorithm.



**Properties:**
- **Type**: Global search
- **Exploration**: Balanced
- **Computational Cost**: Medium

#### SpiralOptimization
Spiral-based optimization algorithm.



#### GeneticAlgorithm
Classical genetic algorithm.



#### EvolutionStrategy
Evolution strategy optimization.



#### DifferentialEvolution
Differential evolution algorithm.



#### ParallelTempering
Parallel tempering/replica exchange method.



### Surrogate-Based Methods

#### BayesianOptimizer
Bayesian optimization with Gaussian processes.



**Properties:**
- **Type**: Global search with surrogate model
- **Exploration**: Balanced, adaptive
- **Computational Cost**: High

#### TreeStructuredParzenEstimators
Tree-structured Parzen Estimator (TPE) algorithm.



#### ForestOptimizer  
Random forest-based optimization.



## Optuna Backend

The Optuna backend provides modern optimization algorithms with advanced features.

### TPEOptimizer
Tree-structured Parzen Estimator via Optuna.



### RandomOptimizer
Random sampling via Optuna.



### CmaEsOptimizer
Covariance Matrix Adaptation Evolution Strategy.



### GPOptimizer
Gaussian Process optimization via Optuna.



### GridOptimizer
Grid search via Optuna.



### Multi-Objective Optimization

#### NSGAIIOptimizer
NSGA-II multi-objective optimization.



#### NSGAIIIOptimizer
NSGA-III multi-objective optimization.



### QMCOptimizer
Quasi-Monte Carlo sampling.



## Scikit-learn Backend

Sklearn-style search; not thin wrappers over sklearn CV utilities. Evaluation is performed by a `SklearnCvExperiment` and parallelization by Hyperactive backends.

### GridSearchSk
Sklearn-style exhaustive grid evaluation (evaluation via `SklearnCvExperiment`).



### RandomSearchSk
Sklearn-style randomized search (evaluation via `SklearnCvExperiment`).



## Choosing the Right Algorithm

### For Quick Prototyping
- **RandomSearch**: Fastest, good baseline
- **GridSearch**: Systematic, interpretable

### For Sample-Efficient Optimization
- **BayesianOptimizer**: Best for expensive functions
- **TPEOptimizer**: Good balance of efficiency and simplicity

### For High-Dimensional Problems
- **DifferentialEvolution**: Good for continuous spaces
- **CmaEsOptimizer**: Excellent for continuous optimization

### For Multi-Modal Functions
- **ParticleSwarmOptimizer**: Good exploration
- **ParallelTempering**: Handles multiple modes well

### For Constrained Problems
- **DirectAlgorithm**: Handles box constraints well
- **PatternSearch**: Good for constrained optimization

## Usage Examples

### Basic Usage Pattern



### Comparing Multiple Algorithms
