# Optimization Algorithms

Hyperactive v5 provides 25+ optimization algorithms across three different backends. Each backend offers different strengths and is optimized for different use cases.

## Backend Overview

### Gradient-Free-Optimizers (GFO) Backend

The GFO backend provides 16 optimization algorithms with consistent interfaces and comprehensive parameter control.

**Import from:** `hyperactive.opt.gfo`

### Optuna Backend  

The Optuna backend provides 8 modern optimization algorithms with advanced features like multi-objective optimization.

**Import from:** `hyperactive.opt.optuna`

### Scikit-learn Backend

Direct integration with scikit-learn's optimization tools for familiar workflows.

**Import from:** `hyperactive.opt.gridsearch` and `hyperactive.opt`

## Gradient-Free-Optimizers Backend

### Hill Climbing Variants

#### HillClimbing
Basic hill climbing with local search.

```python
--8<-- "api_optimizers_example.py"
```

**Properties:**
- **Type**: Local search
- **Exploration**: Low
- **Computational Cost**: Low

#### StochasticHillClimbing
Hill climbing with random steps.

```python
--8<-- "api_optimizers_example_2.py"
```

#### RepulsingHillClimbing
Hill climbing with repulsion mechanism to avoid local minima.

```python
--8<-- "api_optimizers_example_3.py"
```

#### RandomRestartHillClimbing
Hill climbing with periodic random restarts.

```python
--8<-- "api_optimizers_example_4.py"
```

### Annealing Methods

#### SimulatedAnnealing
Classical simulated annealing algorithm.

```python
--8<-- "api_optimizers_example_5.py"
```

**Properties:**
- **Type**: Global search with cooling
- **Exploration**: High initially, decreases over time
- **Computational Cost**: Low

### Direct Methods

#### DownhillSimplexOptimizer
Nelder-Mead simplex method for derivative-free optimization.

```python
--8<-- "api_optimizers_example_6.py"
```

#### PowellsMethod
Powell's conjugate direction method.

```python
--8<-- "api_optimizers_example_7.py"
```

#### PatternSearch
Pattern search optimization.

```python
--8<-- "api_optimizers_example_8.py"
```

### Sampling Methods

#### RandomSearch
Pure random sampling of the search space.

```python
--8<-- "api_optimizers_example_9.py"
```

**Properties:**
- **Type**: Global search
- **Exploration**: Maximum
- **Computational Cost**: Low

#### GridSearch
Systematic grid search over parameter space.

```python
--8<-- "api_optimizers_example_10.py"
```

### Advanced Methods

#### DirectAlgorithm
DIRECT (Dividing Rectangles) global optimization.

```python
--8<-- "api_optimizers_example_11.py"
```

#### LipschitzOptimizer
Optimization using Lipschitz constraint.

```python
--8<-- "api_optimizers_example_12.py"
```

### Population-Based Methods

#### ParticleSwarmOptimizer
Particle swarm optimization algorithm.

```python
--8<-- "api_optimizers_example_13.py"
```

**Properties:**
- **Type**: Global search
- **Exploration**: Balanced
- **Computational Cost**: Medium

#### SpiralOptimization
Spiral-based optimization algorithm.

```python
--8<-- "api_optimizers_example_14.py"
```

#### GeneticAlgorithm
Classical genetic algorithm.

```python
--8<-- "api_optimizers_example_15.py"
```

#### EvolutionStrategy
Evolution strategy optimization.

```python
--8<-- "api_optimizers_example_16.py"
```

#### DifferentialEvolution
Differential evolution algorithm.

```python
--8<-- "api_optimizers_example_17.py"
```

#### ParallelTempering
Parallel tempering/replica exchange method.

```python
--8<-- "api_optimizers_example_18.py"
```

### Surrogate-Based Methods

#### BayesianOptimizer
Bayesian optimization with Gaussian processes.

```python
--8<-- "api_optimizers_example_19.py"
```

**Properties:**
- **Type**: Global search with surrogate model
- **Exploration**: Balanced, adaptive
- **Computational Cost**: High

#### TreeStructuredParzenEstimators
Tree-structured Parzen Estimator (TPE) algorithm.

```python
--8<-- "api_optimizers_example_20.py"
```

#### ForestOptimizer  
Random forest-based optimization.

```python
--8<-- "api_optimizers_example_21.py"
```

## Optuna Backend

The Optuna backend provides modern optimization algorithms with advanced features.

### TPEOptimizer
Tree-structured Parzen Estimator via Optuna.

```python
--8<-- "api_optimizers_example_22.py"
```

### RandomOptimizer
Random sampling via Optuna.

```python
--8<-- "api_optimizers_example_23.py"
```

### CmaEsOptimizer
Covariance Matrix Adaptation Evolution Strategy.

```python
--8<-- "api_optimizers_example_24.py"
```

### GPOptimizer
Gaussian Process optimization via Optuna.

```python
--8<-- "api_optimizers_example_25.py"
```

### GridOptimizer
Grid search via Optuna.

```python
--8<-- "api_optimizers_example_26.py"
```

### Multi-Objective Optimization

#### NSGAIIOptimizer
NSGA-II multi-objective optimization.

```python
--8<-- "api_optimizers_example_27.py"
```

#### NSGAIIIOptimizer
NSGA-III multi-objective optimization.

```python
--8<-- "api_optimizers_example_28.py"
```

### QMCOptimizer
Quasi-Monte Carlo sampling.

```python
--8<-- "api_optimizers_example_29.py"
```

## Scikit-learn Backend

Direct integration with scikit-learn optimization tools.

### GridSearchSk
Direct sklearn GridSearchCV integration.

```python
--8<-- "api_optimizers_example_30.py"
```

### RandomSearchSk
Direct sklearn RandomizedSearchCV integration.

```python
--8<-- "api_optimizers_example_31.py"
```

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

```python
--8<-- "api_optimizers_example_32.py"
```

### Comparing Multiple Algorithms

```python
--8<-- "api_optimizers_example_33.py"
```