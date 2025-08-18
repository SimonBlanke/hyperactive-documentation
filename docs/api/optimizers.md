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
from hyperactive.opt.gfo import HillClimbing

optimizer = HillClimbing(
    experiment=experiment,
    epsilon=0.1,              # Step size
    distribution="normal",    # Step distribution
    n_neighbours=4           # Number of neighbors to try
)
```

**Properties:**
- **Type**: Local search
- **Exploration**: Low
- **Computational Cost**: Low

#### StochasticHillClimbing
Hill climbing with random steps.

```python
from hyperactive.opt.gfo import StochasticHillClimbing

optimizer = StochasticHillClimbing(
    experiment=experiment,
    p_accept=0.1,            # Acceptance probability
    norm_factor=1.0          # Normalization factor
)
```

#### RepulsingHillClimbing
Hill climbing with repulsion mechanism to avoid local minima.

```python
from hyperactive.opt.gfo import RepulsingHillClimbing

optimizer = RepulsingHillClimbing(
    experiment=experiment,
    repulsion_factor=3,      # Strength of repulsion
    n_neighbours=4           # Number of neighbors
)
```

#### RandomRestartHillClimbing
Hill climbing with periodic random restarts.

```python
from hyperactive.opt.gfo import RandomRestartHillClimbing

optimizer = RandomRestartHillClimbing(
    experiment=experiment,
    n_iter_restart=100,      # Iterations before restart
    rand_rest_p=0.1         # Probability of random restart
)
```

### Annealing Methods

#### SimulatedAnnealing
Classical simulated annealing algorithm.

```python
from hyperactive.opt.gfo import SimulatedAnnealing

optimizer = SimulatedAnnealing(
    experiment=experiment,
    start_temp=10.0,         # Initial temperature
    annealing_rate=0.98,     # Temperature decay rate
    min_temp=0.01           # Minimum temperature
)
```

**Properties:**
- **Type**: Global search with cooling
- **Exploration**: High initially, decreases over time
- **Computational Cost**: Low

### Direct Methods

#### DownhillSimplexOptimizer
Nelder-Mead simplex method for derivative-free optimization.

```python
from hyperactive.opt.gfo import DownhillSimplexOptimizer

optimizer = DownhillSimplexOptimizer(
    experiment=experiment,
    alpha=1.0,               # Reflection coefficient
    gamma=2.0,               # Expansion coefficient
    rho=0.5,                 # Contraction coefficient
    sigma=0.5               # Shrink coefficient
)
```

#### PowellsMethod
Powell's conjugate direction method.

```python
from hyperactive.opt.gfo import PowellsMethod

optimizer = PowellsMethod(
    experiment=experiment,
    iters_p_dim=10          # Iterations per dimension
)
```

#### PatternSearch
Pattern search optimization.

```python
from hyperactive.opt.gfo import PatternSearch

optimizer = PatternSearch(
    experiment=experiment,
    pattern_size=0.1,        # Size of search pattern
    reduction=0.9           # Pattern size reduction factor
)
```

### Sampling Methods

#### RandomSearch
Pure random sampling of the search space.

```python
from hyperactive.opt.gfo import RandomSearch

optimizer = RandomSearch(experiment=experiment)
```

**Properties:**
- **Type**: Global search
- **Exploration**: Maximum
- **Computational Cost**: Low

#### GridSearch
Systematic grid search over parameter space.

```python
from hyperactive.opt.gfo import GridSearch

optimizer = GridSearch(
    experiment=experiment,
    step_size=0.1           # Grid step size
)
```

### Advanced Methods

#### DirectAlgorithm
DIRECT (Dividing Rectangles) global optimization.

```python
from hyperactive.opt.gfo import DirectAlgorithm

optimizer = DirectAlgorithm(
    experiment=experiment,
    epsilon=0.01,            # Convergence tolerance
    n_positions=100         # Number of positions to evaluate
)
```

#### LipschitzOptimizer
Optimization using Lipschitz constraint.

```python
from hyperactive.opt.gfo import LipschitzOptimizer

optimizer = LipschitzOptimizer(
    experiment=experiment,
    lip_const=1.0           # Lipschitz constant
)
```

### Population-Based Methods

#### ParticleSwarmOptimizer
Particle swarm optimization algorithm.

```python
from hyperactive.opt.gfo import ParticleSwarmOptimizer

optimizer = ParticleSwarmOptimizer(
    experiment=experiment,
    population=20,           # Swarm size
    inertia=0.9,            # Inertia weight
    cognitive_weight=2.0,    # Cognitive parameter
    social_weight=2.0       # Social parameter
)
```

**Properties:**
- **Type**: Global search
- **Exploration**: Balanced
- **Computational Cost**: Medium

#### SpiralOptimization
Spiral-based optimization algorithm.

```python
from hyperactive.opt.gfo import SpiralOptimization

optimizer = SpiralOptimization(
    experiment=experiment,
    population=10,           # Population size
    decay_rate=0.99         # Spiral decay rate
)
```

#### GeneticAlgorithm
Classical genetic algorithm.

```python
from hyperactive.opt.gfo import GeneticAlgorithm

optimizer = GeneticAlgorithm(
    experiment=experiment,
    population=20,           # Population size
    mutation_rate=0.1,       # Mutation probability
    crossover_rate=0.9      # Crossover probability
)
```

#### EvolutionStrategy
Evolution strategy optimization.

```python
from hyperactive.opt.gfo import EvolutionStrategy

optimizer = EvolutionStrategy(
    experiment=experiment,
    population=15,           # Population size
    mutation_rate=0.2,       # Mutation rate
    selection_rate=0.5      # Selection rate
)
```

#### DifferentialEvolution
Differential evolution algorithm.

```python
from hyperactive.opt.gfo import DifferentialEvolution

optimizer = DifferentialEvolution(
    experiment=experiment,
    population=20,           # Population size
    mutation_rate=0.8,       # Differential weight
    crossover_rate=0.9      # Crossover probability
)
```

#### ParallelTempering
Parallel tempering/replica exchange method.

```python
from hyperactive.opt.gfo import ParallelTempering

optimizer = ParallelTempering(
    experiment=experiment,
    population=10,           # Number of replicas
    n_iter_swap=100,        # Swap attempt frequency
    temp_weight=0.1         # Temperature scaling
)
```

### Surrogate-Based Methods

#### BayesianOptimizer
Bayesian optimization with Gaussian processes.

```python
from hyperactive.opt.gfo import BayesianOptimizer

optimizer = BayesianOptimizer(
    experiment=experiment,
    gpr=None,               # Custom Gaussian process regressor
    xi=0.01,                # Exploration parameter
    warm_start_smbo=None    # Warm start points
)
```

**Properties:**
- **Type**: Global search with surrogate model
- **Exploration**: Balanced, adaptive
- **Computational Cost**: High

#### TreeStructuredParzenEstimators
Tree-structured Parzen Estimator (TPE) algorithm.

```python
from hyperactive.opt.gfo import TreeStructuredParzenEstimators

optimizer = TreeStructuredParzenEstimators(
    experiment=experiment,
    gamma_tpe=0.25,         # Quantile for good/bad split
    warm_start_smbo=None    # Warm start points
)
```

#### ForestOptimizer  
Random forest-based optimization.

```python
from hyperactive.opt.gfo import ForestOptimizer

optimizer = ForestOptimizer(
    experiment=experiment,
    xi=0.01,                # Exploration parameter
    tree_regressor="extra_tree"  # Tree regressor type
)
```

## Optuna Backend

The Optuna backend provides modern optimization algorithms with advanced features.

### TPEOptimizer
Tree-structured Parzen Estimator via Optuna.

```python
from hyperactive.opt.optuna import TPEOptimizer

optimizer = TPEOptimizer(
    experiment=experiment,
    n_startup_trials=10,     # Random trials before TPE
    n_ei_candidates=24      # Expected improvement candidates
)
```

### RandomOptimizer
Random sampling via Optuna.

```python
from hyperactive.opt.optuna import RandomOptimizer

optimizer = RandomOptimizer(experiment=experiment)
```

### CmaEsOptimizer
Covariance Matrix Adaptation Evolution Strategy.

```python
from hyperactive.opt.optuna import CmaEsOptimizer

optimizer = CmaEsOptimizer(
    experiment=experiment,
    sigma0=1.0,             # Initial step size
    population_size=None    # Population size (auto if None)
)
```

### GPOptimizer
Gaussian Process optimization via Optuna.

```python
from hyperactive.opt.optuna import GPOptimizer

optimizer = GPOptimizer(experiment=experiment)
```

### GridOptimizer
Grid search via Optuna.

```python
from hyperactive.opt.optuna import GridOptimizer

optimizer = GridOptimizer(experiment=experiment)
```

### Multi-Objective Optimization

#### NSGAIIOptimizer
NSGA-II multi-objective optimization.

```python
from hyperactive.opt.optuna import NSGAIIOptimizer

optimizer = NSGAIIOptimizer(
    experiment=experiment,  # Must return multiple objectives
    population_size=50,     # Population size
    mutation_prob=0.1      # Mutation probability
)
```

#### NSGAIIIOptimizer
NSGA-III multi-objective optimization.

```python
from hyperactive.opt.optuna import NSGAIIIOptimizer

optimizer = NSGAIIIOptimizer(
    experiment=experiment,
    population_size=50,
    mutation_prob=0.1
)
```

### QMCOptimizer
Quasi-Monte Carlo sampling.

```python
from hyperactive.opt.optuna import QMCOptimizer

optimizer = QMCOptimizer(
    experiment=experiment,
    qmc_type="sobol"        # QMC sequence type
)
```

## Scikit-learn Backend

Direct integration with scikit-learn optimization tools.

### GridSearchSk
Direct sklearn GridSearchCV integration.

```python
from hyperactive.opt import GridSearchSk

optimizer = GridSearchSk(
    experiment=experiment,
    n_jobs=1,               # Parallel jobs
    cv=3                   # Cross-validation folds
)
```

### RandomSearchSk
Direct sklearn RandomizedSearchCV integration.

```python
from hyperactive.opt import RandomSearchSk

optimizer = RandomSearchSk(
    experiment=experiment,
    n_jobs=1,
    cv=3,
    n_iter=100             # Number of random samples
)
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
from hyperactive.opt.gfo import BayesianOptimizer
from hyperactive.experiment.bench import Ackley

# Create experiment
experiment = Ackley(dimensions=2, bounds=(-5, 5))

# Create and run optimizer
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Best score:", experiment.score(best_params)[0])
```

### Comparing Multiple Algorithms

```python
from hyperactive.opt.gfo import BayesianOptimizer, RandomSearch, ParticleSwarmOptimizer

algorithms = [
    BayesianOptimizer(experiment=experiment),
    RandomSearch(experiment=experiment), 
    ParticleSwarmOptimizer(experiment=experiment, population=20)
]

results = {}
for optimizer in algorithms:
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    results[optimizer.__class__.__name__] = (best_params, score)

for name, (params, score) in results.items():
    print(f"{name}: {score}")
```