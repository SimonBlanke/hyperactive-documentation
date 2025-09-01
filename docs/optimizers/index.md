# Optimization Algorithms Overview

Hyperactive provides access to 25+ optimization algorithms through three specialized backends, each designed for different optimization scenarios. This modular architecture allows you to choose the best optimization engine for your specific needs while maintaining a consistent API across all algorithms.

## Optimization Backends

### Gradient-Free-Optimizers 
The primary optimization engine with 16+ algorithms spanning classical to cutting-edge methods. Ideal for users who need maximum flexibility, algorithm variety, and direct control over optimization parameters.

**Best for:** Custom optimization workflows, continuous problems, research applications, and when you need access to algorithm internals.

### Optuna
Integration with the popular Optuna framework, providing 8+ state-of-the-art algorithms optimized for hyperparameter tuning. Excels at mixed parameter spaces and includes advanced features like multi-objective optimization.

**Best for:** Machine learning hyperparameter optimization, mixed parameter spaces (continuous + categorical), and when you need experiment tracking and visualization.

### Scikit-learn
Direct integration with sklearn's optimization methods, offering seamless compatibility with existing sklearn workflows through familiar GridSearchCV and RandomizedSearchCV interfaces.

**Best for:** Existing sklearn pipelines, guaranteed sklearn compatibility, and users who prefer the familiar sklearn API.

## Quick Algorithm Selection

**Just starting?** Try these proven combinations:

- **Hyperparameter tuning**: [Optuna TPE Optimizer](optuna/tpe_optimizer.md)
- **General optimization**: [GFO Bayesian Optimization](gfo/bayesian_optimization.md)
- **Sklearn integration**: [Sklearn Grid Search](sklearn/grid_search_sk.md)
- **Quick exploration**: [GFO Random Search](gfo/random_search.md)

## Algorithm Categories

**Local Search**: Hill climbing variants for fast local optimization
**Global Search**: Methods designed to find global optima
**Population-Based**: Evolutionary and swarm intelligence algorithms
**Surrogate-Based**: Machine learning-guided optimization (Bayesian, TPE)
**Multi-Objective**: Algorithms that optimize multiple objectives simultaneously

## Key Features by Backend

| Feature | GFO | Optuna | Sklearn |
|---------|-----|--------|---------|
| Algorithm Count | 20+ | 8+ | 2 |
| Parameter Types | All | All | All |
| Multi-Objective | Limited | Yes | No |
| Experiment Tracking | Basic | Advanced | Basic |
| Sklearn Integration | Yes | Yes | Native |

## Getting Started

1. **Identify your problem type** (continuous, discrete, mixed, multi-objective)
2. **Choose a backend** based on your requirements and preferences
3. **Select an algorithm** from the backend's available options
4. **Define your optimization experiment** with search space and objective function
5. **Run optimization** and analyze results

