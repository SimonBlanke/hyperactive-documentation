# Advanced Patterns

## Introduction

This page demonstrates sophisticated optimization patterns and techniques that go beyond basic usage. These patterns are essential for tackling complex real-world optimization problems that require advanced strategies, custom implementations, and integration with existing systems.

## Meta-Optimization Patterns

### Optimizer Selection Optimization

```python
from hyperactive.base import BaseExperiment
from hyperactive.opt.gfo import BayesianOptimizer, RandomSearch, ParticleSwarmOptimizer
from hyperactive.opt.optuna import TPEOptimizer, CMAESOptimizer
import numpy as np

class MetaOptimizationExperiment(BaseExperiment):
    """Optimize which optimizer to use for a given problem"""
    
    def __init__(self, target_problem, evaluation_budget=50):
        super().__init__()
        self.target_problem = target_problem
        self.evaluation_budget = evaluation_budget
        self.optimizer_registry = {
            0: lambda exp: BayesianOptimizer(experiment=exp),
            1: lambda exp: RandomSearch(experiment=exp),
            2: lambda exp: ParticleSwarmOptimizer(experiment=exp, population=20),
            3: lambda exp: TPEOptimizer(experiment=exp),
            4: lambda exp: CMAESOptimizer(experiment=exp)
        }
    
    def _paramnames(self):
        return ["optimizer_id", "early_stop_patience"]
    
    def _evaluate(self, params):
        try:
            optimizer_id = int(params["optimizer_id"]) % len(self.optimizer_registry)
            patience = max(5, int(params["early_stop_patience"]))
            
            # Create optimizer for target problem
            optimizer_factory = self.optimizer_registry[optimizer_id]
            optimizer = optimizer_factory(self.target_problem)
            
            # Run optimization with budget
            best_params = optimizer.solve()
            final_score = self.target_problem.score(best_params)[0]
            
            # Performance metrics
            convergence_score = final_score
            optimizer_name = optimizer.__class__.__name__
            
            return convergence_score, {
                "optimizer_used": optimizer_name,
                "optimizer_id": optimizer_id,
                "target_score": final_score,
                "patience_used": patience
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Example target problem
class TestTargetProblem(BaseExperiment):
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        x, y, z = params["x"], params["y"], params["z"]
        # Himmelblau's function (multimodal)
        result = -((x**2 + y - 11)**2 + (x + y**2 - 7)**2) - z**2
        return result, {}

# Run meta-optimization
target_problem = TestTargetProblem()
meta_experiment = MetaOptimizationExperiment(target_problem)
meta_optimizer = BayesianOptimizer(experiment=meta_experiment)

best_meta_params = meta_optimizer.solve()
meta_result = meta_experiment.score(best_meta_params)

print("Meta-Optimization Results:")
print(f"Best optimizer: {meta_result[1]['optimizer_used']}")
print(f"Target problem score: {meta_result[1]['target_score']:.6f}")
print(f"Meta parameters: {best_meta_params}")
```

### Hyperparameter Optimization for Optimizers

```python
class OptimizerHyperparameterOptimization(BaseExperiment):
    """Optimize hyperparameters of optimization algorithms themselves"""
    
    def __init__(self, target_experiment, base_optimizer_class):
        super().__init__()
        self.target_experiment = target_experiment
        self.base_optimizer_class = base_optimizer_class
    
    def _paramnames(self):
        # These would vary based on the optimizer being tuned
        if "ParticleSwarm" in self.base_optimizer_class.__name__:
            return ["population", "inertia", "cognitive", "social"]
        else:
            return ["param1", "param2", "param3"]  # Generic parameters
    
    def _evaluate(self, params):
        try:
            if "ParticleSwarm" in self.base_optimizer_class.__name__:
                # PSO-specific parameter tuning
                population = max(10, int(params["population"]))
                inertia = max(0.1, min(0.9, params["inertia"]))
                cognitive = max(0.5, min(2.5, params["cognitive"]))
                social = max(0.5, min(2.5, params["social"]))
                
                # Create PSO with custom parameters
                optimizer = self.base_optimizer_class(
                    experiment=self.target_experiment,
                    population=population
                    # Note: Actual PSO implementation may have different parameter names
                )
            else:
                # Generic optimizer creation
                optimizer = self.base_optimizer_class(experiment=self.target_experiment)
            
            # Run optimization
            best_params = optimizer.solve()
            target_score = self.target_experiment.score(best_params)[0]
            
            return target_score, {
                "target_score": target_score,
                "optimizer_params": params.copy(),
                "target_best_params": best_params
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Example usage
target_exp = TestTargetProblem()
optimizer_tuning = OptimizerHyperparameterOptimization(target_exp, ParticleSwarmOptimizer)
tuning_optimizer = BayesianOptimizer(experiment=optimizer_tuning)

best_optimizer_params = tuning_optimizer.solve()
print("Optimal optimizer hyperparameters:", best_optimizer_params)
```

## Dynamic and Adaptive Optimization

### Adaptive Parameter Bounds

```python
class AdaptiveBoundsExperiment(BaseExperiment):
    """Experiment that adapts parameter bounds based on performance"""
    
    def __init__(self, initial_bounds=None):
        super().__init__()
        self.evaluation_count = 0
        self.performance_history = []
        self.parameter_history = []
        self.current_bounds = initial_bounds or {"x": (-10, 10), "y": (-10, 10)}
        self.adaptation_frequency = 25  # Adapt every N evaluations
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        self.evaluation_count += 1
        x, y = params["x"], params["y"]
        
        # Objective function (Rosenbrock)
        score = -((1 - x)**2 + 100 * (y - x**2)**2)
        
        # Store history
        self.performance_history.append(score)
        self.parameter_history.append(params.copy())
        
        # Adapt bounds periodically
        if self.evaluation_count % self.adaptation_frequency == 0:
            self._adapt_bounds()
        
        return score, {
            "bounds_x": self.current_bounds["x"],
            "bounds_y": self.current_bounds["y"],
            "adaptation_count": self.evaluation_count // self.adaptation_frequency
        }
    
    def _adapt_bounds(self):
        """Adapt parameter bounds based on recent performance"""
        if len(self.performance_history) < self.adaptation_frequency:
            return
        
        # Get recent best parameters
        recent_scores = self.performance_history[-self.adaptation_frequency:]
        recent_params = self.parameter_history[-self.adaptation_frequency:]
        
        # Find best performing region
        best_idx = np.argmax(recent_scores)
        best_params = recent_params[best_idx]
        
        # Calculate parameter statistics
        x_values = [p["x"] for p in recent_params]
        y_values = [p["y"] for p in recent_params]
        
        x_mean, x_std = np.mean(x_values), np.std(x_values)
        y_mean, y_std = np.mean(y_values), np.std(y_values)
        
        # Adapt bounds (focus on promising regions)
        shrink_factor = 0.8
        expand_factor = 1.2
        
        # Shrink bounds around promising regions
        x_range = shrink_factor * max(x_std, 0.1)
        y_range = shrink_factor * max(y_std, 0.1)
        
        self.current_bounds["x"] = (
            max(best_params["x"] - x_range, self.current_bounds["x"][0]),
            min(best_params["x"] + x_range, self.current_bounds["x"][1])
        )
        
        self.current_bounds["y"] = (
            max(best_params["y"] - y_range, self.current_bounds["y"][0]), 
            min(best_params["y"] + y_range, self.current_bounds["y"][1])
        )
        
        print(f"Adapted bounds after {self.evaluation_count} evaluations:")
        print(f"  x: {self.current_bounds['x']}")
        print(f"  y: {self.current_bounds['y']}")

# Run adaptive optimization
adaptive_experiment = AdaptiveBoundsExperiment()
adaptive_optimizer = BayesianOptimizer(experiment=adaptive_experiment)
adaptive_best = adaptive_optimizer.solve()

print("Adaptive bounds optimization completed")
print("Final bounds:", adaptive_experiment.current_bounds)
print("Best parameters:", adaptive_best)
```

### Multi-Stage Optimization

```python
class MultiStageOptimization(BaseExperiment):
    """Optimization with multiple stages using different strategies"""
    
    def __init__(self):
        super().__init__()
        self.stage = 1
        self.stage_transitions = [20, 50, 100]  # Evaluation counts for stage transitions
        self.evaluation_count = 0
        self.best_so_far = float('-inf')
        self.best_params_so_far = None
    
    def _paramnames(self):
        return ["learning_rate", "batch_size", "dropout", "hidden_size"]
    
    def _evaluate(self, params):
        self.evaluation_count += 1
        
        # Determine current stage
        current_stage = 1
        for threshold in self.stage_transitions:
            if self.evaluation_count > threshold:
                current_stage += 1
            else:
                break
        
        if current_stage != self.stage:
            print(f"Transitioning to stage {current_stage} at evaluation {self.evaluation_count}")
            self.stage = current_stage
        
        # Stage-specific objective function modifications
        lr = max(1e-5, min(1.0, params["learning_rate"]))
        batch_size = max(8, int(params["batch_size"]))
        dropout = max(0, min(0.9, params["dropout"]))
        hidden_size = max(16, int(params["hidden_size"]))
        
        # Base score
        base_score = self._compute_base_score(lr, batch_size, dropout, hidden_size)
        
        # Stage-specific modifications
        if self.stage == 1:
            # Stage 1: Broad exploration - encourage diversity
            diversity_bonus = abs(lr - 0.01) + abs(dropout - 0.5)
            score = base_score + 0.1 * diversity_bonus
            stage_info = "exploration"
        elif self.stage == 2:
            # Stage 2: Focus on promising regions
            if self.best_params_so_far:
                proximity_penalty = (
                    abs(lr - self.best_params_so_far["learning_rate"]) +
                    abs(dropout - self.best_params_so_far["dropout"])
                )
                score = base_score - 0.05 * proximity_penalty
            else:
                score = base_score
            stage_info = "focused_search"
        else:
            # Stage 3: Fine-tuning around best solution
            if self.best_params_so_far:
                fine_tune_bonus = -abs(lr - self.best_params_so_far["learning_rate"]) * 10
                score = base_score + fine_tune_bonus
            else:
                score = base_score
            stage_info = "fine_tuning"
        
        # Update best solution
        if score > self.best_so_far:
            self.best_so_far = score
            self.best_params_so_far = params.copy()
        
        return score, {
            "stage": self.stage,
            "stage_info": stage_info,
            "base_score": base_score,
            "evaluation_count": self.evaluation_count
        }
    
    def _compute_base_score(self, lr, batch_size, dropout, hidden_size):
        """Simulate neural network training performance"""
        # Simplified performance model
        lr_penalty = -abs(np.log10(lr) + 3)**2  # Prefer lr around 1e-3
        batch_penalty = -abs(batch_size - 64)**2 / 1000  # Prefer batch size around 64
        dropout_bonus = dropout * (1 - dropout) * 4  # Inverted U-shape
        size_penalty = -abs(hidden_size - 128)**2 / 10000  # Prefer around 128
        
        return lr_penalty + batch_penalty + dropout_bonus + size_penalty

# Run multi-stage optimization
multistage_experiment = MultiStageOptimization()
multistage_optimizer = BayesianOptimizer(experiment=multistage_experiment)
multistage_best = multistage_optimizer.solve()

print("Multi-stage optimization completed")
print("Best parameters:", multistage_best)
print("Final stage:", multistage_experiment.stage)
```

## Ensemble and Parallel Optimization

### Optimizer Ensemble

```python
import concurrent.futures
from threading import Lock

class EnsembleOptimization:
    """Run multiple optimizers in parallel and combine results"""
    
    def __init__(self, experiment, optimizers, combination_strategy="best"):
        self.experiment = experiment
        self.optimizers = optimizers
        self.combination_strategy = combination_strategy
        self.results = {}
        self.lock = Lock()
    
    def run_single_optimizer(self, name, optimizer):
        """Run a single optimizer"""
        try:
            best_params = optimizer.solve()
            score = self.experiment.score(best_params)[0]
            
            with self.lock:
                self.results[name] = {
                    "parameters": best_params,
                    "score": score,
                    "optimizer": optimizer.__class__.__name__
                }
                print(f"Completed {name}: score = {score:.6f}")
            
            return name, best_params, score
        except Exception as e:
            with self.lock:
                self.results[name] = {
                    "error": str(e),
                    "score": float('-inf')
                }
            return name, None, float('-inf')
    
    def run_ensemble(self, max_workers=None):
        """Run all optimizers in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_single_optimizer, name, optimizer): name
                for name, optimizer in self.optimizers.items()
            }
            
            completed_results = []
            for future in concurrent.futures.as_completed(futures):
                name, params, score = future.result()
                completed_results.append((name, params, score))
        
        return self._combine_results()
    
    def _combine_results(self):
        """Combine results from multiple optimizers"""
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}
        
        if not valid_results:
            return None, float('-inf')
        
        if self.combination_strategy == "best":
            # Return best single result
            best_result = max(valid_results.items(), key=lambda x: x[1]["score"])
            return best_result[1]["parameters"], best_result[1]["score"]
        
        elif self.combination_strategy == "average":
            # Average parameters (only works for numerical parameters)
            param_names = list(next(iter(valid_results.values()))["parameters"].keys())
            averaged_params = {}
            
            for param_name in param_names:
                values = [result["parameters"][param_name] for result in valid_results.values()]
                averaged_params[param_name] = np.mean(values)
            
            avg_score = self.experiment.score(averaged_params)[0]
            return averaged_params, avg_score
        
        elif self.combination_strategy == "weighted_average":
            # Weight by performance
            total_weight = sum(max(0, result["score"]) for result in valid_results.values())
            if total_weight == 0:
                return self._combine_results()  # Fall back to best
            
            param_names = list(next(iter(valid_results.values()))["parameters"].keys())
            weighted_params = {}
            
            for param_name in param_names:
                weighted_sum = sum(
                    result["parameters"][param_name] * max(0, result["score"])
                    for result in valid_results.values()
                )
                weighted_params[param_name] = weighted_sum / total_weight
            
            weighted_score = self.experiment.score(weighted_params)[0]
            return weighted_params, weighted_score

# Example ensemble optimization
test_experiment = TestTargetProblem()

ensemble_optimizers = {
    "Bayesian": BayesianOptimizer(experiment=test_experiment),
    "Random": RandomSearch(experiment=test_experiment),
    "PSO": ParticleSwarmOptimizer(experiment=test_experiment, population=20),
    "TPE": TPEOptimizer(experiment=test_experiment)
}

# Run ensemble
ensemble = EnsembleOptimization(test_experiment, ensemble_optimizers, "best")
best_params, best_score = ensemble.run_ensemble(max_workers=2)

print("Ensemble Optimization Results:")
print("Individual results:")
for name, result in ensemble.results.items():
    if "error" not in result:
        print(f"  {name}: {result['score']:.6f}")
    else:
        print(f"  {name}: Error - {result['error']}")

print(f"\nBest ensemble result: {best_score:.6f}")
print(f"Best parameters: {best_params}")
```

### Distributed Optimization

```python
class DistributedOptimization:
    """Coordinate optimization across multiple workers"""
    
    def __init__(self, experiment, n_workers=4):
        self.experiment = experiment
        self.n_workers = n_workers
        self.shared_best = {"score": float('-inf'), "params": None}
        self.worker_results = []
        
    def worker_optimization(self, worker_id, optimizer_class, n_evaluations=25):
        """Single worker optimization process"""
        print(f"Worker {worker_id} starting optimization...")
        
        # Each worker uses a different random seed for diversity
        np.random.seed(worker_id * 42)
        
        # Create worker-specific optimizer
        optimizer = optimizer_class(experiment=self.experiment)
        
        # Run optimization
        best_params = optimizer.solve()
        score = self.experiment.score(best_params)[0]
        
        result = {
            "worker_id": worker_id,
            "best_params": best_params,
            "score": score,
            "optimizer": optimizer_class.__name__
        }
        
        self.worker_results.append(result)
        
        # Update shared best (in real distributed setup, this would be coordinated)
        if score > self.shared_best["score"]:
            self.shared_best = {"score": score, "params": best_params}
            print(f"Worker {worker_id} found new best: {score:.6f}")
        
        return result
    
    def run_distributed(self):
        """Run distributed optimization"""
        optimizer_classes = [
            BayesianOptimizer,
            RandomSearch, 
            ParticleSwarmOptimizer,
            TPEOptimizer
        ]
        
        # In a real distributed setup, this would use actual distributed computing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for i in range(self.n_workers):
                optimizer_class = optimizer_classes[i % len(optimizer_classes)]
                future = executor.submit(self.worker_optimization, i, optimizer_class)
                futures.append(future)
            
            # Wait for all workers to complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(f"Worker {result['worker_id']} completed with score {result['score']:.6f}")
        
        return self.shared_best, self.worker_results

# Run distributed optimization
distributed_opt = DistributedOptimization(test_experiment, n_workers=4)
best_result, all_results = distributed_opt.run_distributed()

print("\nDistributed Optimization Summary:")
print(f"Best overall score: {best_result['score']:.6f}")
print(f"Best parameters: {best_result['params']}")

print("\nWorker performance:")
for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
    print(f"  Worker {result['worker_id']} ({result['optimizer']}): {result['score']:.6f}")
```

## Custom Algorithm Development

### Hybrid Algorithm Implementation

```python
class HybridOptimizer:
    """Custom hybrid optimization algorithm combining multiple strategies"""
    
    def __init__(self, experiment, exploration_phase_ratio=0.3):
        self.experiment = experiment
        self.exploration_ratio = exploration_phase_ratio
        self.evaluation_count = 0
        self.max_evaluations = 100
        self.population = []
        self.best_solution = {"params": None, "score": float('-inf')}
        
    def solve(self):
        """Main optimization loop"""
        exploration_evaluations = int(self.max_evaluations * self.exploration_ratio)
        
        # Phase 1: Exploration using random search
        print("Phase 1: Exploration")
        self._exploration_phase(exploration_evaluations)
        
        # Phase 2: Exploitation using local search around best solutions
        print("Phase 2: Exploitation")
        remaining_evaluations = self.max_evaluations - exploration_evaluations
        self._exploitation_phase(remaining_evaluations)
        
        return self.best_solution["params"]
    
    def _exploration_phase(self, n_evaluations):
        """Random exploration to find promising regions"""
        param_names = self.experiment.paramnames()
        
        for _ in range(n_evaluations):
            # Generate random parameters
            random_params = {}
            for param_name in param_names:
                # Assume parameters are in [-10, 10] range
                random_params[param_name] = np.random.uniform(-10, 10)
            
            # Evaluate
            score, metadata = self.experiment.score(random_params)
            self.evaluation_count += 1
            
            # Update population
            solution = {"params": random_params, "score": score}
            self.population.append(solution)
            
            # Update best
            if score > self.best_solution["score"]:
                self.best_solution = solution.copy()
                print(f"  New best: {score:.6f}")
        
        # Keep only top solutions for exploitation
        self.population.sort(key=lambda x: x["score"], reverse=True)
        self.population = self.population[:10]  # Keep top 10
    
    def _exploitation_phase(self, n_evaluations):
        """Local search around best solutions"""
        if not self.population:
            return
        
        evaluations_per_solution = n_evaluations // len(self.population)
        
        for solution in self.population:
            self._local_search(solution["params"], evaluations_per_solution)
    
    def _local_search(self, center_params, n_evaluations):
        """Local search around a center point"""
        param_names = list(center_params.keys())
        step_size = 1.0
        
        for _ in range(n_evaluations):
            # Generate neighbor
            neighbor_params = center_params.copy()
            
            # Perturb parameters
            for param_name in param_names:
                perturbation = np.random.normal(0, step_size)
                neighbor_params[param_name] += perturbation
            
            # Evaluate neighbor
            score, metadata = self.experiment.score(neighbor_params)
            self.evaluation_count += 1
            
            # Update best if improved
            if score > self.best_solution["score"]:
                self.best_solution = {
                    "params": neighbor_params,
                    "score": score
                }
                print(f"  Local search improvement: {score:.6f}")
                
                # Successful move - update center
                center_params = neighbor_params.copy()
            else:
                # Reduce step size if no improvement
                step_size *= 0.95

# Test hybrid algorithm
hybrid_optimizer = HybridOptimizer(test_experiment)
hybrid_best = hybrid_optimizer.solve()

print("Hybrid Algorithm Results:")
print(f"Best score: {hybrid_optimizer.best_solution['score']:.6f}")
print(f"Best parameters: {hybrid_best}")
print(f"Total evaluations: {hybrid_optimizer.evaluation_count}")
```

### Adaptive Learning Rate Optimization

```python
class AdaptiveLearningRateOptimizer:
    """Optimizer with adaptive learning rate based on performance"""
    
    def __init__(self, experiment, initial_lr=0.1):
        self.experiment = experiment
        self.current_lr = initial_lr
        self.current_solution = None
        self.performance_history = []
        self.gradient_estimates = {}
        
    def solve(self, max_iterations=100):
        """Gradient-free optimization with adaptive learning rate"""
        param_names = self.experiment.paramnames()
        
        # Initialize random solution
        self.current_solution = {
            param: np.random.uniform(-5, 5) for param in param_names
        }
        
        current_score, _ = self.experiment.score(self.current_solution)
        self.performance_history.append(current_score)
        
        print(f"Initial solution: score = {current_score:.6f}")
        
        for iteration in range(max_iterations):
            # Estimate gradients using finite differences
            gradients = self._estimate_gradients()
            
            # Update solution using estimated gradients
            new_solution = {}
            for param_name in param_names:
                gradient = gradients.get(param_name, 0)
                new_solution[param_name] = (
                    self.current_solution[param_name] + 
                    self.current_lr * gradient
                )
            
            # Evaluate new solution
            new_score, _ = self.experiment.score(new_solution)
            
            # Adaptive learning rate
            if new_score > current_score:
                # Success - increase learning rate
                self.current_lr *= 1.1
                self.current_solution = new_solution
                current_score = new_score
                print(f"Iteration {iteration}: improved to {current_score:.6f}, lr = {self.current_lr:.4f}")
            else:
                # Failure - decrease learning rate
                self.current_lr *= 0.9
                # Don't update solution
                print(f"Iteration {iteration}: no improvement, lr = {self.current_lr:.4f}")
            
            self.performance_history.append(current_score)
            
            # Early stopping
            if self.current_lr < 1e-6:
                print("Learning rate too small, stopping")
                break
        
        return self.current_solution
    
    def _estimate_gradients(self, epsilon=1e-3):
        """Estimate gradients using finite differences"""
        gradients = {}
        base_score, _ = self.experiment.score(self.current_solution)
        
        for param_name in self.current_solution.keys():
            # Forward difference
            perturbed_solution = self.current_solution.copy()
            perturbed_solution[param_name] += epsilon
            
            perturbed_score, _ = self.experiment.score(perturbed_solution)
            gradient = (perturbed_score - base_score) / epsilon
            gradients[param_name] = gradient
        
        return gradients

# Test adaptive learning rate optimizer
adaptive_optimizer = AdaptiveLearningRateOptimizer(test_experiment)
adaptive_best = adaptive_optimizer.solve()

print("Adaptive Learning Rate Results:")
print(f"Best parameters: {adaptive_best}")
print(f"Final learning rate: {adaptive_optimizer.current_lr:.6f}")
print(f"Performance improvement: {adaptive_optimizer.performance_history[-1] - adaptive_optimizer.performance_history[0]:.6f}")
```

## Advanced Integration Patterns

### Optimization with External Constraints

```python
class ConstrainedSystemOptimization(BaseExperiment):
    """Optimization with external system constraints and real-world feedback"""
    
    def __init__(self, constraint_checker=None):
        super().__init__()
        self.constraint_checker = constraint_checker
        self.constraint_violations = []
        self.feasible_solutions = []
        
    def _paramnames(self):
        return ["system_param1", "system_param2", "system_param3"]
    
    def _evaluate(self, params):
        try:
            # Check external constraints first
            if self.constraint_checker:
                constraint_status = self.constraint_checker(params)
                if not constraint_status["feasible"]:
                    self.constraint_violations.append(params.copy())
                    return float('-inf'), {
                        "feasible": False,
                        "constraint_violations": constraint_status["violations"]
                    }
            
            # Simulate expensive system evaluation
            system_performance = self._simulate_system_performance(params)
            
            # Check system-specific constraints
            if system_performance["temperature"] > 100:  # Overheating
                return float('-inf'), {
                    "feasible": False, 
                    "reason": "overheating",
                    "temperature": system_performance["temperature"]
                }
            
            if system_performance["power_consumption"] > 500:  # Power limit
                return float('-inf'), {
                    "feasible": False,
                    "reason": "power_limit",
                    "power": system_performance["power_consumption"]
                }
            
            # Calculate objective (efficiency)
            efficiency = (system_performance["output"] / 
                         system_performance["power_consumption"])
            
            self.feasible_solutions.append({
                "params": params.copy(),
                "efficiency": efficiency,
                "performance": system_performance
            })
            
            return efficiency, {
                "feasible": True,
                "efficiency": efficiency,
                "output": system_performance["output"],
                "power": system_performance["power_consumption"],
                "temperature": system_performance["temperature"]
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}
    
    def _simulate_system_performance(self, params):
        """Simulate complex system behavior"""
        p1, p2, p3 = params["system_param1"], params["system_param2"], params["system_param3"]
        
        # Simulate nonlinear system response
        output = 100 * (1 - np.exp(-p1/5)) * (1 + p2/10) * (1 - abs(p3)/20)
        power = 50 + p1**2 + p2**2 + abs(p3) * 10
        temperature = 20 + power/10 + np.random.normal(0, 5)  # Noisy temperature
        
        return {
            "output": max(0, output),
            "power_consumption": max(1, power), 
            "temperature": temperature
        }

def external_constraint_checker(params):
    """External constraint checking function"""
    violations = []
    
    # Physical constraints
    if params["system_param1"] < 0 or params["system_param1"] > 20:
        violations.append("param1_out_of_range")
    
    if params["system_param2"] < -10 or params["system_param2"] > 15:
        violations.append("param2_out_of_range")
    
    # Interaction constraints
    if params["system_param1"] + params["system_param2"] > 25:
        violations.append("combined_limit_exceeded")
    
    return {
        "feasible": len(violations) == 0,
        "violations": violations
    }

# Run constrained optimization
constrained_experiment = ConstrainedSystemOptimization(external_constraint_checker)
constrained_optimizer = BayesianOptimizer(experiment=constrained_experiment)
constrained_best = constrained_optimizer.solve()

print("Constrained System Optimization:")
if constrained_experiment.feasible_solutions:
    best_feasible = max(constrained_experiment.feasible_solutions, 
                       key=lambda x: x["efficiency"])
    print(f"Best feasible efficiency: {best_feasible['efficiency']:.4f}")
    print(f"Best parameters: {best_feasible['params']}")
    print(f"System performance: {best_feasible['performance']}")
else:
    print("No feasible solutions found")

print(f"Constraint violations encountered: {len(constrained_experiment.constraint_violations)}")
```

### Real-time Optimization with Feedback

```python
import time

class RealTimeOptimization:
    """Real-time optimization with continuous feedback and adaptation"""
    
    def __init__(self, experiment, feedback_frequency=5):
        self.experiment = experiment
        self.feedback_frequency = feedback_frequency
        self.optimization_history = []
        self.performance_trend = []
        self.current_best = {"params": None, "score": float('-inf')}
        self.running = False
        
    def start_optimization(self, duration_seconds=30):
        """Start real-time optimization for specified duration"""
        self.running = True
        start_time = time.time()
        iteration = 0
        
        print(f"Starting real-time optimization for {duration_seconds} seconds...")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            iteration += 1
            
            # Get current system state (simulated)
            system_state = self._get_system_state()
            
            # Adaptive parameter generation based on current state
            candidate_params = self._generate_adaptive_candidate(system_state)
            
            # Evaluate candidate
            score, metadata = self.experiment.score(candidate_params)
            
            # Update history
            self.optimization_history.append({
                "iteration": iteration,
                "timestamp": time.time() - start_time,
                "params": candidate_params,
                "score": score,
                "system_state": system_state
            })
            
            # Update best solution
            if score > self.current_best["score"]:
                self.current_best = {"params": candidate_params, "score": score}
                print(f"Iteration {iteration}: New best score {score:.6f}")
            
            # Periodic feedback and adaptation
            if iteration % self.feedback_frequency == 0:
                self._process_feedback()
            
            # Simulate real-time delay
            time.sleep(0.1)
        
        self.running = False
        print(f"Real-time optimization completed after {iteration} iterations")
        return self.current_best
    
    def _get_system_state(self):
        """Get current system state (simulated)"""
        return {
            "load": np.random.uniform(0.5, 1.5),
            "temperature": 20 + np.random.normal(0, 5),
            "noise_level": np.random.uniform(0, 0.1)
        }
    
    def _generate_adaptive_candidate(self, system_state):
        """Generate candidate parameters adapted to current system state"""
        param_names = self.experiment.paramnames()
        
        if self.current_best["params"] is None:
            # Initial random candidate
            return {param: np.random.uniform(-5, 5) for param in param_names}
        
        # Adaptive candidate based on system state and current best
        adaptive_params = self.current_best["params"].copy()
        
        # Adapt based on system load
        load_factor = system_state["load"]
        adaptation_scale = 0.1 * load_factor
        
        for param_name in param_names:
            # Add adaptive noise
            noise = np.random.normal(0, adaptation_scale)
            adaptive_params[param_name] += noise
            
            # Clip to reasonable bounds
            adaptive_params[param_name] = np.clip(adaptive_params[param_name], -10, 10)
        
        return adaptive_params
    
    def _process_feedback(self):
        """Process feedback and adapt optimization strategy"""
        if len(self.optimization_history) < self.feedback_frequency:
            return
        
        # Analyze recent performance
        recent_scores = [h["score"] for h in self.optimization_history[-self.feedback_frequency:]]
        trend = np.mean(np.diff(recent_scores)) if len(recent_scores) > 1 else 0
        
        self.performance_trend.append(trend)
        
        # Adaptation based on trend
        if trend > 0:
            print(f"  Positive trend detected: {trend:.6f}")
        elif trend < -0.01:
            print(f"  Negative trend detected: {trend:.6f} - increasing exploration")
        
    def stop_optimization(self):
        """Stop real-time optimization"""
        self.running = False

# Run real-time optimization
realtime_experiment = TestTargetProblem()
realtime_optimizer = RealTimeOptimization(realtime_experiment)

# Start optimization (shorter duration for demo)
final_result = realtime_optimizer.start_optimization(duration_seconds=10)

print("Real-time Optimization Results:")
print(f"Final best score: {final_result['score']:.6f}")
print(f"Final parameters: {final_result['params']}")
print(f"Total iterations: {len(realtime_optimizer.optimization_history)}")

# Analyze performance over time
if realtime_optimizer.optimization_history:
    scores = [h["score"] for h in realtime_optimizer.optimization_history]
    print(f"Score improvement: {scores[-1] - scores[0]:.6f}")
    print(f"Best score achieved: {max(scores):.6f}")
```

## Best Practices Summary

### Advanced Optimization Checklist

1. **Meta-Optimization**: Consider optimizing the optimization process itself
2. **Multi-Stage Strategies**: Use different approaches for exploration and exploitation
3. **Ensemble Methods**: Combine multiple optimizers for robustness
4. **Adaptive Techniques**: Adjust parameters based on performance feedback
5. **Constraint Handling**: Implement proper constraint checking and penalty methods
6. **Real-time Considerations**: Design for continuous optimization in dynamic environments
7. **Performance Monitoring**: Track and analyze optimization progress
8. **Distributed Computing**: Leverage parallel processing for complex problems

### Integration Guidelines

1. **Modular Design**: Keep optimization components separate and reusable
2. **Error Handling**: Implement robust error handling for production systems
3. **Logging and Monitoring**: Track optimization progress and performance
4. **Configuration Management**: Make optimization parameters configurable
5. **Testing**: Test optimization components thoroughly
6. **Documentation**: Document custom algorithms and their parameters

## References

- Advanced optimization techniques in literature
- Meta-learning and AutoML research
- Distributed computing patterns for optimization
- Real-time system optimization methodologies