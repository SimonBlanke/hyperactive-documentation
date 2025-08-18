# Custom Algorithms

## Introduction

While Hyperactive provides a rich set of optimization algorithms through its backends, there are scenarios where you need to implement custom optimization strategies. This page guides you through creating custom algorithms, extending existing ones, and integrating domain-specific optimization techniques into the Hyperactive framework.

## Custom Algorithm Architecture

### BaseOptimizer Interface

```python
from hyperactive.base import BaseOptimizer
import numpy as np

class CustomOptimizer(BaseOptimizer):
    """Template for implementing custom optimization algorithms"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment)
        
        # Custom algorithm parameters
        self.max_iterations = kwargs.get('max_iterations', 100)
        self.population_size = kwargs.get('population_size', 20)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        
        # Algorithm state
        self.iteration = 0
        self.population = []
        self.best_solution = None
        self.best_score = float('-inf')
        self.convergence_history = []
        
    def solve(self):
        """Main optimization loop - implement this method"""
        # Initialize algorithm
        self._initialize()
        
        # Main optimization loop
        for self.iteration in range(self.max_iterations):
            # Generate new candidates
            candidates = self._generate_candidates()
            
            # Evaluate candidates
            evaluated_candidates = self._evaluate_candidates(candidates)
            
            # Update population/best solution
            self._update_population(evaluated_candidates)
            
            # Check convergence
            if self._check_convergence():
                break
                
            # Algorithm-specific updates
            self._update_algorithm_state()
        
        return self.best_solution
    
    def _initialize(self):
        """Initialize algorithm state"""
        param_names = self.experiment.paramnames()
        
        # Generate initial population
        for _ in range(self.population_size):
            solution = {}
            for param_name in param_names:
                # Random initialization (customize bounds as needed)
                solution[param_name] = np.random.uniform(-10, 10)
            self.population.append(solution)
        
        # Evaluate initial population
        evaluated_pop = self._evaluate_candidates(self.population)
        self._update_population(evaluated_pop)
    
    def _generate_candidates(self):
        """Generate new candidate solutions - override this method"""
        raise NotImplementedError("Subclasses must implement _generate_candidates")
    
    def _evaluate_candidates(self, candidates):
        """Evaluate candidate solutions"""
        evaluated = []
        for candidate in candidates:
            score, metadata = self.experiment.score(candidate)
            evaluated.append({
                'solution': candidate,
                'score': score,
                'metadata': metadata
            })
        return evaluated
    
    def _update_population(self, evaluated_candidates):
        """Update population and track best solution"""
        for candidate in evaluated_candidates:
            if candidate['score'] > self.best_score:
                self.best_score = candidate['score']
                self.best_solution = candidate['solution'].copy()
        
        self.convergence_history.append(self.best_score)
    
    def _check_convergence(self):
        """Check convergence criteria"""
        if len(self.convergence_history) < 10:
            return False
        
        # Check if improvement has stagnated
        recent_scores = self.convergence_history[-10:]
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < 1e-6
    
    def _update_algorithm_state(self):
        """Update algorithm-specific state - override as needed"""
        pass
```

## Evolutionary Algorithms

### Custom Genetic Algorithm

```python
class CustomGeneticAlgorithm(CustomOptimizer):
    """Custom Genetic Algorithm implementation"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # GA-specific parameters
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.tournament_size = kwargs.get('tournament_size', 3)
        self.elitism_ratio = kwargs.get('elitism_ratio', 0.1)
        
        # Track population fitness
        self.population_fitness = []
    
    def _generate_candidates(self):
        """Generate new candidates through genetic operations"""
        if not self.population:
            return []
        
        new_population = []
        elite_count = int(self.population_size * self.elitism_ratio)
        
        # Elitism: keep best individuals
        sorted_pop = sorted(
            zip(self.population, self.population_fitness),
            key=lambda x: x[1], reverse=True
        )
        
        for i in range(elite_count):
            new_population.append(sorted_pop[i][0].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _tournament_selection(self):
        """Tournament selection for parent selection"""
        tournament_indices = np.random.choice(
            len(self.population), 
            size=self.tournament_size, 
            replace=False
        )
        
        best_idx = max(tournament_indices, key=lambda i: self.population_fitness[i])
        return self.population[best_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Uniform crossover between two parents"""
        child1, child2 = parent1.copy(), parent2.copy()
        param_names = list(parent1.keys())
        
        for param_name in param_names:
            if np.random.random() < 0.5:
                # Swap parameter values
                child1[param_name], child2[param_name] = \
                    child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual):
        """Gaussian mutation"""
        mutated = individual.copy()
        param_names = list(individual.keys())
        
        for param_name in param_names:
            if np.random.random() < self.mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, 1)
                mutated[param_name] += noise
                
                # Apply bounds (customize as needed)
                mutated[param_name] = np.clip(mutated[param_name], -20, 20)
        
        return mutated
    
    def _update_population(self, evaluated_candidates):
        """Update population and fitness tracking"""
        super()._update_population(evaluated_candidates)
        
        # Update population and fitness arrays
        self.population = [c['solution'] for c in evaluated_candidates]
        self.population_fitness = [c['score'] for c in evaluated_candidates]

# Example usage
from hyperactive.base import BaseExperiment

class TestFunction(BaseExperiment):
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        x, y, z = params["x"], params["y"], params["z"]
        # Rastrigin function
        A = 10
        n = 3
        result = A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in [x, y, z])
        return -result, {"rastrigin_value": result}

# Test custom GA
test_experiment = TestFunction()
custom_ga = CustomGeneticAlgorithm(
    experiment=test_experiment,
    max_iterations=50,
    population_size=30,
    mutation_rate=0.15,
    crossover_rate=0.85
)

ga_best = custom_ga.solve()
print("Custom Genetic Algorithm Results:")
print(f"Best parameters: {ga_best}")
print(f"Best score: {custom_ga.best_score:.6f}")
print(f"Convergence achieved at iteration: {custom_ga.iteration}")
```

### Differential Evolution

```python
class DifferentialEvolution(CustomOptimizer):
    """Custom Differential Evolution algorithm"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # DE-specific parameters
        self.F = kwargs.get('F', 0.8)  # Mutation factor
        self.CR = kwargs.get('CR', 0.9)  # Crossover probability
        self.strategy = kwargs.get('strategy', 'DE/rand/1')
        
    def _generate_candidates(self):
        """Generate trial vectors using DE mutation and crossover"""
        if not self.population:
            return []
        
        param_names = list(self.population[0].keys())
        trial_vectors = []
        
        for i, target in enumerate(self.population):
            # Mutation
            mutant = self._mutate_de(i)
            
            # Crossover
            trial = self._crossover_de(target, mutant, param_names)
            
            trial_vectors.append(trial)
        
        return trial_vectors
    
    def _mutate_de(self, target_idx):
        """DE mutation operation"""
        pop_size = len(self.population)
        param_names = list(self.population[0].keys())
        
        # Select random individuals (different from target)
        candidates = list(range(pop_size))
        candidates.remove(target_idx)
        r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
        
        mutant = {}
        if self.strategy == 'DE/rand/1':
            for param_name in param_names:
                mutant[param_name] = (
                    self.population[r1][param_name] +
                    self.F * (self.population[r2][param_name] - self.population[r3][param_name])
                )
        elif self.strategy == 'DE/best/1':
            best_idx = np.argmax(self.population_fitness)
            for param_name in param_names:
                mutant[param_name] = (
                    self.population[best_idx][param_name] +
                    self.F * (self.population[r1][param_name] - self.population[r2][param_name])
                )
        
        return mutant
    
    def _crossover_de(self, target, mutant, param_names):
        """DE crossover operation"""
        trial = {}
        j_rand = np.random.randint(len(param_names))  # Ensure at least one parameter from mutant
        
        for j, param_name in enumerate(param_names):
            if np.random.random() < self.CR or j == j_rand:
                trial[param_name] = mutant[param_name]
            else:
                trial[param_name] = target[param_name]
        
        return trial
    
    def _update_population(self, evaluated_candidates):
        """Update population using selection"""
        if not hasattr(self, 'population_fitness'):
            self.population_fitness = [float('-inf')] * len(self.population)
        
        # Selection: replace if trial is better than target
        for i, trial_candidate in enumerate(evaluated_candidates):
            if trial_candidate['score'] > self.population_fitness[i]:
                self.population[i] = trial_candidate['solution']
                self.population_fitness[i] = trial_candidate['score']
                
                # Update global best
                if trial_candidate['score'] > self.best_score:
                    self.best_score = trial_candidate['score']
                    self.best_solution = trial_candidate['solution'].copy()
        
        self.convergence_history.append(self.best_score)

# Test Differential Evolution
de_optimizer = DifferentialEvolution(
    experiment=test_experiment,
    max_iterations=100,
    population_size=30,
    F=0.7,
    CR=0.8,
    strategy='DE/rand/1'
)

de_best = de_optimizer.solve()
print("\nDifferential Evolution Results:")
print(f"Best parameters: {de_best}")
print(f"Best score: {de_optimizer.best_score:.6f}")
```

## Swarm Intelligence Algorithms

### Custom Particle Swarm Optimization

```python
class CustomParticleSwarmOptimization(CustomOptimizer):
    """Custom Particle Swarm Optimization implementation"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # PSO-specific parameters
        self.w = kwargs.get('w', 0.7)  # Inertia weight
        self.c1 = kwargs.get('c1', 2.0)  # Cognitive coefficient
        self.c2 = kwargs.get('c2', 2.0)  # Social coefficient
        self.velocity_max = kwargs.get('velocity_max', 5.0)
        
        # PSO state
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_scores = []
        self.global_best_position = None
        self.global_best_score = float('-inf')
    
    def _initialize(self):
        """Initialize PSO with positions and velocities"""
        super()._initialize()
        
        param_names = self.experiment.paramnames()
        
        # Initialize velocities
        self.velocities = []
        for _ in range(self.population_size):
            velocity = {}
            for param_name in param_names:
                velocity[param_name] = np.random.uniform(-1, 1)
            self.velocities.append(velocity)
        
        # Initialize personal bests
        self.personal_best_positions = [p.copy() for p in self.population]
        
        # Evaluate initial population for personal bests
        initial_evaluated = self._evaluate_candidates(self.population)
        self.personal_best_scores = [c['score'] for c in initial_evaluated]
        
        # Find initial global best
        best_idx = np.argmax(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]
    
    def _generate_candidates(self):
        """Update particle positions using PSO equations"""
        param_names = list(self.population[0].keys())
        new_positions = []
        
        for i in range(self.population_size):
            new_position = {}
            
            for param_name in param_names:
                # PSO velocity update equation
                r1, r2 = np.random.random(), np.random.random()
                
                cognitive_component = (self.c1 * r1 * 
                    (self.personal_best_positions[i][param_name] - self.population[i][param_name]))
                
                social_component = (self.c2 * r2 * 
                    (self.global_best_position[param_name] - self.population[i][param_name]))
                
                # Update velocity
                self.velocities[i][param_name] = (
                    self.w * self.velocities[i][param_name] +
                    cognitive_component +
                    social_component
                )
                
                # Clamp velocity
                self.velocities[i][param_name] = np.clip(
                    self.velocities[i][param_name], 
                    -self.velocity_max, 
                    self.velocity_max
                )
                
                # Update position
                new_position[param_name] = (
                    self.population[i][param_name] + self.velocities[i][param_name]
                )
                
                # Apply bounds (customize as needed)
                new_position[param_name] = np.clip(new_position[param_name], -20, 20)
            
            new_positions.append(new_position)
        
        return new_positions
    
    def _update_population(self, evaluated_candidates):
        """Update PSO population and personal/global bests"""
        # Update positions
        self.population = [c['solution'] for c in evaluated_candidates]
        
        # Update personal bests
        for i, candidate in enumerate(evaluated_candidates):
            if candidate['score'] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = candidate['score']
                self.personal_best_positions[i] = candidate['solution'].copy()
        
        # Update global best
        best_idx = np.argmax(self.personal_best_scores)
        if self.personal_best_scores[best_idx] > self.global_best_score:
            self.global_best_score = self.personal_best_scores[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            self.best_solution = self.global_best_position.copy()
            self.best_score = self.global_best_score
        
        self.convergence_history.append(self.best_score)
    
    def _update_algorithm_state(self):
        """Update inertia weight (linearly decreasing)"""
        # Linearly decrease inertia weight
        w_max, w_min = 0.9, 0.4
        self.w = w_max - (w_max - w_min) * (self.iteration / self.max_iterations)

# Test Custom PSO
pso_optimizer = CustomParticleSwarmOptimization(
    experiment=test_experiment,
    max_iterations=100,
    population_size=25,
    w=0.7,
    c1=2.0,
    c2=2.0
)

pso_best = pso_optimizer.solve()
print("\nCustom PSO Results:")
print(f"Best parameters: {pso_best}")
print(f"Best score: {pso_optimizer.best_score:.6f}")
```

### Ant Colony Optimization for Continuous Problems

```python
class ContinuousAntColonyOptimization(CustomOptimizer):
    """Ant Colony Optimization adapted for continuous problems"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # ACO parameters
        self.archive_size = kwargs.get('archive_size', 10)
        self.gaussian_std = kwargs.get('gaussian_std', 1.0)
        self.std_decay = kwargs.get('std_decay', 0.99)
        
        # ACO state
        self.archive = []  # Archive of good solutions
        self.weights = []  # Weights for archive solutions
    
    def _generate_candidates(self):
        """Generate candidates using Gaussian kernels around archive solutions"""
        if not self.archive:
            # Random initialization if no archive
            return super()._generate_candidates()
        
        param_names = list(self.archive[0]['solution'].keys())
        candidates = []
        
        for _ in range(self.population_size):
            # Select archive solution based on weights
            selected_idx = np.random.choice(len(self.archive), p=self.weights)
            selected_solution = self.archive[selected_idx]['solution']
            
            # Generate candidate around selected solution
            candidate = {}
            for param_name in param_names:
                # Add Gaussian noise
                noise = np.random.normal(0, self.gaussian_std)
                candidate[param_name] = selected_solution[param_name] + noise
                
                # Apply bounds
                candidate[param_name] = np.clip(candidate[param_name], -20, 20)
            
            candidates.append(candidate)
        
        return candidates
    
    def _update_population(self, evaluated_candidates):
        """Update archive and weights"""
        super()._update_population(evaluated_candidates)
        
        # Add new solutions to archive
        for candidate in evaluated_candidates:
            self.archive.append(candidate)
        
        # Sort archive by score and keep best solutions
        self.archive.sort(key=lambda x: x['score'], reverse=True)
        self.archive = self.archive[:self.archive_size]
        
        # Update weights (quality-based)
        if self.archive:
            scores = [sol['score'] for sol in self.archive]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score > min_score:
                # Normalize scores to [0, 1] and convert to weights
                normalized_scores = [(s - min_score) / (max_score - min_score) 
                                   for s in scores]
                weights = np.array(normalized_scores) + 0.1  # Add small base weight
                self.weights = weights / np.sum(weights)
            else:
                # Equal weights if all scores are the same
                self.weights = np.ones(len(self.archive)) / len(self.archive)
    
    def _update_algorithm_state(self):
        """Decay Gaussian standard deviation"""
        self.gaussian_std *= self.std_decay

# Test ACO
aco_optimizer = ContinuousAntColonyOptimization(
    experiment=test_experiment,
    max_iterations=100,
    population_size=20,
    archive_size=8,
    gaussian_std=2.0
)

aco_best = aco_optimizer.solve()
print("\nAnt Colony Optimization Results:")
print(f"Best parameters: {aco_best}")
print(f"Best score: {aco_optimizer.best_score:.6f}")
```

## Hybrid Algorithms

### Memetic Algorithm (GA + Local Search)

```python
class MemeticAlgorithm(CustomGeneticAlgorithm):
    """Hybrid algorithm combining GA with local search"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # Memetic-specific parameters
        self.local_search_prob = kwargs.get('local_search_prob', 0.3)
        self.local_search_iterations = kwargs.get('local_search_iterations', 10)
        self.step_size = kwargs.get('step_size', 0.1)
    
    def _update_population(self, evaluated_candidates):
        """Update population and apply local search to some individuals"""
        super()._update_population(evaluated_candidates)
        
        # Apply local search to selected individuals
        improved_population = []
        for i, individual in enumerate(self.population):
            if np.random.random() < self.local_search_prob:
                # Apply local search
                improved = self._local_search(individual, self.population_fitness[i])
                improved_population.append(improved)
            else:
                improved_population.append({
                    'solution': individual,
                    'score': self.population_fitness[i]
                })
        
        # Update population with improved solutions
        self.population = [ind['solution'] for ind in improved_population]
        self.population_fitness = [ind['score'] for ind in improved_population]
        
        # Update global best
        best_idx = np.argmax(self.population_fitness)
        if self.population_fitness[best_idx] > self.best_score:
            self.best_score = self.population_fitness[best_idx]
            self.best_solution = self.population[best_idx].copy()
    
    def _local_search(self, solution, current_score):
        """Hill climbing local search"""
        param_names = list(solution.keys())
        best_local = solution.copy()
        best_local_score = current_score
        
        for _ in range(self.local_search_iterations):
            # Generate neighbor
            neighbor = best_local.copy()
            
            # Perturb one random parameter
            param_to_change = np.random.choice(param_names)
            perturbation = np.random.normal(0, self.step_size)
            neighbor[param_to_change] += perturbation
            
            # Apply bounds
            neighbor[param_to_change] = np.clip(neighbor[param_to_change], -20, 20)
            
            # Evaluate neighbor
            score, _ = self.experiment.score(neighbor)
            
            # Update if better
            if score > best_local_score:
                best_local = neighbor
                best_local_score = score
        
        return {'solution': best_local, 'score': best_local_score}

# Test Memetic Algorithm
memetic_optimizer = MemeticAlgorithm(
    experiment=test_experiment,
    max_iterations=50,
    population_size=20,
    local_search_prob=0.4,
    local_search_iterations=5
)

memetic_best = memetic_optimizer.solve()
print("\nMemetic Algorithm Results:")
print(f"Best parameters: {memetic_best}")
print(f"Best score: {memetic_optimizer.best_score:.6f}")
```

## Adaptive and Self-Organizing Algorithms

### Adaptive Parameter Control

```python
class AdaptiveGeneticAlgorithm(CustomGeneticAlgorithm):
    """GA with adaptive parameter control"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # Adaptive parameters
        self.min_mutation_rate = kwargs.get('min_mutation_rate', 0.01)
        self.max_mutation_rate = kwargs.get('max_mutation_rate', 0.3)
        self.adaptation_window = kwargs.get('adaptation_window', 10)
        
        # Tracking for adaptation
        self.performance_history = []
        self.diversity_history = []
    
    def _update_algorithm_state(self):
        """Adapt mutation rate based on population diversity and performance"""
        # Calculate population diversity
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        # Calculate performance trend
        if len(self.convergence_history) >= 2:
            performance_trend = (self.convergence_history[-1] - 
                               self.convergence_history[-2])
            self.performance_history.append(performance_trend)
        
        # Adapt mutation rate
        if len(self.diversity_history) >= self.adaptation_window:
            recent_diversity = np.mean(self.diversity_history[-self.adaptation_window:])
            recent_performance = np.mean(self.performance_history[-self.adaptation_window:]) if self.performance_history else 0
            
            # Increase mutation if diversity is low or performance is stagnant
            if recent_diversity < 0.1 or recent_performance < 1e-6:
                self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.1)
            else:
                self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.95)
            
            print(f"Iteration {self.iteration}: Adapted mutation rate to {self.mutation_rate:.4f}")
    
    def _calculate_diversity(self):
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0
        
        param_names = list(self.population[0].keys())
        total_variance = 0
        
        for param_name in param_names:
            values = [ind[param_name] for ind in self.population]
            variance = np.var(values)
            total_variance += variance
        
        return total_variance / len(param_names)

# Test Adaptive GA
adaptive_ga = AdaptiveGeneticAlgorithm(
    experiment=test_experiment,
    max_iterations=100,
    population_size=30,
    mutation_rate=0.1,
    adaptation_window=15
)

adaptive_best = adaptive_ga.solve()
print("\nAdaptive GA Results:")
print(f"Best parameters: {adaptive_best}")
print(f"Best score: {adaptive_ga.best_score:.6f}")
print(f"Final mutation rate: {adaptive_ga.mutation_rate:.4f}")
```

## Domain-Specific Algorithms

### Simulated Annealing with Custom Cooling Schedule

```python
class CustomSimulatedAnnealing(CustomOptimizer):
    """Simulated Annealing with custom cooling schedules"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # SA parameters
        self.initial_temperature = kwargs.get('initial_temperature', 100.0)
        self.cooling_schedule = kwargs.get('cooling_schedule', 'exponential')
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)
        self.min_temperature = kwargs.get('min_temperature', 0.01)
        
        # SA state
        self.current_solution = None
        self.current_score = float('-inf')
        self.temperature = self.initial_temperature
        self.accepted_moves = 0
        self.rejected_moves = 0
    
    def solve(self):
        """SA main loop"""
        # Initialize
        param_names = self.experiment.paramnames()
        self.current_solution = {
            param: np.random.uniform(-10, 10) for param in param_names
        }
        self.current_score, _ = self.experiment.score(self.current_solution)
        
        self.best_solution = self.current_solution.copy()
        self.best_score = self.current_score
        
        # SA loop
        for self.iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = self._generate_neighbor()
            neighbor_score, _ = self.experiment.score(neighbor)
            
            # Acceptance criterion
            if self._accept_move(neighbor_score):
                self.current_solution = neighbor
                self.current_score = neighbor_score
                self.accepted_moves += 1
                
                # Update best
                if neighbor_score > self.best_score:
                    self.best_solution = neighbor.copy()
                    self.best_score = neighbor_score
            else:
                self.rejected_moves += 1
            
            # Update temperature
            self._update_temperature()
            
            self.convergence_history.append(self.best_score)
            
            # Early stopping
            if self.temperature < self.min_temperature:
                break
        
        return self.best_solution
    
    def _generate_neighbor(self):
        """Generate neighbor solution"""
        neighbor = self.current_solution.copy()
        param_names = list(neighbor.keys())
        
        # Perturb random parameter
        param_to_change = np.random.choice(param_names)
        
        # Temperature-dependent step size
        step_size = self.temperature / self.initial_temperature * 2.0
        perturbation = np.random.normal(0, step_size)
        
        neighbor[param_to_change] += perturbation
        neighbor[param_to_change] = np.clip(neighbor[param_to_change], -20, 20)
        
        return neighbor
    
    def _accept_move(self, neighbor_score):
        """Simulated annealing acceptance criterion"""
        if neighbor_score > self.current_score:
            return True  # Always accept better solutions
        
        # Probabilistic acceptance for worse solutions
        delta = neighbor_score - self.current_score
        probability = np.exp(delta / self.temperature)
        return np.random.random() < probability
    
    def _update_temperature(self):
        """Update temperature based on cooling schedule"""
        if self.cooling_schedule == 'exponential':
            self.temperature *= self.cooling_rate
        elif self.cooling_schedule == 'linear':
            self.temperature = (self.initial_temperature * 
                              (1 - self.iteration / self.max_iterations))
        elif self.cooling_schedule == 'logarithmic':
            self.temperature = self.initial_temperature / np.log(2 + self.iteration)
        elif self.cooling_schedule == 'adaptive':
            # Adaptive cooling based on acceptance rate
            if self.iteration > 0 and (self.iteration % 20) == 0:
                acceptance_rate = self.accepted_moves / (self.accepted_moves + self.rejected_moves)
                if acceptance_rate > 0.6:
                    self.temperature *= 0.9  # Cool faster if accepting too many
                elif acceptance_rate < 0.2:
                    self.temperature *= 1.1  # Cool slower if rejecting too many
                
                # Reset counters
                self.accepted_moves = 0
                self.rejected_moves = 0

# Test custom SA with different cooling schedules
cooling_schedules = ['exponential', 'linear', 'logarithmic', 'adaptive']

print("\nSimulated Annealing Comparison:")
for schedule in cooling_schedules:
    sa_optimizer = CustomSimulatedAnnealing(
        experiment=test_experiment,
        max_iterations=200,
        initial_temperature=50.0,
        cooling_schedule=schedule,
        cooling_rate=0.98
    )
    
    sa_best = sa_optimizer.solve()
    print(f"{schedule.capitalize()} cooling: {sa_optimizer.best_score:.6f} "
          f"(final temp: {sa_optimizer.temperature:.4f})")
```

## Integration with Hyperactive Framework

### Custom Optimizer Factory

```python
class CustomOptimizerFactory:
    """Factory for creating custom optimizers"""
    
    @staticmethod
    def create_optimizer(algorithm_name, experiment, **kwargs):
        """Create custom optimizer by name"""
        optimizers = {
            'custom_ga': CustomGeneticAlgorithm,
            'differential_evolution': DifferentialEvolution,
            'custom_pso': CustomParticleSwarmOptimization,
            'ant_colony': ContinuousAntColonyOptimization,
            'memetic': MemeticAlgorithm,
            'adaptive_ga': AdaptiveGeneticAlgorithm,
            'simulated_annealing': CustomSimulatedAnnealing
        }
        
        if algorithm_name not in optimizers:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        return optimizers[algorithm_name](experiment, **kwargs)
    
    @staticmethod
    def get_available_algorithms():
        """Get list of available custom algorithms"""
        return [
            'custom_ga', 'differential_evolution', 'custom_pso',
            'ant_colony', 'memetic', 'adaptive_ga', 'simulated_annealing'
        ]

# Example usage of factory
print("\nCustom Optimizer Factory Demo:")
available_algorithms = CustomOptimizerFactory.get_available_algorithms()

# Test multiple algorithms
results = {}
for alg_name in available_algorithms[:3]:  # Test first 3 for demo
    try:
        optimizer = CustomOptimizerFactory.create_optimizer(
            alg_name, test_experiment, max_iterations=30
        )
        best_params = optimizer.solve()
        results[alg_name] = optimizer.best_score
        print(f"{alg_name}: {optimizer.best_score:.6f}")
    except Exception as e:
        print(f"{alg_name}: Error - {e}")

# Find best performing algorithm
if results:
    best_algorithm = max(results.items(), key=lambda x: x[1])
    print(f"\nBest performing custom algorithm: {best_algorithm[0]} "
          f"with score {best_algorithm[1]:.6f}")
```

## Best Practices for Custom Algorithms

### Algorithm Validation and Testing

```python
class AlgorithmValidator:
    """Validation toolkit for custom algorithms"""
    
    def __init__(self, algorithm_class, test_functions):
        self.algorithm_class = algorithm_class
        self.test_functions = test_functions
    
    def validate_algorithm(self, **algorithm_params):
        """Validate algorithm on multiple test functions"""
        results = {}
        
        for func_name, test_func in self.test_functions.items():
            print(f"Testing on {func_name}...")
            
            # Multiple runs for statistical significance
            scores = []
            for run in range(5):
                optimizer = self.algorithm_class(
                    experiment=test_func,
                    **algorithm_params
                )
                
                best_params = optimizer.solve()
                score = test_func.score(best_params)[0]
                scores.append(score)
            
            results[func_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'best_score': max(scores),
                'worst_score': min(scores)
            }
        
        return results
    
    def benchmark_against_baseline(self, baseline_class, **params):
        """Compare custom algorithm against baseline"""
        custom_results = self.validate_algorithm(**params)
        
        # Test baseline
        baseline_results = {}
        for func_name, test_func in self.test_functions.items():
            baseline_optimizer = baseline_class(experiment=test_func, **params)
            baseline_best = baseline_optimizer.solve()
            baseline_score = test_func.score(baseline_best)[0]
            baseline_results[func_name] = baseline_score
        
        # Compare results
        comparison = {}
        for func_name in custom_results.keys():
            custom_mean = custom_results[func_name]['mean_score']
            baseline_score = baseline_results[func_name]
            improvement = custom_mean - baseline_score
            comparison[func_name] = {
                'custom_mean': custom_mean,
                'baseline_score': baseline_score,
                'improvement': improvement,
                'improvement_percent': (improvement / abs(baseline_score)) * 100 if baseline_score != 0 else 0
            }
        
        return comparison

# Create test functions
class SphereFunction(BaseExperiment):
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        return -(params["x"]**2 + params["y"]**2), {}

class RosenbrockFunction(BaseExperiment):
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        return -((1-x)**2 + 100*(y-x**2)**2), {}

test_functions = {
    'sphere': SphereFunction(),
    'rosenbrock': RosenbrockFunction()
}

# Validate custom GA
validator = AlgorithmValidator(CustomGeneticAlgorithm, test_functions)
validation_results = validator.validate_algorithm(
    max_iterations=50,
    population_size=20,
    mutation_rate=0.1
)

print("\nAlgorithm Validation Results:")
for func_name, results in validation_results.items():
    print(f"{func_name.capitalize()}:")
    print(f"  Mean score: {results['mean_score']:.6f} ± {results['std_score']:.6f}")
    print(f"  Best score: {results['best_score']:.6f}")
```

## Performance Optimization Tips

### Efficient Implementation Guidelines

```python
class OptimizedCustomOptimizer(CustomOptimizer):
    """Example of performance-optimized custom optimizer"""
    
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        
        # Pre-allocate arrays for efficiency
        self.param_names = self.experiment.paramnames()
        self.n_params = len(self.param_names)
        
        # Use numpy arrays instead of dictionaries for internal operations
        self.population_array = None
        self.fitness_array = None
        
        # Caching
        self.evaluation_cache = {}
        self.cache_enabled = kwargs.get('cache_enabled', True)
    
    def _dict_to_array(self, param_dict):
        """Convert parameter dictionary to numpy array"""
        return np.array([param_dict[name] for name in self.param_names])
    
    def _array_to_dict(self, param_array):
        """Convert numpy array to parameter dictionary"""
        return {name: param_array[i] for i, name in enumerate(self.param_names)}
    
    def _evaluate_with_cache(self, params_dict):
        """Evaluate with caching support"""
        if not self.cache_enabled:
            return self.experiment.score(params_dict)
        
        # Create cache key
        cache_key = tuple(sorted(params_dict.items()))
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Evaluate and cache
        result = self.experiment.score(params_dict)
        self.evaluation_cache[cache_key] = result
        return result
    
    def _vectorized_operations(self, population_array):
        """Example of vectorized operations on population"""
        # Vectorized perturbation
        noise = np.random.normal(0, 0.1, population_array.shape)
        perturbed_population = population_array + noise
        
        # Vectorized bounds checking
        perturbed_population = np.clip(perturbed_population, -20, 20)
        
        return perturbed_population

print("\nPerformance Optimization Guidelines:")
print("1. Use numpy arrays for numerical operations")
print("2. Implement caching for expensive evaluations") 
print("3. Vectorize operations when possible")
print("4. Pre-allocate memory for large data structures")
print("5. Profile your code to identify bottlenecks")
print("6. Consider parallel evaluation of candidates")
```

## Summary

Creating custom algorithms in Hyperactive provides flexibility to address domain-specific optimization challenges. Key considerations include:

1. **Inherit from BaseOptimizer** for consistency with the framework
2. **Implement required methods** (`solve`, `_generate_candidates`, etc.)
3. **Choose appropriate algorithm components** (selection, variation, replacement)
4. **Add adaptive mechanisms** for improved performance
5. **Validate thoroughly** on benchmark functions
6. **Optimize for performance** when dealing with expensive evaluations
7. **Document algorithm parameters** and their effects

Custom algorithms are particularly valuable when:
- Standard algorithms don't work well for your problem structure
- You have domain knowledge that can guide the search
- You need specialized operators or representations
- You want to combine multiple optimization strategies

The examples provided serve as templates that can be adapted and extended for specific optimization challenges.

## References

- Evolutionary Algorithms: Theory and Practice
- Swarm Intelligence algorithms and implementations
- Optimization algorithm design principles
- Performance optimization techniques for metaheuristics