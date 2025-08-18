# Multi-Objective Optimization

## Introduction

Multi-objective optimization involves simultaneously optimizing multiple, often conflicting, objectives. Unlike single-objective optimization where the goal is to find one optimal solution, multi-objective optimization seeks to find a set of trade-off solutions known as the Pareto front. Hyperactive provides specialized algorithms and techniques for tackling these complex optimization problems.

## Fundamentals of Multi-Objective Optimization

### Key Concepts

- **Pareto Optimality**: A solution is Pareto optimal if no other solution can improve one objective without worsening another
- **Pareto Front**: The set of all Pareto optimal solutions
- **Dominance**: Solution A dominates solution B if A is better in all objectives or equal in some and better in at least one
- **Trade-offs**: The balance between competing objectives

### Mathematical Foundation

For a multi-objective problem with objectives $f_1, f_2, ..., f_k$:

$$\text{minimize/maximize } \mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), ..., f_k(\mathbf{x})]$$

Subject to constraints $g_i(\mathbf{x}) \leq 0$ and $h_j(\mathbf{x}) = 0$.

## Multi-Objective Algorithms in Hyperactive

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

```python
from hyperactive.base import BaseExperiment
from hyperactive.opt.optuna import NSGAIIOptimizer
import numpy as np

class MultiObjectiveMLExperiment(BaseExperiment):
    """Multi-objective machine learning optimization"""
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf"]
    
    def _evaluate(self, params):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        # Create model with parameters
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42
        )
        
        # Objective 1: Accuracy (maximize)
        accuracy_scores = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy')
        accuracy = accuracy_scores.mean()
        
        # Objective 2: Model simplicity (minimize complexity → maximize simplicity)
        complexity = params["n_estimators"] * (params["max_depth"] if params["max_depth"] > 0 else 10)
        simplicity = 1000 / (1 + complexity)  # Convert to maximization problem
        
        # Objective 3: Training speed (minimize training time → maximize speed)
        estimated_training_time = complexity / 100  # Simplified estimate
        speed = 10 / (1 + estimated_training_time)
        
        # Return tuple of objectives for multi-objective optimization
        return (accuracy, simplicity, speed), {
            "accuracy": accuracy,
            "complexity": complexity,
            "simplicity": simplicity,
            "estimated_training_time": estimated_training_time,
            "speed": speed
        }

# Load example data
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create multi-objective experiment
multi_obj_experiment = MultiObjectiveMLExperiment(X_train, y_train)

# Use NSGA-II for multi-objective optimization
nsga2_optimizer = NSGAIIOptimizer(
    experiment=multi_obj_experiment,
    population_size=50,  # Larger population for better diversity
    n_generations=20     # Number of evolutionary generations
)

# Run optimization
best_params = nsga2_optimizer.solve()
result = multi_obj_experiment.score(best_params)

print("Multi-Objective Optimization Results (NSGA-II):")
print(f"Best parameters: {best_params}")
print(f"Objectives achieved:")
print(f"  Accuracy: {result[1]['accuracy']:.4f}")
print(f"  Simplicity: {result[1]['simplicity']:.4f}")
print(f"  Speed: {result[1]['speed']:.4f}")
```

### NSGA-III for Many-Objective Problems

```python
from hyperactive.opt.optuna import NSGAIIIOptimizer

class ManyObjectiveExperiment(BaseExperiment):
    """Many-objective optimization (>3 objectives)"""
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]
    
    def _evaluate(self, params):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=min(1.0, max(0.1, params["max_features"])),
            random_state=42
        )
        
        # Multiple objectives for many-objective optimization
        
        # Objective 1: Accuracy
        accuracy = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy').mean()
        
        # Objective 2: Precision (macro average)
        precision = cross_val_score(model, self.X, self.y, cv=3, scoring='precision_macro').mean()
        
        # Objective 3: Recall (macro average)
        recall = cross_val_score(model, self.X, self.y, cv=3, scoring='recall_macro').mean()
        
        # Objective 4: F1 Score
        f1 = cross_val_score(model, self.X, self.y, cv=3, scoring='f1_macro').mean()
        
        # Objective 5: Model simplicity
        complexity = (params["n_estimators"] * 
                     (params["max_depth"] if params["max_depth"] > 0 else 10) * 
                     params["max_features"])
        simplicity = 1000 / (1 + complexity)
        
        # Objective 6: Robustness (inverse of standard deviation)
        accuracy_std = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy').std()
        robustness = 1 / (1 + accuracy_std)
        
        return (accuracy, precision, recall, f1, simplicity, robustness), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "simplicity": simplicity,
            "robustness": robustness,
            "complexity": complexity
        }

# Many-objective optimization with NSGA-III
many_obj_experiment = ManyObjectiveExperiment(X_train, y_train)

nsga3_optimizer = NSGAIIIOptimizer(
    experiment=many_obj_experiment,
    population_size=100,  # Larger population for many objectives
    n_generations=30
)

many_obj_best = nsga3_optimizer.solve()
many_obj_result = many_obj_experiment.score(many_obj_best)

print("\nMany-Objective Optimization Results (NSGA-III):")
print(f"Best parameters: {many_obj_best}")
print("Objectives achieved:")
for metric in ["accuracy", "precision", "recall", "f1", "simplicity", "robustness"]:
    print(f"  {metric.capitalize()}: {many_obj_result[1][metric]:.4f}")
```

## Scalarization Approaches

### Weighted Sum Method

```python
class WeightedSumExperiment(BaseExperiment):
    """Convert multi-objective to single-objective using weighted sum"""
    
    def __init__(self, X, y, weights=None):
        super().__init__()
        self.X = X
        self.y = y
        self.weights = weights or [0.5, 0.3, 0.2]  # Default weights
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf"]
    
    def _evaluate(self, params):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42
        )
        
        # Compute individual objectives
        accuracy = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy').mean()
        
        complexity = params["n_estimators"] * (params["max_depth"] if params["max_depth"] > 0 else 10)
        simplicity = 1 / (1 + complexity / 1000)  # Normalize
        
        training_time_estimate = complexity / 100
        speed = 1 / (1 + training_time_estimate)  # Normalize
        
        # Weighted sum (all objectives should be maximized)
        weighted_score = (self.weights[0] * accuracy + 
                         self.weights[1] * simplicity + 
                         self.weights[2] * speed)
        
        return weighted_score, {
            "weighted_score": weighted_score,
            "accuracy": accuracy,
            "simplicity": simplicity,
            "speed": speed,
            "individual_contributions": [
                self.weights[0] * accuracy,
                self.weights[1] * simplicity,
                self.weights[2] * speed
            ]
        }

# Test different weight combinations
weight_combinations = [
    [0.7, 0.2, 0.1],  # Emphasize accuracy
    [0.3, 0.5, 0.2],  # Emphasize simplicity
    [0.3, 0.2, 0.5],  # Emphasize speed
    [0.33, 0.33, 0.34]  # Equal weights
]

from hyperactive.opt.gfo import BayesianOptimizer

print("\nWeighted Sum Approach:")
print("=" * 40)

weighted_results = {}
for i, weights in enumerate(weight_combinations):
    print(f"\nWeight combination {i+1}: {weights}")
    
    weighted_experiment = WeightedSumExperiment(X_train, y_train, weights)
    weighted_optimizer = BayesianOptimizer(experiment=weighted_experiment)
    
    best_params = weighted_optimizer.solve()
    result = weighted_experiment.score(best_params)
    
    weighted_results[f"combination_{i+1}"] = {
        "weights": weights,
        "params": best_params,
        "weighted_score": result[1]["weighted_score"],
        "objectives": {
            "accuracy": result[1]["accuracy"],
            "simplicity": result[1]["simplicity"],
            "speed": result[1]["speed"]
        }
    }
    
    print(f"  Weighted score: {result[1]['weighted_score']:.4f}")
    print(f"  Accuracy: {result[1]['accuracy']:.4f}")
    print(f"  Simplicity: {result[1]['simplicity']:.4f}")
    print(f"  Speed: {result[1]['speed']:.4f}")

# Analyze trade-offs
print("\nTrade-off Analysis:")
for name, result in weighted_results.items():
    obj = result["objectives"]
    print(f"{name}: Acc={obj['accuracy']:.3f}, Simp={obj['simplicity']:.3f}, Speed={obj['speed']:.3f}")
```

### Constraint-Based Method

```python
class ConstraintBasedMultiObjective(BaseExperiment):
    """Multi-objective using constraint method (optimize one, constrain others)"""
    
    def __init__(self, X, y, constraints=None):
        super().__init__()
        self.X = X
        self.y = y
        # Default constraints: minimum acceptable levels for secondary objectives
        self.constraints = constraints or {"min_simplicity": 0.3, "min_speed": 0.4}
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf"]
    
    def _evaluate(self, params):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42
        )
        
        # Primary objective: Accuracy (to maximize)
        accuracy = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy').mean()
        
        # Secondary objectives (constraints)
        complexity = params["n_estimators"] * (params["max_depth"] if params["max_depth"] > 0 else 10)
        simplicity = 1 / (1 + complexity / 1000)
        
        training_time_estimate = complexity / 100
        speed = 1 / (1 + training_time_estimate)
        
        # Check constraints
        constraint_violations = []
        if simplicity < self.constraints["min_simplicity"]:
            constraint_violations.append(f"simplicity_violation: {simplicity:.3f} < {self.constraints['min_simplicity']}")
        
        if speed < self.constraints["min_speed"]:
            constraint_violations.append(f"speed_violation: {speed:.3f} < {self.constraints['min_speed']}")
        
        # Apply penalty for constraint violations
        if constraint_violations:
            penalty = len(constraint_violations) * 0.5
            penalized_accuracy = accuracy - penalty
            feasible = False
        else:
            penalized_accuracy = accuracy
            feasible = True
        
        return penalized_accuracy, {
            "accuracy": accuracy,
            "penalized_accuracy": penalized_accuracy,
            "simplicity": simplicity,
            "speed": speed,
            "feasible": feasible,
            "constraint_violations": constraint_violations
        }

# Test constraint-based approach
constraint_experiment = ConstraintBasedMultiObjective(
    X_train, y_train, 
    constraints={"min_simplicity": 0.4, "min_speed": 0.3}
)

constraint_optimizer = BayesianOptimizer(experiment=constraint_experiment)
constraint_best = constraint_optimizer.solve()
constraint_result = constraint_experiment.score(constraint_best)

print("\nConstraint-Based Multi-Objective:")
print(f"Best parameters: {constraint_best}")
print(f"Primary objective (accuracy): {constraint_result[1]['accuracy']:.4f}")
print(f"Simplicity: {constraint_result[1]['simplicity']:.4f}")
print(f"Speed: {constraint_result[1]['speed']:.4f}")
print(f"Feasible: {constraint_result[1]['feasible']}")
if not constraint_result[1]['feasible']:
    print(f"Violations: {constraint_result[1]['constraint_violations']}")
```

## Pareto Front Analysis

### Pareto Front Approximation

```python
class ParetoFrontAnalysis:
    """Tools for analyzing and visualizing Pareto fronts"""
    
    def __init__(self):
        self.solutions = []
    
    def add_solution(self, objectives, parameters):
        """Add a solution to the analysis"""
        self.solutions.append({
            "objectives": objectives,
            "parameters": parameters
        })
    
    def find_pareto_front(self):
        """Find Pareto optimal solutions"""
        pareto_front = []
        
        for i, solution_i in enumerate(self.solutions):
            is_dominated = False
            
            for j, solution_j in enumerate(self.solutions):
                if i != j and self._dominates(solution_j["objectives"], solution_i["objectives"]):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution_i)
        
        return pareto_front
    
    def _dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (assuming maximization)"""
        better_in_all = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
        better_in_at_least_one = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
        return better_in_all and better_in_at_least_one
    
    def calculate_hypervolume(self, reference_point):
        """Calculate hypervolume indicator"""
        pareto_front = self.find_pareto_front()
        
        # Simplified 2D hypervolume calculation
        if len(pareto_front[0]["objectives"]) == 2:
            # Sort by first objective
            pareto_front.sort(key=lambda x: x["objectives"][0], reverse=True)
            
            hypervolume = 0
            prev_obj1 = reference_point[0]
            
            for solution in pareto_front:
                obj1, obj2 = solution["objectives"]
                width = prev_obj1 - obj1
                height = obj2 - reference_point[1]
                hypervolume += width * height
                prev_obj1 = obj1
            
            return max(0, hypervolume)
        else:
            # For higher dimensions, return placeholder
            return len(pareto_front)
    
    def get_solution_diversity(self):
        """Calculate diversity of solutions"""
        if len(self.solutions) < 2:
            return 0
        
        distances = []
        for i in range(len(self.solutions)):
            for j in range(i + 1, len(self.solutions)):
                obj1 = np.array(self.solutions[i]["objectives"])
                obj2 = np.array(self.solutions[j]["objectives"])
                distance = np.linalg.norm(obj1 - obj2)
                distances.append(distance)
        
        return np.mean(distances)

# Generate multiple solutions for Pareto analysis
pareto_analyzer = ParetoFrontAnalysis()

# Run optimization multiple times with different random seeds
for seed in range(10):
    np.random.seed(seed)
    
    # Create experiment
    experiment = MultiObjectiveMLExperiment(X_train, y_train)
    
    # Run optimization with different approach each time
    optimizers = [
        BayesianOptimizer(experiment=experiment),
        NSGAIIOptimizer(experiment=experiment, population_size=20, n_generations=5)
    ]
    
    optimizer = optimizers[seed % len(optimizers)]
    best_params = optimizer.solve()
    objectives, metadata = experiment.score(best_params)
    
    pareto_analyzer.add_solution(objectives, best_params)

# Analyze Pareto front
pareto_front = pareto_analyzer.find_pareto_front()
print(f"\nPareto Front Analysis:")
print(f"Total solutions evaluated: {len(pareto_analyzer.solutions)}")
print(f"Pareto optimal solutions: {len(pareto_front)}")

# Display Pareto optimal solutions
print("\nPareto Optimal Solutions:")
for i, solution in enumerate(pareto_front):
    obj = solution["objectives"]
    print(f"Solution {i+1}: Accuracy={obj[0]:.4f}, Simplicity={obj[1]:.4f}, Speed={obj[2]:.4f}")

# Calculate metrics
if len(pareto_front) >= 2:
    # For 2D analysis, use first two objectives
    reference_point = [0.5, 0.1]  # Lower bounds for objectives
    simplified_objectives = [(sol["objectives"][0], sol["objectives"][1]) for sol in pareto_front]
    
    # Create temporary analyzer for 2D analysis
    temp_analyzer = ParetoFrontAnalysis()
    for obj, sol in zip(simplified_objectives, pareto_front):
        temp_analyzer.add_solution(obj, sol["parameters"])
    
    hypervolume = temp_analyzer.calculate_hypervolume(reference_point)
    diversity = pareto_analyzer.get_solution_diversity()
    
    print(f"\nPareto Front Metrics:")
    print(f"Hypervolume (2D approximation): {hypervolume:.4f}")
    print(f"Solution diversity: {diversity:.4f}")
```

## Decision Making with Multiple Objectives

### TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

```python
class TOPSISDecisionMaking:
    """TOPSIS method for multi-criteria decision making"""
    
    def __init__(self, alternatives, criteria_weights, beneficial_criteria):
        """
        alternatives: list of solutions with their objective values
        criteria_weights: weights for each criterion
        beneficial_criteria: list of booleans indicating if criterion is beneficial (True) or cost (False)
        """
        self.alternatives = alternatives
        self.weights = np.array(criteria_weights)
        self.beneficial = beneficial_criteria
        
    def calculate_topsis_scores(self):
        """Calculate TOPSIS scores for all alternatives"""
        # Create decision matrix
        decision_matrix = np.array([alt["objectives"] for alt in self.alternatives])
        
        # Normalize decision matrix
        normalized_matrix = self._normalize_matrix(decision_matrix)
        
        # Apply weights
        weighted_matrix = normalized_matrix * self.weights
        
        # Determine ideal and anti-ideal solutions
        ideal_solution = self._get_ideal_solution(weighted_matrix)
        anti_ideal_solution = self._get_anti_ideal_solution(weighted_matrix)
        
        # Calculate distances and TOPSIS scores
        scores = []
        for i, row in enumerate(weighted_matrix):
            distance_to_ideal = np.linalg.norm(row - ideal_solution)
            distance_to_anti_ideal = np.linalg.norm(row - anti_ideal_solution)
            
            if distance_to_ideal + distance_to_anti_ideal == 0:
                topsis_score = 0
            else:
                topsis_score = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)
            
            scores.append({
                "alternative_index": i,
                "topsis_score": topsis_score,
                "distance_to_ideal": distance_to_ideal,
                "distance_to_anti_ideal": distance_to_anti_ideal,
                "parameters": self.alternatives[i]["parameters"],
                "objectives": self.alternatives[i]["objectives"]
            })
        
        # Sort by TOPSIS score (higher is better)
        scores.sort(key=lambda x: x["topsis_score"], reverse=True)
        return scores
    
    def _normalize_matrix(self, matrix):
        """Normalize decision matrix using vector normalization"""
        column_sums = np.sqrt(np.sum(matrix**2, axis=0))
        return matrix / column_sums
    
    def _get_ideal_solution(self, weighted_matrix):
        """Get ideal solution (best value for each criterion)"""
        ideal = []
        for j in range(weighted_matrix.shape[1]):
            if self.beneficial[j]:
                ideal.append(np.max(weighted_matrix[:, j]))  # Max for beneficial
            else:
                ideal.append(np.min(weighted_matrix[:, j]))  # Min for cost
        return np.array(ideal)
    
    def _get_anti_ideal_solution(self, weighted_matrix):
        """Get anti-ideal solution (worst value for each criterion)"""
        anti_ideal = []
        for j in range(weighted_matrix.shape[1]):
            if self.beneficial[j]:
                anti_ideal.append(np.min(weighted_matrix[:, j]))  # Min for beneficial
            else:
                anti_ideal.append(np.max(weighted_matrix[:, j]))  # Max for cost
        return np.array(anti_ideal)

# Apply TOPSIS to Pareto front solutions
if len(pareto_front) > 1:
    # Define criteria weights and types
    criteria_weights = [0.5, 0.3, 0.2]  # Accuracy, Simplicity, Speed
    beneficial_criteria = [True, True, True]  # All are beneficial (higher is better)
    
    # Apply TOPSIS
    topsis = TOPSISDecisionMaking(pareto_front, criteria_weights, beneficial_criteria)
    topsis_scores = topsis.calculate_topsis_scores()
    
    print("\nTOPSIS Decision Making Results:")
    print("Ranked solutions (best to worst):")
    
    for i, score_info in enumerate(topsis_scores[:5]):  # Show top 5
        obj = score_info["objectives"]
        print(f"Rank {i+1}: TOPSIS Score = {score_info['topsis_score']:.4f}")
        print(f"  Objectives: Acc={obj[0]:.4f}, Simp={obj[1]:.4f}, Speed={obj[2]:.4f}")
        print(f"  Parameters: {score_info['parameters']}")
        print()
```

## Engineering Applications

### Structural Design Optimization

```python
class StructuralDesignMultiObjective(BaseExperiment):
    """Multi-objective structural design optimization"""
    
    def __init__(self, load=1000, material_cost_per_kg=10):
        super().__init__()
        self.load = load
        self.material_cost = material_cost_per_kg
        
    def _paramnames(self):
        return ["width", "height", "thickness"]
    
    def _evaluate(self, params):
        width = max(0.01, params["width"])  # Minimum 1cm
        height = max(0.01, params["height"])
        thickness = max(0.001, min(width/2, height/2, params["thickness"]))
        
        # Calculate structural properties
        outer_area = width * height
        inner_area = (width - 2*thickness) * (height - 2*thickness)
        cross_sectional_area = outer_area - inner_area
        
        # Mass (minimize → maximize negative)
        density = 7850  # Steel density kg/m³
        length = 3.0    # Beam length
        mass = cross_sectional_area * length * density
        
        # Cost (minimize → maximize negative)
        cost = mass * self.material_cost
        
        # Structural performance (maximize)
        moment_of_inertia = ((width * height**3) - 
                           ((width - 2*thickness) * (height - 2*thickness)**3)) / 12
        
        if moment_of_inertia <= 0:
            return (float('-inf'), float('-inf'), float('-inf')), {"invalid": True}
        
        # Maximum stress under load
        max_moment = self.load * length**2 / 8  # Simply supported beam
        max_stress = (max_moment * height/2) / moment_of_inertia
        
        # Safety factor (maximize)
        yield_strength = 250e6  # Steel yield strength
        safety_factor = yield_strength / max_stress if max_stress > 0 else float('inf')
        safety_factor = min(safety_factor, 10)  # Cap for numerical stability
        
        # Objectives (all converted to maximization)
        obj1_mass = -mass  # Minimize mass
        obj2_cost = -cost  # Minimize cost
        obj3_safety = safety_factor  # Maximize safety
        
        return (obj1_mass, obj2_cost, obj3_safety), {
            "mass_kg": mass,
            "cost_currency": cost,
            "safety_factor": safety_factor,
            "max_stress_pa": max_stress,
            "moment_of_inertia": moment_of_inertia,
            "valid": max_stress < yield_strength
        }

# Structural optimization
structural_experiment = StructuralDesignMultiObjective(load=5000, material_cost_per_kg=15)

# Use NSGA-II for structural optimization
structural_optimizer = NSGAIIOptimizer(
    experiment=structural_experiment,
    population_size=50,
    n_generations=25
)

structural_best = structural_optimizer.solve()
structural_result = structural_experiment.score(structural_best)

print("\nStructural Design Multi-Objective Optimization:")
print(f"Best parameters: {structural_best}")
if structural_result[1].get("valid", False):
    print(f"Mass: {structural_result[1]['mass_kg']:.2f} kg")
    print(f"Cost: ${structural_result[1]['cost_currency']:.2f}")
    print(f"Safety Factor: {structural_result[1]['safety_factor']:.2f}")
    print(f"Max Stress: {structural_result[1]['max_stress_pa']/1e6:.1f} MPa")
else:
    print("Invalid design - constraints violated")
```

### Portfolio Optimization

```python
class PortfolioMultiObjective(BaseExperiment):
    """Multi-objective portfolio optimization"""
    
    def __init__(self, returns_data, risk_free_rate=0.02):
        super().__init__()
        self.returns = np.array(returns_data)
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_data)
        
    def _paramnames(self):
        return [f"weight_{i}" for i in range(self.n_assets)]
    
    def _evaluate(self, params):
        # Extract and normalize weights
        weights = np.array([params[f"weight_{i}"] for i in range(self.n_assets)])
        weights = np.abs(weights)  # Ensure positive
        
        if np.sum(weights) == 0:
            return (float('-inf'), float('-inf'), float('-inf')), {"invalid": True}
        
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(self.returns.T, weights)
        
        # Objective 1: Expected return (maximize)
        expected_return = np.mean(portfolio_returns)
        
        # Objective 2: Risk (minimize → maximize negative)
        portfolio_risk = np.std(portfolio_returns)
        risk_objective = -portfolio_risk
        
        # Objective 3: Sharpe ratio (maximize)
        sharpe_ratio = ((expected_return - self.risk_free_rate) / portfolio_risk 
                       if portfolio_risk > 0 else 0)
        
        # Additional metrics
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return (expected_return, risk_objective, sharpe_ratio), {
            "expected_return": expected_return,
            "portfolio_risk": portfolio_risk,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "weights": weights.tolist(),
            "diversification_ratio": 1 / np.sum(weights**2)
        }

# Generate sample portfolio data
np.random.seed(42)
n_assets = 4
n_periods = 252
portfolio_returns = []

# Create assets with different risk-return profiles
for i in range(n_assets):
    mu = 0.05 + i * 0.03  # Increasing expected returns
    sigma = 0.15 + i * 0.05  # Increasing volatility
    asset_returns = np.random.normal(mu/252, sigma/np.sqrt(252), n_periods)
    portfolio_returns.append(asset_returns)

# Portfolio optimization
portfolio_experiment = PortfolioMultiObjective(portfolio_returns)
portfolio_optimizer = NSGAIIOptimizer(
    experiment=portfolio_experiment,
    population_size=40,
    n_generations=20
)

portfolio_best = portfolio_optimizer.solve()
portfolio_result = portfolio_experiment.score(portfolio_best)

print("\nPortfolio Multi-Objective Optimization:")
print(f"Optimal weights: {[f'{w:.3f}' for w in portfolio_result[1]['weights']]}")
print(f"Expected return: {portfolio_result[1]['expected_return']*252:.1%}")
print(f"Portfolio risk: {portfolio_result[1]['portfolio_risk']*np.sqrt(252):.1%}")
print(f"Sharpe ratio: {portfolio_result[1]['sharpe_ratio']:.3f}")
print(f"Max drawdown: {portfolio_result[1]['max_drawdown']:.1%}")
print(f"Diversification ratio: {portfolio_result[1]['diversification_ratio']:.2f}")
```

## Best Practices

### Algorithm Selection Guidelines

1. **NSGA-II**: Best for 2-3 objectives, well-established algorithm
2. **NSGA-III**: Better for many objectives (>3), uses reference points
3. **Weighted Sum**: Simple, good for initial exploration
4. **Constraint Method**: When one objective is clearly primary
5. **TOPSIS**: Excellent for final decision making among Pareto solutions

### Implementation Tips

```python
# Best practices for multi-objective optimization

def multi_objective_best_practices():
    """
    Guidelines for effective multi-objective optimization
    """
    
    guidelines = {
        "objective_formulation": [
            "Ensure objectives are conflicting (trade-offs exist)",
            "Scale objectives appropriately (similar ranges)",
            "Minimize number of objectives when possible",
            "Consider objective interactions"
        ],
        
        "algorithm_configuration": [
            "Use larger population sizes for multi-objective algorithms",
            "Allow more generations/iterations",
            "Consider multiple runs for statistical significance",
            "Tune algorithm-specific parameters"
        ],
        
        "result_analysis": [
            "Always analyze the Pareto front",
            "Use decision-making methods (TOPSIS, AHP) for final selection",
            "Validate solutions in real-world scenarios",
            "Consider robustness of solutions"
        ],
        
        "computational_considerations": [
            "Multi-objective algorithms are more expensive",
            "Parallel evaluation can help",
            "Consider early stopping criteria",
            "Archive non-dominated solutions"
        ]
    }
    
    return guidelines

best_practices = multi_objective_best_practices()
print("\nMulti-Objective Optimization Best Practices:")
for category, practices in best_practices.items():
    print(f"\n{category.replace('_', ' ').title()}:")
    for practice in practices:
        print(f"  • {practice}")
```

## References

- Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms
- Coello, C. A. C. (2006). Evolutionary multi-objective optimization: A historical view of the field
- NSGA-II and NSGA-III algorithm papers
- TOPSIS methodology for decision making
- Pareto optimality theory and applications

## Summary

Multi-objective optimization is essential for real-world problems where multiple, often conflicting, goals must be balanced. Hyperactive provides robust algorithms like NSGA-II and NSGA-III for finding Pareto optimal solutions, along with decision-making tools to select the best solution for your specific needs. The key is understanding the trade-offs inherent in your problem and choosing the appropriate algorithm and decision-making approach for your domain.