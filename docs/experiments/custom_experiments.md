# Custom Experiments

## Introduction

Creating custom experiments in Hyperactive allows you to optimize any objective function or complex system. By inheriting from `BaseExperiment`, you can define domain-specific optimization problems that go beyond standard machine learning hyperparameter tuning.

## BaseExperiment Overview

All experiments inherit from `BaseExperiment`, which provides:
- **Parameter space definition**: Define what parameters to optimize
- **Objective evaluation**: Implement your evaluation logic
- **Metadata handling**: Return additional information about evaluations
- **Tag system**: Specify experiment properties and requirements

## Basic Custom Experiment

```python
from hyperactive.base import BaseExperiment
import numpy as np

class SimpleCustomExperiment(BaseExperiment):
    """Optimize a custom mathematical function"""
    
    def __init__(self, dimensions=2, bounds=(-10, 10)):
        super().__init__()
        self.dimensions = dimensions
        self.bounds = bounds
    
    def _paramnames(self):
        """Define parameter names"""
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        """Evaluate the objective function"""
        # Extract parameter values
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        
        # Custom objective function (e.g., Rosenbrock function)
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        # Return negative for minimization (Hyperactive maximizes)
        return -result, {"evaluations": len(x)}

# Use the custom experiment
experiment = SimpleCustomExperiment(dimensions=3, bounds=(-2, 2))

from hyperactive.opt.gfo import BayesianOptimizer
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
print("Best parameters:", best_params)
```

## Advanced Custom Experiment

```python
import time
import json
import subprocess

class SystemOptimizationExperiment(BaseExperiment):
    """Optimize system configuration parameters"""
    
    _tags = {
        **BaseExperiment._tags,
        "python_dependencies": ["subprocess", "psutil"],
        "property:randomness": "random",  # System performance can be stochastic
        "property:higher_or_lower_is_better": "higher"
    }
    
    def __init__(self, config_template, benchmark_command):
        super().__init__()
        self.config_template = config_template
        self.benchmark_command = benchmark_command
        self.evaluation_count = 0
    
    def _paramnames(self):
        """Define system parameters to optimize"""
        return ["memory_limit", "cache_size", "thread_count", "batch_size"]
    
    def _evaluate(self, params):
        """Run system benchmark with given parameters"""
        self.evaluation_count += 1
        
        try:
            # Create configuration file
            config = self.config_template.copy()
            config.update({
                "memory_limit": int(params["memory_limit"]),
                "cache_size": int(params["cache_size"]),
                "thread_count": int(params["thread_count"]),
                "batch_size": int(params["batch_size"])
            })
            
            # Write config to temporary file
            config_file = f"/tmp/config_{self.evaluation_count}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Run benchmark
            start_time = time.time()
            result = subprocess.run(
                [self.benchmark_command, "--config", config_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            if result.returncode != 0:
                return float('-inf'), {
                    "error": result.stderr,
                    "runtime": end_time - start_time
                }
            
            # Parse performance metric from output
            performance = self._parse_performance(result.stdout)
            
            return performance, {
                "runtime": end_time - start_time,
                "config_used": config,
                "evaluation_id": self.evaluation_count
            }
            
        except subprocess.TimeoutExpired:
            return float('-inf'), {"error": "timeout", "runtime": 300}
        except Exception as e:
            return float('-inf'), {"error": str(e), "runtime": 0}
    
    def _parse_performance(self, output):
        """Extract performance metric from benchmark output"""
        # Custom parsing logic based on your benchmark output
        lines = output.strip().split('\n')
        for line in lines:
            if "Performance:" in line:
                return float(line.split()[-1])
        return 0.0

# Use system optimization experiment
config_template = {
    "algorithm": "optimization_algorithm",
    "max_iterations": 1000
}

experiment = SystemOptimizationExperiment(
    config_template=config_template,
    benchmark_command="./my_benchmark"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Machine Learning Custom Experiment

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class CustomMLExperiment(BaseExperiment):
    """Custom ML experiment with preprocessing and ensemble"""
    
    def __init__(self, X, y, cv=5):
        super().__init__()
        self.X = X
        self.y = y
        self.cv = cv
    
    def _paramnames(self):
        return [
            "n_estimators", "max_depth", "min_samples_split",
            "scale_data", "feature_selection_k"
        ]
    
    def _evaluate(self, params):
        """Custom ML pipeline evaluation"""
        try:
            # Create pipeline based on parameters
            pipeline_steps = []
            
            # Optional scaling
            if params["scale_data"] > 0.5:  # Treat as boolean
                pipeline_steps.append(('scaler', StandardScaler()))
            
            # Optional feature selection
            if params["feature_selection_k"] > 0:
                from sklearn.feature_selection import SelectKBest, f_classif
                k = min(int(params["feature_selection_k"]), self.X.shape[1])
                pipeline_steps.append(('selector', SelectKBest(f_classif, k=k)))
            
            # Main classifier
            classifier = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                min_samples_split=int(params["min_samples_split"]),
                random_state=42,
                n_jobs=1
            )
            pipeline_steps.append(('classifier', classifier))
            
            # Create and evaluate pipeline
            pipeline = Pipeline(pipeline_steps)
            scores = cross_val_score(pipeline, self.X, self.y, cv=self.cv, scoring='f1_weighted')
            
            return scores.mean(), {
                "std_score": scores.std(),
                "individual_scores": scores.tolist(),
                "pipeline_steps": len(pipeline_steps)
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Use custom ML experiment
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

experiment = CustomMLExperiment(X, y, cv=5)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Multi-Objective Custom Experiment

```python
class MultiObjectiveExperiment(BaseExperiment):
    """Experiment with multiple conflicting objectives"""
    
    _tags = {
        **BaseExperiment._tags,
        "property:higher_or_lower_is_better": "mixed"  # Multiple objectives
    }
    
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def _paramnames(self):
        return ["n_estimators", "max_depth", "min_samples_leaf"]
    
    def _evaluate(self, params):
        """Return multiple objectives: accuracy and model complexity"""
        try:
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                min_samples_leaf=int(params["min_samples_leaf"]),
                random_state=42
            )
            
            # Objective 1: Accuracy (maximize)
            scores = cross_val_score(model, self.X, self.y, cv=3, scoring='accuracy')
            accuracy = scores.mean()
            
            # Objective 2: Model simplicity (minimize complexity, so maximize simplicity)
            complexity = params["n_estimators"] * (params["max_depth"] if params["max_depth"] > 0 else 10)
            simplicity = 1.0 / (1.0 + complexity / 1000.0)  # Normalize
            
            # Return as tuple for multi-objective optimization
            return (accuracy, simplicity), {
                "accuracy": accuracy,
                "complexity": complexity,
                "simplicity": simplicity
            }
            
        except Exception as e:
            return (0.0, 0.0), {"error": str(e)}

# Use with multi-objective optimizers
from hyperactive.opt.optuna import NSGAIIOptimizer

experiment = MultiObjectiveExperiment(X, y)
optimizer = NSGAIIOptimizer(experiment=experiment, population_size=50)
best_params = optimizer.solve()
```

## Simulation-Based Experiment

```python
import random
import numpy as np

class MonteCarloSimulationExperiment(BaseExperiment):
    """Optimize parameters of a stochastic simulation"""
    
    _tags = {
        **BaseExperiment._tags,
        "property:randomness": "random",
        "python_dependencies": ["numpy", "random"]
    }
    
    def __init__(self, n_simulations=1000):
        super().__init__()
        self.n_simulations = n_simulations
    
    def _paramnames(self):
        return ["strategy_param1", "strategy_param2", "risk_threshold"]
    
    def _evaluate(self, params):
        """Run Monte Carlo simulation with given parameters"""
        results = []
        
        for _ in range(self.n_simulations):
            # Simulate some stochastic process
            outcome = self._simulate_single_run(
                params["strategy_param1"],
                params["strategy_param2"],
                params["risk_threshold"]
            )
            results.append(outcome)
        
        # Calculate performance metrics
        mean_outcome = np.mean(results)
        std_outcome = np.std(results)
        success_rate = sum(1 for r in results if r > 0) / len(results)
        
        # Multi-criteria objective (mean performance with risk adjustment)
        risk_adjusted_score = mean_outcome - 0.1 * std_outcome
        
        return risk_adjusted_score, {
            "mean_outcome": mean_outcome,
            "std_outcome": std_outcome,
            "success_rate": success_rate,
            "min_outcome": min(results),
            "max_outcome": max(results)
        }
    
    def _simulate_single_run(self, param1, param2, risk_threshold):
        """Single simulation run"""
        # Custom simulation logic
        base_outcome = param1 * random.gauss(1.0, 0.2)
        
        if random.random() < param2:
            # Strategy trigger
            outcome = base_outcome * 1.5
        else:
            outcome = base_outcome
        
        # Risk management
        if outcome < -risk_threshold:
            outcome = -risk_threshold  # Stop loss
        
        return outcome

# Use simulation experiment
experiment = MonteCarloSimulationExperiment(n_simulations=500)
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Experiment with External Data

```python
import pandas as pd
from pathlib import Path

class DataDrivenExperiment(BaseExperiment):
    """Experiment that loads and processes external data"""
    
    def __init__(self, data_path, target_column):
        super().__init__()
        self.data_path = Path(data_path)
        self.target_column = target_column
        self._data_cache = None
    
    def _load_data(self):
        """Lazy load data"""
        if self._data_cache is None:
            self._data_cache = pd.read_csv(self.data_path)
        return self._data_cache
    
    def _paramnames(self):
        return ["feature_threshold", "window_size", "smoothing_factor"]
    
    def _evaluate(self, params):
        """Process data with given parameters and return performance"""
        try:
            data = self._load_data()
            
            # Custom data processing based on parameters
            processed_data = self._process_data(
                data, 
                params["feature_threshold"],
                int(params["window_size"]),
                params["smoothing_factor"]
            )
            
            # Calculate performance metric
            performance = self._calculate_performance(processed_data)
            
            return performance, {
                "data_size": len(processed_data),
                "processing_params": params.copy()
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}
    
    def _process_data(self, data, threshold, window_size, smoothing):
        """Custom data processing logic"""
        # Example processing steps
        filtered_data = data[data['feature'] > threshold]
        
        # Rolling window processing
        if len(filtered_data) >= window_size:
            windowed = filtered_data.rolling(window=window_size).mean()
            
            # Smoothing
            smoothed = windowed * smoothing + filtered_data * (1 - smoothing)
            return smoothed.dropna()
        
        return filtered_data
    
    def _calculate_performance(self, processed_data):
        """Calculate domain-specific performance metric"""
        if len(processed_data) == 0:
            return float('-inf')
        
        # Example: optimize for some data quality metric
        target_values = processed_data[self.target_column]
        performance = -target_values.std() + target_values.mean()  # Low variance, high mean
        
        return performance

# Use data-driven experiment
experiment = DataDrivenExperiment(
    data_path="data/experiment_data.csv",
    target_column="performance_metric"
)

optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

## Best Practices for Custom Experiments

### Error Handling
```python
class RobustExperiment(BaseExperiment):
    def _evaluate(self, params):
        try:
            # Your evaluation logic
            result = self._compute_objective(params)
            return result, {"status": "success"}
        
        except ValueError as e:
            # Handle parameter validation errors
            return float('-inf'), {"error": f"Invalid parameters: {e}"}
        
        except TimeoutError as e:
            # Handle timeout errors
            return float('-inf'), {"error": "evaluation_timeout"}
        
        except Exception as e:
            # Handle unexpected errors
            return float('-inf'), {"error": f"unexpected_error: {e}"}
```

### Parameter Validation
```python
class ValidatedExperiment(BaseExperiment):
    def _validate_params(self, params):
        """Validate parameter ranges and types"""
        if params["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not isinstance(params["n_layers"], int):
            raise ValueError("n_layers must be integer")
        
        if params["dropout_rate"] < 0 or params["dropout_rate"] > 1:
            raise ValueError("dropout_rate must be between 0 and 1")
    
    def _evaluate(self, params):
        self._validate_params(params)
        # Continue with evaluation
        return self._compute_objective(params), {}
```

### Caching and Memoization
```python
import hashlib
import pickle

class CachedExperiment(BaseExperiment):
    def __init__(self, cache_dir="experiment_cache"):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, params):
        """Generate cache key from parameters"""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _evaluate(self, params):
        cache_key = self._get_cache_key(params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            return cached_result["score"], cached_result["metadata"]
        
        # Compute result
        score, metadata = self._compute_objective(params)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({"score": score, "metadata": metadata}, f)
        
        return score, metadata
```

## Testing Custom Experiments

```python
import unittest

class TestCustomExperiment(unittest.TestCase):
    def setUp(self):
        self.experiment = SimpleCustomExperiment(dimensions=2)
    
    def test_paramnames(self):
        """Test parameter names are correct"""
        expected_names = ["x0", "x1"]
        self.assertEqual(self.experiment.paramnames(), expected_names)
    
    def test_evaluate_valid_params(self):
        """Test evaluation with valid parameters"""
        params = {"x0": 1.0, "x1": 1.0}
        score, metadata = self.experiment.evaluate(params)
        
        self.assertIsInstance(score, float)
        self.assertIsInstance(metadata, dict)
        self.assertNotEqual(score, float('-inf'))  # Should not be error value
    
    def test_evaluate_invalid_params(self):
        """Test evaluation with invalid parameters"""
        params = {"wrong_param": 1.0}
        
        with self.assertRaises(ValueError):
            self.experiment.evaluate(params)

# Run tests
if __name__ == "__main__":
    unittest.main()
```

## Integration with Different Optimizers

```python
# Test custom experiment with various optimizers
from hyperactive.opt.gfo import RandomSearch, ParticleSwarmOptimizer
from hyperactive.opt.optuna import TPEOptimizer

experiment = SimpleCustomExperiment(dimensions=3)

optimizers = [
    RandomSearch(experiment=experiment),
    BayesianOptimizer(experiment=experiment),
    TPEOptimizer(experiment=experiment),
    ParticleSwarmOptimizer(experiment=experiment, population=20)
]

results = {}
for optimizer in optimizers:
    best_params = optimizer.solve()
    score = experiment.score(best_params)[0]
    results[optimizer.__class__.__name__] = (best_params, score)
    print(f"{optimizer.__class__.__name__}: {score:.6f}")

# Find best optimizer for this experiment
best_optimizer_result = max(results.items(), key=lambda x: x[1][1])
print(f"Best optimizer: {best_optimizer_result[0]}")
```

## References

- Hyperactive base classes: BaseExperiment and BaseOptimizer documentation
- Python subprocess module: [https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html)
- Scikit-learn custom scorers: [https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions)
