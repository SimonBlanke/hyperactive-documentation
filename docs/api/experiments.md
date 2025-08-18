# Experiments and Integrations

Hyperactive v5 introduces an experiment-based approach where optimization problems are defined as experiment objects. This provides better modularity and makes it easier to create reusable optimization setups.

## Experiment Types

### Built-in Benchmark Experiments

These experiments implement standard optimization test functions commonly used for algorithm benchmarking.

#### Ackley Function

The Ackley function is a widely used multimodal test function.

```python
from hyperactive.experiment.bench import Ackley

experiment = Ackley(
    dimensions=2,           # Problem dimensionality
    bounds=(-5, 5)         # Search bounds for each dimension
)

# Use with any optimizer
from hyperactive.opt.gfo import BayesianOptimizer
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

**Properties:**
- **Global minimum**: f(0, 0, ..., 0) = 0
- **Multimodal**: Many local minima
- **Separable**: No
- **Differentiable**: Yes

#### Sphere Function

Simple quadratic function for testing convergence.

```python
from hyperactive.experiment.bench import Sphere

experiment = Sphere(
    dimensions=3,
    bounds=(-10, 10)
)
```

**Properties:**
- **Global minimum**: f(0, 0, ..., 0) = 0  
- **Unimodal**: Single optimum
- **Separable**: Yes
- **Differentiable**: Yes

#### Parabola Function

Basic parabolic test function.

```python
from hyperactive.experiment.bench import Parabola

experiment = Parabola(
    dimensions=2,
    bounds=(-5, 5)
)
```

### Framework Integration Experiments  

These experiments integrate optimization with machine learning frameworks.

## Scikit-learn Integration

### SklearnCvExperiment

Cross-validation based hyperparameter optimization for scikit-learn estimators.

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load data
X, y = load_wine(return_X_y=True)

# Define experiment
experiment = SklearnCvExperiment(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    X=X,                    # Training features
    y=y,                    # Training targets
    cv=5,                   # Cross-validation folds
    scoring="accuracy",     # Scoring metric
    n_jobs=1               # Parallel jobs for CV
)

# Use with any optimizer
from hyperactive.opt.gfo import BayesianOptimizer
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()

print("Best parameters:", best_params)
print("Best CV score:", experiment.score(best_params)[0])
```

**Parameters:**
- **estimator**: Scikit-learn estimator to optimize
- **param_grid**: Dictionary defining the search space
- **X, y**: Training data
- **cv**: Cross-validation strategy (int or CV object)
- **scoring**: Scoring function or string
- **n_jobs**: Number of parallel jobs for cross-validation

### OptCV - Scikit-learn Compatible Optimizer

For users who prefer the familiar scikit-learn interface, `OptCV` provides a drop-in replacement for `GridSearchCV` and `RandomizedSearchCV`.

```python
from hyperactive.integrations.sklearn import OptCV
from hyperactive.opt.gfo import BayesianOptimizer
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Define parameter space
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
    "kernel": ["rbf", "linear", "poly"]
}

# Create experiment for the optimizer
from hyperactive.experiment.integrations import SklearnCvExperiment
experiment = SklearnCvExperiment(
    estimator=SVC(),
    param_grid=param_grid,
    X=X, y=y,
    cv=3
)

# Create OptCV with Bayesian optimization
opt_cv = OptCV(
    estimator=SVC(),
    optimizer=BayesianOptimizer(experiment=experiment),
    cv=3,
    refit=True
)

# Use like sklearn GridSearchCV
opt_cv.fit(X, y)

print("Best parameters:", opt_cv.best_params_)
print("Best score:", opt_cv.best_score_)

# Make predictions
y_pred = opt_cv.predict(X)
```

**Key Features:**
- **Familiar Interface**: Drop-in replacement for GridSearchCV
- **Any Optimizer**: Use any Hyperactive optimizer
- **Refit Support**: Automatically refit best estimator
- **Full sklearn Compatibility**: Supports all sklearn estimator methods

## Time Series Integration

### SktimeForecastingExperiment

Specialized experiment for time series forecasting with sktime.

```python
from hyperactive.experiment.integrations import SktimeForecastingExperiment
from sktime.forecasting.arima import ARIMA
from sktime.datasets import load_airline

# Load time series data
y = load_airline()

# Define experiment
experiment = SktimeForecastingExperiment(
    forecaster=ARIMA(),
    param_grid={
        "order": [(1,1,1), (2,1,1), (1,1,2), (2,1,2)],
        "seasonal_order": [(1,1,1,12), (0,1,1,12)]
    },
    y=y,                    # Time series data
    cv=5,                   # Time series cross-validation
    scoring="mean_squared_error"
)

# Optimize with any algorithm
from hyperactive.opt.gfo import RandomSearch
optimizer = RandomSearch(experiment=experiment)
best_params = optimizer.solve()
```

## Creating Custom Experiments

You can create custom experiments by inheriting from `BaseExperiment`.

### Basic Custom Experiment

```python
from hyperactive.base import BaseExperiment
import numpy as np

class RosenbrockExperiment(BaseExperiment):
    """Rosenbrock function optimization experiment."""
    
    def __init__(self, dimensions=2, bounds=(-5, 5)):
        super().__init__()
        self.dimensions = dimensions
        self.bounds = bounds
    
    def _paramnames(self):
        """Define parameter names."""
        return [f"x{i}" for i in range(self.dimensions)]
    
    def _evaluate(self, params):
        """Implement Rosenbrock function."""
        x = np.array([params[f"x{i}"] for i in range(self.dimensions)])
        
        # Rosenbrock function: sum of (100*(x[i+1] - x[i]^2)^2 + (1-x[i])^2)
        result = 0
        for i in range(len(x)-1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        
        # Return negative for minimization (Hyperactive maximizes)
        return -result, {"dimensions": self.dimensions}

# Use custom experiment
experiment = RosenbrockExperiment(dimensions=3)
from hyperactive.opt.gfo import BayesianOptimizer
optimizer = BayesianOptimizer(experiment=experiment)
best_params = optimizer.solve()
```

### Machine Learning Custom Experiment

```python
from hyperactive.base import BaseExperiment
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

class NeuralNetworkExperiment(BaseExperiment):
    """Custom neural network hyperparameter optimization."""
    
    def __init__(self, X, y, cv=5):
        super().__init__()
        self.X = X
        self.y = y
        self.cv = cv
    
    def _paramnames(self):
        return ["hidden_layer_size", "learning_rate", "alpha"]
    
    def _evaluate(self, params):
        """Evaluate neural network with given parameters."""
        # Create model with parameters
        model = MLPClassifier(
            hidden_layer_sizes=(int(params["hidden_layer_size"]),),
            learning_rate_init=params["learning_rate"],
            alpha=params["alpha"],
            random_state=42,
            max_iter=200
        )
        
        # Cross-validation score
        scores = cross_val_score(model, self.X, self.y, cv=self.cv)
        mean_score = scores.mean()
        
        return mean_score, {
            "std_score": scores.std(),
            "individual_scores": scores.tolist()
        }

# Usage
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

experiment = NeuralNetworkExperiment(X, y, cv=3)

# Define search space bounds (handled by experiment)
# You might need to implement bounds checking in _evaluate
optimizer = SomeOptimizer(experiment=experiment)  
best_params = optimizer.solve()
```

### Experiment with External Dependencies

```python
from hyperactive.base import BaseExperiment
import subprocess
import tempfile
import os

class ExternalToolExperiment(BaseExperiment):
    """Experiment that optimizes external tool parameters."""
    
    _tags = {
        **BaseExperiment._tags,
        "python_dependencies": ["subprocess"],
        "property:randomness": "random"  # External tool might be stochastic
    }
    
    def __init__(self, input_file, reference_output):
        super().__init__()
        self.input_file = input_file
        self.reference_output = reference_output
    
    def _paramnames(self):
        return ["param1", "param2", "param3"]
    
    def _evaluate(self, params):
        """Run external tool and evaluate results."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
            f.write(f"param1={params['param1']}\n")
            f.write(f"param2={params['param2']}\n") 
            f.write(f"param3={params['param3']}\n")
            config_file = f.name
        
        try:
            # Run external tool
            result = subprocess.run([
                "external_tool",
                "--config", config_file,
                "--input", self.input_file,
                "--output", "/tmp/output.txt"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return float('-inf'), {"error": result.stderr}
            
            # Evaluate output quality
            score = self._evaluate_output("/tmp/output.txt")
            return score, {"external_tool_output": result.stdout}
            
        except subprocess.TimeoutExpired:
            return float('-inf'), {"error": "timeout"}
        finally:
            # Cleanup
            os.unlink(config_file)
    
    def _evaluate_output(self, output_file):
        """Compare output with reference."""
        # Your evaluation logic here
        pass
```

## Advanced Usage Patterns

### Multi-Objective Experiments

```python
from hyperactive.base import BaseExperiment

class MultiObjectiveExperiment(BaseExperiment):
    """Example multi-objective experiment."""
    
    _tags = {
        **BaseExperiment._tags,
        "property:higher_or_lower_is_better": "mixed"
    }
    
    def _evaluate(self, params):
        # Calculate multiple objectives
        obj1 = self._objective1(params)  # Maximize
        obj2 = -self._objective2(params)  # Minimize (negated)
        
        # Return tuple for multi-objective
        return (obj1, obj2), {"obj1_raw": obj1, "obj2_raw": -obj2}

# Use with multi-objective optimizers
from hyperactive.opt.optuna import NSGAIIOptimizer
optimizer = NSGAIIOptimizer(experiment=experiment)
```

### Experiment Composition

```python
class CompositeExperiment(BaseExperiment):
    """Combine multiple experiments."""
    
    def __init__(self, experiments, weights):
        super().__init__()
        self.experiments = experiments
        self.weights = weights
    
    def _paramnames(self):
        # Combine parameter names from all experiments
        all_params = set()
        for exp in self.experiments:
            all_params.update(exp.paramnames())
        return list(all_params)
    
    def _evaluate(self, params):
        """Weighted combination of experiment scores."""
        total_score = 0
        metadata = {}
        
        for i, exp in enumerate(self.experiments):
            # Filter params for this experiment
            exp_params = {k: v for k, v in params.items() 
                         if k in exp.paramnames()}
            
            score, meta = exp.evaluate(exp_params)
            total_score += self.weights[i] * score
            metadata[f"exp_{i}"] = {"score": score, "meta": meta}
        
        return total_score, metadata

# Usage
composite = CompositeExperiment(
    experiments=[experiment1, experiment2], 
    weights=[0.7, 0.3]
)
```

## Best Practices

### Experiment Design
1. **Clear parameter names**: Use descriptive parameter names
2. **Proper bounds checking**: Validate parameters in `_evaluate`
3. **Error handling**: Return meaningful scores for invalid parameters  
4. **Metadata logging**: Include useful information in metadata dict
5. **Reproducibility**: Set random seeds where appropriate

### Performance Optimization
1. **Caching**: Cache expensive computations when possible
2. **Early stopping**: Return early for obviously poor parameters
3. **Parallel evaluation**: Use n_jobs for cross-validation
4. **Memory management**: Clean up resources in long-running experiments

### Integration Guidelines
1. **Framework compatibility**: Follow framework conventions
2. **Type hints**: Use proper type annotations
3. **Documentation**: Document parameter spaces and expected behavior
4. **Testing**: Include unit tests for custom experiments