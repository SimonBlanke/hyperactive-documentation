# Extensions

## Introduction

Hyperactive's modular architecture makes it highly extensible. This page demonstrates how to extend Hyperactive with custom components, integrate with external systems, and build domain-specific optimization solutions. Whether you need custom experiments, new optimization backends, or specialized integrations, this guide provides the blueprints for extending Hyperactive's capabilities.

## Custom Experiment Extensions

### Database-Backed Experiments

```python
import sqlite3
import json
from pathlib import Path
from hyperactive.base import BaseExperiment
import numpy as np

class DatabaseExperiment(BaseExperiment):
    """Experiment that stores all evaluations in a database"""
    
    def __init__(self, db_path="optimization_results.db", table_name="evaluations"):
        super().__init__()
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._setup_database()
        
    def _setup_database(self):
        """Initialize database and create tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create evaluations table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    parameters TEXT NOT NULL,
                    score REAL NOT NULL,
                    metadata TEXT,
                    experiment_id TEXT,
                    algorithm_name TEXT
                )
            ''')
            
            # Create index for faster queries
            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_score 
                ON {self.table_name}(score)
            ''')
            
            conn.commit()
    
    def _paramnames(self):
        return ["learning_rate", "hidden_size", "dropout"]
    
    def _evaluate(self, params):
        # Simulate ML model evaluation
        lr = max(1e-6, min(1.0, params["learning_rate"]))
        hidden_size = max(8, int(params["hidden_size"]))
        dropout = max(0, min(0.95, params["dropout"]))
        
        # Performance model
        lr_penalty = -abs(np.log10(lr) + 3)**2
        size_bonus = min(hidden_size / 128, 1.0)
        dropout_bonus = dropout * (1 - dropout) * 4
        
        score = lr_penalty + size_bonus + dropout_bonus
        
        metadata = {
            "lr_used": lr,
            "hidden_size_used": hidden_size,
            "dropout_used": dropout,
            "lr_penalty": lr_penalty,
            "size_bonus": size_bonus,
            "dropout_bonus": dropout_bonus
        }
        
        # Store in database
        self._store_evaluation(params, score, metadata)
        
        return score, metadata
    
    def _store_evaluation(self, params, score, metadata, experiment_id=None, algorithm_name=None):
        """Store evaluation in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
                INSERT INTO {self.table_name} 
                (parameters, score, metadata, experiment_id, algorithm_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                json.dumps(params),
                score,
                json.dumps(metadata),
                experiment_id or "default",
                algorithm_name or "unknown"
            ))
            
            conn.commit()
    
    def get_best_evaluations(self, limit=10):
        """Retrieve best evaluations from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
                SELECT parameters, score, metadata, timestamp
                FROM {self.table_name}
                ORDER BY score DESC
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'parameters': json.loads(row[0]),
                    'score': row[1],
                    'metadata': json.loads(row[2]),
                    'timestamp': row[3]
                })
            
            return results
    
    def get_optimization_history(self):
        """Get complete optimization history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(f'''
                SELECT parameters, score, timestamp
                FROM {self.table_name}
                ORDER BY timestamp ASC
            ''')
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'parameters': json.loads(row[0]),
                    'score': row[1],
                    'timestamp': row[2]
                })
            
            return history
    
    def analyze_parameter_importance(self):
        """Analyze parameter importance from stored data"""
        history = self.get_optimization_history()
        
        if len(history) < 10:
            return "Insufficient data for analysis"
        
        # Simple correlation analysis
        param_names = self._paramnames()
        correlations = {}
        
        for param_name in param_names:
            param_values = [eval_data['parameters'][param_name] for eval_data in history]
            scores = [eval_data['score'] for eval_data in history]
            
            correlation = np.corrcoef(param_values, scores)[0, 1]
            correlations[param_name] = correlation
        
        return correlations

# Test database experiment
db_experiment = DatabaseExperiment()

from hyperactive.opt.gfo import BayesianOptimizer

print("Database-Backed Experiment:")
db_optimizer = BayesianOptimizer(experiment=db_experiment)
db_best = db_optimizer.solve()

print(f"Best parameters: {db_best}")

# Analyze results
best_evals = db_experiment.get_best_evaluations(5)
print(f"\nTop 5 evaluations:")
for i, eval_data in enumerate(best_evals):
    print(f"{i+1}. Score: {eval_data['score']:.6f}, Params: {eval_data['parameters']}")

# Parameter importance analysis
param_importance = db_experiment.analyze_parameter_importance()
print(f"\nParameter importance (correlation with score):")
for param, corr in param_importance.items():
    print(f"  {param}: {corr:.4f}")
```

### Web API Integration

```python
import requests
import time
from typing import Dict, Any

class WebAPIExperiment(BaseExperiment):
    """Experiment that evaluates parameters via web API"""
    
    def __init__(self, api_url, api_key=None, timeout=30, retry_attempts=3):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.request_count = 0
        
    def _paramnames(self):
        return ["model_config", "training_epochs", "batch_size"]
    
    def _evaluate(self, params):
        """Evaluate parameters by calling external API"""
        self.request_count += 1
        
        # Prepare API request
        payload = {
            "parameters": params,
            "request_id": f"hyperactive_{self.request_count}_{int(time.time())}",
            "timeout": self.timeout
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Make API request with retries
        for attempt in range(self.retry_attempts):
            try:
                response = self._make_api_request(payload, headers)
                
                if response["status"] == "success":
                    return response["score"], response.get("metadata", {})
                elif response["status"] == "error":
                    return float('-inf'), {"api_error": response.get("message", "Unknown error")}
                    
            except requests.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    return float('-inf'), {"network_error": str(e)}
                
                # Exponential backoff
                time.sleep(2 ** attempt)
        
        return float('-inf'), {"error": "max_retries_exceeded"}
    
    def _make_api_request(self, payload, headers):
        """Make single API request"""
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

# Mock API server for demonstration
class MockAPIServer:
    """Mock API server for testing"""
    
    @staticmethod
    def evaluate_parameters(params):
        """Mock evaluation endpoint"""
        # Simulate processing time
        time.sleep(0.1)
        
        # Mock evaluation logic
        model_config = params.get("model_config", 0.5)
        epochs = params.get("training_epochs", 10)
        batch_size = params.get("batch_size", 32)
        
        # Simulate model performance
        config_bonus = model_config * 0.8
        epoch_bonus = min(epochs / 100, 0.3)  # Diminishing returns
        batch_penalty = abs(batch_size - 64) / 1000  # Optimal around 64
        
        score = config_bonus + epoch_bonus - batch_penalty
        
        # Add some noise
        score += np.random.normal(0, 0.05)
        
        return {
            "status": "success",
            "score": score,
            "metadata": {
                "config_bonus": config_bonus,
                "epoch_bonus": epoch_bonus,
                "batch_penalty": batch_penalty,
                "processing_time": 0.1
            }
        }

# Example usage (would normally use real API)
print("\nWeb API Integration Example:")
print("(Using mock API for demonstration)")

# In real usage, you would have:
# api_experiment = WebAPIExperiment("https://your-api.com/evaluate", api_key="your_key")

# For demo, we'll simulate the API calls
class MockWebAPIExperiment(BaseExperiment):
    def _paramnames(self):
        return ["model_config", "training_epochs", "batch_size"]
    
    def _evaluate(self, params):
        result = MockAPIServer.evaluate_parameters(params)
        return result["score"], result["metadata"]

mock_api_experiment = MockWebAPIExperiment()
api_optimizer = BayesianOptimizer(experiment=mock_api_experiment)
api_best = api_optimizer.solve()

print(f"API-based optimization result: {api_best}")
```

## Custom Backend Integration

### Ray Tune Integration

```python
try:
    import ray
    from ray import tune
    from ray.tune.suggest.hyperopt import HyperOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

if RAY_AVAILABLE:
    class RayTuneBackend:
        """Custom backend using Ray Tune"""
        
        def __init__(self, experiment, search_algorithm="hyperopt", num_samples=100):
            self.experiment = experiment
            self.search_algorithm = search_algorithm
            self.num_samples = num_samples
            self.results = []
        
        def _hyperactive_to_ray_config(self):
            """Convert Hyperactive experiment to Ray Tune config"""
            param_names = self.experiment.paramnames()
            
            # Define search space (simplified - would need more sophisticated mapping)
            config = {}
            for param_name in param_names:
                config[param_name] = tune.uniform(-10, 10)  # Default range
            
            return config
        
        def _ray_objective(self, config):
            """Ray Tune objective function"""
            score, metadata = self.experiment.score(config)
            
            # Ray Tune expects to minimize, Hyperactive maximizes
            tune.report(score=-score, **metadata)
        
        def solve(self):
            """Solve using Ray Tune"""
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Configure search algorithm
            if self.search_algorithm == "hyperopt":
                search_alg = HyperOptSearch(metric="score", mode="max")
            else:
                search_alg = None
            
            # Run optimization
            analysis = tune.run(
                self._ray_objective,
                config=self._hyperactive_to_ray_config(),
                search_alg=search_alg,
                num_samples=self.num_samples,
                verbose=1
            )
            
            # Get best result
            best_trial = analysis.get_best_trial("score", "max")
            return best_trial.config
    
    print("\nRay Tune Backend Integration:")
    
    # Test with simple experiment
    class SimpleRayExperiment(BaseExperiment):
        def _paramnames(self):
            return ["x", "y"]
        
        def _evaluate(self, params):
            return -(params["x"]**2 + params["y"]**2), {}
    
    ray_experiment = SimpleRayExperiment()
    ray_backend = RayTuneBackend(ray_experiment, num_samples=20)
    
    try:
        ray_best = ray_backend.solve()
        print(f"Ray Tune result: {ray_best}")
    except Exception as e:
        print(f"Ray Tune integration demo failed: {e}")
    
    # Cleanup
    ray.shutdown()

else:
    print("\nRay Tune not available - install with: pip install ray[tune]")
```

### MLflow Integration

```python
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

if MLFLOW_AVAILABLE:
    class MLflowTrackingExperiment(BaseExperiment):
        """Experiment with MLflow tracking integration"""
        
        def __init__(self, experiment_name="hyperactive_optimization"):
            super().__init__()
            self.experiment_name = experiment_name
            self._setup_mlflow()
            
        def _setup_mlflow(self):
            """Setup MLflow experiment"""
            mlflow.set_experiment(self.experiment_name)
            
        def _paramnames(self):
            return ["n_estimators", "max_depth", "learning_rate"]
        
        def _evaluate(self, params):
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params)
                
                # Simulate model training and evaluation
                score = self._train_and_evaluate(params)
                
                # Log metrics
                mlflow.log_metric("validation_score", score)
                mlflow.log_metric("cv_score", score * 0.95)  # Simulated CV score
                
                # Log artifacts (model, plots, etc.)
                self._log_artifacts(params, score)
                
                return score, {"mlflow_run_id": mlflow.active_run().info.run_id}
        
        def _train_and_evaluate(self, params):
            """Simulate model training"""
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.datasets import make_regression
            from sklearn.model_selection import cross_val_score
            
            # Generate synthetic data
            X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
            
            # Create model with parameters
            model = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] > 0 else None,
                random_state=42
            )
            
            # Evaluate with cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            score = cv_scores.mean()
            
            # Log model
            model.fit(X, y)
            mlflow.sklearn.log_model(model, "model")
            
            return score
        
        def _log_artifacts(self, params, score):
            """Log additional artifacts"""
            import matplotlib.pyplot as plt
            import tempfile
            import os
            
            # Create a simple plot
            fig, ax = plt.subplots(figsize=(8, 6))
            param_names = list(params.keys())
            param_values = [params[name] for name in param_names]
            
            ax.bar(param_names, param_values)
            ax.set_title(f"Parameters for Score: {score:.4f}")
            ax.set_ylabel("Parameter Value")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save and log plot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name)
                mlflow.log_artifact(tmp_file.name, "plots")
                os.unlink(tmp_file.name)
            
            plt.close()
            
            # Log parameter summary
            with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as tmp_file:
                tmp_file.write(f"Optimization Results\n")
                tmp_file.write(f"Score: {score:.6f}\n")
                tmp_file.write(f"Parameters:\n")
                for name, value in params.items():
                    tmp_file.write(f"  {name}: {value}\n")
                
                tmp_file.flush()
                mlflow.log_artifact(tmp_file.name, "summaries")
                os.unlink(tmp_file.name)
    
    print("\nMLflow Integration Example:")
    
    try:
        mlflow_experiment = MLflowTrackingExperiment()
        mlflow_optimizer = BayesianOptimizer(experiment=mlflow_experiment)
        mlflow_best = mlflow_optimizer.solve()
        
        print(f"MLflow tracked optimization result: {mlflow_best}")
        print("Check MLflow UI for detailed tracking results")
        
    except Exception as e:
        print(f"MLflow integration demo failed: {e}")

else:
    print("\nMLflow not available - install with: pip install mlflow")
```

## Domain-Specific Extensions

### AutoML Pipeline Extension

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class AutoMLExperiment(BaseExperiment):
    """Automated ML pipeline optimization"""
    
    def __init__(self, X, y, cv=5):
        super().__init__()
        self.X = X
        self.y = y
        self.cv = cv
        self.pipeline_components = self._get_pipeline_components()
        
    def _paramnames(self):
        return [
            "scaler_type",
            "feature_selection",
            "feature_k_ratio", 
            "classifier_type",
            "classifier_param1",
            "classifier_param2",
            "classifier_param3"
        ]
    
    def _get_pipeline_components(self):
        """Define available pipeline components"""
        return {
            "scalers": {
                0: None,
                1: StandardScaler(),
                2: StandardScaler(with_mean=False)
            },
            "feature_selectors": {
                0: None,
                1: SelectKBest(f_classif)
            },
            "classifiers": {
                0: RandomForestClassifier(random_state=42),
                1: LogisticRegression(random_state=42, max_iter=1000),
                2: SVC(random_state=42, probability=True)
            }
        }
    
    def _evaluate(self, params):
        try:
            # Build pipeline based on parameters
            pipeline_steps = []
            
            # Scaler selection
            scaler_id = int(params["scaler_type"]) % len(self.pipeline_components["scalers"])
            scaler = self.pipeline_components["scalers"][scaler_id]
            if scaler is not None:
                pipeline_steps.append(("scaler", scaler))
            
            # Feature selection
            fs_enabled = params["feature_selection"] > 0.5
            if fs_enabled:
                k_ratio = max(0.1, min(1.0, params["feature_k_ratio"]))
                k = max(1, int(self.X.shape[1] * k_ratio))
                feature_selector = SelectKBest(f_classif, k=k)
                pipeline_steps.append(("feature_selection", feature_selector))
            
            # Classifier selection and configuration
            clf_id = int(params["classifier_type"]) % len(self.pipeline_components["classifiers"])
            
            if clf_id == 0:  # Random Forest
                classifier = RandomForestClassifier(
                    n_estimators=max(10, int(params["classifier_param1"])),
                    max_depth=int(params["classifier_param2"]) if params["classifier_param2"] > 0 else None,
                    min_samples_split=max(2, int(params["classifier_param3"])),
                    random_state=42
                )
            elif clf_id == 1:  # Logistic Regression
                classifier = LogisticRegression(
                    C=max(0.001, params["classifier_param1"]),
                    solver='liblinear' if params["classifier_param2"] > 0.5 else 'lbfgs',
                    random_state=42,
                    max_iter=1000
                )
            else:  # SVM
                classifier = SVC(
                    C=max(0.001, params["classifier_param1"]),
                    gamma='scale' if params["classifier_param2"] > 0.5 else 'auto',
                    kernel='rbf' if params["classifier_param3"] > 0.5 else 'linear',
                    random_state=42,
                    probability=True
                )
            
            pipeline_steps.append(("classifier", classifier))
            
            # Create and evaluate pipeline
            pipeline = Pipeline(pipeline_steps)
            
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(pipeline, self.X, self.y, cv=self.cv, scoring='f1_weighted')
            score = cv_scores.mean()
            
            return score, {
                "cv_std": cv_scores.std(),
                "pipeline_length": len(pipeline_steps),
                "scaler_used": scaler_id,
                "feature_selection_used": fs_enabled,
                "classifier_used": clf_id,
                "classifier_name": classifier.__class__.__name__
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}

# Test AutoML experiment
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

automl_experiment = AutoMLExperiment(X_train, y_train)

print("\nAutoML Pipeline Optimization:")
automl_optimizer = BayesianOptimizer(experiment=automl_experiment)
automl_best = automl_optimizer.solve()

result = automl_experiment.score(automl_best)
print(f"Best AutoML pipeline score: {result[0]:.4f}")
print(f"Pipeline configuration: {result[1]}")
```

### Neural Architecture Search (NAS)

```python
class NeuralArchitectureSearchExperiment(BaseExperiment):
    """Simplified Neural Architecture Search experiment"""
    
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
    def _paramnames(self):
        return [
            "num_layers",
            "layer1_size", "layer1_activation", "layer1_dropout",
            "layer2_size", "layer2_activation", "layer2_dropout", 
            "layer3_size", "layer3_activation", "layer3_dropout",
            "optimizer_type", "learning_rate", "batch_size"
        ]
    
    def _evaluate(self, params):
        try:
            # Architecture configuration
            num_layers = max(1, min(3, int(params["num_layers"])))
            
            architecture = self._build_architecture(params, num_layers)
            training_config = self._get_training_config(params)
            
            # Simulate neural network training
            score = self._simulate_training(architecture, training_config)
            
            return score, {
                "num_layers": num_layers,
                "total_params": self._count_parameters(architecture),
                "architecture": architecture,
                "training_config": training_config
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}
    
    def _build_architecture(self, params, num_layers):
        """Build neural network architecture"""
        architecture = []
        current_size = self.input_dim
        
        for i in range(num_layers):
            layer_size = max(8, int(params[f"layer{i+1}_size"]))
            activation = self._get_activation(params[f"layer{i+1}_activation"])
            dropout = max(0, min(0.9, params[f"layer{i+1}_dropout"]))
            
            architecture.append({
                "type": "dense",
                "input_size": current_size,
                "output_size": layer_size,
                "activation": activation,
                "dropout": dropout
            })
            
            current_size = layer_size
        
        # Output layer
        architecture.append({
            "type": "dense",
            "input_size": current_size,
            "output_size": self.num_classes,
            "activation": "softmax",
            "dropout": 0
        })
        
        return architecture
    
    def _get_activation(self, activation_param):
        """Map parameter to activation function"""
        activations = ["relu", "tanh", "sigmoid", "leaky_relu"]
        idx = int(activation_param * len(activations)) % len(activations)
        return activations[idx]
    
    def _get_training_config(self, params):
        """Extract training configuration"""
        optimizers = ["adam", "sgd", "rmsprop"]
        opt_idx = int(params["optimizer_type"] * len(optimizers)) % len(optimizers)
        
        return {
            "optimizer": optimizers[opt_idx],
            "learning_rate": max(1e-6, min(1e-1, params["learning_rate"])),
            "batch_size": max(8, int(params["batch_size"]))
        }
    
    def _count_parameters(self, architecture):
        """Count total parameters in architecture"""
        total_params = 0
        for layer in architecture:
            if layer["type"] == "dense":
                layer_params = layer["input_size"] * layer["output_size"] + layer["output_size"]
                total_params += layer_params
        return total_params
    
    def _simulate_training(self, architecture, training_config):
        """Simulate neural network training performance"""
        # Model complexity score
        num_layers = len([l for l in architecture if l["type"] == "dense"]) - 1  # Exclude output layer
        total_params = self._count_parameters(architecture)
        
        # Complexity penalty (very large models perform poorly)
        if total_params > 100000:
            complexity_penalty = -0.5
        elif total_params > 50000:
            complexity_penalty = -0.2
        else:
            complexity_penalty = 0
        
        # Layer configuration scoring
        layer_score = 0
        for layer in architecture[:-1]:  # Exclude output layer
            # Prefer ReLU activation
            if layer["activation"] == "relu":
                layer_score += 0.1
            
            # Moderate dropout is good
            dropout = layer["dropout"]
            if 0.1 <= dropout <= 0.5:
                layer_score += 0.05
            
            # Reasonable layer sizes
            size = layer["output_size"]
            if 32 <= size <= 512:
                layer_score += 0.05
        
        # Training configuration scoring
        training_score = 0
        
        # Learning rate (prefer around 0.001)
        lr = training_config["learning_rate"]
        lr_score = -abs(np.log10(lr) + 3)**2 / 10
        training_score += lr_score
        
        # Optimizer preference
        if training_config["optimizer"] == "adam":
            training_score += 0.1
        
        # Batch size (prefer moderate sizes)
        batch_size = training_config["batch_size"]
        if 16 <= batch_size <= 128:
            training_score += 0.05
        
        # Combine scores
        total_score = layer_score + training_score + complexity_penalty
        
        # Add some noise to simulate training variance
        total_score += np.random.normal(0, 0.02)
        
        return total_score

# Test NAS experiment
nas_experiment = NeuralArchitectureSearchExperiment()

print("\nNeural Architecture Search:")
nas_optimizer = BayesianOptimizer(experiment=nas_experiment)
nas_best = nas_optimizer.solve()

nas_result = nas_experiment.score(nas_best)
print(f"Best architecture score: {nas_result[0]:.4f}")
print(f"Architecture details:")
print(f"  Layers: {nas_result[1]['num_layers']}")
print(f"  Total parameters: {nas_result[1]['total_params']:,}")
print(f"  Training config: {nas_result[1]['training_config']}")
```

## Plugin System

### Plugin Architecture

```python
import importlib
from abc import ABC, abstractmethod

class HyperactivePlugin(ABC):
    """Base class for Hyperactive plugins"""
    
    @abstractmethod
    def get_name(self):
        """Return plugin name"""
        pass
    
    @abstractmethod
    def get_version(self):
        """Return plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, hyperactive_instance):
        """Initialize plugin with Hyperactive instance"""
        pass
    
    @abstractmethod
    def get_capabilities(self):
        """Return list of capabilities provided by plugin"""
        pass

class PluginManager:
    """Manager for Hyperactive plugins"""
    
    def __init__(self):
        self.plugins = {}
        self.loaded_plugins = set()
        
    def register_plugin(self, plugin_class):
        """Register a plugin class"""
        plugin_instance = plugin_class()
        plugin_name = plugin_instance.get_name()
        
        if plugin_name in self.plugins:
            raise ValueError(f"Plugin {plugin_name} already registered")
        
        self.plugins[plugin_name] = plugin_instance
        print(f"Registered plugin: {plugin_name} v{plugin_instance.get_version()}")
    
    def load_plugin(self, plugin_name, hyperactive_instance):
        """Load and initialize a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        if plugin_name in self.loaded_plugins:
            print(f"Plugin {plugin_name} already loaded")
            return
        
        plugin = self.plugins[plugin_name]
        plugin.initialize(hyperactive_instance)
        self.loaded_plugins.add(plugin_name)
        
        print(f"Loaded plugin: {plugin_name}")
        print(f"Capabilities: {', '.join(plugin.get_capabilities())}")
    
    def get_available_plugins(self):
        """Get list of available plugins"""
        return list(self.plugins.keys())
    
    def get_plugin_info(self, plugin_name):
        """Get detailed plugin information"""
        if plugin_name not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_name]
        return {
            "name": plugin.get_name(),
            "version": plugin.get_version(),
            "capabilities": plugin.get_capabilities(),
            "loaded": plugin_name in self.loaded_plugins
        }

# Example plugins
class VisualizationPlugin(HyperactivePlugin):
    """Plugin for optimization visualization"""
    
    def get_name(self):
        return "visualization"
    
    def get_version(self):
        return "1.0.0"
    
    def get_capabilities(self):
        return ["plot_convergence", "plot_parameter_space", "plot_correlation"]
    
    def initialize(self, hyperactive_instance):
        """Add visualization methods to Hyperactive instance"""
        import matplotlib.pyplot as plt
        
        def plot_convergence(self, history):
            """Plot optimization convergence"""
            plt.figure(figsize=(10, 6))
            plt.plot(history)
            plt.title("Optimization Convergence")
            plt.xlabel("Iteration")
            plt.ylabel("Best Score")
            plt.grid(True)
            plt.show()
        
        def plot_parameter_distribution(self, parameter_data):
            """Plot parameter distribution"""
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.ravel()
            
            for i, (param_name, values) in enumerate(parameter_data.items()):
                if i >= 4:
                    break
                axes[i].hist(values, bins=20, alpha=0.7)
                axes[i].set_title(f"Distribution of {param_name}")
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.show()
        
        # Add methods to hyperactive instance
        hyperactive_instance.plot_convergence = lambda history: plot_convergence(hyperactive_instance, history)
        hyperactive_instance.plot_parameter_distribution = lambda data: plot_parameter_distribution(hyperactive_instance, data)

class LoggingPlugin(HyperactivePlugin):
    """Plugin for enhanced logging"""
    
    def get_name(self):
        return "logging"
    
    def get_version(self):
        return "1.0.0"
    
    def get_capabilities(self):
        return ["structured_logging", "performance_metrics", "error_tracking"]
    
    def initialize(self, hyperactive_instance):
        """Add logging capabilities"""
        import logging
        import json
        
        # Setup structured logger
        logger = logging.getLogger("hyperactive_optimization")
        logger.setLevel(logging.INFO)
        
        # Add structured logging handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        def log_optimization_start(self, experiment_name, algorithm_name):
            """Log optimization start"""
            logger.info(json.dumps({
                "event": "optimization_start",
                "experiment": experiment_name,
                "algorithm": algorithm_name,
                "timestamp": time.time()
            }))
        
        def log_evaluation(self, iteration, params, score, metadata):
            """Log individual evaluation"""
            logger.info(json.dumps({
                "event": "evaluation",
                "iteration": iteration,
                "parameters": params,
                "score": score,
                "metadata": metadata,
                "timestamp": time.time()
            }))
        
        def log_optimization_complete(self, best_params, best_score, total_time):
            """Log optimization completion"""
            logger.info(json.dumps({
                "event": "optimization_complete",
                "best_parameters": best_params,
                "best_score": best_score,
                "total_time": total_time,
                "timestamp": time.time()
            }))
        
        # Add methods to hyperactive instance
        hyperactive_instance.log_optimization_start = lambda exp, alg: log_optimization_start(hyperactive_instance, exp, alg)
        hyperactive_instance.log_evaluation = lambda i, p, s, m: log_evaluation(hyperactive_instance, i, p, s, m)
        hyperactive_instance.log_optimization_complete = lambda p, s, t: log_optimization_complete(hyperactive_instance, p, s, t)

# Plugin usage example
plugin_manager = PluginManager()

# Register plugins
plugin_manager.register_plugin(VisualizationPlugin)
plugin_manager.register_plugin(LoggingPlugin)

print("\nPlugin System Example:")
print(f"Available plugins: {plugin_manager.get_available_plugins()}")

# Mock hyperactive instance for demo
class MockHyperactive:
    def __init__(self):
        pass

hyperactive_instance = MockHyperactive()

# Load plugins
plugin_manager.load_plugin("logging", hyperactive_instance)
plugin_manager.load_plugin("visualization", hyperactive_instance)

# Test plugin functionality
if hasattr(hyperactive_instance, 'log_optimization_start'):
    hyperactive_instance.log_optimization_start("test_experiment", "BayesianOptimizer")
    print("Logging plugin working!")

print("\nPlugin info:")
for plugin_name in plugin_manager.get_available_plugins():
    info = plugin_manager.get_plugin_info(plugin_name)
    print(f"  {info['name']} v{info['version']}: {', '.join(info['capabilities'])}")
```

## Integration Utilities

### Configuration Management

```python
import yaml
import json
from pathlib import Path

class HyperactiveConfigManager:
    """Configuration management for Hyperactive extensions"""
    
    def __init__(self, config_dir="hyperactive_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configs = {}
    
    def create_config_template(self, config_name, template_type="optimization"):
        """Create configuration template"""
        templates = {
            "optimization": {
                "experiment": {
                    "type": "SklearnCvExperiment",
                    "parameters": {
                        "cv": 5,
                        "scoring": "accuracy",
                        "n_jobs": -1
                    }
                },
                "optimizer": {
                    "type": "BayesianOptimizer",
                    "parameters": {}
                },
                "parameter_space": {
                    "param1": {"type": "uniform", "low": -10, "high": 10},
                    "param2": {"type": "choice", "options": [1, 2, 3, 4]},
                    "param3": {"type": "log_uniform", "low": 1e-6, "high": 1e-1}
                },
                "execution": {
                    "max_evaluations": 100,
                    "timeout": 3600,
                    "parallel_workers": 4
                }
            },
            "multi_objective": {
                "experiment": {
                    "type": "MultiObjectiveExperiment",
                    "objectives": ["accuracy", "model_size", "inference_time"],
                    "weights": [0.6, 0.2, 0.2]
                },
                "optimizer": {
                    "type": "NSGAIIOptimizer",
                    "parameters": {
                        "population_size": 50,
                        "n_generations": 30
                    }
                }
            }
        }
        
        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(templates[template_type], f, default_flow_style=False, indent=2)
        
        print(f"Created config template: {config_path}")
        return config_path
    
    def load_config(self, config_name):
        """Load configuration from file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.configs[config_name] = config
        return config
    
    def save_config(self, config_name, config_data):
        """Save configuration to file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        self.configs[config_name] = config_data
        print(f"Saved config: {config_path}")
    
    def validate_config(self, config_name):
        """Validate configuration"""
        if config_name not in self.configs:
            self.load_config(config_name)
        
        config = self.configs[config_name]
        
        # Basic validation
        required_sections = ["experiment", "optimizer", "parameter_space"]
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            return False, f"Missing required sections: {missing_sections}"
        
        # Validate parameter space
        param_space = config["parameter_space"]
        for param_name, param_config in param_space.items():
            if "type" not in param_config:
                return False, f"Parameter {param_name} missing type specification"
        
        return True, "Configuration is valid"
    
    def get_available_configs(self):
        """Get list of available configuration files"""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]

# Test configuration management
config_manager = HyperactiveConfigManager()

print("\nConfiguration Management System:")

# Create templates
config_manager.create_config_template("ml_optimization", "optimization")
config_manager.create_config_template("multi_obj_optimization", "multi_objective")

# Load and validate config
try:
    config = config_manager.load_config("ml_optimization")
    is_valid, message = config_manager.validate_config("ml_optimization")
    print(f"Config validation: {message}")
    
    print(f"Available configs: {config_manager.get_available_configs()}")
    
except Exception as e:
    print(f"Config management demo failed: {e}")
```

## Extension Best Practices

### Extension Development Guidelines

```python
class ExtensionBestPractices:
    """Best practices for developing Hyperactive extensions"""
    
    @staticmethod
    def get_development_guidelines():
        return {
            "Architecture": [
                "Follow the plugin architecture pattern",
                "Implement proper interfaces and abstract base classes",
                "Use dependency injection for configurable components",
                "Design for extensibility and modularity"
            ],
            
            "Error Handling": [
                "Implement graceful error handling and recovery",
                "Provide meaningful error messages",
                "Log errors appropriately without breaking the optimization",
                "Handle network failures and timeouts"
            ],
            
            "Performance": [
                "Cache expensive operations when possible",
                "Use lazy loading for heavy resources",
                "Implement proper cleanup and resource management",
                "Consider memory usage for long-running optimizations"
            ],
            
            "Testing": [
                "Write comprehensive unit tests",
                "Test integration with different optimizers",
                "Test error conditions and edge cases",
                "Provide mock implementations for testing"
            ],
            
            "Documentation": [
                "Document all public APIs clearly",
                "Provide usage examples and tutorials",
                "Document configuration options and parameters",
                "Include performance characteristics and limitations"
            ],
            
            "Compatibility": [
                "Maintain backward compatibility when possible",
                "Support multiple versions of dependencies",
                "Test with different Python versions",
                "Handle optional dependencies gracefully"
            ]
        }
    
    @staticmethod
    def create_extension_template():
        """Generate template for new extension"""
        template = '''
"""
Hyperactive Extension Template

This template provides a starting point for creating Hyperactive extensions.
"""

from hyperactive.base import BaseExperiment, BaseOptimizer
from abc import ABC, abstractmethod
import logging

class CustomExtension(ABC):
    """Base class for custom extensions"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for extension"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def initialize(self):
        """Initialize the extension"""
        pass
    
    @abstractmethod
    def get_capabilities(self):
        """Return list of capabilities"""
        pass
    
    def validate_config(self):
        """Validate extension configuration"""
        return True, "Valid configuration"
    
    def cleanup(self):
        """Cleanup resources"""
        pass

class CustomExperimentExtension(CustomExtension):
    """Template for experiment extensions"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.evaluation_count = 0
    
    def initialize(self):
        """Initialize experiment extension"""
        self.logger.info("Initializing experiment extension")
    
    def get_capabilities(self):
        """Return experiment capabilities"""
        return ["custom_evaluation", "result_tracking", "metadata_collection"]
    
    def pre_evaluation_hook(self, params):
        """Hook called before parameter evaluation"""
        self.evaluation_count += 1
        self.logger.debug(f"Pre-evaluation hook: {self.evaluation_count}")
    
    def post_evaluation_hook(self, params, score, metadata):
        """Hook called after parameter evaluation"""
        self.logger.debug(f"Post-evaluation hook: score={score}")

class CustomOptimizerExtension(CustomExtension):
    """Template for optimizer extensions"""
    
    def initialize(self):
        """Initialize optimizer extension"""
        self.logger.info("Initializing optimizer extension")
    
    def get_capabilities(self):
        """Return optimizer capabilities"""
        return ["custom_search", "adaptive_parameters", "multi_objective"]
    
    def modify_search_strategy(self, current_strategy):
        """Modify the search strategy"""
        return current_strategy
    
    def update_parameters(self, iteration, performance_data):
        """Update optimizer parameters based on performance"""
        pass

# Example usage
if __name__ == "__main__":
    # Create custom extension
    experiment_ext = CustomExperimentExtension({
        "evaluation_limit": 100,
        "early_stopping": True
    })
    
    # Initialize and use
    experiment_ext.initialize()
    print(f"Capabilities: {experiment_ext.get_capabilities()}")
'''
        
        return template

# Display best practices
practices = ExtensionBestPractices()
guidelines = practices.get_development_guidelines()

print("\nExtension Development Best Practices:")
for category, practices_list in guidelines.items():
    print(f"\n{category}:")
    for practice in practices_list:
        print(f"  • {practice}")

# Generate extension template
template = practices.create_extension_template()
print("\nExtension template generated (see code above for full template)")
```

## Summary

Hyperactive's extension system provides powerful ways to customize and extend optimization capabilities:

1. **Custom Experiments**: Database integration, web APIs, domain-specific evaluations
2. **Backend Integration**: Ray Tune, MLflow, custom optimization libraries
3. **Domain Extensions**: AutoML, NAS, specialized optimization domains
4. **Plugin System**: Modular architecture for reusable components
5. **Configuration Management**: Structured configuration and templates
6. **Best Practices**: Guidelines for robust extension development

The key to successful extensions is:
- Following established patterns and interfaces
- Implementing proper error handling and logging
- Designing for modularity and reusability
- Providing comprehensive documentation and examples
- Testing thoroughly with different scenarios

These extensions enable Hyperactive to adapt to virtually any optimization scenario while maintaining consistency and reliability.

## References

- Plugin architecture patterns
- API integration best practices  
- Configuration management systems
- Extension development methodologies