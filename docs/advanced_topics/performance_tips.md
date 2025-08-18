# Performance Tips

## Introduction

Optimization performance is critical when dealing with expensive evaluations, large parameter spaces, or time-constrained scenarios. This page provides comprehensive strategies for maximizing the efficiency and effectiveness of your optimization processes using Hyperactive.

## Computational Performance Optimization

### Parallel Evaluation Strategies

```python
import concurrent.futures
import multiprocessing as mp
from hyperactive.base import BaseExperiment
import numpy as np
import time

class ParallelizableExperiment(BaseExperiment):
    """Experiment designed for parallel evaluation"""
    
    def __init__(self, computation_time=0.1):
        super().__init__()
        self.computation_time = computation_time
    
    def _paramnames(self):
        return ["x", "y", "z"]
    
    def _evaluate(self, params):
        # Simulate expensive computation
        time.sleep(self.computation_time)
        
        x, y, z = params["x"], params["y"], params["z"]
        result = -(x**2 + y**2 + z**2)  # Simple sphere function
        
        return result, {"computation_time": self.computation_time}

class ParallelOptimizer:
    """Wrapper for parallel optimization evaluation"""
    
    def __init__(self, experiment, base_optimizer_class, n_workers=None):
        self.experiment = experiment
        self.base_optimizer_class = base_optimizer_class
        self.n_workers = n_workers or mp.cpu_count()
        
    def _evaluate_batch(self, param_list):
        """Evaluate a batch of parameters in parallel"""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all evaluations
            future_to_params = {
                executor.submit(self._single_evaluation, params): params 
                for params in param_list
            }
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    score, metadata = future.result()
                    results.append((params, score, metadata))
                except Exception as e:
                    print(f"Evaluation failed for {params}: {e}")
                    results.append((params, float('-inf'), {"error": str(e)}))
            
            return results
    
    def _single_evaluation(self, params):
        """Single parameter evaluation (for process pool)"""
        return self.experiment.score(params)
    
    def solve_with_parallel_evaluation(self, batch_size=10):
        """Solve optimization with parallel batch evaluation"""
        # This is a simplified example - real implementation would integrate 
        # with the optimizer's evaluation loop
        
        param_names = self.experiment.paramnames()
        best_score = float('-inf')
        best_params = None
        
        # Generate and evaluate batches
        for batch_num in range(5):  # 5 batches for demo
            # Generate batch of random parameters
            param_batch = []
            for _ in range(batch_size):
                params = {name: np.random.uniform(-10, 10) for name in param_names}
                param_batch.append(params)
            
            # Evaluate batch in parallel
            start_time = time.time()
            batch_results = self._evaluate_batch(param_batch)
            batch_time = time.time() - start_time
            
            # Find best in batch
            for params, score, metadata in batch_results:
                if score > best_score:
                    best_score = score
                    best_params = params
            
            print(f"Batch {batch_num + 1}: {batch_time:.2f}s for {batch_size} evaluations")
        
        return best_params, best_score

# Example: Compare sequential vs parallel evaluation
experiment = ParallelizableExperiment(computation_time=0.05)

# Sequential evaluation
start_time = time.time()
sequential_results = []
for _ in range(20):
    params = {"x": np.random.uniform(-10, 10), 
              "y": np.random.uniform(-10, 10), 
              "z": np.random.uniform(-10, 10)}
    score, metadata = experiment.score(params)
    sequential_results.append(score)
sequential_time = time.time() - start_time

# Parallel evaluation
parallel_optimizer = ParallelOptimizer(experiment, None, n_workers=4)
start_time = time.time()
best_params, best_score = parallel_optimizer.solve_with_parallel_evaluation(batch_size=20)
parallel_time = time.time() - start_time

print(f"\nPerformance Comparison:")
print(f"Sequential time: {sequential_time:.2f}s")
print(f"Parallel time: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time / parallel_time:.2f}x")
```

### Memory-Efficient Implementation

```python
class MemoryEfficientExperiment(BaseExperiment):
    """Memory-efficient experiment implementation"""
    
    def __init__(self, large_dataset_size=1000000):
        super().__init__()
        self.dataset_size = large_dataset_size
        self._cached_data = None
        self.evaluation_count = 0
        
    def _paramnames(self):
        return ["learning_rate", "batch_size", "regularization"]
    
    def _lazy_load_data(self):
        """Lazy loading of large datasets"""
        if self._cached_data is None:
            print("Loading large dataset...")
            # Simulate loading large dataset
            self._cached_data = np.random.random((self.dataset_size, 10))
        return self._cached_data
    
    def _evaluate(self, params):
        self.evaluation_count += 1
        
        # Memory-efficient parameter extraction
        lr = np.float32(params["learning_rate"])  # Use float32 instead of float64
        batch_size = max(1, int(params["batch_size"]))
        reg = np.float32(params["regularization"])
        
        # Process data in chunks to avoid memory overload
        data = self._lazy_load_data()
        chunk_size = min(batch_size * 100, 10000)
        
        total_loss = 0.0
        num_chunks = 0
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Simulate model training on chunk
            chunk_loss = self._compute_chunk_loss(chunk, lr, reg)
            total_loss += chunk_loss
            num_chunks += 1
            
            # Clear intermediate variables to free memory
            del chunk
        
        avg_loss = total_loss / num_chunks
        
        return -avg_loss, {
            "avg_loss": avg_loss,
            "num_chunks": num_chunks,
            "memory_efficient": True
        }
    
    def _compute_chunk_loss(self, chunk, lr, reg):
        """Compute loss for a data chunk"""
        # Simulate computation
        weights = np.random.random(chunk.shape[1]).astype(np.float32)
        predictions = chunk.dot(weights)
        targets = np.random.random(len(chunk)).astype(np.float32)
        
        # Mean squared error with regularization
        mse = np.mean((predictions - targets)**2)
        l2_penalty = reg * np.sum(weights**2)
        
        return mse + l2_penalty

# Memory profiling helper
import psutil
import os

def monitor_memory_usage(func):
    """Decorator to monitor memory usage"""
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB "
              f"(delta: {mem_after - mem_before:.1f}MB)")
        
        return result
    return wrapper

# Test memory efficiency
@monitor_memory_usage
def test_memory_efficient_optimization():
    experiment = MemoryEfficientExperiment(large_dataset_size=100000)
    
    from hyperactive.opt.gfo import RandomSearch
    optimizer = RandomSearch(experiment=experiment)
    
    best_params = optimizer.solve()
    return best_params

print("Memory-Efficient Optimization Test:")
best_result = test_memory_efficient_optimization()
```

### Evaluation Caching

```python
import hashlib
import pickle
from pathlib import Path

class CachedExperiment(BaseExperiment):
    """Experiment with intelligent caching of evaluations"""
    
    def __init__(self, cache_dir="optimization_cache", cache_enabled=True):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _paramnames(self):
        return ["param1", "param2", "param3"]
    
    def _get_cache_key(self, params):
        """Generate unique cache key for parameters"""
        # Round parameters to avoid precision issues
        rounded_params = {k: round(v, 6) for k, v in params.items()}
        param_str = str(sorted(rounded_params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _save_to_cache(self, cache_key, result):
        """Save evaluation result to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Cache save failed: {e}")
    
    def _load_from_cache(self, cache_key):
        """Load evaluation result from cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load failed: {e}")
        return None
    
    def _evaluate(self, params):
        cache_key = self._get_cache_key(params)
        
        # Check cache first
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            return cached_result
        
        # Cache miss - compute result
        self.cache_misses += 1
        
        # Expensive computation
        time.sleep(0.01)  # Simulate computation time
        p1, p2, p3 = params["param1"], params["param2"], params["param3"]
        score = -(p1**2 + p2**2 + p3**2)  # Simple function
        
        result = (score, {"computed": True, "cache_key": cache_key})
        
        # Save to cache
        self._save_to_cache(cache_key, result)
        
        return result
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_evaluations = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_evaluations if total_evaluations > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_evaluations": total_evaluations
        }

# Test caching effectiveness
cached_experiment = CachedExperiment()

# First optimization run
print("First optimization run (cold cache):")
from hyperactive.opt.gfo import BayesianOptimizer

start_time = time.time()
optimizer1 = BayesianOptimizer(experiment=cached_experiment)
best_params1 = optimizer1.solve()
first_run_time = time.time() - start_time

print(f"First run time: {first_run_time:.2f}s")
print(f"Cache stats: {cached_experiment.get_cache_stats()}")

# Second optimization run (should benefit from cache)
print("\nSecond optimization run (warm cache):")
start_time = time.time()
optimizer2 = BayesianOptimizer(experiment=cached_experiment)
best_params2 = optimizer2.solve()
second_run_time = time.time() - start_time

print(f"Second run time: {second_run_time:.2f}s")
print(f"Cache stats: {cached_experiment.get_cache_stats()}")
print(f"Speedup from caching: {first_run_time / second_run_time:.2f}x")
```

## Algorithm-Specific Performance Tuning

### Bayesian Optimization Efficiency

```python
class EfficientBayesianExperiment(BaseExperiment):
    """Experiment optimized for Bayesian optimization"""
    
    def __init__(self):
        super().__init__()
        self.evaluation_history = []
        self.cheap_evaluations = 0
        self.expensive_evaluations = 0
    
    def _paramnames(self):
        return ["learning_rate", "hidden_size", "dropout", "batch_size"]
    
    def _evaluate(self, params):
        # Early stopping for clearly bad parameters
        if self._is_clearly_bad(params):
            self.cheap_evaluations += 1
            return float('-inf'), {"early_stopped": True}
        
        # Full evaluation for promising parameters
        self.expensive_evaluations += 1
        score = self._full_evaluation(params)
        
        self.evaluation_history.append({
            "params": params.copy(),
            "score": score,
            "evaluation_type": "full"
        })
        
        return score, {"early_stopped": False}
    
    def _is_clearly_bad(self, params):
        """Quick heuristics to identify obviously bad parameters"""
        # Learning rate too high or too low
        if params["learning_rate"] > 1.0 or params["learning_rate"] < 1e-6:
            return True
        
        # Hidden size unreasonable
        if params["hidden_size"] < 8 or params["hidden_size"] > 1024:
            return True
        
        # Dropout too extreme
        if params["dropout"] < 0 or params["dropout"] > 0.95:
            return True
        
        return False
    
    def _full_evaluation(self, params):
        """Expensive full evaluation"""
        time.sleep(0.02)  # Simulate expensive computation
        
        # Simulate neural network performance
        lr = params["learning_rate"]
        hidden_size = params["hidden_size"]
        dropout = params["dropout"]
        batch_size = params["batch_size"]
        
        # Performance model (simplified)
        lr_penalty = -abs(np.log10(lr) + 3)**2  # Optimal around 1e-3
        size_bonus = min(hidden_size / 128, 1.0)  # Diminishing returns
        dropout_bonus = dropout * (1 - dropout) * 4  # Inverted U-shape
        batch_penalty = -abs(batch_size - 64)**2 / 1000  # Optimal around 64
        
        return lr_penalty + size_bonus + dropout_bonus + batch_penalty
    
    def get_efficiency_stats(self):
        """Get evaluation efficiency statistics"""
        total_evals = self.cheap_evaluations + self.expensive_evaluations
        savings = self.cheap_evaluations / total_evals if total_evals > 0 else 0
        
        return {
            "cheap_evaluations": self.cheap_evaluations,
            "expensive_evaluations": self.expensive_evaluations,
            "total_evaluations": total_evals,
            "computational_savings": savings
        }

# Test efficient Bayesian optimization
efficient_experiment = EfficientBayesianExperiment()

start_time = time.time()
efficient_optimizer = BayesianOptimizer(experiment=efficient_experiment)
efficient_best = efficient_optimizer.solve()
efficient_time = time.time() - start_time

print("Efficient Bayesian Optimization Results:")
print(f"Best parameters: {efficient_best}")
print(f"Optimization time: {efficient_time:.2f}s")
print(f"Efficiency stats: {efficient_experiment.get_efficiency_stats()}")
```

### Population-Based Algorithm Tuning

```python
class TunedPopulationExperiment(BaseExperiment):
    """Experiment for tuning population-based algorithms"""
    
    def __init__(self):
        super().__init__()
        self.generation_stats = []
    
    def _paramnames(self):
        return ["x", "y"]
    
    def _evaluate(self, params):
        x, y = params["x"], params["y"]
        
        # Rosenbrock function (challenging for population algorithms)
        result = -((1 - x)**2 + 100 * (y - x**2)**2)
        
        return result, {"rosenbrock_value": -result}

def compare_population_sizes():
    """Compare different population sizes for PSO"""
    from hyperactive.opt.gfo import ParticleSwarmOptimizer
    
    experiment = TunedPopulationExperiment()
    population_sizes = [10, 20, 30, 50, 100]
    results = {}
    
    for pop_size in population_sizes:
        print(f"Testing population size: {pop_size}")
        
        # Multiple runs for statistical significance
        scores = []
        times = []
        
        for run in range(3):
            start_time = time.time()
            
            optimizer = ParticleSwarmOptimizer(
                experiment=experiment,
                population=pop_size
            )
            
            best_params = optimizer.solve()
            score = experiment.score(best_params)[0]
            
            end_time = time.time()
            
            scores.append(score)
            times.append(end_time - start_time)
        
        results[pop_size] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_time": np.mean(times),
            "best_score": max(scores)
        }
    
    return results

# Analyze population size impact
print("Population Size Analysis for PSO:")
pop_results = compare_population_sizes()

for pop_size, stats in pop_results.items():
    print(f"Population {pop_size}: Score = {stats['mean_score']:.6f} ± {stats['std_score']:.6f}, "
          f"Time = {stats['mean_time']:.2f}s")

# Find optimal population size
best_pop_size = max(pop_results.items(), key=lambda x: x[1]['mean_score'])
print(f"\nOptimal population size: {best_pop_size[0]} "
      f"(score: {best_pop_size[1]['mean_score']:.6f})")
```

## Early Stopping and Convergence Detection

### Adaptive Early Stopping

```python
class EarlyStoppingExperiment(BaseExperiment):
    """Experiment with intelligent early stopping"""
    
    def __init__(self, patience=10, min_improvement=1e-6):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.score_history = []
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.stopped_early = False
    
    def _paramnames(self):
        return ["param1", "param2", "param3"]
    
    def _evaluate(self, params):
        # Check if we should stop early
        if self.should_stop_early():
            self.stopped_early = True
            return float('-inf'), {"early_stopped": True}
        
        # Normal evaluation
        p1, p2, p3 = params["param1"], params["param2"], params["param3"]
        score = -(p1**2 + p2**2 + p3**2)
        
        # Update stopping criteria
        self._update_stopping_criteria(score)
        
        return score, {"early_stopped": False}
    
    def should_stop_early(self):
        """Check if optimization should stop early"""
        return self.no_improvement_count >= self.patience
    
    def _update_stopping_criteria(self, score):
        """Update early stopping criteria"""
        self.score_history.append(score)
        
        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

class EarlyStoppingOptimizer:
    """Optimizer wrapper with convergence detection"""
    
    def __init__(self, base_optimizer, convergence_window=20, convergence_threshold=1e-8):
        self.base_optimizer = base_optimizer
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.iteration_scores = []
    
    def solve(self):
        """Solve with convergence monitoring"""
        # This is a simplified example - real implementation would hook into
        # the optimizer's iteration loop
        
        param_names = self.base_optimizer.experiment.paramnames()
        best_score = float('-inf')
        best_params = None
        
        for iteration in range(200):  # Max iterations
            # Generate random candidate (simplified)
            params = {name: np.random.uniform(-10, 10) for name in param_names}
            
            # Evaluate
            score, metadata = self.base_optimizer.experiment.score(params)
            self.iteration_scores.append(score)
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params
            
            # Check convergence
            if self._check_convergence():
                print(f"Converged at iteration {iteration}")
                break
        
        return best_params
    
    def _check_convergence(self):
        """Check if optimization has converged"""
        if len(self.iteration_scores) < self.convergence_window:
            return False
        
        # Check variance in recent scores
        recent_scores = self.iteration_scores[-self.convergence_window:]
        score_variance = np.var(recent_scores)
        
        # Check trend
        if len(recent_scores) >= 2:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            if abs(trend) < self.convergence_threshold:
                return True
        
        return score_variance < self.convergence_threshold

# Test early stopping
early_stop_experiment = EarlyStoppingExperiment(patience=15)

print("Early Stopping Optimization:")
early_stop_optimizer = EarlyStoppingOptimizer(
    base_optimizer=type('MockOptimizer', (), {'experiment': early_stop_experiment})(),
    convergence_window=25
)

best_params = early_stop_optimizer.solve()
print(f"Optimization completed")
print(f"Early stopped: {early_stop_experiment.stopped_early}")
print(f"Total evaluations: {len(early_stop_experiment.score_history)}")
```

## Resource Management

### GPU Memory Management

```python
class GPUAwareExperiment(BaseExperiment):
    """Experiment that manages GPU memory efficiently"""
    
    def __init__(self, gpu_memory_limit=0.8):
        super().__init__()
        self.gpu_memory_limit = gpu_memory_limit
        self.gpu_available = self._check_gpu_availability()
        
    def _paramnames(self):
        return ["model_size", "batch_size", "sequence_length"]
    
    def _check_gpu_availability(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _estimate_gpu_memory_usage(self, params):
        """Estimate GPU memory usage for given parameters"""
        model_size = params["model_size"]
        batch_size = params["batch_size"]
        sequence_length = params["sequence_length"]
        
        # Simplified memory estimation (in GB)
        model_memory = model_size * 4 / 1e9  # 4 bytes per parameter
        batch_memory = batch_size * sequence_length * model_size * 8 / 1e9  # Forward + backward
        
        return model_memory + batch_memory
    
    def _evaluate(self, params):
        if self.gpu_available:
            estimated_memory = self._estimate_gpu_memory_usage(params)
            
            # Get available GPU memory
            try:
                import torch
                torch.cuda.empty_cache()  # Clear cache
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                available_memory = total_memory * self.gpu_memory_limit
                
                if estimated_memory > available_memory:
                    return float('-inf'), {
                        "error": "insufficient_gpu_memory",
                        "estimated": estimated_memory,
                        "available": available_memory
                    }
            except ImportError:
                pass
        
        # Simulate model training
        score = self._simulate_training(params)
        
        return score, {
            "estimated_memory_gb": self._estimate_gpu_memory_usage(params) if self.gpu_available else 0,
            "gpu_used": self.gpu_available
        }
    
    def _simulate_training(self, params):
        """Simulate model training performance"""
        model_size = params["model_size"]
        batch_size = params["batch_size"]
        sequence_length = params["sequence_length"]
        
        # Simulate performance relationship
        base_performance = np.log(model_size) / np.log(10)  # Larger models are better
        batch_penalty = -abs(batch_size - 32)**2 / 1000  # Optimal around 32
        sequence_bonus = min(sequence_length / 512, 1.0)  # Diminishing returns
        
        return base_performance + batch_penalty + sequence_bonus

# Test GPU-aware optimization
gpu_experiment = GPUAwareExperiment()

print("GPU-Aware Optimization:")
print(f"GPU available: {gpu_experiment.gpu_available}")

# Test with different parameter ranges
test_params = {
    "model_size": 1e8,  # 100M parameters
    "batch_size": 16,
    "sequence_length": 256
}

score, metadata = gpu_experiment.score(test_params)
print(f"Test evaluation: score={score:.4f}, metadata={metadata}")
```

### Disk Space and I/O Optimization

```python
class IOOptimizedExperiment(BaseExperiment):
    """Experiment optimized for disk I/O"""
    
    def __init__(self, data_directory="optimization_data", cleanup_frequency=50):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.cleanup_frequency = cleanup_frequency
        self.evaluation_count = 0
        self.temp_files = []
        
        # Create data directory
        self.data_directory.mkdir(exist_ok=True)
    
    def _paramnames(self):
        return ["data_preprocessing", "model_complexity", "output_detail"]
    
    def _evaluate(self, params):
        self.evaluation_count += 1
        
        # Generate temporary files for this evaluation
        temp_file = self.data_directory / f"temp_eval_{self.evaluation_count}.dat"
        
        try:
            # Simulate data processing
            data_size = int(params["data_preprocessing"] * 1000)
            complexity = params["model_complexity"]
            detail = params["output_detail"]
            
            # Write temporary data
            self._write_temp_data(temp_file, data_size)
            self.temp_files.append(temp_file)
            
            # Simulate computation
            score = self._compute_with_temp_data(temp_file, complexity, detail)
            
            # Periodic cleanup
            if self.evaluation_count % self.cleanup_frequency == 0:
                self._cleanup_temp_files()
            
            return score, {
                "temp_file_size_mb": temp_file.stat().st_size / (1024 * 1024),
                "total_temp_files": len(self.temp_files)
            }
            
        except Exception as e:
            return float('-inf'), {"error": str(e)}
    
    def _write_temp_data(self, temp_file, data_size):
        """Write temporary data efficiently"""
        with open(temp_file, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = 8192
            for _ in range(0, data_size, chunk_size):
                chunk = np.random.bytes(min(chunk_size, data_size))
                f.write(chunk)
    
    def _compute_with_temp_data(self, temp_file, complexity, detail):
        """Compute using temporary data"""
        # Simulate reading and processing
        file_size = temp_file.stat().st_size
        
        # Performance model
        size_penalty = file_size / (1024 * 1024 * 10)  # Penalty for large files
        complexity_bonus = min(complexity, 1.0)
        detail_cost = detail * 0.1
        
        return complexity_bonus - size_penalty - detail_cost
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        cleaned = 0
        remaining_files = []
        
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    cleaned += 1
            except Exception as e:
                print(f"Failed to clean {temp_file}: {e}")
                remaining_files.append(temp_file)
        
        self.temp_files = remaining_files
        print(f"Cleaned up {cleaned} temporary files")
    
    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup_temp_files()

# Test I/O optimized experiment
io_experiment = IOOptimizedExperiment(cleanup_frequency=10)

print("I/O Optimized Optimization:")
from hyperactive.opt.gfo import RandomSearch

io_optimizer = RandomSearch(experiment=io_experiment)
io_best = io_optimizer.solve()

print(f"Best parameters: {io_best}")
print(f"Total evaluations: {io_experiment.evaluation_count}")
print(f"Remaining temp files: {len(io_experiment.temp_files)}")

# Final cleanup
io_experiment._cleanup_temp_files()
```

## Scaling Strategies

### Distributed Optimization

```python
import socket
import json
from threading import Thread
import queue

class DistributedOptimizationCoordinator:
    """Coordinator for distributed optimization"""
    
    def __init__(self, experiment, port=8888):
        self.experiment = experiment
        self.port = port
        self.worker_connections = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        
    def start_coordinator(self):
        """Start the coordination server"""
        self.running = True
        
        # Start server thread
        server_thread = Thread(target=self._run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Start task distribution thread
        task_thread = Thread(target=self._distribute_tasks)
        task_thread.daemon = True
        task_thread.start()
        
        print(f"Coordinator started on port {self.port}")
    
    def _run_server(self):
        """Run the coordination server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', self.port))
            server_socket.listen(5)
            
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    self.worker_connections.append(client_socket)
                    
                    # Handle worker in separate thread
                    worker_thread = Thread(
                        target=self._handle_worker, 
                        args=(client_socket, address)
                    )
                    worker_thread.daemon = True
                    worker_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"Server error: {e}")
    
    def _handle_worker(self, client_socket, address):
        """Handle communication with a worker"""
        print(f"Worker connected from {address}")
        
        try:
            while self.running:
                # Send task to worker
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    task_data = json.dumps(task).encode()
                    client_socket.send(task_data + b'\n')
                    
                    # Receive result
                    response = client_socket.recv(4096).decode()
                    if response:
                        result = json.loads(response.strip())
                        self.result_queue.put(result)
                
                time.sleep(0.1)  # Prevent busy waiting
                
        except Exception as e:
            print(f"Worker {address} error: {e}")
        finally:
            client_socket.close()
    
    def _distribute_tasks(self):
        """Distribute optimization tasks"""
        param_names = self.experiment.paramnames()
        
        # Generate tasks
        for i in range(50):  # 50 evaluations
            task = {
                "task_id": i,
                "parameters": {name: np.random.uniform(-10, 10) for name in param_names}
            }
            self.task_queue.put(task)
        
        print("Tasks distributed to queue")
    
    def collect_results(self, timeout=30):
        """Collect results from workers"""
        results = []
        start_time = time.time()
        
        while len(results) < 50 and (time.time() - start_time) < timeout:
            try:
                result = self.result_queue.get(timeout=1)
                results.append(result)
                print(f"Received result {len(results)}/50")
            except queue.Empty:
                continue
        
        return results
    
    def stop(self):
        """Stop the coordinator"""
        self.running = False
        for conn in self.worker_connections:
            conn.close()

class DistributedWorker:
    """Worker for distributed optimization"""
    
    def __init__(self, experiment, coordinator_host='localhost', coordinator_port=8888):
        self.experiment = experiment
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        
    def start_worker(self):
        """Start worker and connect to coordinator"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as worker_socket:
                worker_socket.connect((self.coordinator_host, self.coordinator_port))
                print("Connected to coordinator")
                
                while True:
                    # Receive task
                    data = worker_socket.recv(4096).decode()
                    if not data:
                        break
                    
                    task = json.loads(data.strip())
                    
                    # Evaluate task
                    params = task["parameters"]
                    score, metadata = self.experiment.score(params)
                    
                    # Send result
                    result = {
                        "task_id": task["task_id"],
                        "score": score,
                        "metadata": metadata
                    }
                    
                    response = json.dumps(result).encode()
                    worker_socket.send(response + b'\n')
                    
        except Exception as e:
            print(f"Worker error: {e}")

# Example usage (simplified - would normally run on separate machines)
def simulate_distributed_optimization():
    """Simulate distributed optimization"""
    
    experiment = ParallelizableExperiment(computation_time=0.01)
    
    # Start coordinator
    coordinator = DistributedOptimizationCoordinator(experiment)
    coordinator.start_coordinator()
    
    time.sleep(1)  # Let coordinator start
    
    # Simulate workers (in practice, these would be separate processes/machines)
    workers = []
    for i in range(3):
        worker = DistributedWorker(experiment)
        worker_thread = Thread(target=worker.start_worker)
        worker_thread.daemon = True
        worker_thread.start()
        workers.append(worker_thread)
    
    # Collect results
    results = coordinator.collect_results(timeout=10)
    
    # Stop coordinator
    coordinator.stop()
    
    return results

print("\nDistributed Optimization Simulation:")
try:
    distributed_results = simulate_distributed_optimization()
    print(f"Collected {len(distributed_results)} results from distributed workers")
    
    if distributed_results:
        best_result = max(distributed_results, key=lambda x: x["score"])
        print(f"Best distributed result: {best_result['score']:.6f}")
except Exception as e:
    print(f"Distributed optimization simulation failed: {e}")
```

## Performance Monitoring and Profiling

### Optimization Performance Profiler

```python
import cProfile
import pstats
from contextlib import contextmanager

class OptimizationProfiler:
    """Profiler for optimization performance analysis"""
    
    def __init__(self):
        self.profiles = {}
        self.timing_data = {}
        
    @contextmanager
    def profile_optimization(self, name):
        """Context manager for profiling optimization runs"""
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            end_time = time.time()
            
            # Store profiling data
            self.profiles[name] = profiler
            self.timing_data[name] = end_time - start_time
    
    def print_profile_stats(self, name, top_n=10):
        """Print profiling statistics"""
        if name not in self.profiles:
            print(f"No profile data for {name}")
            return
        
        print(f"\nProfile Statistics for {name}:")
        print(f"Total time: {self.timing_data[name]:.3f} seconds")
        print("-" * 50)
        
        stats = pstats.Stats(self.profiles[name])
        stats.sort_stats('cumulative')
        stats.print_stats(top_n)
    
    def compare_profiles(self, name1, name2):
        """Compare two optimization profiles"""
        if name1 not in self.timing_data or name2 not in self.timing_data:
            print("Missing profile data for comparison")
            return
        
        time1 = self.timing_data[name1]
        time2 = self.timing_data[name2]
        
        print(f"\nProfile Comparison:")
        print(f"{name1}: {time1:.3f}s")
        print(f"{name2}: {time2:.3f}s")
        print(f"Speedup: {time1/time2:.2f}x" if time2 < time1 else f"Slowdown: {time2/time1:.2f}x")

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"{func.__name__} Performance:")
        print(f"  Time: {end_time - start_time:.3f}s")
        print(f"  Memory: {start_memory:.1f}MB -> {end_memory:.1f}MB")
        print(f"  Memory delta: {end_memory - start_memory:.1f}MB")
        
        return result
    return wrapper

# Example profiling usage
profiler = OptimizationProfiler()

# Profile different optimization approaches
@monitor_performance
def test_baseline_optimization():
    experiment = TunedPopulationExperiment()
    optimizer = BayesianOptimizer(experiment=experiment)
    return optimizer.solve()

@monitor_performance  
def test_cached_optimization():
    experiment = CachedExperiment(cache_enabled=True)
    optimizer = BayesianOptimizer(experiment=experiment)
    return optimizer.solve()

print("Performance Profiling Results:")

# Profile baseline
with profiler.profile_optimization("baseline"):
    baseline_result = test_baseline_optimization()

# Profile cached version
with profiler.profile_optimization("cached"):
    cached_result = test_cached_optimization()

# Compare profiles
profiler.compare_profiles("baseline", "cached")
```

## Best Practices Summary

### Performance Optimization Checklist

```python
def optimization_performance_checklist():
    """Comprehensive performance optimization checklist"""
    
    checklist = {
        "Evaluation Efficiency": [
            "✓ Implement evaluation caching for expensive functions",
            "✓ Use early stopping for obviously bad parameters",
            "✓ Consider surrogate models for very expensive evaluations",
            "✓ Batch evaluations when possible",
            "✓ Implement parameter validation before expensive computation"
        ],
        
        "Memory Management": [
            "✓ Use lazy loading for large datasets",
            "✓ Process data in chunks to avoid memory overload",
            "✓ Clean up temporary files and variables",
            "✓ Monitor GPU memory usage",
            "✓ Use appropriate data types (float32 vs float64)"
        ],
        
        "Parallelization": [
            "✓ Evaluate candidates in parallel when possible",
            "✓ Use appropriate number of workers (typically # of cores)",
            "✓ Consider distributed optimization for large-scale problems",
            "✓ Balance communication overhead vs parallelism benefits",
            "✓ Use process pools for CPU-bound tasks"
        ],
        
        "Algorithm Selection": [
            "✓ Choose algorithms appropriate for problem characteristics",
            "✓ Tune algorithm hyperparameters (population size, learning rates)",
            "✓ Use adaptive algorithms when problem dynamics change",
            "✓ Consider hybrid approaches for complex problems",
            "✓ Implement convergence detection"
        ],
        
        "Resource Management": [
            "✓ Monitor and limit memory usage",
            "✓ Implement disk space cleanup",
            "✓ Use appropriate timeout values",
            "✓ Consider computation vs communication trade-offs",
            "✓ Profile code to identify bottlenecks"
        ],
        
        "Scalability": [
            "✓ Design experiments to scale with problem size",
            "✓ Use efficient data structures",
            "✓ Implement incremental learning when possible",
            "✓ Consider approximation methods for large-scale problems",
            "✓ Design for distributed/cloud deployment"
        ]
    }
    
    return checklist

# Print performance checklist
performance_checklist = optimization_performance_checklist()

print("\nOptimization Performance Checklist:")
print("=" * 50)

for category, items in performance_checklist.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

print("\nKey Performance Metrics to Monitor:")
print("• Evaluations per second")
print("• Memory usage per evaluation") 
print("• Cache hit rate")
print("• Convergence speed")
print("• Resource utilization")
print("• Scalability with problem size")
```

## Summary

Performance optimization in Hyperactive involves multiple dimensions:

1. **Computational Efficiency**: Parallel evaluation, caching, early stopping
2. **Memory Management**: Lazy loading, chunking, cleanup
3. **Algorithm Tuning**: Appropriate parameter selection and adaptation
4. **Resource Management**: GPU memory, disk space, network bandwidth
5. **Scalability**: Distributed computing, incremental processing

The key is to profile your specific optimization problem and apply the most relevant techniques. Start with the highest-impact optimizations (typically evaluation caching and parallelization) and progressively add more sophisticated techniques as needed.

Remember that premature optimization can be counterproductive - focus on optimizations that provide measurable benefits for your specific use case.

## References

- High-Performance Computing for optimization
- Distributed optimization algorithms
- Memory management in scientific computing
- Profiling and performance analysis tools