# Migration Guide: v4 to v5

This guide helps you migrate from Hyperactive v4 to v5. The v5 release introduces significant architectural changes that improve modularity and extensibility, but require updates to existing code.

## Overview of Changes

### Major Changes in v5

1. **No more `Hyperactive` class**: Individual optimizer classes are used directly
2. **Experiment-based approach**: Optimization problems are defined as experiment objects
3. **Backend-specific imports**: Optimizers are imported from specific backend modules
4. **New base class architecture**: Uses skbase framework for extensibility
5. **Enhanced integration layer**: Better support for ML frameworks

## Before You Start

### Check Your Dependencies

V5 has updated dependencies:

```bash
# Install v5
pip install hyperactive>=5.0.0

# Key new dependencies
pip install scikit-base
pip install gradient-free-optimizers>=1.2.4
```

## Basic API Migration

### v4 Pattern

```python
--8<-- "tutorials_migration_guide_example.py"
```

### v5 Pattern

```python
--8<-- "tutorials_migration_guide_example_2.py"
```

## Migration Examples

### Example 1: Basic Hyperparameter Optimization

**v4 Code:**
```python
--8<-- "tutorials_migration_guide_example_3.py"
```

**v5 Code:**
```python
--8<-- "tutorials_migration_guide_example_4.py"
```

### Example 2: Custom Objective Function

**v4 Code:**
```python
--8<-- "tutorials_migration_guide_example_5.py"
```

**v5 Code:**
```python
--8<-- "tutorials_migration_guide_example_6.py"
```

### Example 3: Multiple Search Runs

**v4 Code:**
```python
--8<-- "tutorials_migration_guide_example_7.py"
```

**v5 Code:**
```python
--8<-- "tutorials_migration_guide_example_8.py"
```

## Algorithm Name Mapping

Many algorithm names have changed between versions:

| v4 Name | v5 Import | v5 Class Name |
|---------|-----------|---------------|
| `"HillClimbing"` | `from hyperactive.opt.gfo import HillClimbing` | `HillClimbing` |
| `"BayesianOptimization"` | `from hyperactive.opt.gfo import BayesianOptimizer` | `BayesianOptimizer` |
| `"RandomSearch"` | `from hyperactive.opt.gfo import RandomSearch` | `RandomSearch` |
| `"GridSearch"` | `from hyperactive.opt.gfo import GridSearch` | `GridSearch` |
| `"SimulatedAnnealing"` | `from hyperactive.opt.gfo import SimulatedAnnealing` | `SimulatedAnnealing` |
| `"ParticleSwarm"` | `from hyperactive.opt.gfo import ParticleSwarmOptimizer` | `ParticleSwarmOptimizer` |
| `"GeneticAlgorithm"` | `from hyperactive.opt.gfo import GeneticAlgorithm` | `GeneticAlgorithm` |

### New Algorithms in v5

V5 adds many new algorithms not available in v4:

```python
--8<-- "tutorials_migration_guide_example_9.py"
```

## Parameter Configuration

### v4 Parameter Passing

```python
--8<-- "tutorials_migration_guide_example_10.py"
```

### v5 Parameter Passing

```python
--8<-- "tutorials_migration_guide_example_11.py"
```

## Advanced Migration Patterns

### Memory and Warm Starting

**v4 Code:**
```python
--8<-- "tutorials_migration_guide_example_12.py"
```

**v5 Code:**
```python
--8<-- "tutorials_migration_guide_example_13.py"
```

### Parallel Processing

**v4 Code:**
```python
--8<-- "tutorials_migration_guide_example_14.py"
```

**v5 Code:**
```python
--8<-- "tutorials_migration_guide_example_15.py"
```

## Integration with ML Frameworks

### Scikit-learn Integration

**v4 Approach:**
```python
--8<-- "tutorials_migration_guide_example_16.py"
```

**v5 Approach:**
```python
--8<-- "tutorials_migration_guide_example_17.py"
```

## Common Migration Issues

### 1. Search Space Definition

**Problem:** v4 used various formats for search spaces.

**Solution:** v5 uses consistent dict format:
```python
--8<-- "tutorials_migration_guide_example_18.py"
```

### 2. Result Access

**Problem:** Different result access patterns.

**Solution:**
```python
--8<-- "tutorials_migration_guide_example_19.py"
```

### 3. Progress Monitoring

**Problem:** v5 doesn't have built-in progress bars.

**Solution:** Implement custom monitoring:
```python
--8<-- "tutorials_migration_guide_example_20.py"
```

## Testing Your Migration

Create a simple test to verify your migration:

```python
--8<-- "tutorials_migration_guide_example_21.py"
```

## Performance Considerations

V5 may have different performance characteristics:

1. **Startup time**: May be slightly higher due to skbase framework
2. **Memory usage**: Generally lower due to more efficient architecture  
3. **Evaluation speed**: Similar or faster for most use cases
4. **Extensibility**: Much better - easier to add custom components

## Getting Help

If you encounter issues during migration:

1. **Check examples**: Look at the example files in v5 repository
2. **Read documentation**: This documentation covers all new features
3. **Compare patterns**: Use this guide's before/after examples
4. **Community support**: Ask questions on GitHub issues

## Summary Checklist

- [ ] Update imports to use specific backend modules
- [ ] Convert objective functions to experiment classes
- [ ] Replace `Hyperactive()` with individual optimizer classes  
- [ ] Update algorithm names and parameter passing
- [ ] Test migrated code with simple examples
- [ ] Update any custom extensions to use new base classes
- [ ] Consider new v5 features like OptCV for sklearn integration

The migration requires some effort, but v5's improved architecture provides much better extensibility and cleaner interfaces for long-term use.