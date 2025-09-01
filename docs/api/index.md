# Base Classes

Hyperactive v5 is built on a foundation of base classes that provide the core functionality for optimization algorithms and experiments. These classes use the skbase framework and provide a consistent interface across all components.

## Architecture Overview

The base class system in Hyperactive follows a modular design pattern where each component has clearly defined responsibilities:

- **BaseOptimizer**: Abstract base class for all optimization algorithms
- **BaseExperiment**: Abstract base class for defining optimization problems
- **Tag System**: Metadata framework for algorithm properties and capabilities
- **Plugin Architecture**: Extensible system for adding custom optimizers and experiments

## Design Principles

### Separation of Concerns
The architecture separates the optimization logic (BaseOptimizer) from the problem definition (BaseExperiment), allowing for flexible combinations of algorithms and problems.

### Consistent Interface
All optimizers and experiments follow the same API patterns, making it easy to swap algorithms or experiment types without changing the core optimization code.

### Extensibility
The base classes provide extension points through abstract methods and the tag system, allowing developers to create custom optimizers and experiments that integrate seamlessly with the framework.

