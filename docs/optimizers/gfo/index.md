# Gradient-Free-Optimizers

The GFO backend is Hyperactive's primary optimization engine with 20+ algorithms spanning the complete spectrum of optimization methods. This is your go-to choice for maximum algorithmic flexibility and research applications.

## Why Choose GFO?

- **Largest Algorithm Collection**: 20+ methods from simple hill climbing to advanced Bayesian optimization
- **Full Customization**: Direct access to all algorithm parameters and settings
- **Research-Friendly**: Perfect for experimenting with different optimization strategies

## Usage Pattern

Combine a GFO optimizer with an experiment and call `solve()`:

```python
from hyperactive.experiment.integrations import SklearnCvExperiment
from hyperactive.opt.gfo import HillClimbing

exp = SklearnCvExperiment(...)
opt = HillClimbing(experiment=exp)
best = opt.solve()
```

See the Quick Start for end‑to‑end examples and the Experiments section for building custom problems.
