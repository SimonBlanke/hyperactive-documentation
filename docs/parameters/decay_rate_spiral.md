### `decay_rate`

 The `decay_rate` is called $r(k)$ in the spiral-optimization equation and is usually referred to as a step-size, but behaves more like a modification factor of the radius of the spiral movement of the particles in this implementation. This parameter of the spiral-optimization algorithm is a factor, that influences the radius of the particles during their spiral movement. Lower values accelerates the convergence of the particles to the best known position, while values above 1 eventually lead to a movement where the particles spiral away from each other. 

  - **type**: float
  - **default**: 0.99
  - **typical range**: 0.85 ... 1.15

---