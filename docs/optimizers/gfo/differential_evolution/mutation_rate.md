### `mutation_rate`

Mutation Rate (F), also known as the differential weight, is a parameter in Differential Evolution (DE) that controls the amplification of the differential variation between individuals. It is a scaling factor applied to the difference between two randomly selected population vectors before adding the result to a third vector to create a mutant vector. The mutation rate influences the algorithm's ability to explore the search space; a higher value of F increases the diversity of the mutant vectors, leading to broader exploration, while a lower value encourages convergence by making smaller adjustments. The typical range for F is between 0.4 and 1.0, though values outside this range can be used depending on the problem characteristics.

$$
\mathbf{v}_{i,G+1} = \mathbf{x}_{r1,G} + F \cdot (\mathbf{x}_{r2,G} - \mathbf{x}_{r3,G})
$$



  - **type**: float
  - **default**: 0.9
  - **typical range**: 0.4 ... 1.0

---