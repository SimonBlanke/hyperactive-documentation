### `mutation_rate`

The mutation rate is a parameter in genetic algorithms that specifies the probability of randomly altering the value of a gene in a chromosome. Mutation helps in maintaining genetic diversity within the population and prevents the algorithm from getting stuck in local optima.

$$
x'_i = 
\begin{cases} 
x_i & \text{if } \text{rand} > p_m \\
1 - x_i & \text{if } \text{rand} \leq p_m
\end{cases}
$$


  - **type**: float
  - **default**: 0.5
  - **typical range**: 0.0 ... 1.0

---