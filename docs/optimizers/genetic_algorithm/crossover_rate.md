### `crossover_rate`

The crossover rate is a parameter in genetic algorithms that determines the probability with which crossover (recombination) occurs between pairs of parent chromosomes. It controls how often parts of the parent chromosomes are exchanged to produce offspring. A higher crossover rate increases the diversity of the offspring, which can help in exploring the solution space more effectively.

$$
u_{i,j}^{(g)} = 
\begin{cases} 
v_{i,j}^{(g)} & \text{if } \text{rand}_j \leq C_r \text{ or } j = j_{\text{rand}} \\
x_{i,j}^{(g)} & \text{otherwise}
\end{cases}
$$


  - **type**: float
  - **default**: 0.5
  - **typical range**: 0.0 ... 1.0

---