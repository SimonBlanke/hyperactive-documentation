### `crossover_rate`

Crossover Rate (CR) is a parameter in the Differential Evolution (DE) algorithm that controls the probability of mixing components from the target vector and the mutant vector to form a trial vector. It determines how much of the trial vector inherits its components from the mutant vector versus the target vector. A high crossover rate means that more components will come from the mutant vector, promoting exploration of new solutions. Conversely, a low crossover rate results in more components being taken from the target vector, which can help maintain existing solutions and refine them. The typical range for CR is between 0.0 and 1.0, and its optimal value often depends on the specific problem being solved.


$$
u_{i,j,G+1} =
\begin{cases} 
v_{i,j,G+1} & \text{if } \text{rand}_j(0,1) \leq CR \text{ or } j = j_{\text{rand}} \\
x_{i,j,G} & \text{otherwise}
\end{cases}
$$

  - **type**: float
  - **default**: 0.9
  - **typical range**: 0.0 ... 1.0

---