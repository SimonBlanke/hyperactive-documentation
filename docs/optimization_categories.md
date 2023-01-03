


## Optimization Categories


| Algorithm                         | Local | Global | Convergence | Exploration | Expensive |
|-----------------------------------|-------|--------|-------------|-------------|-----------|
| Hill Climbing                     | 🟢    | 🔴     | 🟢          | 🔴          | 🔴        |
| Stochastic Hill Climbing          | 🟢    | 🔴     | 🟢          | 🔴          | 🔴        |
| Simulated Annealing               | 🟢    | 🔴     | 🟢          | 🔴          | 🔴        |
| Repulsing Hill Climbing           | 🟢    | 🔵     | 🟢          | 🔴          | 🔴        |
| Downhill Simplex Optimization     | 🟢    | 🔵     | 🟢          | 🔵          | 🔴        |
| Pattern Search                    | 🟢    | 🔵     | 🟢          | 🔵          | 🔴        |
| Powell's Method                   | 🟢    | 🔴     | 🟢          | 🔴          | 🔴        |
| Random Restart Hill Climbing      | 🟢    | 🔵     | 🔵          | 🟢          | 🔴        |
| Random Search                     | 🔴    | 🟢     | 🔴          | 🟢          | 🔴        |
| Random Annealing                  | 🟢    | 🔵     | 🟢          | 🟢          | 🔴        |
| Grid Search                       | 🔵    | 🔵     | 🔴          | 🟢          | 🔴        |
| Parallel Tempering                | 🔵    | 🔵     | 🟢          | 🔵          | 🔴        |
| Particle Swarm Optimization       | 🔵    | 🔵     | 🟢          | 🔴          | 🔴        |
| Spiral Optimization               | 🔵    | 🔵     | 🟢          | 🔵          | 🔴        |
| Evolution Strategy                | 🔵    | 🔵     | 🟢          | 🔵          | 🔴        |
| Bayesian Optimization             | 🔴    | 🟢     | 🔵          | 🟢          | 🟢        |
| Lipschitz Optimization            | 🔴    | 🟢     | 🔴          | 🟢          | 🟢        |
| DIRECT Algorithm                  | 🔴    | 🟢     | 🟢          | 🔵          | 🔵        |
| Tree Structured Parzen Estimators | 🟢    | 🔵     | 🔵          | 🔵          | 🟢        |
| Forest Optimization               | 🔴    | 🟢     | 🔴          | 🔵          | 🟢        |


!!! warning

    The classification of the optimization algorithms into the categories above is not necessarily scientific, but aims to provide an overview for users. The goal is to give an indication of which algorithm might be a good fit to a given problem.




The algorithms can be described by the following categories:

- **Local** - The algorithm works by choosing new positions within a neighbourhood of the previous positions. It is recommended to use this algorithm for convex optimization problems.
- **Global** - The algorithm works by choosing new positions independently of the previous positions. It is recommended to use this algorithm for non-convex optimization problems.
- **Convergence** - The algorithm tends to focus on finding the optimum in the search-space. It is recommended to use this algorithm to find a good solution quickly.
- **Exploration** - The algorithm tends to explore the search-space. It is recommended to use this algorithm to get an overview of the optimization problem.
- **Expensive** - The algorithm is computationally expensive. It is recommended, that it is used for objective-functions that are also computationally expensive (e.g. hyperparameter optimization).


The following markers describes how well the algorithm fits into the category:

- 🟢 - The algorithm fits into this category
- 🔵 - The algorithm fits somewhat into this category
- 🔴 - The algorithm does not fit into this category
  
