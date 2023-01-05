# Objective Function

Each iteration consists of two steps:

- The optimization step: decides what position in the search space (parameter set) to evaluate next
- The evaluation step: calls the objective function, which returns the score for the given position in the search space

The objective function has one argument that is often called "para", "params", "opt" or "access". This argument is your access to the parameter set that the optimizer has selected in the corresponding iteration.

!!! example 
    ```python
    def objective_function(opt):
        # get x1 and x2 from the argument "opt"
        x1 = opt["x1"]
        x2 = opt["x2"]

        # calculate the score with the parameter set
        score = -(x1 * x1 + x2 * x2)

        # return the score
        return score
    ```


The objective function always needs a score, which shows how "good" or "bad" the current parameter set is. But you can also return some additional information with a dictionary:

!!! example 
    ```python
    def objective_function(opt):
        x1 = opt["x1"]
        x2 = opt["x2"]

        score = -(x1 * x1 + x2 * x2)

        other_info = {
        "x1 squared" : x1**2,
        "x2 squared" : x2**2,
        }

        return score, other_info
    ```

When you take a look at the results (a pandas dataframe with all iteration information) after the run has ended you will see the additional information in it. The reason we need a dictionary for this is because Hyperactive needs to know the names of the additonal parameters. The score does not need that, because it is always called "score" in the results.