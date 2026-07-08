The normal workflow for defining a model with Pyomo and then solving using Gurobi is:
![Control Flow Diagram](diagram1/diagram1.pdf)

However, we may build and solve a Pyomo model just to discover that the model is infeasible or unbounded. In this case, we may want to iterate on the Pyomo model to fix the issue. Then, we may prefer a workflow that is more like the following, where we build and solve a Gurobi model first, and then build a Pyomo model retroactively if the Gurobi model is feasible:
![Control Flow Diagram](diagram2/diagram2.pdf)

"Run transpilation" is a nontrivial step. Traditionally, this step may have taken hundreds of hours of development time to implement, and it may have been difficult to maintain. However, since it's 2026, we were able to generate this step cheaply using Claude Code.

To verify that the transpilation step is correct, we can run a test suite with several small datasets to check that the Gurobi models built using both workflows are equivalent. Once we are confident that the transpilation step is correct for *a specific Pyomo model*, then we can adopt the transpiled code going forward for the large datasets.

There are two main advantages to the transpiled workflow. The first being that we can write models in Pyomo syntax (interpretable, lightweight) while still benefiting from the performance of direct Gurobi calls. We can still get a Pyomo model after the solve, which is useful for backward compatibility and due to Pyomo's transformations, algorithmic, and post-processing capabilities. (It is worth noting that these transformations can also be transpiled and verified pro re nata, but this is not the focus of this paper.)

The second advantage is that the transpiled workflow resolves a fundamental performance tradeoff inherent in Pyomo's native Gurobi interfaces. In a standard Pyomo pipeline, users must select a backend—typically gurobi_direct or gurobi_persistent. While Pyomo offers 'templatization' to significantly accelerate the generation of model instances, this feature is incompatible with the gurobi_persistent backend. Consequently, the standard workflow forces a choice: optimize for initial build speed (using templatization) or optimize for iterative resolve speed (using persistence). The transpiled workflow circumvents this limitation entirely. By bypassing the Pyomo backend, the direct CVaR Gurobi implementation achieves generation speeds strictly faster than Pyomo's templatization, while inherently preserving the persistent in-memory state required for rapid iterative updates.

This is useful for scenarios where we want to iterate on a model, such as in a branch-and-bound algorithm, or when we want to solve the same model with different data.