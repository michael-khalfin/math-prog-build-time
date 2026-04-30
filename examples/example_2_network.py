import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- THE DATA ---
data = {
    'Nodes': ['N1', 'N2', 'N3'],
    'Edges': [('N1', 'N2'), ('N2', 'N3'), ('N1', 'N3')],
    'Costs': {('N1', 'N2'): 2.5, ('N2', 'N3'): 1.5, ('N1', 'N3'): 4.0},
    'MaxBudget': 100.0
}

# --- THE PYOMO TARGET (The agent ingests this) ---
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.Nodes = pyo.Set(initialize=data['Nodes'])
    m.Edges = pyo.Set(dimen=2, initialize=data['Edges'])
    m.x = pyo.Var(m.Edges, domain=pyo.NonNegativeReals)
    m.Cost = pyo.Param(m.Edges, initialize=data['Costs'])
    
    # MaxBudget must be a parameter
    m.MaxBudget = pyo.Param(initialize=data['MaxBudget'])
    
    def budget_rule(m):
        return sum(m.Cost[i, j] * m.x[i, j] for i, j in m.Edges) <= m.MaxBudget
    
    m.budget_constr = pyo.Constraint(rule=budget_rule)
    
    return m

# --- THE PANDAS EQUIVALENT (The agent must generate this) ---
def build_vectorized_model(data):
    m = gp.Model()
    
    # Edges Set -> Index setup
    idx = pd.MultiIndex.from_tuples(data['Edges'], names=['i', 'j'])
    df = pd.DataFrame(index=idx)
    df = df.gppd.add_vars(m, name="x")
    
    # Params -> Columns / Scalars
    df['Cost'] = pd.Series(data['Costs'])
    max_budget = data['MaxBudget']
    
    # Rule Translation: sum() -> groupby().sum()
    lhs = (df['Cost'] * df['x']).sum()
    
    # Instantiation
    m.addLConstr(lhs <= max_budget, name="budget_constr")
    
    return m