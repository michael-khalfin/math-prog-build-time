import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- THE DATA ---
data = {
    'I': ['Plant_A', 'Plant_B'],
    'J': ['Market_1', 'Market_2', 'Market_3'],
    'Supply': {'Plant_A': 100, 'Plant_B': 150}
}

# --- THE PYOMO TARGET (The agent ingests this) ---
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=data['I'])
    m.J = pyo.Set(initialize=data['J'])
    m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)
    m.Supply = pyo.Param(m.I, initialize=data['Supply'])

    def supply_rule(m, i):
        return sum(m.x[i, j] for j in m.J) <= m.Supply[i]
    m.supply_constr = pyo.Constraint(m.I, rule=supply_rule)
    
    return m

# --- THE PANDAS EQUIVALENT (The agent must generate this) ---
def build_vectorized_model(data):
    m = gp.Model()
    
    # Set -> Index setup
    idx = pd.MultiIndex.from_product([data['I'], data['J']], names=['i', 'j'])
    df_vars = pd.DataFrame(index=idx)
    df_vars = df_vars.gppd.add_vars(m, name="x")  
      
    # Param -> Column/Series setup
    s_supply = pd.Series(data['Supply'], name="Supply").rename_axis('i')
    
    # Rule Translation: sum() -> groupby().sum()
    lhs = df_vars.groupby('i')['x'].sum()
    
    # Instantiation
    gppd.add_constrs(m, lhs, gp.GRB.LESS_EQUAL, s_supply, name="supply_constr")
    
    return m