import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

raw_data = {
    'Nodes': ['N1', 'N2'],
    'Time': [1, 2],
    'Capacity': {'N1': 100, 'N2': 150}
}
data = raw_data

# ==========================================
# PYOMO TARGET
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=data['Nodes'])
    m.T = pyo.Set(initialize=data['Time'])
    m.Cap = pyo.Param(m.N, initialize=data['Capacity'])
    
    m.x = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)

    # PATTERN: Intra-sum addition (ast.BinOp inside the generator)
    def state_rule(m, n):
        return sum(m.x[n, t] + m.y[n, t] for t in m.T) <= m.Cap[n]
    m.state_constr = pyo.Constraint(m.N, rule=state_rule)
    
    return m

# ==========================================
# PANDAS EQUIVALENT
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_nt = pd.MultiIndex.from_product([data['Nodes'], data['Time']], names=['n', 't'])
    
    df_x = pd.DataFrame(index=idx_nt).gppd.add_vars(m, name="x")
    df_y = pd.DataFrame(index=idx_nt).gppd.add_vars(m, name="y")
    
    s_cap = pd.Series(data['Capacity'], name='cap').rename_axis('n')
    
    # --- Constraint: Intra-sum Addition ---
    # Because x and y share the exact same index, add them BEFORE grouping
    combined_vars = df_x['x'] + df_y['y']
    lhs_state = combined_vars.groupby('n').sum()
    
    gppd.add_constrs(m, lhs_state, gp.GRB.LESS_EQUAL, s_cap, name="state_constr")
    
    return m