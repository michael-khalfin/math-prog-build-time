import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

raw_data = {
    'Plants': ['P1', 'P2'],
    'StdLines': ['S1', 'S2'],
    'ExpLines': ['E1'],
    
    'Cap': {'P1': 100, 'P2': 150},
    'CostStd': {'S1': 10, 'S2': 12},
    'CostExp': {'E1': 20}
}
data = raw_data

# ==========================================
# PYOMO TARGET
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.P = pyo.Set(initialize=data['Plants'])
    m.S = pyo.Set(initialize=data['StdLines'])
    m.E = pyo.Set(initialize=data['ExpLines'])
    
    m.Cap = pyo.Param(m.P, initialize=data['Cap'])
    m.CostStd = pyo.Param(m.S, initialize=data['CostStd'])
    m.CostExp = pyo.Param(m.E, initialize=data['CostExp'])
    
    m.x_std = pyo.Var(m.P, m.S, domain=pyo.NonNegativeReals)
    m.x_exp = pyo.Var(m.P, m.E, domain=pyo.NonNegativeReals)

    # PATTERN: Inter-sum addition (ast.BinOp where left/right are Call(sum))
    def cap_rule(m, p):
        return sum(m.x_std[p, s] for s in m.S) + sum(m.x_exp[p, e] for e in m.E) <= m.Cap[p]
    m.cap_constr = pyo.Constraint(m.P, rule=cap_rule)
    
    # PATTERN: Multi-term objective function
    m.obj = pyo.Objective(
        expr=sum(m.CostStd[s] * m.x_std[p, s] for p in m.P for s in m.S) + 
             sum(m.CostExp[e] * m.x_exp[p, e] for p in m.P for e in m.E),
        sense=pyo.minimize
    )
    return m

# ==========================================
# PANDAS EQUIVALENT
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()
    
    idx_ps = pd.MultiIndex.from_product([data['Plants'], data['StdLines']], names=['p', 's'])
    df_std = pd.DataFrame(index=idx_ps).gppd.add_vars(m, name="x_std")
    
    idx_pe = pd.MultiIndex.from_product([data['Plants'], data['ExpLines']], names=['p', 'e'])
    df_exp = pd.DataFrame(index=idx_pe).gppd.add_vars(m, name="x_exp")
    
    # Required: Parameters must be named for .join() to work without ValueError
    s_cap = pd.Series(data['Cap'], name='cap').rename_axis('p')
    s_cost_std = pd.Series(data['CostStd'], name='cost_std').rename_axis('s')
    s_cost_exp = pd.Series(data['CostExp'], name='cost_exp').rename_axis('e')
    
    # --- Constraint: Inter-sum Addition ---
    lhs_term1 = df_std.groupby('p')['x_std'].sum()
    lhs_term2 = df_exp.groupby('p')['x_exp'].sum()
    
    # Safely combine them using .add(fill_value=0)
    lhs_total = lhs_term1.add(lhs_term2, fill_value=0)
    
    gppd.add_constrs(m, lhs_total, gp.GRB.LESS_EQUAL, s_cap, name="cap_constr")
    
    # --- Objective: Multi-Term Addition ---
    df_std_cost = df_std.join(s_cost_std, on='s')
    obj_term1 = (df_std_cost['x_std'] * df_std_cost['cost_std']).sum()
    
    df_exp_cost = df_exp.join(s_cost_exp, on='e')
    obj_term2 = (df_exp_cost['x_exp'] * df_exp_cost['cost_exp']).sum()
    
    m.setObjective(obj_term1 + obj_term2, gp.GRB.MINIMIZE)
    return m