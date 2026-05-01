import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA ---
# Workers assigned to products + flexible overtime shared across all products.
#   assign[p, w]  — hours worker w contributes to product p  (2-D: P x W)
#   flex[w]       — extra flexible hours of worker w, available to any product  (1-D: W)
#
# Demand constraint: for each product p,
#   sum_{w} (assign[p,w] + flex[w]) >= Demand[p]
# = sum_{w} assign[p,w]  +  sum_{w} flex[w]  >= Demand[p]
#
# flex is 1-D (indexed only by w), assign is 2-D (p, w).
# The merge approach merges on 'w' (the shared loop column), broadcasting
# flex[w] across all products before groupby('p').sum().

data = {
    'Products': ['P1', 'P2', 'P3'],
    'Workers':  ['W1', 'W2'],
    'Demand':   {'P1': 10.0, 'P2': 15.0, 'P3': 8.0},
}

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()

    m.P = pyo.Set(initialize=data['Products'])
    m.W = pyo.Set(initialize=data['Workers'])

    m.Demand = pyo.Param(m.P, initialize=data['Demand'])

    # assign is (P x W), flex is (W,) — DIFFERENT index shapes
    m.assign = pyo.Var(m.P, m.W, domain=pyo.NonNegativeReals)
    m.flex   = pyo.Var(m.W,      domain=pyo.NonNegativeReals)

    # PATTERN: Intra-sum with differently-shaped variables.
    # assign[p,w] is 2-D; flex[w] is 1-D.
    # Merge on 'w' broadcasts flex across all products.
    def demand_rule(m, p):
        return sum(m.assign[p, w] + m.flex[w] for w in m.W) >= m.Demand[p]
    m.demand_constr = pyo.Constraint(m.P, rule=demand_rule)

    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()

    idx_pw = pd.MultiIndex.from_product([data['Products'], data['Workers']], names=['p', 'w'])
    df_assign = pd.DataFrame(index=idx_pw).gppd.add_vars(m, name='assign')

    idx_w = pd.Index(data['Workers'], name='w')
    df_flex = pd.DataFrame(index=idx_w).gppd.add_vars(m, name='flex')

    s_demand = pd.Series(data['Demand'], name='demand').rename_axis('p')

    # Flatten each variable, merge on shared column 'w'
    _fi0 = df_assign['assign'].reset_index()   # [p, w, assign]
    _fi1 = df_flex['flex'].reset_index()        # [w, flex]
    _mi  = pd.merge(_fi0, _fi1, on='w')         # [p, w, assign, flex]

    lhs_demand_constr = (_mi['assign'] + _mi['flex']).groupby('p').sum()

    gppd.add_constrs(m, lhs_demand_constr, gp.GRB.GREATER_EQUAL, s_demand, name='demand_constr')
    return m
