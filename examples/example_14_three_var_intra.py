import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA ---
# Three resource types consumed by activities at each node in each period.
# Tests that the intra-sum parser handles 3+ terms from a left-associative
# BinOp(Add) tree: ((x + y) + z) is the AST for `x + y + z`.

data = {
    'Nodes': ['N1', 'N2'],
    'Time':  [1, 2, 3],
    'Cap':   {'N1': 50.0, 'N2': 80.0},
}

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()

    m.N = pyo.Set(initialize=data['Nodes'])
    m.T = pyo.Set(initialize=data['Time'])

    m.Cap = pyo.Param(m.N, initialize=data['Cap'])

    m.x = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    m.z = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)

    # PATTERN: Three-variable intra-sum (same MultiIndex shape)
    # AST elt = BinOp(BinOp(x, Add, y), Add, z) — left-associative tree
    def cap_rule(m, n):
        return sum(m.x[n, t] + m.y[n, t] + m.z[n, t] for t in m.T) <= m.Cap[n]
    m.cap_constr = pyo.Constraint(m.N, rule=cap_rule)

    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()

    idx_nt = pd.MultiIndex.from_product([data['Nodes'], data['Time']], names=['n', 't'])
    df_x = pd.DataFrame(index=idx_nt).gppd.add_vars(m, name='x')
    df_y = pd.DataFrame(index=idx_nt).gppd.add_vars(m, name='y')
    df_z = pd.DataFrame(index=idx_nt).gppd.add_vars(m, name='z')

    s_cap = pd.Series(data['Cap'], name='cap').rename_axis('n')

    # Flatten, merge, combine
    _fi0 = df_x['x'].reset_index()
    _fi1 = df_y['y'].reset_index()
    _mi = pd.merge(_fi0, _fi1, on=['n', 't'])
    _fi2 = df_z['z'].reset_index()
    _mi = pd.merge(_mi, _fi2, on=['n', 't'])
    lhs_cap_constr = (_mi['x'] + _mi['y'] + _mi['z']).groupby('n').sum()

    gppd.add_constrs(m, lhs_cap_constr, gp.GRB.LESS_EQUAL, s_cap, name='cap_constr')
    return m
