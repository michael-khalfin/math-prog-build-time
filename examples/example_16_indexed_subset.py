import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA ---
# Workers (W) can be assigned to products (P).
# Not every worker can work on every product — SubSet[p] lists the eligible workers.
# The constraint is: for each product p, the total hours across eligible workers <= Cap[p].
#
# This tests P5 where the OUTER constraint index 'p' also appears inside
# the variable subscript: sum(m.x[p, w] for w in m.SubSet[p]).
# The variable is 2-D (P × W) but only eligible (p, w) pairs contribute.

data = {
    'Products': ['P1', 'P2', 'P3'],
    'Workers':  ['W1', 'W2', 'W3', 'W4'],
    'SubSet': {
        'P1': ['W1', 'W2'],
        'P2': ['W2', 'W3'],
        'P3': ['W1', 'W4'],
    },
    'Cap': {'P1': 10.0, 'P2': 15.0, 'P3': 8.0},
}

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()

    m.P = pyo.Set(initialize=data['Products'])
    m.W = pyo.Set(initialize=data['Workers'])

    m.SubSet = pyo.Set(m.P, initialize=data['SubSet'])
    m.Cap    = pyo.Param(m.P, initialize=data['Cap'])

    # Variable is 2-D: (product, worker).
    # PATTERN: outer constraint index 'p' appears IN the variable subscript.
    # sum(m.x[p, w] for w in m.SubSet[p]) — P5 with x[outer, inner].
    m.x = pyo.Var(m.P, m.W, domain=pyo.NonNegativeReals)

    def cap_rule(m, p):
        return sum(m.x[p, w] for w in m.SubSet[p]) <= m.Cap[p]
    m.cap_constr = pyo.Constraint(m.P, rule=cap_rule)

    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()

    idx_pw = pd.MultiIndex.from_product(
        [data['Products'], data['Workers']], names=['p', 'w']
    )
    df_x = pd.DataFrame(index=idx_pw).gppd.add_vars(m, name='x')

    s_cap = pd.Series(data['Cap'], name='cap').rename_axis('p')

    # Build (p, w) mapping from the indexed subset dict, then merge on BOTH 'p' and 'w'
    # so that only eligible (p, w) pairs are selected.
    _mapping = [(p, w) for p, ws in data['SubSet'].items() for w in ws]
    df_map = pd.DataFrame(_mapping, columns=['p', 'w'])
    df_reset = df_x['x'].reset_index()                   # ['p', 'w', 'x']
    df_lagged = pd.merge(df_map, df_reset, on=['p', 'w'])  # no duplicate columns

    lhs_cap_constr = df_lagged.groupby('p')['x'].sum()
    lhs_cap_constr = lhs_cap_constr.reindex(
        pd.Index(data['Products'], name='p'), fill_value=0.0
    )

    gppd.add_constrs(m, lhs_cap_constr, gp.GRB.LESS_EQUAL, s_cap, name='cap_constr')
    return m
