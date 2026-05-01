import pyomo.environ as pyo
import gurobipy as gp
import gurobipy_pandas as gppd
import pandas as pd

# --- DATA & PREPROCESSING ---
# Single-commodity min-cost flow on a small directed network.
# Stresses the translator's P3 (flow balance) detection when the rule
# writes the subtraction *inline* rather than assigning intermediate names:
#
#   return (sum(...) - sum(...)) == m.Balance[n]
#
# This is semantically identical to example_3_multiflow's named-intermediate
# style but parses differently — the AST has BinOp(Call, Sub, Call) instead
# of BinOp(Name, Sub, Name), which the original P3 check missed.

raw_data = {
    'Nodes':   ['S', 'A', 'B', 'T'],
    'Arcs':    [('S','A'), ('S','B'), ('A','T'), ('B','T'), ('A','B')],
    'Balance': {'S': 10, 'A': 0, 'B': 0, 'T': -10},
}

def preprocess_data(data):
    out_arcs = {n: [] for n in data['Nodes']}
    in_arcs  = {n: [] for n in data['Nodes']}
    for (i, j) in data['Arcs']:
        out_arcs[i].append(j)
        in_arcs[j].append(i)
    data['OutArcs'] = out_arcs
    data['InArcs']  = in_arcs
    return data

data = preprocess_data(raw_data)

# ==========================================
# THE PYOMO TARGET (The Ingestion File)
# ==========================================
def build_pyomo_model(data):
    m = pyo.ConcreteModel()

    m.N       = pyo.Set(initialize=data['Nodes'])
    m.Arcs    = pyo.Set(dimen=2, initialize=data['Arcs'])
    m.OutArcs = pyo.Set(m.N, initialize=data['OutArcs'])
    m.InArcs  = pyo.Set(m.N, initialize=data['InArcs'])
    m.Balance = pyo.Param(m.N, initialize=data['Balance'])

    m.flow = pyo.Var(m.Arcs, domain=pyo.NonNegativeReals)

    # PATTERN: Inline P3 — subtraction of two sums written directly in return,
    # no intermediate variable assignments.
    def balance_rule(m, n):
        return (sum(m.flow[n, j] for j in m.OutArcs[n]) -
                sum(m.flow[i, n] for i in m.InArcs[n])) == m.Balance[n]

    m.balance_constr = pyo.Constraint(m.N, rule=balance_rule)

    return m

# ==========================================
# THE PANDAS EQUIVALENT (The Output Goal)
# ==========================================
def build_vectorized_model(data):
    m = gp.Model()

    idx_flow = pd.MultiIndex.from_tuples(data['Arcs'], names=['i', 'j'])
    df_flow = pd.DataFrame(index=idx_flow)
    df_flow = df_flow.gppd.add_vars(m, name='flow')

    s_balance = pd.Series(data['Balance'], name='balance').rename_axis('n')

    # flow_out[n] = sum of flow leaving node n  (groupby source 'i', rename to 'n')
    flow_out = df_flow.groupby('i')['flow'].sum().rename_axis(index={'i': 'n'})
    # flow_in[n]  = sum of flow arriving at node n (groupby dest 'j', rename to 'n')
    flow_in  = df_flow.groupby('j')['flow'].sum().rename_axis(index={'j': 'n'})

    lhs_balance = flow_out.sub(flow_in, fill_value=0)

    gppd.add_constrs(m, lhs_balance, gp.GRB.EQUAL, s_balance, name='balance_constr')

    return m
