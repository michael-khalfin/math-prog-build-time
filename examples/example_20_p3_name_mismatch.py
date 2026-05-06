"""
Regression test for P3 naming bug: cap_rule uses (src, dst) while flow_rule
loops over 'j'/'i'. The old name-based filter on all_names broke when loop
variable names didn't match registry names from cap_rule's args.
"""
import pyomo.environ as pyo

raw_data = {
    'Nodes': ['A', 'B', 'C'],
    'Commodities': ['X', 'Y'],
    'Edges': [('A', 'B'), ('B', 'C'), ('A', 'C')],
    'Capacity': {('A', 'B'): 10, ('B', 'C'): 15, ('A', 'C'): 5},
    'Demand': {
        ('A', 'X'): 5,  ('C', 'X'): -5,
        ('A', 'Y'): 8,  ('C', 'Y'): -8,
        ('B', 'X'): 0,  ('B', 'Y'): 0,
    },
}

def preprocess_data(data):
    out_arcs = {n: [] for n in data['Nodes']}
    in_arcs  = {n: [] for n in data['Nodes']}
    for src, dst in data['Edges']:
        out_arcs[src].append(dst)
        in_arcs[dst].append(src)
    data['OutArcs'] = out_arcs
    data['InArcs']  = in_arcs
    return data

data = preprocess_data(raw_data)


def build_pyomo_model(data):
    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=data['Nodes'])
    m.K = pyo.Set(initialize=data['Commodities'])
    m.E = pyo.Set(dimen=2, initialize=data['Edges'])

    m.OutArcs = pyo.Set(m.N, initialize=data['OutArcs'])
    m.InArcs  = pyo.Set(m.N, initialize=data['InArcs'])

    m.x   = pyo.Var(m.E, m.K, domain=pyo.NonNegativeReals)
    m.Cap = pyo.Param(m.E, initialize=data['Capacity'])
    m.Dem = pyo.Param(m.N, m.K, initialize=data['Demand'])

    # cap_rule uses (src, dst) — registry will record E -> ['src', 'dst']
    def cap_rule(m, src, dst):
        return sum(m.x[src, dst, k] for k in m.K) <= m.Cap[src, dst]
    m.cap_constr = pyo.Constraint(m.E, rule=cap_rule)

    # flow_rule loops with 'j'/'i' — different names than cap_rule's src/dst
    def flow_rule(m, node, k):
        flow_out = sum(m.x[node, j, k] for j in m.OutArcs[node])
        flow_in  = sum(m.x[i, node, k] for i in m.InArcs[node])
        return flow_out - flow_in == m.Dem[node, k]
    m.flow_constr = pyo.Constraint(m.N, m.K, rule=flow_rule)

    return m
